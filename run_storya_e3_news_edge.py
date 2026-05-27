#!/usr/bin/env python
"""run_storya_e3_news_edge.py — Story A E3 (news-as-edge co-occurrence).

Plan §1.2: tests "news-as-feature catastrophic at 21d, news-as-EDGE helps" hypothesis.

50 NEW cells: SAGE-Mean × Universe B × {correlation + news_cooccurrence} edge
config × 10 canonical seeds × 5 walk-forward folds.
(The corresponding 50 baseline cells (correlation-only) are REUSED from E1-B
SAGE-Mean — no re-training; analysis-side stat tests pair the two cell families.)

Reuses (via import from run_storya_e1_anchor):
  - constants (HORIZON, CANONICAL_SEEDS, WALK_FORWARD_FOLDS, NN_HPARAMS, GRAPH_HPARAMS)
  - data loaders (load_core_data, build_universe_B, build_labels)
  - graph builder (build_correlation_snapshots, get_frozen_snapshot_idx)
  - model (make_nn_model — only SAGE-Mean branch used)
  - fold helpers (create_fold_masks, winsorize_train_only, standardize_train_only)
  - metrics (compute_daily_ic, compute_cost_ladder_sharpe)
  - leak assertion (assert_purge_no_leak)

E3-specific additions:
  - load_news_edge_source(): reads data/fullscale/sp500_news_edge_source.parquet
  - nyse_session_close_utc(): PIT cutoff per news_edge_source_schema.md v2 (Codex D-03 fix)
  - build_per_day_news_edges(): co-occurrence edges per prediction date
  - union_edges_per_day(): corr_frozen ∪ news_per_day, deduplicated, symmetric
  - train_sage_per_day_edges(): training loop that switches edge_index per day

PIT contract (LOCKED per experiments/storya_e3_news_edge/news_edge_source_schema.md):
  At prediction date t, eligible articles satisfy publication_timestamp <= cutoff,
  where cutoff = NYSE session_close(t-1) in UTC (DST-aware, not naive UTC midnight).
  Articles in [cutoff, t+21] are EXCLUDED — guaranteed by the filter, not by post-hoc check.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from collections import defaultdict
from itertools import combinations
from typing import Optional

# macOS OpenMP segfault workaround per CR-A-02 follow-up (Darwin only; no-op on Linux Colab)
if platform.system() == 'Darwin':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# cudnn determinism per CR-A-05
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# pandas_market_calendars for NYSE session close (DST-aware) per Codex D-03 fix
import pandas_market_calendars as mcal

# Reuse from E1 anchor (importing it has no side effects beyond setting env vars +
# cudnn flags, both idempotent). main() is guarded so no training runs.
import run_storya_e1_anchor as e1
from run_storya_e1_anchor import (
    CANONICAL_SEEDS, WALK_FORWARD_FOLDS, HORIZON, TRAIN_START,
    NN_HPARAMS, GRAPH_HPARAMS,
    set_seed, get_device,
    load_core_data, build_universe_B, build_labels,
    build_correlation_snapshots, get_frozen_snapshot_idx,
    make_nn_model,
    create_fold_masks, winsorize_train_only, standardize_train_only,
    compute_daily_ic, compute_cost_ladder_sharpe,
    assert_purge_no_leak,
    COST_LEVELS_BPS, COST_CONVENTION,
)


# ══════════════════════════════════════════════════════════════
# CONFIG — E3-specific
# ══════════════════════════════════════════════════════════════

MODEL = 'SAGE-Mean'                 # E3 fixed: only SAGE-Mean
UNIVERSE = 'B'                      # E3 fixed: only Universe B (10-dim hc)
EDGE_CONFIG = 'corr+news_cooccur'   # E3 treatment

NEWS_LOOKBACK_TRADING_DAYS = 5      # plan §1.2 PIT lookback (≈ 7 calendar days)
NEWS_LOOKBACK_CALENDAR_DAYS = NEWS_LOOKBACK_TRADING_DAYS + 2  # +2 buffer for weekends

# I/O paths (E3-specific output dir; data inputs are shared with E1)
PATHS = {
    'prices': 'data/reference/sp500_5y_prices.csv',
    'sectors': 'data/reference/sp500_sectors.csv',
    'phase5_npy': 'data/reference/sp500_5y_phase5_features.npy',
    'news_edge_source': 'data/fullscale/sp500_news_edge_source.parquet',  # built by scripts/build_news_edge_source.py
}
OUT_DIR = 'experiments/storya_e3_news_edge'
RESULTS_CSV = f'{OUT_DIR}/results.csv'
MANIFEST_CSV = f'{OUT_DIR}/manifest.csv'
PER_DAY_IC_DIR = f'{OUT_DIR}/per_day_ic'
HP_GRID_JSON = f'{OUT_DIR}/hp_grid.json'
META_JSON = f'{OUT_DIR}/_meta.json'
NEWS_SNAPSHOT_CACHE = f'{OUT_DIR}/news_snapshots_cache.npz'

# Cell schema: 50 cells = 10 seeds × 5 folds (universe + model + edge_config fixed)
RESULTS_COLUMNS = (
    ['cell_id', 'universe', 'model', 'edge_config', 'seed', 'fold', 'test_period',
     'IC_mean', 'IC_std', 'n_test_days', 'Sharpe_gross']
    + [f'Sharpe_net_{c}bps' for c in COST_LEVELS_BPS]
    + ['mean_turnover_L1', 'n_periods', 'best_val_loss', 'epochs_run',
       'wall_time_sec', 'converged_flag', 'cost_convention',
       'n_news_edges_avg', 'n_news_articles_avg']
)
MANIFEST_COLUMNS = ['cell_id', 'fold', 'seed', 'status', 'start_ts', 'end_ts', 'wall_time_sec', 'err']


def cell_id_e3(fold_idx: int, seed_idx: int) -> int:
    """E3 cell_id: range [0, 49]. Injective by radix construction."""
    return fold_idx * 10 + seed_idx


def assert_cell_id_e3_injective() -> None:
    seen = set()
    for f in range(5):
        for s in range(10):
            cid = cell_id_e3(f, s)
            assert cid not in seen, f"cell_id collision at f={f}, s={s}"
            seen.add(cid)
    assert max(seen) == 49 and min(seen) == 0 and len(seen) == 50
    print(f"✓ E3 cell_id formula injective, range [0, 49], n=50 cells")


# ══════════════════════════════════════════════════════════════
# PIT-SAFE NEWS EDGE BUILDER
# ══════════════════════════════════════════════════════════════

_NYSE_CACHE = {'schedule': None}


def _to_naive(ts: pd.Timestamp) -> pd.Timestamp:
    """Strip tz info — for comparing against pandas_market_calendars' tz-naive index."""
    return ts.tz_localize(None) if ts.tz is not None else ts


def _get_nyse_schedule(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """Cache the NYSE schedule so we only fetch once. Index is tz-naive."""
    s_naive = _to_naive(start_date)
    e_naive = _to_naive(end_date)
    cache = _NYSE_CACHE['schedule']
    if cache is None or cache.index.min() > s_naive or cache.index.max() < e_naive:
        nyse = mcal.get_calendar('NYSE')
        _NYSE_CACHE['schedule'] = nyse.schedule(
            start_date=s_naive - pd.Timedelta(days=30),
            end_date=e_naive + pd.Timedelta(days=30),
        )
    return _NYSE_CACHE['schedule']


def nyse_session_close_utc(date_like) -> pd.Timestamp:
    """PIT cutoff per news_edge_source_schema.md v2 (Codex D-03 fix).

    Returns UTC timestamp of NYSE regular session close for the trading day
    on-or-before `date_like`. Handles DST automatically via pandas_market_calendars.
    Handles early-close days (e.g., day after Thanksgiving, 13:00 ET = 18:00 UTC).
    """
    d = pd.Timestamp(date_like).normalize()
    d_naive = _to_naive(d)
    # Look back up to 14 calendar days to find prior trading day (handles weekends + holidays)
    sched = _get_nyse_schedule(d_naive - pd.Timedelta(days=14), d_naive)
    on_or_before = sched[sched.index <= d_naive]
    if on_or_before.empty:
        raise ValueError(f"No NYSE trading session on-or-before {d_naive}")
    last_close = on_or_before.iloc[-1]['market_close']
    # market_close is already tz-aware UTC (verified via pmc inspection). Defensive convert.
    if last_close.tz is None:
        last_close = last_close.tz_localize('America/New_York').tz_convert('UTC')
    else:
        last_close = last_close.tz_convert('UTC')
    return last_close


def load_news_edge_source(path: str = None) -> pd.DataFrame:
    """Load the PIT-safe news edge source artifact built by scripts/build_news_edge_source.py.

    Asserts schema invariants per news_edge_source_schema.md.
    """
    p = path or PATHS['news_edge_source']
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"News edge source not built: {p}. "
            f"Run: python scripts/build_news_edge_source.py"
        )
    df = pd.read_parquet(p)
    # Invariants
    expected_cols = {'article_id', 'publication_timestamp', 'tickers_mentioned', 'n_tickers'}
    if set(df.columns) != expected_cols:
        raise ValueError(f"news_edge_source columns {set(df.columns)} != expected {expected_cols}")
    forbidden = ['return_next', 'label', 'polarity', 'neg', 'neu', 'pos', 'tags']
    for col in forbidden:
        if col in df.columns:
            raise AssertionError(f"FORBIDDEN forward field {col} in news_edge_source — re-build artifact")
    if not isinstance(df['publication_timestamp'].dtype, pd.DatetimeTZDtype):
        raise TypeError("publication_timestamp must be tz-aware datetime64")
    if str(df['publication_timestamp'].dt.tz) != 'UTC':
        raise ValueError(f"publication_timestamp tz = {df['publication_timestamp'].dt.tz}, expected UTC")
    return df


def build_per_day_news_edges(edge_source_df: pd.DataFrame,
                              trading_dates: pd.DatetimeIndex,
                              ticker_to_idx: dict[str, int],
                              lookback_calendar_days: int = NEWS_LOOKBACK_CALENDAR_DAYS):
    """For each trading date index t > 0, return co-occurrence edge_index (shape (2, E))
    over the PIT-eligible articles (publication_timestamp ∈ (cutoff - lookback, cutoff]).

    Edges are symmetric AND deduplicated. No self-loops.

    CODEX-CR-E3RUN-A-01 fix: now ALSO returns per-day eligible article counts so
    downstream can populate `n_news_articles_avg` honestly.

    Returns: (snapshots dict {day_idx: edge_index_np}, article_counts dict {day_idx: int}).
    For day_idx = 0 (no t-1 reference), no edges/counts are computed (skipped).
    """
    print(f"[news] building per-day PIT-safe news edge snapshots for {len(trading_dates)} dates ...")
    t0 = time.time()
    snapshots = {}
    article_counts = {}  # CR-E3RUN-A-01: track eligible article count per day
    # Sort source by publication_timestamp for efficient range queries
    edge_source_df = edge_source_df.sort_values('publication_timestamp').reset_index(drop=True)
    ts_arr = edge_source_df['publication_timestamp'].values  # numpy datetime64[ns]
    tickers_arr = edge_source_df['tickers_mentioned'].values  # numpy object array
    n_tickers_arr = edge_source_df['n_tickers'].values
    co_eligible_mask = n_tickers_arr >= 2  # only multi-ticker articles can form edges

    pit_max_seen = pd.Timestamp.min.tz_localize('UTC')
    for t_idx in range(1, len(trading_dates)):
        t_date = trading_dates[t_idx]
        cutoff_utc = nyse_session_close_utc(t_date - pd.Timedelta(days=1))
        lookback_start = cutoff_utc - pd.Timedelta(days=lookback_calendar_days)
        # Numpy datetime64 comparison (must convert pd.Timestamp UTC → np.datetime64)
        cutoff_np = np.datetime64(cutoff_utc.tz_convert(None) if cutoff_utc.tz else cutoff_utc, 'ns')
        lookback_np = np.datetime64(lookback_start.tz_convert(None) if lookback_start.tz else lookback_start, 'ns')
        # Eligible articles: lookback_start < ts <= cutoff AND n_tickers >= 2
        in_window = (ts_arr > lookback_np) & (ts_arr <= cutoff_np) & co_eligible_mask
        n_eligible = int(in_window.sum())
        article_counts[t_idx] = n_eligible
        if n_eligible == 0:
            snapshots[t_idx] = np.zeros((2, 0), dtype=np.int64)
            continue
        # PIT runtime assertion (per schema)
        window_max = ts_arr[in_window].max()
        assert window_max <= cutoff_np, (
            f"PIT VIOLATION at prediction date {t_date}: max article ts {window_max} > cutoff {cutoff_np}"
        )
        if pd.Timestamp(window_max, tz='UTC') > pit_max_seen:
            pit_max_seen = pd.Timestamp(window_max, tz='UTC')

        # Aggregate co-occurrence edges
        edge_count = defaultdict(int)
        for tickers in tickers_arr[in_window]:
            # Map to ticker_to_idx, drop unknown
            idxs = sorted({ticker_to_idx[t] for t in tickers if t in ticker_to_idx})
            for i, j in combinations(idxs, 2):
                edge_count[(i, j)] += 1
        if not edge_count:
            snapshots[t_idx] = np.zeros((2, 0), dtype=np.int64)
            continue
        directed = np.array(list(edge_count.keys()), dtype=np.int64).T  # (2, E)
        # Symmetrize for SAGE message passing
        ei = np.concatenate([directed, directed[[1, 0], :]], axis=1)
        snapshots[t_idx] = ei

        if t_idx % 200 == 0:
            print(f"  [news] t_idx={t_idx} ({t_date.date()}): {ei.shape[1]} edges; "
                  f"{n_eligible} eligible articles; cutoff={cutoff_utc}")

    print(f"[news] built {len(snapshots)} snapshots in {time.time()-t0:.1f}s  "
          f"(max PIT-eligible ts seen = {pit_max_seen})")
    return snapshots, article_counts


def union_edges_per_day(corr_frozen_np: np.ndarray, news_snapshots: dict[int, np.ndarray],
                         day_indices: np.ndarray) -> dict[int, torch.Tensor]:
    """For each day in day_indices, return torch tensor of (corr_frozen ∪ news_per_day) edges.

    Both inputs are np.int64 shape (2, E). Output is torch.long shape (2, E_total),
    deduped via np.unique on transposed pairs.
    """
    out = {}
    for d in day_indices:
        d = int(d)
        if d in news_snapshots:
            news = news_snapshots[d]
            if news.shape[1] > 0:
                combined = np.concatenate([corr_frozen_np, news], axis=1)
                # Dedupe pairs (transpose to (E, 2), unique rows, transpose back)
                pairs = np.unique(combined.T, axis=0)
                ei_np = pairs.T
            else:
                ei_np = corr_frozen_np
        else:
            ei_np = corr_frozen_np
        out[d] = torch.from_numpy(ei_np).long()
    return out


# ══════════════════════════════════════════════════════════════
# E3 TRAINING: SAGE-Mean with per-day edge_index
# ══════════════════════════════════════════════════════════════

def train_sage_per_day_edges(features_t: torch.Tensor, labels_t: torch.Tensor,
                              label_valid_t: torch.Tensor,
                              train_days: np.ndarray, val_days: np.ndarray, test_days: np.ndarray,
                              per_day_edges_cpu: dict[int, torch.Tensor],
                              num_days: int, num_stocks: int,
                              seed: int, device: torch.device):
    """SAGE-Mean training with per-day edge_index lookup. Mirrors train_nn (E1) but
    switches edge_index per day instead of using one frozen snapshot.
    """
    set_seed(seed)
    in_ch = features_t.shape[-1]
    model = make_nn_model('SAGE-Mean', in_ch, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=NN_HPARAMS['lr'],
                                 weight_decay=NN_HPARAMS['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-5)

    # Pre-move edges to device — but only for days actually used (train ∪ val ∪ test)
    used_days = set(int(d) for d in train_days) | set(int(d) for d in val_days) | set(int(d) for d in test_days)
    per_day_edges = {d: per_day_edges_cpu[d].to(device) for d in used_days if d in per_day_edges_cpu}

    best_val = float('inf')
    best_state = None
    no_improve = 0
    epochs_run = 0

    for epoch in range(NN_HPARAMS['epochs']):
        epochs_run = epoch + 1
        model.train()
        optimizer.zero_grad()
        accum_steps = 0
        day_order = train_days[np.random.permutation(len(train_days))]

        for step, d in enumerate(day_order):
            d_int = int(d)
            ei = per_day_edges.get(d_int)
            if ei is None:
                continue
            x = features_t[d_int].to(device)
            pred = model(x, ei)
            mask = label_valid_t[d_int].to(device)
            if mask.sum() < 10:
                continue
            target = labels_t[d_int].to(device)
            loss = F.mse_loss(pred[mask], target[mask])
            (loss / NN_HPARAMS['grad_accum']).backward()
            accum_steps += 1
            if accum_steps >= NN_HPARAMS['grad_accum'] or step == len(day_order) - 1:
                if 0 < accum_steps < NN_HPARAMS['grad_accum']:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(NN_HPARAMS['grad_accum'] / accum_steps)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                accum_steps = 0

        model.eval()
        v_loss, v_cnt = 0.0, 0
        with torch.no_grad():
            for d in val_days:
                d_int = int(d)
                ei = per_day_edges.get(d_int)
                if ei is None:
                    continue
                x = features_t[d_int].to(device)
                pred = model(x, ei)
                mask = label_valid_t[d_int].to(device)
                if mask.sum() < 10:
                    continue
                v_loss += F.mse_loss(pred[mask], labels_t[d_int].to(device)[mask]).item()
                v_cnt += 1

        avg_val = v_loss / max(v_cnt, 1)
        scheduler.step(avg_val)
        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= NN_HPARAMS['patience']:
            break

    assert best_state is not None, f"SAGE-Mean seed={seed} produced no valid state"
    model.load_state_dict(best_state)
    model.to(device).eval()

    preds = np.zeros((num_days, num_stocks), dtype=np.float32)
    with torch.no_grad():
        for d in test_days:
            d_int = int(d)
            ei = per_day_edges.get(d_int)
            if ei is None:
                continue
            x = features_t[d_int].to(device)
            preds[d_int] = model(x, ei).cpu().numpy()

    del model, per_day_edges
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return preds, {'best_val_loss': best_val, 'epochs_run': epochs_run}


# ══════════════════════════════════════════════════════════════
# MANIFEST + CSV (E3-specific, locked column schemas)
# ══════════════════════════════════════════════════════════════

def init_csv_files() -> None:
    if not os.path.exists(RESULTS_CSV):
        pd.DataFrame(columns=RESULTS_COLUMNS).to_csv(RESULTS_CSV, index=False)
    if not os.path.exists(MANIFEST_CSV):
        pd.DataFrame(columns=MANIFEST_COLUMNS).to_csv(MANIFEST_CSV, index=False)


def append_results(row: dict) -> None:
    df = pd.DataFrame([row], columns=RESULTS_COLUMNS)
    df.to_csv(RESULTS_CSV, mode='a', header=False, index=False)


def append_manifest(row: dict) -> None:
    df = pd.DataFrame([row], columns=MANIFEST_COLUMNS)
    df.to_csv(MANIFEST_CSV, mode='a', header=False, index=False)


def load_manifest_done() -> set[tuple[int, int]]:
    """CODEX-CR-E3RUN-A-02 fix: return set of (fold, seed) pairs, not cell_id.
    cell_id is derived from (fold, seed) but if the formula ever changes, stale
    manifests with old cell_ids would silently skip the wrong cell. (fold, seed)
    is the canonical identity. Resume check below uses this pair.
    """
    if not os.path.exists(MANIFEST_CSV):
        return set()
    df = pd.read_csv(MANIFEST_CSV)
    if 'status' not in df.columns:
        return set()
    done_rows = df[df['status'] == 'done']
    return set(zip(done_rows['fold'].astype(int).tolist(),
                    done_rows['seed'].astype(int).tolist()))


def write_meta_json() -> None:
    meta = {
        'experiment_id': 'storya_e3_news_edge_v3',
        'plan_ref': 'plan §1.2 (E3 news-as-edge) + §1.4 (statistics)',
        'horizon_days': HORIZON,
        'canonical_seeds': CANONICAL_SEEDS,
        'model': MODEL,
        'universe': UNIVERSE,
        'edge_config': EDGE_CONFIG,
        'news_lookback_trading_days': NEWS_LOOKBACK_TRADING_DAYS,
        'news_lookback_calendar_days': NEWS_LOOKBACK_CALENDAR_DAYS,
        'pit_cutoff_spec': 'NYSE session_close(t-1) in UTC, DST-aware (pandas_market_calendars)',
        'pit_schema_ref': 'experiments/storya_e3_news_edge/news_edge_source_schema.md (v2, Codex D-03 fix)',
        'baseline_cells_reused_from': 'experiments/storya_e1_anchor/results.csv (Universe B, SAGE-Mean — 50 cells)',
        'cost_ladder': {
            'levels_bps': list(COST_LEVELS_BPS),
            'convention': COST_CONVENTION,
            'turnover_definition': 'L1-norm: turnover_L1 = sum_i|p_i(t) - p_i(t-1)|; at full L-S rotation = 4',
            'relation_to_archived': 'L1 = 2 * one_side',
            'cost_formula': 'cost_per_period(t, c_bps) = turnover_L1(t) * c_bps * 1e-4',
            'annualization': 'sqrt(252/horizon)',
        },
    }
    with open(META_JSON, 'w') as f:
        json.dump(meta, f, indent=2)


def write_hp_grid_json() -> None:
    """Pre-register hyperparameters (plan §1.7 LOCKED — same as E1 SAGE-Mean)."""
    hp = {
        'model': MODEL,
        'sage_mean': NN_HPARAMS,
        'graph': GRAPH_HPARAMS,
        'edge_config': EDGE_CONFIG,
        'news_lookback_calendar_days': NEWS_LOOKBACK_CALENDAR_DAYS,
    }
    with open(HP_GRID_JSON, 'w') as f:
        json.dump(hp, f, indent=2)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Skip cells already in manifest with status=done (default: True)')
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    parser.add_argument('--smoke', action='store_true',
                        help='Smoke mode: 1 seed × 1 fold only')
    parser.add_argument('--news-cache', default=NEWS_SNAPSHOT_CACHE,
                        help='Path to cached news snapshots (.npz). Built once and reused across seeds.')
    args = parser.parse_args()

    # ── Setup ──
    print(f"Working dir: {os.getcwd()}")
    device = get_device()
    print(f"Device: {device.type}")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PER_DAY_IC_DIR, exist_ok=True)
    init_csv_files()
    write_hp_grid_json()
    write_meta_json()

    assert_cell_id_e3_injective()

    # ── Load core data (prices, returns, dates) via E1 anchor's loader ──
    print("\nLoading core data ...")
    core = load_core_data()
    prices       = core['prices']
    returns      = core['returns']
    trading_dates = core['all_dates']
    num_days     = core['num_days']
    num_stocks   = core['num_stocks']
    print(f"  Loaded: {num_stocks} stocks × {num_days} days, "
          f"date range {trading_dates.min().date()} → {trading_dates.max().date()}")

    assert_purge_no_leak(trading_dates, horizon=HORIZON)

    labels_np, label_valid_np = build_labels(prices, horizon=HORIZON)
    print(f"Labels: h={HORIZON}d, {label_valid_np.sum():,} valid (day, stock) entries")

    # Universe B features (10-dim, hand-crafted)
    features_np, feature_names = build_universe_B(prices, returns)
    print(f"Universe B features built: {features_np.shape}")

    # Correlation graph snapshots (same as E1; returns 3-tuple)
    corr_snapshots, _day_to_si, snapshot_points = build_correlation_snapshots(returns, num_days)
    print(f"Correlation graph: {len(corr_snapshots)} snapshots ready")

    # ── Load news edge source + build per-day PIT-safe snapshots ──
    print("\nLoading news edge source ...")
    news_df = load_news_edge_source()
    print(f"  News edge source: {len(news_df):,} articles, "
          f"{(news_df['n_tickers'] >= 2).sum():,} with ≥2 tickers")

    # Build ticker_to_idx aligned with E1's load_core_data result (price ∩ sector universe)
    ticker_to_idx = core['ticker_to_id']

    # Per-day PIT-safe news edge snapshots (built once, cached, reused across seeds).
    # Cache file contains both edges (per-day E_t shape (2,E)) and an article-count array
    # (length num_days, dtype int32) at key '__article_counts__'.
    if os.path.exists(args.news_cache):
        print(f"Loading cached news snapshots: {args.news_cache} ...")
        cache = np.load(args.news_cache, allow_pickle=True)
        if '__article_counts__' in cache.files:
            article_count_arr = cache['__article_counts__']
            article_counts = {i: int(c) for i, c in enumerate(article_count_arr) if c >= 0}
            snapshot_keys = [k for k in cache.files if k != '__article_counts__']
        else:
            # Backward-compat: old cache without article counts → set to 0 (will be re-written)
            article_counts = {}
            snapshot_keys = list(cache.files)
        news_snapshots = {int(k): cache[k] for k in snapshot_keys}
        print(f"  loaded {len(news_snapshots)} snapshots, {len(article_counts)} article-count entries")
    else:
        news_snapshots, article_counts = build_per_day_news_edges(news_df, trading_dates, ticker_to_idx)
        # Cache for reuse across runs (saves ~5-15 min per re-run)
        print(f"Caching news snapshots to {args.news_cache} ...")
        # Build a dense article_counts array (length=num_days, sentinel -1 = "not computed")
        ac_arr = np.full(len(trading_dates), -1, dtype=np.int32)
        for d, c in article_counts.items():
            ac_arr[d] = c
        np.savez_compressed(args.news_cache,
                             __article_counts__=ac_arr,
                             **{str(k): v for k, v in news_snapshots.items()})

    # ── Prepare resumable iteration ──
    done = load_manifest_done() if args.resume else set()
    print(f"\nResume mode {'ON' if args.resume else 'OFF'}: "
          f"{len(done)} cells already completed; will skip them")

    if args.smoke:
        seeds = CANONICAL_SEEDS[:1]
        folds_to_run = [WALK_FORWARD_FOLDS[0]]
        print("SMOKE mode: 1 seed × 1 fold")
    else:
        seeds = CANONICAL_SEEDS
        folds_to_run = WALK_FORWARD_FOLDS

    planned = []
    for fold_cfg in folds_to_run:
        fold_idx = fold_cfg['id']
        for seed_idx, seed in enumerate(seeds):
            cid = cell_id_e3(fold_idx, seed_idx)
            # CR-E3RUN-A-02 fix: skip based on (fold, seed) identity, not cell_id integer
            if (fold_idx, seed) in done:
                continue
            planned.append((cid, fold_cfg, seed_idx, seed))
    print(f"Planned cells (after resume filter): {len(planned)}")

    # ── Per-fold preprocessing (winsor + standardize on TRAIN only; reuse across seeds) ──
    fold_cache = {}
    for fold_cfg in folds_to_run:
        fold_idx = fold_cfg['id']
        train_days, val_days, test_days = create_fold_masks(fold_cfg, trading_dates, horizon=HORIZON)
        print(f"\n[fold {fold_idx} ({fold_cfg['desc']})] "
              f"train={len(train_days)}d (purged {HORIZON}d), val={len(val_days)}d, test={len(test_days)}d")
        # Winsorize then standardize, train-only
        feats_winsor = winsorize_train_only(features_np, train_days)
        feats_std = standardize_train_only(feats_winsor, train_days)
        # Frozen correlation snapshot for this fold (same as E1)
        frozen_si = get_frozen_snapshot_idx(train_days[-1], snapshot_points)
        corr_frozen_tensor = corr_snapshots[frozen_si]
        corr_frozen_np = corr_frozen_tensor.numpy() if hasattr(corr_frozen_tensor, 'numpy') else np.asarray(corr_frozen_tensor)
        # Build per-day UNION (corr_frozen ∪ news_per_day) edges for all (train, val, test) days
        all_days = np.concatenate([train_days, val_days, test_days])
        per_day_edges_cpu = union_edges_per_day(corr_frozen_np, news_snapshots, all_days)
        # CODEX-CR-E3RUN-A-03 (CONCERN) fix: pre-validate every needed day has an edge tensor,
        # so missing-key silent-zero predictions cannot happen at training/inference time.
        missing_days = [int(d) for d in all_days if int(d) not in per_day_edges_cpu]
        if missing_days:
            raise RuntimeError(
                f"Fold {fold_idx}: {len(missing_days)} train/val/test days have no edge tensor "
                f"(union_edges_per_day did not populate them). First 5: {missing_days[:5]}"
            )
        # Stats: avg #news edges + avg #articles per test day  (CR-E3RUN-A-01 fix for articles)
        test_news_edges = [news_snapshots.get(int(d), np.zeros((2, 0), dtype=np.int64)).shape[1] // 2  # half for symmetry
                           for d in test_days]
        avg_news_edges = float(np.mean(test_news_edges)) if test_news_edges else 0.0
        test_article_counts = [article_counts.get(int(d), 0) for d in test_days]
        avg_news_articles = float(np.mean(test_article_counts)) if test_article_counts else 0.0
        print(f"  frozen_si={frozen_si}, corr edges={corr_frozen_np.shape[1]//2}, "
              f"avg news edges/test_day={avg_news_edges:.0f}, "
              f"avg news articles/test_day={avg_news_articles:.0f}")

        fold_cache[fold_idx] = {
            'features_t': torch.from_numpy(feats_std).float(),
            'labels_t': torch.from_numpy(labels_np).float(),
            'label_valid_t': torch.from_numpy(label_valid_np),
            'train_days': train_days,
            'val_days': val_days,
            'test_days': test_days,
            'per_day_edges_cpu': per_day_edges_cpu,
            'avg_news_edges': avg_news_edges,
            'avg_news_articles': avg_news_articles,
        }

    # ── Run cells ──
    for cid, fold_cfg, seed_idx, seed in planned:
        fold_idx = fold_cfg['id']
        cache = fold_cache[fold_idx]
        start_ts = time.strftime('%Y-%m-%dT%H:%M:%S')
        t_cell = time.time()
        print(f"\n  [cid={cid:03d}] U{UNIVERSE}  {MODEL}  edge={EDGE_CONFIG}  seed={seed}  fold={fold_idx} ...")

        append_manifest({'cell_id': cid, 'fold': fold_idx, 'seed': seed,
                          'status': 'running', 'start_ts': start_ts,
                          'end_ts': '', 'wall_time_sec': 0, 'err': ''})

        try:
            preds, train_info = train_sage_per_day_edges(
                cache['features_t'], cache['labels_t'], cache['label_valid_t'],
                cache['train_days'], cache['val_days'], cache['test_days'],
                cache['per_day_edges_cpu'],
                num_days, num_stocks, seed, device,
            )

            ic_arr = compute_daily_ic(preds, cache['test_days'], labels_np, label_valid_np)
            sh = compute_cost_ladder_sharpe(
                preds, cache['test_days'], prices, label_valid_np,
                num_stocks, num_days, horizon=HORIZON,
                cost_levels_bps=COST_LEVELS_BPS,
            )

            wall = round(time.time() - t_cell, 1)
            ic_path = f'{PER_DAY_IC_DIR}/{UNIVERSE}_{MODEL}_s{seed}_f{fold_idx}.npy'
            np.save(ic_path, ic_arr)

            row = {
                'cell_id': cid, 'universe': UNIVERSE, 'model': MODEL,
                'edge_config': EDGE_CONFIG, 'seed': seed, 'fold': fold_idx,
                'test_period': fold_cfg['desc'],
                'IC_mean': round(float(ic_arr.mean()), 6) if len(ic_arr) else 0.0,
                'IC_std': round(float(ic_arr.std()), 6) if len(ic_arr) else 0.0,
                'n_test_days': int(len(ic_arr)),
                'Sharpe_gross': round(sh['Sharpe_gross'], 4),
                **{k: round(v, 4) for k, v in sh.items() if k.startswith('Sharpe_net_')},
                'mean_turnover_L1': round(sh['mean_turnover_L1'], 4),
                'n_periods': int(sh['n_periods']),
                'best_val_loss': round(float(train_info['best_val_loss']), 6),
                'epochs_run': int(train_info['epochs_run']),
                'wall_time_sec': wall,
                'converged_flag': 1,
                'cost_convention': COST_CONVENTION,
                'n_news_edges_avg': round(cache['avg_news_edges'], 1),
                'n_news_articles_avg': round(cache['avg_news_articles'], 1),  # CR-E3RUN-A-01 fix
            }
            append_results(row)

            end_ts = time.strftime('%Y-%m-%dT%H:%M:%S')
            append_manifest({'cell_id': cid, 'fold': fold_idx, 'seed': seed,
                              'status': 'done', 'start_ts': start_ts, 'end_ts': end_ts,
                              'wall_time_sec': wall, 'err': ''})
            print(f"  done: IC={row['IC_mean']:+.4f}, Sh_g={row['Sharpe_gross']:.3f}, "
                  f"Sh_n@10={row['Sharpe_net_10bps']:.3f}, wall={wall}s")

        except Exception as e:
            wall = round(time.time() - t_cell, 1)
            end_ts = time.strftime('%Y-%m-%dT%H:%M:%S')
            err_msg = f"{type(e).__name__}: {e}"
            append_manifest({'cell_id': cid, 'fold': fold_idx, 'seed': seed,
                              'status': 'failed', 'start_ts': start_ts, 'end_ts': end_ts,
                              'wall_time_sec': wall, 'err': err_msg[:500]})
            print(f"  FAILED ({wall}s): {err_msg}", file=sys.stderr)
            import traceback; traceback.print_exc()

    print(f"\n✓ E3 run complete. See {RESULTS_CSV} + {MANIFEST_CSV}")


if __name__ == '__main__':
    main()
