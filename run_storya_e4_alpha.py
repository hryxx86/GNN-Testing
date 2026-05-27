#!/usr/bin/env python
"""run_storya_e4_alpha.py — Story A E4-α (clean edge ablation).

Plan §1.3: fixed SAGE-Mean architecture, vary 4 edge configs to isolate edge
contribution from architecture confounding. Replaces v2 HGT 21d rerun.

| config | edges | status | new cells |
|--------|-------|--------|-----------|
| α1: corr-only       | correlation                                | REUSED from E1-B SAGE-Mean | 0 |
| α2: corr+sector     | correlation ∪ sector (GICS 11, static)     | NEW                       | 50 |
| α3: corr+news       | correlation ∪ news cooccurrence (PIT)      | REUSED from E3            | 0 |
| α4: corr+sector+news| α1 ∪ α2 ∪ α3                                | NEW                       | 50 |

Total NEW cells: 100 = 2 configs × 10 seeds × 5 folds.
α1 + α3 reused from other experiments at analysis time (E6 reads all results.csv files).

Fixed (per plan §1.3):
- model: SAGE-Mean (only)
- universe: Universe B (10-dim hc) only
- horizon: 21d
- seeds: canonical 10
- folds: same 5-fold walk-forward as E1
- hyperparameters: locked per §1.7 (same as E1 SAGE-Mean)

PIT contract for news edges: same as E3 (NYSE session_close(t-1) in UTC, per
news_edge_source_schema.md v2). Sector edges are STATIC (no PIT issue).

Reuses (via import):
- E1 anchor: constants, data loaders, model builder, training utilities, metrics
- E3 runner: load_news_edge_source, build_per_day_news_edges, union_edges_per_day, train_sage_per_day_edges
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

if platform.system() == 'Darwin':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Reuse from E1 anchor
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

# Reuse E3's news edge machinery + per-day training
from run_storya_e3_news_edge import (
    load_news_edge_source, build_per_day_news_edges,
    train_sage_per_day_edges,
    NEWS_LOOKBACK_CALENDAR_DAYS,
)


# ══════════════════════════════════════════════════════════════
# CONFIG — E4-α
# ══════════════════════════════════════════════════════════════

MODEL = 'SAGE-Mean'
UNIVERSE = 'B'

# Edge configs to RUN (α1 and α3 reused from E1/E3, not re-run here)
NEW_EDGE_CONFIGS = ['corr+sector', 'corr+sector+news']

# Path setup
PATHS = {
    'prices': 'data/reference/sp500_5y_prices.csv',
    'sectors': 'data/reference/sp500_sectors.csv',
    'phase5_npy': 'data/reference/sp500_5y_phase5_features.npy',
    'news_edge_source': 'data/fullscale/sp500_news_edge_source.parquet',
}
OUT_DIR = 'experiments/storya_e4_alpha'
RESULTS_CSV = f'{OUT_DIR}/results.csv'
MANIFEST_CSV = f'{OUT_DIR}/manifest.csv'
PER_DAY_IC_DIR = f'{OUT_DIR}/per_day_ic'
HP_GRID_JSON = f'{OUT_DIR}/hp_grid.json'
META_JSON = f'{OUT_DIR}/_meta.json'
NEWS_SNAPSHOT_CACHE = f'experiments/storya_e3_news_edge/news_snapshots_cache.npz'  # share E3 cache

RESULTS_COLUMNS = (
    ['cell_id', 'universe', 'model', 'edge_config', 'seed', 'fold', 'test_period',
     'IC_mean', 'IC_std', 'n_test_days', 'Sharpe_gross']
    + [f'Sharpe_net_{c}bps' for c in COST_LEVELS_BPS]
    + ['mean_turnover_L1', 'n_periods', 'best_val_loss', 'epochs_run',
       'wall_time_sec', 'converged_flag', 'cost_convention',
       'n_corr_edges', 'n_sector_edges', 'n_news_edges_avg']
)
MANIFEST_COLUMNS = ['cell_id', 'edge_config', 'fold', 'seed',
                    'status', 'start_ts', 'end_ts', 'wall_time_sec', 'err']


def cell_id_e4(config_idx: int, fold_idx: int, seed_idx: int) -> int:
    """E4 cell_id: range [0, 99]. Injective by radix (2 configs × 5 folds × 10 seeds = 100)."""
    return config_idx * 50 + fold_idx * 10 + seed_idx


def assert_cell_id_e4_injective() -> None:
    seen = set()
    for c in range(2):
        for f in range(5):
            for s in range(10):
                cid = cell_id_e4(c, f, s)
                assert cid not in seen, f"cell_id collision at c={c},f={f},s={s}"
                seen.add(cid)
    assert max(seen) == 99 and min(seen) == 0 and len(seen) == 100
    print(f"✓ E4 cell_id formula injective, range [0, 99], n=100 cells")


# ══════════════════════════════════════════════════════════════
# SECTOR EDGE BUILDER (static, no PIT issue)
# ══════════════════════════════════════════════════════════════

def build_sector_edges(sectors_csv: str, ticker_to_idx: dict[str, int]) -> np.ndarray:
    """For each pair of stocks in the same GICS sector, add an undirected edge.

    Static (no time variation). Returns symmetric edge_index shape (2, E).
    Edge weights are not used (SAGE-Mean is unweighted), so we only return topology.
    """
    df = pd.read_csv(sectors_csv)
    # Locate the column names
    sector_col = [c for c in df.columns if 'sector' in c.lower()][0]
    ticker_col = [c for c in df.columns if c != sector_col][0]
    # Build sector → list of ticker idx
    sector_to_idxs = defaultdict(list)
    for _, row in df.iterrows():
        t = str(row[ticker_col])
        sec = row[sector_col]
        if t in ticker_to_idx and pd.notna(sec):
            sector_to_idxs[sec].append(ticker_to_idx[t])
    # Build edges
    edge_pairs = []
    for sec, idxs in sector_to_idxs.items():
        idxs = sorted(set(idxs))
        for i, j in combinations(idxs, 2):
            edge_pairs.append((i, j))
    if not edge_pairs:
        return np.zeros((2, 0), dtype=np.int64)
    directed = np.array(edge_pairs, dtype=np.int64).T  # (2, E)
    # Symmetrize
    sym = np.concatenate([directed, directed[[1, 0], :]], axis=1)
    return sym


# ══════════════════════════════════════════════════════════════
# UNION (corr_frozen ∪ sector_static [∪ news_per_day])
# ══════════════════════════════════════════════════════════════

def union_static_edges(corr_frozen_np: np.ndarray, sector_np: np.ndarray) -> np.ndarray:
    """Union + dedupe two static edge sets, return (2, E_total)."""
    if sector_np.shape[1] == 0:
        return corr_frozen_np
    combined = np.concatenate([corr_frozen_np, sector_np], axis=1)
    pairs = np.unique(combined.T, axis=0)
    return pairs.T


def union_edges_per_day_e4(static_combined: np.ndarray,
                            news_snapshots: Optional[dict[int, np.ndarray]],
                            day_indices: np.ndarray) -> dict[int, torch.Tensor]:
    """For α2 (sector but no news): news_snapshots=None → same static_combined each day.
    For α4 (sector + news): news_snapshots provided → per-day union.
    """
    out = {}
    for d in day_indices:
        d = int(d)
        if news_snapshots is None:
            ei_np = static_combined
        else:
            news = news_snapshots.get(d, np.zeros((2, 0), dtype=np.int64))
            if news.shape[1] > 0:
                combined = np.concatenate([static_combined, news], axis=1)
                pairs = np.unique(combined.T, axis=0)
                ei_np = pairs.T
            else:
                ei_np = static_combined
        out[d] = torch.from_numpy(ei_np).long()
    return out


# ══════════════════════════════════════════════════════════════
# CSV / MANIFEST  (E4-specific)
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


def load_manifest_done() -> set[tuple[str, int, int]]:
    """Mirror of E3's CR-E3RUN-A-02 fix: identity is (edge_config, fold, seed), NOT cell_id,
    so a future cell_id formula change cannot silently skip the wrong cell."""
    if not os.path.exists(MANIFEST_CSV):
        return set()
    df = pd.read_csv(MANIFEST_CSV)
    if 'status' not in df.columns:
        return set()
    done_rows = df[df['status'] == 'done']
    return set(zip(done_rows['edge_config'].astype(str).tolist(),
                    done_rows['fold'].astype(int).tolist(),
                    done_rows['seed'].astype(int).tolist()))


def write_meta_json() -> None:
    meta = {
        'experiment_id': 'storya_e4_alpha_v3',
        'plan_ref': 'plan §1.3 (E4-α clean edge ablation; SAGE-Mean fixed, vary edges only)',
        'horizon_days': HORIZON,
        'canonical_seeds': CANONICAL_SEEDS,
        'model': MODEL,
        'universe': UNIVERSE,
        'edge_configs_run_here': NEW_EDGE_CONFIGS,
        'edge_configs_reused_elsewhere': {
            'corr_only_alpha1': 'experiments/storya_e1_anchor/results.csv (SAGE-Mean, Universe B)',
            'corr_news_alpha3': 'experiments/storya_e3_news_edge/results.csv',
        },
        'sector_source': 'data/reference/sp500_sectors.csv (GICS 11)',
        'news_lookback_calendar_days': NEWS_LOOKBACK_CALENDAR_DAYS,
        'cost_ladder': {
            'levels_bps': list(COST_LEVELS_BPS),
            'convention': COST_CONVENTION,
        },
    }
    with open(META_JSON, 'w') as f:
        json.dump(meta, f, indent=2)


def write_hp_grid_json() -> None:
    hp = {
        'model': MODEL,
        'sage_mean': NN_HPARAMS,
        'graph': GRAPH_HPARAMS,
        'edge_configs': NEW_EDGE_CONFIGS,
    }
    with open(HP_GRID_JSON, 'w') as f:
        json.dump(hp, f, indent=2)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    parser.add_argument('--smoke', action='store_true',
                        help='Smoke mode: 1 seed × 1 fold × 1 config (the first NEW config)')
    parser.add_argument('--configs', nargs='*', default=NEW_EDGE_CONFIGS,
                        help=f'Subset of edge configs to run (default: {NEW_EDGE_CONFIGS})')
    parser.add_argument('--news-cache', default=NEWS_SNAPSHOT_CACHE,
                        help='Path to news snapshots cache (shared with E3)')
    args = parser.parse_args()

    print(f"Working dir: {os.getcwd()}")
    device = get_device()
    print(f"Device: {device.type}")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PER_DAY_IC_DIR, exist_ok=True)
    init_csv_files()
    write_hp_grid_json()
    write_meta_json()

    assert_cell_id_e4_injective()

    # ── Core data ──
    print("\nLoading core data ...")
    core = load_core_data()
    prices       = core['prices']
    returns      = core['returns']
    trading_dates = core['all_dates']
    num_days     = core['num_days']
    num_stocks   = core['num_stocks']
    ticker_to_idx = core['ticker_to_id']
    print(f"  Loaded: {num_stocks} stocks × {num_days} days")

    assert_purge_no_leak(trading_dates, horizon=HORIZON)
    labels_np, label_valid_np = build_labels(prices, horizon=HORIZON)
    print(f"Labels: h={HORIZON}d, {label_valid_np.sum():,} valid")

    features_np, feature_names = build_universe_B(prices, returns)
    print(f"Universe B features built: {features_np.shape}")

    corr_snapshots, _day_to_si, snapshot_points = build_correlation_snapshots(returns, num_days)
    print(f"Correlation graph: {len(corr_snapshots)} snapshots ready")

    # ── Build static sector edges (once) ──
    sector_np = build_sector_edges(PATHS['sectors'], ticker_to_idx)
    print(f"Sector edges: {sector_np.shape[1] // 2} undirected pairs "
          f"(symmetric edge_index shape {sector_np.shape})")

    # ── News snapshots (needed for α4 only) ──
    news_snapshots = None
    if 'corr+sector+news' in args.configs:
        if os.path.exists(args.news_cache):
            print(f"Loading cached news snapshots: {args.news_cache} ...")
            cache = np.load(args.news_cache, allow_pickle=True)
            # CR-E3RUN-A-01 fix propagation: ignore '__article_counts__' sidecar (E4 doesn't use it)
            snapshot_keys = [k for k in cache.files if k != '__article_counts__']
            news_snapshots = {int(k): cache[k] for k in snapshot_keys}
            print(f"  loaded {len(news_snapshots)} snapshots")
        else:
            print("Building news snapshots (no cache found) ...")
            news_df = load_news_edge_source()
            # build_per_day_news_edges now returns (snapshots, article_counts) per CR-E3RUN-A-01
            news_snapshots, article_counts = build_per_day_news_edges(news_df, trading_dates, ticker_to_idx)
            os.makedirs(os.path.dirname(args.news_cache), exist_ok=True)
            ac_arr = np.full(len(trading_dates), -1, dtype=np.int32)
            for d, c in article_counts.items():
                ac_arr[d] = c
            np.savez_compressed(args.news_cache,
                                 __article_counts__=ac_arr,
                                 **{str(k): v for k, v in news_snapshots.items()})
            print(f"  cached to {args.news_cache}")

    # ── Plan cells ──
    done = load_manifest_done() if args.resume else set()
    print(f"\nResume mode {'ON' if args.resume else 'OFF'}: "
          f"{len(done)} cells already completed; will skip them")

    if args.smoke:
        configs = args.configs[:1]
        seeds = CANONICAL_SEEDS[:1]
        folds_to_run = [WALK_FORWARD_FOLDS[0]]
        print(f"SMOKE mode: 1 config ({configs[0]}) × 1 seed × 1 fold")
    else:
        configs = args.configs
        seeds = CANONICAL_SEEDS
        folds_to_run = WALK_FORWARD_FOLDS

    planned = []
    for config_idx, edge_config in enumerate(NEW_EDGE_CONFIGS):
        if edge_config not in configs:
            continue
        for fold_cfg in folds_to_run:
            fold_idx = fold_cfg['id']
            for seed_idx, seed in enumerate(seeds):
                cid = cell_id_e4(config_idx, fold_idx, seed_idx)
                # CR-E3RUN-A-02 fix propagation: skip by (edge_config, fold, seed) tuple,
                # not by cell_id integer (which is formula-dependent)
                if (edge_config, fold_idx, seed) in done:
                    continue
                planned.append((cid, edge_config, config_idx, fold_cfg, seed_idx, seed))
    print(f"Planned cells (after resume filter): {len(planned)}")

    # ── Per-fold preprocessing (winsor + standardize on TRAIN; reuse across seeds + configs) ──
    fold_cache = {}
    for fold_cfg in folds_to_run:
        fold_idx = fold_cfg['id']
        train_days, val_days, test_days = create_fold_masks(fold_cfg, trading_dates, horizon=HORIZON)
        print(f"\n[fold {fold_idx} ({fold_cfg['desc']})] "
              f"train={len(train_days)}d (purged {HORIZON}d), val={len(val_days)}d, test={len(test_days)}d")
        feats_winsor = winsorize_train_only(features_np, train_days)
        feats_std = standardize_train_only(feats_winsor, train_days)
        frozen_si = get_frozen_snapshot_idx(train_days[-1], snapshot_points)
        corr_frozen_tensor = corr_snapshots[frozen_si]
        corr_frozen_np = corr_frozen_tensor.numpy() if hasattr(corr_frozen_tensor, 'numpy') else np.asarray(corr_frozen_tensor)

        # corr ∪ sector (static union — used by α2 and as base for α4)
        static_combined = union_static_edges(corr_frozen_np, sector_np)
        n_corr = corr_frozen_np.shape[1] // 2
        n_sector = sector_np.shape[1] // 2
        n_combined = static_combined.shape[1] // 2
        print(f"  frozen_si={frozen_si}, corr={n_corr}, sector={n_sector}, "
              f"static_union(after dedupe)={n_combined}")

        fold_cache[fold_idx] = {
            'features_t': torch.from_numpy(feats_std).float(),
            'labels_t': torch.from_numpy(labels_np).float(),
            'label_valid_t': torch.from_numpy(label_valid_np),
            'train_days': train_days,
            'val_days': val_days,
            'test_days': test_days,
            'corr_frozen_np': corr_frozen_np,
            'sector_np': sector_np,
            'static_combined': static_combined,
            'n_corr_edges': n_corr,
            'n_sector_edges': n_sector,
        }

    # ── Run cells ──
    for cid, edge_config, config_idx, fold_cfg, seed_idx, seed in planned:
        fold_idx = fold_cfg['id']
        cache = fold_cache[fold_idx]
        start_ts = time.strftime('%Y-%m-%dT%H:%M:%S')
        t_cell = time.time()
        print(f"\n  [cid={cid:03d}] U{UNIVERSE}  {MODEL}  edge={edge_config}  seed={seed}  fold={fold_idx} ...")

        append_manifest({'cell_id': cid, 'edge_config': edge_config, 'fold': fold_idx, 'seed': seed,
                          'status': 'running', 'start_ts': start_ts,
                          'end_ts': '', 'wall_time_sec': 0, 'err': ''})

        try:
            all_days = np.concatenate([cache['train_days'], cache['val_days'], cache['test_days']])
            if edge_config == 'corr+sector':
                per_day_edges_cpu = union_edges_per_day_e4(cache['static_combined'], None, all_days)
                avg_news_edges = 0.0
            elif edge_config == 'corr+sector+news':
                if news_snapshots is None:
                    raise RuntimeError("news_snapshots not loaded; needed for corr+sector+news config")
                per_day_edges_cpu = union_edges_per_day_e4(cache['static_combined'], news_snapshots, all_days)
                test_news_edges = [news_snapshots.get(int(d), np.zeros((2, 0), dtype=np.int64)).shape[1] // 2
                                   for d in cache['test_days']]
                avg_news_edges = float(np.mean(test_news_edges)) if test_news_edges else 0.0
            else:
                raise ValueError(f"Unknown edge_config: {edge_config}")

            preds, train_info = train_sage_per_day_edges(
                cache['features_t'], cache['labels_t'], cache['label_valid_t'],
                cache['train_days'], cache['val_days'], cache['test_days'],
                per_day_edges_cpu,
                num_days, num_stocks, seed, device,
            )

            ic_arr = compute_daily_ic(preds, cache['test_days'], labels_np, label_valid_np)
            sh = compute_cost_ladder_sharpe(
                preds, cache['test_days'], prices, label_valid_np,
                num_stocks, num_days, horizon=HORIZON,
                cost_levels_bps=COST_LEVELS_BPS,
            )

            wall = round(time.time() - t_cell, 1)
            ic_path = f'{PER_DAY_IC_DIR}/{edge_config.replace("+","_")}_{MODEL}_s{seed}_f{fold_idx}.npy'
            np.save(ic_path, ic_arr)

            row = {
                'cell_id': cid, 'universe': UNIVERSE, 'model': MODEL,
                'edge_config': edge_config, 'seed': seed, 'fold': fold_idx,
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
                'n_corr_edges': cache['n_corr_edges'],
                'n_sector_edges': cache['n_sector_edges'],
                'n_news_edges_avg': round(avg_news_edges, 1),
            }
            append_results(row)

            end_ts = time.strftime('%Y-%m-%dT%H:%M:%S')
            append_manifest({'cell_id': cid, 'edge_config': edge_config, 'fold': fold_idx, 'seed': seed,
                              'status': 'done', 'start_ts': start_ts, 'end_ts': end_ts,
                              'wall_time_sec': wall, 'err': ''})
            print(f"  done: IC={row['IC_mean']:+.4f}, Sh_g={row['Sharpe_gross']:.3f}, "
                  f"Sh_n@10={row['Sharpe_net_10bps']:.3f}, wall={wall}s")

        except Exception as e:
            wall = round(time.time() - t_cell, 1)
            end_ts = time.strftime('%Y-%m-%dT%H:%M:%S')
            err_msg = f"{type(e).__name__}: {e}"
            append_manifest({'cell_id': cid, 'edge_config': edge_config, 'fold': fold_idx, 'seed': seed,
                              'status': 'failed', 'start_ts': start_ts, 'end_ts': end_ts,
                              'wall_time_sec': wall, 'err': err_msg[:500]})
            print(f"  FAILED ({wall}s): {err_msg}", file=sys.stderr)
            import traceback; traceback.print_exc()

    print(f"\n✓ E4-α run complete. See {RESULTS_CSV} + {MANIFEST_CSV}")


if __name__ == '__main__':
    main()
