#!/usr/bin/env python
"""Plan AAA v1 — 168-feature grouped permutation Δ-IC ranking.

168 features = 158 Alpha158 (qlib, raw _raw.npy) + 10 hand-curated
(inherited from Plan Z++ Part A code path). Methodology verbatim Part A:
grouped cross-sectional permutation at inference, non-retrain, on
production SAGE-Mean + MLP cells.

See docs/plan_aaa_v1_2026-05-23.md (sections referenced inline).
See artifacts/reviews/2026-05-23_codex_plan_A.md for finding history.

Modes
-----
  --mode smoke   1 fold × 1 seed × 1 model × 5 epochs × 5 groups, with
                 negative control 1 (identity permutation), negative
                 control 2 (noise feature group; requires retraining), and
                 ms benchmark + projection. Plan v1 §9.4.
  --mode full    2 archs × 3 seeds × 5 folds = 30 cells, all K groups.

Outputs under artifacts/plan_aaa/ per Plan v1 §6.
"""
import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import norm, spearmanr
from sklearn.metrics import adjusted_rand_score

# Self-adaptive chdir (matches part_a:37-44).
_ROOT_CANDIDATES = [
    '/content/drive/MyDrive/GNN测试',
    '/Users/heruixi/Desktop/GNN-Testing',
]
for _p in _ROOT_CANDIDATES:
    if os.path.exists(os.path.join(_p, 'data', 'reference', 'sp500_5y_prices.csv')):
        os.chdir(_p)
        break

# Inherit model classes + training utilities from part_a so methodology is
# verbatim (Plan v1 §1). Anything Plan AAA changes is reimplemented inline
# below: data loader (168 features), permutation seeding (cell_id triple),
# convergence telemetry, aggregation order, smoke negative controls.
import run_step3_plan_z_part_a as pa  # noqa: E402
from run_tier1_phase_a import per_fold_winsorize  # noqa: E402 — canonical Plan Z++ helper


# ─────────────────────────────── configuration ───────────────────────────────

HORIZON = 21
SEEDS = [86, 123, 456]            # Plan v1 §3.2 — canonical seed list
ARCHS = ['SAGE-Mean', 'MLP']
N_FOLDS = 5
CALIBRATION_DAYS = list(range(0, 252))   # Plan v1 §3.1 — first 252 trading days
CLUSTER_THRESHOLD = 0.6                  # Plan v1 §3.1 — Plan Z++ Part A threshold
CLUSTER_METHOD = 'complete'              # complete-link on 1 − |Spearman|

HPARAMS = dict(
    hidden=64, num_layers=2, dropout=0.3,
    lr=1e-3, weight_decay=1e-4,
    epochs=50, patience=10, grad_accum=4,
    corr_window=126, corr_step=21, corr_threshold=0.6,
)

# Convergence + halt rule (Plan v1 §3.2)
CONVERGENCE_TRAIN_LOSS_DECREASE_FRAC = 0.01   # >1% drop epoch 1 → early-stop
CONVERGENCE_VAL_IC_FLOOR = -0.05
HALT_FRAC_FAILED = 0.20                       # >20% non-converged ⇒ raise

# Multiple-testing + bootstrap (Plan v1 §3.4)
BH_FDR_Q = 0.05
BLOCK_BOOTSTRAP_BLOCK_LEN = 21
BLOCK_BOOTSTRAP_N = 1000

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

ALPHA158_PATH = Path('data/reference/sp500_5y_alpha158_features_raw.npy')
ALPHA158_META = Path('data/reference/sp500_5y_alpha158_features_meta.json')
FOLD_MANIFEST_EXPANDING = Path('data/reference/fold_manifest_expanding.json')

ARTIFACT_DIR = Path('artifacts/plan_aaa')
AUDIT_DIR = ARTIFACT_DIR / 'audit'
PERMUTED_IC_DIR = ARTIFACT_DIR / 'permuted_ic'
SMOKE_DIR = ARTIFACT_DIR / 'smoke'
for _d in (ARTIFACT_DIR, AUDIT_DIR, PERMUTED_IC_DIR, SMOKE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────── 168-feature loader ───────────────────────────────

def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_data_and_features_168() -> dict:
    """Plan v1 §2.4–§2.5 — 158 Alpha158 (raw) + 10 hc features with runtime
    raw-signature assertion + provenance audit.
    """
    # Plan v1 §2.4 — runtime assertions on Alpha158 source path
    assert ALPHA158_PATH.name == 'sp500_5y_alpha158_features_raw.npy', \
        f"Plan AAA must use raw Alpha158; got {ALPHA158_PATH.name}"
    assert '_raw.npy' in ALPHA158_PATH.name, \
        "Plan AAA refuses build-time-winsorized Alpha158 (2026-05-22-a bug class)"

    # 10 hc features via inherited part_a code path (matches Plan v1 §2.5 formula
    # table: shift(1) + rolling, no same-day close leakage).
    base = pa.load_data_and_features()
    hc_tensor = base['features_np']                     # (T, N, 10) float32
    hc_names = [f'hc_{n}' for n in base['feature_names']]
    valid_tickers = base['valid_tickers']

    # 158 Alpha158 raw.
    alpha = np.load(ALPHA158_PATH).astype(np.float32)
    alpha_meta = json.load(open(ALPHA158_META))
    a_names = list(alpha_meta['feature_order'])
    assert alpha.shape[-1] == 158 and len(a_names) == 158, \
        f"Alpha158 expected 158 cols, got {alpha.shape[-1]}"
    T_hc, N_hc, _ = hc_tensor.shape
    T_a, N_a, _ = alpha.shape
    assert T_hc == T_a, f"date dim mismatch: hc {T_hc} vs alpha {T_a}"
    assert N_hc == N_a, (
        f"ticker dim mismatch: hc {N_hc} vs alpha {N_a}. Alpha158 build script "
        f"must use the same valid_tickers intersection as part_a."
    )

    # Plan v1 §2.4 follow-up (Touchpoint 2 fix FINGNN-CODE-A-02): runtime
    # ticker-order verification beyond cardinality, in 2 layers:
    # (a) Replicate the build script's intersection logic and assert == part_a's
    #     valid_tickers. Guarantees logic consistency across scripts.
    # (b) Numeric spot-check: compute KMID = (close - open) / open from OHLC for
    #     a sample of tickers on a known date and compare to alpha[date, idx, 0].
    #     Catches any file-level ticker mis-ordering even if logic appears consistent.
    sector_df = pd.read_csv('data/reference/sp500_sectors.csv')
    sec_col = [c for c in sector_df.columns if 'sector' in c.lower()][0]
    tic_col = [c for c in sector_df.columns if c != sec_col][0]
    events_tickers = set(pd.read_parquet(
        'data/fullscale/sp500_news_events.parquet', columns=['ticker'])['ticker'].unique())
    prices_full = pd.read_csv('data/reference/sp500_5y_prices.csv',
                                index_col=0, parse_dates=True)
    alpha_expected_tickers = sorted(set(prices_full.columns) & events_tickers
                                      & set(sector_df[tic_col]))
    assert alpha_expected_tickers == valid_tickers, (
        f"Alpha158 ticker intersection differs from part_a's valid_tickers "
        f"(first diff at {next((i, a, b) for i, (a, b) in enumerate(zip(alpha_expected_tickers, valid_tickers)) if a != b)}). "
        f"Refuse to proceed."
    )

    # Numeric spot-check via KMID full-period correlation. Per-date deltas drift
    # at ex-dividend/split events (yfinance adj_ohlc vs EODHD close — build script
    # comment notes ~0.1% systematic drift, larger on ex-div dates). But ticker
    # mis-alignment would scramble the WHOLE time series → Pearson corr collapses
    # to near zero. We require ρ > 0.9 per sampled ticker.
    ohlcv = pd.read_parquet('data/reference/sp500_5y_ohlcv.parquet')
    spot_tickers_idx = [1, 100, 300]  # early/mid/late alphabetical
    spot_results = []
    for ti in spot_tickers_idx:
        if ti >= len(valid_tickers):
            continue
        tk = valid_tickers[ti]
        try:
            if 'ticker' in ohlcv.index.names:
                tk_ohlcv = ohlcv.xs(tk, level='ticker')
            else:
                tk_ohlcv = ohlcv.loc[tk]
        except (KeyError, AttributeError):
            spot_results.append({'ticker': tk, 'idx': ti, 'check': 'skipped — lookup failed'})
            continue
        close_col = 'close' if 'close' in tk_ohlcv.columns else 'Close'
        open_col = 'open' if 'open' in tk_ohlcv.columns else 'Open'
        # Align by date with base['all_dates']
        tk_ohlcv = tk_ohlcv.reindex(base['all_dates'])
        expected_kmid = ((tk_ohlcv[close_col] - tk_ohlcv[open_col])
                          / tk_ohlcv[open_col].replace(0, np.nan)).values.astype(np.float64)
        actual_kmid = alpha[:, ti, 0].astype(np.float64)
        mask = np.isfinite(expected_kmid) & np.isfinite(actual_kmid) & (np.abs(actual_kmid) > 1e-8)
        if mask.sum() < 100:
            spot_results.append({'ticker': tk, 'idx': ti,
                                  'check': 'skipped — < 100 finite paired obs'})
            continue
        rho = float(np.corrcoef(expected_kmid[mask], actual_kmid[mask])[0, 1])
        spot_results.append({
            'ticker': tk, 'idx': ti, 'n_obs': int(mask.sum()),
            'pearson_corr': rho, 'pass': rho > 0.9,
        })
    spot_checked = sum(1 for r in spot_results if 'pass' in r)
    spot_pass = (spot_checked > 0
                  and all(r.get('pass', True) for r in spot_results if 'pass' in r))
    if not spot_pass:
        raise RuntimeError(
            f"Plan v1 Touchpoint 2 fix (FINGNN-CODE-A-02): KMID ticker-alignment "
            f"spot-check FAILED (Pearson ρ > 0.9 required). Results: {spot_results}"
        )
    print(f'[data] Alpha158 ticker alignment: intersection-logic PASS + KMID '
          f'time-series correlation PASS ({spot_checked} tickers, '
          f'ρ range [{min(r["pearson_corr"] for r in spot_results if "pass" in r):.4f}, '
          f'{max(r["pearson_corr"] for r in spot_results if "pass" in r):.4f}])')

    features = np.concatenate([hc_tensor, alpha], axis=-1).astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    feature_names = hc_names + a_names
    assert features.shape[-1] == 168, f"expected 168, got {features.shape[-1]}"

    # Plan v1 §2.4 — raw signature check: ROC5 raw max is ~2.4; winsorized is
    # capped at ~1.14. Hard-fail if ROC5 column looks pre-winsorized.
    roc5_col = len(hc_names) + a_names.index('ROC5')
    roc5_max = float(features[:, :, roc5_col].max())
    is_raw_sig = roc5_max > 1.5

    audit = {
        'alpha158_source_path': str(ALPHA158_PATH),
        'alpha158_md5': _md5(ALPHA158_PATH),
        'alpha158_shape': list(alpha.shape),
        'hc_source': 'run_step3_plan_z_part_a.load_data_and_features (inherited)',
        'hc_feature_count': hc_tensor.shape[-1],
        'alpha_feature_count': alpha.shape[-1],
        'total_features': features.shape[-1],
        'num_tickers': N_hc,
        'feature_value_range': [float(features.min()), float(features.max())],
        'roc5_col_index': roc5_col,
        'roc5_max_raw_signature_check': roc5_max,
        'is_raw_signature_pass': is_raw_sig,
        'numpy_version': np.__version__,
        'torch_version': torch.__version__,
        'platform': platform.platform(),
        'mps_available': bool(torch.backends.mps.is_available()),
        'cuda_available': bool(torch.cuda.is_available()),
    }
    json.dump(audit, open(AUDIT_DIR / 'data_provenance.json', 'w'), indent=2)
    if not is_raw_sig:
        raise RuntimeError(
            f"Plan v1 §2.4 raw signature check FAILED: ROC5 max={roc5_max:.4f} ≤ 1.5; "
            f"Alpha158 file appears winsorized. HALT."
        )
    print(f'[data] 168-feature tensor {features.shape}, raw-sig PASS '
          f'(ROC5 max {roc5_max:.3f})')

    return {
        'features_np': features,
        'feature_names': feature_names,
        'hc_offset': hc_tensor.shape[-1],
        'labels_np': base['labels_np'],
        'label_valid_np': base['label_valid_np'],
        'all_dates': base['all_dates'],
        'returns': base['returns'],
        'sector_groups': base['sector_groups'],
        'num_days': base['num_days'],
        'num_stocks': base['num_stocks'],
        'valid_tickers': valid_tickers,
        'audit': audit,
    }


# ─────────────────────────────── clustering ───────────────────────────────

def _winsor_p1_p99(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.nanpercentile(arr, [1, 99], axis=0)
    return np.clip(arr, lo, hi)


def compute_groups(features_np: np.ndarray, feature_names: list, day_indices: list,
                    label_valid_np: np.ndarray, threshold: float = CLUSTER_THRESHOLD,
                    method: str = CLUSTER_METHOD) -> dict:
    """Spearman complete-link grouping over `day_indices` valid-stock rows.

    Plan v1 §3.1 — for the calibration window (not a training fold), apply a
    single global p1/p99 winsor over the entire calibration slice. Returns a
    dict with deterministic group_id 0..K−1.
    """
    rows = [features_np[d][label_valid_np[d]] for d in day_indices
             if label_valid_np[d].sum() > 0]
    X = np.vstack(rows).astype(np.float64)
    X = _winsor_p1_p99(X)
    # Replace any residual NaNs (shouldn't exist after part_a's nan_to_num, but defensive)
    X = np.nan_to_num(X, nan=0.0)

    # Identify zero-variance feature columns up-front (these would cause NaN rho
    # entries from spearmanr's internal stddev division). For Plan AAA these are
    # treated as uncorrelated with everything (rho=0) so they become singleton
    # groups — a defensible neutral handling that preserves the K-group ranking.
    col_std = np.nanstd(X, axis=0)
    zero_var_cols = np.where(col_std < 1e-12)[0].tolist()

    rho, _ = spearmanr(X, axis=0)
    if isinstance(rho, float):
        rho = np.array([[1.0, rho], [rho, 1.0]])
    rho = np.asarray(rho, dtype=np.float64)
    n_nan_rho = int(np.isnan(rho).sum())
    rho = np.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(rho, 1.0)
    if zero_var_cols or n_nan_rho:
        print(f'[cluster] zero-variance features: {len(zero_var_cols)} '
              f'(indices {zero_var_cols[:8]}{"..." if len(zero_var_cols)>8 else ""}), '
              f'NaN rho entries replaced: {n_nan_rho}')

    dist = 1.0 - np.abs(rho)
    np.fill_diagonal(dist, 0.0)
    dist = np.maximum(dist, 0.0)
    cond = squareform(dist, checks=False)
    Z = linkage(cond, method=method)
    cluster_ids = fcluster(Z, t=1.0 - threshold, criterion='distance')

    grouping = defaultdict(list)
    for fi, ci in enumerate(cluster_ids):
        grouping[int(ci)].append(fi)
    # Deterministic group_id: sort clusters by min member index
    sorted_cluster_ids = sorted(grouping.keys(), key=lambda c: min(grouping[c]))
    groups = []
    for gid, ci in enumerate(sorted_cluster_ids):
        members_idx = sorted(grouping[ci])
        members = [feature_names[i] for i in members_idx]
        label = members[0] if len(members) == 1 else f'{members[0]}+{len(members) - 1}'
        groups.append({
            'group_id': gid,
            'label': label,
            'members': members,
            'members_idx': members_idx,
            'size': len(members),
        })
    return {
        'threshold': threshold,
        'clustering_method': 'complete-link on 1 − |Spearman|',
        'pooled_panel': True,
        'pooled_panel_note': (
            'Spearman computed on the (n_days × n_valid_stocks) × n_features '
            'pooled matrix. Inherited from Plan Z++ Part A protocol. ARI vs. '
            'fold-0 train slice (adjusted_rand_index.json) is the empirical '
            'robustness gate — concern flag fires if ARI < 0.85.'),
        'feature_order': list(feature_names),
        'num_features': len(feature_names),
        'num_groups': len(groups),
        'groups': groups,
        'spearman_matrix': rho.tolist(),
    }


def _groups_to_labels(groups_obj: dict, num_features: int) -> np.ndarray:
    labels = np.full(num_features, -1, dtype=np.int64)
    for g in groups_obj['groups']:
        for i in g['members_idx']:
            labels[i] = g['group_id']
    assert (labels >= 0).all(), "every feature must belong to a group"
    return labels


# ─────────────────────────────── stat utilities (inline, no cross-script import) ───────────────────────────────

def newey_west_hac(d_series: np.ndarray, lag: int) -> tuple:
    """Two-sided NW-HAC for H0: E[d]=0. Bartlett kernel.
    Returns (mean, se, t, p_two_sided).
    """
    d = np.asarray(d_series, dtype=np.float64)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < lag + 5:
        return (float(d.mean()) if n else np.nan, np.nan, np.nan, np.nan)
    mean = float(d.mean())
    centered = d - mean
    gamma_0 = float(np.mean(centered ** 2))
    long_run_var = gamma_0
    for l in range(1, lag + 1):
        gamma_l = float(np.mean(centered[l:] * centered[:-l]))
        weight = 1.0 - l / (lag + 1.0)
        long_run_var += 2.0 * weight * gamma_l
    # Touchpoint 2 fix FINGNN-CODE-A-03: non-positive finite-sample HAC estimate
    # surfaces as missing p instead of artificially huge t (clamp to 1e-12 would
    # poison BH-FDR over K≈61 groups).
    if long_run_var <= 0:
        return mean, np.nan, np.nan, np.nan
    se = float(np.sqrt(long_run_var / n))
    t = mean / se if se > 0 else 0.0
    p = 2.0 * (1.0 - norm.cdf(abs(t)))
    return mean, se, t, p


def nw_auto_lag(t: int) -> int:
    """Newey-West (1994) automatic lag = floor(4 (T/100)^(2/9))."""
    return max(int(np.floor(4.0 * (t / 100.0) ** (2.0 / 9.0))), 1)


def block_bootstrap_mean_ci(series: np.ndarray, n_boot: int = BLOCK_BOOTSTRAP_N,
                             block_len: int = BLOCK_BOOTSTRAP_BLOCK_LEN,
                             seed: int = 42, alpha: float = 0.05) -> tuple:
    """Fixed-length block (Künsch 1989) bootstrap CI for E[series].

    NB: Plan v1 §3.4 mentioned 'stationary block bootstrap'; this implementation
    matches analyze_tier1_phase_a.py's Künsch fixed-block variant (block_len=21).
    CI quality is similar for our setting; naming corrected per Touchpoint 2 A-10.
    """
    s = np.asarray(series, dtype=np.float64)
    s = s[~np.isnan(s)]
    n = len(s)
    if n < block_len * 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_len))
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        starts = rng.integers(0, n - block_len + 1, size=n_blocks)
        sample = np.concatenate([s[s_:s_ + block_len] for s_ in starts])[:n]
        means[b] = sample.mean()
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def bh_fdr(p_values: list, q: float = BH_FDR_Q) -> tuple:
    """Benjamini-Hochberg. Returns (rejected[bool], adjusted_p[float])."""
    p = np.asarray(p_values, dtype=np.float64)
    n = len(p)
    order = np.argsort(p)
    p_sorted = p[order]
    p_adj = np.minimum.accumulate((p_sorted * n / np.arange(1, n + 1))[::-1])[::-1]
    p_adj = np.minimum(p_adj, 1.0)
    rejected_sorted = p_adj <= q
    p_adj_unsorted = np.empty_like(p_adj)
    rejected_unsorted = np.empty(p_adj.shape, dtype=bool)
    p_adj_unsorted[order] = p_adj
    rejected_unsorted[order] = rejected_sorted
    return rejected_unsorted.tolist(), p_adj_unsorted.tolist()


# ─────────────────────────────── training with convergence telemetry ───────────────────────────────

def train_one_with_telemetry(model_type, features_t, labels_t, label_valid_t,
                              labels_np, label_valid_np,
                              train_days, val_days, test_days,
                              snap_tensors, snaps, sector_edge_index,
                              fold_id, seed, snap_points, max_epochs=None):
    """Wrap part_a.train_one's loop with epoch-1 / last train loss + best
    val-IC telemetry so Plan v1 §3.2 convergence criteria can be checked.
    """
    if max_epochs is None:
        max_epochs = HPARAMS['epochs']
    pa.set_seed(seed)
    in_ch = features_t.shape[-1]
    if model_type == 'MLP':
        model = pa.RankingMLP(in_ch, HPARAMS['hidden'], HPARAMS['num_layers'], HPARAMS['dropout'])
        use_graph = False
    else:
        model = pa.RankingGNN(in_ch, HPARAMS['hidden'], HPARAMS['num_layers'], HPARAMS['dropout'])
        use_graph = True
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=HPARAMS['lr'],
                            weight_decay=HPARAMS['weight_decay'])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=5, min_lr=1e-5)

    pa.assert_graph_train_only(snap_points, snaps, train_days,
                                corr_window=HPARAMS['corr_window'], fold_id=fold_id)
    frozen_si = snaps[int(train_days.max())]
    if use_graph:
        frozen_corr_ei = snap_tensors[frozen_si].to(DEVICE)
        sector_ei = sector_edge_index.to(DEVICE)
        full_ei = torch.cat([frozen_corr_ei, sector_ei], dim=1)
    else:
        full_ei = None

    best_val_mse, best_state, bad = float('inf'), None, 0
    epoch1_train_loss = None
    last_train_loss = None
    best_val_ic = -np.inf
    n_epochs_run = 0
    stopped_early = False

    for ep in range(max_epochs):
        model.train()
        opt.zero_grad()
        accum = 0
        ep_loss_sum, ep_loss_cnt = 0.0, 0
        day_order = train_days[np.random.permutation(len(train_days))]
        for step, d in enumerate(day_order):
            x = features_t[d].to(DEVICE)
            pred = model(x, full_ei)
            mask = label_valid_t[d].to(DEVICE)
            target = labels_t[d].to(DEVICE)
            if mask.sum() < 10:
                continue
            loss = F.mse_loss(pred[mask], target[mask])
            (loss / HPARAMS['grad_accum']).backward()
            ep_loss_sum += float(loss.item())
            ep_loss_cnt += 1
            accum += 1
            if accum >= HPARAMS['grad_accum'] or step == len(day_order) - 1:
                if 0 < accum < HPARAMS['grad_accum']:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(HPARAMS['grad_accum'] / accum)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
                accum = 0
        ep_train_loss = ep_loss_sum / max(ep_loss_cnt, 1)
        if ep == 0:
            epoch1_train_loss = ep_train_loss
        last_train_loss = ep_train_loss

        # Val MSE for early-stop + Val IC for §3.2 convergence floor
        model.eval()
        v_loss, v_cnt = 0.0, 0
        v_preds = np.zeros((len(val_days), features_t.shape[1]), dtype=np.float32)
        with torch.no_grad():
            for i, d in enumerate(val_days):
                x = features_t[d].to(DEVICE)
                pred = model(x, full_ei)
                v_preds[i] = pred.cpu().numpy()
                mask = label_valid_t[d].to(DEVICE)
                if mask.sum() < 10:
                    continue
                v_loss += F.mse_loss(pred[mask], labels_t[d].to(DEVICE)[mask]).item()
                v_cnt += 1
        avg_val_mse = v_loss / max(v_cnt, 1)
        v_ic = pa.daily_ic(v_preds, val_days, labels_np, label_valid_np)
        avg_val_ic = float(np.nanmean(v_ic)) if np.any(~np.isnan(v_ic)) else np.nan
        if np.isfinite(avg_val_ic) and avg_val_ic > best_val_ic:
            best_val_ic = avg_val_ic
        sched.step(avg_val_mse)
        if avg_val_mse < best_val_mse:
            best_val_mse = avg_val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        n_epochs_run = ep + 1
        if bad >= HPARAMS['patience']:
            stopped_early = True
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    test_preds = np.zeros((len(test_days), features_t.shape[1]), dtype=np.float32)
    with torch.no_grad():
        for i, d in enumerate(test_days):
            x = features_t[d].to(DEVICE)
            test_preds[i] = model(x, full_ei).cpu().numpy()

    telemetry = {
        'epoch1_train_loss': float(epoch1_train_loss) if epoch1_train_loss is not None else np.nan,
        'last_train_loss': float(last_train_loss) if last_train_loss is not None else np.nan,
        'best_val_ic': float(best_val_ic) if np.isfinite(best_val_ic) else np.nan,
        'best_val_mse': float(best_val_mse),
        'n_epochs_run': n_epochs_run,
        'stopped_early': stopped_early,
    }
    return model, test_preds, full_ei, telemetry


def check_convergence(telemetry: dict) -> tuple:
    """Plan v1 §3.2 — train loss >1% drop AND best val IC > −0.05."""
    e1 = telemetry['epoch1_train_loss']
    last = telemetry['last_train_loss']
    val_ic = telemetry['best_val_ic']
    if not np.isfinite(e1) or not np.isfinite(last):
        return False, 'non-finite train loss'
    reasons = []
    if e1 <= 0 or (e1 - last) / e1 < CONVERGENCE_TRAIN_LOSS_DECREASE_FRAC:
        reasons.append(f'train loss decrease {(e1-last)/max(e1,1e-9)*100:.2f}% < 1%')
    if not np.isfinite(val_ic):
        reasons.append('best val IC NaN')
    elif val_ic < CONVERGENCE_VAL_IC_FLOOR:
        reasons.append(f'best val IC {val_ic:.4f} < {CONVERGENCE_VAL_IC_FLOOR}')
    return (len(reasons) == 0, '; '.join(reasons) if reasons else 'OK')


# ─────────────────────────────── permutation Δ-IC inference ───────────────────────────────

def grouped_permutation_ic_aaa(model, features_t, label_valid_t, label_valid_np,
                                 labels_np, test_days, full_ei, groups,
                                 num_features, *, cell_id, cell_seed, arch_label,
                                 fold_idx, seed_idx) -> tuple:
    """Plan v1 §3.3 — shared-row permutation + SeedSequence([cell_id, group_id, date]).

    Notes
    -----
    • cell_id is globally unique 0..29 = arch_idx*15 + fold_idx*3 + seed_idx
      (Plan v1 §3.2 / §3.3 post-6th-pass). cell_seed (86/123/456) is decorative only.
    • SeedSequence is passed directly to default_rng so PCG64 gets the full 128-bit
      state via SHA256 input mixing — no uint32 truncation (post-5th-pass).
    • Non-group columns must remain bit-identical to features_t[d] after permutation
      (asserted per (group, day)).
    """
    model.eval()
    num_stocks = features_t.shape[1]
    out_ic = {}
    audit_rows = []
    for grp in groups:
        gid = int(grp['group_id'])
        grp_indices = list(grp['members_idx'])
        grp_indices_t = torch.as_tensor(grp_indices, dtype=torch.long)
        non_grp_indices_t = torch.as_tensor(
            sorted(set(range(num_features)) - set(grp_indices)), dtype=torch.long)

        permuted_preds = np.zeros((len(test_days), num_stocks), dtype=np.float32)
        with torch.no_grad():
            for i, d in enumerate(test_days):
                d_int = int(d)
                ss = np.random.SeedSequence([int(cell_id), gid, d_int])
                rng = np.random.default_rng(ss)
                perm = rng.permutation(num_stocks)
                perm_t = torch.as_tensor(perm, dtype=torch.long)

                x_orig = features_t[d]
                x = x_orig.clone()
                x_permuted = x[perm_t]
                x[:, grp_indices_t] = x_permuted[:, grp_indices_t]
                # Non-group cols must be untouched (Plan v1 §3.3 runtime assertion)
                assert torch.equal(x[:, non_grp_indices_t],
                                    x_orig[:, non_grp_indices_t]), (
                    f"non-group features were permuted (cell={cell_id}, "
                    f"group_id={gid}, date={d_int})"
                )
                x_dev = x.to(DEVICE)
                permuted_preds[i] = model(x_dev, full_ei).cpu().numpy()

                audit_rows.append({
                    'cell_id': int(cell_id),
                    'arch': arch_label,
                    'fold_idx': int(fold_idx),
                    'seed_idx': int(seed_idx),
                    'cell_seed_value': int(cell_seed),
                    'group_id': gid,
                    'group_label': grp['label'],
                    'date': d_int,
                    'perm_first10': perm[:10].tolist(),
                })
        out_ic[grp['label']] = pa.daily_ic(permuted_preds, test_days, labels_np, label_valid_np)
    return out_ic, audit_rows


# ─────────────────────────────── aggregation ───────────────────────────────

def aggregate_delta_ic(baseline_df: pd.DataFrame, permuted_df: pd.DataFrame,
                        groups_obj: dict) -> tuple:
    """Plan v1 §3.4 — collapse cells first, then NW-HAC over date dim per group."""
    paired = permuted_df.merge(
        baseline_df,
        on=['cell_id', 'arch', 'fold_idx', 'seed_idx', 'cell_seed_value', 'day_idx'],
        how='left', validate='many_to_one',
    )
    n_before = len(paired)
    paired = paired[np.isfinite(paired['IC']) & np.isfinite(paired['IC_perm'])].copy()
    n_dropped = n_before - len(paired)
    if n_dropped > 0:
        print(f'[aggregate] WARN: {n_dropped} rows dropped due to NaN IC '
              f'({n_dropped/n_before*100:.2f}% of {n_before}) — Round B B-01 guard')
    paired['delta_IC'] = paired['IC'] - paired['IC_perm']

    # 4b — cell-mean per (group, date)
    daily = paired.groupby(['group_label', 'day_idx'])['delta_IC'].mean().reset_index()
    daily = daily.rename(columns={'delta_IC': 'cell_mean_delta_IC'})

    # 4c-4d — per group: NW-HAC, BH-FDR, block bootstrap CI
    group_records = []
    pvals = []
    for g in groups_obj['groups']:
        sub = daily[daily['group_label'] == g['label']].sort_values('day_idx')
        series = sub['cell_mean_delta_IC'].values
        T = int(np.sum(~np.isnan(series)))
        lag = nw_auto_lag(T)
        mean, se, t_stat, p_two = newey_west_hac(series, lag=lag)
        ci_lo, ci_hi = block_bootstrap_mean_ci(series, seed=42 + g['group_id'])
        group_records.append({
            'group_id': g['group_id'],
            'group_label': g['label'],
            'group_size': g['size'],
            'group_members': ','.join(g['members']),
            'n_dates': T,
            'mean_delta_IC': mean,
            'nw_lag': lag,
            'nw_se': se,
            'nw_t': t_stat,
            'nw_p_two_sided': p_two,
            'bootstrap_ci_lo': ci_lo,
            'bootstrap_ci_hi': ci_hi,
        })
        pvals.append(p_two if np.isfinite(p_two) else 1.0)

    # Touchpoint 2 FINGNN-CODE-A-03 follow-up: count how many groups produced a
    # non-positive long-run-variance (NaN p) — these are mapped to p=1 for BH-FDR
    # but surfaced here for transparency.
    n_hac_degenerate = int(sum(1 for r in group_records
                                if not np.isfinite(r['nw_p_two_sided'])))

    rejected, p_adj = bh_fdr(pvals, q=BH_FDR_Q)
    for rec, rej, padj in zip(group_records, rejected, p_adj):
        rec['bh_fdr_rejected'] = bool(rej)
        rec['bh_fdr_p_adj'] = float(padj)

    ranking = pd.DataFrame(group_records).sort_values(
        'mean_delta_IC', ascending=False).reset_index(drop=True)
    ranking.insert(0, 'rank', np.arange(1, len(ranking) + 1))

    summary = {
        'n_paired_rows': int(len(paired)),
        'n_groups': int(len(groups_obj['groups'])),
        'n_cells_present': int(paired['cell_id'].nunique()),
        'n_hac_degenerate': n_hac_degenerate,
        'bh_fdr_q': BH_FDR_Q,
        'block_bootstrap_block_len': BLOCK_BOOTSTRAP_BLOCK_LEN,
        'block_bootstrap_n': BLOCK_BOOTSTRAP_N,
    }
    return daily, ranking, summary


# ─────────────────────────────── smoke mode ───────────────────────────────

def run_smoke_mode():
    """Plan v1 §9.4 — 1 fold × 1 seed × 1 model × 5 epochs × 5 groups, with
    negative control 1 (identity perm), negative control 2 (noise feature
    group, requires retrain), real perm sanity, and ms-benchmark projection.
    """
    print('═' * 70)
    print('SMOKE MODE — Plan AAA v1 §9.4')
    print('═' * 70)
    t_start = time.time()

    data = load_data_and_features_168()
    features_np = data['features_np']
    feature_names = data['feature_names']
    labels_np = data['labels_np']
    label_valid_np = data['label_valid_np']
    num_stocks = data['num_stocks']
    num_features = features_np.shape[-1]

    # Build full 168-feature groupings on calibration window (so smoke uses real
    # cluster structure rather than synthetic groups).
    print('[smoke] clustering 168 features on calibration window [0, 251]')
    groups_obj = compute_groups(features_np, feature_names, CALIBRATION_DAYS,
                                 label_valid_np)
    print(f'[smoke] {groups_obj["num_groups"]} groups discovered')
    smoke_groups = [dict(g) for g in groups_obj['groups'][:5]]

    # Use fold-0 only
    manifest = json.load(open(FOLD_MANIFEST_EXPANDING))
    fold = manifest['folds'][0]
    train_days = np.array(fold['train_days'])
    val_days = np.array(fold['val_days'])
    test_days = np.array(fold['test_days'])

    # Plan v1 §2.4 — per-fold train-only winsor at p1/p99, then per-fold scaler.
    # Winsor uses canonical run_tier1_phase_a.per_fold_winsorize (Plan Z++ Phase 0).
    features_winsor, winsor_bounds = per_fold_winsorize(features_np, train_days)
    mean, std = pa.fit_feature_scaler(features_winsor, label_valid_np, train_days)
    features_t = pa.apply_scaler(torch.tensor(features_winsor, dtype=torch.float32), mean, std)
    labels_t = torch.tensor(labels_np, dtype=torch.float32)
    label_valid_t = torch.tensor(label_valid_np, dtype=torch.bool)

    snap_points, snap_tensors, snaps = pa.build_correlation_snapshots(
        data['returns'], data['num_days'])
    sector_edge_index = pa.build_sector_edges(data['sector_groups'])

    arch_label = 'SAGE-Mean'
    arch_idx = ARCHS.index(arch_label)
    seed_idx = 0
    fold_idx = 0
    seed = SEEDS[seed_idx]
    cell_id = arch_idx * 15 + fold_idx * 3 + seed_idx
    print(f'[smoke] training cell_id={cell_id} ({arch_label}, fold={fold_idx}, '
          f'seed={seed}) — 5 epochs')

    model, test_preds, full_ei, telemetry = train_one_with_telemetry(
        arch_label, features_t, labels_t, label_valid_t, labels_np, label_valid_np,
        train_days, val_days, test_days,
        snap_tensors, snaps, sector_edge_index, fold_idx, seed,
        snap_points=snap_points, max_epochs=5,
    )
    print(f'[smoke] telemetry: {telemetry}')
    baseline_ic = pa.daily_ic(test_preds, test_days, labels_np, label_valid_np)
    print(f'[smoke] baseline mean IC = {np.nanmean(baseline_ic):+.4f}')

    # ─── 4a — identity permutation ───
    print('\n[smoke 4a] negative control 1 — identity permutation, expect ΔIC ≈ 0')
    identity_grp = dict(smoke_groups[0])
    ic_identity = np.zeros(len(test_days), dtype=np.float64)
    grp_t = torch.as_tensor(identity_grp['members_idx'], dtype=torch.long)
    perm_identity = torch.arange(num_stocks, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for i, d in enumerate(test_days):
            x = features_t[d].clone()
            x[:, grp_t] = x[perm_identity][:, grp_t]   # no-op
            x_dev = x.to(DEVICE)
            pred = model(x_dev, full_ei).cpu().numpy()
            m = label_valid_np[d]
            if m.sum() < 30:
                ic_identity[i] = np.nan
                continue
            rho, _ = spearmanr(pred[m], labels_np[d][m])
            ic_identity[i] = rho if np.isfinite(rho) else np.nan
    delta_identity = float(np.nanmean(baseline_ic - ic_identity))
    pass_identity = abs(delta_identity) < 0.005
    print(f'[smoke 4a] mean ΔIC (identity) = {delta_identity:+.5f}  '
          f'(threshold |Δ| < 0.005)  →  {"PASS" if pass_identity else "FAIL"}')

    # ─── 4b — noise feature group (requires retrain on 173-dim) ───
    print('\n[smoke 4b] negative control 2 — noise feature group (5 N(0,1) cols), '
          'requires retrain on 173-dim, expect ΔIC ≈ 0')
    rng_noise = np.random.default_rng(12345)
    noise = rng_noise.standard_normal(
        size=(features_np.shape[0], features_np.shape[1], 5)).astype(np.float32)
    noise_features = np.concatenate([features_np, noise], axis=-1)
    # Plan v1 §2.4 — per-fold winsor → scaler (also applied to 173-dim noise panel
    # so smoke noise control mirrors full-mode preprocessing exactly).
    noise_winsor, _ = per_fold_winsorize(noise_features, train_days)
    mean_n, std_n = pa.fit_feature_scaler(noise_winsor, label_valid_np, train_days)
    features_t_n = pa.apply_scaler(
        torch.tensor(noise_winsor, dtype=torch.float32), mean_n, std_n)
    noise_group = {
        'group_id': 9001, 'label': 'noise_5',
        'members': [f'noise_{i}' for i in range(5)],
        'members_idx': list(range(num_features, num_features + 5)),
        'size': 5,
    }
    model_n, preds_n, full_ei_n, tel_n = train_one_with_telemetry(
        arch_label, features_t_n, labels_t, label_valid_t,
        labels_np, label_valid_np,
        train_days, val_days, test_days,
        snap_tensors, snaps, sector_edge_index, fold_idx, seed,
        snap_points=snap_points, max_epochs=5,
    )
    baseline_n = pa.daily_ic(preds_n, test_days, labels_np, label_valid_np)
    perm_ic_noise, _ = grouped_permutation_ic_aaa(
        model_n, features_t_n, label_valid_t, label_valid_np, labels_np,
        test_days, full_ei_n, [noise_group], features_t_n.shape[-1],
        cell_id=cell_id, cell_seed=seed, arch_label=arch_label,
        fold_idx=fold_idx, seed_idx=seed_idx,
    )
    delta_noise = float(np.nanmean(baseline_n - perm_ic_noise['noise_5']))
    pass_noise_soft = abs(delta_noise) < 0.01
    print(f'[smoke 4b] mean ΔIC (noise group) = {delta_noise:+.5f}  '
          f'(soft threshold |Δ| < 0.01 at 5 epochs)  →  '
          f'{"PASS" if pass_noise_soft else "WARN"}')
    del model_n, features_t_n, noise_features
    gc.collect()

    # ─── 4c — real permutation on first non-singleton group ───
    print('\n[smoke 4c] real permutation on first multi-member group (expect ΔIC > 0)')
    real_group = next((dict(g) for g in smoke_groups if g['size'] >= 2),
                       dict(smoke_groups[0]))
    perm_ic, audit_rows = grouped_permutation_ic_aaa(
        model, features_t, label_valid_t, label_valid_np, labels_np,
        test_days, full_ei, [real_group], num_features,
        cell_id=cell_id, cell_seed=seed, arch_label=arch_label,
        fold_idx=fold_idx, seed_idx=seed_idx,
    )
    delta_real = float(np.nanmean(baseline_ic - perm_ic[real_group['label']]))
    pass_real_sign = delta_real > 0
    print(f'[smoke 4c] group "{real_group["label"]}" (size={real_group["size"]})  '
          f'mean ΔIC = {delta_real:+.5f}  (expect > 0)  →  '
          f'{"PASS" if pass_real_sign else "WARN — pipeline detects no effect at 5 epochs"}')

    # ─── 4d — forward-pass ms benchmark + projection ───
    print('\n[smoke 4d] forward-pass ms benchmark on 168-dim')
    bench_iters = min(30, len(test_days))
    model.eval()
    with torch.no_grad():
        t0 = time.time()
        for d in test_days[:bench_iters]:
            x = features_t[d].to(DEVICE)
            _ = model(x, full_ei)
        ms_per_pass = (time.time() - t0) * 1000.0 / bench_iters
    n_test_per_fold_avg = int(np.mean([len(m['test_days']) for m in manifest['folds']]))
    est_baseline_passes = len(ARCHS) * N_FOLDS * len(SEEDS) * n_test_per_fold_avg
    est_perm_passes = (len(ARCHS) * N_FOLDS * len(SEEDS)
                        * groups_obj['num_groups'] * n_test_per_fold_avg)
    total_passes = est_baseline_passes + est_perm_passes
    est_inference_h = total_passes * ms_per_pass / 1000.0 / 3600.0
    print(f'[smoke 4d] {ms_per_pass:.2f} ms/forward-pass on {DEVICE}')
    print(f'[smoke 4d] projected full-mode inference: {total_passes:,} passes × '
          f'{ms_per_pass:.2f} ms ≈ {est_inference_h:.2f} h  '
          f'(NOT including {len(ARCHS)*N_FOLDS*len(SEEDS)} cells of training)')

    # Audit uniqueness assertion (Plan v1 §3.3)
    triples = {(r['cell_id'], r['group_id'], r['date']) for r in audit_rows}
    audit_pass = len(triples) == len(audit_rows)
    print(f'[smoke 4d] audit uniqueness {"PASS" if audit_pass else "FAIL"} '
          f'({len(audit_rows)} rows, {len(triples)} unique triples)')

    pd.DataFrame(audit_rows).to_parquet(
        SMOKE_DIR / 'permutations_smoke.parquet', index=False)
    smoke_report = {
        'cell_id': cell_id, 'arch': arch_label, 'fold_idx': fold_idx, 'seed': seed,
        'epochs_max': 5,
        'telemetry_real': telemetry,
        'telemetry_noise': tel_n,
        'baseline_mean_ic': float(np.nanmean(baseline_ic)),
        'neg_control_identity_delta_ic': delta_identity,
        'neg_control_noise_delta_ic': delta_noise,
        'real_group_label': real_group['label'],
        'real_group_size': real_group['size'],
        'real_group_delta_ic': delta_real,
        'ms_per_forward_pass': ms_per_pass,
        'projected_full_inference_hours': est_inference_h,
        'projected_total_forward_passes': total_passes,
        'n_groups_full': groups_obj['num_groups'],
        'pass_identity': pass_identity,
        'pass_noise_soft': pass_noise_soft,
        'pass_real_sign': pass_real_sign,
        'pass_audit_uniqueness': audit_pass,
    }
    json.dump(smoke_report, open(SMOKE_DIR / 'smoke_report.json', 'w'), indent=2)
    elapsed = time.time() - t_start
    print(f'\n[smoke] DONE in {elapsed:.1f}s → {SMOKE_DIR / "smoke_report.json"}')
    return smoke_report


# ─────────────────────────────── full mode ───────────────────────────────

def run_full_mode():
    print('═' * 70)
    print('FULL MODE — Plan AAA v1 §3')
    print('═' * 70)
    t_start = time.time()

    data = load_data_and_features_168()
    features_np = data['features_np']
    feature_names = data['feature_names']
    labels_np = data['labels_np']
    label_valid_np = data['label_valid_np']
    num_features = features_np.shape[-1]

    # Environment audit (Plan v1 §3.3, §5 hedge #13)
    import scipy
    env_audit = {
        'numpy_version': np.__version__,
        'torch_version': torch.__version__,
        'scipy_version': scipy.__version__,
        'pandas_version': pd.__version__,
        'python_version': sys.version,
        'platform': platform.platform(),
        'device': str(DEVICE),
        'mps_available': bool(torch.backends.mps.is_available()),
        'cuda_available': bool(torch.cuda.is_available()),
        'seeds': SEEDS,
        'archs': ARCHS,
        'n_folds': N_FOLDS,
        'cell_id_formula': 'arch_idx*15 + fold_idx*3 + seed_idx (range 0..29)',
        'hparams': HPARAMS,
        'calibration_days': [int(CALIBRATION_DAYS[0]), int(CALIBRATION_DAYS[-1])],
        'cluster_threshold': CLUSTER_THRESHOLD,
        'cluster_method': CLUSTER_METHOD,
        'bh_fdr_q': BH_FDR_Q,
        'block_bootstrap_block_len': BLOCK_BOOTSTRAP_BLOCK_LEN,
        'block_bootstrap_n': BLOCK_BOOTSTRAP_N,
        'convergence_train_loss_decrease_frac': CONVERGENCE_TRAIN_LOSS_DECREASE_FRAC,
        'convergence_val_ic_floor': CONVERGENCE_VAL_IC_FLOOR,
        'halt_frac_failed': HALT_FRAC_FAILED,
    }
    json.dump(env_audit, open(AUDIT_DIR / 'environment.json', 'w'), indent=2)

    # ─── Step 1: clustering on calibration window ───
    print('\n[step 1] clustering 168 features on calibration window [0, 251]')
    groups_obj = compute_groups(features_np, feature_names, CALIBRATION_DAYS,
                                 label_valid_np, threshold=CLUSTER_THRESHOLD)
    groups_obj['calibration_window_indices'] = [int(CALIBRATION_DAYS[0]),
                                                  int(CALIBRATION_DAYS[-1])]
    groups_obj['calibration_window_dates'] = [
        str(data['all_dates'][CALIBRATION_DAYS[0]].date()),
        str(data['all_dates'][CALIBRATION_DAYS[-1]].date()),
    ]
    spearman_matrix = groups_obj.pop('spearman_matrix')
    json.dump(groups_obj, open(ARTIFACT_DIR / 'groups_168.json', 'w'), indent=2)
    pd.DataFrame(spearman_matrix, index=feature_names, columns=feature_names
                  ).to_csv(ARTIFACT_DIR / 'calibration_window_spearman_168.csv')
    # universe_definition_168 (Plan v1 §6)
    universe_def = {
        'features': [
            {'index': i, 'name': n,
             'provenance': 'hc_inherited_from_part_a' if n.startswith('hc_') else 'alpha158_raw'}
            for i, n in enumerate(feature_names)
        ],
        'num_features': len(feature_names),
        'hc_offset': data['hc_offset'],
    }
    json.dump(universe_def, open(ARTIFACT_DIR / 'universe_definition_168.json', 'w'), indent=2)
    print(f'[step 1] {groups_obj["num_groups"]} groups discovered')

    # Plan v1 §3.3 runtime assertion: group_id 0..K−1 contiguous
    gids = [g['group_id'] for g in groups_obj['groups']]
    assert gids == list(range(len(gids))), f'group_ids not 0..K-1 contiguous: {gids}'

    # ─── Step 1b: supplementary fold-0 clustering + ARI ───
    print('\n[step 1b] supplementary clustering on fold-0 train slice + ARI')
    manifest = json.load(open(FOLD_MANIFEST_EXPANDING))
    # Round B B-03 guard: calibration window must be disjoint from any fold's test_days
    _cal_set = set(CALIBRATION_DAYS)
    for _fi, _f in enumerate(manifest['folds']):
        _overlap = _cal_set & set(_f['test_days'])
        assert not _overlap, (
            f'fold {_fi} test_days overlap calibration window: {len(_overlap)} days '
            f'(would leak test-period correlation structure into clustering)')
    fold0_train_days = list(manifest['folds'][0]['train_days'])
    groups_obj_f0 = compute_groups(features_np, feature_names, fold0_train_days,
                                    label_valid_np, threshold=CLUSTER_THRESHOLD)
    groups_obj_f0.pop('spearman_matrix', None)
    json.dump(groups_obj_f0, open(ARTIFACT_DIR / 'groups_168_fold0.json', 'w'), indent=2)
    ari = float(adjusted_rand_score(
        _groups_to_labels(groups_obj, len(feature_names)),
        _groups_to_labels(groups_obj_f0, len(feature_names)),
    ))
    json.dump({
        'ari_calibration_vs_fold0': ari,
        'flag_concern_if_ari_lt': 0.85,
        'concern_triggered': bool(ari < 0.85),
        'num_groups_calibration': groups_obj['num_groups'],
        'num_groups_fold0': groups_obj_f0['num_groups'],
    }, open(ARTIFACT_DIR / 'adjusted_rand_index.json', 'w'), indent=2)
    print(f'[step 1b] ARI(calib, fold0) = {ari:.4f}  '
          f'({"CONCERN" if ari < 0.85 else "OK"} vs 0.85 threshold)')

    # ─── Step 2-3: train 30 cells + baseline + permutation ───
    print(f'\n[step 2-3] {len(ARCHS)*N_FOLDS*len(SEEDS)} cells × '
          f'{groups_obj["num_groups"]} groups')
    labels_t = torch.tensor(labels_np, dtype=torch.float32)
    label_valid_t = torch.tensor(label_valid_np, dtype=torch.bool)
    snap_points, snap_tensors, snaps = pa.build_correlation_snapshots(
        data['returns'], data['num_days'])
    sector_edge_index = pa.build_sector_edges(data['sector_groups'])

    convergence_log = []
    baseline_rows = []
    permuted_rows = []
    audit_perm_all = []
    scaler_log = []
    failed_cells = 0
    total_cells = len(ARCHS) * N_FOLDS * len(SEEDS)
    cell_counter = 0

    for fold_idx in range(N_FOLDS):
        fold = manifest['folds'][fold_idx]
        train_days = np.array(fold['train_days'])
        val_days = np.array(fold['val_days'])
        test_days = np.array(fold['test_days'])
        # Plan v1 §2.4 — per-fold train-only winsor (p1/p99) → per-fold scaler.
        features_winsor, winsor_bounds = per_fold_winsorize(features_np, train_days)
        mean, std = pa.fit_feature_scaler(features_winsor, label_valid_np, train_days)
        features_t = pa.apply_scaler(torch.tensor(features_winsor, dtype=torch.float32),
                                       mean, std)
        scaler_log.append({'fold': fold_idx,
                            'feature_mean': mean.tolist(),
                            'feature_std': std.tolist(),
                            'winsor_bounds': winsor_bounds.tolist()})

        for arch_idx, arch in enumerate(ARCHS):
            for seed_idx, seed in enumerate(SEEDS):
                cell_id = arch_idx * 15 + fold_idx * 3 + seed_idx
                assert 0 <= cell_id <= 29, f'cell_id {cell_id} out of [0,29] — Round B B-02 guard'
                cell_counter += 1
                t_cell = time.time()

                model, test_preds, full_ei, telemetry = train_one_with_telemetry(
                    arch, features_t, labels_t, label_valid_t,
                    labels_np, label_valid_np,
                    train_days, val_days, test_days,
                    snap_tensors, snaps, sector_edge_index, fold_idx, seed,
                    snap_points=snap_points,
                )
                converged, reason = check_convergence(telemetry)
                if not converged:
                    failed_cells += 1
                    print(f'  ⚠ cell_id={cell_id} {arch} fold={fold_idx} seed={seed} '
                          f'NON-CONVERGED: {reason}')
                convergence_log.append({
                    'cell_id': cell_id, 'arch': arch, 'fold_idx': fold_idx,
                    'seed_idx': seed_idx, 'seed': seed,
                    'converged': converged, 'reason': reason,
                    **telemetry,
                })
                if failed_cells > HALT_FRAC_FAILED * total_cells:
                    json.dump(convergence_log,
                               open(AUDIT_DIR / 'convergence.json', 'w'), indent=2)
                    raise RuntimeError(
                        f'HALT (Plan v1 §3.2): {failed_cells} non-converged cells '
                        f'> {HALT_FRAC_FAILED*100:.0f}% of {total_cells}. '
                        f'See artifacts/plan_aaa/audit/convergence.json'
                    )

                # Baseline IC (per cell, per test day)
                base_ic = pa.daily_ic(test_preds, test_days, labels_np, label_valid_np)
                for i, d in enumerate(test_days):
                    baseline_rows.append({
                        'cell_id': cell_id, 'arch': arch, 'fold_idx': fold_idx,
                        'seed_idx': seed_idx, 'cell_seed_value': int(seed),
                        'day_idx': int(d),
                        'IC': float(base_ic[i]) if np.isfinite(base_ic[i]) else np.nan,
                    })

                # Permuted IC per group
                perm_ic, audit_rows = grouped_permutation_ic_aaa(
                    model, features_t, label_valid_t, label_valid_np, labels_np,
                    test_days, full_ei, groups_obj['groups'], num_features,
                    cell_id=cell_id, cell_seed=seed, arch_label=arch,
                    fold_idx=fold_idx, seed_idx=seed_idx,
                )
                audit_perm_all.extend(audit_rows)
                for grp_label, ic_arr in perm_ic.items():
                    for i, d in enumerate(test_days):
                        permuted_rows.append({
                            'cell_id': cell_id, 'arch': arch, 'fold_idx': fold_idx,
                            'seed_idx': seed_idx, 'cell_seed_value': int(seed),
                            'group_label': grp_label, 'day_idx': int(d),
                            'IC_perm': float(ic_arr[i]) if np.isfinite(ic_arr[i]) else np.nan,
                        })

                elapsed = time.time() - t_cell
                print(f'  [{cell_counter}/{total_cells}] cell_id={cell_id} {arch} '
                      f'fold={fold_idx} seed={seed}  baseline IC={np.nanmean(base_ic):+.4f}  '
                      f'val_ic_best={telemetry["best_val_ic"]:+.4f}  '
                      f'epochs={telemetry["n_epochs_run"]}  conv={"Y" if converged else "N"}  '
                      f'{elapsed:.1f}s')
                del model
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()

    json.dump(convergence_log, open(AUDIT_DIR / 'convergence.json', 'w'), indent=2)
    pd.DataFrame(scaler_log).to_json(AUDIT_DIR / 'per_fold_scaler.json',
                                       orient='records', indent=2)

    # Audit uniqueness assertion (Plan v1 §3.3)
    triples = {(r['cell_id'], r['group_id'], r['date']) for r in audit_perm_all}
    assert len(triples) == len(audit_perm_all), (
        f'audit uniqueness FAILED: {len(audit_perm_all)} rows but '
        f'{len(triples)} unique triples — cell_id assignment bug'
    )
    print(f'[audit] uniqueness PASS ({len(audit_perm_all):,} rows, '
          f'{len(triples):,} unique triples)')
    pd.DataFrame(audit_perm_all).to_parquet(
        AUDIT_DIR / 'permutations.parquet', index=False)

    # Persist raw IC tables
    base_df = pd.DataFrame(baseline_rows)
    perm_df = pd.DataFrame(permuted_rows)
    base_df.to_csv(ARTIFACT_DIR / 'baseline_ic_per_cell.csv', index=False)
    for g in groups_obj['groups']:
        sub = perm_df[perm_df['group_label'] == g['label']]
        safe = g['label'].replace('/', '_').replace(' ', '_')
        sub.to_csv(PERMUTED_IC_DIR / f'{safe}.csv', index=False)

    # ─── Step 4: aggregate ───
    print('\n[step 4] aggregating Δ-IC + NW-HAC + BH-FDR + block bootstrap')
    daily_delta, ranking, summary = aggregate_delta_ic(base_df, perm_df, groups_obj)
    daily_delta.to_csv(ARTIFACT_DIR / 'daily_delta_ic_per_group.csv', index=False)
    ranking.to_csv(ARTIFACT_DIR / 'ranking.csv', index=False)
    json.dump({
        'summary': summary,
        'groups_ranked': ranking.to_dict(orient='records'),
    }, open(ARTIFACT_DIR / 'ranking.json', 'w'), indent=2, default=float)
    print(ranking[['rank', 'group_label', 'group_size', 'mean_delta_IC',
                    'nw_t', 'nw_p_two_sided', 'bh_fdr_p_adj', 'bh_fdr_rejected']
                  ].to_string(index=False))

    # ─── Step 5: hand-curated → group mapping ───
    hc_names = [n for n in feature_names if n.startswith('hc_')]
    feat_to_group = {m: g for g in groups_obj['groups'] for m in g['members']}
    hc_mapping = []
    for hn in hc_names:
        g = feat_to_group.get(hn)
        if g is None:
            continue
        rank_row = ranking[ranking['group_label'] == g['label']].iloc[0]
        hc_mapping.append({
            'hc_feature': hn,
            'group_id': int(g['group_id']),
            'group_label': g['label'],
            'group_size': int(g['size']),
            'group_members': g['members'],
            'rank_out_of_K': int(rank_row['rank']),
            'mean_delta_IC': float(rank_row['mean_delta_IC']),
            'nw_t': float(rank_row['nw_t']),
            'nw_p_two_sided': float(rank_row['nw_p_two_sided']),
            'bh_fdr_p_adj': float(rank_row['bh_fdr_p_adj']),
            'bh_fdr_rejected': bool(rank_row['bh_fdr_rejected']),
        })
    json.dump(hc_mapping,
               open(ARTIFACT_DIR / 'hand_curated_mapping_168.json', 'w'),
               indent=2, default=float)
    print(f'\n[step 5] hand_curated_mapping_168.json — {len(hc_mapping)} hc features mapped')

    elapsed = time.time() - t_start
    print(f'\n[done] Plan AAA full mode in {elapsed/60:.1f} min  '
          f'(failed cells: {failed_cells}/{total_cells})')


# ─────────────────────────────── entry ───────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['smoke', 'full'], default='smoke')
    args = parser.parse_args()
    print(f'[device] {DEVICE}')
    if args.mode == 'smoke':
        run_smoke_mode()
    else:
        run_full_mode()


if __name__ == '__main__':
    main()
