#!/usr/bin/env python
"""
sanity_common.py — shared library for the Sanity-Check Suite (E0–E4).

Purpose (paper thesis defense): before publishing the Story A NULL result
("predefined graph relations add no incremental info for cross-sectional ranking"),
falsify the alternative hypothesis H2 (the null is a broken-pipeline artifact) vs
H1 (the null is a real task property). See:
  - plan: /Users/heruixi/.claude/plans/sanity-check-sorted-lark.md
  - prereg: sanity_check_preregistration.md
  - Codex Touchpoint 1 review: artifacts/reviews/2026-06-10_codex_plan_A.md

This module provides:
  - Four graph/signal builders (all reuse the anchor's fold/snapshot machinery,
    NOTHING in run_storya_e1_anchor.py is modified — import-only):
      E1   build_oracle_edges_per_fold       (return-corr oracle; UPPER-BOUND DIAGNOSTIC only)
      E1b  build_label_similarity_oracle_edges (leaked forward-label-similarity; NECESSARY control)
      E2   build_shuffled_edges              (degree-preserving rewiring; negative control)
      E3   planted-signal helpers            (synthetic graph-borne signal; NECESSARY control)
  - Verdict / threshold logic (single source of truth for the LOCKED pre-registered
    thresholds; consumed by analyze_sanity.py).

Codex Round A fixes baked in:
  A-01  E3 label is SAME-INDEX neighbor-mean of OBSERVED features (model can see it).
  A-02  graph-provenance canary (independently recomputes alpha1 edges) lives in run_sanity.py E0.
  A-03  E1 demoted to upper-bound diagnostic (no sick branch); E1b added as necessary control.
  A-04  E3 inference unit = seed-average-per-fold paired daily delta-IC, HLN+BH-FDR.
  A-05  E2 equivalence gate (one-sided 95% block-bootstrap CI < unit AND TOST +/-0.005).
  A-06  shuffled builder: undirected-canonical swap + re-symmetrize + assertions.
  A-07  E4 collinearity AUC is descriptive only (not a verdict input).
"""

import numpy as np
import torch
from scipy.stats import spearmanr, rankdata

# ── Import-only reuse of the frozen anchor (its main() is __main__-guarded) ──
from run_storya_e1_anchor import (  # noqa: F401
    create_fold_masks,
    get_frozen_snapshot_idx,
    build_correlation_snapshots,
    HORIZON,
    GRAPH_HPARAMS,
    CANONICAL_SEEDS,
    WALK_FORWARD_FOLDS,
)

# ══════════════════════════════════════════════════════════════
# LOCKED PRE-REGISTERED THRESHOLDS (single source of truth)
# ══════════════════════════════════════════════════════════════
# Unit edge lift = max mean_delta_ic over the 3 alpha-vs-alpha1 pairs, regime='full'
# (source: artifacts/storya_e6_edge_ablation/edge_pairs_dm.csv = 0.01001 for alpha3).
# Loaded from CSV at analyze time (load_edge_lift_unit) so the constant below is only a
# documented fallback / sanity bound — NOT used for the verdict if the CSV is present.
EDGE_LIFT_UNIT_FALLBACK = 0.01001
ORACLE_PASS_MULTIPLE = 3.0          # E1b Result-A (pipeline innocent) requires lift >= 3x unit
E3_RECOVERY_FRACTION = 0.70         # E3 pass requires GNN IC >= 0.70 x measured achievable
E3_TARGET_ACHIEVABLE_IC = 0.05      # planted-signal calibration target
E3_CALIB_BAND = (0.04, 0.06)        # acceptable measured-oracle-IC band after calibration
E2_TOST_MARGIN = 0.005              # E2 equivalence margin (|shuffled - no_graph| ΔIC)
ORACLE_CORR_THRESHOLD = GRAPH_HPARAMS['corr_threshold']  # 0.6, parity with alpha1


# ══════════════════════════════════════════════════════════════
# Spearman correlation helper (rank + Pearson; robust to constant cols)
# ══════════════════════════════════════════════════════════════

def _spearman_corr_matrix(window: np.ndarray) -> np.ndarray:
    """Spearman correlation matrix across columns of `window` (T, N) -> (N, N).
    Rank each column then Pearson-correlate; constant/NaN columns -> 0 correlation.
    Used for both E1 (returns) and E1b (labels)."""
    T, N = window.shape
    w = np.where(np.isfinite(window), window, np.nan)
    # Rank along time (axis 0); columns that are constant or all-nan -> rank variance 0.
    ranks = np.empty_like(w, dtype=np.float64)
    finite_col = np.zeros(N, dtype=bool)
    for j in range(N):
        col = w[:, j]
        m = np.isfinite(col)
        if m.sum() >= 3 and np.nanstd(col) > 1e-12:
            r = np.full(T, np.nan)
            r[m] = rankdata(col[m])
            ranks[:, j] = r
            finite_col[j] = True
        else:
            ranks[:, j] = np.nan
    # Pairwise Pearson on ranks, ignoring nan rows pairwise is expensive; since the
    # anchor's returns/labels have no intra-window nans on valid tickers, fill nan->col mean.
    ranks = np.where(np.isfinite(ranks), ranks, np.nan)
    col_mean = np.nanmean(ranks, axis=0)
    inds = np.where(np.isnan(ranks))
    ranks[inds] = np.take(col_mean, inds[1])
    ranks = np.nan_to_num(ranks, nan=0.0)
    cm = np.corrcoef(ranks.T)
    cm = np.nan_to_num(cm, nan=0.0)
    # Zero out columns/rows that were not finite (degenerate stocks).
    bad = ~finite_col
    cm[bad, :] = 0.0
    cm[:, bad] = 0.0
    return cm


def _corr_matrix_to_edge_index(cm: np.ndarray, threshold: float,
                               signed: bool = False) -> torch.Tensor:
    """Off-diagonal -> symmetric (2, E) long edge_index. np.where on a symmetric matrix
    already emits both (i,j) and (j,i).
    signed=False: |cm| > threshold (magnitude — E1 return-corr parity with alpha1).
    signed=True : cm > threshold (POSITIVE only — E1b label oracle: connect SAME-direction
                  co-movers only; anti-correlated stocks must NOT be smoothed together)."""
    cm = cm.copy()
    np.fill_diagonal(cm, 0.0)
    mask = (cm > threshold) if signed else (np.abs(cm) > threshold)
    src, dst = np.where(mask)
    if len(src) == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor(np.stack([src, dst]), dtype=torch.long)


# ══════════════════════════════════════════════════════════════
# E1  ORACLE EDGES (return-corr) — UPPER-BOUND DIAGNOSTIC ONLY (Codex A-03)
# ══════════════════════════════════════════════════════════════

def build_oracle_edges_per_fold(returns, fold_cfg, all_dates,
                                threshold: float = ORACLE_CORR_THRESHOLD) -> torch.Tensor:
    """E1 oracle: edges from |Spearman rho| > threshold over THIS fold's TEST-window returns.

    DELIBERATELY uses future (test-window) information. DIAGNOSTIC UPPER BOUND ONLY:
    never reported as a result; NOT a pass/fail necessary control (a contemporaneous
    correlation graph need not carry 21d-FORWARD rank info even when message passing works
    — Codex A-03). The sole decisive necessary control is E3 (E1/E1b are supporting diagnostics).

    returns : pd.DataFrame (T, N) daily simple returns (anchor `core['returns']`).
    """
    _, _, test_days = create_fold_masks(fold_cfg, all_dates, HORIZON)
    window = returns.iloc[test_days[0]:test_days[-1] + 1].values  # (T_test, N)
    cm = _spearman_corr_matrix(window)
    return _corr_matrix_to_edge_index(cm, threshold)


# ══════════════════════════════════════════════════════════════
# E1b  LABEL-SIMILARITY ORACLE EDGES — NECESSARY control (Codex A-03)
# ══════════════════════════════════════════════════════════════

def build_label_similarity_oracle_edges(labels_np, fold_cfg, all_dates,
                                        threshold: float = ORACLE_CORR_THRESHOLD) -> torch.Tensor:
    """E1b oracle: connect i,j if their TEST-window FORWARD-LABEL series correlate > threshold.

    Leaks the forward label (oracle), fenced identically to E1 (never reported as a result).
    Unlike E1, this IS a valid NECESSARY control: a working pipeline MUST convert a
    maximally-predictive (cheating) topology into IC. If E1b shows no lift -> SICK.

    labels_np : (T, N) market-demeaned 21d-forward return z-scores (anchor build_labels).
    """
    _, _, test_days = create_fold_masks(fold_cfg, all_dates, HORIZON)
    window = labels_np[test_days[0]:test_days[-1] + 1]  # (T_test, N)
    cm = _spearman_corr_matrix(window)
    # POSITIVE correlation only (signed): connect same-direction label co-movers; smoothing
    # anti-correlated stocks together would corrupt the oracle (smoke finding 2026-06-10).
    return _corr_matrix_to_edge_index(cm, threshold, signed=True)


# ══════════════════════════════════════════════════════════════
# E2  DEGREE-PRESERVING SHUFFLED EDGES — negative control (Codex A-06)
# ══════════════════════════════════════════════════════════════

def _to_undirected_pairs(edge_index: torch.Tensor) -> set:
    """Directed symmetric (2, E) -> set of undirected (i, j) with i < j, self-loops dropped."""
    ei = edge_index.cpu().numpy()
    pairs = set()
    for k in range(ei.shape[1]):
        i, j = int(ei[0, k]), int(ei[1, k])
        if i == j:
            continue
        pairs.add((i, j) if i < j else (j, i))
    return pairs


def _undirected_degree(pairs: set, num_stocks: int) -> np.ndarray:
    deg = np.zeros(num_stocks, dtype=np.int64)
    for i, j in pairs:
        deg[i] += 1
        deg[j] += 1
    return deg


def _symmetrize_pairs(pairs: set) -> torch.Tensor:
    """Undirected pair set -> directed symmetric (2, 2M) long edge_index."""
    if not pairs:
        return torch.zeros((2, 0), dtype=torch.long)
    src, dst = [], []
    for i, j in pairs:
        src += [i, j]
        dst += [j, i]
    return torch.tensor(np.stack([np.array(src), np.array(dst)]), dtype=torch.long)


def build_shuffled_edges(edge_index_alpha1: torch.Tensor, num_stocks: int,
                         n_swaps_mult: int = 10, seed: int = 86) -> torch.Tensor:
    """E2: degree-preserving double-edge-swap of the alpha1 graph (numpy self-contained).

    Canonicalize to undirected i<j, swap on a simple undirected graph (reject self-loops
    & duplicates), then re-symmetrize. Preserves the EXACT undirected degree sequence
    (Codex A-06). Deterministic for a fixed seed.

    Returns directed symmetric (2, 2M) long edge_index.
    """
    pairs0 = _to_undirected_pairs(edge_index_alpha1)
    M = len(pairs0)
    if M < 2:
        return _symmetrize_pairs(pairs0)
    rng = np.random.default_rng(seed)
    edge_set = set(pairs0)
    edge_list = list(pairs0)
    n_swaps = n_swaps_mult * M
    done = 0
    attempts = 0
    max_attempts = n_swaps * 20  # safety bound against degenerate stalls
    while done < n_swaps and attempts < max_attempts:
        attempts += 1
        e1 = edge_list[rng.integers(len(edge_list))]
        e2 = edge_list[rng.integers(len(edge_list))]
        if e1 == e2:
            continue
        a, b = e1
        c, d = e2
        # Randomly orient the second edge to allow both swap pairings.
        if rng.random() < 0.5:
            c, d = d, c
        # Proposed new undirected edges (canonical i<j).
        n1 = (a, d) if a < d else (d, a)
        n2 = (c, b) if c < b else (b, c)
        if a == d or c == b:           # self-loop
            continue
        if n1 == n2:                   # would create a duplicate within the pair
            continue
        if n1 in edge_set or n2 in edge_set:
            continue
        # Apply swap.
        edge_set.discard(e1)
        edge_set.discard(e2)
        edge_set.add(n1)
        edge_set.add(n2)
        # Keep edge_list in sync (rebuild lazily is costly; do targeted update).
        edge_list = list(edge_set)
        done += 1
    return _symmetrize_pairs(edge_set)


def assert_shuffled_valid(shuffled: torch.Tensor, edge_index_alpha1: torch.Tensor,
                          num_stocks: int) -> None:
    """Codex A-06: 5 assertions — undirected degree sequence preserved, 2E directed
    count, no self-loops, no duplicate directed edges."""
    pairs_s = _to_undirected_pairs(shuffled)
    pairs_0 = _to_undirected_pairs(edge_index_alpha1)
    deg_s = _undirected_degree(pairs_s, num_stocks)
    deg_0 = _undirected_degree(pairs_0, num_stocks)
    assert np.array_equal(deg_s, deg_0), "shuffled degree sequence != alpha1"
    ei = shuffled.cpu().numpy()
    assert ei.shape[1] == 2 * len(pairs_s), "directed count != 2 * |undirected|"
    assert not np.any(ei[0] == ei[1]), "shuffled contains self-loops"
    directed = set(zip(ei[0].tolist(), ei[1].tolist()))
    assert len(directed) == ei.shape[1], "shuffled contains duplicate directed edges"


# ══════════════════════════════════════════════════════════════
# E3  PLANTED-SIGNAL helpers — NECESSARY control (Codex A-01)
# ══════════════════════════════════════════════════════════════

def make_planted_features(num_days: int, num_stocks: int, feat_dim: int, seed: int) -> np.ndarray:
    """X ~ N(0,1) iid, shape (T, N, D), deterministic per seed."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((num_days, num_stocks, feat_dim)).astype(np.float32)


def adjacency_norm(edge_index: torch.Tensor, num_stocks: int) -> np.ndarray:
    """Row-normalized dense adjacency (N, N): A_norm[i, j] = 1/deg(i) if edge, else 0.
    Isolated nodes -> all-zero row (signal s_i = 0)."""
    A = np.zeros((num_stocks, num_stocks), dtype=np.float64)
    ei = edge_index.cpu().numpy()
    A[ei[0], ei[1]] = 1.0
    np.fill_diagonal(A, 0.0)
    deg = A.sum(axis=1)
    nz = deg > 0
    A[nz] = A[nz] / deg[nz, None]
    return A


def planted_signal(X: np.ndarray, A_norm: np.ndarray) -> np.ndarray:
    """s[t, i] = (A_norm @ X[t, :, 0])_i = sum_j A_norm[i,j] X[t,j,0].
    Vectorized: s = X0 @ A_norm.T. Shape (T, N)."""
    X0 = X[:, :, 0].astype(np.float64)        # (T, N)
    return (X0 @ A_norm.T).astype(np.float32)  # (T, N)


def calibrate_beta(s: np.ndarray, day_indices: np.ndarray,
                   target_ic: float = E3_TARGET_ACHIEVABLE_IC,
                   connected: np.ndarray = None) -> float:
    """Closed-form beta so that achievable oracle IC ~ target.
    With y = beta*s + eps, sigma_eps=1: corr(s, y) ~ beta*sigma_s for small beta*sigma_s.
    sigma_s = median over days of the cross-sectional std of s (over CONNECTED stocks only —
    isolated stocks have s=0 and are excluded from the planted-recovery evaluation)."""
    day_s = s[day_indices]                          # (T_sub, N)
    if connected is not None:
        day_s = day_s[:, connected]
    sigma_s = float(np.median(day_s.std(axis=1)))
    if sigma_s < 1e-8:
        return 0.0
    return target_ic / sigma_s


def plant_labels(s: np.ndarray, beta: float, seed: int, sigma_eps: float = 1.0) -> np.ndarray:
    """y[t] = beta*s[t] + eps[t]  (SAME index as features — Codex A-01). eps seeded.
    Returns y (T, N) float32."""
    rng = np.random.default_rng(seed + 10_007)      # offset so eps != feature noise stream
    eps = rng.standard_normal(s.shape).astype(np.float32) * sigma_eps
    return (beta * s + eps).astype(np.float32)


def measure_oracle_ic(s: np.ndarray, y: np.ndarray, day_indices: np.ndarray,
                      valid_mask: np.ndarray = None, min_valid: int = 30) -> float:
    """Mean daily Spearman(s, y) over day_indices — the E3 pass DENOMINATOR (measured
    achievable IC, NOT the nominal 0.05). s is the true planted signal. If valid_mask
    (T,N) or (N,) is given, evaluate only over valid (connected) stocks."""
    ics = []
    for d in day_indices:
        if valid_mask is not None:
            m = valid_mask[d] if valid_mask.ndim == 2 else valid_mask
            if m.sum() < min_valid:
                continue
            ic, _ = spearmanr(s[d][m], y[d][m])
        else:
            ic, _ = spearmanr(s[d], y[d])
        if not np.isnan(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else 0.0


def build_planted_fold(edge_index_alpha1: torch.Tensor, X: np.ndarray, num_stocks: int,
                       train_days: np.ndarray, test_days: np.ndarray, seed: int,
                       target_ic: float = E3_TARGET_ACHIEVABLE_IC, max_rescale: int = 3):
    """Assemble one fold's planted-recovery dataset on the fold's frozen alpha1 topology.

    ISOLATED stocks (degree 0, s=0) are marked y_valid=False — they have no graph-borne
    signal, so E3 measures recovery only among stocks that HAVE neighbors (cleaner GNN-vs-MLP
    discriminator + stable calibration). beta is calibrated on TRAIN days over connected
    stocks, then rescaled toward target_ic on TEST days up to `max_rescale` times (Codex A-01).

    Returns dict: y_np, y_valid_np, beta, achievable_ic, A_norm, s, n_connected.
    """
    A_norm = adjacency_norm(edge_index_alpha1, num_stocks)
    connected = A_norm.sum(axis=1) > 0                            # (N,) bool
    s = planted_signal(X, A_norm)                                 # (T, N)
    y_valid = np.zeros(s.shape, dtype=bool)
    y_valid[:, connected] = True                                 # isolated -> invalid

    beta = calibrate_beta(s, train_days, target_ic, connected)
    y = plant_labels(s, beta, seed)
    achievable = measure_oracle_ic(s, y, test_days, y_valid)
    for _ in range(max_rescale):
        if achievable <= 1e-6 or (E3_CALIB_BAND[0] <= achievable <= E3_CALIB_BAND[1]):
            break
        beta = beta * (target_ic / achievable)
        y = plant_labels(s, beta, seed)
        achievable = measure_oracle_ic(s, y, test_days, y_valid)
    return {
        'y_np': y, 'y_valid_np': y_valid, 'beta': float(beta),
        'achievable_ic': float(achievable), 'A_norm': A_norm, 's': s,
        'n_connected': int(connected.sum()),
    }


# ══════════════════════════════════════════════════════════════
# A-02 PROVENANCE CANARY — independently recompute alpha1 edges
# ══════════════════════════════════════════════════════════════

def recompute_alpha1_frozen_edges(returns, all_dates, fold_cfg, num_days):
    """Independently rebuild the frozen alpha1 edge set for `fold_cfg` from returns/ticker
    order, mirroring the anchor's build_correlation_snapshots + get_frozen_snapshot_idx.
    Used by E0 to assert the edges train_nn actually consumes are provenance-correct
    (Codex A-02). Returns (edge_index (2,E) long, frozen_si)."""
    snaps, _, snapshot_points = build_correlation_snapshots(returns, num_days)
    train_days, _, _ = create_fold_masks(fold_cfg, all_dates, HORIZON)
    frozen_si = get_frozen_snapshot_idx(train_days[-1], snapshot_points)
    return snaps[frozen_si], frozen_si


def edge_index_signature(edge_index: torch.Tensor) -> tuple:
    """(n_directed_edges, density-proxy, sorted-edge hash) — a provenance fingerprint
    invariant to row order but sensitive to ticker permutation / snapshot off-by-one."""
    ei = edge_index.cpu().numpy()
    n = ei.shape[1]
    pairs = sorted(set(zip(ei[0].tolist(), ei[1].tolist())))
    h = hash(tuple(pairs))
    return (n, len(pairs), h)


def make_block_correlation_fixture(num_stocks: int, num_days: int, block_sizes,
                                   rho: float = 0.9, seed: int = 0):
    """Synthetic returns with KNOWN correlated blocks: stocks within a block share a
    common factor (corr ~ rho), across blocks ~ 0. build_correlation_snapshots MUST recover
    the block structure; a ticker permutation or +/-1 snapshot off-by-one MUST break it.
    Returns a pd.DataFrame (T, N) of synthetic returns and the block label array."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    assert sum(block_sizes) == num_stocks, "block_sizes must sum to num_stocks"
    labels = np.concatenate([np.full(b, k) for k, b in enumerate(block_sizes)])
    R = np.zeros((num_days, num_stocks), dtype=np.float64)
    a = np.sqrt(rho)
    for k, b in enumerate(block_sizes):
        common = rng.standard_normal((num_days, 1))
        idio = rng.standard_normal((num_days, b))
        cols = np.where(labels == k)[0]
        R[:, cols] = a * common + np.sqrt(1 - rho) * idio
    dates = pd.date_range('2020-01-01', periods=num_days, freq='B')
    return pd.DataFrame(R, index=dates), labels


# ══════════════════════════════════════════════════════════════
# VERDICT / THRESHOLD LOGIC (consumed by analyze_sanity.py)
# ══════════════════════════════════════════════════════════════

def load_edge_lift_unit(edge_pairs_csv: str = 'artifacts/storya_e6_edge_ablation/edge_pairs_dm.csv') -> float:
    """The '1x' unit = max mean_delta_ic over the 3 alpha{2,3,4}-vs-alpha1 pairs, regime='full'.
    Provenance for the E1b 3x threshold. Falls back to EDGE_LIFT_UNIT_FALLBACK if CSV absent."""
    import os
    import pandas as pd
    if not os.path.exists(edge_pairs_csv):
        return EDGE_LIFT_UNIT_FALLBACK
    df = pd.read_csv(edge_pairs_csv)
    sub = df[(df['regime_condition'] == 'full') &
             (df['pair_id'].str.contains('vs_alpha1_corr_only'))]
    if len(sub) == 0:
        return EDGE_LIFT_UNIT_FALLBACK
    return float(sub['mean_delta_ic'].max())


def load_alpha1_baseline(results_csv: str = 'experiments/storya_e1_anchor/results.csv'):
    """Per (universe, model, fold, seed) IC_mean from the frozen alpha1 anchor run."""
    import pandas as pd
    df = pd.read_csv(results_csv)
    return df[['universe', 'model', 'seed', 'fold', 'IC_mean']].copy()


def _block_bootstrap_mean_ci(x: np.ndarray, block: int = 21, n_boot: int = 5000,
                             alpha: float = 0.05, seed: int = 42):
    """Stationary-ish moving-block bootstrap CI for the mean of a daily series `x`.
    Returns (lo, hi) two-sided (1-alpha) percentile CI. Used for E2 equivalence gate."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return (0.0, 0.0)
    if n <= block:
        block = max(1, n // 2)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block + 1
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        means[b] = x[idx].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (lo, hi)


def e1b_oracle_verdict(oracle_per_day_ics: dict, alpha1_baseline,
                       unit_lift: float, multiple: float = ORACLE_PASS_MULTIPLE) -> dict:
    """E1b SUPPORTING diagnostic (NOT a necessary control — smoke finding 2026-06-10 +
    H博士 directive: E3 is the sole decisive necessary control).

    A leaked label-co-movement topology transmits label info only SECOND-ORDER
    (lift ≈ feature-predictiveness × co-movement-tightness), so its lift is inherently
    modest — it can even sit below the alpha1 baseline on a perfectly healthy pipeline.
    Therefore NO 3× pass/fail gate (that would false-negative a healthy pipeline and
    re-commit the Codex A-03 error). Reported descriptively: direction + magnitude only.

    lift = oracle IC_mean - alpha1 IC_mean matched per (universe, model, fold, seed).
    """
    import pandas as pd
    a1 = alpha1_baseline.set_index(['universe', 'model', 'seed', 'fold'])['IC_mean'].to_dict()
    rows = []
    for (u, m, s, f), oic in oracle_per_day_ics.items():
        base = a1.get((u, m, int(s), int(f)))
        if base is None:
            continue
        rows.append({'universe': u, 'model': m, 'seed': int(s), 'fold': int(f),
                     'oracle_ic': oic, 'alpha1_ic': base, 'lift': oic - base})
    if not rows:
        return {'verdict': 'NO_DATA', 'role': 'supporting_diagnostic', 'mean_lift': None}
    df = pd.DataFrame(rows)
    mean_lift = float(df['lift'].mean())
    frac_pos = float((df['lift'] > 0).mean())
    verdict = 'SUPPORTING_POSITIVE' if mean_lift > 0 else 'SUPPORTING_NULL_OR_NEGATIVE'
    return {'verdict': verdict, 'role': 'supporting_diagnostic', 'mean_lift': mean_lift,
            'unit_lift': unit_lift, 'lift_in_units': (mean_lift / unit_lift) if unit_lift else None,
            'frac_seed_positive': frac_pos, 'n_cells': len(df), 'per_cell': df}


def e3_recovery_verdict(gnn_test_ic: float, mlp_test_ic: float, achievable_ic: float,
                        hln_p: float, bh_reject: bool,
                        recovery_fraction: float = E3_RECOVERY_FRACTION) -> dict:
    """E3 necessary control: pass if GNN test IC >= recovery_fraction*achievable AND
    GNN significantly > MLP. achievable = MEASURED oracle IC.

    Codex C-01 fix: 'significantly > MLP' uses the BH-FDR REJECT boolean directly, NOT a
    reconstructed p<threshold comparison (the latter strict-< failed the marginal rejected
    model, making it impossible for both GNNs to pass together)."""
    recovered = (achievable_ic > 1e-6) and (gnn_test_ic >= recovery_fraction * achievable_ic)
    beats_mlp = (gnn_test_ic > mlp_test_ic) and bool(bh_reject)
    verdict = 'PASS_PIPELINE_INNOCENT' if (recovered and beats_mlp) else 'FAIL_PIPELINE_SICK'
    return {'verdict': verdict, 'gnn_test_ic': gnn_test_ic, 'mlp_test_ic': mlp_test_ic,
            'achievable_ic': achievable_ic, 'recovery_target': recovery_fraction * achievable_ic,
            'recovered': bool(recovered), 'beats_mlp': bool(beats_mlp),
            'hln_p': hln_p, 'bh_reject': bool(bh_reject)}


def e2_shuffled_verdict(shuffled_minus_nograph_daily: np.ndarray, unit_lift: float,
                        tost_margin: float = E2_TOST_MARGIN, seed: int = 42) -> dict:
    """E2 equivalence gate (Codex A-05): paired daily (shuffled - no_graph) delta-IC.
    Approx no-graph if the two-sided 95% block-bootstrap CI of the mean lies within
    [-tost_margin, +tost_margin] (TOST) AND its upper bound < unit_lift. If the lower
    bound > 0 by more than the margin -> 'STRUCTURAL_REG' independent finding."""
    lo, hi = _block_bootstrap_mean_ci(shuffled_minus_nograph_daily, seed=seed)
    mean = float(np.mean(shuffled_minus_nograph_daily)) if len(shuffled_minus_nograph_daily) else 0.0
    equivalent = (hi < unit_lift) and (-tost_margin <= lo) and (hi <= tost_margin)
    structural = lo > tost_margin
    if equivalent:
        verdict = 'APPROX_NO_GRAPH'
    elif structural:
        verdict = 'STRUCTURAL_REGULARIZATION'
    else:
        verdict = 'INCONCLUSIVE'
    return {'verdict': verdict, 'mean_delta': mean, 'ci95': (lo, hi),
            'unit_lift': unit_lift, 'tost_margin': tost_margin}
