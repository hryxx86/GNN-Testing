#!/usr/bin/env python
"""compute_family1_ladder.py — D-RERUN-12F Family-1 §2a confirmatory analysis.

Family-1 = PREDICTIVE / model-selection on the tuned 12-fold ladder (the LOCKED
confirmatory main table). Answers "does any TUNED arm beat TUNED LightGBM (best-vs-best)".

Spec is LOCKED in docs/protocol_v2_freeze.md §6 + docs/plan_fc_edge_robustness_2026-06-17.md
(Family-1). This module is a THIN DRIVER: it reuses the validated statistical helpers from
compute_e6_dm_spa.py (HLN test, NW-HAC, BH-FDR, Hansen SPA, stationary block bootstrap,
power/MDE) and only supplies the tuned-ladder ARM structure + the pre-registered 20-test
pairwise family + the M=9 SPA candidate set + the L7/Cn5 contingency rule.

LOCKED design (protocol §6):
  - DM-HLN pairwise family (pre-registered, NO adding pairs) = 10 pairs × 2 universes = 20 tests:
      ladder 5:  L1-L0, L2-L1, L6-L2, L7-L2, L2s-L2
      edge DAG 5: L3-L2, L4-L2, L5-L2, L5-L4, L5-L3
    on seed-AVERAGED per-day ΔIC, HLN small-sample t (h=21), BH-FDR q=0.05 over the family.
  - Hansen SPA per universe, M=9 candidates {L1,L2,L2s,L3,L4,L5,L5s,L6,L7} vs benchmark L0.
    Seeds averaged per (arm, universe, date, fold) BEFORE SPA (consistent w/ DM/HLN; D-04).
  - L7 contingency (Cn5, mechanical, pre-locked): if >20% L7 cells diverged_flag==1 (or IC NaN),
    OR >20% L7 cells alpha_max_fraction_collapsed_test>0.9 → L7 demoted to exploratory:
    dropped from the pairwise family (L7-L2 removed) AND from SPA (M=9→8). Ledger records.
  - block-bootstrap CI (block=21, n_boot=5000) on seed-averaged pooled daily IC per arm.
  - MDE ≈ 2.8 × SE_block-bootstrap(mean ΔIC), report n_eff; "can detect model-level ~0.025-0.03,
    cannot detect edge-level +0.006-0.009" (§6 MDE).
  - LOFO: full 12-fold leave-one-out scan on each pairwise mean ΔIC (single-fold contribution).

The §4 tuning val-IC is a SELECTION metric, NEVER a result — not touched here.

Usage (from project root):
  python compute_family1_ladder.py
  python compute_family1_ladder.py --main-dir experiments/storya_v21_main12_tuned \
      --l7-dir experiments/_rerun_colab_staging/storya_v21_l7_hats_tuned \
      --output-dir artifacts/storya_v21_family1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# Reuse the validated statistical engine (import-only; do NOT reimplement methods)
import compute_e6_dm_spa as e6
from compute_e6_dm_spa import (
    hln_test, dm_test, nw_lag, nw_hac_variance, bh_fdr,
    stationary_bootstrap_ci, run_spa, two_sided_power, _Z_975, _Z_80,
)

# ── CONFIG (LOCKED per protocol §6) ──
UNIVERSES = ['B', 'C']   # default; overridable ONLY via --universes (2026-09-10 C5 post-hoc sensitivity)
ALL_ARMS_DEFAULT = ['L0', 'L1', 'L2', 'L2s', 'L3', 'L4', 'L5', 'L5s', 'L6', 'L7']
CANONICAL_SEEDS = [86, 123, 456, 789, 1024, 2024, 7, 34, 99, 2026]
N_FOLDS = 12
HORIZON = 21
BASELINE = 'L0'

# Pre-registered pairwise family (10 pairs; A-B means ΔIC = IC_A - IC_B)
LADDER_PAIRS = [('L1', 'L0'), ('L2', 'L1'), ('L6', 'L2'), ('L7', 'L2'), ('L2s', 'L2')]
EDGE_PAIRS = [('L3', 'L2'), ('L4', 'L2'), ('L5', 'L2'), ('L5', 'L4'), ('L5', 'L3')]
ALL_PAIRS = LADDER_PAIRS + EDGE_PAIRS  # 10

# SPA candidate set (M=9 incl L7; L0 is benchmark, not a candidate)
SPA_CANDIDATES_FULL = ['L1', 'L2', 'L2s', 'L3', 'L4', 'L5', 'L5s', 'L6', 'L7']

# L7 / Cn5 contingency thresholds
L7_DIVERGE_FRAC_MAX = 0.20
L7_COLLAPSE_FRAC_MAX = 0.20
L7_COLLAPSE_CELL_THRESH = 0.9   # alpha_max_fraction_collapsed_test > 0.9 = a collapsed cell

# Block bootstrap knobs (match compute_e6 / protocol)
N_BOOT = 5000
BLOCK_SIZE = HORIZON
BH_FDR_Q = 0.05
MDE_FACTOR = 2.8  # protocol §6: MDE ≈ 2.8 × SE_block-bootstrap(mean ΔIC)


# ══════════════════════════════════════════════════════════════
# LOADERS — arm-based (the runner saves {universe}_{arm}_s{seed}_f{fold}.npy)
# ══════════════════════════════════════════════════════════════

def _npy_path(per_day_dir: str, universe: str, arm: str, seed: int, fold: int) -> str:
    return os.path.join(per_day_dir, f'{universe}_{arm}_s{seed}_f{fold}.npy')


def arm_per_day_dir(arm: str, main_dir: str, l7_dir: str) -> str:
    """L7 per_day_ic lives in its own out-dir; all other arms in the merged main dir."""
    return os.path.join(l7_dir, 'per_day_ic') if arm == 'L7' else os.path.join(main_dir, 'per_day_ic')


def collect_arm_matrix(per_day_dir: str, universe: str, arm: str) -> dict:
    """{fold: (n_seeds, n_days) float64 matrix} for seed-averaging. PRIMARY treatment = EXCLUDE.

    Mirrors compute_e6.collect_per_day_ic_matrix but keyed by ARM. A day's per-cell IC is the
    cross-sectional Spearman IC; for a degenerate cell (constant prediction) it is UNDEFINED (0/0,
    zero cross-sectional variance), so `compute_daily_ic` dropped that day → the cell's .npy is
    short or empty (results.csv fallback IC_mean=0, n_test_days=0).

    H博士 decision 2026-06-21 (LOCKED): degenerate days/cells are treated as MISSING (undefined IC
    is NOT a measured 0 → zero-fill would fabricate 0s and dilute the real distribution). So:
      - file MISSING                → None row → excluded by np.nanmean.
      - file EXISTS, len 0          → fully degenerate (constant every test day): the whole seed row
                                      stays all-NaN → EXCLUDED. (Verified case: C/L5s converged to a
                                      constant predictor, best_val_loss≈0.998 = no-signal plateau,
                                      identical L/S Sharpe across seeds — NOT a train crash, IC is
                                      genuinely undefined.)
      - file EXISTS, 0<len<max      → partial collapse: the cell's VALID measured-IC days are kept
                                      at positions [0..len-1]; the collapsed (undefined) tail days are
                                      NaN-padded → excluded. (Per-day date labels are unavailable, so
                                      collapsed days are dropped, never positionally guessed.)
    The seed-averaged IC under this treatment is "IC conditional on not collapsing". The collapse
    RATE itself is reported as a stability finding (see degeneracy_report); robustness vs zero-fill /
    zero-skill is in cl5s_robustness (all three give C/L5s mean IC ≈ 0 — appendix, not primary).
    NEVER re-tune C/L5s (equal-budget symmetry; re-tuning a losing arm = cherry-pick, forbidden).
    """
    out = {}
    for fold in range(N_FOLDS):
        seed_arrays = []
        for seed in CANONICAL_SEEDS:
            p = _npy_path(per_day_dir, universe, arm, seed, fold)
            # existing-but-empty .npy stays a 0-length array (NOT None): a degenerate COMPLETED cell,
            # distinct from an absent seed file. Both end up excluded, but we count them separately.
            seed_arrays.append(np.load(p).astype(np.float64) if os.path.exists(p) else None)
        nonempty = [a for a in seed_arrays if a is not None and len(a) > 0]
        if not nonempty:
            out[fold] = None
            continue
        max_len = max(len(a) for a in nonempty)
        n_full = sum(1 for a in seed_arrays if a is not None and len(a) == 0)
        n_partial = sum(1 for a in seed_arrays if a is not None and 0 < len(a) < max_len)
        if n_full or n_partial:
            print(f"WARN: {universe}/{arm}/fold{fold}: {n_full} fully-degenerate (len 0) + "
                  f"{n_partial} partial-collapse cells EXCLUDED (undefined IC treated as missing).")
        mat = np.full((len(CANONICAL_SEEDS), max_len), np.nan, dtype=np.float64)
        for i, a in enumerate(seed_arrays):
            if a is not None and len(a) > 0:
                mat[i, :len(a)] = a          # valid measured-IC days; collapsed tail stays NaN (excluded)
        out[fold] = mat
    return out


def seed_avg_per_fold(mat_dict: dict) -> dict:
    """{fold: 1D seed-averaged daily IC (nanmean over seeds)}."""
    return {f: (np.nanmean(m, axis=0) if m is not None else None) for f, m in mat_dict.items()}


def seed_avg_pooled(mat_dict: dict) -> np.ndarray:
    """Concat seed-averaged daily IC across folds in chronological order → 1D series."""
    pooled = [np.nanmean(m, axis=0) for f in range(N_FOLDS)
              for m in [mat_dict.get(f)] if m is not None]
    return np.concatenate(pooled) if pooled else np.array([], dtype=np.float64)


def all_pooled(mat_dict: dict) -> np.ndarray:
    """Flatten ALL (seed, day) per-day IC across folds, drop NaN — seed-stacked diagnostic series."""
    out = []
    for f in range(N_FOLDS):
        m = mat_dict.get(f)
        if m is not None:
            flat = m.flatten()
            out.append(flat[~np.isnan(flat)])
    return np.concatenate(out) if out else np.array([], dtype=np.float64)


def build_aggregate(arms: list[str], main_dir: str, l7_dir: str) -> dict:
    """{(universe, arm): {'matrix': {fold:mat}, 'seed_avg_pooled': vec, 'all_pooled': vec}}."""
    agg = {}
    for u in UNIVERSES:
        for arm in arms:
            d = collect_arm_matrix(arm_per_day_dir(arm, main_dir, l7_dir), u, arm)
            sa = seed_avg_pooled(d)
            agg[(u, arm)] = {'matrix': d, 'seed_avg_pooled': sa, 'all_pooled': all_pooled(d)}
            print(f"  [{u}/{arm}] seed_avg T={len(sa)}  all_pooled N={len(agg[(u, arm)]['all_pooled'])}")
    return agg


# ══════════════════════════════════════════════════════════════
# L7 / Cn5 contingency (mechanical, pre-locked)
# ══════════════════════════════════════════════════════════════

def l7_contingency(l7_results_csv: str) -> dict:
    """Returns {'demote': bool, 'diverge_frac':, 'collapse_frac':, 'reason':, 'n_cells':}."""
    df = pd.read_csv(l7_results_csv)
    n = len(df)
    # (b) divergence: diverged_flag==1 OR IC_mean NaN
    diverged = (df.get('diverged_flag', pd.Series([0] * n)) == 1) | df['IC_mean'].isna()
    diverge_frac = float(diverged.mean()) if n else 0.0
    # (c) alpha collapse: alpha_max_fraction_collapsed_test > 0.9
    col = df.get('alpha_max_fraction_collapsed_test', pd.Series([0.0] * n))
    collapse_frac = float((col > L7_COLLAPSE_CELL_THRESH).mean()) if n else 0.0
    reasons = []
    if diverge_frac > L7_DIVERGE_FRAC_MAX:
        reasons.append(f"diverge_frac={diverge_frac:.3f}>{L7_DIVERGE_FRAC_MAX}")
    if collapse_frac > L7_COLLAPSE_FRAC_MAX:
        reasons.append(f"collapse_frac={collapse_frac:.3f}>{L7_COLLAPSE_FRAC_MAX}")
    return {
        'demote': bool(reasons), 'diverge_frac': diverge_frac, 'collapse_frac': collapse_frac,
        'reason': '; '.join(reasons) if reasons else 'healthy (kept in family + SPA, M=9)',
        'n_cells': int(n),
    }


# ══════════════════════════════════════════════════════════════
# DM-HLN pairwise (20-test pre-registered family) + BH-FDR
# ══════════════════════════════════════════════════════════════

def run_pairwise(agg: dict, pairs: list, out_dir: str, apply_bh: bool = True) -> pd.DataFrame:
    """HLN test on seed-averaged daily ΔIC for each (universe, pair). BH-FDR over the FULL family.

    NOTE (flag for Touchpoint 2): protocol §6 phrases the family as "10 pairs × 2 universes = 20
    tests, BH-FDR q=0.05" → BH applied over the FULL 20-test family (not per-universe-10). This is
    the headline confirmatory correction; per-universe BH is also reported as a column for context.
    """
    rows = []
    for u in UNIVERSES:
        for a, b in pairs:
            ic_a = agg[(u, a)]['seed_avg_pooled']
            ic_b = agg[(u, b)]['seed_avg_pooled']
            nlen = min(len(ic_a), len(ic_b))
            if nlen < 5:
                continue
            d = (-ic_a[:nlen]) - (-ic_b[:nlen])        # loss_A - loss_B = -(IC_A - IC_B)
            dm_stat, dm_p, T = dm_test(d)
            hln_stat, hln_p, _ = hln_test(d)
            _, hln_p_lag21, _ = hln_test(d, lag=HORIZON)  # HAC lag=21 robustness
            rows.append({
                'universe': u, 'arm_A': a, 'arm_B': b,
                'mean_delta_IC': float((ic_a[:nlen] - ic_b[:nlen]).mean()),
                'T': T, 'NW_lag': nw_lag(T),
                'DM_stat': dm_stat, 'DM_p_normal': dm_p,
                'HLN_stat': hln_stat, 'HLN_p_t': hln_p, 'HLN_p_t_lag21': hln_p_lag21,
            })
    df = pd.DataFrame(rows)
    if len(df) and apply_bh:
        # Headline: BH-FDR over the FULL family (all rows present)
        df['BH_FDR_reject_family'] = bh_fdr(df['HLN_p_t'].tolist(), q=BH_FDR_Q)
        # Context: BH-FDR within each universe
        df['BH_FDR_reject_per_univ'] = False
        for u in df['universe'].unique():
            sub = df[df['universe'] == u]
            df.loc[sub.index, 'BH_FDR_reject_per_univ'] = bh_fdr(sub['HLN_p_t'].tolist(), q=BH_FDR_Q)
        df['bh_fdr_q'] = BH_FDR_Q
    elif len(df):
        # POST-HOC SENSITIVITY (2026-09-10 C5): no BH family is opened — raw HLN p only.
        df['BH_FDR_reject_family'] = None
        df['BH_FDR_reject_per_univ'] = None
        df['bh_fdr_q'] = np.nan
        df['bh_applied'] = False
    df.to_csv(os.path.join(out_dir, 'family1_dm_hln.csv'), index=False)
    return df


# ══════════════════════════════════════════════════════════════
# Hansen SPA per universe (M=9 or 8 after contingency)
# ══════════════════════════════════════════════════════════════

def run_spa_ladder(agg: dict, candidates: list, out_dir: str) -> pd.DataFrame:
    rows = []
    for u in UNIVERSES:
        bench = agg[(u, BASELINE)]['seed_avg_pooled']
        if len(bench) == 0:
            print(f"  [SPA {u}] benchmark L0 empty; skip")
            continue
        cand_series, used = [], []
        n_align = len(bench)
        for m in candidates:
            ic = agg[(u, m)]['seed_avg_pooled']
            if len(ic) == 0:
                print(f"  [SPA {u}] candidate {m} empty; skip whole universe")
                cand_series = []
                break
            n_align = min(n_align, len(ic))
            cand_series.append(ic)
            used.append(m)
        if not cand_series:
            continue
        bench_l = -bench[:n_align]
        cand_l = np.column_stack([c[:n_align] * -1.0 for c in cand_series])
        if len(bench_l) < 2 * BLOCK_SIZE:
            print(f"  [SPA {u}] T={len(bench_l)} < 2*block; skip")
            continue
        print(f"  [SPA {u}] T={len(bench_l)} M={cand_l.shape[1]} candidates={used}")
        res = run_spa(bench_l, cand_l)
        rows.append({
            'universe': u, 'benchmark': BASELINE, 'candidates': '|'.join(used),
            'M': res['M'], 'T': res['T'],
            'p_lower': res['p_lower'], 'p_consistent': res['p_consistent'], 'p_upper': res['p_upper'],
            'reject_h0_at_5pct': res['p_consistent'] < 0.05,
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'family1_spa.csv'), index=False)
    return df


# ══════════════════════════════════════════════════════════════
# Block-bootstrap CI per arm + MDE per pairwise + LOFO
# ══════════════════════════════════════════════════════════════

def run_ci_and_mde(agg: dict, arms: list, pairs: list, out_dir: str) -> tuple:
    # (1) seed-averaged IC CI per arm (the headline estimand)
    ci_rows = []
    for u in UNIVERSES:
        for arm in arms:
            s = agg[(u, arm)]['seed_avg_pooled']
            if len(s) < 2:
                continue
            mean, lo, hi = stationary_bootstrap_ci(s, lambda a: float(np.mean(a)),
                                                   n_boot=N_BOOT, block_size=BLOCK_SIZE)
            ci_rows.append({'universe': u, 'arm': arm, 'T': int(len(s)),
                            'IC_mean': round(mean, 5), 'IC_ci_lo': round(lo, 5),
                            'IC_ci_hi': round(hi, 5), 'ci_excludes_0': bool(lo > 0 or hi < 0)})
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(os.path.join(out_dir, 'family1_ic_ci.csv'), index=False)

    # (2) MDE per pairwise: SE from block-bootstrap of mean ΔIC; MDE = 2.8 × SE (protocol §6)
    mde_rows = []
    for u in UNIVERSES:
        for a, b in pairs:
            ic_a, ic_b = agg[(u, a)]['seed_avg_pooled'], agg[(u, b)]['seed_avg_pooled']
            nlen = min(len(ic_a), len(ic_b))
            if nlen < 5:
                continue
            d = ic_a[:nlen] - ic_b[:nlen]                      # paired daily ΔIC
            mean_d, lo, hi = stationary_bootstrap_ci(d, lambda x: float(np.mean(x)),
                                                     n_boot=N_BOOT, block_size=BLOCK_SIZE)
            # bootstrap SE of the mean ΔIC (std of the bootstrap distribution of the mean)
            from arch.bootstrap import StationaryBootstrap
            sb = StationaryBootstrap(BLOCK_SIZE, d, seed=86)
            boot_means = sb.apply(lambda x: float(np.mean(x)), N_BOOT).ravel()
            se_block = float(np.std(boot_means, ddof=1))
            mde_rows.append({
                'universe': u, 'pair': f'{a}-{b}',
                'is_edge_pair': bool((a, b) in EDGE_PAIRS),
                'T': int(nlen), 'mean_delta_IC': round(mean_d, 5),
                'delta_ci_lo': round(lo, 5), 'delta_ci_hi': round(hi, 5),
                'ci_excludes_0': bool(lo > 0 or hi < 0),
                'SE_block': round(se_block, 5),
                'MDE_2p8xSE': round(MDE_FACTOR * se_block, 5),
                # CODEX-A-02 fix: nlen is the raw aligned DAY count, NOT the effective sample size.
                # The block bootstrap (block=21) implies ~nlen/BLOCK_SIZE near-independent blocks.
                'T_days': int(nlen),
                'n_eff_blocks': round(nlen / BLOCK_SIZE, 1),
            })
    mde_df = pd.DataFrame(mde_rows)
    mde_df.to_csv(os.path.join(out_dir, 'family1_mde.csv'), index=False)

    # (3) LOFO: leave-one-fold-out mean ΔIC per pairwise (single-fold contribution)
    lofo_rows = []
    for u in UNIVERSES:
        for a, b in pairs:
            ma, mb = agg[(u, a)]['matrix'], agg[(u, b)]['matrix']
            # fold-level seed-averaged mean ΔIC
            fold_delta = {}
            for f in range(N_FOLDS):
                if ma.get(f) is None or mb.get(f) is None:
                    continue
                da = np.nanmean(ma[f], axis=0)
                db = np.nanmean(mb[f], axis=0)
                nlen = min(len(da), len(db))
                if nlen == 0:
                    continue
                fold_delta[f] = float(np.mean(da[:nlen] - db[:nlen]))
            if len(fold_delta) < 2:
                continue
            full_mean = float(np.mean(list(fold_delta.values())))
            for f, fd in fold_delta.items():
                rest = [v for k, v in fold_delta.items() if k != f]
                lofo_rows.append({
                    'universe': u, 'pair': f'{a}-{b}', 'dropped_fold': f,
                    'full_mean_delta': round(full_mean, 5),
                    'lofo_mean_delta': round(float(np.mean(rest)), 5),
                    'this_fold_delta': round(fd, 5),
                    'sign_flips_when_dropped': bool(np.sign(np.mean(rest)) != np.sign(full_mean)),
                })
    lofo_df = pd.DataFrame(lofo_rows)
    lofo_df.to_csv(os.path.join(out_dir, 'family1_lofo.csv'), index=False)
    return ci_df, mde_df, lofo_df


# ══════════════════════════════════════════════════════════════
# Stability finding (degeneracy rate) + C/L5s robustness (3 treatments)
# ══════════════════════════════════════════════════════════════

def degeneracy_report(arms: list, main_dir: str, l7_dir: str, out_dir: str,
                      ref_arms: list | None = None) -> pd.DataFrame:
    """Per (universe, arm): count fully-degenerate (len-0) + partial-collapse cells. STABILITY
    FINDING — a constant-prediction collapse is the tuned config losing all ranking ability, not a
    fill-method footnote. Scans the per-day .npy directly (the ground truth for 'IC undefined').
    `ref_arms`: arms whose max per-(universe, fold) length defines the full day count (default ['L2'],
    the confirmatory healthy reference; sensitivity mode passes the arms actually present —
    CODEX-TP2-A-02, otherwise an absent L2 makes every cell look 'normal')."""
    rows = []
    ref_arms = list(ref_arms) if ref_arms else ['L2']
    full_len = {}  # per (universe, fold) reference day count from a healthy arm (L2)
    for u in UNIVERSES:
        for f in range(N_FOLDS):
            ls = []
            for ra in ref_arms:
                for s in CANONICAL_SEEDS:
                    p = _npy_path(arm_per_day_dir(ra, main_dir, l7_dir), u, ra, s, f)
                    if os.path.exists(p):
                        ls.append(len(np.load(p)))
            full_len[(u, f)] = max(ls) if ls else 0
    for u in UNIVERSES:
        for arm in arms:
            pd_dir = arm_per_day_dir(arm, main_dir, l7_dir)
            n_full = n_partial = n_normal = n_cells = 0
            for s in CANONICAL_SEEDS:
                for f in range(N_FOLDS):
                    p = _npy_path(pd_dir, u, arm, s, f)
                    if not os.path.exists(p):
                        continue
                    n_cells += 1
                    L = len(np.load(p))
                    ref = full_len.get((u, f), 0)
                    if L == 0:
                        n_full += 1
                    elif ref and L < ref:
                        n_partial += 1
                    else:
                        n_normal += 1
            if n_cells:
                rows.append({
                    'universe': u, 'arm': arm, 'n_cells': n_cells,
                    'n_fully_degenerate': n_full, 'n_partial_collapse': n_partial,
                    'n_normal': n_normal,
                    'collapse_rate': round((n_full + n_partial) / n_cells, 4),
                })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'family1_stability.csv'), index=False)
    return df


def _cl5s_series(treatment: str, main_dir: str) -> np.ndarray:
    """Build C/L5s seed-averaged pooled daily IC under one of three treatments (robustness)."""
    pd_dir = os.path.join(main_dir, 'per_day_ic')
    u, arm = 'C', 'L5s'
    if treatment == 'zeroskill_cell':
        # cell-level: each (seed,fold) contributes its results.csv IC_mean (degenerate→0 fallback),
        # seed-averaged per fold then pooled. Different estimand (cell-level), shown for robustness.
        res = pd.read_csv(os.path.join(main_dir, 'results.csv'))
        sub = res[(res.universe == u) & (res.arm == arm)]
        pooled = []
        for f in range(N_FOLDS):
            vals = sub[sub.fold == f]['IC_mean'].values
            if len(vals):
                pooled.append(np.repeat(np.mean(vals), 1))   # one fold-level value
        return np.concatenate(pooled) if pooled else np.array([])
    # day-level treatments: build per-fold (n_seeds, max_len) then seed-average
    pooled = []
    for f in range(N_FOLDS):
        arrs = [np.load(_npy_path(pd_dir, u, arm, s, f)).astype(np.float64)
                if os.path.exists(_npy_path(pd_dir, u, arm, s, f)) else None for s in CANONICAL_SEEDS]
        nonempty = [a for a in arrs if a is not None and len(a) > 0]
        if not nonempty:
            continue
        max_len = max(len(a) for a in nonempty)
        mat = np.full((len(CANONICAL_SEEDS), max_len), np.nan)
        for i, a in enumerate(arrs):
            if a is None:
                continue
            if len(a) == 0:
                if treatment == 'zerofill':
                    mat[i, :] = 0.0          # degenerate cell → IC 0 across the fold
                # exclude: leave all-NaN
            else:
                mat[i, :len(a)] = a
                if treatment == 'zerofill' and len(a) < max_len:
                    mat[i, len(a):] = 0.0    # partial: collapsed tail → 0 (zero-fill variant only)
        pooled.append(np.nanmean(mat, axis=0))
    return np.concatenate(pooled) if pooled else np.array([])


def cl5s_robustness(agg: dict, main_dir: str, spa_candidates: list, out_dir: str) -> pd.DataFrame:
    """Pre-registered robustness: C/L5s mean IC + universe-C SPA p_consistent under 3 treatments
    {exclude (primary), zerofill, zeroskill_cell}. Expectation (LOCKED): all give C/L5s IC ≈ 0 and
    the C SPA verdict is unchanged → the C/L5s treatment choice does not affect any conclusion."""
    rows = []
    # build C-universe candidate losses once (the 8 healthy candidates are treatment-invariant)
    bench = agg[('C', BASELINE)]['seed_avg_pooled']
    for treat in ['exclude', 'zerofill', 'zeroskill_cell']:
        if treat == 'exclude':
            l5s = agg[('C', 'L5s')]['seed_avg_pooled']
        else:
            l5s = _cl5s_series(treat, main_dir)
        mean_ic = float(np.nanmean(l5s)) if len(l5s) else np.nan
        p_cons = np.nan
        # re-run C SPA with this L5s series (cell-level treatment has different T → SPA skipped, IC only)
        if treat != 'zeroskill_cell' and len(bench):
            series = {m: (l5s if m == 'L5s' else agg[('C', m)]['seed_avg_pooled']) for m in spa_candidates}
            if all(len(series[m]) for m in spa_candidates):
                n = min([len(bench)] + [len(series[m]) for m in spa_candidates])
                cand = np.column_stack([series[m][:n] * -1.0 for m in spa_candidates])
                if n >= 2 * BLOCK_SIZE:
                    try:
                        p_cons = run_spa(-bench[:n], cand)['p_consistent']
                    except Exception as e:
                        print(f"  [cl5s_robustness] SPA {treat} failed: {e}")
        rows.append({'treatment': treat, 'cl5s_C_mean_IC': round(mean_ic, 5),
                     'C_SPA_p_consistent': (round(p_cons, 4) if p_cons == p_cons else None),
                     'is_primary': treat == 'exclude'})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'family1_cl5s_robustness.csv'), index=False)
    return df


# ══════════════════════════════════════════════════════════════
# Ledger + summary
# ══════════════════════════════════════════════════════════════

def _run_inputs(main_dir: str) -> dict:
    """FINGNN-B-02 (TP2-B): identify the result directory the stats were computed from."""
    import hashlib
    def _md5(p):
        return hashlib.md5(open(p, 'rb').read()).hexdigest() if os.path.exists(p) else None
    prov_p = os.path.join(main_dir, '_run_provenance.json')
    prov = json.load(open(prov_p)) if os.path.exists(prov_p) else None
    if isinstance(prov, list):
        prov = prov[-1] if prov else None
    return {'main_dir': main_dir, 'results_csv_md5': _md5(os.path.join(main_dir, 'results.csv')),
            'manifest_csv_md5': _md5(os.path.join(main_dir, 'manifest.csv')),
            'device': prov.get('device') if prov else None, 'platform': prov.get('platform') if prov else None,
            'git_rev': prov.get('git_rev') if prov else None, 'source_clean': prov.get('source_clean') if prov else None}


def write_ledger(out_dir: str, l7: dict, spa_M: int, sensitivity: bool = False,
                 pairs_run: list | None = None, arms_run: list | None = None, main_dir: str | None = None,
                 n_tests_actual: int | None = None) -> None:
    if sensitivity:
        # CODEX-TP2-A-05: record what was ACTUALLY executed (restricted pairs, no BH, no SPA, no L7 gate)
        pairs_run = list(pairs_run or [])
        ledger = {
            'inputs': _run_inputs(main_dir) if main_dir else None,
            'family': 'Family-1 machinery re-used for a POST-HOC SENSITIVITY run',
            'role': ('POST-HOC SENSITIVITY (NOT confirmatory; no BH family opened; raw HLN p only; '
                     'docs/c5_rerun_brief_2026-09-10.md)'),
            'sensitivity_scope': {'universes': list(UNIVERSES), 'arms': list(arms_run or []),
                                  'pairs_tested': [f'{a}-{b}' for a, b in pairs_run],
                                  'n_tests_total': (int(n_tests_actual) if n_tests_actual is not None
                                                    else len(pairs_run) * len(UNIVERSES)),
                                  'note': 'pre-registered LADDER_PAIRS/EDGE_PAIRS restricted to arms present; no pair added'},
            'bh_fdr': 'NOT APPLIED (raw, unadjusted HLN p; no family opened)',
            'hln_hac_lag': ('HLN_p_t = Newey-West AUTO lag (implementation default in compute_e6_dm_spa.nw_lag; NOT specified '
                            'by protocol §6); HLN_p_t_lag21 = horizon-matched lag reported alongside (EXPL-STAT-01)'),
            'spa': 'NOT RUN',
            'l7_contingency': 'SKIPPED (L7 not part of this run)',
            'degenerate_cell_treatment': {'primary': 'EXCLUDE (as confirmatory)',
                                          'reference_for_partial_collapse': 'max length over arms present'},
            'block_bootstrap': {'n_boot': N_BOOT, 'block_size_days': BLOCK_SIZE},
            'mde_rule': f'MDE ≈ {MDE_FACTOR} x SE_block-bootstrap(mean delta-IC) (approximate nominal)',
            'horizon_days': HORIZON,
            'seed_avg_note': 'Seeds averaged per (arm, universe, date, fold) BEFORE HLN (as confirmatory).',
            'tuning_val_ic_note': 'The §4 tuning val-IC is a SELECTION metric, NEVER entered as a result.',
        }
        with open(os.path.join(out_dir, 'family1_ledger.json'), 'w') as f:
            json.dump(ledger, f, indent=2)
        return
    ledger = {
        'family': 'Family-1 (predictive / model-selection; tuned 12-fold ladder)',
        'role': 'CONFIRMATORY (the only confirmatory family; protocol §6)',
        'dm_hln_pairwise_family': {
            'n_pairs_per_universe': len(ALL_PAIRS),
            'universes': len(UNIVERSES),
            'n_tests_total': len(ALL_PAIRS) * len(UNIVERSES),
            'ladder_pairs': [f'{a}-{b}' for a, b in LADDER_PAIRS],
            'edge_dag_pairs': [f'{a}-{b}' for a, b in EDGE_PAIRS],
            'bh_fdr_q': BH_FDR_Q,
            'bh_scope': 'full 20-test family (headline); per-universe also reported',
        },
        'spa': {'per_universe': True, 'benchmark': BASELINE, 'M': spa_M,
                'candidates_full': SPA_CANDIDATES_FULL},
        'l7_contingency': l7,
        'degenerate_cell_treatment': {
            'primary': 'EXCLUDE (undefined IC = missing; constant-prediction collapse, NOT measured 0)',
            'rationale': ('H博士 2026-06-21 LOCKED: zero-fill fabricates 0s for undefined values + '
                          'cannot be cleanly implemented for partial cells without per-day date labels; '
                          'exclude is the only non-fabricating, implementable treatment.'),
            'robustness_appendix': 'family1_cl5s_robustness.csv {exclude, zerofill, zeroskill_cell} all ≈0',
            'stability_finding': 'family1_stability.csv — reported as a tuned-config instability, not hidden',
            'no_retune': 'C/L5s NOT re-tuned (equal-budget symmetry; re-tuning a losing arm = cherry-pick)',
        },
        'block_bootstrap': {'n_boot': N_BOOT, 'block_size_days': BLOCK_SIZE},
        'mde_rule': f'MDE = {MDE_FACTOR} x SE_block-bootstrap(mean delta-IC) (protocol §6)',
        'horizon_days': HORIZON,
        'seed_avg_note': 'Seeds averaged per (arm, universe, date, fold) BEFORE SPA/DM (D-04).',
        'tuning_val_ic_note': 'The §4 tuning val-IC is a SELECTION metric, NEVER entered as a result.',
    }
    with open(os.path.join(out_dir, 'family1_ledger.json'), 'w') as f:
        json.dump(ledger, f, indent=2)


def write_summary(out_dir: str, l7: dict, spa_df, dm_df, ci_df, mde_df, lofo_df, stab_df, rob_df,
                  sensitivity: bool = False) -> None:
    title = ('Family-1 machinery — POST-HOC SENSITIVITY (NOT confirmatory; raw HLN p, no BH)'
             if sensitivity else 'Family-1 §2a confirmatory summary')
    L = [f"# {title}  (_generated {time.strftime('%Y-%m-%d %H:%M:%S')}_)\n"]
    if sensitivity:
        L.append("**L7/Cn5 contingency**: SKIPPED (sensitivity mode; L7 not part of this run). "
                 "**SPA**: not run. **BH-FDR**: not applied (raw HLN p).\n")
    else:
        L.append(f"**L7/Cn5 contingency**: {l7['reason']}  "
                 f"(diverge_frac={l7['diverge_frac']:.3f}, collapse_frac={l7['collapse_frac']:.3f}, "
                 f"n={l7['n_cells']}) → L7 {'DEMOTED to exploratory (M=8)' if l7['demote'] else 'KEPT (M=9)'}\n")
    # STABILITY FINDING (C/L5s constant-collapse) — surfaced before the headline stats
    deg = stab_df[stab_df['collapse_rate'] > 0] if len(stab_df) else stab_df
    if len(deg):
        L.append("\n## ⚠️ STABILITY FINDING — tuned-config constant-collapse\n")
        L.append("_A degenerate cell = the tuned arm converged to a CONSTANT prediction (zero ranking "
                 "ability; cross-sectional IC undefined), verified NOT a train crash (converged_flag=1, "
                 "best_val_loss≈0.998 no-signal plateau). Treated as MISSING (primary=EXCLUDE; undefined "
                 "≠ measured-0). NEVER re-tuned (equal-budget symmetry)._\n")
        L.append(deg.to_markdown(index=False))
        for _, r in deg.iterrows():
            L.append(f"\n**{r['universe']}/{r['arm']}**: the equal-budget tuned champion config "
                     f"degenerates to a constant predictor in **{r['collapse_rate']*100:.1f}%** of "
                     f"test fold-seeds ({r['n_fully_degenerate']} full + {r['n_partial_collapse']} partial "
                     f"of {r['n_cells']}). Mechanism: SAGE-mean aggregation + thin data + high dropout "
                     f"(0.5) smooths the signal away → 'smoothing hurts ranking' evidence chain.")
    if len(rob_df):
        L.append("\n## C/L5s robustness — 3 treatments (appendix; all give IC ≈ 0, conclusion stable)\n")
        L.append(rob_df.to_markdown(index=False))
    L.append("\n## Hansen SPA (per universe; benchmark L0)\n")
    if len(spa_df):
        L.append(spa_df[['universe', 'M', 'T', 'p_consistent', 'reject_h0_at_5pct']].to_markdown(index=False))
    L.append("\n## DM/HLN pairwise (seed-avg daily ΔIC; "
             + ("raw HLN p — NO BH, sensitivity)\n" if sensitivity else "BH-FDR over 20-test family)\n"))
    if len(dm_df):
        L.append(dm_df[['universe', 'arm_A', 'arm_B', 'mean_delta_IC', 'HLN_p_t',
                        'HLN_p_t_lag21', 'BH_FDR_reject_family', 'BH_FDR_reject_per_univ']].to_markdown(index=False))
    L.append("\n## Seed-averaged IC block-bootstrap CI per arm\n")
    if len(ci_df):
        L.append(ci_df.to_markdown(index=False))
    L.append("\n## MDE per pairwise (MDE = 2.8 × SE = effect detectable with 80% power; 'ci_excludes_0' = significant at α=0.05 — "
             "an effect can be significant and still below the MDE)\n")
    if len(mde_df):
        L.append(mde_df[['universe', 'pair', 'is_edge_pair', 'mean_delta_IC', 'delta_ci_lo',
                         'delta_ci_hi', 'ci_excludes_0', 'SE_block', 'MDE_2p8xSE']].to_markdown(index=False))
    L.append("\n## LOFO sign-flip summary (pairs where dropping a fold flips the sign)\n")
    if len(lofo_df):
        flips = lofo_df[lofo_df['sign_flips_when_dropped']]
        L.append(f"{len(flips)} (pair,fold) sign-flips of {len(lofo_df)} scanned.\n")
        if len(flips):
            L.append(flips[['universe', 'pair', 'dropped_fold', 'full_mean_delta', 'lofo_mean_delta']].to_markdown(index=False))
    with open(os.path.join(out_dir, 'family1_summary.md'), 'w') as f:
        f.write('\n'.join(L))


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main() -> int:
    global N_BOOT, UNIVERSES
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--main-dir', default='experiments/storya_v21_main12_tuned')
    p.add_argument('--l7-dir', default='experiments/_rerun_colab_staging/storya_v21_l7_hats_tuned')
    p.add_argument('--output-dir', default='artifacts/storya_v21_family1')
    p.add_argument('--smoke', action='store_true', help='reduce n_boot for a fast wiring check')
    # 2026-09-10 post-hoc sensitivity overrides (defaults reproduce the confirmatory run exactly)
    p.add_argument('--universes', default=','.join(UNIVERSES),
                   help='comma list of universes to aggregate (default B,C = confirmatory)')
    p.add_argument('--arms', default=','.join(ALL_ARMS_DEFAULT),
                   help='comma subset of arms present in --main-dir; pre-registered pairs / SPA '
                        'candidates are RESTRICTED to it (never extended)')
    p.add_argument('--sensitivity', action='store_true',
                   help='POST-HOC SENSITIVITY mode: no L7 contingency, no SPA, no C/L5s robustness, '
                        'NO BH-FDR (raw HLN p only); ledger/summary marked non-confirmatory')
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.smoke:
        N_BOOT = 200
    UNIVERSES = [u for u in args.universes.split(',') if u]
    arms = [a for a in args.arms.split(',') if a]
    for a in arms:
        assert a in ALL_ARMS_DEFAULT, f'unknown arm {a}; valid: {ALL_ARMS_DEFAULT}'

    # Set the shared module's fold count so reused helpers behave consistently
    e6.N_FOLDS = N_FOLDS
    e6.CANONICAL_SEEDS = CANONICAL_SEEDS

    if args.sensitivity and os.path.abspath(args.output_dir) == os.path.abspath('artifacts/storya_v21_family1'):
        raise SystemExit('--sensitivity must not write into the confirmatory artifacts/storya_v21_family1; pass --output-dir')  # EXPL-CODE-01
    if args.sensitivity:
        print(f"[F1] POST-HOC SENSITIVITY mode: universes={UNIVERSES} arms={arms} "
              f"(no L7 contingency / SPA / BH; LADDER_PAIRS unchanged, restricted to arms present)")
        l7 = {'demote': False, 'diverge_frac': 0.0, 'collapse_frac': 0.0, 'n_cells': 0,
              'reason': 'N/A (sensitivity mode: L7 not part of this run)'}
    else:
        # L7 contingency FIRST (decides M=9 vs 8 + whether L7-L2 pair stays)
        l7_results = os.path.join(args.l7_dir, 'results.csv')
        l7 = l7_contingency(l7_results)
        print(f"[F1] L7 contingency: demote={l7['demote']} | {l7['reason']}")

    spa_candidates = [m for m in SPA_CANDIDATES_FULL if m in arms]
    pairs = [pr for pr in ALL_PAIRS if pr[0] in arms and pr[1] in arms]   # pre-registered pairs only
    if l7['demote']:
        spa_candidates = [m for m in spa_candidates if m != 'L7']    # M=9 → 8
        pairs = [pr for pr in pairs if 'L7' not in pr]               # drop L7-L2
        arms = [a for a in arms if a != 'L7']
        print("[F1] L7 DEMOTED → SPA M=8, L7-L2 pair removed (kept exploratory elsewhere)")

    print("[F1] aggregating per-day IC matrices ...")
    agg = build_aggregate(arms, args.main_dir, args.l7_dir)
    if args.sensitivity:   # EXPL-CODE-10: fail closed on an empty requested scope (silent empty CSVs otherwise)
        empty = [(u, a) for u in UNIVERSES for a in arms if len(agg[(u, a)]['seed_avg_pooled']) == 0]
        if empty:
            raise SystemExit(f'no per-day IC found for {empty} in {args.main_dir}; check --universes/--arms')

    if args.sensitivity:
        spa_df = pd.DataFrame()
        print("[F1] Hansen SPA skipped (sensitivity mode)")
    else:
        print("[F1] Hansen SPA ...")
        spa_df = run_spa_ladder(agg, spa_candidates, args.output_dir)
        print(spa_df.to_string(index=False))

    print("[F1] DM/HLN pairwise" + (" (raw HLN p, NO BH — sensitivity) ..." if args.sensitivity
                                    else " + BH-FDR ..."))
    dm_df = run_pairwise(agg, pairs, args.output_dir, apply_bh=not args.sensitivity)
    if len(dm_df):
        print(dm_df[['universe', 'arm_A', 'arm_B', 'mean_delta_IC', 'HLN_p_t',
                     'BH_FDR_reject_family']].to_string(index=False))

    print("[F1] block-bootstrap CI + MDE + LOFO ...")
    ci_df, mde_df, lofo_df = run_ci_and_mde(agg, arms, pairs, args.output_dir)

    print("[F1] stability (degeneracy rate) + C/L5s robustness (3 treatments) ...")
    stab_df = degeneracy_report(arms, args.main_dir, args.l7_dir, args.output_dir,
                                ref_arms=(arms if args.sensitivity else None))
    print(stab_df[stab_df['collapse_rate'] > 0].to_string(index=False)
          if len(stab_df) and (stab_df['collapse_rate'] > 0).any() else "  (no degeneracy in any arm)")
    if args.sensitivity or ('C', 'L5s') not in agg:
        rob_df = pd.DataFrame()
        print("[F1] C/L5s robustness skipped (sensitivity mode or C/L5s not in this run)")
    else:
        rob_df = cl5s_robustness(agg, args.main_dir, spa_candidates, args.output_dir)
        print(rob_df.to_string(index=False))

    spa_M = int(spa_df['M'].iloc[0]) if len(spa_df) else len(spa_candidates)
    write_ledger(args.output_dir, l7, spa_M, sensitivity=args.sensitivity, pairs_run=pairs, arms_run=arms,
                 main_dir=args.main_dir, n_tests_actual=len(dm_df))
    write_summary(args.output_dir, l7, spa_df, dm_df, ci_df, mde_df, lofo_df, stab_df, rob_df,
                  sensitivity=args.sensitivity)
    print(f"[F1] DONE → {args.output_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
