#!/usr/bin/env python
"""compute_e6_dm_spa.py — Story A E6 post-process (plan §1.4).

Reads E1 anchor results (and optionally E3 / E4-α results once available) and produces
paper-ready statistical artifacts:

  artifacts/storya_e6_dm_spa/spa_results.csv               Hansen SPA per universe + joint
  artifacts/storya_e6_dm_spa/dm_hln_results.csv            5 pairwise DM/HLN tests with BH-FDR
  artifacts/storya_e6_dm_spa/bootstrap_ci.csv              Per (model, universe) IC + Sharpe with stationary-block CI
  artifacts/storya_e6_dm_spa/cost_ladder.csv               Net Sharpe per (model, universe, cost_level)
  artifacts/storya_e6_dm_spa/multiple_testing_ledger.json  Honest disclosure of all trials
  artifacts/storya_e6_dm_spa/summary.md                    Human-readable Table 1-4

Methodology (LOCKED per plan §1.4):
  - SPA candidates seed-AVERAGED per (model, universe, date, fold) (M=3 per universe, M=6 joint)
    per Codex Round D D-04 fix. Benchmark = LightGBM.
  - DM/HLN family = 5 pairwise tests: {GAT,SAGE-Mean,MLP} vs LightGBM + {GAT,SAGE-Mean} vs MLP
    on seed-aggregated per-day IC series (T=313 not T=3130).
    HLN small-sample correction at T=313, h=HORIZON=21 → factor ≈ 0.935.
    BH-FDR at q=0.05.
  - Stationary block bootstrap CI: block_size=21 (= horizon, captures intra-label autocorr),
    n_boot=5000, two-sided 95%. IC CI uses pooled per-day IC across 10 seeds × 5 folds.
  - Cost ladder Sharpe: per-cell Sharpe_net_{c}bps already in results.csv (turnover_L1 convention,
    annualization sqrt(252/HORIZON); plan §1.4(d) D-05). CI from 50 cell-level Sharpe values.
  - Multi-testing ledger: SPA M=3/3/6, DM family=5, BH-FDR q=0.05.

Usage (from project root):
  python compute_e6_dm_spa.py
  python compute_e6_dm_spa.py --results-csv experiments/storya_e1_anchor/results.csv
  python compute_e6_dm_spa.py --include-e3 experiments/storya_e3_news_edge/results.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

# ── arch is REQUIRED (Hansen SPA) ──
try:
    from arch.bootstrap import SPA, StationaryBootstrap
except ImportError:
    print("ERROR: arch package required. pip install arch", file=sys.stderr)
    sys.exit(2)


# ══════════════════════════════════════════════════════════════
# CONFIG — must match run_storya_e1_anchor.py
# ══════════════════════════════════════════════════════════════

EXPECTED_MODELS = ['GAT', 'SAGE-Mean', 'MLP', 'LightGBM']
EXPECTED_UNIVERSES = ['B', 'C']
BASELINE = 'LightGBM'
SECONDARY_BASELINE = 'MLP'
CANONICAL_SEEDS = [86, 123, 456, 789, 1024, 2024, 7, 34, 99, 2026]
N_FOLDS = 5

HORIZON = 21
COST_LEVELS_BPS = (0, 5, 10, 15, 20, 30)

# Statistical knobs (LOCKED per plan §1.4)
N_BOOT = 5000
BLOCK_SIZE = HORIZON          # block bootstrap block length = horizon
SPA_REPS = 10000
SPA_BLOCK_SIZE = HORIZON
BH_FDR_Q = 0.05

# Newey-West auto-lag for DM (Newey-West 1994): L = floor(4 * (T/100)^(2/9))
def nw_lag(T: int) -> int:
    return int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))


# ══════════════════════════════════════════════════════════════
# LOADERS
# ══════════════════════════════════════════════════════════════

def load_results_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {'cell_id', 'universe', 'model', 'seed', 'fold',
              'IC_mean', 'Sharpe_gross', 'converged_flag', 'n_periods'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    for c in COST_LEVELS_BPS:
        col = f'Sharpe_net_{c}bps'
        if col not in df.columns:
            raise ValueError(f"{path} missing cost-ladder column {col}")
    if df['converged_flag'].sum() < len(df):
        bad = df[df['converged_flag'] != 1]
        print(f"WARN: {len(bad)} cells failed converged_flag check; downstream stats may be biased")
    return df


def load_per_day_ic(per_day_ic_dir: str, universe: str, model: str, seed: int, fold: int) -> Optional[np.ndarray]:
    """Returns 1D float array of per-day ICs for this cell, or None if file missing."""
    path = os.path.join(per_day_ic_dir, f'{universe}_{model}_s{seed}_f{fold}.npy')
    if not os.path.exists(path):
        return None
    return np.load(path).astype(np.float64)


def collect_per_day_ic_matrix(per_day_ic_dir: str, universe: str, model: str) -> dict:
    """For (universe, model), build dict {fold: 2D array (n_seeds, n_test_days_in_fold)}.

    Within a fold, all seeds should produce the same length per_day_ic (because the
    label_valid mask is identical across seeds; only models with degenerate predictions
    would produce NaN ICs that get skipped — defensive). If lengths differ within a fold,
    we pad with NaN to max length and use np.nanmean for seed aggregation downstream.
    """
    out = {}
    for fold in range(N_FOLDS):
        seed_arrays = []
        for seed in CANONICAL_SEEDS:
            arr = load_per_day_ic(per_day_ic_dir, universe, model, seed, fold)
            if arr is None:
                seed_arrays.append(None)
            else:
                seed_arrays.append(arr)
        present = [a for a in seed_arrays if a is not None]
        if not present:
            out[fold] = None
            continue
        lens = [len(a) for a in present]
        max_len = max(lens)
        if len(set(lens)) > 1:
            print(f"WARN: {universe}/{model}/fold{fold}: per-day IC lengths differ across seeds "
                  f"(min={min(lens)}, max={max_len}); padding with NaN.")
        # Build (n_seeds, max_len) matrix, pad missing with all-NaN
        mat = np.full((len(CANONICAL_SEEDS), max_len), np.nan, dtype=np.float64)
        for i, a in enumerate(seed_arrays):
            if a is None:
                continue
            mat[i, :len(a)] = a
        out[fold] = mat
    return out


# ══════════════════════════════════════════════════════════════
# SEED AGGREGATION
# ══════════════════════════════════════════════════════════════

def seed_aggregate_pooled(per_day_ic_dict: dict) -> np.ndarray:
    """Concat per (date, fold) seed-averaged IC across all folds → 1D series of length T.

    For each fold:
      seed-averaged_per_day_ic[fold] = nanmean(mat, axis=0)  # length n_test_days_in_fold
    Then concat across folds in chronological order.
    """
    pooled = []
    for fold in range(N_FOLDS):
        mat = per_day_ic_dict.get(fold)
        if mat is None:
            continue
        avg = np.nanmean(mat, axis=0)
        pooled.append(avg)
    if not pooled:
        return np.array([], dtype=np.float64)
    return np.concatenate(pooled)


def pool_per_day_ic_full(per_day_ic_dict: dict) -> np.ndarray:
    """For bootstrap CI: flatten ALL (seed, fold, day) per-day IC into one long series.

    Length ≈ N_SEEDS × T_pooled (T_pooled ≈ 313). Block bootstrap on this captures both
    seed AND day-to-day variation as the noise sources for the headline CI.
    """
    out = []
    for fold in range(N_FOLDS):
        mat = per_day_ic_dict.get(fold)
        if mat is None:
            continue
        # Flatten seed × day; drop NaN
        flat = mat.flatten()
        flat = flat[~np.isnan(flat)]
        out.append(flat)
    if not out:
        return np.array([], dtype=np.float64)
    return np.concatenate(out)


# ══════════════════════════════════════════════════════════════
# NW-HAC variance + DM/HLN
# ══════════════════════════════════════════════════════════════

def nw_hac_variance(x: np.ndarray, L: int) -> float:
    """Newey-West HAC variance estimator of sample mean. Bartlett kernel.

    NW_var = γ_0 + 2 * sum_{l=1}^{L} w_l * γ_l,  w_l = 1 - l/(L+1)
    Where γ_l = (1/T) * sum_{t=l+1}^{T} (x_t - x_bar)(x_{t-l} - x_bar)

    CODEX-CR-E6-A-03 fix: must divide each γ_l by T (full series length), not T-l.
    Using .mean() on the truncated product array divides by T-l, biasing the estimator
    slightly upward. Newey-West 1987 standard uses 1/T for all lags.

    Returns the variance ON THE MEAN (= variance of mean estimator).  To get
    SE(mean), do sqrt(nw_var / T) externally.
    """
    T = len(x)
    if T < 2:
        return 0.0
    xc = x - x.mean()
    gamma_0 = float((xc * xc).sum() / T)  # γ_0 divisor is T (equivalent to .mean() here)
    var = gamma_0
    for lag in range(1, L + 1):
        if lag >= T:
            break
        w = 1.0 - lag / (L + 1.0)
        # Divide by T (NOT T-lag); CODEX-CR-E6-A-03 fix
        gamma_l = float((xc[lag:] * xc[:-lag]).sum() / T)
        var += 2.0 * w * gamma_l
    return max(var, 0.0)  # truncate at zero (occasional negative under small T)


def dm_test(d: np.ndarray) -> tuple[float, float, int]:
    """Standard Diebold-Mariano test on loss-difference series d_t.

    Returns (DM_stat, p_two_sided_normal, T). p uses N(0,1) — to be REPLACED by HLN-t below.
    """
    T = len(d)
    if T < 5:
        return np.nan, np.nan, T
    L = nw_lag(T)
    nw_var = nw_hac_variance(d, L)
    if nw_var <= 0:
        return np.nan, np.nan, T
    dm_stat = d.mean() / np.sqrt(nw_var / T)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p), T


def hln_test(d: np.ndarray, h: int = HORIZON) -> tuple[float, float, int]:
    """Harvey-Leybourne-Newbold small-sample-corrected DM.

    HLN_stat = DM × sqrt((T + 1 - 2h + h(h-1)/T) / T)
    p-value from t_{T-1} distribution (two-sided).
    """
    dm_stat, _, T = dm_test(d)
    if np.isnan(dm_stat) or T < 5:
        return np.nan, np.nan, T
    factor = np.sqrt(max((T + 1 - 2 * h + h * (h - 1) / T) / T, 1e-12))
    hln_stat = dm_stat * factor
    p = 2.0 * (1.0 - stats.t.cdf(abs(hln_stat), df=T - 1))
    return float(hln_stat), float(p), T


def bh_fdr(pvals: list[float], q: float = BH_FDR_Q) -> list[bool]:
    """Benjamini-Hochberg FDR correction. Returns boolean reject array."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    ranks = np.empty(m, dtype=int)
    ranks[order] = np.arange(m)
    crit = (ranks + 1) / m * q
    reject_sorted = pvals[order] <= ((np.arange(m) + 1) / m * q)
    # Step-up: find largest k where p_(k) <= k/m * q
    if not reject_sorted.any():
        return [False] * m
    last_k = np.where(reject_sorted)[0].max()
    threshold = pvals[order][last_k]
    return list(pvals <= threshold)


# ══════════════════════════════════════════════════════════════
# Stationary block bootstrap CI
# ══════════════════════════════════════════════════════════════

def stationary_bootstrap_ci(x: np.ndarray, statistic, n_boot: int = N_BOOT,
                             block_size: int = BLOCK_SIZE, alpha: float = 0.05,
                             seed: int = 86) -> tuple[float, float, float]:
    """Stationary block bootstrap CI on a 1D series.

    Uses arch.bootstrap.StationaryBootstrap with geometric block lengths (mean = block_size).
    Returns (point_estimate, ci_lower, ci_upper) for `statistic` (callable, takes 1D array).
    """
    if len(x) < 2:
        return float(statistic(x)) if len(x) else 0.0, np.nan, np.nan
    sb = StationaryBootstrap(block_size, x, seed=seed)
    results = sb.apply(statistic, n_boot)
    # `results` is shape (n_boot,) for scalar statistic
    ci_lo = float(np.percentile(results, 100 * alpha / 2))
    ci_hi = float(np.percentile(results, 100 * (1 - alpha / 2)))
    return float(statistic(x)), ci_lo, ci_hi


# ══════════════════════════════════════════════════════════════
# Hansen SPA wrapper
# ══════════════════════════════════════════════════════════════

def run_spa(benchmark_losses: np.ndarray, candidate_losses: np.ndarray,
            reps: int = SPA_REPS, block_size: int = SPA_BLOCK_SIZE,
            seed: int = 86) -> dict:
    """Hansen SPA (2005) via arch.bootstrap.SPA.

    H0: max_k E[L_benchmark - L_k] <= 0 (benchmark not worse than any candidate)
    Reject → at least one candidate has lower expected loss (higher IC).

    Returns dict with p_lower, p_consistent, p_upper (Hansen 2005 nomenclature).
    """
    if candidate_losses.ndim == 1:
        candidate_losses = candidate_losses.reshape(-1, 1)
    T_b = len(benchmark_losses)
    T_c = candidate_losses.shape[0]
    if T_b != T_c:
        raise ValueError(f"SPA length mismatch: benchmark T={T_b}, candidates T={T_c}")
    spa = SPA(benchmark_losses, candidate_losses, reps=reps, block_size=block_size, seed=seed)
    spa.compute()
    # arch returns pvalues as a pandas.Series with index ['lower', 'consistent', 'upper'],
    # each value is a numpy.float64. Just cast directly.
    pv = spa.pvalues
    return {
        'p_lower': float(pv['lower']),
        'p_consistent': float(pv['consistent']),
        'p_upper': float(pv['upper']),
        'M': int(candidate_losses.shape[1]),
        'T': int(T_b),
    }


# ══════════════════════════════════════════════════════════════
# PIPELINE — per-universe + joint
# ══════════════════════════════════════════════════════════════

def aggregate_e1(per_day_ic_dir: str) -> dict:
    """Build {(universe, model): {'seed_avg_pooled': T-vec, 'all_pooled': flat}}."""
    out = {}
    for universe in EXPECTED_UNIVERSES:
        for model in EXPECTED_MODELS:
            d = collect_per_day_ic_matrix(per_day_ic_dir, universe, model)
            seed_avg = seed_aggregate_pooled(d)
            all_pooled = pool_per_day_ic_full(d)
            out[(universe, model)] = {
                'seed_avg_pooled': seed_avg,
                'all_pooled': all_pooled,
                'per_fold_seed_matrix': d,
            }
            n_seed_avg = len(seed_avg)
            n_all = len(all_pooled)
            print(f"  [{universe}/{model}] seed_avg T={n_seed_avg}  all_pooled N={n_all}")
    return out


def run_spa_per_universe(agg: dict, out_dir: str) -> pd.DataFrame:
    rows = []
    for universe in EXPECTED_UNIVERSES:
        candidates = [m for m in EXPECTED_MODELS if m != BASELINE]  # 3 candidates
        bench = agg[(universe, BASELINE)]['seed_avg_pooled']
        if len(bench) == 0:
            print(f"  [SPA universe={universe}] benchmark empty; skip")
            continue
        # Skip if any candidate has empty series (partial smoke data)
        if any(len(agg[(universe, m)]['seed_avg_pooled']) == 0 for m in candidates):
            print(f"  [SPA universe={universe}] one or more candidates empty; skip")
            continue
        cand_losses_list = []
        for m in candidates:
            ic = agg[(universe, m)]['seed_avg_pooled']
            if len(ic) != len(bench):
                # Align by truncating to min — happens if some fold is missing
                n = min(len(ic), len(bench))
                bench = bench[:n]
                cand_losses_list = [c[:n] for c in cand_losses_list]
                ic = ic[:n]
            cand_losses_list.append(-ic)  # loss = -IC
        bench_losses = -bench
        cand_losses = np.column_stack(cand_losses_list)
        # SPA bootstrap needs T >= block_size + some headroom; refuse if T < 2 * block_size
        if len(bench_losses) < 2 * SPA_BLOCK_SIZE:
            print(f"  [SPA universe={universe}] T={len(bench_losses)} < 2*block_size={2*SPA_BLOCK_SIZE}; "
                  f"skip (insufficient samples for block bootstrap)")
            continue
        print(f"  [SPA universe={universe}] T={len(bench_losses)}, M={cand_losses.shape[1]}")
        res = run_spa(bench_losses, cand_losses)
        rows.append({
            'universe': universe,
            'benchmark': BASELINE,
            'candidates': '|'.join(candidates),
            'M': res['M'],
            'T': res['T'],
            'p_lower': res['p_lower'],
            'p_consistent': res['p_consistent'],
            'p_upper': res['p_upper'],
            'reject_h0_at_5pct': res['p_consistent'] < 0.05,
        })
    # Joint SPA (M=6: 3 candidates × 2 universes). CODEX-CR-E6-A-07(a) fix: skip joint
    # SPA entirely if ANY of the 6 expected (universe, candidate) series is empty,
    # because np.column_stack would otherwise raise on a length-0 element.
    cand_losses_list = []
    cand_labels = []
    bench_losses_list = []
    skip_joint = False
    for universe in EXPECTED_UNIVERSES:
        bench = agg[(universe, BASELINE)]['seed_avg_pooled']
        if len(bench) == 0:
            print(f"  [SPA joint] {universe}/{BASELINE} empty; SKIP joint SPA")
            skip_joint = True
            break
        # use this universe's LGB losses as part of joint benchmark
        bench_losses_list.append(-bench)
        for m in [x for x in EXPECTED_MODELS if x != BASELINE]:
            ic = agg[(universe, m)]['seed_avg_pooled']
            if len(ic) == 0:
                print(f"  [SPA joint] {universe}/{m} empty; SKIP joint SPA")
                skip_joint = True
                break
            if len(ic) != len(bench):
                ic = ic[:len(bench)]
            cand_losses_list.append(-ic)
            cand_labels.append(f"{universe}.{m}")
        if skip_joint:
            break
    if cand_losses_list and not skip_joint:
        # joint benchmark = pooled LightGBM loss across both universes (per plan §1.4(a))
        T_min = min(len(b) for b in bench_losses_list)
        bench_pool = np.mean(np.stack([b[:T_min] for b in bench_losses_list]), axis=0)
        cand_trimmed = np.column_stack([c[:T_min] for c in cand_losses_list])
        print(f"  [SPA joint B+C] T={T_min}, M={cand_trimmed.shape[1]}")
        res = run_spa(bench_pool, cand_trimmed)
        rows.append({
            'universe': 'JOINT(B+C)',
            'benchmark': f'{BASELINE}_pooled',
            'candidates': '|'.join(cand_labels),
            'M': res['M'],
            'T': res['T'],
            'p_lower': res['p_lower'],
            'p_consistent': res['p_consistent'],
            'p_upper': res['p_upper'],
            'reject_h0_at_5pct': res['p_consistent'] < 0.05,
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'spa_results.csv'), index=False)
    return df


def run_dm_hln_pairwise(agg: dict, out_dir: str) -> pd.DataFrame:
    """5-test family: GAT/SAGE/MLP vs LightGBM + GAT/SAGE vs MLP, per universe."""
    rows = []
    for universe in EXPECTED_UNIVERSES:
        # Family of 5: 3 vs LGB + 2 vs MLP
        pairs = [(m, BASELINE) for m in EXPECTED_MODELS if m != BASELINE] + \
                [(m, SECONDARY_BASELINE) for m in ['GAT', 'SAGE-Mean']]
        for a, b in pairs:
            ic_a = agg[(universe, a)]['seed_avg_pooled']
            ic_b = agg[(universe, b)]['seed_avg_pooled']
            n = min(len(ic_a), len(ic_b))
            if n < 5:
                continue
            d = (-ic_a[:n]) - (-ic_b[:n])  # loss_A - loss_B = -(IC_A - IC_B)
            dm_stat, dm_p, T = dm_test(d)
            hln_stat, hln_p, _ = hln_test(d)
            rows.append({
                'universe': universe,
                'model_A': a,
                'model_B': b,
                'mean_delta_IC': float((ic_a[:n] - ic_b[:n]).mean()),
                'T': T,
                'NW_lag': nw_lag(T),
                'DM_stat': dm_stat,
                'DM_p_normal': dm_p,
                'HLN_stat': hln_stat,
                'HLN_p_t': hln_p,
            })
    df = pd.DataFrame(rows)
    if len(df):
        # BH-FDR over the joint family (10 tests = 5 per universe × 2 universes; but BH should
        # be applied per universe per plan §1.4(b) — the 5-test family is INSIDE each universe).
        # Apply BH-FDR per universe, then mark.
        df['BH_FDR_reject'] = False
        for universe in df['universe'].unique():
            sub = df[df['universe'] == universe]
            rejected = bh_fdr(sub['HLN_p_t'].tolist(), q=BH_FDR_Q)
            df.loc[sub.index, 'BH_FDR_reject'] = rejected
        df['bh_fdr_q'] = BH_FDR_Q
    df.to_csv(os.path.join(out_dir, 'dm_hln_results.csv'), index=False)
    return df


def run_bootstrap_ci(agg: dict, results_df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """Block bootstrap CI on pooled IC + cell-level Sharpe distribution."""
    rows = []
    for universe in EXPECTED_UNIVERSES:
        for model in EXPECTED_MODELS:
            # IC pooled (per-seed × per-day) → block bootstrap
            x = agg[(universe, model)]['all_pooled']
            if len(x) < 2:
                continue
            ic_mean, ic_lo, ic_hi = stationary_bootstrap_ci(
                x, statistic=lambda a: float(np.mean(a)),
            )
            # cell-level Sharpe distribution (one value per (seed, fold))
            sub = results_df[(results_df['universe'] == universe) & (results_df['model'] == model)]
            if len(sub) == 0:
                continue
            row = {
                'universe': universe,
                'model': model,
                'n_per_day_obs': int(len(x)),
                'n_cells': int(len(sub)),
                'IC_mean': ic_mean,
                'IC_mean_ci_lo': ic_lo,
                'IC_mean_ci_hi': ic_hi,
                'Sharpe_gross_mean': float(sub['Sharpe_gross'].mean()),
                # CODEX-CR-E6-A-07(b) fix: pandas .std() returns NaN for N<2 by default (ddof=1),
                # but be defensive: explicitly return NaN if only 1 cell.
                'Sharpe_gross_std': float(sub['Sharpe_gross'].std()) if len(sub) >= 2 else np.nan,
            }
            # Bootstrap CI on cell-level Sharpe_gross
            sg = sub['Sharpe_gross'].dropna().values
            if len(sg) >= 2:
                _, sg_lo, sg_hi = stationary_bootstrap_ci(
                    sg, statistic=lambda a: float(np.mean(a)), block_size=1
                )
                row['Sharpe_gross_ci_lo'] = sg_lo
                row['Sharpe_gross_ci_hi'] = sg_hi
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'bootstrap_ci.csv'), index=False)
    return df


def run_cost_ladder(results_df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """Aggregate per-cell Sharpe_net_{c}bps → per (universe, model, cost) mean ± bootstrap CI."""
    rows = []
    for universe in EXPECTED_UNIVERSES:
        for model in EXPECTED_MODELS:
            sub = results_df[(results_df['universe'] == universe) & (results_df['model'] == model)]
            if len(sub) == 0:
                continue
            for c in COST_LEVELS_BPS:
                col = f'Sharpe_net_{c}bps'
                vals = sub[col].dropna().values
                if len(vals) == 0:
                    continue
                if len(vals) >= 2:
                    # block_size=1 because cell-level Sharpe values are independent (different seeds/folds)
                    _, lo, hi = stationary_bootstrap_ci(
                        vals, statistic=lambda a: float(np.mean(a)), block_size=1
                    )
                else:
                    lo, hi = np.nan, np.nan
                rows.append({
                    'universe': universe,
                    'model': model,
                    'cost_bps': c,
                    'n_cells': int(len(vals)),
                    'Sharpe_net_mean': float(vals.mean()),
                    # CODEX-CR-E6-A-07(b) fix: numpy .std() returns 0.0 for N=1 (ddof=0 default);
                    # explicit NaN when fewer than 2 observations (no spread can be estimated).
                    'Sharpe_net_std': float(vals.std(ddof=1)) if len(vals) >= 2 else np.nan,
                    'Sharpe_net_ci_lo': lo,
                    'Sharpe_net_ci_hi': hi,
                })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'cost_ladder.csv'), index=False)
    return df


def write_multiple_testing_ledger(out_dir: str, e1_n_cells: int, e3_n_cells: int = 0, e4_n_cells: int = 0) -> None:
    ledger = {
        'primary_storya_trials': {
            'models': len(EXPECTED_MODELS),
            'seeds_per_model': len(CANONICAL_SEEDS),
            'folds': N_FOLDS,
            'universes': len(EXPECTED_UNIVERSES),
            'total_E1_cells': e1_n_cells,
        },
        'ablation_storya_trials': {
            'E3_news_encoding_configs': 2,
            'E3_cells': e3_n_cells,
            'E4_alpha_edge_configs': 4,
            'E4_cells_new': e4_n_cells,
        },
        'spa_application_universe_B_M': 3,
        'spa_application_universe_C_M': 3,
        'spa_application_joint_M': 6,
        'spa_seed_aggregation_note': (
            'M counts NON-baseline candidates only (LightGBM is benchmark, not candidate). '
            'Seeds are averaged per (model, universe, date, fold) BEFORE SPA — consistent with '
            'DM/HLN treatment. Fixed per Codex Round D D-04.'
        ),
        'dm_hln_family_size_per_universe': 5,
        'dm_hln_pairs_per_universe': [
            'GAT vs LightGBM', 'SAGE-Mean vs LightGBM', 'MLP vs LightGBM',
            'GAT vs MLP', 'SAGE-Mean vs MLP',
        ],
        'bh_fdr_q': BH_FDR_Q,
        'cost_ladder_bps': list(COST_LEVELS_BPS),
        'cost_convention': 'L1_one_way',
        'block_bootstrap': {
            'n_boot': N_BOOT,
            'block_size_days': BLOCK_SIZE,
            'spa_reps': SPA_REPS,
        },
        'horizon_days': HORIZON,
        'claim_robustness_note': (
            'Headline claims use SPA multi-comparison test (M=3 per universe, M=6 joint); '
            'pairwise comparisons use BH-FDR over family of 5 DM/HLN tests per universe. '
            'Historical exploratory trials (horizon ablation, Plan AAA, phase5 step3 subsets) '
            'are disclosed but NOT entered into SPA family — declared as pre-experiment '
            'exploratory per Codex Round A disposition.'
        ),
    }
    path = os.path.join(out_dir, 'multiple_testing_ledger.json')
    with open(path, 'w') as f:
        json.dump(ledger, f, indent=2)
    print(f"  [ledger] wrote {path}")


def write_summary(out_dir: str, spa_df: pd.DataFrame, dm_df: pd.DataFrame,
                  ci_df: pd.DataFrame, cost_df: pd.DataFrame) -> None:
    lines = ["# E6 Story A statistical summary\n", f"_generated {time.strftime('%Y-%m-%d %H:%M:%S')}_\n"]
    lines.append("## Hansen SPA (multi-comparison cherry-pick defense)\n")
    if len(spa_df):
        lines.append(spa_df[['universe', 'M', 'T', 'p_consistent', 'reject_h0_at_5pct']].to_markdown(index=False))
    lines.append("\n## DM/HLN pairwise (paired ΔIC, seed-aggregated T=313)\n")
    if len(dm_df):
        lines.append(dm_df[['universe', 'model_A', 'model_B', 'mean_delta_IC', 'HLN_p_t', 'BH_FDR_reject']].to_markdown(index=False))
    lines.append("\n## IC + Sharpe with block-bootstrap CI\n")
    if len(ci_df):
        lines.append(ci_df.to_markdown(index=False))
    lines.append("\n## Cost ladder (Net Sharpe per cost level)\n")
    if len(cost_df):
        # Pivot for readability
        pivot = cost_df.pivot_table(index=['universe', 'model'], columns='cost_bps',
                                    values='Sharpe_net_mean')
        lines.append(pivot.to_markdown())
    with open(os.path.join(out_dir, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines))


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--results-csv', default='experiments/storya_e1_anchor/results.csv')
    p.add_argument('--per-day-ic-dir', default='experiments/storya_e1_anchor/per_day_ic')
    p.add_argument('--output-dir', default='artifacts/storya_e6_dm_spa')
    p.add_argument('--include-e3', default=None, help='Optional: experiments/storya_e3_news_edge/results.csv')
    p.add_argument('--include-e4', default=None, help='Optional: experiments/storya_e4_alpha/results.csv')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[E6] loading {args.results_csv} ...")
    results_df = load_results_csv(args.results_csv)
    print(f"[E6]   {len(results_df)} cells; converged={int(results_df['converged_flag'].sum())}; "
          f"universes={sorted(results_df['universe'].unique())}; "
          f"models={sorted(results_df['model'].unique())}")

    e3_n = 0
    e4_n = 0
    if args.include_e3 and os.path.exists(args.include_e3):
        e3_df = pd.read_csv(args.include_e3)
        e3_n = len(e3_df)
        results_df = pd.concat([results_df, e3_df], ignore_index=True)
        print(f"[E6] merged E3: +{e3_n} cells")
    if args.include_e4 and os.path.exists(args.include_e4):
        e4_df = pd.read_csv(args.include_e4)
        e4_n = len(e4_df)
        results_df = pd.concat([results_df, e4_df], ignore_index=True)
        print(f"[E6] merged E4: +{e4_n} cells")

    print(f"\n[E6] aggregating per-day IC matrices ...")
    agg = aggregate_e1(args.per_day_ic_dir)

    print(f"\n[E6] running Hansen SPA ...")
    spa_df = run_spa_per_universe(agg, args.output_dir)
    print(spa_df.to_string(index=False))

    print(f"\n[E6] running DM/HLN pairwise + BH-FDR ...")
    dm_df = run_dm_hln_pairwise(agg, args.output_dir)
    if len(dm_df):
        print(dm_df[['universe', 'model_A', 'model_B', 'mean_delta_IC',
                     'HLN_p_t', 'BH_FDR_reject']].to_string(index=False))

    print(f"\n[E6] running block-bootstrap CI ...")
    ci_df = run_bootstrap_ci(agg, results_df, args.output_dir)
    if len(ci_df):
        print(ci_df[['universe', 'model', 'n_per_day_obs', 'IC_mean',
                     'IC_mean_ci_lo', 'IC_mean_ci_hi']].to_string(index=False))

    print(f"\n[E6] running cost-ladder Net Sharpe aggregation ...")
    cost_df = run_cost_ladder(results_df, args.output_dir)
    if len(cost_df):
        pivot = cost_df.pivot_table(index=['universe', 'model'], columns='cost_bps',
                                    values='Sharpe_net_mean')
        print(pivot.round(3))

    print(f"\n[E6] writing multi-testing ledger ...")
    write_multiple_testing_ledger(args.output_dir, e1_n_cells=400, e3_n_cells=e3_n, e4_n_cells=e4_n)

    print(f"\n[E6] writing summary.md ...")
    write_summary(args.output_dir, spa_df, dm_df, ci_df, cost_df)

    print(f"\n[E6] DONE → {args.output_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
