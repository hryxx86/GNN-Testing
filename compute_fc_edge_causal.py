#!/usr/bin/env python
"""compute_fc_edge_causal.py — D-RERUN-12F Family-2 FC causal edge-attribution.

Family-2 = CAUSAL edge attribution at the FROZEN L2 operating point (the fixed-capacity
arm). Estimand = the LOCAL pure-edge effect at the frozen L2 architecture (CODEX-A-04:
labelled local-to-L2, NOT global edge superiority). matched-ΔIC is the CAUSAL PRIMARY;
the tuned-ΔIC for the same edge arms is a DESCRIPTIVE complement only (no post-hoc
primary-switching — protocol/plan LOCKED).

Spec is LOCKED in docs/plan_fc_edge_robustness_2026-06-17.md (Family-2). This module is a
THIN DRIVER reusing compute_e6_dm_spa.py helpers (BH-FDR, stationary block bootstrap).

LOCKED design (plan §Family-2):
  - 6 confirmatory contrasts: {L3fc (corr+news), L4fc (corr+sector), L5fc (corr+sec+news)} − L2,
    × {B, C}. (L6 EXCLUDED; L2-L1 EXCLUDED — those are Family-1.)
  - Baseline = the FROZEN tuned-ladder L2 per-day predictions, REUSED (the FC arms share L2's
    HP vector; only the edge set varies → the ΔIC isolates the edge effect at fixed capacity).
  - Inference = paired FOLD-LEVEL seed-averaged ΔIC (effective n ≈ 12 fold blocks, NOT 120 cells):
      per (universe, fc_arm, fold): seed-average daily IC → fold-mean IC for FC and for L2;
      ΔIC_fold = fold-mean(FC) − fold-mean(L2). 12 fold-level ΔIC values per contrast.
      → stationary block bootstrap over the 12 fold blocks for the CI (fold = resample unit).
      → BH-FDR q=0.05 over the 6 contrasts.
  - Report matched-ΔIC point + CI + same-sign vs the tuned-ΔIC (L3/L4/L5 tuned − L2 tuned).
  - Honest power (plan): n_eff ≈ 12 → MDE@80% ≈ 0.008-0.032 vs edge ΔIC ~0.005-0.016 → several
    contrasts likely "directional but not reliable" — expected, the FC arm is a pre-registered
    causal frame + reviewer defense, not a guaranteed resolution.

Usage (from project root):
  python compute_fc_edge_causal.py
  python compute_fc_edge_causal.py --fc-dir experiments/_rerun_colab_staging/storya_v21_main12_fc \
      --main-dir experiments/storya_v21_main12_tuned --output-dir artifacts/storya_v21_family2_fc
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

from compute_e6_dm_spa import bh_fdr, stationary_bootstrap_ci

# ── CONFIG (LOCKED per plan §Family-2) ──
UNIVERSES = ['B', 'C']
CANONICAL_SEEDS = [86, 123, 456, 789, 1024, 2024, 7, 34, 99, 2026]
N_FOLDS = 12
FC_ARMS = ['L3', 'L4', 'L5']           # L3=+news, L4=+sector, L5=+sec+news (vs L2 corr-only)
L2_BASE = 'L2'
N_BOOT = 5000
FOLD_BLOCK_SIZE = 1                     # 12 fold-blocks are the resample unit (block=1 fold)
BH_FDR_Q = 0.05
_Z_975 = float(stats.norm.ppf(0.975))
_Z_80 = float(stats.norm.ppf(0.80))


def _load(per_day_dir: str, u: str, arm: str, seed: int, fold: int):
    p = os.path.join(per_day_dir, f'{u}_{arm}_s{seed}_f{fold}.npy')
    return np.load(p).astype(np.float64) if os.path.exists(p) else None


def fold_seedavg_mean(per_day_dir: str, u: str, arm: str, fold: int):
    """Seed-average the daily IC for (u, arm, fold) then take the fold mean. None if absent."""
    arrs = [_load(per_day_dir, u, arm, s, fold) for s in CANONICAL_SEEDS]
    present = [a for a in arrs if a is not None]
    if not present:
        return None
    max_len = max(len(a) for a in present)
    mat = np.full((len(present), max_len), np.nan)
    for i, a in enumerate(present):
        mat[i, :len(a)] = a
    seed_avg_daily = np.nanmean(mat, axis=0)           # seed-average per day
    return float(np.nanmean(seed_avg_daily))           # fold-mean of the seed-averaged daily IC


def pooled_seedavg_mean(per_day_dir: str, u: str, arm: str):
    """Tuned-ΔIC helper: seed-avg daily IC pooled across all folds, then overall mean."""
    daily = []
    for f in range(N_FOLDS):
        arrs = [_load(per_day_dir, u, arm, s, f) for s in CANONICAL_SEEDS]
        present = [a for a in arrs if a is not None]
        if not present:
            continue
        max_len = max(len(a) for a in present)
        mat = np.full((len(present), max_len), np.nan)
        for i, a in enumerate(present):
            mat[i, :len(a)] = a
        daily.append(np.nanmean(mat, axis=0))
    return float(np.nanmean(np.concatenate(daily))) if daily else np.nan


def run(fc_dir: str, main_dir: str, out_dir: str) -> pd.DataFrame:
    fc_pd = os.path.join(fc_dir, 'per_day_ic')
    main_pd = os.path.join(main_dir, 'per_day_ic')
    rows = []
    for u in UNIVERSES:
        # L2 fold-level seed-averaged means (the frozen baseline, reused)
        l2_fold = {f: fold_seedavg_mean(main_pd, u, L2_BASE, f) for f in range(N_FOLDS)}
        l2_tuned_pooled = pooled_seedavg_mean(main_pd, u, L2_BASE)
        for arm in FC_ARMS:
            # FC arm fold-level seed-averaged means (from the FC out-dir; L2-capacity, varied edges)
            fc_fold = {f: fold_seedavg_mean(fc_pd, u, arm, f) for f in range(N_FOLDS)}
            # paired fold-level ΔIC (drop folds missing on either side)
            deltas = np.array([fc_fold[f] - l2_fold[f] for f in range(N_FOLDS)
                               if fc_fold[f] is not None and l2_fold[f] is not None], dtype=np.float64)
            if len(deltas) < 2:
                continue
            matched_delta = float(np.mean(deltas))
            _, lo, hi = stationary_bootstrap_ci(deltas, lambda x: float(np.mean(x)),
                                                n_boot=N_BOOT, block_size=FOLD_BLOCK_SIZE, seed=86)
            se = float(np.std(deltas, ddof=1) / np.sqrt(len(deltas)))   # SE over n_eff fold blocks
            mde80 = (_Z_975 + _Z_80) * se
            # tuned-ΔIC (DESCRIPTIVE complement): the same arm's TUNED main-table IC − tuned L2
            fc_tuned_pooled = pooled_seedavg_mean(main_pd, u, arm)
            tuned_delta = float(fc_tuned_pooled - l2_tuned_pooled)
            rows.append({
                'universe': u, 'fc_arm': arm, 'contrast': f'{arm}fc-L2',
                'edge_added': {'L3': 'news', 'L4': 'sector', 'L5': 'sector+news'}[arm],
                'n_fold_blocks': int(len(deltas)),
                'matched_delta_IC': round(matched_delta, 5),         # CAUSAL PRIMARY
                'ci_lo': round(lo, 5), 'ci_hi': round(hi, 5),
                'ci_excludes_0': bool(lo > 0 or hi < 0),
                'SE_fold': round(se, 5), 'MDE_80pct': round(mde80, 5),
                'underpowered_vs_effect': bool(abs(matched_delta) < mde80),
                'tuned_delta_IC': round(tuned_delta, 5),             # DESCRIPTIVE only
                'same_sign_matched_vs_tuned': bool(np.sign(matched_delta) == np.sign(tuned_delta)),
            })
    df = pd.DataFrame(rows)
    if len(df):
        # BH-FDR over the 6 contrasts on a two-sided p from the fold-block t-test
        # (paired one-sample t on the 12 fold deltas; consistent estimand with the bootstrap CI)
        pvals = []
        for _, r in df.iterrows():
            n = r['n_fold_blocks']
            t = r['matched_delta_IC'] / r['SE_fold'] if r['SE_fold'] > 0 else 0.0
            pvals.append(float(2 * (1 - stats.t.cdf(abs(t), df=n - 1))))
        df['t_p_two_sided'] = [round(p, 5) for p in pvals]
        df['BH_FDR_reject'] = bh_fdr(pvals, q=BH_FDR_Q)
        df['bh_fdr_q'] = BH_FDR_Q
    df.to_csv(os.path.join(out_dir, 'family2_fc_causal.csv'), index=False)
    return df


def write_ledger(out_dir: str, df: pd.DataFrame) -> None:
    ledger = {
        'family': 'Family-2 (FC fixed-capacity edge causal attribution)',
        'role': 'CAUSAL PRIMARY for edge attribution (matched-ΔIC); tuned-ΔIC is DESCRIPTIVE only',
        'estimand': 'local pure-edge effect at the frozen L2 operating point (local-to-L2, NOT global)',
        'contrasts': [f'{a}fc-L2 × {u}' for u in UNIVERSES for a in FC_ARMS],
        'n_contrasts': len(FC_ARMS) * len(UNIVERSES),
        'inference': {
            'unit': 'paired fold-level seed-averaged ΔIC',
            'n_eff': 'approx 12 fold blocks (NOT 120 cells)',
            'ci': f'stationary block bootstrap over fold blocks (block={FOLD_BLOCK_SIZE} fold), n_boot={N_BOOT}',
            'multiplicity': f'BH-FDR q={BH_FDR_Q} over the {len(FC_ARMS) * len(UNIVERSES)} contrasts',
        },
        'baseline': 'FROZEN tuned-ladder L2 per-day predictions, REUSED (edge set varies, HP fixed at L2)',
        'honest_power_note': ('n_eff≈12 → MDE@80%≈0.008-0.032 vs edge ΔIC~0.005-0.016; several '
                              'contrasts likely directional-but-not-reliable — expected, not a failure.'),
        'no_post_hoc_switch': 'matched-ΔIC is the locked causal primary; tuned-ΔIC never post-hoc-promoted.',
    }
    with open(os.path.join(out_dir, 'family2_ledger.json'), 'w') as f:
        json.dump(ledger, f, indent=2)


def write_summary(out_dir: str, df: pd.DataFrame) -> None:
    L = [f"# Family-2 FC causal edge-attribution  (_generated {time.strftime('%Y-%m-%d %H:%M:%S')}_)\n",
         "matched-ΔIC = CAUSAL PRIMARY (fold-level seed-avg, n≈12 blocks, block bootstrap CI, BH-FDR/6). "
         "tuned-ΔIC = DESCRIPTIVE complement (no post-hoc primary-switching).\n"]
    if len(df):
        cols = ['universe', 'fc_arm', 'edge_added', 'n_fold_blocks', 'matched_delta_IC',
                'ci_lo', 'ci_hi', 'ci_excludes_0', 'MDE_80pct', 'underpowered_vs_effect',
                't_p_two_sided', 'BH_FDR_reject', 'tuned_delta_IC', 'same_sign_matched_vs_tuned']
        L.append(df[cols].to_markdown(index=False))
        n_rej = int(df['BH_FDR_reject'].sum())
        n_under = int(df['underpowered_vs_effect'].sum())
        L.append(f"\n**{n_rej}/{len(df)} contrasts survive BH-FDR q=0.05.** "
                 f"{n_under}/{len(df)} are underpowered (|matched ΔIC| < MDE@80%) → "
                 f"'directional but not reliable' (expected per locked power analysis).")
    with open(os.path.join(out_dir, 'family2_summary.md'), 'w') as f:
        f.write('\n'.join(L))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--fc-dir', default='experiments/_rerun_colab_staging/storya_v21_main12_fc')
    p.add_argument('--main-dir', default='experiments/storya_v21_main12_tuned')
    p.add_argument('--output-dir', default='artifacts/storya_v21_family2_fc')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.smoke:
        global N_BOOT
        N_BOOT = 200
    print("[F2-FC] computing fold-level seed-averaged ΔIC + block bootstrap + BH-FDR/6 ...")
    df = run(args.fc_dir, args.main_dir, args.output_dir)
    if len(df):
        print(df[['universe', 'fc_arm', 'matched_delta_IC', 'ci_lo', 'ci_hi',
                  'ci_excludes_0', 'BH_FDR_reject', 'tuned_delta_IC',
                  'same_sign_matched_vs_tuned']].to_string(index=False))
    write_ledger(args.output_dir, df)
    write_summary(args.output_dir, df)
    print(f"[F2-FC] DONE → {args.output_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
