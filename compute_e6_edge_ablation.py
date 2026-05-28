#!/usr/bin/env python
"""compute_e6_edge_ablation.py

Story A E6 post-process v2 — Edge ablation framework for E3 + E4-α (per plan §1.3
+ H博士 2026-05-27-b decisions):

  5 paired comparisons × 3 regime conditions (full 5-fold / LOFO-4 / Fold-4-only):
    1. α2=corr+sector vs α1=corr only      → "Sector edge benefit"
    2. α3=corr+news   vs α1                 → "News edge benefit"
    3. α4=corr+sector+news vs α1            → "Full edge bundle benefit"
    4. α4 vs α2                              → "News on top of corr+sector"
    5. α4 vs α3                              → "Sector on top of corr+news"

  Statistical tests:
    - DM/HLN paired ΔIC per day (seed-aggregated; HLN small-sample correction h=21)
    - BH-FDR over 5 pairs at q=0.05 (full 5-fold condition only)
    - Block-bootstrap CI (block_size=21 on per-day IC, block_size=1 on per-cell Sharpe)
    - Fold-4-only condition reports CIs only (no DM/HLN p-value — T=63 too small for HLN)

  Cost ladder: 4 configs × 6 cost levels × 3 regime conditions = 72 rows

Option Y per H博士 decision: this script IMPORTS helpers from compute_e6_dm_spa.py
(NW-HAC variance, DM/HLN, BH-FDR, stationary bootstrap CI), keeping E1's E6 script
untouched and zero-risk. Single source of truth for methodology via imports.

Inputs (read-only):
  α1: /Users/heruixi/Library/CloudStorage/.../experiments/storya_e1_anchor/{results.csv, per_day_ic/B_SAGE-Mean_*}
  α2: .../experiments/storya_e4_alpha/{results.csv [edge_config==corr+sector], per_day_ic/corr_sector_SAGE-Mean_*}
  α3: .../experiments/storya_e3_news_edge/{results.csv, per_day_ic/B_SAGE-Mean_*}
  α4: .../experiments/storya_e4_alpha/{results.csv [edge_config==corr+sector+news], per_day_ic/corr_sector_news_SAGE-Mean_*}

Outputs:
  artifacts/storya_e6_edge_ablation/edge_pairs_dm.csv       — 5 pairs × 3 conditions
  artifacts/storya_e6_edge_ablation/edge_bootstrap_ci.csv   — 5 pairs × 3 conditions × {IC, Sharpe}
  artifacts/storya_e6_edge_ablation/edge_cost_ladder.csv    — 4 configs × 6 bps × 3 conditions
  artifacts/storya_e6_edge_ablation/edge_summary.md         — human readable
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/heruixi/Desktop/GNN-Testing")
from compute_e6_dm_spa import (
    CANONICAL_SEEDS,
    N_FOLDS,
    HORIZON,
    BLOCK_SIZE,
    N_BOOT,
    BH_FDR_Q,
    nw_lag,
    nw_hac_variance,
    dm_test,
    hln_test,
    bh_fdr,
    stationary_bootstrap_ci,
)

DRIVE = "/Users/heruixi/Library/CloudStorage/GoogleDrive-hryxx86@gmail.com/我的云端硬盘/GNN测试"
OUT_DIR = "artifacts/storya_e6_edge_ablation"
COST_LEVELS_BPS = (0, 5, 10, 15, 20, 30)

os.makedirs(OUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────
# Config definitions: which experiment dir + per_day_ic prefix + results filter
# ──────────────────────────────────────────────────────────────────────────

CONFIGS = {
    'alpha1_corr_only':       {'exp_dir': 'storya_e1_anchor',     'per_day_prefix': 'B_SAGE-Mean',          'results_filter': lambda r: (r.universe == 'B') & (r.model == 'SAGE-Mean')},
    'alpha2_corr_sector':     {'exp_dir': 'storya_e4_alpha',      'per_day_prefix': 'corr_sector_SAGE-Mean','results_filter': lambda r: r.edge_config == 'corr+sector'},
    'alpha3_corr_news':       {'exp_dir': 'storya_e3_news_edge',  'per_day_prefix': 'B_SAGE-Mean',          'results_filter': lambda r: r.edge_config == 'corr+news_cooccur'},
    'alpha4_corr_sector_news':{'exp_dir': 'storya_e4_alpha',      'per_day_prefix': 'corr_sector_news_SAGE-Mean','results_filter': lambda r: r.edge_config == 'corr+sector+news'},
}

PAIRS = [
    ('alpha2_corr_sector',     'alpha1_corr_only',       'sector adds to corr'),
    ('alpha3_corr_news',       'alpha1_corr_only',       'news adds to corr'),
    ('alpha4_corr_sector_news','alpha1_corr_only',       'full bundle adds to corr'),
    ('alpha4_corr_sector_news','alpha2_corr_sector',     'news on top of corr+sector'),
    ('alpha4_corr_sector_news','alpha3_corr_news',       'sector on top of corr+news'),
]

REGIME_CONDITIONS = {
    'full':       [0, 1, 2, 3, 4],
    'lofo4':      [0, 1, 2, 3],
    'fold4_only': [4],
}


# ──────────────────────────────────────────────────────────────────────────
# Loader: per (config), build per_day_ic_dict {fold: (n_seeds, n_days)} matrix
#         and results subset (50 cells)
# ──────────────────────────────────────────────────────────────────────────

def load_config_per_day_ic(config_name: str) -> dict:
    cfg = CONFIGS[config_name]
    per_day_dir = f"{DRIVE}/experiments/{cfg['exp_dir']}/per_day_ic"
    prefix = cfg['per_day_prefix']
    out = {}
    for fold in range(N_FOLDS):
        seed_arrays = []
        for seed in CANONICAL_SEEDS:
            path = f"{per_day_dir}/{prefix}_s{seed}_f{fold}.npy"
            seed_arrays.append(np.load(path).astype(np.float64) if os.path.exists(path) else None)
        present = [a for a in seed_arrays if a is not None]
        if not present:
            out[fold] = None
            continue
        lens = [len(a) for a in present]
        max_len = max(lens)
        mat = np.full((len(CANONICAL_SEEDS), max_len), np.nan, dtype=np.float64)
        for i, a in enumerate(seed_arrays):
            if a is None:
                continue
            mat[i, :len(a)] = a
        out[fold] = mat
    return out


def load_config_results(config_name: str) -> pd.DataFrame:
    cfg = CONFIGS[config_name]
    r = pd.read_csv(f"{DRIVE}/experiments/{cfg['exp_dir']}/results.csv")
    sub = r[cfg['results_filter'](r)].copy()
    return sub


# ──────────────────────────────────────────────────────────────────────────
# Helpers: seed aggregation within fold; paired ΔIC series within regime subset
# ──────────────────────────────────────────────────────────────────────────

def seed_aggregate_fold(per_day_ic_dict: dict, fold: int) -> Optional[np.ndarray]:
    """For one fold, average across seeds → 1D length n_days_in_fold (skip NaN)."""
    mat = per_day_ic_dict.get(fold)
    if mat is None:
        return None
    return np.nanmean(mat, axis=0)  # length = n_days_in_fold


def paired_delta_ic_series(per_day_dict_A: dict, per_day_dict_B: dict, fold_subset: list) -> np.ndarray:
    """For each fold in subset, seed-aggregate both configs and compute per-day ΔIC = A − B.
    Concatenate across folds in chronological order. Returns 1D series.
    """
    out = []
    for fold in fold_subset:
        avgA = seed_aggregate_fold(per_day_dict_A, fold)
        avgB = seed_aggregate_fold(per_day_dict_B, fold)
        if avgA is None or avgB is None:
            continue
        if len(avgA) != len(avgB):
            # CODEX-CR-EDGE-A-01 fix 2026-05-27: hard error rather than silent truncate.
            # Within-fold length mismatch across configs indicates a temporal-contract
            # violation (different test calendars / label masks) — abort post-processing.
            raise RuntimeError(
                f"Fold {fold} length mismatch A={len(avgA)} B={len(avgB)} — "
                f"likely indicates temporal contract violation (different test "
                f"calendars or label masks across configs); abort post-processing"
            )
        d = avgA - avgB
        d = d[~np.isnan(d)]
        out.append(d)
    if not out:
        return np.array([], dtype=np.float64)
    return np.concatenate(out)


def paired_sharpe_diff(results_A: pd.DataFrame, results_B: pd.DataFrame, fold_subset: list,
                        col: str = 'Sharpe_net_10bps') -> np.ndarray:
    """Paired Sharpe difference per (seed, fold) for fold_subset. Returns 1D length up to 50."""
    A_sub = results_A[results_A.fold.isin(fold_subset)]
    B_sub = results_B[results_B.fold.isin(fold_subset)]
    A_keyed = A_sub.set_index(['seed', 'fold'])[col]
    B_keyed = B_sub.set_index(['seed', 'fold'])[col]
    diff = (A_keyed - B_keyed).dropna()
    return diff.values.astype(np.float64)


# ──────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────

def main() -> int:
    print(f"[edge] loading per_day_ic + results for 4 configs ...")
    per_day_dicts = {name: load_config_per_day_ic(name) for name in CONFIGS}
    results_dfs   = {name: load_config_results(name) for name in CONFIGS}
    for name in CONFIGS:
        n_cells = len(results_dfs[name])
        n_folds_present = sum(1 for f in range(N_FOLDS) if per_day_dicts[name].get(f) is not None)
        print(f"[edge]   {name}: {n_cells} cells, {n_folds_present}/5 folds with per_day_ic")
        if n_cells != 50:
            print(f"WARN: {name} expected 50 cells, got {n_cells}")

    # ───────────────────────────────────────────────────
    # 1) Paired DM/HLN per pair × regime condition
    # ───────────────────────────────────────────────────
    print(f"\n[edge] computing 5 pairs × 3 regime conditions (DM/HLN + bootstrap CI) ...")
    pair_rows = []
    bootstrap_rows = []

    for pair_a, pair_b, pair_desc in PAIRS:
        for cond_name, fold_subset in REGIME_CONDITIONS.items():
            d_ic = paired_delta_ic_series(per_day_dicts[pair_a], per_day_dicts[pair_b], fold_subset)
            d_sh = paired_sharpe_diff(results_dfs[pair_a], results_dfs[pair_b], fold_subset, 'Sharpe_net_10bps')

            # DM/HLN — only meaningful for full (T~313) and lofo4 (T~250); Fold-4-only T~63 too small for HLN
            if cond_name == 'fold4_only':
                dm_stat, dm_p_norm, T = (np.nan, np.nan, len(d_ic))
                hln_stat, hln_p, T2 = (np.nan, np.nan, len(d_ic))
            else:
                dm_stat, dm_p_norm, T = dm_test(d_ic)
                hln_stat, hln_p, T2 = hln_test(d_ic, h=HORIZON)

            pair_rows.append({
                'pair_id': f"{pair_a}_vs_{pair_b}",
                'description': pair_desc,
                'regime_condition': cond_name,
                'n_days_paired': len(d_ic),
                'n_cells_paired': len(d_sh),
                'mean_delta_ic': round(float(d_ic.mean()) if len(d_ic) else np.nan, 5),
                'mean_delta_sharpe_net10bps': round(float(d_sh.mean()) if len(d_sh) else np.nan, 4),
                'DM_stat': round(dm_stat, 4) if not np.isnan(dm_stat) else np.nan,
                'HLN_stat': round(hln_stat, 4) if not np.isnan(hln_stat) else np.nan,
                'HLN_p_two_sided': round(hln_p, 6) if not np.isnan(hln_p) else np.nan,
            })

            # Bootstrap CI
            ic_mean, ic_lo, ic_hi = stationary_bootstrap_ci(d_ic, np.mean, n_boot=N_BOOT, block_size=BLOCK_SIZE)
            sh_mean, sh_lo, sh_hi = stationary_bootstrap_ci(d_sh, np.mean, n_boot=N_BOOT, block_size=1)
            bootstrap_rows.append({
                'pair_id': f"{pair_a}_vs_{pair_b}",
                'description': pair_desc,
                'regime_condition': cond_name,
                'delta_ic_mean': round(ic_mean, 5),
                'delta_ic_ci_lo': round(ic_lo, 5),
                'delta_ic_ci_hi': round(ic_hi, 5),
                'delta_sharpe_net10bps_mean': round(sh_mean, 4),
                'delta_sharpe_net10bps_ci_lo': round(sh_lo, 4),
                'delta_sharpe_net10bps_ci_hi': round(sh_hi, 4),
            })

    pair_df = pd.DataFrame(pair_rows)
    boot_df = pd.DataFrame(bootstrap_rows)

    # BH-FDR over 5 pairs in 'full' condition only (paper-headline test family)
    full_mask = pair_df['regime_condition'] == 'full'
    full_pvals = pair_df.loc[full_mask, 'HLN_p_two_sided'].tolist()
    rejected = bh_fdr([p for p in full_pvals if not np.isnan(p)], q=BH_FDR_Q)
    bh_col = []
    rej_idx = 0
    for _, row in pair_df.iterrows():
        if row['regime_condition'] == 'full' and not np.isnan(row['HLN_p_two_sided']):
            bh_col.append(bool(rejected[rej_idx]))
            rej_idx += 1
        else:
            bh_col.append(None)
    pair_df['BH_FDR_rejected_q05_full_family5'] = bh_col

    pair_df.to_csv(f"{OUT_DIR}/edge_pairs_dm.csv", index=False)
    boot_df.to_csv(f"{OUT_DIR}/edge_bootstrap_ci.csv", index=False)
    print(f"[edge] wrote edge_pairs_dm.csv ({len(pair_df)} rows = 5 pairs × 3 conditions)")
    print(f"[edge] wrote edge_bootstrap_ci.csv ({len(boot_df)} rows)")

    # ───────────────────────────────────────────────────
    # 2) Cost ladder per config × cost level × regime condition
    # ───────────────────────────────────────────────────
    print(f"\n[edge] computing cost ladder per (config, bps, regime) ...")
    cost_rows = []
    for config_name in CONFIGS:
        sub_all = results_dfs[config_name]
        for cond_name, fold_subset in REGIME_CONDITIONS.items():
            sub = sub_all[sub_all.fold.isin(fold_subset)]
            for bps in COST_LEVELS_BPS:
                col = f'Sharpe_net_{bps}bps'
                if col not in sub.columns:
                    continue
                vals = sub[col].values.astype(np.float64)
                if len(vals) == 0:
                    continue
                mean_, lo, hi = stationary_bootstrap_ci(vals, np.mean, n_boot=N_BOOT, block_size=1)
                cost_rows.append({
                    'config': config_name,
                    'regime_condition': cond_name,
                    'bps': bps,
                    'n_cells': len(vals),
                    'Sharpe_net_mean': round(mean_, 3),
                    'Sharpe_net_ci_lo': round(lo, 3),
                    'Sharpe_net_ci_hi': round(hi, 3),
                    'Sharpe_net_std': float(vals.std(ddof=1)) if len(vals) >= 2 else np.nan,
                })
    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(f"{OUT_DIR}/edge_cost_ladder.csv", index=False)
    print(f"[edge] wrote edge_cost_ladder.csv ({len(cost_df)} rows = 4 configs × 6 bps × 3 conditions)")

    # ───────────────────────────────────────────────────
    # 3) Human-readable summary
    # ───────────────────────────────────────────────────
    with open(f"{OUT_DIR}/edge_summary.md", "w") as f:
        f.write("# Story A E3+E4-α — Edge Ablation E6 Post-Process\n\n")
        f.write("Per plan §1.3 + H博士 2026-05-27-b decisions: 5 paired comparisons × 3 regime conditions.\n\n")
        f.write("## DM/HLN paired tests + BH-FDR (family=5 over 'full' condition)\n\n")
        f.write(pair_df.to_markdown(index=False))
        f.write("\n\n## Bootstrap CI on paired ΔIC and ΔSharpe @10bps\n\n")
        f.write("Block-bootstrap: block_size=21 for IC (per-day series), block_size=1 for Sharpe (per-cell exchangeable).\n\n")
        f.write(boot_df.to_markdown(index=False))
        f.write("\n\n## Cost ladder per (config, bps, regime condition)\n\n")
        cost_pivot = cost_df.pivot_table(values='Sharpe_net_mean', index=['config','regime_condition'],
                                          columns='bps').round(3)
        f.write(cost_pivot.to_markdown())
    print(f"[edge] wrote edge_summary.md")

    print(f"\n[edge] DONE → {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
