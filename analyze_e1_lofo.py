#!/usr/bin/env python
"""analyze_e1_lofo.py — LOFO + per-fold + per-cell Sharpe decomposition for E1.

Addresses Codex Touchpoint 3 Round A-bis MAJOR + CONCERN findings:
  - CR-E1E6-A-bis-02: Fold 4 uniformity attribution
  - CR-E1E6-A-bis-04: Univ C GAT Net Sharpe 3.08 (CI [1.05, 6.19]) — is it Fold 4 driven?
  - CR-E1E6-A-bis-05: LightGBM Univ B failure scope

Inputs (read-only):
  /Users/heruixi/Library/CloudStorage/.../experiments/storya_e1_anchor/results.csv
  /Users/heruixi/Library/CloudStorage/.../experiments/storya_e1_anchor/per_day_ic/*.npy

Outputs:
  artifacts/storya_e6_dm_spa/lofo_diagnostic.csv  — per (universe, model, leave-out-fold)
  artifacts/storya_e6_dm_spa/per_fold_table.csv   — fold × (universe, model) IC + Sharpe
  artifacts/storya_e6_dm_spa/per_cell_distribution.csv  — for outlier flagging
  artifacts/storya_e6_dm_spa/lofo_summary.md       — human-readable
"""

import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

DRIVE = "/Users/heruixi/Library/CloudStorage/GoogleDrive-hryxx86@gmail.com/我的云端硬盘/GNN测试"
RESULTS_CSV = f"{DRIVE}/experiments/storya_e1_anchor/results.csv"
OUT_DIR = "artifacts/storya_e6_dm_spa"

FOLDS = [0, 1, 2, 3, 4]
COST_LEVELS_BPS = (0, 5, 10, 15, 20, 30)

os.makedirs(OUT_DIR, exist_ok=True)

print(f"[lofo] loading {RESULTS_CSV} ...")
r = pd.read_csv(RESULTS_CSV)
print(f"[lofo]   {len(r)} cells; cols: {list(r.columns)}")
assert len(r) == 400, f"expected 400 cells, got {len(r)}"

# ───────────────────────────────────────────────────────────────────────────────
# 1) Per-fold table: IC + Sharpe per (universe, model, fold), averaged over 10 seeds
# ───────────────────────────────────────────────────────────────────────────────

per_fold_cols = ['IC_mean', 'Sharpe_gross'] + [f'Sharpe_net_{c}bps' for c in COST_LEVELS_BPS]
per_fold = r.groupby(['universe', 'model', 'fold'])[per_fold_cols].agg(['mean', 'std']).round(4)
per_fold.columns = [f"{m}_{s}" for m, s in per_fold.columns]
per_fold.reset_index(inplace=True)
per_fold.to_csv(f"{OUT_DIR}/per_fold_table.csv", index=False)
print(f"[lofo] wrote per_fold_table.csv ({len(per_fold)} rows = 2 univ × 4 models × 5 folds)")

# ───────────────────────────────────────────────────────────────────────────────
# 2) LOFO: leave-one-fold-out IC & Sharpe per (universe, model)
# ───────────────────────────────────────────────────────────────────────────────

lofo_rows = []
for universe in ['B', 'C']:
    for model in ['GAT', 'SAGE-Mean', 'MLP', 'LightGBM']:
        sub = r[(r['universe'] == universe) & (r['model'] == model)]
        if len(sub) == 0:
            continue
        # All 5 folds × 10 seeds = 50 cells; per-cell metrics
        ic_all = sub['IC_mean'].values
        sh_all = sub['Sharpe_gross'].values
        sh_10_all = sub['Sharpe_net_10bps'].values

        # Full (no LOFO)
        lofo_rows.append({
            'universe': universe, 'model': model, 'left_out_fold': 'none',
            'n_cells': len(sub),
            'IC_mean': round(ic_all.mean(), 4),
            'IC_std_across_cells': round(ic_all.std(ddof=1), 4),
            'Sharpe_gross_mean': round(sh_all.mean(), 4),
            'Sharpe_gross_std_across_cells': round(sh_all.std(ddof=1), 4),
            'Sharpe_net_10bps_mean': round(sh_10_all.mean(), 4),
            'Sharpe_net_10bps_std_across_cells': round(sh_10_all.std(ddof=1), 4),
        })

        # Drop one fold at a time
        for leave_out in FOLDS:
            sub_lofo = sub[sub['fold'] != leave_out]
            ic = sub_lofo['IC_mean'].values
            sh = sub_lofo['Sharpe_gross'].values
            sh10 = sub_lofo['Sharpe_net_10bps'].values
            lofo_rows.append({
                'universe': universe, 'model': model,
                'left_out_fold': str(leave_out),
                'n_cells': len(sub_lofo),
                'IC_mean': round(ic.mean(), 4),
                'IC_std_across_cells': round(ic.std(ddof=1), 4),
                'Sharpe_gross_mean': round(sh.mean(), 4),
                'Sharpe_gross_std_across_cells': round(sh.std(ddof=1), 4),
                'Sharpe_net_10bps_mean': round(sh10.mean(), 4),
                'Sharpe_net_10bps_std_across_cells': round(sh10.std(ddof=1), 4),
            })

lofo_df = pd.DataFrame(lofo_rows)
lofo_df.to_csv(f"{OUT_DIR}/lofo_diagnostic.csv", index=False)
print(f"[lofo] wrote lofo_diagnostic.csv ({len(lofo_df)} rows = (5+1) × 2 univ × 4 models)")

# ───────────────────────────────────────────────────────────────────────────────
# 3) Per-cell outlier flagging: especially for Univ C GAT Sharpe 3.08 / std 9.94
# ───────────────────────────────────────────────────────────────────────────────

# For each (universe, model), list top-3 high-Sharpe + bot-3 low-Sharpe cells
outlier_rows = []
for universe in ['B', 'C']:
    for model in ['GAT', 'SAGE-Mean', 'MLP', 'LightGBM']:
        sub = r[(r['universe'] == universe) & (r['model'] == model)].copy()
        sub = sub.sort_values('Sharpe_gross', ascending=False)
        for label, idx_range in [('TOP3_Sharpe', range(3)), ('BOT3_Sharpe', range(-3, 0))]:
            for cell in sub.iloc[idx_range].itertuples():
                outlier_rows.append({
                    'universe': universe, 'model': model, 'rank_class': label,
                    'cell_id': cell.cell_id, 'fold': cell.fold, 'seed': cell.seed,
                    'IC_mean': round(cell.IC_mean, 4),
                    'Sharpe_gross': round(cell.Sharpe_gross, 4),
                    'Sharpe_net_10bps': round(cell.Sharpe_net_10bps, 4),
                    'mean_turnover_L1': round(cell.mean_turnover_L1, 3),
                })

outlier_df = pd.DataFrame(outlier_rows)
outlier_df.to_csv(f"{OUT_DIR}/per_cell_distribution.csv", index=False)
print(f"[lofo] wrote per_cell_distribution.csv ({len(outlier_df)} rows)")

# ───────────────────────────────────────────────────────────────────────────────
# 4) Headline summary markdown
# ───────────────────────────────────────────────────────────────────────────────

with open(f"{OUT_DIR}/lofo_summary.md", "w") as f:
    f.write("# E1 LOFO + per-fold + per-cell diagnostics\n\n")
    f.write("Addresses Codex Touchpoint 3 Round A-bis findings 02 (Fold 4 uniformity), "
            "04 (Univ C GAT Sharpe 3.08), 05 (LGB Univ B failure).\n\n")
    f.write("## Per-fold IC means (averaged over 10 seeds)\n\n")
    pivot_ic = r.pivot_table(values='IC_mean', index=['universe', 'model'], columns='fold', aggfunc='mean').round(4)
    f.write(pivot_ic.to_markdown())

    f.write("\n\n## Per-fold Sharpe_gross means (averaged over 10 seeds)\n\n")
    pivot_sh = r.pivot_table(values='Sharpe_gross', index=['universe', 'model'], columns='fold', aggfunc='mean').round(3)
    f.write(pivot_sh.to_markdown())

    f.write("\n\n## Per-fold Net Sharpe @10bps (averaged over 10 seeds)\n\n")
    pivot_sh10 = r.pivot_table(values='Sharpe_net_10bps', index=['universe', 'model'], columns='fold', aggfunc='mean').round(3)
    f.write(pivot_sh10.to_markdown())

    f.write("\n\n## LOFO sensitivity — IC (drop each fold; if value drops a lot, that fold drives the result)\n\n")
    lofo_ic = lofo_df.pivot_table(values='IC_mean', index=['universe', 'model'], columns='left_out_fold').round(4)
    # Reorder columns: none first, then 0,1,2,3,4
    col_order = ['none'] + [str(i) for i in range(5)]
    lofo_ic = lofo_ic[col_order]
    f.write(lofo_ic.to_markdown())

    f.write("\n\n## LOFO sensitivity — Sharpe_gross\n\n")
    lofo_sh = lofo_df.pivot_table(values='Sharpe_gross_mean', index=['universe', 'model'], columns='left_out_fold').round(3)
    lofo_sh = lofo_sh[col_order]
    f.write(lofo_sh.to_markdown())

    f.write("\n\n## LOFO sensitivity — Sharpe_net @10bps\n\n")
    lofo_sh10 = lofo_df.pivot_table(values='Sharpe_net_10bps_mean', index=['universe', 'model'], columns='left_out_fold').round(3)
    lofo_sh10 = lofo_sh10[col_order]
    f.write(lofo_sh10.to_markdown())

    f.write("\n\n## Outlier cells — top-3 / bot-3 by Sharpe_gross per (universe, model)\n\n")
    f.write(outlier_df.to_markdown(index=False))

print(f"[lofo] wrote lofo_summary.md")
print(f"[lofo] DONE → {OUT_DIR}/")
