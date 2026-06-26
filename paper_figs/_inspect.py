"""Throwaway schema inspector for paper figs source CSVs."""
import pandas as pd
paths = [
    '/Users/heruixi/Desktop/GNN-Testing/experiments/horizon_ablation_results.csv',
    '/Users/heruixi/Desktop/GNN-Testing/experiments/graph_ablation_results.csv',
    '/Users/heruixi/Desktop/GNN-Testing/experiments/selectivenet_results.csv',
    '/Users/heruixi/Desktop/GNN-Testing/experiments/diag_phase5_permutation_importance_lgb.csv',
    '/Users/heruixi/Desktop/GNN-Testing/experiments/diag_sector_attribution_sage_mean.csv',
    '/Users/heruixi/Desktop/GNN-Testing/experiments/diag_sector_attribution_sage_sum.csv',
    '/Users/heruixi/Desktop/GNN-Testing/experiments/diag_sector_composition.csv',
    '/Users/heruixi/Desktop/GNN-Testing/artifacts/plan_aaa/ranking.csv',
    '/Users/heruixi/Desktop/GNN-Testing/artifacts/plan_aaa_t1_diagnostic/group_ranking_comparison.csv',
    '/Users/heruixi/Desktop/GNN-Testing/experiments/step3_plan_z/hansen_spa_results.csv',
    '/Users/heruixi/Desktop/GNN-Testing/experiments/step3_plan_z/part_a_daily_ic.csv',
    '/Users/heruixi/Desktop/GNN-Testing/experiments/step3_plan_z/part_b_summary.csv',
    '/Users/heruixi/Desktop/GNN-Testing/experiments/step3_plan_z/sensitivity_per_fold_ranking.csv',
    '/Users/heruixi/Desktop/GNN-Testing/experiments/step3_plan_z/pairwise_fdr.csv',
    '/Users/heruixi/Desktop/GNN-Testing/experiments/loss_horserace/results.csv',
    '/Users/heruixi/Desktop/GNN-Testing/experiments/loss_horserace/paired_delta_ic.csv',
    '/Users/heruixi/Desktop/GNN-Testing/experiments/loss_horserace/results_diagnostic_price.csv',
    '/Users/heruixi/Desktop/GNN-Testing/experiments/loss_horserace/sharpe_per_run.csv',
    '/Users/heruixi/Desktop/GNN-Testing/artifacts/tier1a_phase_b/results.csv',
    '/Users/heruixi/Desktop/GNN-Testing/artifacts/tier1b_h2_phase_b/results.csv',
    '/Users/heruixi/Desktop/GNN-Testing/artifacts/tier1c_phase_b/results.csv',
]
for path in paths:
    try:
        df = pd.read_csv(path)
        print(path)
        print(' shape:', df.shape)
        print(' cols:', list(df.columns))
        print(df.head(2).to_string())
    except Exception as exc:
        print(path, 'ERROR:', exc)
    print('---')
