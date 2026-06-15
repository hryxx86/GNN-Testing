# Univ-C GAT anomaly decomposition (Codex T3 R9-A-07)

_source: /Users/heruixi/Desktop/GNN-Testing/experiments/storya_e1_anchor/results.csv (Univ C, 480 cells)_

_LIMITATION: no per-day portfolio returns / decile positions stored → no decile-level attribution; this is the cell/fold/seed-level characterisation only._

## (1) Univ-C cross-model: IC vs Sharpe vs turnover

| model     |     IC |   IC_sd_cells |   turnover |   Sh_gross |   Sh_net10 |
|:----------|-------:|--------------:|-----------:|-----------:|-----------:|
| GAT       | 0.0182 |        0.0633 |     2.9238 |     1.8189 |     1.2908 |
| SAGE-Mean | 0.026  |        0.0574 |     2.8447 |     0.7197 |     0.2981 |
| MLP       | 0.0317 |        0.0527 |     2.9663 |     0.8239 |     0.3127 |
| LightGBM  | 0.027  |        0.0453 |     2.4637 |     1.0513 |     0.6502 |

GAT has the lowest IC (0.0182) but highest gross Sharpe (1.8189); its turnover (2.9238) is high — WELL ABOVE LightGBM's (2.4637) — so the turnover-cost story ('LightGBM is penalised for over-trading') does NOT explain GAT beating LightGBM net: the higher-turnover model still wins net.


## (2) C-GAT per-fold (regime concentration check)

|   fold | test_period   |     IC |   Sh_gross |   Sh_net10 |   turnover |   n_periods |
|-------:|:--------------|-------:|-----------:|-----------:|-----------:|------------:|
|      0 | Q2-2024       | -0.013 |     -0.166 |     -0.795 |      2.646 |           3 |
|      1 | Q3-2024       |  0.007 |      0.303 |      0.108 |      2.723 |           4 |
|      2 | Q4-2024       |  0.088 |      4.65  |      4.132 |      3.139 |           4 |
|      3 | Q1-2025       | -0.018 |      0.147 |     -0.035 |      2.782 |           3 |
|      4 | Q2-2025       |  0.15  |     13.184 |     11.994 |      3.196 |           3 |
|      5 | Q1-2023       | -0     |     -1.718 |     -1.887 |      3.154 |           3 |
|      6 | Q2-2023       | -0.054 |     -0.422 |     -0.796 |      3.064 |           3 |
|      7 | Q3-2023       | -0.02  |     -3.058 |     -4.101 |      3.132 |           3 |
|      8 | Q4-2023       |  0.043 |      3.273 |      2.405 |      3.164 |           3 |
|      9 | Q1-2024       |  0.067 |      6.238 |      5.865 |      2.808 |           3 |
|     10 | Q3-2025       | -0.037 |     -1.198 |     -1.616 |      2.751 |           4 |
|     11 | Q4-2025       |  0.006 |      0.593 |      0.215 |      2.528 |           3 |

Per-fold gross Sharpe range [-3.058, 13.184]; best fold = 4 (Q2-2025); #folds with Sh_gross>1.0 = 4/12, #folds negative = 5/12.


## (3) C-GAT per-seed (seed concentration check)

|   seed |     IC |   Sh_gross |   Sh_net10 |
|-------:|-------:|-----------:|-----------:|
|      7 |  0.022 |      0.923 |      0.515 |
|     34 |  0.023 |      0.556 |      0.224 |
|     86 |  0.023 |      7.36  |      6.242 |
|     99 |  0.025 |      2.884 |      2.179 |
|    123 |  0.013 |      1.927 |      1.433 |
|    456 |  0.017 |      2.072 |      1.578 |
|    789 |  0.006 |      0.111 |     -0.475 |
|   1024 |  0.021 |      0.559 |      0.173 |
|   2024 | -0.004 |      0.018 |     -0.391 |
|   2026 |  0.037 |      1.779 |      1.429 |

Per-seed gross Sharpe std = 2.162 (mean 1.819) → seed-dispersed.


## (4) Cell-level IC↔Sharpe_gross correlation by model (Univ C)

| model     |   n_cells |   corr(IC, Sh_gross) |   mean_IC |   mean_Sh_gross |
|:----------|----------:|---------------------:|----------:|----------------:|
| GAT       |       120 |                0.499 |    0.0182 |           1.819 |
| SAGE-Mean |       120 |                0.554 |    0.026  |           0.72  |
| MLP       |       120 |                0.663 |    0.0317 |           0.824 |
| LightGBM  |       120 |                0.791 |    0.027  |           1.051 |

If corr(IC, Sharpe) is weak/negative for GAT but positive for others, the IC→return mapping is GAT-specific (GAT earns L/S return without high rank-IC — i.e. it separates the extreme deciles but ranks the middle poorly). A positive corr across all models would instead point to a universe-wide IC-vs-Sharpe divergence.


## Honest takeaway

- **The C-GAT high Sharpe is a Fold-4 (Q2-2025) small-sample artefact.** That fold's gross Sharpe is 13.18 at only 3 periods (an annualised Sharpe that size at n=3 is statistically meaningless). Per-fold mean gross Sharpe = 1.82; dropping the single best fold (LOFO) collapses it to 0.79. The advantage is regime-concentrated, not a robust tradeable edge.

- It is NOT a turnover-cost artefact: C-GAT beats LightGBM on NET Sharpe despite TRADING MORE than LightGBM (turnover ~2.92 vs ~2.46), so 'LightGBM over-trades' cannot explain it.

- It is an IC-vs-Sharpe divergence amplified by one regime: a model can earn a top-minus-bottom L/S return (Sharpe) in a few quarters while having a low full-cross-section rank-IC overall. GAT's cell-level corr(IC, Sharpe)=0.50 is the weakest of the four models. IC is the paper's headline ranking metric; the SPA/DM null is on IC and is unaffected.

- A mechanistic decile-level attribution (which deciles drive C-GAT's L/S return) requires a targeted re-run that logs per-day portfolio returns / decile positions — NOT available in the current anchor outputs. Flagged as a follow-up; cite net-Sharpe as a separate economic endpoint, not as IC evidence.
