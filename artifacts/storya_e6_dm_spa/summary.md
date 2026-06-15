# E6 Story A statistical summary

_generated 2026-06-13 18:43:01_

## Hansen SPA (multi-comparison cherry-pick defense)

_Per-universe rows are PRIMARY; JOINT(B+C) is SUPPLEMENTARY (R9-A-03: pooled-benchmark construction is not a clean matched test)._

| universe   | role          |   M |   T |   p_consistent | reject_h0_at_5pct   |
|:-----------|:--------------|----:|----:|---------------:|:--------------------|
| B          | primary       |   3 | 749 |         0.2948 | False               |
| C          | primary       |   3 | 749 |         0.3377 | False               |
| JOINT(B+C) | supplementary |   6 | 749 |         0.4661 | False               |

## DM/HLN pairwise (paired ΔIC, seed-averaged per (date, fold), pooled across all folds; T per row)

_`HLN_p_t_lag21` = HAC lag fixed at horizon=21 (R9-A-09 robustness vs the Newey-West auto lag)._

| universe   | model_A   | model_B   |   mean_delta_IC |     HLN_p_t |   HLN_p_t_lag21 | BH_FDR_reject   |
|:-----------|:----------|:----------|----------------:|------------:|----------------:|:----------------|
| B          | GAT       | LightGBM  |     0.00984797  | 0.321631    |      0.510248   | False           |
| B          | SAGE-Mean | LightGBM  |     0.00969565  | 0.288841    |      0.46746    | False           |
| B          | MLP       | LightGBM  |     0.00603306  | 0.472496    |      0.630727   | False           |
| B          | GAT       | MLP       |     0.00381491  | 0.365166    |      0.518317   | False           |
| B          | SAGE-Mean | MLP       |     0.00366259  | 0.390258    |      0.525557   | False           |
| C          | GAT       | LightGBM  |    -0.0087558   | 0.151588    |      0.271833   | False           |
| C          | SAGE-Mean | LightGBM  |    -0.000945899 | 0.859325    |      0.88988    | False           |
| C          | MLP       | LightGBM  |     0.00476581  | 0.381609    |      0.481808   | False           |
| C          | GAT       | MLP       |    -0.0135216   | 0.000228794 |      0.00478433 | True            |
| C          | SAGE-Mean | MLP       |    -0.0057117   | 0.0672674   |      0.148753   | False           |

## HEADLINE IC CI — seed-AVERAGED (T≈749, matches SPA/DM estimand) [R9-A-04]

| universe   | model     |   T |   IC_mean |   IC_ci_lo |   IC_ci_hi | ci_excludes_0   |
|:-----------|:----------|----:|----------:|-----------:|-----------:|:----------------|
| B          | GAT       | 749 |    0.0321 |     0.0002 |     0.0623 | True            |
| B          | SAGE-Mean | 749 |    0.032  |     0.0036 |     0.0585 | True            |
| B          | MLP       | 749 |    0.0283 |     0.0033 |     0.0526 | True            |
| B          | LightGBM  | 749 |    0.0223 |    -0.0027 |     0.0461 | False           |
| C          | GAT       | 749 |    0.0182 |    -0.0139 |     0.0507 | False           |
| C          | SAGE-Mean | 749 |    0.026  |    -0.0038 |     0.0562 | False           |
| C          | MLP       | 749 |    0.0317 |     0.002  |     0.0613 | True            |
| C          | LightGBM  | 749 |    0.027  |    -0.0007 |     0.0546 | False           |

## Power / MDE per pairwise comparison (paired ΔIC; `is_edge_test`=GAT/SAGE vs non-graph MLP) [R9-A-05/02]

_`power_at_delta_0.01` = two-sided α=0.05 power to detect a true +0.01 IC edge; `MDE_80pct_power` = smallest IC effect detectable at 80% power. Non-rejection with low power = UNRESOLVED, not 'no effect'._

| universe   | pair               | is_edge_test   |   T |   mean_delta_IC |   delta_ci_lo |   delta_ci_hi | ci_excludes_0   |     SE |   power_at_delta_0.01 |   MDE_80pct_power |   SE_lag21 |   power_at_delta_0.01_lag21 |   MDE_80pct_power_lag21 |
|:-----------|:-------------------|:---------------|----:|----------------:|--------------:|--------------:|:----------------|-------:|----------------------:|------------------:|-----------:|----------------------------:|------------------------:|
| B          | GAT-LightGBM       | False          | 749 |          0.0098 |       -0.0174 |        0.0397 | False           | 0.0097 |                 0.179 |            0.0271 |     0.0145 |                       0.106 |                  0.0407 |
| B          | SAGE-Mean-LightGBM | False          | 749 |          0.0097 |       -0.0137 |        0.0366 | False           | 0.0089 |                 0.203 |            0.0249 |     0.013  |                       0.12  |                  0.0363 |
| B          | MLP-LightGBM       | False          | 749 |          0.006  |       -0.0173 |        0.0299 | False           | 0.0082 |                 0.232 |            0.0229 |     0.0122 |                       0.13  |                  0.0342 |
| B          | GAT-MLP            | True           | 749 |          0.0038 |       -0.0081 |        0.0162 | False           | 0.0041 |                 0.685 |            0.0115 |     0.0057 |                       0.414 |                  0.0161 |
| B          | SAGE-Mean-MLP      | True           | 749 |          0.0037 |       -0.0069 |        0.0159 | False           | 0.0041 |                 0.675 |            0.0116 |     0.0056 |                       0.43  |                  0.0157 |
| C          | GAT-LightGBM       | False          | 749 |         -0.0088 |       -0.0224 |        0.0054 | False           | 0.0059 |                 0.392 |            0.0166 |     0.0077 |                       0.252 |                  0.0217 |
| C          | SAGE-Mean-LightGBM | False          | 749 |         -0.0009 |       -0.013  |        0.0107 | False           | 0.0052 |                 0.487 |            0.0145 |     0.0066 |                       0.325 |                  0.0186 |
| C          | MLP-LightGBM       | False          | 749 |          0.0048 |       -0.0071 |        0.0161 | False           | 0.0053 |                 0.472 |            0.0148 |     0.0066 |                       0.33  |                  0.0185 |
| C          | GAT-MLP            | True           | 749 |         -0.0135 |       -0.0223 |       -0.005  | True            | 0.0036 |                 0.804 |            0.01   |     0.0046 |                       0.576 |                  0.013  |
| C          | SAGE-Mean-MLP      | True           | 749 |         -0.0057 |       -0.0121 |        0.001  | False           | 0.003  |                 0.91  |            0.0085 |     0.0038 |                       0.74  |                  0.0108 |

## IC block-bootstrap CI — seed-STACKED (N≈7490) — DIAGNOSTIC ONLY (anti-conservative, R9-A-04)

_Treats 10 non-independent seeds as independent days → understates width ~3x. Use the seed-averaged headline CI above for inference._

| universe   | model     |   n_per_day_obs |   n_cells |   IC_mean |   IC_mean_ci_lo |   IC_mean_ci_hi |   Sharpe_gross_mean |   Sharpe_gross_std |   Sharpe_gross_ci_lo |   Sharpe_gross_ci_hi |
|:-----------|:----------|----------------:|----------:|----------:|----------------:|----------------:|--------------------:|-------------------:|---------------------:|---------------------:|
| B          | GAT       |            7490 |       120 | 0.0321029 |      0.0214287  |       0.0423182 |            1.20709  |            2.95338 |           0.690728   |              1.74076 |
| B          | SAGE-Mean |            7490 |       120 | 0.0319506 |      0.0216554  |       0.0421381 |            1.41236  |            2.78888 |           0.919444   |              1.92947 |
| B          | MLP       |            7490 |       120 | 0.028288  |      0.0198921  |       0.0365609 |            1.50539  |            3.72299 |           0.922039   |              2.25869 |
| B          | LightGBM  |            7490 |       120 | 0.0222549 |      0.0146258  |       0.0299894 |            0.646885 |            3.26031 |           0.00824996 |              1.20245 |
| C          | GAT       |            7490 |       120 | 0.0182262 |      0.00742364 |       0.0289755 |            1.81887  |            7.79388 |           0.671403   |              3.4501  |
| C          | SAGE-Mean |            7490 |       120 | 0.0260361 |      0.0162771  |       0.0358595 |            0.719734 |            4.48155 |          -0.111928   |              1.52687 |
| C          | MLP       |            7490 |       120 | 0.0317478 |      0.0225632  |       0.0409174 |            0.823912 |            4.25611 |           0.0741308  |              1.59037 |
| C          | LightGBM  |            7490 |       120 | 0.026982  |      0.0183014  |       0.0354228 |            1.05133  |            3.10846 |           0.532848   |              1.63772 |

## Cost ladder (Net Sharpe per cost level)

|                    |        0 |        5 |       10 |        15 |        20 |        30 |
|:-------------------|---------:|---------:|---------:|----------:|----------:|----------:|
| ('B', 'GAT')       | 1.20709  | 1.05291  | 0.898577 | 0.743971  |  0.588941 |  0.276926 |
| ('B', 'LightGBM')  | 0.646885 | 0.458117 | 0.267632 | 0.0752058 | -0.119442 | -0.516727 |
| ('B', 'MLP')       | 1.50539  | 1.32763  | 1.14847  | 0.968542  |  0.788484 |  0.430108 |
| ('B', 'SAGE-Mean') | 1.41236  | 1.26066  | 1.1087   | 0.956452  |  0.803909 |  0.497864 |
| ('C', 'GAT')       | 1.81887  | 1.55686  | 1.29079  | 1.02194   |  0.751604 |  0.210399 |
| ('C', 'LightGBM')  | 1.05133  | 0.851426 | 0.650222 | 0.447954  |  0.244952 | -0.161547 |
| ('C', 'MLP')       | 0.823912 | 0.568129 | 0.312731 | 0.0578283 | -0.19648  | -0.702985 |
| ('C', 'SAGE-Mean') | 0.719734 | 0.509117 | 0.298062 | 0.0857608 | -0.128638 | -0.567109 |