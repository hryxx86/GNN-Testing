# Cost-口径 (gross/net) crosswalk — confirmatory tuned ladder  (_generated 2026-06-21 17:16:25_)

**DESCRIPTIVE economic-sensitivity layer.** IC stays the sole confirmatory metric; net Sharpe is reported as cost sensitivity, NOT a second confirmatory family. Net口径 headline = 10bps (L1-one-way). Gross ΔIC + BH copied verbatim from family1_dm_hln.csv.

## ⚠️ Cost-sensitive findings (gross-IC and net-Sharpe@10bps DISAGREE in sign)

**(a) BH-significant IC claims whose口径 does NOT carry to net Sharpe — the ones that MUST be flagged in the paper:**

| universe   | pair   | claim                  |   gross_delta_IC |   net_dSharpe_10bps |   net_ci_lo |   net_ci_hi | net_ci_excludes_0   | net_sign_stable_lofo   |
|:-----------|:-------|:-----------------------|-----------------:|--------------------:|------------:|------------:|:--------------------|:-----------------------|
| C          | L3-L2  | +news edge vs corr-GAT |         -0.01231 |              0.0821 |     -0.7713 |      0.8868 | False               | False                  |

_Read carefully (PRIMARY signal = the net bootstrap CI): a BH-significant IC harm whose net ΔSharpe@10bps CI straddles 0 is NOT evidence the edge helps economically — the IC-口径 conclusion simply does not reproduce in net economic口径, and the net difference is itself indistinguishable from zero. `net_sign_stable_lofo` is a STRICT supplementary check (sign survives EVERY single-fold drop); False here reflects sign-split per-fold ΔSharpe around a near-zero mean, NOT a robust reversal._


**(b) Non-significant gross pairs that also disagree (noise-level — NOT headline claims either口径):**

| universe   | pair   | claim                  |   gross_delta_IC |   net_dSharpe_10bps | net_sign_stable_lofo   |
|:-----------|:-------|:-----------------------|-----------------:|--------------------:|:-----------------------|
| B          | L5-L4  | +news on top of sector |          -0.0019 |              0.0573 | False                  |
| C          | L5-L4  | +news on top of sector |          -0.0011 |              0.1517 | False                  |

## Headline gross/net crosswalk (all 20 pre-registered pairs)

| universe   | pair   | claim                                    |   gross_delta_IC | gross_BH_FDR_reject   |   net_dSharpe_10bps |   net_ci_lo |   net_ci_hi | net_sign_stable_lofo   | cost_sensitive   |
|:-----------|:-------|:-----------------------------------------|-----------------:|:----------------------|--------------------:|------------:|------------:|:-----------------------|:-----------------|
| B          | L1-L0  | MLP (non-graph NN) vs tuned LightGBM     |          0.01428 | False                 |              0.4321 |     -0.141  |      0.9741 | True                   | False            |
| B          | L2-L1  | corr-GAT (add graph) vs MLP              |         -0.01327 | True                  |             -0.1411 |     -0.5307 |      0.2902 | True                   | False            |
| B          | L6-L2  | complete-graph attention vs corr-GAT     |          0.00517 | False                 |              0.0821 |     -0.7769 |      0.8991 | False                  | False            |
| B          | L7-L2  | HATS relation-attention vs corr-GAT      |         -0.00914 | False                 |             -0.3464 |     -1.394  |      0.5009 | False                  | False            |
| B          | L2s-L2 | SAGE-Mean vs corr-GAT (aggregation swap) |          0.00785 | False                 |              0.0917 |     -0.6829 |      0.8962 | False                  | False            |
| B          | L3-L2  | +news edge vs corr-GAT                   |         -0.01493 | True                  |             -0.3457 |     -1.3687 |      0.6308 | True                   | False            |
| B          | L4-L2  | +sector edge vs corr-GAT                 |         -0.00813 | False                 |             -0.1006 |     -1.2435 |      0.9448 | False                  | False            |
| B          | L5-L2  | +sector+news vs corr-GAT                 |         -0.01004 | False                 |             -0.0433 |     -1.0896 |      0.9232 | False                  | False            |
| B          | L5-L4  | +news on top of sector                   |         -0.0019  | False                 |              0.0573 |     -0.2336 |      0.3275 | False                  | True             |
| B          | L5-L3  | +sector on top of news                   |          0.00489 | False                 |              0.3024 |     -0.2204 |      0.901  | True                   | False            |
| C          | L1-L0  | MLP (non-graph NN) vs tuned LightGBM     |          0.01477 | True                  |              1.1733 |      0.356  |      2.0793 | True                   | False            |
| C          | L2-L1  | corr-GAT (add graph) vs MLP              |         -0.01193 | True                  |             -0.7208 |     -1.4207 |     -0.1017 | True                   | False            |
| C          | L6-L2  | complete-graph attention vs corr-GAT     |          0.01736 | True                  |              0.4984 |     -0.7708 |      1.7774 | True                   | False            |
| C          | L7-L2  | HATS relation-attention vs corr-GAT      |          0.01015 | True                  |              0.1447 |     -1.4047 |      1.4013 | False                  | False            |
| C          | L2s-L2 | SAGE-Mean vs corr-GAT (aggregation swap) |          0.00891 | True                  |              0.0653 |     -0.6276 |      0.741  | False                  | False            |
| C          | L3-L2  | +news edge vs corr-GAT                   |         -0.01231 | True                  |              0.0821 |     -0.7713 |      0.8868 | False                  | True             |
| C          | L4-L2  | +sector edge vs corr-GAT                 |          0.01625 | True                  |              0.809  |     -0.3601 |      2.0802 | True                   | False            |
| C          | L5-L2  | +sector+news vs corr-GAT                 |          0.01516 | True                  |              0.9607 |     -0.1866 |      2.095  | True                   | False            |
| C          | L5-L4  | +news on top of sector                   |         -0.0011  | False                 |              0.1517 |     -0.4162 |      0.7114 | False                  | True             |
| C          | L5-L3  | +sector on top of news                   |          0.02747 | True                  |              0.8786 |      0.118  |      1.6509 | True                   | False            |

## Per-arm net cost-ladder (descriptive; mean + median + LOFO-Q2-2025 + heavy-tail flag)

_`max_abs_Sharpe_gross_cell` flags heavy-tailed cells (the prior E6 had a single Sharpe=75 cell inflate a mean); compare `Sharpe_net_mean` vs `Sharpe_net_median` and `Sharpe_net_drop_q2_2025` for fragility._

| universe   | arm   | model         |   n_cells |   Sharpe_net_mean |   Sharpe_net_median |   Sharpe_net_ci_lo |   Sharpe_net_ci_hi |   Sharpe_net_drop_q2_2025 |   mean_turnover_L1 |   max_abs_Sharpe_gross_cell |
|:-----------|:------|:--------------|----------:|------------------:|--------------------:|-------------------:|-------------------:|--------------------------:|-------------------:|----------------------------:|
| B          | L0    | LightGBM      |       120 |            0.5892 |              0.5825 |             0.1568 |             1.0132 |                    0.4723 |              2.007 |                        8.42 |
| B          | L1    | MLP           |       120 |            1.0213 |              0.4132 |             0.5641 |             1.4903 |                    0.7726 |              2.612 |                       12.03 |
| B          | L2    | GAT           |       120 |            0.8802 |              0.4962 |             0.3711 |             1.3755 |                    0.7135 |              2.75  |                       11.33 |
| B          | L2s   | SAGE-Mean     |       120 |            0.9719 |              0.7174 |             0.533  |             1.4299 |                    0.9733 |              2.454 |                       11.43 |
| B          | L3    | GAT           |       120 |            0.5345 |              0.5554 |            -0.0449 |             1.1569 |                    0.5188 |              3.03  |                       19.99 |
| B          | L4    | GAT           |       120 |            0.7797 |              0.8055 |             0.2663 |             1.3141 |                    0.8777 |              2.954 |                       10.93 |
| B          | L5    | GAT           |       120 |            0.8369 |              0.5715 |             0.3105 |             1.4103 |                    0.9111 |              2.994 |                       18.46 |
| B          | L5s   | SAGE-Mean     |       120 |            0.8816 |              0.7953 |             0.3157 |             1.4204 |                    0.8187 |              2.396 |                       12.04 |
| B          | L6    | GAT           |       120 |            0.9624 |              0.7423 |             0.5226 |             1.3934 |                    0.7625 |              2.602 |                        9.7  |
| B          | L7    | HATS-3R-adapt |       120 |            0.5338 |              0.7835 |             0.0595 |             0.9849 |                    0.8002 |              2.408 |                       11.73 |
| C          | L0    | LightGBM      |       120 |           -0.2191 |             -0.3333 |            -0.8161 |             0.3644 |                   -0.536  |              2.253 |                       10.59 |
| C          | L1    | MLP           |       120 |            0.9542 |              0.1818 |             0.1751 |             1.7738 |                    0.2973 |              2.895 |                       31.69 |
| C          | L2    | GAT           |       120 |            0.2334 |             -0.0403 |            -0.5902 |             0.9722 |                   -0.2391 |              2.913 |                       18.92 |
| C          | L2s   | SAGE-Mean     |       120 |            0.2987 |              0.12   |            -0.8733 |             1.1016 |                   -0.148  |              2.917 |                       38.8  |
| C          | L3    | GAT           |       120 |            0.3155 |             -0.0106 |            -0.2977 |             0.9876 |                   -0.1691 |              3.19  |                       20.55 |
| C          | L4    | GAT           |       120 |            1.0423 |              0.1302 |             0.3193 |             1.8975 |                    0.1444 |              2.703 |                       27.82 |
| C          | L5    | GAT           |       120 |            1.194  |              0.6534 |             0.5487 |             1.8633 |                    0.465  |              3.016 |                       19    |
| C          | L5s   | SAGE-Mean     |        87 |           -0.446  |             -0.2852 |            -1.251  |             0.3096 |                   -0.4686 |              2.783 |                       15.59 |
| C          | L6    | GAT           |       120 |            0.7318 |              0.7063 |            -0.2027 |             1.5747 |                    0.1737 |              2.85  |                       23.91 |
| C          | L7    | HATS-3R-adapt |       120 |            0.378  |              0.1578 |            -1.0792 |             1.504  |                   -0.3634 |              2.829 |                       45.88 |

_(net @ 10bps shown; full 0–30bps ladder in cost_ladder_by_arm.csv)_


## FC arm net ΔSharpe (Family-2 contrasts; descriptive)

| universe   | fc_arm   | edge_added   |   n_fold_blocks |   dSharpe_net_mean |   ci_lo |   ci_hi | ci_excludes_0   |
|:-----------|:---------|:-------------|----------------:|-------------------:|--------:|--------:|:----------------|
| B          | L3       | news         |              12 |             0.0053 | -0.3955 |  0.4331 | False           |
| B          | L4       | sector       |              12 |             0.2098 | -0.3801 |  0.8327 | False           |
| B          | L5       | sector+news  |              12 |             0.1176 | -0.448  |  0.7669 | False           |
| C          | L3       | news         |              12 |            -0.3756 | -0.9061 |  0.0588 | False           |
| C          | L4       | sector       |              12 |             0.8566 | -0.2788 |  2.0237 | False           |
| C          | L5       | sector+news  |              12 |             0.8133 | -0.3218 |  1.9682 | False           |