# Family-1 §2a confirmatory summary  (_generated 2026-06-21 15:25:42_)

**L7/Cn5 contingency**: healthy (kept in family + SPA, M=9)  (diverge_frac=0.000, collapse_frac=0.000, n=240) → L7 KEPT (M=9)


## ⚠️ STABILITY FINDING — tuned-config constant-collapse

_A degenerate cell = the tuned arm converged to a CONSTANT prediction (zero ranking ability; cross-sectional IC undefined), verified NOT a train crash (converged_flag=1, best_val_loss≈0.998 no-signal plateau). Treated as MISSING (primary=EXCLUDE; undefined ≠ measured-0). NEVER re-tuned (equal-budget symmetry)._

| universe   | arm   |   n_cells |   n_fully_degenerate |   n_partial_collapse |   n_normal |   collapse_rate |
|:-----------|:------|----------:|---------------------:|---------------------:|-----------:|----------------:|
| C          | L5s   |       120 |                   25 |                    8 |         87 |           0.275 |

**C/L5s**: the equal-budget tuned champion config degenerates to a constant predictor in **27.5%** of test fold-seeds (25 full + 8 partial of 120). Mechanism: SAGE-mean aggregation + thin data + high dropout (0.5) smooths the signal away → 'smoothing hurts ranking' evidence chain.

## C/L5s robustness — 3 treatments (appendix; all give IC ≈ 0, conclusion stable)

| treatment      |   cl5s_C_mean_IC |   C_SPA_p_consistent | is_primary   |
|:---------------|-----------------:|---------------------:|:-------------|
| exclude        |          0.00182 |               0.0774 | True         |
| zerofill       |          0.00087 |               0.0795 | False        |
| zeroskill_cell |          0.00021 |             nan      | False        |

## Hansen SPA (per universe; benchmark L0)

| universe   |   M |   T |   p_consistent | reject_h0_at_5pct   |
|:-----------|----:|----:|---------------:|:--------------------|
| B          |   9 | 749 |         0.2767 | False               |
| C          |   9 | 749 |         0.0774 | False               |

## DM/HLN pairwise (seed-avg daily ΔIC; BH-FDR over 20-test family)

| universe   | arm_A   | arm_B   |   mean_delta_IC |     HLN_p_t |   HLN_p_t_lag21 | BH_FDR_reject_family   | BH_FDR_reject_per_univ   |
|:-----------|:--------|:--------|----------------:|------------:|----------------:|:-----------------------|:-------------------------|
| B          | L1      | L0      |      0.014282   | 0.0523635   |     0.181214    | False                  | False                    |
| B          | L2      | L1      |     -0.0132676  | 0.000396805 |     0.011162    | True                   | True                     |
| B          | L6      | L2      |      0.00516993 | 0.355914    |     0.509686    | False                  | False                    |
| B          | L7      | L2      |     -0.00914242 | 0.19502     |     0.40306     | False                  | False                    |
| B          | L2s     | L2      |      0.00785303 | 0.118199    |     0.305039    | False                  | False                    |
| B          | L3      | L2      |     -0.0149259  | 0.0058975   |     0.0488423   | True                   | True                     |
| B          | L4      | L2      |     -0.00813226 | 0.226833    |     0.391599    | False                  | False                    |
| B          | L5      | L2      |     -0.0100352  | 0.11013     |     0.260386    | False                  | False                    |
| B          | L5      | L4      |     -0.0019029  | 0.172348    |     0.30141     | False                  | False                    |
| B          | L5      | L3      |      0.00489078 | 0.0658161   |     0.155424    | False                  | False                    |
| C          | L1      | L0      |      0.0147651  | 0.0108525   |     0.0631869   | True                   | True                     |
| C          | L2      | L1      |     -0.0119255  | 6.87282e-06 |     0.000981683 | True                   | True                     |
| C          | L6      | L2      |      0.0173639  | 0.00103819  |     0.0177002   | True                   | True                     |
| C          | L7      | L2      |      0.01015    | 0.00184811  |     0.0277337   | True                   | True                     |
| C          | L2s     | L2      |      0.00890957 | 0.000184262 |     0.00406934  | True                   | True                     |
| C          | L3      | L2      |     -0.0123063  | 0.00862585  |     0.0700648   | True                   | True                     |
| C          | L4      | L2      |      0.0162542  | 0.0124669   |     0.0696577   | True                   | True                     |
| C          | L5      | L2      |      0.0151588  | 0.000102086 |     0.00850731  | True                   | True                     |
| C          | L5      | L4      |     -0.00109541 | 0.831268    |     0.876863    | False                  | False                    |
| C          | L5      | L3      |      0.0274651  | 5.89528e-08 |     8.50736e-05 | True                   | True                     |

## Seed-averaged IC block-bootstrap CI per arm

| universe   | arm   |   T |   IC_mean |   IC_ci_lo |   IC_ci_hi | ci_excludes_0   |
|:-----------|:------|----:|----------:|-----------:|-----------:|:----------------|
| B          | L0    | 749 |   0.02279 |    0.00236 |    0.04294 | True            |
| B          | L1    | 749 |   0.03707 |    0.01179 |    0.06297 | True            |
| B          | L2    | 749 |   0.02381 |   -0.00539 |    0.05284 | False           |
| B          | L2s   | 749 |   0.03166 |    0.00438 |    0.0596  | True            |
| B          | L3    | 749 |   0.00888 |   -0.01475 |    0.03118 | False           |
| B          | L4    | 749 |   0.01567 |   -0.00783 |    0.03827 | False           |
| B          | L5    | 749 |   0.01377 |   -0.01098 |    0.03732 | False           |
| B          | L5s   | 749 |   0.02721 |   -0.00173 |    0.05669 | False           |
| B          | L6    | 749 |   0.02898 |    0.00146 |    0.05642 | True            |
| B          | L7    | 749 |   0.01466 |   -0.01999 |    0.04941 | False           |
| C          | L0    | 749 |   0.01953 |   -0.00702 |    0.04808 | False           |
| C          | L1    | 749 |   0.03429 |    0.00437 |    0.06694 | True            |
| C          | L2    | 749 |   0.02237 |   -0.00798 |    0.0559  | False           |
| C          | L2s   | 749 |   0.03128 |    0.00094 |    0.06538 | True            |
| C          | L3    | 749 |   0.01006 |   -0.01221 |    0.0329  | False           |
| C          | L4    | 749 |   0.03862 |    0.00024 |    0.08115 | True            |
| C          | L5    | 749 |   0.03753 |    0.00613 |    0.07214 | True            |
| C          | L5s   | 749 |   0.00182 |   -0.01903 |    0.02359 | False           |
| C          | L6    | 749 |   0.03973 |    0.00731 |    0.07561 | True            |
| C          | L7    | 749 |   0.03252 |   -0.00027 |    0.06775 | False           |

## MDE per pairwise (MDE = 2.8 × SE; 'ci_excludes_0' = detected at this design)

| universe   | pair   | is_edge_pair   |   mean_delta_IC |   delta_ci_lo |   delta_ci_hi | ci_excludes_0   |   SE_block |   MDE_2p8xSE |
|:-----------|:-------|:---------------|----------------:|--------------:|--------------:|:----------------|-----------:|-------------:|
| B          | L1-L0  | False          |         0.01428 |      -0.00507 |       0.03414 | False           |    0.00982 |      0.02749 |
| B          | L2-L1  | False          |        -0.01327 |      -0.02272 |      -0.00439 | True            |    0.00485 |      0.01359 |
| B          | L6-L2  | False          |         0.00517 |      -0.00963 |       0.02103 | False           |    0.00781 |      0.02188 |
| B          | L7-L2  | False          |        -0.00914 |      -0.03571 |       0.01031 | False           |    0.01205 |      0.03373 |
| B          | L2s-L2 | False          |         0.00785 |      -0.00777 |       0.0221  | False           |    0.00762 |      0.02133 |
| B          | L3-L2  | True           |        -0.01493 |      -0.02998 |       0.00021 | False           |    0.00777 |      0.02177 |
| B          | L4-L2  | True           |        -0.00813 |      -0.02687 |       0.01011 | False           |    0.00945 |      0.02647 |
| B          | L5-L2  | True           |        -0.01004 |      -0.02796 |       0.00722 | False           |    0.00896 |      0.0251  |
| B          | L5-L4  | True           |        -0.0019  |      -0.00538 |       0.00153 | False           |    0.00176 |      0.00494 |
| B          | L5-L3  | True           |         0.00489 |      -0.00134 |       0.01108 | False           |    0.00314 |      0.00878 |
| C          | L1-L0  | False          |         0.01477 |      -0.00036 |       0.03041 | False           |    0.00786 |      0.02201 |
| C          | L2-L1  | False          |        -0.01193 |      -0.01813 |      -0.00564 | True            |    0.00318 |      0.00891 |
| C          | L6-L2  | False          |         0.01736 |       0.00395 |       0.03019 | True            |    0.00681 |      0.01906 |
| C          | L7-L2  | False          |         0.01015 |       0.00073 |       0.01809 | True            |    0.00446 |      0.01249 |
| C          | L2s-L2 | False          |         0.00891 |       0.00373 |       0.01413 | True            |    0.00267 |      0.00748 |
| C          | L3-L2  | True           |        -0.01231 |      -0.02569 |      -1e-05   | True            |    0.00656 |      0.01838 |
| C          | L4-L2  | True           |         0.01625 |      -0.00033 |       0.03267 | False           |    0.00845 |      0.02366 |
| C          | L5-L2  | True           |         0.01516 |       0.00437 |       0.02644 | True            |    0.00564 |      0.01578 |
| C          | L5-L4  | True           |        -0.0011  |      -0.01467 |       0.01208 | False           |    0.00691 |      0.01935 |
| C          | L5-L3  | True           |         0.02747 |       0.01419 |       0.04157 | True            |    0.00698 |      0.01955 |

## LOFO sign-flip summary (pairs where dropping a fold flips the sign)

4 (pair,fold) sign-flips of 240 scanned.

| universe   | pair   |   dropped_fold |   full_mean_delta |   lofo_mean_delta |
|:-----------|:-------|---------------:|------------------:|------------------:|
| B          | L7-L2  |              9 |          -0.00945 |           0.00526 |
| C          | L5-L4  |              2 |          -0.00105 |           0.00048 |
| C          | L5-L4  |              3 |          -0.00105 |           0.00072 |
| C          | L5-L4  |              9 |          -0.00105 |           0.00382 |