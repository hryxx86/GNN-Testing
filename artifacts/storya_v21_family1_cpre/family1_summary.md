# Family-1 machinery — POST-HOC SENSITIVITY (NOT confirmatory; raw HLN p, no BH)  (_generated 2026-09-12 04:01:29_)

**L7/Cn5 contingency**: SKIPPED (sensitivity mode; L7 not part of this run). **SPA**: not run. **BH-FDR**: not applied (raw HLN p).


## Hansen SPA (per universe; benchmark L0)


## DM/HLN pairwise (seed-avg daily ΔIC; raw HLN p — NO BH, sensitivity)

| universe   | arm_A   | arm_B   |   mean_delta_IC |   HLN_p_t |   HLN_p_t_lag21 | BH_FDR_reject_family   | BH_FDR_reject_per_univ   |
|:-----------|:--------|:--------|----------------:|----------:|----------------:|:-----------------------|:-------------------------|
| CPRE       | L1      | L0      |     -0.00241929 |  0.785537 |        0.846583 |                        |                          |

## Seed-averaged IC block-bootstrap CI per arm

| universe   | arm   |   T |   IC_mean |   IC_ci_lo |   IC_ci_hi | ci_excludes_0   |
|:-----------|:------|----:|----------:|-----------:|-----------:|:----------------|
| CPRE       | L0    | 749 |   0.00568 |   -0.02301 |    0.03759 | False           |
| CPRE       | L1    | 749 |   0.00326 |   -0.01807 |    0.02449 | False           |

## MDE per pairwise (MDE = 2.8 × SE = effect detectable with 80% power; 'ci_excludes_0' = significant at α=0.05 — an effect can be significant and still below the MDE)

| universe   | pair   | is_edge_pair   |   mean_delta_IC |   delta_ci_lo |   delta_ci_hi | ci_excludes_0   |   SE_block |   MDE_2p8xSE |
|:-----------|:-------|:---------------|----------------:|--------------:|--------------:|:----------------|-----------:|-------------:|
| CPRE       | L1-L0  | False          |        -0.00242 |      -0.02563 |       0.01784 | False           |    0.01119 |      0.03134 |

## LOFO sign-flip summary (pairs where dropping a fold flips the sign)

3 (pair,fold) sign-flips of 12 scanned.

| universe   | pair   |   dropped_fold |   full_mean_delta |   lofo_mean_delta |
|:-----------|:-------|---------------:|------------------:|------------------:|
| CPRE       | L1-L0  |              1 |          -0.00214 |           4e-05   |
| CPRE       | L1-L0  |              9 |          -0.00214 |           0.01027 |
| CPRE       | L1-L0  |             10 |          -0.00214 |           0.00088 |