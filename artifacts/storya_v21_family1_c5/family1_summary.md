# Family-1 machinery — POST-HOC SENSITIVITY (NOT confirmatory; raw HLN p, no BH)  (_generated 2026-09-11 03:15:08_)

**L7/Cn5 contingency**: SKIPPED (sensitivity mode; L7 not part of this run). **SPA**: not run. **BH-FDR**: not applied (raw HLN p).


## Hansen SPA (per universe; benchmark L0)


## DM/HLN pairwise (seed-avg daily ΔIC; raw HLN p — NO BH, sensitivity)

| universe   | arm_A   | arm_B   |   mean_delta_IC |   HLN_p_t |   HLN_p_t_lag21 | BH_FDR_reject_family   | BH_FDR_reject_per_univ   |
|:-----------|:--------|:--------|----------------:|----------:|----------------:|:-----------------------|:-------------------------|
| C5         | L1      | L0      |       0.0134327 | 0.0080228 |       0.0537258 |                        |                          |

## Seed-averaged IC block-bootstrap CI per arm

| universe   | arm   |   T |   IC_mean |   IC_ci_lo |   IC_ci_hi | ci_excludes_0   |
|:-----------|:------|----:|----------:|-----------:|-----------:|:----------------|
| C5         | L0    | 749 |   0.02031 |   -0.00373 |    0.04717 | False           |
| C5         | L1    | 749 |   0.03374 |    0.00208 |    0.07045 | True            |

## MDE per pairwise (MDE = 2.8 × SE = effect detectable with 80% power; 'ci_excludes_0' = significant at α=0.05 — an effect can be significant and still below the MDE)

| universe   | pair   | is_edge_pair   |   mean_delta_IC |   delta_ci_lo |   delta_ci_hi | ci_excludes_0   |   SE_block |   MDE_2p8xSE |
|:-----------|:-------|:---------------|----------------:|--------------:|--------------:|:----------------|-----------:|-------------:|
| C5         | L1-L0  | False          |         0.01343 |       0.00075 |       0.02833 | True            |    0.00704 |       0.0197 |

## LOFO sign-flip summary (pairs where dropping a fold flips the sign)

0 (pair,fold) sign-flips of 12 scanned.
