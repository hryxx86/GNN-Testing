# Family-1 machinery — POST-HOC SENSITIVITY (NOT confirmatory; raw HLN p, no BH)  (_generated 2026-09-11 02:14:21_)

**L7/Cn5 contingency**: SKIPPED (sensitivity mode; L7 not part of this run). **SPA**: not run. **BH-FDR**: not applied (raw HLN p).


## Hansen SPA (per universe; benchmark L0)


## DM/HLN pairwise (seed-avg daily ΔIC; raw HLN p — NO BH, sensitivity)

| universe   | arm_A   | arm_B   |   mean_delta_IC |   HLN_p_t |   HLN_p_t_lag21 | BH_FDR_reject_family   | BH_FDR_reject_per_univ   |
|:-----------|:--------|:--------|----------------:|----------:|----------------:|:-----------------------|:-------------------------|
| C5         | L1      | L0      |       0.0131759 | 0.0125196 |        0.067178 |                        |                          |

## Seed-averaged IC block-bootstrap CI per arm

| universe   | arm   |   T |   IC_mean |   IC_ci_lo |   IC_ci_hi | ci_excludes_0   |
|:-----------|:------|----:|----------:|-----------:|-----------:|:----------------|
| C5         | L0    | 749 |   0.02031 |   -0.00373 |    0.04717 | False           |
| C5         | L1    | 749 |   0.03349 |    0.00201 |    0.07076 | True            |

## MDE per pairwise (MDE = 2.8 × SE; 'ci_excludes_0' = detected at this design)

| universe   | pair   | is_edge_pair   |   mean_delta_IC |   delta_ci_lo |   delta_ci_hi | ci_excludes_0   |   SE_block |   MDE_2p8xSE |
|:-----------|:-------|:---------------|----------------:|--------------:|--------------:|:----------------|-----------:|-------------:|
| C5         | L1-L0  | False          |         0.01318 |       0.00021 |       0.02786 | True            |    0.00707 |       0.0198 |

## LOFO sign-flip summary (pairs where dropping a fold flips the sign)

0 (pair,fold) sign-flips of 12 scanned.
