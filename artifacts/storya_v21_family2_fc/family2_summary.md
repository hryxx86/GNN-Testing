# Family-2 FC causal edge-attribution  (_generated 2026-06-21 15:25:55_)

matched-ΔIC = CAUSAL PRIMARY (fold-level seed-avg, n≈12 blocks, block bootstrap CI, BH-FDR/6). tuned-ΔIC = DESCRIPTIVE complement (no post-hoc primary-switching).

| universe   | fc_arm   | edge_added   |   n_fold_blocks |   matched_delta_IC |    ci_lo |   ci_hi | ci_excludes_0   |   MDE_80pct | underpowered_vs_effect   |   t_p_two_sided | BH_FDR_reject   |   tuned_delta_IC | same_sign_matched_vs_tuned   |
|:-----------|:---------|:-------------|----------------:|-------------------:|---------:|--------:|:----------------|------------:|:-------------------------|----------------:|:----------------|-----------------:|:-----------------------------|
| B          | L3       | news         |              12 |            0.00153 | -0.00347 | 0.00649 | False           |     0.00737 | True                     |         0.57246 | False           |         -0.01493 | False                        |
| B          | L4       | sector       |              12 |            0.00549 | -0.00623 | 0.01754 | False           |     0.01778 | True                     |         0.40574 | False           |         -0.00813 | False                        |
| B          | L5       | sector+news  |              12 |            0.00417 | -0.00698 | 0.01631 | False           |     0.01747 | True                     |         0.51708 | False           |         -0.01004 | False                        |
| C          | L3       | news         |              12 |            0.00114 | -0.00407 | 0.00621 | False           |     0.00756 | True                     |         0.681   | False           |         -0.01231 | False                        |
| C          | L4       | sector       |              12 |            0.01369 |  0.00048 | 0.0274  | True            |     0.02    | True                     |         0.08152 | False           |          0.01625 | True                         |
| C          | L5       | sector+news  |              12 |            0.01412 |  0.00102 | 0.02841 | True            |     0.02022 | True                     |         0.07638 | False           |          0.01516 | True                         |

**0/6 contrasts survive BH-FDR q=0.05.** 6/6 are underpowered (|matched ΔIC| < MDE@80%) → 'directional but not reliable' (expected per locked power analysis).