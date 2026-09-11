# C5 device replication — primary vs replicate (same frozen HPs, same code)

| arm   |   n_cells |   corr_cell_IC |   mean_IC_primary |   mean_IC_replicate |   mean_abs_diff |   max_abs_diff |   n_identical |   wall_primary_s |   wall_replicate_s |
|:------|----------:|---------------:|------------------:|--------------------:|----------------:|---------------:|--------------:|-----------------:|-------------------:|
| L0    |       120 |         1      |           0.02026 |             0.02026 |         0       |        0       |           120 |              1.3 |                0.6 |
| L1    |       120 |         0.9511 |           0.03368 |             0.03341 |         0.01768 |        0.10272 |             0 |             29.4 |               51.2 |

{"primary_dir": "experiments/storya_v21_main12_c5_t4", "replicate_dir": "experiments/storya_v21_main12_c5", "pooled_delta_L1_L0_primary": 0.01343, "pooled_delta_L1_L0_replicate": 0.01318}
