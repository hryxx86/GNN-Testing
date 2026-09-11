# C5 device replication — same frozen HPs (md5 cdb4d923…), same code; Colab T4 (CUDA, primary) vs Mac M4 (MPS/CPU, replicate)

| arm   |   n_cells |   corr_cell_IC |   mean_IC_mac |   mean_IC_t4 |   mean_abs_diff |   max_abs_diff |   n_identical |   wall_mac_s |   wall_t4_s |
|:------|----------:|---------------:|--------------:|-------------:|----------------:|---------------:|--------------:|-------------:|------------:|
| L0    |       120 |         1      |       0.02026 |      0.02026 |         0       |        0       |           120 |          0.6 |         1.3 |
| L1    |       120 |         0.9511 |       0.03341 |      0.03368 |         0.01768 |        0.10272 |             0 |         51.2 |        29.4 |

{"pooled_delta_L1_L0_mac": 0.01318, "pooled_delta_L1_L0_t4": 0.01343}

(source: experiments/storya_v21_main12_c5_t4/results.csv vs experiments/storya_v21_main12_c5/results.csv; cell-level IC_mean joined on arm/seed/fold)
