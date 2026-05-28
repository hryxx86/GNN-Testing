# Story A E3+E4-α — Edge Ablation E6 Post-Process

Per plan §1.3 + H博士 2026-05-27-b decisions: 5 paired comparisons × 3 regime conditions.

## DM/HLN paired tests + BH-FDR (family=5 over 'full' condition)

| pair_id                                       | description                | regime_condition   |   n_days_paired |   n_cells_paired |   mean_delta_ic |   mean_delta_sharpe_net10bps |   DM_stat |   HLN_stat |   HLN_p_two_sided | BH_FDR_rejected_q05_full_family5   |
|:----------------------------------------------|:---------------------------|:-------------------|----------------:|-----------------:|----------------:|-----------------------------:|----------:|-----------:|------------------:|:-----------------------------------|
| alpha2_corr_sector_vs_alpha1_corr_only        | sector adds to corr        | full               |             313 |               50 |         0.00969 |                       0.1424 |    1.6717 |     1.5622 |          0.119246 | False                              |
| alpha2_corr_sector_vs_alpha1_corr_only        | sector adds to corr        | lofo4              |             251 |               40 |         0.00457 |                      -0.0371 |    0.7091 |     0.6512 |          0.515547 |                                    |
| alpha2_corr_sector_vs_alpha1_corr_only        | sector adds to corr        | fold4_only         |              62 |               10 |         0.03044 |                       0.8606 |  nan      |   nan      |        nan        |                                    |
| alpha3_corr_news_vs_alpha1_corr_only          | news adds to corr          | full               |             313 |               50 |         0.01001 |                       0.2054 |    2.2156 |     2.0705 |          0.039232 | False                              |
| alpha3_corr_news_vs_alpha1_corr_only          | news adds to corr          | lofo4              |             251 |               40 |         0.00497 |                      -0.1673 |    1.0372 |     0.9525 |          0.341756 |                                    |
| alpha3_corr_news_vs_alpha1_corr_only          | news adds to corr          | fold4_only         |              62 |               10 |         0.03041 |                       1.6961 |  nan      |   nan      |        nan        |                                    |
| alpha4_corr_sector_news_vs_alpha1_corr_only   | full bundle adds to corr   | full               |             313 |               50 |         0.00714 |                       1.1429 |    1.1724 |     1.0956 |          0.274103 | False                              |
| alpha4_corr_sector_news_vs_alpha1_corr_only   | full bundle adds to corr   | lofo4              |             251 |               40 |         0.0022  |                      -0.2225 |    0.3226 |     0.2963 |          0.767272 |                                    |
| alpha4_corr_sector_news_vs_alpha1_corr_only   | full bundle adds to corr   | fold4_only         |              62 |               10 |         0.02714 |                       6.6043 |  nan      |   nan      |        nan        |                                    |
| alpha4_corr_sector_news_vs_alpha2_corr_sector | news on top of corr+sector | full               |             313 |               50 |        -0.00255 |                       1.0004 |   -1.4159 |    -1.3232 |          0.186739 | False                              |
| alpha4_corr_sector_news_vs_alpha2_corr_sector | news on top of corr+sector | lofo4              |             251 |               40 |        -0.00237 |                      -0.1854 |   -1.1338 |    -1.0412 |          0.298781 |                                    |
| alpha4_corr_sector_news_vs_alpha2_corr_sector | news on top of corr+sector | fold4_only         |              62 |               10 |        -0.0033  |                       5.7437 |  nan      |   nan      |        nan        |                                    |
| alpha4_corr_sector_news_vs_alpha3_corr_news   | sector on top of corr+news | full               |             313 |               50 |        -0.00287 |                       0.9375 |   -0.7895 |    -0.7378 |          0.461201 | False                              |
| alpha4_corr_sector_news_vs_alpha3_corr_news   | sector on top of corr+news | lofo4              |             251 |               40 |        -0.00278 |                      -0.0552 |   -0.6739 |    -0.6189 |          0.536577 |                                    |
| alpha4_corr_sector_news_vs_alpha3_corr_news   | sector on top of corr+news | fold4_only         |              62 |               10 |        -0.00327 |                       4.9082 |  nan      |   nan      |        nan        |                                    |

## Bootstrap CI on paired ΔIC and ΔSharpe @10bps

Block-bootstrap: block_size=21 for IC (per-day series), block_size=1 for Sharpe (per-cell exchangeable).

| pair_id                                       | description                | regime_condition   |   delta_ic_mean |   delta_ic_ci_lo |   delta_ic_ci_hi |   delta_sharpe_net10bps_mean |   delta_sharpe_net10bps_ci_lo |   delta_sharpe_net10bps_ci_hi |
|:----------------------------------------------|:---------------------------|:-------------------|----------------:|-----------------:|-----------------:|-----------------------------:|------------------------------:|------------------------------:|
| alpha2_corr_sector_vs_alpha1_corr_only        | sector adds to corr        | full               |         0.00969 |         -0.01206 |          0.02805 |                       0.1424 |                       -0.6411 |                        1.0666 |
| alpha2_corr_sector_vs_alpha1_corr_only        | sector adds to corr        | lofo4              |         0.00457 |         -0.02089 |          0.02559 |                      -0.0371 |                       -0.6771 |                        0.5095 |
| alpha2_corr_sector_vs_alpha1_corr_only        | sector adds to corr        | fold4_only         |         0.03044 |          0.02655 |          0.03446 |                       0.8606 |                       -1.8728 |                        4.7323 |
| alpha3_corr_news_vs_alpha1_corr_only          | news adds to corr          | full               |         0.01001 |         -0.00695 |          0.02409 |                       0.2054 |                       -0.4806 |                        0.9632 |
| alpha3_corr_news_vs_alpha1_corr_only          | news adds to corr          | lofo4              |         0.00497 |         -0.01442 |          0.02066 |                      -0.1673 |                       -0.7061 |                        0.2696 |
| alpha3_corr_news_vs_alpha1_corr_only          | news adds to corr          | fold4_only         |         0.03041 |          0.02219 |          0.03755 |                       1.6961 |                       -0.6436 |                        4.7592 |
| alpha4_corr_sector_news_vs_alpha1_corr_only   | full bundle adds to corr   | full               |         0.00714 |         -0.0162  |          0.02562 |                       1.1429 |                       -0.7479 |                        4.3934 |
| alpha4_corr_sector_news_vs_alpha1_corr_only   | full bundle adds to corr   | lofo4              |         0.0022  |         -0.02551 |          0.02412 |                      -0.2225 |                       -0.851  |                        0.3    |
| alpha4_corr_sector_news_vs_alpha1_corr_only   | full bundle adds to corr   | fold4_only         |         0.02714 |          0.02328 |          0.03099 |                       6.6043 |                       -1.8624 |                       21.7354 |
| alpha4_corr_sector_news_vs_alpha2_corr_sector | news on top of corr+sector | full               |        -0.00255 |         -0.00601 |          0.00096 |                       1.0004 |                       -0.2778 |                        3.3538 |
| alpha4_corr_sector_news_vs_alpha2_corr_sector | news on top of corr+sector | lofo4              |        -0.00237 |         -0.00676 |          0.00198 |                      -0.1854 |                       -0.4412 |                        0.0288 |
| alpha4_corr_sector_news_vs_alpha2_corr_sector | news on top of corr+sector | fold4_only         |        -0.0033  |         -0.00498 |         -0.0017  |                       5.7437 |                       -0.1344 |                       17.0691 |
| alpha4_corr_sector_news_vs_alpha3_corr_news   | sector on top of corr+news | full               |        -0.00287 |         -0.01264 |          0.0062  |                       0.9375 |                       -0.4841 |                        3.4822 |
| alpha4_corr_sector_news_vs_alpha3_corr_news   | sector on top of corr+news | lofo4              |        -0.00278 |         -0.01491 |          0.00803 |                      -0.0552 |                       -0.3038 |                        0.1959 |
| alpha4_corr_sector_news_vs_alpha3_corr_news   | sector on top of corr+news | fold4_only         |        -0.00327 |         -0.01013 |          0.0036  |                       4.9082 |                       -1.852  |                       17.3789 |

## Cost ladder per (config, bps, regime condition)

|                                           |      0 |      5 |     10 |     15 |    20 |    30 |
|:------------------------------------------|-------:|-------:|-------:|-------:|------:|------:|
| ('alpha1_corr_only', 'fold4_only')        |  5.023 |  4.817 |  4.606 |  4.389 | 4.167 | 3.708 |
| ('alpha1_corr_only', 'full')              |  1.912 |  1.768 |  1.623 |  1.477 | 1.33  | 1.033 |
| ('alpha1_corr_only', 'lofo4')             |  1.134 |  1.006 |  0.878 |  0.749 | 0.621 | 0.364 |
| ('alpha2_corr_sector', 'fold4_only')      |  5.955 |  5.711 |  5.466 |  5.223 | 4.979 | 4.496 |
| ('alpha2_corr_sector', 'full')            |  2.023 |  1.894 |  1.766 |  1.638 | 1.51  | 1.256 |
| ('alpha2_corr_sector', 'lofo4')           |  1.04  |  0.94  |  0.841 |  0.742 | 0.643 | 0.447 |
| ('alpha3_corr_news', 'fold4_only')        |  6.822 |  6.564 |  6.302 |  6.036 | 5.766 | 5.215 |
| ('alpha3_corr_news', 'full')              |  2.102 |  1.966 |  1.829 |  1.691 | 1.554 | 1.277 |
| ('alpha3_corr_news', 'lofo4')             |  0.922 |  0.816 |  0.71  |  0.605 | 0.501 | 0.293 |
| ('alpha4_corr_sector_news', 'fold4_only') | 12.87  | 12.126 | 11.21  | 10.231 | 9.269 | 7.558 |
| ('alpha4_corr_sector_news', 'full')       |  3.255 |  3.028 |  2.766 |  2.492 | 2.222 | 1.724 |
| ('alpha4_corr_sector_news', 'lofo4')      |  0.852 |  0.753 |  0.655 |  0.557 | 0.46  | 0.266 |