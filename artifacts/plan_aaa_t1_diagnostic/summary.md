# Plan AAA T-1 Stability Diagnostic

Run date: 2026-05-27 | Wall time: 0.3 min

## Question

If Plan AAA's input `sp500_5y_alpha158_features_raw.npy` had been T-1-shifted (leak-free), would the same top-15 groups still appear in the top-15 by importance? Universe C composition derives from Plan AAA top-15; if top-15 is unstable under T-1 shift, the composition basis is weakened.

## Method

- Proxy importance per feature: mean per-day spearman IC vs 21d forward labels over the last 313 valid days (matches Plan AAA n_dates=313 in ranking.csv)
- Group importance: mean(|feature_IC|) over alpha158 member features
- Proxy ≠ Plan AAA permutation Δ-IC (model-dependent); single-feature IC is a directional indicator

## Top-15 Overlap Results

- Plan AAA original top-15 (from `ranking.csv`): 15 groups
- Proxy-raw (leaky alpha158) top-15: 15 groups
- Proxy-T1 (shifted alpha158) top-15: 15 groups

- **Overlap(Plan AAA orig ∩ proxy-raw) = 5/15** — sanity check that proxy aligns with Plan AAA permutation framework
- **Overlap(Plan AAA orig ∩ proxy-T1) = 5/15** — KEY: stability under leak removal
- **Overlap(proxy-raw ∩ proxy-T1) = 15/15** — direct leak-effect-on-proxy measurement

## Interpretation

- Verdict: **LOW STABILITY** (Plan AAA orig top-15 ∩ proxy-T1 top-15 = 5/15)
- Action: Universe C composition basis is leak-driven; full Plan AAA re-run required before submission, OR Universe C must be re-defined.

## Caveats

1. Single-feature IC ≠ Plan AAA permutation Δ-IC. A group can be important via member-feature interaction effects that single-feature IC misses.
2. The proxy uses |IC| aggregation; sign is discarded. Plan AAA's Δ-IC is signed.
3. Test window matched at length (313 days) but specific calendar dates may differ if data has been updated since Plan AAA ranking was generated. ranking.csv n_dates=313 is the reference target.
4. Pure-hc groups (e.g., rank-1 `hc_mom12m`) are NOT affected by alpha158 leak; they retain their Plan AAA rank by construction.

## Top-15 group-by-group detail

|   plan_aaa_orig_rank |   group_id | group_label     |   n_alpha158_members |   plan_aaa_orig_mean_delta_ic |   proxy_group_abs_ic_raw_leaky |   proxy_group_abs_ic_t1_shifted |   proxy_ic_drop_abs |   proxy_rank_raw |   proxy_rank_t1 | in_proxy_raw_top15   | in_proxy_t1_top15   |
|---------------------:|-----------:|:----------------|---------------------:|------------------------------:|-------------------------------:|--------------------------------:|--------------------:|-----------------:|----------------:|:---------------------|:--------------------|
|                    1 |          4 | hc_mom12m       |                    0 |                    0.00789921 |                     nan        |                      nan        |          nan        |               58 |              58 | False                | False               |
|                    2 |         47 | ROC30+5         |                    6 |                    0.00432301 |                       0.042036 |                        0.04244  |           -0.000403 |                7 |               8 | True                 | True                |
|                    3 |         57 | CNTP60+1        |                    2 |                    0.00338091 |                       0.012684 |                        0.012647 |            3.7e-05  |               31 |              33 | False                | False               |
|                    4 |          7 | KMID+6          |                    7 |                    0.00337225 |                       0.035343 |                        0.032509 |            0.002834 |               11 |              13 | True                 | True                |
|                    5 |         53 | RESI60          |                    1 |                    0.00335005 |                       0.02291  |                        0.029558 |           -0.006648 |               19 |              16 | False                | False               |
|                    6 |         35 | BETA20+8        |                    9 |                    0.0026512  |                       0.019118 |                        0.02262  |           -0.003502 |               21 |              19 | False                | False               |
|                    7 |         21 | CNTP5+5         |                    6 |                    0.0024592  |                       0.006662 |                        0.009215 |           -0.002553 |               42 |              37 | False                | False               |
|                    8 |         50 | ROC60+3         |                    4 |                    0.0024448  |                       0.017393 |                        0.01667  |            0.000723 |               25 |              27 | False                | False               |
|                    9 |         45 | WVMA20+1        |                    2 |                    0.00242559 |                       0.02621  |                        0.026921 |           -0.000711 |               17 |              17 | False                | False               |
|                   10 |          9 | KUP+1           |                    2 |                    0.00230692 |                       0.045769 |                        0.045646 |            0.000123 |                5 |               6 | True                 | True                |
|                   11 |         54 | RANK60+2        |                    3 |                    0.00205645 |                       0.017536 |                        0.018544 |           -0.001008 |               24 |              23 | False                | False               |
|                   12 |         41 | CNTP20+3        |                    4 |                    0.00172323 |                       0.028968 |                        0.030034 |           -0.001065 |               15 |              15 | True                 | True                |
|                   13 |          2 | hc_ret_std_5d+1 |                    0 |                    0.00150429 |                     nan        |                      nan        |          nan        |               58 |              58 | False                | False               |
|                   14 |         36 | RSQR20          |                    1 |                    0.00127327 |                       0.011172 |                        0.011618 |           -0.000446 |               35 |              35 | False                | False               |
|                   15 |         55 | CORR60          |                    1 |                    0.00106813 |                       0.036905 |                        0.037856 |           -0.000951 |               10 |               9 | True                 | True                |