# C-pre selection — pre-evaluation feature re-selection (post-hoc sensitivity)

Run: 2026-09-12 02:04:38 | git 044dd09cb7ae6a62d1045638d8387058f181e8b2 (source_clean=True) | wall 7s

Window: 231 feature dates 2021-07-01..2022-05-31, label end ≤ 2022-06-30 (TUNE_FOLD train days; val 2022H2 and the 12-fold test window are NOT used). τ = 0.5; top-15 groups.

Candidates: 168 (10 hc + 158 Alpha158, T−1). Unscored: hc_mom12m (coverage 0.368)

**Selected: 48 columns** from 15 groups; overlap with Universe C = 22 columns, with C5 = 4.

| rank | group | score = mean member abs-IC | scored/members | members |
|---|---|---|---|---|
| 1 | hc_ret_std_5d+1 | 0.05384 | 2/2 | hc_ret_std_5d,hc_ret_std_10d |
| 2 | hc_dolvol | 0.05233 | 1/1 | hc_dolvol |
| 3 | hc_ret_std_21d+1 | 0.05152 | 2/2 | hc_ret_std_21d,hc_maxret |
| 4 | STD5+1 | 0.04695 | 2/2 | STD5,STD10 |
| 5 | KLEN | 0.04174 | 1/1 | KLEN |
| 6 | MAX20+3 | 0.04157 | 4/4 | MAX20,QTLU20,MAX30,QTLU30 |
| 7 | CNTN20+1 | 0.03885 | 2/2 | CNTN20,CNTN30 |
| 8 | CNTP20+3 | 0.03873 | 4/4 | CNTP20,CNTD20,CNTP30,CNTD30 |
| 9 | BETA20+8 | 0.03841 | 9/9 | BETA20,RANK20,RSV20,IMAX20,IMXD20,SUMP20,SUMD20,RANK30,RSV30 |
| 10 | MAX5+5 | 0.03586 | 6/6 | MAX5,RSV5,MAX10,QTLU10,RANK10,RSV10 |
| 11 | STD20+1 | 0.03442 | 2/2 | STD20,STD30 |
| 12 | WVMA60 | 0.03324 | 1/1 | WVMA60 |
| 13 | CNTP5+5 | 0.03294 | 6/6 | CNTP5,CNTN5,CNTD5,CNTP10,CNTN10,CNTD10 |
| 14 | ROC20+4 | 0.03292 | 5/5 | ROC20,IMIN20,SUMN20,IMIN30,SUMN30 |
| 15 | RESI60 | 0.03198 | 1/1 | RESI60 |

Robustness (rankings only; the frozen rule is τ = 0.50 regardless):

|   tau | frozen   |   n_scored_features |   top15_overlap_with_frozen | identical_to_frozen   |
|------:|:---------|--------------------:|----------------------------:|:----------------------|
|  0    | False    |                 168 |                          14 | False                 |
|  0.5  | True     |                 167 |                          15 | True                  |
|  0.75 | False    |                 167 |                          15 | True                  |

(source: feature_scores.csv / group_scores.csv / selection.json / selector_robustness.csv in this directory)
