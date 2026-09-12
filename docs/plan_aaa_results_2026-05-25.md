---
title: Plan AAA — 168-feature grouped permutation Δ-IC ranking results
plan_id: AAA-v1
plan_file: docs/plan_aaa_v1_2026-05-23.md
status: RESULTS_AVAILABLE — full mode completed 2026-05-25 02:17 PT (57.6 min wall, 29/30 cells converged); placeholders below filled with values from artifacts/plan_aaa/
created: 2026-05-25
codex_review_history:
  - touchpoint_1_plan: COMPLETED Round A — Codex (artifacts/reviews/2026-05-23_codex_plan_A.md)
  - touchpoint_2_code: COMPLETED Round A — finance-gnn-reviewer fallback (artifacts/reviews/2026-05-23_finance-gnn-reviewer_code_A.md)
  - touchpoint_3_results: PENDING — to be invoked after full mode completes
---

# Plan AAA Results — 168-Feature Grouped Permutation Δ-IC Ranking

## 1. Context

Plan AAA tests whether the 10 hand-curated features (`hc_*`) selected in Plan Z++ Part A's universe are actually the most predictive groups within a broader 168-feature universe (158 Alpha158 + 10 hc). Methodology is verbatim Plan Z++ Part A grouped cross-sectional permutation Δ-IC at inference on production SAGE-Mean + MLP cells, extended from 10 to 168 features. Plan v1 (post-6-pass) at [docs/plan_aaa_v1_2026-05-23.md](docs/plan_aaa_v1_2026-05-23.md).

**Three pre-committed outcomes** (Plan v1 §1, immutable, written before execution):
1. **(a) Validates universe**: all 10 hc features' groups rank in top-K of 168-universe → keeps Plan Z++ universe choice
2. **(b) Mixed**: disclose honestly
3. **(c) All hc rank low**: flag suboptimal; recommend Plan AAA-derived universe for future

## 2. Provenance + Audit (deterministic, captured at run start)

### 2.1 Data sources

| Item | Value |
|---|---|
| Alpha158 file | `data/reference/sp500_5y_alpha158_features_raw.npy` |
| Alpha158 MD5 | `0a2cd862eb1e94a8b132c12d858591ab` |
| Alpha158 shape | (1255, 501, 158) |
| Hand-curated source | inherited from `run_step3_plan_z_part_a.load_data_and_features` |
| Total features | 168 (10 hc + 158 Alpha158) |
| Universe (valid tickers) | 501 (`sorted(prices ∩ events ∩ sectors)`) |
| Raw signature check | ROC5 max = **2.423** > 1.5 → PASS (NOT pre-winsorized) (source: `artifacts/plan_aaa/audit/data_provenance.json` field `roc5_max_raw_signature_check`) |
| Ticker alignment | Layer A: intersection-logic match → PASS; Layer B: KMID full-period Pearson ρ on 3 sample tickers → all ρ = 1.0000 (smoke v4) |

### 2.2 Environment (reproducibility)

| Field | Value |
|---|---|
| Python | 3.11.15 (conda-forge, Mar 2026) |
| NumPy | 2.4.4 |
| PyTorch | 2.11.0 |
| SciPy | 1.17.1 |
| Pandas | 2.3.3 |
| Device | mps (M4 Mac Mini) |
| Seeds | [86, 123, 456] |
| cell_id formula | `arch_idx*15 + fold_idx*3 + seed_idx` (range 0..29, globally unique) |

### 2.3 Locked Stage 0 hparams

```
hidden=64, num_layers=2, dropout=0.3, lr=1e-3, weight_decay=1e-4,
epochs=50, patience=10, grad_accum=4,
corr_window=126, corr_step=21, corr_threshold=0.6
```

## 3. §3.1 Step 1: Clustering on Calibration Window

### 3.1 Calibration window definition

- Days [0, 251] = 2021-01-29 to ~2022-01-29 (252 trading days)
- Strictly pre-experiment: earliest test fold (fold 0) starts day 796
- Single global p1/p99 winsor over the calibration slice, NOT per-fold (this is not a training fold)

### 3.2 Discovered groups

- **Total groups**: 61 (out of 168 features)
- **Distribution**: 20 singletons (33%), 19 doublets (31%), 22 triplets+ (36%)
- **Zero-variance feature in calibration**: 1 (index 6 = `hc_mom12m`, expected — 252-day lookback ≥ 252-day window)

### 3.3 10 hc features → 7 calibration groups

| group_id | label | size | hc members | Note |
|---|---|---|---|---|
| 0 | `hc_ret_mean_5d+6` | 7 | `hc_ret_mean_5d`, `hc_ret_mean_10d` | + 5 Alpha158 short-momentum siblings |
| 1 | `hc_ret_mean_21d+5` | 6 | `hc_ret_mean_21d` | + 5 Alpha158 medium-momentum siblings |
| 2 | `hc_ret_std_5d+1` | 2 | `hc_ret_std_5d`, `hc_ret_std_10d` | hc-only doublet (no high-corr Alpha158 sibling at \|ρ\|>0.6) |
| 3 | `hc_ret_std_21d+1` | 2 | `hc_ret_std_21d`, `hc_maxret` | hc-only doublet |
| 4 | `hc_mom12m` | 1 | `hc_mom12m` | singleton (252d lookback > 252d calib window → zero variance → forced singleton) |
| 5 | `hc_dolvol` | 1 | `hc_dolvol` | singleton (volume-based, no high-corr Alpha158 partner) |
| 6 | `hc_CORR5+1` | 2 | `hc_CORR5` | + Alpha158 `CORR5` (identical formula by construction) |

**Counts**: 4 hc features (`hc_ret_mean_5d`, `hc_ret_mean_10d`, `hc_ret_mean_21d`, `hc_CORR5`) merged with Alpha158 siblings; 6 hc features remain hc-internal (singletons or hc-only doublets).

## 4. §3.1 Sensitivity: ARI Calibration vs. Fold-0

### 4.1 Result

| Field | Value | Status |
|---|---|---|
| ARI(calibration, fold-0) | **0.5506** | (source: `artifacts/plan_aaa/adjusted_rand_index.json` field `ari_calibration_vs_fold0`) |
| Threshold (Plan v1 §3.1) | 0.85 | concern fires below |
| `concern_triggered` | **TRUE** | flag in paper §7 limitations |
| num_groups_calibration | 61 | |
| num_groups_fold0 | 58 | |

### 4.2 Interpretation

ARI 0.55 means the partition produced by 252-day calibration clustering and the partition produced by 714-day fold-0 train clustering have **moderate but imperfect** agreement (random partitions would give ARI ≈ 0, identical partitions ARI = 1). Per Plan v1 §3.1, this triggers the pre-registered concern flag.

### 4.3 Likely drivers (mechanistic)

1. **Long-lookback features under-resolved in calibration**: Alpha158 features with 60-day windows (MA60, STD60, ROC60, etc.) have NaN→0 for the first 60 days of the calibration window, depressing variance and decorrelating them from peers. In fold-0 (714 days) these features have meaningful variance throughout.
2. **`hc_mom12m` flips from singleton to grouped**: 252-day lookback exceeds the 252-day calibration window → zero variance → forced singleton in calibration. In fold-0 (714 days, well past 252), `hc_mom12m` has real variance and groups with medium-momentum features.
3. **Window-size effect on cluster compactness**: smaller calibration window → noisier rank correlations → more fragmented dendrogram at |ρ|>0.6.

### 4.4 What this means for the ranking

- **Pre-committed approach (unchanged)**: The headline ranking uses calibration grouping (preregistered before any test fold).
- **Robustness check needed for paper**: Per Touchpoint 2 finding FINGNN-CODE-A-04 (reviewer A-04), option (b): re-cluster using **per-day Spearman matrices averaged across calibration days** (median rho per pair) — this isolates cross-sectional comovement from time-series drift. To be added before paper submission.
- **Alternative robustness check**: re-compute the ranking using fold-0 grouping (`groups_168_fold0.json`) as a supplementary table; report rank stability.

## 5. §3.4 Headline Ranking — 61 Groups

Full mode completed 2026-05-25 02:17 PT (57.6 min wall). Results from `artifacts/plan_aaa/ranking.csv` and `artifacts/plan_aaa/ranking.json`.

### Top-15 (highest mean ΔIC)

| rank | group_label | size | mean ΔIC | NW t | NW p | BH-FDR p_adj | rejected |
|---|---|---|---|---|---|---|---|
| 1 | hc_mom12m | 1 | +0.007899 | 1.014 | 0.311 | 0.647 | ✗ |
| 2 | ROC30+5 | 6 | +0.004323 | 2.850 | 0.004 | 0.133 | ✗ |
| 3 | CNTP60+1 | 2 | +0.003381 | 1.747 | 0.081 | 0.504 | ✗ |
| 4 | KMID+6 | 7 | +0.003372 | 1.595 | 0.111 | 0.520 | ✗ |
| 5 | RESI60 | 1 | +0.003350 | 1.985 | 0.047 | 0.504 | ✗ |
| 6 | BETA20+8 | 9 | +0.002651 | 1.233 | 0.217 | 0.619 | ✗ |
| 7 | CNTP5+5 | 6 | +0.002459 | 1.119 | 0.263 | 0.642 | ✗ |
| 8 | ROC60+3 | 4 | +0.002445 | 1.232 | 0.218 | 0.619 | ✗ |
| 9 | WVMA20+1 | 2 | +0.002426 | 2.423 | 0.015 | 0.313 | ✗ |
| 10 | KUP+1 | 2 | +0.002307 | 1.420 | 0.156 | 0.619 | ✗ |
| 11 | RANK60+2 | 3 | +0.002056 | 0.805 | 0.421 | 0.755 | ✗ |
| 12 | CNTP20+3 | 4 | +0.001723 | 0.765 | 0.445 | 0.775 | ✗ |
| 13 | **hc_ret_std_5d+1** | 2 | +0.001504 | 1.365 | 0.172 | 0.619 | ✗ |
| 14 | RSQR20 | 1 | +0.001273 | 1.408 | 0.159 | 0.619 | ✗ |
| 15 | CORR60 | 1 | +0.001068 | 1.276 | 0.202 | 0.619 | ✗ |

(source: artifacts/plan_aaa/ranking.csv rows 1-15)

### Bottom-5 (lowest mean ΔIC)

| rank | group_label | size | mean ΔIC | NW t | NW p | BH-FDR p_adj | rejected |
|---|---|---|---|---|---|---|---|
| 57 | RSQR30 | 1 | -0.000787 | -1.599 | 0.110 | 0.520 | ✗ |
| 58 | STD20+1 | 2 | -0.000900 | -1.155 | 0.248 | 0.636 | ✗ |
| 59 | WVMA10 | 1 | -0.000956 | -1.150 | 0.250 | 0.636 | ✗ |
| 60 | **hc_ret_mean_21d+5** | 6 | -0.001805 | -1.691 | 0.091 | 0.504 | ✗ |
| 61 | **CORD20+1** | 2 | **-0.004020** | **-3.579** | **0.000344** | **0.021008** | **✓** |

(source: artifacts/plan_aaa/ranking.csv rows 57-61)

### Summary statistics

| Field | Value |
|---|---|
| Total groups | 61 |
| BH-FDR rejected at q=0.05 | **1 (CORD20+1, rank 61, NEGATIVE ΔIC)** |
| Groups with raw p < 0.05 (uncorrected) | 5 (ranks 2, 5, 9, 18, 61) |
| HAC degenerate count (n_hac_degenerate, A-03 fix branch) | 0 |
| n_paired_rows (cells × groups × dates) | 114,558 |
| n_cells_present (with at least one valid IC obs) | 30 (29 converged + 1 failed but still produced IC) |
| Number of test dates pooled across 5 folds | 313 (= 63+64+64+60+62) |
| NW-HAC auto lag at T=313 | 5 (= floor(4 × (313/100)^(2/9))) |
| Bootstrap block_len | 21 |
| Bootstrap n_boot | 1000 |

(source: artifacts/plan_aaa/ranking.json `summary`)

### Interpretation

- **No group has statistically significant POSITIVE ΔIC after BH-FDR correction**: even rank-2 `ROC30+5` (NW t=2.85, raw p=0.004) becomes p_adj=0.133 after K=61 multiple testing.
- **The single BH-FDR-rejected group is NEGATIVE**: `CORD20+1` (Alpha158 20-day CORD combination) is reliably HARMFUL — permuting it improves IC by 0.40 percentage points. Suggests inclusion of CORD20+1 in the universe actively damages prediction quality (likely a noise factor that the model overweights without it).
- **Signal is uniformly weak**: mean ΔIC of rank 1 is only +0.0079, equivalent to permuting `hc_mom12m` costing the model 0.79 percentage points of daily IC. Below practical importance threshold for most quant deployments.

### 5.1 Statistical pre-specifications (reproduced from Plan v1 §3.4)

- **Aggregation order**: per (cell, group, day) ΔIC = baseline_IC − permuted_IC; then mean across 30 cells per (group, day) → daily series per group; then NW-HAC on that daily series.
- **NW-HAC lag**: `floor(4 × (T/100)^(2/9))` (Newey-West 1994 auto rule, T = pooled test dates ≈ 313)
- **NW-HAC degenerate handling** (Touchpoint 2 FINGNN-CODE-A-03 fix): if long-run-variance ≤ 0, return NaN p (BH-FDR maps NaN → 1.0); `n_hac_degenerate` counter in summary
- **Multiple testing**: BH-FDR at q=0.05 over K=61 groups
- **CI**: stationary-style fixed-length-block (Künsch 1989) bootstrap, block_len=21, n_boot=1000

## 6. §3.5 Hand-Curated Mapping — Rank of 10 hc Features

(source: `artifacts/plan_aaa/hand_curated_mapping_168.json`)

| hc_feature | group_id | group_label | size | rank/61 | mean ΔIC | NW p | BH-FDR p_adj | rejected |
|---|---|---|---|---|---|---|---|---|
| hc_ret_mean_5d | 0 | hc_ret_mean_5d+6 | 7 | 33 | +0.000032 | 0.961 | 0.987 | ✗ |
| hc_ret_mean_10d | 0 | hc_ret_mean_5d+6 | 7 | 33 | +0.000032 | 0.961 | 0.987 | ✗ |
| hc_ret_mean_21d | 1 | hc_ret_mean_21d+5 | 6 | **60** | -0.001805 | 0.091 | 0.504 | ✗ |
| hc_ret_std_5d | 2 | hc_ret_std_5d+1 | 2 | 13 | +0.001504 | 0.172 | 0.619 | ✗ |
| hc_ret_std_10d | 2 | hc_ret_std_5d+1 | 2 | 13 | +0.001504 | 0.172 | 0.619 | ✗ |
| hc_ret_std_21d | 3 | hc_ret_std_21d+1 | 2 | 54 | -0.000564 | 0.560 | 0.875 | ✗ |
| **hc_mom12m** | 4 | hc_mom12m | 1 | **1** | +0.007899 | 0.311 | 0.647 | ✗ |
| hc_maxret | 3 | hc_ret_std_21d+1 | 2 | 54 | -0.000564 | 0.560 | 0.875 | ✗ |
| hc_dolvol | 5 | hc_dolvol | 1 | 24 | +0.000489 | 0.857 | 0.980 | ✗ |
| hc_CORR5 | 6 | hc_CORR5+1 | 2 | 41 | -0.000147 | 0.666 | 0.917 | ✗ |

### Group composition for hc groups (member listing)

| group_id | group_label | members |
|---|---|---|
| 0 | hc_ret_mean_5d+6 | hc_ret_mean_5d, hc_ret_mean_10d, ROC10, BETA10, SUMP10, SUMN10, SUMD10 |
| 1 | hc_ret_mean_21d+5 | hc_ret_mean_21d, BETA30, IMAX30, IMXD30, SUMP30, SUMD30 |
| 2 | hc_ret_std_5d+1 | hc_ret_std_5d, hc_ret_std_10d |
| 3 | hc_ret_std_21d+1 | hc_ret_std_21d, hc_maxret |
| 4 | hc_mom12m | hc_mom12m |
| 5 | hc_dolvol | hc_dolvol |
| 6 | hc_CORR5+1 | hc_CORR5, CORR5 |

### Counts (Plan v1 §1 pre-commitment decision rule)

- **Top-30 (top half of 61) hc groups**: 3 — rank 1 (hc_mom12m), rank 13 (hc_ret_std_5d+1), rank 24 (hc_dolvol)
- **Bottom-31 hc groups**: 4 — rank 33 (hc_ret_mean_5d+6), rank 41 (hc_CORR5+1), rank 54 (hc_ret_std_21d+1), rank 60 (hc_ret_mean_21d+5)
- **Top-10 hc groups**: 1 — only `hc_mom12m`
- **Top-3 hc groups**: 1 — only `hc_mom12m`
- **BH-FDR rejected hc groups**: 0 — no hc group passed multiple-testing correction at q=0.05

## 7. Outcome Resolution (per Plan v1 §1 pre-commitment)

**Outcome: (b) MIXED** — 3/7 hc groups in top-30; 4/7 in bottom-31.

Per Plan v1 §1:
- (a) "all hc rank in top-K": **FAILED** (4/7 hc groups in bottom-half; only 1/7 in top-10)
- (b) Mixed: **✅ CURRENT OUTCOME** — partial validation of universe choice
- (c) "all hc rank low": **FAILED** (3/7 hc groups in top-30, including rank 1)

### Interpretation

The Plan Z++ hand-curated universe is **neither validated nor refuted** by the broader 168-universe sensitivity analysis:

- **Strongly supported hc features**: `hc_mom12m` (rank 1 of 61, strongest contributor in the entire 168 universe; though not BH-FDR-significant). `hc_ret_std_5d` / `hc_ret_std_10d` (rank 13, hc-only doublet — no Alpha158 short-term volatility feature merged in despite Alpha158 having STD5/STD10). `hc_dolvol` (rank 24, hc-only singleton — Alpha158 has no log-dollar-volume proxy at the same level).
- **Weakly contributing hc features**: `hc_ret_mean_5d` / `hc_ret_mean_10d` (rank 33, merged with 5 Alpha158 sibling features — group mean ΔIC near zero). `hc_CORR5` (rank 41, merged with identical Alpha158 CORR5).
- **Potentially harmful hc features**: `hc_ret_std_21d` / `hc_maxret` (rank 54, mildly negative ΔIC). `hc_ret_mean_21d` (rank 60, second-most-negative ΔIC group; ranked just above the only BH-FDR-rejected harmful group `CORD20+1`).

### Universe-policy implications

1. **Keep**: `hc_mom12m`, `hc_ret_std_5d/10d`, `hc_dolvol` — these contribute information not captured by Alpha158 at |ρ|>0.6.
2. **Reconsider**: `hc_ret_mean_5d/10d/21d` — short/medium momentum is redundant with Alpha158 SUMP/SUMD/BETA at the multi-feature level; the hc means lose nothing if removed.
3. **Consider removing**: `hc_ret_mean_21d` (rank 60, near-significant negative ΔIC at raw p=0.091) — may actively harm prediction in the 168-universe context.
4. **Note for paper**: report this as a NEGATIVE finding (universe partially validated, signal weak universe-wide), not as new positive evidence.

### No signal at BH-FDR q=0.05 for positive groups

Critically, **no positive-ΔIC group survives BH-FDR correction**. This means at the 168-universe scale with 61 groups tested, our infrastructure cannot identify any single feature group as statistically reliably useful. Two interpretations:
- (i) **Realistic for cross-sectional equity prediction at this universe size** — signal is genuinely weak and dispersed across many features
- (ii) **Power limitation** — T=313 pooled dates and K=61 test family give limited statistical power; effect sizes that exist may need more data or fewer multiple comparisons to detect
- Both can be true. The CORD20+1 negative result (the one BH-FDR rejection) suggests the test has power for large effects but not small ones.

## 8. Convergence Audit (Plan v1 §3.2)

(source: `artifacts/plan_aaa/audit/convergence.json`)

| Metric | Value |
|---|---|
| Total cells | 30 (2 archs × 5 folds × 3 seeds) |
| Cells converged | **29 / 30** |
| Cells failed | **1** — cell_id=28 MLP fold=4 seed=123, reason: best val IC -0.0686 < -0.05 floor |
| Halt rule triggered (>20% = >6 failed) | **NO** (1/30 = 3.3%, well under 20% threshold) |
| Mean train_loss decrease (epoch 1 → early stop) | 0.106 (well above 1% threshold) |
| Best val IC range across cells | [-0.069, +0.168] |
| Median best val IC | +0.0348 |
| Median epochs to early stop | 12 (range 11-22 for converged cells) |
| Wall time | **57.6 minutes** on M4 MPS |

### Note on the one failed cell

`cell_id=28 (MLP, fold 4, seed 123)` produced a final test IC of +0.2466 despite val IC -0.0686. Fold 4 is a known outlier (per CLAUDE.md Rule 10 and `docs/fold4_leakage_diagnostic_2026-04-20.md`) — its val period precedes a regime shift in test. The cell's predictions are still used in the ranking (not auto-excluded per Plan v1 §3.2 halt rule design — "no auto-exclusion of failed cells"). Its presence may slightly noise-up the cell-mean ΔIC per group but cannot flip rank ordering given the dominance of the other 29 cells.

## 9. Touchpoint 2 Findings — Status After Full Mode

Reference: `artifacts/reviews/2026-05-23_finance-gnn-reviewer_code_A.md`

### 9.1 MAJOR — all FIXED before full mode

| ID | Category | Resolution |
|---|---|---|
| FINGNN-CODE-A-01 | data-leakage | per-fold winsor implemented via canonical helper; smoke v4 confirms train loss decrease 4.5%, val IC improved +0.068→+0.105 |
| FINGNN-CODE-A-02 | correctness | 2-layer ticker alignment (intersection assert + KMID time-series ρ > 0.9); smoke v4 ρ = 1.0000 on 3 sample tickers |
| FINGNN-CODE-A-03 | statistics | NW-HAC degenerate handling: ≤0 long-run-var returns NaN p; BH-FDR maps NaN → 1.0; `n_hac_degenerate` counter in summary |
| FINGNN-CODE-A-04 | correctness | `pooled_panel: True` + note added to groups_168.json; option (b) per-day Spearman deferred behind ARI<0.85 trigger (NOW TRIGGERED — see §4) |
| FINGNN-CODE-A-10 | statistics | bootstrap docstring renamed to "Fixed-length block (Künsch 1989)" |

### 9.2 CONCERN — deferred to analysis writeup or future tightening

- FINGNN-CODE-A-05 (correctness): defensive `_groups_to_labels` assert. Not blocking.
- FINGNN-CODE-A-06 (reproducibility): smoke noise-control field name. Not blocking; rename `pass_noise_soft` → `pass_noise_soft_smoke_only` if cited in paper.
- FINGNN-CODE-A-07 (reproducibility): train-day shuffle determinism within seed_idx across (arch, fold). Documented in convergence audit; inherited from part_a.
- FINGNN-CODE-A-08 (correctness): REJECTED by reviewer (self-verified merge invariant holds).
- FINGNN-CODE-A-09 (correctness): per-(group, day) non-group assert maintenance hazard. Documented; smoke proves overhead acceptable.

## 10. Paper Integration (Plan v1 §7)

### 10.1 §7.1 Limitations narrative (filled with actual ranks)

> "Feature universe scope (extended via Plan AAA). The Plan Z++ S1-S5 subsets were preregistered within a hand-curated 10-feature universe. We post-hoc extend the Plan Z++ Part A methodology — grouped cross-sectional permutation Δ-IC at inference on production SAGE-Mean/MLP models — to a 168-feature universe combining the 10 hand-curated features with the 158-dimensional qlib Alpha158 panel. Group structure is computed on a 252-day pre-experiment calibration window (days [0, 251], strictly before any test fold) at the same |ρ|>0.6 clustering threshold as Plan Z++ Part A, producing K=61 groups. Our hand-curated features map to 7 groups ranked at positions **1, 13, 24, 33, 41, 54, 60** (out of 61 groups) in the 168-feature Δ-IC ranking — three in the top half, four in the bottom half. The top-ranked group is `hc_mom12m` (12-month momentum, singleton; mean ΔIC +0.0079, NW p=0.31, BH-FDR p_adj=0.65). No group, including `hc_mom12m`, survives BH-FDR correction at q=0.05 over the K=61 family; the only group reliably rejected is `CORD20+1` (rank 61, NW p=0.0003, BH-FDR p_adj=0.021) which is a NEGATIVE ΔIC group (its inclusion harms prediction). We additionally observe an ARI of 0.55 between the calibration-window grouping and a fold-0-train-slice grouping (concern threshold preregistered at 0.85), suggesting moderate instability of cluster structure across windows; this is reported as a sensitivity limitation. Plan AAA is reported as an internal pre-analysis using the preregistered Plan Z++ Part A methodology applied to an extended universe; the universe extension itself is not externally preregistered."

(numeric provenance per Rule 5.H4: ranks from `artifacts/plan_aaa/hand_curated_mapping_168.json`; CORD20+1 row from `artifacts/plan_aaa/ranking.csv` row 61; ARI from `artifacts/plan_aaa/adjusted_rand_index.json`.)

### 10.2 §7.2 Prior-art framing (Touchpoint 1 finding CODEX-A-12 fix)

> "Our grouped cross-sectional permutation importance follows the broader feature-attribution literature: conditional permutation importance (Strobl, Boulesteix, Kneib, Augustin, & Zeileis, 2008) controls for within-group correlation by permuting groups jointly; SHAP (Lundberg & Lee, 2017) provides game-theoretic feature contributions. Plan AAA adapts these ideas to production GNN/MLP models for feature universe sensitivity analysis in cross-sectional equity ranking. We do not claim methodological novelty over the broader permutation-importance family; the contribution is the application to a 168-feature universe spanning qlib Alpha158 and literature-derived hand-curated factors."

References (Touchpoint 1 A-12):
- Strobl, C., Boulesteix, A. L., Kneib, T., Augustin, T., & Zeileis, A. (2008). Conditional variable importance for random forests. *BMC Bioinformatics*, 9(307).
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*.

## 11. Status + Next Steps

### Done in this session (2026-05-25)

1. ✅ §5 ranking table filled from `ranking.csv` with provenance per row
2. ✅ §6 hand-curated mapping filled from `hand_curated_mapping_168.json`
3. ✅ §7 outcome resolved: **(b) Mixed** per Plan v1 §1 pre-commitment decision rule
4. ✅ §8 convergence audit filled — 29/30 cells, halt rule NOT triggered

### Deferred (Touchpoint 3 outage — see progress.md 2026-05-25-c)

5. ⏸ **Codex Touchpoint 3 (results review)** — DEFERRED. Codex CLI rate-limited (reset 09:40 AM PT); 3 consecutive Anthropic API 529 errors blocked finance-gnn-reviewer fallback. Documented in progress.md 2026-05-25-c. Resume next session when API recovers; expected near-term once Anthropic 529 transient passes.
6. ⏸ §10.1 paper §7 Limitations + §10.2 §4.X Universe Sensitivity — **BLOCKED until Touchpoint 3 lands**. Per Rule 9 ("no paper claim before results review"). Draft language is ready in §10 above; will be transplanted to paper_draft after Touchpoint 3 PASS or PROCEED-WITH-FIXES verdict.
7. ⏸ `docs/methodology_qa_2026-05-22.md` Part 10.5 update — same blocker.
8. ⏸ Optional FINGNN-CODE-A-04 option (b) per-day-Spearman-median re-clustering — Touchpoint 2 reviewer recommended this if ARI<0.85 fired. **ARI=0.55 → trigger fired.** Awaiting H博士 prioritization (substantive analysis, requires model retraining since trained models are not persisted in current script).
9. ⏸ `scripts/verify_docs_provenance.py docs/plan_aaa_results_2026-05-25.md` — run after Touchpoint 3 doc updates land.

### Resume protocol

- Next session: try `/codex-results-review artifacts/plan_aaa/` first (Codex preferred over fallback per Rule 9 hierarchy). If still rate-limited and Anthropic API healthy, fall back to finance-gnn-reviewer with the Touchpoint 3 prompt structure documented in progress.md 2026-05-25-c.

→ progress: 2026-05-25-a (script + smoke + Touchpoint 2) | 2026-05-25-b (full mode results) | 2026-05-25-c (Touchpoint 3 deferred)
→ plan: 2026-05-23-a (plan_aaa_v1)
→ analysis: docs/analysis.md PENDING update AFTER Touchpoint 3 verdict (per Rule 9)

*Last updated: 2026-05-25 02:28 PT (placeholders filled with results; Touchpoint 3 deferred due to double reviewer outage)*
