# Phase 5 Diagnostic 3 — 9-dim Feature Importance + Collinearity

**Date**: 2026-04-16
**Companion CSVs**: `diag_phase5_collinearity.csv`, `diag_phase5_feature_importance.csv`, `diag_phase5_single_feature_lgb.csv`, `diag_phase5_permutation_importance_lgb.csv`

---

## TL;DR

1. **3 pairs of features are perfectly rank-redundant** (cross-sectional Pearson corr = 1.00): `ret_mean_Nd` and `momentum_Nd` for N ∈ {5, 10, 21}. Note: they differ by a scale factor (momentum ≈ N × ret_mean for small returns) but this does not affect rank-based or normalized models.
2. **Effective rank of the 9-dim feature set is ~3 (corrected via eigendecomposition)**: top 3 principal components explain **89.7%** of variance, top 4 explain **95.0%**. Participation ratio = 2.91; Shannon effective rank = 3.66. The three PCs have clean financial interpretation: PC1 = momentum/trend (49.5%), PC2 = volatility (28.4%), PC3 = short-vs-long momentum spread (11.8%).
3. **Volatility dominates signal.** `ret_std_10d` alone delivers IC = 0.028 on Fold 0; shuffling it drops full-LGB IC by -0.024 (2-3× the effect of any other feature). **Caveat: single-fold, 63-day test sample; IC SE ≈ 0.013, so 0.028 vs 0.021 is within noise — don't over-read.**
4. **Momentum features contribute near-zero signal for 21d horizon on Fold 0.** `momentum_21d` and `ret_mean_10d` shuffling *improves* IC by +0.006 — they are noise or mild anti-predictors in this fold.
5. **Implication for Phase 5**: adding more momentum variants (mom12m) is low-ROI — they'll load on the existing dominant PC1. **Volume-based features (dolvol) and range features (RSV5) are orthogonal to the existing 3 PCs — those are the high-leverage additions.**

---

## 1. Collinearity matrix (cross-sectional Pearson, averaged over 1212 valid days)

|              | ret_m5 | std_5 | mom_5 | ret_m10 | std_10 | mom_10 | ret_m21 | std_21 | mom_21 |
|--------------|--------|-------|-------|---------|--------|--------|---------|--------|--------|
| ret_mean_5d  | 1.00   | 0.02  | **1.00** | 0.69    | 0.02   | 0.69   | 0.47    | 0.02   | 0.47   |
| ret_std_5d   | 0.02   | 1.00  | 0.00  | 0.01    | 0.81   | -0.01  | -0.01   | 0.68   | -0.02  |
| momentum_5d  | **1.00** | 0.00 | 1.00 | 0.69    | 0.01   | 0.69   | 0.47    | 0.02   | 0.47   |
| ret_mean_10d | 0.69   | 0.01  | 0.69  | 1.00    | 0.02   | **1.00** | 0.67    | 0.03   | 0.67   |
| ret_std_10d  | 0.02   | 0.81  | 0.01  | 0.02    | 1.00   | 0.00   | 0.00    | 0.84   | -0.02  |
| momentum_10d | 0.69   | -0.01 | 0.69  | **1.00** | 0.00  | 1.00   | 0.67    | 0.02   | 0.67   |
| ret_mean_21d | 0.47   | -0.01 | 0.47  | 0.67    | 0.00   | 0.67   | 1.00    | 0.02   | **1.00** |
| ret_std_21d  | 0.02   | 0.68  | 0.02  | 0.03    | 0.84   | 0.02   | 0.02    | 1.00   | 0.00   |
| momentum_21d | 0.47   | -0.02 | 0.47  | 0.67    | -0.02  | 0.67   | **1.00** | 0.00  | 1.00   |

### Key structure
- **Exact redundancy (corr=1.00)**: `ret_mean_Nd ≡ momentum_Nd`. Formulas: `ret_mean_Nd = returns.rolling(N).mean().shift(1)` vs `momentum_Nd = prices.shift(1)/prices.shift(N+1) - 1`. Arithmetic mean of simple returns ≈ compound return for small returns — numerically identical at ~3-4 decimals.
- **Volatility cluster**: `ret_std_5d / ret_std_10d / ret_std_21d` pairwise correlations 0.68-0.84. Strongly correlated but not identical; the fast (5d) vs slow (21d) vol spread carries marginal info.
- **Momentum/return spans**: `ret_mean_5d / ret_mean_10d / ret_mean_21d` pairwise 0.47-0.69. More independence than within vol, but still substantial overlap.

### Effective rank (actually computed, corrected)

Eigendecomposition of the 9×9 averaged cross-sectional correlation matrix (1212 valid days):

| k | λ_k    | % variance | cumulative |
|---|--------|-----------|-----------|
| 1 | 4.453  | 49.5%     | 49.5%     |
| 2 | 2.557  | 28.4%     | 77.9%     |
| 3 | 1.059  | 11.8%     | **89.7%** |
| 4 | 0.482  | 5.4%      | **95.0%** |
| 5 | 0.317  | 3.5%      | 98.5%     |
| 6 | 0.126  | 1.4%      | 99.9%     |
| 7-9 | <0.005 | <0.1% each | 100.0%   |

**Effective-rank measures (nominal = 9)**:
- Participation ratio `(Σλ)² / Σλ²` = **2.91**
- Shannon effective rank `exp(−Σ pₖ log pₖ)` = **3.66**
- k for 80% / 90% / 95% variance = **3 / 4 / 4**

**Interpretation via top-3 PC loadings** (clean financial meaning):
- **PC1** (49.5%, λ=4.45): All 6 mean/momentum features load with ~−0.40 each; ret_std's near zero. This is the **momentum/trend factor**.
- **PC2** (28.4%, λ=2.56): All 3 ret_std features load +0.56 to +0.60; mean/momentum near zero. This is the **volatility factor**.
- **PC3** (11.8%, λ=1.06): Contrasts short (5/10d momentum, +0.49) vs long (21d momentum, −0.51). This is the **momentum horizon-spread factor**.

Three factors — trend, vol, horizon-spread — cover 90% of the information in our 9-dim feature set. See `experiments/diag_phase5_effective_rank.csv`.

*Correction note*: earlier version of this report speculated "4-5 effective dim" without running eigendecomposition. The actual number is 3 (for 90% var) or 4 (for 95% var), cleanly interpretable as 3 financial factors.

---

## 2. Single-feature LightGBM IC (Fold 0 test)

LGB trained with 1 feature only, tested on Fold 0 test period (63 days):

| Feature       | Fold 0 test IC |
|---------------|----------------|
| **ret_std_10d** | **+0.028**   |
| ret_std_5d    | +0.016         |

**⚠️ Statistical caveat**: SE(mean IC) at N=63 days is approximately `σ_daily / √N ≈ 0.10 / 8 = 0.013`. So the gap between ret_std_10d (+0.028) and full-LGB (+0.021) is ~0.5 SE — not statistically distinguishable. The direction is suggestive (vol dominates) but the "single feature beats full model" framing is over-stated without a multi-fold replication.
| momentum_10d  | +0.001         |
| ret_mean_5d   | +0.001         |
| ret_std_21d   | -0.001         |
| ret_mean_10d  | -0.003         |
| momentum_5d   | -0.004         |
| momentum_21d  | -0.005         |
| ret_mean_21d  | -0.006         |

### Full 9-feature LGB baseline IC: **+0.021**

The single best feature (ret_std_10d, IC=0.028) **beats the full 9-feature LGB (IC=0.021)**. This is a strong signal that the additional features are net-noise for this horizon / this fold / LGB's inductive bias.

---

## 3. Permutation importance (shuffle one feature in test, measure IC drop)

Full LGB trained on all 9 features; then for each feature, shuffle its test-period values and re-predict. Delta_IC = shuffled IC - baseline IC (more negative = more important).

| Feature       | shuffled IC | delta IC |
|---------------|-------------|----------|
| **ret_std_10d** | -0.004    | **-0.024** |
| ret_mean_21d  | +0.011      | -0.010   |
| ret_std_5d    | +0.011      | -0.009   |
| ret_mean_5d   | +0.014      | -0.007   |
| momentum_10d  | +0.019      | -0.002   |
| ret_std_21d   | +0.019      | -0.002   |
| momentum_5d   | +0.019      | -0.002   |
| momentum_21d  | +0.027      | **+0.006** |
| ret_mean_10d  | +0.027      | **+0.006** |

### Observations
1. **ret_std_10d carries an order of magnitude more signal than anything else** (delta -0.024 vs next -0.010).
2. Two features — `momentum_21d` and `ret_mean_10d` — show **positive delta_IC when shuffled**. Shuffling them makes predictions *better*. In the Fold 0 regime, LGB over-weighted these features; noise in them hurts predictions.
3. The `ret_mean_21d` vs `momentum_21d` inconsistency (one helps, one hurts, despite being corr=1.00 identical) reveals that LGB's split choices between near-identical features are effectively random — one is "used" and the other is "parked." Classic redundant-feature artifact.

---

## 4. Implications

### 4a. Feature pruning candidate
Dropping one of each mean/momentum pair costs zero information. A 6-dim feature set (`ret_mean_{5,10,21}d` + `ret_std_{5,10,21}d`) is equivalent to current 9-dim. Consider for cleanup.

### 4b. Phase 5 new features — MEASURED PC loadings (2026-04-16 update)

**Method**: Computed 14×14 cross-sectional correlation matrix averaged over 982 valid days (post feature-build). Reported each new feature's maximum correlation with any of the 9 original features, and its orthogonal residual. Source: `data/reference/phase5_feature_14x14_collinearity.csv` + `build_phase5_features.py`.

| New feature | Max \|corr\| with old 9 | Orthogonal component | Revised ROI assessment |
|-------------|-------------------------|----------------------|------------------------|
| **mom12m** | <0.05 (all 9) | **~0.99** | **High** — fully orthogonal to all 3 existing PCs; earlier "loads PC1" hypothesis was wrong |
| **dolvol** | 0.13 (ret_std_21d) | **~0.98** | **High** — near-fully orthogonal |
| **CORR5** | 0.27 (ret_mean_5d, momentum_5d) | ~0.83 | **Medium-high** — partial PC1 overlap but mostly new |
| **maxret** | 0.80 (ret_std_21d) | low | **Low-medium** — mostly in PC2 (vol) span, as hypothesized |
| **RSV5** | 0.66 (ret_mean_5d, momentum_5d) | low | **Low-medium** — substantially in PC1 (short momentum) span; earlier "medium-high" assessment was overstated |

### 4c. Effective rank of combined 14-dim feature set

Eigendecomposition of 14×14 correlation matrix (averaged over 982 days):

| k | cum var |
|---|---------|
| 3 | 67.5% |
| 4 | 75.5% |
| 7 | **92.3%** |
| 8 | **95.4%** |

- 9-dim prior: k90=4, k95=4 (effective rank ≈ 3)
- 14-dim new: **k90=7, k95=8** (effective rank ≈ 7)
- **Adding the 5 features roughly doubles the effective rank.** Most of the gain comes from mom12m (horizon extension) and dolvol (volume dimension).

### 4d. Revised Phase 5 Step 2/3 guidance

### 4e. Observations relevant to Phase 5 Step 2/3 scope (not recommendations)

1. **The 9-dim feature set is effectively 3-dim** (PC1 momentum, PC2 volatility, PC3 horizon-spread). The 14-dim expanded set has effective rank ≈ 7 (after measured PC loadings in 4b-4c).
2. **Cross-sectional normalization is regime-dependent (see Diag 1)**. Do not treat as mandatory.
3. **Clean 6-dim baseline** (drop one of each mean/momentum pair since corr=1.00) is a reasonable additional ablation row. Expected null result (LGB is rank-invariant to such scale-only redundancy, NN may care slightly), but a clean paper story.
4. **Full Alpha158 is out of scope** for now — feature selection is H博士's call, not derivable from this single-fold diagnostic.
5. If run-count budget is tight, **mom12m and dolvol are the highest-orthogonality additions**; maxret and RSV5 are most likely to replicate existing PC information.

### 4d. Caveat

This analysis is **Fold 0 only** for permutation & single-feature LGB. The cross-sectional IC importance (`diag_phase5_feature_importance.csv`) covers all 5 folds and shows similar pattern but with fold-to-fold variation. A future pass could compute permutation importance per fold if the full-5-fold view changes the story.

---

*Written 2026-04-16 by Claude based on local diagnostic run.*
