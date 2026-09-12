# Phase 5 Diagnostic 2 — Fold-level Anomaly Root Cause

**Date**: 2026-04-16
**Inputs**: `experiments/wf5_results.csv` + `experiments/arch_comparison_results.csv` (90+150=240 runs)
**Companion CSVs**: `diag_phase5_fold_regime.csv`, `diag_phase5_label_dist.csv`, `diag_phase5_feature_dist.csv`, `diag_phase5_train_test_shift.csv`, `diag_phase5_ic_by_fold.csv`

---

## TL;DR

Fold 4 (Q2-2025) is **NOT systematically bad** — its mean IC is positive (0.024) and mean Sharpe is the highest among all folds (+2.22). The anomaly is **variance explosion, not mean collapse**: IC std_across_runs = 0.088 vs 0.02-0.03 for other folds (3-4× higher). Different seeds and architectures produce wildly different outcomes, from IC=-0.145 to IC=+0.223.

**Root cause**: Fold 4 test period contains the **April 2025 tariff shock**. Market regime metrics:
- Realized daily vol **1.81%** (2-3× other folds' 0.65-0.87%)
- Max drawdown **-12.7%** (vs -5% to -8% for other folds)
- **Lockstep regime (corrected calc, all 501 stocks, signed Pearson over 62 test days)**: signed mean pairwise correlation **0.496** (Folds 0-3: 0.18-0.22); **96.9% of stock-pairs have positive correlation** (Folds 0-3: 83-86%); **54.3% of pairs have correlation > 0.5** (Folds 0-3: 3.6-8.3%). This is a sharp qualitative regime shift, not a small quantitative one.
- ret_std_21d feature: train mean=0.018 → test mean=0.024 (KS=0.22, p≈0) — largest shift across all folds and features

**Note on correlation calculation correction**: First version of this report used first-100 alphabetical stocks + `np.abs(corr)`. Subsample bias was small (Fold 4 changed 0.500 → 0.496) but `|corr|` vs signed correlation is conceptually important: "lockstep selloff" means positive co-movement (signed), not just magnitude. The corrected signed mean + positive-pair fraction tell a cleaner story. See `experiments/diag_phase5_pairwise_corr_v2.csv`.

This is a **regime-shift stress test**: models trained on 2021-07 to 2024-12 encounter a high-vol, high-correlation environment at test time. The fact that IC mean stays positive but variance explodes is consistent with "some models generalize, others don't" rather than "all models fail."

**Secondary anomaly**: Fold 1 (Q3-2024) has **systematically negative IC** (mean -0.015). This is a different failure mode, deserves a brief note in the paper.

**Fold 3 (Q1-2025)** has near-zero IC (-0.002) but catastrophic Sharpe (-3.22). Portfolio-construction problem, not ranking problem. Flagged for later investigation.

---

## 1. Per-fold IC + Sharpe summary (n=48 runs/fold from combined wf5 + arch_comparison)

| Fold | Period  | mean IC | median IC | std IC | min IC | max IC | mean Sharpe_net | std Sharpe_net |
|------|---------|---------|-----------|--------|--------|--------|-----------------|----------------|
| 0    | Q2-2024 | +0.013  | +0.007    | 0.021  | -0.033 | +0.036 | -0.84           | 4.18           |
| 1    | Q3-2024 | **-0.015** | -0.021 | 0.022  | -0.046 | +0.078 | -0.78           | 3.95           |
| 2    | Q4-2024 | **+0.048** | +0.048 | 0.029  | -0.043 | +0.106 | **+1.39**       | 1.50           |
| 3    | Q1-2025 | -0.002  | +0.004    | 0.023  | -0.100 | +0.034 | **-3.22**       | 5.74           |
| 4    | Q2-2025 | +0.024  | +0.006    | **0.088** | **-0.145** | **+0.223** | **+2.22** | **5.10** |

**Observations**:
- Only Fold 2 is unambiguously "easy" (high IC, high Sharpe, low variance).
- Fold 1 is the systematically-negative-IC fold — models are wrong, not just noisy.
- Fold 3 is the IC-to-Sharpe disconnect fold — ranking ok, portfolio disaster.
- Fold 4 is the variance-explosion fold — high mean but unstable.

---

## 2. Market regime per fold (test period)

| Fold | Period  | eq_ret_cum | daily vol | Sharpe_ann | max DD | signed mean corr | frac pairs corr>0 | frac pairs corr>0.5 |
|------|---------|------------|-----------|------------|--------|------------------|-------------------|--------------------|
| 0    | Q2-2024 | -1.4%      | 0.65%     | -0.50      | -5.3%  | 0.181            | 86%               | 3.6%               |
| 1    | Q3-2024 | +10.2%     | 0.85%     | +2.89      | -5.4%  | 0.216            | 85%               | 8.3%               |
| 2    | Q4-2024 | -0.5%      | 0.75%     | -0.12      | -7.0%  | 0.188            | 85%               | 5.7%               |
| 3    | Q1-2025 | -0.4%      | 0.87%     | -0.05      | -7.8%  | 0.204            | 83%               | 7.4%               |
| 4    | Q2-2025 | +6.9%      | **1.81%** | +1.08      | **-12.7%** | **0.496**    | **97%**           | **54.3%**          |

**Fold 4 is a clear outlier on all volatility/correlation metrics.** The +6.9% cumulative return hides a severe drawdown (-12.7%) followed by sharp recovery — classic V-shape crash-rebound. When 54% of stock-pairs have correlation > 0.5 (vs <9% in other folds), cross-sectional ranking becomes fundamentally noisier: when everything moves together, relative rankings compress.

---

## 3. Train→Test feature distribution shift (top-3 per fold by KS)

| Fold | Feature       | KS     | train_mean | test_mean | train_std | test_std |
|------|---------------|--------|------------|-----------|-----------|----------|
| 0    | ret_std_21d   | 0.215  | 0.0188     | 0.0157    | 0.0098    | 0.0085   |
| 0    | ret_std_10d   | 0.191  | 0.0183     | 0.0152    | 0.0108    | 0.0094   |
| 1    | ret_mean_21d  | 0.137  | 0.0004     | 0.0011    | 0.0044    | 0.0037   |
| 1    | momentum_21d  | 0.136  | 0.0082     | 0.0233    | 0.0945    | 0.0787   |
| 2    | ret_std_5d    | 0.098  | 0.0169     | 0.0154    | 0.0119    | 0.0121   |
| 3    | momentum_21d  | 0.096  | 0.0094     | -0.0104   | 0.0928    | 0.0913   |
| 4    | **ret_std_21d** | **0.222** | 0.0181 | **0.0241** | 0.0096 | **0.0130** |
| 4    | ret_std_10d   | 0.138  | 0.0176     | 0.0225    | 0.0106    | 0.0153   |

**Fold 4 is the only fold where test volatility is dramatically HIGHER than train** (+33% for ret_std_21d). Folds 0 and 2 have test volatility *lower* than train (easier regime for volatility-dominant models). Fold 1 has a momentum regime shift (test mean 3× train mean) — possibly why IC goes negative: models learned to short 2022-2023 underperformers, but Q3-2024 reversed.

---

## 4. Label distribution (21d forward excess return, cross-sectional z-score)

By construction z-scores are mean=0, std=1 per day. Interesting variation in higher moments:

| Fold | skew | kurt | excess_std (raw, not z) |
|------|------|------|-------------------------|
| 0 Q2-2024 | +0.66 | 4.31  | 0.075 |
| 1 Q3-2024 | +0.17 | 4.81  | 0.078 |
| 2 Q4-2024 | +1.34 | **10.32** | 0.088 |
| 3 Q1-2025 | -0.02 | 2.64  | 0.088 |
| 4 Q2-2025 | +0.50 | 3.62  | **0.089** |

Fold 2's extreme kurtosis (10.3) is consistent with post-election concentrated winners driving the rally. Fold 4 has the widest absolute excess return distribution (highest dispersion), which creates both opportunity (high max IC 0.22) and risk (low min IC -0.15).

---

## 5. Recommendations for paper / next experiments

### 5a. Report Fold 4 honestly in paper
- Fold 4 is NOT a bug to fix. It is a **regime stress test**.
- Show per-fold IC + Sharpe table with confidence bars. Do NOT hide the variance.
- Narrative: "Our method is robust in calm regimes (Folds 0-3) but shows high seed/architecture sensitivity under extreme stress (Fold 4, April 2025 tariff shock). This is a limitation and an interesting avenue for regime-aware models."

### 5b. Investigate why Fold 1 has negative mean IC
Not part of current diagnostic scope, but the momentum regime shift (Q3-2024) is a potential explanation. Worth a single paragraph in paper.

### 5c. Investigate Fold 3 IC-Sharpe disconnect
`mean_Sharpe_net = -3.22` with `mean_IC ≈ 0` implies top-30 portfolio consistently underperforms. Likely sector concentration similar to the SAGE-Sum issue previously diagnosed. Flag for Week 3 follow-up.

### 5d. Implications for Phase 5 feature expansion
- Adding features that help in **high-vol / high-correlation regimes** has higher leverage than adding more momentum variants.
- Volume-based features (dolvol, CORR5) may capture liquidity dislocations during stress — good.
- RSV5 (OHLC-based range) directly encodes intraday stress — potentially highest value in Fold 4.
- **Do not expect feature expansion to stabilize Fold 4 variance** without explicit regime conditioning (VIX overlay, Priority 3 of Phase 5 plan).

---

*Written 2026-04-16 by Claude based on local diagnostic run.*
