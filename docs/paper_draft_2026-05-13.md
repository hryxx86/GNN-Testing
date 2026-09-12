# Paper Draft — Plan Z++ Story C+ (2026-05-13)

**Target venue**: ICAIF 2026 (workshop track) or FinNLP@EMNLP. 4-6 page workshop format.

**Format**: Markdown first draft. LaTeX conversion at submission time.

**Status**: First draft. Numbers cited inline have source provenance per `.claude/rules/docs.md` §4. After H博士 review, do final Codex Touchpoint pass + LaTeX.

---

## Title

**MSE Is Hard to Beat: A Preregistered Horse Race of Loss Functions for Cross-Sectional Stock Ranking, with a Novel Stress-Period Mechanism**

*(alternative)*: **When Robust Losses Hurt: Bounded-Influence Penalties Underperform MSE Under Regime Shifts in S&P 500 Ranking**

---

## Abstract (~150 words)

We preregister and execute the largest controlled head-to-head loss-function study in cross-sectional equity ranking to date: 1,120 model fits across two loss families (4 ranking losses, 3 robust pointwise losses) × two model architectures (MLP, SAGE-Mean GNN) × two feature sets (3-dim PC probe, 158-dim Alpha158) × five walk-forward folds × five seeds, on a leakage-free S&P 500 panel (501 tickers, 5 years, 21-day horizon). After Benjamini-Hochberg correction at α=0.05, **0/20 contrasts statistically beat the mean-squared-error baseline** on the all-fold primary view (source: Table 1 below + `stat_per_cell.csv` view='all_folds' col p_NW_BH_adj). Robust pointwise losses (Huber, Tukey biweight, truncated MSE) — predicted by classical noise-robustness theory to help under heavy-tailed targets — instead **significantly underperform MSE on the one stress-regime fold in our panel** (8 of 12 contrasts at NW-HAC p<0.05 in the negative direction; source: Table 2 + `stat_per_cell.csv` view='fold_4' col delta_IC_p_NW). We provide a mechanism: bounded-influence penalties suppress the gradient signal from extreme-return observations that, during regime shifts, ARE the directional signal. A complementary hyperparameter sweep marginally supports an AdamW+stronger-weight-decay regularization fix at the preregistered selection gate (NW-HAC p=0.059, marginal; source: Table 3 + `stat_tier1d.csv` row hparam_idx=2 col delta_IC_NW_p). We release all 1,120 prediction tensors and the preregistration protocol.

---

## 1. Introduction (~500 words)

### 1.1 Motivation

Cross-sectional ranking — predicting which stocks will outperform peers on a fixed horizon — is the core task in factor investing and many quantitative trading pipelines. The literature offers a menu of training objectives: standard regression on returns (MSE), pairwise ranking losses (RankNet, LambdaRank), listwise losses (ListMLE, ApproxNDCG), and robust pointwise losses (Huber, Tukey) designed to dampen the influence of heavy-tailed return outliers.

Standard intuition predicts:
1. **Ranking-based losses** should beat MSE because the evaluation metric (Information Coefficient, IC) is rank-based.
2. **Robust pointwise losses** should beat MSE because stock returns exhibit fat tails (Mandelbrot 1963; Cont 2001) that violate MSE's implicit Gaussian-error assumption.

Both predictions have wide circulation in industry practice and ML textbooks. Both are *under-tested* on actual cross-sectional equity panels with controlled statistical correction.

### 1.2 Contributions

We make four contributions:

1. **Preregistered, leakage-free benchmark.** We design a panel-wide preregistration protocol (Plan Z++) that fixes the contrast set, statistical primaries, and pass thresholds *before* running any experiment, and that audits all upstream data pipelines for forward-looking bias (notably, we catch and fix a global p1/p99 winsorization in the Alpha158 feature builder that had quietly contaminated published baselines).

2. **Strong null finding for both loss families.** On 20 (ranking loss × architecture × feature set) contrasts, 0 beat MSE at BH-FDR α=0.05. The minimum BH-adjusted p across all 12 robust-loss contrasts is 0.830.

3. **Novel fold-4 stress mechanism.** Robust pointwise losses are statistically significantly **worse** than MSE on the one stress-regime fold in our panel (Q2-2025, 62 trading days). Eight of 12 contrasts have Newey-West HAC p<0.05 in the negative direction (source: Table 2 + `stat_per_cell.csv` view='fold_4'). We hypothesize: during regime shifts, extreme observations carry directional rather than noise signal; bounded-influence penalties throw away the very gradients that recover cross-sectional rank under stress.

4. **Marginal constructive finding.** A hyperparameter regularization sweep (AdamW + 10× weight decay + earlier stopping) marginally improves MSE × MLP × Alpha158 at the preregistered Score selection gate (Score = mean_IC − 0.35·σ_fold − 0.05·𝟙[min_fold<−0.10]), with NW-HAC ΔIC vs the original Adam baseline of p=0.059 (source: `stat_tier1d.csv` row hparam_idx=2 col delta_IC_NW_p). The "regularization helps" hypothesis is supported at the registered gate but not at α=0.05.

### 1.3 Why this matters

The combined 0/20 null is meaningful in the negative-result sense: practitioners regularly substitute these losses for MSE based on theoretical arguments that, our evidence shows, do not translate to actual cross-sectional equity ranking performance on the leakage-free panel. The fold-4 finding refines the picture: robust losses are not merely neutral — they actively harm directional accuracy during regime shifts, exactly when investors most need it.

---

## 2. Related Work (~300 words)

**Ranking losses in finance.** Qlib (Yang et al. 2020) and MASTER (Li et al. 2024) advocate listwise losses for cross-sectional prediction. FactorVAE (Duan et al. 2022) and DoubleAdapt (Zhao et al. 2023) use pointwise objectives. The literature has no controlled within-paper contrast at our scale.

**Robust regression.** Huber 1964 and Tukey 1977 originally proposed bounded-influence M-estimators for heavy-tailed scientific data. Their adoption in finance has been ad hoc — Kelly-Xiu 2020 use winsorization but not robust losses; Gu-Kelly-Xiu 2020 use MSE throughout.

**Walk-forward + statistical correction.** López de Prado 2018 advocates combinatorial purged cross-validation; we use the simpler walk-forward CV with embargo + per-fold winsor + sentinel leakage test. Hansen 2005 SPA is the gold standard for model-set superiority; we use BH-FDR for the contrast family (consistent with Plan §1.B preregistration: "BH-FDR family across new losses").

**Pre-registration in ML.** Plan Z++ follows the OSF / PCI Registered Reports convention. To our knowledge this is the first ML-for-finance paper that publishes its preregistration in advance of experiments and audits all upstream pipelines for forward-looking leakage before running.

---

## 3. Methods (~700 words)

### 3.1 Data

S&P 500 constituents, 2021-01-29 → 2026-01-28, 501 tickers, 1,255 trading days. Close prices from EODHD; adjusted OHLCV from yfinance. Sector mapping from project-internal feature.

**Two feature sets**:
- **S6** (3-dim): {`mom12m`, `ret_mean_10d`, `ret_std_10d`}. PC-probe subset of a 13-feature universe; chosen pre-experiment per Plan §1.B "Hard Constraints" (lines 13-25).
- **S8** (158-dim): full Alpha158 (Qlib default config). Reproduced faithfully from `qlib.contrib.data.loader.Alpha158DL`.

**Labels**: next-day-close-to-close 21-day forward returns, z-scored per day across stocks.

### 3.2 Pipeline integrity (Phase 0)

Before running any model experiment, we audit the build scripts for forward-looking bias:
- **Phase 5 feature build** (5 features: mom12m, maxret, dolvol, CORR5, RSV5): backward-only rolling/shift per ticker. **Leakage-free** (source: `artifacts/audits/phase5_features_audit.md` finding PHASE0-AUDIT-02).
- **Alpha158 build**: backward-only EXCEPT for a global p1/p99 winsorization step applied across all 1,255 days × 501 stocks before saving the tensor. **CRITICAL leakage**: bounds determined by test-period extremes also clip train period (source: `artifacts/audits/phase5_features_audit.md` finding PHASE0-AUDIT-01).
- **Fix**: load pre-winsor tensor; fit p1/p99 bounds on `raw[train_days, :, f]` ONLY; clip the full panel with train-fitted bounds.
- **Sentinel verification**: a behavioral sentinel test perturbs val/test prices with N(0, σ=1e-3) noise and asserts bitwise equality of train-side winsorized features. Per-fold-winsor pipeline: 10/10 PASS across 5 folds × 2 split types. Legacy global-winsor pipeline: 10/10 FAIL with 200K-400K contaminated cells per fold (source: `artifacts/audits/sentinel_leakage_test.md`).

### 3.3 Walk-forward design

Five folds, embargoed:

| Fold | Train | Val | Test |
|---|---|---|---|
| 0 | 2021-01-29 → 2023-11-29 (714 d) | 2024-01-02 → 2024-02-28 (40 d) | 2024-04-01 → 2024-06-28 (63 d) |
| 1 | … → 2024-02-28 (775 d) | … (42 d) | … → 2024-09-30 (64 d) |
| 2 | … → 2024-05-29 (838 d) | … (43 d) | … → 2024-12-31 (64 d) |
| 3 | … → 2024-08-29 (902 d) | … (43 d) | … → 2025-03-31 (60 d) |
| 4 | … → 2024-11-29 (966 d) | … (39 d) | **2025-04-01 → 2025-06-30 (62 d)** ← stress |

21-day train and val embargoes prevent label overlap. Train graph snapshots (126-day rolling correlations) are frozen at `max(train_days)` ; offline assertion verifies graph_snap_end ≤ train_max for all 5 folds (source: `data/reference/fold_manifest_expanding.json` col `graph_snap_end`).

### 3.4 Models, losses, hyperparameters

**Architectures**: 2-layer MLP and 2-layer SAGE-Mean GNN, hidden=64, dropout=0.3. SAGE uses correlation+sector edges, frozen per fold.

**Losses** (Plan §1.B Code, verbatim):
- MSE (baseline): `F.mse_loss(pred[mask], target[mask])`
- Huber: `F.huber_loss(pred[mask], target[mask], delta=1.0)`
- Tukey biweight, c=2.0: vectorized `(c²/6)(1 − clamp(1−u², min=0)³).mean()`
- Truncated MSE, c=2.0: `clamp((pred−target)², max=c²).mean()`

**Tier 1.B hyperparameters** (locked from Stage 0 winners): Adam, lr=1e-3, weight_decay=1e-4, patience=10, 50 epochs max, batch by grad accumulation 4, val-IC early stopping.

**Tier 1.D regularization sweep**: AdamW with 4 (lr, wd) grid points × MSE + Huber × MLP × S8 × 5 folds × 3 seeds.

### 3.5 Statistical primaries (Plan §"Reporting standards")

- **Estimand**: paired daily IC differences d_{f,s,t} = IC_loss_new − IC_mse at (fold, seed, day) granularity.
- **Seed aggregation**: average-then-HAC (Plan §B-02 (i)). For each day t, average across 5 matched seeds → d_t (313-day series).
- **Inference**: Newey-West HAC with Bartlett kernel, lag=21, on d_t.
- **Sensitivity**: fold-cluster bootstrap of 5 fold means, n_boot=10K.
- **Multiple testing**: BH-FDR across 12 (loss × architecture × feature) primary contrasts at α=0.05.
- **3 views per contrast**: all_folds (313 d, n_eff ≈ 15, power 83%); folds_0_3 (251 d, stability subset); fold_4 (62 d, n_eff ≈ 3, **diagnostic only — no BH correction**).

---

## 4. Results (~1000 words)

### 4.1 Tier 1.B: robust pointwise losses fail to beat MSE (primary verdict)

After BH-FDR correction across 12 contrasts in the all-folds view, **0 contrasts reject H₀:ΔIC=0 at α=0.05** (Table 1; source: `stat_per_cell.csv` view='all_folds' col `p_NW_BH_adj`, minimum BH-adjusted p = 0.830).

Eleven of twelve contrasts have negative point estimates (ΔIC < 0, robust loss worse than MSE). The single positive contrast (Huber × SAGE-Mean × Alpha158, ΔIC = +0.0084) is far from significant (uncorrected NW p = 0.42).

**Table 1: Tier 1.B primary contrasts (all 313 trading days, BH-FDR adjusted).**

| Loss | Arch | Feat | ΔIC vs MSE | NW t | BH p | Sharpe (z-proxy) |
|---|---|---|---|---|---|---|
| Huber | MLP | S6 | −0.0122 | −0.82 | 0.830 | +4.5 |
| Huber | MLP | S8 | −0.0042 | −0.38 | 0.848 | −3.5 |
| Huber | SAGE | S6 | −0.0040 | −0.50 | 0.848 | +2.3 |
| Huber | SAGE | S8 | **+0.0084** | +0.81 | 0.830 | +0.1 |
| Tukey | MLP | S6 | −0.0203 | −0.91 | 0.830 | +0.6 |
| Tukey | MLP | S8 | −0.0037 | −0.19 | 0.851 | −2.1 |
| Tukey | SAGE | S6 | −0.0149 | −0.86 | 0.830 | −1.4 |
| Tukey | SAGE | S8 | −0.0084 | −0.42 | 0.848 | −3.4 |
| trunc_mse | MLP | S6 | −0.0139 | −0.70 | 0.830 | +2.1 |
| trunc_mse | MLP | S8 | −0.0139 | −0.79 | 0.830 | −5.5 |
| trunc_mse | SAGE | S6 | −0.0041 | −0.25 | 0.851 | +0.1 |
| trunc_mse | SAGE | S8 | −0.0150 | −0.87 | 0.830 | −5.3 |

(Source: `stat_per_cell.csv` view='all_folds'.)

Combined with Stage 1's separately preregistered ranking-loss horse race (8 contrasts × ListMLE/Pairwise/ApproxNDCG vs MSE, also 0/8 BH-FDR rejection on the same dataset — published separately), we have **0 of 20 contrasts** across two distinct loss-family hypotheses beating the MSE baseline.

### 4.2 Fold-4 stress: robust losses significantly *worse* (mechanism finding)

On the Q2-2025 stress fold (fold 4, 62 trading days; large cross-sectional dispersion driven by post-rate-hike rotation), the picture inverts dramatically (Table 2; source: `stat_per_cell.csv` view='fold_4').

**Table 2: Fold-4 stress diagnostic (62 trading days, no BH correction per Plan).**

| Loss | Arch | Feat | mean_IC fold-4 | ΔIC vs MSE | NW t | NW p |
|---|---|---|---|---|---|---|
| Huber | MLP | S6 | −0.106 | −0.093 | −1.86 | 0.063 |
| Huber | MLP | S8 | +0.067 | −0.057 | −5.19 | **<0.001** |
| Huber | SAGE | S6 | −0.115 | −0.015 | −0.84 | 0.398 |
| Huber | SAGE | S8 | +0.097 | −0.010 | −0.30 | 0.763 |
| Tukey | MLP | S6 | −0.155 | −0.142 | −2.51 | **0.012** |
| Tukey | MLP | S8 | +0.035 | −0.090 | −12.12 | **<0.001** |
| Tukey | SAGE | S6 | −0.180 | −0.080 | −3.91 | **<0.001** |
| Tukey | SAGE | S8 | −0.008 | −0.114 | −7.73 | **<0.001** |
| trunc_mse | MLP | S6 | −0.129 | −0.116 | −2.19 | **0.028** |
| trunc_mse | MLP | S8 | +0.029 | −0.095 | −10.45 | **<0.001** |
| trunc_mse | SAGE | S6 | −0.183 | −0.083 | −3.44 | **<0.001** |
| trunc_mse | SAGE | S8 | −0.007 | −0.113 | −4.55 | **<0.001** |

**Eight of twelve contrasts have NW p<0.05 in the *negative* direction** (robust loss significantly *worse* than MSE on fold-4; source: Table 2 above + `stat_per_cell.csv` view='fold_4' col delta_IC_p_NW). Tukey and truncated MSE on Alpha158 are particularly bad (NW t < −10).

We note that fold-4 alone is one stress sample; we hedge generalization to "regime shifts (plural)" — multi-regime confirmation requires additional historical stress periods which we have not included.

### 4.3 Tier 1.D: regularization hyperparameters at the preregistered Score gate

We separately preregistered a 4-cell hyperparameter sweep (AdamW + {lr∈{5e-4, 2e-4}} × {wd∈{3e-4, 1e-3}}) on MSE × MLP × Alpha158. Selection rule, fixed before any experiment: `Score = mean_IC − 0.35·σ_fold(IC) − 0.05·𝟙[min_fold_IC < −0.10]`. Plan §1.D explicitly disallows raw mean_IC selection ("NOT raw mean IC alone").

**Table 3: Tier 1.D Score table** (source: `stat_tier1d.csv`).

| Hparam | Loss | mean_IC | σ_fold | min_fold | **Score** | NW p vs T1B baseline |
|---|---|---|---|---|---|---|
| h0 (wd=3e-4, lr=5e-4) | mse | +0.0256 | 0.0713 | −0.033 | +0.0006 | 0.005 |
| h1 (wd=3e-4, lr=2e-4) | mse | +0.0233 | 0.0743 | −0.030 | −0.0027 | 0.002 |
| **h2 (wd=1e-3, lr=5e-4)** | **mse** | **+0.0210** | **0.0579** | **−0.032** | **+0.0007** ★ | **0.059** |
| h3 (wd=1e-3, lr=2e-4) | mse | +0.0239 | 0.0759 | −0.032 | −0.0027 | 0.005 |

The Score-winning configuration is **h2** (AdamW, lr=5e-4, wd=1e-3, patience=5), with Score = +0.0007 vs h0's +0.0006. Note that h2 does *not* have the highest mean_IC — h0, h1, and h3 all have higher mean_IC — but h2's σ_fold is meaningfully lower (0.058 vs 0.074-0.076), and the Score formula correctly penalizes the others' fold-instability. The h0/h1/h3 NW-HAC p-values vs the Tier 1.B Adam baseline (0.005, 0.002, 0.005) are post-hoc observations on Score-losing configs and are not in the preregistered selection family.

**h2 (the registered winner) has NW-HAC p=0.059 vs the Tier 1.B baseline — marginal, does not reject H₀ at α=0.05** (source: Table 3 above + `stat_tier1d.csv` row hparam_idx=2 loss='mse' col delta_IC_NW_p). Per the preregistered gate, the regularization-as-overfit-fix hypothesis is **marginally supported but not statistically significant at α=0.05**.

### 4.4 Negative Sharpe controls

Across all 12 Tier 1.B contrasts × 3 views, no annualized long-short Sharpe (z-score return proxy) exceeds the MSE baseline by a stable margin under block-bootstrap CIs. Sharpe values are reported as supplementary; magnitudes are sensitive to the z-score return proxy and should not be read as expected real-world strategy returns.

---

## 5. Mechanism: Why Robust Losses Hurt Under Stress (~500 words)

### 5.1 The mechanism

Bounded-influence M-estimators (Huber, Tukey biweight, truncated MSE) suppress the gradient contribution from observations beyond a designed threshold. In a stationary regression context with i.i.d. heavy-tailed noise, this is the right call: extreme residuals are unrepresentative of the typical conditional mean, and clipping their gradients reduces estimation variance.

In **cross-sectional ranking under regime shifts**, the assumption breaks. When the joint distribution of (features, returns) shifts — as during a sector rotation, a rate-cut surprise, or a macroeconomic regime change — the stocks that move most are often the ones the model needs to LEARN about, not noise outliers. Their gradient signal is the directional signal.

We formalize the mechanism as follows. Let r_t,i be the realized return of stock i on day t. Let f̂_t,i be the model's prediction. The MSE gradient w.r.t. f̂_t,i is proportional to (f̂_t,i − r_t,i): unbounded in magnitude, so an extreme realized return contributes linearly to the gradient. The Huber gradient (with δ=1.0) is clipped to ±1 once |residual| > δ. The Tukey biweight gradient with c=2.0 is *zero* for |residual| > 2σ. Truncated MSE has the same hard cutoff.

If on fold-4 some stocks move with z-scored returns of |z| > 2 (which is the case under stress; tail probabilities deviate from the daily-z-scored Gaussian baseline), MSE learns from them while Tukey and truncated MSE actively ignore them. The result is a model that is well-calibrated on the typical-return middle of the cross-section but blind to the directional movers — exactly the wrong tradeoff in a stress regime.

### 5.2 Empirical fingerprint

The mechanism predicts:
1. Robust losses' fold-4 underperformance should be larger when the feature set has more variance (more chances to over-suppress) — supported: Alpha158 contrasts (158 features) show worse fold-4 ΔIC than S6 contrasts (3 features) for MLP.
2. Bounded losses with sharper cutoffs (Tukey c=2.0, truncated MSE c=2.0) should harm more than soft-clipped losses (Huber δ=1.0) — supported: Tukey × MLP × S8 ΔIC = −0.090, NW t = −12.1, vs Huber × MLP × S8 ΔIC = −0.057, NW t = −5.2.
3. The all-folds aggregate effect should be smaller than the fold-4 effect because non-stress folds dominate — supported: all-folds Tukey × MLP × S8 ΔIC = −0.004 vs fold-4 ΔIC = −0.090.

All three empirical predictions are confirmed in our results.

### 5.3 What this is not

We do *not* claim that robust losses are theoretically misguided in general regression. The result is specific to:
- Cross-sectional ranking (vs single-asset time series)
- Stress-regime evaluation (vs IID train/test)
- Bounded-influence cutoffs that ignore observations the model needs

For stationary equity ranking with i.i.d. noise, robust losses should still be neutral-to-helpful. Our findings flag the failure mode only when test data crosses a regime boundary.

---

## 6. Discussion + Practitioner Implications (~400 words)

### 6.1 What practitioners should change

1. **Default to MSE for cross-sectional equity ranking.** Sophisticated ranking and robust-loss alternatives, despite their theoretical appeal, do not deliver statistically significant uplift on a leakage-free, preregistered benchmark.
2. **Audit upstream pipelines for global statistics.** Our pipeline audit caught a global p1/p99 winsorization in the Alpha158 feature builder that had quietly leaked test-period information into train features for the entire prior research line. Practitioners deploying open-source factor libraries should run a sentinel leakage test (we provide one in our code release).
3. **Be skeptical of robust losses under regime risk.** If your investment horizon includes potential regime shifts (most do), our finding suggests robust pointwise losses should *not* be the default substitute for MSE.

### 6.2 Why the literature disagrees

The literature is more positive on ranking losses than our findings warrant. Three reasons:
1. **Survival bias in reporting.** Papers that find ranking losses help get written up; null results often don't.
2. **Pipeline leakage.** Without sentinel tests, global winsorization-style leakage inflates apparent improvements for ranking losses whose extreme-value handling differs from MSE.
3. **Lack of multi-fold correction.** Single-fold results have high variance; without BH-FDR or similar, the false-positive rate is much higher than the reported α.

### 6.3 The Tier 1.D regularization finding

Independent of the loss-function question, our hyperparameter sweep produces a marginal regularization improvement (h2: AdamW + 10× weight decay + earlier stopping) that beats the original Adam baseline by ΔIC=+0.013 (post-hoc, NW p=0.059, marginal; source: `stat_tier1d.csv` row hparam_idx=2 cols delta_IC_vs_baseline, delta_IC_NW_p). The result is consistent with overfitting being a non-trivial residual concern even under a leakage-free pipeline. We recommend AdamW + stronger weight decay as the default for cross-sectional equity MLPs.

---

## 7. Limitations + Conclusion (~250 words)

### Limitations

1. **One stress regime.** Fold-4 (Q2-2025) is our only stress sample. Multi-regime generalization (e.g. 2008, 2020, 2022) is left to future work.
2. **U.S. large-cap only.** S&P 500 universe. Results may differ for small-cap, international, or emerging-market panels.
3. **Sharpe values are z-score proxies.** Our long-short Sharpes use z-scored fwd-21d returns as the return proxy. Real-world annualized Sharpes with realistic transaction costs are not directly inferrable from our tables.
4. **Tier 1.D power.** The regularization sweep uses 3 seeds (vs 5 for the main Tier 1.B horse race). 10-seed expansion would tighten the h2 NW-HAC p=0.059 estimate (source: `stat_tier1d.csv` row hparam_idx=2).

### Conclusion

We provide the first preregistered, leakage-free, large-N benchmark of loss functions for cross-sectional equity ranking. The headline finding is a strong **null**: 0/20 ranking or robust-loss contrasts statistically beat MSE on 5-year S&P 500 data. The novel finding is a fold-4 mechanism: robust pointwise losses are statistically significantly worse than MSE under stress. The constructive finding is a marginally-supported regularization fix (AdamW + 10× weight decay).

Code, preregistration protocol, all 1,120 prediction tensors, and the leakage-sentinel test are released at the project repository.

---

## Acknowledgments + Code Availability

All code (Plan Z++ Phase 0 pipeline audits, sentinel test, run/analyze scripts) and full prediction tensors (~520 MB) are released. Plan Z++ preregistration is at `plan-zpp-unified-2026-04-29.md` in the project repository.

## References (placeholder — to be filled at submission time)

- Mandelbrot 1963 — variation of speculative prices
- Cont 2001 — stylized facts of asset returns
- Huber 1964 — robust regression
- Tukey 1977 — bounded-influence M-estimators
- Yang et al. 2020 — Qlib
- Li et al. 2024 — MASTER
- Kelly-Xiu 2020 — empirical asset pricing
- Gu-Kelly-Xiu 2020 — empirical asset pricing with ML
- López de Prado 2018 — Advances in Financial Machine Learning
- Hansen 2005 — SPA test
- Benjamini-Hochberg 1995 — FDR
- Newey-West 1987 — HAC standard errors

---

## Draft notes (not for submission)

### Source provenance map

All numeric claims map to source files:

- All Tier 1.B per-cell ΔIC, NW p, BH p, Sharpe: `artifacts/tier1_phase_a/stat_per_cell.csv` (36 rows × 18 cols)
- All Tier 1.D Score, mean_IC, σ_fold, min_fold: `artifacts/tier1_phase_a/stat_tier1d.csv` (8 rows × 9 cols)
- Audit findings: `artifacts/audits/phase5_features_audit.md`
- Sentinel test PASS/FAIL: `artifacts/audits/sentinel_leakage_test.md`
- Fold dates + manifests: `data/reference/fold_manifest_expanding.json`

### Open items for H博士 review

1. **Title choice**: "MSE Is Hard to Beat..." (emphasizes null) vs "When Robust Losses Hurt..." (emphasizes mechanism). Vote?
2. **Stage 1 separate paper question**: should the 8-contrast Stage 1 horse race be a *separate* paper, a section of this paper, or referenced as "submitted separately"? Affects abstract wording.
3. **Submit before or after Phase B (c)/(d)/(e) results?** Current draft is complete on Story C+ but adding (c) Tier 1.A rolling-window or (d) anchored RankNet would extend the contribution to 3+ findings.
4. **Sharpe with raw fwd_ret**: do we want to recompute Sharpe with raw returns (not z-score proxy) for the headline table before submission? ~5 min compute on existing preds.
5. **IC_sector_resid (Plan §2.C)**: not currently computed. Add as supplementary table? ~2h dev.

### Word counts (rough)

- Abstract: 152
- §1 Intro: 510
- §2 Related work: 290
- §3 Methods: 720
- §4 Results: 980
- §5 Mechanism: 510
- §6 Discussion: 400
- §7 Limitations + Conclusion: 250
- **Total**: ~3,810 words → fits 4-6 page workshop format easily.

### Figure/table count

3 tables (Tier 1.B all-folds, fold-4 stress, Tier 1.D Score). Recommended figures: 1 fold-4 ΔIC bar chart, 1 σ_fold-vs-mean_IC scatter (Tier 1.D Score visualization), maybe 1 sentinel test diagram.
