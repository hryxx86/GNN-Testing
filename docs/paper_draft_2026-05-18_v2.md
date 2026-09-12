# Paper Draft v2 — Plan Z++ Story C+ (2026-05-18)

**Update from v1 (2026-05-14)**: adds Tier 2.C (IC_sector_resid secondary metric across 1,720 cells) + Tier 1.E (pre-registered regime-stratified forensic on ListMLE fold-4 collapse). **Key narrative change**: Tier 1.E REJECTS the pre-registered lagged-cs-dispersion regime hypothesis for the ListMLE collapse (0/4 cells pass primary gate). The "bounded-influence under regime shifts" mechanism narrative in v1 is revised: the empirical fold-4 harm pattern is robust (11/12 NW-significant at h2), but the **specific** mechanism (high-dispersion regime stress) is rejected. The collapse mechanism remains unexplained — a paper-grade negative finding.

**Target venue**: ICAIF 2026 workshop / FinNLP@EMNLP. 4-6 page workshop format.

**Status**: Markdown v2. Provenance-clean.

---

## Title

**MSE Is Hard to Beat: A 28-Contrast Preregistered Benchmark of Loss Functions for Cross-Sectional Stock Ranking, with Three Mechanism Tests and a Rejected Regime Hypothesis**

*(alternative)*: **Three Failed Mechanisms: A Preregistered Forensic of Why Robust and Ranking Losses Lose to MSE on Cross-Sectional Equity Data**

---

## Abstract (~180 words)

We preregister and execute the largest controlled benchmark of loss functions for cross-sectional equity ranking: 1,720 model fits across three loss families on a leakage-free S&P 500 panel (501 tickers, 5 years, 21-day horizon). After Benjamini-Hochberg correction at α=0.05, **0 of 28 (loss × architecture × feature) contrasts beat mean-squared error** (source: `phase_b_finalize/stat_*.csv` view='all_folds' col p_NW_BH_adj). The null is robust to two hyperparameter baselines (Adam, AdamW+10×wd) and to a sector-adjusted IC secondary metric (Plan §2.C). Robust pointwise losses (Huber, Tukey biweight, truncated MSE) are statistically significantly **worse** than MSE on the one stress-regime fold in our panel, with 11 of 12 contrasts at NW-HAC p<0.05 in the negative direction under the stronger baseline. Anchored Bradley-Terry pairwise with explicit σ-guard fails the preregistered scale-guard gate in 0 of 4 cells. A pre-registered regime-stratification test (Plan §1.E) **rejects** the lagged-cross-sectional-dispersion hypothesis for the ListMLE catastrophic fold-4 collapse: degradation_share is below 0.50 in 0 of 4 cells, with three of four cells showing NEGATIVE degradation_share. **The ListMLE collapse mechanism remains unexplained after a pre-registered regime forensic.** We release all 1,720 prediction tensors and the preregistration protocol.

---

## 1. Introduction (~520 words)

### 1.1 Motivation

Cross-sectional ranking — predicting which stocks will outperform peers on a fixed horizon — is the core task in factor investing and quantitative trading. The literature offers a menu of training objectives: MSE, pairwise ranking (RankNet, LambdaRank), listwise (ListMLE, ApproxNDCG), robust pointwise (Huber, Tukey biweight, truncated MSE), and pairwise losses with explicit scale guards.

Three intuitive predictions:
1. **Ranking losses** should help because the evaluation metric (Information Coefficient, IC) is rank-based.
2. **Robust pointwise losses** should help under heavy-tailed returns (Mandelbrot 1963; Cont 2001).
3. **Pairwise losses with σ-guards** should avoid the prediction-scale collapse failure mode.

All three are wide industry practice. All three are *under-tested* on cross-sectional equity panels with controlled statistical correction.

### 1.2 Contributions

1. **Preregistered, leakage-free benchmark.** Plan Z++ fixes contrast sets, statistical primaries, and pass thresholds *before* any experiment. We catch and fix a global p1/p99 winsorization in the Alpha158 feature builder that had quietly contaminated prior baselines (source: `artifacts/audits/phase5_features_audit.md`).

2. **Strong cumulative null across three loss families.** Across 28 (loss × architecture × feature) contrasts at BH-FDR α=0.05, **zero alternative losses beat MSE**. Three loss families × two hparam baselines (where applicable) all fail (source: `tier1_phase_a/stat_per_cell.csv` + `phase_b_finalize/stat_tier1b_h2.csv` + `phase_b_finalize/stat_tier1c.csv`).

3. **Cross-baseline robustness.** The 12 robust-loss contrasts are re-run at a Tier 1.D registered-Score-winner baseline (AdamW + 10× weight decay + earlier stopping). Result: 0/12 BH-FDR rejections again; 12/12 ΔIC negative (vs 11/12 at Adam). The null is robust to hparam tuning.

4. **Sector-adjusted robustness.** Plan §2.C secondary metric (IC_sector_resid: Spearman vs sector-residualized z-scored fwd returns) is computed for all 1,720 cells. Mean per-experiment IC_sector_resid is comparable to or smaller than IC_abs across all loss families; loss-family orderings preserved (source: `artifacts/phase_b_finalize/ic_sector_resid_per_cell.csv`). The null holds under sector adjustment.

5. **Fold-4 stress effect on robust losses.** Robust pointwise losses are statistically significantly worse than MSE on fold-4 (Q2-2025): 8/12 contrasts at the Adam baseline, **11/12** at the h2 baseline (source: `phase_b_finalize/stat_tier1b_h2.csv` view='fold_4'). The effect is hparam-agnostic. We interpret this as a mechanism finding — but see Contribution 6 for the regime-mechanism caveat.

6. **Pre-registered regime hypothesis REJECTED for ListMLE collapse (NEW v2 finding).** Plan §1.E pre-registered a primary mechanism test: lagged 21-day cross-sectional return dispersion explains ≥ 50% of the fold-4 IC degradation. Across all four ListMLE × (architecture × feature) cells, degradation_share is below 0.50 (mean −0.36; three of four cells NEGATIVE; placebo p > 0.05 in 3/4 cells; source: `artifacts/phase_b_finalize/tier1e_regime_forensic.csv`). Two secondary regime variables (drawdown, market volatility) also fail. **The ListMLE catastrophic collapse mechanism is empirically unexplained**, despite a pre-registered test specifically designed to identify it.

7. **σ-guard mechanism failure.** Anchored RankNet with σ_penalty=0.05 designed to enforce min prediction std ≥ 0.05 produces empirical median cross-sectional std of 0.022-0.036 across 4/4 cells — fails Gate 1.C universally. The explicit anti-collapse mechanism is empirically inadequate.

8. **Regime-conditional rolling-window finding.** Rolling 2-year windows produce statistically significant ListMLE fold-4 attenuation (+0.092 IC, NW p=0.009; source: `phase_b_finalize/stat_tier1a.csv`) but do not pass the all-folds + folds 0-3 general-preference gate. Rolling-window training partially attenuates the collapse but the mechanism is not "stale-regime contamination" alone.

### 1.3 Why this matters

The 0/28 null and the rejected regime mechanism together provide a clean negative result on three of the most popular alternative-loss families. The benchmark establishes a methodological standard (preregistration + leakage audit + sentinel test + per-fold winsor + BH-FDR + NW-HAC + placebo) that future cross-sectional equity ranking studies should match before claiming improvements over MSE.

---

## 2. Related Work (~300 words)

(Same as v1 §2 — Ranking losses in finance / Robust regression / Pairwise scale stabilization / Walk-forward + statistical correction / Pre-registration in ML.)

---

## 3. Methods (~720 words)

(Same as v1 §3, with addition of Tier 2.C IC_sector_resid in §3.6 statistical primaries:)

§3.6 Statistical primaries (updated):

- **Estimand**: paired daily IC differences d_{f,s,t}, average-then-HAC (Plan §B-02 (i)), NW-HAC Bartlett kernel lag=21.
- **Sensitivity**: fold-cluster bootstrap n_boot=10K.
- **Multiple testing**: BH-FDR within each experiment's contrast family at α=0.05.
- **Three views per cell**: all_folds (313 d), folds_0_3 (251 d), fold_4 (62 d, diagnostic).
- **Plan §2.C secondary metric (NEW)**: IC_sector_resid = Spearman correlation between predictions and sector-residualized z-scored 21-day forward returns. Computed for all 1,720 cells.

§3.7 Pre-registered regime forensic (NEW)

Plan §1.E specifies a one-shot pre-registered mechanism test for the ListMLE catastrophic fold-4 collapse:
- **Primary regime variable**: lagged 21-day cross-sectional return dispersion (per Codex A Q3 highest priority).
- **Degradation share formula**:
  ```
  degradation_share = (IC_high_disp_fold4 - IC_low_disp_fold4) / (IC_baseline_folds_0to3 - IC_fold4)
  ```
- **Primary gate**: degradation_share ≥ 0.50 → trigger Tier 2.A (Group-DRO). degradation_share < 0.50 → null finding, Tier 2.A skipped.
- **Random-label placebo**: shuffle dispersion tercile assignments 10K times; observed must exceed placebo distribution at one-sided α=0.05.
- **Secondary diagnostics** (Bonferroni α/3 = 0.0167): lagged 21-day market drawdown, lagged 21-day realized volatility.

---

## 4. Results (~1050 words)

### 4.1 Tier 1.B Adam: 0/12 BH-FDR, 8/12 fold-4 NW-significant

(Same as v1 §4.1.)

### 4.2 Tier 1.B h2 (cross-baseline): null stronger under stronger baseline

(Same as v1 §4.2: 0/12 BH-FDR, 12/12 ΔIC negative, 11/12 fold-4 NW-significant negative.)

### 4.3 Tier 1.C: anchored RankNet 0/4 scale-guard gate

(Same as v1 §4.3: 0/4 BH-FDR, 0/4 Gate 1.C — σ-guard fails universally.)

### 4.4 Tier 1.A: regime-conditional ListMLE rolling-window attenuation

(Same as v1 §4.4: ListMLE fold-4 attenuation NW-significant p=0.009 but FAILS general-preference gate.)

### 4.5 Tier 2.C (NEW): null robust to sector adjustment

For each of the 28 Tier 1 contrasts plus Stage 1 600 cells, we compute IC_sector_resid using sector-residualized z-scored 21-day forward returns (Plan §2.C; sector mapping from `data/reference/sp500_sectors.csv` GICS sectors).

Per-experiment aggregates (mean across all cells in the experiment; source: `phase_b_finalize/ic_sector_resid_per_cell.csv`):

| Experiment | N cells | mean IC_abs | mean IC_sector_resid | Δ |
|---|---|---|---|---|
| Stage 1 | 599 | −0.012 | −0.008 | +0.004 |
| Tier 1.A (rolling+expanding) | 100 | −0.009 | −0.007 | +0.003 |
| Tier 1.B Adam | 400 | +0.002 | +0.001 | −0.001 |
| Tier 1.B h2 | 400 | +0.005 | +0.003 | −0.002 |
| Tier 1.C | 200 | +0.014 | +0.008 | −0.006 |

(Source: `phase_b_finalize/ic_sector_resid_per_cell.csv`.)

**Sector-residualization absorbs a small fraction of IC** (typical |Δ| < 0.01). **The loss-family ordering is preserved**: within Tier 1.B Adam, MSE still has the highest mean IC; within Tier 1.B h2, MSE still has the highest mean IC; etc. **All 28 BH-FDR null verdicts remain unchanged** under the sector-adjusted secondary metric.

This is a paper-grade robustness check: the null is not an artifact of sector-loading concentration in predictions.

### 4.6 Tier 1.E (NEW): pre-registered regime hypothesis REJECTED for ListMLE collapse

Plan §1.E primary gate test on the four ListMLE × (architecture × feature) catastrophic-collapse cells (Stage 1 600-cell horse race):

| Cell | IC_baseline_0-3 | IC_fold4 | Degradation magnitude | deg_share_primary | Placebo p (one-sided) | PASS gate? |
|---|---|---|---|---|---|---|
| listmle/MLP/S6 | +0.029 | −0.308 | 0.337 | **−0.64** | 1.000 | **NO** |
| listmle/MLP/S8 | +0.006 | −0.296 | 0.302 | **−0.06** | 0.711 | **NO** |
| listmle/SAGE-Mean/S6 | +0.018 | −0.287 | 0.305 | **−0.89** | 1.000 | **NO** |
| listmle/SAGE-Mean/S8 | +0.002 | −0.278 | 0.280 | **+0.17** | 0.099 | **NO** |

(Source: `artifacts/phase_b_finalize/tier1e_regime_forensic.csv` rows experiment='stage1' loss='listmle'.)

**0/4 cells pass the primary gate** (degradation_share ≥ 0.50). Three of four cells have NEGATIVE degradation_share — meaning that within fold-4, high-dispersion days have LOWER IC than low-dispersion days for ListMLE. The hypothesis "ListMLE collapse is driven by lagged cross-sectional dispersion stress" is **rejected by the data**.

The placebo test (10K shuffles of tercile labels within fold-4) gives one-sided p > 0.05 in 3/4 cells, consistent with no signal. The MLP/S8 cell's placebo p of 0.099 is the closest to significance and is positive (+0.17), but the magnitude is far below the 0.50 gate threshold and does not survive the binary gate.

**Secondary diagnostics** (Plan §1.E secondary, Bonferroni α/3 ≈ 0.017):

| Cell | deg_share_drawdown | deg_share_market_vol |
|---|---|---|
| listmle/MLP/S6 | −0.59 | −0.67 |
| listmle/MLP/S8 | −0.15 | −0.13 |
| listmle/SAGE-Mean/S6 | −0.65 | −0.92 |
| listmle/SAGE-Mean/S8 | +0.19 | +0.09 |

(Source: same CSV.)

**Both secondary regime variables also fail** to support the regime hypothesis: 3/4 cells NEGATIVE degradation_share for drawdown and market vol; only the SAGE-Mean/S8 cell shows weak positive support across all three regime variables, but at magnitudes far below 0.50.

**Verdict**: Plan §1.E pre-registered hypothesis is REJECTED across 4 (architecture × feature) cells × 3 regime variables = 12 mechanism tests. Tier 2.A (Group-DRO, conditional on Tier 1.E PASS) is correctly skipped per the pre-registered protocol.

**The ListMLE catastrophic collapse mechanism is empirically unexplained.** The lagged cross-sectional dispersion stress hypothesis is intuitive (Stivers & Sun 2010 link dispersion to factor instability) but is not supported on our data. We disclose this as a paper-grade negative mechanism finding.

A nominal-PASS subtlety: Tier 1.B Adam tukey × SAGE-Mean × S8 and trunc_mse × SAGE-Mean × S8 cells nominally pass the primary gate (deg_share 19.2 and 67.0 respectively), but the denominators are very small (fold-4 IC ≈ baseline IC), and the large deg_share is a numerical artifact of the formula under near-zero denominator rather than a genuine mechanism signal. Disclosed as caveat — not interpreted as a positive mechanism finding.

### 4.7 Cumulative summary

Across 28 (loss × architecture × feature) contrasts in Tier 1.B Adam + Tier 1.B h2 + Tier 1.C: **0/28 BH-FDR rejections**. Plus a pre-registered ListMLE-collapse regime hypothesis test (Tier 1.E): **0/4 cells pass primary gate**. Sector-adjusted IC secondary metric (Tier 2.C): null preserved across all 1,720 cells.

---

## 5. Mechanism: Three Failed Hypotheses (~700 words)

### 5.1 Bounded-influence under fold-4 stress (PARTIAL support)

Bounded-influence M-estimators (Huber, Tukey, truncated MSE) suppress gradient contributions from observations beyond a designed threshold. In stationary regression with i.i.d. heavy-tailed noise, this reduces estimation variance.

**Empirical finding (from §4.1 + §4.2)**: 8/12 contrasts (Adam) and 11/12 contrasts (h2) have NW-HAC p < 0.05 in the NEGATIVE direction on fold-4 (Q2-2025) — i.e., robust losses are statistically significantly worse than MSE on the one stress-regime fold in our panel.

**Plausible mechanism**: under regime shift, the stocks that move with |z| > 2 are not noise outliers but the model's directional signal. MSE learns from them; Tukey/trunc_mse with cutoff c=2.0 actively ignore them.

**Important caveat (Tier 1.E)**: the *specific* mechanism — high lagged-cross-sectional-dispersion days within fold-4 driving the harm — is rejected by Tier 1.E. The fold-4 harm pattern is robust empirically, but our pre-registered proxy for "regime stress" does not stratify fold-4 days in a way that explains the harm differentially. **The harm mechanism for robust losses on fold-4 is therefore empirically observed but the specific within-fold mechanism is unconfirmed.**

### 5.2 Anchored RankNet σ-guard mechanism failure (CLEAN negative finding)

Plan §1.C designed an anchored RankNet loss specifically to address the prediction-scale-collapse failure mode of pairwise hinge losses, combining (a) Bradley-Terry pairwise log-loss on top/bottom-k pairs, (b) Huber anchor, (c) **σ_penalty=0.05 · ReLU(σ_min − std(pred))²** with σ_min=0.05.

**Empirical result**: predicted std at h2-trained anchored RankNet is 0.022-0.036 across 4/4 cells (source: `stat_tier1c.csv` col median_pred_cs_std_new) — well below the 0.05 floor.

**Mechanism**: the σ_penalty=0.05 coefficient is insufficient to overcome the pairwise loss's natural compression incentive. Practitioners should use much larger penalty coefficients (10×-100× our value) or hard rank-margin constraints instead of soft σ-guards.

### 5.3 Stale-regime contamination of ListMLE training set (PARTIAL support; INCOMPLETE)

Plan §1.A tested whether rolling 2y windows attenuate ListMLE's fold-4 collapse by removing stale 2021-2022 regime contamination.

**Empirical finding (from §4.4)**: ListMLE fold-4 IC improves from −0.282 (expanding) to −0.190 (rolling 2y), an attenuation of +0.092 IC that is NW-significant (p=0.009). But the rolling window does NOT pass the all-folds + folds 0-3 stability gate.

**Mechanism**: rolling-window training removes some — but not all — of the fold-4 collapse. The mechanism is **partial** stale-regime contamination, not the dominant driver.

### 5.4 Lagged-dispersion regime stress for ListMLE (REJECTED)

Plan §1.E tested whether ListMLE's fold-4 catastrophic collapse is concentrated in high lagged cross-sectional dispersion days within fold-4. Pre-registered primary gate: degradation_share ≥ 0.50 (high-vs-low dispersion IC difference ≥ half the fold-4 vs folds 0-3 baseline degradation).

**Empirical result (from §4.6)**: 0/4 ListMLE cells pass the gate. Three of four cells have NEGATIVE degradation_share. Secondary regime variables (drawdown, market vol) also fail.

**Interpretation**: The lagged cross-sectional dispersion mechanism — intuitive and supported by prior factor instability literature (Stivers & Sun 2010) — does NOT explain the ListMLE collapse. The collapse mechanism is **unexplained** after this pre-registered forensic.

Possible non-rejected hypotheses (left for future work):
1. Softmax-over-ranking surrogate's sensitivity to ties in test labels (we add tie-perturbation but the effect persists)
2. Fold-4 test distribution shift in feature space (not in label space) that lagged dispersion doesn't capture
3. Selection bias in the 21-day forward return label under regime shift
4. Optimization-path-dependent local minimum reached during training on stale-regime data (we partially confirm with rolling-window attenuation in §5.3)

### 5.5 Synthesis

We test three pre-registered mechanism hypotheses. **Two are partial-fail and one is clean-reject:**

| Hypothesis | Result | Evidence |
|---|---|---|
| Bounded-influence loss × regime stress | Empirical fold-4 harm: **YES**; specific dispersion mechanism: **NO** | §4.2 + §4.6 |
| σ-guard prevents scale collapse | **NO** (0/4 Gate 1.C) | §4.3 |
| Stale-regime training → ListMLE collapse | **Partial** (+0.092 attenuation, p=0.009; but not general) | §4.4 |
| Lagged cross-sectional dispersion → ListMLE collapse (specific) | **NO** (0/4 primary gate) | §4.6 |

**Net: three negative mechanism findings, plus partial-support for rolling-window remedy for ListMLE only.** The ListMLE catastrophic collapse mechanism remains unexplained.

---

## 6. Discussion + Practitioner Implications (~430 words)

### 6.1 What practitioners should change

1. **Default to MSE for cross-sectional equity ranking.** 28 contrasts at BH-FDR α=0.05; no alternative beats MSE on a leakage-free benchmark.
2. **Audit upstream pipelines for global statistics.** Sentinel test artifact released; the global p1/p99 winsorization in Alpha158 we caught was a real bug in prior research.
3. **Be skeptical of robust losses under regime risk.** Empirical fold-4 harm is large (ΔIC −0.03 to −0.15) and statistically significant. The specific mechanism is uncertain, but the warning is concrete.
4. **σ-guard mechanisms in pairwise losses need calibration.** Our σ_penalty=0.05 with σ_min=0.05 produces empirical median std 0.022-0.036. Future pairwise losses should use larger penalty coefficients or hard rank-margin constraints.
5. **Rolling-window training is regime-conditional.** Plan §1.A "rolling generally preferable" gate fails. Rolling helps fold-4 ListMLE significantly but does not generalize. Consider regime-aware deployment, not a universal switch.
6. **NEW (Tier 1.E): the ListMLE collapse mechanism is empirically unexplained.** Practitioners using ListMLE for cross-sectional ranking should monitor for catastrophic fold-level collapse even in cases where lagged dispersion does not flag elevated stress.

### 6.2 Why the literature disagrees

(Same as v1 §6.2.)

### 6.3 The Tier 1.D regularization finding

(Same as v1 §6.3 — h2 marginally helps MSE at registered Score gate, NW p=0.059.)

### 6.4 NEW: the value of pre-registered mechanism tests

Our Tier 1.E pre-registered regime-stratification test rejects an intuitive and literature-supported mechanism for the ListMLE collapse. This is exactly the kind of finding that pre-registration is designed to surface: without pre-registration, a researcher might select a different regime variable post-hoc that yields a more favorable result (a forking-paths problem). The pre-registered failure is also an invitation: the field should test alternative ListMLE-collapse mechanisms before adopting ListMLE as a default ranking loss.

---

## 7. Limitations + Conclusion (~280 words)

### Limitations

1. **One stress regime** (fold-4 Q2-2025). Multi-regime confirmation requires additional historical periods.
2. **U.S. large-cap only.** Results may differ for small-cap, international, or emerging-market panels.
3. **Sharpe values are z-score proxies.**
4. **5 seeds.** With 0/28 BH-FDR rejections, no 10-seed expansion is authorized.
5. **Stage 1 (8 contrasts) is reported separately** but referenced here for the Tier 1.E ListMLE primary target.
6. **Tier 1.E's small-denominator artifact**: 2 robust-loss cells nominally pass the primary gate with deg_share = 19.2 and 67.0, but the small denominators (fold-4 ≈ baseline IC) make the result a formula artifact rather than a mechanism finding. Disclosed in §4.6.
7. **Tier 2.C secondary metric**: confirms null but does not establish a positive alternative.

### Conclusion

We provide the first preregistered, leakage-free, large-N benchmark of loss functions for cross-sectional equity ranking, with explicit hparam-baseline sensitivity, sector-adjusted robustness check, and a pre-registered regime-stratified mechanism forensic. The headline finding is a strong cumulative null: **0/28 BH-FDR rejections** across three distinct alternative-loss families on the same panel. **Three mechanism hypotheses are tested and three are non-supportive**: bounded-influence regime-stress (specific lagged-dispersion mechanism rejected, fold-4 harm pattern robust); σ-guard scale prevention (universal failure); stale-regime contamination for ListMLE (partial attenuation only). **The ListMLE catastrophic collapse mechanism is empirically unexplained**, despite a pre-registered forensic test specifically designed to identify it.

We release all 1,720 prediction tensors, the preregistration protocol, the leakage-sentinel test, and the regime-forensic analysis script at the project repository.

---

## Source provenance map

(Same as v1 + Tier 1.E + Tier 2.C additions:)

- Stage 1 Tier 1.E regime forensic: `artifacts/phase_b_finalize/tier1e_regime_forensic.csv` (21 rows × 17 cols)
- Tier 2.C sector IC: `artifacts/phase_b_finalize/ic_sector_resid_per_cell.csv` (1,720 rows × 8 cols)
- All other sources same as paper v1.

## Word counts (rough)

- Abstract: 185
- §1 Intro: 600 (8 contributions now)
- §2 Related work: 305
- §3 Methods: 745 (added §3.7 regime forensic spec)
- §4 Results: 1,080 (added §4.5 Tier 2.C + §4.6 Tier 1.E + §4.7 cumulative)
- §5 Mechanism: 720 (added §5.4 lagged-dispersion REJECTED + §5.5 synthesis)
- §6 Discussion: 430 (added §6.4 value of preregistered mechanism tests)
- §7 Limitations + Conclusion: 290
- **Total**: ~4,355 words → fits 6-page workshop format with tables.

## Open items for H博士 review

1. **Title choice** (3 alternatives in current draft)
2. **Stage 1 integration**: this v2 references Stage 1 as separately preregistered but also includes Stage 1's ListMLE cells in Tier 1.E §4.6. Consider whether to fold Stage 1 fully into this paper for a 36-contrast version.
3. **Sharpe with raw fwd_ret** vs z-score proxy: still using z-score; switch for headline table?
4. **Final Codex Touchpoint 2 + 3 on paper v2**: pending Codex quota reset.
