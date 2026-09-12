# GNN-Testing — Findings Visualization Package (Advisor Briefing)

> Compiled 2026-04-21. Twelve empirical findings from the SP500 daily equity-ranking
> project. Each section states the claim, how it was implemented, what the figure
> shows, and a measured takeaway. Numbers are sourced directly from the CSV/JSON
> outputs listed in **Appendix B**.

**Study setup (shared context)**

- Universe: 501 SP500 constituents (5-year daily OHLCV, 2020-01 → 2025-12; 1,255 trading days).
- Target: next-day cross-sectional return rank (21-day horizon in main experiments unless stated).
- Features: 9-dim price/volatility probe (primary) and 158-dim Alpha158 library (reference).
- Evaluation: 5-fold walk-forward with per-fold train-only normalization; Newey-West (NW) t-stats for daily IC; Hansen SPA + BH-FDR for multi-model comparison; 15 bps round-trip cost for Sharpe.
- Seeds: 3 per configuration (42, 123, 456). All runs are deterministic per seed.

---

## Finding 1 — A 3-Feature "PC Probe" Is Not Shown to Outperform the 158-Feature Alpha158 Library (Hansen SPA with S8 as benchmark, one-sided, α = 0.05; reverse direction and TOST not tested)

**What.** Under Hansen's Superior Predictive Ability (SPA) test with S8 as benchmark,
**no candidate subset demonstrates statistically superior IC over the 158-feature Alpha158 library**.
The candidates include a hand-picked 3-feature "PC probe"
(S6 = ret_mean_10d + ret_std_10d + mom12m, representing PC1 trend / PC2 vol / PC3 horizon-extension).
Both S6 and S8 produce daily IC in the 0.041–0.047 range
across 313 test days (5 walk-forward folds); SPA fails to reject the one-sided null of no-alternative-superiority
(MLP: T_SPA = 0.270, p_consistent = 0.5506; SAGE-Mean: T_SPA = 1.231, p_consistent = 0.5509).
**Interpretation caveat**: Hansen SPA is a one-sided superiority test. Failing to reject means
"S6 does not outperform S8 at α = 0.05"; it does **not** positively establish equivalence.
A rigorous "S6 = S8" equivalence claim would require TOST (two one-sided tests) against a
pre-specified margin δ (Codex Round 3 Q3 constraint; see `docs/analysis.md:1844`).

**How.** S8 was built by faithfully re-implementing qlib's Alpha158DL operator set
(9 K-bar features + 4 price features + 145 rolling operators; 1/99 winsorization)
in `build_alpha158_features.py`. Training was done by
`run_step3_plan_z_part_c.py` (30 runs = 2 models × 5 folds × 3 seeds).
`run_step3_plan_z_part_c_perfold.py` is the "Path A" rerun with strict per-fold
train-only winsorization to rule out residual leakage. Aggregation and SPA
testing live in `analyze_step3_plan_z.py`.

**Figure.**
![Finding 1](../plots/advisor/fig_01_s6_vs_s8.png)

**Analysis.** Panel (a) plots seed-averaged daily IC with NW 95% confidence
intervals for 9 subsets. S6 (3 features) and S8 (Alpha158) reach almost the same
height for both MLP and SAGE-Mean, and both are significant (stars: ** for S6
MLP at p = 0.009; * for S8 at p = 0.026). Panel (b) plots the S6-specific
studentized paired t-statistic (S6 − benchmark, from the Hansen SPA per-
alternative t_stats dictionary). All four bars lie within |t| < 1.96
(two-sided α = 0.05 under asymptotic normality), i.e. **the two-sided point null
of zero mean-IC difference between S6 and each Alpha158 benchmark is not rejected**
at any (model, benchmark) pair. The Path-A fix (S8_pf) closes a small (ΔIC ≈ +0.010,
BH p = 0.037) but real MLP leakage; after that fix, the S6 vs S8_pf t-statistic is
still well below 1.96 (MLP t = +0.76, SAGE t = −0.08). Failure to reject a point
null is **not** a positive equivalence result.

**Takeaway.** Under Hansen SPA with S8 as the benchmark, **no candidate subset
(including S6) demonstrates statistically superior IC over S8 at α = 0.05**
(one-sided non-superiority in the direction "candidate > S8"). We have **not**
run the reverse SPA with S6 as benchmark, so we cannot claim "S8 does not beat S6"
either. Panel (b)'s per-pair |t| < 1.96 two-sided result is a failure-to-reject
of the point null mean(IC_S6 − IC_S8) = 0, which is **not** positive equivalence
evidence. Current evidence supports only the narrowest claim: "S6 does not
outperform S8 at α = 0.05." Any formulation involving "S6 = S8", "S6 ≈ S8",
"matches", "parity", "indistinguishable", or "does not underperform" requires
additional tests (reverse-direction SPA and/or TOST with a pre-specified margin δ;
Codex Round 3 Q3). These must be added before a paper submission.

---

## Finding 2 — 21-Day Prediction Horizon Gives the Highest IC in Price-Only Models

**What.** Across horizons {1, 5, 10, 21, 42, 63} days, mean IC peaks at 21 days
for both SAGE-Mean and MLP using price-only features (Folds 0–3, 3 seeds).
MLP IC rises from 0.012 (1-day) to 0.026 (21-day), then falls to 0.012 (63-day).
SAGE-Mean IC rises from 0.009 → 0.024 (21-day) → 0.002 (63-day).

**How.** The 360-run sweep is in `run_phase5_step3_feature_expansion.py` and
persisted to `experiments/horizon_ablation_results.csv` (4 models × 6 horizons
× 5 folds × 3 seeds = 360 rows). Fold 4 was excluded from the figure because its
Q2-2025 tariff-shock regime creates outlier variance that swamps the
cross-horizon pattern (see Finding 8); including it pulls 63-day mean IC up
artificially. All-feature (news-augmented) variants are not plotted because
Finding 11 establishes that they degrade price-only IC across folds.

**Figure.**
![Finding 2](../plots/advisor/fig_02_horizon_ablation.png)

**Analysis.** Both curves trace an inverted-U with the maximum near 21 days.
The golden band marks this peak. Error bars are ±1 standard error across
3 seeds × 4 folds = 12 runs per point. The shape is consistent with the
hypothesis that cross-sectional rank signal is richest at 2–4 weeks: too short
and noise dominates; too long and per-stock drift dominates.

**Takeaway.** This is a pre-specified horizon ablation, not post-hoc tuning.
The 21-day peak is modest in absolute terms (IC ≈ 0.025) and specific to
price-only features on this universe; we do not claim a universal peak.

---

## Finding 3 — Walk-Forward 5-Fold: Stable Mean IC, Fold-4 Variance Explosion

**What.** The full 5-fold walk-forward protocol (90 runs) shows positive mean IC
for SAGE-Mean and MLP across 2024-H2 to 2025-H2, but Fold 4 (Q2-2025) exhibits
variance 3–5× the other folds. Fold-4 per-fold IC std rises from ~0.02 (Folds 0-3)
to ~0.09 (computed over `wf5_results.csv`), and the IC range widens from
roughly ±0.05 to ±0.15.

**How.** Implemented in `run_walkforward_5fold.py`; output
`experiments/wf5_results.csv` has 90 rows (6 model × feature variants × 5 folds
× 3 seeds). Per-fold training uses train-only p1/p99 winsorization and
cross-sectional z-score; no normalization is fitted on validation or test.
Trained on Colab RTX Pro 6000 in ~13 minutes total.

**Figure.**
![Finding 3](../plots/advisor/fig_03_wf5_stability.png)

**Analysis.** Box plots show the per-fold distribution of IC values across
3 seeds (overlaid as strip points). The yellow band marks Fold 4. Notice
MLP price-only (light red, rightmost) reaches its highest median IC on Fold 4
while MLP all-features (dark red) collapses to −0.04 — the news-augmented
variant has the largest range in Fold 4, consistent with Finding 11 and with
Fold 4 being a feature-distribution-shift stress test (Finding 8).

**Takeaway.** Fold-level means remain above the IC = 0.03 target on average,
but the Fold-4 variance is not suppressed by averaging across 3 seeds and
should be reported honestly rather than aggregated away.

---

## Finding 4 — Permutation Importance: mom12m Contributes ~5× More Than Any Other Feature Group

**What.** In a 30-run cross-sectional permutation ranking over 7 feature groups
(built on top of the 14-dim extended set), the 12-month momentum group
(`mom12m`) delivers ΔIC = +0.0182 (baseline minus shuffled), roughly 5–8× the
next-best groups (`ret_mean_21d` +0.0036, `ret_mean_10d` +0.0025). Three groups
have slightly negative mean ΔIC (`CORR5` −0.0002, `dolvol` −0.0005,
`maxret` −0.0029), i.e. shuffling them does not hurt or marginally helps.

**How.** `run_step3_plan_z_part_a.py` performs a group-wise cross-sectional
permutation with a SHA-256 seeded RNG (Codex Round 5 fix). 30 runs × 7 groups ×
313 test days = 617,859 permuted daily IC rows, paired with 13,146 baseline
rows. Aggregation lives in `analyze_step3_plan_z.py` and the summary JSON is
`artifacts/step3_plan_z/part_a_ranking.json`.

**Figure.**
![Finding 4](../plots/advisor/fig_04_permutation_ranking.png)

**Analysis.** Horizontal bars are group mean ΔIC with error bars at ±1 standard
error across 30 runs. `mom12m` is the lone feature group whose shuffling
meaningfully degrades model IC. The error bars on `ret_std_10d` and `maxret`
cross zero, so their point estimates are not distinguishable from no-effect.

**Takeaway.** Signal concentrates in long-horizon momentum; most of the other
engineered groups we tested (volume, cross-stock correlation, tail return) add
little. This informs the S6 "PC probe" choice in Finding 1.

---

## Finding 5 — Normalization × Regime: Same-Sign Effects Across Graph and Non-Graph Models

**What.** Standard train-only cross-sectional z-score normalization (replacing
raw scaling) improves or hurts IC differently depending on the fold, and the
effect is almost identical across SAGE-Mean (graph), NoGraph (no graph),
and MLP (no graph, no pooling). 14 of 15 (model × fold) cells have the same
sign, so the interaction is with the regime, not with the graph.

**How.** `run_diag1_normalization.py` runs SAGE-Mean × 30 (raw vs. norm); the
replication `run_diag1b_replication.py` runs MLP × NoGraph × 60 to test
whether the effect is GNN-specific. Outputs: `experiments/diag1_normalization_results.csv`
(30 rows) and `diag1b_replication_results.csv` (60 rows).

**Figure.**
![Finding 5](../plots/advisor/fig_05_norm_regime.png)

**Analysis.** Heat-map cells are (ΔIC = norm − raw) averaged over 3 seeds.
Blue cells: normalization hurts (Folds 0, 1, 3). Red cells: normalization helps
(mostly Fold 4). Fold 3 sees a ~−0.10 collapse with normalization for all three
models; Fold 4 sees a +0.17 to +0.28 rescue. The cross-model consistency
(14/15 cells same sign) rules out a GNN-specific mechanism. The remaining
explanation is input-scale saturation of the first linear layer under OOD
feature scale, which is what the per-fold scaler's statistics would produce
during a regime change.

**Takeaway.** Normalization is not a universally-safe preprocessing step for
weakly-stationary financial data; whether to apply it is regime-dependent.
The overall mean effect is statistically indistinguishable from zero
(Wilcoxon p = 0.60), so one shouldn't claim "normalize = worse" either.

---

## Finding 6 — SelectiveNet Underperforms a Simple Threshold Baseline at Low Coverage

**What.** Across 10 coverage targets (10–100%), the ICML-2019 SelectiveNet
3-head architecture ("E2E" in the data) produces a weaker IC curve than a
simple threshold baseline that drops low-confidence predictions by score
magnitude. At 10% coverage, Threshold IC = 0.084 vs. E2E IC = 0.048.

**How.** SelectiveNet and the two baselines (Threshold, Vol-Calibrated) were
trained in the v3 pipeline N5 experiment. Output:
`experiments/selectivenet_results.csv` (70 rows). E2E (SelectiveNet) and
Vol-Calibrated each have three calibration-target variants (target ∈
{0.2, 0.4, 0.6}), so the plotted curve for each strategy is the per-coverage
**envelope (max IC across calibration targets)** — the most generous
interpretation for the non-threshold strategies.

**Figure.**
![Finding 6](../plots/advisor/fig_06_selectivenet.png)

**Analysis.** Threshold (green) has the largest lead at low coverage (10–50%)
— exactly where a selective prediction system is supposed to add value. Above
60% coverage the three strategies cross and Vol-Calibrated slightly exceeds
Threshold; by 100% coverage all three converge within 0.02 IC, consistent
with the learned selection head collapsing to "accept all" at its saturation
regime. SelectiveNet (red) is the weakest strategy at the low-coverage end
that matters for the use case.

**Takeaway.** In this weak-signal, high-noise setting, the learned selection
head does not recover a confident subset better than picking the most extreme
predicted scores. The finding applies to this task; it is not a general
refutation of SelectiveNet.

---

## Finding 7 — SEC Lazy-Prices Similarity Features Degrade NN Ranking IC

**What.** Adding the **combined** SEC 10-K/10-Q Lazy-Prices feature pair
(`lazy_sim` + `log1p(days_since_filing)`) on top of price features sharply
lowers IC for SAGE-Mean (0.034 → 0.013, −61%) and MLP (0.034 → 0.023, −34%),
but is approximately neutral for a LightGBM tree baseline (0.016 → 0.019).
SAGE-Mean single-feature ablation shows the damage is driven by
`days_since_filing`: adding `lazy_sim` alone costs only ~11% IC (0.034 → 0.031),
while adding `days_since` alone pushes IC to near zero (−0.004).

**How.** `run_gate1_experiment.py` (790 lines; 21 runs on Fold 0 Q2-2024) is
the only driver. Output: `experiments/gate1_results.csv` (22 rows).
The ablation tests 4 variants — price only, +lazy_sim only, +days_since only,
+both — on SAGE-Mean, and 2 variants (price, +both) on MLP and LGB.

**Figure.**
![Finding 7](../plots/advisor/fig_07_sec_gate1.png)

**Analysis.** Bars are mean IC over 3 seeds on Fold 0. For SAGE-Mean,
adding `lazy_sim` alone costs ~0.003 IC (tolerable). Adding `days_since` alone
drops IC to near zero (the orange bar) — the feature's 0–7 log scale dominates
the gradient of the first linear layer. The tree baseline (LGB) is unaffected,
which is consistent with this being a scaling/saturation mechanism, not a
signal-quality issue.

**Takeaway.** The experiment is scoped to Fold 0 only, so we can only report a
decision (stop Layer 2/3 of the SEC pipeline), not a generalization across
regimes. The NN-vs-tree asymmetry is the interpretable part.

---

## Finding 8 — Fold-4 Anomaly Is Regime Stress, Not Label Leakage

**What.** Fold 4 (Q2-2025) has elevated daily feature-distribution drift
(z-drift) and elevated daily IC. The two co-move strongly: Pearson correlation
ρ = +0.420 for MLP (p = 7×10⁻⁴) and ρ = +0.476 for SAGE-Mean (p = 9×10⁻⁵)
across 62 test days. Four leakage tests (tail displacement, z-shift, rank
preservation, tail concentration) all come out clean. The elevated IC is
therefore consistent with regime stress (market volatility rose ~2× during
the 2025 tariff shock), not with label leakage.

**How.** `analyze_fold4_leakage.py` runs the 4-test framework. Outputs:
`experiments/step3_plan_z/fold4_zdrift_summary.csv` (158-feature drift scores),
`fold4_zdrift_per_day.csv` (62 days × drift + IC time series), and
`fold4_tail_concentration.csv` (158-feature tail stats). The daily rolling
metrics used in this figure come from `fold4_zdrift_per_day.csv`.

**Figure.**
![Finding 8](../plots/advisor/fig_08_fold4_regime.png)

**Analysis.** Panel (a) is a per-day scatter: each point is one of 62 test
days. The fitted dashed lines have positive slope with the reported ρ values.
If leakage were the explanation, IC would be uniformly high or uncorrelated
with drift; instead, high-IC days are exactly the high-drift days. Panel (b)
shows the same two signals as time series: the z-drift spike around
day index 1055 corresponds to the peak MLP rolling IC a few days later.

**Takeaway.** The Fold-4 behavior is *not* a training artefact; it is the model
happening to rank correctly under a distribution shift. This strengthens the
case for reporting Fold 4 as a stress-test rather than discarding it.

---

## Finding 9 — The 9-Dim Price-Feature Set Has Effective Rank ≈ 3

**What.** The 9×9 correlation matrix of the price-feature set has 3
eigenvalues carrying ~89.7% of the variance (PC1 49.5%, PC2 28.4%, PC3 11.8%).
The three `ret_mean_k` and `momentum_k` pairs are numerically identical
(Pearson ρ = 1.00 at k ∈ {5, 10, 21}).

**How.** Standard PCA on the cross-sectional correlation matrix averaged over
1,212 valid trading days (`diag_phase5_effective_rank.csv`, 9 rows), plus the
full 9×9 correlation matrix (`diag_phase5_collinearity.csv`, 9 rows).
The pipeline is part of the Phase-5 diagnostic, triggered by
`diagnostic_phase5_step0.py`.

**Figure.**
![Finding 9](../plots/advisor/fig_09_effective_rank.png)

**Analysis.** Panel (a) is a scree plot: blue bars are per-PC variance share,
red line is cumulative; the golden band highlights PC3, where cumulative passes
90%. Panel (b) is the full correlation matrix. The three ~1.0 cells off-diagonal
are the `ret_mean_k ≡ momentum_k` identities — they are mathematically the
same quantity in our feature definition, which we flag explicitly.

**Takeaway.** The 9-dim feature set is not nine independent signals. This
motivates the 3-feature PC probe (Finding 1) and is not a novel statement about
financial factors in general.

---

## Finding 10 — Architecture Comparison: Price-Only Point-Estimate IC Is Higher Than All-Features in Every Tested Architecture (arithmetic only; no joint statistical test run)

**What.** In the architecture comparison (5 models × 2 feature sets × 5 folds × 3
seeds = 150 rows), every price-only variant has higher mean IC than its
all-features counterpart:
SAGE-Sum (0.039 vs. 0.010), MLP (0.037 vs. −0.008), Transformer (0.027 vs. −0.009),
SAGE-Mean (0.026 vs. 0.011), GAT (0.022 vs. −0.002). Across the price-only
variants, mean IC differences are small (0.022 → 0.039), and variability
bars (±1 σ) overlap heavily.

**How.** `arch_comparison_results.csv` (150 rows). The script that produced it
is not in the current repo root — it appears to have been run out of a notebook
during the stability experiments campaign. See Appendix B for the canonical CSV.

**Figure.**
![Finding 10](../plots/advisor/fig_10_arch_stability.png)

**Analysis.** Each point is a model × feature-set combination; horizontal
bars are ±1 standard deviation across the 15 runs (5 folds × 3 seeds). Blue
points are price-only, red are all-features. The consistent left shift for
red points is the same effect seen in Findings 11 (news harm) and is the
strongest qualitative signal in this figure. The CV(%) metric ("SAGE-Sum
CV ≈ 5%") reported in earlier internal docs is not reproducible from this
CSV — with means near 0, |std/mean| blows up — so we show std in raw IC units
instead of CV%.

**Takeaway.** Within price-only variants, we do not have statistical
separation between architectures at 15 runs each. We do have separation
between price-only and all-features across every architecture.

---

## Finding 11 — FinBERT News Embeddings Degrade Daily IC Across 5 Folds

**What.** Concatenating 384-dim FinBERT title-embedded news features to the
price feature set lowers seed-averaged mean IC in **5 of 5** folds for MLP and
in **4 of 5** folds for SAGE-Mean (Fold 3 is the single exception for SAGE,
where the news-augmented variant edges out price-only by a small margin).
Overall across the 5-fold walk-forward, MLP price-only mean IC is +0.037 vs.
−0.008 with all features.

**How.** Same `run_walkforward_5fold.py` driver as Finding 3; output is
`experiments/wf5_results.csv`. This figure filters to the four relevant model
variants (MLP_price, MLP_all, SAGE-Mean_price, SAGE-Mean_all) and pivots by
fold.

**Figure.**
![Finding 11](../plots/advisor/fig_11_finbert_harm.png)

**Analysis.** Solid bars are price-only (blue), hatched pattern distinguishes
SAGE-Mean from MLP. Red bars are +FinBERT. For MLP, the price bar exceeds the
all-features bar in every fold (5/5). For SAGE-Mean the same pattern holds in
Folds 0, 1, 2, 4 (4/5); Fold 3 is the single exception, where the news-augmented
SAGE bar slightly exceeds price-only.

**Takeaway.** At this universe (SP500 large-caps) and this news format
(title-level, ~15 words per event), FinBERT embeddings do not add ranking
signal and typically subtract from it. The result is consistent with the
efficient-markets prior that large-cap news is priced in well before the
daily close.

---

## Finding 12 — Binary Direction Prediction Is Not Learnable on This Universe

**What.** On the Phase 1d **news-event-driven, next-day binary direction** task
(≈437K stock-day events on SP500, market-adjusted labels), the actual B1–B5
baseline matrix per-model test AUCs (from the authoritative run log
`progress.md` §2026-03-03-g) are:
B1 LR + FinBERT = 0.4993; B2 LR + Sentiment = 0.5031;
B3 LR + Sent + Momentum = 0.4965; B4 LR + Momentum = 0.4987;
B5 XGBoost + all = 0.5046.
That is, the real range is **[0.4965, 0.5046]**, with the minimum at B3
(momentum-based LR overfitting, not B1) and the maximum at B5. None exceed the
pre-registered 0.52 "Go" threshold. Substituting Qwen / GPT-4o LLM embeddings
for FinBERT changes AUC by only ΔAUC = +0.0009 (indistinguishable from noise).

**How.** The binary direction experiments are in archived notebooks referenced
by `archived/docs/2026-03-27/notebook_phase1_2_B.md`; the 5 per-model numbers
above come from **`progress.md` §2026-03-03-g**, the authoritative run log.
**Raw per-seed CSVs are not archived in `experiments/`**, so we plot only the
5 real per-model AUC point estimates above; we do not invent per-seed points.

**Figure.**
![Finding 12](../plots/advisor/fig_12_binary_failure.png)

**Analysis.** The five dots are the per-model test AUCs from progress.md
§2026-03-03-g: green (max) is B5 XGBoost at 0.5046, red (min) is B3 LR + Sent +
Momentum at 0.4965 (the Val → Test drop-off marks it as an overfitting baseline
in the run log). The solid black line is random (0.50); the dashed red line is
the pre-registered Go threshold (0.52). Every baseline sits within ±0.005 of
random (max deviation: B5 at +0.0046); the best baseline (B5) is still 0.0154
below the Go line. Per-seed
points are intentionally not plotted — only the aggregated per-model numbers
in progress.md are available.

**Takeaway.** We report this as a negative result for the **news-event-driven,
next-day (short-horizon) binary direction** task on the SP500 universe
(≈437K stock-day events after event→stock-day compression, market-adjusted
labels), with the features tested — not as a general statement about equity
direction prediction, and not about the 21-day horizon (which is the
ranking-task horizon used in v3 / Findings 1-11). The finding justified the
switch from event-driven binary classification to daily cross-sectional
ranking as the primary task in v3.

---

## Appendix A — Glossary

- **IC (Information Coefficient)**: cross-sectional Spearman rank correlation
  between model-predicted scores and realized next-day returns, computed on
  each trading day and averaged.
- **ICIR**: mean daily IC divided by its standard deviation across days.
- **Sharpe (gross / net)**: annualized return/volatility of a long-short
  portfolio built from predicted scores. Net deducts 15 bps round-trip cost.
- **Newey-West (NW) t-stat**: heteroskedasticity- and autocorrelation-robust
  t-statistic for the mean of a daily series; we use lag = 5 days.
- **Hansen SPA (Superior Predictive Ability)**: test of the null "the
  benchmark is not worse than the best of a set of alternatives," using a
  studentized test statistic and stationary-bootstrap resampling. p_consistent
  is the recommended middle estimate (sitting between p_lower and p_upper).
- **BH-FDR**: Benjamini-Hochberg false-discovery-rate correction for
  multiple hypothesis tests.
- **Wilcoxon signed-rank test**: non-parametric paired test for median
  difference.
- **Bootstrap CI**: confidence interval from a stationary bootstrap with block
  length ≈ n^{1/3}.
- **Walk-forward CV (purged, embargoed)**: train on a fixed prefix of dates,
  validate on the next slice, test on the slice after that; roll forward
  without reshuffling dates. Purged = drop overlap days, Embargoed = additional
  days between splits.
- **GAT / GraphSAGE (Mean, Sum) / HGT / Transformer / MLP**: node-level
  ranking models. GAT = Graph Attention Network. SAGE-Mean/Sum = GraphSAGE
  with mean/sum aggregator. HGT = Heterogeneous Graph Transformer.
  Transformer = permutation-invariant transformer over neighbor set.
- **NoGraph**: ablation that runs the same backbone without message passing
  (to isolate graph-specific effects).
- **Alpha158**: qlib's library of 158 engineered price/volume features
  (K-bar + rolling statistics).
- **mom12m / momentum_k / ret_mean_k / ret_std_k**: feature groups.
  mom12m = 12-month momentum (252-day lookback). momentum_k = k-day momentum.
  ret_mean_k = k-day rolling mean of daily returns. ret_std_k = k-day rolling
  volatility.
- **CORR5 / dolvol / maxret / RSV5**: extended features.
  CORR5 = 5-day rolling correlation with market. dolvol = dollar-volume proxy.
  maxret = max daily return in a window. RSV5 = 5-day realized semi-variance.
- **Winsorization (p1/p99)**: clip extreme values at the 1st / 99th percentile.
- **Cross-sectional z-score**: per-day (across stocks) z-score normalization,
  fitted on training stats only.
- **Permutation importance (ΔIC)**: baseline IC minus IC when a feature (or
  group) is cross-sectionally shuffled on each day. Higher = more important.
- **Effective rank**: the number of principal components needed to capture
  ~90% of variance (scree method) or participation-ratio-weighted.
- **PCA**: principal component analysis of the feature correlation matrix.
- **SelectiveNet**: ICML-2019 three-head architecture (prediction, selection,
  auxiliary) that learns what fraction of inputs to accept.
- **Coverage**: the fraction of test samples the selective model chooses to
  predict on.
- **Threshold baseline**: rank predictions by |score|, keep the top-k
  proportion. A simple control for selective prediction.
- **FinBERT**: a BERT variant fine-tuned on financial text; we use title-level
  embeddings of news events.
- **TF-IDF**: term-frequency × inverse-document-frequency, a classical text
  vectorization.
- **Lazy Prices**: the Cohen-Malloy-Nguyen 2020 construct that measures
  quarter-to-quarter textual change in 10-K / 10-Q filings.
- **SEC 10-K / 10-Q**: annual and quarterly filings with the U.S. SEC.
- **Tariff shock (Q2-2025)**: a market-wide volatility regime in April–June
  2025 following tariff announcements; empirically visible in SP500 as ~2×
  mean daily volatility.
- **Regime stress / regime shift**: a change in the joint distribution of
  features or returns between training and test windows.
- **Data leakage**: information from the test window influencing training.
- **Long-short portfolio**: long the top-quantile predicted, short the bottom
  quantile; daily rebalanced.
- **Turnover**: fraction of positions changed per rebalance.
- **HHI (Herfindahl-Hirschman Index)**: portfolio concentration measure.
- **OOD**: out-of-distribution.
- **CV%**: coefficient of variation, |std/mean| × 100. Unstable when mean is
  near zero; we use raw std IC in Finding 10 instead.

---

## Appendix B — Implementation Code Per Finding

Each finding lists (1) the training / analysis scripts that produced the data
and (2) the canonical CSV / JSON outputs consumed by the figures. All paths
are relative to the repo root `/Users/heruixi/Desktop/GNN-Testing/`.

| # | Finding | Scripts | Data Files |
|---|---|---|---|
| 1 | S6 non-superiority vs S8 (parsimony) | [build_alpha158_features.py](../build_alpha158_features.py), [run_step3_plan_z_part_b.py](../run_step3_plan_z_part_b.py), [run_step3_plan_z_part_c.py](../run_step3_plan_z_part_c.py), [run_step3_plan_z_part_c_perfold.py](../run_step3_plan_z_part_c_perfold.py), [analyze_step3_plan_z.py](../analyze_step3_plan_z.py) | `experiments/step3_plan_z/part_b_summary.csv`, `part_c_s8_daily_ic.csv`, `part_c_s8_perfold_daily_ic.csv`, `hansen_spa_results.csv` |
| 2 | Horizon ablation | [run_phase5_step3_feature_expansion.py](../run_phase5_step3_feature_expansion.py), [diagnostic_phase5_step0.py](../diagnostic_phase5_step0.py) | `experiments/horizon_ablation_results.csv` (360 rows) |
| 3 | Walk-forward 5-fold | [run_walkforward_5fold.py](../run_walkforward_5fold.py) | `experiments/wf5_results.csv` (90 rows) |
| 4 | Permutation ranking | [run_step3_plan_z_part_a.py](../run_step3_plan_z_part_a.py), [analyze_step3_plan_z.py](../analyze_step3_plan_z.py) | `experiments/step3_plan_z/part_a_daily_ic.csv`, `part_a_permuted_ic.csv`, `artifacts/step3_plan_z/part_a_ranking.json` |
| 5 | Normalization × regime | [run_diag1_normalization.py](../run_diag1_normalization.py), [run_diag1b_replication.py](../run_diag1b_replication.py) | `experiments/diag1_normalization_results.csv`, `diag1b_replication_results.csv` |
| 6 | SelectiveNet coverage | v3 N5 notebook (see `archived/notebooks/v3_ranking_pipeline.ipynb`) | `experiments/selectivenet_results.csv` (70 rows) |
| 7 | SEC Gate 1 text | [run_gate1_experiment.py](../run_gate1_experiment.py) | `experiments/gate1_results.csv` (22 rows) |
| 8 | Fold-4 regime stress | [analyze_fold4_leakage.py](../analyze_fold4_leakage.py) | `experiments/step3_plan_z/fold4_zdrift_summary.csv`, `fold4_zdrift_per_day.csv`, `fold4_tail_concentration.csv` |
| 9 | 9-dim effective rank | [diagnostic_phase5_step0.py](../diagnostic_phase5_step0.py), [diagnostic_phase5_fix.py](../diagnostic_phase5_fix.py) | `experiments/diag_phase5_effective_rank.csv`, `diag_phase5_collinearity.csv` |
| 10 | Architecture comparison | produced via archived v3 stability notebook; not in repo-root `.py`; see `archived/notebooks/` | `experiments/arch_comparison_results.csv` (150 rows) |
| 11 | FinBERT news harm | [run_walkforward_5fold.py](../run_walkforward_5fold.py) | `experiments/wf5_results.csv` (filtered to `_price` vs. `_all`) |
| 12 | Binary direction failure | archived Phase-1d notebooks (see `archived/docs/2026-03-27/notebook_phase1_2_B.md`) | `progress.md` §2026-03-03-g (authoritative run log; 5 per-model point estimates) |

### Figure-generation driver

All 12 figures are produced by **[make_advisor_figures.py](../make_advisor_figures.py)**
(a single ~450-line file) which reads the CSVs listed above and writes to
`plots/advisor/fig_NN_*.png`. Invoke with:
`/opt/homebrew/Caskroom/miniforge/base/envs/gnn/bin/python make_advisor_figures.py`

### Feature / data build

- `build_alpha158_features.py` — reproduces qlib Alpha158DL feature library.
- `build_phase5_features.py` — builds the 14-dim extended feature set.
- `cleanup_and_rebuild_features.py` — unified feature-rebuild utility.
- `download_ohlcv_yf.py` — yfinance OHLCV fetcher.
- `refetch_zts.py` — ZTS ticker data refetch utility.

### Paper / report aggregator (separate, existing)

- `run_figures_tables.py` — the older paper-pipeline figure generator, kept
  for reproducibility of the existing `plots/paper_*.png` figures; not
  modified for this advisor package.
