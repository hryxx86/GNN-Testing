# Discussion A: Data Length, Regimes, and Supplementary Experiments

This document is written for the current cross-sectional stock prediction paper, not as generic ML guidance. The fixed facts are: 500 U.S. stocks, daily OHLCV plus Alpha158-style features, 2020-01 to 2025-06, 21-day forward percentage-return labels, cross-sectional z-scored targets, and five expanding walk-forward folds whose final test fold is 2025-Q2. The locked Stage 1 result is a preregistered 600-cell horse race with 0/8 co-primary rejections for ListMLE or pairwise loss against MSE. That verdict cannot be amended. Any new work below is supplementary only.

The most important local empirical fact is not "MSE is strong." MSE is weak but stable: current mean IC is roughly +0.013 to +0.020, versus the project-provided CIKM 2025 comparator of +0.075 on a different 1-day horizon and 110 top-cap universe. The important fact is that ListMLE has a systematic fold-4 collapse: 6/6 architecture-feature combinations have 2025-Q2 fold-4 IC between -0.28 and -0.36, with small seed dispersion. Fold 4 is therefore a regime-stress diagnostic, not just another noisy quarter.

## Q1: Should We Extend the Dataset from 5y to 10y or 20y?

My answer: test 10y only as a narrow supplementary ablation; do not spend deadline time building a 20y main dataset. A 10y extension is worth one focused experiment because it directly tests whether the fold-4 ListMLE collapse is data-poverty. A 20y extension is more likely to create a second paper's worth of data-quality and stationarity problems than to rescue this one.

The case for more data is real but easy to overstate. With 500 stocks and daily panels, the row count looks large, but the effective number of independent time regimes is small. A 21-day forward label creates heavy overlap across adjacent days, and the number of independent market episodes in 2020-2025 is closer to dozens than to 1.6M. Moving from 5y to 10y adds calendar regimes: late zero-rate bull market, 2018 volatility spike, 2019 easing, COVID, 2022 hikes, 2023 AI concentration, and 2025-Q2. That can stabilize simple MSE estimation and reduce seed variance.

But the best recent paper directly on this issue argues against the naive "longer is better" view. [Capponi et al. 2025, arXiv](https://arxiv.org/abs/2512.23596) formalize a nonstationarity-complexity tradeoff: prediction error decomposes into misspecification, estimation uncertainty, and nonstationarity, so longer windows reduce variance but add stale-regime bias. Their empirical setting is industry-portfolio return prediction rather than cross-sectional single-stock IC, but the lesson transfers cleanly: model class and training-window length must be selected jointly. A 20y window is not "more evidence" if the learner treats pre-GFC, QE, COVID, and rate-hike markets as exchangeable samples from one function.

The empirical asset-pricing literature does use long samples successfully, but usually for monthly characteristics, broad panels, and explicitly regularized models. [Gu, Kelly, and Xiu 2020, RFS](https://academic.oup.com/rfs/article/33/5/2223/5758276) use 1957-2016 U.S. equities and show that trees and neural nets improve stock-level monthly out-of-sample R2, with dominant predictors including momentum, liquidity, and volatility. That is evidence that long panels can help when the target is monthly risk premia and the design is aggressively regularized. It is not evidence that a daily ListMLE surrogate trained on technical Alpha158 features should pool 2005 with 2025. [Pesaran and Timmermann 2007, Journal of Econometrics](https://ideas.repec.org/a/eee/econom/v137y2007i1p134-161.html) is closer in spirit: under breaks, estimation-window selection is itself part of the forecasting problem.

For cross-sectional IC specifically, I am not aware of a literature result that says "10y is optimal" or "20y is optimal." What exists is indirect. [Green, Hand, and Zhang 2017, RFS](https://ideas.repec.org/a/oup/rfinst/v30y2017i12p4389-4436..html) find that U.S. characteristic predictability fell sharply after 2003, with only two independent determinants in non-microcaps after that point and insignificant hedge returns outside microcaps. [McLean and Pontiff 2016, Journal of Finance](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2156623) estimate that returns to published predictors are 26% lower out-of-sample and 58% lower post-publication. Those are not training-window papers, but they are strong warnings against treating older anomaly behavior as stationary signal.

Given that fold 4 is 2025-Q2, 10y data may help MSE a little and may not help ListMLE at all. The collapse looks like a directional rank inversion under a specific regime, not high variance from too few rows. If the ListMLE softmax ranking likelihood has learned "recent winners remain top-ranked" from 2020-2024 and 2025-Q2 rewards the opposite, adding 2015-2019 can either dilute the recent pattern or strengthen an even older pattern. There is no reason to assume the added years contain the relevant analogue. A 2025-Q2 tariff/rate/concentration unwind is not well represented by a pre-COVID zero-rate market. If the mechanism is regime-mismatch, indiscriminate lengthening is the wrong fix.

Opinionated recommendation: do a 3y vs 5y vs 10y window ablation on MLP-S8 only, with MSE and ListMLE, fixed 2024-Q2 to 2025-Q2 test folds, and three seeds. If 10y does not materially reduce ListMLE fold-4 collapse, stop. Do not build 20y before submission.

Publishable supplementary finding: "Longer history did not rescue likelihood-ranking collapse; fold-4 IC remained strongly negative under matched forward folds." Inconclusive noise: "10y mean IC is +0.006 higher over three seeds but fold-4 remains negative and cluster bootstrap CI crosses zero."

## Q2: Should We Adopt Regime-Conditioned Mixture-of-Experts?

My answer: not for the paper deadline. A full MoE is a plausible next project, but it is a bad use of the remaining 30h M4 budget. The paper's contribution is currently sharper as a preregistered null plus a mechanistic ListMLE regime-failure diagnostic. A rushed MoE risks looking like post-hoc rescue modeling.

The MoE literature is suggestive but not directly transferable. [Yu et al. 2024, arXiv](https://arxiv.org/abs/2410.02241) propose MIGA and report that MIGA-Conv reaches 24% excess annual return on CSI300, 8 percentage points above their previous SOTA benchmark. That is a Chinese-index benchmark, not U.S. 500-stock 21-day rank IC, and it is an arXiv preprint rather than a settled empirical finance result. [Liu et al. 2025, WWW Companion](https://www.sigweb.sigweb.hosting.acm.org/toc/www25b.html) propose MERA, a retrieval-augmented MoE with GateNet for diversified stock patterns; the paper claims significant improvements and releases code, but the WWW Companion abstract does not give enough detail to treat the effect size as a reliable expected IC gain here. [Vallarino 2025, arXiv](https://arxiv.org/abs/2508.02686) reports up to 33% MSE improvement for volatile assets and 28% for stable assets using a volatility-aware RNN-plus-linear MoE, but that is a 30-stock U.S. price-forecasting preprint, not a cross-sectional portfolio IC paper.

Your sample size is simultaneously large and small. At the row level, 500 stocks x about 5 years is roughly 1.6M stock-day observations. With 3-5 experts, a balanced assignment gives perhaps 300K-500K rows per expert, enough for a shallow MLP. But the relevant independent unit is the date/regime panel, not the stock row. A high-volatility expert might get 5-15% of days, which is only about 60-190 daily panels in the current sample. For SAGE experts, each expert also needs graph construction, edge stability, and enough regime-specific dates for early stopping. That is thin.

The gating problem is the core risk. A hard regime classifier can multiply errors: wrong gate plus expert trained on the wrong conditional distribution. Soft gating is more robust, but then the implementation becomes a real MoE training problem with load balancing, leakage control, and expert collapse diagnostics. A noisy gate can still help if the regimes are coarse, pre-specified, and strongly tied to the failure mechanism. But "VIX high/low" is not enough; the model must show that fold-4-like rank inversions are allocated differently before the return labels arrive.

The realistic development cost is about one month for a correct implementation with forward validation. The minimum correct version needs: lagged regime features, a gate trained only on train/validation data, no use of future realized returns in labels or routing, regime-specific early stopping, matched seeds, fold-level clustered inference, and an ablation against a single global model with the same parameter budget. Anything less will be easy for a reviewer to dismiss as post-hoc overfit.

If you still want a minimum valid MoE-style supplement, do not implement full MIGA/MERA. Implement a regime-conditioned shallow ensemble:

- Use only MLP-S8 and MSE first.
- Define regimes from lagged observables: realized cross-sectional dispersion, VIX level/change if available, and market 21-day drawdown.
- Train two experts: normal and stress. Use soft weights from a logistic gate trained on train/val only, or a pre-specified hard threshold learned on train/val and frozen for test.
- Compare against a single MLP with the same hidden size and against an ensemble with random regime labels.
- Evaluate on the existing five forward folds, not an in-sample split.

Statistical validity requires paired daily IC differences by fold and seed, fold-cluster bootstrap or block bootstrap, and explicit fold-4 reporting. A claim is publishable only if regime conditioning improves fold-4 ListMLE or MSE behavior without degrading folds 0-3, and the effect survives a random-gate placebo. A within-sample regime fit is not evidence.

## Q3: What Macro Indicators Should Drive Regime Classification?

For a 21-day cross-sectional rank-IC problem, the best regime variables are not deep macro variables. They are market-state variables that proxy near-term breadth, dispersion, and stress. My preferred minimal set is:

1. Lagged 21-day realized cross-sectional return dispersion.
2. Lagged 21-day market return or drawdown.
3. Lagged VIX level and 5/21-day VIX change, if available with clean timestamps.
4. Lagged market realized volatility over 21 or 63 days.
5. 10Y-2Y yield spread only as a slow background variable, not as a primary gate.

The strongest evidence for cross-sectional rank behavior points to dispersion and volatility, not the yield curve. [Stivers and Sun 2010, JFQA](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/crosssectional-return-dispersion-and-time-variation-in-value-and-momentum-premiums/77E2E3B09BDA5992C29BBCE2CEDC08FE) find that recent cross-sectional return dispersion is positively related to subsequent value premia and negatively related to subsequent momentum premia, controlling for macro state variables. That is highly relevant because Alpha158 contains many price-trend and volatility-style features; a dispersion regime can literally flip which technical ranks are rewarded. [Gorman, Sapra, and Weigand 2010, Journal of Investing](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1444868) report that cross-sectional dispersion and VIX forecast alpha dispersion, though they also warn that higher active risk can offset alpha in information-ratio terms. That maps directly to IC versus Sharpe tension.

VIX is useful but should be treated as a near-term stress and expected-volatility state, not a clean leading predictor of cross-sectional rank IC. It is derived from S&P 500 options and targets roughly 30-day expected volatility, so its horizon is close to a 21-trading-day label. But VIX often spikes contemporaneously with market drawdowns. For regime classification, its level and change are plausible, while a claim that VIX "predicts" ListMLE collapse should be made cautiously. Use VIX as a conditioning variable observed at t-1, not as an ex post explanation.

The yield curve is slower and more dangerous. [Ferson and Harvey 1999, Journal of Finance](https://www.nber.org/papers/w7009) show that lagged conditioning variables have cross-sectional explanatory power for stock portfolio returns. But for a 21-day daily-stock IC problem, the 10Y-2Y spread will mostly identify macro cycle background, not the abrupt fold-4 rank inversion. [Welch and Goyal 2008, RFS](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1211941) also show that many aggregate equity-premium predictors are unstable out of sample. The yield curve belongs in a robustness table, not as the first gate.

The look-ahead risk in yield-curve labels is not the yield observation itself if you use a market-observed t-1 value. The look-ahead risk comes from ex post regime definitions: NBER recession labels, future 21-day realized volatility, final revised macro series, or thresholds chosen after seeing fold-4 performance. If using FRED or Treasury constant-maturity data, lag by at least one trading day and document timestamp availability. If using recession labels, call them ex post diagnostics, not tradable regimes.

Best practical regime definition for this project: a two-dimensional stress label based on lagged realized cross-sectional dispersion and lagged market drawdown, with VIX as a sensitivity if data are already clean. This is directly computable from the same stock panel, has no external data latency, and is closest to the ListMLE failure mechanism.

## Q4: How Stationary Is Cross-Sectional Rank IC Over Multi-Decade Windows?

Not stationary enough to justify blind pooling. Some factors are robust across markets and decades, but their realized IC is episodic, crowded, and regime-dependent. A deep daily model trained on Alpha158-style features is more fragile than the canonical monthly factor evidence.

Momentum is the most defensible feature family, but even momentum has horizon structure and reversals. [Jegadeesh and Titman 2001, Journal of Finance](https://www.nber.org/papers/w7159) show that momentum profits continued in the 1990s, reducing the data-snooping concern, but they also find significant reversals 4-5 years after formation. [Asness, Moskowitz, and Pedersen 2013, Journal of Finance](https://www.aqr.com/insights/research/journal-article/value-and-momentum-everywhere) find value and momentum premia across eight markets and asset classes, with negative correlation between value and momentum. That supports including trend and value-like signals, but not assuming a constant daily mapping from 5/10/21-day technical features to next-month ranks.

Value is slower and more regime-cyclical. It can be structurally useful over long horizons, but it has long drawdowns and is sensitive to sector composition, rates, and accounting definitions. This project is not using rich accounting value features as the main signal; it is using daily OHLCV plus Alpha158. Extending to 20y will add value-cycle history only indirectly, while adding major microstructure and monetary-policy shifts.

Volatility and low-risk signals are especially conditional. [Ang, Hodrick, Xing, and Zhang 2006, Journal of Finance](https://www.nber.org/papers/w10852) document that high idiosyncratic-volatility stocks earn abnormally low average returns in their sample, while [Frazzini and Pedersen 2014, Journal of Financial Economics](https://www.sciencedirect.com/science/article/pii/S0304405X13002675) explain betting-against-beta through leverage constraints. These are robust enough to motivate volatility features, but their sign and payoff can vary in crisis rebounds and speculative episodes. In a 21-day z-scored return task, volatility can be a risk proxy, a reversal proxy, or a distress lottery proxy depending on regime.

There is no credible single "half-life of alpha decay" for momentum, value, and volatility that can be plugged into this project. The honest answer is:

- Momentum signal horizon: 3-12 month continuation is canonical; short-term reversal and 4-5 year reversal are documented in [Jegadeesh and Titman 2001, Journal of Finance](https://www.nber.org/papers/w7159). For 5/10/21-day technical momentum, the half-life is much shorter and more microstructure-dependent; I am not aware of a direct multi-decade half-life estimate for Alpha158-style daily features.
- Value signal horizon: multi-year and slow-moving; useful for long samples but not a direct stabilizer of 21-day technical IC.
- Volatility/low-risk horizon: conditional and regime-sensitive; evidence supports a volatility relation, not a stable daily IC half-life.
- Broad anomaly decay: [McLean and Pontiff 2016, Journal of Finance](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2156623) give the cleanest quantitative decay estimate: 26% lower out-of-sample returns and 58% lower post-publication returns for published predictors. That is a decay estimate, not a physical half-life.

Pre-2015 data are informationally useful only if treated as a regularizer or as a candidate window in a forward-selected design. They are actively harmful if pooled unconditionally into a high-capacity model that assumes the same feature-return map. Pre-2010 data add the GFC, but also pre-modern ETF, pre-zero-commission, different option-market structure, different Fed reaction function, and different factor crowding. For a top-500 U.S. daily technical model, I would distrust pre-2010 more than I would value the extra rows.

The structural breaks matter even though the labels are cross-sectionally z-scored. Z-scoring removes the market-level return drift each day; it does not remove a change in which stocks win. 2008, 2020, 2022, and 2025-Q2 can all preserve a zero-mean daily cross-section while flipping the rank relation between momentum, volatility, size, liquidity, and next-month return. That is exactly the failure mode ListMLE appears to expose.

For Alpha158 specifically, the literature is weaker than people often imply. Alpha158 is a useful standardized technical feature set popularized in open-source quant ML workflows such as Qlib, but I am not aware of a peer-reviewed multi-decade U.S. stationarity study showing that Alpha158-style daily features have stable cross-sectional rank IC across GFC, QE, COVID, and post-2022 rates. The closest broad evidence is that [Gu, Kelly, and Xiu 2020, RFS](https://academic.oup.com/rfs/article/33/5/2223/5758276) find momentum, liquidity, and volatility among dominant predictor families; that supports the feature families, not the stationarity of this specific daily representation.

## Q5: Practical Prioritized Recommendation

The minimum viable experiment should not try to "beat MSE" in the main paper. It should answer two supplementary questions:

1. Does longer training history reduce the ListMLE fold-4 collapse?
2. Is the collapse concentrated in pre-specified lagged stress/dispersion regimes?

The right design is small, matched, and forward-only. Use MLP-S8 as the main probe because S8 is the Alpha158 feature set and MLP is cheaper than SAGE. Keep folds 0-4 exactly fixed: tests ending 2024-Q2, 2024-Q3, 2024-Q4, 2025-Q1, and 2025-Q2. Use the same 21-day horizon and embargo logic. Use three seeds for new supplementary runs, and compare against the existing 10-seed Stage 1 baseline only as context; where possible, rerun the sampled seeds to make paired comparisons clean.

The data-window experiment should be:

- Windows: 3y rolling/expanding cap, current 5y, and 10y if the data can be added cleanly.
- Models/losses: MLP-S8 with MSE and ListMLE.
- Cells: 3 windows x 2 losses x 5 folds x 3 seeds = 90 cells.
- Primary supplementary metrics: mean daily rank IC, fold-4 IC, fold standard deviation, and ListMLE-vs-MSE delta IC.
- Test: paired daily IC differences, clustered by fold and seed; report fold-cluster bootstrap because only five folds exist.
- Stopping criterion: if 10y ListMLE fold-4 remains below -0.15 IC, stop all further data-extension work.

The regime experiment should be:

- First pass: no retraining. Stratify existing predictions by pre-specified lagged stress labels: top tercile of 21-day cross-sectional return dispersion, top tercile of 21-day market realized volatility, and negative 21-day market return. Use VIX only if timestamp-clean data are already available.
- Second pass if first pass is strong: train a two-regime MLP-S8 MSE model using stress/normal labels from train/val only; do not implement full MoE.
- Metrics: IC by regime, fold-by-regime heatmap, and whether fold-4 ListMLE collapse is explained by stress-regime days or by a broader quarter-level shift.
- Test: permutation or bootstrap over dates within folds, plus a random-regime placebo.
- Stopping criterion: if regime labels do not isolate at least 50% of the fold-4 IC degradation or fail placebo, do not train a regime-conditioned model.

Publishable supplementary finding: "The ListMLE collapse is not resolved by longer windows and is concentrated in lagged high-dispersion/stress states, supporting a regime-conditioned failure interpretation." Also publishable: "A 10y window materially reduces fold-4 ListMLE collapse without improving average MSE, implying a data-window interaction specific to likelihood ranking." Inconclusive: any mean IC improvement below +0.005 with fold-cluster CI crossing zero, or a regime result that only appears after thresholds are chosen using fold 4.

## Prioritized Action List

### Tier 1

1. Window-length ablation: 3y vs 5y vs 10y on MLP-S8 with MSE and ListMLE.
   - Estimated dev hours: 4-8h if 10y Alpha158/OHLCV data are already accessible; 12-18h if data ingestion must be patched.
   - Estimated compute: 3-8h M4 for 90 MLP cells, depending on 10y preprocessing and training length.
   - Expected IC gain or null probability: expected MSE gain +0.000 to +0.005 IC; estimated 70% probability of null for mean IC. For ListMLE fold-4, estimated 60% probability that collapse remains below -0.15 IC.
   - Concrete design: fixed folds 0-4, three matched seeds, MLP-S8 only, metrics = mean daily rank IC, fold-4 IC, fold-sigma, and ListMLE-vs-MSE delta IC.
   - Minimum statistical validity requirement: fixed forward folds, three matched seeds, paired daily IC deltas, fold-cluster bootstrap, and explicit fold-4 table. No post-hoc fold exclusion.
   - Stopping criterion: stop data-extension work if 10y ListMLE fold-4 IC remains below -0.15 or if the 10y mean IC gain is below +0.005 with bootstrap CI crossing zero.
   - Publishable if: 10y changes fold-4 ListMLE IC by at least +0.10 while not reducing mean IC, or if it cleanly fails and supports the nonstationarity argument.

2. Regime-stratified forensic analysis of existing predictions.
   - Estimated dev hours: 3-5h using panel-derived dispersion/volatility/drawdown; 5-8h if VIX/yield data are added with timestamp checks.
   - Expected IC gain or null probability: no model IC gain because this is diagnostic; estimated 50-60% probability it explains a meaningful portion of fold-4 degradation because the collapse is large and systematic.
   - Concrete design: fixed folds 0-4, no retraining, labels = lagged dispersion/volatility/drawdown terciles, metrics = regime-wise IC, fold-by-regime IC heatmap, and fold-4 degradation share.
   - Minimum statistical validity requirement: thresholds pre-specified before viewing results, labels lagged by one day, fold-by-regime IC table, random-label placebo, and bootstrap CIs clustered by fold/date.
   - Stopping criterion: stop regime-model work if no pre-specified regime explains at least 50% of fold-4 degradation or if random-label placebo performs similarly.
   - Publishable if: high-dispersion or drawdown regimes account for at least 50% of the ListMLE fold-4 degradation and the same rule has directionally consistent effects across folds.

3. Two-regime MLP-S8 MSE stress/normal model, only if Tier 1 item 2 is strong.
   - Estimated dev hours: 8-12h.
   - Expected IC gain or null probability: estimated +0.002 to +0.008 IC if regime labels are meaningful; 65% probability of null or overfit given only five years.
   - Concrete design: fixed folds 0-4, three matched seeds, two experts or regime-specific heads trained on train/val labels only, metrics = mean IC, fold-4 IC, fold 0-3 average IC, and random-regime placebo delta.
   - Minimum statistical validity requirement: gate/threshold learned only on train/val, frozen on test, same folds and seeds as the window ablation, comparison to global MSE and random-regime placebo.
   - Stopping criterion: stop if fold-4 IC improves by less than +0.05, if folds 0-3 lose more than -0.005 mean IC, or if the random-regime placebo matches the effect.
   - Publishable if: fold-4 IC improves by at least +0.05 and average IC does not fall on folds 0-3, with bootstrap CI excluding a zero or negative fold-4 effect at least at a clearly labeled exploratory p < 0.10.

### Tier 2

1. Regime-augmented features rather than MoE: append lagged dispersion, market drawdown, realized volatility, and optionally VIX to each stock-day feature vector.
   - Estimated dev hours: 6-10h.
   - Expected IC gain or null probability: estimated +0.001 to +0.006 IC; 70% probability of null because MLP may ignore weak global features.
   - Minimum statistical validity requirement: all regime features lagged, no future realized-vol labels, matched folds/seeds, and ablation against stock-only S8.

2. 10y MSE-only robustness across MLP and SAGE.
   - Estimated dev hours: 8-14h after data extension exists.
   - Expected IC gain or null probability: estimated +0.000 to +0.005 IC; 75% probability of null for SAGE because graph construction may absorb little from old data.
   - Minimum statistical validity requirement: same five folds, at least three seeds, separate report for MLP and SAGE, and no change to Stage 1 primary verdict.

3. ListMLE temperature or loss-stabilization sensitivity.
   - Estimated dev hours: 8-16h.
   - Expected IC gain or null probability: possible fold-4 gain +0.05 to +0.15 if collapse is softmax overconfidence; 70% probability of no robust mean gain.
   - Minimum statistical validity requirement: pre-specify one or two temperatures only, run all five folds, and compare to original ListMLE and MSE without selecting on fold 4.

### Tier 3

1. Full 3-5 expert MoE with learned gate and SAGE/MLP experts.
   - Estimated dev hours: 40-80h for a defensible version.
   - Expected IC gain or null probability: upside maybe +0.005 to +0.015 IC, but estimated 75% probability of inconclusive or overfit under this sample and deadline.
   - Minimum statistical validity requirement: forward-only gate training, load-balancing diagnostics, random-gate placebo, matched parameter-budget baseline, all five folds, and fold-cluster inference. Skip for this submission.

2. 20y data extension.
   - Estimated dev hours: 30-60h including data QA, survivorship checks, feature recomputation, and split validation.
   - Expected IC gain or null probability: estimated 80% probability of null or harmful shift for daily Alpha158-style models; possible value only for a separate stationarity paper.
   - Minimum statistical validity requirement: survivor-bias audit, pre-2010/pre-2015 split diagnostics, rolling-window selection, and a paper section on structural breaks. Skip now.

3. Reframing the paper around a new regime-conditioned method.
   - Estimated dev hours: more than 80h including experiments and writing.
   - Expected IC gain or null probability: not estimable from current evidence; high risk of post-hoc narrative contamination.
   - Minimum statistical validity requirement: new preregistration or clearly separated exploratory study. Skip for the locked Stage 1 paper.
