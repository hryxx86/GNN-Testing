---
reviewer: codex
touchpoint: plan
round: A
date: 2026-04-29
target_plan: /Users/heruixi/.claude/plans/seed-level-diagnostic-2026-04-29.md
overall_verdict: BLOCK-EXECUTION
summary_3line: |
  Phase 2's permutation test is invalid as written because permuting seed labels within a config leaves the best-vs-median order statistic unchanged.
  Test C reverses the expected-max correction: a gap smaller than the null expected maximum is evidence for sampling luck, not a real best seed.
  Reporting guardrails still allow post-hoc seed results into paper supplement without enough controls against narrative leakage.
---

## Findings

## Finding 1: Best-vs-median permutation is non-identifying
- severity: CRITICAL
- category: statistics
- claim: Test A does not generate a valid null distribution as written. If `best_seed` and `median_seed` are recomputed after permuting labels within each config, the set of IC values and therefore the best-vs-median gap are unchanged; if the plan intended fixed seed identities, the statistic and blocking scheme are not specified.
- evidence: Phase 2 defines Test A as `IC(best_seed) - IC(median_seed)` per config and specifies "10,000 permutations of seed labels within each config; recompute statistic" at lines 57-61.
- suggested_fix: Replace Test A with a seed-identity global statistic aligned to Phase 1 Q1/Q4, such as max seed mean IC across config/fold blocks, max seed win count across configs, or mean rank by seed; permute seed labels within each config/fold block and use the max statistic across seeds/configs for the p-value.
- status: OPEN

## Finding 2: Selection-corrected gap decision is reversed
- severity: CRITICAL
- category: correctness
- claim: Test C inverts the selection-bias correction. An actual best-vs-median gap that is much smaller than the null expected maximum is evidence that the observed best seed is explainable by selection/sampling luck, not that the best seed is real.
- evidence: Phase 2 Test C computes `E[max(IC_seed)] - median(IC_seed)` under a null at lines 68-70, but line 71 states, "If actual gap << expected-under-null: best seed is real; otherwise just sampling luck." The plan's own review question frames this as an expected-max versus BH-FDR/Tukey choice at line 152.
- suggested_fix: Use the expected-max/null resampling as a one-sided max-statistic correction: report the observed gap, null mean/quantiles, and `Pr(null max gap >= observed gap)`. Treat BH-FDR or Tukey HSD only as optional post-hoc pairwise seed comparisons after a valid global seed-effect test, not as the primary correction for selecting the winner after looking.
- status: OPEN

## Finding 3: Non-significance is over-interpreted as exchangeability
- severity: MAJOR
- category: statistics
- claim: The decision rule converts failure to reject into a positive conclusion that all seeds are exchangeable and that seed effects are pure noise. With 10 seeds and 14 configs, this is too strong and risks reporting an equivalence claim not supported by the proposed tests.
- evidence: Lines 75-77 say that if Test A and Test B are non-significant, the report should state "no significant seed effect, all 10 seeds are exchangeable; observed best is sampling luck" and stop. Lines 63-66 similarly say a non-significant Friedman test means "seed effects are pure noise across configs."
- suggested_fix: Change the null-result language to "no statistically detected stable seed effect under the planned diagnostics; the observed best seed is consistent with selection/sampling luck." Report Phase 1 effect sizes and uncertainty descriptively, but do not claim proven exchangeability or pure noise.
- status: OPEN

## Finding 4: Paper-supplement allowance is not fenced tightly enough
- severity: MAJOR
- category: reporting
- claim: The Use Case E guardrail is directionally correct, but it does not fully prevent post-hoc seed findings from leaking into paper claims. The plan allows paper supplementary reporting and only requires general disclosures/citations, which controls numeric provenance but not narrative use of a high-IC seed as evidence.
- evidence: The constraint says seed-level findings cannot enter the paper primary verdict but "may enter paper supplementary" at lines 14-18, and the Phase 2 null branch explicitly says to report to "paper supplementary" at line 75. Required disclosures are listed at lines 121-127, while out-of-scope rules at lines 139-143 prohibit changing the Stage 1 verdict or claiming a magic seed without forward-validation, but do not explicitly fence main-text/abstract/discussion language.
- suggested_fix: Add a reporting rule that any seed-level material may appear only as a clearly labeled exploratory sensitivity/diagnostic supplement, with no change to the abstract, main-text conclusions, model ranking, Stage 1 Scenario B verdict, or production/deployment claims. Put the exploratory disclosure on every seed table/figure, not only in surrounding prose.
- status: OPEN

## Finding 5: Phase 3 remains a selected-extremes fishing expedition
- severity: MAJOR
- category: methodology
- claim: Phase 3 is conditional on significance and labeled hypothesis generation, which helps, but the proposed best-N versus worst-N rerun still selects seeds and likely configs after seeing the outcome. That can produce mechanism stories from post-selection noise, especially with only 2-3 selected seeds per side.
- evidence: Phase 3 runs only after Phase 2 significance at lines 79-81, then examines "best-N vs worst-N seeds (N=2 or 3 each)" at lines 83-89. Because internals were not saved, the plan proposes rerunning "best 2 + worst 2 seeds x 1-2 configs x 5 folds" at lines 90-92, while the small-sample caveat at lines 94-95 says findings are hypothesis generation.
- suggested_fix: Keep Phase 3 only if Phase 2 identifies a stable seed-identity effect, and freeze the seed selection rule, config(s), fold set, and internal metrics before rerunning. If the Phase 2 signal is config-local or fold-4-only, skip Phase 3 or report that no stable seed-level mechanism was sought.
- status: OPEN

## Finding 6: Forward-validation success criterion overclaims
- severity: CONCERN
- category: reporting
- claim: The Phase 4 default skip is appropriate for Use Case E, but the success language overstates what a small fresh-data check can establish. This is an inference from the plan: because candidate seeds are selected post hoc and fresh data may represent one market regime, "matches within 1 sd" is not enough to conclude the seed effect is real and predictive.
- evidence: Phase 4 is optional and conditional on H博士 sign-off at lines 97-100, compares fresh-data IC to Stage 1 IC at lines 101-105, and says "If fresh-data IC matches Stage 1 IC (within 1 sd): seed effect is real and predictive" at line 107. Line 112 correctly states that the default is to skip because Use Case E only requires Phases 1-2.
- suggested_fix: Retain the default skip unless H博士 requests deployment-grade evidence. If Phase 4 is run, label it as a forward-validation sanity check and report regression/no-regression descriptively; do not use "real and predictive" language without a pre-specified holdout decision rule and uncertainty statement.
- status: OPEN

## Overall Assessment

The plan should not execute as written because the Phase 2 inferential core has two blocking errors. Test A's permutation scheme does not change the best-vs-median order statistic under the stated per-config recomputation, and Test C reverses the expected-max interpretation. These issues directly affect review questions 1 and 5 and would make any "particular seeds are notably higher IC" answer statistically unreliable.

The Use Case E constraint is present and substantively important, but the reporting controls need sharper boundaries. The plan correctly says the Stage 1 preregistered 10-seed mean verdict cannot change, but allowing paper-supplement seed results without explicit main-text and narrative fences leaves a p-hacking channel open. Numeric CSV citations are necessary for provenance; they are not sufficient to prevent cherry-picked seed interpretation.

Phase 3 is only defensible as tightly labeled hypothesis generation after a valid global seed-identity signal. As drafted, rerunning selected extremes on 1-2 configs risks creating mechanistic stories from the same post-hoc selection process. Phase 4's default skip is cost-benefit appropriate for Use Case E, but if it is run, the plan should avoid treating one fresh-data comparison as proof that a seed effect is real and predictive.
