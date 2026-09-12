---
reviewer: codex
touchpoint: plan
round: B
date: 2026-04-29
target_plan: /Users/heruixi/.claude/plans/plan-zpp-unified-2026-04-29.md
source_discussions:
  - 2026-04-29_codex_discussion_A_data_length_regime.md
  - 2026-04-29_codex_discussion_B_split_methodology.md
  - 2026-04-29_codex_discussion_C_loss_noise.md
findings:
  - id: B-01
    severity: MAJOR
    category: stop_criteria
    claim: >-
      The Tier 1.C anchored RankNet gate is weaker than the plan's mandatory ranking-loss fold-4 viability gate.
    evidence: >-
      Plan "Hard Constraints" lines 19-22 and "Statistical guards against p-hacking" lines 419-422 require fold-4 IC > -0.15, sigma_fold <= 2 x sigma_fold(MSE), and median(pred_cs_std) >= 0.05 for any new ranking loss, but Tier 1.C lines 214-216 only requires fold-4 IC not below MSE by 0.05 and median(pred_cs_std) >= 0.05.
    suggested_fix: >-
      Replace the Tier 1.C gate with the global three-part gate and add the relative-to-MSE condition only as an additional fourth condition.
    status: OPEN
  - id: B-02
    severity: MAJOR
    category: statistical_rigor
    claim: >-
      The Newey-West/block-bootstrap specification does not state how seed-level paired daily differences are aggregated, creating a pseudo-replication risk.
    evidence: >-
      Plan Tier 1.A lines 117-120 and Reporting standards lines 394-413 describe paired daily IC differences and HAC/bootstrap inference, but omit the seed index even though Tier 1.A uses 10 seeds at lines 107-109 and Tier 1.B/C/D use 3-5 seed pilots at lines 137-139, 179-180, and 232-233.
    suggested_fix: >-
      Define the estimand as seed-paired daily differences by fold/seed/day, then either average across matched seeds per date before HAC or use a hierarchical date-block bootstrap with seed resampling nested inside date/fold blocks; do not treat seed x day rows as independent days.
    status: OPEN
  - id: B-03
    severity: MAJOR
    category: statistical_rigor
    claim: >-
      The pilot-to-expansion and Tier 2 triggers are under-specified, allowing adaptive selection and post-selection inference on the same folds.
    evidence: >-
      Plan Tier 1.B says seeds=5 and "expand to 10 only if Tier 1 gates pass" at lines 137-139, but no pass gate is defined for robust pointwise losses; Phase A asks "Huber > MSE? wd > 0 helps?" at line 435 without a threshold; Tier 2.A triggers on "Huber stable" at line 296 and Tier 2.B triggers on "Tier 1.B + 1.C pass" at line 315 without defining "stable" or "pass."
    suggested_fix: >-
      Freeze explicit pilot gates before running: for each candidate loss/config, specify the primary metric, minimum effect size, allowed fold-4/folds-0-3 degradation, and whether passing permits only expansion or also a claim; report hparam 1.D as model selection only unless a locked confirmatory rerun is performed.
    status: OPEN
  - id: B-04
    severity: MAJOR
    category: reporting_fence
    claim: >-
      The regime forensic gate is vulnerable to post-hoc cherry-picking because fold-4 is already known and the degradation-share rule is not operationally defined across multiple regime variables.
    evidence: >-
      Plan Tier 1.E lines 263-270 gates regime-model work on whether a pre-specified regime explains at least 50% of fold-4 IC degradation and says thresholds are frozen before viewing fold-4 results, while Stage 1 fold-4 collapse is already stated as a known fact at lines 29-32; lines 252-259 and 271-275 list several candidate regime variables without saying which one is the primary gate or how the 50% degradation share is computed.
    suggested_fix: >-
      Reword the freeze point to "before computing regime-stratified fold-4 tables"; define the degradation-share numerator and denominator; designate one primary regime gate, preferably the first-priority lagged 21-day cross-sectional dispersion, and treat drawdown/VIX/yield variables as ordered secondary diagnostics unless multiplicity correction is applied.
    status: OPEN
  - id: B-05
    severity: MAJOR
    category: compute_budget
    claim: >-
      The Tier 1 compute budget assumes the trimmed 1.A pilot while the 1.A design still specifies the full 400-cell experiment.
    evidence: >-
      Plan Tier 1.A lines 111-115 defines both a 400-cell full design and a 100-cell trimmed pilot, but the Tier 1 budget table lines 279-286 and execution total lines 430-440 budget only the 100-cell pilot; the plan does not state whether the full 400-cell design is deferred, conditional, or still part of Tier 1.
    suggested_fix: >-
      Declare the 100-cell 1.A pilot as the only Tier 1 execution item under the 30h M4 budget, add a promotion gate for any 400-cell expansion, and exclude that expansion from the Phase 0-C budget unless explicitly triggered.
    status: OPEN
  - id: B-06
    severity: CONCERN
    category: cross_discussion_conflict
    claim: >-
      The plan labels 1.A as a synthesis of Discussion A and B, but it resolves the A/B window disagreement without documenting that A's longer-history data-length test was dropped.
    evidence: >-
      Plan Tier 1.A heading line 98 cites Codex A Tier 1.1 plus B Tier 2.1, but the design at lines 102-115 is an expanding-vs-2y-rolling SAGE-Mean test; Discussion A's Tier 1 recommendation was a 3y/current/10y MLP-S8 data-window ablation, while Discussion B recommended the 2y rolling paired comparison.
    suggested_fix: >-
      State explicitly that 1.A answers the recency/stale-regime question from Discussion B, not the longer-history data-poverty question from Discussion A; either relabel the source attribution or add a written rationale that the 10y/3y data-length ablation is deferred/skipped for this paper.
    status: OPEN
  - id: B-07
    severity: CONCERN
    category: tier0_completeness
    claim: >-
      Tier 0 lacks an executable sentinel leakage test, so the leakage controls remain mostly provenance assertions rather than behavioral verification.
    evidence: >-
      Plan Tier 0 lines 41-92 includes a feature audit, rolling manifest support, graph provenance assertions, and expanding-vs-rolling manifest assertions, but it does not include the perturb-after-boundary sentinel test recommended in Discussion B's zero-leakage verification protocol.
    suggested_fix: >-
      Add a Tier 0 sentinel test: perturb prices/features strictly after min(val_days) and assert that train features, train labels, train scaler state, and frozen train graph are bitwise unchanged for every fold/split type.
    status: OPEN
  - id: B-08
    severity: CONCERN
    category: stop_criteria
    claim: >-
      The 1.A stop criterion can overstate a fold-4-only repair as a "strong signal" without tying it back to the preceding all-fold decision rule.
    evidence: >-
      Plan Tier 1.A lines 123-125 correctly says 2y rolling is preferable only if it improves the all-5 aggregate and is not materially negative on folds 0-3, but line 127 then says fold-4 ListMLE IC > -0.15 is a "strong signal" without specifying that this is only a fold-4 collapse-attenuation signal.
    suggested_fix: >-
      Reword line 127 to "strong signal for fold-4 collapse attenuation only"; reserve "2y rolling preferable" for the line 123-125 all-fold and folds-0-3 conditions.
    status: OPEN
summary:
  critical: 0
  major: 5
  concern: 3
  total: 8
overall_verdict: PROCEED-WITH-FIXES
verdict_rationale: >-
  The plan is directionally sound and keeps the Stage 1 verdict locked, but several execution gates are not yet tight enough for unambiguous supplementary experimentation. Fix the ranking-loss gate inconsistency, seed/date inference specification, adaptive pilot triggers, and regime-analysis pre-analysis protocol before spending compute.
---

Round B question coverage: Q1 dependencies mostly hang together, but B-01, B-03, B-05, and B-07 need tightening before execution. Q2 Newey-West/block bootstrap are appropriate in principle for 21-day overlapping labels, but B-02 and B-03 must be fixed to avoid pseudo-replication and post-selection inference. Q3 the main-text reporting fence is directionally good, but B-04 leaves a p-hacking path inside the supplementary regime narrative. Q4 most gate directions are sensible, with B-01 and B-08 as the main reversed/ambiguous logic risks. Q5 per-cell rates are broadly consistent for the trimmed plan, but B-05 makes the total budget ambiguous. Q6 Tier 0 is close but missing B-07. Q7 the only clear A/B/C synthesis conflict I found is B-06; other differences appear to be intentional budget pruning rather than unresolved disagreement.

## B-01: Anchored RankNet Gate Is Weaker Than The Global Ranking-Loss Gate

Exact problem: the plan defines a mandatory fold-4 viability gate for any new ranking loss, then gives anchored RankNet a different and weaker gate. The global gate requires `fold-4 IC > -0.15`, `sigma_fold(IC) <= 2 x sigma_fold(MSE)`, and `median(pred_cs_std) >= 0.05` in the Hard Constraints section, lines 19-22, and repeats the same gate in Statistical guards, lines 419-422. Tier 1.C, lines 214-216, only requires fold-4 IC not to be below MSE by 0.05 and the prediction-scale floor.

Why it matters: anchored RankNet is exactly the kind of new ranking loss that could reproduce the prior pairwise/listwise failure modes. Omitting the absolute fold-4 floor and fold-sigma condition allows a configuration to pass even if it remains unstable across folds, especially if the MSE fold-4 comparator is itself weak. That would undermine the plan's stated protection against another ListMLE-style fold-4 collapse.

Specific citation: `/Users/heruixi/.claude/plans/plan-zpp-unified-2026-04-29.md`, "Hard Constraints", lines 19-22; "1.C - Anchored RankNet pilot", lines 214-216; "Statistical guards against p-hacking", lines 419-422.

Concrete fix: make the Tier 1.C gate exactly the global ranking-loss gate and append the relative-to-MSE condition as an additional requirement: pass only if `fold-4 IC > -0.15`, `sigma_fold(IC) <= 2 x sigma_fold(MSE)`, `median(pred_cs_std) >= 0.05`, and fold-4 IC is not below MSE by more than 0.05 in any architecture-feature pilot cell.

## B-02: Seed Handling Is Under-Specified For HAC And Bootstrap Inference

Exact problem: the plan repeatedly says to use paired daily IC differences with Newey-West lag 21 or moving-block bootstrap, but it does not specify how to handle multiple seeds. Tier 1.A has 10 seeds at lines 107-109, Tier 1.B and 1.C use 5-seed pilots at lines 137-139 and 179-180, and Tier 1.D uses 3 seeds at lines 232-233. The statistical-analysis text at lines 117-120 and the reporting standards at lines 394-413 refer only to daily differences, not `fold x seed x day` differences.

Why it matters: seeds are not independent market histories. Treating 10 seeds on the same day as 10 independent daily observations would shrink standard errors and make weak IC deltas look more precise than they are. Discussion B's paired-test definition explicitly uses `d_{fold, seed, day}` and then calls for Newey-West or block bootstrap inference; the unified plan needs to preserve that hierarchy.

Specific citation: plan "1.A - Window ablation", lines 107-120; "1.B - Robust pointwise sweep", lines 137-139; "1.C - Anchored RankNet pilot", lines 179-180; "1.D - Hparam regularization sweep", lines 232-233; "Reporting standards", lines 394-413. Source cross-check: `2026-04-29_codex_discussion_B_split_methodology.md`, lines 278-284.

Concrete fix: define the estimator before execution. Acceptable choices are: average matched seed-paired daily deltas per calendar day and apply HAC over dates, or use a hierarchical bootstrap that samples fold/date blocks and resamples seeds within those blocks. In either case, keep the fold-cluster bootstrap as a sensitivity, not the primary precision estimate.

## B-03: Pilot Expansion Gates Are Not Frozen

Exact problem: the plan says robust pointwise losses use a 5-seed pilot and expand to 10 seeds only if Tier 1 gates pass, but no gate is specified for those losses. Phase A asks "Huber > MSE? wd > 0 helps?" without thresholds. Tier 2.A then triggers on "Huber stable", and Tier 2.B triggers on "Tier 1.B + 1.C pass", but neither "stable" nor "pass" is operationally defined for 1.B.

Why it matters: this is the most direct double-dipping path in the plan. If the team looks at 5-seed results, decides informally which result is "stable", expands that winner, and then reports the expanded result as evidence, the final claim inherits unreported selection pressure. The hparam sweep has the same issue if the selected score is later described as an improvement rather than as model selection.

Specific citation: plan "1.B - Robust pointwise sweep", lines 133-143; "1.D - Hparam regularization sweep", lines 239-243; "2.A - Group-DRO + Huber", line 296; "2.B - Anchored soft Spearman IC", line 315; "Execution order", line 435.

Concrete fix: add a small pre-analysis gate table. For 1.B, specify, for example, the primary comparison `Huber/Tukey/trunc_mse vs same-cell MSE`, the minimum all-fold point estimate, the maximum tolerated folds-0-3 and fold-4 degradation, and whether a pass only authorizes 10-seed expansion. For 1.D, state that the stability score is only a selection heuristic unless a locked confirmatory rerun or held-out seed expansion is run.

## B-04: Regime Forensic Gate Still Allows Cherry-Picked Explanations

Exact problem: the plan says regime thresholds are frozen before viewing fold-4 results, but the document itself lists fold-4 collapse as a known Stage 1 fact. It also lists multiple regime variables, including dispersion, volatility, drawdown, optional VIX, and yield-spread background, without defining which variable is the primary gate or how "regime contributes >= 50% of fold-4 IC drop vs folds 0-3 average" is computed.

Why it matters: the regime analysis is supplementary and forensic, so some post-hoc context is unavoidable. The integrity risk is presenting a selected regime explanation as if it had been prospectively tested. With several plausible regime variables and tercile cuts, an "any variable explains 50%" rule can discover an explanation by search, especially because the target event, fold-4 ListMLE collapse, is already known.

Specific citation: plan "Paper-known Stage 1 facts", lines 29-32; "1.E - Regime-stratified forensic analysis", lines 252-270; "Macro indicator priority", lines 271-275; "Statistical guards against p-hacking", lines 423-425.

Concrete fix: change the freeze language to "before computing any regime-stratified fold-4 tables." Define the degradation-share formula explicitly, including the denominator and sign convention. Make lagged 21-day cross-sectional dispersion the primary gate if that is the intended mechanism, because it is listed as highest priority at lines 271-273. Treat drawdown, VIX, and yield spread as ordered secondary diagnostics unless the plan applies multiplicity correction and reports all failures.

## B-05: Tier 1 Budget Depends On An Ambiguous 1.A Scope

Exact problem: Tier 1.A specifies a full 400-cell design, then describes a 100-cell trimmed pilot. The Tier 1 budget and execution table count the trimmed pilot, but the design section still leaves the full 400-cell version in Tier 1 without a promotion gate.

Why it matters: the stated compute budget is about 30h M4. The trimmed plan is plausible against Discussion B's diagnostic rate of 200 cells in 439 minutes, but the full 1.A run plus the other Tier 1 experiments would consume most or all of the budget before any conditional Tier 2 work. Ambiguity here can cause the team to overspend compute on the first experiment and then selectively drop later experiments, which is another form of analysis-order bias.

Specific citation: plan "1.A - Window ablation", lines 111-115; "Tier 1 budget", lines 279-288; "Execution order", lines 430-440. Source cross-check: `2026-04-29_codex_discussion_B_split_methodology.md`, line 272 reports the diagnostic run rate.

Concrete fix: state that Tier 1 includes only the 100-cell 1.A pilot. Add a promotion rule for the full 400-cell expansion, such as passing the fold-4 attenuation diagnostic while satisfying the all-fold/folds-0-3 decision rule and leaving enough budget. If the full expansion is not intended before submission, move it out of Tier 1.

## B-06: Discussion A's Data-Length Recommendation Is Not Actually Covered

Exact problem: the plan's 1.A heading cites both Discussion A and Discussion B, but the actual experiment follows B's 2-year rolling-vs-expanding design. Discussion A's high-priority data-length recommendation was a 3y/current/10y window ablation on MLP-S8 with MSE and ListMLE, not a SAGE-Mean 2y rolling recency test.

Why it matters: this is not a reason to add the 10y experiment under the deadline. It is a paper-credibility issue about what the supplement can claim. A 2y rolling test answers whether stale 2021-2022 regimes hurt fold 4; it does not answer whether longer history cures data poverty. If the write-up later says Plan Z++ tested Discussion A's data-length concern, that would overclaim the evidence.

Specific citation: plan "1.A - Window ablation", lines 98-115. Source cross-check: `2026-04-29_codex_discussion_A_data_length_regime.md`, lines 21 and 122-128; `2026-04-29_codex_discussion_B_split_methodology.md`, lines 257-270 and 322-324.

Concrete fix: relabel 1.A as the B-style recency/stale-regime ablation and explicitly state that A's 3y/5y/10y data-length ablation is deferred or skipped for this paper. Do not describe the 2y rolling result as evidence about longer-history data extension.

## B-07: Tier 0 Needs A Behavioral Leakage Sentinel

Exact problem: Tier 0 audits the precomputed feature file and adds manifest/graph assertions, but it does not include a behavioral sentinel test that proves future-period perturbations cannot affect training artifacts. Discussion B's zero-leakage protocol specifically recommends perturbing data strictly after `min(val_days)` and checking that train features, labels, scaler, and frozen graph are unchanged.

Why it matters: provenance review can miss pipeline bugs, cached arrays, hidden global transforms, or indexing mistakes. A sentinel test is cheap and catches leakage mechanically. Given the paper's locked null result and the plan's reliance on supplementary claims, this should be part of the pre-experiment validity layer rather than an optional QA step.

Specific citation: plan "Tier 0 - Mandatory pre-experiment fixes", lines 37-92. Source cross-check: `2026-04-29_codex_discussion_B_split_methodology.md`, lines 211-218.

Concrete fix: add a Tier 0 item after 0.4: for each fold and split type, perturb prices/features after `min(val_days)` and assert bitwise equality for train features, train labels, train scaler parameters, graph snapshot index/window, and frozen graph edges. Save the result in the audit artifact next to the phase5 feature audit.

## B-08: The 1.A Fold-4 Stop Criterion Needs Narrower Language

Exact problem: Tier 1.A has a good general decision rule: call 2y rolling preferable only if it improves the all-5 aggregate and is not materially negative on folds 0-3. The stop criterion immediately below it then says fold-4 ListMLE IC above -0.15 is a "strong signal" without specifying that this is only a fold-4 collapse-attenuation signal.

Why it matters: this is a subtle narrative-leakage risk. A fold-4-only improvement is scientifically useful, but the plan already says it should be reported as regime-conditional if improvement only appears in fold 4. Calling it a strong signal without the qualifier creates room to overstate the rolling window as generally better.

Specific citation: plan "1.A - Window ablation", lines 123-127.

Concrete fix: rewrite the stop criterion as: "If 2y-rolling ListMLE fold-4 IC > -0.15, report a strong signal for fold-4 collapse attenuation; call 2y rolling generally preferable only if the all-5 aggregate improves and folds 0-3 are not materially negative." This preserves the intended diagnostic while preventing a reversed interpretation.
