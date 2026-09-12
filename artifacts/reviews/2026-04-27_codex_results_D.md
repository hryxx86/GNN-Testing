---
reviewer: codex
touchpoint: results
round: D
date: 2026-04-27
target_files:
  - /Users/heruixi/.claude/plans/loss-function-s6-research-jaunty-nova.md
  - /Users/heruixi/Desktop/GNN-Testing/.claude/rules/docs.md
  - /Users/heruixi/Desktop/GNN-Testing/experiments/loss_horserace/per_cell_stats.csv
  - /Users/heruixi/Desktop/GNN-Testing/experiments/loss_horserace/mixed_effects_ic.csv
  - /Users/heruixi/Desktop/GNN-Testing/experiments/loss_horserace/mixed_effects_pred_cs_std.csv
  - /Users/heruixi/Desktop/GNN-Testing/experiments/loss_horserace/block_bootstrap_sharpe.csv
  - /Users/heruixi/Desktop/GNN-Testing/experiments/loss_horserace/fold4_lofo_stats.csv
  - /Users/heruixi/Desktop/GNN-Testing/experiments/loss_horserace/analysis_scenario.json
  - /Users/heruixi/Desktop/GNN-Testing/experiments/loss_horserace/results.csv
  - /Users/heruixi/Desktop/GNN-Testing/experiments/loss_horserace/local_analyze_log.txt
  - /Users/heruixi/Desktop/GNN-Testing/analyze_loss_horserace.py
  - /Users/heruixi/Desktop/GNN-Testing/progress.md
findings:
  - id: CODEX-D-01
    severity: MAJOR
    category: statistics
    claim: "The ΔSharpe bootstrap is not the preregistered studentized bootstrap of the same ΔSharpe estimand used for the point estimate."
    evidence: "Plan lines 281-285 register ΔSharpe_net with a studentized block bootstrap; analyze_loss_horserace.py lines 320-355 set point = Sharpe(loss)-Sharpe(mse) but boot_deltas = Sharpe(resampled daily Δreturn), and block_bootstrap_sharpe.csv row 3 reports ci_hi=-1.291372607442722 while p_delta_sharpe=0.1174."
    suggested_fix: "Before paper use, relabel this as a non-studentized Sharpe-of-difference sensitivity or recompute the bootstrap on Sharpe(loss)-Sharpe(mse) with the registered studentized statistic."
    status: OPEN
  - id: CODEX-D-02
    severity: MAJOR
    category: statistics
    claim: "The MixedLM specification is preregistered and captures fold plus fold-day clustering, but SEs are fragile because seed-level repeated-measure correlation is omitted and the final log contains convergence/Hessian warnings."
    evidence: "Plan lines 259-272 and analyze_loss_horserace.py lines 236-242 use groups=fold plus vc_formula fold_day with no seed term; local_analyze_log.txt lines 63-68 show MixedLM gradient convergence failure and lines 149-154 show another MixedLM optimization failure."
    suggested_fix: "Report the warning status and frame MixedLM p-values, especially the supporting Δpred_cs_std p-values, as qualified unless a seed-aware or cluster-robust sensitivity agrees."
    status: OPEN
  - id: CODEX-D-03
    severity: MAJOR
    category: correctness
    claim: "The Scenario A decision code omits the registered direction check that a ranking loss must beat MSE."
    evidence: "Plan line 323 says Scenario A requires a ranking loss to beat MSE on both ΔIC and ΔSharpe_net, while analyze_loss_horserace.py lines 403-404 define co_primary_reject from p-values and Bonferroni flags only, without beta_delta_ic > 0 or delta_sharpe > 0."
    suggested_fix: "Add positive-direction checks before any Scenario A/A-regime/A-partial classification; current Scenario B is unchanged because all co_primary_reject values are False in per_cell_stats.csv rows 2-9."
    status: OPEN
  - id: CODEX-D-04
    severity: MAJOR
    category: interpretation
    claim: "The auto-verdict phrase 'scale collapse diagnostic confirmed' is too broad for the observed supporting endpoint pattern."
    evidence: "analysis_scenario.json line 3 says scale collapse diagnostic confirmed, but mixed_effects_pred_cs_std.csv rows 7 and 9 are the only significant pairwise SAGE cells, while rows 2-6 and 8 do not reject."
    suggested_fix: "Rewrite as 'SAGE-Mean pairwise shows a scale-collapse diagnostic on S6 and S8; the other six pred-scale contrasts do not reject.'"
    status: OPEN
  - id: CODEX-D-05
    severity: MAJOR
    category: interpretation
    claim: "The phrase 'portfolio gain marginal' reads like an equivalence or marginal-benefit claim, but the current tests only support no detected ranking-loss superiority."
    evidence: "analysis_scenario.json line 3 says portfolio gain marginal; block_bootstrap_sharpe.csv rows 2-9 have all delta_sharpe values negative, and per_cell_stats.csv rows 2-9 have co_primary_reject=False."
    suggested_fix: "Use 'no portfolio improvement detected; ΔSharpe point estimates favor MSE but are not declared significant under the registered primary gate.'"
    status: OPEN
  - id: CODEX-D-06
    severity: CONCERN
    category: regime
    claim: "Fold-4 LOFO reverses the ΔIC direction, so the plain Scenario B headline hides a material regime caveat."
    evidence: "Full-data per_cell_stats.csv rows 2-9 have seven negative beta_delta_ic values and one tiny positive value (+0.002691576464148498), while fold4_lofo_stats.csv rows 2-9 have all beta_delta_ic values positive, from +0.0005995451837252644 to +0.025507447676958703."
    suggested_fix: "Headline the result as 'Scenario B with fold-4 caveat'; keep full-data as primary because plan lines 316-318 only make LOFO mandatory for Scenario A declarations."
    status: OPEN
  - id: CODEX-D-07
    severity: CONCERN
    category: correctness
    claim: "Pairwise ΔIC mixed-effects sample sizes are lower than the expected 3130 observations, so missing-IC handling should be disclosed."
    evidence: "mixed_effects_ic.csv rows 3, 5, 7, and 9 report pairwise n=3077, 3066, 3061, and 2997, while mixed_effects_pred_cs_std.csv rows 3, 5, 7, and 9 report n=3130 for the same pairwise cells."
    suggested_fix: "Report the dropped IC counts and likely constant-prediction/Spearman-undefined mechanism alongside the scale-collapse discussion."
    status: OPEN
  - id: CODEX-D-08
    severity: CONCERN
    category: prior-art
    claim: "The paper framing is defensible only as a setup-scoped null against these two ranking losses, not as a general claim that ranking losses fail for stock selection."
    evidence: "Plan lines 27-36 cite MSE as a field default and ListMLE/pairwise as literature-backed contenders, while per_cell_stats.csv rows 2-9 show no co-primary rejection in this US 500-stock S6/S8 setup."
    suggested_fix: "Frame the result as 'ranking losses did not improve MSE in this 10-seed US 500-stock S6/S8 horse race; SAGE pairwise collapses prediction scale without portfolio benefit.'"
    status: OPEN
summary:
  critical: 0
  major: 5
  concern: 3
overall_verdict: PROCEED-WITH-FIXES
---

**1. Credibility**
The raw `results.csv` is structurally consistent with the run matrix: local aggregation found 37,560 data rows, 600 unique `(model, loss, feature, fold, seed)` runs, and fold day counts of 63, 64, 64, 60, and 62. The paired Δpred_cs_std mixed-effects table has the expected 3,130 observations per contrast, matching 10 seeds times 313 total test days. ΔIC is mostly consistent, but pairwise IC drops rows: `mixed_effects_ic.csv` reports 3,077, 3,066, 3,061, and 2,997 observations for pairwise cells. That is not a run-completeness problem, because the pred-scale table remains at 3,130 for the same cells; it is an IC-missingness problem that should be disclosed. Recomputing the full MixedLM outputs from `results.csv` reproduced the saved β values exactly. Half-seed subsamples were qualitatively stable for ΔIC β direction and magnitude, but produced additional convergence warnings, so I would treat the subsample exercise as a sanity check, not as a new inferential table.

**2. Methodology**
The registered MixedLM structure is implemented as planned: fold random intercept plus a fold-day variance component. That captures the main crossed dependency from seeds sharing the same test day within a fold. It is still not a complete dependence model because the same seed IDs recur across folds and days, and no seed random effect or seed fixed effect is included. The plan explicitly chose "seeds as fixed replicates," so this is not a preregistration breach, but it can understate SE when seed-specific training behavior persists across days. The higher-risk issue is that the final analysis log contains MixedLM convergence and non-positive-definite Hessian warnings. That matters most for Δpred_cs_std, where the scenario label depends on supporting p-values.

The block bootstrap preserves the fold structure in the right broad sense: blocks are resampled within non-overlapping walk-forward folds and then aggregated, which is appropriate for the five time windows. The implementation problem is narrower but important: the point estimate is `Sharpe(loss)-Sharpe(mse)`, while bootstrap draws use the Sharpe ratio of the daily return difference. The plan also says "studentized"; the code uses percentile quantiles plus a centered p-value. I would not use the current ΔSharpe p-values or CIs as final paper inference without relabeling or recomputing the bootstrap. The BH-FDR family itself is consistent with the plan: 8 cells times the two co-primary endpoints gives 16 tests, and the Bonferroni co-primary gate is applied per endpoint.

**3. Interpretation**
"MSE wins all 8 ΔSharpe contrasts numerically" is defensible only as a point-estimate direction statement: all eight `delta_sharpe` values are negative. It must not be written as statistical superiority because all co-primary gates are false. The 2026-04-21-c `T_SPA` incident is the relevant precedent: `.claude/rules/docs.md` lines 82-84 and `progress.md` lines 305-308 document how an over-read scalar propagated into advisor docs. Apply the same discipline here. "Scale collapse confirmed" should be narrowed to SAGE-Mean pairwise on S6 and S8. "Portfolio gain marginal" should be replaced; the current tests support no detected portfolio improvement, not equivalence and not a marginal benefit claim. Equivalence would require a TOST-style design that was not run.

**4. Regime/Fold Sensitivity**
The fold-4 LOFO table is important even though no Scenario A trigger exists. Dropping fold 4 makes all ΔIC β values positive, while the full-data table is mostly negative. That says MSE's numerical ΔIC advantage is fold-4-driven. I would keep the preregistered full-data result as primary, because the plan only makes LOFO a mandatory downgrade rule for Scenario A. But the honest headline is "Scenario B with fold-4 caveat," not plain Scenario B.

**5. Pre-Registration Honesty**
`apply_multiple_testing` matches the registered 16-test BH-FDR family and Bonferroni co-primary structure. The verdict logic matches Scenario B for this run because no co-primary cell rejects and the supporting pred-scale family has two BH rejections. The latent deviation is Scenario A directionality: `co_primary_reject` does not check that βIC and ΔSharpe are positive. It does not alter this result, but it must be fixed before the function is reused.

**6. Prior-Art Framing**
The defensible paper claim is scoped and negative: on this US 500-stock, S6/S8, 10-seed setup, ListMLE and pairwise losses did not improve over MSE under the registered co-primary gate; SAGE-Mean pairwise collapsed prediction scale without portfolio benefit. Do not generalize to "ranking losses do not work for stock selection." The plan itself cites literature-backed reasons for testing ListMLE and pairwise losses, so the contribution is a careful null/robustness result under this design, not a contradiction of the broader learning-to-rank literature.
