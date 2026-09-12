<!-- Rule 9 Touchpoint 3, Round A, for the C-pre production results. Reviewer = Codex CLI (gpt-6-astra, xhigh),
`codex exec --sandbox read-only`, 2026-09-12 04:04–04:11 local (≈6.7 min, no usage-limit interruption). Prompt archived
in the session scratchpad (codex/cpre_results_A_prompt.md). Status/resolution_notes filled in by Claude after verifying
each claim against the artifacts (cpre_seed_robustness.csv, CPRE_L1.json top_table) and applying the wording. -->
---
reviewer: codex
touchpoint: results
round: A
target_files:
  - artifacts/storya_v21_family1_cpre/cpre_comparison.md
  - artifacts/storya_v21_family1_cpre/cpre_run_integrity.json
  - experiments/storya_v21_main12_cpre/results.csv
findings:
  - id: CODEX-A-01
    severity: MAJOR
    category: statistics
    claim: "The proposed per-arm signal interpretation and unqualified non-reproduction sentence exceed the evidence."
    evidence: "CPRE's per-arm intervals contain zero and substantial positive IC values; its contrast interval also contains all three comparator point estimates. All paired comparator-minus-CPRE intervals contain zero. These results establish neither negligible predictive signal nor a reduction in the underlying MLP-LightGBM contrast."
    suggested_fix: "Describe low pooled point estimates under the evaluated configurations. Qualify non-reproduction as a statement about the positive pooled point estimate, immediately followed by the inconclusive paired comparisons. Do not infer a statistically established per-arm decline from separate confidence intervals."
    status: FIXED
    resolution_notes: "Accepted. docs/analysis.md 2026-09-12-a uses Codex's PERMITTED sentences verbatim: non-reproduction is stated about the positive pooled POINT ESTIMATE and immediately followed by the three zero-containing paired intervals; the per-arm levels are reported as low pooled point estimates with their intervals and the explicit note that this descriptive pattern establishes neither negligible predictive content nor the cause. The 'little cross-sectional signal … while B does' sentence from the review request is NOT used."
  - id: CODEX-A-02
    severity: CONCERN
    category: correctness
    claim: "The per-seed CPRE range in the review request is incorrect."
    evidence: "cpre_seed_robustness.csv and cpre_seed_robustness_per_seed.json report -0.03143 to +0.01731. Independent reconstruction from daily arrays gives -0.03142888 for seed 456 and +0.01731492 for seed 7, rather than -0.0119 to +0.0089."
    suggested_fix: "Use -0.0314 to +0.0173 in downstream prose. Retain 5/10 same-sign seeds and 1/10 LOSO flips, which reproduce correctly."
    status: FIXED
    resolution_notes: "Verified: cpre_seed_robustness.csv per_seed_min −0.03143 (seed 456) / per_seed_max +0.01731 (seed 7); the range in the review REQUEST was a transcription error by Claude (no artifact carried the wrong numbers). docs/analysis.md uses −0.0314…+0.0173; k = 5/10 and m = 1/10 unchanged."
  - id: CODEX-A-03
    severity: CONCERN
    category: statistics
    claim: "Positive finalist averages support a descriptive tuning disclosure, not exclusion of tuning limitations."
    evidence: "CPRE_L1.json records positive three-seed averages for all five finalists, but the winner's individual tuning ICs are +0.03440, +0.01223 and -0.00624. These are selection metrics. The runner's converged_flag only checks that best_val_loss is finite."
    suggested_fix: "Specify positive three-seed finalist averages and distinguish protocol compliance from search adequacy, optimization quality and generalization. Replace the tuning-failure exclusion with the narrower observation that CPRE lacks C5's all-negative-finalist-average pattern."
    status: FIXED
    resolution_notes: "Verified: CPRE_L1.json top_table[0].tune_seed_ics = [+0.03440, +0.01223, −0.00624] (mean +0.01346); L0 winner [+0.03109 × 3]. docs/analysis.md states 'all five finalists in each arm had positive THREE-SEED AVERAGE tuning ICs (the L1 winner's individual tuning-seed ICs were +0.034 / +0.012 / −0.006); these selected validation metrics establish neither independent validation nor search adequacy' and only contrasts this descriptively with C5's all-negative finalist averages. The 'not a tuning-failure signature' clause is dropped."
  - id: CODEX-A-04
    severity: CONCERN
    category: correctness
    claim: "Adding CPRE requires updating the existing L1 account of the earlier diagnostic and the scope of pending re-selection."
    evidence: "paper/iclr2027/main.tex:290, :998 and :1012 still describe five groups as surviving strict T-1 re-ranking and re-selection as pending. The archived proxy top-15 sets are identical before and after the shift; five-of-fifteen is an overlap between different ranking methods. CPRE now supplies retrospective re-selection evidence only for L0/L1."
    suggested_fix: "Correct the five-group description, acknowledge the completed L0/L1 sensitivity, retain the qualification on original C results, and explicitly leave graph/edge re-selection unresolved."
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "Paper-side action for H博士 (main.tex is edited on the paper-side clone; not touched here). Already tracked since 2026-09-11 (plan 2026-09-11-b item 2, report §6.1); the scope sentence ('L0/L1 only; graph/edge re-selection unresolved; original C results keep their qualification') is added to the paper to-do in plan.md 2026-09-12-b and to docs/analysis.md 2026-09-12-a's paper paragraph."
summary:
  critical: 0
  major: 1
  concern: 3
  fixed_before_reply: 0
overall_verdict: PROCEED-WITH-FIXES
---

# Codex review body (verbatim)

**The production results are credible enough to report. The required fixes concern interpretation and reporting; I found no defect requiring new model runs.** The pre-declared branch-(c) reading is correct. The additional explanatory sentences need the qualifications below.

I performed a read-only audit of the saved outputs and source. I independently reproduced the headline, both CPRE per-arm intervals, all three paired contrasts, and all four ex-fold-9 comparisons using the daily arrays. No repository files were changed.

**Credibility and statistical computation**

The checks support the reported run identity:

- The five recorded runtime source modules match their archived hashes and commit `46ca6e3`. Tuning-module hashes, the selection hash chain, archived tuning JSONs and databases, and frozen-parameter copies agree. Applied parameters match the frozen winners.
- The CPRE manifest contains 240 unique completed cells, matching the results. All 240 CPRE arrays and all 720 comparator L0/L1 arrays are finite and have their full expected fold lengths. CSV means agree with array means within rounding precision.
- Reconstructing label-eligible dates from prices and the frozen calendar gives the reported fold counts and 749 days. Alignment is supported by that reconstruction and the runner's chronological evaluation loop; the `.npy` files themselves do not contain date stamps.
- The change between the production commit and the results commit is the analyzer's fold-share display guard, not a change to the reviewed training or statistical calculations.

`source_clean=True` concerns the recorded imported source modules at invocation, not the cleanliness of every file in the current workspace. It is supported in that intended sense.

A **near-zero sample mean is not a problem for these mean-based calculations**. The bootstrap resamples the daily series and estimates uncertainty in its mean; HLN divides the mean by an estimated standard error. Neither divides by the estimated contrast. The unstable quantity here is the fold-share ratio. Suppressing its percentage display is appropriate; the raw CSV ratio of `4.7441` should not become a substantive "474%" claim.

The paired calculations correctly form the daily **difference of contrasts before resampling**, preserving their shared-day covariance. The ex-fold calculation correctly removes fold 9 and pools the remaining 687 observations with daily weighting.

These remain nominal time-series procedures. A stationary bootstrap with expected block length 21 and the two HAC specifications does not guarantee validity under arbitrary regime changes. The ex-fold procedure also concatenates the retained observations across the removed quarter. Those limitations reinforce its role as a sensitivity diagnostic; they do not invalidate the reported arithmetic or require replacing the approved method for this restrained interpretation.

The CI describes temporal uncertainty in the **average of the ten seeds' daily ICs**, conditional on the fitted runs. It does not resample feature selection, tuning, or training seeds. It is also not the IC of an ensemble formed by averaging predictions.

**Application of plan §1**

Branch (c) applies exactly: ΔIC = −0.0024, CI [−0.0256, +0.0178], with nominal HLN p = 0.786 / 0.847. There is no CPRE disagreement among these zero-exclusion decisions. Preserve the sentence that containing zero does not establish absence of a contrast.

The paired reading is also correct. All three intervals include zero, so the conditional "comparator exceeds C-pre; consistent with selection inflation" sentence is **not triggered**. The paired MDEs indicate limited resolution for changes of roughly the observed 0.016–0.017 magnitude. They do not establish equivalence.

Keep MDE as an **approximate nominal 80%-power scale**, not a significance threshold. An estimate below its MDE can still reject under a particular test; this explains why the C/C5 findings and their MDE comparisons are not logically inconsistent. When reporting the comparator table, retain its existing disagreements: C's auto-lag rejection versus its zero-containing bootstrap CI, and C5's positive bootstrap interval versus lag-21 non-rejection.

The inventory correctly reports 22 published nominal HLN p-values. They are correlated specifications and contrasts, not 22 independent replications.

**Per-arm levels, fold structure and tuning**

I would replace "the 48 columns carry little cross-sectional signal … while B does" with a statement about **what the evaluated models achieved**. The pooled estimates are low, but the intervals are wide, and performance varies sharply across quarters. Separate per-arm intervals also do not test whether B or C has higher underlying IC than CPRE. The existing paired tables test changes in **L1−L0**, not changes in each arm's level.

The selector chose the top 15 groups by the declared empirical marginal-IC score and included all their members. It did not identify an optimal 48-feature representation for either model. Sampling variation in selection, regime dependence, representation differences and tuning limitations remain possible explanations. This experiment does not distinguish them.

Fold 9 is legitimately influential. The exact decomposition is approximately −0.00242 = −0.01148 (fold 9 contribution) + 0.00906 (remaining folds' contribution). It is the largest negative contribution, and excluding it reverses the pooled point estimate to +0.0099. That supports "sensitive to 2025Q2." It does not establish why LightGBM performed unusually well then, or justify replacing the full-period result with the exclusion result.

All finalists have positive **three-seed average** tuning ICs. That differs descriptively from C5, but positive selected validation averages do not demonstrate tuning adequacy. Protocol consistency follows from the audited procedure and frozen-parameter chain.

The parameter-count statement is defensible when phrased precisely: CPRE has 31,361 parameters versus C's 31,745, a difference of about 1.2%. Similar parameter counts do not equalize effective capacity, regularization or optimization.

**Paper treatment and necessary work before writing**

The requested re-selection sensitivity has now been conducted for **MLP versus LightGBM**. It yields an inconclusive estimate and does not provide independent confirmation of C's positive contrast. L1 should acknowledge that completed evidence while retaining the original qualification and leaving graph/edge claims unresolved.

The existing L1 paragraph (paper/iclr2027/main.tex:290) and associated diagnostic text need the scope correction in finding A-04. C-pre does not justify assertions that negative contrasts are immune to selection effects.

**No additional model runs or inferential analyses are necessary before writing the restrained account below.** Correct the seed range, narrow the proposed interpretations, preserve the full-period result alongside the fold diagnostic, and update L1 consistently. Per-arm paired testing would become necessary only if you choose to claim a statistically established decline in either arm's level. L2 on CPRE is outside this review's approved scope.

**PERMITTED sentences for docs/analysis.md and the paper appendix**

- "We conducted a retrospective sensitivity analysis using feature re-selection whose scoring and grouping inputs were restricted to information through June 2022, conditional on the study's fixed stock panel, and re-tuned LightGBM and MLP under the existing evaluation protocol."
- "On C-pre, the seed-averaged daily MLP−LightGBM contrast was −0.0024 (95% stationary-block-bootstrap CI [−0.0256, +0.0178]; nominal HLN p = 0.786 at the automatic HAC lag and 0.847 at lag 21). The interval contains zero; this does not establish absence of a contrast. The estimate's magnitude is below C-pre's own approximate nominal MDE of 0.0313."
- "The positive pooled point estimate observed in C, C5 and B was not reproduced on C-pre. However, all three paired difference intervals contain zero, so the analysis does not establish that the underlying contrasts differ across these bases."
- "The comparator-minus-C-pre contrasts were +0.0172 for C, +0.0167 for B and +0.0158 for C5, with respective 95% intervals [−0.0111, +0.0494], [−0.0133, +0.0509] and [−0.0095, +0.0481]; these comparisons remain imprecise."
- "Both evaluated models had low pooled IC point estimates on C-pre: LightGBM 0.0057 [−0.0230, +0.0376] and MLP 0.0033 [−0.0181, +0.0245]. Their point estimates were lower than on C and B; this descriptive pattern does not establish negligible predictive content or identify the cause of the differences."
- "The pooled contrast was sensitive to 2025Q2, when LightGBM's IC was 0.2042 and MLP's was 0.0656. Excluding that quarter gave ΔIC = +0.0099 [−0.0069, +0.0265]; the full-period result remains primary."
- "CPRE's per-seed contrasts ranged from −0.0314 to +0.0173, with five of ten matching the pooled sign and one leave-one-seed-out sign reversal. These summarize initialization sensitivity on the same evaluation data."
- "All five finalists in each arm had positive three-seed average tuning ICs. These selected validation metrics establish neither independent validation nor search adequacy. The CPRE and C MLPs had similar parameter counts, 31,361 and 31,745."
- "C-pre does not estimate how much leakage inflated C; it provides the clean-basis estimate of the same contrast, where 'clean basis' means feature re-selection whose scoring and grouping inputs are bounded by 2022-06-30, conditional on the study's fixed stock panel and on a protocol chosen retrospectively. B remains the leak-free anchor."
- "This sensitivity addresses feature re-selection for the MLP−LightGBM comparison only. It does not re-evaluate graph or edge contrasts, remove the fixed-panel limitation, or convert the original C results into clean confirmatory evidence."

**FORBIDDEN sentences for docs/analysis.md and the paper appendix**

- "The MLP−LightGBM advantage is not reproduced on a pre-evaluation basis." — Without the point-estimate qualification and inconclusive paired comparison.
- "C-pre shows that the selected features contain little or no predictive signal, whereas B contains signal."
- "Removing leakage eliminated, reversed, halved or reduced the MLP advantage by a known amount."
- "The models are equivalent on C-pre," or "LightGBM is superior on C-pre."
- "Positive tuning finalists rule out tuning failure," or "all tuning-seed ICs were positive."
- "Similar parameter counts rule out capacity or optimization explanations."
- "Excluding the anomalous quarter reveals the true MLP advantage."
- "Five groups survived removal of the T−1 leakage." — The historical five-group overlap compares ranking methods.
- "C-pre resolves L1," "provides leak-free confirmation," or "confirms that graphs hurt on a clean re-selected basis."
- "Selection leakage cannot affect null or negative within-universe contrasts."

---

## Claude disposition (2026-09-12)

- A-01 MAJOR accepted and applied (wording); A-02 verified and corrected (Claude's transcription error in the request, not in any artifact); A-03 verified (winner's tuning-seed ICs +0.034 / +0.012 / −0.006) and applied; A-04 accepted as the pending paper-side action. Codex's independent reproduction of the headline, per-arm intervals, paired contrasts and ex-fold statistics matches the published artifacts; no new runs.
- docs/analysis.md 2026-09-12-a uses the PERMITTED list verbatim and carries the FORBIDDEN list.
