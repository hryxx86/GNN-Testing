<!-- Rule 9 Touchpoint 1, Round A, for the C-pre plan (docs/c_pre_plan_2026-09-11.md). Reviewer = Codex CLI
(gpt-6-astra, xhigh), `codex exec --sandbox read-only`, 2026-09-11 18:36–18:41 local (≈5 min, no usage-limit
interruption). Prompt archived in the session scratchpad (codex/cpre_A_prompt.md). Statuses and resolution_notes
filled in by Claude after reading each cited location (compute_family1_ladder.py:329-345,
analyze_c5_sensitivity.py:395-440 and :590-600, run_step3_plan_z_part_a.py:81-92) and editing the plan. -->
---
reviewer: codex
touchpoint: plan
round: A
target_plan: docs/c_pre_plan_2026-09-11.md
findings:
  - id: CODEX-A-01
    severity: MAJOR
    category: statistics
    claim: "The pre-declared reporting branches omit a statistically detected sign reversal and assume an MDE conclusion that is not known for C-pre."
    evidence: "docs/c_pre_plan_2026-09-11.md:22-25 requires exactly one of (a)/(b), but neither covers a CI entirely below zero. Sentence (b) asserts that the MDE exceeds every observed contrast. compute_family1_ladder.py:331-343 instead estimates a separate bootstrap SE and MDE for each contrast."
    suggested_fix: "Declare three exhaustive outcomes: CI entirely positive, entirely negative, or containing zero. Compute each contrast's approximate nominal MDE as 2.8 times its own bootstrap SE; retain 0.020 only as a historical planning reference. Generate C-pre reporting from its actual results, including both HAC lags, rather than inheriting C5-specific conclusions."
    status: FIXED
    resolution_notes: "Verified: family1 computes SE_block/MDE per (universe, pair) row (compute_family1_ladder.py:329-345), so the 0.020 figure is C5/C/B-specific. Plan §1 rewritten: three exhaustive CI branches; |Δ| compared with C-pre's OWN 2.8×SE; CI-vs-HLN disagreement reported as such; both HAC lags; nothing inherited from C5/C/B."
  - id: CODEX-A-02
    severity: MAJOR
    category: statistics
    claim: "The proposed paired-difference CI does not by itself establish whether the underlying contrast was halved or doubled."
    evidence: "docs/c_pre_plan_2026-09-11.md:26 promises a halving/doubling interpretation from the C minus C-pre interval. analyze_c5_sensitivity.py:431-439 currently generates this interpretation using paired MDE greater than the absolute C point estimate plus CI inclusion of zero, which does not test either proportional hypothesis."
    suggested_fix: "Either report the absolute paired change and leave proportional comparisons descriptive, or explicitly define halving/doubling relative to each comparator and bootstrap the paired daily series d_CPRE minus k times d_comparator for k=0.5 and k=2. Do not infer equivalence from non-rejection or use MDE as an interval-membership test."
    status: FIXED
    resolution_notes: "Verified: analyze_c5_sensitivity.py:431-439 derives the sentence from (paired MDE > |C contrast|) and (CI includes 0). Plan §1 now reports the absolute paired change only (CI, both lags, own MDE) and forbids any halving/doubling statement; §5.6 requires the analyzer's paired note to be replaced by the absolute-change wording when generalised to CPRE. Paper-side note for H博士: the same sentence in the C5 paragraph draft should become descriptive (the values +0.007 / −0.015 that would correspond to a halved / doubled C contrast lie inside the paired interval) or be dropped."
  - id: CODEX-A-03
    severity: CONCERN
    category: statistics
    claim: "A model-free selector removes direct NN-based importance scoring but does not establish that feature selection cannot favor the NN arm."
    evidence: "docs/c_pre_plan_2026-09-11.md:69 infers absence of NN-favoring selection bias from single-feature IC scoring. The criterion in line 55 selects marginally predictive features and therefore changes the representation on which the two model classes compete."
    suggested_fix: "State that the selector does not use either evaluated model's fitted importance or evaluation-period performance. Describe the result as conditional on this common selection rule; remove the guarantee of architecture neutrality."
    status: FIXED
    resolution_notes: "Plan §4 note rewritten exactly as suggested; the val-IC sentence now says 'outside the feature-selection sample, while remaining the HP-search / early-stopping metric rather than an independent validation'."
  - id: CODEX-A-04
    severity: CONCERN
    category: reproducibility
    claim: "The selector archive does not yet identify its executed source and resolved sample as completely as the downstream provenance."
    evidence: "docs/c_pre_plan_2026-09-11.md:61 records git revision but no selector source snapshot, while line 95 places the commit after execution. run_step3_plan_z_part_a.py:83-88 also uses sectors and news-ticker membership, absent from the selector's listed input hashes. Feature-specific eligible dates requested in the previous review are reduced to counts."
    suggested_fix: "Before the selector runs, preserve its reviewed source through a commit or content-addressed snapshot. Archive the resolved ordered ticker/date axes or their determining inputs, and each feature's eligible-date indices. Link this record to the selection, tuning, and evaluation provenance."
    status: FIXED
    resolution_notes: "Verified: part_a.load_data_and_features() intersects prices columns with news-event tickers and the sectors csv (run_step3_plan_z_part_a.py:82-88). Plan §3.6 now archives eligible_dates.json (per-feature indices), the resolved ordered ticker and date axes (count/first/last/md5), md5 of prices/sectors/news-ticker membership/groups_168/alpha158/phase5, and the git rev of the COMMITTED selector; §7 orders TP2 → commit selector → run; tune execution block and main12 provenance record selection.json's md5."
summary:
  critical: 0
  major: 2
  concern: 2
  fixed_before_reply: 0
overall_verdict: PROCEED-WITH-FIXES
---

# Codex review body (verbatim)

H博士，**PROCEED-WITH-FIXES.** I accept the proposed selector. It resolves CODEX-B-01 and the previously open score, grouping, and candidate-scope decisions. The necessary changes concern statistical interpretation and the selector's execution record; they do not require a different experiment.

**C-pre delivers temporally separated feature re-selection for the L0/L1 comparison, conditional on the existing fixed panel.** That satisfies the narrow feature-selection component of L1's request. It does **not** fully discharge L1's qualification of all Universe-C positive results: no graph or edge contrast is re-evaluated, the analysis was designed after seeing test results, and the panel retains its documented non-point-in-time composition.

I independently checked the calendar, feature eligibility, group partition, and ticker axes without computing selection ICs or rankings:

| Check | Result |
|---|---|
| Selection feature dates | 231 dates, 2021-07-01–2022-05-31 |
| Latest selection-label endpoint | 2022-06-30 |
| Purged tuning-validation dates | 106 dates, 2022-07-01–2022-11-30 |
| Latest tuning-validation label endpoint | 2022-12-30 |
| Feature eligibility | 167 features: 231/231; `hc_mom12m`: 85/231 |
| Group partition | 61 groups covering 168 unique features |
| Momentum membership | Singleton group 4 |
| Anchor versus part_a ticker axes | Identical ordered 501-ticker lists |

The Alpha158 roll and hc construction respect the stated feature-time boundary. Reusing the groups is also temporally acceptable: calibration ends 2022-01-27, and the label-validity mask used during clustering reaches only 2022-02-28. Neither reaches the selection cutoff. Using selection observations again for model training is legitimate here.

The six decisions are defensible as follows:

| Decision | Verdict | Evidence and interpretation |
|---|---|---|
| **D1 — coverage** | **Agree** | The observed counts match the disclosed consequence exactly. All 167 scored features use the same 231 dates, so unequal scoring windows do not affect the primary ranking on these artifacts. Excluding momentum is an eligibility decision, not evidence that momentum lacks predictive value. |
| **D2 — score** | **Agree** | §3.4 unambiguously specifies mean member absolute time-mean IC, includes eligible hc members, and defines missing-member handling. The deterministic tie-break avoids test-ranked CSV order. Remove only the neutrality inference identified in A-03. |
| **D3 — grouping** | **Agree** | The membership calibration and its label-validity inputs remain before cutoff. Group IDs are assigned by minimum member index in the grouping implementation (run_plan_aaa_168_ranking.py:326), independently of test importance. Historical same-day calibration features do not introduce post-cutoff information into these fixed memberships. |
| **D4 — candidates and columns** | **Agree** | Full-168 selection avoids inheriting C's test-selected restriction. Union-of-all-members preserves the declared group selection unit. Accept the resulting width and membership without adjustment toward C or C5. |
| **D5 — device** | **Agree** | §6 declares the primary-device rule before results and retains a replicate when T4 is primary. Preserve the designated primary result when assembling comparisons. |
| **D6 — comparators** | **Agree** | Paired daily contrasts against C, B, and C5 answer the stated sensitivity question. Their estimand is a difference between tuned pipeline contrasts, not identified leakage inflation. A-02 changes their interpretation, not their inclusion. |

The τ table is sensible but has a precise limitation: **τ = 0.50 and 0.75 must produce identical rankings on the current artifacts.** At τ = 0, only the momentum singleton becomes newly scoreable. Thus this table measures sensitivity to admitting that short-history feature; it does not establish general selector stability. The advance disclosure is adequate, provided the primary rule remains frozen irrespective of those rankings.

The statistical machinery can remain the C5 machinery: average seeds within each date, use paired daily differences, preserve the calendar checks, report the stationary-bootstrap interval and both HAC specifications, and calculate MDE separately for each series. These intervals condition on the chosen pipelines and seed average; they do not account for retrospective protocol choice. The nominal-p inventory is appropriate transparency, but does not make the analysis confirmatory.

For the four pre-declared sentences:

- **(a): change.** Include a negative-CI branch and use the computed C-pre MDE. CI exclusion and the two HLN tests may disagree; report that disagreement without upgrading it to robust detection.
- **(b): change.** "The interval includes zero; this does not establish absence" is defensible. "The MDE exceeds every observed contrast" cannot be frozen before observing C-pre.
- **(c): change.** The absolute paired estimate and interval are valid. Halving or doubling concerns E[d_CPRE] = k·E[d_U], requiring inference on d_CPRE,t − k·d_U,t. The comparator is estimated, so its uncertainty cannot be treated as a fixed benchmark implicitly. The simplest repair is to omit the proportional inference.
- **(d): agree with a scope qualification.** The non-identification statement is correct. Define "clean basis" as feature re-selection with scoring and grouping inputs bounded by the cutoff, conditional on the existing panel and retrospective protocol.

Likewise, the finalists' validation IC is outside the **feature-selection** sample, as stated. It remains the metric used for hyperparameter search, finalist selection, and early stopping—not independent validation of the selected models.

For TP2, the code-change list is broadly sufficient, including the reserved **[3600, 4799]** block, separate sensitivity defaults, and frozen-HP gate. Two implementation details directly affect correctness:

- Generalizing the analyzer must cover its result-dependent prose and test inventory. It currently contains literal C5 conclusions (analyze_c5_sensitivity.py:401) and a fixed assertion that the headline p is smallest (analyze_c5_sensitivity.py:597). Changing universe names and paths alone would propagate false conclusions.
- Preserve the selector's actual source before execution. Downstream provenance referencing `selection.json` cannot reconstruct an unidentified selector process retrospectively. The planned SQLite backups and hashes are sufficient if preserved durably; they need not enter the confirmatory artifact set.

For the paper, a defensible description is:

> We conducted a retrospective sensitivity analysis using feature re-selection whose scoring and grouping inputs were restricted to information through June 2022, conditional on the study's fixed stock universe. We re-tuned and evaluated LightGBM and MLP under the existing evaluation protocol.

Retain the paper's L8 composition limitation (paper/iclr2027/main.tex:294), the post-hoc designation, and L1's qualification for unevaluated graph/edge claims. C-pre can supply the requested re-selection evidence for MLP versus LightGBM; it cannot retroactively turn C's original results into clean confirmatory evidence.

This review was read-only. No files were changed, selection rankings computed, or models trained.

---

## Claude disposition (2026-09-11, after the review)

- All four findings ACCEPTED and applied to `docs/c_pre_plan_2026-09-11.md` (§0 scope paragraph and paper sentence; §1 exhaustive branches / own-MDE / no proportional inference / "clean basis" definition; §3.3 verified coverage facts and the τ-table limitation; §3.6 archive; §4 selector note; §5.6 analyzer generalisation must regenerate all result-dependent prose; §7 commit-before-run; §8 verdict line; §9 deviations).
- Claude's own verification of the plan's facts (before the review, `run_storya_e1_anchor.create_fold_masks(TUNE_FOLD)`): D_sel = 231 dates 2021-07-01…2022-05-31; val = 106 dates 2022-07-01…2022-11-30; coverage < 1.0 for exactly one of 168 candidates (hc_mom12m, 0.368). Matches Codex's independent table.
- Round B is deferred until H博士 approves the plan (run together with Touchpoint 2), so no review quota is spent on a plan that may be declined. The plan cannot proceed to code without H博士's go in any case (CLAUDE.md Rule 2).
