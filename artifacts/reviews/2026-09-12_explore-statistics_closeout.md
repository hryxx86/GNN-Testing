<!-- Rule 9 session-closeout audit 2/4 (Explore agent, independent context), 2026-09-12. Scope = the C-pre statistics
code + reporting. Statuses filled in by Claude after verifying each claim and applying the fixes. -->
---
reviewer: explore-statistics
touchpoint: closeout
round: closeout
target_files:
  - analyze_c5_sensitivity.py
  - compute_family1_ladder.py
  - compute_e6_dm_spa.py
  - run_storya_cpre_select.py
  - artifacts/storya_v21_family1_cpre/*.csv
  - artifacts/storya_cpre_select/*.csv
  - docs/analysis.md:7-46
findings:
  - id: EXPL-STAT-01
    severity: CONCERN
    category: statistics
    claim: "The selector's group scores are not separable from sampling noise, and the 'selector robustness' table varies only tau (one feature), so nothing quantifies how arbitrary the top-15 cut is."
    evidence: "group_scores.csv rank 15 RESI60 S=0.031983 vs rank 16 IMIN10 S=0.031502 (gap 0.00048). feature_scores.csv median ic_daily_sd 0.1501 -> naive SE 0.0099, overlap-adjusted (x sqrt 21) 0.0453; 60 of 61 group scores lie within one adjusted SE of the rank-15 score."
    suggested_fix: "Add the plan's own caveat plus a computed sentence giving the rank-15/16 gap against the adjusted SE."
    status: FIXED
    resolution_notes: "Verified on the artifacts (gap 0.00048; median ic_daily_sd 0.1498 -> SE 0.0099 / adjusted 0.0452; 60/60 rankable groups within one adjusted SE). docs/analysis.md 2026-09-12-a now carries both the caveat that the tau table only measures admitting one short-history feature and the near-tie quantification."
  - id: EXPL-STAT-02
    severity: CONCERN
    category: statistics
    claim: "The fold-share n/a guard suppresses only the CPRE ratio; the C/B/C5 percentages it still prints have denominators whose own 95% CI contains zero, and they carry a substantive sentence."
    evidence: "analyze_c5_sensitivity.py:520-524 guarded on abs(pooled) < SE_block; cpre_comparison.md printed C 44%, B 31%, C5 53% while C's delta CI is [-0.00036, +0.03041]."
    suggested_fix: "Print the contribution rank next to every share, or attach an interval to the ratio."
    status: FIXED
    resolution_notes: "Every share now prints with its contribution rank, and the source line states the denominator carries the same uncertainty as the headline. The guard itself moved into ex_fold_stats (see EXPL-CODE-01) so the flag is computed once and published in the CSV."
  - id: EXPL-STAT-03
    severity: CONCERN
    category: statistics
    claim: "The headlined interval covers day-to-day variance only; the across-seed spread is wider than the interval, and the analysis entry did not carry the conditioning caveat the artifact does."
    evidence: "CPRE CI width 0.04347 vs per-seed spread 0.04874 (-0.03143 to +0.01731), k 5/10, 1 LOSO flip."
    suggested_fix: "Copy reading note (vi)'s clause into the entry."
    status: FIXED
    resolution_notes: "docs/analysis.md 2026-09-12-a now states the interval is for the seed-averaged daily series conditional on the fitted runs, resamples neither selection nor tuning nor training seeds, and gives the per-seed range with its span against the interval width."
  - id: EXPL-STAT-04
    severity: CONCERN
    category: statistics
    claim: "The percentile-asymmetry boundary-case check is computed only for the target universe, so C5's zero-excluding interval is re-published in the CPRE comparison without the flag the same code would raise if C5 were the target."
    evidence: "analyze_c5_sensitivity.py:446-454 computed note (ii) from the target row only; for C5 |mean| 0.01343 < 1.96 x SE 0.01380 while the CI excludes 0."
    suggested_fix: "Iterate note (ii) over every row of the comparison."
    status: FIXED
    resolution_notes: "reading_notes now emits a per-universe boundary check (target first); both regenerated markdowns carry the C5 boundary-case sentence, and docs/analysis.md 2026-09-12-a states it in the MDE paragraph."
  - id: EXPL-STAT-05
    severity: CONCERN
    category: reproducibility
    claim: "L0's 'three-seed average' tuning val-IC has zero seed variance, so the phrase implies seed evidence that exists only for L1."
    evidence: "CPRE_L0.json: every finalist's tune_seed_ics are three identical values (winner [0.03109]x3); CPRE_L1.json winner [+0.03440, +0.01223, -0.00624]."
    suggested_fix: "One clause in the disclosure."
    status: FIXED
    resolution_notes: "Verified (all five L0 finalists degenerate, no L1 finalist degenerate). The analyzer now computes the degeneracy flag per arm and appends the clause to the disclosure; docs/analysis.md says LightGBM is deterministic under the frozen parameters so the three-seed average carries no initialisation information for L0."
  - id: EXPL-STAT-06
    severity: CONCERN
    category: statistics
    claim: "The 22-test inventory labels all entries 'nominal, unadjusted ... no multiplicity correction applied', but four of them are re-published members of the confirmatory 20-test BH family."
    evidence: "cpre_tests_reported.json includes C and B L1-L0 auto-lag p, which are BH-family members in artifacts/storya_v21_family1/family1_dm_hln.csv."
    suggested_fix: "Add a per-test role field and restate the count."
    status: FIXED
    resolution_notes: "The inventory now carries a roles map and n_new_nominal_this_run / n_republished (CPRE: 16 new + 6 republished; C5: 12 + 4), with a note distinguishing the two; docs/analysis.md restates the count the same way."
  - id: EXPL-STAT-07
    severity: CONCERN
    category: statistics
    claim: "TOP_K = 15 is the one selector constant whose provenance is not recorded as pre-evaluation; it is inherited from the test-informed Plan-AAA -> Universe C construction."
    evidence: "run_storya_cpre_select.py:60; selection.json['rules'] records top_k_groups: 15 without provenance. Everything else verified pre-evaluation (window, group calibration, pre-winsorisation alpha158 input, no scaler on hc, per-day cross-sectional labels)."
    suggested_fix: "One provenance sentence."
    status: FIXED
    resolution_notes: "Comment added at the constant and a sentence in docs/analysis.md; see also EXPL-LEAK-03, which generalises it to the whole selection rule."
  - id: EXPL-STAT-08
    severity: CONCERN
    category: statistics
    claim: "The ex-fold-9 series is a concatenation across the excised quarter; the artificial seam is disclosed in the Codex review but reached neither the artifact nor the entry."
    evidence: "analyze_c5_sensitivity.py:316-334 joins folds 0-8 and 10-11 and applies hln_test and StationaryBootstrap(21) to the 687-day series."
    suggested_fix: "One line under the fold-concentration table."
    status: FIXED
    resolution_notes: "The markdown source line and docs/analysis.md now say the retained observations are joined across the removed quarter so the HAC window and 21-day blocks straddle one artificial seam, and that the ex-fold row is a diagnostic with the full-period row primary."
summary:
  critical: 0
  major: 0
  concern: 8
  fixed_before_reply: 0
overall_verdict: PASS-WITH-CONCERNS
---

## Verified clean (agent): 114 numeric cells checked, 114 exact, 0 mismatches

Everything in docs/analysis.md 2026-09-12-a was re-derived from the per-day .npy arrays (CPRE, tuned, c5_t4) or cross-read against the source family CSVs: headline ΔIC, CI, both p, SE/MDE; both per-arm intervals; all three paired contrasts; all four ex-fold-9 rows including contribution ranks and largest-contribution folds; fold-9 arm ICs; the per-seed vector; and the decomposition -0.00242 = -0.01148 + 0.00906. The comparator rows in cpre_comparison.csv are byte-identical to the confirmatory and C5 family tables - nothing was re-estimated. Selector: 48 columns, overlaps 22 / 4, 231 dates, hc_mom12m 85/231, tau variants as archived. Tuning: winners and finalist ranges, 31,361 vs 31,745 params. Integrity: 240 cells, cell_id [3600, 3839], 749-day calendar, 48 names, frozen md5.

Also verified clean: the estimand chain (per-day IC -> seed average -> fold-chronological concatenation -> mean) with bootstrap and HLN applied to the daily series (neither divides by the estimate); the paired difference formed before resampling with per-fold calendar asserts on both sides; the 21-day overlap handled by reporting lag-21 alongside the auto lag everywhere, with the auto lag labelled an implementation default and all CI/HLN disagreements preserved; no BH family opened (bh_applied False, ledger NOT APPLIED, SPA NOT RUN, pairs_tested [L1-L0]); LADDER_PAIRS untouched and the confirmatory artifacts clean in git status; plan §1 branch (c) applied with the Codex PERMITTED sentences and no FORBIDDEN sentence present; MDE stated as an approximate nominal 80%-power scale; the analyzer's prose genuinely computed; and the selector's inputs free of any test-informed channel (pre-winsorisation alpha158, no scaler on hc, undefined IC never zero-filled, deterministic tie-break).
