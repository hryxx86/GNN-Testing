<!-- Rule 9 session-closeout audit 1/4 (Explore agent, independent context), 2026-09-12. Scope = the C-pre session
diff (b969a62..HEAD) + the feature-build and Plan-AAA lineage it depends on. Statuses filled in by Claude after
applying each fix. -->
---
reviewer: explore-leakage
touchpoint: closeout
round: closeout
target_files:
  - run_storya_cpre_select.py
  - run_storya_e1_anchor.py:494-552
  - run_storya_v21_main12.py:692-700, 809-822
  - run_storya_v21_tune.py:157-195
  - analyze_c5_sensitivity.py:245-300
  - build_alpha158_features.py:362-398
  - build_phase5_features.py:78-96
  - run_step3_plan_z_part_a.py:81-137
  - artifacts/storya_cpre_select/selection.json
  - experiments/storya_v21_main12_cpre/_run_provenance.json
findings:
  - id: EXPL-LEAK-01
    severity: CONCERN
    category: correctness
    claim: "The selector's eligibility rule 'stocks with a finite feature value' can never fire: both candidate sources are NaN->0-filled upstream, so missing observations enter the selection ICs as literal zeros, and coverage = 231/231 for 167/168 candidates is partly an artifact of that fill."
    evidence: "build_alpha158_features.py:362 nan_to_num BEFORE the _raw.npy save at :378-380; run_step3_plan_z_part_a.py:113. The selector mask run_storya_cpre_select.py:164 therefore equals label_valid_np[d]. Build-time NaN rate for the 43 selected Alpha158 columns: median 1.67%, max 3.34% (data/reference/sp500_5y_alpha158_qa.csv). The tau machinery binds only through the non-constant test, which is what excludes hc_mom12m."
    suggested_fix: "State in the plan / selector / analysis entry that NaN->0 is applied upstream so 'finite' is vacuous and up to ~3% of a column's selection-window observations are imputed zeros."
    status: FIXED
    resolution_notes: "Disclosed in run_storya_cpre_select.py's docstring (eligibility block) and in docs/analysis.md 2026-09-12-a's selector paragraph with the rates. The selector was NOT re-run: selection.json's md5 is chained into the tune JSONs and the 240-cell provenance, and the selection itself is unchanged by the disclosure."
  - id: EXPL-LEAK-02
    severity: CONCERN
    category: reproducibility
    claim: "The grouping input artifacts/plan_aaa/groups_168.json and its producer run_plan_aaa_168_ranking.py are untracked/gitignored, so the pre-cutoff calibration claim is auditable only from the artifact's self-reported field, and the selection is not reproducible from a clean clone."
    evidence: ".gitignore:29 artifacts/** (the whitelist did not cover artifacts/plan_aaa/); git ls-files --error-unmatch fails for both. selection.json records only groups_168_json md5 c50c221f..."
    suggested_fix: "Whitelist artifacts/plan_aaa/groups_168.json and commit run_plan_aaa_168_ranking.py."
    status: FIXED
    resolution_notes: "Both whitelisted/committed this closeout, together with artifacts/plan_aaa/ranking.csv (the group_members source the C5 selection quotes). md5 on disk unchanged, so every recorded selector_inputs_md5 still matches."
  - id: EXPL-LEAK-03
    severity: CONCERN
    category: data-leakage
    claim: "The selector's INPUTS are bounded by 2022-06-30 but its RULE is not: the group score, the group unit, the eligibility thresholds and the top-15 cut were carried over from a diagnostic computed on evaluation-window days and from the test-informed Plan-AAA construction. The paper-facing sentence about 'scoring and grouping inputs restricted to information through June 2022' can be read as covering the whole procedure."
    evidence: "docs/c_pre_plan_2026-09-11.md §3.4 D2 justifies the score form as continuity with analyze_plan_aaa_t1_diagnostic.py, whose window is valid_days[-313:] ~ 2024-09-27..2025-12-26, inside the 12-fold test period. TOP_K = 15 is inherited from the Plan-AAA -> Universe C cut."
    suggested_fix: "Add an explicit clause: only the inputs are bounded; the rule itself was carried over from an evaluation-window diagnostic."
    status: FIXED
    resolution_notes: "Clause added to docs/analysis.md 2026-09-12-a (selector paragraph) and to run_storya_cpre_select.py's header. It sharpens, and does not contradict, the plan's existing post-hoc disclosure; the Codex-permitted paper sentence stays accurate because it speaks about inputs, and the entry now says so explicitly."
  - id: EXPL-LEAK-04
    severity: CONCERN
    category: statistics
    claim: "The 48-column basis sits on a knife-edge: the top-15 cut is decided by a 0.00048 gap in the group score, so the column set is not stable to trivial perturbations. Only the tau-variant instability was disclosed."
    evidence: "rank 15 RESI60 S = 0.031983 vs rank 16 IMIN10 S = 0.031497 (archived group_scores.csv)."
    suggested_fix: "Report the rank-15/16 margin alongside the tau table."
    status: FIXED
    resolution_notes: "Independently raised as EXPL-STAT-01 and fixed in the same edit: docs/analysis.md 2026-09-12-a now reports the 0.00048 margin against the overlap-adjusted SE (about 0.045) and states that all 60 rankable groups lie within one such SE of the rank-15 score, i.e. the top-15 cut is one draw from many near-ties."
  - id: EXPL-LEAK-05
    severity: CONCERN
    category: correctness
    claim: "np.nan_to_num(x, 0.0) binds 0.0 to `copy`, so the call fills in place; at run_storya_e1_anchor.py:546 the argument is a view into part_a's shared hc tensor, so the verification assert can write into the array it verifies."
    evidence: "run_storya_e1_anchor.py:541-546, :549; numpy signature (x, copy=True, nan=0.0, ...)."
    suggested_fix: "Use the keyword form."
    status: FIXED
    resolution_notes: "Also raised as EXPL-CODE-03; all new call sites now use nan=0.0, posinf=0.0, neginf=0.0 (and copy=True at the assert). Values unchanged (the inputs are NaN-free), so no artifact was regenerated for this."
summary:
  critical: 0
  major: 0
  concern: 5
  fixed_before_reply: 0
overall_verdict: PASS-WITH-CONCERNS
---

## Verified clean (agent)

- Selection window reconstructed from the raw price CSV: 252 raw train dates 2021-07-01..2022-06-30, purged by HORIZON=21 to 231 feature dates 2021-07-01..2022-05-31; window/ticker/date-axis md5s all equal the archived selection.json values; the last selection label ends exactly on 2022-06-30 (the boundary, not past it).
- Full independent re-derivation of the selection from the raw artifacts: max |IC difference| vs feature_scores.csv = 5.0e-7, 0 n_eligible mismatches, identical top-15 group ids and identical 48-column list (columns_md5 0ac94cf6... = the frozen UNIVERSE_CPRE_NAMES md5).
- Panel-truncation test: recomputing the hc columns and the 21d label on a price panel truncated at 2022-06-30 gives bit-identical values and valid-mask on all 231 selection dates.
- The full-panel p1/p99 winsorization in build_alpha158_features.py is applied AFTER the _raw.npy save, and the raw file is what the whole pipeline (and the selector) reads, so no full-sample clip bound enters features, groups or selection. build_phase5_features.py applies no normalization; build_labels is purely per-day cross-sectional.
- T-1 alignment verified empirically: raw Alpha158 correlates with the same-day return (RSV5 +0.551, MAX5 -0.486) while the rolled version correlates ~0 (+0.021 / -0.019) and reproduces the raw value at t-1. Roll-then-slice and slice-then-roll are equivalent; row 0 is outside every mask.
- The part_a news-events ticker intersection is a no-op on this panel (prices∩sectors == prices∩news∩sectors == the same 501 tickers) and both loaders assert axis equality, so evaluation-period news membership cannot redefine the selection panel.
- CPRE uses the same fold masks, 21-day purge and per-fold train-only winsorize/standardize as B/C/C5; tuning scores 2022H2 only with the frozen corr snapshot <= train_end.
- Paired alignment: every L0/L1 per-day .npy for CPRE (240), C+B (480) and C5 (240) matches the confirmatory per-fold calendar (62,62,63,63,61,63,64,64,60,62,64,61; total 749) with zero short or long files.
- Provenance chain intact: selection.json md5 identical in both tune JSONs and in _run_provenance.json; columns_md5 matches UNIVERSE_CPRE_NAMES and the run's feature_names; the selector ran on committed source (044dd09, source_clean true).
