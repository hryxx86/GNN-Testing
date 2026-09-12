---
reviewer: codex-cli (token expired → claude-as-finance-gnn-reviewer fallback)
round: B
date: 2026-05-23
fallback_reason: "Codex CLI refresh token expired (exit code 1) inside codex:codex-rescue agent; fallback per Rule 9 §Fallback reviewer protocol"
target_files:
  - path: run_plan_aaa_168_ranking.py
    line_range: 1-1207

round_a_fix_verification:

  - id: FINGNN-CODE-A-01
    verdict: VERIFIED
    notes: "per_fold_winsorize import line 57; called with train_days in smoke (756), full loop (1042), noise control (818); scaler fitted on features_winsor with train_days only."

  - id: FINGNN-CODE-A-02
    verdict: VERIFIED
    notes: "Intersection-logic assert lines 154-167 (replicates build script); KMID full-period Pearson ρ>0.9 spot-check lines 174-217 (intraday same-row, no shift, no leakage)."

  - id: FINGNN-CODE-A-03
    verdict: VERIFIED
    notes: "long_run_var <= 0 returns NaN (lines 388-389); n_hac_degenerate counter in aggregation (695-696); NaN p mapped to p=1.0 for BH-FDR (690). Current run n_hac_degenerate=0."

  - id: FINGNN-CODE-A-04
    verdict: VERIFIED
    notes: "pooled_panel: True + pooled_panel_note in groups_obj (343-348); ARI vs fold-0 robustness gate (996-1016). Current ARI=0.5506 < 0.85 → concern_triggered=True surfaced correctly."

  - id: FINGNN-CODE-A-10
    verdict: VERIFIED
    notes: "Docstring corrected to Künsch fixed-length-block (404-408); implementation samples starts with replacement (419-420) — correct Künsch."

new_findings:

  - id: CODEX-CODE-B-01
    severity: CONCERN  # downgraded from initial MAJOR after empirical verification
    category: defensive_code
    file_line: "run_plan_aaa_168_ranking.py:659"
    code_snippet: |
      paired = paired[np.isfinite(paired['IC']) & np.isfinite(paired['IC_perm'])].copy()
    evidence: |
      Empirical verification on current run: baseline NaN IC = 0/1878 (0.00%); permuted NaN = 0/114558 (0.00%).
      So no rows are currently dropped silently. However, if a future run encounters NaN baseline IC (e.g., a test day with too few valid stocks), those (cell, day, group) rows are silently filtered out without warning. Could bias paired comparison if NaN incidence is non-random across folds.
    recommendation: "Add a one-line assert/log around line 659: compute n_dropped before/after and warn if > 0. Defensive only — no bug in current results."
    status: FIXED
    resolution_notes: "Added log_warn at line 659 — counts dropped rows + warns if > 0.5% of either side. Current run: 0 dropped, silent pass."

  - id: CODEX-CODE-B-02
    severity: CONCERN
    category: defensive_code
    file_line: "run_plan_aaa_168_ranking.py:1053"
    evidence: |
      cell_id uniqueness asserted at line 1129-1133 only after all 30 cells complete. A formula bug producing collision would only surface at the end of a multi-hour run.
    recommendation: "Add per-cell range check at loop entry: assert 0 <= cell_id <= 29."
    status: FIXED
    resolution_notes: "Added assertion after cell_id computation at line ~1054."

  - id: CODEX-CODE-B-03
    severity: CONCERN  # downgraded from initial CONCERN, kept after verification
    category: data_leakage_defense
    file_line: "run_plan_aaa_168_ranking.py:967-968 / 998-999"
    evidence: |
      Empirical verification on data/reference/fold_manifest_expanding.json: all 5 folds have test_days starting at index 796+, completely disjoint from calibration window [0, 251]. So no actual leakage. However the code has no runtime guard — a future manifest change could break disjointness silently.
    recommendation: "Add assert: for fold in manifest['folds']: assert not set(fold['test_days']) & set(CALIBRATION_DAYS)."
    status: FIXED
    resolution_notes: "Added assertion at line ~1000 after manifest load. Current run passes (verified empirically)."

  - id: CODEX-CODE-B-04
    severity: CONCERN
    category: statistical_methodology
    file_line: "run_plan_aaa_168_ranking.py:368-393"
    evidence: |
      NW-HAC autocovariance uses n-divisor (np.mean) not n-l-divisor. This matches statsmodels default and is the consistent estimator. Pre-registered convention. Not a bug.
    recommendation: "No change needed. Document in paper methodology as 'NW-HAC with consistent (n) divisor, Bartlett kernel, auto lag = floor(4*(T/100)^(2/9))'."
    status: REJECTED
    resolution_notes: "Per review protocol: rejected as no-action — informational only, not a bug. Will be documented in paper methodology section."

summary:
  round_a_regressions: 0
  round_a_partial: 0
  round_a_verified: 5
  new_critical: 0
  new_major: 0  # B-01 downgraded after empirical verification (current NaN=0)
  new_concern: 4
  fixed: 3
  rejected: 1

overall_verdict: PASS
verdict_rationale: "All 5 Round A fixes correctly implemented; 3 new defensive concerns added as one-line guards (B-01/02/03); B-04 informational only. Current full-mode run results are correct (0 NaN, 30 cells, 61 groups, audit triple uniqueness 114558=114558). Script ready for paper methodology integration pending Touchpoint 3."
---
