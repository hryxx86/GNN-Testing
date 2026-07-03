---
reviewer: codex
touchpoint: results
round: A
target_files:
  - analyze_paper_eval_robustness.py
  - artifacts/audits/paper_eval_robustness.csv
target_plan: docs/paper_evaluation_2026-07-02.md §5-§6 (T1 sentences S1/S2/S3 entering paper/main.tex)
findings:
  - id: CODEX-A-01
    severity: CONCERN
    category: statistics
    claim: "S1 could be read as giving a post-hoc pooled 26-test BH family confirmatory status."
    evidence: "26-test BH is a sensitivity (analyze_paper_eval_robustness.py:83-98; audit row check=pooled_bh_26: 26 tests, 11 rejections, identical_to_preregistered=True). Not one of the two primary pre-registered families."
    suggested_fix: "Mark explicitly as post-hoc multiplicity sensitivity that does not replace the two pre-registered families."
    status: FIXED
    resolution_notes: "S1 final wording: 'Two post-hoc multiplicity sensitivities, which do not replace the pre-registered families, bound the layering...' Applied to main.tex §5.2."
  - id: CODEX-A-02
    severity: CONCERN
    category: statistics
    claim: "S1's BY clause should identify the exact family and survivor count."
    evidence: "BY = BH at q/c(m), m=20, c(m)=3.5977; exactly 7 survivors, 4 BH-only drops (audit rows check=by_20)."
    suggested_fix: "State 'BY over the 20-test DM family leaves 7 of the 11 BH rejections' before naming examples."
    status: FIXED
    resolution_notes: "S1 final wording includes 'leaves 7 of the 11 rejections'. Applied."
  - id: CODEX-A-03
    severity: CONCERN
    category: statistics
    claim: "S2's '8–10 of 10 seeds' compresses the two weakest cases (B L2-L1 and B L3-L2 at 8/10, opposite-sign deltas up to +0.0136/+0.0257)."
    evidence: "Audit rows check=per_seed_sign: B L2-L1 8/10, B L3-L2 8/10, C L2-L1 9/10, C L3-L2 9/10, C L1-L0 10/10, C L5-L3 10/10; LOSO 0/10 all six."
    suggested_fix: "Unpack the per-universe breakdown."
    status: FIXED
    resolution_notes: "S2 final wording gives the explicit 8/10 (Universe-B penalties) / 9/10 (Universe-C counterparts) / 10/10 (C L1-L0, C L5-L3) breakdown. Applied."
  - id: CODEX-A-04
    severity: CONCERN
    category: correctness
    claim: "S3 should stay explicitly gross-pipeline / pilot / synthetic; avoid 'validates the graph path' and 'HLN p≈0'."
    evidence: "verdicts.json E3: achievable=0.04691195, GAT=0.0383799 (82%), SAGE=0.0426937 (91%), MLP=0.00244185, hln_p=0.0, bh_reject=true. E3 = Universe B, 5 folds, 4 seeds, synthetic features; GAT and SAGE positive in 20/20 fold-seed cells (independently re-verified by Claude from experiments/sanity_e3_planted/results.csv: GAT 20/20 range [0.0248,0.0536], SAGE 20/20 [0.0353,0.0577], MLP 12/20 [-0.0106,0.0205])."
    suggested_fix: "'pre-confirmatory planted-signal control' + 'rules out a grossly nonfunctional graph path'; replace 'HLN p≈0' with 'BH-significant'."
    status: FIXED
    resolution_notes: "S3 final wording adopts both phrasings and adds 'positive in all 20 fold-seed cells'. Applied to main.tex §4."
summary:
  critical: 0
  major: 0
  concern: 4
  fixed_before_reply: 0
overall_verdict: PROCEED-WITH-FIXES
---

# Codex TP3 Round A — paper-eval robustness checks (2026-07-02)

**Scope**: 4 zero-rerun robustness checks (per-seed sign / LOSO / pooled 26-test BH / BY) produced by
`analyze_paper_eval_robustness.py`, entering `paper/main.tex` as three passages (S1 §5.2, S2 §5.3, S3 §4).

**Codex independent verification (read-only sandbox, opened source files itself)**:
- Six contrasts' per-seed sign counts and LOSO 0/10 match `artifacts/audits/paper_eval_robustness.csv`.
- The n_test_days-weighted fold-mean reconstruction is valid for these six contrasts: each involved arm has
  120 rows, 10 seeds, 12 folds, no zero-test-day cells; C/L5s degenerate rows are not used → no contamination.
- Pooled 26-test BH: 26 p-values, 11 rejections, all from the DM family, no Family-2 rejection — correct as sensitivity.
- BY correctly implemented (q / Σ1/i over 20 tests), exactly the 7 claimed survivors.
- E3 numbers match `experiments/sanity_summary/verdicts.json`.

**Verdict**: PROCEED-WITH-FIXES — 0 CRITICAL, 0 MAJOR, 4 CONCERN (all wording discipline). All four accepted
and applied; final sentence texts recorded in resolution_notes above. Full Codex transcript (304KB, includes
its own CSV-verification trail): session scratchpad `codex_tp3_output.md`; tokens used 120,461.
