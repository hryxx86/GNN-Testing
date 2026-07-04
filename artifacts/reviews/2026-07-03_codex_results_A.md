---
reviewer: codex
touchpoint: results
round: A
target_files:
  - artifacts/storya_v21_family1_m14/family1_dm_hln.csv
  - experiments/storya_v21_main12_m14_retune/results.csv
  - experiments/storya_v21_tune/B_L2.json
  - experiments/storya_v21_tune/C_L2.json
target_plan: /Users/heruixi/.claude/plans/handoff-paperjury-paper-folder-codex-co-snuggly-robin.md  # M14 plan, primary=B pre-committed 2026-06-30
scope: "M14 GAT trials-sensitivity sweep — re-tune L2 at 90 trials (density-matched), verify whether the confirmatory L2-L1<0 (graph underperforms MLP) is a search-budget artifact"
model: gpt-5.5 (reasoning xhigh); tokens 178,463
findings:
  - id: CODEX-M14-A-01
    severity: MAJOR
    category: methodology
    claim: "The 90-trial study does NOT reproduce the confirmatory first 30 trials then extend them (as originally worded)."
    evidence: "Study DBs have 90 COMPLETE trials but their first-30 top search values do not match the 30-trial backups: B_L2.db number<30 top=0.0664 vs B_L2.json.30trial.bak top search_val≈0.0595; C_L2.db number<30 top=0.0674 vs backup≈0.0637. Winner params also differ (B dropout 0.3/hidden 64/2L -> 0.1/32/1L). Mechanism: run_storya_v21_tune.py:267 restarts the TPE sampler RNG per process."
    suggested_fix: "Frame M14 as an independent deterministic 90-trial L2 retune under the same nominal protocol; do not claim reproduce/extend the first 30."
    status: FIXED
    resolution_notes: "Verified DB/params differ. analysis.md 2026-07-03-a Method reworded to 'independent 90-trial retune, NOT a superset'. Does not affect the verdict (an independent 90-trial retune is a valid, cleaner budget stress test)."
  - id: CODEX-M14-A-02
    severity: MAJOR
    category: interpretation
    claim: "Density-matching at 90 trials does not FULLY rule out a search-budget artifact; it is a heuristic under adaptive TPE, not an optimizer-fairness theorem."
    evidence: "GAT adds gat_heads at run_storya_v21_tune.py:128-135; B still rejects after M14 (family1_dm_hln.csv:3), but 'defuses I-02' overclaims."
    suggested_fix: "Say M14 addresses the specific equal-30 / categorical-density objection in primary Universe-B; do not claim exhaustive tuning fairness."
    status: FIXED
    resolution_notes: "analysis.md interpretation (1) softened from 'directly answers I-02' to 'addresses the specific equal-30/density objection in B; does not prove exhaustive fairness'."
  - id: CODEX-M14-A-03
    severity: CONCERN
    category: regime-sensitivity
    claim: "Universe C is search-sensitive — but direction/near-nominal should be stated."
    evidence: "M14 C L2-L1 mean_delta_IC=-0.006860, HLN_p_t=0.059428, reject=False (family1_dm_hln.csv:13); confirmatory -0.011925/6.87e-06/True (storya_v21_family1/family1_dm_hln.csv:13)."
    suggested_fix: "Write C's statistical significance is search-budget-sensitive; direction remains negative and near nominal two-sided significance."
    status: FIXED
    resolution_notes: "analysis.md interpretation (2)+(3) now note ΔIC stays negative and near-nominal."
  - id: CODEX-M14-A-04
    severity: CONCERN
    category: pre-registration
    claim: "Primary=B provenance not visible in the repo scan (only post-result logs); cite the dated pre-result plan to avoid a B/C cherry-pick attack."
    evidence: "Codex scan found no separate pre-result M14 plan artifact in-repo; user states an approved plan exists."
    suggested_fix: "Cite the dated approved M14 plan (primary=B pre-committed before the run) in the manuscript/review packet."
    status: FIXED
    resolution_notes: "Plan file dated 2026-06-30 (before the 2026-07-02 run) with primary=B in the pre-registered decision rule; analysis.md interpretation (4) now cites it explicitly."
  - id: CODEX-M14-A-05
    severity: CONCERN
    category: credibility
    claim: "Confirmatory Universe-B L2-L1 p was mis-cited as 6.9e-4; the source value is ~4.0e-4 (6.9e-6 is the C row)."
    evidence: "artifacts/storya_v21_family1/family1_dm_hln.csv:3 B,L2,L1 HLN_p_t=0.00039680; line 13 C,L2,L1 HLN_p_t=6.8728e-06."
    suggested_fix: "Use ~4.0e-4 for confirmatory B."
    status: FIXED
    resolution_notes: "Verified from source (3.97e-4). analysis.md 2026-07-03-a line corrected 6.9e-4 -> 4.0e-4 with source citation. (progress.md did not contain this error.)"
summary: {critical: 0, major: 2, concern: 3}
overall_verdict: PROCEED-WITH-FIXES
---

# Codex TP3 Results Review — M14 GAT trials-sensitivity (Round A)

Codex independently verified the core numbers (wrote its own Python): val-IC B 0.05429→0.05595,
C 0.05503→0.06304; M14 L2 pooled test IC B 0.02805, C 0.02743; confirmatory L1 B 0.03707, C 0.03429;
240 rows, 0 nonconverged; merged input substitutes only L2@90. All matched the claimed numbers.

**Primary verdict (Codex):** defensible if pre-registration provenance is real — in leak-free B,
L2@90 still trails L1 and survives full-family BH (ΔIC=−0.00902, p=0.002109, reject=True). Supports
"robust to this 3× L2-only budget stress test," NOT "GAT was fully/fairly optimized." Recomputing the
full 20-test BH family with L2@90 substituted is acceptable and conservative for the B claim; label it
a sensitivity, not a replacement confirmatory family. Proposed manuscript wording is directionally
right after fixes: "robustly significant in leak-free B; search-budget-sensitive in leak-selected C"
(add that C's direction stays negative, near nominal). The result helps a "graphs were under-tuned"
reviewer somewhat on magnitude but mostly helps the primary clean-universe claim.

**Disposition:** all 5 findings ACCEPTED and FIXED in `docs/analysis.md` 2026-07-03-a (verified each
number against source before applying). No CRITICAL. Verdict PROCEED-WITH-FIXES → manuscript edit may
proceed with the corrected, softened wording, pending H博士 sign-off.
