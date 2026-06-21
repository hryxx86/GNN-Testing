---
reviewer: codex
touchpoint: results
round: A
target_files:
  - "artifacts/storya_v21_family1/family1_spa.csv"
  - "artifacts/storya_v21_family1/family1_dm_hln.csv"
  - "artifacts/storya_v21_family1/family1_ic_ci.csv"
  - "artifacts/storya_v21_family1/family1_mde.csv"
  - "artifacts/storya_v21_family1/family1_lofo.csv"
  - "artifacts/storya_v21_family1/family1_stability.csv"
  - "artifacts/storya_v21_family1/family1_cl5s_robustness.csv"
  - "artifacts/storya_v21_family1/family1_ledger.json"
  - "artifacts/storya_v21_family1/family1_summary.md"
  - "artifacts/storya_v21_family2_fc/family2_fc_causal.csv"
  - "artifacts/storya_v21_family2_fc/family2_ledger.json"
  - "artifacts/storya_v21_family2_fc/family2_summary.md"
  - "compute_family1_ladder.py"
  - "compute_fc_edge_causal.py"
  - "docs/protocol_v2_freeze.md"
  - "docs/plan_fc_edge_robustness_2026-06-17.md"
findings:
  - id: "CODEX-A-01"
    severity: MAJOR
    category: interpretation
    claim: "The headline narrative is credible only if framed as local tuned-ladder evidence, not as a global or causal claim that graphs/news hurt."
    evidence: "C L1-L0 is positive and BH-significant; C L2-L1 and C L3-L2 are negative and BH-significant. But C L4/L5/L6 are positive versus L2, C SPA remains non-rejecting at p_consistent=0.0774, and Family-2 matched news effects are near zero/non-significant rather than harmful."
    suggested_fix: "Write: tuned corr-GAT underperforms tuned MLP, and tuned news-edge arm underperforms corr-GAT in the ladder. Do not generalize to all graph structures or causal edge harm; reserve causal edge claims for Family-2."
    status: OPEN
  - id: "CODEX-A-02"
    severity: CONCERN
    category: statistics
    claim: "Several important DM-HLN findings are sensitive to the HAC lag/bootstrap view, so strong language would overstate robustness."
    evidence: "Examples: C L1-L0 HLN_p_t=0.01085 but lag21 p=0.06319; C L3-L2 HLN_p_t=0.00863 but lag21 p=0.07006; C L4-L2 HLN_p_t=0.01247 but lag21 p=0.06966. Some BH-significant DM pairs also have block-bootstrap delta CIs crossing zero."
    suggested_fix: "Keep the pre-registered DM-HLN/BH result as headline, but disclose lag21/bootstrap sensitivity and avoid calling marginal pairs robust."
    status: OPEN
  - id: "CODEX-A-03"
    severity: CONCERN
    category: statistics
    claim: "Family-2 mixes bootstrap CIs with fold-level t-test p-values for BH, which is not fatal but needs careful labeling."
    evidence: "C L4 and C L5 matched bootstrap CIs exclude zero, but t_p_two_sided=0.08152 and 0.07638 and BH_FDR_reject=False. All six contrasts are also marked underpowered versus MDE."
    suggested_fix: "Present FC CIs as unadjusted descriptive intervals and BH_FDR_reject as the family-level confirmatory decision; do not describe CI-excluding-zero rows as significant."
    status: OPEN
  - id: "CODEX-A-04"
    severity: CONCERN
    category: reproducibility
    claim: "C/L5s EXCLUDE-primary handling is defensible, but partial-collapse alignment remains a residual limitation."
    evidence: "C/L5s has 25 fully degenerate and 8 partial-collapse cells of 120. The source pads short arrays as valid positions 0..len-1 because per-day date labels are unavailable; robustness treatments give IC≈0 and C SPA p≈0.077-0.080."
    suggested_fix: "Keep EXCLUDE primary with collapse rate and three-treatment robustness. Treat C/L5s IC as conditional-on-defined-ranking and avoid relying on its daily path for mechanism claims."
    status: OPEN
  - id: "CODEX-A-05"
    severity: CONCERN
    category: reproducibility
    claim: "The multiplicity ledger is substantively complete, but Family-1 wording can confuse the two-family confirmatory hierarchy."
    evidence: "Family-1 ledger says it is 'the only confirmatory family', while the locked FC plan and Family-2 ledger define a separate confirmatory causal family with 6 BH-adjusted contrasts."
    suggested_fix: "Phrase Family-1 as the only confirmatory predictive/model-selection family, and Family-2 as the separate confirmatory causal edge-attribution family."
    status: OPEN
summary:
  critical: 0
  major: 1
  concern: 4
  fixed_before_reply: 0
overall_verdict: PASS-WITH-CONCERNS
---

**Discussion**

I do not see an arithmetic or file-integrity red flag. The artifacts have the expected 2160 main cells, 240 L7 cells, and 720 FC cells, with no duplicate `cell_id`s. Direct reads of the `.npy` IC files reproduce the reported IC means, pairwise deltas, C/L5s treatment means, and FC matched deltas to rounding.

The SPA/DM contrast is coherent, not suspicious. SPA asks the tougher universe-level question “does any candidate beat L0 after data-snooping adjustment over M=9?” DM-HLN asks pre-registered local pairwise rung questions. So C can have many pairwise BH rejections while SPA still lands at p=0.077. The correct paper framing is: local rung evidence is present; global SPA superiority over L0 is not confirmed at 5%.

The “MLP not graph” story is partly supported but easy to over-read. Supported: in C, MLP beats LGB and corr-GAT loses to MLP; tuned news edge hurts versus corr-GAT. Not supported: a broad claim that graphs hurt, or that news edges causally hurt. C L4/L5/L6 recover above L2, and FC says news matched effects are tiny/non-rejecting.

Family-2 is honestly framed if kept as “0/6 survive BH; underpowered; fail-to-reject is not no effect.” The B sign reversal between matched and tuned deltas is useful evidence of the capacity confound. For C sector and sector+news, the right wording is “directionally positive but not family-significant.”

C/L5s handling is defensible: EXCLUDE is mathematically reasonable for undefined Spearman IC, and the collapse rate plus three-treatment robustness prevents cherry-picking. The only residual concern is partial-cell date alignment; it does not appear conclusion-changing because L5s is poor under all treatments and SPA p is stable.

A serious reviewer would likely accept this as a confirmatory, mixed/null result if the paper is conservative. The single biggest credibility risk is not the computation; it is letting the narrative run ahead of the hierarchy: pairwise ladder signals must not be promoted into global SPA success or causal graph/edge conclusions.
