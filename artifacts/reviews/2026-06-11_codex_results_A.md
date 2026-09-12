---
reviewer: codex
touchpoint: results
round: A
target_files:
  - experiments/sanity_summary/verdicts.json
  - experiments/sanity_e3_planted/results.csv
  - experiments/sanity_e2_shuffled/results.csv
  - experiments/sanity_e1_oracle/results.csv
  - experiments/sanity_e1b_label_oracle/results.csv
findings:
  - id: CODEX-R-A-01
    severity: MAJOR
    category: other
    claim: "E3 PASS -> 'the Story A null is a genuine task property' is over-claimed."
    evidence: "E3 plants y = beta*(A_norm @ X[:,0]) + eps — a pure one-step neighbor aggregation that is exactly what a GNN is built to compute, with the MLP structurally denied edges. PASS proves the graph message-passing PATH is operational (rules out edges-not-wired / gross training failure / bottleneck / lookahead-broken aggregation), but does NOT prove the architecture/hyperparameters are optimal for REAL features, so it cannot certify the null is definitively a task property."
    suggested_fix: "Scope the claim: E3 establishes the graph pipeline is OPERATIONAL and converts graph-borne signal to IC; the null is therefore not an artifact of a NON-FUNCTIONAL graph pipeline (gross H2 modes ruled out). State the residual caveat (subtle hyperparameter/architecture mis-specification on real features is not excluded by E3)."
    status: ACCEPTED
    resolution_notes: "Claim-scoping accepted. docs/analysis.md 2026-06-11-a written with the narrowed claim ('pipeline operational, gross H2 ruled out', explicit residual-caveat line); NOT 'null proven real'. Paper §Methods/§Limitations will carry the same scoping."
  - id: CODEX-R-A-02
    severity: MAJOR
    category: statistics
    claim: "E2's formal equivalence gate FAILED, so 'consistent with no effect' is not a licensed statement."
    evidence: "verdicts.json E2 all four = INCONCLUSIVE; CIs cross zero AND exceed the +/-0.005 TOST margin (e.g. C_GAT mean_delta -0.0145 [-0.043,0.0034]). Underpowered (4 seeds)."
    suggested_fix: "Report E2 as 'no significant structural-regularization effect detected; formal equivalence to no-graph NOT established (underpowered)'. Flag the B_GAT/B_SAGE positive point estimates and check Fold-4 dependence."
    status: ACCEPTED
    resolution_notes: "analysis.md uses 'equivalence not established / no significant effect detected, underpowered' verbatim — not 'consistent with no effect'. E2 is supporting context only; E3 carries the verdict."
  - id: CODEX-R-A-03
    severity: MAJOR
    category: reproducibility
    claim: "E1b _meta.json still labels it a 'necessary control', contradicting sanity_summary.md + code which demoted it to supporting diagnostic."
    evidence: "run_sanity.py write_meta E1b branch NOTE_A03 = 'leaked ... oracle IS a necessary control ... no lift => SICK' — stale, not updated after the 2026-06-10 demotion."
    suggested_fix: "Unify all three (meta/summary/code) to 'supporting diagnostic, not a necessary control'."
    status: FIXED
    resolution_notes: "Verified run_sanity.py:127-129 (read in-session) — stale text confirmed. write_meta E1b NOTE_A03 rewritten to 'SUPPORTING DIAGNOSTIC, NOT a necessary control'; E1/E1b _meta.json regenerated on disk. Now consistent with sanity_common docstrings + verdict + analyze summary + plan."
  - id: CODEX-R-A-04
    severity: CONCERN
    category: statistics
    claim: "0.7x recovery threshold (pre-registered, OK) needs a sensitivity table; disclose beta calibration."
    evidence: "Single threshold reported; beta calibration (closed-form + <=3 rescales to measured oracle IC) not surfaced in results."
    suggested_fix: "Report recovery at 0.7/0.8/0.9x and the beta calibration procedure."
    status: ACCEPTED
    resolution_notes: "analysis.md adds sensitivity: at 0.7x both PASS; 0.8x both PASS (target 0.0375; GAT 0.0384, SAGE 0.0427); 0.9x (target 0.0422) SAGE PASS, GAT marginal-fail. beta calibration disclosed (sanity_common.build_planted_fold)."
  - id: CODEX-R-A-05
    severity: CONCERN
    category: other
    claim: "Fully hiding oracle absolute IC from all tables can read as selective reporting."
    evidence: "Prereg fencing said oracle numbers NEVER in a table; only the relative inequality in the headline."
    suggested_fix: "Put oracle absolute IC in an APPENDIX with an explicit leaked-oracle warning; keep the headline on the relative inequality."
    status: ACCEPTED
    resolution_notes: "Fencing softened in write_meta + plan: oracle absolute IC -> appendix with leaked-oracle warning (NOT main results table). analysis.md (internal honest record) includes the oracle numbers with the leaked flag."
  - id: CODEX-R-A-06
    severity: CONCERN
    category: correctness
    claim: "E0 independent provenance recompute only covered fold-0; plan intended per-fold."
    evidence: "run_e0 independent recompute was a single fold-0 block."
    suggested_fix: "Extend the independent recompute to all 5 folds or document the limitation."
    status: FIXED
    resolution_notes: "Extended to all 5 folds; re-ran E0 -> 'independent recompute match (all 5 folds)=True', 14/14 PASS. Fixed a stale cosmetic print (was comparing fold-4 indep vs fold-0 pipe)."
summary:
  critical: 0
  major: 3
  concern: 3
  fixed_before_reply: 2
overall_verdict: OVERSTATED-REVISE
---

# Codex Results Review — Sanity-Check Suite E0-E4, Touchpoint 3 Round A

**Reviewer**: Codex (primary; ~5 min). **Verdict**: OVERSTATED-REVISE.

## The key distinction (Codex)

The RESULT is real and defensible; the CLAIM around it was too strong. The experiment is sound — Codex found NO statistical or correctness error in E3's recovery test (HLN+BH-FDR correct, seed-averaged-per-fold pairing avoids pseudo-replication, 0.7x pre-registered, achievable=measured-oracle-IC is the right denominator). The revisions are claim-scoping (R-A-01, R-A-02) and presentation/consistency (R-A-03, R-A-05) plus two cheap completeness items (R-A-04, R-A-06).

## Disposition

| ID | Sev | Disposition |
|----|-----|-------------|
| R-A-01 | MAJOR | ACCEPTED — scope to "pipeline operational, gross H2 ruled out" (not "null proven real"); residual hyperparameter caveat stated. |
| R-A-02 | MAJOR | ACCEPTED — E2 reported as "equivalence not established (underpowered)", not "consistent with no effect". |
| R-A-03 | MAJOR | FIXED — E1b _meta.json unified to "supporting diagnostic"; verified the stale text + regenerated. |
| R-A-04 | CONCERN | ACCEPTED — recovery 0.7/0.8/0.9x sensitivity + beta calibration disclosed in analysis.md. |
| R-A-05 | CONCERN | ACCEPTED — oracle absolute IC -> appendix w/ leaked warning. |
| R-A-06 | CONCERN | FIXED — E0 independent recompute extended to all 5 folds; 14/14 PASS. |

## Verification performed (Rule 9 #5)

- R-A-03: read run_sanity.py:127-129 — confirmed stale "IS a necessary control" text; fixed + regenerated _meta.json.
- R-A-06: extended recompute loop; re-ran E0 → all-5-fold match=True, 14/14 PASS.
- R-A-01/02/04/05: claim/presentation items — applied in docs/analysis.md 2026-06-11-a with the scoped language.

## Net

E3 (decisive necessary control) PASS stands: GAT 82% / SAGE 91% recovery, MLP ~0, both BH-rejected. The defensible headline is **"the graph pipeline is operational — the gross broken-pipeline failure modes (H2) are ruled out"**, with the residual caveat that E3 does not certify hyperparameter optimality on real features. With that scoping, the null is publishable; the un-scoped "null is proven a task property" is not.
