---
reviewer: codex
touchpoint: plan
round: A
target_plan: docs/plan_fc_edge_robustness_2026-06-17.md
findings:
  - id: CODEX-A-01
    severity: CRITICAL
    category: statistics
    claim: "Plan is internally inconsistent: matched-FC ΔIC called PRIMARY for the causal edge claim while assigned NO formal inference (robustness-only, no p/FDR/SPA)."
    evidence: "A primary causal claim needs an uncertainty statement. 6 FC contrasts (3 edge × 2 univ); descriptive ΔIC can't separate signal from fold/seed noise."
    suggested_fix: "Make FC matched-ΔIC the primary causal estimand AND give it a pre-registered paired/block inference with multiplicity control across the FC family."
    status: ACCEPTED
    resolution_notes: "ACCEPTED — genuine logical flaw I made. Revised plan v2: FC is NOT robustness-only; it carries its own confirmatory inference (Family 2: paired fold-level seed-avg ΔIC + block bootstrap over 12 folds + BH-FDR over 6 contrasts). The sliding-axis (robustness-only) treatment was the wrong template for a PRIMARY causal claim."
  - id: CODEX-A-02
    severity: CRITICAL
    category: design
    claim: "Switching the primary scientific estimand AFTER the tuned ladder completed + after seeing the capacity confound = garden-of-forking-paths unless FC is locked before inspecting/using confirmatory outcomes."
    evidence: "Frozen tuning already revealed the confound (B L3/L4/L5/L6, C L3/L6). Choosing the favorable interpretation post hoc invalidates the causal claim."
    suggested_fix: "Pre-register before FC execution: tuned ladder = primary for predictive selection; FC matched contrasts = primary ONLY for causal edge attribution. Lock estimands/contrasts/inference/hierarchy first."
    status: ACCEPTED
    resolution_notes: "ACCEPTED. Revised plan v2 = the pre-registration record; H博士 approval BEFORE the 720 FC cells run = the lock. Two-family hierarchy fixed in advance; tuned-ΔIC for edge arms is DESCRIPTIVE only (not a post-hoc-selectable primary)."
  - id: CODEX-A-03
    severity: MAJOR
    category: design
    claim: "FC must freeze the COMPLETE L2 HP vector, not just architecture."
    evidence: "Retuning lr/wd/dropout per edge config reintroduces optimization confounding; the estimand is edge-set-only."
    suggested_fix: "Fix B {hid64/2L/heads2/drop0.3/lr2e-4/wd1e-4}, C {hid128/1L/heads4/drop0.1/lr5e-4/wd3.4e-5} for ALL FC edge configs."
    status: ACCEPTED
    resolution_notes: "ACCEPTED (was my leaning in open-Q2). v2 freezes the full L2 winner vector; only the edge set varies."
  - id: CODEX-A-04
    severity: MAJOR
    category: design
    claim: "L2-winner anchor > neutral center, but estimand must be LABELED local-to-L2-architecture."
    evidence: "Neutral hid64/2L is arbitrary for C (L2=hid128/1L; clean comparisons already use 128/1)."
    suggested_fix: "Anchor at frozen L2 winner per universe; label estimand 'local edge effect at the frozen L2 operating point', not global edge superiority."
    status: ACCEPTED
    resolution_notes: "ACCEPTED. v2 anchors at L2 winner + explicit 'local-to-L2' labeling in the estimand definition + paper caveat."
  - id: CODEX-A-05
    severity: MAJOR
    category: statistics
    claim: "120 seed-fold cells are NOT 120 independent samples; effective n ≈ 12 fold blocks."
    evidence: "10 seeds share the same 12 walk-forward folds / market histories; fold/time dependence dominates."
    suggested_fix: "Inference on paired fold-level seed-averaged ΔIC with block/clustered bootstrap over folds; seed dispersion secondary."
    status: ACCEPTED
    resolution_notes: "ACCEPTED — consistent with the existing main-axis fix (analysis.md 2026-06-15-a R9-A-04: seed-AVERAGED T=749 not seed-stacked N=7490). v2 FC inference = paired fold-level seed-avg ΔIC + block bootstrap over 12 folds."
  - id: CODEX-A-06
    severity: MAJOR
    category: design
    claim: "Including L6 in the FC edge-family test contaminates the additive-edge claim."
    evidence: "L6 = dense complete-graph topology/density stress, a different mechanism than additive news/sector info edges."
    suggested_fix: "Confirmatory FC scope = L3fc/L4fc/L5fc vs L2 only. L6fc = separately-labeled exploratory topology stress with its own multiplicity if reported."
    status: ACCEPTED
    resolution_notes: "ACCEPTED. v2 FC confirmatory family = {L3fc,L4fc,L5fc}×{B,C}=6 contrasts. L6 excluded from the family (Cn1 dense-vs-sparse handled separately / exploratory)."
  - id: CODEX-A-07
    severity: CONCERN
    category: correctness
    claim: "L2−L1 (graph-vs-no-graph) should not be forced into the FC edge arm."
    evidence: "L1 is MLP with no edge set; heads/layers/message-passing don't map cleanly onto an MLP; 'FC-MLP at L2 arch' ill-defined."
    suggested_fix: "Handle L2−L1 as predictive-ladder evidence, or a separate architecture-matched non-graph baseline outside the FC edge arm."
    status: ACCEPTED
    resolution_notes: "ACCEPTED. v2 keeps L2−L1 OUT of the FC edge family. Graph-vs-no-graph covered by (a) the tuned SPA family (best-vs-best predictive) and (b) OPTIONAL separate hidden/layers-matched MLP-vs-GAT study (deferred; not in FC scope)."
  - id: CODEX-A-08
    severity: CONCERN
    category: reproducibility
    claim: "Reusing tuned L2 cells as the FC baseline is valid only with identical seeds/folds/preprocessing/eval; else rerun noise enters asymmetrically."
    evidence: "FC baseline IS the L2 config; within-arm paired comparison clean only if L2 cells are evaluation-equivalent."
    suggested_fix: "Reuse FROZEN L2 predictions as the FC baseline; rerun L2 only as an audit reporting max |Δpred|/|ΔIC|."
    status: ACCEPTED
    resolution_notes: "ACCEPTED. v2 FC baseline = the frozen tuned-ladder L2 per-day predictions (reused, paired); L2 rerun only as a repro audit (report max abs discrepancy)."
summary:
  critical: 2
  major: 5
  concern: 2
  fixed_before_reply: 0
overall_verdict: BLOCK-EXECUTION
---

# Codex Plan Review — Touchpoint 1, Round A — FC edge-robustness arm

Verdict: **BLOCK-EXECUTION** — correct. The plan as written had a real logical inconsistency (A-01)
+ a forking-path risk (A-02). All 9 findings ACCEPTED (none rejected); I verified each against the
plan + the project's existing methodology (the seed-averaging precedent in analysis.md 2026-06-15-a
directly supports A-05). Plan revised to v2 (`docs/plan_fc_edge_robustness_2026-06-17.md`).

## Key resolution (Codex Q4, the sharpest point)
The original plan tried to have it both ways: matched-ΔIC "PRIMARY for causality" yet "robustness-only,
no inference." Resolved into a TWO-FAMILY pre-registered hierarchy:
- **Family 1 — predictive/model-selection** (unchanged): tuned ladder, DM-HLN + BH-FDR q=0.05 + SPA M=9.
  "Does any tuned arm beat tuned LightGBM." Capacity confound is NOT a defect (best-vs-best is the
  right question here).
- **Family 2 — causal edge attribution** (NEW, FC carries inference): L3fc/L4fc/L5fc − L2 at the frozen
  full L2 HP vector; paired fold-level seed-averaged ΔIC + block bootstrap over 12 folds + BH-FDR over
  the 6 contrasts. "Do news/sector edges causally add, holding the model fixed (local to L2 arch)."
- tuned-ΔIC for edge arms = DESCRIPTIVE complement only (not a post-hoc-selectable primary).

Both families locked BEFORE the 720 FC cells run (this doc + H博士 approval = the pre-registration).

Codex power analysis (Q5): under effective n ≈ 12 fold blocks, MDE@80% ≈ 0.008–0.032 → ΔIC ~0.005
underpowered, ~0.016 detectable only at low paired-difference sd. Honest expectation: several FC
contrasts will land "directional but not reliable." The FC arm's value is a defensible, pre-registered
causal-attribution frame + reviewer defense, not a guarantee of resolving the edge question.
