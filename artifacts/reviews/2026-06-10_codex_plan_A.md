---
reviewer: codex
touchpoint: plan
round: A
target_plan: /Users/heruixi/.claude/plans/sanity-check-sorted-lark.md
findings:
  - id: CODEX-A-01
    severity: CRITICAL
    category: correctness
    claim: "E3's planted signal is index-misaligned with the anchor trainer, making the planted edge signal unobservable to the model as written."
    evidence: "Plan E3 used y[t]=beta*(A_norm @ X[t-1,:,0])+eps. But run_storya_e1_anchor.py:578-585 trains pred=model(features_t[d], edge_index) against labels_t[d] at the SAME index d, and build_labels (run_storya_e1_anchor.py:397-410) already pre-aligns labels_t[d] = 21d-forward return z-score for features at d. With X iid N(0,1), X[d] carries no info about A@X[d-1] → GNN cannot recover → false-SICK on a working pipeline."
    suggested_fix: "Use y[d]=beta*(A_norm @ X[d,:,0])+eps (same-index, model-observable). Synthetic iid noise has no PIT concern; the recovery target must be a graph-function of the features the model actually sees at index d. Add a hard assertion (E0/smoke) that the EXACT (features_t, labels_t) tensors passed to train_nn yield true-signal Spearman in [0.04,0.06] on the same day indices compute_daily_ic uses."
    status: FIXED
    resolution_notes: "Verified against run_storya_e1_anchor.py:397-410 + :580 (read in-session). Plan revised: E3 label = same-index neighbor-mean of observed features; 'X[t-1]/PIT' framing removed (was an over-applied real-data rule on synthetic noise). Pre-train recoverability assertion added to E0/smoke."
  - id: CODEX-A-02
    severity: CRITICAL
    category: correctness
    claim: "E3 can pass even if the real graph construction is semantically broken, because it plants labels using the same edge_index object later supplied to the GNN."
    evidence: "build_planted_signal(edge_index_alpha1) plants on A_alpha1, then trains with corr_snapshots={0:A_alpha1}. run_storya_e1_anchor.py:564 only does .to(device) — no ticker-mapping/snapshot-provenance/returns-correspondence validation; build_correlation_snapshots (:417-427) indexes edges by returns column order via np.where. A ticker-permutation or snapshot off-by-one bug survives E3 because plant and train share the SAME wrong graph. This is exactly the H2 broken-pipeline class the suite must falsify."
    suggested_fix: "Add a graph-provenance canary to E0 BEFORE E1/E1b/E3: (a) independently recompute the frozen alpha1 edge set from returns/ticker order per fold and assert exact equality (or density/hash match) vs what train_nn would use; (b) a synthetic known-block-correlation fixture build_correlation_snapshots must recover, which would FAIL under a deliberate ticker permutation or +/-1 snapshot index. E3 alone must NOT be allowed to certify graph-builder correctness."
    status: FIXED
    resolution_notes: "Verified :564 + :417-427 (read in-session) — no provenance validation exists. Plan revised: E0 upgraded from swap-sensitivity-only to a graph-PROVENANCE check (provenance canary + planted-block fixture with permutation/off-by-one falsification). Verdict logic states E3 cannot certify graph construction alone."
  - id: CODEX-A-03
    severity: MAJOR
    category: statistics
    claim: "E1 oracle is not a valid NECESSARY positive control for next-day/21d predictive edge value; E1 failure should not be read as a sick pipeline."
    evidence: "build_oracle_edges_per_fold uses fold TEST-window CONTEMPORANEOUS Spearman |rho|>0.6, but the label (build_labels :400) is 21d-FORWARD market-demeaned excess return. Future contemporaneous covariance need not carry forward-rank info even when message passing works → E1 'Result B = sick if lift<1x' is an invalid inference."
    suggested_fix: "Demote E1 to a diagnostic upper-bound / leakage stress test; REMOVE the E1-alone sick branch. Add E1b 'predictive-topology oracle' = leaked forward-LABEL-similarity graph (connect i,j if their test-window forward-label series corr > thr); a working pipeline MUST lift IC under E1b, so E1b (with E3) is the necessary-control pair."
    status: FIXED
    resolution_notes: "Verified :400 forward-return label (read in-session). Matches the concern Claude raised to H博士 pre-review and H博士's decision to promote E3 to co-primary. Plan revised: E1 demoted (no sick branch, upper-bound diagnostic only); E1b leaked-label-similarity oracle ADDED as a topology-based necessary control. +40 cells (~3.5h A100) flagged to H博士 as an optional drop if E3-alone necessary control is preferred."
  - id: CODEX-A-04
    severity: MAJOR
    category: statistics
    claim: "E3 significance under-specified; risks seed-day pseudo-replication if hln_test is applied to pooled per-seed daily IC arrays."
    evidence: "Plan said 'significance via hln_test+bh_fdr' without stating seed aggregation. Project precedent compute_e6_edge_ablation.py:138-170 seed-averages within fold before paired daily DM/HLN to avoid pseudo-replication."
    suggested_fix: "Preregister E3 test unit: seed-average per (fold, model), then paired daily delta-IC (GNN - MLP) across folds before HLN/BH-FDR; report seed-level consistency separately. Do not pool seed-days as independent."
    status: FIXED
    resolution_notes: "Plan revised: E3 inference unit preregistered as seed-averaged-per-fold paired daily delta-IC, HLN+BH-FDR, mirroring compute_e6_edge_ablation.py:138-170. Seed consistency reported separately."
  - id: CODEX-A-05
    severity: MAJOR
    category: statistics
    claim: "E2 'within seed noise' verdict is not operationalized → shuffled placebo interpretable post hoc."
    evidence: "Plan gave no equivalence margin, CI rule, seed-noise estimator, or multiplicity family for E2; allocated only 2 shuffled seeds."
    suggested_fix: "Preregister an E2 equivalence gate: upper one-sided 95% block-bootstrap CI of (shuffled - MLP) delta-IC below the practical edge unit 0.01001, OR TOST with +/-0.005 margin. Estimate seed noise from the 10-seed anchor; run E2 at >=4 seeds (match E3)."
    status: FIXED
    resolution_notes: "Plan revised: E2 equivalence gate preregistered (one-sided 95% block-bootstrap CI of shuffled-minus-MLP delta-IC < 0.01001 AND TOST +/-0.005); E2 seed count raised 2 -> 4 to match E3."
  - id: CODEX-A-06
    severity: CONCERN
    category: reproducibility
    claim: "Degree-preserving shuffled-edge control underspecified for the symmetric directed edge_index representation."
    evidence: "Anchor emits both directions via np.where on a symmetric matrix (run_storya_e1_anchor.py:424-427). A naive directed swap can break undirected symmetry or preserve the wrong degree notion."
    suggested_fix: "Canonicalize edges as undirected i<j, swap on a simple undirected graph with self-loop/duplicate rejection, re-symmetrize. Assert exact undirected degree sequence, exact 2E directed count, no self-loops, no duplicate directed edges, deterministic for fixed seed."
    status: FIXED
    resolution_notes: "Plan revised: shuffled builder spec made explicit (undirected canonicalization + re-symmetrize) with the 5 listed assertions in smoke."
  - id: CODEX-A-07
    severity: CONCERN
    category: statistics
    claim: "E4 edge-feature collinearity AUC is a weak diagnostic; not proof the graph is redundant with node features."
    evidence: "Edge predictability from feature distance != zero incremental predictive value from nonlinear time-varying GNN aggregation."
    suggested_fix: "Label as descriptive 'edge predictability from node features' only; remove causal 'redundant' language. If used as a verdict input, add a matched feature-distance-strata conditional comparison of real vs shuffled/no-graph delta-IC."
    status: FIXED
    resolution_notes: "Plan revised: E4 AUC reframed as descriptive 'edge predictability from node features', no causal language, NOT a standalone verdict input."
summary:
  critical: 2
  major: 3
  concern: 2
  fixed_before_reply: 7
overall_verdict: BLOCK-EXECUTION
---

# Codex Plan Review — Sanity-Check Suite (E0–E4), Touchpoint 1 Round A

**Reviewer**: Codex (primary; no fallback needed — responded ~5 min).
**Verdict**: BLOCK-EXECUTION → all 7 findings accepted and folded into the revised plan. No rebuttals.

## Disposition summary

| ID | Sev | Disposition |
|----|-----|-------------|
| A-01 | CRITICAL | ACCEPT — E3 label re-indexed to same-index neighbor-mean (model-observable); recoverability assertion added. |
| A-02 | CRITICAL | ACCEPT — E0 upgraded to a graph-provenance canary + planted-block falsification fixture; E3 cannot certify graph build alone. |
| A-03 | MAJOR | ACCEPT + EXTEND — E1 demoted to upper-bound diagnostic (no sick branch); E1b leaked-label-similarity oracle added as topology-based necessary control. |
| A-04 | MAJOR | ACCEPT — E3 inference unit = seed-averaged-per-fold paired daily ΔIC, HLN+BH-FDR (matches compute_e6_edge_ablation precedent). |
| A-05 | MAJOR | ACCEPT — E2 equivalence gate preregistered; seeds 2→4. |
| A-06 | CONCERN | ACCEPT — shuffled builder undirected-canonical spec + 5 assertions. |
| A-07 | CONCERN | ACCEPT — E4 AUC reframed descriptive, no causal/verdict use. |

## Verification performed (Rule 9 #5 — actually read, not asserted)

- A-01: read `run_storya_e1_anchor.py:397-410` (build_labels: 21d-forward z-score) + `:578-585` (train loop same-index features↔labels). Bug confirmed.
- A-02: read `:564` (no provenance validation) + `:417-427` (edges indexed by returns column order). Gap confirmed.
- A-03: read `:400` (`prices.shift(-horizon)` forward return). Contemporaneous-corr oracle non-necessity confirmed.
- A-06: read `:424-427` (`np.where` symmetric → both directions). Representation confirmed.

## Key logical point (Codex body, paraphrased)

The co-primary planted control as originally written was simultaneously capable of (i) FAILING a working pipeline (A-01 index bug) and (ii) PASSING a semantically broken graph (A-02 self-consistent plant+train). Both defeat the suite's purpose. The two CRITICAL fixes restore the suite's falsification power; the MAJOR fixes close post-hoc-interpretation loopholes (E2 gate, E3 seed unit) and correct an invalid inference (E1 sick branch). E1 remains useful strictly as a leaked upper-bound diagnostic.

## Next action

Plan revised in place. Recommend Codex Round B re-review of the revised plan before code (or H博士 may waive Round B given all fixes are mechanical and accepted). E1b adds ~40 cells / 3.5h A100 — H博士 to confirm keep-or-drop.
