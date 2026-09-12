---
reviewer: codex
touchpoint: code
round: A
target_files:
  - run_storya_v21_main12.py
target_plan: docs/protocol_v2_freeze.md
findings:
  - id: CODEX-A-01
    severity: CONCERN
    category: correctness
    claim: "E0 canary does not implement the protocol's explicit off-by-one negative test for frozen α1 snapshot provenance (C1 / precheck #7 requires off-by-1 必 FAIL)."
    evidence: "run_storya_v21_main12.py:240-247 (canary-b) compares runner_si/runner_edge to recompute_alpha1_frozen_edges but never asserts that an adjacent snapshot (runner_si±1) would FAIL the match — it only proves the expected path is equal, not that the canary is sensitive to an off-by-one."
    suggested_fix: "In canary-b, when an adjacent snapshot exists, assert edge_index_signature(snaps[runner_si±1]) != edge_index_signature(indep_edge)."
    status: FIXED
    resolution_notes: "Fixed run_storya_v21_main12.py:canary-b: b_pass now requires match AND off-by-1 caught (both runner_si-1 and runner_si+1 signatures must differ from indep) AND n_adj>0. Re-ran --canary: '[canary-b] ... match=True, off-by-1 caught=True (2 adj) -> PASS'. Verified by Claude (actually executed)."
  - id: CODEX-A-02
    severity: CONCERN
    category: correctness
    claim: "Complete-graph canary only samples symmetry (first 5000) and does not check edge uniqueness, so it is weaker than the stated |E|=N(N-1)+no-self+symmetric contract."
    evidence: "run_storya_v21_main12.py:257-260 (canary-c) checks ce.shape[1]==N(N-1), no self-loops, and symmetry over list(fwd)[:5000] only — not all edges and not len(fwd)==expected (uniqueness)."
    suggested_fix: "Exhaustive: len(fwd)==N*(N-1) (uniqueness), all i!=j, all((j,i) in fwd for (i,j) in fwd). Cheap at ~250k edges."
    status: FIXED
    resolution_notes: "Fixed run_storya_v21_main12.py:canary-c: right_count now also requires len(fwd)==expected (uniqueness); symmetric is exhaustive over all edges (no sampling). Re-ran --canary: '[canary-c] |E|=250500 unique=250500 (expect 250500), no_self=True, symmetric=True -> PASS'. Verified by Claude (actually executed)."
summary:
  critical: 0
  major: 0
  concern: 2
  fixed_before_reply: 2
overall_verdict: PASS-WITH-CONCERNS
---

# Codex Review — Code (Touchpoint 2, Round A) — run_storya_v21_main12.py

Rule 9 Touchpoint 2 correctness review of the v2.1-frozen MAIN AXIS 12-fold confirmatory runner.

## Verdict: PASS-WITH-CONCERNS (0 CRITICAL + 0 MAJOR + 2 CONCERN — both FIXED before reply)

## Blocking items — all CLEAN per Codex

- **§5 import-only (C1)**: no reimplemented data/label/correlation construction in the runner — all data construction is imported from `run_storya_e1_anchor.py`. The same-day-leak regression entry point (C1) is closed.
- **12-fold split (§2a)**: mapping matches the expanding train-with-last-quarter-val design; test 2023Q1→2025Q4; `assert_purge_no_leak_12` uses the imported `create_fold_masks` and enforces HORIZON=21 label-window separation (last train/val label end < next split first feature day).
- **Univ-C T-1 asserts**: present via imported `build_universe_C` (a158_slice[1]==raw[0] & row0==0) plus the runner's `assert_univ_c_t1_contract` row-0 reconfirmation.
- **L6**: dispatch is edge-only different from L2 (param-identical GAT on a complete graph); faithful "full-attention-no-graph".
- **cell_id**: injective over [0, 2399].
- **seed / resume / schema**: consistent with the stated design.

## Findings (both CONCERN, both FIXED + re-verified by Claude)

See frontmatter CODEX-A-01 (off-by-one negative test in canary-b) and CODEX-A-02 (exhaustive complete-graph check in canary-c). Both strengthen the E0-canary's C1 rigor (precheck #7: "置换/off-by-1 负测试必 FAIL"). Fixes applied and `--canary` re-run confirms ALL PASS with the stronger assertions:

```
[canary-a] block fixture: within_ok=1.000 (>0.95), within_perm=0.332 (<0.6) -> PASS
[canary-b] provenance fold 2024Q2: runner_si=27 indep_si=27, match=True, off-by-1 caught=True (2 adj) -> PASS
[canary-c] complete graph: |E|=250500 unique=250500 (expect 250500), no_self=True, symmetric=True -> PASS
=== E0-CANARY ALL PASS ✓ ===
```

## Out of scope (explicit follow-up, not in this file)

Edge arms L3/L4/L5/L5s (per-day dynamic edges importing `build_per_day_news_edges` from `run_storya_e3_news_edge.py` + `build_sector_edges` from `run_storya_e4_alpha.py`); L7 HATS (separate runner `run_storya_e1_6_hats.py` under §6 contingency).

## Non-finding (cost, not correctness)

L6 (GAT on 250k-edge complete graph) is the dominant-cost arm; on local MPS a single fold-0 cell did not finish in ~18 min. Not a bug — inherent to the protocol's complete-graph design. A100 needed for representative 单价; §8 budget's "标准神经 60–90s" does not separately account for L6's complete graph. Surfaced to H博士 for budget/sequencing decision.
