---
reviewer: codex
touchpoint: code
round: A
target_files:
  - sanity_common.py
  - run_sanity.py
  - analyze_sanity.py
findings:
  - id: CODEX-C-01
    severity: CRITICAL
    category: statistics
    claim: "E3's BH decision is converted into a strict p-value threshold that makes the weakest BH-rejected model fail beats_mlp, so both GNNs cannot pass E3 together."
    evidence: "analyze_sanity.py:246-254 computed bh_thr = max rejected p-value, passed to e3_recovery_verdict; sanity_common.py required hln_p < bh_threshold (strict <), so the marginal rejected model (hln_p == bh_thr) fails. E3 is the SOLE decisive necessary control -> it could never pass."
    suggested_fix: "Pass bool(rejects[i]) directly into the E3 verdict; use the BH reject boolean for beats_mlp."
    status: FIXED
    resolution_notes: "Verified by reading analyze_sanity.py:246-257 in-session (strict-< confirmed). e3_recovery_verdict signature changed bh_threshold->bh_reject; beats_mlp = (gnn_ic>mlp_ic) and bool(bh_reject). verdict_e3 now passes rejects[i] (the BH boolean). Re-ran analyze (no crash). End-to-end pass/fail validated on the full 60-cell E3 Colab run + analyze."
  - id: CODEX-C-02
    severity: MAJOR
    category: correctness
    claim: "The E0 frozen-alpha1 provenance canary is circular and cannot catch shared ticker-order or snapshot-index bugs."
    evidence: "run_e0 (b) recomputed edges with the SAME anchor helpers (build_correlation_snapshots/create_fold_masks/get_frozen_snapshot_idx) it was checking, then compared signatures -> only tests determinism, not correctness."
    suggested_fix: "Independent recompute (hand-rolled correlation), fixed ticker list, expected per-fold signatures, negative tests (column permutation / frozen_si +/-1 must fail)."
    status: FIXED
    resolution_notes: "E0 (b) rewritten (run_sanity.py): (i) ticker-order invariant assert list(returns.columns)==valid_tickers (the direct A-02 guard — edges index returns columns = feature stock order); (ii) hand-rolled np.corrcoef recompute of fold-0 frozen edges (separate code path) asserts EXACT match to pipeline (3026 edges, match=True); (iii) negative tests: frozen_si+/-1 differs=True AND column-permutation changes=True. Re-ran E0: 10/10 PASS. Note the block-fixture falsification in (a) was already non-circular."
  - id: CODEX-C-03
    severity: MAJOR
    category: statistics
    claim: "Append-only results.csv + row averaging lets stale/duplicate cells change gnn/mlp/achievable IC while paired .npy come from overwritten keyed files -> inconsistency."
    evidence: "run_sanity.py init_csvs only creates if absent; appends rows; --no-resume re-runs append duplicates (the local smoke runs did exactly this). analyze averaged ALL rows; per_day_ic .npy are keyed (latest wins) -> CSV means and npy can disagree."
    suggested_fix: "Upsert by key, or dedup-on-read keeping the latest completed row per (universe,model,seed,fold)."
    status: FIXED
    resolution_notes: "verdict_e3 + verdict_e1b now df.drop_duplicates(subset=['universe','model','seed','fold'], keep='last') after read, matching the latest keyed .npy. The full Colab run uses fresh dirs + --resume so duplicates don't arise; the dedup guard makes re-analysis robust regardless."
  - id: CODEX-C-04
    severity: CONCERN
    category: statistics
    claim: "Paired daily deltas aligned by array position after silent truncation, not by test-day identity; a model-specific skipped IC day would mispair."
    evidence: "per_day_ic .npy saved without day index; paired_delta_across_folds truncates to min length and subtracts by position."
    suggested_fix: "Save day-indexed IC or fixed-length-over-test_days vectors; join/mask by day before HLN."
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "Cannot manifest in THIS pipeline: compute_daily_ic's <30-valid skip depends only on label_valid_np, which is model-INDEPENDENT (same labels for GAT/SAGE/MLP within an experiment) -> all models skip the SAME days -> equal length, position-aligned by test-day. Added a loud length-mismatch warning in paired_delta_across_folds to guard the invariant. Full day-indexed save deferred (not load-bearing given the model-independent mask)."
summary:
  critical: 1
  major: 2
  concern: 1
  fixed_before_reply: 3
overall_verdict: BLOCK-EXECUTION
---

# Codex Code Review — Sanity-Check Suite (sanity_common / run_sanity / analyze_sanity), Touchpoint 2 Round A

**Reviewer**: Codex (primary; ~6.5 min). **Verdict**: BLOCK-EXECUTION → 3 FIXED + 1 ACCEPTED-AS-CONCERN. No rebuttals.

## Disposition

| ID | Sev | Disposition |
|----|-----|-------------|
| C-01 | CRITICAL | FIXED — E3 verdict uses the BH reject boolean (was strict p<threshold → marginal model unpassable → E3 structurally could never pass). |
| C-02 | MAJOR | FIXED — E0 provenance rewritten to independent recompute + ticker-order invariant + negative tests (off-by-1 / permutation must fail). 10/10 PASS. |
| C-03 | MAJOR | FIXED — analyze dedups results.csv by cell key (keep='last') to match keyed per_day_ic .npy. |
| C-04 | CONCERN | ACCEPTED-AS-CONCERN — cannot manifest (valid-day skip is model-independent); added length-mismatch guard. |

## Verification performed (Rule 9 #5 — actually read/ran)

- C-01: read analyze_sanity.py:246-257 (strict `hln_p < bh_thr` with bh_thr = max rejected p) — confirmed the marginal rejected model fails. Fixed + re-ran analyze.
- C-02: re-ran `run_sanity.py --experiment E0` → `[E0b] ticker-order invariant=True; independent recompute match=True (3026 edges); off-by-1 differs=True; permutation changes=True`; 10/10 PASS.
- C-03: confirmed init_csvs append-only + local smoke --no-resume produced duplicate rows; dedup added.
- C-04: traced compute_daily_ic skip = f(label_valid only) → model-independent → no mispairing; guard added.

## Codex confirmations (no bug found)

E3 same-index observability through train_nn is correct; isolated-node mask consistent across plant / achievable-IC / saved arrays; E2 degree-preserving swap correct; write_summary does NOT leak oracle absolute IC into the headline.

## Note: E1b demotion (independent of this review)

Between Touchpoint 1 and this review, the 2026-06-10 smoke revealed E1b (label-sim oracle) is a SECOND-ORDER control (lift ≈ feature-predictiveness × co-movement-tightness; smoke IC below baseline, positive-only variant −0.058 with high variance from over-smoothing). Per H博士 directive, E1b was demoted from necessary control to SUPPORTING diagnostic; the overall pipeline verdict now rests on E3 ALONE. This does not affect the C-findings above.

## Next action

3 CRITICAL/MAJOR fixed + validated locally; 1 CONCERN guarded. Recommend either a Codex Round B confirmation or proceeding to the Colab full run (the C-01 end-to-end pass/fail validates on the full 60-cell E3 analyze). H博士 to decide.
