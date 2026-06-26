---
reviewer: codex
touchpoint: code
round: A
target_files:
  - compute_cost_confirmatory.py:170-205
  - compute_cost_confirmatory.py:243-285
  - compute_cost_confirmatory.py:300-335
target_plan: /Users/heruixi/.claude/plans/handoff-reactive-knuth.md
findings:
  - id: CODEX-A-01
    severity: CRITICAL
    category: correctness
    claim: "sign_stable_under_lofo uses all() (unanimous) but the intended robustness rule is a majority vote; the BH-significant C L3-L2 flip is labelled LOFO-fragile when a majority of LOFO means keep the sign."
    evidence: "compute_cost_confirmatory.py: sign_stable = all(np.sign(v)==np.sign(full) for v in lofo). For C L3-L2 @10bps, 10/12 LOFO means are positive → majority-stable, but all()=False."
    suggested_fix: "Change all(...) to a majority vote: sum(sign(v)==sign(full)) > len(lofo)/2."
    status: REJECTED
    resolution_notes: >
      No "majority vote" semantics was ever specified — Codex invented the intended rule. The column
      is named sign_stable_under_lofo and the docstring says the sign "survives leave-one-fold-out",
      i.e. the STRICT criterion: survives EVERY single-fold removal. Verified the actual data
      (independent re-derivation, not vibes): C L3-L2 @10bps per-fold ΔSharpe =
      [-0.42,0.06,2.59,2.40,-2.79,0.87,-0.13,-2.13,-0.30,0.22,-0.17,0.79] → 6 positive / 6 negative,
      range -2.8..+2.6, full mean +0.0821, bootstrap CI [-0.7713,+0.8868] STRADDLES 0. This is a
      near-zero noisy mean. all()=False is the HONEST label (the +0.08 sign is not robust — dropping
      the two big positive folds flips it). Adopting majority-vote would label it "stable", directly
      CONTRADICTING the bootstrap CI — that would be the real error, making the paper-facing artifact
      LESS honest. all() also correctly labels the genuine main claim C L1-L0 stable (12/12 LOFO
      positive, CI [+0.36,+2.08] excludes 0). KEPT all(); the primary inferential signal is the
      bootstrap CI (ci_excludes_0), with LOFO as a strict supplementary flag — narrative + summary
      reworded to lead with the CI and to define the LOFO flag precisely (not "driven by a single
      fold" but "sign not robust to every single-fold drop; per-fold ΔSharpe sign-split around a
      near-zero mean").
  - id: CODEX-A-02
    severity: MAJOR
    category: reproducibility
    claim: "Crosswalk completeness check only prints WARN; a missing L7 csv or a failed pair-merge would silently emit an incomplete (claim-dropping) crosswalk."
    evidence: "compute_cost_confirmatory.py build_crosswalk: previously `if missing: print(WARN ...)`."
    suggested_fix: "raise ValueError on missing pairs instead of warning."
    status: FIXED
    resolution_notes: >
      Fixed in build_crosswalk: split into dropped_no_gross (legitimately absent from Family-1, e.g.
      L7 demoted → printed, allowed) vs missing_net (a gross IC claim with no net sibling → INTEGRITY
      FAIL). Now `raise ValueError` on missing_net, plus `assert len(out) == len(gmap)` (every Family-1
      pair gets exactly one net sibling). Re-run confirms all 20 pairs present, asserts pass.
  - id: CODEX-A-03
    severity: CONCERN
    category: correctness
    claim: "Degenerate-cell verification asserts excluded arms ⊆ {C/L5s} but not the exact count; a rerun that under-excludes a few partial cells would pass silently."
    evidence: "compute_cost_confirmatory.py main(): assert bad_arms <= {('C','L5s')} with no count check."
    suggested_fix: "assert len(excl) == 33."
    status: FIXED
    resolution_notes: >
      Fixed more robustly than suggested (avoid the magic 33): cross-check len(excl) against the
      Family-1 ground truth — read family1_stability.csv, n_expected = sum(n_fully_degenerate +
      n_partial_collapse), assert len(excl) == n_expected. Re-run prints "EXCLUDE check: 33 cells ==
      Family-1 stability total 33, all C/L5s". Catches degeneracy drift vs the IC family without
      hardcoding a number.
summary:
  critical: 1
  major: 1
  concern: 1
  fixed_before_reply: 0
overall_verdict: PASS-WITH-CONCERNS
---

# Review body — Touchpoint 2, cost-口径 analyzer (compute_cost_confirmatory.py)

Codex Round A returned BLOCK-EXECUTION with 3 findings. After independent verification (reading the
code + re-deriving the C L3-L2 fold-level data), the resolution is: **1 REJECTED with evidence
(CODEX-A-01), 2 FIXED (A-02, A-03)**. The single CRITICAL was a misread of intended semantics, not a
real bug; the two hardening findings were legitimate and fixed. Post-fix verdict: **PASS-WITH-CONCERNS**.

## CODEX-A-01 (CRITICAL) — REJECTED

Codex asserted the LOFO sign-stability flag "should be" a majority vote and that `all()` mislabels the
C L3-L2 cost-sensitive flip as fragile. This was a hallucinated intent. The flag is, by design and by
name (`sign_stable_under_lofo` / "survives leave-one-fold-out"), the strict criterion: the sign holds
under EVERY single-fold removal.

Independent evidence (re-derived from `experiments/storya_v21_main12_tuned/results.csv`):
- C L3-L2 @10bps per-fold ΔSharpe is 6 positive / 6 negative, range −2.8..+2.6, full mean +0.0821,
  bootstrap CI [−0.77, +0.89] straddles 0 → indistinguishable from zero.
- `all()` → False correctly flags this as not-robust. Majority-vote (10/12 LOFO means positive) would
  flag it "stable", contradicting the CI → that would be the real, more-misleading error.
- `all()` correctly labels the genuine main claim C L1-L0 stable (12/12 LOFO positive, CI [+0.36,+2.08]
  excludes 0).

Action: kept `all()`; reworded the summary/narrative to lead with the bootstrap CI as the PRIMARY
signal and to define the LOFO flag precisely. The conclusion the artifact conveys (C L3-L2 net is
near-zero noise, not a robust reversal of the IC "news hurts" finding) is correct and is independently
supported by the CI regardless of the all-vs-majority debate.

## CODEX-A-02 (MAJOR) — FIXED
Silent-incomplete crosswalk hardened to a hard ValueError on any gross-claim-without-net-sibling, plus
a `len(out)==len(gmap)` integrity assert. Re-run: all 20 pairs present.

## CODEX-A-03 (CONCERN) — FIXED
EXCLUDE count now cross-checked against `family1_stability.csv` (n_fully_degenerate + n_partial_collapse),
not a hardcoded 33. Re-run: "33 cells == Family-1 stability total 33".

## What Codex verified clean (kept)
bootstrap helper usage; BH-FDR import (unused — descriptive layer, intentional); ΔSharpe orientation
(arm_A − arm_B, +Δ = arm_A better, matches gross ΔIC orientation); FC contrast direction; L7 merge
coverage; the EXCLUDE mask flags only C/L5s; gross values copied verbatim from family1_dm_hln.csv
(0 mismatches across all 20 pairs, independently confirmed).
