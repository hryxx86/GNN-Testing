---
reviewer: codex
touchpoint: code
round: C
target_files:
  - run_storya_v21_l7_hats.py
target_plan: docs/protocol_v2_freeze.md
findings:
  - id: CODEX-C-01
    severity: CRITICAL
    category: statistics
    claim: "Divergence proxy `epochs_run >= HATS_HPARAMS['epochs']` flags every full-budget HATS cell as diverged even when val kept improving late, risking a false >20% Cn5 demotion. The protocol says max-epoch WITH NO val improvement / IC NaN, not max-epoch alone."
    evidence: "run_storya_v21_l7_hats.py divergence line used `info['epochs_run'] >= HATS_HPARAMS['epochs']`; train_hats (run_storya_e1_6_hats.py:482) returns only best_val_loss/epochs_run/alpha; protocol Cn5 docs/protocol_v2_freeze.md:72."
    suggested_fix: "Redefine divergence as a genuine health failure; drop epochs_run from the criterion."
    status: FIXED
    resolution_notes: "FIXED + reasoned: under patience=15 early-stopping, epochs_run==max means the run was STILL improving within the last `patience` epochs (else early-stop fires) — i.e. a LATE-IMPROVER, the OPPOSITE of diverged; a truly non-learning run early-stops near `patience`. So the literal 'max-epoch no val improvement' is inapplicable/backwards here. New definition: diverged := (no valid eval days / IC NaN) OR (not np.isfinite(best_val_loss)) — genuine health failure, no train_hats change. epochs_run still recorded as a column. DEVIATION FROM LITERAL Cn5 WORDING — H博士 CONFIRMED & ACCEPTED 2026-06-12 (the literal 'max-epoch no val improvement' is backwards/inapplicable under patience early-stopping). Verified by Claude."
  - id: CODEX-C-02
    severity: MAJOR
    category: statistics
    claim: "evaluate_contingency uses n=len(results_df); failed/missing cells are excluded from numerator AND denominator, but Cn5 is defined over 240 cells and 'IC absent' should not silently shrink the denominator."
    evidence: "run_storya_v21_l7_hats.py evaluate_contingency used `n = len(results_df)` over completed rows only; exception path writes only a manifest 'failed' row, no results row; Cn5 '240 cells' docs/protocol_v2_freeze.md:72."
    suggested_fix: "Evaluate over the full 240-cell grid; count missing/failed cells as diverged health failures; emit INCOMPLETE/PENDING until all attempted."
    status: FIXED
    resolution_notes: "FIXED: evaluate_contingency(results_df, manifest_df, expected_n=240) now uses denominator=expected_n; failed cells (manifest status=='failed') counted as diverged; collapse fraction over expected_n; verdict = KEEP-PENDING while n_attempted<expected_n (no premature KEEP), DEMOTE if a trigger fires against the full grid. Unit-tested 7 cases incl. 190 done + 50 failed → 20.8% diverged → DEMOTE; 100/240 → KEEP-PENDING; exactly 20% → KEEP. Verified by Claude."
summary:
  critical: 1
  major: 1
  concern: 0
  fixed_before_reply: 2
overall_verdict: PROCEED-WITH-FIXES
---

# Codex Review — Code (Touchpoint 2, Round C) — run_storya_v21_l7_hats.py

Rule 9 Touchpoint 2 (fresh thread) correctness review of the NEW L7 HATS-3R-adapt runner on the v2.1 12-fold main axis (240 cells, separate runner per §8, joins DM-HLN/SPA as arm 'L7').

## Verdict: PROCEED-WITH-FIXES → both findings FIXED + re-verified (post-fix = PASS, modulo 1 H博士-pending deviation)

## Codex confirmations (clean)

- **import-only (§5)**: data builders, fold masks, metrics, sector/news builders, 3-relation edge construction (`build_three_relation_edges_per_fold`), the HATS model, and `train_hats` are all imported, not reimplemented. Training call wiring matches `train_hats`'s signature.
- **cell_id_l7** delegates to the imported `cell_id`; ARM_ORDER index L7==7 → structurally disjoint block (range [840, 2159]).
- **Injection canary sufficient** for a global relation slot-swap: checks corr/sector/news signatures in order + HATS forward → alpha (N,3). (A multi-day sample would be marginally stronger but not blocking — the imported builder returns [ei_corr, ei_sector, ei_news] directly.)
- **NaN IC handling consistent**: no valid eval days → IC_mean=NaN, round() guarded by np.isnan, contingency reads diverged_flag (not IC).
- **Repro**: train_hats sets seed internally; resume key (universe, seed, fold); L7_RESULTS_COLUMNS carries the L7 diagnostics.

## Findings (both FIXED + re-verified)

CODEX-C-01 (CRITICAL) and CODEX-C-02 (MAJOR) — see frontmatter. Both fixed; contingency unit-tested (7 cases pass).

## ⚠️ H博士 decision pending (CODEX-C-01 deviation)

The frozen Cn5 wording "max-epoch 无 val 改善" is **backwards/inapplicable under patience-based early-stopping** (no-improvement → early-stop, so a max-epoch run is a late-improver, not a diverged one). The fix defines divergence as a genuine health failure (no valid IC / non-finite val loss). This is a correctness-driven deviation from the literal protocol text — flagged for H博士 confirmation per Rule 2. If H博士 wants the literal "ran full budget AND no improvement in last K epochs" signal instead, that requires extending the imported `train_hats` to return best_epoch/no_improve (additive change to run_storya_e1_6_hats.py).

## NOT executed (deferred — local Mac busy with a concurrent run)

The actual HATS training smoke (train-loop runtime). Static correctness review + no-training validation only (syntax / import / cell_id injective / injection canary ALL PASS / contingency unit tests). The L7 training smoke + a final per-cell divergence sanity will run when the Mac frees up or on Colab, before the full 240-cell run.

## Out of scope: imported HATS model/train internals (T2'd with e1_6_hats), main12 runner (rounds A/B), Optuna tuning (not built).
