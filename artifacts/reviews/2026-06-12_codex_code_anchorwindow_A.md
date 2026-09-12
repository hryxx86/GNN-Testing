---
reviewer: codex
touchpoint: code
round: A
target_files:
  - run_storya_e1_anchor.py:81-110   # WALK_FORWARD_FOLDS window extension (folds 5-11)
  - run_storya_e1_anchor.py:213-239  # cell_id formula + assert_cell_id_injective
  - run_storya_e1_anchor.py:245-289  # assert_purge_no_leak
  - run_storya_e1_anchor.py:441-471  # build_correlation_snapshots / get_frozen_snapshot_idx
  - run_storya_e1_anchor.py:528-539  # create_fold_masks (train+val 21d purge)
note: "Distinct from 2026-06-12_codex_code_A.md (which reviews run_storya_v21_main12.py, the v2.1 ladder runner). This reviews the SIMPLE anchor window-only extension (4 models, untuned, reuses the 400 existing cells)."
findings:
  - id: CODEX-A-01
    severity: MAJOR
    category: correctness
    claim: "Fold 11 (Q4-2025, test_end 2025-12-31) silently evaluates only 61 of 64 test days — the last 3 (2025-12-29/30/31) have no 21d-forward label because the price file ends 2026-01-28."
    evidence: "run_storya_e1_anchor.py:109 test_end=2025-12-31; labels use a 21d forward shift; create_fold_masks includes all test days <= test_end; train loop skips invalid-label days instead of asserting. Verified: last valid-label idx = num_days-1-21 = 1233 = 2025-12-26; fold-11 days 2025-12-29/30/31 exceed it."
    suggested_fix: "Truncate fold 11 test_end to 2025-12-26 so declared == evaluated."
    status: FIXED
    resolution_notes: "run_storya_e1_anchor.py:109 test_end '2025-12-31'→'2025-12-26' + inline note. Re-validated (script): fold 11 now 61 test days [2025-10-01..2025-12-26], 0 invalid-label days; assert_purge_no_leak PASS all 12 folds. Boundary independently confirmed before fix."
  - id: CODEX-A-02
    severity: CONCERN
    category: correctness
    claim: "Mixed old/new cell_id formula produces duplicate cell_id values in results.csv / manifest.csv (170 = old (B,LightGBM,f2,s86) and new (B,SAGE-Mean,f5,s86)). Not a resume/training bug (those key on the (universe,model,seed,fold) tuple) but analysis grouping by cell_id alone would merge unrelated cells."
    evidence: "cell_id widened *200/*50→*480/*120; folds 0-4 cached rows keep old ids; resume + per_day_ic filenames key on the tuple. Verified: results.csv had cell_id 170 on 2 rows (1 dup / 401)."
    suggested_fix: "Backfill cell_id on existing rows to the new formula, or use the tuple key downstream."
    status: FIXED
    resolution_notes: "Backfilled cell_id deterministically (u*480+m*120+fold*10+seed_idx) across results.csv + manifest.csv (backups .bak_20260612). Re-validated: 401 rows, 0 dup; (B,LightGBM,f2,s86)→380, (B,SAGE-Mean,f5,s86)→170. per_day_ic/resume unaffected (tuple-keyed)."
summary:
  critical: 0
  major: 1
  concern: 1
  fixed_before_reply: 2
overall_verdict: PROCEED-WITH-FIXES
---

# Codex Code Review — Anchor window extension (5→12 fold), Round A

Rule 9 Touchpoint 2 review of the **window-only** extension of `run_storya_e1_anchor.py`
(same 4 models / hyperparameters / TRAIN_START; +7 quarterly expanding folds ids 5–11; cell_id
widened; injectivity assert generalized). Distinct from the v2.1 ladder runner review (`_A`).

**Leak verdict (Codex + independently confirmed): NO leak path in the new folds.** Train+val
purged 21d (`create_fold_masks:537-538`); frozen correlation snapshot selected with
`t_end ≤ train_days[-1]` (already-purged), so the injected graph stays inside the fold's training
window even though `build_correlation_snapshots` precomputes all 54 snapshots on the full returns
array. Winsor/scale fit train-only; Alpha158 T-1 shift fold-independent. `assert_purge_no_leak`
PASS for all 12 folds. No remaining 5-fold/400-cell assumption on the run path (only stale
`--smoke`-mode print strings, not execution control).

**Both findings fixed and re-validated in-session before reply** (resolution_notes). Verdict
PROCEED-WITH-FIXES → fixes applied → cleared to launch the 7 new folds (560 cells) once compute
(A100) is available. Note: existing downstream analysis scripts still hardcode 5 folds and must
not be reused unchanged for 12-fold inference.
