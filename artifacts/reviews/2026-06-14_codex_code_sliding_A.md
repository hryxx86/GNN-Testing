---
reviewer: codex
touchpoint: code
round: A
date: 2026-06-14
target_files:
  - run_storya_anchor_sliding.py        # NEW: 副轴 sliding-252d runner (import-only monkeypatch of anchor)
  - run_storya_e1_anchor.py             # overridden functions (read for monkeypatch-takes-effect check)
context: "副轴 (secondary axis) sliding-252d robustness runner — Option A (4-model anchor, H博士 2026-06-14).
  Import-only reuse of run_storya_e1_anchor: overrides output paths, WALK_FORWARD_FOLDS (12 sliding folds),
  create_fold_masks (per-fold train_start), assert_purge_no_leak. Verdict gates a ~15h MPS launch."
findings:
  - id: F3
    severity: MAJOR
    category: correctness
    claim: "Anchor argparse default is --folds 0,1,2,3,4; running the sliding runner without explicit --folds silently trains only 5 of 12 folds → incomplete robustness run."
    evidence: "run_storya_e1_anchor.py:974 default '0,1,2,3,4'; :1066-1068 fold loop skips ids not in folds_run."
    suggested_fix: "Inject --folds <all 12 ids> into sys.argv when neither --folds nor --smoke supplied."
    status: FIXED
    resolution_notes: "run_storya_anchor_sliding.py __main__: detects absent --folds (and not --smoke) → appends '--folds 0,1,...,11'. Re-validated: injection logic fires on bare argv."
  - id: F1
    severity: CONCERN
    category: provenance
    claim: "write_run_meta_json hardcodes experiment_id='storya_e1_anchor_v3'; sliding dir's _meta.json would mislabel the run (path is correct, only the label is wrong)."
    evidence: "run_storya_e1_anchor.py:899 hardcoded experiment_id; :986 main writes under (monkeypatched) OUT_DIR."
    suggested_fix: "Monkeypatch a.write_run_meta_json to relabel."
    status: FIXED
    resolution_notes: "Added sliding_write_run_meta_json wrapper: calls original then sets experiment_id='storya_anchor_sliding' + axis/sliding_window_td/train_start_semantics/main_axis_ref. Override confirmed in effect (a.write_run_meta_json is the wrapper)."
  - id: F2
    severity: CONCERN
    category: data_leakage
    claim: "The snapshot leak assert used train_end−126, but the frozen snapshot is selected from train_days[-1] (=te_i−HORIZON after purge) snapped DOWN to the corr_step=21 grid; the assert did not prove the actual selected snapshot's 126d window starts ≥ train_start."
    evidence: "run_storya_e1_anchor.py:445 snapshot_points=range(126,N,21); :448 window=[sp−126,sp); :1070 frozen from train_days[-1]."
    suggested_fix: "Compute purged_last_i=te_i−HORIZON, snap_pt=max grid point ≤ purged_last_i, assert snap_pt−126 ≥ train_start."
    status: FIXED
    resolution_notes: "Replaced with the exact bound in _build_sliding_folds. Re-validated: import (which runs the assert for all 12 folds) PASSES → every fold's actual frozen snapshot window starts ≥ its sliding train_start. No pre-window correlation leak."
verified_PASS:
  - "Q1 monkeypatch takes effect: anchor reads OUT_DIR/RESULTS_CSV/MANIFEST/PER_DAY_IC_DIR/SMOKE/HP_GRID + WALK_FORWARD_FOLDS + create_fold_masks + assert_purge_no_leak as module globals at call time (Codex traced each call site); no closure/default-arg capture."
  - "Q2b winsor/scale fit on sliding train_days (single source = overridden create_fold_masks)."
  - "Q2c 21d purge preserved ([:-horizon] in the override)."
  - "Q2d feature lookback (Alpha158 60d, T-1 shifted) reaching before train_start is ACCEPTABLE under standard sliding-window semantics (window = admitted (feature_date,label) PAIRS; features keep causal lookback). Codex verdict: leakage-free, no fix."
  - "Q3 cell_id unchanged, 12 ids 0-11, separate out-dir → no collision; resume tuple-keyed."
  - "Q4 TRAIN_START only used by the overridden create_fold_masks + its assert; no other main-axis assumption (besides the F3 --folds default, fixed)."
summary:
  critical: 0
  major: 1
  concern: 2
  fixed_before_launch: 3
overall_verdict: PROCEED-WITH-FIXES → all 3 fixed + re-validated → CLEARED to launch (after E4 resume completes, to avoid MPS contention)
---

# Codex Code Review — 副轴 sliding-252d runner, Round A

Rule 9 Touchpoint 2 on `run_storya_anchor_sliding.py` (import-only monkeypatch of the main-axis anchor).
Verdict **PROCEED-WITH-FIXES**: 0 CRITICAL, 1 MAJOR (F3), 2 CONCERN (F1, F2) — all fixed + re-validated
in-session before launch.

Codex confirmed the monkeypatch takes effect (anchor reads all overridden names as module globals at call
time, no capture), the sliding scheme is leak-safe (snapshot stays in window after the F2 tightening;
winsor/scale/purge propagate; feature lookback before train_start is acceptable sliding-window semantics),
and cell_id/resume are sound. The MAJOR (F3: silent 5-fold default) is the launch-critical fix — now the
runner injects all 12 fold ids when --folds is omitted. Cleared to launch the ~960-cell sliding run once
the E4 resume frees the MPS device.
