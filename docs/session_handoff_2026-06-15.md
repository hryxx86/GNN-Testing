---
handoff_date: 2026-06-15
last_completed: "2026-06-15-d: §4 search space frozen (protocol→v2.2) + tuning harness BUILT & smoke-validated (L0 LGB + L2 GAT, both pass leak assert + all dims flow) + anchor train_lightgbm lambda fix (3 verifications pass)"
in_flight:
  - id: step-4-tuning-RUN
    file: run_storya_v21_tune.py + run_v21_tune_launcher.py
    status: "BUILT + smoke-validated (L0/L2). NOT yet Touchpoint-2-reviewed; NOT yet run. NEXT conversation: (1) Touchpoint 2 (Codex) on the 3 new/changed files, (2) launch parallel tuning Mac+T4, (3) merge → frozen_hparams.json."
    blockers: ["Rule 9 Touchpoint 2 MUST pass before the tuning run (the 3 verifications below are evidence, not a substitute for the review)."]
  - id: storya-v21-main12-TUNED-rerun
    file: run_storya_v21_main12.py
    status: "after frozen_hparams.json exists: re-run the 2160-cell ladder under tuned HPs (write to a NEW dir, e.g. experiments/storya_v21_main12_tuned/, do NOT overwrite the pilot). --resume cannot reuse pilot cells (different HPs)."
    blockers: ["step-4-tuning-RUN must produce frozen_hparams.json"]
  - id: storya-v21-main12-analysis
    file: compute_e6_dm_spa.py
    status: "§2a (DM-HLN + BH-FDR q=0.05 + Hansen SPA M=9) on the TUNED rerun (LOCAL, no GPU)."
    blockers: ["tuned rerun complete", "decide reuse vs adapt compute_e6_dm_spa.py → if adapted, Touchpoint 2"]
open_questions:
  - "§2a tooling: reuse compute_e6_dm_spa.py (anchor E1-E6) or build a 9-arm-ladder driver? Code change → Touchpoint 2."
  - "RESOLVED this session: tuning is mandatory (§4 + D-RERUN-12F); search space frozen v2.2 (center=pilot default); top-5 (NOT top-8, H博士 2026-06-15); LGB lambda is a REAL dim (anchor fix)."
file_state:
  new_files:
    - "run_storya_v21_tune.py (single-study §4 tuner, import-only)"
    - "run_v21_tune_launcher.py (study-level parallel launcher + merge)"
    - "docs/session_handoff_2026-06-15.md"
    - ".claude/ (infra, untracked)"
    - "experiments/storya_v21_main12/ (PILOT data, untracked; md5-verified local copy of Colab/Drive)"
  modified_since_last_commit:
    - "run_storya_e1_anchor.py (train_lightgbm: +lambda_l1/lambda_l2, additive, default 0.0 = no-op when absent)"
    - "docs/protocol_v2_freeze.md (→ v2.2: §4 search space explicit, §11 row)"
    - "progress.md (2026-06-15-d), plan.md (Decision Log)"
    - "(concurrent pilot track) run_storya_e1_anchor.py other?, compute_e6_dm_spa.py, artifacts/storya_e6_dm_spa/*, docs/analysis.md"
rule9_status:
  touchpoint_1_plan: PASSED       # v2.1/v2.2 protocol disposition
  touchpoint_2_code: PARTIAL      # main12 PILOT runner PASSED (2026-06-12). PENDING: tuning harness + launcher + anchor lambda-fix (3 verifications done as evidence; Codex review still owed BEFORE the tuning run).
  touchpoint_3_results: PENDING   # §2a on the TUNED rerun. (progress 2026-06-13-c T3 = concurrent untuned-anchor pilot, NOT this ladder.)
next_actions:
  - "Touchpoint 2 (Codex; finance-gnn-reviewer fallback >15min) on: run_storya_v21_tune.py, run_v21_tune_launcher.py, run_storya_e1_anchor.py train_lightgbm diff. Attach the 3 verifications (no-op / path-consistency / ghost-dim audit) as evidence."
  - "Launch parallel tuning — Mac: `python run_v21_tune_launcher.py --machine mac --concurrency 2` ; T4: `LD_LIBRARY_PATH=/usr/lib64-nvidia python run_v21_tune_launcher.py --machine t4 --concurrency 4` (after re-installing deps + optuna)."
  - "Merge: `python run_v21_tune_launcher.py --merge` → experiments/storya_v21_tune/frozen_hparams.json (expect 20 studies)."
  - "Tuned rerun of run_storya_v21_main12.py under frozen_hparams (new output dir)."
  - "§2a (local) → Touchpoint 3 → docs/analysis.md + progress.md."
---

# Session Handoff — 2026-06-15  (START HERE to run the §4 tuning)

## TL;DR — where we are

1. **PILOT done**: `run_storya_v21_main12.py` completed 2160 cells under the OLD anchor HPs (perfect integrity, byte-verified local copy in `experiments/storya_v21_main12/`). This is a **pre-tuning pilot, NOT the confirmatory main table.**
2. **Search space FROZEN** (protocol → **v2.2**, `docs/protocol_v2_freeze.md` §4): 6-dim NN/GAT/L6/L7 + 6-dim LGB, every dim's **center = the pilot default** (so the pilot is the N=1 center-point and pilot-vs-tuned stays comparable). 3 deviations from v2-frozen logged in §11 (dropout discrete; LGB 8→6 dims; hidden {32,64,128}).
3. **Anchor lambda fix DONE + verified**: `anchor.train_lightgbm` now consumes `lambda_l1/lambda_l2` (additive, `.get(...,0.0)` → no-op when absent). Was a ghost dim. 3 verifications pass (below).
4. **Tuning harness BUILT + smoke-validated**: `run_storya_v21_tune.py` (single study) + `run_v21_tune_launcher.py` (parallel). Smoke: L0 (LGB) + L2 (GAT) both end-to-end OK, leak assert fires, all dims flow.

**Next conversation's job**: Touchpoint 2 → run parallel tuning → merge → frozen_hparams.json → tuned 2160-cell rerun → §2a.

## The tuning harness (what was built)

- **`run_storya_v21_tune.py --arm X --universe Y`** runs ONE self-contained Optuna study:
  - import-only from anchor (data/snapshot/C1 asserts) + main12 (run_arm_cell / build_fold_edges / complete graph) → trains EXACTLY like the confirmatory cell.
  - tuning window: train `TRAIN_START..2022-06-30`, val `2022H2` (early-stop + Rank-IC selection metric); scores Rank IC on the val days.
  - **leak protection**: corr graph = frozen snapshot at train_end=2022-06 (explicit assert `snap_point ≤ train_end`); C1 asserts ride along via the imports.
  - search space v2.2 (arm-aware: heads only for GAT/L7; MLP/SAGE drop heads; LGB its own 6 dims). HP injection = monkeypatch `anchor.NN_HPARAMS/LGB_HPARAMS` per trial.
  - procedure: N=30 single-seed search (seed 11) → top-5 finalists × 3 tuning seeds [11,22,33] → winner = max mean val Rank IC. Persists FULL study (sqlite) + top-5 table per study + `{U}_{arm}.json`.
  - tuning seeds [11,22,33] **disjoint** from canonical 10 (startup assert).
- **`run_v21_tune_launcher.py`** = study-level parallelism (one process per study, sequential trials inside → TPE determinism preserved; concurrency fills the GPU). Split: `--machine mac` = L0/L1/L2s/L5s ×{B,C} (8 studies, LGB CPU + MLP/SAGE MPS); `--machine t4` = L2/L3/L4/L5/L6/L7 ×{B,C} (12 studies, GAT/L6/L7). `--merge` collects → frozen_hparams.json.

## Smoke evidence (this session)

- L0 B (LGB): winner mean val-IC(3seed)=+0.08218, params include `lambda_l2≈0.09` → lambda dim real & flowing; 6s.
- L2 B (GAT): winner `{lr 2.8e-3, wd 3e-4, dropout 0.2, hidden 32, layers 3, heads 8}` → all 6 NN dims flow via monkeypatch; leak assert `snap≤train_end ✓`; 575s on MPS (T4 much faster).
- Smoke artifacts deleted; `experiments/storya_v21_tune/` starts empty for the real run.

## Anchor lambda fix — the 3 Touchpoint-2 verifications (evidence for the review)

1. **no-op when absent**: synthetic LGB, explicit `lambda_l1=l2=0.0` vs keys absent → `max|Δpred| = 0.000e+00` (byte-identical). Concurrent/pilot callers unaffected. And `lambda_l2=1.0` shifts preds 6.16e-2 → lambda is genuinely consumed.
2. **path consistency**: `lgb.train` only in `anchor.train_lightgbm`; main12 L0 (line 489) + the tuning harness (via run_arm_cell) both route through it. (`option_b_lgbm_importance.py` uses LGBMRegressor but is an unrelated importance diagnostic, out of the ladder path.)
3. **ghost-dim audit (all 12 dims)**: every NN dim has a consumption point (lr/wd train_nn:585-586; dropout/hidden/num_layers/heads make_nn_model:483-496) — weight_decay & num_layers confirmed real, NOT ghosts. LGB num_leaves/lr/min_data:728-730, n_estimators:744; lambda_l1/l2 were the ONLY ghosts → fixed (now train_lightgbm:731).

## How to RUN the tuning (new conversation)

1. **Touchpoint 2 first** (Rule 9 — before the run): Codex review of `run_storya_v21_tune.py`, `run_v21_tune_launcher.py`, and the `run_storya_e1_anchor.py` train_lightgbm diff. The 3 verifications above are evidence, not a substitute.
2. **Mac** (no GPU needed for LGB; MPS for MLP/SAGE): `python run_v21_tune_launcher.py --machine mac --concurrency 2`
3. **Colab T4** (fresh runtime: re-install `torch_geometric pandas_market_calendars lightgbm optuna`; export `LD_LIBRARY_PATH=/usr/lib64-nvidia`; run in tmux): `python run_v21_tune_launcher.py --machine t4 --concurrency 4`
4. **Merge** when both done: `python run_v21_tune_launcher.py --merge` → `experiments/storya_v21_tune/frozen_hparams.json` (expect 20/20 studies).
5. Each study is resumable (sqlite `load_if_exists`); a failed study just re-run its `--arm/--universe`.

## Colab infra notes (from the pilot marathon)

trycloudflare quick tunnels dropped 6× (hostnames change each re-run of the SSH cell; `bad handshake`=origin down, `no such host`=tunnel gone — needs H博士 to re-run the tunnel cell). T4 ≈ A100 for these tiny GNNs (kernel-launch-bound). Drive + `--resume` = zero data loss across all the drops. Distinguish *tunnel* reconnect (VM+tmux survive; same PID, deps present) from *runtime* recycle (deps + tmux wiped → re-install + relaunch).

## Two tracks (don't conflate)

THIS track = `run_storya_v21_main12.py` ladder (the confirmatory main table, AFTER tuning). Concurrent track = `run_storya_e1_anchor.py` 5→12 + sliding (untuned PILOT, progress 2026-06-13/-15-a). Both untuned-pilot results carry the **PILOT-CENTER / pending-tuning** label — do NOT let them flow as the main result.

## Data location (PILOT)

`experiments/storya_v21_main12/`: results.csv (2160) + manifest.csv (0 failed) + _meta.json + per_day_ic/×2160 + v21main.tar.gz backup. Cross-verified byte-identical to Colab/Drive via `/tmp/verify_match.py`.
