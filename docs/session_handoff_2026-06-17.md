---
handoff_date: 2026-06-17
last_completed: "2026-06-17-b: D-RERUN-12F frozen-HP injection built in main12 + l7_hats (Codex T2 PROCEED-WITH-FIXES, all 3 fixed) + code pushed 84176ff. §4 tuning 20/20 done; frozen_hparams.json git-preserved."
in_flight:
  - id: rerun-local-light
    file: run_storya_v21_main12.py
    status: "NOT launched. LOCAL Mac, light arms L0/L1/L2s/L5s × {B,C} = 960 cells. 2 procs by universe → own out-dirs → merge. ~13h. H博士 runs in a NEW window."
    blockers: []
  - id: rerun-colab-heavy
    file: run_storya_v21_main12.py + run_storya_v21_l7_hats.py
    status: "NOT launched. Colab T4: tuned heavy L2/L3/L4/L5/L6 (1200) + L7 HATS (240) + FC arm L3fc/L4fc/L5fc (720). Needs SSH hostname + tmux."
    blockers: ["Colab SSH hostname (trycloudflare, changes per runtime) from H博士"]
  - id: rerun-merge-analysis
    file: compute_e6_dm_spa.py (+ a new FC Family-2 analyzer, TBD)
    status: "After all cells done: merge Mac+Colab results → experiments/storya_v21_main12_tuned/ → Family-1 §2a + Family-2 FC inference → Touchpoint 3."
    blockers: ["all rerun cells complete"]
open_questions:
  - "Colab SSH hostname needed to launch the heavy/L7/FC part on T4."
  - "Family-2 FC inference (fold-level seed-avg ΔIC + block bootstrap + BH-FDR/6): reuse compute_e6_dm_spa.py building blocks or new small analyzer? → if new code, Touchpoint 2."
file_state:
  new_files:
    - "docs/session_handoff_2026-06-17.md, docs/plan_fc_edge_robustness_2026-06-17.md"
    - "artifacts/reviews/2026-06-17_codex_plan_A.md, artifacts/reviews/2026-06-17_codex_code_A.md"
    - "artifacts/storya_v21_tune/ (frozen_hparams.json + 20 winners + README, committed 2fd6e3c)"
    - "scripts/colab_v21_tune_db_sync.py (committed 62b9edc; tuning-only, NOT needed for the rerun)"
  pushed_to_main: ["84176ff (injection + reviews + FC plan)", "2fd6e3c (frozen_hparams)", "6c139cd (STUDY_DIR env)", "62b9edc (dbsync)"]
  modified_uncommitted: ["progress.md, plan.md (this session's entries — commit at closeout)"]
rule9_status:
  touchpoint_1_plan: PASSED       # FC two-family v2 (2026-06-17_codex_plan_A), BLOCK→fixed→locked
  touchpoint_2_code: PASSED       # injection (2026-06-17_codex_code_A), PROCEED-WITH-FIXES all fixed
  touchpoint_3_results: PENDING   # on the tuned rerun (Family-1 §2a + Family-2 FC)
next_actions:
  - "LOCAL: launch the 2 light-arm procs (see §A). Mac, no Colab needed."
  - "COLAB: get SSH hostname → bootstrap + pip install → launch heavy + L7 + FC in tmux (see §B). Recycle-safe via Drive manifest + --resume (NO dbsync needed for the rerun)."
  - "MERGE + ANALYZE: §C."
---

# Session Handoff — 2026-06-17 (START HERE to run D-RERUN-12F)

## TL;DR
§4 equal-budget tuning is DONE (20/20 → `frozen_hparams.json`, git-preserved at
`artifacts/storya_v21_tune/`, md5 59ddd0a2). The confirmatory tuned rerun code is built + Codex-T2-passed
(main12 + l7_hats `--frozen-hparams`/`--fc-fix-arm`/`--out-dir`; no-flag = byte-identical pilot). Code is
pushed (84176ff). **This session's job for the NEW window: run the rerun — light arms LOCAL, heavy/L7/FC on
Colab — then merge + §2a + FC Family-2 inference + Touchpoint 3.**

**LOCKED pre-registration (honor it):** two families — Family-1 (tuned ladder, predictive/model-selection,
SPA M=9) + Family-2 (FC fixed-capacity edge ablation, causal edge attribution). Full spec:
`docs/plan_fc_edge_robustness_2026-06-17.md`. tuned-ΔIC for edge arms is DESCRIPTIVE only; matched-ΔIC (FC)
is the causal primary. Do NOT switch primaries post-hoc.

## §A — LOCAL (Mac): light arms (L0/L1/L2s/L5s), 2-way parallel
PY=/opt/homebrew/Caskroom/miniforge/base/envs/gnn/bin/python (env `gnn`). frozen_hparams is the local repo
copy. Each proc → its OWN out-dir (main12 appends to ONE results.csv → parallel procs to the SAME dir would
race; separate dirs + merge later is the safe pattern).
```
cd /Users/heruixi/Desktop/GNN-Testing
FZ=artifacts/storya_v21_tune/frozen_hparams.json
nohup $PY run_storya_v21_main12.py --universe B --arms L0,L1,L2s,L5s \
  --frozen-hparams $FZ --out-dir experiments/storya_v21_main12_tuned_macB \
  > /tmp/rerun_macB.log 2>&1 &
nohup $PY run_storya_v21_main12.py --universe C --arms L0,L1,L2s,L5s \
  --frozen-hparams $FZ --out-dir experiments/storya_v21_main12_tuned_macC \
  > /tmp/rerun_macC.log 2>&1 &
```
- Each proc ~1GB RAM (16GB fine). ~13h for the pair. Resume: re-run the same cmd (reads that out-dir's manifest).
- Smoke already confirmed end-to-end: B/L0 tuned ran (λ2=0.4659 injected), L2 GAT injection table correct.
- Startup prints the injection table + writes `_frozen_hp_provenance.json` (mode/md5/applied) in each out-dir.

## §B — COLAB T4: heavy arms + L7 + FC
3 cells per runtime (CLAUDE.md Rule 7): mount Drive → `colab_bootstrap.sh` → `colab_ssh_tunnel.sh`. Then over SSH
(`sshpass -p "GNNTEST" ssh <HOST>.trycloudflare.com "..."`):
```
cd /content/GNN-Testing && git fetch origin -q && git reset --hard origin/main -q   # → 84176ff (or later)
pip install -q torch_geometric pandas_market_calendars                              # (optuna NOT needed for rerun)
```
**Colab path GOTCHA**: the runners `setup_workdir()` → chdir to the Drive root, so use an ABSOLUTE
`--frozen-hparams` (repo path) and a RELATIVE `--out-dir` (resolves to Drive → persisted). Launch in tmux:
```
FZ=/content/GNN-Testing/artifacts/storya_v21_tune/frozen_hparams.json
LD=/usr/lib64-nvidia
# (1) tuned heavy GAT/dense
tmux new-session -d -s t4heavy "cd /content/GNN-Testing && LD_LIBRARY_PATH=$LD python run_storya_v21_main12.py \
  --universe both --arms L2,L3,L4,L5,L6 --frozen-hparams $FZ \
  --out-dir experiments/storya_v21_main12_tuned_t4 2>&1 | tee experiments/storya_v21_main12_tuned_t4/run.log"
# (2) L7 HATS tuned
tmux new-session -d -s t4l7 "cd /content/GNN-Testing && LD_LIBRARY_PATH=$LD python run_storya_v21_l7_hats.py \
  --universe both --frozen-hparams $FZ --out-dir experiments/storya_v21_l7_hats_tuned 2>&1 | tee experiments/storya_v21_l7_hats_tuned/run.log"
# (3) FC arm (Family-2, fixed-capacity edge ablation — vary only edges, HPs frozen at L2)
tmux new-session -d -s t4fc "cd /content/GNN-Testing && LD_LIBRARY_PATH=$LD python run_storya_v21_main12.py \
  --universe both --arms L3,L4,L5 --fc-fix-arm L2 --frozen-hparams $FZ \
  --out-dir experiments/storya_v21_main12_fc 2>&1 | tee experiments/storya_v21_main12_fc/run.log"
```
- Run (1)+(2)+(3) concurrently if GPU/RAM allow (T4 15GB, each ~1-2GB GPU; tune-time ran 8-way fine). L6 is the
  long pole (Colab ~184s/cell mean, 734s max). ~1-2 days total on T4 with the recycles.
- **Recycle resilience (rerun)**: results/manifest/npy write to Drive (FUSE-fine; the pilot did this). On a
  recycle: re-bootstrap + re-pip + relaunch the SAME tmux cmds — `--resume` (default ON) reads the Drive
  manifest.csv and skips completed cells. **NO sqlite, NO dbsync needed** (that was tuning-only).
- `--out-dir` is enforced NON-pilot (fail-closed); `--frozen-hparams` aborts on a partial HP set.

## §C — MERGE + ANALYSIS (after all cells done)
1. Pull Colab outputs to Mac (scp the 3 t4 out-dirs' results.csv + manifest.csv + per_day_ic/ from Drive).
2. **Tuned main table** = concat {macB, macC, t4heavy} results.csv → `experiments/storya_v21_main12_tuned/`
   (cell_ids disjoint across arms/universes → clean union; merge per_day_ic npys; expect 2160 cells, manifest 0 failed).
   L7 (240) joins the DM/SPA family as a candidate arm (subject to §6 contingency on its health diagnostics).
3. **Family-1 §2a** (predictive): DM-HLN pairwise + BH-FDR q=0.05 + Hansen SPA M=9 on the tuned table
   (reuse/adapt `compute_e6_dm_spa.py` — if adapted, Touchpoint 2).
4. **Family-2 FC** (causal edge): on `experiments/storya_v21_main12_fc/` vs the tuned L2 cells (reuse frozen L2
   predictions as baseline). Inference = paired FOLD-LEVEL seed-averaged ΔIC (effective n≈12 fold blocks, NOT
   120 cells) + block bootstrap over 12 folds + BH-FDR over the 6 contrasts (L3fc/L4fc/L5fc−L2 × {B,C}).
   Report matched-ΔIC + same-sign vs tuned-ΔIC. (New analyzer → Touchpoint 2.)
5. **Touchpoint 3** (Codex results review) → `docs/analysis.md` (the confirmatory headline; the pilot
   2026-06-15-b/-a stays labelled PILOT-CENTER).

## Reading red lines
- The §4 tuning val-IC (in frozen_hparams) is a SELECTION metric (2022H2, optimistic) — NOT a result; never a finding.
- Capacity confound: edge-ablation tuned-ΔIC mixes edge effect with capacity (B all confounded; C only L3−L2, L6−L2).
  The FC arm (Family-2) is the clean causal read. Honest power: MDE@80%≈0.008–0.032 vs edge ΔIC ~0.005–0.016 →
  several FC contrasts likely "directional but not reliable" — that's expected, the FC arm is a defensible
  pre-registered frame + reviewer defense, not a guaranteed resolution.
