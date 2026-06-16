---
reviewer: codex
touchpoint: code
round: A
target_files:
  - run_storya_v21_tune.py
  - run_v21_tune_launcher.py
  - run_storya_e1_anchor.py:728-736
findings:
  - id: CODEX-A-01
    severity: MAJOR
    category: reproducibility
    claim: "merge() writes frozen_hparams.json even when studies are missing or failed, so downstream can consume a partial frozen HP set."
    evidence: "run_v21_tune_launcher.py:111-138 — merge() collects whatever JSON files exist, writes frozen_hparams.json at 131-134, only prints a WARNING at 135-138 when len(rows)!=20. run_pool reports failed studies (103-108) but main (163) discards the return without a nonzero exit."
    suggested_fix: "Make merge fail closed unless exactly the 20 expected keys are present; launcher exit nonzero on failed studies; partial only under an explicit --allow-partial flag writing a differently-named artifact."
    status: FIXED
    resolution_notes: "FIXED run_v21_tune_launcher.py. merge(allow_partial): if n!=20 and not allow_partial → print ERROR + missing list + sys.exit(1), does NOT write frozen_hparams.json; with --allow-partial writes frozen_hparams.PARTIAL.json (complete:false). run_pool failures now propagate: main() sys.exit(1) if any study rc!=0. Verified: `--merge` on a 1-study dir → ERROR, exit 1, no frozen_hparams.json written; on a 20-study stub → writes frozen_hparams.json. (Directly closes the L7-crash→18/20→silent-partial path Claude flagged in pre-review.)"
  - id: CODEX-A-02
    severity: MAJOR
    category: reproducibility
    claim: "Smoke and production tuning share the same study database and per-study JSON path, so smoke-only 2-trial winners can contaminate resume or merge."
    evidence: "run_storya_v21_tune.py:247-248 same study_name/storage for all runs; 301 smoke=2 trials/top1; 307 same OUT_DIR/{u}_{a}.json. run_v21_tune_launcher.py:116-124 merges per-study JSON without checking r['smoke']."
    suggested_fix: "Separate smoke study names/output paths; make merge reject smoke=true or sub-budget studies."
    status: FIXED
    resolution_notes: "FIXED both files. tune: run_study(smoke) → study_name f'{u}_{a}_smoke' (separate sqlite db); main() writes smoke JSON to OUT_DIR/_smoke_{u}_{a}.json. So smoke NEVER touches the production db/json. launcher merge(): excludes basenames starting '_smoke', AND belt-and-suspenders skips any JSON with smoke==True (logs skipped). Re-smoke (L7 B) confirms: smoke writes _smoke_B_L7.json + studies/B_L7_smoke.db only; production paths untouched."
  - id: CODEX-A-03
    severity: CONCERN
    category: reproducibility
    claim: "Optuna resume counts COMPLETE trials correctly, but the TPE sampler is recreated rather than restored, so a resumed study is not guaranteed to match an uninterrupted deterministic TPE trajectory."
    evidence: "run_storya_v21_tune.py:256-262 — fresh TPESampler(seed) each invocation; load_if_exists for trial storage; resume by COMPLETE count only. Sampler RNG state is not persisted."
    suggested_fix: "Persist/restore sampler state, or require production studies to run uninterrupted from a clean DB and document that interrupted resume defines a different (still deterministic-from-that-point) trajectory."
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "Accepted + documented (comment at run_storya_v21_tune.py resume block). Disposition: a CLEAN uninterrupted study is fully reproducible (md5-seeded TPE, fix MAJOR-1); resume conditions on stored trials but restarts the sampler RNG, so a resumed run is a valid-but-different trajectory. Studies are short (≤45 trainings), so the operator guidance is: for a FAILED study prefer re-run from a clean db (delete its .db); resume is the crash-recovery path across Colab runtime recycles, not a bit-reproducibility guarantee. Not worth pickling TPE sampler state for a 30-trial search. H博士 informed."
summary:
  critical: 0
  major: 2
  concern: 1
  fixed_before_reply: 0
overall_verdict: PROCEED-WITH-FIXES
---

# Codex Code Review — Touchpoint 2, Round A (§4 tuning harness)

Target: `run_storya_v21_tune.py` (new), `run_v21_tune_launcher.py` (new), `run_storya_e1_anchor.py` train_lightgbm lambda diff (L728-736).

## Verdict: PROCEED-WITH-FIXES — 0 CRITICAL / 2 MAJOR / 1 CONCERN. All MAJOR FIXED, CONCERN accepted+documented.

## Codex confirmations (Claude pre-review items re-validated, no new issue)
- **A. L7 apples-to-apples**: PASS. Tuner uses the SAME HATS edge builder (`build_three_relation_edges_per_fold`, tune L197-199) and HP source (`hats.HATS_HPARAMS`, tune L140-141 → consumed e1_6_hats L326-335) as the confirmatory L7 runner (run_storya_v21_l7_hats.py L332-335).
- **B. val=early-stop=scoring**: acceptable; no 2023+ leak — TUNE_FOLD val_end 2022-12-31 (tune L82), create_fold_masks purges last 21 val td (anchor L537-539), scoring only on val_days.
- **C. monkeypatch concurrency**: safe — launcher uses subprocess.Popen (process-local module globals), trials sequential (no n_jobs).
- **F. ghost dims**: none — LGB lambda_l1/l2 consumed (anchor L728-736); NN/GAT dims in make_nn_model (L483-497)+optimizer (585-586); L7 six dims via HATS_HPARAMS (e1_6_hats L326-335).

## Fixes applied
- **CODEX-A-01 (MAJOR→FIXED)**: merge fail-closed + launcher nonzero exit on failed studies. See resolution_notes.
- **CODEX-A-02 (MAJOR→FIXED)**: smoke physically isolated from production db/json + merge skips smoke. See resolution_notes.
- **CODEX-A-03 (CONCERN→ACCEPTED)**: documented; clean run reproducible, resume = recovery path. See resolution_notes.

All fixes re-smoke-validated (L7 B rc=0, isolated paths) + merge fail-closed/partial behavior unit-checked. Cleared for the 20-study tuning launch (pending H博士 + Colab SSH hostname).
