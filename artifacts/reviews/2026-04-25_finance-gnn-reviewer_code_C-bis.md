---
reviewer: finance-gnn-reviewer
touchpoint: code
round: C-bis
target_files:
  - run_loss_horserace.py:890-976
  - run_loss_horserace.py:981-1106
  - run_loss_horserace.py:700-706
  - artifacts/loss_horserace/hparams.json
target_plan: /Users/heruixi/.claude/plans/loss-function-s6-research-jaunty-nova.md
findings:
  - id: FINGNN-Cbis-01
    severity: pass
    category: correctness
    claim: Stage 0a aggregation uses mean across 3 seeds (per H博士 D3), not median.
    evidence: run_loss_horserace.py:896 `agg = sub.groupby(group_cols, dropna=False)['val_ic'].mean().reset_index()`. groupby keys = ['lr','dropout'] (+ ['margin'] for pairwise) so the only non-grouped axis collapsed by `.mean()` is seed (3 PILOT_SEEDS). dropna=False ensures NaN margin (listmle/approxndcg) does not silently drop rows.
    suggested_fix: none
    status: ACCEPTED-AS-CONCERN
    resolution_notes: One robustness observation — `mean()` with no `min_count` will return the mean of however-many-seeds-completed if a seed run errored. Currently no run-failure path is observable, but if Stage 0 is ever resumed mid-seed, a config with 1-of-3 seeds done would still be aggregated. Not a blocker because Stage 0 is already complete; flag for future stage-0 re-runs.

  - id: FINGNN-Cbis-02
    severity: pass
    category: correctness
    claim: ApproxNDCG decision direction is correct — DISCARDED iff ApproxNDCG falls behind ListMLE by more than Δ=0.003. Robust to ApproxNDCG-leads case.
    evidence: run_loss_horserace.py:904 `approxndcg_survives = (listmle_win_ic - approxndcg_win_ic) <= APPROXNDCG_KEEP_DELTA`. With listmle=+0.1175 and approxndcg=+0.0041, gap = +0.1134 > 0.003 → False → DISCARDED. If ApproxNDCG had led ListMLE (e.g. listmle=+0.05, approxndcg=+0.08), `listmle - approxndcg = -0.03 <= 0.003` → True → SURVIVES (correct: a leading ApproxNDCG must always be kept). Direction verified.
    suggested_fix: none
    status: REJECTED
    resolution_notes: No bug. Plan §"Stage 0a outputs": "if val IC < ListMLE by noise threshold Δ=0.003, drop ApproxNDCG ... else include as 4th loss" — code matches.

  - id: FINGNN-Cbis-03
    severity: pass
    category: correctness
    claim: Stage 1 hparam loading — ranking-loss cells receive winner lr/dropout; MSE cell uses Part B defaults (lr=1e-3, dropout=0.3).
    evidence: run_loss_horserace.py:1036-1045 `hp = default_hparams()` (always); only when `loss_type != 'mse'` is `hp['lr']` / `hp['dropout']` overwritten from `winners[loss_type]`. default_hparams() at line 700-706 returns `lr=1e-3, dropout=0.3, hidden=64, num_layers=2, weight_decay=1e-4, epochs=50, patience=10, grad_accum=4`. MSE cell therefore = Part B locked.
    suggested_fix: none
    status: REJECTED
    resolution_notes: Note: epochs=50, patience=10 in default_hparams differs from plan §"Stage 1" text "epochs=100, patience=15". Plan text contradicts Part B's actual frozen config — verify by checking run_step3_plan_z_part_b.py. If Part B was indeed epochs=50/patience=10, plan text is stale; if Part B was 100/15 then default_hparams understates training budget. Out of scope for this review (Round C C-bis is narrowly Stage-0→Stage-1 transition), but H博士 should confirm before Stage 1 launch.

  - id: FINGNN-Cbis-04
    severity: pass
    category: correctness
    claim: SAGE lr factor is applied multiplicatively to the loss-winner lr, isolated per-iteration via dict copy.
    evidence: run_loss_horserace.py:1052-1054 `hp_run = dict(hp); if model_type == 'SAGE-Mean' and loss_type != 'mse': hp_run['lr'] = hp['lr'] * sage_lr_factor`. `dict(hp)` is a shallow copy — sufficient because hp only contains scalar values (no nested dicts). MLP iteration leaves hp['lr'] = winner's lr (e.g. 0.002); then SAGE-Mean iteration creates fresh hp_run with lr = 0.002 × 1.0 = 0.002. Next outer-loop loss_type re-initializes hp = default_hparams() (line 1036), so no leak across loss iterations.
    suggested_fix: none
    status: REJECTED
    resolution_notes: Verified — hp_run is per (loss × model_type) cell, isolated. With sage_factors = {listmle:1.0, pairwise:1.0}, SAGE lr = MLP lr; with non-1.0 factors the multiplicative behavior is correct.

  - id: FINGNN-Cbis-05
    severity: pass
    category: correctness
    claim: active_losses correctly excludes ApproxNDCG when survives=false.
    evidence: run_loss_horserace.py:999 `active_losses = ['mse', 'listmle', 'pairwise'] + (['approxndcg'] if approxndcg_in else [])`. Given Stage 0 produced approxndcg_survives=false → approxndcg_in=False → active_losses = ['mse', 'listmle', 'pairwise']. Later line 1034 `for loss_type in active_losses` iterates only these three.
    suggested_fix: none
    status: REJECTED
    resolution_notes: Confirmed clean.

  - id: FINGNN-Cbis-06
    severity: pass
    category: correctness
    claim: Total run count = 600 (3 losses × 2 models × 2 features × 5 folds × 10 seeds), not 800.
    evidence: run_loss_horserace.py:1028 `total = (len(active_losses) * 2 * 2 * len(manifest['folds']) * len(STAGE1_SEEDS))`. With active_losses=3, folds=5 (per fold_manifest), STAGE1_SEEDS=10 (line 81): 3 × 2 × 2 × 5 × 10 = 600. Matches plan §"Stage 1": "{3 or 4} losses × 2 models × 2 feature sets × 5 folds × 10 seeds = 600 or 800 runs".
    suggested_fix: none
    status: REJECTED
    resolution_notes: Confirmed.

  - id: FINGNN-Cbis-07
    severity: major
    category: reproducibility
    claim: day_idx written to results.csv corresponds to the trading-day index in the global panel (fold_manifest test_days), and matches preds .npy ordering. However, results.csv currently has NO ordering guarantee within (model,loss,feat,fold,seed); downstream join must use (..., day_idx) tuple — verify analyze_loss_horserace.py does this.
    evidence: run_loss_horserace.py:1073 `for i, d in enumerate(r['test_days']):` writes `day_idx=int(d)` where `d` comes from `fold['test_days']` (line 720). Plan §"Critical infrastructure reuse": "Walk-forward folds | artifacts/step3_plan_z/fold_manifest.json | Triple-set identity check ...; day_idx integrity". The same `test_days` array is used for ic_per_day computation (line 730 daily_ic) and for the .npy save indexing — so per-day rows ARE row-aligned with preds_*.npy[i, :]. Resume safe: fold['test_days'] is read fresh from manifest each launch. Concern: line 1014 `key_cols=['model','loss','feature_set','fold','seed','day_idx']` — recover_fallback_csv merges on 6-tuple including day_idx, which is correct.
    suggested_fix: |
      No code change required. Add a one-line comment near line 1073 documenting the alignment guarantee:
      `# day_idx = global trading-day index from fold_manifest test_days; preds_*.npy[i] is the same i.`
      Downstream analyze_loss_horserace.py must load preds_*.npy in fold['test_days'] order and merge on day_idx — verify before Stage 2 SPA.
    status: ACCEPTED-AS-CONCERN
    resolution_notes: Stable across resumes because fold_manifest.json is the immutable source of truth. The `day_idx=int(d)` cast assumes d is already a global day index (np.int64); confirmed by line 720 `np.array(fold['test_days'])` where fold_manifest stores trading-day indices.

  - id: FINGNN-Cbis-08
    severity: major
    category: reproducibility
    claim: Resume key (model, loss, feature_set, fold, seed) does NOT include hparams. If hparams.json is regenerated mid-pipeline (e.g. Stage 0 re-run with different winner), Stage 1 resume will skip cells that should be retrained under new hparams.
    evidence: run_loss_horserace.py:1016-1024 `done_keys = set()  # (model, loss, feat, fold, seed)`. The hparam values (lr, dropout, margin) are read from hparams.json (line 992) but NOT incorporated into the resume key. Concrete failure mode: launch Stage 1 with hparams.json v1 (listmle lr=0.002). Some cells complete. Stop; re-run Stage 0 producing hparams.json v2 (listmle lr=0.001). Restart Stage 1 — completed cells will be skipped despite different hparams.
    suggested_fix: |
      Either (a) add a `hparams_hash` column to results.csv computed from sha256(json.dumps(pilot_out, sort_keys=True)) and include it in done_keys; or (b) abort if hparams.json mtime > results.csv mtime (cheap guard). Minimal fix:
      ```python
      # at line ~993 after json.load
      hparams_hash = hashlib.sha256(json.dumps(pilot_out, sort_keys=True, default=str).encode()).hexdigest()[:12]
      # at line 1016-1024 after building done_keys, add:
      if results_csv.exists() and 'hparams_hash' in existing.columns:
          stale = (existing['hparams_hash'] != hparams_hash)
          if stale.any():
              raise RuntimeError(f'results.csv has {stale.sum()} rows from a different hparams.json hash. Inspect before resuming.')
      # at line 1076 dict(...): add `hparams_hash=hparams_hash`
      ```
    status: OPEN
    resolution_notes: For the immediate Stage 1 launch this is low-risk because Stage 0 just completed and hparams.json will not be regenerated mid-run. Flagged MAJOR because the reproducibility guarantee silently breaks if a future operator re-runs Stage 0 partway. H博士 may defer if commitment is "one-shot Stage 1 with no re-tuning."

  - id: FINGNN-Cbis-09
    severity: pass
    category: correctness
    claim: MSE-SAGE cell uses lr=1e-3 (Part B default), no SAGE factor applied.
    evidence: run_loss_horserace.py:1049 `if model_type == 'SAGE-Mean' and loss_type != 'mse':` — both clauses must hold for sage_lr_factor lookup; for MSE+SAGE-Mean the second clause fails → sage_lr_factor stays at default 1.0 (line 1048). Then line 1053 `if model_type == 'SAGE-Mean' and loss_type != 'mse':` — same gate, so hp_run['lr'] is NOT overwritten for MSE+SAGE → keeps hp['lr'] = default 1e-3. Plan §"Stage 1" hparams table: "MSE cell = Part B frozen (... lr=1e-3 ...)" — both MLP and SAGE. Verified.
    suggested_fix: none
    status: REJECTED
    resolution_notes: Even if hparams.json carried `mse: {sage_lr_factor: 1.0}` (it does, line 967), it's never consulted because the loss_type=='mse' guard skips that branch. Defensive: the `mse` key in hparams_out is dead-code at Stage 1, only documentary. Acceptable.

  - id: FINGNN-Cbis-10
    severity: concern
    category: correctness
    claim: ApproxNDCG `sage_lr_factor=null` edge case — if approxndcg ever survives, sage_factors['approxndcg'] is set to `float(best['lr_factor'])` (line 961) = real number. But if approxndcg is discarded (current case), sage_factors['approxndcg']=None (line 956). At Stage 1 active_losses excludes approxndcg, so the `None × float` path is never reached. Future-proofing only.
    evidence: run_loss_horserace.py:1050 `sage_lr_factor = sage_factors.get(loss_type, 1.0)`. If a future bug causes active_losses to include 'approxndcg' while sage_factors['approxndcg']=None, line 1054 `hp['lr'] * sage_lr_factor` = `float * None` → TypeError. Currently unreachable because active_losses construction (line 999) is gated on the same approxndcg_in flag.
    suggested_fix: |
      Defensive: `sage_lr_factor = sage_factors.get(loss_type) or 1.0` (line 1050) — handles None → 1.0 gracefully. Or assert-as-precondition:
      ```python
      assert sage_lr_factor is not None, f'sage_lr_factor is None for {loss_type} but loss is active'
      ```
      Either is one line. Not a blocker for current Stage 1.
    status: ACCEPTED-AS-CONCERN
    resolution_notes: Flagged as CONCERN per H博士 audit checklist item J. No action required for current Stage 1 launch.

  - id: FINGNN-Cbis-11
    severity: pass
    category: correctness
    claim: Codex's prior claim "epochs/patience/weight_decay/grad_accum not frozen between Stage 0 and Stage 1" — REJECTED with independent verification.
    evidence: run_loss_horserace.py:700-706 default_hparams() returns dict(hidden=64, num_layers=2, dropout=0.3, lr=1e-3, weight_decay=1e-4, epochs=50, patience=10, grad_accum=4). Stage 0 invocation line 864 `hparams = default_hparams() | hp_over` where hp_over only contains lr/dropout (lines 850, 854, 858) → epochs/patience/weight_decay/grad_accum/hidden/num_layers are ALL inherited from default_hparams, identical in Stage 0a and 0b. Stage 1 invocation line 1036 `hp = default_hparams()` then overrides only `hp['lr']` and `hp['dropout']` (lines 1040-1041) for ranking losses; everything else inherited. So epochs/patience/weight_decay/grad_accum are byte-identical between Stage 0 pilot and Stage 1 production runs. The only freeze-boundary leak risk would be if someone modified default_hparams() between Stage 0 and Stage 1 — neither code change nor source diff shows this. Claude's rebuttal stands; Codex's concern is REJECTED.
    suggested_fix: none
    status: REJECTED
    resolution_notes: Independent line-by-line trace confirms Claude's argument. Note: the current default_hparams (epochs=50, patience=10) may not match plan §"Stage 1" text (epochs=100, patience=15) — that's a plan-vs-code documentation issue (FINGNN-Cbis-03 resolution_notes), not a freeze-boundary bug.

  - id: FINGNN-Cbis-12
    severity: concern
    category: reproducibility
    claim: Stage 0 hparams.json artifact does not exist at /Users/heruixi/Desktop/GNN-Testing/artifacts/loss_horserace/hparams.json on local disk.
    evidence: `ls -la /Users/heruixi/Desktop/GNN-Testing/artifacts/loss_horserace/` returns empty directory. Stage 0 ran on Colab; hparams.json presumably written under /content/drive/MyDrive/GNN测试/artifacts/loss_horserace/. Reviewer cannot independently audit the on-disk values; relying on user-supplied "ListMLE: lr=0.002, dropout=0.3, val_ic=+0.1175 / Pairwise: lr=0.002, dropout=0.3, margin=0.01, val_ic=+0.0479 / ApproxNDCG val_ic=+0.0041 / SAGE factors {listmle:1.0, pairwise:1.0} / approxndcg_survives=false".
    suggested_fix: |
      Before launching Stage 1: sync hparams.json from Drive to local for code-archive reproducibility, or commit to repo. Verify schema:
      ```python
      python -c "import json; d=json.load(open('artifacts/loss_horserace/hparams.json')); assert d['approxndcg_survives'] is False; assert d['stage0a_winners']['listmle']['lr']==0.002; assert d['stage0a_winners']['pairwise']['margin']==0.01; print('OK')"
      ```
    status: FIXED
    resolution_notes: Synced from Drive 2026-04-25 02:24, md5=fd5ced40fe93e01a031d7d9b68fcde3d, file size 765 bytes at artifacts/loss_horserace/hparams.json. Schema verified: approxndcg_survives=False, listmle.lr=0.002, pairwise.margin=0.01, sage_factors {listmle:1.0, pairwise:1.0}. Logged in progress.md 2026-04-25-a §"Stage 1 launch 前 3 项准备" #2.

summary:
  critical: 0
  major: 2  # 1 OPEN (Cbis-08, deferred under one-shot launch commitment) + 1 FIXED post-review (Cbis-12, hparams.json synced 2026-04-25)
  concern: 3
  fixed_before_reply: 0
  fixed_post_review: 1  # Cbis-12 (synced from Drive 2026-04-25 02:24)
overall_verdict: PASS-WITH-CONCERNS
---

# Review body — Stage 0 → Stage 1 transition audit

## Scope and method

Independent line-by-line verification of Stage 0a winner selection, ApproxNDCG decision, Stage 0b SAGE factor selection, hparams.json write, and Stage 1 hparam loading + active_losses construction + run-iteration + per-day result row writing. Per Round C-bis instructions: persistence layer, loss-function bodies, _ensure_pkg, build_shared_state, _single_run internals, and smoke mode are explicitly out of scope (Codex Round C + 6 stop-time rounds passed those).

## Per-focus verdicts

**A (Stage 0a aggregation)**: PASS. `groupby([lr,dropout,margin?])['val_ic'].mean()` collapses on the 3 PILOT_SEEDS axis. dropna=False prevents margin=NaN row drops for listmle/approxndcg. One robustness note logged in FINGNN-Cbis-01.

**B (ApproxNDCG decision direction)**: PASS. `(listmle - approxndcg) <= 0.003` correctly DISCARDS when approxndcg lags by more than Δ; SURVIVES when approxndcg leads (negative gap satisfies <= Δ trivially). Plan-aligned. Stage 0 outputs (gap = +0.1134) match the discarded result.

**C (Stage 1 hparam loading)**: PASS structurally. MSE cell inherits Part B defaults; ranking cells override lr/dropout from winners. Caveat: plan text says epochs=100/patience=15 but default_hparams returns 50/10 — flagged in FINGNN-Cbis-03 resolution_notes. Out of Round C-bis scope per instructions, but H博士 should confirm Part B's actual frozen budget before Stage 1.

**D (SAGE lr factor application)**: PASS. `dict(hp)` per (loss × model) cell guarantees isolation. With both factors=1.0 in current Stage 0 output, MLP and SAGE share lr=0.002 for listmle and pairwise — defensible and matches the "no transfer adjustment needed" pilot conclusion.

**E (active_losses)**: PASS. With approxndcg_survives=false → 3 losses → 600 Stage-1 runs.

**F (run count)**: PASS. 3×2×2×5×10 = 600. Verified at line 1028.

**G (day_idx alignment)**: PASS-WITH-CONCERN (FINGNN-Cbis-07). day_idx is the global trading-day index from fold_manifest['test_days'] and aligns row-for-row with preds_*.npy[i, :]. Stable across resumes because manifest is immutable. Suggested adding a one-line comment documenting the alignment guarantee.

**H (resume key uniqueness)**: MAJOR FLAG (FINGNN-Cbis-08). The 5-tuple (model, loss, feat, fold, seed) excludes hparams. If Stage 0 is ever re-run between Stage 1 launches (e.g. mid-week pilot adjustment), Stage 1 would silently skip stale completed cells. Low-probability for current launch but worth a 5-line hparams_hash defensive guard. H博士 may defer if Stage 1 is one-shot.

**I (MSE-SAGE cell)**: PASS. Conjunction `loss_type != 'mse'` correctly skips both the sage_factors lookup and the lr override. MSE-SAGE-Mean = MSE-MLP = lr=1e-3, dropout=0.3 (Part B locked).

**J (ApproxNDCG sage_lr_factor=null)**: CONCERN. Defensively guard with `or 1.0` or assert; not reachable in current Stage 1.

## Codex prior claim — independent verification

Codex (per user-supplied summary) claimed Stage 1 "only overrides lr/dropout/margin/T but epochs/patience/weight_decay/grad_accum are not frozen — violates plan freeze-boundary." 

I traced default_hparams() (line 700-706) → Stage 0 invocation (line 864 `default_hparams() | hp_over`) → Stage 1 invocation (line 1036 `hp = default_hparams()` followed by selective overrides at lines 1040-1041, 1054). Across all three invocation paths, only lr and dropout (and not even those for MSE) are overridden; epochs/patience/weight_decay/grad_accum/hidden/num_layers come from the same default_hparams() function call site. **Claude's rebuttal is correct. Codex finding REJECTED.** Logged as FINGNN-Cbis-11 with line-by-line trace.

The only residual concern in this area is the plan-vs-code mismatch on epochs (50 vs 100) — that's a documentation drift, not a freeze-boundary bug, and it is consistent across Stage 0 and Stage 1 (so it doesn't change the apples-to-apples claim of the pilot).

## Bottom line

Two MAJOR concerns (FINGNN-Cbis-08 hparams_hash in resume key; FINGNN-Cbis-12 hparams.json not synced to local for audit). Three CONCERNS (Cbis-01 partial-seed aggregation edge case; Cbis-07 day_idx documentation; Cbis-10 ApproxNDCG None defensive). Zero CRITICAL.

The Stage 0 → Stage 1 transition is logically sound and plan-aligned. Stage 1 launch is unblocked **conditional on**:
1. Confirming default_hparams epochs/patience values match Part B's actual frozen config (FINGNN-Cbis-03 resolution_notes; quick `git log run_step3_plan_z_part_b.py` or H博士 memory check).
2. Syncing hparams.json from Colab Drive → local repo, optionally committing for archival reproducibility.
3. Acknowledging FINGNN-Cbis-08 as a one-shot-launch assumption: do NOT re-run Stage 0 mid-Stage-1 without manually clearing results.csv.

If the operator commits to one-shot Stage 1, no code changes are required. **Verdict: PASS-WITH-CONCERNS — proceed to Colab launch.**
