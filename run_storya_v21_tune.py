#!/usr/bin/env python
"""
Story A v2.2 §4 EQUAL-BUDGET TUNING — single-study Optuna runner (one arm × one universe).

Per docs/protocol_v2_freeze.md v2.2-frozen §4 (search space EXPLICITLY frozen 2026-06-15) and
progress.md 2026-06-15-d. Decision D-RERUN-12F: the confirmatory 12-fold ladder runs under the
§4 TUNED-then-frozen hyperparameters; the completed 2160-cell PILOT used the old anchor defaults.

WHY A NEW FILE + IMPORT-ONLY (§5 铁律 / C1): this harness does NOT rewrite any data / edge /
snapshot / training logic. It IMPORTS the data builders + the frozen-snapshot helper + the C1
runtime asserts from run_storya_e1_anchor (E0-validated), and the per-arm cell dispatch
(run_arm_cell), per-fold edge construction (build_fold_edges) and complete-graph builder from
run_storya_v21_main12 — so a tuning trial trains EXACTLY the same way the confirmatory cell will,
keeping tuning apples-to-apples with the eventual main table.

LEAK PROTECTION (Touchpoint-2 #1, the most important item):
  - Tuning window: train = TRAIN_START..2022-06-30, val = 2022H2 (early-stop AND selection metric).
  - The correlation graph is the FROZEN snapshot at train_end=2022-06-30 via the imported
    get_frozen_snapshot_idx — so it can ONLY see data ≤ 2022-06 (an explicit assert below proves it;
    NO 2022H2 data leaks into the graph).
  - The two C1 runtime asserts travel with the imports: Univ-C T-1 contract fires inside
    build_universe_C (+ assert_univ_c_t1_contract); the news-edge PIT assert (max(pub_ts) <=
    session_close(t-1)) fires inside build_per_day_news_edges. A tuning script does NOT escape PIT.

SEARCH SPACE (v2.2 §4, center = pilot default → pilot is the N=1 center-point sample):
  NN/GAT/L6/L7: lr log[1e-4,1e-2] · weight_decay log[1e-5,1e-3] · dropout {0.1,0.2,0.3,0.5}
                · hidden_channels {32,64,128} · num_layers {1,2,3} · heads {2,4,8} (GAT/L7 only)
  LGB:          num_leaves {15,31,63,127} · learning_rate log[0.01,0.1] · min_data_in_leaf
                {10,20,50,100} · lambda_l1 log[1e-8,1] · lambda_l2 log[1e-8,1] (n_estimators=early-stop)
  HP injection: monkeypatch anchor.NN_HPARAMS / anchor.LGB_HPARAMS per trial (module-global read at
  call time; one study == one process, sequential trials → safe, deterministic). lambda_l1/l2 are
  REAL dims (anchor.train_lightgbm extended 2026-06-15 to consume them; no-op when absent).

PROCEDURE (§4): N=30 single-seed search (seed=TUNE_SEEDS[0]); top-5 finalists re-run with all 3
  TUNE_SEEDS; winner = max mean-over-3-seeds val Rank IC. Tuning seeds [11,22,33] DISJOINT from the
  canonical 10 evaluation seeds (precheck #6; startup assert). Persists the FULL Optuna study
  (all trials) + the top-5 table (regime-mismatch ammunition; study discarded == lost).

Parallelism is at the STUDY level (one process per arm×universe, sequential trials inside → TPE
  determinism preserved). Concurrency / cross-machine split is the launcher's job
  (run_v21_tune_launcher.py); this file runs ONE study.

CLI:  --arm {L0..L6,L2s,L5s,L7}  --universe {B|C}   [--n-trials 30] [--top-k 5] [--smoke]
"""

import os
import sys
import time
import json
import argparse
import hashlib

import numpy as np
import pandas as pd
import torch
import optuna

# ── import-only reuse (§5): data / snapshot / C1 asserts from the E0-validated anchor ──
import run_storya_e1_anchor as anchor
from run_storya_e1_anchor import (
    CANONICAL_SEEDS, HORIZON, TRAIN_START,
    load_core_data, build_universe_B, build_universe_C, build_labels,
    build_correlation_snapshots, get_frozen_snapshot_idx, create_fold_masks,
    winsorize_train_only, standardize_train_only, compute_daily_ic,
    get_device, set_seed,
)
# per-arm cell dispatch + edge construction + complete graph — reuse the confirmatory path verbatim
import run_storya_v21_main12 as main12
from run_storya_v21_main12 import (
    run_arm_cell, build_fold_edges, build_complete_graph_edge_index,
    assert_univ_c_t1_contract, ARM_SPEC, IMPLEMENTED_ARMS,
    EDGE_CONFIGS_NEWS, EDGE_CONFIGS_SECTOR,
)
from run_storya_e3_news_edge import load_news_edge_source, build_per_day_news_edges
from run_storya_e4_alpha import build_sector_edges
# L7 HATS (§4 frozen lists L7): tuned on the shared NN 6-dim space via its OWN module-global
# HATS_HPARAMS; per-relation GAT needs 3 SEPARATE edge lists, so it can't route through run_arm_cell.
import run_storya_e1_6_hats as hats
from run_storya_e1_6_hats import train_hats, build_three_relation_edges_per_fold

# ══════════════════════════════════════════════════════════════
# §4 TUNING CONFIG (frozen)
# ══════════════════════════════════════════════════════════════
# Tuning window: train = TRAIN_START..2022-06-30 ; val = 2022H2 (used for BOTH early-stop and the
# Rank-IC selection metric, per protocol §1 "调参 val 2022H2 ... 选模/early-stop"). test_end == val_end
# so create_fold_masks yields an empty test slice; we SCORE Rank IC on the val days themselves.
TUNE_FOLD = {'train_end': '2022-06-30', 'val_end': '2022-12-31', 'test_end': '2022-12-31'}
TUNE_SEEDS = [11, 22, 33]          # MUST be disjoint from CANONICAL_SEEDS (startup assert)
N_TRIALS_DEFAULT = 30              # §4 N=30
TOP_K_DEFAULT = 5                  # §4 top-5 (H博士 confirmed 2026-06-15: top-5, NOT top-8)
ALL_ARMS = IMPLEMENTED_ARMS + ['L7']  # L7 HATS tunes the shared NN 6-dim space (structure fixed)

OUT_DIR = 'experiments/storya_v21_tune'
# Optuna sqlite study DBs need POSIX file locking → "unable to open database file" on a Google Drive
# FUSE mount (Colab `experiments/` is a Drive symlink). Env override puts the DBs on local disk; the
# per-study JSON winners still go to OUT_DIR (Drive, persisted across runtime recycles). Default = the
# in-tree path, so local Mac runs (experiments/ on real disk) are byte-identical to before.
STUDY_DIR = os.environ.get('V21_TUNE_STUDY_DIR', f'{OUT_DIR}/studies')

ORIG_NN = dict(anchor.NN_HPARAMS)     # pilot defaults = search-space centers (preserved)
ORIG_LGB = dict(anchor.LGB_HPARAMS)
ORIG_HATS = dict(hats.HATS_HPARAMS)   # = dict(NN_HPARAMS) + num_relations/rel_attn_arch (L7 center)


def assert_tune_seeds_disjoint() -> None:
    """Precheck #6: tuning seeds must not overlap the canonical evaluation seeds."""
    overlap = set(TUNE_SEEDS) & set(CANONICAL_SEEDS)
    assert not overlap, (f'TUNE_SEEDS {TUNE_SEEDS} overlap CANONICAL_SEEDS {CANONICAL_SEEDS}: '
                         f'{sorted(overlap)} — tuning would peek at evaluation seeds (precheck #6).')


# ══════════════════════════════════════════════════════════════
# FROZEN SEARCH SPACE (v2.2 §4) — arm-aware (heads only for GAT/L7; LGB its own dims)
# ══════════════════════════════════════════════════════════════

def sample_hparams(trial: optuna.Trial, model: str) -> dict:
    """Return HP overrides for one trial. Centers = pilot defaults (so the pilot is the N=1
    center-point). MLP/SAGE drop `gat_heads` (no consumption point → would be a ghost dim)."""
    if model == 'LightGBM':
        return {
            'num_leaves': trial.suggest_categorical('num_leaves', [15, 31, 63, 127]),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'min_data_in_leaf': trial.suggest_categorical('min_data_in_leaf', [10, 20, 50, 100]),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 1.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 1.0, log=True),
        }
    hp = {
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
        'dropout': trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.5]),
        'hidden_channels': trial.suggest_categorical('hidden_channels', [32, 64, 128]),
        'num_layers': trial.suggest_categorical('num_layers', [1, 2, 3]),
    }
    if model == 'GAT':   # GAT arms (L2-L6) + L7 HATS (per-relation GAT shares this `heads`)
        hp['gat_heads'] = trial.suggest_categorical('gat_heads', [2, 4, 8])
    return hp


def apply_hparams(model: str, overrides: dict, arm: str = None) -> None:
    """Monkeypatch the module-global HP dict read at call time (anchor.train_nn/make_nn_model/
    train_lightgbm; hats.train_hats for L7). Always starts from the preserved pilot defaults, so
    untouched keys = center. L7 patches hats.HATS_HPARAMS (overrides = the 6 NN dims from the 'GAT'
    search space; num_relations/rel_attn_arch are preserved from ORIG_HATS)."""
    if arm == 'L7':
        hats.HATS_HPARAMS = {**ORIG_HATS, **overrides}
    elif model == 'LightGBM':
        anchor.LGB_HPARAMS = {**ORIG_LGB, **overrides}
    else:
        anchor.NN_HPARAMS = {**ORIG_NN, **overrides}


# ══════════════════════════════════════════════════════════════
# DATA CONTEXT — built ONCE per study; mirrors main12.main() for the single tuning fold
# ══════════════════════════════════════════════════════════════

def build_data_ctx(universe: str, arm: str, device) -> dict:
    anchor.setup_workdir()
    core = load_core_data()
    prices, returns, all_dates = core['prices'], core['returns'], core['all_dates']
    num_days, num_stocks = core['num_days'], core['num_stocks']
    ticker_to_idx = core['ticker_to_id']

    labels_np, label_valid_np = build_labels(prices, HORIZON)
    labels_t = torch.tensor(labels_np, dtype=torch.float32)
    label_valid_t = torch.tensor(label_valid_np, dtype=torch.bool)
    corr_snapshots, _, snapshot_points = build_correlation_snapshots(returns, num_days)

    # single tuning split (expanding from TRAIN_START to 2022-06; val = 2022H2)
    train_days, val_days, _test_empty = create_fold_masks(TUNE_FOLD, all_dates, HORIZON)
    assert len(train_days) > 0 and len(val_days) > 0, 'empty tuning split'
    frozen_si = get_frozen_snapshot_idx(int(train_days[-1]), snapshot_points)
    # LEAK ASSERT (#1): the frozen corr snapshot must end strictly inside the train window
    # (≤ train_end), so NO 2022H2 data enters the graph used during tuning.
    assert snapshot_points[frozen_si] <= int(train_days[-1]), (
        f'LEAK: corr snapshot point {snapshot_points[frozen_si]} > train_end idx {int(train_days[-1])}')

    # features (build_universe_C fires C1 assert (a) at construction; re-confirm per-run)
    if universe == 'B':
        feats_raw, _ = build_universe_B(prices, returns)
    else:
        feats_raw, _ = build_universe_C(prices, returns)
        assert_univ_c_t1_contract(feats_raw)
    feats_winz = winsorize_train_only(feats_raw, train_days)
    feats_std = standardize_train_only(feats_winz, train_days)
    feats_std_t = torch.tensor(feats_std, dtype=torch.float32)

    # arm-specific edges for the tuning window (build_per_day_news_edges fires C1 assert (b) PIT)
    spec = ARM_SPEC[arm] if arm != 'L7' else {'edge': 'corr_sector_news'}
    need_sector = spec['edge'] in EDGE_CONFIGS_SECTOR
    need_news = spec['edge'] in EDGE_CONFIGS_NEWS
    sector_np = build_sector_edges(anchor.PATHS['sectors'], ticker_to_idx) if need_sector else None
    news_snapshots = None
    if need_news:
        news_df = load_news_edge_source()
        news_snapshots, _ = build_per_day_news_edges(news_df, all_dates, ticker_to_idx)
    used_days = np.concatenate([train_days, val_days])
    fold_edges, hats_edges = None, None
    if arm == 'L7':
        # HATS needs 3 SEPARATE relation edge lists per day (corr / sector / news), NOT the union —
        # its own builder, mirroring the confirmatory L7 runner (run_storya_v21_l7_hats.py).
        hats_edges = build_three_relation_edges_per_fold(
            corr_snapshots[frozen_si].cpu().numpy(), sector_np, news_snapshots,
            train_days, val_days, val_days)   # test := val_days (score Rank IC on 2022H2)
    else:
        fold_edges = build_fold_edges([arm], corr_snapshots[frozen_si],
                                      sector_np, news_snapshots, used_days)
    complete_snap = {0: build_complete_graph_edge_index(num_stocks)}

    print(f'[ctx U{universe} {arm}] train={len(train_days)}d val(2022H2)={len(val_days)}d '
          f'frozen_si={frozen_si} (snap≤train_end ✓) stocks={num_stocks}')
    return dict(
        feats_winz=feats_winz, feats_std_t=feats_std_t, labels_np=labels_np, labels_t=labels_t,
        label_valid_np=label_valid_np, label_valid_t=label_valid_t,
        train_days=train_days, val_days=val_days, corr_snapshots=corr_snapshots,
        frozen_si=frozen_si, complete_snap=complete_snap, fold_edges=fold_edges,
        hats_edges=hats_edges,
        num_days=num_days, num_stocks=num_stocks, device=device,
    )


def eval_config(arm: str, ctx: dict, seed: int) -> float:
    """Train the arm at the CURRENT HP under one seed; return mean val Rank IC over 2022H2.
    Scores on val_days (passed as test_days) — early-stop on val, predict on it. L7 routes to the
    HATS trainer (3-relation edges, hats.HATS_HPARAMS); every other arm goes through the confirmatory
    run_arm_cell so a tuning trial trains EXACTLY like the eventual main-table cell."""
    val_days = ctx['val_days']
    if arm == 'L7':
        preds, _info = train_hats(
            ctx['feats_std_t'], ctx['labels_t'], ctx['label_valid_t'],
            ctx['train_days'], val_days, val_days,   # test_days := val_days → Rank IC on 2022H2
            ctx['hats_edges'], ctx['num_days'], ctx['num_stocks'], seed, ctx['device'])  # alpha_log=None
    else:
        preds, _info = run_arm_cell(
            arm, ctx['feats_winz'], ctx['feats_std_t'], ctx['labels_np'], ctx['labels_t'],
            ctx['label_valid_np'], ctx['label_valid_t'],
            ctx['train_days'], val_days, val_days,   # test_days := val_days → score Rank IC on 2022H2
            ctx['corr_snapshots'], ctx['frozen_si'], ctx['complete_snap'], ctx['fold_edges'],
            ctx['num_days'], ctx['num_stocks'], seed, ctx['device'])
    ic = compute_daily_ic(preds, val_days, ctx['labels_np'], ctx['label_valid_np'])
    return float(np.mean(ic)) if len(ic) else float('nan')


# ══════════════════════════════════════════════════════════════
# STUDY: N=30 single-seed search → top-K × 3-seed re-run → winner
# ══════════════════════════════════════════════════════════════

def run_study(arm: str, universe: str, n_trials: int, top_k: int, smoke: bool = False) -> dict:
    model = ARM_SPEC[arm]['model'] if arm != 'L7' else 'GAT'
    ctx = build_data_ctx(universe, arm, get_device())
    search_seed = TUNE_SEEDS[0]
    # CODEX-A-02: smoke uses a SEPARATE study db ({u}_{a}_smoke.db) so a 2-trial wiring check can never
    # contaminate the production study's resume or the launcher merge. Production = {u}_{a}.db.
    study_name = f'{universe}_{arm}_smoke' if smoke else f'{universe}_{arm}'
    storage = f'sqlite:///{STUDY_DIR}/{study_name}.db'

    def objective(trial: optuna.Trial) -> float:
        apply_hparams(model, sample_hparams(trial, model), arm)
        return eval_config(arm, ctx, search_seed)

    # deterministic TPE per study: md5(study_name) is stable across processes, UNLIKE Python's
    # builtin hash() which is PYTHONHASHSEED-salted (would break reproducibility + resume continuity).
    seed = int(hashlib.md5(study_name.encode()).hexdigest(), 16) % (2 ** 31)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='maximize', study_name=study_name,
                                storage=storage, load_if_exists=True, sampler=sampler)
    # CODEX-A-03 (accepted-as-concern): a CLEAN uninterrupted study is fully reproducible (md5-seeded
    # TPE above). load_if_exists resume conditions on the stored trials but the sampler RNG restarts,
    # so a RESUMED study is a valid-but-different trajectory. Resume is the crash-recovery path (Colab
    # runtime recycles); for strict reproducibility re-run a failed study from a clean db.
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if done < n_trials:
        study.optimize(objective, n_trials=n_trials - done)

    # top-K finalists by single-seed search value → re-run each with all 3 tuning seeds
    completed = [t for t in study.trials if t.value is not None and np.isfinite(t.value)]
    completed.sort(key=lambda t: t.value, reverse=True)
    finalists = completed[:top_k]
    top_table = []
    for rank, t in enumerate(finalists):
        apply_hparams(model, t.params, arm)
        seed_ics = [eval_config(arm, ctx, s) for s in TUNE_SEEDS]
        mean_ic = float(np.nanmean(seed_ics))
        top_table.append({'rank': rank, 'params': t.params, 'search_val_ic': float(t.value),
                          'tune_seed_ics': seed_ics, 'mean_val_ic_3seed': mean_ic})
    top_table.sort(key=lambda r: (r['mean_val_ic_3seed'] if np.isfinite(r['mean_val_ic_3seed'])
                                  else -1e9), reverse=True)
    winner = top_table[0]
    return {
        'universe': universe, 'arm': arm, 'model': model,
        'n_trials': len(completed), 'top_k': top_k,
        'tune_seeds': TUNE_SEEDS, 'tune_window': TUNE_FOLD,
        'winner_params': winner['params'], 'winner_mean_val_ic_3seed': winner['mean_val_ic_3seed'],
        'top_table': top_table,   # ALL top-K (regime-mismatch ammunition, not just winner)
        'search_space_ref': 'docs/protocol_v2_freeze.md v2.2 §4',
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=ALL_ARMS)
    ap.add_argument('--universe', required=True, choices=['B', 'C'])
    ap.add_argument('--n-trials', type=int, default=N_TRIALS_DEFAULT)
    ap.add_argument('--top-k', type=int, default=TOP_K_DEFAULT)
    ap.add_argument('--smoke', action='store_true', help='2 trials + top-1 (wiring check only)')
    args = ap.parse_args()

    assert_tune_seeds_disjoint()
    os.makedirs(STUDY_DIR, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_trials, top_k = (2, 1) if args.smoke else (args.n_trials, args.top_k)
    t0 = time.time()
    result = run_study(args.arm, args.universe, n_trials, top_k, smoke=args.smoke)
    result['wall_time_sec'] = round(time.time() - t0, 1)
    result['smoke'] = bool(args.smoke)

    # CODEX-A-02: smoke artifacts go to _smoke_{u}_{a}.json (+ {u}_{a}_smoke.db) — excluded by the
    # launcher merge — so a wiring check can never enter frozen_hparams.json.
    study_tag = f'{args.universe}_{args.arm}_smoke' if args.smoke else f'{args.universe}_{args.arm}'
    out = (f'{OUT_DIR}/_smoke_{args.universe}_{args.arm}.json' if args.smoke
           else f'{OUT_DIR}/{args.universe}_{args.arm}.json')
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    w = result['winner_params']
    print(f'✓ [{args.universe} {args.arm}]{" SMOKE" if args.smoke else ""} winner mean val-IC(3seed)='
          f'{result["winner_mean_val_ic_3seed"]:+.5f}  params={w}  ({result["wall_time_sec"]:.0f}s)')
    print(f'  written {out}  (+ full study {STUDY_DIR}/{study_tag}.db)')


if __name__ == '__main__':
    main()
