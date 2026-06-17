#!/usr/bin/env python
"""
Story A v2.1 MAIN AXIS — expanding 12-fold confirmatory runner.

Per `docs/protocol_v2_freeze.md` v2.1-frozen (§2a main axis, §3 ladder, §5 training, §8 budget)
and Touchpoint 1 disposition `artifacts/reviews/2026-06-12_codex_plan_T1.md`.

WHY A NEW FILE (H博士 2026-06-12 decision, §5 import-only 铁律 / C1):
  The 12-fold confirmatory main axis must NOT re-write any data / edge / snapshot construction
  logic (a large rewrite is the same-day-leak regression entry point — finding C1). This runner
  therefore IMPORTS every data-construction primitive from the E0-validated 5-fold anchor
  (`run_storya_e1_anchor.py`) and writes ONLY: the 12-fold definition (config, not data), the
  complete-graph builder for L6 (the single new architecture, §3), cell_id remap, the per-fold
  runtime asserts, and the E0-canary-on-new-runner. Decision D-RERUN-12F: the 12-fold main table
  is run fresh under frozen hyperparameters; the old 5-fold anchor is pilot/smoke only.

AXIS (§2a): expanding window, train start fixed 2021-07-01, test 2023Q1 → 2025Q4 (12 quarterly
  folds). The "train →YYYY-MM" column of the protocol table = val_end (the expanding window ends
  with its last quarter used as the early-stop val; test follows immediately). This maps exactly
  onto the anchor's create_fold_masks(train/val/test + purge HORIZON) semantics — hence reusable.

ARMS IMPLEMENTED (9 = full ladder except L7 HATS). Frozen-snapshot path = train_nn (single edge
set/fold); per-day path = train_gnn_per_day_edges (edge set switches per day):
  L0  LightGBM            (no graph)              -- non-neural baseline
  L1  MLP                 (no graph)              -- neural value
  L2  GAT  + α1 (corr)    (frozen corr snapshot)  -- graph value
  L3  GAT  + α1∪news      (per-day: corr∪news)    -- news edge marginal value
  L4  GAT  + α2           (frozen: corr∪sector)   -- sector edge marginal value (static union)
  L5  GAT  + α4           (per-day: corr∪sec∪news)-- edge stacking
  L2s SAGE-Mean + α1      (frozen corr snapshot)  -- aggregator control
  L5s SAGE-Mean + α4      (per-day: corr∪sec∪news)-- aggregator control on α4
  L6  GAT  + complete     (complete graph)        -- attention-vs-structure adjudicator (NEW CODE)
      L6 is make_nn_model('GAT') — PARAMETER-IDENTICAL to L2 — fed a complete-graph edge_index
      instead of the corr snapshot. The diff vs the GAT arm is ONLY the edge set ("mask"),
      per protocol §3 / precheck #4. claim_scope: dense learned attention (MASTER/AD-GAT), NOT
      learned-sparse (Cn1).
  Edge CONSTRUCTION is import-only (§5): build_per_day_news_edges (carries C1 assert b:
  max(pub_ts)<=session_close(t-1)) from run_storya_e3_news_edge.py; build_sector_edges/union_*
  from run_storya_e4_alpha.py. train_gnn_per_day_edges mirrors train_nn (only model_name + per-day
  edge lookup differ) so confirmatory pairs (L3−L2, L5−L2, ...) stay apples-to-apples.

FOLLOW-UP (NOT in this file): L7 HATS — SEPARATE runner (run_storya_e1_6_hats.py) under §6 contingency.

CLI:
  --universe {B|C|both}   default both
  --arms L0,L1,L2,...     default all 9 implemented
  --seeds ...             default canonical 10 (must be subset)
  --folds 0..11           default all twelve
  --smoke                 implemented arms × 1 fold × 1 seed (--smoke-fold default 11 = max train)
  --smoke-fold N          which fold to smoke (default 11, the largest expanding window, for §8 单价)
  --canary                E0-canary-on-new-runner (no training): block-fixture mis-map detection +
                          real-data provenance signature match + complete-graph structural assert.
  --resume / --no-resume  default resume ON (manifest.csv)
"""

import os
import sys
import time
import argparse
import json
import hashlib
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # used by train_gnn_per_day_edges (per-day MSE loss)

# ── import-only reuse of the E0-validated anchor (§5 铁律): NO data/edge/snapshot rewrite ──
import run_storya_e1_anchor as anchor
from run_storya_e1_anchor import (
    CANONICAL_SEEDS, HORIZON, TRAIN_START, COST_LEVELS_BPS, COST_CONVENTION,
    load_core_data, build_universe_B, build_universe_C, build_labels,
    build_correlation_snapshots, get_frozen_snapshot_idx, create_fold_masks,
    winsorize_train_only, standardize_train_only, train_nn, train_lightgbm,
    compute_daily_ic, compute_cost_ladder_sharpe, get_device, set_seed,
)
# edge-arm builders (L3/L4/L5/L5s) — import-only edge CONSTRUCTION (§5). build_per_day_news_edges
# carries the C1 assert (b): max(pub_ts) <= session_close(t-1), fired per-day at construction.
from run_storya_e3_news_edge import (
    load_news_edge_source, build_per_day_news_edges, union_edges_per_day,
)
from run_storya_e4_alpha import (
    build_sector_edges, union_static_edges, union_edges_per_day_e4,
)

# ══════════════════════════════════════════════════════════════
# 12-FOLD DEFINITION (§2a) — config only, NOT data construction
# ══════════════════════════════════════════════════════════════
# train_end = last day of train proper; val = the quarter (train_end, val_end]; test = (val_end, test_end].
# Expanding: train_start fixed at TRAIN_START; train_end marches one quarter per fold.
# Protocol table "train →YYYY-MM" == val_end here (expanding window's last quarter = early-stop val).
WALK_FORWARD_FOLDS_12 = [
    {'id': 0,  'train_end': '2022-09-30', 'val_end': '2022-12-31', 'test_end': '2023-03-31', 'desc': '2023Q1'},
    {'id': 1,  'train_end': '2022-12-31', 'val_end': '2023-03-31', 'test_end': '2023-06-30', 'desc': '2023Q2'},
    {'id': 2,  'train_end': '2023-03-31', 'val_end': '2023-06-30', 'test_end': '2023-09-30', 'desc': '2023Q3'},
    {'id': 3,  'train_end': '2023-06-30', 'val_end': '2023-09-30', 'test_end': '2023-12-31', 'desc': '2023Q4'},
    {'id': 4,  'train_end': '2023-09-30', 'val_end': '2023-12-31', 'test_end': '2024-03-31', 'desc': '2024Q1'},
    {'id': 5,  'train_end': '2023-12-31', 'val_end': '2024-03-31', 'test_end': '2024-06-30', 'desc': '2024Q2'},
    {'id': 6,  'train_end': '2024-03-31', 'val_end': '2024-06-30', 'test_end': '2024-09-30', 'desc': '2024Q3'},
    {'id': 7,  'train_end': '2024-06-30', 'val_end': '2024-09-30', 'test_end': '2024-12-31', 'desc': '2024Q4'},
    {'id': 8,  'train_end': '2024-09-30', 'val_end': '2024-12-31', 'test_end': '2025-03-31', 'desc': '2025Q1'},
    {'id': 9,  'train_end': '2024-12-31', 'val_end': '2025-03-31', 'test_end': '2025-06-30', 'desc': '2025Q2'},
    {'id': 10, 'train_end': '2025-03-31', 'val_end': '2025-06-30', 'test_end': '2025-09-30', 'desc': '2025Q3'},
    {'id': 11, 'train_end': '2025-06-30', 'val_end': '2025-09-30', 'test_end': '2025-12-31', 'desc': '2025Q4'},
]
N_FOLDS = len(WALK_FORWARD_FOLDS_12)

# Ladder arm → (torch model_name for make_nn_model, edge_config). Stable ARM_ORDER for cell_id.
ARM_ORDER = ['L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L2s', 'L5s']  # indices locked
ARM_SPEC = {
    'L0':  {'model': 'LightGBM',  'edge': 'none'},
    'L1':  {'model': 'MLP',       'edge': 'none'},
    'L2':  {'model': 'GAT',       'edge': 'corr'},
    'L3':  {'model': 'GAT',       'edge': 'corr_news'},         # α1∪news (per-day dynamic)
    'L4':  {'model': 'GAT',       'edge': 'corr_sector'},       # α2 corr∪sector (static union)
    'L5':  {'model': 'GAT',       'edge': 'corr_sector_news'},  # α4 corr∪sector∪news (per-day)
    'L2s': {'model': 'SAGE-Mean', 'edge': 'corr'},
    'L5s': {'model': 'SAGE-Mean', 'edge': 'corr_sector_news'},  # α4 aggregator control
    'L6':  {'model': 'GAT',       'edge': 'complete'},  # param-identical to L2, edge-only diff
}
# All except L7 (separate HATS runner). Run order: ladder spine, edge DAG, aggregators, L6.
IMPLEMENTED_ARMS = ['L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L2s', 'L5s', 'L6']
EDGE_CONFIGS_NEWS = {'corr_news', 'corr_sector_news'}        # arms needing per-day news edges
EDGE_CONFIGS_SECTOR = {'corr_sector', 'corr_sector_news'}    # arms needing sector edges
ALL_UNIVERSES = ['B', 'C']

OUT_DIR = 'experiments/storya_v21_main12'
RESULTS_CSV = f'{OUT_DIR}/results.csv'
MANIFEST_CSV = f'{OUT_DIR}/manifest.csv'
PER_DAY_IC_DIR = f'{OUT_DIR}/per_day_ic'

# ── §4 frozen-HP injection (D-RERUN-12F + FC arm) ──
# Preserved pilot defaults = the NO-FLAG baseline. With no --frozen-hparams the runner is byte-identical
# to the untuned pilot (the pilot is the comparison). Injection monkeypatches the anchor module-global
# HP dicts per (universe, arm), exactly like run_storya_v21_tune.apply_hparams (verified to reach all
# training paths in the tuning smoke).
_ORIG_NN = dict(anchor.NN_HPARAMS)
_ORIG_LGB = dict(anchor.LGB_HPARAMS)


def load_frozen_hparams(path):
    """Load frozen_hparams.json (run_v21_tune_launcher --merge). Refuse a partial HP set.
    CODEX-A-03: explicit raise (not assert) so the completeness gate survives `python -O`."""
    d = json.load(open(path))
    if not (d.get('complete') and d.get('n_studies') == d.get('expected')):
        raise SystemExit(f"frozen_hparams not complete: {d.get('n_studies')}/{d.get('expected')} "
                         f"(missing {d.get('missing')}); refuse to rerun on a partial HP set")
    return d['studies']


def inject_frozen_hparams(frozen, universe, src_arm, target_model):
    """Monkeypatch the anchor module-global HP dict (read at call time by train_nn / make_nn_model /
    train_lightgbm) with the frozen winner for (universe, src_arm). Starts from the preserved pilot
    defaults so untouched keys = pilot center. `src_arm` == arm for the TUNED rerun; == the FC fixed
    arm (e.g. L2) for the FC arm. `target_model` = the ACTUAL arm's model (selects NN vs LGB dict)."""
    wp = frozen[f'{universe}_{src_arm}']['winner_params']
    if target_model == 'LightGBM':
        anchor.LGB_HPARAMS = {**_ORIG_LGB, **wp}
    else:
        anchor.NN_HPARAMS = {**_ORIG_NN, **wp}
    return wp

V21_RESULTS_COLUMNS = (
    ['cell_id', 'universe', 'arm', 'model', 'edge_config', 'seed', 'fold', 'test_period',
     'IC_mean', 'IC_std', 'n_test_days', 'Sharpe_gross']
    + [f'Sharpe_net_{c}bps' for c in COST_LEVELS_BPS]
    + ['mean_turnover_L1', 'n_periods', 'best_val_loss', 'epochs_run',
       'wall_time_sec', 'converged_flag', 'cost_convention']
)
MANIFEST_COLUMNS = ['cell_id', 'universe', 'arm', 'model', 'seed', 'fold',
                    'status', 'start_ts', 'end_ts', 'wall_time_sec', 'err']


# ══════════════════════════════════════════════════════════════
# CELL_ID (universe × arm × fold × seed) — injective radix, stable across arm additions
# ══════════════════════════════════════════════════════════════

def cell_id(universe_idx: int, arm: str, fold_idx: int, seed_idx: int) -> int:
    """universe*1200 + arm_idx*120 + fold*10 + seed. Range [0, 2399], injective by radix
    (seed<10, fold<12<12→fold*10+seed<120, arm<10→arm*120<1200, universe<2)."""
    arm_idx = ARM_ORDER.index(arm)
    return universe_idx * 1200 + arm_idx * 120 + fold_idx * 10 + seed_idx


def assert_cell_id_injective() -> None:
    """Enumerate the FULL (2 universe × 10 arm × 12 fold × 10 seed) space; confirm injective
    and range [0, 2399] — validates the formula regardless of which arms run this session."""
    seen = set()
    for u in range(2):
        for arm in ARM_ORDER:
            for f in range(N_FOLDS):
                for s in range(10):
                    cid = cell_id(u, arm, f, s)
                    assert cid not in seen, f"cell_id collision u={u} arm={arm} f={f} s={s}"
                    seen.add(cid)
    assert min(seen) == 0 and max(seen) == 2399 and len(seen) == 2400, \
        f"cell_id range broken: min={min(seen)} max={max(seen)} n={len(seen)}"
    print(f'✓ cell_id injective over 2×10×12×10=2400 space, range [0, 2399]')


# ══════════════════════════════════════════════════════════════
# PER-FOLD RUNTIME ASSERTS (§5) — leak prevention on the 12-fold list
# ══════════════════════════════════════════════════════════════

def assert_purge_no_leak_12(all_dates: pd.DatetimeIndex, horizon: int = HORIZON) -> None:
    """§5 purge/embargo check on the 12-fold list (mirrors anchor.assert_purge_no_leak, but the
    anchor version iterates its OWN 5-fold global — so we run the same logic over our 12 folds
    using the imported create_fold_masks for the actual split). Verification only; no data build."""
    for fold in WALK_FORWARD_FOLDS_12:
        train_days, val_days, test_days = create_fold_masks(fold, all_dates, horizon)
        fid = fold['id']
        assert len(train_days) > horizon, f"Fold {fid}: train too short ({len(train_days)}d)"
        assert len(val_days) > horizon, f"Fold {fid}: val too short ({len(val_days)}d)"
        assert len(test_days) > 0, f"Fold {fid}: empty test"
        # feature at trading-day index i has label window trading_dates[i+1..i+horizon];
        # create_fold_masks already purges last `horizon` from train/val, so last train/val
        # feature day's label window must end strictly before the next split's first feature day.
        last_train_label_end = all_dates[train_days[-1] + horizon]
        first_val_feat = all_dates[val_days[0]]
        assert last_train_label_end < first_val_feat, \
            f"Fold {fid}: train label end {last_train_label_end.date()} >= val start {first_val_feat.date()} — LEAK"
        last_val_label_end = all_dates[val_days[-1] + horizon]
        first_test_feat = all_dates[test_days[0]]
        assert last_val_label_end < first_test_feat, \
            f"Fold {fid}: val label end {last_val_label_end.date()} >= test start {first_test_feat.date()} — LEAK"
    print(f'✓ purge/embargo passed for all {N_FOLDS} folds (h={horizon}d)')


def assert_univ_c_t1_contract(features_C: np.ndarray) -> None:
    """§5 assert (a) re-check on the assembled Universe C tensor: the T-1 shift baked in by the
    imported build_universe_C must leave row 0 zeroed (no prior day to lag from). The full
    a158_slice[1]==raw[0] elementwise assert fires INSIDE build_universe_C at construction;
    this is the cheap per-run re-confirmation that the tensor we hold is the shifted one."""
    assert np.all(features_C[0] == 0.0), "Univ-C T-1 contract broken: row 0 not zeroed (CR-A-01)"
    print('✓ Univ-C T-1 shift contract re-confirmed (row 0 zeroed)')


# ══════════════════════════════════════════════════════════════
# L6 — COMPLETE GRAPH (the only new construction; NOT a data/feature rewrite)
# ══════════════════════════════════════════════════════════════

def build_complete_graph_edge_index(num_stocks: int) -> torch.Tensor:
    """All ordered pairs i≠j (GATConv adds self-loops). L6 feeds this to the SAME GATConv as L2,
    so L6 ≡ GAT-on-complete-graph (attention over all stocks, no graph structure). Symmetric."""
    idx = torch.arange(num_stocks, dtype=torch.long)
    src = idx.repeat_interleave(num_stocks)
    dst = idx.repeat(num_stocks)
    keep = src != dst
    return torch.stack([src[keep], dst[keep]], dim=0)


# ══════════════════════════════════════════════════════════════
# PER-DAY DYNAMIC-EDGE TRAINER (L3/L5/L5s) — training code, NOT data construction.
# Mirrors run_storya_e3_news_edge.train_sage_per_day_edges EXACTLY (same optim/scheduler/
# grad-accum/early-stop/purge as the imported train_nn) with model_name parameterized, so the
# confirmatory pairs (L3−L2, L5−L2, ...) are apples-to-apples (only the edge set differs).
# The edge SETS are built by imported builders (union_edges_per_day / union_edges_per_day_e4);
# this function only consumes them.
# ══════════════════════════════════════════════════════════════

def train_gnn_per_day_edges(model_name, features_t, labels_t, label_valid_t,
                            train_days, val_days, test_days, per_day_edges_cpu,
                            num_days, num_stocks, seed, device):
    from run_storya_e1_anchor import make_nn_model, NN_HPARAMS
    set_seed(seed)
    in_ch = features_t.shape[-1]
    model = make_nn_model(model_name, in_ch, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=NN_HPARAMS['lr'],
                                 weight_decay=NN_HPARAMS['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-5)

    used = set(int(d) for d in train_days) | set(int(d) for d in val_days) | set(int(d) for d in test_days)
    per_day_edges = {d: per_day_edges_cpu[d].to(device) for d in used if d in per_day_edges_cpu}

    best_val, best_state, no_improve, epochs_run = float('inf'), None, 0, 0
    for epoch in range(NN_HPARAMS['epochs']):
        epochs_run = epoch + 1
        model.train()
        optimizer.zero_grad()
        accum_steps = 0
        day_order = train_days[np.random.permutation(len(train_days))]
        for step, d in enumerate(day_order):
            d_int = int(d)
            ei = per_day_edges.get(d_int)
            if ei is None:
                continue
            x = features_t[d_int].to(device)
            pred = model(x, ei)
            mask = label_valid_t[d_int].to(device)
            if mask.sum() < 10:
                continue
            loss = F.mse_loss(pred[mask], labels_t[d_int].to(device)[mask])
            (loss / NN_HPARAMS['grad_accum']).backward()
            accum_steps += 1
            if accum_steps >= NN_HPARAMS['grad_accum'] or step == len(day_order) - 1:
                if 0 < accum_steps < NN_HPARAMS['grad_accum']:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(NN_HPARAMS['grad_accum'] / accum_steps)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                accum_steps = 0
        model.eval()
        v_loss, v_cnt = 0.0, 0
        with torch.no_grad():
            for d in val_days:
                d_int = int(d)
                ei = per_day_edges.get(d_int)
                if ei is None:
                    continue
                x = features_t[d_int].to(device)
                pred = model(x, ei)
                mask = label_valid_t[d_int].to(device)
                if mask.sum() < 10:
                    continue
                v_loss += F.mse_loss(pred[mask], labels_t[d_int].to(device)[mask]).item()
                v_cnt += 1
        avg_val = v_loss / max(v_cnt, 1)
        scheduler.step(avg_val)
        if avg_val < best_val:
            best_val, best_state, no_improve = avg_val, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
        if no_improve >= NN_HPARAMS['patience']:
            break

    assert best_state is not None, f"{model_name} per-day seed={seed} produced no valid state"
    model.load_state_dict(best_state)
    model.to(device).eval()
    preds = np.zeros((num_days, num_stocks), dtype=np.float32)
    with torch.no_grad():
        for d in test_days:
            d_int = int(d)
            ei = per_day_edges.get(d_int)
            if ei is None:
                continue
            x = features_t[d_int].to(device)
            preds[d_int] = model(x, ei).cpu().numpy()
    del model, per_day_edges
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return preds, {'best_val_loss': best_val, 'epochs_run': epochs_run}


def build_fold_edges(arms_run, corr_frozen_edge, sector_np, news_snapshots, used_days):
    """Pre-build per-fold edge structures for edge arms (once per fold, reused across seeds).
    All edge CONSTRUCTION via imported builders (§5). Returns dict keyed by edge_config.
      corr_sector       -> {0: static_union_tensor}  (static → frozen-snapshot train_nn path)
      corr_news         -> {day: tensor}              (per-day dynamic)
      corr_sector_news  -> {day: tensor}              (per-day dynamic)
    """
    out = {}
    need = {ARM_SPEC[a]['edge'] for a in arms_run}
    corr_np = corr_frozen_edge.cpu().numpy()
    static_combined = None
    if need & EDGE_CONFIGS_SECTOR:
        static_combined = union_static_edges(corr_np, sector_np)  # corr ∪ sector (static)
    if 'corr_sector' in need:
        out['corr_sector'] = {0: torch.from_numpy(static_combined).long()}
    if 'corr_news' in need:
        out['corr_news'] = union_edges_per_day(corr_np, news_snapshots, used_days)
    if 'corr_sector_news' in need:
        out['corr_sector_news'] = union_edges_per_day_e4(static_combined, news_snapshots, used_days)
    # CODEX-B-01: confirmatory invariant — every used day MUST have an edge set for the dynamic
    # configs. Otherwise train_gnn_per_day_edges' `ei is None -> skip` would silently drop days for
    # L3/L5/L5s but not for L2 (frozen, always-present edge), confounding the edge-effect estimate.
    # Under the current imported union builders every requested day is populated; this makes the
    # invariant CHECKED rather than assumed (mirrors the E3 runner's pre-validation guard).
    used_set = set(int(d) for d in used_days)
    for ec in ('corr_news', 'corr_sector_news'):
        if ec in out:
            missing = used_set - set(int(k) for k in out[ec].keys())
            assert not missing, (f"build_fold_edges: {ec} missing edge sets for {len(missing)} used "
                                 f"days (e.g. {sorted(missing)[:3]}) — per-day skip would confound the pair")
    return out


# ══════════════════════════════════════════════════════════════
# E0-CANARY-ON-NEW-RUNNER (precheck #7 / C1) — no training
# ══════════════════════════════════════════════════════════════

def _within_block_frac(edge_index: torch.Tensor, labels: np.ndarray) -> float:
    ei = edge_index.cpu().numpy()
    if ei.shape[1] == 0:
        return 0.0
    return float(np.mean(labels[ei[0]] == labels[ei[1]]))


def run_canary(core, returns, all_dates, num_days, num_stocks) -> bool:
    """E0-canary-on-new-runner: confirm THIS runner's edge-feeding path is provenance-correct and
    its mis-map detector is live. Three checks, all must PASS:
      (a) block-fixture mis-map sensitivity: build_correlation_snapshots on a known block fixture
          recovers within-block edges (>0.95); the SAME edges scored against permuted labels drop
          (<0.6) — i.e. a ticker/label mis-map WOULD be caught (off-by-permutation negative test).
      (b) real-data provenance match: the frozen corr edge this runner will feed for a fold equals
          the independent recompute (sanity_common.recompute_alpha1_frozen_edges) by signature.
      (c) L6 complete-graph structural integrity: |E| = N(N-1), no self-loops, symmetric.
    """
    import sanity_common as sc
    print('\n=== E0-CANARY-ON-NEW-RUNNER (no training) ===')
    ok = True

    # (a) block-fixture mis-map sensitivity
    N_fix, T_fix = 60, 400
    df_fix, labels = sc.make_block_correlation_fixture(N_fix, T_fix, [20, 20, 20], rho=0.9, seed=0)
    snaps_fix, _, sps_fix = build_correlation_snapshots(df_fix, T_fix)
    si = len(sps_fix) // 2  # mid snapshot index (dict key 0..len-1)
    e_fix = snaps_fix[si]
    within_ok = _within_block_frac(e_fix, labels)
    rng = np.random.default_rng(7)
    perm = rng.permutation(N_fix)
    within_perm = _within_block_frac(e_fix, labels[perm])
    a_pass = (within_ok > 0.95) and (within_perm < 0.6)
    ok &= a_pass
    print(f'[canary-a] block fixture: within_ok={within_ok:.3f} (>0.95), '
          f'within_perm={within_perm:.3f} (<0.6) -> {"PASS" if a_pass else "FAIL"}')

    # (b) real-data provenance match + OFF-BY-ONE NEGATIVE TEST (C1 / precheck #7: off-by-1 必 FAIL).
    #     CODEX-A-01 fix: it is not enough that the expected snapshot matches — the canary must PROVE
    #     it is sensitive to a snapshot off-by-one, i.e. an adjacent snapshot MUST NOT match.
    fold = WALK_FORWARD_FOLDS_12[5]
    snaps, _, snapshot_points = build_correlation_snapshots(returns, num_days)
    train_days, _, _ = create_fold_masks(fold, all_dates, HORIZON)
    runner_si = get_frozen_snapshot_idx(train_days[-1], snapshot_points)
    runner_edge = snaps[runner_si]
    indep_edge, indep_si = sc.recompute_alpha1_frozen_edges(returns, all_dates, fold, num_days)
    sig_indep = sc.edge_index_signature(indep_edge)
    match = (runner_si == indep_si) and (sc.edge_index_signature(runner_edge) == sig_indep)
    offby1_caught, n_adj = True, 0
    for adj in (runner_si - 1, runner_si + 1):
        if 0 <= adj < len(snapshot_points):
            n_adj += 1
            if sc.edge_index_signature(snaps[adj]) == sig_indep:
                offby1_caught = False   # an adjacent snapshot looked identical -> canary would miss off-by-1
    b_pass = match and offby1_caught and (n_adj > 0)
    ok &= b_pass
    print(f'[canary-b] provenance fold {fold["desc"]}: runner_si={runner_si} indep_si={indep_si}, '
          f'match={match}, off-by-1 caught={offby1_caught} ({n_adj} adj) -> {"PASS" if b_pass else "FAIL"}')

    # (c) L6 complete-graph structural integrity — EXHAUSTIVE (CODEX-A-02 fix: was sampled).
    #     Verify |E| AND uniqueness AND no-self AND full symmetry over all edges (cheap at ~250k).
    ce = build_complete_graph_edge_index(num_stocks)
    eu = ce.cpu().numpy()
    no_self = not np.any(eu[0] == eu[1])
    fwd = set(zip(eu[0].tolist(), eu[1].tolist()))
    expected = num_stocks * (num_stocks - 1)
    right_count = (ce.shape[1] == expected) and (len(fwd) == expected)   # count AND uniqueness
    symmetric = all((j, i) in fwd for (i, j) in fwd)                     # exhaustive, not sampled
    c_pass = no_self and right_count and symmetric
    ok &= c_pass
    print(f'[canary-c] complete graph: |E|={ce.shape[1]} unique={len(fwd)} (expect {expected}), '
          f'no_self={no_self}, symmetric={symmetric} -> {"PASS" if c_pass else "FAIL"}')

    print(f'=== E0-CANARY {"ALL PASS ✓" if ok else "FAIL ✗"} ===\n')
    return ok


# ══════════════════════════════════════════════════════════════
# MANIFEST / RESULTS IO
# ══════════════════════════════════════════════════════════════

def init_csv_files() -> None:
    if not os.path.exists(RESULTS_CSV):
        pd.DataFrame(columns=V21_RESULTS_COLUMNS).to_csv(RESULTS_CSV, index=False)
    if not os.path.exists(MANIFEST_CSV):
        pd.DataFrame(columns=MANIFEST_COLUMNS).to_csv(MANIFEST_CSV, index=False)


def load_manifest_done(path: str) -> set:
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    if 'status' not in df.columns or len(df) == 0:
        return set()
    done = df[df['status'] == 'completed']
    return set(zip(done['universe'].astype(str), done['arm'].astype(str),
                   done['seed'].astype(int), done['fold'].astype(int)))


def append_row(path: str, row: dict, cols) -> None:
    pd.DataFrame([row], columns=cols).to_csv(path, mode='a', header=False, index=False)


def write_meta_json() -> None:
    meta = {
        'experiment_id': 'storya_v21_main12',
        'protocol_ref': 'docs/protocol_v2_freeze.md v2.1-frozen §2a/§3/§5/§8',
        'disposition_ref': 'artifacts/reviews/2026-06-12_codex_plan_T1.md',
        'decision': 'D-RERUN-12F (12-fold fresh under frozen hparams; old 5-fold = pilot/smoke)',
        'axis': 'expanding 12-fold, train_start=2021-07-01, test 2023Q1..2025Q4',
        'horizon_days': HORIZON,
        'canonical_seeds': CANONICAL_SEEDS,
        'arms_implemented': IMPLEMENTED_ARMS,
        'arms_followup': ['L7 (separate HATS runner under §6 contingency)'],
        'import_only_source': 'run_storya_e1_anchor.py (E0-validated; no data/edge/snapshot rewrite)',
        'cost_convention': COST_CONVENTION,
        'cost_levels_bps': list(COST_LEVELS_BPS),
        'l6_note': 'L6 = make_nn_model(GAT) on complete graph; param-identical to L2, edge-only diff',
    }
    with open(f'{OUT_DIR}/_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)


# ══════════════════════════════════════════════════════════════
# ARM DISPATCH
# ══════════════════════════════════════════════════════════════

def run_arm_cell(arm, feats_winz, feats_std_t, labels_np, labels_t, label_valid_np, label_valid_t,
                 train_days, val_days, test_days, corr_snapshots, frozen_si, complete_snap,
                 fold_edges, num_days, num_stocks, seed, device):
    """Returns (preds, train_info). Dispatch by edge_config:
      none/corr/complete/corr_sector -> frozen-snapshot train_nn (single edge set per fold);
      corr_news/corr_sector_news     -> train_gnn_per_day_edges (per-day dynamic edges).
    fold_edges holds the per-fold edge sets built by build_fold_edges (imported constructions)."""
    spec = ARM_SPEC[arm]
    ec = spec['edge']
    if spec['model'] == 'LightGBM':
        return train_lightgbm(feats_winz, labels_np, label_valid_np,
                              train_days, val_days, test_days, num_days, num_stocks, seed)
    if ec in ('corr_news', 'corr_sector_news'):   # L3 / L5 / L5s — per-day dynamic edges
        return train_gnn_per_day_edges(spec['model'], feats_std_t, labels_t, label_valid_t,
                                       train_days, val_days, test_days, fold_edges[ec],
                                       num_days, num_stocks, seed, device)
    if ec == 'complete':                          # L6
        snaps, si = complete_snap, 0
    elif ec == 'corr_sector':                     # L4 — static corr∪sector, frozen path
        snaps, si = fold_edges['corr_sector'], 0
    else:                                         # 'corr' (L2/L2s) or 'none' (MLP/L1 ignores edges)
        snaps, si = corr_snapshots, frozen_si
    return train_nn(spec['model'], feats_std_t, labels_t, label_valid_t,
                    train_days, val_days, test_days, snaps, si,
                    num_days, num_stocks, seed, device)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--universe', choices=['B', 'C', 'both'], default='both')
    parser.add_argument('--arms', type=str, default=','.join(IMPLEMENTED_ARMS),
                        help=f'Comma-separated subset of {IMPLEMENTED_ARMS}')
    parser.add_argument('--seeds', type=str, default=','.join(str(s) for s in CANONICAL_SEEDS))
    parser.add_argument('--folds', type=str, default=','.join(str(i) for i in range(N_FOLDS)))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-fold', type=int, default=11,
                        help='Fold to smoke (default 11 = largest expanding window, for §8 单价)')
    parser.add_argument('--canary', action='store_true', help='Run E0-canary only, then exit')
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    parser.add_argument('--frozen-hparams', type=str, default=None,
                        help='Path to frozen_hparams.json → inject per-(universe,arm) tuned HPs '
                             '(D-RERUN-12F). Absent = pilot defaults (byte-identical to the untuned pilot).')
    parser.add_argument('--fc-fix-arm', type=str, default=None,
                        help='FC arm: inject the frozen HPs of THIS arm (e.g. L2) for ALL run arms '
                             '(fixed-capacity edge ablation, Family 2). Requires --frozen-hparams; all '
                             'run arms must share fc-fix-arm model.')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Override output dir (tuned → experiments/storya_v21_main12_tuned/, '
                             'FC → .../_fc/). Absent = the pilot dir (do NOT overwrite the pilot).')
    args = parser.parse_args()

    global OUT_DIR, RESULTS_CSV, MANIFEST_CSV, PER_DAY_IC_DIR
    _pilot_out = OUT_DIR
    if args.out_dir:
        OUT_DIR = args.out_dir.rstrip('/')
        RESULTS_CSV = f'{OUT_DIR}/results.csv'
        MANIFEST_CSV = f'{OUT_DIR}/manifest.csv'
        PER_DAY_IC_DIR = f'{OUT_DIR}/per_day_ic'
    # CODEX-A-01 (fail closed): frozen/FC must NOT write into the untuned pilot dir — resume would skip
    # pilot-completed cells (same cell_ids) or mix tuned/FC rows into the pilot manifest.
    if args.frozen_hparams and OUT_DIR == _pilot_out:
        raise SystemExit(f'--frozen-hparams requires a NEW --out-dir (not the pilot dir {_pilot_out}); '
                         f'e.g. --out-dir experiments/storya_v21_main12_tuned (or .../_fc for the FC arm)')

    anchor.setup_workdir()
    for d in [OUT_DIR, PER_DAY_IC_DIR]:
        os.makedirs(d, exist_ok=True)
    device = get_device()
    print(f'Device: {device}')

    # mandatory startup asserts
    assert_cell_id_injective()

    core = load_core_data()
    prices, returns, all_dates = core['prices'], core['returns'], core['all_dates']
    num_days, num_stocks = core['num_days'], core['num_stocks']

    assert_purge_no_leak_12(all_dates, HORIZON)

    if args.canary:
        ok = run_canary(core, returns, all_dates, num_days, num_stocks)
        sys.exit(0 if ok else 1)

    # labels + corr snapshots (imported builders)
    labels_np, label_valid_np = build_labels(prices, HORIZON)
    labels_t = torch.tensor(labels_np, dtype=torch.float32)
    label_valid_t = torch.tensor(label_valid_np, dtype=torch.bool)
    corr_snapshots, _, snapshot_points = build_correlation_snapshots(returns, num_days)
    complete_snap = {0: build_complete_graph_edge_index(num_stocks)}
    print(f'Labels h={HORIZON}d, {int(label_valid_np.sum()):,} valid; '
          f'{len(snapshot_points)} corr snapshots; complete graph |E|={complete_snap[0].shape[1]}')

    # resolve CLI
    if args.smoke:
        universes_run, arms_run = ['B'], IMPLEMENTED_ARMS
        seeds_run, folds_run = [CANONICAL_SEEDS[0]], [args.smoke_fold]
        print(f'\n=== SMOKE: {len(arms_run)} arms × fold {args.smoke_fold} × seed {seeds_run[0]} (Univ B) ===\n')
    else:
        universes_run = ALL_UNIVERSES if args.universe == 'both' else [args.universe]
        arms_run = [a for a in args.arms.split(',') if a]
        seeds_run = [int(s) for s in args.seeds.split(',') if s]
        folds_run = [int(f) for f in args.folds.split(',') if f != '']
        for a in arms_run:
            assert a in IMPLEMENTED_ARMS, f'Arm {a} not implemented this build; have {IMPLEMENTED_ARMS}'
        for s in seeds_run:
            assert s in CANONICAL_SEEDS, f'Seed {s} not canonical: {CANONICAL_SEEDS}'

    init_csv_files()
    write_meta_json()
    universe_idx_map = {'B': 0, 'C': 1}
    seed_idx_map = {s: i for i, s in enumerate(CANONICAL_SEEDS)}
    done_cells = load_manifest_done(MANIFEST_CSV) if args.resume else set()
    print(f'Resume {"ON" if args.resume else "OFF"}: {len(done_cells)} cells already done')

    # ── frozen-HP injection setup (D-RERUN-12F tuned rerun / FC edge arm) ──
    frozen = load_frozen_hparams(args.frozen_hparams) if args.frozen_hparams else None
    if args.fc_fix_arm:   # CODEX-A-03: explicit raises (survive python -O)
        if frozen is None:
            raise SystemExit('--fc-fix-arm requires --frozen-hparams')
        fc_model = ARM_SPEC[args.fc_fix_arm]['model']
        for a in arms_run:
            if ARM_SPEC[a]['model'] != fc_model:
                raise SystemExit(f'--fc-fix-arm {args.fc_fix_arm} ({fc_model}) but arm {a} is '
                                 f'{ARM_SPEC[a]["model"]}; FC requires same-model arms (edge family is all GAT)')
    if frozen:
        mode = f'FC fixed-arm={args.fc_fix_arm}' if args.fc_fix_arm else 'TUNED per-arm'
        fz_md5 = hashlib.md5(open(args.frozen_hparams, 'rb').read()).hexdigest()
        # CODEX-A-02: MERGE provenance (subset invocations union, never clobber) + validate mode/md5.
        prov_path = f'{OUT_DIR}/_frozen_hp_provenance.json'
        if os.path.exists(prov_path):
            prov = json.load(open(prov_path))
            if prov.get('mode') != mode or prov.get('frozen_md5') != fz_md5:
                raise SystemExit(f'{prov_path} exists with mode={prov.get("mode")}/md5={prov.get("frozen_md5","?")[:8]} '
                                 f'!= this run mode={mode}/md5={fz_md5[:8]}; refuse to mix HP modes/files in one dir')
        else:
            prov = {'mode': mode, 'frozen_hparams': args.frozen_hparams, 'frozen_md5': fz_md5,
                    'fc_fix_arm': args.fc_fix_arm, 'applied': {}}
        print(f'Frozen-HP injection [{mode}] from {args.frozen_hparams} (md5 {fz_md5[:8]}):')
        for u in universes_run:
            for a in arms_run:
                src = args.fc_fix_arm if args.fc_fix_arm else a
                wp = frozen[f'{u}_{src}']['winner_params']
                prov['applied'][f'{u}_{a}'] = {'src': f'{u}_{src}', 'params': wp}
                print(f'  U{u} {a:>3s} ← {u}_{src}: {wp}')
        with open(prov_path, 'w') as f:
            json.dump(prov, f, indent=2)
    else:
        print('Frozen-HP injection: OFF (pilot defaults — untuned baseline)')

    # build features per universe once (per-fold winsor/scale inside)
    features_raw = {}
    if 'B' in universes_run:
        fB, _ = build_universe_B(prices, returns)
        features_raw['B'] = fB
        print(f'Universe B features: {fB.shape}')
    if 'C' in universes_run:
        fC, _ = build_universe_C(prices, returns)   # fires C1 assert (a) at construction
        assert_univ_c_t1_contract(fC)               # §5 per-run re-confirmation
        features_raw['C'] = fC
        print(f'Universe C features: {fC.shape}')

    # ── edge-arm setup (once): sector (static) + per-day news edges. build_per_day_news_edges
    #    fires the C1 assert (b) max(pub_ts)<=session_close(t-1) per-day at construction. ──
    ticker_to_idx = core['ticker_to_id']
    need_sector = any(ARM_SPEC[a]['edge'] in EDGE_CONFIGS_SECTOR for a in arms_run)
    need_news = any(ARM_SPEC[a]['edge'] in EDGE_CONFIGS_NEWS for a in arms_run)
    sector_np = build_sector_edges(anchor.PATHS['sectors'], ticker_to_idx) if need_sector else None
    news_snapshots = None
    if need_news:
        news_df = load_news_edge_source()
        news_snapshots, _ = build_per_day_news_edges(news_df, all_dates, ticker_to_idx)
        print(f'News per-day edges built (C1 assert b PIT-checked at construction); '
              f'sector edges={"on" if need_sector else "off"}')

    t0 = time.time()
    smoke_rows = []
    for universe in universes_run:
        feats_raw = features_raw[universe]
        u_idx = universe_idx_map[universe]
        for fold in WALK_FORWARD_FOLDS_12:
            if fold['id'] not in folds_run:
                continue
            train_days, val_days, test_days = create_fold_masks(fold, all_dates, HORIZON)
            frozen_si = get_frozen_snapshot_idx(train_days[-1], snapshot_points)
            used_days = np.concatenate([train_days, val_days, test_days])
            fold_edges = build_fold_edges(arms_run, corr_snapshots[frozen_si],
                                          sector_np, news_snapshots, used_days)
            feats_winz = winsorize_train_only(feats_raw, train_days)
            feats_std = standardize_train_only(feats_winz, train_days)
            feats_std_t = torch.tensor(feats_std, dtype=torch.float32)
            print(f'\n[U{universe}] fold {fold["id"]} ({fold["desc"]}): train={len(train_days)}d '
                  f'val={len(val_days)}d test={len(test_days)}d frozen_si={frozen_si}')

            for arm in arms_run:
                spec = ARM_SPEC[arm]
                if frozen:   # inject (universe,arm) tuned HPs — or the FC fixed arm's HPs — before training
                    inject_frozen_hparams(frozen, universe,
                                          args.fc_fix_arm if args.fc_fix_arm else arm, spec['model'])
                for seed in seeds_run:
                    s_idx = seed_idx_map[seed]
                    cid = cell_id(u_idx, arm, fold['id'], s_idx)
                    key = (universe, arm, int(seed), int(fold['id']))
                    if key in done_cells:
                        continue
                    print(f'  [cid={cid:04d}] U{universe} {arm:>3s}/{spec["model"]:<9s} '
                          f'seed={seed:<5d} fold={fold["id"]} ...', end=' ', flush=True)
                    start_ts = time.time()
                    status, err = 'running', ''
                    try:
                        preds, info = run_arm_cell(
                            arm, feats_winz, feats_std_t, labels_np, labels_t,
                            label_valid_np, label_valid_t, train_days, val_days, test_days,
                            corr_snapshots, frozen_si, complete_snap,
                            fold_edges, num_days, num_stocks, seed, device)
                        ic_arr = compute_daily_ic(preds, test_days, labels_np, label_valid_np)
                        ic_mean = float(ic_arr.mean()) if len(ic_arr) else 0.0
                        ic_std = float(ic_arr.std()) if len(ic_arr) else 0.0
                        sh = compute_cost_ladder_sharpe(preds, test_days, prices, label_valid_np,
                                                        num_stocks, num_days, horizon=HORIZON)
                        np.save(f'{PER_DAY_IC_DIR}/{universe}_{arm}_s{seed}_f{fold["id"]}.npy', ic_arr)
                        wall = time.time() - start_ts
                        row = {
                            'cell_id': cid, 'universe': universe, 'arm': arm, 'model': spec['model'],
                            'edge_config': spec['edge'], 'seed': seed, 'fold': fold['id'],
                            'test_period': fold['desc'],
                            'IC_mean': round(ic_mean, 6), 'IC_std': round(ic_std, 6),
                            'n_test_days': len(ic_arr), 'Sharpe_gross': round(sh['Sharpe_gross'], 4),
                            **{k: round(v, 4) for k, v in sh.items() if k.startswith('Sharpe_net_')},
                            'mean_turnover_L1': round(sh['mean_turnover_L1'], 4), 'n_periods': sh['n_periods'],
                            'best_val_loss': round(info['best_val_loss'], 6), 'epochs_run': info['epochs_run'],
                            'wall_time_sec': round(wall, 1),
                            'converged_flag': int(-np.inf < info['best_val_loss'] < np.inf),
                            'cost_convention': COST_CONVENTION,
                        }
                        append_row(RESULTS_CSV, row, V21_RESULTS_COLUMNS)
                        status = 'completed'
                        print(f'IC={ic_mean:+.5f}±{ic_std:.5f} Sh_g={sh["Sharpe_gross"]:.3f} ({wall:.0f}s)')
                        if args.smoke:
                            smoke_rows.append({'arm': arm, 'model': spec['model'],
                                               'wall_time_sec': round(wall, 1), 'IC_mean': round(ic_mean, 5)})
                    except Exception as e:
                        status, err = 'failed', str(e)[:200]
                        import traceback; traceback.print_exc()
                        print(f'  FAILED: {err}')
                    end_ts = time.time()
                    append_row(MANIFEST_CSV, {
                        'cell_id': cid, 'universe': universe, 'arm': arm, 'model': spec['model'],
                        'seed': seed, 'fold': fold['id'], 'status': status,
                        'start_ts': round(start_ts, 1), 'end_ts': round(end_ts, 1),
                        'wall_time_sec': round(end_ts - start_ts, 1), 'err': err,
                    }, MANIFEST_COLUMNS)
                    gc.collect()
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
            del feats_winz, feats_std, feats_std_t
            gc.collect()

    if args.smoke and smoke_rows:
        total = sum(r['wall_time_sec'] for r in smoke_rows)
        print(f'\n=== SMOKE SUMMARY (fold {args.smoke_fold}) ===')
        for r in smoke_rows:
            print(f'  {r["arm"]:>3s}/{r["model"]:<9s}: {r["wall_time_sec"]:.0f}s  IC={r["IC_mean"]:+.5f}')
        print(f'  total {len(smoke_rows)} cells: {total:.0f}s ({total/60:.1f} min)')
        pd.DataFrame(smoke_rows).to_csv(f'{OUT_DIR}/smoke_benchmark.csv', index=False)

    print(f'\nTotal wall: {(time.time()-t0)/3600:.2f}h')
    print('=== run_storya_v21_main12.py DONE ===')


if __name__ == '__main__':
    main()
