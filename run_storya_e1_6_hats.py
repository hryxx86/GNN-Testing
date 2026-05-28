#!/usr/bin/env python
"""run_storya_e1_6_hats.py — Story A §1.6 STRETCH: HATS-3R-adapt baseline.

This script is an ADAPTATION (not a literal reproduction) of:
    Kim, Raehyun et al., 2019.
    "HATS: A Hierarchical Graph Attention Network for Stock Movement Prediction"
    arXiv:1908.07999.

ADAPTATIONS vs Kim 2019 (locked per Codex Plan Round A disposition,
artifacts/reviews/2026-05-27_codex_plan_A.md, findings A-05/A-06/A-07):
  1. No per-stock GRU/LSTM encoder. Universe B 10-dim hc features are used directly
     as node features (apples-to-apples with E1 GAT).
  2. Relation set substituted: Wikidata corporate relations (75 types) -> 3 project
     relations: correlation_frozen, sector_GICS11_static, news_cooccurrence_PIT.
  3. Task adapted: classification (up/neutral/down) -> regression on 21d CS-z-scored
     forward returns. MSE loss instead of cross-entropy.
  4. Relation attention scoring simplified: Linear(64, 1) shared across relations
     instead of Kim Section 3.2 relation-embedding-concatenation scorer.

The `model` column in results.csv is "HATS-3R-adapt". Any positive/negative
Template-1 conclusion speaks ONLY to this adapted module on strict 21d ranking,
NOT to Kim's published HATS performance.

50 cells = 1 model x 1 universe (B) x 10 canonical seeds x 5 walk-forward folds.
cell_id = 400 + fold_idx*10 + seed_idx, range [400, 449], offset to avoid E1 [0,399].

Outputs (mirrors E3 schema):
  experiments/storya_e1_6_hats/
    results.csv       one row per cell, schema = E1 RESULTS_COLUMNS + alpha/edge cols
    manifest.csv      resume bookkeeping
    per_day_ic/       per-cell per-day IC arrays
    hp_grid.json      locked HATS-3R-adapt hyperparameters
    prereg.json       locked decision rules + claim scope
    _meta.json        run metadata
    alpha_diag/       per-cell per-epoch alpha statistics (DIAGNOSTIC only)

CLI:
  --seeds, --folds  subset of canonical 10 seeds / 5 folds (default: all)
  --smoke           1 cell on fold 0 seed 86 (shape + train + eval smoke)
  --resume          default ON; honors manifest.csv

Rule 9 chain: this code is written under Plan Touchpoint 1 PROCEED-WITH-FIXES verdict
(artifacts/reviews/2026-05-27_codex_plan_A.md). Touchpoint 2 code review must run
BEFORE 50-cell Colab launch.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from collections import defaultdict
from itertools import combinations
from typing import Optional

# macOS OpenMP segfault workaround (Darwin only) — must be BEFORE numpy/torch import
if platform.system() == 'Darwin':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Reuse E1 anchor (import is idempotent; main() is __name__ guarded)
import run_storya_e1_anchor as e1
from run_storya_e1_anchor import (
    CANONICAL_SEEDS, WALK_FORWARD_FOLDS, HORIZON, TRAIN_START,
    NN_HPARAMS, GRAPH_HPARAMS,
    set_seed, get_device,
    load_core_data, build_universe_B, build_labels,
    build_correlation_snapshots, get_frozen_snapshot_idx,
    create_fold_masks, winsorize_train_only, standardize_train_only,
    compute_daily_ic, compute_cost_ladder_sharpe,
    assert_purge_no_leak,
    COST_LEVELS_BPS, COST_CONVENTION,
)

# Reuse E3 news edge machinery
from run_storya_e3_news_edge import (
    load_news_edge_source, build_per_day_news_edges,
    NEWS_LOOKBACK_TRADING_DAYS, NEWS_LOOKBACK_CALENDAR_DAYS,
)

# Reuse E4 sector edge builder (Codex A-04 deferred; sector PIT is project-level §Limitations)
from run_storya_e4_alpha import build_sector_edges


# ══════════════════════════════════════════════════════════════
# CONFIG — HATS-3R-adapt (LOCKED per plan + Codex Round A disposition)
# ══════════════════════════════════════════════════════════════

MODEL = 'HATS-3R-adapt'  # renamed per Codex A-05/06/07 disposition
UNIVERSE = 'B'           # Universe B only (per H博士 2026-05-27 decision)

# HATS_HPARAMS = E1 NN_HPARAMS + 2 HATS-specific keys.
# LOCKED per plan §"Pre-registration"; mirrors E1 GAT for apples-to-apples.
HATS_HPARAMS = dict(NN_HPARAMS)
HATS_HPARAMS['num_relations'] = 3
HATS_HPARAMS['rel_attn_arch'] = 'linear_shared'  # Linear(hidden, 1) shared across relations
RELATIONS_IN_ORDER = ['correlation_frozen', 'sector_GICS11_static', 'news_cooccurrence_PIT']

PATHS = {
    'prices': 'data/reference/sp500_5y_prices.csv',
    'sectors': 'data/reference/sp500_sectors.csv',
    'phase5_npy': 'data/reference/sp500_5y_phase5_features.npy',
    'news_edge_source': 'data/fullscale/sp500_news_edge_source.parquet',
}

OUT_DIR = 'experiments/storya_e1_6_hats'
RESULTS_CSV = f'{OUT_DIR}/results.csv'
MANIFEST_CSV = f'{OUT_DIR}/manifest.csv'
PER_DAY_IC_DIR = f'{OUT_DIR}/per_day_ic'
ALPHA_DIAG_DIR = f'{OUT_DIR}/alpha_diag'
HP_GRID_JSON = f'{OUT_DIR}/hp_grid.json'
PREREG_JSON = f'{OUT_DIR}/prereg.json'
META_JSON = f'{OUT_DIR}/_meta.json'
NEWS_SNAPSHOT_CACHE = 'experiments/storya_e3_news_edge/news_snapshots_cache.npz'

# Schema = E1 RESULTS_COLUMNS + 4 HATS-specific diagnostic columns.
RESULTS_COLUMNS = (
    ['cell_id', 'universe', 'model', 'seed', 'fold', 'test_period',
     'IC_mean', 'IC_std', 'n_test_days', 'Sharpe_gross']
    + [f'Sharpe_net_{c}bps' for c in COST_LEVELS_BPS]
    + ['mean_turnover_L1', 'n_periods', 'best_val_loss', 'epochs_run',
       'wall_time_sec', 'converged_flag', 'cost_convention',
       'n_corr_edges', 'n_sector_edges', 'n_news_edges_avg',
       'alpha_mean_corr_test', 'alpha_mean_sector_test', 'alpha_mean_news_test',
       'alpha_max_fraction_collapsed_test']
)
MANIFEST_COLUMNS = ['cell_id', 'fold', 'seed', 'status', 'start_ts', 'end_ts',
                    'wall_time_sec', 'err']


def cell_id_hats(fold_idx: int, seed_idx: int) -> int:
    """HATS-3R-adapt cell_id: 400 + fold_idx*10 + seed_idx, range [400, 449].
    Offset by 400 to avoid collision with E1 cell_id range [0, 399]
    (Codex Plan Round A finding CODEX-A-04 disposition).
    """
    return 400 + fold_idx * 10 + seed_idx


def assert_cell_id_hats_injective_and_disjoint_from_e1() -> None:
    """Codex A-04 disposition: HATS ids must be injective AND non-overlapping with
    E1 ids [0, 399]. Enumerate all 50 HATS cells; assert range == [400, 449]."""
    seen = set()
    for f in range(5):
        for s in range(10):
            cid = cell_id_hats(f, s)
            assert cid not in seen, f"cell_id_hats collision at f={f}, s={s}"
            assert 400 <= cid <= 449, f"cell_id_hats({f},{s})={cid} outside [400,449]"
            seen.add(cid)
    assert len(seen) == 50 and min(seen) == 400 and max(seen) == 449
    print(f"✓ HATS cell_id formula injective in [400, 449], n=50; disjoint from E1 [0, 399]")


# ══════════════════════════════════════════════════════════════
# HATS-3R-ADAPT MODULE
# ══════════════════════════════════════════════════════════════

class HATS3RAdapt(nn.Module):
    """HATS-style three-relation attention adaptation.

    Architecture (LOCKED per plan §"Architecture" + Codex Round A scope clarification):

    forward(x: (N, F), edge_index_per_relation: [ei_corr, ei_sector, ei_news_day]) -> (N,)
      h = ReLU(Linear(F, hidden))(x)
      for layer in {1, 2}:
        for r in {0, 1, 2}:
          h_r = LayerNorm( Dropout(GATConv_r(h, ei_r)) + h )    # per-relation GAT + residual + LN
        h_stack = stack(h_0, h_1, h_2)                          # (N, 3, hidden)
        alpha = softmax( Linear(hidden, 1, shared)(h_stack), dim=relation )   # (N, 3)
        h = sum_r alpha_r * h_r                                 # hierarchical aggregation
      return Sequential(Linear, ReLU, Dropout, Linear)(h).squeeze(-1)

    Diagnostic instrumentation:
      .last_alpha is a (N, num_relations) tensor from the second layer's softmax,
      captured at every forward pass (cleared each call). Used for per-epoch alpha
      logging in train_hats (mean/max across relations).

    Adaptation note (Codex A-05/06/07 disposition): The relation attention scorer is
    a SHARED Linear(hidden, 1) which simplifies Kim Section 3.2's relation-embedding
    concatenation. This is intentional — see prereg.json claim_scope.
    """

    def __init__(self, in_channels: int, hidden: int = 64, num_relations: int = 3,
                 heads: int = 4, dropout: float = 0.3, num_layers: int = 2):
        super().__init__()
        from torch_geometric.nn import GATConv

        assert num_relations >= 1, "HATS requires >=1 relation"
        assert hidden % heads == 0, f"hidden={hidden} must divide heads={heads} cleanly"
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.dropout = dropout

        self.lin_in = nn.Linear(in_channels, hidden)

        # Per-layer × per-relation GATConv stack
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            relation_convs = nn.ModuleList([
                GATConv(hidden, hidden // heads, heads=heads,
                        dropout=dropout, add_self_loops=True)
                for _ in range(num_relations)
            ])
            relation_norms = nn.ModuleList([
                nn.LayerNorm(hidden) for _ in range(num_relations)
            ])
            self.layers.append(relation_convs)
            self.norms.append(relation_norms)

        # Shared relation-attention scorer (simplified vs Kim §3.2)
        self.rel_attn_score = nn.Linear(hidden, 1, bias=False)

        # Regression head — identical signature to E1 make_nn_model head (L480-483)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden // 2, 1),
        )

        # Diagnostic: captured each forward pass
        self.last_alpha: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor,
                edge_index_per_relation: list[torch.Tensor]) -> torch.Tensor:
        assert len(edge_index_per_relation) == self.num_relations, (
            f"Expected {self.num_relations} edge_index tensors, got "
            f"{len(edge_index_per_relation)}"
        )

        h = F.relu(self.lin_in(x))  # (N, hidden)

        last_alpha = None
        for layer_idx, (relation_convs, relation_norms) in enumerate(zip(self.layers, self.norms)):
            h_rels = []
            for r, (conv, norm) in enumerate(zip(relation_convs, relation_norms)):
                ei = edge_index_per_relation[r]
                # Edge case: empty edge_index (E_r == 0). add_self_loops=True ensures
                # GATConv still produces a valid output (self-attention only on this relation).
                h_r = conv(h, ei)
                h_r = F.dropout(h_r, p=self.dropout, training=self.training)
                h_r = norm(h_r + h)  # residual + LN, matching E1 _NN pattern
                h_rels.append(h_r)
            h_stack = torch.stack(h_rels, dim=1)  # (N, R, hidden)

            # Hierarchical relation-attention (softmax across relation dim per node)
            attn_logits = self.rel_attn_score(h_stack).squeeze(-1)  # (N, R)
            alpha = F.softmax(attn_logits, dim=-1)                  # (N, R)
            h = (alpha.unsqueeze(-1) * h_stack).sum(dim=1)          # (N, hidden)

            last_alpha = alpha  # only the final layer's alpha is exported as diagnostic

        self.last_alpha = last_alpha.detach() if last_alpha is not None else None
        return self.head(h).squeeze(-1)


# ══════════════════════════════════════════════════════════════
# THREE-RELATION EDGE BUILDER (per-fold)
# ══════════════════════════════════════════════════════════════

def build_three_relation_edges_per_fold(
    corr_frozen_np: np.ndarray,
    sector_static_np: np.ndarray,
    news_snapshots: dict[int, np.ndarray],
    train_days: np.ndarray,
    val_days: np.ndarray,
    test_days: np.ndarray,
) -> dict[int, list[np.ndarray]]:
    """Per-day per-relation edge_index dict.

    Output: { day_idx: [ei_corr (2,E1), ei_sector (2,E2), ei_news_day (2,E3)] }

    The three relations are kept SEPARATE (NOT unioned into a single edge_index).
    HATS needs per-relation GATConv to compute per-relation attention; if we unioned
    the edges, we'd lose the relation identity.

    Edge sources:
      - corr_frozen_np: frozen at fold's train_end snapshot, constant for the fold
      - sector_static_np: GICS-11 fully-connected within-sector, constant for the run
      - news_snapshots: per-day PIT-filtered co-occurrence from E3 build_per_day_news_edges
    """
    all_days = np.concatenate([train_days, val_days, test_days])
    out: dict[int, list[np.ndarray]] = {}
    for d in all_days:
        d = int(d)
        # Per-relation edge_index (np.int64 shape (2, E))
        ei_corr = corr_frozen_np                                         # frozen
        ei_sector = sector_static_np                                     # static
        ei_news = news_snapshots.get(d, np.zeros((2, 0), dtype=np.int64))  # per-day
        out[d] = [ei_corr, ei_sector, ei_news]
    return out


# ══════════════════════════════════════════════════════════════
# TRAIN HATS-3R-ADAPT
# ══════════════════════════════════════════════════════════════

def train_hats(features_t: torch.Tensor, labels_t: torch.Tensor,
               label_valid_t: torch.Tensor,
               train_days: np.ndarray, val_days: np.ndarray, test_days: np.ndarray,
               per_day_edges_cpu: dict[int, list[np.ndarray]],
               num_days: int, num_stocks: int,
               seed: int, device: torch.device,
               alpha_log_path: Optional[str] = None) -> tuple[np.ndarray, dict]:
    """HATS-3R-adapt training with per-day three-relation edge_index lookup.

    Mirrors train_sage_per_day_edges (E3) but threads a *list* of 3 edge_index tensors
    per day (one per relation) instead of a single tensor.

    Diagnostic: writes per-epoch (alpha_mean_r0, alpha_mean_r1, alpha_mean_r2,
    alpha_max_frac_collapsed) to alpha_log_path if provided (CSV).
    """
    set_seed(seed)
    in_ch = features_t.shape[-1]
    model = HATS3RAdapt(
        in_channels=in_ch,
        hidden=HATS_HPARAMS['hidden_channels'],
        num_relations=HATS_HPARAMS['num_relations'],
        heads=HATS_HPARAMS['gat_heads'],
        dropout=HATS_HPARAMS['dropout'],
        num_layers=HATS_HPARAMS['num_layers'],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=HATS_HPARAMS['lr'],
                                 weight_decay=HATS_HPARAMS['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5, min_lr=1e-5
    )

    # Pre-move all per-day per-relation edges to device once (memory cost: ~3x E3,
    # but constant for the fold across epochs).
    used_days = (set(int(d) for d in train_days)
                 | set(int(d) for d in val_days)
                 | set(int(d) for d in test_days))
    per_day_edges: dict[int, list[torch.Tensor]] = {}
    for d in used_days:
        if d not in per_day_edges_cpu:
            continue
        per_day_edges[d] = [
            torch.from_numpy(ei).long().to(device) if ei.shape[1] > 0
            else torch.zeros((2, 0), dtype=torch.long, device=device)
            for ei in per_day_edges_cpu[d]
        ]

    best_val = float('inf')
    best_state = None
    no_improve = 0
    epochs_run = 0

    alpha_log_rows = []  # diagnostic per-epoch alpha stats

    for epoch in range(HATS_HPARAMS['epochs']):
        epochs_run = epoch + 1
        model.train()
        optimizer.zero_grad()
        accum_steps = 0
        day_order = train_days[np.random.permutation(len(train_days))]

        for step, d in enumerate(day_order):
            d_int = int(d)
            ei_list = per_day_edges.get(d_int)
            if ei_list is None:
                continue
            x = features_t[d_int].to(device)
            pred = model(x, ei_list)
            mask = label_valid_t[d_int].to(device)
            if mask.sum() < 10:
                continue
            target = labels_t[d_int].to(device)
            loss = F.mse_loss(pred[mask], target[mask])
            (loss / HATS_HPARAMS['grad_accum']).backward()
            accum_steps += 1
            if accum_steps >= HATS_HPARAMS['grad_accum'] or step == len(day_order) - 1:
                if 0 < accum_steps < HATS_HPARAMS['grad_accum']:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(HATS_HPARAMS['grad_accum'] / accum_steps)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                accum_steps = 0

        # Validation + alpha diagnostic
        model.eval()
        v_loss, v_cnt = 0.0, 0
        # Aggregate alpha statistics across all val days for this epoch
        epoch_alpha_means = np.zeros(HATS_HPARAMS['num_relations'], dtype=np.float64)
        epoch_alpha_max_frac_collapsed = 0.0
        v_alpha_day_count = 0
        with torch.no_grad():
            for d in val_days:
                d_int = int(d)
                ei_list = per_day_edges.get(d_int)
                if ei_list is None:
                    continue
                x = features_t[d_int].to(device)
                pred = model(x, ei_list)
                mask = label_valid_t[d_int].to(device)
                if mask.sum() < 10:
                    continue
                v_loss += F.mse_loss(pred[mask], labels_t[d_int].to(device)[mask]).item()
                v_cnt += 1

                # Alpha diagnostic on this val day
                if model.last_alpha is not None:
                    a = model.last_alpha.cpu().numpy()  # (N, R)
                    epoch_alpha_means += a.mean(axis=0)
                    # Fraction of nodes with one relation > 0.9 (collapse indicator)
                    epoch_alpha_max_frac_collapsed += float((a.max(axis=1) > 0.9).mean())
                    v_alpha_day_count += 1

        avg_val = v_loss / max(v_cnt, 1)
        if v_alpha_day_count > 0:
            epoch_alpha_means /= v_alpha_day_count
            epoch_alpha_max_frac_collapsed /= v_alpha_day_count

        scheduler.step(avg_val)
        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        alpha_log_rows.append({
            'epoch': epoch + 1,
            'avg_val_loss': float(avg_val),
            'alpha_mean_corr': float(epoch_alpha_means[0]),
            'alpha_mean_sector': float(epoch_alpha_means[1]),
            'alpha_mean_news': float(epoch_alpha_means[2]),
            'alpha_max_frac_collapsed': float(epoch_alpha_max_frac_collapsed),
            'is_best_so_far': int(avg_val == best_val),
        })

        if no_improve >= HATS_HPARAMS['patience']:
            break

    assert best_state is not None, f"HATS-3R-adapt seed={seed} produced no valid state"

    # Persist per-epoch alpha diagnostic CSV (if path given)
    if alpha_log_path is not None and alpha_log_rows:
        pd.DataFrame(alpha_log_rows).to_csv(alpha_log_path, index=False)

    # Re-load best state and predict on test
    model.load_state_dict(best_state)
    model.to(device).eval()

    preds = np.zeros((num_days, num_stocks), dtype=np.float32)
    # Aggregate test-set alpha statistics
    test_alpha_means = np.zeros(HATS_HPARAMS['num_relations'], dtype=np.float64)
    test_alpha_max_frac_collapsed = 0.0
    t_alpha_day_count = 0
    with torch.no_grad():
        for d in test_days:
            d_int = int(d)
            ei_list = per_day_edges.get(d_int)
            if ei_list is None:
                continue
            x = features_t[d_int].to(device)
            preds[d_int] = model(x, ei_list).cpu().numpy()

            if model.last_alpha is not None:
                a = model.last_alpha.cpu().numpy()
                test_alpha_means += a.mean(axis=0)
                test_alpha_max_frac_collapsed += float((a.max(axis=1) > 0.9).mean())
                t_alpha_day_count += 1

    if t_alpha_day_count > 0:
        test_alpha_means /= t_alpha_day_count
        test_alpha_max_frac_collapsed /= t_alpha_day_count

    train_info = {
        'best_val_loss': best_val,
        'epochs_run': epochs_run,
        'alpha_mean_corr_test': float(test_alpha_means[0]),
        'alpha_mean_sector_test': float(test_alpha_means[1]),
        'alpha_mean_news_test': float(test_alpha_means[2]),
        'alpha_max_fraction_collapsed_test': float(test_alpha_max_frac_collapsed),
    }

    del model, per_day_edges
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return preds, train_info


# ══════════════════════════════════════════════════════════════
# I/O HELPERS (CSV + JSON)
# ══════════════════════════════════════════════════════════════

def init_csv_files() -> None:
    if not os.path.exists(RESULTS_CSV):
        pd.DataFrame(columns=RESULTS_COLUMNS).to_csv(RESULTS_CSV, index=False)
    if not os.path.exists(MANIFEST_CSV):
        pd.DataFrame(columns=MANIFEST_COLUMNS).to_csv(MANIFEST_CSV, index=False)


def append_results(row: dict) -> None:
    df = pd.DataFrame([row], columns=RESULTS_COLUMNS)
    df.to_csv(RESULTS_CSV, mode='a', header=False, index=False)


def append_manifest(row: dict) -> None:
    df = pd.DataFrame([row], columns=MANIFEST_COLUMNS)
    df.to_csv(MANIFEST_CSV, mode='a', header=False, index=False)


def load_manifest_done() -> set[tuple[int, int]]:
    """Mirrors E3 CR-E3RUN-A-02 pattern: identity by (fold, seed), not cell_id."""
    if not os.path.exists(MANIFEST_CSV):
        return set()
    df = pd.read_csv(MANIFEST_CSV)
    if 'status' not in df.columns:
        return set()
    done_rows = df[df['status'] == 'done']
    return set(zip(done_rows['fold'].astype(int).tolist(),
                   done_rows['seed'].astype(int).tolist()))


def write_hp_grid_json() -> None:
    hp = {
        'model': MODEL,
        'paper_inspired_by': 'Kim et al. 2019 arXiv:1908.07999 (HATS)',
        'is_literal_reproduction': False,
        'hats_hparams': HATS_HPARAMS,
        'relations_in_order': RELATIONS_IN_ORDER,
        'graph_hparams': GRAPH_HPARAMS,
        'news_lookback_trading_days': NEWS_LOOKBACK_TRADING_DAYS,
        'news_lookback_calendar_days': NEWS_LOOKBACK_CALENDAR_DAYS,
        'universe': UNIVERSE,
        'horizon_days': HORIZON,
        'seeds': CANONICAL_SEEDS,
    }
    with open(HP_GRID_JSON, 'w') as f:
        json.dump(hp, f, indent=2)


def write_meta_json() -> None:
    meta = {
        'experiment_id': 'storya_e1_6_hats_3r_adapt',
        'plan_ref': '/Users/heruixi/.claude/plans/hats-baseline-reproduction-delightful-lighthouse.md',
        'codex_plan_review_ref': 'artifacts/reviews/2026-05-27_codex_plan_A.md',
        'paper_inspired_by': 'Kim et al. 2019 arXiv:1908.07999',
        'is_literal_reproduction': False,
        'adaptations_vs_kim_2019': [
            'No GRU/LSTM encoder (uses Universe B 10-dim aggregated features directly)',
            '3 substitute relations (corr/sector/news) instead of Wikidata 75 types',
            'Regression on 21d CS-z-scored returns instead of up/neutral/down classification',
            'Linear(64, 1) shared relation-attention scorer instead of Kim §3.2 relation-embedding concat',
        ],
        'horizon_days': HORIZON,
        'canonical_seeds': CANONICAL_SEEDS,
        'universe': UNIVERSE,
        'relations_in_order': RELATIONS_IN_ORDER,
        'cell_id_range': [400, 449],
        'cell_id_formula': 'cell_id = 400 + fold_idx*10 + seed_idx (Codex A-04 disposition)',
        'cost_ladder': {
            'levels_bps': list(COST_LEVELS_BPS),
            'convention': COST_CONVENTION,
            'turnover_definition': 'L1-norm; at full L-S rotation = 4',
            'annualization': 'sqrt(252/horizon)',
        },
        'sector_pit_limitation': {
            'sector_file': PATHS['sectors'],
            'fetch_date_estimate': '2026-02-09 (file mtime)',
            'codex_finding': 'CODEX-A-01 (CRITICAL) ACCEPTED-AS-CONCERN',
            'treatment': 'Project-level §Limitations; same as Plan AAA Alpha158 same-day OHLC leak',
        },
    }
    with open(META_JSON, 'w') as f:
        json.dump(meta, f, indent=2)


def write_prereg_json() -> None:
    """LOCKED decision rules + claim scope. Written BEFORE first non-smoke cell.

    Per Codex Plan Round A finding CODEX-A-08 disposition + H博士 2026-05-27 decision.
    """
    import subprocess
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = 'unknown'

    prereg = {
        'experiment_id': 'storya_e1_6_hats',
        'model': MODEL,
        'paper_inspired_by': 'Kim et al. 2019 arXiv:1908.07999 (HATS)',
        'claim_scope': (
            'This is an ADAPTATION of Kim 2019 HATS, NOT a literal reproduction. '
            'Differences: no GRU encoder, 3 substitute relations (corr/sector/news) '
            'instead of Wikidata 75-type, regression on 21d CS-z-scored returns '
            'instead of up/neutral/down classification, simplified Linear(hidden, 1) '
            'relation-attention scoring instead of Kim §3.2 relation-embedding '
            'concatenation. Any Template-1 conclusion speaks ONLY to the adapted '
            'module on this project\'s strict ranking eval; does NOT speak to Kim\'s '
            'published HATS performance.'
        ),
        'scope': 'Universe B only, 50 cells (10 seeds x 5 folds), cell_id range [400, 449]',
        'spa_family_expansion': {
            'M_universe_B_old': 3, 'M_universe_B_new': 4,
            'M_universe_C_old': 3, 'M_universe_C_new': 3,
            'M_joint_old': 6,      'M_joint_new': 6,
            'rationale_codex_A_02_disposition': (
                'HATS-3R-adapt is EXCLUDED from joint SPA (H博士 2026-05-27). Adds '
                'to per-universe B SPA only (M=3->4). Joint SPA stays E1-only M=6. '
                'compute_e6_dm_spa.py:582-584 spa_application_joint_M=6 unchanged.'
            ),
        },
        'decision_rules_locked_2026_05_27': {
            'primary_comparator': 'GAT (Universe B, E1 anchor)',
            'primary_metric': 'IC_mean (full 5-fold + LOFO-4)',
            'secondary_metric_narrative_only': 'Sharpe_net_10bps',
            'POSITIVE_validation': (
                'delta_IC(HATS-3R-adapt - GAT) > +0.005 in full condition AND '
                'BH-adjusted HLN p < 0.05 in per-universe-B 4-pair family AND '
                'LOFO-4 delta_IC sign preserved (>0)'
            ),
            'NEGATIVE_template1': (
                'delta_IC < -0.005 OR (BH-adjusted HLN p > 0.20 in full AND '
                'LOFO-4 delta_IC <= 0)'
            ),
            'TIE': (
                '|delta_IC| <= 0.005 AND BH-adjusted HLN p > 0.05 '
                '(neither rejected nor strongly negative)'
            ),
            'attention_diagnostic_NOT_a_gate': (
                'alpha statistics (mean per relation per epoch, max-collapsed fraction) '
                'logged for transparency; attention-specific claims are DEFERRED unless '
                'a uniform-alpha control run is added per Codex A-11 extension rule below.'
            ),
        },
        'uniform_alpha_extension_rule_codex_A_11': {
            'trigger': (
                'If primary 50-cell HATS-3R-adapt result is POSITIVE per decision '
                'rule AND H博士 wants to claim attention-specific benefit'
            ),
            'action': (
                'Run +50 cells (same seeds x folds) with frozen alpha = 1/3 uniform '
                'across relations; compare paired delta_IC to learned-alpha '
                'HATS-3R-adapt'
            ),
            'no_attention_claims_without_this': True,
        },
        'sector_pit_limitation_codex_A_01_disposition': {
            'sector_file': PATHS['sectors'],
            'sector_file_fetch_date_estimate': '2026-02-09 (file mtime)',
            'as_of_basis': (
                'Snapshot of S&P GICS sectors at 2026-02-09; sector reclassifications '
                'between 2024-04 and 2026-02 (~1-3% of universe per year) are NOT '
                'corrected; constitutes a small look-ahead leak in sector edges '
                '(same magnitude/handling as Plan AAA Alpha158 same-day OHLC leak).'
            ),
            'treatment': (
                'Documented in Story A §Limitations alongside Alpha158 leak; '
                'HATS-3R-adapt and existing E4-alpha both inherit this limitation; '
                'not re-running E4-alpha to fix.'
            ),
        },
        'no_post_hoc_tuning': True,
        'no_attention_specific_claims': True,
        'wall_time_estimate': 'PROVISIONAL ~13 min/cell A100; must be re-locked from 1-cell smoke benchmark (Codex A-10 disposition)',
        'git_hash': git_hash,
        'written_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(PREREG_JSON, 'w') as f:
        json.dump(prereg, f, indent=2)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def setup_workdir() -> None:
    try:
        import google.colab  # noqa: F401
        os.chdir('/content/drive/MyDrive/GNN测试')
    except ImportError:
        os.chdir('/Users/heruixi/Desktop/GNN-Testing')
    print(f"Working dir: {os.getcwd()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--seeds', type=str, default=','.join(str(s) for s in CANONICAL_SEEDS),
                        help='Comma-separated seed list (subset of canonical 10)')
    parser.add_argument('--folds', type=str, default='0,1,2,3,4',
                        help='Comma-separated fold indices (subset of 0..4)')
    parser.add_argument('--smoke', action='store_true',
                        help='1 cell on fold 0 seed 86 (shape+train+eval smoke)')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Skip cells with status=done in manifest (default ON)')
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    parser.add_argument('--news-cache', default=NEWS_SNAPSHOT_CACHE,
                        help='Path to E3 news snapshot cache')
    args = parser.parse_args()

    # ── Setup ──
    setup_workdir()
    device = get_device()
    print(f"Device: {device.type}")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PER_DAY_IC_DIR, exist_ok=True)
    os.makedirs(ALPHA_DIAG_DIR, exist_ok=True)
    init_csv_files()
    write_hp_grid_json()
    write_meta_json()
    write_prereg_json()  # MUST be before first cell per Codex A-08

    # Codex A-04 disposition: assert HATS ids are disjoint from E1
    assert_cell_id_hats_injective_and_disjoint_from_e1()

    # ── Load core data via E1 anchor's loader ──
    print("\nLoading core data ...")
    core = load_core_data()
    prices = core['prices']
    returns = core['returns']
    trading_dates = core['all_dates']
    num_days = core['num_days']
    num_stocks = core['num_stocks']
    ticker_to_idx = core['ticker_to_id']
    print(f"  Loaded: {num_stocks} stocks x {num_days} days, "
          f"date range {trading_dates.min().date()} -> {trading_dates.max().date()}")

    assert_purge_no_leak(trading_dates, horizon=HORIZON)

    labels_np, label_valid_np = build_labels(prices, horizon=HORIZON)
    print(f"Labels: h={HORIZON}d, {label_valid_np.sum():,} valid (day, stock) entries")

    # Universe B features (10-dim hc)
    features_np, feature_names = build_universe_B(prices, returns)
    print(f"Universe B features built: {features_np.shape}")

    # Correlation graph snapshots
    corr_snapshots, _day_to_si, snapshot_points = build_correlation_snapshots(returns, num_days)
    print(f"Correlation graph: {len(corr_snapshots)} snapshots ready")

    # Sector edges (static GICS-11, fully-connected within sector)
    sector_static_np = build_sector_edges(PATHS['sectors'], ticker_to_idx)
    print(f"Sector edges: {sector_static_np.shape[1] // 2} undirected (={sector_static_np.shape[1]} directed)")

    # News edge source + per-day PIT-safe snapshots (cached from E3 if available)
    print("\nLoading news edge source ...")
    news_df = load_news_edge_source()
    print(f"  News edge source: {len(news_df):,} articles, "
          f"{(news_df['n_tickers'] >= 2).sum():,} with >=2 tickers")

    if os.path.exists(args.news_cache):
        print(f"Loading cached news snapshots: {args.news_cache} ...")
        cache = np.load(args.news_cache, allow_pickle=True)
        snapshot_keys = [k for k in cache.files if k != '__article_counts__']
        news_snapshots = {int(k): cache[k] for k in snapshot_keys}
        if '__article_counts__' in cache.files:
            ac_arr = cache['__article_counts__']
            article_counts = {i: int(c) for i, c in enumerate(ac_arr) if c >= 0}
        else:
            article_counts = {}
        print(f"  loaded {len(news_snapshots)} snapshots, {len(article_counts)} article-count entries")
    else:
        news_snapshots, article_counts = build_per_day_news_edges(
            news_df, trading_dates, ticker_to_idx
        )
        # Persist cache for future re-use
        ac_arr = np.full(len(trading_dates), -1, dtype=np.int32)
        for d, c in article_counts.items():
            ac_arr[d] = c
        os.makedirs(os.path.dirname(args.news_cache), exist_ok=True)
        np.savez_compressed(args.news_cache,
                            __article_counts__=ac_arr,
                            **{str(k): v for k, v in news_snapshots.items()})
        print(f"Cached news snapshots to {args.news_cache}")

    # ── Resumable iteration ──
    done = load_manifest_done() if args.resume else set()
    print(f"\nResume mode {'ON' if args.resume else 'OFF'}: "
          f"{len(done)} cells already completed; will skip them")

    requested_seeds = [int(s) for s in args.seeds.split(',') if s]
    requested_folds = [int(f) for f in args.folds.split(',') if f]
    for s in requested_seeds:
        assert s in CANONICAL_SEEDS, f"Seed {s} not in canonical list {CANONICAL_SEEDS}"
    for f in requested_folds:
        assert 0 <= f <= 4, f"Fold {f} outside [0, 4]"

    if args.smoke:
        seeds_run = [CANONICAL_SEEDS[0]]  # seed=86
        folds_run = [WALK_FORWARD_FOLDS[0]]
        print("\n=== SMOKE MODE: 1 cell (fold 0, seed 86, Universe B) ===")
    else:
        seeds_run = [s for s in CANONICAL_SEEDS if s in requested_seeds]
        folds_run = [f for f in WALK_FORWARD_FOLDS if f['id'] in requested_folds]

    seed_idx_map = {s: i for i, s in enumerate(CANONICAL_SEEDS)}

    planned = []
    for fold_cfg in folds_run:
        f_idx = fold_cfg['id']
        for seed in seeds_run:
            s_idx = seed_idx_map[seed]
            cid = cell_id_hats(f_idx, s_idx)
            if (f_idx, seed) in done:
                continue
            planned.append((cid, fold_cfg, s_idx, seed))
    print(f"Planned cells (after resume filter): {len(planned)}")

    # ── Per-fold preprocessing (winsor + standardize on TRAIN only; reuse across seeds) ──
    fold_cache = {}
    for fold_cfg in folds_run:
        f_idx = fold_cfg['id']
        train_days, val_days, test_days = create_fold_masks(fold_cfg, trading_dates, horizon=HORIZON)
        print(f"\n[fold {f_idx} ({fold_cfg['desc']})] "
              f"train={len(train_days)}d (purged {HORIZON}d), "
              f"val={len(val_days)}d, test={len(test_days)}d")

        feats_winsor = winsorize_train_only(features_np, train_days)
        feats_std = standardize_train_only(feats_winsor, train_days)

        frozen_si = get_frozen_snapshot_idx(train_days[-1], snapshot_points)
        corr_frozen_tensor = corr_snapshots[frozen_si]
        corr_frozen_np = (corr_frozen_tensor.numpy()
                          if hasattr(corr_frozen_tensor, 'numpy')
                          else np.asarray(corr_frozen_tensor))

        # Codex A-01 disposition (project-level §Limitations): sector edges accepted with
        # documented as-of date 2026-02-09. Asserted in prereg.json, not runtime-aborted.
        per_day_edges_cpu = build_three_relation_edges_per_fold(
            corr_frozen_np, sector_static_np, news_snapshots,
            train_days, val_days, test_days,
        )

        all_days = np.concatenate([train_days, val_days, test_days])
        missing_days = [int(d) for d in all_days if int(d) not in per_day_edges_cpu]
        if missing_days:
            raise RuntimeError(
                f"Fold {f_idx}: {len(missing_days)} days have no edge tensors. "
                f"First 5: {missing_days[:5]}"
            )

        # Stats per fold
        n_corr_edges = corr_frozen_np.shape[1] // 2  # symmetric -> halve for "undirected count"
        n_sector_edges = sector_static_np.shape[1] // 2
        test_news_counts = [
            news_snapshots.get(int(d), np.zeros((2, 0), dtype=np.int64)).shape[1] // 2
            for d in test_days
        ]
        avg_news_edges = float(np.mean(test_news_counts)) if test_news_counts else 0.0

        print(f"  frozen_si={frozen_si}, n_corr={n_corr_edges}, n_sector={n_sector_edges}, "
              f"avg news/test_day={avg_news_edges:.0f}")

        fold_cache[f_idx] = {
            'features_t': torch.from_numpy(feats_std).float(),
            'labels_t': torch.from_numpy(labels_np).float(),
            'label_valid_t': torch.from_numpy(label_valid_np),
            'train_days': train_days,
            'val_days': val_days,
            'test_days': test_days,
            'per_day_edges_cpu': per_day_edges_cpu,
            'n_corr_edges': n_corr_edges,
            'n_sector_edges': n_sector_edges,
            'avg_news_edges': avg_news_edges,
            'test_period': fold_cfg['desc'],
        }

    # ── Run cells ──
    smoke_rows = []
    for cid, fold_cfg, s_idx, seed in planned:
        f_idx = fold_cfg['id']
        cache = fold_cache[f_idx]
        start_ts = time.strftime('%Y-%m-%dT%H:%M:%S')
        t_cell = time.time()
        print(f"\n  [cid={cid:03d}] U{UNIVERSE}  {MODEL}  seed={seed}  fold={f_idx} ...")

        append_manifest({
            'cell_id': cid, 'fold': f_idx, 'seed': seed,
            'status': 'running', 'start_ts': start_ts,
            'end_ts': '', 'wall_time_sec': 0, 'err': '',
        })

        alpha_log_path = f'{ALPHA_DIAG_DIR}/alpha_s{seed}_f{f_idx}.csv'

        try:
            preds, train_info = train_hats(
                cache['features_t'], cache['labels_t'], cache['label_valid_t'],
                cache['train_days'], cache['val_days'], cache['test_days'],
                cache['per_day_edges_cpu'],
                num_days, num_stocks, seed, device,
                alpha_log_path=alpha_log_path,
            )

            ic_arr = compute_daily_ic(preds, cache['test_days'], labels_np, label_valid_np)
            sh = compute_cost_ladder_sharpe(
                preds, cache['test_days'], prices, label_valid_np,
                num_stocks, num_days, horizon=HORIZON,
                cost_levels_bps=COST_LEVELS_BPS,
            )

            wall = round(time.time() - t_cell, 1)
            ic_path = f'{PER_DAY_IC_DIR}/{UNIVERSE}_{MODEL}_s{seed}_f{f_idx}.npy'
            np.save(ic_path, ic_arr)

            row = {
                'cell_id': cid, 'universe': UNIVERSE, 'model': MODEL,
                'seed': seed, 'fold': f_idx, 'test_period': cache['test_period'],
                'IC_mean': round(float(ic_arr.mean()), 6) if len(ic_arr) else 0.0,
                'IC_std': round(float(ic_arr.std()), 6) if len(ic_arr) else 0.0,
                'n_test_days': int(len(ic_arr)),
                'Sharpe_gross': round(sh['Sharpe_gross'], 4),
                **{k: round(v, 4) for k, v in sh.items() if k.startswith('Sharpe_net_')},
                'mean_turnover_L1': round(sh['mean_turnover_L1'], 4),
                'n_periods': int(sh['n_periods']),
                'best_val_loss': round(float(train_info['best_val_loss']), 6),
                'epochs_run': int(train_info['epochs_run']),
                'wall_time_sec': wall,
                'converged_flag': 1,
                'cost_convention': COST_CONVENTION,
                'n_corr_edges': cache['n_corr_edges'],
                'n_sector_edges': cache['n_sector_edges'],
                'n_news_edges_avg': round(cache['avg_news_edges'], 1),
                'alpha_mean_corr_test': round(train_info['alpha_mean_corr_test'], 4),
                'alpha_mean_sector_test': round(train_info['alpha_mean_sector_test'], 4),
                'alpha_mean_news_test': round(train_info['alpha_mean_news_test'], 4),
                'alpha_max_fraction_collapsed_test': round(
                    train_info['alpha_max_fraction_collapsed_test'], 4
                ),
            }
            append_results(row)

            end_ts = time.strftime('%Y-%m-%dT%H:%M:%S')
            append_manifest({
                'cell_id': cid, 'fold': f_idx, 'seed': seed,
                'status': 'done', 'start_ts': start_ts, 'end_ts': end_ts,
                'wall_time_sec': wall, 'err': '',
            })
            print(f"  done: IC={row['IC_mean']:+.4f}, Sh_g={row['Sharpe_gross']:.3f}, "
                  f"Sh_n@10={row['Sharpe_net_10bps']:.3f}, wall={wall}s, "
                  f"alpha=[corr={row['alpha_mean_corr_test']:.2f} "
                  f"sec={row['alpha_mean_sector_test']:.2f} "
                  f"news={row['alpha_mean_news_test']:.2f}]")

            if args.smoke:
                smoke_rows.append({
                    'cell_id': cid,
                    'wall_time_sec': wall,
                    'IC_mean': row['IC_mean'],
                    'Sharpe_gross': row['Sharpe_gross'],
                    'epochs_run': row['epochs_run'],
                    'alpha_max_frac_collapsed': row['alpha_max_fraction_collapsed_test'],
                })

        except Exception as e:
            wall = round(time.time() - t_cell, 1)
            end_ts = time.strftime('%Y-%m-%dT%H:%M:%S')
            err_msg = f"{type(e).__name__}: {e}"
            append_manifest({
                'cell_id': cid, 'fold': f_idx, 'seed': seed,
                'status': 'failed', 'start_ts': start_ts, 'end_ts': end_ts,
                'wall_time_sec': wall, 'err': err_msg[:500],
            })
            print(f"  FAILED ({wall}s): {err_msg}", file=sys.stderr)
            import traceback; traceback.print_exc()

        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ── Smoke summary ──
    if args.smoke and smoke_rows:
        smoke_csv = f'{OUT_DIR}/smoke_benchmark.csv'
        pd.DataFrame(smoke_rows).to_csv(smoke_csv, index=False)
        total = sum(r['wall_time_sec'] for r in smoke_rows)
        print(f"\n=== SMOKE SUMMARY ===")
        print(f"1 cell wall: {total:.0f}s ({total / 60:.1f} min)")
        print(f"Decision gate (plan §Verification step 1): wall < 45 min M4? "
              f"{'PASS' if total < 45 * 60 else 'FAIL'}")
        ic = smoke_rows[0]['IC_mean']
        print(f"IC sanity check: {ic:+.4f} in (-0.05, 0.10)? "
              f"{'PASS' if -0.05 < ic < 0.10 else 'INVESTIGATE'}")
        full_estimate = 50 * (total / len(smoke_rows)) / 3600
        print(f"Full 50-cell estimate (M4 scaling): {full_estimate:.1f}h "
              f"(A100 likely 2-4x faster)")

    print(f"\nDONE. See {RESULTS_CSV} + {MANIFEST_CSV}")


if __name__ == '__main__':
    main()
