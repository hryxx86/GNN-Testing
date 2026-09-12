#!/usr/bin/env python
"""Phase 5 Step 3 Plan Z++ — Part A.

Train full 10-feature SAGE-Mean + MLP on 5-fold walk-forward with 3 seeds
(30 runs). Compute grouped permutation delta-IC at inference to rank the
7 preregistered feature groups, and freeze the ranking for Part B subset
construction.

See plan.md 2026-04-18-a.

Outputs:
  artifacts/step3_plan_z/fold_manifest.json
  experiments/step3_plan_z/part_a_daily_ic.csv      — (model, fold, seed, day, IC_baseline)
  experiments/step3_plan_z/part_a_permuted_ic.csv   — (model, fold, seed, group, day, IC_permuted)
  experiments/step3_plan_z/part_a_ranking.json      — aggregate group ranking
"""
import gc
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch_geometric.nn import SAGEConv

# Self-adaptive chdir — must probe for project-root marker (data/reference/ + sp500 csv)
# not just path existence, because Colab VMs can have a stale empty /Users/heruixi/... from
# previous mount attempts. Prefer Colab Drive path first on Colab, then local Mac path.
_ROOT_CANDIDATES = [
    '/content/drive/MyDrive/GNN测试',
    '/Users/heruixi/Desktop/GNN-Testing',
]
for _p in _ROOT_CANDIDATES:
    if os.path.exists(os.path.join(_p, 'data', 'reference', 'sp500_5y_prices.csv')):
        os.chdir(_p)
        break

# ─────────────────────────────── configuration ───────────────────────────────
HORIZON = 21
TRAIN_START = '2021-01-29'
SEEDS = [42, 123, 2024]
# CUDA > MPS > CPU fallback (2026-04-22 fix for Colab A100)
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

HPARAMS = dict(
    hidden=64, num_layers=2, dropout=0.3,
    lr=1e-3, weight_decay=1e-4,
    epochs=50, patience=10, grad_accum=4,
    corr_window=126, corr_step=21, corr_threshold=0.6,
)

FOLDS = [
    dict(id=0, train_end='2023-12-31', val_end='2024-03-31', test_end='2024-06-30'),
    dict(id=1, train_end='2024-03-31', val_end='2024-06-30', test_end='2024-09-30'),
    dict(id=2, train_end='2024-06-30', val_end='2024-09-30', test_end='2024-12-31'),
    dict(id=3, train_end='2024-09-30', val_end='2024-12-31', test_end='2025-03-31'),
    dict(id=4, train_end='2024-12-31', val_end='2025-03-31', test_end='2025-06-30'),
]

ARTIFACT_DIR = Path('/Users/heruixi/Desktop/GNN-Testing/artifacts/step3_plan_z')
EXPER_DIR = Path('/Users/heruixi/Desktop/GNN-Testing/experiments/step3_plan_z')
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
EXPER_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────── data loading ───────────────────────────────

def load_data_and_features():
    prices = pd.read_csv('data/reference/sp500_5y_prices.csv', index_col=0, parse_dates=True)
    sector_df = pd.read_csv('data/reference/sp500_sectors.csv')
    sec_col = [c for c in sector_df.columns if 'sector' in c.lower()][0]
    tic_col = [c for c in sector_df.columns if c != sec_col][0]
    events = pd.read_parquet('data/fullscale/sp500_news_events.parquet', columns=['ticker'])
    valid_tickers = sorted(set(prices.columns) & set(events['ticker'].unique()) &
                           set(sector_df[tic_col]))
    prices = prices[valid_tickers]
    returns = prices.pct_change()
    returns.iloc[0] = 0
    all_dates = prices.index
    num_days, num_stocks = prices.shape
    ticker_to_id = {t: i for i, t in enumerate(valid_tickers)}
    sector_map = dict(zip(sector_df[tic_col], sector_df[sec_col]))
    sector_groups = defaultdict(list)
    for t in valid_tickers:
        if t in sector_map:
            sector_groups[sector_map[t]].append(ticker_to_id[t])

    # 10 features: ret_mean_{5,10,21}d, ret_std_{5,10,21}d, mom12m, maxret, dolvol, CORR5
    frames = []
    for w in [5, 10, 21]:
        frames.append(returns.rolling(w).mean().shift(1))
    for w in [5, 10, 21]:
        frames.append(returns.rolling(w).std().shift(1))
    old_tensor = np.stack([f.values for f in frames], axis=-1)

    new_arr = np.load('data/reference/sp500_5y_phase5_features.npy')
    new_tensor = new_arr[:, :, :4]  # drop RSV5

    features_np = np.concatenate([old_tensor, new_tensor], axis=-1).astype(np.float32)
    features_np = np.nan_to_num(features_np, 0.0)

    feature_names = [f'ret_mean_{w}d' for w in [5, 10, 21]] + \
                    [f'ret_std_{w}d' for w in [5, 10, 21]] + \
                    ['mom12m', 'maxret', 'dolvol', 'CORR5']
    assert features_np.shape == (num_days, num_stocks, len(feature_names))

    # Labels: cross-sectionally standardized 21d forward returns
    # (daily CS demean + CS z-score; NOT risk-free-rate "excess" returns)
    fwd_ret = prices.shift(-HORIZON) / prices - 1
    day_mean = fwd_ret.mean(axis=1)
    day_std = fwd_ret.std(axis=1)
    day_std[day_std < 1e-8] = 1.0
    z = fwd_ret.sub(day_mean, axis=0).div(day_std, axis=0)
    valid = ~z.isna()
    labels_np = np.nan_to_num(z.values.astype(np.float32), 0.0)
    label_valid_np = valid.values

    return {
        'prices': prices, 'returns': returns, 'all_dates': all_dates,
        'valid_tickers': valid_tickers, 'sector_groups': sector_groups,
        'num_days': num_days, 'num_stocks': num_stocks,
        'features_np': features_np, 'feature_names': feature_names,
        'labels_np': labels_np, 'label_valid_np': label_valid_np,
    }


def build_correlation_snapshots(returns: pd.DataFrame, num_days: int):
    """Per-day correlation edge snapshot. Returns {day_idx: edge_index_tensor}."""
    snaps = {}
    snap_points = list(range(HPARAMS['corr_window'], num_days, HPARAMS['corr_step']))
    snap_tensors = {}
    for si, t_end in enumerate(snap_points):
        w = returns.iloc[t_end - HPARAMS['corr_window']:t_end].values
        cm = np.corrcoef(w.T)
        np.fill_diagonal(cm, 0.0)
        src, dst = np.where(np.abs(cm) > HPARAMS['corr_threshold'])
        snap_tensors[si] = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    si = 0
    for di in range(num_days):
        while si + 1 < len(snap_points) and snap_points[si + 1] <= di:
            si += 1
        snaps[di] = si if snap_points[si] <= di else 0
    return snap_points, snap_tensors, snaps


def build_sector_edges(sector_groups):
    src, dst = [], []
    for members in sector_groups.values():
        for i in members:
            for j in members:
                if i != j:
                    src.append(i); dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)


# ─────────────────────────────── fold manifest ───────────────────────────────

def assert_graph_train_only(snap_points: list, snaps: dict, train_days: np.ndarray,
                             corr_window: int = None, fold_id: int = None) -> tuple[int, tuple[int, int]]:
    """Assert frozen graph snapshot ends at or before max(train_days). Plan Z++ §0.3.

    Returns (snap_end, snap_window) so callers can record provenance.
    """
    if corr_window is None:
        corr_window = HPARAMS['corr_window']
    max_train = int(train_days.max())
    frozen_si = snaps[max_train]
    snap_end = int(snap_points[frozen_si])
    snap_window = (snap_end - corr_window, snap_end)
    fold_label = f'fold {fold_id}' if fold_id is not None else 'fold'
    assert snap_end - 1 <= max_train, (
        f"{fold_label}: graph leakage — snap_end={snap_end} > max(train_days)={max_train}; "
        f"frozen_si={frozen_si}, snap_window={snap_window}"
    )
    return snap_end, snap_window


def build_fold_manifest(folds, feature_names, all_dates,
                         snap_points: list = None, snaps: dict = None):
    """Walk-forward fold manifest with leakage assertions (Codex Round 3).

    Plan Z++ §0.3: when snap_points + snaps are provided, also record
    graph_snap_end and graph_snap_window per fold (graph provenance).
    """
    manifest = []
    for cfg in folds:
        ts = pd.Timestamp(TRAIN_START)
        te = pd.Timestamp(cfg['train_end'])
        ve = pd.Timestamp(cfg['val_end'])
        test_e = pd.Timestamp(cfg['test_end'])
        tr_days = np.where((all_dates >= ts) & (all_dates <= te))[0]
        val_days = np.where((all_dates > te) & (all_dates <= ve))[0]
        test_days = np.where((all_dates > ve) & (all_dates <= test_e))[0]

        # 21-day embargo at tail of BOTH train and val (Codex Round 5 CRITICAL #1):
        # train labels use prices T+HORIZON, so the last HORIZON days of train
        # would leak val-period price info; same for val->test.
        if len(tr_days) > HORIZON:
            tr_days = tr_days[:-HORIZON]
        else:
            raise AssertionError(f'fold {cfg["id"]}: train window < HORIZON, cannot embargo')
        if len(val_days) > HORIZON:
            val_days = val_days[:-HORIZON]
        else:
            raise AssertionError(f'fold {cfg["id"]}: val window < HORIZON, cannot embargo')

        assert len(tr_days) > 0, f'fold {cfg["id"]}: empty train days'
        assert len(val_days) > 0, f'fold {cfg["id"]}: empty val days after embargo'
        assert len(test_days) > 0, f'fold {cfg["id"]}: empty test days'
        assert tr_days.max() < val_days.min(), f'fold {cfg["id"]}: train/val overlap'
        assert val_days.max() < test_days.min(), f'fold {cfg["id"]}: val/test overlap'
        # HORIZON-day label overlap checks after embargo
        assert tr_days.max() + HORIZON < val_days.min(), \
            f'fold {cfg["id"]}: train label window overlaps val'
        assert val_days.max() + HORIZON < test_days.min(), \
            f'fold {cfg["id"]}: val label window overlaps test'

        entry = {
            'fold_id': cfg['id'],
            'train_start': str(all_dates[tr_days.min()].date()),
            'train_end': str(all_dates[tr_days.max()].date()),
            'val_start': str(all_dates[val_days.min()].date()),
            'val_end': str(all_dates[val_days.max()].date()),
            'test_start': str(all_dates[test_days.min()].date()),
            'test_end': str(all_dates[test_days.max()].date()),
            'n_train': int(len(tr_days)),
            'n_val': int(len(val_days)),
            'n_test': int(len(test_days)),
            'train_days': tr_days.tolist(),
            'val_days': val_days.tolist(),
            'test_days': test_days.tolist(),
        }

        if snap_points is not None and snaps is not None:
            snap_end, snap_window = assert_graph_train_only(
                snap_points, snaps, tr_days, corr_window=HPARAMS['corr_window'],
                fold_id=cfg['id'])
            entry['graph_snap_end'] = snap_end
            entry['graph_snap_window'] = list(snap_window)

        manifest.append(entry)
    out = {
        'horizon': HORIZON,
        'feature_names': feature_names,
        'num_features': len(feature_names),
        'n_folds': len(manifest),
        'folds': manifest,
    }
    with open(ARTIFACT_DIR / 'fold_manifest.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'[manifest] wrote {ARTIFACT_DIR / "fold_manifest.json"} — {len(manifest)} folds')
    return out


# ─────────────────────────────── models ───────────────────────────────

class RankingGNN(nn.Module):
    def __init__(self, in_ch, hidden=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lin = nn.Linear(in_ch, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden, hidden, aggr='mean'))
            self.norms.append(nn.LayerNorm(hidden))
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden // 2, 1),
        )

    def forward(self, x, edge_index):
        h = F.relu(self.lin(x))
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index)
            h = norm(F.dropout(h_new, p=self.dropout, training=self.training) + h)
        return self.head(h).squeeze(-1)


class RankingMLP(nn.Module):
    def __init__(self, in_ch, hidden=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lin = nn.Linear(in_ch, hidden)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Linear(hidden, hidden))
            self.norms.append(nn.LayerNorm(hidden))
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden // 2, 1),
        )

    def forward(self, x, edge_index=None):
        h = F.relu(self.lin(x))
        for layer, norm in zip(self.layers, self.norms):
            h_new = layer(h)
            h = norm(F.dropout(h_new, p=self.dropout, training=self.training) + h)
        return self.head(h).squeeze(-1)


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(s)


# ─────────────────────────────── training ───────────────────────────────

def fit_feature_scaler(features_np: np.ndarray, label_valid_np: np.ndarray,
                       train_days: np.ndarray):
    """Fit per-feature mean/std using only valid-stock training-day values.

    Returns (mean, std) 1-D arrays of length num_features. NaN-robust; std floored
    at 1e-8. Addresses Codex Round 5 Q5: 10,000x scale disparity between dolvol
    (~20) and ret_mean_Nd (~1e-3) would make first-layer weights dominated by
    scale rather than signal.
    """
    vals = []
    for d in train_days:
        m = label_valid_np[d]
        if m.sum() > 0:
            vals.append(features_np[d][m])
    X = np.vstack(vals)
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def apply_scaler(features_t: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    """Broadcast (F,) scaler across (D, N, F) tensor without copying labels."""
    mean_t = torch.as_tensor(mean, dtype=features_t.dtype).view(1, 1, -1)
    std_t = torch.as_tensor(std, dtype=features_t.dtype).view(1, 1, -1)
    return (features_t - mean_t) / std_t


def train_one(model_type, features_t, labels_t, label_valid_t,
              train_days, val_days, test_days,
              snap_tensors, snaps, sector_edge_index, fold_id, seed,
              snap_points: list = None):
    """Train one (model, fold, seed) combination. Returns (model, test_preds ndarray).

    Plan Z++ §0.3: when snap_points is provided, asserts frozen graph snapshot
    end <= max(train_days) (graph train-only invariant). Optional for back-compat.
    """
    set_seed(seed)
    in_ch = features_t.shape[-1]
    if model_type == 'MLP':
        model = RankingMLP(in_ch, HPARAMS['hidden'], HPARAMS['num_layers'], HPARAMS['dropout'])
        use_graph = False
    else:
        model = RankingGNN(in_ch, HPARAMS['hidden'], HPARAMS['num_layers'], HPARAMS['dropout'])
        use_graph = True
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=HPARAMS['lr'], weight_decay=HPARAMS['weight_decay'])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=1e-5)

    # Freeze correlation snapshot at fold's train_end (Codex trap: no dynamic lookahead)
    # Plan Z++ §0.3: assert graph_snap_end <= max(train_days) — runtime guard
    if snap_points is not None:
        _snap_end, _snap_window = assert_graph_train_only(
            snap_points, snaps, train_days,
            corr_window=HPARAMS['corr_window'], fold_id=fold_id)
    frozen_si = snaps[int(train_days.max())]
    frozen_corr_ei = snap_tensors[frozen_si].to(DEVICE) if use_graph else None
    sector_ei = sector_edge_index.to(DEVICE) if use_graph else None
    if use_graph:
        full_ei = torch.cat([frozen_corr_ei, sector_ei], dim=1)
    else:
        full_ei = None

    best_val, best_state, bad = float('inf'), None, 0
    for ep in range(HPARAMS['epochs']):
        model.train()
        opt.zero_grad()
        accum = 0
        day_order = train_days[np.random.permutation(len(train_days))]
        for step, d in enumerate(day_order):
            x = features_t[d].to(DEVICE)
            pred = model(x, full_ei)
            mask = label_valid_t[d].to(DEVICE)
            target = labels_t[d].to(DEVICE)
            if mask.sum() < 10:
                continue
            loss = F.mse_loss(pred[mask], target[mask])
            (loss / HPARAMS['grad_accum']).backward()
            accum += 1
            if accum >= HPARAMS['grad_accum'] or step == len(day_order) - 1:
                if 0 < accum < HPARAMS['grad_accum']:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(HPARAMS['grad_accum'] / accum)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
                accum = 0

        model.eval()
        v_loss, v_cnt = 0.0, 0
        with torch.no_grad():
            for d in val_days:
                x = features_t[d].to(DEVICE)
                pred = model(x, full_ei)
                mask = label_valid_t[d].to(DEVICE)
                if mask.sum() < 10:
                    continue
                v_loss += F.mse_loss(pred[mask], labels_t[d].to(DEVICE)[mask]).item()
                v_cnt += 1
        avg_val = v_loss / max(v_cnt, 1)
        sched.step(avg_val)
        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= HPARAMS['patience']:
            break

    model.load_state_dict(best_state)
    model.eval()
    test_preds = np.zeros((len(test_days), features_t.shape[1]), dtype=np.float32)
    with torch.no_grad():
        for i, d in enumerate(test_days):
            x = features_t[d].to(DEVICE)
            test_preds[i] = model(x, full_ei).cpu().numpy()

    return model, test_preds, full_ei


def daily_ic(preds: np.ndarray, days: np.ndarray, labels_np: np.ndarray,
             label_valid_np: np.ndarray) -> np.ndarray:
    """Spearman IC per test day. Returns array aligned to `days`."""
    ic_arr = np.full(len(days), np.nan, dtype=np.float64)
    for i, d in enumerate(days):
        mask = label_valid_np[d]
        if mask.sum() < 30:
            continue
        p = preds[i][mask]
        a = labels_np[d][mask]
        ic, _ = spearmanr(p, a)
        if not np.isnan(ic):
            ic_arr[i] = ic
    return ic_arr


# ─────────────────────────────── grouped permutation ───────────────────────────────

def _group_rng(seed: int, group_label: str) -> np.random.Generator:
    """Deterministic RNG per (training_seed, group_label). Removes Codex Q3 order
    dependence — each group gets its own independent permutation sequence.

    Uses SHA-256 over the stable string representation rather than Python's
    `hash()`, which is process-randomized by default (PYTHONHASHSEED) and would
    break cross-run reproducibility.
    """
    key = f'{seed}|{group_label}'.encode('utf-8')
    digest = hashlib.sha256(key).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], 'big'))


def compute_grouped_permutation_ic(
    model, features_t, label_valid_t, label_valid_np, labels_np,
    test_days: np.ndarray, full_ei, groups: list[dict], feature_names: list[str],
    seed: int,
) -> dict[str, np.ndarray]:
    """For each group, joint per-day cross-sectional shuffle of group features on the
    test set. Returns {group_label: permuted_daily_IC_array}."""
    model.eval()
    feat_idx = {n: i for i, n in enumerate(feature_names)}
    out = {}
    for grp in groups:
        rng = _group_rng(seed, grp['label'])
        grp_indices = [feat_idx[m] for m in grp['members']]
        grp_indices_t = torch.as_tensor(grp_indices, dtype=torch.long)
        permuted_preds = np.zeros((len(test_days), features_t.shape[1]), dtype=np.float32)
        with torch.no_grad():
            for i, d in enumerate(test_days):
                x = features_t[d].clone()
                perm = torch.as_tensor(rng.permutation(x.shape[0]), dtype=torch.long)
                x[:, grp_indices_t] = x[perm][:, grp_indices_t]
                x = x.to(DEVICE)
                permuted_preds[i] = model(x, full_ei).cpu().numpy()
        out[grp['label']] = daily_ic(permuted_preds, test_days, labels_np, label_valid_np)
    return out


# ─────────────────────────────── main orchestrator ───────────────────────────────

def main():
    t0 = time.time()
    print(f'[device] {DEVICE}')

    data = load_data_and_features()
    features_np = data['features_np']
    feature_names = data['feature_names']
    labels_np = data['labels_np']
    label_valid_np = data['label_valid_np']
    print(f'[data] features {features_np.shape}, labels valid {label_valid_np.sum():,}')

    snap_points, snap_tensors, snaps = build_correlation_snapshots(
        data['returns'], data['num_days']
    )
    sector_edge_index = build_sector_edges(data['sector_groups'])
    print(f'[graph] {len(snap_points)} corr snapshots, {sector_edge_index.shape[1]} sector edges')

    # Plan Z++ §0.3: build manifest AFTER graph snapshots so graph_snap_end /
    # graph_snap_window are stamped per fold and the assertion fires before training.
    manifest = build_fold_manifest(FOLDS, feature_names, data['all_dates'],
                                    snap_points=snap_points, snaps=snaps)

    # NOTE: feature scaling is per-fold, applied in the fold loop (Codex Round 5 Q5).
    features_raw_t = torch.tensor(features_np, dtype=torch.float32)
    labels_t = torch.tensor(labels_np, dtype=torch.float32)
    label_valid_t = torch.tensor(label_valid_np, dtype=torch.bool)

    with open(ARTIFACT_DIR / 'groups.json') as f:
        groups_obj = json.load(f)
    groups = groups_obj['groups']
    print(f'[groups] {len(groups)} groups: {[g["label"] for g in groups]}')

    scaler_log = []

    # Collect results
    baseline_rows = []      # (model, fold, seed, test_day_idx, IC)
    permuted_rows = []      # (model, fold, seed, group_label, test_day_idx, IC_permuted)

    models = ['SAGE-Mean', 'MLP']
    total_runs = len(models) * len(manifest['folds']) * len(SEEDS)
    run_idx = 0
    for fold in manifest['folds']:
        fold_id = fold['fold_id']
        train_days = np.array(fold['train_days'])
        val_days = np.array(fold['val_days'])
        test_days = np.array(fold['test_days'])
        print(f'\n[fold {fold_id}] train {fold["train_start"]}..{fold["train_end"]}, '
              f'val ..{fold["val_end"]}, test ..{fold["test_end"]} '
              f'(n_train={fold["n_train"]}, n_val={fold["n_val"]}, n_test={fold["n_test"]})')
        # Fit feature scaler on train-only, apply to all splits within this fold
        mean, std = fit_feature_scaler(features_np, label_valid_np, train_days)
        features_t = apply_scaler(features_raw_t, mean, std)
        scaler_log.append({
            'fold': fold_id,
            'feature_mean': mean.tolist(),
            'feature_std': std.tolist(),
        })
        print(f'  [scaler] fit on {len(train_days)} train days. '
              f'feature std range: [{std.min():.4f}, {std.max():.4f}]')
        for model_type in models:
            for seed in SEEDS:
                run_idx += 1
                t_run = time.time()
                model, test_preds, full_ei = train_one(
                    model_type, features_t, labels_t, label_valid_t,
                    train_days, val_days, test_days,
                    snap_tensors, snaps, sector_edge_index, fold_id, seed,
                    snap_points=snap_points,
                )
                # Baseline IC
                ic = daily_ic(test_preds, test_days, labels_np, label_valid_np)
                for i, d in enumerate(test_days):
                    baseline_rows.append({
                        'model': model_type, 'fold': fold_id, 'seed': seed,
                        'day_idx': int(d), 'IC': float(ic[i]) if not np.isnan(ic[i]) else np.nan,
                    })

                # Grouped permutation
                perm_ic = compute_grouped_permutation_ic(
                    model, features_t, label_valid_t, label_valid_np, labels_np,
                    test_days, full_ei, groups, feature_names, seed=seed,
                )
                for label, ic_arr in perm_ic.items():
                    for i, d in enumerate(test_days):
                        permuted_rows.append({
                            'model': model_type, 'fold': fold_id, 'seed': seed,
                            'group': label,
                            'day_idx': int(d),
                            'IC_perm': float(ic_arr[i]) if not np.isnan(ic_arr[i]) else np.nan,
                        })

                elapsed = time.time() - t_run
                print(f'  [{run_idx}/{total_runs}] {model_type} fold={fold_id} seed={seed} '
                      f'baseline_mean_IC={np.nanmean(ic):+.4f} in {elapsed:.1f}s')
                del model; gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

    # Save
    base_df = pd.DataFrame(baseline_rows)
    perm_df = pd.DataFrame(permuted_rows)
    base_df.to_csv(EXPER_DIR / 'part_a_daily_ic.csv', index=False)
    perm_df.to_csv(EXPER_DIR / 'part_a_permuted_ic.csv', index=False)

    # Aggregate ranking — paired per-day delta_IC to avoid NaN-skew (Codex Round 5 CRITICAL #2)
    paired = perm_df.merge(
        base_df, on=['model', 'fold', 'seed', 'day_idx'],
        how='left', validate='many_to_one',
    )
    paired = paired[np.isfinite(paired['IC']) & np.isfinite(paired['IC_perm'])].copy()
    paired['delta_IC'] = paired['IC'] - paired['IC_perm']  # positive = shuffling hurts
    assert len(paired) > 0, 'no paired baseline/permuted IC rows after NaN filter'

    perm_agg = paired.groupby(['model', 'fold', 'seed', 'group'])['delta_IC'].mean().reset_index()
    ranking = perm_agg.groupby('group')['delta_IC'].agg(['mean', 'std', 'count']).reset_index()
    ranking = ranking.sort_values('mean', ascending=False)
    ranking_dict = {row['group']: {
        'mean_delta_IC': float(row['mean']),
        'std_delta_IC': float(row['std']),
        'n_obs': int(row['count']),
    } for _, row in ranking.iterrows()}

    per_fold_ranking = perm_agg.groupby(['fold', 'group'])['delta_IC'].mean().reset_index()
    per_fold_ranking = per_fold_ranking.pivot(index='group', columns='fold', values='delta_IC')

    with open(ARTIFACT_DIR / 'part_a_ranking.json', 'w') as f:
        json.dump({
            'aggregate_ranking': ranking_dict,
            'per_fold_mean_delta_IC': per_fold_ranking.to_dict(),
            'n_paired_rows': int(len(paired)),
            'n_model_fold_seed_runs': int(len(perm_agg[['model', 'fold', 'seed']].drop_duplicates())),
            'note': 'delta_IC positive means shuffling group hurts IC (group is important). '
                    'Pairs baseline/permuted IC per (model,fold,seed,day) before averaging.',
        }, f, indent=2)
    pd.DataFrame(scaler_log).to_json(ARTIFACT_DIR / 'per_fold_scaler.json', orient='records', indent=2)

    print('\n[ranking] group delta-IC (mean across runs):')
    print(ranking.to_string(index=False))
    print(f'\n[done] {time.time() - t0:.1f}s, wrote part_a_daily_ic.csv, part_a_permuted_ic.csv, part_a_ranking.json')


if __name__ == '__main__':
    main()
