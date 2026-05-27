#!/usr/bin/env python
"""
Story A E1 anchor — 4 models × 10 canonical seeds × 5 folds × 2 universes = 400 cells

Per plan §1.1 (v3 + Codex Round D D-01..D-05 fixes + Round E E-01 fix).

Models : {GAT, SAGE-Mean, MLP, LightGBM}      (NO LSTM — see §1.1 limitation note)
Seeds  : [86, 123, 456, 789, 1024, 2024, 7, 34, 99, 2026]  (canonical, locked)
Folds  : 5-fold walk-forward, Q2-2024 .. Q2-2025 test periods
Univ.  : B (10-dim hc)   or  C (51-dim Plan AAA top-15 members)
Horizon: 21d (locked per CLAUDE.md Rule 8)
Edge   : correlation only (single relation; sector/news handled in E4-α / E3)
Label  : market-demeaned 21d forward return → CS z-score

Outputs (atomic per cell):
- experiments/storya_e1_anchor/results.csv               (one row per cell)
- experiments/storya_e1_anchor/manifest.csv              (resume bookkeeping)
- experiments/storya_e1_anchor/per_day_ic/{univ}_{model}_s{seed}_f{fold}.npy
- experiments/storya_e1_anchor/smoke_benchmark.csv       (--smoke mode only)

CLI:
- --universe {B|C|both}     default: both
- --models GAT,SAGE-Mean,MLP,LightGBM   default: all four
- --seeds 86,123,...        default: canonical 10
- --folds 0,1,2,3,4         default: all five
- --smoke                   1 cell per model on Universe B fold 0 seed 86 (§1.10 gate)
- --resume                  default ON; honors manifest.csv
- --no-resume               force re-run all cells
"""

import os
import sys
import platform

# CR-A-02 fix follow-up (macOS OpenMP conflict, 2026-05-26 night):
# When PyTorch (Intel OpenMP / libiomp5) and LightGBM (Homebrew libomp) initialize
# their OpenMP runtimes in the same Python process on macOS, the second initialization
# silently segfaults LightGBM workers (reproduced on M4 with lightgbm 4.6.0 + torch +
# 30K+ row datasets — exit 139). Fix: force single-thread OpenMP on macOS before any
# numpy/torch/lightgbm import. On Linux (Colab A100), this env var has no harmful
# effect because we're GPU-bound for NN training and LightGBM cells are CPU-light;
# but we restrict the patch to macOS to preserve full A100 throughput. MUST be set
# BEFORE `import numpy` (numpy loads OpenMP via BLAS at import time).
if platform.system() == 'Darwin':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import random
import warnings
import gc
import argparse
import json
from collections import defaultdict

# lightgbm MUST be imported BEFORE torch on macOS to avoid the OpenMP conflict above.
# The OMP_NUM_THREADS=1 env var alone is sufficient but import order is the documented
# secondary safeguard.
import lightgbm as lgb  # noqa: F401  (module also re-imported inside train_lightgbm)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# PARAMETERS (v3 LOCKED — see plan §1.1, §1.7, §1.8)
# ══════════════════════════════════════════════════════════════

CANONICAL_SEEDS = [86, 123, 456, 789, 1024, 2024, 7, 34, 99, 2026]
ALL_MODELS = ['GAT', 'SAGE-Mean', 'MLP', 'LightGBM']
ALL_UNIVERSES = ['B', 'C']
HORIZON = 21  # locked per CLAUDE.md Rule 8

# Walk-forward folds (port from archived/scripts/run_horizon_ablation.py:72-83)
WALK_FORWARD_FOLDS = [
    {'id': 0, 'train_end': '2023-12-31', 'val_end': '2024-03-31', 'test_end': '2024-06-30',
     'desc': 'Q2-2024'},
    {'id': 1, 'train_end': '2024-03-31', 'val_end': '2024-06-30', 'test_end': '2024-09-30',
     'desc': 'Q3-2024'},
    {'id': 2, 'train_end': '2024-06-30', 'val_end': '2024-09-30', 'test_end': '2024-12-31',
     'desc': 'Q4-2024'},
    {'id': 3, 'train_end': '2024-09-30', 'val_end': '2024-12-31', 'test_end': '2025-03-31',
     'desc': 'Q1-2025'},
    {'id': 4, 'train_end': '2024-12-31', 'val_end': '2025-03-31', 'test_end': '2025-06-30',
     'desc': 'Q2-2025'},
]
TRAIN_START = '2021-07-01'

NN_HPARAMS = {  # plan §1.7 LOCKED — same for GAT / SAGE-Mean / MLP
    'hidden_channels': 64,
    'num_layers': 2,
    'dropout': 0.3,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'epochs': 100,
    'patience': 15,
    'grad_accum': 32,
    'gat_heads': 4,
}
LGB_HPARAMS = {  # plan §1.7 LOCKED — Qlib LGBModel defaults
    'num_leaves': 31,
    'n_estimators': 100,
    'learning_rate': 0.05,
    'min_data_in_leaf': 20,
    'patience': 10,
    'metric': 'val_IC',
}
GRAPH_HPARAMS = {
    'corr_window': 126,
    'corr_threshold': 0.6,
    'corr_step': 21,
}

PATHS = {
    'prices': 'data/reference/sp500_5y_prices.csv',
    'sectors': 'data/reference/sp500_sectors.csv',
    'phase5_npy': 'data/reference/sp500_5y_phase5_features.npy',
    'phase5_meta': 'data/reference/sp500_5y_phase5_features_meta.json',
    'alpha158_npy': 'data/reference/sp500_5y_alpha158_features_raw.npy',
    'alpha158_meta': 'data/reference/sp500_5y_alpha158_features_meta.json',
}

OUT_DIR = 'experiments/storya_e1_anchor'
RESULTS_CSV = f'{OUT_DIR}/results.csv'
MANIFEST_CSV = f'{OUT_DIR}/manifest.csv'
PER_DAY_IC_DIR = f'{OUT_DIR}/per_day_ic'
SMOKE_CSV = f'{OUT_DIR}/smoke_benchmark.csv'
HP_GRID_JSON = f'{OUT_DIR}/hp_grid.json'

# ══════════════════════════════════════════════════════════════
# Universe C feature list (plan §1.1, top-15 Plan AAA groups → 51 unique features)
# ══════════════════════════════════════════════════════════════

UNIVERSE_C_ALPHA158_NAMES = [
    'ROC30', 'MA60', 'MAX60', 'MIN60', 'QTLU60', 'QTLD60',
    'CNTP60', 'CNTD60',
    'KMID', 'KMID2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'VWAP0',
    'RESI60',
    'BETA20', 'RANK20', 'RSV20', 'IMAX20', 'IMXD20', 'SUMP20', 'SUMD20', 'RANK30', 'RSV30',
    'CNTP5', 'CNTN5', 'CNTD5', 'CNTP10', 'CNTN10', 'CNTD10',
    'ROC60', 'IMIN60', 'CNTN60', 'SUMN60',
    'WVMA20', 'WVMA30',
    'KUP', 'KUP2',
    'RANK60', 'RSV60', 'IMAX60',
    'CNTP20', 'CNTD20', 'CNTP30', 'CNTD30',
    'RSQR20',
    'CORR60',
]
# +3 hc_ features (mom12m from phase5; ret_std_5d / ret_std_10d computed)
UNIVERSE_C_EXTRA_NAMES = ['hc_mom12m', 'hc_ret_std_5d', 'hc_ret_std_10d']


# ══════════════════════════════════════════════════════════════
# SEEDING + DEVICE
# ══════════════════════════════════════════════════════════════

# CR-A-05 fix: deterministic CUDA kernels (matches archived/scripts/run_horizon_ablation.py:33-34)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def setup_workdir() -> None:
    try:
        import google.colab  # noqa: F401
        os.chdir('/content/drive/MyDrive/GNN测试')
    except ImportError:
        os.chdir('/Users/heruixi/Desktop/GNN-Testing')
    print(f'Working dir: {os.getcwd()}')


# ══════════════════════════════════════════════════════════════
# CELL_ID ASSERTION (Codex Round D D-01 fix)
# ══════════════════════════════════════════════════════════════

def cell_id(universe_idx: int, model_idx: int, fold_idx: int, seed_idx: int) -> int:
    """Per plan §1.1 v3 + D-01 fix: universe_idx*200 + model_idx*50 + fold_idx*10 + seed_idx.
    Range [0, 399], injective by radix construction."""
    return universe_idx * 200 + model_idx * 50 + fold_idx * 10 + seed_idx


def assert_cell_id_injective() -> None:
    """Startup assertion per plan §1.1: enumerate all 400 cell ids; confirm no collisions."""
    seen = set()
    for u in range(2):
        for m in range(4):
            for f in range(5):
                for s in range(10):
                    cid = cell_id(u, m, f, s)
                    assert cid not in seen, f"cell_id collision at u={u},m={m},f={f},s={s}"
                    seen.add(cid)
    assert max(seen) == 399 and min(seen) == 0 and len(seen) == 400, \
        f"cell_id range broken: min={min(seen)}, max={max(seen)}, n={len(seen)}"
    print(f'✓ cell_id formula injective, range [0, 399], n=400 cells')


# ══════════════════════════════════════════════════════════════
# PURGE / EMBARGO SANITY CHECK (Codex Round D D-02 fix)
# ══════════════════════════════════════════════════════════════

def assert_purge_no_leak(trading_dates: pd.DatetimeIndex, horizon: int = HORIZON) -> None:
    """Per plan §1.8 v3 + D-02 fix: explicit label_end vs feature_start comparison
    on actual trading_dates array. Within-fold purge of last `horizon` train/val days
    is the PRIMARY leak prevention (expanding-window cross-fold is by design)."""
    train_start_ts = pd.Timestamp(TRAIN_START)
    for fold in WALK_FORWARD_FOLDS:
        fid = fold['id']
        te_ts = pd.Timestamp(fold['train_end'])
        ve_ts = pd.Timestamp(fold['val_end'])
        test_e_ts = pd.Timestamp(fold['test_end'])

        train_mask = (trading_dates >= train_start_ts) & (trading_dates <= te_ts)
        val_mask = (trading_dates > te_ts) & (trading_dates <= ve_ts)
        test_mask = (trading_dates > ve_ts) & (trading_dates <= test_e_ts)

        train_dates = trading_dates[train_mask]
        val_dates = trading_dates[val_mask]
        test_dates = trading_dates[test_mask]

        assert len(train_dates) > horizon, f"Fold {fid}: train too short ({len(train_dates)}d) for {horizon}d purge"
        assert len(val_dates) > horizon, f"Fold {fid}: val too short ({len(val_dates)}d) for {horizon}d purge"
        assert len(test_dates) > 0, f"Fold {fid}: test split is empty"

        # Purge last HORIZON trading days
        train_feat_dates = train_dates[:-horizon]
        val_feat_dates = val_dates[:-horizon]

        # Each feature at trading-day index i has label window using trading_dates[i+1 .. i+horizon]
        last_train_feat_idx = np.where(trading_dates == train_feat_dates[-1])[0][0]
        last_train_label_end_date = trading_dates[last_train_feat_idx + horizon]
        first_val_feat_date = val_feat_dates[0]
        assert last_train_label_end_date < first_val_feat_date, (
            f"Fold {fid}: train label window ends at {last_train_label_end_date} "
            f">= first val feature date {first_val_feat_date} — LEAK"
        )

        last_val_feat_idx = np.where(trading_dates == val_feat_dates[-1])[0][0]
        last_val_label_end_date = trading_dates[last_val_feat_idx + horizon]
        first_test_feat_date = test_dates[0]
        assert last_val_label_end_date < first_test_feat_date, (
            f"Fold {fid}: val label window ends at {last_val_label_end_date} "
            f">= first test feature date {first_test_feat_date} — LEAK"
        )
    print(f'✓ purge/embargo sanity check passed for all {len(WALK_FORWARD_FOLDS)} folds (h={horizon}d)')


# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════

def load_core_data():
    """Returns dict with prices, returns, all_dates, valid_tickers, ticker_to_id,
    num_days, num_stocks, sector_groups (for downstream E4-α reuse)."""
    print('Loading prices + sectors ...')
    t0 = time.time()
    prices = pd.read_csv(PATHS['prices'], index_col=0, parse_dates=True)
    sector_df = pd.read_csv(PATHS['sectors'])
    sector_col = [c for c in sector_df.columns if 'sector' in c.lower()][0]
    ticker_col = [c for c in sector_df.columns if c != sector_col][0]
    sector_map = dict(zip(sector_df[ticker_col], sector_df[sector_col]))

    price_tickers = set(prices.columns)
    sector_tickers = set(sector_map.keys())
    valid_tickers = sorted(price_tickers & sector_tickers)
    ticker_to_id = {t: i for i, t in enumerate(valid_tickers)}

    prices = prices[valid_tickers]
    returns = prices.pct_change()
    returns.iloc[0] = 0.0
    all_dates = prices.index
    num_stocks = len(valid_tickers)
    num_days = len(all_dates)

    sector_groups = defaultdict(list)
    for t in valid_tickers:
        if t in sector_map:
            sector_groups[sector_map[t]].append(ticker_to_id[t])

    print(f'  Loaded: {num_stocks} stocks × {num_days} days in {time.time()-t0:.1f}s')
    return {
        'prices': prices, 'returns': returns, 'all_dates': all_dates,
        'valid_tickers': valid_tickers, 'ticker_to_id': ticker_to_id,
        'num_stocks': num_stocks, 'num_days': num_days,
        'sector_groups': sector_groups,
    }


def build_universe_B(prices: pd.DataFrame, returns: pd.DataFrame):
    """Universe B: 10-dim hc per run_step3_plan_z_part_a.py:101-117.
    All features T-1 lagged (rolling.shift(1) pattern). Returns (features_np, names)."""
    frames = []
    names = []
    for w in [5, 10, 21]:
        frames.append(returns.rolling(w).mean().shift(1))
        names.append(f'ret_mean_{w}d')
    for w in [5, 10, 21]:
        frames.append(returns.rolling(w).std().shift(1))
        names.append(f'ret_std_{w}d')
    old_tensor = np.stack([f.values for f in frames], axis=-1)

    phase5 = np.load(PATHS['phase5_npy'])  # (T, N, 5) = [mom12m, maxret, dolvol, CORR5, RSV5]
    new_tensor = phase5[:, :, :4]  # drop RSV5
    names.extend(['mom12m', 'maxret', 'dolvol', 'CORR5'])

    features = np.concatenate([old_tensor, new_tensor], axis=-1).astype(np.float32)
    features = np.nan_to_num(features, 0.0)
    assert features.shape[2] == 10, f"Universe B expected 10 features, got {features.shape[2]}"
    return features, names


def build_universe_C(prices: pd.DataFrame, returns: pd.DataFrame):
    """Universe C: 51-dim = 48 Alpha158 (sliced by name, T-1 SHIFTED at runtime) +
    3 hc_ (mom12m from phase5, ret_std_{5,10}d computed inline — all already lagged).

    CRITICAL TEMPORAL CONTRACT (Codex Touchpoint 2 Round A CR-A-01 fix, H博士 option A,
    2026-05-26 night):

    The source artifact `data/reference/sp500_5y_alpha158_features_raw.npy` is built by
    `build_alpha158_features.py` which evaluates qlib expressions like
    `($close - $open) / $open`, `Mean($close, d) / $close`, etc. on SAME-DAY prices
    (no .shift(1) anywhere in the build pipeline; lines 209-251 of build script).
    Row t of the npy therefore encodes features computed from prices at index t.

    Our pipeline requires features at feature_date t to be known by close_of_day(t-1)
    (CLAUDE.md Rule 8 universal invariant + plan §1.8 contract). To enforce this WITHOUT
    rebuilding the Alpha158 artifact (which would invalidate Plan AAA results — H博士
    option A defers that to paper §Limitations), we apply np.roll along the time axis
    immediately after slicing. This matches the news-feature T-1 lag pattern in
    `archived/scripts/run_horizon_ablation.py:169-171` (post-load corrective shift).

    Phase5 features (mom12m, maxret, dolvol, CORR5) DO have .shift(1) / .shift(22)
    already baked in per `build_phase5_features.py:78-96` and need NO additional shift.

    Plan AAA limitation (carry to paper §Limitations): the Plan AAA ranking pipeline
    `run_plan_aaa_168_ranking.py:219` loads the same npy WITHOUT applying this shift, so
    its 158 alpha158 features have a same-day leak. Plan AAA BH-FDR result (0/61 positive
    groups pass) is directionally robust to this leak (a leaky IC overestimates magnitude
    but does not create false negatives against the null). Plan AAA does NOT need re-running
    to produce defensible Universe C composition.
    """
    alpha158_meta = json.load(open(PATHS['alpha158_meta']))
    alpha158_order = alpha158_meta['feature_order']
    alpha158_arr = np.load(PATHS['alpha158_npy'])  # (T, N, 158)

    name_to_idx = {n: i for i, n in enumerate(alpha158_order)}
    for n in UNIVERSE_C_ALPHA158_NAMES:
        assert n in name_to_idx, f"Universe C feature {n} missing from alpha158_meta"
    a158_cols = [name_to_idx[n] for n in UNIVERSE_C_ALPHA158_NAMES]
    a158_slice_raw = alpha158_arr[:, :, a158_cols].astype(np.float32)  # (T, N, 48), SAME-DAY

    # T-1 shift along time axis (CR-A-01 fix). Row 0 becomes zeros (no prior day to lag from).
    a158_slice = np.roll(a158_slice_raw, shift=1, axis=0)
    a158_slice[0] = 0.0

    # Sanity check on the shift: a158_slice[1] must equal a158_slice_raw[0] elementwise.
    assert np.array_equal(a158_slice[1], a158_slice_raw[0]), \
        "Alpha158 T-1 shift failed: a158_slice[1] != a158_slice_raw[0]"
    assert np.all(a158_slice[0] == 0.0), \
        "Alpha158 T-1 shift failed: row 0 should be zeroed"

    # hc_mom12m comes from phase5 (already shifted at build time per build_phase5_features.py:78)
    phase5 = np.load(PATHS['phase5_npy'])  # (T, N, 5)
    hc_mom12m = phase5[:, :, 0:1].astype(np.float32)
    # hc_ret_std_{5,10}d computed inline from returns (T-1 lagged via .shift(1))
    ret_std_5 = returns.rolling(5).std().shift(1).values[:, :, None].astype(np.float32)
    ret_std_10 = returns.rolling(10).std().shift(1).values[:, :, None].astype(np.float32)
    hc_extra = np.concatenate([hc_mom12m, ret_std_5, ret_std_10], axis=-1)  # (T, N, 3)

    features = np.concatenate([a158_slice, hc_extra], axis=-1)
    features = np.nan_to_num(features, 0.0)
    names = list(UNIVERSE_C_ALPHA158_NAMES) + list(UNIVERSE_C_EXTRA_NAMES)
    assert features.shape[2] == 51, f"Universe C expected 51 features, got {features.shape[2]}"
    return features, names


def build_labels(prices: pd.DataFrame, horizon: int = HORIZON):
    """Market-demeaned 21d forward return → CS z-score per plan §1.1 + run_horizon_ablation.py:184-189.
    Returns (labels_np, label_valid_np)."""
    fwd_ret = prices.shift(-horizon) / prices - 1
    market_ret = fwd_ret.mean(axis=1)
    excess = fwd_ret.sub(market_ret, axis=0)
    day_mean = excess.mean(axis=1)
    day_std = excess.std(axis=1)
    day_std[day_std < 1e-8] = 1.0
    z = excess.sub(day_mean, axis=0).div(day_std, axis=0)
    valid = ~z.isna()
    labels_np = np.nan_to_num(z.values.astype(np.float32), 0.0)
    label_valid_np = valid.values
    return labels_np, label_valid_np


# ══════════════════════════════════════════════════════════════
# CORRELATION GRAPH (single relation per plan §1.1 E1)
# ══════════════════════════════════════════════════════════════

def build_correlation_snapshots(returns: pd.DataFrame, num_days: int):
    """Per-snapshot correlation edge index (last snapshot ≤ end_of_day(t-1) at use site).
    Returns (snapshots dict, day_to_snapshot dict, snapshot_points list)."""
    snapshot_points = list(range(GRAPH_HPARAMS['corr_window'], num_days, GRAPH_HPARAMS['corr_step']))
    snaps = {}
    for si, t_end in enumerate(snapshot_points):
        w_ret = returns.iloc[t_end - GRAPH_HPARAMS['corr_window']:t_end].values
        cm = np.corrcoef(w_ret.T)
        np.fill_diagonal(cm, 0)
        src, dst = np.where(np.abs(cm) > GRAPH_HPARAMS['corr_threshold'])
        snaps[si] = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    day_to_snapshot = {}
    si = 0
    for di in range(num_days):
        while si + 1 < len(snapshot_points) and snapshot_points[si + 1] <= di:
            si += 1
        day_to_snapshot[di] = si if snapshot_points[si] <= di else 0

    return snaps, day_to_snapshot, snapshot_points


def get_frozen_snapshot_idx(train_end_idx: int, snapshot_points: list) -> int:
    """C3 fix: freeze correlation graph to last snapshot inside train window."""
    best_si = 0
    for si, sp in enumerate(snapshot_points):
        if sp <= train_end_idx:
            best_si = si
        else:
            break
    return best_si


# ══════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════

def make_nn_model(model_name: str, in_channels: int, device: torch.device):
    """Returns a torch model on device. model_name ∈ {'GAT', 'SAGE-Mean', 'MLP'}."""
    from torch_geometric.nn import SAGEConv, GATConv

    hidden = NN_HPARAMS['hidden_channels']
    num_layers = NN_HPARAMS['num_layers']
    dropout = NN_HPARAMS['dropout']

    class _NN(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin_in = nn.Linear(in_channels, hidden)
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            for _ in range(num_layers):
                if model_name == 'GAT':
                    self.convs.append(GATConv(hidden, hidden // NN_HPARAMS['gat_heads'],
                                              heads=NN_HPARAMS['gat_heads'], dropout=dropout,
                                              add_self_loops=True))
                elif model_name == 'SAGE-Mean':
                    self.convs.append(SAGEConv(hidden, hidden, aggr='mean'))
                elif model_name == 'MLP':
                    self.convs.append(nn.Linear(hidden, hidden))
                else:
                    raise ValueError(f'Unknown NN model_name: {model_name}')
                self.norms.append(nn.LayerNorm(hidden))
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden // 2), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden // 2, 1),
            )
            self.dropout = dropout
            self._is_mlp = (model_name == 'MLP')

        def forward(self, x, edge_index=None):
            h = F.relu(self.lin_in(x))
            for conv, norm in zip(self.convs, self.norms):
                if self._is_mlp:
                    h_new = conv(h)
                else:
                    h_new = conv(h, edge_index)
                h = norm(F.dropout(h_new, p=self.dropout, training=self.training) + h)
            return self.head(h).squeeze(-1)

    return _NN().to(device)


# ══════════════════════════════════════════════════════════════
# FOLD MASKS (within-fold purge per plan §1.8 C1 fix)
# ══════════════════════════════════════════════════════════════

def create_fold_masks(fold_cfg: dict, all_dates: pd.DatetimeIndex, horizon: int = HORIZON):
    ts = pd.Timestamp(TRAIN_START)
    te = pd.Timestamp(fold_cfg['train_end'])
    ve = pd.Timestamp(fold_cfg['val_end'])
    test_e = pd.Timestamp(fold_cfg['test_end'])
    train_days = np.where((all_dates >= ts) & (all_dates <= te))[0]
    val_days = np.where((all_dates > te) & (all_dates <= ve))[0]
    test_days = np.where((all_dates > ve) & (all_dates <= test_e))[0]
    # C1 FIX: purge last `horizon` trading days from train + val
    train_days = train_days[:-horizon]
    val_days = val_days[:-horizon]
    return train_days, val_days, test_days


# ══════════════════════════════════════════════════════════════
# FEATURE PRE-PROCESSING (per-fold, train-only fit)
# ══════════════════════════════════════════════════════════════

def winsorize_train_only(features: np.ndarray, train_days: np.ndarray, p_low=0.01, p_high=0.99):
    """Per-fold winsorization: compute p1/p99 bounds on TRAIN days only;
    clip all days. Per plan §1.8. Returns winsorized copy."""
    train_slice = features[train_days]  # (T_train, N, D)
    flat = train_slice.reshape(-1, train_slice.shape[-1])
    finite_mask = np.isfinite(flat)
    bounds_low = np.zeros(flat.shape[1], dtype=np.float32)
    bounds_high = np.zeros(flat.shape[1], dtype=np.float32)
    for d in range(flat.shape[1]):
        col = flat[:, d][finite_mask[:, d]]
        if len(col) > 0:
            bounds_low[d] = np.quantile(col, p_low)
            bounds_high[d] = np.quantile(col, p_high)
    out = np.clip(features, bounds_low, bounds_high)
    return out.astype(np.float32)


def standardize_train_only(features: np.ndarray, train_days: np.ndarray):
    """Per-fold StandardScaler fit on TRAIN days only (per plan §1.8). NN-only."""
    train_slice = features[train_days]
    flat_train = train_slice.reshape(-1, train_slice.shape[-1])
    scaler = StandardScaler()
    scaler.fit(flat_train)
    out = scaler.transform(features.reshape(-1, features.shape[-1])).reshape(features.shape).astype(np.float32)
    return out


# ══════════════════════════════════════════════════════════════
# TRAIN: NN MODELS (GAT, SAGE-Mean, MLP)
# ══════════════════════════════════════════════════════════════

def train_nn(model_name: str, features_t: torch.Tensor, labels_t: torch.Tensor,
             label_valid_t: torch.Tensor, train_days: np.ndarray, val_days: np.ndarray,
             test_days: np.ndarray, corr_snapshots: dict, frozen_si: int,
             num_days: int, num_stocks: int, seed: int, device: torch.device):
    set_seed(seed)
    in_ch = features_t.shape[-1]
    model = make_nn_model(model_name, in_ch, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=NN_HPARAMS['lr'],
                                 weight_decay=NN_HPARAMS['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=1e-5)

    edge_index = corr_snapshots[frozen_si].to(device) if model_name != 'MLP' else None

    best_val = float('inf')
    best_state = None
    no_improve = 0
    epochs_run = 0

    for epoch in range(NN_HPARAMS['epochs']):
        epochs_run = epoch + 1
        model.train()
        optimizer.zero_grad()
        accum_steps = 0
        day_order = train_days[np.random.permutation(len(train_days))]

        for step, d in enumerate(day_order):
            x = features_t[d].to(device)
            pred = model(x, edge_index)
            mask = label_valid_t[d].to(device)
            if mask.sum() < 10:
                continue
            target = labels_t[d].to(device)
            loss = F.mse_loss(pred[mask], target[mask])
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
                x = features_t[d].to(device)
                pred = model(x, edge_index)
                mask = label_valid_t[d].to(device)
                if mask.sum() < 10:
                    continue
                v_loss += F.mse_loss(pred[mask], labels_t[d].to(device)[mask]).item()
                v_cnt += 1

        avg_val = v_loss / max(v_cnt, 1)
        scheduler.step(avg_val)
        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= NN_HPARAMS['patience']:
            break

    assert best_state is not None, f"{model_name} seed={seed} produced no valid state"
    model.load_state_dict(best_state)
    model.to(device).eval()

    preds = np.zeros((num_days, num_stocks), dtype=np.float32)
    with torch.no_grad():
        for d in test_days:
            x = features_t[d].to(device)
            preds[d] = model(x, edge_index).cpu().numpy()

    del model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return preds, {'best_val_loss': best_val, 'epochs_run': epochs_run}


# ══════════════════════════════════════════════════════════════
# TRAIN: LIGHTGBM
# ══════════════════════════════════════════════════════════════

def train_lightgbm(features_np: np.ndarray, labels_np: np.ndarray, label_valid_np: np.ndarray,
                   train_days: np.ndarray, val_days: np.ndarray, test_days: np.ndarray,
                   num_days: int, num_stocks: int, seed: int):
    """LightGBM regressor on flattened (day, stock) samples.

    No standardization (trees are scale-invariant); winsorization already applied upstream.

    CR-A-02 fix (Codex Touchpoint 2): switched from sklearn-API LGBMRegressor + custom
    callback (which raised lgb.callback.EarlyStopException with the brittle
    env.model.boost_round attribute and was wrapped in a bare `except Exception: pass`
    that could silently swallow real failures) to the official `lgb.train()` API with
    `feval` for the custom val-IC metric and `lgb.early_stopping` for the documented
    early-stop pattern. Across modern lightgbm versions, this works without attribute
    introspection and any genuine fit error now surfaces instead of being masked.
    """
    import lightgbm as lgb
    set_seed(seed)

    def flatten(days):
        X_list, y_list, day_idx_list = [], [], []
        for d in days:
            mask = label_valid_np[d]
            if mask.sum() < 10:
                continue
            X_list.append(features_np[d][mask])
            y_list.append(labels_np[d][mask])
            day_idx_list.extend([d] * int(mask.sum()))
        if not X_list:
            return (np.zeros((0, features_np.shape[-1]), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64))
        X = np.concatenate(X_list, axis=0).astype(np.float32)
        y = np.concatenate(y_list, axis=0).astype(np.float32)
        day_idx = np.array(day_idx_list, dtype=np.int64)
        return X, y, day_idx

    X_train, y_train, _ = flatten(train_days)
    X_val, y_val, val_day_idx = flatten(val_days)

    if len(X_train) == 0 or len(X_val) == 0:
        raise RuntimeError(f'LightGBM train_lightgbm: empty train ({len(X_train)}) '
                           f'or val ({len(X_val)}) — cannot fit')

    def val_ic_feval(preds, eval_data):
        # Called by lgb.train on the validation set. Returns (name, value, is_higher_better).
        # preds: 1-D array aligned with val rows; eval_data: the val lgb.Dataset.
        ic_list = []
        for d in np.unique(val_day_idx):
            sel = (val_day_idx == d)
            if sel.sum() < 30:
                continue
            ic, _ = spearmanr(preds[sel], y_val[sel])
            if not np.isnan(ic):
                ic_list.append(ic)
        return ('val_IC', float(np.mean(ic_list)) if ic_list else 0.0, True)  # higher_is_better

    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    params = {
        'objective': 'regression',
        'num_leaves': LGB_HPARAMS['num_leaves'],
        'learning_rate': LGB_HPARAMS['learning_rate'],
        'min_data_in_leaf': LGB_HPARAMS['min_data_in_leaf'],
        'seed': seed,
        'deterministic': True,
        'force_row_wise': True,
        'verbose': -1,
        'metric': 'None',  # disable built-in metrics; use feval exclusively
    }

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=LGB_HPARAMS['n_estimators'],
        valid_sets=[val_set],
        valid_names=['val'],
        feval=val_ic_feval,
        callbacks=[
            lgb.early_stopping(stopping_rounds=LGB_HPARAMS['patience'],
                               first_metric_only=True, verbose=False),
            lgb.log_evaluation(period=0),  # silent
        ],
    )

    best_iter = booster.best_iteration if booster.best_iteration else booster.current_iteration()
    # booster.best_score['val']['val_IC'] is the best val IC value
    best_val_ic = float(booster.best_score['val']['val_IC']) if booster.best_score else 0.0

    # Predict on test
    preds = np.zeros((num_days, num_stocks), dtype=np.float32)
    for d in test_days:
        mask = label_valid_np[d]
        if mask.sum() == 0:
            continue
        pred_d = booster.predict(features_np[d][mask], num_iteration=best_iter)
        full = np.zeros(num_stocks, dtype=np.float32)
        full[mask] = pred_d.astype(np.float32)
        preds[d] = full

    # Sign convention: best_val_loss = -best_val_ic (lower = better val loss).
    # Both NN and LightGBM share this column; downstream consumers should not be surprised.
    return preds, {'best_val_loss': -best_val_ic, 'epochs_run': best_iter}


# ══════════════════════════════════════════════════════════════
# METRICS (per-day IC + cost-ladder Sharpe)
# ══════════════════════════════════════════════════════════════

def compute_daily_ic(predictions: np.ndarray, day_indices: np.ndarray,
                     labels_np: np.ndarray, label_valid_np: np.ndarray):
    ic_list = []
    for d in day_indices:
        mask = label_valid_np[d]
        if mask.sum() < 30:
            continue
        pred = predictions[d][mask]
        actual = labels_np[d][mask]
        ic, _ = spearmanr(pred, actual)
        if not np.isnan(ic):
            ic_list.append(ic)
    return np.array(ic_list, dtype=np.float32)


def compute_cost_ladder_sharpe(predictions: np.ndarray, day_indices: np.ndarray,
                                prices: pd.DataFrame, label_valid_np: np.ndarray,
                                num_stocks: int, num_days: int,
                                horizon: int = HORIZON, top_pct: float = 0.10,
                                cost_levels_bps=(0, 5, 10, 15, 20, 30)):
    """Equal-weight decile L/S portfolio, 21d non-overlapping rebalance, L1-norm turnover
    convention per plan §1.4(d) (Codex D-05 fix).

    Returns dict: Sharpe_gross + Sharpe_net_{c}bps for each c in cost_levels_bps,
    plus mean_turnover_L1, n_periods (no annualization mismatch — uses sqrt(252/21)).
    """
    K = max(1, int(np.floor(top_pct * num_stocks)))
    rebal_days = day_indices[::horizon]
    rets = []
    turnovers_L1 = []
    prev_long = set()
    prev_short = set()

    for d in rebal_days:
        mask = label_valid_np[d]
        if mask.sum() < 2 * K:
            continue
        pred = predictions[d].copy().astype(np.float64)
        pred[~mask] = np.nan
        vi = np.where(mask)[0]
        scores = pred[vi]
        tiebreak = np.arange(len(scores)) * 1e-10
        ranks = np.argsort(np.argsort(-(scores + tiebreak)))
        top_idx = set(vi[ranks < K].tolist())
        bot_idx = set(vi[ranks >= len(scores) - K].tolist())

        d_end = min(d + horizon, num_days - 1)
        if d_end >= prices.shape[0]:
            continue
        ps = prices.iloc[d].values
        pe = prices.iloc[d_end].values
        sr = (pe - ps) / np.where(ps > 0, ps, 1e-8)
        lr = np.nanmean([sr[i] for i in top_idx])
        shr = np.nanmean([sr[i] for i in bot_idx])
        rets.append(lr - shr)

        # L1-norm turnover per D-05: counts both entries and exits, at full rotation = 4
        # = 2 * one-side definition used in horizon_ablation.py:370
        oneside = (1 - len(top_idx & prev_long) / K) + (1 - len(bot_idx & prev_short) / K) if prev_long else 2.0
        turnovers_L1.append(2.0 * oneside)
        prev_long = top_idx
        prev_short = bot_idx

    if not rets:
        out = {'Sharpe_gross': 0.0, 'mean_turnover_L1': 0.0, 'n_periods': 0}
        for c in cost_levels_bps:
            out[f'Sharpe_net_{c}bps'] = 0.0
        return out

    rets = np.array(rets, dtype=np.float64)
    turnovers_L1 = np.array(turnovers_L1, dtype=np.float64)
    annualizer = np.sqrt(252.0 / horizon)
    mean_gross = rets.mean()
    std_gross = rets.std() if len(rets) > 1 else 1e-8

    out = {
        'Sharpe_gross': float(mean_gross / max(std_gross, 1e-8) * annualizer) if std_gross > 1e-8 else 0.0,
        'mean_turnover_L1': float(turnovers_L1[1:].mean()) if len(turnovers_L1) > 1 else 0.0,
        'n_periods': int(len(rets)),
    }
    for c in cost_levels_bps:
        net_rets = rets - turnovers_L1 * (c / 10000.0)
        std_net = net_rets.std() if len(net_rets) > 1 else 1e-8
        out[f'Sharpe_net_{c}bps'] = float(net_rets.mean() / max(std_net, 1e-8) * annualizer) if std_net > 1e-8 else 0.0

    return out


# ══════════════════════════════════════════════════════════════
# MANIFEST / RESUME
# ══════════════════════════════════════════════════════════════

# Schemas for header pre-initialization (CR-A-04 fix — eliminates race condition where
# concurrent sessions both call os.path.exists at the same instant and both write headers).
COST_LEVELS_BPS = (0, 5, 10, 15, 20, 30)
RESULTS_COLUMNS = (
    ['cell_id', 'universe', 'model', 'seed', 'fold', 'test_period',
     'IC_mean', 'IC_std', 'n_test_days', 'Sharpe_gross']
    + [f'Sharpe_net_{c}bps' for c in COST_LEVELS_BPS]
    + ['mean_turnover_L1', 'n_periods', 'best_val_loss', 'epochs_run',
       'wall_time_sec', 'converged_flag', 'cost_convention']
)
MANIFEST_COLUMNS = ['cell_id', 'universe', 'model', 'seed', 'fold',
                    'status', 'start_ts', 'end_ts', 'wall_time_sec', 'err']
COST_CONVENTION = 'L1_one_way'  # plan §1.4(d) D-05 fix


def init_csv_files() -> None:
    """CR-A-04 fix: pre-write headers if files don't exist. Single-shot at startup,
    eliminating per-cell os.path.exists race condition that would corrupt CSVs under
    concurrent multi-session runs."""
    if not os.path.exists(RESULTS_CSV):
        pd.DataFrame(columns=RESULTS_COLUMNS).to_csv(RESULTS_CSV, index=False)
    if not os.path.exists(MANIFEST_CSV):
        pd.DataFrame(columns=MANIFEST_COLUMNS).to_csv(MANIFEST_CSV, index=False)


def write_run_meta_json(path: str) -> None:
    """CR-A-03 fix: document cost-ladder convention + temporal contract for downstream
    consumers (compute_e6_dm_spa.py, analyze_storya_results.py). Without this, the
    L1-norm turnover convention (2x the archived horizon_ablation oneside) is invisible
    in results.csv schema."""
    meta = {
        'experiment_id': 'storya_e1_anchor_v3',
        'plan_ref': 'plan §1.1 (E1) + §1.4 (statistics) + §1.7 (hparams) + §1.8 (temporal contract)',
        'horizon_days': HORIZON,
        'canonical_seeds': CANONICAL_SEEDS,
        'models': ALL_MODELS,
        'universes': ALL_UNIVERSES,
        'cost_ladder': {
            'levels_bps': list(COST_LEVELS_BPS),
            'convention': COST_CONVENTION,
            'turnover_definition': 'L1-norm: turnover_L1 = sum_i|p_i(t) - p_i(t-1)|; at full L-S rotation = 4',
            'relation_to_archived': 'L1 = 2 * one_side (used in archived/scripts/run_horizon_ablation.py:370)',
            'cost_formula': 'cost_per_period(t, c_bps) = turnover_L1(t) * c_bps * 1e-4',
            'annualization': 'sqrt(252/horizon)',
            'note': 'Per plan §1.4(d) D-05 fix. NOT directly comparable to existing experiments/horizon_ablation_results.csv Sharpe_net (oneside convention).',
        },
        'temporal_contract_ref': 'plan §1.8 v3 (Codex Round D D-02 + D-03 fixes); see run_storya_e1_anchor.py:assert_purge_no_leak',
        'universe_C_alpha158_t1_shift_note': 'Alpha158 npy is built same-day; build_universe_C applies np.roll T-1 shift at runtime per Codex Touchpoint 2 CR-A-01 fix',
    }
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)


def load_manifest(path: str):
    if not os.path.exists(path):
        return pd.DataFrame(columns=MANIFEST_COLUMNS), set()
    df = pd.read_csv(path)
    done = set()
    if 'status' in df.columns and len(df) > 0:
        done_rows = df[df['status'] == 'completed']
        if len(done_rows) > 0:
            done = set(zip(done_rows['universe'].astype(str),
                           done_rows['model'].astype(str),
                           done_rows['seed'].astype(int),
                           done_rows['fold'].astype(int)))
    return df, done


def append_manifest(path: str, row: dict) -> None:
    """CR-A-04 fix: header=False unconditionally (file pre-initialized by init_csv_files)."""
    df = pd.DataFrame([row], columns=MANIFEST_COLUMNS)
    df.to_csv(path, mode='a', header=False, index=False)


def append_results(path: str, row: dict) -> None:
    """CR-A-04 fix: header=False unconditionally (file pre-initialized by init_csv_files)."""
    df = pd.DataFrame([row], columns=RESULTS_COLUMNS)
    df.to_csv(path, mode='a', header=False, index=False)


def write_hp_grid_json(path: str) -> None:
    hp = {
        'GAT': {**{k: v for k, v in NN_HPARAMS.items() if k != 'gat_heads'},
                'heads': NN_HPARAMS['gat_heads']},
        'SAGE-Mean': {**{k: v for k, v in NN_HPARAMS.items() if k != 'gat_heads'}, 'aggr': 'mean'},
        'MLP': {**{k: v for k, v in NN_HPARAMS.items() if k != 'gat_heads'}},
        'LightGBM': dict(LGB_HPARAMS),
        '_source': 'plan §1.7 LOCKED per Codex A-07 fix + Round D D-01 LSTM drop residual cleanup',
        '_seeds': CANONICAL_SEEDS,
        '_horizon_days': HORIZON,
    }
    with open(path, 'w') as f:
        json.dump(hp, f, indent=2)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--universe', choices=['B', 'C', 'both'], default='both')
    parser.add_argument('--models', type=str, default=','.join(ALL_MODELS),
                        help='Comma-separated subset of GAT,SAGE-Mean,MLP,LightGBM')
    parser.add_argument('--seeds', type=str, default=','.join(str(s) for s in CANONICAL_SEEDS),
                        help='Comma-separated seed list (must be subset of canonical 10)')
    parser.add_argument('--folds', type=str, default='0,1,2,3,4')
    parser.add_argument('--smoke', action='store_true',
                        help='Smoke benchmark: 4 cells (one per model) on B fold 0 seed 86')
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    args = parser.parse_args()

    # --- startup setup ---
    setup_workdir()
    for d in [OUT_DIR, PER_DAY_IC_DIR]:
        os.makedirs(d, exist_ok=True)
    write_hp_grid_json(HP_GRID_JSON)
    write_run_meta_json(f'{OUT_DIR}/_meta.json')  # CR-A-03 fix
    init_csv_files()  # CR-A-04 fix
    device = get_device()
    print(f'Device: {device}')

    # --- mandatory startup assertions (D-01 + D-02) ---
    assert_cell_id_injective()

    # --- load data ---
    core = load_core_data()
    prices = core['prices']
    returns = core['returns']
    all_dates = core['all_dates']
    num_days = core['num_days']
    num_stocks = core['num_stocks']

    assert_purge_no_leak(all_dates, horizon=HORIZON)

    # --- build labels (once, h=21d) ---
    labels_np, label_valid_np = build_labels(prices, horizon=HORIZON)
    labels_t = torch.tensor(labels_np, dtype=torch.float32)
    label_valid_t = torch.tensor(label_valid_np, dtype=torch.bool)
    print(f'Labels: h={HORIZON}d, {int(label_valid_np.sum()):,} valid (day, stock) entries')

    # --- build correlation snapshots (once) ---
    corr_snapshots, _, snapshot_points = build_correlation_snapshots(returns, num_days)
    print(f'Correlation graph: {len(snapshot_points)} snapshots ready')

    # --- resolve cli args ---
    if args.smoke:
        universes_run = ['B']
        models_run = ALL_MODELS
        seeds_run = [CANONICAL_SEEDS[0]]
        folds_run = [0]
        print('\n=== SMOKE MODE: 4 cells (B × all-models × fold 0 × seed 86) ===\n')
    else:
        universes_run = ALL_UNIVERSES if args.universe == 'both' else [args.universe]
        models_run = [m for m in args.models.split(',') if m]
        seeds_run = [int(s) for s in args.seeds.split(',') if s]
        folds_run = [int(f) for f in args.folds.split(',') if f]
        for m in models_run:
            assert m in ALL_MODELS, f'Unknown model {m}; valid: {ALL_MODELS}'
        for s in seeds_run:
            assert s in CANONICAL_SEEDS, f'Seed {s} not in canonical list {CANONICAL_SEEDS}'

    universe_idx_map = {'B': 0, 'C': 1}
    model_idx_map = {m: i for i, m in enumerate(ALL_MODELS)}
    seed_idx_map = {s: i for i, s in enumerate(CANONICAL_SEEDS)}

    # --- resume ---
    manifest_df, done_cells = load_manifest(MANIFEST_CSV)
    if args.resume:
        print(f'Resume mode ON: {len(done_cells)} cells already completed; will skip them')
    else:
        print(f'Resume mode OFF: re-running all cells (existing results.csv left intact, will append)')
        done_cells = set()

    # --- pre-build features per universe (once outside fold loop; per-fold winsor/scale inside) ---
    features_raw = {}
    if 'B' in universes_run:
        feats, names = build_universe_B(prices, returns)
        features_raw['B'] = {'features': feats, 'names': names}
        print(f'Universe B features built: {feats.shape}')
    if 'C' in universes_run:
        feats, names = build_universe_C(prices, returns)
        features_raw['C'] = {'features': feats, 'names': names}
        print(f'Universe C features built: {feats.shape}')

    # --- smoke output buffer ---
    smoke_rows = []

    # --- main loop ---
    t0_global = time.time()
    n_cells_planned = len(universes_run) * len(models_run) * len(folds_run) * len(seeds_run)
    print(f'\nPlanned cells (after CLI filter, before resume skip): {n_cells_planned}')

    for universe in universes_run:
        feats_raw = features_raw[universe]['features']
        u_idx = universe_idx_map[universe]

        for fold_cfg in WALK_FORWARD_FOLDS:
            if fold_cfg['id'] not in folds_run:
                continue
            train_days, val_days, test_days = create_fold_masks(fold_cfg, all_dates, HORIZON)
            frozen_si = get_frozen_snapshot_idx(train_days[-1], snapshot_points)

            # Per-fold winsor + scale (train-only fit)
            feats_winz = winsorize_train_only(feats_raw, train_days)
            feats_std = standardize_train_only(feats_winz, train_days)
            feats_std_t = torch.tensor(feats_std, dtype=torch.float32)
            print(f'\n[Universe {universe}] fold {fold_cfg["id"]} ({fold_cfg["desc"]}): '
                  f'train={len(train_days)}d (purged {HORIZON}d), '
                  f'val={len(val_days)}d, test={len(test_days)}d, frozen_si={frozen_si}')

            for model_name in models_run:
                m_idx = model_idx_map[model_name]
                for seed in seeds_run:
                    s_idx = seed_idx_map[seed]
                    cid = cell_id(u_idx, m_idx, fold_cfg['id'], s_idx)
                    key = (universe, model_name, int(seed), int(fold_cfg['id']))
                    if key in done_cells:
                        continue

                    label = f'[cid={cid:03d}] U{universe} {model_name:>10s} seed={seed:<5d} fold={fold_cfg["id"]}'
                    print(f'  {label} ...', end=' ', flush=True)
                    start_ts = time.time()
                    status = 'running'
                    err_msg = ''

                    try:
                        if model_name == 'LightGBM':
                            # LightGBM uses winsorized but UN-scaled features (trees are scale-invariant)
                            preds, train_info = train_lightgbm(
                                feats_winz, labels_np, label_valid_np,
                                train_days, val_days, test_days,
                                num_days, num_stocks, seed,
                            )
                        else:
                            preds, train_info = train_nn(
                                model_name, feats_std_t, labels_t, label_valid_t,
                                train_days, val_days, test_days,
                                corr_snapshots, frozen_si, num_days, num_stocks,
                                seed, device,
                            )

                        ic_arr = compute_daily_ic(preds, test_days, labels_np, label_valid_np)
                        ic_mean = float(ic_arr.mean()) if len(ic_arr) > 0 else 0.0
                        ic_std = float(ic_arr.std()) if len(ic_arr) > 0 else 0.0

                        sh = compute_cost_ladder_sharpe(
                            preds, test_days, prices, label_valid_np,
                            num_stocks, num_days, horizon=HORIZON,
                        )

                        # Save per-day IC array for downstream block-bootstrap + DM/HLN
                        ic_path = f'{PER_DAY_IC_DIR}/{universe}_{model_name}_s{seed}_f{fold_cfg["id"]}.npy'
                        np.save(ic_path, ic_arr)

                        wall = time.time() - start_ts
                        row = {
                            'cell_id': cid,
                            'universe': universe,
                            'model': model_name,
                            'seed': seed,
                            'fold': fold_cfg['id'],
                            'test_period': fold_cfg['desc'],
                            'IC_mean': round(ic_mean, 6),
                            'IC_std': round(ic_std, 6),
                            'n_test_days': len(ic_arr),
                            'Sharpe_gross': round(sh['Sharpe_gross'], 4),
                            **{k: round(v, 4) for k, v in sh.items() if k.startswith('Sharpe_net_')},
                            'mean_turnover_L1': round(sh['mean_turnover_L1'], 4),
                            'n_periods': sh['n_periods'],
                            'best_val_loss': round(train_info['best_val_loss'], 6),
                            'epochs_run': train_info['epochs_run'],
                            'wall_time_sec': round(wall, 1),
                            'converged_flag': int(train_info['best_val_loss'] > -np.inf and train_info['best_val_loss'] < np.inf),
                            'cost_convention': COST_CONVENTION,  # CR-A-03 fix
                        }
                        append_results(RESULTS_CSV, row)
                        status = 'completed'
                        print(f'IC={ic_mean:+.5f} ± {ic_std:.5f}, Sh_g={sh["Sharpe_gross"]:.3f}, '
                              f'Sh_n@10={sh["Sharpe_net_10bps"]:.3f}, n_periods={sh["n_periods"]} '
                              f'({wall:.0f}s)')

                        if args.smoke:
                            smoke_rows.append({
                                'model': model_name, 'wall_time_sec': round(wall, 1),
                                'IC_mean': round(ic_mean, 5), 'Sharpe_gross': round(sh['Sharpe_gross'], 3),
                            })

                    except Exception as e:
                        status = 'failed'
                        err_msg = str(e)[:200]
                        import traceback
                        traceback.print_exc()
                        print(f'  FAILED: {err_msg}')

                    end_ts = time.time()
                    append_manifest(MANIFEST_CSV, {
                        'cell_id': cid,
                        'universe': universe,
                        'model': model_name,
                        'seed': seed,
                        'fold': fold_cfg['id'],
                        'status': status,
                        'start_ts': round(start_ts, 1),
                        'end_ts': round(end_ts, 1),
                        'wall_time_sec': round(end_ts - start_ts, 1),
                        'err': err_msg,
                    })

                    gc.collect()
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

            del feats_winz, feats_std, feats_std_t
            gc.collect()

    # --- smoke summary ---
    if args.smoke and smoke_rows:
        pd.DataFrame(smoke_rows).to_csv(SMOKE_CSV, index=False)
        total = sum(r['wall_time_sec'] for r in smoke_rows)
        print(f'\n=== SMOKE SUMMARY ===')
        print(f'4 cells total wall: {total:.0f}s ({total/60:.1f} min)')
        print(f'Decision gate (plan §1.10): if total < 25 min → proceed to full E1 (400 cells × ~6 min ≈ 40h)')
        print(f'Per-cell wall times: {[r["wall_time_sec"] for r in smoke_rows]}')
        if total < 25 * 60:
            print(f'✓ Smoke PASSED: full E1 ≈ {400 * (total / 4) / 3600:.1f}h estimated')
        else:
            print(f'⚠ Smoke OVER BUDGET: investigate per-cell wall outlier before launching full E1')

    elapsed_h = (time.time() - t0_global) / 3600.0
    print(f'\nTotal wall time: {elapsed_h:.2f}h')
    print('=== run_storya_e1_anchor.py DONE ===')


if __name__ == '__main__':
    main()
