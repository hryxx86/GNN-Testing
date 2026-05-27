#!/usr/bin/env python
"""
Phase 5 Step 2 — Build 5 new features from OHLCV.

Inputs:
  data/reference/sp500_5y_prices.csv       — close (EODHD, canonical)
  data/reference/sp500_5y_ohlcv.parquet    — O/H/L/C/V + Adj Close (yfinance)
  data/reference/sp500_5y_adj_ohlc.parquet — split/div-adjusted OHLC (yfinance)

Output:
  data/reference/sp500_5y_phase5_features.npy      — shape (num_days, num_stocks, N)
  data/reference/sp500_5y_phase5_features_meta.json — feature names + computation notes
  data/reference/phase5_feature_diag.csv           — per-feature NaN rate, scale, PC loading

Feature definitions (from Alpha158 + GKX):
  mom12m   = prices.shift(22) / prices.shift(252) - 1               (close only)
  maxret   = returns.rolling(21).max().shift(1)                     (close only)
  dolvol   = log(mean(close*volume, 63d)).shift(1)                   (close+volume)
  CORR5    = Corr(close, log(volume+1), 5).shift(1)                  (close+volume)
  RSV5     = (close − Min(low,5)) / (Max(high,5) − Min(low,5)).shift(1)  (adj OHLC)

Does NOT apply cross-sectional normalization here — per Diag 1/1b, that's a modeling
choice deferred to training time, not baked into features.

Run:
  /opt/homebrew/Caskroom/miniforge/base/envs/gnn/bin/python build_phase5_features.py
"""

import os, sys, warnings, json, time
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from collections import defaultdict

os.chdir('/Users/heruixi/Desktop/GNN-Testing')
t0 = time.time()

# ── load prices (canonical date/ticker axes) ─────────────────────
prices = pd.read_csv('data/reference/sp500_5y_prices.csv', index_col=0, parse_dates=True)
sector_df = pd.read_csv('data/reference/sp500_sectors.csv')
sec_col = [c for c in sector_df.columns if 'sector' in c.lower()][0]
tic_col = [c for c in sector_df.columns if c != sec_col][0]
events = pd.read_parquet('data/fullscale/sp500_news_events.parquet', columns=['ticker'])
valid_tickers = sorted(set(prices.columns) & set(events['ticker'].unique()) & set(sector_df[tic_col]))
prices = prices[valid_tickers]
returns = prices.pct_change(); returns.iloc[0] = 0
all_dates = prices.index
num_days = len(all_dates); num_stocks = len(valid_tickers)
print(f'[axes] {num_stocks} tickers × {num_days} days')

# ── load OHLCV + adjusted OHLC ───────────────────────────────────
ohlcv = pd.read_parquet('data/reference/sp500_5y_ohlcv.parquet')
adj_ohlc = pd.read_parquet('data/reference/sp500_5y_adj_ohlc.parquet')
print(f'[load] ohlcv {ohlcv.shape}, adj_ohlc {adj_ohlc.shape}')

# Reshape to wide (date, ticker) matrices aligned to prices
def to_wide(long_df, col, tickers, dates):
    w = long_df[col].unstack(level='ticker')
    w.index = pd.to_datetime(w.index).tz_localize(None) if w.index.tz is not None else pd.to_datetime(w.index)
    return w.reindex(index=dates, columns=tickers)

volume    = to_wide(ohlcv,    'Volume',    valid_tickers, all_dates)
adj_close = to_wide(adj_ohlc, 'adj_close', valid_tickers, all_dates)
adj_high  = to_wide(adj_ohlc, 'adj_high',  valid_tickers, all_dates)
adj_low   = to_wide(adj_ohlc, 'adj_low',   valid_tickers, all_dates)

print(f'[wide] volume {volume.shape}, NaN rate {volume.isna().mean().mean():.3%}')
print(f'[wide] adj_close NaN rate {adj_close.isna().mean().mean():.3%}')

# Sanity: volume should have zero NaN on valid trading days
nan_by_day = volume.isna().mean(axis=1)
print(f'[wide] days with any volume NaN: {(nan_by_day > 0).sum()}')

# ── compute features ────────────────────────────────────────────
def shift1(df): return df.shift(1)

# 1. mom12m — long-horizon skip-month momentum
mom12m = (prices.shift(22) / prices.shift(252) - 1)
# note: per Alpha158 / Jegadeesh-Titman, skip the most-recent month

# 2. maxret — highest daily return in trailing 21d
maxret = returns.rolling(21).max().shift(1)

# 3. dolvol — log average dollar volume over 63d
dollar_volume = prices * volume
dolvol = np.log(dollar_volume.rolling(63).mean() + 1).shift(1)

# 4. CORR5 — rolling 5d corr between close and log(volume+1)
log_vol = np.log(volume + 1)
# pandas' rolling.corr with another df works pairwise aligned
corr5_records = {}
for t in valid_tickers:
    if t not in volume.columns: continue
    s_close = prices[t]; s_lv = log_vol[t]
    corr5_records[t] = s_close.rolling(5).corr(s_lv)
corr5 = pd.DataFrame(corr5_records).reindex(columns=valid_tickers).shift(1)

# 5. RSV5 — where does today's close sit in 5d High-Low range
rsv_numer = adj_close - adj_low.rolling(5).min()
rsv_denom = adj_high.rolling(5).max() - adj_low.rolling(5).min()
rsv5 = (rsv_numer / rsv_denom.replace(0, np.nan)).shift(1)

# ── per-feature diagnostics ─────────────────────────────────────
FEATURES = {
    'mom12m': mom12m,
    'maxret': maxret,
    'dolvol': dolvol,
    'CORR5':  corr5,
    'RSV5':   rsv5,
}

diag_rows = []
for name, df in FEATURES.items():
    vals = df.values
    finite = vals[np.isfinite(vals)]
    diag_rows.append(dict(
        feature=name, shape=str(df.shape),
        nan_rate=float(np.mean(~np.isfinite(vals))),
        mean=float(np.mean(finite)) if len(finite) else np.nan,
        std=float(np.std(finite))  if len(finite) else np.nan,
        p01=float(np.percentile(finite, 1))  if len(finite) else np.nan,
        p50=float(np.percentile(finite, 50)) if len(finite) else np.nan,
        p99=float(np.percentile(finite, 99)) if len(finite) else np.nan,
    ))
pd.DataFrame(diag_rows).to_csv('data/reference/phase5_feature_diag.csv', index=False)
print('\nPer-feature diagnostics:')
print(pd.DataFrame(diag_rows).round(5).to_string(index=False))

# ── stack into tensor + NaN → 0 (pipeline convention) ───────────
feature_order = ['mom12m', 'maxret', 'dolvol', 'CORR5', 'RSV5']
arr = np.stack([FEATURES[n].values for n in feature_order], axis=-1).astype(np.float32)
print(f'\n[stack] new features shape {arr.shape}, pre-fill NaN rate {np.mean(~np.isfinite(arr)):.3%}')

arr = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
np.save('data/reference/sp500_5y_phase5_features.npy', arr)

meta = {
    'shape': list(arr.shape),
    'feature_order': feature_order,
    'num_features': len(feature_order),
    'dates_range': [str(all_dates[0].date()), str(all_dates[-1].date())],
    'num_tickers': num_stocks,
    'formulas': {
        'mom12m': 'prices.shift(22) / prices.shift(252) - 1   # Jegadeesh-Titman skip-month momentum',
        'maxret': 'returns.rolling(21).max().shift(1)         # 21d trailing max daily return',
        'dolvol': 'log(mean(close*volume, 63d) + 1).shift(1)   # log 63d avg dollar volume',
        'CORR5':  'rolling_corr(close, log(volume+1), 5).shift(1)',
        'RSV5':   '(adj_close - Min(adj_low, 5d)) / (Max(adj_high, 5d) - Min(adj_low, 5d))).shift(1)',
    },
    'data_sources': {
        'close': 'data/reference/sp500_5y_prices.csv (EODHD)',
        'ohlcv': 'data/reference/sp500_5y_ohlcv.parquet (yfinance)',
        'adj_ohlc': 'data/reference/sp500_5y_adj_ohlc.parquet (yfinance, adjusted via adj_close/close factor)',
    },
    'alignment_ref': 'data/reference/ohlcv_alignment_report.csv — 500/500 tickers return-corr > 0.9999 vs EODHD',
    'post_processing': 'NaN filled with 0.0 (matches existing pipeline convention)',
    'note_on_normalization': 'Cross-sectional normalization NOT applied here — deferred to training per Diag 1/1b finding that it is regime-dependent',
}
with open('data/reference/sp500_5y_phase5_features_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print(f'\nSaved:')
print(f'  data/reference/sp500_5y_phase5_features.npy       {arr.shape}')
print(f'  data/reference/sp500_5y_phase5_features_meta.json')
print(f'  data/reference/phase5_feature_diag.csv')

# ── PC loading check — are new features orthogonal to existing 3 PCs? ──
print('\n=== PC loading check: new features on existing 9-dim PC basis ===')
# Re-compute existing 9-dim features + their PC basis (same as Diag 3)
pf = {}
for w in [5, 10, 21]:
    pf[f'ret_mean_{w}d']  = returns.rolling(w).mean().shift(1)
    pf[f'ret_std_{w}d']   = returns.rolling(w).std().shift(1)
    pf[f'momentum_{w}d']  = (prices.shift(1) / prices.shift(1 + w) - 1)
OLD_FN = ['ret_mean_5d','ret_std_5d','momentum_5d',
          'ret_mean_10d','ret_std_10d','momentum_10d',
          'ret_mean_21d','ret_std_21d','momentum_21d']
old_arr = np.stack([pf[n].values for n in OLD_FN], axis=-1).astype(np.float32)
old_arr = np.nan_to_num(old_arr, 0.0)

# Build valid mask (non-NaN labels, same rule as pipeline)
fwd_ret = prices.shift(-21) / prices - 1
valid = ~fwd_ret.isna().values  # (D, S)

# Combined 14-dim matrix, compute cross-sectional correlations per day
combined = np.concatenate([old_arr, arr], axis=-1)  # (D, S, 14)
ALL_FN = OLD_FN + feature_order

corr_sum = np.zeros((14, 14)); n_days = 0
for d in range(num_days):
    if valid[d].sum() < 30: continue
    mat = combined[d, valid[d], :]
    if np.std(mat, axis=0).min() < 1e-10: continue
    c = np.corrcoef(mat.T)
    if not np.isnan(c).any():
        corr_sum += c; n_days += 1
collin = corr_sum / n_days
coll_df = pd.DataFrame(collin, index=ALL_FN, columns=ALL_FN)
print(f'14x14 cross-sectional correlation (averaged over {n_days} days):')
print(coll_df.round(2).to_string())
coll_df.to_csv('data/reference/phase5_feature_14x14_collinearity.csv')

# eigendecompose 14x14 for effective rank
eigvals_14 = np.linalg.eigvalsh(collin)[::-1]
cum = np.cumsum(eigvals_14) / eigvals_14.sum()
print(f'\n14-dim eigenvalues (descending):')
for i, (e, c) in enumerate(zip(eigvals_14, cum)):
    print(f'  λ_{i+1} = {e:.3f} ({100*e/14:.1f}% var, cum {100*c:.1f}%)')
k95 = int(np.argmax(cum >= 0.95)) + 1
k90 = int(np.argmax(cum >= 0.90)) + 1
print(f'k for 90% var: {k90}; k for 95% var: {k95} (prior 9-dim: k90=4, k95=4)')

# For each new feature, project onto old 9-dim PC basis and measure residual
eigvals_9, eigvecs_9 = np.linalg.eigh(collin[:9, :9])
eigvals_9 = eigvals_9[::-1]; eigvecs_9 = eigvecs_9[:, ::-1]
print('\nEach new feature — its corr with old 9-dim PCs (sign-agnostic, top-3 PCs):')
header = f'  {"feature":<10s}  {"|corr PC1|":>10s}  {"|corr PC2|":>10s}  {"|corr PC3|":>10s}  {"|corr PC4+|":>12s}'
print(header)
for i, fn in enumerate(feature_order):
    col = i + 9  # position in 14x14
    # Project: load_k = (corr with feature_fn) · eigvec_k, but we have already got
    # the 14x14 corr matrix; the corr of the new feature with each old feature is
    # in row `col`. Its coordinate on PC_k (in old 9-feature space) is:
    #     proj_k = (corr_with_old_features) · eigvec_k
    old_corrs = collin[col, :9]
    projs = np.abs(eigvecs_9.T @ old_corrs)
    pc_rest = float(np.sqrt(max(0.0, 1 - np.sum(projs**2))))
    print(f'  {fn:<10s}  {projs[0]:>10.3f}  {projs[1]:>10.3f}  {projs[2]:>10.3f}  {pc_rest:>12.3f}')

print(f'\n[done] total {time.time()-t0:.1f}s')
