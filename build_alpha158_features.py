#!/usr/bin/env python
"""Build Alpha158 features on our OHLCV data, faithfully matching qlib's definitions.

Source: qlib.contrib.data.loader.Alpha158DL.get_feature_config (default config).
Reference: https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py

Exactly 158 features: 9 KBAR + 4 PRICE (window=[0], feature=OPEN/HIGH/LOW/VWAP) +
145 ROLLING (29 operator types × 5 windows [5,10,20,30,60]).

Output:
  data/reference/sp500_5y_alpha158_features.npy          — (num_days, num_tickers, 158)
  data/reference/sp500_5y_alpha158_features_meta.json    — feature names + QA log
  data/reference/sp500_5y_alpha158_qa.csv                — per-factor NaN/coverage/stats

Note on $vwap: our yfinance data has no true VWAP. We use the typical price
(high+low+close)/3 as proxy — standard substitute when tick data unavailable.
"""
import argparse
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import numpy as np
import pandas as pd

os.chdir('/Users/heruixi/Desktop/GNN-Testing')

# ──────────────────────────── qlib expression ops ────────────────────────────
# Each op takes DataFrame(s) of shape (num_days, num_tickers) and returns same shape.
# Rolling ops operate ALONG TIME AXIS (axis=0) per stock (per column).

def Ref(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Lag by d days. qlib: Ref($close, 5) = close 5 days ago."""
    return x.shift(d)


def Mean(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(window=d, min_periods=max(1, d // 2)).mean()


def Std(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(window=d, min_periods=max(2, d // 2)).std()


def Sum(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(window=d, min_periods=max(1, d // 2)).sum()


def Max(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(window=d, min_periods=max(1, d // 2)).max()


def Min(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(window=d, min_periods=max(1, d // 2)).min()


def Quantile(x: pd.DataFrame, d: int, q: float) -> pd.DataFrame:
    return x.rolling(window=d, min_periods=max(1, d // 2)).quantile(q)


def Rank(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Rolling rank (percentile) of current value within past d window."""
    return x.rolling(window=d, min_periods=max(1, d // 2)).apply(
        lambda s: (s.rank(pct=True).iloc[-1] if len(s) > 0 else np.nan), raw=False
    )


def _slope_1d(y: np.ndarray) -> float:
    """OLS slope of y on integer x. Robust to short series."""
    n = len(y)
    if n < 2 or not np.all(np.isfinite(y)):
        return np.nan
    x = np.arange(n, dtype=np.float64)
    vx = x.var()
    if vx < 1e-12:
        return np.nan
    return np.cov(x, y, ddof=0)[0, 1] / vx


def Slope(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(window=d, min_periods=max(2, d // 2)).apply(_slope_1d, raw=True)


def _rsquare_1d(y: np.ndarray) -> float:
    n = len(y)
    if n < 2 or not np.all(np.isfinite(y)):
        return np.nan
    x = np.arange(n, dtype=np.float64)
    vx, vy = x.var(), y.var()
    if vx < 1e-12 or vy < 1e-12:
        return np.nan
    c = np.cov(x, y, ddof=0)[0, 1]
    return (c * c) / (vx * vy)


def Rsquare(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(window=d, min_periods=max(2, d // 2)).apply(_rsquare_1d, raw=True)


def _resi_1d(y: np.ndarray) -> float:
    """Last residual of OLS fit y ~ a + b*x on past d points."""
    n = len(y)
    if n < 2 or not np.all(np.isfinite(y)):
        return np.nan
    x = np.arange(n, dtype=np.float64)
    vx = x.var()
    if vx < 1e-12:
        return np.nan
    b = np.cov(x, y, ddof=0)[0, 1] / vx
    a = y.mean() - b * x.mean()
    return float(y[-1] - (a + b * x[-1]))


def Resi(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(window=d, min_periods=max(2, d // 2)).apply(_resi_1d, raw=True)


def IdxMax(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Index of max within past d window (0 = d-1 days ago, d-1 = today)."""
    return x.rolling(window=d, min_periods=max(1, d // 2)).apply(
        lambda s: float(np.nanargmax(s)) if len(s) > 0 and np.any(np.isfinite(s)) else np.nan, raw=True
    )


def IdxMin(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(window=d, min_periods=max(1, d // 2)).apply(
        lambda s: float(np.nanargmin(s)) if len(s) > 0 and np.any(np.isfinite(s)) else np.nan, raw=True
    )


def _pairwise_rolling(a: pd.DataFrame, b: pd.DataFrame, d: int, fn) -> pd.DataFrame:
    """Apply per-column rolling window combining a and b."""
    out = pd.DataFrame(index=a.index, columns=a.columns, dtype=np.float64)
    for col in a.columns:
        sa = a[col].values
        sb = b[col].values
        vals = np.full(len(sa), np.nan)
        for t in range(d - 1, len(sa)):
            wa = sa[t - d + 1: t + 1]
            wb = sb[t - d + 1: t + 1]
            vals[t] = fn(wa, wb)
        out[col] = vals
    return out


def _corr_1d(wa: np.ndarray, wb: np.ndarray) -> float:
    if not (np.all(np.isfinite(wa)) and np.all(np.isfinite(wb))):
        return np.nan
    sa, sb = wa.std(), wb.std()
    if sa < 1e-12 or sb < 1e-12:
        return np.nan
    return float(np.corrcoef(wa, wb)[0, 1])


def Corr(a: pd.DataFrame, b: pd.DataFrame, d: int) -> pd.DataFrame:
    return _pairwise_rolling(a, b, d, _corr_1d)


def Log(x: pd.DataFrame) -> pd.DataFrame:
    return np.log(x)


def Abs(x: pd.DataFrame) -> pd.DataFrame:
    return x.abs()


def Greater(a, b):
    """Element-wise max. Accepts DataFrame/Series/scalar."""
    if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
        return pd.DataFrame(np.maximum(
            a.values if isinstance(a, pd.DataFrame) else a,
            b.values if isinstance(b, pd.DataFrame) else b,
        ), index=(a.index if isinstance(a, pd.DataFrame) else b.index),
           columns=(a.columns if isinstance(a, pd.DataFrame) else b.columns))
    return np.maximum(a, b)


def Less(a, b):
    if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
        return pd.DataFrame(np.minimum(
            a.values if isinstance(a, pd.DataFrame) else a,
            b.values if isinstance(b, pd.DataFrame) else b,
        ), index=(a.index if isinstance(a, pd.DataFrame) else b.index),
           columns=(a.columns if isinstance(a, pd.DataFrame) else b.columns))
    return np.minimum(a, b)


# ──────────────────────────── Alpha158 factor list ────────────────────────────
# Faithful reproduction of qlib.contrib.data.loader.Alpha158DL.get_feature_config
# with default config: kbar={}, price={windows=[0], feature=OPEN/HIGH/LOW/VWAP},
# rolling={} (default windows=[5,10,20,30,60]).
ROLLING_WINDOWS = [5, 10, 20, 30, 60]
ROLLING_OPS = [
    'ROC', 'MA', 'STD', 'BETA', 'RSQR', 'RESI', 'MAX', 'LOW', 'QTLU', 'QTLD',
    'RANK', 'RSV', 'IMAX', 'IMIN', 'IMXD', 'CORR', 'CORD', 'CNTP', 'CNTN', 'CNTD',
    'SUMP', 'SUMN', 'SUMD', 'VMA', 'VSTD', 'WVMA', 'VSUMP', 'VSUMN', 'VSUMD',
]


def alpha158_expressions():
    """Return (fields, names) for the default 158-factor Alpha158 config."""
    fields, names = [], []

    # KBAR (9)
    kbar = [
        ("($close-$open)/$open", "KMID"),
        ("($high-$low)/$open", "KLEN"),
        ("($close-$open)/($high-$low+1e-12)", "KMID2"),
        ("($high-Greater($open, $close))/$open", "KUP"),
        ("($high-Greater($open, $close))/($high-$low+1e-12)", "KUP2"),
        ("(Less($open, $close)-$low)/$open", "KLOW"),
        ("(Less($open, $close)-$low)/($high-$low+1e-12)", "KLOW2"),
        ("(2*$close-$high-$low)/$open", "KSFT"),
        ("(2*$close-$high-$low)/($high-$low+1e-12)", "KSFT2"),
    ]
    for f, n in kbar:
        fields.append(f); names.append(n)

    # PRICE (4, window=[0])
    for feat in ['open', 'high', 'low', 'vwap']:
        fields.append(f"${feat}/$close")
        names.append(feat.upper() + "0")

    # ROLLING (29 ops × 5 windows = 145)
    for d in ROLLING_WINDOWS:
        fields.append(f"Ref($close, {d})/$close"); names.append(f"ROC{d}")
        fields.append(f"Mean($close, {d})/$close"); names.append(f"MA{d}")
        fields.append(f"Std($close, {d})/$close"); names.append(f"STD{d}")
        fields.append(f"Slope($close, {d})/$close"); names.append(f"BETA{d}")
        fields.append(f"Rsquare($close, {d})"); names.append(f"RSQR{d}")
        fields.append(f"Resi($close, {d})/$close"); names.append(f"RESI{d}")
        fields.append(f"Max($high, {d})/$close"); names.append(f"MAX{d}")
        fields.append(f"Min($low, {d})/$close"); names.append(f"MIN{d}")
        fields.append(f"Quantile($close, {d}, 0.8)/$close"); names.append(f"QTLU{d}")
        fields.append(f"Quantile($close, {d}, 0.2)/$close"); names.append(f"QTLD{d}")
        fields.append(f"Rank($close, {d})"); names.append(f"RANK{d}")
        fields.append(f"($close-Min($low, {d}))/(Max($high, {d})-Min($low, {d})+1e-12)"); names.append(f"RSV{d}")
        fields.append(f"IdxMax($high, {d})/{d}"); names.append(f"IMAX{d}")
        fields.append(f"IdxMin($low, {d})/{d}"); names.append(f"IMIN{d}")
        fields.append(f"(IdxMax($high, {d})-IdxMin($low, {d}))/{d}"); names.append(f"IMXD{d}")
        fields.append(f"Corr($close, Log($volume+1), {d})"); names.append(f"CORR{d}")
        fields.append(f"Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), {d})"); names.append(f"CORD{d}")
        fields.append(f"Mean($close>Ref($close, 1), {d})"); names.append(f"CNTP{d}")
        fields.append(f"Mean($close<Ref($close, 1), {d})"); names.append(f"CNTN{d}")
        fields.append(f"Mean($close>Ref($close, 1), {d})-Mean($close<Ref($close, 1), {d})"); names.append(f"CNTD{d}")
        fields.append(f"Sum(Greater($close-Ref($close, 1), 0), {d})/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)"); names.append(f"SUMP{d}")
        fields.append(f"Sum(Greater(Ref($close, 1)-$close, 0), {d})/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)"); names.append(f"SUMN{d}")
        fields.append(f"(Sum(Greater($close-Ref($close, 1), 0), {d})-Sum(Greater(Ref($close, 1)-$close, 0), {d}))/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)"); names.append(f"SUMD{d}")
        fields.append(f"Mean($volume, {d})/($volume+1e-12)"); names.append(f"VMA{d}")
        fields.append(f"Std($volume, {d})/($volume+1e-12)"); names.append(f"VSTD{d}")
        fields.append(f"Std(Abs($close/Ref($close, 1)-1)*$volume, {d})/(Mean(Abs($close/Ref($close, 1)-1)*$volume, {d})+1e-12)"); names.append(f"WVMA{d}")
        fields.append(f"Sum(Greater($volume-Ref($volume, 1), 0), {d})/(Sum(Abs($volume-Ref($volume, 1)), {d})+1e-12)"); names.append(f"VSUMP{d}")
        fields.append(f"Sum(Greater(Ref($volume, 1)-$volume, 0), {d})/(Sum(Abs($volume-Ref($volume, 1)), {d})+1e-12)"); names.append(f"VSUMN{d}")
        fields.append(f"(Sum(Greater($volume-Ref($volume, 1), 0), {d})-Sum(Greater(Ref($volume, 1)-$volume, 0), {d}))/(Sum(Abs($volume-Ref($volume, 1)), {d})+1e-12)"); names.append(f"VSUMD{d}")

    assert len(fields) == len(names) == 158, f'got {len(fields)} fields, want 158'
    return fields, names


def evaluate(expr: str, ctx: dict) -> pd.DataFrame:
    """Evaluate a qlib expression string. ctx must contain $close/$open/$high/$low/$volume/$vwap."""
    py_expr = expr
    for field in ('close', 'open', 'high', 'low', 'volume', 'vwap'):
        py_expr = py_expr.replace(f'${field}', field)
    ns = {
        'close': ctx['close'], 'open': ctx['open'], 'high': ctx['high'],
        'low': ctx['low'], 'volume': ctx['volume'], 'vwap': ctx['vwap'],
        'Ref': Ref, 'Mean': Mean, 'Std': Std, 'Sum': Sum,
        'Max': Max, 'Min': Min, 'Quantile': Quantile, 'Rank': Rank,
        'Slope': Slope, 'Rsquare': Rsquare, 'Resi': Resi,
        'IdxMax': IdxMax, 'IdxMin': IdxMin, 'Corr': Corr,
        'Log': Log, 'Abs': Abs, 'Greater': Greater, 'Less': Less,
    }
    return eval(py_expr, {'__builtins__': {}}, ns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-raw', action='store_true',
                        help='Also save pre-winsor (post-NaN/inf-fill) array to '
                             'sp500_5y_alpha158_features_raw.npy for Fold 4 leakage diagnostics.')
    args = parser.parse_args()
    save_raw = args.save_raw

    t0 = time.time()

    # Load OHLCV (yfinance) + canonical prices (EODHD) for ticker alignment
    prices = pd.read_csv('data/reference/sp500_5y_prices.csv', index_col=0, parse_dates=True)
    sector_df = pd.read_csv('data/reference/sp500_sectors.csv')
    sec_col = [c for c in sector_df.columns if 'sector' in c.lower()][0]
    tic_col = [c for c in sector_df.columns if c != sec_col][0]
    events = pd.read_parquet('data/fullscale/sp500_news_events.parquet', columns=['ticker'])
    valid_tickers = sorted(set(prices.columns) & set(events['ticker'].unique()) & set(sector_df[tic_col]))
    print(f'[data] {len(valid_tickers)} tickers, {len(prices)} days')

    # Load adjusted OHLC (split/div-adjusted yfinance) — needed so open/high/low
    # are in the same adjustment space as EODHD close. Mixing RAW open with
    # adjusted close produces spurious offsets (KMID systematically off by the
    # adjustment factor) and corrupts every K-line feature.
    adj = pd.read_parquet('data/reference/sp500_5y_adj_ohlc.parquet')
    ohlcv = pd.read_parquet('data/reference/sp500_5y_ohlcv.parquet')  # for raw volume only
    print(f'[adj_ohlc] {adj.shape}; [ohlcv raw] {ohlcv.shape}')

    def to_wide_from(df, col):
        w = df[col].unstack(level='ticker')
        w.index = pd.to_datetime(w.index).tz_localize(None) if w.index.tz is not None else pd.to_datetime(w.index)
        return w.reindex(index=prices.index, columns=valid_tickers)

    open_raw = to_wide_from(adj, 'adj_open')
    high_raw = to_wide_from(adj, 'adj_high')
    low_raw = to_wide_from(adj, 'adj_low')
    adj_close_yf = to_wide_from(adj, 'adj_close')
    volume = to_wide_from(ohlcv, 'Volume')

    # Sanity check: OHLC consistency in adjusted space
    hi_ok = ((high_raw >= np.maximum(open_raw, adj_close_yf) - 1e-6) | high_raw.isna()).all().all()
    lo_ok = ((low_raw <= np.minimum(open_raw, adj_close_yf) + 1e-6) | low_raw.isna()).all().all()
    print(f'[sanity] adj High >= max(O,C): {hi_ok}; adj Low <= min(O,C): {lo_ok}')
    if not (hi_ok and lo_ok):
        print('[WARN] adjusted OHLC inconsistency — check ticker data')

    # Close: use EODHD canonical (aligns with Plan Z features + labels). Adjusted
    # ~0.1% from yfinance adj_close — acceptable drift, KMID etc. now meaningful.
    close = prices[valid_tickers]

    # VWAP proxy: typical price in ADJUSTED space (high+low+close)/3
    vwap = (high_raw + low_raw + close) / 3.0

    ctx = {
        'close': close.astype(np.float64),
        'open': open_raw.astype(np.float64),
        'high': high_raw.astype(np.float64),
        'low': low_raw.astype(np.float64),
        'volume': volume.astype(np.float64),
        'vwap': vwap.astype(np.float64),
    }

    fields, names = alpha158_expressions()
    print(f'[alpha158] 158 factors defined')

    # Evaluate
    num_days = len(prices)
    num_stocks = len(valid_tickers)
    out = np.zeros((num_days, num_stocks, len(names)), dtype=np.float32)
    qa = []
    for i, (f, n) in enumerate(zip(fields, names)):
        ts = time.time()
        try:
            result = evaluate(f, ctx)
        except Exception as e:
            print(f'[FAIL] factor {n}: {e}')
            qa.append({'factor': n, 'expr': f, 'nan_rate': 1.0, 'mean': np.nan,
                       'std': np.nan, 'p_finite': 0.0, 'error': str(e)})
            continue
        if not isinstance(result, pd.DataFrame):
            print(f'[WARN] factor {n} returned {type(result)}, coercing')
            continue
        arr = result.reindex(columns=valid_tickers).values.astype(np.float32)
        out[:, :, i] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        nan_rate = float(np.mean(~np.isfinite(arr)))
        finite = arr[np.isfinite(arr)]
        qa.append({
            'factor': n, 'expr': f,
            'nan_rate': nan_rate,
            'mean': float(finite.mean()) if finite.size > 0 else np.nan,
            'std': float(finite.std()) if finite.size > 0 else np.nan,
            'p_finite': 1.0 - nan_rate,
            'error': '',
        })
        if (i + 1) % 20 == 0:
            print(f'  [{i + 1}/158] last factor {n}: nan={nan_rate:.3f} in {time.time() - ts:.2f}s')

    Path('data/reference').mkdir(parents=True, exist_ok=True)

    if save_raw:
        raw_path = 'data/reference/sp500_5y_alpha158_features_raw.npy'
        np.save(raw_path, out)
        print(f'[save-raw] pre-winsor array saved to {raw_path}')

    # Winsorize 1/99 percentile per feature before saving. Volume-normalized
    # factors (VMA, VSTD) divide by $volume and produce 1e15+ magnitudes on flat
    # trading days (halts, edge cases). Without winsorization, per-fold
    # StandardScaler will have mean/std dominated by a handful of outliers and
    # training will fail. Winsorization is standard in quant pipelines (Kelly-Xiu,
    # Han et al.) and preserves all rank-based info.
    print('[winsorize] clipping each feature to [p1, p99] over all valid observations')
    for i in range(out.shape[-1]):
        arr = out[:, :, i]
        valid = arr[np.isfinite(arr) & (arr != 0)]  # exclude the NaN->0 fills
        if valid.size < 100:
            continue
        lo, hi = np.percentile(valid, [1, 99])
        out[:, :, i] = np.clip(arr, lo, hi)

    np.save('data/reference/sp500_5y_alpha158_features.npy', out)
    pd.DataFrame(qa).to_csv('data/reference/sp500_5y_alpha158_qa.csv', index=False)
    meta = {
        'source': 'qlib.contrib.data.loader.Alpha158DL (default config, commit fetched 2026-04-19)',
        'shape': list(out.shape),
        'feature_order': names,
        'num_features': len(names),
        'dates_range': [str(prices.index.min().date()), str(prices.index.max().date())],
        'num_tickers': num_stocks,
        'vwap_note': 'VWAP proxied as (high+low+close)/3 (yfinance has no tick-level data)',
        'close_source': 'EODHD canonical close (data/reference/sp500_5y_prices.csv)',
        'ohlcv_source': 'yfinance data/reference/sp500_5y_ohlcv.parquet',
        'post_processing': 'NaN/inf filled with 0.0 (matches existing pipeline convention)',
    }
    with open('data/reference/sp500_5y_alpha158_features_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'[done] {time.time() - t0:.1f}s')
    print(f'  {out.shape} alpha158 features saved')
    print(f'  QA summary: median NaN rate = {np.median([r["nan_rate"] for r in qa]):.3f}, '
          f'max NaN rate = {max(r["nan_rate"] for r in qa):.3f}')
    worst = sorted(qa, key=lambda r: -r['nan_rate'])[:5]
    print('  Worst NaN factors:')
    for r in worst:
        print(f'    {r["factor"]}: nan_rate={r["nan_rate"]:.3f}')


if __name__ == '__main__':
    main()
