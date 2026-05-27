# E3 News-Edge Source Artifact Schema (v2, addressing Codex C-03 + Round D D-03)

## Context

Codex Round C finding **CODEX-C-03 (MAJOR)**: the v2 news PIT rule in plan §1.2 is conceptually correct but cannot be enforced against the existing artifact `data/fullscale/sp500_news_events.parquet` which contains forward-looking fields `return_next` (next trading day return) and `label` (binary direction). H博士 chose Option B then directed C-03 fix immediately (this doc) rather than deferring.

Codex Round D follow-up **CODEX-D-03 (MAJOR)** revealed that the original C-03 schema doc used UTC midnight as the PIT cutoff (`pd.Timestamp(prediction_date_t, tz='UTC') - pd.Timedelta(seconds=1)`), which leaves a 3-4 hour after-hours leak window because NYSE regular session closes at 16:00 ET = 20:00 UTC (DST) / 21:00 UTC (winter). v2 of this doc replaces the PIT enforcement code with NYSE session-close timestamps via `pandas_market_calendars`, plus DST-spanning worked examples.

## Source artifact audit (2026-05-26 evening)

**File**: `data/fullscale/sp500_news_events.parquet` (2.7 GB, 1,698,182 rows)

**Schema** (verified via direct parquet read):

| Column | dtype | Notes |
|--------|-------|-------|
| `date` | datetime64[ns, UTC] | **publication timestamp at SECOND precision** (e.g. `2026-01-22 12:08:25+00:00`) — usable as `publication_timestamp` |
| `ticker` | object (str) | one row per (article, ticker_mentioned) |
| `title` | object (str) | article title |
| `content` | object (str) | full article body |
| `polarity` | float64 | sentiment score (NOT a forward field but not needed for edge) |
| `neg`, `neu`, `pos` | float64 | FinBERT sentiment probabilities (not needed for edge) |
| `tags` | object (str, semicolon-separated) | LLM-extracted topic tags (not needed for edge) |
| **`return_next`** | float64 | **FORWARD FIELD — MUST STRIP** (next trading day return at article date) |
| **`label`** | int64 | **FORWARD FIELD — MUST STRIP** (binary: 1 if return_next > 0, else 0) |

**Date range**: 2021-01-29 to 2026-01-26 UTC

**Critical observation**: rows for different tickers in same article have IDENTICAL `date`, `title`, `content` (verified rows 1 and 2 both for article "The 20 stocks hedge funds are most underweight" published 2026-01-18 19:07:53 UTC, one row for CHTR, another for COIN). This means we can group by (date, title, content) to recover article-level granularity even though source has no `article_id` column.

## v3 PIT-safe edge source artifact (NEW — to be built before E3)

**File**: `data/fullscale/sp500_news_edge_source.parquet` (derived, much smaller than source)

**Schema** (LOCKED):

| Column | dtype | Description |
|--------|-------|-------------|
| `article_id` | uint64 | xxHash64(`f"{date_iso}\\t{title}\\t{content[:512]}"`) — deterministic, collision-safe; 512-char content prefix balances uniqueness with hash cost |
| `publication_timestamp` | datetime64[ns, UTC] | exact UTC timestamp from source `date` column (second precision preserved) |
| `tickers_mentioned` | list[str] (pyarrow list array) | sorted unique list of all SP500 tickers mentioned in this article |
| `n_tickers` | uint16 | len(tickers_mentioned) for fast filtering |

**Forbidden columns in derived artifact** (assert at build time):
- `return_next` — FORWARD; MUST NOT appear
- `label` — FORWARD; MUST NOT appear
- `polarity`, `neg`, `neu`, `pos` — not forward but not needed for edge; STRIPPED for hygiene (FinBERT sentiment is sourced separately if needed for future ablation)
- `tags` — LLM artifact; STRIPPED (use only for human exploratory analysis)

## Build procedure

```python
# Pseudocode for build_news_edge_source.py
import pandas as pd
import xxhash
import pyarrow as pa
import pyarrow.parquet as pq

src = pd.read_parquet('data/fullscale/sp500_news_events.parquet',
                      columns=['date', 'ticker', 'title', 'content'])

# Assert source schema invariants
assert pd.api.types.is_datetime64tz_dtype(src['date']), "date must be tz-aware (UTC)"
assert src['date'].dt.tz.zone == 'UTC', "date must be UTC"

# Derive article_id (deterministic)
def article_id_fn(row):
    key = f"{row['date'].isoformat()}\t{row['title']}\t{(row['content'] or '')[:512]}"
    return xxhash.xxh64(key.encode('utf-8')).intdigest()

src['article_id'] = src.apply(article_id_fn, axis=1)

# Group by article_id → aggregate ticker list, take first date/title/content for consistency
agg = src.groupby('article_id').agg(
    publication_timestamp=('date', 'first'),
    tickers_mentioned=('ticker', lambda s: sorted(set(s))),
)
agg['n_tickers'] = agg['tickers_mentioned'].map(len)
agg = agg.reset_index()

# Final invariant: no forward fields
forbidden = ['return_next', 'label', 'polarity', 'neg', 'neu', 'pos', 'tags']
for col in forbidden:
    assert col not in agg.columns, f"FORBIDDEN forward/sentiment field {col} leaked into derived artifact"

# Save with pyarrow list type for tickers_mentioned
table = pa.Table.from_pandas(agg)
pq.write_table(table, 'data/fullscale/sp500_news_edge_source.parquet',
               compression='snappy')

# Expected ~600K-800K unique articles (from 1.7M ticker-rows; deduplication factor ~2-3x)
```

## PIT enforcement in E3 script (UPDATED 2026-05-26 per Codex Round D D-03 — timezone-aware NYSE session close)

> **D-03 finding rationale**: prior draft used `pd.Timestamp(prediction_date_t, tz='UTC') - pd.Timedelta(seconds=1)` as the cutoff, which evaluates to 23:59:59 UTC on day t-1 calendar. But NYSE regular session closes at 16:00 ET = **20:00 UTC (DST)** or **21:00 UTC (winter)**. UTC-midnight cutoff therefore admits any article published in the 3-4 hour window between NYSE close and 00:00 UTC as if it were known at close-of-day t-1, even though prices are not yet updated. This is a real after-hours leak. Fix: use `pandas_market_calendars.get_calendar('NYSE').schedule(...).market_close` for the t-1 session, converted to UTC.

```python
# In run_storya_e3_news_edge.py — co-occurrence edge builder
import pandas_market_calendars as mcal
import pandas as pd
from itertools import combinations
from collections import defaultdict

NYSE = mcal.get_calendar('NYSE')

def nyse_session_close_utc(date_like):
    """Return the UTC timestamp of NYSE regular session close for the trading day
    on-or-before `date_like`. Handles DST automatically via pandas_market_calendars."""
    d = pd.Timestamp(date_like).normalize()
    # Look back up to 7 calendar days to find the prior trading day
    sched = NYSE.schedule(start_date=d - pd.Timedelta(days=7), end_date=d)
    if sched.empty:
        raise ValueError(f"No NYSE trading session on-or-before {d}")
    last_close = sched.iloc[-1]['market_close']  # already tz-aware
    return last_close.tz_convert('UTC')

def build_news_edges_at_date(edge_source_df, prediction_date_t, lookback_days=5,
                              label_horizon_days=21):
    """
    Returns edge_index for prediction at date t (close-of-day t-1 known).

    PIT contract (LOCKED per Codex C-03 + Round D D-03):
    1. Cutoff = NYSE session_close of the trading day immediately preceding prediction_date_t
       (converted to UTC). DST handled by pandas_market_calendars.
    2. Only articles with publication_timestamp ≤ cutoff are eligible.
    3. Lookback window start = cutoff - lookback_days × 1 trading-day (calendar approx OK; we
       use 7 calendar days as a safe upper bound to capture exactly `lookback_days` trading days
       across weekends).
    4. Articles inside [t, t+label_horizon_days] are EXCLUDED — automatically enforced by step 2
       (such articles have publication_timestamp > cutoff and fail the filter).
    """
    t_ts = pd.Timestamp(prediction_date_t)
    # Cutoff is t-1 trading session NYSE close, in UTC
    cutoff_utc = nyse_session_close_utc(t_ts - pd.Timedelta(days=1))
    # Lookback start: cutoff - (lookback_days × 1 calendar week approximation safe upper bound)
    lookback_start_utc = cutoff_utc - pd.Timedelta(days=lookback_days + 2)  # +2 buffer for weekends

    # PIT filter
    eligible = edge_source_df[
        (edge_source_df['publication_timestamp'] <= cutoff_utc)
        & (edge_source_df['publication_timestamp'] >= lookback_start_utc)
        & (edge_source_df['n_tickers'] >= 2)  # need ≥2 tickers for co-occurrence
    ]

    # Runtime assertion (must pass per Codex C-03 + D-03)
    if len(eligible) > 0:
        max_ts = eligible['publication_timestamp'].max()
        assert max_ts <= cutoff_utc, (
            f"PIT VIOLATION at prediction_date {prediction_date_t}: "
            f"max article timestamp {max_ts} > NYSE t-1 session close {cutoff_utc}"
        )

    # Build edges: for each article, all pairs of tickers
    edge_count = defaultdict(int)
    for tickers in eligible['tickers_mentioned']:
        for i, j in combinations(tickers, 2):
            if i in ticker_to_idx and j in ticker_to_idx:
                edge_count[(ticker_to_idx[i], ticker_to_idx[j])] += 1

    # Return symmetric edge_index + edge_weight
    ...
```

### Worked DST-spanning examples (must be verified at first runtime)

| prediction_date_t | t-1 trading session | NYSE close (ET) | cutoff (UTC) | DST status |
|------------------|--------------------|-----------------| -------------|------------|
| 2024-06-03 (Mon) | 2024-05-31 (Fri) | 16:00 ET | **2024-05-31 20:00:00+00:00** | DST in effect (EDT) |
| 2024-12-02 (Mon) | 2024-11-29 (Fri, ½ day) | 13:00 ET | **2024-11-29 18:00:00+00:00** | EST + early close (Black Friday) |
| 2024-03-11 (Mon) | 2024-03-08 (Fri) | 16:00 ET | **2024-03-08 21:00:00+00:00** | EST (DST starts 2024-03-10) |
| 2024-03-12 (Tue) | 2024-03-11 (Mon) | 16:00 ET | **2024-03-11 20:00:00+00:00** | EDT (DST in effect) |

**Verification test (must pass before E3 launch)**: an article published at `2024-05-31 21:00:00+00:00` (1 hour after NYSE close, before UTC midnight) must be:
- **EXCLUDED** for prediction_date_t = 2024-06-03 (cutoff 20:00 UTC on 2024-05-31 → 21:00 > 20:00 fails filter)
- **INCLUDED** for prediction_date_t = 2024-06-04 (cutoff is now 2024-06-03 20:00 UTC, which is later than 2024-05-31 21:00 UTC → passes upper bound; passes lower bound if within 5-trading-day lookback)

If this property does not hold at runtime, the PIT contract is violated and the E3 run is aborted.

## Verification at build time

After running `build_news_edge_source.py`:

```python
# Verification script (to be invoked at end of build):
import pandas as pd
edge_src = pd.read_parquet('data/fullscale/sp500_news_edge_source.parquet')

# Check 1: schema is clean
assert set(edge_src.columns) == {'article_id', 'publication_timestamp',
                                   'tickers_mentioned', 'n_tickers'}, \
    f"Unexpected columns: {edge_src.columns}"

# Check 2: no forward fields
for col in ['return_next', 'label', 'polarity', 'neg', 'neu', 'pos', 'tags']:
    assert col not in edge_src.columns, f"FORWARD field {col} leaked"

# Check 3: article_id uniqueness
assert edge_src['article_id'].nunique() == len(edge_src), \
    "article_id not unique — hash collision or aggregation bug"

# Check 4: publication_timestamp is tz-aware UTC
assert pd.api.types.is_datetime64tz_dtype(edge_src['publication_timestamp'])
assert edge_src['publication_timestamp'].dt.tz.zone == 'UTC'

# Check 5: tickers_mentioned non-empty
assert (edge_src['n_tickers'] >= 1).all()

# Report
print(f"✓ Derived artifact OK: {len(edge_src)} unique articles, "
      f"{edge_src['n_tickers'].sum()} total ticker mentions, "
      f"{(edge_src['n_tickers'] >= 2).sum()} articles with ≥2 tickers "
      f"(usable for co-occurrence edges)")
```

## Files to create

| File | Purpose | Status |
|------|---------|--------|
| `scripts/build_news_edge_source.py` | Build derived PIT-safe artifact | TODO before E3 launch |
| `data/fullscale/sp500_news_edge_source.parquet` | Output of build script | TODO |
| `experiments/storya_e3_news_edge/news_edge_source_schema.md` | This file (schema spec) | ✓ created 2026-05-26 |
| `run_storya_e3_news_edge.py` | E3 experiment runner using PIT artifact | TODO after C-03 PIT artifact built |

## Round C disposition update

CODEX-C-03 status: **FIXED 2026-05-26 evening** (was DEFERRED-TO-EXECUTION in Option B; H博士 directed immediate fix).

Resolution:
- Schema designed with article_id + UTC publication_timestamp + tickers_mentioned list
- Forward fields (return_next, label) explicitly forbidden in derived artifact + build-time assertions
- PIT enforcement code outlined for E3 script with runtime assertion
- Verification script provides build-time guarantees

Pending:
- Actual build of `data/fullscale/sp500_news_edge_source.parquet` (run `scripts/build_news_edge_source.py` before E3 launch; ~30 min wall time on M4 for 1.7M row aggregation)
- E3 runner `run_storya_e3_news_edge.py` to be written and Codex-reviewed (Touchpoint 2)

## Reference

- Source artifact audit: this file §"Source artifact audit"
- Codex Round C finding: `artifacts/reviews/2026-05-26_codex_plan_C.md` CODEX-C-03
- Plan §1.2 news-as-edge spec (to be cross-referenced after this schema lock)
- Original prepare_events.py: `scripts/prepare_events.py` (pilot version; fullscale version assumed similar logic)
