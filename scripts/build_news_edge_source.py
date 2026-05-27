#!/usr/bin/env python
"""build_news_edge_source.py — Build PIT-safe news co-occurrence source artifact.

Implements the schema spec at:
  experiments/storya_e3_news_edge/news_edge_source_schema.md
(v2, addresses Codex C-03 + D-03; LOCKED 2026-05-26 evening.)

Input:
  data/fullscale/sp500_news_events.parquet
    1,698,182 (article, ticker_mention) rows; columns date, ticker, title, content,
    plus FORWARD fields (return_next, label) and sentiment fields that MUST be stripped.

Output:
  data/fullscale/sp500_news_edge_source.parquet
    ~600-800K unique articles, one row per article, with:
      - article_id (uint64, xxhash64 of iso_date + title + content[:512])
      - publication_timestamp (datetime64[ns, UTC], second precision preserved)
      - tickers_mentioned (list[str], sorted unique SP500 tickers)
      - n_tickers (uint16)

Hard invariants (asserted at runtime — failures abort the build):
  1. Source date column tz-aware UTC.
  2. No FORBIDDEN forward/sentiment columns in output.
  3. article_id uniquely identifies each output row.
  4. Output publication_timestamp tz-aware UTC.
  5. n_tickers >= 1 for every output row.

Usage (from project root):
  python scripts/build_news_edge_source.py
  python scripts/build_news_edge_source.py --src <alt_src> --dst <alt_dst>

Required deps: pandas, pyarrow, xxhash.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xxhash


SRC_DEFAULT = "data/fullscale/sp500_news_events.parquet"
DST_DEFAULT = "data/fullscale/sp500_news_edge_source.parquet"
SECTORS_CSV = "data/reference/sp500_sectors.csv"

FORBIDDEN_COLS = [
    "return_next", "label",
    "polarity", "neg", "neu", "pos", "tags",
]


def load_universe(sectors_csv: str) -> set[str]:
    sectors = pd.read_csv(sectors_csv)
    # sp500_sectors.csv uses 'Symbol' column; accept 'ticker' as an alias for forward compat.
    if "Symbol" in sectors.columns:
        col = "Symbol"
    elif "ticker" in sectors.columns:
        col = "ticker"
    else:
        raise ValueError(f"{sectors_csv} missing 'Symbol' (or 'ticker') column; "
                         f"got: {list(sectors.columns)}")
    universe = set(sectors[col].astype(str).unique())
    return universe


def assert_source_schema(src: pd.DataFrame) -> None:
    if "date" not in src.columns:
        raise ValueError("source missing 'date' column")
    if not isinstance(src["date"].dtype, pd.DatetimeTZDtype):
        raise TypeError("source 'date' must be tz-aware datetime")
    tz = src["date"].dt.tz
    if str(tz) != "UTC":
        raise ValueError(f"source 'date' tz must be UTC, got {tz}")


def hash_article_ids(dates: pd.Series, titles: pd.Series, contents: pd.Series) -> np.ndarray:
    """Hash each (date_iso, title, content[:512]) triple to uint64 via xxh64.

    Matches schema spec key = f"{date.isoformat()}\\t{title}\\t{content[:512]}".
    Uses list comp (not Series.apply) to keep the Python loop tight; xxh64 itself is C-speed.
    """
    n = len(dates)
    out = np.empty(n, dtype=np.uint64)
    # `Timestamp.isoformat()` on a UTC-aware ts returns e.g. '2026-01-22T12:08:25+00:00'.
    # We zip Series → Python objects (iterating pd.Series yields elements in C-speed).
    for i, (d, t, c) in enumerate(zip(dates, titles, contents)):
        # d is pd.Timestamp (UTC). isoformat preserves whatever sub-second precision exists.
        key = f"{d.isoformat()}\t{t}\t{c}".encode("utf-8")
        out[i] = xxhash.xxh64(key).intdigest()
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", default=SRC_DEFAULT, help=f"source parquet (default: {SRC_DEFAULT})")
    p.add_argument("--dst", default=DST_DEFAULT, help=f"output parquet (default: {DST_DEFAULT})")
    p.add_argument("--sectors", default=SECTORS_CSV, help=f"SP500 universe csv (default: {SECTORS_CSV})")
    p.add_argument("--force", action="store_true", help="overwrite existing destination")
    args = p.parse_args()

    if not os.path.exists(args.src):
        print(f"ERROR: source not found: {args.src} (cwd={os.getcwd()})", file=sys.stderr)
        return 2
    if not os.path.exists(args.sectors):
        print(f"ERROR: sectors not found: {args.sectors}", file=sys.stderr)
        return 2
    if os.path.exists(args.dst) and not args.force:
        print(f"WARN: destination exists: {args.dst} — pass --force to overwrite, exiting.",
              file=sys.stderr)
        return 1

    print(f"[build] cwd={os.getcwd()}")
    print(f"[build]  src={args.src}")
    print(f"[build]  dst={args.dst}")

    # ── Step 1: load SP500 universe ──
    universe = load_universe(args.sectors)
    print(f"[build] SP500 universe: {len(universe)} tickers")

    # ── Step 2: load source (only needed columns; reduces RAM) ──
    t0 = time.time()
    print(f"[build] reading source parquet ...")
    src = pd.read_parquet(args.src, columns=["date", "ticker", "title", "content"])
    print(f"[build]   loaded {len(src):,} ticker-rows in {time.time()-t0:.1f}s")

    assert_source_schema(src)
    print(f"[build]   source schema OK (date is tz-aware UTC)")

    # ── Step 3: filter to SP500 universe ──
    n_before = len(src)
    src = src[src["ticker"].astype(str).isin(universe)].reset_index(drop=True)
    print(f"[build] SP500 filter: {n_before:,} → {len(src):,} ticker-rows "
          f"(dropped {n_before-len(src):,} non-SP500 mentions)")

    if len(src) == 0:
        print("ERROR: no SP500 ticker-rows remain after filter — check universe", file=sys.stderr)
        return 3

    # ── Step 4: derive article_id (xxh64 of iso_date + title + content[:512]) ──
    t0 = time.time()
    print(f"[build] hashing article_id (xxh64) ...")
    titles = src["title"].fillna("").astype(str)
    contents = src["content"].fillna("").astype(str).str.slice(0, 512)
    src["article_id"] = hash_article_ids(src["date"], titles, contents)
    print(f"[build]   hashed {len(src):,} rows in {time.time()-t0:.1f}s "
          f"({len(src) / max(time.time()-t0, 1e-9):,.0f} rows/s)")

    # ── Step 5: aggregate by article_id → unique-article rows ──
    # CODEX-CR-E3BUILD-A-03 fix: groupby('first') is order-dependent. Even though the
    # article_id hash INCLUDES the date so each article_id should have invariant date,
    # we sort by ('article_id', 'date') before groupby to make the contract explicit and
    # robust to any future hash-key change.
    t0 = time.time()
    print(f"[build] sorting + aggregating by article_id ...")
    src = src.sort_values(["article_id", "date"], kind="mergesort").reset_index(drop=True)
    agg = (
        src.groupby("article_id", sort=False)
           .agg(
               publication_timestamp=("date", "first"),
               tickers_mentioned=("ticker", lambda s: sorted(set(str(x) for x in s))),
           )
           .reset_index()
    )
    # Defense in depth: assert per-article_id date uniqueness post-groupby. If this
    # fires it means our hash key did NOT cover all timestamp-distinguishing info.
    if not (src.groupby("article_id")["date"].nunique() == 1).all():
        bad = src.groupby("article_id")["date"].nunique()
        bad = bad[bad > 1]
        raise AssertionError(
            f"PIT-invariant violation: {len(bad)} article_ids have multiple distinct dates. "
            f"This means the hash key does not uniquely identify a publication time. "
            f"First 3: {bad.head(3).to_dict()}"
        )
    agg["n_tickers"] = agg["tickers_mentioned"].map(len).astype("uint16")
    print(f"[build]   aggregated in {time.time()-t0:.1f}s")

    # ── Step 6: drop zero-ticker articles (defensive; should be 0 after SP500 filter) ──
    n_before = len(agg)
    agg = agg[agg["n_tickers"] >= 1].reset_index(drop=True)
    dropped = n_before - len(agg)
    if dropped:
        print(f"[build]   dropped {dropped} zero-ticker articles (defensive)")

    print(f"[build] unique articles: {len(agg):,} "
          f"(dedup factor: {len(src) / max(len(agg),1):.2f}× ticker-rows per article)")
    n_ge2 = int((agg["n_tickers"] >= 2).sum())
    print(f"[build]   articles with ≥2 SP500 tickers (usable for co-occurrence): {n_ge2:,}")
    print(f"[build]   n_tickers distribution: "
          f"min={agg['n_tickers'].min()}  median={int(agg['n_tickers'].median())}  "
          f"mean={agg['n_tickers'].mean():.1f}  max={agg['n_tickers'].max()}")

    # ── Step 7: invariants ──
    for col in FORBIDDEN_COLS:
        if col in agg.columns:
            raise AssertionError(
                f"FORBIDDEN forward/sentiment field '{col}' leaked into derived artifact "
                f"— bug in build script."
            )
    if agg["article_id"].nunique() != len(agg):
        raise AssertionError(
            f"article_id collision: {agg['article_id'].nunique():,} unique vs {len(agg):,} rows"
        )
    if not isinstance(agg["publication_timestamp"].dtype, pd.DatetimeTZDtype):
        raise AssertionError("publication_timestamp lost its tz-aware dtype during groupby")
    if str(agg["publication_timestamp"].dt.tz) != "UTC":
        raise AssertionError(f"publication_timestamp tz = {agg['publication_timestamp'].dt.tz}, expected UTC")
    if not (agg["n_tickers"] >= 1).all():
        raise AssertionError("n_tickers < 1 row present after filter")

    # ── Step 8: lock column order + dtypes + write ──
    agg = agg[["article_id", "publication_timestamp", "tickers_mentioned", "n_tickers"]]
    agg["article_id"] = agg["article_id"].astype("uint64")
    agg["n_tickers"] = agg["n_tickers"].astype("uint16")

    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    print(f"[build] writing {args.dst} ...")
    t0 = time.time()
    table = pa.Table.from_pandas(agg, preserve_index=False)
    pq.write_table(table, args.dst, compression="snappy")
    sz_mb = os.path.getsize(args.dst) / 1e6
    print(f"[build]   wrote {sz_mb:.1f} MB in {time.time()-t0:.1f}s")

    # ── Step 9: readback verification ──
    # CODEX-CR-E3BUILD-A-05 fix: verify locked dtype contract for article_id (uint64),
    # n_tickers (uint16), and parquet schema for tickers_mentioned (list<string>).
    print(f"[build] verifying readback ...")
    rb = pd.read_parquet(args.dst)
    expected = {"article_id", "publication_timestamp", "tickers_mentioned", "n_tickers"}
    if set(rb.columns) != expected:
        raise AssertionError(f"readback columns {set(rb.columns)} != expected {expected}")
    for col in FORBIDDEN_COLS:
        if col in rb.columns:
            raise AssertionError(f"readback has FORBIDDEN field '{col}'")
    if rb["article_id"].nunique() != len(rb):
        raise AssertionError("readback article_id collision")
    if not isinstance(rb["publication_timestamp"].dtype, pd.DatetimeTZDtype):
        raise AssertionError("readback publication_timestamp lost tz")
    if str(rb["publication_timestamp"].dt.tz) != "UTC":
        raise AssertionError(f"readback tz = {rb['publication_timestamp'].dt.tz}, expected UTC")
    if not (rb["n_tickers"] >= 1).all():
        raise AssertionError("readback has n_tickers < 1")

    # CR-A-05 dtype assertions
    if rb["article_id"].dtype != np.uint64:
        raise AssertionError(f"readback article_id dtype = {rb['article_id'].dtype}, expected uint64")
    if rb["n_tickers"].dtype != np.uint16:
        raise AssertionError(f"readback n_tickers dtype = {rb['n_tickers'].dtype}, expected uint16")

    # CR-A-05 parquet schema check via pyarrow (verifies tickers_mentioned is list<string>)
    schema = pq.read_schema(args.dst)
    tm_field = schema.field("tickers_mentioned")
    expected_list_type = pa.list_(pa.string())
    if not tm_field.type.equals(expected_list_type):
        raise AssertionError(
            f"readback parquet schema: tickers_mentioned is {tm_field.type}, expected {expected_list_type}"
        )

    # Spot-check 1 row: re-hash from src (post-sort order) and compare.
    # We re-derive title/content from src directly, not from the original titles/contents
    # Series (those still hold pre-sort order after step 5's sort_values).
    spot = src.iloc[0]
    spot_title = "" if pd.isna(spot["title"]) else str(spot["title"])
    spot_content = "" if pd.isna(spot["content"]) else str(spot["content"])[:512]
    spot_key = f"{spot['date'].isoformat()}\t{spot_title}\t{spot_content}".encode("utf-8")
    spot_aid = xxhash.xxh64(spot_key).intdigest()
    if spot_aid != int(spot["article_id"]):
        raise AssertionError("spot-check hash mismatch — non-deterministic hashing?")
    print(f"[build]   ✓ readback OK; {len(rb):,} articles; sample article_id={int(rb['article_id'].iloc[0])}; "
          f"dtypes: article_id={rb['article_id'].dtype}, n_tickers={rb['n_tickers'].dtype}, "
          f"tickers_mentioned={tm_field.type}")

    print(f"[build] DONE → {args.dst} ({sz_mb:.1f} MB, {len(agg):,} articles)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
