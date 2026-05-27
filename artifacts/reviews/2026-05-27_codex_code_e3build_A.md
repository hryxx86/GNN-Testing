---
reviewer: codex
touchpoint: code
round: A
target: scripts/build_news_edge_source.py
date: 2026-05-26
verdict: PROCEED-WITH-FIXES
---

## Summary
The PIT projection and hash construction are correct, including explicit parquet column selection, TAB-separated UTF-8 xxh64 input, UTC timestamp `isoformat()`, 512-character content truncation, and null text handling. The script should still be fixed before relying on the artifact because `publication_timestamp=('date','first')` is order-dependent without an explicit pre-group sort, and the readback verification does not assert the locked parquet/list and integer dtypes.

## Findings

### CODEX-CR-E3BUILD-A-01: PIT fields are projected out and rechecked
severity: PASS
check: PIT-INTEGRITY
evidence: lines 130, 158-165, 184-190, 217-223: `pd.read_parquet(args.src, columns=["date", "ticker", "title", "content"])`; aggregation only emits `publication_timestamp` and `tickers_mentioned`; forbidden-column checks run on `agg` and readback.
finding: PASS: `return_next`, `label`, sentiment fields, and `tags` cannot enter through the parquet load, and no merge/join reintroduces them.
fix: N/A

### CODEX-CR-E3BUILD-A-02: Article hash key matches the locked formula
severity: PASS
check: HASH-DETERMINISM
evidence: lines 149-151 and 93-96: `titles = src["title"].fillna("").astype(str)`, `contents = src["content"].fillna("").astype(str).str.slice(0, 512)`, `key = f"{d.isoformat()}\t{t}\t{c}".encode("utf-8")`, `xxhash.xxh64(key).intdigest()`.
finding: PASS: the hash uses the UTC `date` timestamp object's `isoformat()`, TAB separators, a 512-character content prefix, UTF-8 bytes, and pre-hash null handling for title/content.
fix: N/A

### CODEX-CR-E3BUILD-A-03: `first` timestamp aggregation is order-dependent
severity: CONCERN
check: AGGREGATION
evidence: lines 158-162: `src.groupby("article_id", sort=False).agg(publication_timestamp=("date", "first"), tickers_mentioned=("ticker", lambda s: sorted(set(str(x) for x in s))))`; no preceding `sort_values(["article_id", "date"])` appears between hashing at line 151 and groupby at line 159.
finding: `publication_timestamp=("date", "first")` is applied to the current parquet row order, not to a deterministic `(article_id, date)` ordering. For normal non-colliding article IDs the date should be invariant because the hash includes `date`, but the code does not enforce that invariant before using `first`.
fix: Sort stably by `article_id` and `date` before groupby, or assert per-`article_id` date uniqueness and use an order-independent aggregation such as `min`.

### CODEX-CR-E3BUILD-A-04: Ticker aggregation preserves non-null article groups
severity: PASS
check: AGGREGATION
evidence: lines 89-96, 151, 162, 166: `out = np.empty(n, dtype=np.uint64)`, `src["article_id"] = hash_article_ids(...)`, `tickers_mentioned=("ticker", lambda s: sorted(set(str(x) for x in s)))`, `agg["n_tickers"] = ...astype("uint16")`.
finding: PASS: `article_id` is a dense `uint64` key, so groupby has no NA article IDs to drop; ticker mentions are deduplicated and sorted; `n_tickers` is cast to `uint16`.
fix: N/A

### CODEX-CR-E3BUILD-A-05: Readback schema verification is incomplete
severity: CONCERN
check: SCHEMA
evidence: lines 203-211 and 217-231: the script locks column order and casts `article_id`/`n_tickers` before write, then readback checks column set, forbidden fields, uniqueness, timestamp tz, and `n_tickers >= 1`, but does not assert `rb["article_id"].dtype == uint64`, `rb["n_tickers"].dtype == uint16`, or parquet `tickers_mentioned` is `list<string>`.
finding: The post-aggregation write path likely preserves the intended types, but the readback assertion does not confirm the locked schema contract for `article_id`, `n_tickers`, or the pyarrow list element type.
fix: Add readback dtype checks for `article_id` and `n_tickers`, plus a parquet schema check with `pq.read_schema(args.dst)` confirming `tickers_mentioned` is `pa.list_(pa.string())`.

### CODEX-CR-E3BUILD-A-06: Null text and zero-ticker edge cases are handled
severity: PASS
check: EDGE-CASES
evidence: lines 138, 142-144, 149-150, 169-174: SP500 filtering occurs before aggregation; empty post-filter input aborts; `title`/`content` nulls become `""`; zero-ticker aggregates are dropped defensively.
finding: PASS: `None`/`NaN` title and content values do not reach xxh64 unnormalized, empty strings remain valid hash input, and articles with zero SP500 tickers are not kept in the output.
fix: N/A

## Verdict Rationale
The script satisfies the highest-risk PIT and hash determinism requirements, so this is not a block-execution finding. The remaining gaps are localized and fixable: make timestamp aggregation deterministic or explicitly invariant-checked, and strengthen readback verification so the locked schema is actually asserted after parquet serialization.
