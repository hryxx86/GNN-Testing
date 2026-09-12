---
reviewer: codex
touchpoint: code
round: A
target_files:
  - analyze_m10_universe_gap.py:1-185
findings:
  - id: M10-CODE-A-001
    severity: MAJOR
    category: correctness
    claim: "Stock-day spans include the removal effective date, contradicting the intended half-open membership interval ending at removal_date."
    evidence: "Comment specified [start, removal_date) but trading_days_between used `lo <= d <= hi`; rm_d passed as inclusive hi. All 87 in-window removal dates are trading dates → missing_stock_days inflated by 1 day per gap ticker (87 days)."
    suggested_fix: "Make the counter half-open (`lo <= d < hi`) or pass the prior trading date as inclusive upper bound."
    status: FIXED
    resolution_notes: "Replaced trading_days_between with count_days(dates, lo, hi_excl) using `lo <= d < hi_excl`; removed names counted over [start, removal_date). Re-run: missing_stock_days 51,928 → 51,841 (exactly −87, one per gap ticker). Verified at analyze_m10_universe_gap.py:73-75,118."
  - id: M10-CODE-A-002
    severity: CRITICAL
    category: statistics
    claim: "The stock-days denominator labeled PIT total is not a PIT membership denominator: fixed-snapshot tickers are counted for the full window regardless of their actual add dates, understating the stock-days gap."
    evidence: "our_stock_days = len(valid) * n_trading credited every snapshot ticker all 1,255 days; 76 snapshot tickers have in-window add events, so their pre-add days were wrongly counted as PIT membership → gap % biased low."
    suggested_fix: "Build membership intervals for every superset ticker (snapshot survivors AND removed names); sum membership-days for the denominator; numerator stays the gap-ticker membership-days."
    status: FIXED
    resolution_notes: "Rewrote the stock-days block to iterate the full superset: survivors counted from max(WINDOW_START, latest add≤WINDOW_END) through window end; gap names over [start, removal). pit_total_stock_days now sums per-ticker tenure. Re-run: stock-days gap 7.6% → 8.2% (denominator shrank as 76 mid-window-added snapshot names lost pre-add days). our_universe_stock_days (501×1255 look-ahead count) retained for reference only. Verified at analyze_m10_universe_gap.py:104-135."
summary:
  critical: 1
  major: 1
  concern: 0
  fixed_before_reply: 2
overall_verdict: PROCEED-WITH-FIXES
---

# Codex Code Review — M10 universe-gap audit (Touchpoint 2, Round A)

Target: `analyze_m10_universe_gap.py` (new one-off survivorship-gap audit feeding the paper's M10 Limitations disclosure).

Codex responded < 1 min (no fallback). Two correctness findings, both verified against the cited lines by Claude and both fixed:

- **A-002 (raised CRITICAL by Claude; Codex "high")** — the original denominator `501 × 1255` over-credited snapshot names that only joined the index mid-window, biasing the stock-days gap **low**. Fixed by building true per-ticker PIT membership intervals across the whole superset. Effect: stock-days gap **7.6% → 8.2%** (honest, slightly larger).
- **A-001 (MAJOR)** — removal effective date is the date the name is already out, so membership is half-open `[start, eff)`. Fixed; missing stock-days **51,928 → 51,841** (−87, one per gap ticker).

Codex confirmed the **names-gap set logic** (`valid_set | removed_tickers`, gap = removed ∉ valid) and date/window parsing are correct; no duplicate removals, no coerced-date failures, no blank tickers in the current cached data.

Post-fix headline (data of record, `artifacts/audits/m10_universe_gap.{csv,md}`):
- Names gap = 87 / 588 = **14.8%**
- Stock-days gap = 51,841 / 629,580 = **8.2%**
- Verdict: SOFT-MIDDLE (10–20% names) — below the 20% hard-escalation; disclose with the small-residual-tenure / stock-days argument.

→ progress: 2026-06-26-b | plan: handoff-m (M10) | analysis: 2026-06-26-a
