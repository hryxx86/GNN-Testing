# J Track: CN Data Kill-Gate Criteria (frozen before any data inspection)

> Frozen 2026-07-28 per Rev-2/Rev-14 (`docs/idea_expansions_2026-07-28.md`).
> Rule: ZERO model work on the CN track before this gate passes. Gate report
> → `artifacts/audits/cn_data_gate_<date>.md`. H博士 approved launch
> 2026-07-28. Sources: Qlib CN daily bundle (primary) vs AkShare (independent
> cross-check; no TuShare token required).

## Checks & pass thresholds (all 5 must pass)

1. **PIT membership** — CSI300 constituent history (Qlib instruments file)
   vs AkShare index-constituent history: sample = all rebalances 2020-01→
   2025-12 + 30 random names' membership spans. PASS: rebalance dates
   present within ±5 trading days AND ≥99% agreement of membership stock-days
   on the sample.
2. **Adjusted-price reconciliation** — daily returns from Qlib adjusted
   close vs AkShare qfq close, 30 random constituents × full 2019-07→2026-06
   window. PASS: |Δ daily return| ≤ 1e-3 on ≥99.5% of overlapping stock-days;
   every violation individually documented.
3. **Suspension/delisting completeness** — sampled known suspensions and
   any members leaving the index/market in-window. PASS: no phantom (stale
   nonzero-volume) prices on ≥99% of sampled halt-days; departed members
   retained in historical membership spans (no silent survivor filtering).
4. **Tradability mask (limit rules)** — limit-hit detection rule: main
   board ±10%, ST ±5%, ChiNext/STAR post-2020 ±20%, validated vs AkShare
   high==low==limit days on a 30-name × 3-year sample. PASS: F1 ≥ 0.95.
5. **Calendar & label conventions** — trading calendar identity vs AkShare;
   T+1 label = Ref(close,-2)/Ref(close,-1) reproducible on 10 sampled
   names; 21d-forward label constructible with paper-1 lag discipline.
   PASS: zero calendar mismatches; label spot-checks exact.

## Procedure

Install pyqlib + akshare into the `gnn` env → download Qlib CN daily bundle
→ run checks 1–5 with fixed RNG seed 42 for all sampling → write gate
report with per-check evidence tables. Any FAIL: document → attempt fix via
alternate source/config → re-run gate; unfixable → J BLOCKED, report H博士.
Gate scripts require Rule 9 Touchpoint 2 review before the gate verdict is
accepted (audit code is leakage-load-bearing). MASTER-family reproduction
smoke test (Rev-10) runs only AFTER gate pass.

→ progress: 2026-07-28-f | plan: 2026-07-28 Decision Log | analysis: N/A

---

## RE-FREEZE 2026-07-29 (source pivot — documented amendment, not silent edit)

Trigger: the original primary source (Qlib community bundle) FAILED the
"alternate source/config" branch preconditions — bundle ends 2020-09-25
(~12% window coverage), in-window price errors (2/2 independent vendors
agree against qlib), and no independent membership source in akshare
(progress 2026-07-28-h). Per the frozen procedure, source is repaired and
the affected checks are re-frozen BEFORE the full gate runs:

- **Primary price source** → custom-built qlib-format bundle from
  **baostock k-data** (verified reachable + current through 2025-12 from
  this machine; per-call timeout+retry+cache mandatory since endpoints can
  hang). Factor derived as hfq_close/raw_close. Index sh.000300 included.
- **Membership source** → `experiments/cn_membership_scout/
  csi300_instruments_draft.txt` (CSIndex official announcement-trail
  reconstruction: 592 spans, 540 symbols, 2018-12-17→2026-07-29; 74 exact
  independent point-in-time confirmations — Wayback 5/5, tushare/DoltHub
  61/61 monthly, stale-bundle 8/8 quarterly; zero discrepancies; 3 news
  spot-checks. PROVENANCE.md in the same dir).
- **C1 re-frozen** as: (i) span-integrity audit of the instruments file
  (incl. degenerate floored spans like SH600008 start>end — must be
  handled, not silently dropped); (ii) re-run of the scout's independent
  snapshot diffs as regression; (iii) PIT count == 300 at 6 spot dates.
  Threshold: zero unexplained discrepancies.
- **C2–C5 unchanged in spirit**, re-targeted at the custom bundle over the
  FULL frozen window (coverage_fraction must be ~100% now); C2 cross-check
  construction unchanged (EastMoney raw ÷ Sina qfq-factor, tol 1e-3 @
  ≥99.5%).
- All other thresholds, seed 42 sampling, and the kill rule unchanged.

→ progress: 2026-07-29-a | plan: 2026-07-28 Decision Log | analysis: N/A
