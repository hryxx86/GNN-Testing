---
reviewer: codex
touchpoint: results
round: A
target_files:
  - artifacts/audits/m10_universe_gap.md
  - artifacts/audits/m10_universe_gap.csv
  - analyze_m10_universe_gap.py
findings:
  - id: CODEX-M10-A-01
    severity: MAJOR
    category: statistics
    claim: "8.2% understates total fixed-vs-PIT mismatch; the snapshot also look-ahead-credits current names their pre-addition days."
    evidence: "PIT snapshot-name days = 629,580 − 51,841 = 577,739; fixed universe carries 628,755; so 51,016 pre-add look-ahead days. Two-sided composition mismatch = (51,841 + 51,016)/629,580 = 16.3%."
    suggested_fix: "Report 8.2% as 'omitted removed-member stock-days' and add a 'composition mismatch' metric including pre-add days; do not use 8.2% alone."
    status: FIXED
    resolution_notes: "Script now computes lookahead_days (51,016 / 8.1%) and composition_mismatch_pct (16.3%); md reports survivorship 8.2% + look-ahead 8.1% + total 16.3%. analyze_m10_universe_gap.py:137-145."
  - id: CODEX-M10-A-02
    severity: MAJOR
    category: interpretation
    claim: "'survivorship bias largely cancels in relative contrasts' is not guaranteed under non-uniform attrition (bank failures SIVB/SBNY/FRC, M&A ATVI/TWTR/XLNX/MXIM/PXD)."
    evidence: "Relative contrasts cancel common restriction only if arm-level effects are stable across included/omitted names; GNN/news/sector/MLP arms could differ on distress/contagion/acquisition clusters."
    suggested_fix: "State a conditional-estimand: pairing on the same fixed snapshot reduces mechanical cross-arm imbalance but does not equal a PIT estimand; heterogeneous effects on delisted/M&A/distress names remain unmeasured."
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "Binding spec for the M10 Limitation wording (main.tex). Drop 'cancels'; use conditional-estimand language. Applied in disclosure-text step."
  - id: CODEX-M10-A-03
    severity: CONCERN
    category: reproducibility
    claim: "'true PIT denominator' overclaims source quality — Wikipedia changes table, not official S&P DJI/CRSP."
    evidence: "May miss/normalize ticker-change, share-class, correction events; half-open effective-date convention is ≤1 day/event, immaterial."
    suggested_fix: "Call it 'Wikipedia-reconstructed PIT membership'; disclose source + effective-date convention in the audit footnote."
    status: FIXED
    resolution_notes: "Renamed throughout md to 'Wikipedia-reconstructed PIT membership'; added source caveat + effective-date note + absent-vs-NaN distinction. analyze_m10_universe_gap.py md block."
  - id: CODEX-M10-A-04
    severity: CONCERN
    category: other
    claim: "Disclosure-without-rebuild is defensible only if the estimand is narrowed; not if the paper claims an investable PIT S&P 500 backtest."
    evidence: "Gap is moderate (14.8% names, 16.3% composition). Paper's claims are within-universe paired contrasts → disclosure path plausible."
    suggested_fix: "Methods+Limitations: results conditional on a fixed 501-ticker end-window snapshot, not PIT; avoid live-investability / absolute-Sharpe / S&P-wide claims."
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "Binding spec for M10 disclosure. Paper already treats net Sharpe descriptive + IC as metric; will add explicit 'conditional on fixed snapshot' to Methods §3.1 + Limitation."
  - id: CODEX-M10-A-05
    severity: CONCERN
    category: other
    claim: "Limitation wording over-claim risk: 'largely cancels', 'true PIT', 'delisted→NaN→excluded' (removed names are absent entirely, not NaN-excluded)."
    evidence: "8.2% is moderate not negligible; attrition non-uniform; removed names never in the price file (≠ within-universe NaN exclusion)."
    suggested_fix: "Use the provided neutral wording reporting 14.8% / 8.2% / look-ahead, distinguishing absent-from-universe vs NaN-excluded."
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "Binding spec for M10 disclosure wording. absent-vs-NaN distinction already added to audit md; will mirror in main.tex Methods/Limitation."
summary:
  critical: 0
  major: 2
  concern: 3
  fixed_before_reply: 2
overall_verdict: PROCEED-WITH-FIXES
---

# Codex Results Review — M10 survivorship-gap audit (Touchpoint 3, Round A)

Codex responded < 2 min (no fallback). Verdict PROCEED-WITH-FIXES — no blocker; all five findings improve the disclosure rather than reject the disclose-without-rebuild path.

**Fixed in the audit script/output (A-01, A-03):** added the two-sided composition-mismatch metric (survivorship 8.2% + look-ahead 8.1% = 16.3% total), renamed to "Wikipedia-reconstructed PIT membership", added source + effective-date + absent-vs-NaN caveats.

**Binding specs for the M10 manuscript disclosure (A-02, A-04, A-05) — to apply in the Methods §3.1 + Limitation edit:**
1. Report 14.8% names / 8.2% survivorship stock-days / 8.1% look-ahead / 16.3% composition mismatch — the honest full picture.
2. Drop "largely cancels"; use a conditional-estimand statement (pairing on the fixed snapshot reduces mechanical cross-arm imbalance, ≠ a PIT S&P 500 estimand; heterogeneous effects on delisted/M&A/distress names unmeasured).
3. State results are conditional on a fixed 501-ticker end-window snapshot, not PIT; avoid live-investability / absolute-Sharpe / S&P-wide claims.
4. Distinguish: removed names are absent from the universe entirely vs the within-universe NaN-exclusion of missing forward-return cells.

Data of record: `artifacts/audits/m10_universe_gap.{csv,md}` (names gap 87/588=14.8%, survivorship 51,841/629,580=8.2%, look-ahead 51,016=8.1%, composition mismatch 16.3%).

→ progress: 2026-06-26-b | plan: handoff-m (M10) | analysis: 2026-06-26-a
