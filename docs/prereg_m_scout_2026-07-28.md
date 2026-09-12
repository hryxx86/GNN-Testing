# Pre-registration: M Scout Step-1 (Peer-Momentum Anomaly Audit, bounded panel)

> Frozen 2026-07-28 BEFORE any audit computation, per Rev-4/Rev-19/Rev-20
> (`docs/idea_expansions_2026-07-28.md`, Codex Round A+B).
> Scope: BOUNDED 2021–2026 analysis on the existing paper-1 panel
> (`data/reference/sp500_5y_prices.csv` + `sp500_sectors.csv`).
> The dead-anomaly/decay branch over 2000–2026 is GATED on WRDS/CRSP access
> (Rev-9) and is NOT part of this scout. H博士 approved launch 2026-07-28.

## 1. Primary characteristic (Rev-20: exactly one)

**Sector-peer momentum**, stock i, day t:
`peer_mom_i(t) = mean_{j in sector(i), j != i} cumret_j(t-126, t-1)`
- sector(i): GICS sector from `sp500_sectors.csv` (snapshot; PIT upgrade
  deferred to WRDS branch — disclosed limitation, consistent with paper-1
  L5s edges).
- cumret: 126-trading-day cumulative simple return, measured strictly at
  T-1 close (project invariant).
- Universe/label: paper-1 conventions verbatim — 501-name panel; outcome =
  21d forward close-to-close market-excess z-scored return; 1-day lag.

## 2. Primary outcome & inference

- Daily cross-sectional Spearman rank IC of peer_mom vs the 21d-forward
  label, pooled over all available days (~2021-07→2026-06 given 126d
  formation burn-in).
- CI: stationary block bootstrap, block length 21d, 5000 draws (paper-1
  convention), on the daily IC series mean.
- Controls (secondary description, not branch-relevant): Fama-MacBeth of
  label on peer_mom controlling own-stock 126d momentum, 21d reversal,
  log market cap proxy (dollar-volume rank if cap unavailable), beta
  (252d); Newey-West HAC lag 21.

## 3. Branch rule (Rev-4, three-way, computed in this order)

1. Before looking at the IC point estimate: compute SE_block = block-
   bootstrap SE of the mean daily IC. Detectability floor F = 1.96 ×
   SE_block (a signal indistinguishable from zero at our T cannot be
   harvested). Provisional Step-3 MDE M3 = 2.8 × SE_block (paper-1 80%-power
   convention, main.tex §Power); to be replaced by the measured planted-
   signal detection threshold in Step-3 calibration.
2. DEAD (bounded sense): IC upper 95% CI < F.
3. ALIVE: IC lower 95% CI > M3.
4. Else: MIDDLE → bounded-diagnostic path (report; no mechanism claims).

Bounded-sense caveat (frozen): a DEAD verdict here is "not detectable in
the modern large-cap survivor panel", NOT the historical-decay claim —
that claim requires the WRDS branch.

## 4. Secondary family (FDR q=0.05 within family; cannot flip the branch)

S1 corr-top-20-peer momentum (126d formation; peers by trailing 126d
correlation, T-1); S2 sector-peer momentum with 21d formation; S3 classical
industry-momentum portfolio read-out (VW sector 6-1 monthly, long-short
mean); S4 Lo–MacKinlay weekly lead-lag cross-autocorrelation summary
(large→small within sector). Wikidata links: descriptive-only (Rev-19),
not tested.

## 5. Multiplicity & integrity

- Exactly 1 primary test (branch-relevant) + 4 secondary tests under BH
  q=0.05. No other variants will be computed before the branch is recorded.
- Any deviation from this document is logged in progress.md before results
  are interpreted.
- Outputs → `experiments/m_scout_step1/` + analysis entry in
  docs/analysis.md; script `analyze_m_scout_step1.py` requires Rule 9
  Touchpoint 2 (Codex code review) BEFORE the confirmatory execution.

→ progress: 2026-07-28-f | plan: 2026-07-28 Decision Log | analysis: pending
