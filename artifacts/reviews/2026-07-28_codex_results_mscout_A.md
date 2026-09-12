<!-- Saved verbatim from Codex T3 run (gpt-5.6-sol xhigh), 2026-07-28.
Claude disposition: all 4 CONCERN ACCEPTED; wording constraints applied verbatim
in docs/analysis.md 2026-07-28-a (licensed claims used, blocked claims excluded,
mandatory caveats included). Statuses updated to FIXED-by-wording below is NOT
edited into the YAML to preserve verbatim fidelity; see progress.md 2026-07-28-i. -->

---
reviewer: codex
touchpoint: results
round: A
target_files:
  - experiments/m_scout_step1/
findings:
  - id: CODEX-M3-A-01
    severity: CONCERN
    category: statistics
    claim: "The DEAD verdict is mechanically correct but close enough to the floor that zero/no-anomaly wording would overstate it."
    evidence: "experiments/m_scout_step1/branch_rule.csv reports ci_hi_95=0.028537536719658658 and F_floor=0.031804225641239275, so F-ci_hi=0.003266688921580617; the same row set reports mean_ic=-0.004139401237753028 and ci_lo_95=-0.03525116995256412."
    suggested_fix: "State the frozen bounded verdict as DEAD, but add that the panel cannot rule out positive IC up to about +0.029, just below the preregistered detectability floor."
    status: OPEN
  - id: CODEX-M3-A-02
    severity: CONCERN
    category: interpretation
    claim: "The actual sample window and bounded scope must be stated exactly; this is not a WRDS/CRSP historical-decay result."
    evidence: "experiments/m_scout_step1/summary.json sample.date_range is [2021-01-29, 2026-01-28], while experiments/m_scout_step1/primary_daily_ic.csv runs from 2021-08-02 through 2025-12-26 with 1107 IC rows; docs/prereg_m_scout_2026-07-28.md states the 2000-2026 dead-anomaly/decay branch is gated on WRDS/CRSP and is not part of this scout."
    suggested_fix: "Use the actual panel and IC dates in docs/analysis.md, and explicitly block any historical decay, post-publication decay, or full-S&P anomaly-death claim."
    status: OPEN
  - id: CODEX-M3-A-03
    severity: CONCERN
    category: interpretation
    claim: "The survivor-panel design limits mechanism interpretation and is not a clean lower or upper bound for true peer momentum."
    evidence: "experiments/m_scout_step1/summary.json reports n_tickers=501, missing_volume_tickers=[AXON], and size_proxy_note='PIT market-cap series unavailable ... trailing dollar-volume rank'; docs/prereg_m_scout_2026-07-28.md uses a sector snapshot and defers PIT sector membership to the WRDS branch."
    suggested_fix: "Describe the result as a current-survivor large-cap panel result only; state that survivorship and snapshot sectors can alter peer ranks and labels, with direction not sign-definite for rank IC."
    status: OPEN
  - id: CODEX-M3-A-04
    severity: CONCERN
    category: statistics
    claim: "S4 is a near-signal only and must not be used to fish around the primary branch decision."
    evidence: "experiments/m_scout_step1/secondary_family.csv reports S4 stat=0.05308481937472209, p=0.06062271151691556, p_bh=0.22151548944070232, reject_bh_q05=False, ci_lo=0.001414782604134028, ci_hi=0.11116628512274968, n_obs=246."
    suggested_fix: "Report S4 as a preregistered secondary, hypothesis-generating weekly large-to-small lead-lag read-out; do not let it alter the DEAD branch or imply a confirmed peer-momentum characteristic."
    status: OPEN
summary:
  critical: 0
  major: 0
  concern: 4
overall_verdict: PASS_WITH_CONCERNS
---

**1. DEAD Verdict**

The DEAD verdict is correctly derived under the frozen preregistration rule.

`branch_floor.csv` reports `se_block=0.016226645735326162`, `F_floor=0.031804225641239275`, and `M3_mde=0.04543460805891325`, exactly matching `1.96*SE` and `2.8*SE`. `branch_rule.csv` then reports `mean_ic=-0.004139401237753028`, 95% CI `[-0.03525116995256412, +0.028537536719658658]`, `n_days_ic=1107`, and `verdict=DEAD`.

The strict rule is `ci_hi_95 < F_floor`. Here `0.0285375367 < 0.0318042256`, so DEAD follows. The branch order in `branch_rule.csv` is also correct: SE, F, M3 precede the mean/CI/verdict. `branch_floor.csv` lists `n_days_ic` before SE, but that is not an effect estimate and does not violate the preregistered “SE before point estimate” rule.

**2. Licensed Claims And Blocked Claims**

Licensed: the result supports the bounded claim that the preregistered sector-peer momentum characteristic is not detectable in the modern large-cap survivor panel under paper-1 label conventions. More exactly: 126d sector-peer momentum does not clear the frozen detectability floor for daily rank IC against the 21d forward market-excess z-scored label.

Licensed, but only narrowly: it can be used as mechanism support for paper-1 in the sense that this specific graph-borne anomaly proxy is below the panel’s detectability floor. That makes paper-1’s graph-penalty null less surprising. It does not prove that graph models fail because the anomaly decayed, and it does not rule out other graph-borne mechanisms.

Blocked: historical decay. The WRDS/CRSP branch was not run. No claim should say the anomaly “died,” “decayed post-publication,” or disappeared over 2000-2026.

Mandatory caveats: the actual panel is `2021-01-29` to `2026-01-28`, and the primary IC window is `2021-08-02` to `2025-12-26`. It is a 501-name survivor panel with snapshot sectors, not PIT membership/sector history with delisting returns. Survivorship bias is not sign-definite for rank IC: it can truncate loser/delisting tails and reduce dispersion, but can also make portfolio read-outs look better by omitting failed names. The 21d daily label is not a full classical monthly J/K industry-momentum test. S1/S2 negative point estimates are not evidence of reversal: S1 is `-0.0218` with `p=0.3267`; S2 is `-0.0233` with `p=0.1108`, `p_BH=0.2215`.

**3. Statistical Soundness**

`SE_block=0.0162` is plausible. From `primary_daily_ic.csv`, the daily IC sample standard deviation is about `0.1455`, giving an iid SE near `0.00437`. But the 21d overlapping labels create high serial dependence: computing directly from `primary_daily_ic.csv`, lag-1 autocorrelation is about `0.905` and lag-21 autocorrelation is still about `0.091`. A 21d block-bootstrap SE around `0.016` is therefore reasonable.

Calling the branch DEAD is defensible because the rule was frozen and the inequality holds. The write-up must not make it sound like a wide-margin rejection of any positive peer momentum. The CI allows IC values up to `+0.0285`, about 90% of the detectability floor. Required characterization: “formal bounded DEAD; cannot rule out a small positive IC up to roughly +0.029.”

**4. S4 Near-Signal**

S4 should be reported as a preregistered secondary near-signal: large-to-small weekly lead-lag asymmetry is positive (`+0.0531`) with unadjusted `p=0.0606`, but BH-adjusted `p=0.2215` and `reject_bh_q05=False`. It cannot flip the primary branch.

The positive S4 percentile CI should not be used as a backdoor significance claim, because the preregistered secondary family decision is BH over p-values. The disciplined interpretation is: there may be a weekly size-asymmetric lead-lag structure worth a separately preregistered follow-up, but the 126d sector-peer momentum characteristic still fails at the paper-1 21d label horizon.

**5. Decision Relevance**

Without WRDS/CRSP, M scopes down. This bounded DEAD result is useful as a paper-1 support/diagnostic result, but it is not enough to carry a full historical-decay mechanism paper. With WRDS, the historical decay branch can still open.

For paper-2 spine selection, this weakens M as the immediate main spine and strengthens J, conditional on the J data gate passing. M is not “dead” as a research line, but the bounded scout now looks more like supporting evidence for the paper-1 graph-penalty story than a standalone mechanism paper, unless WRDS produces the historical decay curve.

**Recommended docs/analysis.md Wording**

May write:

“The preregistered M Scout Step-1 primary test yields mean daily rank IC `-0.00414` for 126d sector-peer momentum against the paper-1 21d forward market-excess z-scored label, with 95% 21d block-bootstrap CI `[-0.03525, +0.02854]` over `1,107` daily ICs. The frozen detectability floor is `F=0.03180`; because the CI upper bound is below `F`, the preregistered bounded branch verdict is `DEAD`.”

Must accompany with:

“This is a bounded modern survivor-panel result, not a historical-decay result. The underlying price panel spans `2021-01-29` to `2026-01-28`, and the primary IC window spans `2021-08-02` to `2025-12-26`. The WRDS/CRSP PIT-membership, historical-sector, and delisting-return branch was not run.”

Also may write:

“The result supports the interpretation that the sector-peer momentum signal motivating graph-stock models is not detectable in this paper-1 panel/horizon, which makes the paper-1 graph-penalty null less surprising.”

Must not write:

“The peer-momentum anomaly is dead,” “historical decay is confirmed,” “GNN-stock motivation is stale,” or “there is no graph-borne signal.” The strongest allowed statement is bounded: this preregistered sector-peer characteristic does not clear the detectability floor in the modern large-cap survivor panel, and the analysis cannot rule out a small positive IC up to about `+0.029`.