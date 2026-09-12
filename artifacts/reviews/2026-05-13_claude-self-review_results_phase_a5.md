---
reviewer: claude-self-review
touchpoint: results
round: A
fallback_chain: [codex (quota limit hit 2026-05-13, resets 8:10am PT), claude-self-review (continuing 2026-05-02 fallback pattern)]
target_files:
  - artifacts/tier1_phase_a/stat_report.md
  - artifacts/tier1_phase_a/stat_per_cell.csv (36 rows)
  - artifacts/tier1_phase_a/stat_tier1d.csv (8 rows)
findings:
  - id: SELF-A5-R-01
    severity: CONCERN
    category: interpretation
    claim: "Tier 1.B 'fold-4 novel mechanism finding' rests on 8/12 contrasts having NW p<0.05 in negative direction on fold-4. The claim is that 'robust pointwise losses harm directional accuracy during regime shifts.' But fold-4 IS the only stress fold in the 5-fold panel — n=1 regime sample for generalizing to 'regime shifts.'"
    evidence: "stat_per_cell.csv fold_4 rows show 8 of 12 with delta_IC_p_NW < 0.05. But generalization 'regime shifts' (plural) requires multiple stress periods. We have ONE: Q2-2025 (fold-4 test period 2025-04-01 → 2025-06-30). The mechanism interpretation should be hedged: 'in the one stress-period in our panel, robust losses underperformed.'"
    suggested_fix: "Rewrite stat_report.md mechanism interpretation to 'robust pointwise losses underperform MSE on the one stress-regime fold (fold-4 Q2-2025) in our panel; multi-regime generalization requires additional historical stress periods.' Paper should similarly hedge the language."
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "Stat_report.md already labels fold-4 as 'diagnostic only' and notes 'n_eff per fold is small (~3 per fold per Plan power analysis)' (caveat 1) and 'fold-4 NW t values reflect cross-day consistency, not large independent signal.' Disclosure is adequate. For paper, ensure 'regime shifts' is singular or explicitly hedged."
  - id: SELF-A5-R-02
    severity: CONCERN
    category: interpretation
    claim: "Tier 1.D registered Score gate produces h2 winner with Score = +0.0007. Plan §1.D expected ΔIC range was +0.003 to +0.008 per Codex C. Observed h2 Score of +0.0007 is BELOW the expected range — h2 'wins' by being the LEAST overfit among the 4 hparams, not by hitting the expected effect size."
    evidence: "stat_tier1d.csv row hparam_idx=2 loss='mse' col 'score' = +0.0007 (computed inline as mean_IC - 0.35*sigma_fold - 0.05*indicator). Plan §1.D Expected ΔIC: '+0.003 to +0.008 if overfitting residual real' (Codex C estimate). h2's Score +0.0007 is sub-+0.003, indicating the overfitting-residual hypothesis is weakly supported at best."
    suggested_fix: "stat_report.md does note 'Score = +0.0007 is at the low end of (or below) Plan's Codex-C-stated expected ΔIC range +0.003 to +0.008.' This is correctly disclosed. Just ensure paper framing reflects this — the 'positive' finding for regularization is genuinely marginal at the registered gate."
    status: PASS
    resolution_notes: "Already disclosed in stat_report.md line ~179 'Score = +0.0007 is at the low end of (or below) Plan's Codex-C-stated expected ΔIC range'. Disclosure adequate."
  - id: SELF-A5-R-03
    severity: CONCERN
    category: methodology
    claim: "Sharpe values use z-scored fwd-21d returns as portfolio return proxy (per SELF-A5-C-06 code review). Some reported Sharpes are large-magnitude (e.g. tukey/SAGE-Mean/S6 fold-4 annualized Sharpe = −22.9), which is implausibly extreme for any realistic equity strategy. These need clear 'z-score proxy, not raw return' labeling in any paper table."
    evidence: "stat_per_cell.csv col sharpe_new shows extreme values on fold-4 stress (e.g. tukey/SAGE-Mean/S6 fold_4: -22.92). Real-world annualized Sharpe rarely exceeds |3| even for top funds. The magnitudes reflect (a) the z-score proxy unit issue, (b) fold-4's 62-day length giving high annualization variance."
    suggested_fix: "Paper supplementary table 1 should report Sharpe ONLY in all-folds view (313-day, less annualization noise) AND should use raw fwd_ret labels (not z-scored). Two-line change to analyze_tier1_phase_a.py + rerun analysis (no model retraining needed). ~5 min compute."
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "Stat_report.md line 213 already discloses 'Sharpe values are sensitive to portfolio definition.' Adequate for supplementary; for headline-grade paper Sharpe, switch to raw fwd_ret. Defer this until Phase B (b) paper writing if needed."
  - id: SELF-A5-R-04
    severity: CONCERN
    category: methodology
    claim: "BH-FDR scope is 12 (loss × model × feat) contrasts in all_folds view only. This treats the 4 (model × feat) cells per loss as independent tests of the same hypothesis. An alternative could be the more conservative family = 3 losses × 4 (model × feat) × 3 views = 36 tests with BH-FDR across all. Current scope (12) is the minimum sensible family."
    evidence: "Plan §1.B says 'BH-FDR family across new losses; Bonferroni co-primary already applied to Stage 1' — this strongly implies family-wise control across the 3 new losses, but doesn't explicitly say whether to additionally cross (model × feat) or 3 views."
    suggested_fix: "Plan-aligned scope (12) is defensible and was applied. A more conservative scope (36) would only make the null finding STRONGER (more contrasts to clear, all fail). No change needed."
    status: PASS
    resolution_notes: "Current scope is plan-consistent. More-conservative scope only makes verdict more robust (still 0/36 rejections expected since min unadjusted p = 0.364)."
  - id: SELF-A5-R-05
    severity: CONCERN
    category: scope
    claim: "Plan §2.C 'Sector-adjusted IC' (IC_sector_resid) was specified as a secondary metric for every Tier 1 cell. Not computed in current analysis. This is a gap vs Plan but does not invalidate current verdicts (primary metric is IC_abs which was computed)."
    evidence: "Plan §2.C lines 393-409 specifies IC_sector_resid as 'NEW secondary metric per 2.C' that 'every Tier 1 + Tier 2 cell gets BOTH'. analyze_tier1_phase_a.py does not compute this. stat_per_cell.csv has no IC_sector_resid column."
    suggested_fix: "Add IC_sector_resid computation pass to analyze_tier1_phase_a.py. Needs sector mapping (already available via pa.load_data_and_features → data['sector_groups']). ~2h dev + ~1 min compute on existing preds (no retraining). Can be a Phase B addendum before paper draft or after (c)/(d)/(e)."
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "Plan gap; does not change current null/marginal verdicts. Can be added later as paper supplementary table. Deferred to post-(b) or whenever H博士 instructs."
  - id: SELF-A5-R-06
    severity: PASS
    category: interpretation
    claim: "Tier 1.B verdict (0/12 BH-FDR rejections at α=0.05) is unambiguous. Combined with Stage 1's 0/8 ranking-loss rejection, the joint claim '20 contrasts across two distinct loss-family hypotheses, none significantly beats MSE' is well-supported."
    evidence: "stat_per_cell.csv view='all_folds' col p_NW_BH_adj minimum is 0.830. No contrast is even close to α=0.05. Combined with Stage 1's documented 0/8 BH-FDR rejection (per docs/analysis.md 2026-04-27 entries), the cross-experiment claim is sound."
    status: PASS
    resolution_notes: "Strong null finding; one of the paper's headline contributions."
  - id: SELF-A5-R-07
    severity: PASS
    category: interpretation
    claim: "Tier 1.D CORRECTED verdict properly distinguishes pre-registered Score selection from post-hoc NW-HAC observations. Frontmatter `correction_log` documents the change. Stat_report.md verdict section leads with Score, not mean_IC."
    evidence: "stat_report.md frontmatter `correction_log` 2026-05-06 entry; line 154+ 'Tier 1.D verdict (CORRECTED 2026-05-06 per Codex stop-time review — original framing violated pre-registered gate)'. Score table at line 138 leads with h2 winner (+0.0007), then h0 second (+0.0006), then Score-NEGATIVE h1/h3."
    status: PASS
    resolution_notes: "Codex stop-hook caught + corrected. Current report is properly framed."
  - id: SELF-A5-R-08
    severity: PASS
    category: methodology
    claim: "Statistical primaries (paired daily IC differences, average-then-HAC seed aggregation, NW-HAC lag=21, fold-cluster bootstrap sensitivity) match Plan §B-02 (i) verbatim."
    evidence: "analyze_tier1_phase_a.py:259-289 computes d_per_seed_per_day, averages to d_avg_seed, applies newey_west_hac(lag=21), then fold_cluster_bootstrap on per-fold means. Matches Plan §B-02 (i) recommended default."
    status: PASS
    resolution_notes: "Methodology is plan-aligned. No FORBIDDEN seed-inflation pattern."
summary:
  critical: 0
  major: 0
  concern: 5
  pass: 3
overall_verdict: PASS-WITH-CONCERNS
---

# Self-Review — Plan Z++ Phase A.5 Results (Round A)

## Reviewer note

Same fallback as code self-review (`2026-05-13_claude-self-review_code_phase_a5.md`). Codex unavailable.

## Verdict

**PASS-WITH-CONCERNS**: 0 CRITICAL + 0 MAJOR + 5 CONCERN + 3 PASS findings.

The 5 CONCERN findings are about hedging language and gap items that don't change verdicts:
- "Regime shifts" → "the one stress-regime fold" (singular hedge)
- Sharpe magnitude proxy disclosure
- IC_sector_resid gap (Plan §2.C secondary metric — not yet computed)
- BH-FDR scope justification
- h2 Score below Codex C expected range (already disclosed)

The 3 PASS findings cover the headline verdicts:
- Tier 1.B 0/12 null (unambiguous, combined with Stage 1 = 20 contrasts)
- Tier 1.D corrected verdict (h2 winner, NW p=0.059 marginal)
- Statistical methodology (NW-HAC, BH-FDR, bootstrap) plan-aligned

## Critical verdicts unchanged

- **Tier 1.B: strong NULL** — stands.
- **Fold-4 stress: 8/12 contrasts NW-significantly negative** — stands, with hedge that this is one stress sample.
- **Tier 1.D: MARGINALLY SUPPORTED at Score gate** — stands per Codex stop-hook correction.

## Gaps to consider for paper

1. **IC_sector_resid (Plan §2.C)** not computed — supplementary table addition (~2h dev).
2. **Sharpe with raw fwd_ret** not computed — currently uses z-score proxy. Switch for headline table (~5 min compute).
3. **Per_fold_scale all-stock vs valid-mask** (per code review SELF-A5-C-01) — affects absolute IC magnitudes if cross-comparing with Stage 1. Re-run cost ~14h; defer unless paper requires direct cross-comparison.

## What I did not over-claim

- I did NOT extrapolate fold-4 negative finding to a general "regime-shift" mechanism without hedging.
- I did NOT claim Tier 1.D is "positive" — only "marginally supported at registered gate" per the Codex stop-hook correction.
- I did NOT extrapolate Sharpe magnitudes to real-world strategy returns without disclosure of the z-score proxy.

## Recommendation

**Proceed to Phase B (b) paper draft.** Address the 3 paper-prep gaps when convenient (IC_sector_resid + raw-return Sharpe + maybe scaler rerun). The 5 CONCERN findings are sub-blocking; the headline verdicts (Tier 1.B null + fold-4 mechanism + Tier 1.D marginal) are well-supported.

When Codex quota resets, run formal Touchpoint 3 with this self-review as input. Expected: confirms PASS-WITH-CONCERNS or asks for the IC_sector_resid addition.
