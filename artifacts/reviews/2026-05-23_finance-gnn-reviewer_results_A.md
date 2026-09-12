---
reviewer: finance-gnn-reviewer
touchpoint: results
round: A
date: 2026-05-23
fallback_reason: "Codex CLI rate-limited + Anthropic API 529 outage; finance-gnn-reviewer is Rule 9 §Fallback primary reviewer"
target_files:
  - /Users/heruixi/Desktop/GNN-Testing/artifacts/plan_aaa/ranking.csv
  - /Users/heruixi/Desktop/GNN-Testing/artifacts/plan_aaa/hand_curated_mapping_168.json
  - /Users/heruixi/Desktop/GNN-Testing/artifacts/plan_aaa/adjusted_rand_index.json
  - /Users/heruixi/Desktop/GNN-Testing/artifacts/plan_aaa/audit/convergence.json
  - /Users/heruixi/Desktop/GNN-Testing/artifacts/plan_aaa/audit/data_provenance.json
  - /Users/heruixi/Desktop/GNN-Testing/artifacts/plan_aaa/audit/permutations.parquet
  - /Users/heruixi/Desktop/GNN-Testing/artifacts/plan_aaa/groups_168.json
  - /Users/heruixi/Desktop/GNN-Testing/docs/plan_aaa_results_2026-05-25.md
findings:
  - id: FINGNN-RESULTS-A-01
    severity: MAJOR
    category: interpretation
    claim: "Doc §7 'Universe-policy implications' overstates evidence for action. Bullets 1-3 framed as actionable recommendations, but no group survives BH-FDR except CORD20+1. For everything else the correct stance is 'cannot distinguish from noise at q=0.05'. Risk: a paper reviewer attacks the document as p-hacking via ranks where p-values offer no support."
    evidence: "docs/plan_aaa_results_2026-05-25.md:240-243 (§7 Universe-policy implications, bullets 1-3). hc_mom12m raw p=0.311, BH p_adj=0.647 (ranking.csv row 1); hc_ret_mean_21d+5 raw p=0.091, BH p_adj=0.504 (ranking.csv row 60). Neither rejected."
    suggested_fix: "Reframe §7 as 'descriptive observations, not actionable recommendations': replace 'Keep/Reconsider/Consider removing' with 'Ranks consistent with retention by hc_mom12m...' / 'Ranking weakly inconsistent with retention of hc_ret_mean_21d, but BH-FDR cannot confirm at q=0.05'. Promote the existing 'Note for paper' bullet (line 244)."
    status: OPEN

  - id: FINGNN-RESULTS-A-02
    severity: MAJOR
    category: interpretation
    claim: "CORD20+1 negative-ΔIC interpretation in §5 (line 174) is mechanically correct but causally overreaches. 'Inclusion of CORD20+1 actively damages prediction quality' is a causal claim from a non-interventional permutation test on a trained model. Negative ΔIC means the model is using CORD20+1 in a direction that anti-correlates with truth on this test window — does NOT establish that retraining without CORD20+1 would improve IC. Strobl 2008 (cited in §10.2) is exactly about this misinterpretation class."
    evidence: "docs/plan_aaa_results_2026-05-25.md:174 ('Suggests inclusion of CORD20+1 in the universe actively damages prediction quality')."
    suggested_fix: "Soften to: 'On the test panel, the model's learned use of CORD20+1 is reliably anti-correlated with realized cross-sectional ranks (NW p=0.0003, BH-FDR p_adj=0.021). This does not by itself establish that retraining without CORD20+1 would improve OOS IC — the permutation breaks both feature-target correlation and feature-feature dependencies. A confirmatory drop-CORD20+1 retraining experiment would be needed for a causal pruning claim.'"
    status: OPEN

  - id: FINGNN-RESULTS-A-03
    severity: CONCERN
    category: interpretation
    claim: "§7 outcome label 'Mixed' counting hc top-half placement as 'partial validation' is the kind of asymmetry an ICAIF reviewer will flag. 0/7 hc groups survive BH-FDR; 3 Alpha158 groups (KMID+6 rank 4, BETA20+8 rank 6, CNTP5+5 rank 7) outrank 6 of 7 hc groups."
    evidence: "docs/plan_aaa_results_2026-05-25.md:233 ('neither validated nor refuted'); compare ranking.csv rows 4, 6, 7 vs hc_ret_mean_5d+6 rank 33."
    suggested_fix: "Replace 'partial validation' with 'Plan AAA finds 1/7 hc groups in top-10 (hc_mom12m, rank 1), 3/7 in top-half, and 4/7 in bottom-half. No hc group survives BH-FDR at q=0.05. The methodology identifies one harmful Alpha158 group (CORD20+1) but lacks power to confirm any hc group as beneficial above chance — this is the honest reading of outcome (b).'"
    status: OPEN

  - id: FINGNN-RESULTS-A-04
    severity: CONCERN
    category: regime-sensitivity
    claim: "ARI=0.5506 < 0.85 fired concern gate (correctly reported in §4) but caveat NOT propagated into §6 / §7. §4.3 mechanism #2 (hc_mom12m forced singleton because 252d lookback ≥ 252d window) means on fold-0 (714d window) grouping, hc_mom12m would merge with siblings and the singleton bonus may evaporate — could swap rank 1."
    evidence: "docs/plan_aaa_results_2026-05-25.md:90-114 (§4.1-§4.3); §7.1 paper limitations narrative (line 299) mentions ARI in passing but doesn't connect to rank-1 interpretation."
    suggested_fix: "Add to §7 and §10.1: 'In particular, hc_mom12m's rank-1 placement depends on its forced singleton status in the 252-day calibration window. On fold-0 (714-day) grouping the same feature merges with medium-momentum siblings — a fold-0-grouping replication is identified in §11 next-steps as a robustness check.'"
    status: OPEN

  - id: FINGNN-RESULTS-A-05
    severity: CONCERN
    category: statistics
    claim: "BH-FDR over K=61 cluster-derived groups assumes PRDS (Benjamini-Yekutieli 2001). Doc doesn't state assumption nor BY conservative alternative. With only 1 rejection, doesn't change conclusion, but a careful reviewer will ask."
    evidence: "docs/plan_aaa_results_2026-05-25.md:182 (§5.1: 'Multiple testing: BH-FDR at q=0.05 over K=61 groups'); no PRDS discussion."
    suggested_fix: "Add §5.1 footnote: 'BH-FDR controls FDR under PRDS (Benjamini-Yekutieli 2001); cluster-based groupings typically satisfy this. Under arbitrary dependence the BY-corrected critical value is K·H_K ≈ 4.6× more conservative — applying it would not change conclusions (1 rejection at p_adj=0.021 × 4.6 ≈ 0.097, just above 0.05).'"
    status: OPEN

  - id: FINGNN-RESULTS-A-06
    severity: CONCERN
    category: credibility
    claim: "Doc §8 (line 271) glosses the one failed cell (cell_id=28, MLP fold 4 seed 123) with 'predictions still used per Plan v1 §3.2 halt rule'. Correct per pre-registration. But val IC -0.069 → test IC +0.247 is a fold-4-leakage red flag class. Without leave-fold-4-out sensitivity, rank-1 hc_mom12m and rank-61 CORD20+1 could both have unstated dependence on fold-4 idiosyncrasy."
    evidence: "audit/convergence.json cell_id=28 (best_val_ic=-0.069); docs/plan_aaa_results_2026-05-25.md:271."
    suggested_fix: "Pre-register a leave-fold-4-out re-ranking as supplementary table using daily_delta_ic_per_group.csv (no retraining needed). If ranks hold, headline strengthens; if not, limitations need updating."
    status: OPEN

  - id: FINGNN-RESULTS-A-07
    severity: CONCERN
    category: pre-registration
    claim: "All claimed pre-registration parameters verified against plan_aaa_v1_2026-05-23.md: |ρ|>0.6, BH q=0.05, block_len=21, calibration days [0,251], ARI<0.85 concern threshold. No deviations."
    evidence: "Cross-checked plan_aaa_v1_2026-05-23.md:23, 119-129, 248-253 vs groups_168.json (threshold=0.6, calibration_window_indices=[0,251])."
    suggested_fix: "No change needed. Positive verification."
    status: ACCEPTED-AS-CONCERN

  - id: FINGNN-RESULTS-A-08
    severity: CONCERN
    category: prior-art
    claim: "§10.2 Strobl 2008 + Lundberg-Lee 2017 citations correctly framed and humble. Gap: Plan AAA's within-day grouped permutation conditional on cluster structure is closer to Hooker & Mentch 2019 (Please Stop Permuting Features) than Strobl 2008 proper."
    evidence: "docs/plan_aaa_results_2026-05-25.md:304-309 (§10.2)."
    suggested_fix: "OPTIONAL: add Hooker, G. & Mentch, L. (2019) Please Stop Permuting Features: An Explanation and Alternatives, arXiv 1905.03151. Hardens prior-art framing against permutation-importance critiques."
    status: OPEN

summary:
  critical: 0
  major: 2
  concern: 6
  total: 8

overall_verdict: PROCEED-WITH-FIXES
verdict_rationale: "Data analysis pipeline is sound and matches pre-registration exactly. No CRITICAL. 2 MAJOR are both interpretive wording fixes in docs/plan_aaa_results_2026-05-25.md §5 and §7 — soften the CORD20+1 causal claim per Strobl 2008 (already cited!), reframe §7 'Keep/Reconsider/Remove' bullets as descriptive ranks rather than actionable universe edits. After these and the propagation of ARI=0.55 caveat into §7/§10.1, doc is paper-ready."
---

# Review body

## Credibility (a) — PASS

Independently verified all 6 headline numbers from source files (not quoted text):
- Rank 1 hc_mom12m: ranking.csv row 1 confirmed mean_delta_IC=0.007899, nw_t=1.014, BH p_adj=0.647, NOT rejected ✓
- Rank 61 CORD20+1: ranking.csv row 61 confirmed mean_delta_IC=-0.00402, nw_t=-3.579, BH p_adj=0.021, REJECTED ✓
- hc_ret_mean_5d+6 mean_delta_IC=3.22e-05: hand_curated_mapping_168.json confirmed ✓
- ARI=0.5506: adjusted_rand_index.json confirmed concern_triggered=True ✓
- 29/30 convergence: audit/convergence.json confirmed ✓
- Audit triple uniqueness: 114,558 = 114,558 unique (cell_id, group_id, date) ✓

BH-FDR recomputed via scipy and manual implementation both match `bh_fdr_p_adj` column to 1e-6. n_dates=313 across all 61 groups → balanced panel. ROC5 raw signature 2.42 confirms no pre-winsorization leakage.

## Methodology (b) — PASS with 1 documentation gap (A-05 PRDS)

NW-HAC auto lag = floor(4 × (313/100)^(2/9)) = 5, stored correctly. Künsch block_len=21, n_boot=1000, as pre-registered. CORD20+1 BH p_adj=0.021 matches BH formula at rank=K.

## Interpretation (c) — HIGHEST RISK area

A-01 (universe-policy bullets overstate) and A-02 (CORD20+1 causal overreach) are MAJOR. Both correctable with wording changes; no re-analysis needed. The 2026-04-21-c T_SPA pattern is NOT recurring (no copy-paste of per-row number into header position) but the interpretive overreach is in the same class of failure mode.

## Regime / fold sensitivity (d)

A-04 (ARI caveat not propagated) and A-06 (fold-4 sensitivity not quantified). Both addressable by adding 1-2 paragraphs and one supplementary re-aggregation (no retraining).

## Pre-registration honesty (e) — PASS

A-07 verifies: |ρ|>0.6, BH q=0.05, block_len=21, calibration days [0,251], ARI<0.85 all match plan_aaa_v1.

## Prior-art framing (f) — PASS

A-08 PASS; optional Hooker-Mentch 2019 addition would strengthen.

## Bottom line

PROCEED-WITH-FIXES. The data analysis pipeline is sound. The 2 MAJOR findings are both interpretive wording fixes in `docs/plan_aaa_results_2026-05-25.md` §5 and §7. After applying A-01, A-02, A-04 fixes (and optionally A-06 leave-fold-4-out supplementary), the doc is paper-ready.
