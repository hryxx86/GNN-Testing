---
reviewer: codex
touchpoint: results
round: A
date: 2026-06-13
target_files:
  - artifacts/storya_e6_dm_spa/spa_results.csv
  - artifacts/storya_e6_dm_spa/dm_hln_results.csv
  - artifacts/storya_e6_dm_spa/bootstrap_ci.csv
  - artifacts/storya_e6_dm_spa/cost_ladder.csv
  - artifacts/storya_e6_dm_spa/summary.md
  - artifacts/storya_e6_dm_spa/multiple_testing_ledger.json
  - compute_e6_dm_spa.py
context: "12-fold (T=749 pooled test days) formal stats on the Story A anchor (960 cells). Reviews
  the recomputed Hansen SPA / DM-HLN+BH-FDR / block-bootstrap CI / cost-ladder and Claude's proposed
  interpretation. All 7 MAJOR + 2 CONCERN independently verified by Claude against the artifacts."
findings:
  - id: R9-A-01
    severity: MAJOR
    category: over_interpretation
    claim: "Non-rejection (SPA B/C/JOINT p=0.29/0.34/0.47; DM B all p>0.28) stated as 'null holds / proven equal / +0.01 was a 5-fold artifact'. That is absence-of-evidence, not evidence-of-absence."
    suggested_fix: "Reword to 'no statistically reliable evidence that predefined-edge models improve 21d cross-sectional IC over the prespecified baselines.' The +0.01 Univ-B edge is UNRESOLVED, not disproven."
    status: ACCEPTED
    resolution_notes: "Verified: I overstated in the H博士 report. Will adopt the no-reliable-evidence framing in analysis.md and all paper text. Not yet written to analysis.md (held pending this review)."
  - id: R9-A-02
    severity: MAJOR
    category: estimand
    claim: "SPA vs LightGBM is a strong-baseline test, NOT a clean graph-edge test. The edge-specific contrast is GAT/SAGE vs the non-graph MLP."
    suggested_fix: "Present two distinct claims: (1) no candidate reliably beats LightGBM; (2) graph-edge models do not reliably beat non-graph MLP. The GAT/SAGE-vs-MLP DM rows carry the edge interpretation."
    status: ACCEPTED
    resolution_notes: "Verified in dm_hln_results.csv: B GAT-MLP +0.0038 p=0.37, B SAGE-MLP +0.0037 p=0.39 (no edge benefit in B); C GAT-MLP -0.0135 p=0.0002 (edge harms), C SAGE-MLP -0.0057 p=0.067. The edge-test framing is the correct primary lens."
  - id: R9-A-03
    severity: MAJOR
    category: methodology
    claim: "Joint SPA (M=6, p=0.466) compares universe-specific B.*/C.* candidates against a B/C-AVERAGED LightGBM benchmark — not a clean pooled test."
    suggested_fix: "Either pool each architecture across B+C (M=3 pooled GAT/SAGE/MLP vs pooled LightGBM) or bootstrap six matched universe-specific loss differentials. Do not headline the current joint p."
    status: ACCEPTED
    resolution_notes: "Verified in run_spa_per_universe joint branch (compute_e6_dm_spa.py:421-438): bench_pool = mean of per-universe LGB losses, candidates kept universe-specific. Will reconstruct as matched differentials or de-emphasize. Pending H博士 scope decision."
  - id: R9-A-04
    severity: MAJOR
    category: statistics
    claim: "Block-bootstrap IC CIs pool seed×day (N=7490) from 10 NON-independent seeds; blocks don't span seed boundaries → anti-conservative (CIs ~3x too narrow)."
    suggested_fix: "Use seed-AVERAGED T=749 series for headline IC CIs (matches SPA/DM estimand), or hierarchical/date-block bootstrap. Add paired-difference CIs."
    status: ACCEPTED
    resolution_notes: >
      VERIFIED by recompute: seed-averaged CIs are 2.5-3.5x WIDER and mostly overlap 0.
      B/LightGBM seedavg [-0.0027,0.0461] vs stacked [0.0146,0.0300]; C/GAT [-0.0139,0.0507] vs [0.0074,0.0290];
      C/LightGBM [-0.0007,0.0546] vs [0.0183,0.0354]. The seed-stacked CI is statistically wrong for IC inference.
      FIX: switch run_bootstrap_ci headline to the seed-averaged series; keep seed-stacked only as a training-
      randomness diagnostic. Pending H博士 nod (changes paper central uncertainty numbers).
  - id: R9-A-05
    severity: MAJOR
    category: power
    claim: "Non-rejection of the +0.01 B edge reflects low power (~17-20% for a true +0.01 effect at T=749, ~36 effective blocks), not true absence. MDE for 80% power ≈ 0.025-0.028 IC."
    suggested_fix: "Add a power/MDE section. State the design cannot confirm a +0.01 IC edge; do not call it disproven."
    status: ACCEPTED
    resolution_notes: "Order-of-magnitude consistent (delta +0.01 with p~0.3 ⟹ low power). Will add an explicit MDE computation. Strengthens the paper by making the limitation transparent."
  - id: R9-A-06
    severity: MAJOR
    category: over_generalization
    claim: "GAT<MLP in Univ C (p=0.0002, BH-FDR reject) is real locally but must not be generalized to 'graph harms'. B GAT-MLP/SAGE-MLP are positive & non-significant; C SAGE-MLP p=0.067 not rejected."
    suggested_fix: "Minimal claim: 'In Univ C, GAT underperforms MLP on 21d IC (DM p=0.0002); inconsistent with a robust edge benefit. Univ B shows no such pattern.'"
    status: ACCEPTED
    resolution_notes: "Verified cross-universe asymmetry in dm_hln_results.csv. Will bound the claim to Univ C / GAT."
  - id: R9-A-07
    severity: MAJOR
    category: factual_error
    claim: "Claude's cost-ladder claim ('neural incl MLP beats LightGBM in both universes') is FALSE for Univ C; and the turnover causal story is contradicted."
    suggested_fix: "Present net Sharpe as a separate economic endpoint with paired tests + actual turnover/concentration decomposition. Investigate the C-GAT anomaly (IC 0.018 lowest, net Sharpe 1.29 highest)."
    status: ACCEPTED
    resolution_notes: >
      VERIFIED in cost_ladder.csv: Univ C @10bps GAT 1.29 > LightGBM 0.65 > MLP 0.31 > SAGE 0.30 (MLP/SAGE
      BELOW LGB). My 'both universes' claim was wrong. C-GAT per-cell: IC 0.0182 (lowest), turnover 2.92
      (HIGHEST), Sh_gross 1.82 (highest) → turnover story is contradicted; it is an IC-vs-Sharpe divergence
      (good extreme-decile selection, poor average rank corr). Needs explicit decomposition before citing.
  - id: R9-A-08
    severity: CONCERN
    category: independence
    claim: "SPA and DM use the SAME T=749 seed-averaged series → correlated, not independent confirmation. Don't say 'DM confirms SPA'."
    suggested_fix: "Declare hierarchy: SPA primary, DM secondary localization, cost-ladder economic secondary."
    status: ACCEPTED
    resolution_notes: "Correct. Will state DM 'supports/localizes' rather than 'independently confirms'."
  - id: R9-A-09
    severity: CONCERN
    category: robustness
    claim: "DM HAC uses NW_lag=6 (Newey-West auto) for T=749 while horizon h=21 and block=21 → may understate serial covariance at lags 7-20."
    suggested_fix: "Add NW_lag=21 (or block-bootstrap paired-difference) sensitivity. Confirm B non-rejections and C GAT-MLP p survive."
    status: ACCEPTED
    resolution_notes: "Standard multi-horizon reviewer request. Will add a lag=21 sensitivity row."
summary:
  critical: 0
  major: 7
  concern: 2
  accepted: 9
  rejected: 0
overall_verdict: CONDITIONAL_PASS — all 9 findings accepted; no claim may enter analysis.md/paper until R9-A-01/04/05/07 framing+stats fixes applied
---

# Codex Results Review — 12-fold formal null (anchor), Round A

Rule 9 Touchpoint 3 on the recomputed 12-fold (T=749) Hansen SPA / DM-HLN+BH-FDR / block-bootstrap CI /
cost-ladder. Verdict **CONDITIONAL_PASS**: 0 CRITICAL, 7 MAJOR, 2 CONCERN — all independently verified
by Claude against the artifacts and ALL ACCEPTED. No false positives.

The computation is sound; the issues are (a) over-strong interpretation, (b) one anti-conservative CI
methodology (seed-stacking), (c) one factual error by Claude (cost-ladder Univ C), and (d) missing
power/MDE + HAC-lag robustness that a sharp ICAIF referee would demand.

Honest corrected headline: across 3 years / 12 quarterly folds, there is **no statistically reliable
evidence that predefined graph-edge models improve 21-day cross-sectional IC** over the prespecified
baselines (LightGBM, and the non-graph MLP). The small (+0.01) Univ-B neural lead is statistically
**unresolved** (power ~17-20%), not disproven. In Univ C, GAT is significantly worse than MLP (DM
p=0.0002), inconsistent with a robust edge benefit. Net-Sharpe is a separate economic endpoint with an
unexplained C-GAT anomaly that must be decomposed before citation.

Required before any paper claim: apply R9-A-01 (framing), R9-A-04 (seed-averaged headline CI),
R9-A-05 (power/MDE section), R9-A-07 (cost-ladder correction + C-GAT decomposition).
