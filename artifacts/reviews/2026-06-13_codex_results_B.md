---
reviewer: codex
touchpoint: results
round: B
date: 2026-06-13
prior_round: artifacts/reviews/2026-06-13_codex_results_A.md
target_files:
  - compute_e6_dm_spa.py:run_headline_seedavg_ci_and_power
  - compute_e6_dm_spa.py:dm_test/hln_test (lag param), run_spa_per_universe (role)
  - analyze_cgat_anomaly.py
  - artifacts/storya_e6_dm_spa/{headline_ic_ci_seedavg.csv,pairwise_power_mde.csv,dm_hln_results.csv,spa_results.csv,cgat_anomaly.md,summary.md}
roundA_disposition:
  - {id: R9-A-01, status: FIXED, note: "framing to 'no reliable evidence' adopted (applied in analysis.md write-up)"}
  - {id: R9-A-02, status: FIXED, note: "edge test = GAT/SAGE vs MLP flagged via is_edge_test col in pairwise_power_mde.csv"}
  - {id: R9-A-04, status: FIXED, note: "headline_ic_ci_seedavg.csv = seed-averaged T=749 CIs (~3x wider, half include 0); seed-stacked relabeled DIAGNOSTIC in summary.md"}
  - {id: R9-A-06, status: FIXED, note: "GAT<MLP bounded to Univ C in write-up"}
  - {id: R9-A-08, status: FIXED, note: "summary + write-up state DM localizes, not independently confirms"}
  - {id: R9-A-09, status: FIXED, note: "HLN_p_t_lag21 column; C GAT-MLP survives (0.0002→0.0048<0.05); B stays non-sig"}
findings:
  - id: R9-B-01
    severity: CONCERN
    category: factual_error
    claim: "cgat_anomaly.md said GAT has 'highest turnover (2.92)' but MLP=2.97 > GAT=2.92."
    status: FIXED
    resolution_notes: "Verified: turnover GAT 2.9238 < MLP 2.9663. Reworded to 'high, WELL ABOVE LightGBM 2.46' — the relevant comparison is GAT-vs-LightGBM (GAT trades MORE than LightGBM yet wins net), which holds. No 'highest' claim. Re-ran; md updated."
  - id: R9-B-02
    severity: MAJOR
    category: consistency
    claim: "pairwise_power_mde.csv power used auto-lag NW SE, inconsistent with the R9-A-09 lag=21 sensitivity; the 'edge test well-powered' claim leaned on the optimistic auto-lag SE."
    status: FIXED
    resolution_notes: >
      Added SE_lag21 / power_at_delta_0.01_lag21 / MDE_80pct_power_lag21 columns. Material implication:
      under the conservative lag=21, edge-test power for +0.01 drops 0.68-0.91 → 0.41-0.74 (MDE 0.011-0.016),
      and vs-LightGBM drops 0.18-0.49 → 0.11-0.33 (MDE 0.018-0.041). The reframe survives but is CALIBRATED:
      the edge test (GAT/SAGE vs MLP) is MODERATELY/better-powered, NOT 'high-powered'. Core conclusions hold:
      C GAT-MLP significant at both lags (p 0.0002 / 0.0048); C SAGE-MLP non-sig at 74% lag-21 power = genuine
      no-benefit. The analysis.md headline will quote the conservative (lag-21) MDE range.
  - id: R9-A-03
    severity: MAJOR
    category: methodology
    claim: "Joint SPA pooled-benchmark construction not a clean matched test."
    status: FIXED
    resolution_notes: "Disposition = DOWNGRADE (Codex accepted). spa_results.csv gains role column ('primary' per-universe / 'supplementary' joint); summary.md prints a note. Per-universe SPA is primary evidence."
  - id: R9-A-05
    severity: MAJOR
    category: power
    claim: "Power/MDE present but auto-lag only (Round B partial)."
    status: FIXED
    resolution_notes: "Closed by R9-B-02 lag-21 columns. Power formula Φ(δ/SE−z)+Φ(−δ/SE−z) and MDE=(z.975+z.80)·SE verified correct by Codex Round B."
  - id: R9-A-07
    severity: MAJOR
    category: interpretation
    claim: "C-GAT cost-ladder anomaly (Round B partial: turnover wording)."
    status: FIXED
    resolution_notes: "Decomposition sound (Codex confirmed): C-GAT gross Sharpe 1.82 dominated by Fold-4 (Q2-2025, n=3, Sharpe 13.18); LOFO-best collapses 1.82→0.79; regime-concentrated small-sample artifact, not tradeable edge; IC null unaffected. Turnover wording fixed (R9-B-01). Decile attribution flagged as needing return-logging re-run (not available)."
summary:
  roundA: {fixed: 6, partially_fixed: 3, still_open: 0}
  roundB_new: {major: 1, concern: 1}
  all_now: FIXED
overall_verdict: PASS — all Round A + Round B findings FIXED and self-verified; cleared to write analysis.md with the calibrated (lag-21) framing
---

# Codex Results Review — 12-fold formal null, Round B (verification)

Round B verified the Round A fixes: **6 FIXED, 3 PARTIALLY-FIXED, 0 STILL-OPEN, 2 NEW** (R9-B-01 CONCERN,
R9-B-02 MAJOR). All 5 residual items addressed and self-verified in-session:

- **R9-B-02 (MAJOR)** added lag-21 power columns — this CALIBRATED the reframe: the edge test (GAT/SAGE
  vs MLP) is moderately/better-powered (lag-21 power 0.41-0.74, MDE 0.011-0.016), not "high-powered", but
  still clearly better than the vs-LightGBM benchmark comparison (power 0.11-0.33, MDE 0.018-0.041). The
  analysis.md headline quotes the conservative lag-21 MDE.
- **R9-B-01 (CONCERN)** turnover "highest" → "high, above LightGBM" (GAT 2.92 < MLP 2.97; the GAT-vs-LightGBM
  comparison that matters still holds).
- **R9-A-03** joint SPA downgraded to supplementary (role column + summary note).

Net honest picture cleared for write-up: (1) per-universe SPA — no model reliably beats LightGBM, but this
benchmark test is under-powered (MDE ~0.018-0.041); (2) the edge-specific test (graph vs non-graph MLP) is
better-powered (MDE ~0.011-0.016) and shows no edge benefit in B, significant harm in C (robust to HAC lag);
(3) IC CIs (seed-averaged) are wide — half the models incl. LightGBM not reliably IC>0; (4) C-GAT net-Sharpe
"win" is a Fold-4 small-sample artifact. No claim of proven equality; non-rejections framed as unresolved /
no-reliable-evidence.
