---
reviewer: codex
touchpoint: results
round: A
target_files:
  - artifacts/storya_v21_cost/cost_headline_crosswalk.csv
  - artifacts/storya_v21_cost/cost_pairwise_dsharpe.csv
  - artifacts/storya_v21_cost/cost_ladder_by_arm.csv
  - artifacts/storya_v21_cost/cost_fc_dsharpe.csv
  - artifacts/storya_v21_cost/cost_ledger.json
  - artifacts/storya_v21_cost/cost_summary.md
findings:
  - id: CODEX-A-07
    severity: MAJOR
    category: reproducibility
    claim: "The 'C L3-L2 per-fold ΔSharpe = 6 pos / 6 neg, range -2.8..+2.6' statement is not source-citable — no persisted artifact held the 12 per-fold values (only aggregate columns existed)."
    evidence: "cost_pairwise_dsharpe.csv had n_fold_blocks/lofo_min_mean/lofo_max_mean/sign_stable but not the 12 per-fold ΔSharpe values; the per-fold count/range was computed only in an ad-hoc check."
    suggested_fix: "Persist the 12 per-fold ΔSharpe values as a supplementary artifact and cite it, OR drop the precise per-fold count/range from the docs."
    status: FIXED
    resolution_notes: >
      Added cost_pairwise_folddeltas.csv (per universe × pair × cost × fold ΔSharpe, with test_period).
      Re-run confirms C L3-L2 @10bps = [-0.4161,0.0596,2.5860,2.4004,-2.7851,0.8692,-0.1286,-2.1289,
      -0.3012,0.2151,-0.1744,0.7892] → 6 pos / 6 neg, min -2.7851, max 2.586. The analysis.md
      statement now cites cost_pairwise_folddeltas.csv (provenance per docs.md §4).
  - id: CODEX-A-02
    severity: CONCERN
    category: interpretation
    claim: "'C-MLP beats LightGBM strengthens at net' overstates — the ΔSharpe erodes slightly with cost (0bps 1.2447 > 10bps 1.1733 > 30bps 1.0217); and the heavy-tailed C/MLP per-arm mean must be disclosed."
    evidence: "cost_pairwise_dsharpe.csv C L1-L0: dSharpe 0bps=1.2447, 10bps=1.1733, 30bps=1.0217. cost_ladder_by_arm.csv C/L1: Sharpe_net_mean=0.954 vs median=0.182, max_abs_Sharpe_gross_cell=31.69."
    suggested_fix: "Use 'holds/survives' not 'strengthens'; note ΔSharpe erodes mildly with cost (higher MLP turnover); disclose heavy tail but state the RELATIVE ranking rests on the fold-level paired ΔSharpe (CI excludes 0), not the per-arm mean."
    status: ACCEPTED
    resolution_notes: >
      Accepted — applied in the analysis.md wording: 'C-MLP > tuned LightGBM HOLDS at net@10bps
      (ΔSharpe +1.17, CI [+0.36,+2.08] excludes 0, LOFO-stable); LGB's own net Sharpe is negative
      (-0.22) while MLP's is strongly positive (+0.95), so the economic separation is starker than the
      +0.015 IC gap — though the ΔSharpe itself erodes mildly with cost (1.24→1.17→1.02 over 0→10→30bps)
      as MLP's higher turnover (2.90 vs 2.25) is paid down. C/MLP's per-arm net Sharpe is heavy-tailed
      (mean 0.95 vs median 0.18, max cell |Sharpe|=31.7); the MLP>LGB ranking rests on the fold-level
      paired ΔSharpe (CI excludes 0), not the per-arm mean magnitude.'  No 'strengthens' anywhere.
summary:
  critical: 0
  major: 1
  concern: 1
  fixed_before_reply: 1
overall_verdict: PASS-WITH-CONCERNS
---

# Review body — Touchpoint 3, cost-口径 results (artifacts/storya_v21_cost/)

Codex Round A: **APPROVE_WITH_FIXES** (0 critical, 1 major, 1 concern, 5 pass). Codex opened every CSV
+ family1_dm_hln.csv and re-verified each headline number. After resolution: A-07 FIXED (fold-delta
artifact added), A-02 ACCEPTED (docs wording). Post-fix verdict **PASS-WITH-CONCERNS**.

## PASS items (Codex verified, used as-is)
- **Pre-registration framing**: net Sharpe DESCRIPTIVE only, IC the sole confirmatory metric, no
  BH-FDR on Sharpe, 10bps headline, 33-cell C/L5s EXCLUDE — all matched the locked plan/ledger.
- **C L1-L0 (MLP>LGB)**: net ΔSharpe@10bps +1.1733, CI [+0.356,+2.079] excludes 0, LOFO-stable — a
  defensible SUPERIORITY claim (not equivalence); ranking robust despite the heavy tail (see A-02).
- **C L2-L1 (graph doesn't help)**: net −0.7208, CI [−1.421,−0.102] excludes 0 → holds at net.
- **C L3-L2 (news)**: gross IC harm (−0.01231, BH-reject) does NOT reproduce at net — net +0.0821,
  CI [−0.771,+0.887] STRADDLES 0. Codex confirmed the honest reading: indistinguishable from zero,
  flagged cost-sensitive, **NOT** evidence news helps economically. No non-superiority→equivalence or
  near-zero→"helps" error.
- **C L5-L3 (+sector)**: net +0.8786, CI [+0.118,+1.651] excludes 0.
- **FC arm**: all 6 contrasts net CI straddle 0 — consistent with Family-2 (0/6 BH, 6/6 underpowered).

## Resolution
- A-07 (MAJOR) → FIXED: cost_pairwise_folddeltas.csv added; per-fold detail now source-citable.
- A-02 (CONCERN) → ACCEPTED: 'strengthens'→'holds'; ΔSharpe-erodes-with-cost + heavy-tail disclosed;
  ranking attributed to fold-level paired ΔSharpe, not per-arm mean.

No CRITICAL. No interpretation error of the 2026-04-21-c (non-superiority→equivalence) class. Cleared
to write conclusions into docs/analysis.md with provenance.
