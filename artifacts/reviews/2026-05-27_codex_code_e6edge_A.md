---
reviewer: codex
touchpoint: code
round: A
target_files:
  - /Users/heruixi/Desktop/GNN-Testing/analyze_e1_lofo.py:165-235
  - /Users/heruixi/Desktop/GNN-Testing/compute_e6_edge_ablation.py
findings:
  - id: CODEX-CR-EDGE-A-01
    severity: CONCERN
    category: correctness
    claim: "Length-mismatch between paired configs A/B for the same fold uses warning+truncate rather than hard error"
    evidence: "compute_e6_edge_ablation.py:156-161 if len(avgA) != len(avgB): n = min(...); print WARN; truncate. Smoke test confirms 0 mismatches across all 4 configs × 5 folds (313/251/62 days as expected), so production-safe; but silent truncation could mask a future temporal-contract violation if E3/E4 are re-run with mismatched calendar configuration."
    suggested_fix: "Change to raise ValueError(f'Fold {fold} length mismatch A={len(avgA)} B={len(avgB)} — likely indicates temporal contract violation; abort post-processing'). Locks in the structural invariant. H博士 decided fix-now to be loud rather than silent."
    status: FIXED
    resolution_notes: |
      FIXED 2026-05-27. Changed compute_e6_edge_ablation.py:156-161 from
      truncate+WARN to raise RuntimeError with detailed message. Same-length
      invariant (within-fold across configs) is now hard-enforced. Re-ran the
      full script after fix — all outputs identical (no mismatch in current
      data, as expected). The next time a config is added or re-run, any
      calendar drift will surface as a loud error, not silently corrupt the
      paired ΔIC series.
  - id: CODEX-CR-EDGE-A-02
    severity: OK
    category: correctness
    claim: "α3 filter 'corr+news_cooccur' matches E3 results.csv exactly; α1 baseline correctly reused across multiple pair comparisons (paired-design)"
    status: NOTED
  - id: CODEX-CR-EDGE-A-03
    severity: OK
    category: statistics
    claim: "DM/HLN applied AFTER seed aggregation (T = pooled test days: 313/251/62 for full/lofo4/fold4_only), matching plan §1.4(b) seed-aggregation policy + Codex Round D D-04 fix"
    status: NOTED
  - id: CODEX-CR-EDGE-A-04
    severity: OK
    category: statistics
    claim: "HLN small-sample correction formula numerically verified: T=313, h=21 → factor=0.9345, matches reference ≈0.935"
    status: NOTED
  - id: CODEX-CR-EDGE-A-05
    severity: OK
    category: statistics
    claim: "BH-FDR is standard Benjamini-Hochberg 1995 step-up; applied only to full condition's 5 HLN p-values (paper-headline test family)"
    status: NOTED
  - id: CODEX-CR-EDGE-A-06
    severity: OK
    category: statistics
    claim: "IC bootstrap block_size=21 (per-day autocorr); Sharpe bootstrap block_size=1 (per-cell exchangeable). Assumptions match"
    status: NOTED
  - id: CODEX-CR-EDGE-A-07
    severity: OK
    category: correctness
    claim: "Sharpe paired difference keyed on (seed, fold); all 50 keys present in both arms of each pair (verified by smoke test)"
    status: NOTED
  - id: CODEX-CR-EDGE-A-08
    severity: OK
    category: statistics
    claim: "fold4_only skipping DM/HLN is justified: T=62, h=21 → HLN factor=0.669 (over-correction in small-T regime); bootstrap CI remains meaningful"
    status: NOTED
  - id: CODEX-CR-EDGE-A-09
    severity: OK
    category: reproducibility
    claim: "Output filenames consistent (edge_*); seed=86 default in stationary_bootstrap_ci ensures reproducibility; script doesn't train models so no other random state"
    status: NOTED
summary:
  critical: 0
  major: 0
  concern: 1
  ok: 8
  fixed_before_reply: 1
overall_verdict: PASS-WITH-CONCERNS
---

# Codex Round A Review — compute_e6_edge_ablation.py + analyze_e1_lofo.py §5

**Verdict: PASS-WITH-CONCERNS** (0 CRITICAL + 0 MAJOR + 1 CONCERN, now FIXED before reply).

PIT contract intact (seed aggregation BEFORE pairing per §1.4(b)). HLN small-sample formula numerically verified. BH-FDR family scoped correctly to 5 pairs × full condition only. Bootstrap block_size choices match per-day vs per-cell exchangeability assumptions. fold4_only T=62 too small for HLN (factor 0.669 over-corrects); bootstrap CI is the correct fallback, no DM/HLN reported for that condition.

## Single CONCERN, FIXED

**CR-EDGE-A-01 (CONCERN → FIXED)**: Length-mismatch between paired configs A/B for same fold was using `print WARN; truncate to min length` (compute_e6_edge_ablation.py:156-161). Smoke test confirms 0 mismatches in current data, but silent truncation could mask future temporal-contract violations. Per H博士 decision, changed to `raise RuntimeError` — loud rather than silent. Re-ran script after fix; outputs identical (no current mismatch).

## Headline findings (for context)

- Full 5-fold: 0/5 pairs reject BH-FDR at q=0.05. Smallest HLN p = 0.039 (α3 news vs α1).
- LOFO-4: edge benefit ΔIC drops to +0.002 to +0.005; all HLN p > 0.30.
- Fold-4-only: α2/α3/α4 vs α1 ΔIC CIs all positive and exclude 0 ([+0.022, +0.038]).

Conclusion: GNN edge augmentation provides regime-specific economic value concentrated in Q2-2025; full-sample claims do not survive multi-testing correction.

Ready for Codex Touchpoint 3 (results review).
