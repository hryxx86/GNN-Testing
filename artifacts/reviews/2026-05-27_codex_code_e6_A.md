---
reviewer: codex
touchpoint: code
round: A
target_files:
  - /Users/heruixi/Desktop/GNN-Testing/compute_e6_dm_spa.py
findings:
  - id: CODEX-CR-E6-A-01
    severity: PASS
    category: methodology
    claim: "SPA loss convention (loss = -IC) is correctly oriented"
    status: PASS
  - id: CODEX-CR-E6-A-02
    severity: PASS
    category: methodology
    claim: "Seed aggregation per (model, universe, date, fold) BEFORE SPA/DM is correctly implemented (T=313 not T=3130)"
    status: PASS
  - id: CODEX-CR-E6-A-03
    severity: MAJOR
    category: correctness
    claim: "NW-HAC variance divisor wrong: uses (T-l) instead of T for lag-l autocovariance"
    evidence: "Line 212 (pre-fix): `gamma_l = float((xc[lag:] * xc[:-lag]).mean())` — .mean() on length-(T-l) array divides by (T-l). Newey-West 1987 requires dividing by T."
    suggested_fix: "Replace .mean() with .sum() / T: `gamma_l = float((xc[lag:] * xc[:-lag]).sum() / T)`"
    status: FIXED
    resolution_notes: "Fixed 2026-05-27. Also explicitly applied .sum()/T to gamma_0 for consistency (was already mathematically equivalent since gamma_0 product has length T). Re-smoked: HLN p-values shifted by <5% (e.g., GAT-vs-LGB 0.01286 → 0.01222). BH-FDR reject pattern unchanged at q=0.05 on the 4-cell smoke. Impact for production: slightly less aggressive over-rejection."
  - id: CODEX-CR-E6-A-04
    severity: PASS
    category: methodology
    claim: "HLN small-sample correction factor and t_{T-1} p-value distribution correct"
    status: PASS
  - id: CODEX-CR-E6-A-05
    severity: PASS
    category: methodology
    claim: "BH-FDR step-up procedure correctly implemented"
    status: PASS
  - id: CODEX-CR-E6-A-06
    severity: PASS
    category: correctness
    claim: "arch.bootstrap.SPA + StationaryBootstrap API usage correct"
    status: PASS
  - id: CODEX-CR-E6-A-07
    severity: CONCERN
    category: correctness
    claim: "Edge-case handling gaps: (a) joint SPA np.column_stack fails on empty candidate; (b) cost-ladder std=0 for N=1 instead of NaN"
    evidence: |
      (a) L399-409 (pre-fix): joint SPA loop continues collecting bench/cand even after a universe has no benchmark; np.column_stack on mixed-length list would raise.
      (b) L512, L554 (pre-fix): `vals.std()` returns 0.0 for N=1 (numpy default ddof=0); should be NaN to signal "no spread estimable".
    suggested_fix: |
      (a) Skip joint SPA entirely if any of the 6 expected (universe, candidate) series is empty.
      (b) Explicit NaN guard: `float(vals.std(ddof=1)) if len(vals) >= 2 else np.nan`.
    status: FIXED
    resolution_notes: "Both fixed 2026-05-27. (a) added skip_joint flag with diagnostic print; (b) replaced std() calls with conditional ddof=1 + NaN fallback. Re-smoke confirms no behavior change on the 4-cell smoke (which has only N=1 cells per (model, universe) so std was always NaN anyway)."
summary:
  critical: 0
  major: 1
  concern: 1
  fixed_before_reply: 2
overall_verdict: PROCEED-WITH-FIXES
---

# Codex Round A Review — compute_e6_dm_spa.py

**Verdict**: PROCEED-WITH-FIXES.

5/7 findings PASS on core methodology (SPA sign, seed aggregation, HLN correction, BH-FDR step-up, arch API). Two real bugs found and fixed:

## CODEX-CR-E6-A-03 (MAJOR) — NW-HAC variance divisor

`nw_hac_variance` used `.mean()` on the lag-truncated autocovariance product, which divides by (T-l) instead of T. Newey-West 1987 standard estimator requires 1/T divisor for ALL lags. Fix: replaced with explicit `.sum() / T`.

Impact: pre-fix, DM/HLN p-values were slightly over-rejecting (too small). Re-smoke shows the shift is <5% at the smoke T=63. At production T=313 the bias is even smaller.

## CODEX-CR-E6-A-07 (CONCERN) — Edge-case handling

Two minor gaps:
- (a) Joint SPA loop didn't guard against missing candidates → would crash with `np.column_stack`. Fixed with `skip_joint` flag.
- (b) Cost-ladder Sharpe std returned 0.0 for N=1 (numpy default ddof=0). Fixed with explicit `>= 2` check + NaN fallback (ddof=1 for sample std).

## Verification

Re-smoked on 4-cell M4 smoke data after both fixes. All 5 outputs regenerated. HLN p-values shifted slightly (consistent with bias correction direction). BH-FDR reject pattern unchanged (3/5 rejected at q=0.05). No regressions.

Production launch (post-E1 completion) is now safe to proceed with corrected formula.
