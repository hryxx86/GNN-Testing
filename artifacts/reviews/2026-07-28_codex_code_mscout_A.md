---
reviewer: codex
touchpoint: code
round: A
model: gpt-5.6-sol (xhigh)
target_files:
  - analyze_m_scout_step1.py
target_plan: docs/prereg_m_scout_2026-07-28.md
findings:
  - id: CODEX-M2-A-01
    severity: MAJOR
    category: prereg-fidelity
    claim: "S4 implements equal-weighted large/small tercile returns, not the frozen S4 VW-weighted construction."
    evidence: "analyze_m_scout_step1.py:553-560 computes tercile membership from dollar volume but then uses .mean() for rS/rL; CONFIG choice (e) froze VW weights."
    suggested_fix: "Within each S4 small/large tercile, compute returns with trailing 63d dollar-volume weights, normalized inside the tercile."
    status: FIXED
    resolution_notes: "Claude verified lines 559-560 (.mean() = EW, contradicting CONFIG (e)) and fixed exactly per suggested_fix: np.average with sub_dv[t] weights within each tercile, EW fallback only on degenerate zero-weight groups; docstring EW→VW corrected. Smoke re-run clean."
  - id: CODEX-M2-A-02
    severity: MAJOR
    category: correctness
    claim: "S4 includes an incomplete final weekly bin labeled after the last daily observation."
    evidence: "analyze_m_scout_step1.py:540-543 resamples to W-FRI and immediately computes weekly returns; panel ends 2026-01-28 (Wed) so a partial 2026-01-30 bin is created."
    suggested_fix: "Drop incomplete trailing weekly bins before pct_change and align dv_wk to the filtered weekly index."
    status: FIXED
    resolution_notes: "Fixed per suggested_fix (trailing-bin drop + dv_wk reindex to filtered index). Verified on smoke: panel ends Wed 2022-05-25 → last weekly bin now Fri 2022-05-20 (complete), partial 2022-05-27 bin gone."
  - id: CODEX-M2-A-03
    severity: MAJOR
    category: correctness
    claim: "summary.json can be non-standard JSON because NaN values are emitted directly."
    evidence: "S3 NaN CI fields dumped via default json.dump; smoke summary.json contained bare NaN tokens."
    suggested_fix: "Recursively convert non-finite floats to None and dump with allow_nan=False so invalid JSON fails loudly."
    status: FIXED
    resolution_notes: "Added _json_sanitize (non-finite→None, numpy scalars normalized) + allow_nan=False on both summary.json and config.json. Verified: strict json.load succeeds on regenerated smoke output."
summary:
  critical: 0
  major: 3
  concern: 0
  fixed_before_reply: 0
overall_verdict: BLOCK-EXECUTION
---

# Review body (Codex, verbatim excerpt)

The confirmatory `--full` run should remain blocked. The primary daily signal
path is mostly clean: the label follows the cited paper-1 construction, daily
characteristics are T-1 aligned, S1's correlation window ends at `t-1`, the
branch rule is recorded before secondary tests, and BH is applied to exactly
four secondary p-values. The blockers are concentrated in secondary/output
integrity: S4 forms dollar-volume terciles but computes equal-weighted tercile
returns (conflicts with the frozen VW requirement); S4 admits a partial final
weekly return because the data end mid-week; summary.json is not strict JSON
when NaNs are present.

# Claude disposition (2026-07-28)

All 3 MAJOR verified against the cited lines and fixed by implementing Codex's
suggested_fix verbatim (agreement by adoption); smoke re-run + strict-JSON
parse + weekly-bin check all pass. Primary/branch-rule path was explicitly
cleared by the review. BLOCK-EXECUTION therefore discharged → `--full`
confirmatory executed post-fix (1107 IC days; outputs in
`experiments/m_scout_step1/`). Results interpretation gated on Touchpoint 3.
