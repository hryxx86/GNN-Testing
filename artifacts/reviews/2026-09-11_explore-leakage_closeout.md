<!-- Rule 9 session-closeout audit 1/4 (Explore agent, independent context), 2026-09-11 ~03:25 local. Scope = this
session's diff eb8314e..a903c5e (C5 sensitivity code + housekeeping commit 46b3b8c). Status of the finding filled in by
Claude after implementing the fix (see resolution_notes). -->
---
reviewer: explore-leakage
touchpoint: closeout
round: closeout
target_files:
  - run_storya_e1_anchor.py
  - run_storya_v21_main12.py
  - run_storya_v21_tune.py
  - run_v21_tune_launcher.py
  - compute_family1_ladder.py
  - analyze_c5_sensitivity.py
  - compute_e6_dm_spa.py
  - run_storya_e3_news_edge.py
  - run_storya_e4_alpha.py
findings:
  - id: EXPL-LEAK-01
    severity: CONCERN
    category: correctness
    claim: "The paired C-minus-C5 / B-minus-C5 daily contrast pairs two pooled day series purely by position and guards that pairing with a single total-length equality assert (disabled under --smoke). `_delta_series` truncates the L1/L0 series to min(len) over the whole pooled 12-fold series, so a single short cell would chop the tail of fold 11 instead of the affected fold and silently shift every subsequent day. The C5 side is checked cell by cell against the frozen calendar in run_integrity, but the confirmatory C/B series are never checked per fold — only their aggregate length. No leakage is created and this session's numbers are provably unaffected (240/240 cells, zero cells_not_full_calendar_length, frozen_calendar_total 749; T_days 749 for C5, C and B), but the invariant 'same 749 test days in the same chronological order' is inferred transitively rather than asserted."
    evidence: "analyze_c5_sensitivity.py:177-191, :454-461; compute_family1_ladder.py:136-156; artifacts/storya_v21_family1_c5/c5_run_integrity.json"
    suggested_fix: "Pair per fold: build both sides with seed_avg_per_fold, assert len(series[f]) == fold_calendar_days[f] for BOTH the C5 and the confirmatory universe for every fold, then concatenate the aligned per-fold differences. Under --smoke, apply the same check or refuse to emit c5_paired_contrast.csv rather than falling back to min() truncation."
    status: FIXED
    resolution_notes: "Implemented in analyze_c5_sensitivity.py (_delta_per_fold / _delta_series(calendar) / paired_contrast(calendar=...)): in strict (non-smoke) mode every fold of both arms of BOTH universes must equal the frozen-calendar day count (raise otherwise); the paired series is the concatenation of per-fold differences. Re-run on the T4 primary and the Mac replicate: numbers unchanged (as the audit predicted). Smoke mode still tolerates partial folds (wiring check only) — c5_paired_contrast.csv from a smoke run is labelled by the --smoke integrity gate (PASS=False)."
summary:
  critical: 0
  major: 0
  concern: 1
  fixed_before_reply: 0
overall_verdict: PASS
---

**Verdict body (agent, verbatim summary).** PASS — no new leakage in the C5 runtime pipeline. C5 is a pure by-name column selection of the already T-1-shifted Universe C tensor (elementwise-asserted at construction), and every temporal mechanism downstream (fold masks, 21d purge, train-only winsorize/standardize, frozen corr snapshot, label definition) is universe-agnostic shared code that C5 reaches through the exact same call path as C. The tuning window (train ≤ 2022-06-30, val 2022H2, both purged by 21 days) stays strictly before 2023Q1. One CONCERN on defensive alignment in the new analysis script, with evidence that this session's outputs are unaffected.

**What the agent read.** Full session diff eb8314e..a903c5e for the six target files plus the 46b3b8c housekeeping diff, and the shared runtime the C5 path delegates to (build_universe_C / build_labels / build_correlation_snapshots / get_frozen_snapshot_idx / create_fold_masks / winsorize_train_only / standardize_train_only / train_nn / compute_daily_ic in run_storya_e1_anchor.py; assert_purge_no_leak_12 + assert_univ_c_t1_contract + fold loop in run_storya_v21_main12.py; build_data_ctx / eval_config / run_study in run_storya_v21_tune.py; collect_arm_matrix / seed_avg_pooled / seed_avg_per_fold / run_ci_and_mde in compute_family1_ladder.py), cross-checked against the committed artifacts.

**Verified clean.** Temporal handling C5 vs C (np.roll T-1 shift + row-0 zero inside build_universe_C before selection; per-column elementwise equality; all 20 names ⊂ UNIVERSE_C_ALPHA158_NAMES; hc_ excluded); winsorize/standardize per fold on purged train_days in both main12 and tune; no new rolling/rank/ffill; corr snapshot ≤ purged last train day (tune asserts snapshot_points[frozen_si] ≤ train_days[-1]); identical WALK_FORWARD_FOLDS_12 + assert_purge_no_leak_12; tuning window purged and strictly pre-2023 with HP selection scored only on purged 2022H2 val days; tune seeds disjoint from canonical seeds; Rule 8 label definition untouched; cell_id block disjoint; launcher merge keeps frozen_hparams.json byte-identical; --sensitivity only suppresses BH/SPA/L7/robustness with pairs restricted never extended; housekeeping 46b3b8c is inference-only / ID arithmetic with tuple-keyed resume.

**Explicitly not flagged.** The test-informed selection of the 20 columns is pre-disclosed (docs/analysis.md 2026-09-11-a; runtime provenance; every table header) — the session's known limitation, not a new finding. The sensitivity-mode degeneracy_report ref_arms fallback is a documented CODEX-TP2-A-02 tradeoff whose failure mode is independently caught by run_integrity.
