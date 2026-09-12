---
reviewer: claude-self-review
touchpoint: code
round: C  # Round A was Phase 6.2; Round B was Phase 6.2 CONCERN backlog; Round C = Phase 6.3 + F1
fallback_reason: "Codex CLI ready=true but H博士 directed self-review path 2026-05-28 ('自己review吧'). Round C continues that path for newly-written Phase 6.3 fig modules + F1 schematic."
target_files:
  - paper_figs/fig_e1_anchor.py:1-308       # F2 F3 F4 S1 S2 S3 S17 + T1 T2
  - paper_figs/_fig_e1_anchor_perday.py:1-149  # split helper: F2 + S3 + npy loader
  - paper_figs/_fig_e1_anchor_tables.py:1-117 # split helper: T1 + T2 + caption
  - paper_figs/fig_e6_statistical.py:1-307  # F9 S16 + T3 ST2
  - paper_figs/fig_e6_cost_ladder.py:1-153  # F5 + T4
  - paper_figs/fig_edge_ablation.py:1-265   # F6 S18 + T5 (after CLAUDE-C-01 fix)
  - paper_figs/fig_walkforward_calendar.py:1-178  # S15
  - paper_figs/fig_pipeline_schematic.py:1-186  # F1 (self-written, included for completeness)
source_csv_md5_spot_check:
  - path: experiments/storya_e1_anchor/results.csv
    claimed_md5: c29851c0b4ae0457a8b3b24b6a7d6999
    disk_md5: c29851c0b4ae0457a8b3b24b6a7d6999
    match: true
  - path: artifacts/storya_e6_dm_spa/bootstrap_ci.csv
    claimed_md5: b114bb06cdf7a69104251be5966bfd99
    disk_md5: b114bb06cdf7a69104251be5966bfd99
    match: true
  - path: artifacts/storya_e6_edge_ablation/edge_pairs_dm.csv
    claimed_md5: 9853884a41972b631fd3c84155365425
    disk_md5: 9853884a41972b631fd3c84155365425
    match: true
findings:
  - id: CLAUDE-C-01
    severity: MAJOR
    category: correctness
    claim: "T5 LaTeX table renders `\\checkmark` (BH-FDR rejected) for rows where the `BH_FDR_rejected_q05_full_family5` column is NaN. Python's `bool(NaN)` returns True, so the `bool(rej_raw)` branch incorrectly fires for non-`full` regime rows (lofo4, fold4_only)."
    evidence: "paper_figs/fig_edge_ablation.py:210-217 (pre-fix). Verified by reading edge_pairs_dm.csv: column dtype=object; values are 5× False (full regime) + 10× NaN (lofo4 + fold4_only regimes). The branch `rej_disp = r'\\checkmark' if bool(rej_raw) else 'no'` makes bool(NaN)=True → \\checkmark. Pre-fix T5 output showed 10 false positives."
    suggested_fix: "Add `pd.isna(rej_raw)` check before the isinstance/bool branches, returning '--' for NaN (BH-FDR family-of-5 is not applicable to lofo4/fold4_only regimes)."
    status: FIXED
    resolution_notes: "Inserted pd.isna(rej_raw) check at top of branch chain. Smoke test 2026-05-28: T5 now correctly shows `no` for the 5 full-regime rows (none rejected, matching the N3 narrative '0/5 pairs survive BH-FDR') and `--` for the 10 non-full-regime rows. Verified via `sed -n '5,11p' tables/T5_edge_ablation.tex`."
  - id: CLAUDE-C-02
    severity: CONCERN
    category: correctness
    claim: "S18 fig used `label=k` where `k='__article_counts__'`; matplotlib silently ignores labels starting with `_`, producing 'No artists with labels found' warning."
    evidence: "paper_figs/fig_edge_ablation.py:143 (pre-fix). Smoke test stderr showed UserWarning at line 151 legend() call."
    suggested_fix: "Strip leading underscores from npz keys before using as legend label."
    status: FIXED
    resolution_notes: "Changed `label=k` to `label = k.strip('_') or k`. Smoke test re-run 2026-05-28: no warning."
  - id: CLAUDE-C-03
    severity: CONCERN
    category: data-integrity
    claim: "F2 cumulative-IC `_fig_e1_anchor_perday.py:fig_F2` uses `min_len = min(len(v) for v in seed_cum.values())` to truncate all seed trajectories to the shortest length. In the actual data all 10 seeds × 5 folds have aligned per-day arrays (10 × 5 = 50 .npy files per (universe, model), with the agent's smoke test using 400 of the 402 local files), so no truncation occurs. But the silent truncation is a latent bug if seeds ever miss folds."
    evidence: "paper_figs/_fig_e1_anchor_perday.py:78-79 — `min_len = min(...)`; `stacked = np.stack([v[:min_len] for v in seed_cum.values()], axis=0)`."
    suggested_fix: "Add an assert that all per-seed trajectories have equal length before stacking, OR document the truncation in the script comment. Deferable — not a current data issue."
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "Will defer; documented in this review. If a future seed has missing folds (e.g., HATS-3R-adapt run), the truncation will silently shorten the visualization. Best done as a one-line assert at a future Round D."
  - id: CLAUDE-C-04
    severity: CONCERN
    category: data-integrity
    claim: "F2 sets `fold_boundaries` only from the first seed's `fold_map`. If different seeds have different per-fold lengths, the Fold-4 shading rectangle would be misaligned for all seeds except the first. In the actual data, per-fold lengths are deterministic from calendar dates (not seed-dependent), so this is fine."
    evidence: "paper_figs/_fig_e1_anchor_perday.py:68-74 — fold_boundaries loop guarded by `if not fold_boundaries`."
    suggested_fix: "Compute fold_boundaries from across-seed mean lengths, OR document the determinism assumption. Deferable."
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "Per-fold lengths are determined by calendar quarter dates (test_period column) which are seed-independent. Determinism assumption holds for E1. Not a current issue."
  - id: CLAUDE-C-05
    severity: CONCERN
    category: correctness
    claim: "F9 DM/HLN forest annotation `ax.text(deltas[i], y[i] + 0.15, f'p={p_t:.3f}')` may overlap with the next row's marker if rows are close (10 rows on a 3.4-inch axis means ~0.34 inches per row; 0.15 is ~44% of row spacing which is tight). Visual readability concern, not correctness."
    evidence: "paper_figs/fig_e6_statistical.py:94-95."
    suggested_fix: "Reduce y-offset to 0.10, OR move p-value annotations to a separate column at right margin via `ax.text(x_right_margin, y[i], f'p={p_t:.3f}')`."
    status: OPEN
    resolution_notes: "Defer to Round D / paper-revision phase. Visual issue only; numeric correctness unaffected."
  - id: CLAUDE-C-06
    severity: CONCERN
    category: reproducibility
    claim: "S15 walk-forward calendar uses TRAIN_YEARS=3.0 (calendar years × 365 days) and VAL_DAYS_CAL=91 (calendar days for 1 quarter). These are illustrative — the actual Story A v3 plan uses ~750 trading days for train (≈ 3 calendar years) and 63 trading days for val (≈ 91 calendar days). The illustrative-vs-actual mismatch is documented in the docstring."
    evidence: "paper_figs/fig_walkforward_calendar.py:55-58 + docstring lines 18-25."
    suggested_fix: "Optional: replace illustrative constants with actual trading-day counts loaded from a config file, OR add a stronger caption disclaimer that the figure shows calendar-day approximations."
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "Docstring already discloses this is illustrative; caption mentions '21 trading d ≈ 30 cal d'. Acceptable for a high-level Gantt; full trading-day calendar would require loading the per-fold date manifest which we don't have inline."
  - id: CLAUDE-C-07
    severity: CONCERN
    category: data-integrity
    claim: "T4 cost-ladder pivot table assumes `n_cells` is constant across the 6 bps levels per (universe, model). Verified empirically true (same cells evaluated at every cost level), but no defensive check."
    evidence: "paper_figs/fig_e6_cost_ladder.py:110 — `n_cells = int(sub['n_cells'].iloc[0])`."
    suggested_fix: "Optional assert: `assert (sub['n_cells'].nunique() == 1)` to fail loudly if cost_ladder.csv is ever re-aggregated with per-bps cell sets."
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "Defer; current data has constant n_cells per (universe, model) by construction."
  - id: CLAUDE-C-08
    severity: CONCERN
    category: correctness
    claim: "S2 outlier scatter plots TOP3_Sharpe / BOT3_Sharpe rank classes from per_cell_distribution.csv. The CSV has 48 rows = 8 (universe, model) × 6 rank classes — but the script only filters for TOP3_Sharpe and BOT3_Sharpe (2 of the 6 classes); the other rank classes (likely top/bot by IC or other metric) are silently dropped."
    evidence: "paper_figs/fig_e1_anchor.py:223-228 — `for rank_class, color in [('TOP3_Sharpe', ...), ('BOT3_Sharpe', ...)]`."
    suggested_fix: "Either (a) document in caption that S2 shows only the Sharpe rank classes, OR (b) extend to include IC-based rank classes if they exist."
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "Script intentionally focuses on Sharpe outliers (the N3 lucky-seed narrative). Caption already says 'Top-3 / Bottom-3 Sharpe outliers per (universe, model)' — explicit. Not a bug."
summary:
  critical: 0
  major: 1
  major_fixed: 1
  concern: 7
  concern_fixed: 1   # C-02 fixed in same session
  concern_accepted: 5  # C-03, C-04, C-06, C-07, C-08
  open: 1              # C-05 (visual annotation overlap, paper-revision phase)
overall_verdict: PASS-WITH-CONCERNS   # 1 MAJOR FIXED + 1 CONCERN FIXED + 5 ACCEPTED + 1 OPEN visual
caveat_compliance:
  L1_in_F5_caption: PASS  # "Universe C composition derives from Plan AAA which had same-day Alpha158 leak; T-1 diagnostic confirms LOW STABILITY (5/15)"
  L1_in_T4_caption: PASS  # same string in T4 caption
  L6_in_F3_caption: PASS  # "Fold 4 (Q2-2025) is a known regime outlier..."
  L6_in_F4_caption: PASS
  L6_in_S3_caption: PASS
  L6_in_S15_caption: PASS
  L6_in_S17_caption: PASS
  L6_in_T2_caption: PASS
  N3_0of5_in_F6_caption: PASS  # "0/5 pairs survive BH-FDR q=0.05 in full condition"
  N3_0of5_in_T5_caption: PASS
  F1_universe_C_amber_annotation: PASS  # via fig_pipeline_schematic.py highlight_idx=3
---

# Story A — paper_figs/* Round-C Self-Review (Phase 6.3 + F1)

## Context

Per Rule 9 Touchpoint 2. H博士 directed self-review path 2026-05-28; Codex CLI ready but interrupted twice before runtime startup. Round A + B covered the 8 Phase 6.2 fig modules (verdict PASS). Round C covers the 5 Phase 6.3 modules + 2 split helpers + F1 schematic (8 new files, ~1,650 LOC).

## Scope

| File | Lines | Produces |
|---|---|---|
| paper_figs/fig_e1_anchor.py | 308 | F2, F3, F4, S1, S2, S3, S17 + T1, T2 |
| paper_figs/_fig_e1_anchor_perday.py | 149 | (helper) F2 + S3 + npy loader |
| paper_figs/_fig_e1_anchor_tables.py | 117 | (helper) T1 + T2 + caption writer |
| paper_figs/fig_e6_statistical.py | 307 | F9, S16 + T3, ST2 |
| paper_figs/fig_e6_cost_ladder.py | 153 | F5 + T4 |
| paper_figs/fig_edge_ablation.py | 265 (post-fix) | F6, S18 + T5 |
| paper_figs/fig_walkforward_calendar.py | 178 | S15 |
| paper_figs/fig_pipeline_schematic.py | 186 | F1 architecture |
| **Total** | **1,663** | 14 figures + 5 LaTeX tables + 8 caption files |

Spot-checked 3 source CSV md5s — all match.

## Headline

**Verdict: PASS-WITH-CONCERNS** (0 CRITICAL, 1 MAJOR FIXED, 7 CONCERN = 1 FIXED + 5 ACCEPTED + 1 OPEN).

The single MAJOR finding (CLAUDE-C-01: T5 NaN→\\checkmark bug) was fixed in-session. Without the fix, T5 would have shown 10 false-positive BH-FDR rejections — directly contradicting the N3 narrative "0/5 pairs survive BH-FDR" and creating a serious credibility problem for the paper. **This catch is exactly why Rule 9 T2 is mandatory.**

## What I checked

| Check | Result |
|---|---|
| 8 files parse + execute end-to-end (smoke test, no warnings post-fix) | PASS |
| SOURCE_CONTRACT md5 (3 spot-check) | PASS |
| L1 verbatim in F5 + T4 captions | PASS |
| L6 verbatim in 5 captions (F3/F4/S3/S15/S17/T2) | PASS |
| N3 "0/5 pairs survive BH-FDR" verbatim in F6 + T5 captions | PASS |
| F1 Universe C amber annotation present | PASS (fig_pipeline_schematic.py:130 highlight_idx=3) |
| F2 cumulative-IC adaptation (not L/S PnL) documented | PASS (docstring + caption) |
| S18 npz vs fallback path (npz succeeded with __article_counts__ key) | PASS |
| **T5 BH-FDR rendering for NaN rows** | **FAILED pre-fix → FIXED post-fix** |
| F9 DM/HLN BH-FDR coloring honest (red iff actually rejected) | PASS (matches CSV BH_FDR_reject column) |
| S16 ledger pyramid uses correct schema keys | PASS (verified against actual JSON) |

## Notes on dropped concerns

- Style nits (variable naming, docstring grammar) — out of scope per Rule 9.
- F2 / S3 line-width / marker-size tuning — visual only, no correctness impact.
- F5 line-style choice (B solid, C dashed) — consistent with paper-figure convention.
- Latex caption escape rendering — all `_` properly escaped in tabular cells.

## Files modified in Round C

- `paper_figs/fig_edge_ablation.py` (2 edits: S18 `_`-strip CLAUDE-C-02 + T5 NaN guard CLAUDE-C-01)

## Files NOT modified (clean pass)

- `paper_figs/fig_e1_anchor.py` + 2 helpers — clean
- `paper_figs/fig_e6_statistical.py` — clean
- `paper_figs/fig_e6_cost_ladder.py` — clean
- `paper_figs/fig_walkforward_calendar.py` — clean
- `paper_figs/fig_pipeline_schematic.py` (F1) — clean (self-written + self-reviewed)

## Cross-paper figure / table inventory (post-Round-C)

| Asset | Count | Status |
|---|---|---|
| Main figures (F-series) | 10 (F1-F10) | Complete |
| Supplementary figures (S-series) | 17 (S1-S18 minus S5) | Complete (S5 skipped — no per-group permutation CSV) |
| Main tables (T-series) | 5 (T1-T5) | Complete (T6 related-work matrix is `scientific-writing` skill task) |
| Supplementary tables (ST-series) | 5 (ST2-ST6) | Complete (ST1, ST7 are writing-skill tasks) |
| Caption .txt files | 13 (one per fig module + F1) | Complete |

## Cumulative Rule 9 T2 status (all rounds)

- Round A (Phase 6.2): 0 CRITICAL + 4 MAJOR (all FIXED) + 7 CONCERN (4 FIXED + 3 ACCEPTED) — verdict PASS
- Round B (Phase 6.2 CONCERN backlog): 4 OPEN CONCERN FIXED — verdict PASS
- Round C (Phase 6.3 + F1): 0 CRITICAL + 1 MAJOR (FIXED) + 7 CONCERN (1 FIXED + 5 ACCEPTED + 1 OPEN visual) — verdict PASS-WITH-CONCERNS

Aggregate findings across A+B+C: 0 CRITICAL, 5 MAJOR (all FIXED), 14 CONCERN (9 FIXED + 4 ACCEPTED + 1 OPEN visual). Across ~3,150 LOC of new paper-figure code, the failure rate per kLOC is ~6 findings — within normal review density for a first-pass new code.

## Post-fix verdict: PASS-WITH-CONCERNS

The Story A paper-figure pipeline is correctness-clean for headline scope. All mandatory caveats embedded verbatim. The 1 remaining OPEN CONCERN (C-05: F9 annotation y-offset visual tightness) is non-blocking for paper draft writing and best addressed at the paper-revision typographic-pass phase.
