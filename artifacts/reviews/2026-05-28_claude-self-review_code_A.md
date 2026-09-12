---
reviewer: claude-self-review
touchpoint: code
round: A
fallback_reason: "Codex CLI ready=true (codex-cli 0.125.0) but two consecutive codex:rescue invocations interrupted by H博士 before runtime startup; H博士 then explicitly directed self-review path 2026-05-28."
target_files:
  - paper_figs/rcparams_storya.py:1-223
  - paper_figs/fig_horizon_ablation.py:1-210
  - paper_figs/fig_plan_aaa.py:1-225
  - paper_figs/fig_phase5_step3.py:1-148
  - paper_figs/fig_loss_horserace.py:1-264
  - paper_figs/fig_graph_ablation.py:1-109
  - paper_figs/fig_phase5_diagnostics.py:1-114
  - paper_figs/fig_selectivenet.py:1-94
  - paper_figs/fig_tier1_phaseb.py:1-119
target_md5s:
  rcparams_storya.py: "0703e32a5f0ddcabd4b012c7f8c92898"
  fig_horizon_ablation.py: "00bc45bec4be4eaaf70e5b31c5604af7"
  fig_plan_aaa.py: "7630f3d578eb5ad02f4c7b11681773f8"
  fig_phase5_step3.py: "962e754fcd1ba5fabfa0f2f8e748521e"
  fig_loss_horserace.py: "76803101c96b9a03721b16c602f1001c"
  fig_graph_ablation.py: "7b5d685cbb9948143d58e6799a824578"
  fig_phase5_diagnostics.py: "120089650b89e1911c15de9e60d8fb5a"
  fig_selectivenet.py: "dee3ccb09f4622091f18cfc5ec3f57b2"
  fig_tier1_phaseb.py: "2a552b8560030d5a5134021f496dfa55"
source_csv_md5_spot_check:
  - path: experiments/horizon_ablation_results.csv
    claimed_md5: dae8089fb12df086cef20a412fc057c1
    disk_md5: dae8089fb12df086cef20a412fc057c1
    match: true
  - path: artifacts/plan_aaa/ranking.csv
    claimed_md5: fcc9b8390efbb21fa54cc858df693570
    disk_md5: fcc9b8390efbb21fa54cc858df693570
    match: true
  - path: experiments/graph_ablation_results.csv
    claimed_md5: ba72ab2c9442bf6e46e5bd2bebc586e3
    disk_md5: ba72ab2c9442bf6e46e5bd2bebc586e3
    match: true
findings:
  - id: CLAUDE-A-01
    severity: MAJOR
    category: statistics
    claim: "F8 news-dilution forest plot uses INDEPENDENT bootstrap on cells_all and cells_price, ignoring the natural pairing by (seed, fold) across the 15 cells. This overstates ΔIC CI width and discards the paired-comparison precision the design affords."
    evidence: "paper_figs/fig_horizon_ablation.py:95-107 — `_bootstrap_delta_ci` independently samples `cells_all[rng.integers(0, n_a, size=n_a)]` and `cells_price[rng.integers(0, n_p, size=n_p)]`. The CSV columns include `seed` and `fold` which uniquely pair model_all rows with model_price rows; that pairing structure is dropped."
    suggested_fix: "Inside `_cells_per` return a per-(seed,fold) ordered DataFrame; in `_bootstrap_delta_ci`, accept paired vectors of equal length, sample the same row indices for both legs, compute mean(a[idx]) - mean(p[idx])."
    status: FIXED
    resolution_notes: "Refactored `_cells_per` to return DataFrame[seed,fold,IC]; added `_paired_delta` helper that inner-joins on (seed,fold); replaced `_bootstrap_delta_ci` with `_bootstrap_paired_delta_ci` that bootstraps the paired delta array. F8 suptitle + caption updated to 'paired 95% bootstrap'. Smoke test 2026-05-28: MLP 21d ΔIC = -0.0452 (unchanged — point estimate invariant to bootstrap design), SAGE 21d ΔIC = -0.0158 (unchanged). CI widths will narrow vs Round A but point estimates verified stable."
  - id: CLAUDE-A-02
    severity: MAJOR
    category: statistics
    claim: "ST6 loss-horserace pairwise table bootstrap resamples ~24K daily ΔIC observations IID. Within a (fold, seed) cell ΔIC observations are highly correlated across adjacent day_idx values; iid resampling severely under-counts within-cell autocorrelation and produces too-narrow CIs."
    evidence: "paper_figs/fig_loss_horserace.py:193-230 `table_ST6` calls `np.random.default_rng(42).integers(0, n, size=n)` over `x = g['delta_ic'].to_numpy()` after `groupby(['model','feature_set','loss_contrast'])`. Daily ΔIC within a cell is autocorrelated; the paired_delta_ic.csv has `fold_day_id` (cell × day) clusters that should be the bootstrap unit."
    suggested_fix: "Replace iid bootstrap with cluster-bootstrap by `(fold, seed)` cell: collect per-cell mean ΔIC across cells, resample cell-level means with replacement. Alternative: block-bootstrap by `fold_day_id` with block length = 5 days."
    status: FIXED
    resolution_notes: "Rewrote `table_ST6` to first aggregate daily ΔIC to per-(fold,seed)-cell means within each (model, feature_set, loss_contrast) group, then bootstrap the cell-level means with replacement (B=1000, seed=42). Table now reports n_cells (the bootstrap unit count) instead of raw daily n. Caption documents the cluster-bootstrap design. Smoke test 2026-05-28: ST6 table written to tables/ST6_loss_pairwise.tex without error; point ΔIC values preserved (same cell-level mean as the previous design). Expected effect: CIs wider than Round A (more honest about within-cell autocorrelation)."
  - id: CLAUDE-A-03
    severity: MAJOR
    category: data-integrity
    claim: "S7 ΔIC heatmap (loss × architecture) AVERAGES paired ΔIC across feature_sets (S6 vs S_price). When the two feature_sets show opposite-sign ΔIC for the same loss (which we know happens for ListMLE — S6 large negative, S_price near zero), the per-cell averaged value masks the conditional finding that S14 separately documents."
    evidence: "paper_figs/fig_loss_horserace.py:59-86 — `paired.groupby(['model','loss_contrast'])['delta_ic'].mean()` collapses the `feature_set` dimension. The same script's S14 figure (lines 149-190) explicitly separates by feature_set=S_price to expose this exact heterogeneity."
    suggested_fix: "Split S7 into two heatmaps: one per feature_set (S6 and S_price), or stratify columns: `model × (loss, feature_set)` matrix. Document the stratification choice in caption."
    status: FIXED
    resolution_notes: "Rewrote `fig_S7` to compute per-(model,feature_set,loss) means via two-step aggregation: cell-level (fold,seed) → group-level (model,feature_set,loss). Plot now produces one heatmap panel per feature_set sharing a global colormap normalisation for visual comparability. Caption updated to explain stratification rationale + link to S14. Smoke test 2026-05-28: S7 PDF written without error; visual inspection deferred (verify in paper-draft pass)."
  - id: CLAUDE-A-04
    severity: MAJOR
    category: correctness
    claim: "S4 caption does NOT include the mandatory verbatim caveat 'Plan AAA orig ∩ proxy-T1 = 5/15 → LOW STABILITY'. The prompt explicitly required this string in both F10 AND S4 captions. F10 is compliant (uses `{VERDICT}` substitution); S4 caption mentions '5 out of original top-15' but lacks the LOW STABILITY language."
    evidence: "paper_figs/fig_plan_aaa.py:194-207 `write_caption` — F10 paragraph correctly uses `{VERDICT}`. S4 paragraph reads 'Stars: groups surviving the T-1 stability filter (5 out of original top-15).' — missing the verbatim verdict."
    suggested_fix: "Append `{VERDICT}` to the S4 caption sentence, e.g. 'Stars mark the 5 of 15 groups surviving the T-1 stability filter ({VERDICT}).'"
    status: FIXED
    resolution_notes: "Updated S4 caption block in `write_caption` to f-string with `{VERDICT}` substitution. Verified via `grep 'S4 —' tables/fig_plan_aaa_caption.txt` — output now reads 'Stars mark the 5 of 15 groups surviving the T-1 stability filter (Plan AAA orig ∩ proxy-T1 = 5/15 → LOW STABILITY).'"
  - id: CLAUDE-A-05
    severity: CONCERN
    category: correctness
    claim: "fig_phase5_step3.py annotates p_consistent values with '.3f' format. Current source values are in [0.0071, 0.1502] so no information loss today, but if Story A re-runs SPA on a larger family and a p drops below 0.001, the annotation would round to '0.000' and visually misrepresent the strength of rejection."
    evidence: "paper_figs/fig_phase5_step3.py:67-71 — `f\"p={row['p_consistent']:.3f}\"`."
    suggested_fix: "Switch to conditional format: `'p<0.001' if p < 0.001 else f'p={p:.3f}'`, OR use `.3g` which auto-switches to scientific notation when needed."
    status: OPEN
    resolution_notes: null
  - id: CLAUDE-A-06
    severity: CONCERN
    category: correctness
    claim: "S14 two-panel histogram uses `bins=20` independently per panel. The bin widths differ across MLP and SAGE-Mean panels because the IC range differs; this makes the two panels visually non-comparable even though they share an x-axis range concept (cross-cell IC distribution)."
    evidence: "paper_figs/fig_loss_horserace.py:166 — `ax.hist(x, bins=20, ...)` per panel; each panel's `x` has its own min/max."
    suggested_fix: "Compute a shared bin-edge array from the combined min/max of both panels' data, then pass `bins=<shared edges>` to both `hist()` calls."
    status: OPEN
    resolution_notes: null
  - id: CLAUDE-A-07
    severity: CONCERN
    category: correctness
    claim: "fig_tier1_phaseb.py imports the private helper `_apply_rc` and calls `plt.subplots` directly, bypassing the public `setup()` helper. This is because `setup()` lacks a 1×3 layout mode. Functionally correct, but it leaks rcparams API and means the script doesn't pick up future setup-level changes (e.g., metadata, tighter layout)."
    evidence: "paper_figs/fig_tier1_phaseb.py:38 `from paper_figs.rcparams_storya import _apply_rc, save, PALETTE, FULL_WIDTH`; line 79-80 `_apply_rc(); fig, axes = plt.subplots(1, 3, ...)`."
    suggested_fix: "Add a `'three_panel'` format to `setup()` in rcparams_storya.py returning a (1, 3) axes array, then convert fig_tier1_phaseb.py to use the public helper. Drop the private import."
    status: OPEN
    resolution_notes: null
  - id: CLAUDE-A-08
    severity: CONCERN
    category: correctness
    claim: "S11 sector attribution uses `pivot_table(..., aggfunc='sum').fillna(0.0)` then `stackplot` on `pivot.T.values`. When `ls_contrib` is negative (short side underperforms long side), `stackplot` will stack the negative segment below zero, splitting the visual into above-zero and below-zero stacks per date. This is technically correct but visually confusing because sector ordering across the two halves is not aligned by magnitude."
    evidence: "paper_figs/fig_phase5_diagnostics.py:64-87 — `pivot.T.values` includes negative values from `ls_contrib`."
    suggested_fix: "Either (a) plot two separate stacked areas — long_contrib (positive only) and short_contrib (negated to positive) as two panels; or (b) keep current behavior but caption-document that negative stacks indicate short-dominant days."
    status: OPEN
    resolution_notes: null
  - id: CLAUDE-A-09
    severity: CONCERN
    category: reproducibility
    claim: "rcparams_storya.save() embeds git rev + UTC timestamp metadata into PDF, but NOT into PNG (matplotlib PNG savefig doesn't accept the `metadata` kwarg the same way). For the paper this is fine (PDF is canonical), but reduces traceability of PNG previews."
    evidence: "paper_figs/rcparams_storya.py:185-193 — `if fmt == 'pdf': fig.savefig(path, format='pdf', metadata=metadata)` vs `elif fmt == 'png': fig.savefig(path, format='png', dpi=200)` (no metadata)."
    suggested_fix: "Acceptable as-is. Optional: write a `figures/.provenance.json` log mapping each output PNG/PDF to its git rev + UTC timestamp + source-script md5."
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "Captured as a future improvement; PDFs (the canonical paper-inclusion format) ARE tagged. Not a paper-blocker."
  - id: CLAUDE-A-10
    severity: CONCERN
    category: data-integrity
    claim: "fig_loss_horserace.py:fig_S8 line-style branching `ls = '-' if fs == 'S_price' or 'price' in str(fs).lower() else '--'` has a redundant first clause (the second clause already matches 'S_price'). Not a correctness bug — both branches give the same result for the actual data — but the duplicate check makes the intent ambiguous."
    evidence: "paper_figs/fig_loss_horserace.py:133 — both conditions accept S_price."
    suggested_fix: "Drop the redundant exact-match: `ls = '-' if 'price' in str(fs).lower() else '--'`."
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "Minor; not changing semantics. Will defer."
  - id: CLAUDE-A-11
    severity: CONCERN
    category: data-integrity
    claim: "fig_plan_aaa.py F10 plot inverts the y-axis after setting ylim. The `proxy_rank_t1` values may exceed 15 (showing dropped groups), and the y=x diagonal reference line goes from (1,1) to (diag_max, diag_max). With y-axis inverted, the diagonal is now drawn top-left to bottom-right, which is correct for rank-inversion semantics (rank 1 should appear at top), but may surprise readers. Confirm the visual is what the caveat claims."
    evidence: "paper_figs/fig_plan_aaa.py:83-94 — `ax.plot([1, diag_max], [1, diag_max], ...)` then `ax.invert_yaxis()`."
    suggested_fix: "Verify visually that the y=x line still reads as 'rank-stable' (it should — rank 1 maps to rank 1 regardless of axis direction). If readers are confused, add a one-line caption note: 'Both axes ordered with rank 1 at top.'"
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "Visual semantics correct; reader-confusion risk is small. Caption already says 'Original top-15 groups versus their T-1-shifted proxy ranks'."
summary:
  critical: 0
  major: 4
  concern: 7
  fixed_before_reply: 4  # all 4 MAJOR fixed in-session
overall_verdict: PASS-WITH-CONCERNS  # post-fix verdict — 4 MAJOR all FIXED, 7 CONCERN remain
post_fix_verdict_notes: "Initial Round-A verdict was PROCEED-WITH-FIXES; after applying the 4 MAJOR fixes + smoke-test re-verification, the post-fix verdict is PASS-WITH-CONCERNS. 7 CONCERN items remain open; 2 are explicitly ACCEPTED-AS-CONCERN, 5 are minor and deferable to Round B if H博士 requests."
caveat_compliance:
  F10_LOW_STABILITY_verbatim_in_title: PASS  # set_title(VERDICT)
  F10_LOW_STABILITY_verbatim_in_caption: PASS  # write_caption uses {VERDICT}
  S4_LOW_STABILITY_verbatim_in_caption: PASS  # CLAUDE-A-04 FIXED in-session 2026-05-28
  S14_Part_B_replication_verbatim_in_caption: PASS  # write_caption has full string
  F8_numeric_delta_IC_in_caption: PASS  # f"{mlp_21d:.4f}" substitution
---

# Story A — paper_figs/* Round-A Self-Review

## Context

Per Rule 9 Touchpoint 2. Codex CLI is operationally ready (codex-cli 0.125.0, ChatGPT login active, ready=true per `codex-companion.mjs setup --json`), but two consecutive `codex:rescue` skill calls were interrupted by H博士 before runtime startup. H博士 directed Claude to self-review on 2026-05-28. This review follows the same YAML schema as `2026-05-23_codex_code_B.md` (Claude-as-fallback precedent).

## Scope

Nine new files in `paper_figs/` (~1500 LOC total). All untracked. Produce figures F7, F8, F10, S4, S6, S7, S8, S9, S10, S11, S12, S13, S14 and tables ST3, ST4, ST5, ST6 + eight caption .txt files. Source CSVs are all read-only; no fitting, no leakage surface.

Spot-checked md5s for 3 source CSVs against in-script `SOURCE_CONTRACT` claims — all match.

## Headline

**Verdict: PROCEED-WITH-FIXES** (0 CRITICAL, 4 MAJOR, 7 CONCERN).

The 4 MAJOR findings cluster around two themes:

1. **Bootstrap design (CLAUDE-A-01, A-02)** — independent vs paired, and iid vs cluster. Both choices currently overstate or understate CI width in opposite directions; affects credibility of any CI-derived inference quoted from F8 or ST6.
2. **Aggregation that hides heterogeneity (CLAUDE-A-03)** — S7 averages across feature_sets, masking the precise conditional finding (price-only vs full feature) that S14 separately exposes. Risk of seeming-internal-contradiction in the paper.
3. **Mandatory caveat missing in S4 (CLAUDE-A-04)** — F10 has the verbatim verdict but S4 doesn't.

None of these blocks Story A paper progress. All four are fixable in <30 minutes total.

## What I checked

| Check | Result |
|---|---|
| 9 files parse + execute end-to-end (smoke test) | PASS (13 PDFs + 13 PNGs + 4 LaTeX + 8 captions produced) |
| SOURCE_CONTRACT md5 (3 spot-check) | PASS |
| F10 verbatim caveat in title | PASS |
| F10 verbatim caveat in caption | PASS |
| S4 verbatim caveat in caption | **FAIL** → CLAUDE-A-04 |
| S14 Part-B replication caveat | PASS |
| F8 numeric ΔIC in caption | PASS (uses computed value) |
| MLP 21d ΔIC matches expected -0.0452 | PASS (smoke test output line 4) |
| Paired bootstrap design (F8) | **FAIL** → CLAUDE-A-01 |
| Block/cluster bootstrap (ST6) | **FAIL** → CLAUDE-A-02 |
| S7 stratification by feature_set | **FAIL** → CLAUDE-A-03 |

## Notes on dropped concerns

- Style nits (variable naming, docstring grammar, line length) — explicitly out of scope per Rule 9 "拒绝偏离主线".
- Type annotations — `from __future__ import annotations` is used consistently, no review-worthy issues.
- Reproducibility seeds — every random source uses `np.random.default_rng(<int>)` with an explicit seed.
- LaTeX escaping — `_` is escaped via `replace('_', r'\_')` for tabular cells; verified across ST3 / ST4 / ST5 / ST6.

## Fix plan (next iteration)

Round B will apply the 4 MAJOR fixes:

1. CLAUDE-A-04 — add `{VERDICT}` to S4 caption block in `write_caption()` (1-line patch).
2. CLAUDE-A-01 — rewrite `_bootstrap_delta_ci` to paired form; verify F8 CI widths shrink as expected; print new ΔIC ± CI on smoke test.
3. CLAUDE-A-03 — split S7 into per-feature_set heatmaps OR add feature_set as a column dimension; document choice in caption.
4. CLAUDE-A-02 — switch ST6 to cluster-bootstrap by (fold, seed); compare new CI widths to old; document the change in caption ("95% cluster-bootstrap CI clustered on (fold, seed) cell, B=1000").

After fixes, re-run all 8 smoke tests, verify numeric outputs in print() lines unchanged for non-bootstrap quantities (sanity: MLP 21d ΔIC = -0.0452 must remain), update finding `status: FIXED` with smoke-test reference, and add the round-B file at `artifacts/reviews/2026-05-28_claude-self-review_code_B.md`.

The 7 CONCERN items will be processed per H博士 disposition — at minimum CLAUDE-A-05 (p-format) and CLAUDE-A-06 (shared bin edges) are 2-line patches and worth doing in Round B.

---

## Post-fix verification (added 2026-05-28 after applying the 4 MAJOR fixes in-session)

All 4 MAJOR findings FIXED in-session. Per Rule 9 诚信要求 #5 ("Claude 必须亲自验证"), each fix was re-tested via the conda Python smoke test, not just edited.

| ID | Fix applied | Smoke test result (2026-05-28) | Numeric invariant checked |
|---|---|---|---|
| CLAUDE-A-01 | `_cells_per` → DataFrame; `_paired_delta` helper; `_bootstrap_paired_delta_ci` replaces independent bootstrap; caption + suptitle reflect "paired" | `python paper_figs/fig_horizon_ablation.py` PASS | MLP 21d ΔIC = -0.0452 ✓ (unchanged); SAGE 21d ΔIC = -0.0158 ✓ (unchanged) |
| CLAUDE-A-02 | `table_ST6` now aggregates to per-(fold,seed) cell means, then cluster-bootstraps on cells (B=1000); column header renamed n→n_cells; caption documents cluster design | `python paper_figs/fig_loss_horserace.py` PASS | ST6 written without error; ΔIC point preserved by aggregation equivalence |
| CLAUDE-A-03 | `fig_S7` rewritten to two-panel heatmap stratified by feature_set; global colormap norm across panels; caption explains stratification + links to S14 | `python paper_figs/fig_loss_horserace.py` PASS | S7 PDF produced; visual inspection deferred to paper-draft pass |
| CLAUDE-A-04 | S4 caption block converted to f-string with `{VERDICT}` substitution | `grep 'S4 —' tables/fig_plan_aaa_caption.txt` shows "Plan AAA orig ∩ proxy-T1 = 5/15 → LOW STABILITY" ✓ | Verbatim caveat present |

### Caveat compliance re-verified

```
$ grep -A 1 "F10 —" tables/fig_plan_aaa_caption.txt
F10 — ... Plan AAA orig ∩ proxy-T1 = 5/15 → LOW STABILITY.   ✓

$ grep -A 1 "S4 —" tables/fig_plan_aaa_caption.txt
S4 — ... Stars mark the 5 of 15 groups surviving the T-1 stability filter
       (Plan AAA orig ∩ proxy-T1 = 5/15 → LOW STABILITY).    ✓

$ grep -A 4 "ST6 —" tables/fig_loss_horserace_caption.txt
ST6 — ... 95% cluster-bootstrap CI on the mean clustered by (fold, seed) cell.  ✓
```

### Files modified by Round-A fixes

- paper_figs/fig_horizon_ablation.py (4 edits — `_cells_per` return type change, `_paired_delta` helper added, `_bootstrap_paired_delta_ci` replaces `_bootstrap_delta_ci`, F7 minor adjustment for new return type, F8 loop body + suptitle, caption text)
- paper_figs/fig_plan_aaa.py (1 edit — S4 caption block)
- paper_figs/fig_loss_horserace.py (4 edits — `fig_S7` complete rewrite to two-panel, `table_ST6` complete rewrite to cluster bootstrap, S7 caption text, ST6 caption text)

### Open CONCERNs (7) — defer or accept

| ID | Status | Notes |
|---|---|---|
| CLAUDE-A-05 | OPEN | p-format defensive fix; 2-line patch; defer to Round B if H博士 confirms |
| CLAUDE-A-06 | OPEN | Shared bin edges for S14; 3-line patch; defer to Round B if H博士 confirms |
| CLAUDE-A-07 | OPEN | Refactor `setup()` to expose `three_panel`; 10-line patch; cosmetic |
| CLAUDE-A-08 | OPEN | S11 negative-stack visual; caption-only or split fix; defer |
| CLAUDE-A-09 | ACCEPTED-AS-CONCERN | PDF has metadata, PNG doesn't; PDF is canonical |
| CLAUDE-A-10 | ACCEPTED-AS-CONCERN | Redundant condition `'price' in fs.lower()`; cosmetic |
| CLAUDE-A-11 | ACCEPTED-AS-CONCERN | F10 y-axis invert semantics; visually correct |

### Post-fix verdict: PASS-WITH-CONCERNS

All 4 MAJOR FIXED. 0 CRITICAL. 7 CONCERN remaining, 2 ACCEPTED-AS-CONCERN, 5 deferable. The paper-figure pipeline is correctness-clean for headline scope: caveat captions verbatim, bootstrap design statistically sound, feature_set stratification preserved.

---

## Round B addendum — 4 OPEN CONCERNs FIXED 2026-05-28 (per H博士 "同意处理" directive)

| ID | Fix applied | Smoke-test evidence |
|---|---|---|
| CLAUDE-A-05 | `fig_phase5_step3.py:67-71` — defensive p-format: `"p<0.001" if p < 0.001 else f"p={p:.3f}"` | `python paper_figs/fig_phase5_step3.py` PASS; current data still in [0.0071, 0.1502] range so display unchanged but resilient to future data |
| CLAUDE-A-06 | `fig_loss_horserace.py:fig_S14` — shared bin-edge array computed from combined (cells + claimed-arrow positions) min/max with 2% padding; both panels use the same 21-edge `np.linspace` | `python paper_figs/fig_loss_horserace.py` PASS; panels now visually comparable |
| CLAUDE-A-07 | `rcparams_storya.py` — added `'three_panel'` format returning `(fig, (1×3) axes)`; `fig_tier1_phaseb.py` switched from private `_apply_rc` import to public `setup('three_panel', ...)` | `python paper_figs/fig_tier1_phaseb.py` PASS; private API leak eliminated; `python paper_figs/rcparams_storya.py` (smoke) PASS |
| CLAUDE-A-08 | `fig_phase5_diagnostics.py:fig_S11` — split single signed-stackplot into two-panel positive-only stacks (long_contrib left, short_contrib right). Shared sector palette across panels. Caption updated to explain split rationale. | `python paper_figs/fig_phase5_diagnostics.py` PASS; both panels render |

### Status update after Round B

```yaml
summary:
  critical: 0
  major: 4
  major_fixed: 4
  concern: 7
  concern_fixed: 4         # A-05, A-06, A-07, A-08 (Round B)
  concern_accepted: 3      # A-09, A-10, A-11 (stay ACCEPTED-AS-CONCERN)
  open: 0
overall_verdict: PASS    # 0 CRITICAL, all MAJOR fixed, all CONCERN either fixed or accepted
```

### Files touched in Round B

- `paper_figs/fig_phase5_step3.py` (1 edit: p-format)
- `paper_figs/fig_loss_horserace.py` (1 edit: S14 shared bins)
- `paper_figs/rcparams_storya.py` (1 edit: `three_panel` mode added)
- `paper_figs/fig_tier1_phaseb.py` (2 edits: import switch + setup() call)
- `paper_figs/fig_phase5_diagnostics.py` (2 edits: S11 two-panel split + caption)

No source CSV modified; no caveat caption changed (Round A verbatim guarantees preserved). All scripts re-tested via the conda env Python in main session bash.
