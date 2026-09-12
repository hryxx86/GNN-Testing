---
handoff_date: 2026-06-24
last_completed: "2026-06-23-b: 6 confirmatory paper figures rebuilt (nature-figure + scientific-schematics) + triple QA + bilingual self-contained HTML gallery; legends given opaque white bg, §5.1 legend moved outside axes (2026-06-24 polish)."
in_flight:
  - id: paper-results-rewrite
    file: docs/storya_paper_draft.md
    status: "NOT started. §Results/§Discussion must be rewritten around the 6 rebuilt confirmatory figures + the two-family + cost framing. Each data-figure LaTeX caption MUST carry the ML-stats block (seeds=10 / folds=12 / metric / CI definition / baseline=L0) — nature-figure QA requirement."
    blockers: []
  - id: prior-work-figs-disposition
    file: paper_figs/
    status: "NOT decided. The OLD prior-work figs (horizon F7/F8, loss S7/S8, graph-ablation S9, diagnostics S10/S11, selectivenet S12, tier1 S13, plan-AAA F10/S4, step3 S6) were DELETED in the 2026-06-23 cleanup. Their data is unchanged (not 'wrong') — decide per figure: rebuild as exploratory-appendix (sans style) or drop. Scripts still exist in paper_figs/ (old fig_*.py)."
    blockers: ["paper-results-rewrite (structure decides what's needed)"]
open_questions:
  - "Prior-work figs (horizon/loss/graph-ablation/plan-AAA/step3/selectivenet/tier1): rebuild as 'exploratory' appendix figures, or drop from the confirmatory paper?"
  - "C/L5s 27.5% collapse + SPA C=0.077 wording: apply the LOCKED discipline when writing §Results (C/L5s folds into the 'smoothing hurts ranking' mechanism para, NOT its own §; SPA 0.077 = 'fail to reject', NEVER 'near-significant', add the MDE-underpowered qualifier)."
file_state:
  modified_uncommitted:
    - "docs/analysis.md (2026-06-21-a §5 cost crosswalk + TL;DR gross/net annotation)"
    - "progress.md (2026-06-21-b cost, 2026-06-23-a/b figures), plan.md (Decision Log: cost layer + figure/font decisions)"
  new_files:
    - "compute_cost_confirmatory.py + artifacts/storya_v21_cost/ (cost gross/net crosswalk, 6 outputs + README)"
    - "artifacts/reviews/2026-06-21_codex_code_cost_A.md, 2026-06-21_codex_results_cost_A.md"
    - "paper_figs/: fig_headline_ic.py, fig_f9_confirmatory.py, fig_regime.py, fig_cost.py, fig_family2.py, fig_pipeline.py, build_gallery.py (+ updated rcparams_storya.py: sans-Arial global)"
    - "figures/ (6 rebuilt confirmatory PNG+PDF + figure_gallery.html self-contained bilingual + README), tables/ (cleared, README only)"
rule9_status:
  touchpoint_2_code: PASSED        # cost analyzer: 2026-06-21_codex_code_cost_A (PASS-WITH-CONCERNS, 1 CRIT rejected w/ evidence + 2 fixed)
  touchpoint_3_results: PASSED     # cost results: 2026-06-21_codex_results_cost_A (PASS-WITH-CONCERNS, A-07 fixed + A-02 wording)
  figure_qa: PASSED                # triple QA (Codex correctness + nature-figure QA contract + manual visual) — recorded in progress 2026-06-23-b; plotting scripts are not Rule-9 correctness-critical code, so no separate reviews/ file
next_actions:
  - "Rewrite paper §Results/§Discussion on the 6 confirmatory figures (two-family + cost framing); add ML-stats captions."
  - "Decide prior-work figure disposition (open_question 1)."
  - "Commit this session: compute_cost_confirmatory.py + artifacts/storya_v21_cost + paper_figs/ + figures/ + tables/ + the 3 tri-docs + 2 review files."
---

# Session Handoff — 2026-06-24

## TL;DR
The **cost-口径 BLOCKING item is DONE** and the **paper figure system is rebuilt from scratch** on the confirmatory data. The confirmatory result itself (D-RERUN-12F: Family-1 predictive null + Family-2 underpowered causal edge) was already complete and T3-passed before this session; this session added the missing **net-of-cost crosswalk** and produced a clean, QA'd, bilingual **figure set + gallery**. What remains is **writing the paper §Results/§Discussion** around these figures.

## What this session did

### 1. Cost-口径 (gross/net) crosswalk — the BLOCKING completion item
- New analyzer `compute_cost_confirmatory.py` → `artifacts/storya_v21_cost/` (per-arm net ladder, 20-pair fold-level ΔSharpe + LOFO, FC net, headline gross/net crosswalk + per-fold deltas). **DESCRIPTIVE layer; IC stays the sole confirmatory metric** (no 3rd BH-FDR family on Sharpe). Net口径 headline = 10bps; old E1/E6/pilot Sharpe NOT mixed in.
- **Findings**: C-MLP>LGB and "graph doesn't help" HOLD at net@10bps (CIs exclude 0); the IC "news hurts" (C L3−L2) does NOT reproduce at net (CI straddles 0, fold-fragile) → flagged cost-sensitive, NOT evidence news helps. → `docs/analysis.md` 2026-06-21-a §5.
- Rule 9: T2 + T3 both PASS-WITH-CONCERNS (see reviews). All Codex findings verified by hand (1 CRITICAL rejected with code+data evidence).

### 2. Paper figures — full rebuild
- **Old 2026-05-28 batch DELETED** (pilot data + low quality). Global figure font **LOCKED = sans-Arial** (`paper_figs/rcparams_storya.py`).
- **6 confirmatory figures** built via the `nature-figure` workflow (conclusion-first + self-review) and `scientific-schematics` (pipeline): `headline_ic_ladder` (§5.1), `F9_spa_dm_confirmatory` (§5.2), `regime_perfold_ic` (§5.3), `cost_gross_net` (§5.4), `family2_edge_causal` (§5.5), `pipeline_confirmatory` (Methods). Each script supports `--font serif`.
- **Triple QA** (Codex code/data correctness + nature-figure QA contract + manual visual): all overlaps fixed; legends given opaque white bg; §5.1 legend moved outside the axes so it covers no CI line.

### 3. Bilingual figure gallery
- `figures/figure_gallery.html` — **self-contained** (base64-embedded images, ~1 MB single file, opens anywhere). Per-figure 中文/English: 是什么 / 怎么看 / 结论 + data source.
- Generated by `paper_figs/build_gallery.py` — **to add/update a figure: edit the `FIGS` list and re-run** (gallery stays self-contained).

## How to regenerate (quick reference)
```bash
PY=/opt/homebrew/Caskroom/miniforge/base/envs/gnn/bin/python
$PY paper_figs/fig_headline_ic.py            # §5.1   (--font serif for the alt)
$PY paper_figs/fig_f9_confirmatory.py        # §5.2
$PY paper_figs/fig_regime.py                 # §5.3
$PY paper_figs/fig_cost.py                   # §5.4
$PY paper_figs/fig_family2.py                # §5.5
$PY paper_figs/fig_pipeline.py               # Methods
$PY paper_figs/build_gallery.py              # → figures/figure_gallery.html (self-contained)
```

## Reading red lines (unchanged from 2026-06-21)
- Two SEPARATE confirmatory families: Family-1 = predictive/model-selection (SPA/DM); Family-2 = causal edge (matched-ΔIC). Do not interchange their primaries.
- Net Sharpe is DESCRIPTIVE; IC is the SOLE confirmatory metric.
- SPA C=0.077 is "fail to reject"; NEVER "near-significant" — pair with the MDE-underpowered qualifier.
- DM pairwise rejections are LOCAL ladder rungs, NOT global superiority over LightGBM.
- C/L5s 27.5% collapse is a reported stability finding (EXCLUDE primary), folds into the "smoothing hurts ranking" mechanism para; never re-tuned.

## What's NOT done (next session)
1. **Paper §Results/§Discussion rewrite** on the 6 figures + two-family + cost framing; ML-stats captions (seeds/folds/metric/CI/baseline).
2. **Prior-work figure disposition** — rebuild as exploratory appendix vs drop (open_question 1).
3. **Commit** — large uncommitted surface (analyzer + artifacts + paper_figs/ + figures/ + tables/ + 3 tri-docs + 2 reviews).

## Key paths
- Cost: `compute_cost_confirmatory.py`, `artifacts/storya_v21_cost/`
- Figures: `paper_figs/fig_*.py`, `paper_figs/build_gallery.py`, `figures/figure_gallery.html`, `figures/README.md`
- Confirmatory data: `experiments/storya_v21_main12_tuned/`, `artifacts/storya_v21_family1/`, `artifacts/storya_v21_family2_fc/`
- Reviews: `artifacts/reviews/2026-06-21_codex_{code,results}_cost_A.md`
