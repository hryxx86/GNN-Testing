---
handoff_date: 2026-05-27
handoff_type: paper-figure-and-experiment-mapping-plan
last_completed: "2026-05-27-d: Codex Touchpoint 1 PROCEED-WITH-FIXES on HATS-3R-adapt baseline plan (parallel session); 2026-05-27-e: Honesty pass — analysis.md SPA/CI corrections + plan.md Decision Log rows; 2026-05-27-f: this handoff"
in_flight:
  - id: figure-and-table-plan
    file: docs/session_handoff_2026-05-27_storya_paper_plan.md
    status: "draft handoff complete; awaiting H博士 review before script implementation"
    blockers: []
  - id: storya-paper-figure-scaffolding
    file: analyze_storya_results.py (to be written)
    status: "queued; depends on plan approval + restart-for-nature-skill-loading"
    blockers: ["plan approval", "nature-* skills load on next session"]
  - id: hats-baseline-reproduction
    file: run_hats_baseline.py (to be written)
    status: "scheduled per plan §1.6 GO 2026-05-27"
    blockers: ["~1-1.5 weeks effort budget"]
open_questions:
  - "ALL 4 ORIGINAL OPEN QUESTIONS RESOLVED 2026-05-27 — locked in plan.md Decision Log + this frontmatter for audit:"
  - "Q1 RESOLVED → 图表全出 = exhaustive 25 figures + 13 tables (upper bound; trim at writing time per venue page budget)"
  - "Q2 RESOLVED → 13 modular paper_figs/fig_*.py scripts (Option Y precedent; shared rcparams_storya.py import)"
  - "Q3 RESOLVED → ICAIF 2026 ACM SIG primary (highest-prestige feasible for AI+finance 8-week timeline); QF journal backup if rejected"
  - "Q4 RESOLVED → HATS-3R-adapt (Codex T1 PROCEED-WITH-FIXES; narrowed claim_scope per A-05/A-06/A-07)"
  - "NEW open question 2026-05-27-h: Drive-sync strategy for Story A 400 .npy per-day_ic files — pull from Drive (~50MB) to local for offline fig scripts, OR run scripts on Colab + push small PDFs back? (Phase 6.1 decision)"
file_state:
  modified_since_last_commit: []
  new_files: [docs/session_handoff_2026-05-27_storya_paper_plan.md]
rule9_status:
  touchpoint_1_plan: NOT_YET_TRIGGERED  # this handoff itself is informational, not a "new experiment plan" — touchpoint optional
  touchpoint_2_code: PENDING  # for analyze_storya_results.py when written
  touchpoint_3_results: PENDING  # for figures + tables when produced
next_actions:
  - "Review this handoff with H博士 — confirm Q1-Q4 above"
  - "Restart Claude Code session so nature-figure / nature-writing / nature-polishing skills load"
  - "Write `analyze_storya_results.py` per §6 plan (estimated 2-3 days)"
  - "Write `run_hats_baseline.py` per plan §1.6 (estimated 1-1.5 weeks)"
  - "Verify-docs-provenance pass on docs/storya_paper_draft.md when draft exists"
---

# Story A Paper — Comprehensive Figure / Table / Experiment Handoff Plan

> **Audience**: future Claude session(s) + H博士. This document maps EVERY completed experiment in this repo onto figures and tables for the Story A ICAIF 2026 paper. Both **last-2-days work** (Story A v3 confirmatory: E1/E3/E4 + E6 + Plan AAA T-1 diagnostic) AND **prior work** (horizon ablation, arch comparison, graph ablation, wf5, Phase 5, Plan AAA 168-group ranking, loss horserace, Tier 1 Phase A/B, SelectiveNet, etc.) are explicitly inventoried and slot into the paper's 4-narrative-pillar structure.
>
> **Source of authority**: Story A v3 plan at `/Users/heruixi/.claude/plans/handoff-session-ranking-swirling-lemur.md`; `plan.md` Decision Log 2026-05-26 / 2026-05-27 rows; `progress.md` 2026-05-26-a..k + 2026-05-27-a/b/c/d; `docs/analysis.md` 2026-05-27-a/c; existing experiment artifacts (paths cited inline).
>
> **Key honesty caveat**: figure counts in this handoff are **upper bounds** ("能出多少出多少图" per H博士 directive 2026-05-27); actual paper Figure/Table count will trim per venue page budget at writing time. Numbers cited below are taken from source CSVs unless noted "TBD" or "approximate".

---

## §1. Paper-level Structure

### §1.1 Paper title (working)

**"When Do Graph Neural Networks Help in Cross-Sectional Stock Ranking? A Multi-Seed, Multi-Universe, Cost-Aware Study of US S&P 500"**

### §1.2 Target venue + format

- **Primary**: ICAIF 2026 (full paper, ACM SIG format, ~8-10 pages + refs)
- **Backup**: Quantitative Finance journal (no page limit)
- **Use venue-templates skill** for: ACM SIG `.cls`, bibstyle, page-limit-aware figure sizing

### §1.3 7-section structure

| § | Title | Pages (ACM SIG est.) | Figures | Tables |
|---|-------|---------------------|---------|--------|
| 1 | Introduction | ~1.0 | 1 (Architecture/setup overview) | 0 |
| 2 | Related Work | ~1.0 | 0 | 1 (16-paper matrix) |
| 3 | Methodology | ~1.5 | 0–1 (pipeline diagram) | 1 (statistical-test framework summary) |
| 4 | Data + Experimental Setup | ~0.5 | 0 | 1 (data + cell budget table) |
| 5 | Results | ~3.5 | 7–10 (main result figs) | 4–6 (headline + ablations) |
| 6 | Discussion | ~1.0 | 0 | 0 |
| 7 | Limitations + Future Work | ~0.5 | 0 | 1 (limitation matrix) |
| — | **Supplementary** | unbounded | +5–10 (overflow figs) | +3–5 (overflow tables) |

### §1.4 4 narrative pillars (locked per plan §LOCKED DECISIONS 2026-05-26 evening)

**N1 — Honest baseline IC under strict multi-seed eval.** Establishes IC numbers reviewers can trust because they survive 10-seed × 5-fold × bootstrap CI eval.

**N2 — Conditional findings (horizon × feature universe × news encoding × edge type).** When does GNN beat MLP / LightGBM, and when does it not?

**N3 — Failure-mode catalog (lucky-seed inflation / news-feature dilution / Fold-4 regime collapse / multi-edge bundle harm / Plan AAA T-1 leak provenance).** What goes wrong, why, and how to detect.

**N4 — Methodology framework (walk-forward + multi-seed + Hansen SPA + DM/HLN + BH-FDR + block bootstrap + LOFO + cost ladder + multi-testing ledger).** Reusable scaffold for future GNN-finance work.

---

## §2. Master Figure List (~25 figures, upper bound)

### Main paper figures (target: 10 — trim from this list)

| F# | Title | Type | Narrative | Source experiment |
|----|-------|------|-----------|-------------------|
| F1 | Story A architecture + experimental design overview | Schematic | All | scientific-schematics skill |
| F2 | Cumulative L/S PnL curves, 2×4 panel (universe × model) | Line plot | N1 | storya_e1_anchor |
| F3 | E1 LOFO sensitivity heatmap (model × universe, color=ΔIC%) | Heatmap | N1 + N3 | storya_e6_dm_spa lofo_diagnostic.csv |
| F4 | E1 per-fold IC bar plot with seed-spread error bars | Grouped bar | N3 | storya_e6_dm_spa per_fold_table.csv |
| F5 | Cost-ladder Net Sharpe vs bps cost, 8 lines (univ × model) | Line plot | N1 + N4 | storya_e6_dm_spa cost_ladder.csv |
| F6 | Edge ablation forest plot (5 pairs × 3 regimes, ΔIC + CI) | Forest plot | N2 + N3 | storya_e6_edge_ablation edge_bootstrap_ci.csv |
| F7 | Horizon × architecture conditional matrix (4 models × 6 horizons) | Heatmap | N2 | horizon_ablation_results.csv |
| F8 | News-feature dilution at 21d: ΔIC = -0.045 illustration | Forest / bar | N3 | horizon_ablation MLP_all vs MLP_price |
| F9 | Hansen SPA p-value timeline + DM/HLN paired Δ-IC distribution | Density / step | N4 | storya_e6_dm_spa spa_results + dm_hln_results |
| F10 | Plan AAA T-1 stability scatter (proxy ranking raw vs T-1) | Scatter | N3 | plan_aaa_t1_diagnostic |

### Supplementary figures (S1-S15+, page-budget-flexible)

| F# | Title | Type | Narrative | Source experiment |
|----|-------|------|-----------|-------------------|
| S1 | Per-cell IC vs Sharpe scatter (400-cell distribution) | Scatter | N1 | per_cell_distribution.csv |
| S2 | Per-cell outlier flagging top-3/bot-3 by Sharpe | Scatter + annotations | N3 (Univ C GAT cid=240 Sharpe=75) | per_cell_distribution.csv |
| S3 | E1 per-day IC time series, 8 lines + Fold-4 shaded | Time series | N3 | storya_e1_anchor per_day_ic/*.npy |
| S4 | Plan AAA 168-group ranking dot plot (top-30, NW-t + BH-FDR) | Dot plot | N3 + N4 | artifacts/plan_aaa/ranking.csv |
| S5 | Plan AAA permutation Δ-IC distribution (per group) | Violin | N4 | plan_aaa per-group perm distributions |
| S6 | Phase 5 Step 3 Plan Z subset SPA + DM tree (S1-S8) | Tree / forest | N2 + N4 | step3_plan_z |
| S7 | Loss horserace ΔIC per (model, loss) heatmap | Heatmap | N2 | loss_horserace paired_delta_ic.csv |
| S8 | Loss horserace Fold 4 ListMLE collapse per architecture | Faceted line | N3 | loss_horserace per-fold IC |
| S9 | Graph ablation: corr vs +sector vs multi-edge IC bar | Grouped bar | N2 + N3 | graph_ablation_results.csv |
| S10 | Phase 5 feature importance per fold (LGB perm rank) | Stacked bar | N1 + N2 | diag_phase5_permutation_importance_lgb.csv |
| S11 | Sector attribution daily forensics (long vs short sector mix) | Stacked area | N3 | diag_sector_attribution_*.csv |
| S12 | SelectiveNet coverage vs IC tradeoff curve | Line | N3 | selectivenet_results.csv |
| S13 | Tier 1 Phase B hyperparameter sweep convergence (per-tier IC distribution) | Box / violin | N4 | tier1{a,b_h2,c}_phase_b |
| S14 | Diagnostic_price 200-cell distribution vs Part B v4 claim | Histogram + arrow | N3 (Part B replication failure) | loss_horserace results_diagnostic_price.csv |
| S15 | Walk-forward calendar visualization (5 folds + 21d purge) | Gantt | N4 | run_storya_e1_anchor.py WALK_FORWARD_FOLDS |
| S16 | Multi-testing ledger pyramid (exploratory + confirmatory family sizes) | Pyramid | N4 | multiple_testing_ledger.json |
| S17 | Bootstrap CI overlay: full vs LOFO-4 vs Fold-4-only | Forest | N1 + N3 | e1_three_column_summary.csv |
| S18 | News-edge density temporal profile (eligible articles per trading day) | Time series | N4 | storya_e3_news_edge news_snapshots_cache.npz |

---

## §3. Master Table List (~12 tables, upper bound)

### Main paper tables (target: 5-6)

| T# | Title | Source | Narrative | Cells |
|----|-------|--------|-----------|-------|
| T1 | E1 Headline IC + Sharpe + Bootstrap CI (8 rows = 2 univ × 4 models) | bootstrap_ci.csv | N1 | 8 |
| T2 | E1 3-column robustness: full / LOFO-4 / Fold-4-only IC + Sharpe (8 rows) | e1_three_column_summary.csv | N1 + N3 | 8 |
| T3 | Statistical tests: SPA + DM/HLN family-of-5 + BH-FDR (per universe) | spa_results.csv + dm_hln_results.csv | N4 | 8 + 10 |
| T4 | Cost ladder Net Sharpe @ {0,5,10,15,20,30} bps × 8 (univ × model) | cost_ladder.csv | N1 + N4 | 48 |
| T5 | Edge ablation: 5 pairs × 3 regime conditions + BH-FDR | edge_pairs_dm.csv + edge_bootstrap_ci.csv | N2 + N3 | 15 |
| T6 | 16-paper related-work matrix (TARGET: expand to 19 via literature-review skill — see §10.2 Stage 1) (axes: horizon / feature / graph / regime / seed / PIT / cherry-pick / cost) | plan §1.9 | All | 17 rows |

### Supplementary tables (~6, page-budget-flexible)

| T# | Title | Source | Narrative | Cells |
|----|-------|--------|-----------|-------|
| ST1 | Data + experimental setup summary (universe + fold dates + hyperparameters) | run_storya_e1_anchor.py constants | N4 | 1 page |
| ST2 | Multi-testing ledger (exploratory family enumeration) | multiple_testing_ledger.json | N4 | 1 page |
| ST3 | Horizon × architecture full IC + Sharpe table (24 rows) | horizon_ablation_results.csv | N2 | 24 |
| ST4 | Plan AAA ranking top-20 + BH-FDR | plan_aaa/ranking.csv | N3 + N4 | 20 |
| ST5 | Phase 5 Step 3 Plan Z subset-by-subset breakdown | step3_plan_z | N2 | ~30 rows |
| ST6 | Loss horserace ΔIC + DM/HLN per pair | loss_horserace paired_delta_ic.csv | N2 | ~20 rows |
| ST7 | Limitations + decisions table (LSTM dropped / sector edge null / Plan AAA T-1 caveat / single-market) | plan §1.9 honest caveats | All §Limitations | 7 rows |

---

## §4. Per-Experiment Figure & Table Map

> **Reading guide**: each subsection below is one experiment family. For each, this lists:
> - Experiment ID + scope + date + status
> - Source files (absolute paths)
> - Headline result (verified, with citation)
> - Figures derivable (F# refs main + S# refs supplementary)
> - Tables derivable (T# refs main + ST# refs supplementary)
> - Narrative pillar contribution

### §4.1 Story A E1 anchor — 400 cells, 2026-05-26..27

- **Scope**: 4 models {GAT, SAGE-Mean, MLP, LightGBM} × 10 canonical seeds × 5 walk-forward folds × 2 universes {B: 10-dim hc, C: 51-dim Plan AAA top-15} = **400 cells**; 21d horizon; correlation-only edges
- **Status**: COMPLETE on Colab A100 (5.58h wall, source: `experiments/storya_e1_anchor/_meta.json`); local results.csv has only smoke 4 cells (Drive sync gap, by `.gitignore` design)
- **Source files**:
  - `experiments/storya_e1_anchor/results.csv` (400 rows on Drive; 4-row smoke locally)
  - `experiments/storya_e1_anchor/per_day_ic/*.npy` (400 .npy files on Drive; ≤4 locally)
  - `experiments/storya_e1_anchor/manifest.csv` (atomic checkpoint log)
  - `experiments/storya_e1_anchor/_meta.json`, `hp_grid.json`, `smoke_benchmark.csv`
  - **Derived E6**: `artifacts/storya_e6_dm_spa/{bootstrap_ci, spa_results, dm_hln_results, cost_ladder, lofo_diagnostic, per_fold_table, per_cell_distribution, e1_three_column_summary, multiple_testing_ledger, summary.md, lofo_summary.md}.csv|json|md`
- **Headline result**: 7/8 (univ, model) cells have IC > 0 with bootstrap CI excluding 0 (only B LightGBM excludes; source `bootstrap_ci.csv`); SPA p_consistent = 0.147 / 0.384 / 0.136 for B / C / joint — none reject at 5% (source `spa_results.csv`); LOFO-4 drops most IC by 38-72% (source `lofo_diagnostic.csv`)
- **Figures**: F2 (PnL curves), F3 (LOFO heatmap), F4 (per-fold IC bars), F5 (cost ladder), S1 (cell-level IC vs Sharpe), S2 (outlier flagging), S3 (per-day IC time series), S15 (walk-forward calendar), S17 (3-column CI forest)
- **Tables**: T1 (headline), T2 (3-column robustness), T3 (SPA + DM/HLN), T4 (cost ladder), ST1 (data setup)
- **Narrative**: N1 (primary), N3 (LOFO failure), N4 (methodology demo)

### §4.2 Story A E3 news-as-edge — 50 NEW cells, 2026-05-27

- **Scope**: SAGE-Mean × Universe B × 10 canonical seeds × 5 walk-forward folds × {corr+news_cooccur} = **50 cells**; PIT news edge (NYSE-session-close UTC cutoff per Codex Round D D-03)
- **Status**: COMPLETE on Colab A100
- **Source files**:
  - `experiments/storya_e3_news_edge/results.csv` (50 rows on Drive)
  - `experiments/storya_e3_news_edge/per_day_ic/*.npy`
  - `experiments/storya_e3_news_edge/news_edge_source_schema.md` (PIT spec)
  - `experiments/storya_e3_news_edge/news_snapshots_cache.npz` (12MB derived PIT-safe edges, 313 daily snapshots, avg ~1823 edges + ~807 articles per day per `_meta.json`)
- **Headline result**: news-edge ΔIC vs α1 baseline = +0.010 [-0.007, +0.024], HLN p=0.039 (smallest in family but below rank-1 BH threshold 0.010 — NOT rejected); LOFO-4 collapses to +0.005 with p=0.34 (source `edge_pairs_dm.csv` row α3vα1)
- **Figures**: F6 (forest plot, α3 row), S18 (news-edge density temporal profile)
- **Tables**: T5 (edge ablation main table row α3)
- **Narrative**: N2 (news encoding conditional finding), N3 (LOFO collapse), N4 (PIT methodology)

### §4.3 Story A E4-α edge ablation — 100 NEW cells, 2026-05-27

- **Scope**: SAGE-Mean × Universe B × 10 seeds × 5 folds × 2 NEW configs {corr+sector (α2), corr+sector+news (α4)} = **100 NEW cells** (α1 reuses E1-B SAGE 50 cells; α3 reuses E3 50 cells; total ablation analysis = 200 cells)
- **Status**: COMPLETE on Colab A100
- **Source files**:
  - `experiments/storya_e4_alpha/results.csv` (100 rows on Drive)
  - `experiments/storya_e4_alpha/per_day_ic/*.npy`
  - `experiments/storya_e4_alpha/_meta.json` (records edge counts: ~1513 corr + ~13535 sector + ~1823 news per day)
  - **Derived E6 v2**: `artifacts/storya_e6_edge_ablation/{edge_pairs_dm, edge_bootstrap_ci, edge_cost_ladder, edge_summary.md}` (15 + 15 + 72 rows)
- **Headline result**: 0/5 pairs survive BH-FDR q=0.05 in full condition; LOFO-4 collapses all ΔIC magnitudes; Fold-4-only ΔIC CIs exclude 0 for all 3 augmented-vs-α1 pairs but N=10 cells per arm caps to diagnostic-only (per `edge_pairs_dm.csv`, `edge_bootstrap_ci.csv`)
- **Figures**: F6 (full forest plot, all 5 rows × 3 regimes), S9 (graph_ablation cross-reference comparison)
- **Tables**: T5 (full)
- **Narrative**: N2 (edge-type conditional), N3 (full bundle harm), N4 (BH-FDR + bootstrap methodology)

### §4.4 Story A E6 statistical post-process — 2026-05-27

- **Scope**: SPA + DM/HLN + BH-FDR + block bootstrap + LOFO + per-fold + per-cell + cost ladder + multi-testing ledger
- **Status**: COMPLETE; outputs at `artifacts/storya_e6_dm_spa/` + `artifacts/storya_e6_edge_ablation/`
- **Source files**: `compute_e6_dm_spa.py` (E1) + `compute_e6_edge_ablation.py` (E3/E4, Option Y imports helpers)
- **Headline result**: see §4.1 (SPA + DM/HLN) and §4.3 (edge BH-FDR)
- **Figures**: F9 (SPA + DM/HLN visualization)
- **Tables**: T3, ST2 (ledger)
- **Narrative**: N4 (entire framework)

### §4.5 Plan AAA T-1 stability diagnostic — 15 groups, 2026-05-27  (REVISED 2026-05-27 post Codex T1 A-02 CRITICAL)

> **HONEST RESTATEMENT (per Codex Touchpoint 1 Round A finding CODEX-A-02 CRITICAL)**: prior handoff draft led with "HIGH proxy stability" framing which softened the diagnostic's actual verdict. The artifact `summary.md` line 27 explicitly states **"Verdict: LOW STABILITY (Plan AAA orig top-15 ∩ proxy-T1 top-15 = 5/15)"** and line 28 prescribes the action **"Universe C composition basis is leak-driven; full Plan AAA re-run required before submission, OR Universe C must be re-defined."** H博士 2026-05-27 verdict A chose to defer the full re-run to §Future Work; the paper §Limitations MUST explicitly carry the LOW STABILITY verdict + the deferral decision, not softpedal as "inconclusive."

- **Scope**: 158 Alpha158 features × 313 test days × {raw / T-1-shifted} → proxy single-feature mean per-day IC → group-level proxy importance for top-15 Plan AAA groups
- **Status**: COMPLETE on M4 CPU (0.3 min vectorized)
- **Source files**: `analyze_plan_aaa_t1_diagnostic.py`, `artifacts/plan_aaa_t1_diagnostic/{group_ranking_comparison, proxy_ic_per_feature, summary.md}`
- **Headline result (LITERAL from artifact summary.md)**:
  - Overlap(Plan AAA orig ∩ proxy-raw) = **5/15** (sanity check; proxy does NOT align with Plan AAA permutation framework)
  - Overlap(Plan AAA orig ∩ proxy-T1) = **5/15** (**KEY metric: LOW STABILITY under leak removal**)
  - Overlap(proxy-raw ∩ proxy-T1) = **15/15** (direct leak-effect on proxy is small, IC magnitudes change ≤0.007 absolute)
  - **Artifact verdict (verbatim)**: **"LOW STABILITY"**
  - **Artifact action (verbatim)**: **"Universe C composition basis is leak-driven; full Plan AAA re-run required before submission, OR Universe C must be re-defined."**
- **H博士 2026-05-27 verdict A** (decision-level, not artifact-level): defer full Plan AAA re-run (~12-24h M4) to paper §Future Work; current submission carries the LOW STABILITY verdict as §Limitations Item 7 + §Limitations Item 5 cross-reference. **The diagnostic's "action-required" language is acknowledged but deferred; the paper does NOT claim Universe C composition is leak-free.**
- **Caveat propagation**: §Limitations Item 5 (analysis.md 2026-05-27-a Q4) + Item 7 must both be rewritten to lead with "Per diagnostic verdict: LOW STABILITY" rather than "inconclusive"
- **Figures**: F10 (proxy stability scatter — caption must explicitly state "Plan AAA orig ∩ proxy-T1 = 5/15 → LOW STABILITY"), S4 (Plan AAA ranking dot plot — caption must annotate which 5/15 of top-15 groups survive)
- **Tables**: ST7 (limitations row 5+7 — language rewrite required: lead with LOW STABILITY verdict)
- **Narrative**: N3 (Plan AAA T-1 leak provenance — STRENGTHENED severity per A-02 fix; this becomes one of the strongest "honest failure-mode catalog" entries since it documents a leak in OUR OWN pipeline that we caught + diagnosed + deferred fix on transparently)

### §4.6 Horizon ablation — 360 data rows, 2026-04-XX  (VERIFIED 2026-05-27)

- **Scope**: 4 models {SAGE-Mean_price, SAGE-Mean_all, MLP_price, MLP_all} × 6 horizons {1, 5, 10, 21, 42, 63}d × 3 seeds {42, 123, 456} × 5 folds = **360 data rows + 1 header**. Note: NO GAT, NO LSTM (despite original plan); actual experiment is 4-model 6-horizon. Plan/memory had drifted; correcting here.
- **Status**: COMPLETE; results in `experiments/horizon_ablation_results.csv` (361 lines)
- **Source files**:
  - `experiments/horizon_ablation_results.csv` (360 data rows; columns: model, seed, fold, horizon, test_period, IC, IC_std, n_days, Sharpe_gross, Sharpe_net, n_periods, mean_turnover)
  - `experiments/horizon_preds/*.npy` (per-cell prediction tensors, 60 files)
  - `archived/scripts/run_horizon_ablation.py` (original runner)
- **Headline result** (VERIFIED 2026-05-27 via pandas group-by on actual CSV):
  - **21d horizon IC by model** (n=15 per cell = 5 folds × 3 seeds):
    - **MLP_price: +0.0374 ± 0.0646** (15 cells)
    - **MLP_all: −0.0078 ± 0.0298** (15 cells)
    - **SAGE-Mean_price: +0.0269 ± 0.0417** (15 cells)
    - **SAGE-Mean_all: +0.0111 ± 0.0296** (15 cells)
  - **News-feature dilution at 21d for MLP: ΔIC = +0.0374 − (−0.0078) = −0.0452** (matches plan §1.2 motivation exactly)
  - News-feature dilution at 21d for SAGE-Mean: ΔIC = +0.0269 − 0.0111 = −0.0158 (less catastrophic than MLP, but same sign)
- **Figures**: F7 (horizon × architecture heatmap, 6×4 cells), F8 (news dilution forest plot — MLP_all vs MLP_price and SAGE-Mean_all vs SAGE-Mean_price across all horizons), S7-S8 (cross-reference)
- **Tables**: ST3 (full 24-row {model × horizon} aggregated)
- **Narrative**: N2 (horizon conditional — peak IC at H=1 for SAGE, H=21 for MLP_price; opposite directions by architecture), N3 (news dilution catastrophic for MLP at H≥21)

### §4.7 Architecture comparison (Week 1-2) — ~150 cells, 2026-03-XX

- **Scope**: {SAGE-Mean, GAT, Transformer, MLP, +variants} × seeds × folds = ~150 rows
- **Status**: COMPLETE; results in `experiments/arch_comparison_results.csv`
- **Source files**: `experiments/arch_comparison_results.csv` (~150 rows; verify via `wc -l`)
- **Headline result** (TBD — re-read CSV at script time): per project Rule 10 + prior memory, GNN-on-all-features negative across architectures; price-only positive uniformly
- **Figures**: optional inclusion in S9 or as supplementary cross-reference
- **Tables**: optional; mostly subsumed by E1
- **Narrative**: N1 (architectural robustness of "GNN-all-features doesn't help"), N3

### §4.8 Graph ablation — 27 data rows, 2026-03-XX  (VERIFIED 2026-05-27)

- **Scope**: 8 configs (verified: `0_true_mlp` true-MLP baseline, `current` corr-0.6+dense-sector, +sector, +industry, no-graph, +news variants, all-edges) × 3 seeds {42,123,456} = **27 data rows + 1 header**
- **Status**: COMPLETE; results in `experiments/graph_ablation_results.csv` (28 lines incl. header)
- **Source files**: `experiments/graph_ablation_results.csv` (columns: `config, desc, seed, edges, IC, IC_std, Sharpe_NO, n_periods`)
- **Headline result** (VERIFIED): config=`0_true_mlp` (nn.Linear true MLP baseline) IC mean across 3 seeds = +0.0413, std = 0.0098 (source: read at 2026-05-27); multi-edge configs trend ≈ baseline or below (foreshadows E4-α negative bundle finding). **IMPORTANT**: this experiment includes the `0_true_mlp` row which is THE clearest "no graph" baseline — should be highlighted in F9/S9.
- **Figures**: S9 (cross-reference to E4-α with `0_true_mlp` baseline overlay)
- **Tables**: optional supplementary; could feed ST8 (new — graph_ablation full)
- **Narrative**: N2 (edge-type conditional, pre-Story-A evidence), N3 (multi-edge collapse)

### §4.9 Week 1 5-fold walk-forward baseline (wf5) — 90 cells, 2026-03-XX

- **Scope**: 6 models × 3 seeds × 5 folds = **90 cells** per project Rule 10
- **Status**: COMPLETE; results in `experiments/wf5_results.csv`
- **Source files**: `experiments/wf5_results.csv`
- **Headline result** (TBD): provides original 9-dim S_price + 768-d + 9-dim feature universes' walk-forward IC baseline; **used as comparator for Diagnostic_price 200-cell replication study** (see §4.13)
- **Figures**: optional inclusion in S9 (cross-reference)
- **Tables**: optional supplementary
- **Narrative**: N1 (baseline-credibility predecessor of E1), N4 (walk-forward methodology origin)

### §4.10 Phase 5 Step 3 Plan Z — subset × Hansen SPA analysis, 2026-04-18..20

- **Scope**: 8 feature subsets {S1-S8} × {MLP, SAGE-Mean} × 5 folds × multiple seeds (verify exact count); Hansen SPA + DM/BH-FDR applied to ΔIC family
- **Status**: COMPLETE; results in `experiments/step3_plan_z/`
- **Source files**:
  - `experiments/step3_plan_z/part_a_permuted_ic.csv` (13K rows, per-day per-group IC)
  - `experiments/step3_plan_z/part_a_daily_ic.csv` (1.9K rows, aggregated)
  - `experiments/step3_plan_z/part_c_s8_perfold_daily_ic.csv` (S8 universe per-fold)
  - `experiments/step3_plan_z/fold4_tail_concentration.csv` (Fold 4 regime tail risk)
  - `artifacts/step3_plan_z/{subsets_frozen, fold_manifest, per_fold_scaler}.json` (frozen configs)
- **Headline result** (per `docs/methodology_qa_2026-05-22.md` cited earlier): T_SPA = 1.231 for SAGE-Mean (NOT 0.23 — that was the 2026-04-21-c provenance bug); S6 MLP ΔIC = +0.046, p = 0.009. Provides Story A's first Hansen SPA precedent in this repo.
- **Figures**: S6 (subset SPA forest tree)
- **Tables**: ST5 (subset breakdown)
- **Narrative**: N2 (feature-universe conditional), N4 (SPA methodology precedent)

### §4.11 Plan AAA 168-group ranking — 168 groups, 2026-05-XX

- **Scope**: 168 feature groups × 313 test days × 3 seeds × permutation Δ-IC + Newey-West t-stat + BH-FDR + bootstrap CI
- **Status**: COMPLETE; results in `artifacts/plan_aaa/`
- **Source files**:
  - `artifacts/plan_aaa/ranking.csv` (62 rows of ranked groups; check if 168 vs 62 — possibly 62 = unique non-trivial groups after dedup)
  - `run_plan_aaa_168_ranking.py` (original runner)
  - **Diagnostic addendum**: `artifacts/plan_aaa_t1_diagnostic/` (see §4.5)
- **Headline result**: 0 of 61 (or 62) groups pass BH-FDR q=0.05 (per project memory + plan §1.9 honest caveats); top group hc_mom12m Δ-IC = +0.0079, NW-t = 1.01, p = 0.31 (per agent inventory above; **VERIFY at script time**). 1 group (CORD20+1) passed FDR in NEGATIVE direction. **Caveat**: Plan AAA used same-day Alpha158 (T-1 leak in ranking step), see §4.5 diagnostic.
- **Figures**: S4 (top-30 dot plot), S5 (perm Δ-IC violins)
- **Tables**: ST4 (top-20 ranking)
- **Narrative**: N3 (group-test null result + leak provenance), N4 (permutation + FDR framework)

### §4.12 Loss horserace (Stage 0 + Stage 1) — 600 cells, 2026-04-22..27

- **Scope**: Stage 0 pilot (162 cells) + Stage 1 (5 losses × 6 archs × 5 folds × ~3-10 seeds ≈ 438 cells, total ~600) on MLP / SAGE-Mean × {MSE, Huber, Tukey, ListNet τ=0.2, ListMLE}
- **Status**: COMPLETE; results in `experiments/loss_horserace/` + `artifacts/loss_horserace/`
- **Source files**:
  - `experiments/loss_horserace/results.csv` (~37K daily-IC rows; ~600 cells aggregated)
  - `experiments/loss_horserace/results_diagnostic_price.csv` (12.5K daily-IC rows, ~200 cells diagnostic price)
  - `experiments/loss_horserace/paired_delta_ic.csv` (24.7K rows, pairwise loss comparison)
  - `experiments/loss_horserace/sharpe_per_run.csv` (601 rows, Sharpe summary)
- **Headline result** (per Decision Log 2026-05-20 + analysis.md 2026-04-27-b): 0/36 BH-FDR rejected across loss × arch × subset families; ListMLE shows architecture-independent + feature-independent Fold-4 catastrophic collapse (mean IC ∈ [-0.36, -0.28] across 6/6 arch × feat combos). Robust losses (Tukey/Huber) don't help. Loss family is NOT a key conditional → Story A drops loss-function as a narrative axis.
- **Figures**: S7 (ΔIC heatmap), S8 (ListMLE Fold-4 collapse faceted)
- **Tables**: ST6 (loss horserace pairwise DM/HLN)
- **Narrative**: N2 (loss conditional null), N3 (ListMLE collapse failure mode), N4 (BH-FDR over 36-test family)

### §4.13 Diagnostic_price (Part B v4 replication) — 200 cells, 2026-04-27

- **Scope**: (MSE, ListMLE) × (MLP, SAGE-Mean) × 9-dim S_price × 5 folds × 10 seeds = **200 cells** in Stage 1 framework
- **Status**: COMPLETE; results in `experiments/loss_horserace/results_diagnostic_price.csv`
- **Source files**: `experiments/loss_horserace/results_diagnostic_price.csv` (200 cells × ~62.6 days = 12.5K daily rows); `experiments/loss_horserace/local_diag_price.log`
- **Headline result** (per analysis.md 2026-04-27-b): MLP×S_price IC = -0.004, SAGE×S_price IC = -0.057 — **Part B v4 wf5's headline 21d MLP_price IC=+0.037 / SAGE_price IC=+0.027 DOES NOT replicate** in Stage 1 framework with identical feature set. Part B's apparent advantage was setup-specific artifact (code path / fold timing / model spec).
- **Figures**: S14 (Part B replication failure histogram)
- **Tables**: optional supplementary
- **Narrative**: N3 (replication-failure failure mode — supports Template 1 narrative element via §4.18 HATS-3R-adapt comparator)

### §4.14 Tier 1 Phase A/B (Plan AAA precursor sweeps) — ~1400 cells, 2026-05-XX

- **Scope**: Tier1a Phase B (200) + Tier1b h2 Phase B (800) + Tier1c Phase B (400) = ~1400 cells across hyperparameter / loss / feature variations
- **Status**: COMPLETE; results in `artifacts/tier1{_phase_a, a_phase_b, b_h2_phase_b, c_phase_b}/`
- **Source files**:
  - `artifacts/tier1_phase_a/results.csv` (~1200 rows)
  - `artifacts/tier1a_phase_b/results.csv` (~200 rows)
  - `artifacts/tier1b_h2_phase_b/results.csv` (~800 rows)
  - `artifacts/tier1c_phase_b/results.csv` (~400 rows)
  - `artifacts/phase_b_finalize/{stat_tier1a, stat_tier1b_h2, stat_tier1c, tier1e_regime_forensic, ic_sector_resid_per_cell}.csv` (aggregated stats)
- **Headline result**: Tier1 best mean IC ~0.04 across folds 0-3, Fold 4 consistently -0.05 to -0.10 (regime forensic confirmed). 10-seed expansion exposed 5-seed selection artifact (Decision Log 2026-05-20: Tier 1.D verdict revoked from "marginal positive" to "FULL NULL").
- **Figures**: S13 (Tier 1 hyperparameter convergence)
- **Tables**: optional; mostly subsumed by E1 + Plan AAA
- **Narrative**: N3 (Fold 4 universally negative across Tier 1 hyperparameter sweeps — the strongest pre-Story-A evidence for Fold 4 = regime failure)

### §4.15 SelectiveNet — 70 cells, 2026-04-XX

- **Scope**: 7 SelectiveNet target coverages × 10 thresholds = 70 rows
- **Status**: COMPLETE; results in `experiments/selectivenet_results.csv`
- **Source files**: `experiments/selectivenet_results.csv` (70 rows)
- **Headline result**: Peak IC ~0.08 at 10% coverage (threshold strategy works); E2E target-0.2 drops to 0.02 IC; selective rejection suppresses signal not noise
- **Figures**: S12 (coverage vs IC tradeoff)
- **Tables**: optional supplementary
- **Narrative**: N3 (selective-rejection failure mode)

### §4.16 Sector attribution diagnostics — 2026-04-XX

- **Scope**: SAGE-Mean × {sum, mean} aggregation × 11 sectors × 313 days = ~1409 rows × 2 aggregations
- **Status**: COMPLETE; results in `experiments/diag_sector_attribution_sage_*.csv`
- **Source files**: `experiments/diag_sector_attribution_sage_mean.csv`, `_sum.csv`, `diag_sector_composition.csv`, `diag_sector_ic.csv`
- **Headline result**: Sector contributions ±0.01/day; Tech dominates long-side (10+ stocks/day) with high short overlap (8-15 stocks); sector crowding masks signal
- **Figures**: S11 (sector attribution stacked area)
- **Tables**: optional supplementary
- **Narrative**: N3 (sector-crowding failure mode)

### §4.17 Phase 5 diagnostics suite — 2026-04-XX

- **Scope**: 10+ diagnostic CSVs covering collinearity, feature importance, label distribution, train/test regime shift, permutation importance LGB, fold-regime composition, pairwise correlations
- **Status**: COMPLETE; results in `experiments/diag_phase5_*.csv`
- **Source files**:
  - `experiments/diag_phase5_collinearity.csv`
  - `experiments/diag_phase5_feature_importance.csv` (5.7K rows)
  - `experiments/diag_phase5_ic_by_fold.csv`
  - `experiments/diag_phase5_train_test_shift.csv`
  - `experiments/diag_phase5_permutation_importance_lgb.csv`
  - `experiments/diag_phase5_fold_regime.csv`
  - `experiments/diag_phase5_pairwise_corr_v2.csv`
  - `experiments/diag_phase5_effective_rank.csv`
  - `experiments/diag_phase5_label_dist.csv`
  - `experiments/diag_phase5_single_feature_lgb.csv`
  - `experiments/diag_phase5_feature_dist.csv`
- **Headline result**: Per-feature IC range 0-0.03; Fold 4 shows 0.05-0.10 regime shift via collinearity diagnostic
- **Figures**: S10 (LGB perm rank stacked bar per fold)
- **Tables**: optional supplementary (mostly methodology validation)
- **Narrative**: N1 (baseline credibility), N3 (Fold 4 regime confirmation via independent diagnostic), N4 (multi-modal feature diagnostics)

### §4.18 HATS-3R-adapt baseline — plan locked + Codex Round A PROCEED-WITH-FIXES, 2026-05-27 (parallel session)

- **IMPORTANT name change (per Codex T1 Round A dispositions A-05/A-06/A-07)**: NOT pure "HATS reproduction" — renamed to **"HATS-3R-adapt"** (HATS-style 3-Relation Adaptation) because the implementation differs from Kim et al. 2019 along 4 dimensions: (1) no Wikidata KG ingestion → use sector/correlation/news edges; (2) no GRU sequence encoder → use SAGE-Mean to align with E1; (3) S&P 500 not KOSPI 200; (4) 10-seed × 5-fold + walk-forward purge per Story A methodology. **prereg.json `claim_scope` narrowed** — paper does NOT make a "Template 1 replication-failure" claim against Kim 2019 GRU+Wikidata original; instead positions as "Template 1 adapted-architecture comparator" inside Story A narrative N1
- **Plan file**: `/Users/heruixi/.claude/plans/hats-baseline-reproduction-delightful-lighthouse.md`
- **Codex Touchpoint 1 review**: `artifacts/reviews/2026-05-27_codex_plan_A.md` — Round A verdict: **1 CRITICAL + 8 MAJOR + 2 CONCERN** initial BLOCK-EXECUTION → post-disposition **PROCEED-WITH-FIXES** (8 FIXED + 3 ACCEPTED-AS-CONCERN + 0 REJECTED). Disposition log in progress.md 2026-05-27-d
- **Scope (post-Codex amendments)**: 50 cells = SAGE-Mean + HATS-3R adapter × 10 seeds × 5 folds × Universe B (single-universe, NOT both); horizon 21d; 3 edge relations {correlation, sector, news_cooccur} fused
- **Status**: PLAN APPROVED; awaiting implementation. **A100 1-cell smoke benchmark mandatory before 50-cell launch** per A-10 disposition (wall-time was provisional 13min/cell)
- **Cell IDs**: `cell_id_hats = 400 + fold_idx*10 + seed_idx`, range [400, 449] (avoids E1 [0, 399] collision per A-04 fix)
- **Source files (future after launch)**: `run_hats_3r_adapt.py`, `experiments/storya_hats_3r_adapt/results.csv` + `per_day_ic/*.npy` + `manifest.csv`; `analyze_hats_lofo.py` (mirrors `analyze_e1_lofo.py:167-169` 3-column pattern per A-09 fix); E6 integration via the locked decision rule + SPA family expansion (HATS EXCLUDED from joint SPA per A-02; per-universe B M=3→4 only)
- **Locked decision rule (per A-08 Codex 3-gate)**:
  - POSITIVE: ΔIC > +0.005 AND BH-HLN p < 0.05 AND LOFO-4 sign-preserve
  - NEGATIVE: ΔIC < −0.005 OR (p > 0.20 AND LOFO-4 ≤ 0)
  - TIE: else
- **Locked extensions pre-committed (per A-11)**: +50 cell uniform-α run gates any "attention-specific" claim
- **§Limitations item (per A-01 ACCEPTED-AS-CONCERN)**: sector PIT — `data/reference/sp500_sectors.csv` was fetched 2026-02-09 (one snapshot), no PIT history; Plan §Limitations row added
- **Headline result**: TBD post-run; expected per Codex 3-gate framework + Story A N1 narrative
- **Figures (now confirmed integrated into S+1/S+2 + T1 extension after launch)**:
  - S+1: HATS-3R-adapt IC distribution overlaid on Story A E1 (per-seed dot plot, 50 HATS dots + 200 E1 SAGE-Mean dots for Univ B)
  - S+2: HATS-3R-adapt 3-column robustness (full / LOFO-4 / Fold-4-only) — same 3-column format as E1 Table 2
  - S+3 (optional): edge-relation attention weights heatmap if HATS attention layer instrumented
- **Tables (confirmed)**: T1 extension row "HATS-3R-adapt (10s × 5f, Univ B)" + ST6 extension paired DM/HLN row "HATS-3R-adapt vs SAGE-Mean_corr-only"
- **Narrative**: N1 (4th narrative element — strengthens "honest baseline under strict eval" by adapting a published-GNN family for S&P 500 cross-sectional ranking; narrowed claim_scope avoids overreach into Template 1 false-equivalence territory)

### §4.19 Step 3 Feature Expansion (Phase 5) — 135 rows, 2026-04-XX  (DISCOVERED 2026-05-27 self-audit)

- **Scope**: Feature-expansion sub-experiment (separate from §4.10 Plan Z subset analysis); 135 data rows
- **Status**: COMPLETE; results in `experiments/step3_feature_expansion_results.csv` (136 lines incl. header)
- **Source files**: `experiments/step3_feature_expansion_results.csv`
- **Headline result** (TBD — script-time verification needed)
- **Figures**: ancillary; cross-reference only in S6 (Plan Z subset SPA tree)
- **Tables**: optional supplementary; could feed an extended ST5 row
- **Narrative**: N2 (additional feature-universe conditional data point)

### §4.20 Step 0 Permutation v2 (Phase 5 baseline-null) — 16K shuffles, 2026-04-XX  (DISCOVERED 2026-05-27 self-audit)

- **Scope**: Per Rule 10 "Step 0 Reruns DONE: ... permutation v2 (16K shuffles)"; established IC null distribution for SAGE-Mean × {_price, _all} × 3 seeds {42, 123, 456}
- **Status**: COMPLETE
- **Source files**:
  - `experiments/permutation_v2_results.csv` (16 rows summary)
  - `experiments/perm_v2_ics_*.npy` (12 distribution files: 4 model×feature combos × 3 seeds = 12 files of ~16000 shuffles each, total 16K × 12 ≈ 192K samples)
  - `experiments/permutation_test_results.csv` + `permutation_test_ics.npy` (v1 baseline, predecessor)
- **Headline result** (TBD — read CSV at script time); per Rule 10 spirit, this established that "GNN-on-all-features IC is statistically indistinguishable from random permutation"
- **Figures**: methodology box; could feed S10 (cross-reference) or new figure S_perm
- **Tables**: ST_perm (optional new)
- **Narrative**: N1 (baseline-null calibration) + N4 (permutation methodology precedent)

### §4.21 Step 0 SEC Gate 1 (Lazy Prices) — 22 rows, 2026-04-XX  (DISCOVERED 2026-05-27 self-audit)

- **Scope**: Per Rule 10 "SEC Gate 1 STOP: Layer 1 Lazy Prices 对 NN 有害; Layer 2/3 CANCELLED"; tested SEC Lazy Prices layer-1 features on NN
- **Status**: COMPLETE; Layer 2/3 CANCELLED per gate decision
- **Source files**: `experiments/gate1_results.csv` (23 lines incl. header)
- **Headline result** (TBD — read CSV at script time): negative for NN; led to gate STOP decision
- **Figures**: optional supplementary S_gate1 (Lazy Prices SEC failure illustration)
- **Tables**: optional supplementary
- **Narrative**: N3 (additional failure mode: SEC text feature fails for NN — pre-Story-A evidence aligning with "GNN doesn't help" theme broadened beyond graph alone)

### §4.22 Option B LightGBM Feature Importance (hand-curated rank) — top-30 features, 2026-05-XX  (DISCOVERED 2026-05-27 self-audit)

- **Scope**: LightGBM permutation feature importance with manual curation to top-30 list; basis for Universe C composition selection
- **Status**: COMPLETE
- **Source files**: `artifacts/option_b_lgbm_importance/{hand_curated_ranks.csv, importance_full.csv, summary.md, top_30.csv}`
- **Headline result** (TBD — read summary.md at script time); cross-referenced with Plan AAA top-15 in Universe C composition (§4.1 E1 anchor Universe C list)
- **Figures**: optional supplementary S_lgb_imp (top-30 LGB importance ranked bar)
- **Tables**: optional supplementary
- **Narrative**: N2 (feature-universe conditional provenance — explains how Universe C top-15 groups were chosen; provides external LGB-importance corroboration of Plan AAA's permutation-Δ-IC-based group selection)

### §4.23 Phase 5 Audits (feature audit + sentinel leakage test) — 2026-04-XX  (DISCOVERED 2026-05-27 self-audit)

- **Scope**: (1) Phase 5 features audit — every Alpha158 column inspected for T-1 contract compliance; (2) sentinel leakage test — synthetic "future" sentinel feature injected to detect leakage in pipeline
- **Status**: COMPLETE; methodology validation step that pre-dates the Plan AAA T-1 diagnostic (§4.5)
- **Source files**: `artifacts/audits/{phase5_features_audit.md, sentinel_leakage_test.md}`
- **Headline result** (TBD — read summary docs at script time)
- **Figures**: methodology box; no new figure unless sentinel test result needs visualization
- **Tables**: optional supplementary
- **Narrative**: N4 (methodology pre-validation — supports Story A's claim "we have a leakage-aware pipeline" by citing audit + sentinel test). **Important for §Limitations** — explains the chain of evidence around the Plan AAA T-1 leak: this audit DID NOT catch it because audit was on Phase 5 features (different from Alpha158); the leak was discovered LATER via Plan AAA T-1 diagnostic (§4.5).

### §4.24 Loss Horserace methodology sub-analyses — multiple bootstrap / mixed-effects / LOFO artifacts, 2026-04-XX  (DISCOVERED 2026-05-27 self-audit; partially covered in §4.12)

- **Scope**: Statistical methodology applied to loss horserace data, demonstrating the cluster-bootstrap + mixed-effects + LOFO + block-bootstrap framework that Story A v3 E6 inherits
- **Status**: COMPLETE; outputs in `experiments/loss_horserace/` sub-CSVs
- **Source files**:
  - `experiments/loss_horserace/block_bootstrap_sharpe.csv` — block bootstrap on Sharpe
  - `experiments/loss_horserace/cluster_bootstrap_ic.csv` + `cluster_bootstrap_pred_cs_std.csv` — cluster bootstrap (cluster = fold) on IC and pred CS std
  - `experiments/loss_horserace/mixed_effects_ic.csv` + `mixed_effects_pred_cs_std.csv` — mixed-effects regression (random intercepts per fold)
  - `experiments/loss_horserace/fold4_lofo_stats.csv` — Fold 4 LOFO statistics
  - `experiments/loss_horserace/per_cell_stats.csv` + `analysis_summary.csv` — per-cell + summary
- **Headline result** (per Decision Log 2026-05-20 + Rule 10): 0/36 BH-FDR rejected; ListMLE Fold-4 collapse; mixed-effects validates the Fold-4 random-intercept dominance
- **Figures**: S_loss_methodology (optional: visualization of mixed-effects vs naive aggregation showing how methodology choice changes conclusion)
- **Tables**: optional supplementary (mostly subsumed by §4.12's main tables)
- **Narrative**: N4 (methodology lineage — Story A v3 E6's LOFO + bootstrap + mixed-effects-equivalent (NW-HAC paired tests) pattern comes from this pre-work). Should be CITED in paper §3 Methodology as "the framework was first applied to Loss Horserace (Decision Log 2026-05-20) and refined for Story A v3 E1/E3/E4/E6 (this work)."

### §4.26 Ranking-loss research — 65 rows, 2026-04-XX  (ADDED 2026-05-27-g per Codex T1 A-01 MAJOR)

- **Scope**: {NoGraph_price, MLP_price, SAGE-Mean variants} × {mse, listmle (τ=0.2), pairwise} × seeds × folds = **65 data rows**
- **Status**: COMPLETE; results in `experiments/ranking_loss_results.csv` (66 lines incl. header; columns: model, seed, fold, loss_type, tau, test_period, IC, IC_std, n_days, Sharpe_gross, Sharpe_net, n_periods, mean_turnover)
- **Source files**: `experiments/ranking_loss_results.csv` + `experiments/ranking_loss_combined.csv` (aggregated)
- **Headline result** (TBD — read CSV at script time): listed in inventory as "ListNet τ=0.2 IC range 0.00-0.12 vs MSE 0.00-0.10, no consistent winner; Fold 4 Q2-2025 lucky seed Sharpe 22.76 for MLP+ListNet seed 123 — not reproducible across seeds"
- **Figures**: optional supplementary S_ranking_loss (lucky-seed Sharpe distribution across seeds for ListNet)
- **Tables**: optional supplementary
- **Narrative**: N3 (lucky-seed failure mode — adds pre-Story-A evidence that single-seed Sharpe headlines are unreliable, complements §4.1 E1 LOFO finding)

### §4.27 Comprehensive metrics (Phase 1+2 unified summary) — 12 rows, 2026-03-XX  (ADDED 2026-05-27-g per Codex T1 A-01 MAJOR)

- **Scope**: 12 model-config rows (e.g., sage-mean_all_s42, sage-mean_all_s123, sage-sum_all_s42, ...) with comprehensive metrics including Sharpe at multiple cost levels {0, 5, 10, 15, 20, 30 bps}
- **Status**: COMPLETE; results in `experiments/comprehensive_metrics.csv` (13 lines incl. header; columns include IC, ICIR, IC_std, Sharpe_overlap, Ann_LS_overlap, Sharpe_nonoverlap, n_periods, mean_turnover, Sharpe_{0,5,10,15,20,30}bps + Ann_LS_*)
- **Source files**: `experiments/comprehensive_metrics.csv`
- **Headline result** (TBD — read at script time): provides the FIRST cost-ladder Sharpe analysis in this repo (predecessor of Story A E6 `cost_ladder.csv`); same 6-bps-level convention {0, 5, 10, 15, 20, 30}
- **Figures**: cross-reference in F5 (cost ladder) — possible overlay of Phase 1+2 + E1 cost-ladder curves to show methodology lineage; or new S_comp_metrics
- **Tables**: optional supplementary; supersedes by E1 §4.1 cost_ladder.csv
- **Narrative**: N4 (cost-ladder methodology origin in this repo — informs §3 Methodology paragraph "the 6-level bps ladder is consistent across Phase 1+2 historical baseline and Story A v3 confirmatory")

### §4.25 (Deprecated artifact note) storya_multiseed/ — SUPERSEDED, do not include in paper  (DISCOVERED 2026-05-27 self-audit)

- **Scope**: `experiments/storya_multiseed/prereg.json` ONLY — superseded design v2 pre-registration; replaced by v3 storya_e1_anchor pre-reg (§4.1) per Codex Round D D-01 fix series
- **Status**: DEPRECATED; do NOT include in paper
- **Action**: leave on disk for audit trail; in §Limitations or §Methodology, do NOT cite; if paper reviewer asks "did you change pre-registration", paper response: "yes — pre-reg v3 supersedes earlier drafts per Codex Touchpoint 1 Round A-D iterations recorded in artifacts/reviews/ commits 5bef3b9..8149fab"

---

## §5. Narrative-Pillar → Figure/Table Cross-Index

> **Use case**: when writing each §5 results subsection of the paper, look up the pillar to find every figure/table that supports it.

### N1: Honest baseline IC under strict multi-seed eval

- **Primary evidence**: E1 anchor (§4.1), HATS-3R-adapt (§4.18 — plan locked, Codex T1 PROCEED-WITH-FIXES, awaiting compute launch)
- **Supporting**: Arch comparison (§4.7), wf5 baseline (§4.9), Phase 5 diagnostics (§4.17)
- **Figures**: F2, F3, S1, S3, S15, S17 (+ S+1, S+2 once HATS done)
- **Tables**: T1, T2 (+ T1 extension with HATS row)

### N2: Conditional findings

- **Primary evidence**: Horizon ablation (§4.6), E3 news-as-edge (§4.2), E4-α edge ablation (§4.3), Phase 5 Step 3 Plan Z (§4.10)
- **Supporting**: Graph ablation (§4.8), Loss horserace (§4.12)
- **Figures**: F6, F7, S6, S7, S9
- **Tables**: T5, ST3, ST5, ST6

### N3: Failure-mode catalog

- **Lucky-seed inflation**: E1 LOFO (§4.1), Tier 1 10-seed revocation (§4.14)
- **News-feature dilution**: Horizon ablation MLP_all vs MLP_price (§4.6)
- **Fold-4 regime collapse**: E1 (§4.1), Tier 1 Phase B (§4.14), Phase 5 Step 3 Plan Z (§4.10), Loss horserace ListMLE (§4.12)
- **Multi-edge bundle harm**: E4-α (§4.3), Graph ablation (§4.8)
- **Plan AAA T-1 leak**: Diagnostic (§4.5), Plan AAA 168 (§4.11)
- **Selective-rejection failure**: SelectiveNet (§4.15)
- **Sector-crowding**: Sector attribution (§4.16)
- **Replication failure**: Diagnostic_price (§4.13)
- **Figures**: F3, F4, F8, F10, S2, S8, S11, S12, S14
- **Tables**: T2 (LOFO), T5 (edge), ST7 (limitations matrix)

### N4: Methodology framework

- **Walk-forward purge + embargo**: §4.1 + §4.9
- **Multi-seed canonical 10**: §4.1
- **Hansen SPA**: §4.4 + §4.10 (precedent)
- **DM/HLN paired**: §4.4
- **BH-FDR**: §4.4 + §4.3 + §4.11 + §4.12
- **Block bootstrap CI**: §4.4 + §4.11
- **LOFO sensitivity**: §4.1
- **Cost ladder**: §4.4
- **Multi-testing ledger**: §4.4 (`multiple_testing_ledger.json` with historical exploratory enumeration)
- **Permutation Δ-IC**: §4.11
- **T-1 stability diagnostic**: §4.5
- **Figures**: F9, S5, S15, S16, S18
- **Tables**: T3, T6, ST1, ST2

---

## §6. Implementation plan — sequenced script work

> **Estimated total**: 5-8 days local work + ~1-1.5 weeks HATS (~3-4 days could be parallelized to Colab compute) = ~2.5 weeks before draft writing starts (week 5 per plan §8 timeline).

### Phase 6.1: rcparams preset + skill verification (1 day, BLOCKING)

- **Restart Claude Code session** so nature-* skills auto-load
- Write `paper_figs/rcparams_storya.py` defining ACM-SIG / ICAIF-compatible rcparams (combining `matplotlib` skill defaults + `mpl_sizes.get_format('ICML')` + Times serif + 8pt caption + ≥300 DPI)
- Smoke test: render 1 dummy bar chart, verify `figures/test.pdf` produced with correct dimensions
- Verify all 7 currently-loaded skills + 6 nature-* skills usable via `Skill` tool

### Phase 6.2: Prior-experiment figure ports (3-4 days)

Scope: figures S4-S14 derived from prior experiments with existing result CSVs (no new compute needed).

**Module split (per Q2 above, propose modular)**:

1. `paper_figs/fig_horizon_ablation.py` → F7, F8, ST3 (from §4.6 + §4.13 cross-reference)
2. `paper_figs/fig_plan_aaa.py` → F10, S4, S5, ST4 (from §4.5 + §4.11)
3. `paper_figs/fig_phase5_step3.py` → S6, ST5 (from §4.10)
4. `paper_figs/fig_loss_horserace.py` → S7, S8, S14, ST6 (from §4.12 + §4.13)
5. `paper_figs/fig_graph_ablation.py` → S9 (from §4.8)
6. `paper_figs/fig_phase5_diagnostics.py` → S10, S11 (from §4.16 + §4.17)
7. `paper_figs/fig_selectivenet.py` → S12 (from §4.15)
8. `paper_figs/fig_tier1_phaseb.py` → S13 (from §4.14)

Each module: ~150-300 LOC, imports rcparams_storya, reads CSV, produces PDF + PNG to `figures/`.

### Phase 6.2b: Newly-discovered prior-experiment scripts (ADDED 2026-05-27-g per Codex T1 A-05 MAJOR — §4.19-§4.27 manifest reconciliation, +1 day)

Scope: scripts for the 8 newly-discovered/missing prior experiments (§4.19 step3 feature expansion + §4.20 permutation v2 + §4.21 SEC Gate 1 + §4.22 Option B LGBM importance + §4.23 Phase 5 audits + §4.24 loss horserace methodology + §4.26 ranking-loss + §4.27 comprehensive metrics).

8a. `paper_figs/fig_step3_expansion.py` → optional supplementary; cross-reference within S6 (Plan Z subset SPA tree) — from §4.19
8b. `paper_figs/fig_perm_v2_null.py` → optional new S_perm + ST_perm (null distribution + summary) — from §4.20
8c. `paper_figs/fig_sec_gate1.py` → optional S_gate1 (Lazy Prices SEC failure) — from §4.21
8d. `paper_figs/fig_lgb_importance.py` → optional S_lgb_imp (top-30 LGB importance bars) — from §4.22
8e. `paper_figs/fig_audits.py` → methodology box only (prose); no new figure; from §4.23
8f. `paper_figs/fig_loss_methodology.py` → optional S_loss_methodology (mixed-effects vs naive aggregation comparison) — from §4.24
8g. `paper_figs/fig_ranking_loss.py` → optional S_ranking_loss (lucky-seed Sharpe distribution for ListNet) — from §4.26
8h. `paper_figs/fig_comp_metrics.py` → cross-reference overlay in F5 (Phase 1+2 cost ladder + E1 overlay) — from §4.27

Note on §3 Master Table prose tables (T6 / ST1 / ST7 — NOT auto-generated; clarified per Codex T1 A-05 MAJOR):
- **T6** (16-paper / 19-paper related-work matrix): authored by `scientific-writing` skill + `literature-review` skill in §10.2 Stage 1; NOT by a `paper_figs/fig_*.py` script
- **ST1** (data + experimental setup summary): authored by `scientific-writing` skill; takes inputs from `run_storya_e1_anchor.py` constants + plan §1.7 hp_grid.json
- **ST7** (limitations matrix): authored by `scientific-writing` skill; inputs from plan §1.9 honest caveats + analysis.md §Limitations Items 5+6+7 + Codex T1 dispositions

### Phase 6.3: Story A v3 figures (2 days)

Scope: F2-F6, F9, S1-S3, S15-S18 from E1/E3/E4/E6 artifacts (last 2 days work).

9. `paper_figs/fig_e1_anchor.py` → F2, F3, F4, S1, S2, S3, S17, T1, T2 (from §4.1)
10. `paper_figs/fig_e6_statistical.py` → F9, S16, T3, ST2 (from §4.4)
11. `paper_figs/fig_e6_cost_ladder.py` → F5, T4 (from §4.4)
12. `paper_figs/fig_edge_ablation.py` → F6, S18, T5 (from §4.2 + §4.3)
13. `paper_figs/fig_walkforward_calendar.py` → S15 (Gantt of folds + 21d purge)

### Phase 6.4: Architecture + setup schematic (1 day)

14. Use `scientific-schematics` skill to produce F1 (Story A architecture overview: data flow + model + edge config + walk-forward schematic in a single SVG)

### Phase 6.5: Composite + summary figures (0.5 day)

15. Multi-panel "Story A at a glance" composite combining F2 + F3 + F4 into single PDF page for §5 lead figure

### Phase 6.6: HATS-3R-adapt baseline (~1-1.5 weeks parallel work — plan PROCEED-WITH-FIXES per parallel-session Codex T1)

16. `run_hats_3r_adapt.py` per `/Users/heruixi/.claude/plans/hats-baseline-reproduction-delightful-lighthouse.md` (post-Codex-T1-amendments) + `analyze_hats_lofo.py` (mirrors `analyze_e1_lofo.py:167-169`) — run on Colab A100 (mandatory 1-cell smoke benchmark before 50-cell launch per A-10), integrate into T1 row extension + S+1/S+2 supplementary figures; cell_id range [400, 449] (A-04 fix); HATS EXCLUDED from joint SPA M=6 (A-02 fix); locked Codex 3-gate decision rule (A-08); uniform-α extension pre-committed (A-11)

### Phase 6.7: Verification + provenance check (1 day; UPGRADED 2026-05-27-g per Codex T1 A-07 MAJOR)

> **Per Codex A-07 disposition**: §7.4 header-comment "self-verification" is documentation, not an executable gate. Replace with pytest + pre-commit hook below.

17. Every figure has source citation in its caption (per `.claude/rules/docs.md` §4)
18. `python scripts/verify_docs_provenance.py docs/storya_paper_draft.md` passes before send to H博士
19. **NEW per Codex A-07**: Write `tests/test_paper_figs_provenance.py` (pytest) that:
    - Parses each `paper_figs/fig_*.py` header `# SOURCE_CONTRACT:` block (YAML-in-comment format)
    - Reads the cited CSV at the cited path; extracts the cited row/column
    - Asserts the script's headline numeric output (e.g., main bar height, point estimate annotation) matches the source CSV to ≥3 decimal places
    - Computes source CSV MD5 + git SHA; stores in `paper_figs/.provenance_locks.json` to detect post-hoc CSV mutation
20. **NEW per Codex A-07**: Add pre-commit hook `pre-commit run --all-files` invoking the above pytest before any commit that touches `paper_figs/*.py`. CI / commit fails if any fig script's claimed value drifts from CSV.
21. **NEW per Codex A-07**: Convert §4 "VERIFY"/"TBD" flags into concrete pytest cases. Each TBD becomes a `@pytest.mark.skip(reason="VERIFY at fig-script time")` placeholder that will FAIL once the corresponding fig script is written (forcing the value to be filled-in or the skip removed with rationale). This makes the verification protocol auditable and inversion-resistant — no fig script can ship with a TBD claim.

Concrete header format for fig scripts (NEW spec per A-07):
```python
# SOURCE_CONTRACT:
#   inputs:
#     - path: artifacts/storya_e6_dm_spa/bootstrap_ci.csv
#       columns: [IC_mean, IC_mean_ci_lo, IC_mean_ci_hi]
#       md5: e3a2... (computed at script-write time)
#   outputs:
#     - path: figures/T1_headline.tex
#       headline_values:
#         - B-GAT-IC_mean: 0.035
#         - B-GAT-IC_mean_ci_lo: 0.018
#         # ... etc
#   tolerance_decimal_places: 3
```

---

## §7. Constraints + risks

### §7.1 Drive sync constraint

- **Local** has only smoke 4 cells of E1 (per `.gitignore` design); per-day_ic .npy files for full 400 cells live on Drive at `/content/drive/MyDrive/GNN测试/experiments/storya_e1_anchor/per_day_ic/`
- For figures requiring per-day IC time series (F4, S3): either pull from Drive OR run figure scripts on Colab + push small PDFs back
- **Decision needed at script-writing time**: sync 400 .npy files from Drive (~50MB total) to local for offline figure work, OR script-on-Colab approach

### §7.2 Figure-count vs page-limit

- Upper bound here: ~25 figures + ~12 tables. ICAIF ACM SIG page limit is typically 8 pages + refs (extends to ~10 with supplementary). Realistic main-paper budget: **F1-F10 + T1-T6** (10 figures + 6 tables). Rest go to supplementary / online appendix.
- **H博士 to decide** at writing-time per Q1 in frontmatter

### §7.3 Reproducibility

- All figure scripts use canonical 10 seeds (`.claude/rules/experiments.md`)
- All scripts must include `git rev-parse HEAD` hash in figure metadata
- rcparams preset must be version-locked (matplotlib + mpl_sizes versions pinned via `pip freeze | grep -E "matplotlib|mpl_sizes"`)
- Output paths: ALL figures → `figures/`; ALL tables → `tables/`; PDF + PNG dual export per matplotlib-skill default

### §7.4 Honest-numbers integrity (Rule 9 #5)

- Several headline numbers in §4 above are quoted from project memory / plan citations rather than freshly read from CSVs (flagged "VERIFY" where applicable)
- **MANDATORY**: every figure script begins with a header comment block listing the source CSV columns + verification that the script's headline plot value matches the source value to ≥3 decimal places
- Codex Touchpoint 3 (results review) MUST run before docs/storya_paper_draft.md ships

---

## §8. Skills inventory for paper workflow

### §8.1 Currently auto-loaded (per system list 2026-05-27)

| Skill | Usage in paper workflow |
|-------|------------------------|
| `matplotlib` | Most figure scripts (PnL curves, heatmaps, bar plots, line plots, cost ladder) |
| `scientific-schematics` | F1 architecture diagram, walk-forward Gantt S15 |
| `scientific-writing` | §1-§7 prose drafting |
| `literature-review` | Plan §1.9 16-paper matrix verification (Codex C-06 deferred task) |
| `peer-review` | Self-review pass before submission |
| `citation-management` | BibTeX generation for 16+ refs |
| `venue-templates` | ACM SIG `.cls` + bibstyle |

### §8.2 Awaiting restart-to-load

| Skill | Status | Usage |
|-------|--------|-------|
| `nature-figure` | symlinked at `~/.claude/skills/nature-figure`, awaiting session reload | Alternative figure style for journal backup (Quant Finance has different conventions than ACM) |
| `nature-writing` | symlinked, awaiting reload | Optional secondary writing aid |
| `nature-polishing` | symlinked, awaiting reload | Final prose polish pass |
| `nature-response` | symlinked, awaiting reload | Reviewer-response letter draft (post-submission) |
| `nature-citation` | symlinked, awaiting reload | Cross-check with citation-management |
| `nature-data` | symlinked, awaiting reload | Reproducibility statement / data availability section |

### §8.3 Python packages

- `matplotlib`, `seaborn`, `numpy`, `pandas`, `scipy.stats` (already in gnn conda env)
- `mpl_sizes` 0.0.2 (just installed)
- `arch` (for SPA), `statsmodels` (NW-HAC) — already used in E6 scripts

---

## §9. Decision log + cross-references

### §9.1 Decisions encoded in this plan (need H博士 confirmation)

1. **Figure budget**: upper-bound 25 figures + 12 tables; H博士 to pick subset at writing
2. **Module split**: 13 individual `paper_figs/fig_*.py` scripts (modular) — H博士 confirmed Option Y pattern for compute_e6_edge_ablation.py, same pattern applied here
3. **HATS scope** (RESOLVED 2026-05-27 post Codex T1): HATS-3R-adapt (S&P 500, 10-seed × 5-fold, SAGE-Mean adapter; narrowed claim_scope per Codex A-05/06/07 — NOT pure Kim 2019 reproduction). See §4.18 for full disposition.
4. **rcparams source**: matplotlib-skill defaults + ICML preset from mpl_sizes + manual 5-line ACM-SIG override (Times serif, double-column 3.33" half-width / 7.0" full-width)
5. **PDF + PNG dual export**: keep matplotlib-skill default; PDF for inclusion in paper, PNG for slide / quick share

### §9.2 Tri-doc cross-reference

→ progress: 2026-05-27-d (handoff is informational, no new experiment, but adds §6 work plan) | plan: 2026-05-27 (5 Decision Log rows landed) | analysis: 2026-05-27-c (last analysis entry — handoff references but does not modify)

### §9.3 References

- Plan: `/Users/heruixi/.claude/plans/handoff-session-ranking-swirling-lemur.md` (Story A v3, LOCKED 2026-05-26 evening, Codex Round E PASS-WITH-FIXES)
- Plan §1.9 honest caveats: source for §Limitations + ST7
- Plan §1.6 HATS GO: confirmed 2026-05-27 (§4.18 plan + §6.6 scheduling)
- Plan §1.4 statistical framework: source for §4.4 + N4 narrative
- Recent commits: `5bef3b9` (E1/E6/LOFO/Plan AAA T-1 diagnostic), `706be0d` (E3/E4 edge ablation), `8149fab` (honesty-pass corrections) — all pushed
- Rule 9 Touchpoint 1 (Plan Review) for this paper-figure-plan: OPTIONAL (this is a derivative work plan; Codex review of `run_storya_e1_anchor.py` + the v3 plan already passed Round E)
- Rule 9 Touchpoint 2 (Code Review): WILL FIRE for `analyze_storya_results.py` or per-module `paper_figs/fig_*.py` once written (per H博士 directive)
- Rule 9 Touchpoint 3 (Results Review): already DONE for E1+E6 + E3/E4 — review files at `artifacts/reviews/2026-05-27_codex_results_*.md`

---

## §10. Skill Chain Workflow — paper production pipeline

> **Goal** (per H博士 2026-05-27 directive "建立skill的联动"): orchestrate the 13 installed Claude Code skills into a single coherent paper-production pipeline. Each stage produces outputs consumed by the next stage; each transition has an explicit hand-off + verification checkpoint.

### §10.1 7-stage pipeline

```
┌────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1 — LITERATURE & SETUP                                               │
│ literature-review → citation-management → venue-templates                  │
│ Output: refs.bib + venue-template.cls + 16-paper related-work matrix (TARGET: expand to 19 via literature-review skill — see §10.2 Stage 1).md   │
│ Verification: scripts/verify_docs_provenance.py on related-work matrix    │
└────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2 — FIGURE / SCHEMATIC PRODUCTION                                    │
│ matplotlib (data viz) + scientific-schematics (F1 arch diagram, S15 Gantt) │
│ Style override: mpl_sizes 'ICML' preset + 5-line rcparams_storya.py       │
│ Output: figures/*.pdf + figures/*.png (28 figs upper bound)                │
│ Verification: each fig script's header comment block cross-checks source  │
│              CSV value to ≥3 decimals (per §7.4)                          │
└────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3 — DRAFT WRITING (IMRAD)                                            │
│ scientific-writing (primary) + nature-writing (secondary cross-check)     │
│ Two-stage: outline (using research-lookup) → flowing prose                │
│ Output: docs/storya_paper_draft_v1.md (full 7-section IMRAD)              │
│ Verification: scripts/verify_docs_provenance.py on every numeric claim    │
└────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4 — POLISH                                                           │
│ nature-polishing (prose) + peer-review (self-review simulation)            │
│ Output: docs/storya_paper_draft_v2.md (polished)                          │
│ Verification: peer-review skill produces structured findings (CRITICAL/   │
│              MAJOR/CONCERN per .claude/rules/docs.md §6 schema)           │
└────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ STAGE 5 — EXTERNAL REVIEW (Rule 9 Touchpoint 3)                            │
│ /codex-results-review on full draft + figures + tables                     │
│ Output: artifacts/reviews/2026-XX-XX_codex_results_paper_draft_A.md       │
│ Verification: every CRITICAL in Codex review must be FIXED before stage 6  │
└────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ STAGE 6 — SUBMISSION PREP                                                  │
│ venue-templates (ACM SIG check) + citation-management (BibTeX validate)   │
│ + nature-data (reproducibility statement / data availability)             │
│ Output: docs/storya_paper_submission.tex + refs.bib + figures.zip         │
│ Verification: ACM TAPS validator local pass; venue template compliance    │
└────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ STAGE 7 — POST-SUBMISSION (if reviewer feedback received)                  │
│ nature-response (reviewer-response letter draft)                           │
│ Output: docs/storya_paper_response_v1.md                                   │
└────────────────────────────────────────────────────────────────────────────┘
```

### §10.2 Per-skill invocation pattern (concrete)

| Stage | Skill | Invocation pattern | Inputs | Outputs |
|-------|-------|--------------------|--------|---------|
| 1 | `literature-review` | "Verify plan §1.9 16-paper matrix: confirm FinGAT/HIGSTM/HTAN entries against arXiv abstracts; add 3 missing papers (GRU-PFG, DishFT-GNN, DGT)" | plan §1.9 table | `docs/storya_related_work.md` |
| 1 | `citation-management` | "Generate BibTeX for the 19 papers in storya_related_work.md; cross-check DOI" | related work matrix | `refs.bib` |
| 1 | `venue-templates` | "Fetch ICAIF 2026 ACM SIG template" | venue=ICAIF 2026 | `template/sample-sigconf.cls` |
| 2 | `matplotlib` | "Generate F2 cumulative L/S PnL curves, 2×4 panel (univ × model), input source `experiments/storya_e1_anchor/results.csv` cols={IC_mean, Sharpe_*, n_periods}" + rcparams_storya | E6 result CSVs | `figures/F2_pnl_2x4.pdf` |
| 2 | `scientific-schematics` | "Produce Story A pipeline F1: data → features (Universe B/C) → model (GAT/SAGE/MLP/LGB) → walk-forward fold → eval (IC + Sharpe + SPA + DM + BH-FDR)" | written spec | `figures/F1_pipeline.svg` |
| 3 | `scientific-writing` | "Draft §5 Results subsection N1 using TBD figure refs F2/F3/F4 + table T1/T2; cite all numeric values per source CSV path" | figures + tables + analysis.md | `docs/storya_paper_draft_v1.md §5.N1` |
| 4 | `nature-polishing` | "Polish IMRAD §5 Results paragraph X for clarity + concision; preserve all numeric claims and citation anchors" | §5.X draft | §5.X polished |
| 4 | `peer-review` | "Simulate peer review of full §5 Results: assess methodology rigor (Hansen SPA, DM/HLN, BH-FDR), figure integrity, reporting standards" | full draft v2 | structured YAML review |
| 5 | `codex-results-review` (slash) | "Touchpoint 3 final on docs/storya_paper_draft_v2.md + figures/" | draft + figs | review .md in artifacts/reviews/ |
| 6 | `venue-templates` | "Confirm ACM SIG sample-sigconf submission compliance: 2-column, 7-page count, refs format" | tex source | pass/fail report |
| 6 | `nature-data` | "Generate Data Availability + Code Availability statements" | git rev + Drive paths | section.md |
| 7 | `nature-response` | "Draft reviewer-response letter addressing 3 reviewer comments preserving original claim_scope" | reviewer comments | response.md |

### §10.3 Hand-off contracts (machine-checkable)

Each pipeline transition has a **contract file** in `paper_workflow/contracts/` that lists:
- Required input artifacts (paths + MD5)
- Required output schemas (column names, frontmatter keys, etc.)
- Verification command (must exit 0 before next stage starts)

Example for Stage 2 → Stage 3 transition:

```yaml
# paper_workflow/contracts/stage2_to_stage3.yaml
stage_2_outputs_required:
  - path: figures/F1_pipeline.svg
    min_size_kb: 5
  - path: figures/F2_pnl_2x4.pdf
    min_size_kb: 20
  - path: figures/F3_lofo_heatmap.pdf
  # ... (all 10 main figures)
  - path: tables/T1_headline.tex
  - path: tables/T2_3col_robustness.tex
stage_2_verification_command: |
  python scripts/verify_figures_complete.py --required-list paper_workflow/required_figs.yaml
  python scripts/verify_docs_provenance.py docs/storya_paper_draft_v1.md
```

### §10.4 Implementation NOT in this handoff

This §10 only documents the chain; CONCRETE implementation:
- `paper_workflow/contracts/` directory with 6 stage-to-stage contract YAMLs — TO BE WRITTEN during script work
- `scripts/verify_figures_complete.py` — TO BE WRITTEN (1 day effort) to mechanically check Stage 2 → 3 transition
- Slash command `/paper-stage-advance` that runs the next stage's verification before triggering the next skill — OPTIONAL stretch goal

For now, H博士 + Claude invoke skills manually per §10.2 invocation patterns; the chain documentation prevents skill-overlap or sequence-mistakes.

### §10.5 Verification hooks for Rule 9 integration

- **Stage 2 (figures)** → Rule 9 Touchpoint 2 (Code Review) fires on each `paper_figs/fig_*.py`
- **Stage 5 (external review)** = Rule 9 Touchpoint 3 (Results Review) on full draft
- **Stage 6 (submission)** → final `verify_docs_provenance.py` pass before send-to-H博士
- All 3 Rule 9 touchpoints + the skill-chain transitions log to `progress.md` with the standard entry format

---

## §11. Limitations cross-reference matrix (ADDED 2026-05-27-g per Codex T1 A-03 MAJOR)

> **Goal**: every data-leakage caveat surfaced in §4 must trace EXPLICITLY to a paper §Limitations row in ST7 + a §Methodology disclosure + an analysis.md item. Codex A-03 flagged that the original handoff named "should be in Limitations" without naming WHICH limitation row WHERE. This matrix closes the loop.

| Caveat ID | Source (§4 entry) | Verdict / severity | analysis.md row | Plan §1.9 caveat # | Paper §Limitations row (ST7) | Paper §Results caveat sentence | Paper §Methodology disclosure |
|-----------|-------------------|--------------------|--------------------|--------------------|------------------------------|--------------------------------|-------------------------------|
| L1 | §4.5 Plan AAA T-1 diagnostic | **LOW STABILITY** (5/15) per artifact summary.md line 27; H博士 verdict A defers full re-run | analysis.md 2026-05-27-a Q4 Item 7 (revised 2026-05-27-g per Codex A-02) | Plan §1.9 caveat #5 (Plan AAA Alpha158 same-day leak provenance) + caveat #7 (T-1 diagnostic detail) | ST7 row 5 (composition basis leak); ST7 row 7 (T-1 diagnostic LOW STABILITY verdict) | "Universe C composition derives from a leak-affected Plan AAA ranking; T-1 diagnostic confirms LOW STABILITY (5/15); a definitive re-ranking with T-1-shifted Alpha158 is deferred to future work and the current paper carries this caveat." | §3 Methodology must disclose Universe C composition source + acknowledge that the Plan AAA ranking step used same-day Alpha158 (the runtime T-1 shift in `build_universe_C` keeps E1 IC values leak-free, but the COMPOSITION BASIS — which 51 features were selected — is not) |
| L2 | §4.18 HATS-3R-adapt sector PIT | **ACCEPTED-AS-CONCERN** per Codex T1 (HATS) A-01 | (n/a — pre-experiment; HATS results not yet) | HATS plan §Limitations (added per A-01 disposition) | ST7 row 8 (HATS sector PIT — `data/reference/sp500_sectors.csv` is one snapshot 2026-02-09, no PIT history) | "Sector edges in HATS-3R-adapt use a single-snapshot membership table fetched 2026-02-09; we do not have point-in-time sector composition history. Sector-based edges therefore carry a small look-ahead risk for any stock that changed sector during the 5y window." | §3 Methodology must declare the sector source + snapshot date |
| L3 | §4.18 HATS-3R-adapt claim_scope | **NARROWED** per Codex T1 A-05/06/07 | (n/a — by design) | HATS plan claim_scope block | ST7 row 9 (HATS not pure Kim 2019 reproduction) | "We adapt the HATS architecture by replacing Wikidata KG with sector/correlation/news edges, replacing GRU with SAGE-Mean, and evaluating on S&P 500 (not KOSPI 200). Results characterize HATS-3R-adapt as a Story A baseline comparator, not as a reproduction of Kim et al. 2019." | §2 Related Work must NOT cite as "we reproduce Kim 2019"; §4 Setup must explain the 4 architectural deviations |
| L4 | §4.6 horizon ablation | **VERIFIED 21d horizon selection** (no caveat — methodology choice with literature precedent + ablation evidence) | analysis.md 2026-05-27-a Q1 | (n/a — locked choice) | (n/a — discussed in §3 Methodology as locked choice with sensitivity analysis cited) | (n/a in §Limitations; positive in §3) | §3 Methodology cites horizon ablation as 21d-selection evidence |
| L5 | §4.12 Loss horserace + §4.26 ranking-loss + §4.13 Diagnostic_price | **REPLICATION FAILURE FRAMING** (Part B v4 wf5 headlines did not replicate in Stage 1 framework) | analysis.md 2026-04-27-b | Plan §1.9 caveat #3 (LSTM absence) — related but different | ST7 row 10 (loss-function choice rationale + framework-replication caveat) | "Earlier in this project's history, a Part B v4 walk-forward run reported MLP_price 21d IC = +0.037 / SAGE_price +0.027 (`experiments/wf5_results.csv`); a follow-up replication in the Stage 1 framework with identical 9-dim S_price features (`experiments/loss_horserace/results_diagnostic_price.csv`) recovered MLP IC = −0.004 and SAGE IC = −0.057. We treat this as evidence that 21d cross-sectional IC is framework-sensitive and prefer the more recent Story A E1 framework (which uses canonical 10 seeds, 21d purge, and the §3 Methodology multi-seed protocol) for the headline results." | §3 Methodology must describe framework lineage |
| L6 | §4.1 E1 Fold 4 dominance | **REGIME RISK** (Q2-2025 fold is a known outlier per project Rule 10) | analysis.md 2026-05-27-a Q4 Item 6 | Plan §1.9 caveat #6 | ST7 row 6 (Q2-2025 regime risk) | "Most positive Story A IC and Sharpe results lose 38-72% magnitude when Fold 4 (test period Q2-2025) is dropped (LOFO-4 column in Table 2). We report both full and LOFO-4 columns and treat both as equally informative pending future evaluation on additional regimes." | §3 Methodology cites the LOFO-4 protocol; §4 Setup cites the 5-fold walk-forward with Fold 4 as known regime outlier |
| L7 | §4.3 E4-α full bundle harm | **NEGATIVE FINDING** (multi-edge bundle does not improve over correlation-only) | analysis.md 2026-05-27-c Q2 Finding 1+2 | (n/a — primary finding, not caveat) | (n/a in §Limitations; framed as positive negative finding in §5 Results) | "Bundling sector + news edges on top of the correlation-only baseline does not improve IC under either full or LOFO-4 evaluation; 0/5 pairs survive BH-FDR q=0.05" | §3 Methodology cites the BH-FDR family-of-5 protocol |
| L8 | §4.17 + §4.23 Phase 5 audits did NOT catch Plan AAA T-1 leak | **METHODOLOGICAL CAVEAT** | (n/a — methodology lineage) | (n/a) | ST7 row 11 (audit scope caveat) | "Our Phase 5 features audit (artifacts/audits/phase5_features_audit.md) covered Phase 5 features but NOT the separately-built Alpha158 features used by Plan AAA. The Plan AAA T-1 leak was therefore discovered LATER via the T-1 diagnostic (L1 above), not by the Phase 5 audit." | §3 Methodology must clarify audit scope to avoid implying full-pipeline coverage |

**Action items derived from this matrix** (must close before paper submission):
1. analysis.md §Limitations Item 7 — already revised 2026-05-27-g to lead with LOW STABILITY verdict (per A-02 fix)
2. Plan §1.9 — add caveat #7 explicit row (currently caveats #1-#6 documented; new #7 for T-1 LOW STABILITY needs to land in `/Users/heruixi/.claude/plans/handoff-session-ranking-swirling-lemur.md` §1.9)
3. ST7 prose table — 11-row generation deferred to `scientific-writing` skill in §10.2 Stage 3
4. F10 + S4 captions — explicit "5/15 LOW STABILITY" annotations per §4.5 revised version
5. §3 Methodology + §4 Setup paragraphs — disclosures per the L1-L8 mapping above
