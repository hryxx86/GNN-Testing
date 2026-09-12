---
handoff_date: 2026-05-26
last_completed: "2026-05-26-h: Codex Round D fixes applied + independently verified (5 D-series CRITICAL/MAJOR + 5 C-series re-mappings) across plan + prereg + schema doc + Round C review file; Round E queued"
in_flight:
  - id: codex-round-e-plan-review
    file: /Users/heruixi/.claude/plans/handoff-session-ranking-swirling-lemur.md
    status: "queued — Round E verifies the 10 Round D fixes (D-01..D-05 + C-01..C-05 re-mappings); use /codex-plan-review skill with prompt scoped to verification-only (not redesign)"
    blockers: []
  - id: storya-e1-anchor-script
    file: run_storya_e1_anchor.py (to be created)
    status: "blocked on Codex Round E PASS verdict; will port archived/scripts/run_horizon_ablation.py + add LightGBM_price + Universe B/C switch + 10-canonical-seed list; NO LSTM per §1.1 limitation note"
    blockers: ["codex-round-e-plan-review"]
  - id: storya-e3-news-edge-build
    file: scripts/build_news_edge_source.py (to be created)
    status: "blocked on Round E PASS; ~30 min M4 wall time aggregating 1.7M news rows → ~600-800K unique articles per experiments/storya_e3_news_edge/news_edge_source_schema.md v2 spec (NYSE session_close cutoff per D-03)"
    blockers: ["codex-round-e-plan-review"]
  - id: c06-literature-matrix-reverify
    file: /Users/heruixi/.claude/plans/handoff-session-ranking-swirling-lemur.md §1.9
    status: "DEFERRED to paper §2 writing phase per Round C disposition; not an E1 blocker"
    blockers: []
  - id: c08-bc-hp-transfer-paper-limitation
    file: paper draft §Limitations (not yet started)
    status: "DEFERRED to paper writing phase per Round C disposition; no plan/prereg change needed"
    blockers: []
open_questions:
  - "Universe C (51-dim Plan AAA top-15 members) feature list: extract from artifacts/plan_aaa/ranking.csv top 15 groups; need to enumerate exact members and confirm Alpha158 slicing aligns with sp500_5y_alpha158_features_raw.npy column order"
  - "E3 news-as-edge co-occurrence: exact lookback window (5d default per plan but H博士 not yet confirmed) + sentiment-weighting toggle"
  - "E5 Mamba-SAGE: GO/SKIP decision after E1 + E3 + E4 finish — based on time budget remaining"
  - "Literature matrix axes (Codex A-06 still-open): horizon × feature_universe × graph_relation × regime × PIT × seed_count × overfit_diagnostic — need to populate 8-12 published papers"
file_state:
  modified_since_last_commit:
    - progress.md (added 2026-05-26-a Story A pivot + 2026-05-26-b Codex Round A entries; needs 2026-05-26-c plan v3 simplification entry)
    - plan.md (added 2026-05-26-a phase entry + 6 → 14 Decision Log rows after v2; needs v3 rows: drop adaptive, drop DSR, drop PBO, add SPA, add DM, add cost ladder)
    - docs/session_handoff_2026-05-26.md (this file, updated to v3 frontmatter)
    - /Users/heruixi/.claude/plans/handoff-session-ranking-swirling-lemur.md (STATE/NEXT ACTIONS rewritten to v3; Sections 1.1+ still v2)
    - .gitignore, CLAUDE.md (modified, from prior session)
  new_files:
    - artifacts/reviews/2026-05-26_codex_plan_A.md (Codex Round A; verdict BLOCK-EXECUTION; v3 disposition update body pending)
    - experiments/storya_multiseed/prereg.json (v2 written; v3 rewrite pending)
    - /Users/heruixi/.claude/projects/-Users-heruixi-Desktop-GNN-Testing/memory/project_storya_pivot_2026-05-26.md
    - /Users/heruixi/.claude/projects/-Users-heruixi-Desktop-GNN-Testing/memory/project_gat21d_lucky_seed.md
    - /Users/heruixi/.claude/projects/-Users-heruixi-Desktop-GNN-Testing/memory/feedback_adaptive_seed_design.md (note: v3 supersedes adaptive design entirely; memory note still useful as historical record of why adaptive failed Codex review)
rule9_status:
  touchpoint_1_plan: "ROUND_A_BLOCK_(2026-05-26) → ROUND_C_BLOCK_(2026-05-26) → ROUND_D_BLOCK_(2026-05-26-evening) → ROUND_E_QUEUED (10 Round D fixes applied + independently verified per progress 2026-05-26-h)"
  touchpoint_2_code: PENDING  # blocked on Round E PASS + run_storya_e1_anchor.py creation
  touchpoint_3_results: PENDING  # blocked on T2 + experiment runs
next_actions:
  - "Trigger Codex Round E via /codex-plan-review .claude/plans/handoff-session-ranking-swirling-lemur.md with verification-scope prompt (NOT redesign)"
  - "If Round E verdict ∈ {PASS, PASS-WITH-CONCERNS, PROCEED-WITH-FIXES}: write run_storya_e1_anchor.py (port from archived/scripts/run_horizon_ablation.py, add LightGBM_price + Universe B/C switch + 10-canonical-seed list, NO LSTM)"
  - "If Round E BLOCK-EXECUTION: address Round E findings with same independent-verification protocol per progress 2026-05-26-h lesson"
  - "After E1 script created: Codex Touchpoint 2 code review via /codex-code-review run_storya_e1_anchor.py"
  - "After T2 PASS: write scripts/build_news_edge_source.py (per schema doc v2); build derived parquet (~30 min M4)"
  - "Smoke benchmark: 4 cells (one per model) × 1 seed × 1 fold on Universe B; measure per-cell wall time; gate full E1 launch per plan §1.10"
  - "Full E1 launch: 400 cells, ~35-40h A100 with checkpoint/resume; then E3 (50 new cells, ~4-5h), E4-α (100 new cells, ~8-10h), E6 post-process (~5 min CPU); optional E5 Mamba ablation (250 cells, ~21h, GO/SKIP gated by remaining budget)"
codex_artifacts:
  round_a_review: artifacts/reviews/2026-05-26_codex_plan_A.md
  round_a_verdict: BLOCK-EXECUTION  # 3 CRITICAL + 6 MAJOR + 2 CONCERN
  round_c_review: artifacts/reviews/2026-05-26_codex_plan_C.md
  round_c_verdict: BLOCK-EXECUTION  # 1 CRITICAL + 5 MAJOR + 2 CONCERN
  round_d_review: artifacts/reviews/2026-05-26_codex_plan_D.md
  round_d_verdict: BLOCK-EXECUTION  # 1 CRITICAL + 4 MAJOR (D-series) + 5/8 Round C dispositions failed independent verification
  round_e_review: pending — to be triggered after 10 Round D fixes commit; scope verification-only
---

# Session Handoff — 2026-05-26

## TL;DR

This session was a long strategic discussion that resulted in a **major paper-direction pivot**: from Plan AAA wording fixes (carryover from 2026-05-23-d) to a new Story A "When Do GNNs Help in Cross-Sectional Stock Ranking" paper targeting ICAIF 2026.

The pivot was forced by H博士's direct questioning that surfaced six brutal facts already present in our data but not previously framed honestly together:

1. GAT 21d IC=0.044 is lucky-seed (5-seed mean=0.032, CV=55%, seed=1024 gave IC=0.00182)
2. MLP_price 21d (IC=0.0374, Sharpe=2.35) ≥ GAT 21d (IC=0.0321, Sharpe=0.84) — graph adds nothing on Sharpe
3. News-as-feature catastrophic at 21d: ΔIC = −0.045
4. HGT 4-edge IC=0.003 worse than SAGE 1-edge IC=0.012 (at 1d, not retested at 21d)
5. Plan AAA: 0/61 positive groups pass BH-FDR; 1/61 (CORD20+1) FDR-rejected NEGATIVE direction
6. Plan AAA used only 3 seeds vs canonical 10-seed standard

## What just happened (discussion arc)

The session went through 5 phases:

1. **Plan AAA retrospective** — H博士 asked for full ranking; reviewed Top 10 / Bottom 5 / FDR status
2. **Methodology stress test** — H博士 challenged: does FDR matter? Does cherry-pick detection matter? Do others do it?
3. **Literature deep-dive** — for each of (regime, sector, LSTM, Mamba, Transformer, news encoding), identified what's been done in published GNN-finance work vs genuine gaps
4. **Story brainstorm** — proposed 7 candidate paper stories; converged on Story A (conditional findings) + Mamba-SAGE insurance
5. **Methodology pre-commitment** — H博士 chose "先跑 30 seeds 试水" → I formalized as adaptive 30→100 design with pre-committed extension rule (avoids selection bias)

## Key findings (all derived from existing files, independently re-verified)

### From experiments/horizon_ablation_results.csv (15 runs/model, 3 seeds × 5 folds)

| Model | Features | IC mean | Sharpe_net mean |
|-------|----------|---------|-----------------|
| MLP_price | 9-dim price only | +0.0374 | +2.351 |
| SAGE-Mean_price | 9-dim price only | +0.0269 | +1.009 |
| MLP_all | 9-dim + 772 news | −0.0078 | −0.530 |
| SAGE-Mean_all | 9-dim + 772 news | +0.0111 | −2.766 |

→ news features at 21d are catastrophically harmful for both MLP and SAGE; MLP_price beats SAGE_price.

### From docs/analysis.md lines 1015-1028 (GAT 21d 5-seed stability test, 2026-04-08)

| Seed | IC | Sharpe |
|------|-----|--------|
| 42 | 0.05140 | 1.262 |
| 123 | 0.03800 | 0.984 |
| 456 | 0.04549 | 1.241 |
| 789 | 0.02402 | 0.828 |
| 1024 | 0.00182 | −0.096 |
| Mean | 0.03215 | 0.844 |
| Std | 0.01771 | — |
| CV | 55.1% | — |

→ commonly cited "GAT 21d IC=0.044" comes from seed=42 single run; 5-seed mean is 0.032 with seed=1024 being a full failure.

### From artifacts/plan_aaa/ranking.csv (Plan AAA grouped permutation Δ-IC, 168 features → 61 groups, 30 cells)

- Rank 1: `hc_mom12m` mean_delta_IC = +0.007899, BH p_adj = 0.647 (NOT rejected)
- Rank 61: `CORD20+1` mean_delta_IC = −0.004020, BH p_adj = 0.021 (REJECTED, NEGATIVE direction)
- 0/61 positive groups pass BH-FDR at q=0.05
- 0/7 hand-curated groups pass BH-FDR

## Approved plan

**File**: `/Users/heruixi/.claude/plans/handoff-session-ranking-swirling-lemur.md`

**Story A core experiments** (in execution order):

1. **§1.1 Multi-seed model comparison** — adaptive 30→100 seed × 5 models × 5 folds × 21d
   - Pre-commit rule: extend per-model only if (30-seed mean IC > 0.020) AND (CV > 30%)
   - Models: GAT_price, SAGE-Mean_price, MLP_price, LSTM_price, Mamba-SAGE (insurance)
2. **§1.2 News-as-edge co-occurrence** (test "news-as-feature hurts, news-as-edge helps" hypothesis)
3. **§1.3 HGT 21d rerun** (test if HGT IC=0.003 at 1d was horizon artifact)
4. **§1.4 Cherry-pick detection suite**: DSR + PBO + bootstrap CI (no surveyed GNN-finance paper has these)
5. **§1.5 Mamba-SAGE prefix** (positive anchor / insurance)
6. **§1.6 HATS baseline reproduction** (STRETCH only if budget allows)

**Compute estimate**: 30-80h A100 total across all experiments

**Timeline**: 8 weeks to ICAIF 2026 submission-ready draft (+2 weeks if HATS reproduction included)

## Mamba + Regime + Sector — discussion archive (deferred)

All discussion archived in plan file §3. Key points:

- **Mamba**: Route A (per-stock temporal encoder → SAGE GNN cross-stock); input T=21, D=13-40; vanilla first, then SAMBA Bi-Mamba + AGC upgrade path
- **Regime**: HMM filtered posterior (NOT smoothed) per Cube Exchange / QuantStart standard; soft conditioning (concat as feature) not specialist sub-models
- **Sector**: GICS 11-sector already at `data/reference/sp500_sectors.csv`
- **Regime × Sector fusion**: Route A (regime-conditional sector edge weight) recommended novelty if Story A succeeds

## Negative-result publication framing (insurance)

4 publishable templates identified with precedent:

- **T1: Strict eval reveals overstated claims** (Hou-Xue-Zhang 2020 RFS precedent)
- **T2: When-X-helps-when-X-hurts conditional findings** ← Story A core
- **T3: Failure-mode diagnose + mitigate**
- **T4: Methodology framework** (DSR/PBO precedent: Lopez de Prado)

All four can be combined into Story A paper structure.

## Outstanding items from prior sessions

- **Plan AAA wording fixes (A-01, A-02 from 2026-05-23-d)**: now DEFERRED. Plan AAA becomes a §X feature ablation within Story A, not a standalone deliverable. H博士's decision on wording fixes still pending but no longer paper-blocking.

## Files to read for next session

- `/Users/heruixi/.claude/plans/handoff-session-ranking-swirling-lemur.md` — full approved plan (Story A §1-§8 + Mamba/Regime/Sector archive §3)
- `progress.md` 2026-05-26-a entry
- `plan.md` 2026-05-26-a entry + Decision Log rows
- `experiments/horizon_ablation_results.csv` — 21d horizon ablation source numbers
- `artifacts/plan_aaa/ranking.csv` — Plan AAA 168-feature ranking source
