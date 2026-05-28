# storya_e1_6_hats — HATS-3R-adapt Baseline (Story A §1.6 STRETCH)

## 当前内容

| File | Purpose |
|---|---|
| `prereg.json` | Pre-committed decision rules + claim scope (LOCKED before first non-smoke cell, per Codex Plan Round A finding CODEX-A-08 disposition) |
| `hp_grid.json` | Locked HATS-3R-adapt hyperparameters (mirrors E1 NN_HPARAMS for apples-to-apples) |
| `_meta.json` | Run metadata: experiment id, paper inspired-by, adaptations vs Kim 2019, cost ladder spec, sector PIT limitation note |
| `results.csv` | One row per cell, schema = E1 RESULTS_COLUMNS + diagnostic columns (n_corr_edges, n_sector_edges, n_news_edges_avg, alpha_mean_{corr,sector,news}_test, alpha_max_fraction_collapsed_test) |
| `manifest.csv` | Resume bookkeeping (cell_id, fold, seed, status, timing, err) |
| `per_day_ic/` | `B_HATS-3R-adapt_s{seed}_f{fold}.npy` — 1-D per-day IC arrays for downstream block bootstrap + DM/HLN |
| `alpha_diag/` | Per-cell per-epoch alpha statistics CSV (DIAGNOSTIC ONLY — not gating; no attention-specific claims without uniform-α control per prereg.json) |
| `smoke_benchmark.csv` | `--smoke` mode output (1 cell) |

## 关键文件速查

- **Implementation**: `/Users/heruixi/Desktop/GNN-Testing/run_storya_e1_6_hats.py`
- **Plan**: `/Users/heruixi/.claude/plans/hats-baseline-reproduction-delightful-lighthouse.md`
- **Codex Plan Touchpoint 1 review**: `artifacts/reviews/2026-05-27_codex_plan_A.md`
- **Upstream baselines** (must concat for E6 comparison):
  - E1 anchor: `experiments/storya_e1_anchor/results.csv` (400 cells, all 4 models × B/C)
  - E3 news edge: `experiments/storya_e3_news_edge/results.csv` (50 cells, SAGE-Mean + news)
  - E4-α: `experiments/storya_e4_alpha/results.csv` (100 cells, SAGE-Mean + sector ± news)

## Scope clarification

**This is NOT a literal reproduction of Kim et al. 2019 HATS.** It is an adaptation
that:
- Skips the GRU encoder; uses Universe B 10-dim hc features directly
- Substitutes Wikidata 75-relation set with 3 project relations (correlation_frozen
  / sector_GICS11_static / news_cooccurrence_PIT)
- Regresses on 21d CS-z-scored returns instead of up/neutral/down classification
- Uses a simplified Linear(hidden, 1) shared relation-attention scorer instead of
  Kim §3.2's relation-embedding concatenation

Per `prereg.json` `claim_scope`: any Template-1 conclusion speaks only to this
adapted module on Story A's strict 21d ranking eval; does NOT speak to Kim's
published HATS performance.

## Cell budget

- 1 model × 1 universe (B) × 10 canonical seeds × 5 walk-forward folds = **50 cells**
- `cell_id = 400 + fold_idx*10 + seed_idx`, range **[400, 449]** (offset to avoid E1 [0,399] collision per Codex CODEX-A-04)
- Wall time **PROVISIONAL** ~13 min/cell A100; must be re-locked from 1-cell smoke benchmark before launching 50 cells (Codex CODEX-A-10)

## Decision rules (LOCKED per Codex CODEX-A-08)

See `prereg.json` `decision_rules_locked_2026_05_27`. Three gates:

| Verdict | Condition |
|---|---|
| **POSITIVE** | ΔIC(HATS−GAT) > +0.005 AND BH-adj HLN p < 0.05 AND LOFO-4 ΔIC sign preserved |
| **NEGATIVE** | ΔIC < −0.005 OR (BH-adj HLN p > 0.20 AND LOFO-4 ΔIC ≤ 0) |
| **TIE** | \|ΔIC\| ≤ 0.005 AND BH-adj HLN p > 0.05 |

## E6 integration

HATS-3R-adapt enters per-universe B SPA (M=3→4) and per-universe B DM family (5→8 pairs).
**EXCLUDED from joint SPA** per H博士 2026-05-27 (joint stays M=6, E1-only).

Required additive changes to `compute_e6_dm_spa.py`:
- `--include-hats-csv PATH` flag
- Hard date-alignment (no silent min-length truncate) per Codex CODEX-A-03
- cell_id injectivity assertion across concat per Codex CODEX-A-04
- LOFO-4 / Fold-4-only post-process via `analyze_hats_lofo.py` (mirror `analyze_e1_lofo.py:167-169`)

## 变更日志

- 2026-05-27: 文件夹创建 + plan/code skeleton 写入 (Touchpoint 1 PROCEED-WITH-FIXES applied; Touchpoint 2 PENDING) （→ progress: 2026-05-27-d）
