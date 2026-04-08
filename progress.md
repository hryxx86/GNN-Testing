# Progress Log

> **做了什么。** 按时间记录已完成的工作。每个条目与 `plan.md` 和 `docs/analysis.md` 时间对齐。

---

## 2026-02-XX: Phase 1 + B + 2 Pilot + A (earlier sessions)

- [x] Phase 1: 502-stock correlation network, GCN embedding, visualizations
- [x] Phase B: 636 dynamic graph snapshots, sensitivity heatmaps, hub evolution
- [x] Phase 2 Pilot: 9 hub stocks, 480 Factiva events, MiniLM 384-dim → LR 0.62, GraphSAGE 0.64
- [x] Phase A: EODHD 1.7M articles → 1.06M cleaned → 1.7M events, FinBERT 768-dim + sentiment
- [x] Infrastructure: directory reorg, bug fixes, Drive paths

---

## 2026-02-27-a: Phase C notebook created

- [x] Created `phase_c_model_training.ipynb` (10 cells)
- [x] HeteroGNN 2-layer GraphSAGE, full-batch, 3 edge types
- [x] News 771-dim, stock 12-dim, time split train/val/test
- [x] Switched mini-batch → full-batch (A100 80GB)

→ plan: `2026-02-27-a` | analysis: N/A

## 2026-02-27-b: Phase C v1 experiments run (Colab A100)

- [x] Ran B1, B2, A1, A2, A3, Full — all AUC ≈ 0.50
- [x] Data quality verified (no bug, signal too weak)

| Experiment | Val AUC | Test AUC |
|-----------|---------|----------|
| B1: LR + FinBERT | 0.5018 | 0.4976 |
| B2: LR + Sentiment | 0.5044 | 0.5027 |
| A1: GNN news→stock | 0.5085 | 0.4913 |
| A2: + correlation | 0.5122 | 0.4949 |
| A3: + sector | 0.5133 | 0.4961 |
| Full: all edges | 0.5133 | 0.5069 |

→ plan: `2026-02-27-b` | analysis: `2026-02-27-b`

## 2026-02-27-c: Diagnostic cells + docs restructure

- [x] Added Cell D.1: data-level diagnostics (4 analyses, 4 plots)
- [x] Added Cell D.2: model prediction diagnostics (4 analyses, 4 plots)
- [x] Created `plan.md`, `docs/analysis.md`
- [x] Updated MEMORY.md with tri-doc update rules
- [x] Run D.1 + D.2 on Colab → see 2026-03-03-a

→ plan: `2026-02-27-c` | analysis: `2026-02-27-c`

---

## 2026-03-03-a: D.1 + D.2 Diagnostics Run on Colab

- [x] D.1: Label noise analysis — 26.5% events in noise zone
- [x] D.1: Sentiment alignment — FinBERT alignment ~51.5% (near-random)
- [x] D.1: Per-sector stats — IT dominates (420K events)
- [x] D.1: Temporal stability — clear regime shifts (2022Q2 bear, 2023Q4 rally)
- [x] D.2: LR prediction separation — mean separation = -0.00030 (zero)
- [x] D.2: Per-sector AUC — max 0.512 (Utilities, 9K events)
- [x] D.2: Sentiment confidence AUC — no improvement at any level
- [x] D.2: Return magnitude AUC — no improvement for large moves

**Key finding**: FinBERT title-level sentiment has zero predictive power for next-day returns across ALL conditions.

-> plan: `2026-03-03-a` | analysis: `2026-03-03-a`

## 2026-03-03-b: Literature Review — NLP+GNN Stock Prediction Papers

- [x] Searched & analyzed 6 papers: THGNN, DGRCL, DASF-Net, ChatGPT-GNN, Kengmegni 2024, Sentiment-Size Nexus
- [x] Identified key reasons our AUC~0.50 is expected (not a bug)
- [x] Found: most GNN papers use price-only (no NLP); NLP papers use 12-30 stocks
- [x] Found: DGRCL on 1,026 NASDAQ stocks gets only 53% acc (same regime as us)
- [x] Found: multiple 2024-2025 papers confirm FinBERT sentiment lacks predictive power for large-cap
- [x] Updated analysis.md with full comparison table and 7 critical findings

-> plan: `2026-03-03-b` | analysis: `2026-03-03-b`

## 2026-03-03-c: Plan Revision — Signal-First Roadmap

- [x] Diagnosed plan_v2 fatal ordering issue (selective prediction before signal exists)
- [x] Researched financial LLM benchmarks (FinBERT F1=0.88 > GPT-4o zero-shot 0.86)
- [x] Identified LLM value = impact prediction, not sentiment replacement
- [x] Selected GPT-4o-mini ($0.45/7K samples, best academic credibility + JSON schema)
- [x] Rewrote plan.md with 4-phase roadmap: signal fix → LLM validation → selective prediction → paper
- [x] Added Go/Stop gates at Phase 1 exit

**Key decisions**:
- Reorder: signal fix (market-adjusted labels, dedup, momentum) BEFORE selective prediction
- Drop Option B (shrink stock universe) — doesn't solve EMH, weakens paper
- GPT-4o-mini for impact prediction, not sentiment (FinBERT already strong at sentiment)

→ plan: `2026-03-03-c` | analysis: N/A

## 2026-03-03-d: Phase 1 Signal Fix — Code Written

- [x] Phase 1a: News deduplication cell (sparse matrix averaging, groupby)
  - Same (date, ticker) → one stock-day record
  - FinBERT embeddings: mean-pooled via scipy sparse matrix
  - Expected: 1.7M events → ~250-500K stock-days
- [x] Phase 1b: Market-adjusted labels cell
  - Label: (stock_return - equal_weight_market_return) > 0
  - SPY not in prices file → used equal-weight S&P 500 mean as market proxy
  - Reports noise zone comparison (raw vs market-adjusted)
- [x] Phase 1c: Momentum/volatility features cell (9 features)
  - 3 windows (5/10/21d) × 3 stats (return mean, return std, momentum)
  - All use T-1 close via shift(1) — no look-ahead
  - Merged per-event via (trading_day, ticker) lookup
- [x] Modified build-graph cell: news features 771-dim → 780-dim
- [x] Updated baselines cell: added B3 (sent+momentum), B4 (momentum only), B5 (XGBoost)
- [x] **Phase 1 preprocessing run on Colab A100** (see 2026-03-03-f)

→ plan: `2026-03-03-d` | analysis: `2026-03-03-f`

## 2026-03-03-f: Phase 1 Preprocessing — Colab Results

- [x] 1a: 1,698,182 events → 437,194 stock-days (3.88:1 compression)
- [x] 1b: Market-adjusted labels — pos_rate 0.5164→0.4925, noise zone 27.6%→23.0%
- [x] 1c: Momentum features — 99.5% coverage (434,833/437,194)
- [x] Total pipeline: 40.9s, all shapes verified

→ plan: `2026-03-03-f` | analysis: `2026-03-03-f`

## 2026-03-03-g: Phase 1d Baseline Matrix — All Test AUC ≈ 0.50

- [x] B1 LR+FinBERT: Test AUC 0.4993 (random)
- [x] B2 LR+Sentiment: Test AUC 0.5031 (random)
- [x] B3 LR+Sent+Momentum: Val 0.5182 → Test 0.4965 (**overfitting**)
- [x] B4 LR+Momentum: Val 0.5178 → Test 0.4987 (**overfitting**)
- [x] B5 XGBoost: Test AUC 0.5046 (best, still below 0.52 Go threshold)

**Go/Stop verdict**: Stop condition triggered — all test AUC < 0.51 after signal fix.

→ plan: `2026-03-03-g` | analysis: `2026-03-03-g`

## 2026-03-03-h: Selective AUC + GNN v2 — STOP Confirmed

- [x] Added selective AUC analysis cell to notebook (Cell 12)
- [x] Ran on Colab: B1-B5 selective AUC + GNN Full (780-dim)
- [x] GNN Full test AUC = 0.5002 (random, graph adds nothing)
- [x] Max selective AUC@10% = 0.5071 (B1), far below 0.54 Go threshold
- [x] Max selective AUC@5% = 0.5154 (B1), within statistical noise (~2K samples)
- [x] Momentum features hurt selective AUC (B3/B4 @10% < 0.50)

**STOP condition confirmed**: All three Go criteria unmet. No signal in tails.
**Remaining low-cost option**: Phase 2 LLM features (~$0.45) — different signal dimension.

→ plan: `2026-03-03-h` | analysis: `2026-03-03-h`

## 2026-03-03-i: Phase 2 LLM Validation — Code Written

- [x] Phase 2a cell: GPT-4o-mini structured output on dev-holdout (2023-Q4)
  - Reloads original events with titles
  - Samples ~7K events, calls API with json_schema structured output
  - Caches results to JSON for resume/reuse
  - Reports LLM output distributions (impact/direction/reasoning)
- [x] Phase 2b cell: 5-fold CV comparison
  - Encodes LLM output as 10-dim features
  - Compares: FinBERT 3-dim vs LLM 10-dim vs Combined 13-dim vs FinBERT emb 768-dim vs LLM+emb 778-dim
  - Impact-level subset analysis (high/medium/low AUC)
  - LLM direction prediction accuracy
  - Auto Go/Stop assessment (delta > 0.02 = worth full-scale)
- [x] **Colab run complete** — via OpenRouter API

→ plan: `2026-03-03-i` | analysis: `2026-03-04-a`

## 2026-03-04-a: Phase 2 LLM Results — NO Signal

- [x] GPT-4o-mini on 7K dev-holdout (2023-Q4) via OpenRouter — 0 errors, ~$0.45
- [x] LLM output: impact (med 49.5%, low 37.6%, high 13.0%), direction (neutral 44.3%, pos 37.1%, neg 18.6%)
- [x] LLM structured (10d) AUC = 0.5034 vs FinBERT (3d) AUC = 0.5025 → delta = +0.0009 (no signal)
- [x] High-impact subset AUC = 0.4762 (WORSE than random)
- [x] LLM direction accuracy = 0.5208 (random); high-impact+directional = 0.4989 (random)
- [x] Combined (13d) AUC = 0.5019, LLM+emb (778d) AUC = 0.5102 — no improvement
- [x] **Go/Stop: STOP confirmed. LLM delta < 0.02 → skip full-scale run, save $19**

**Conclusion**: All avenues exhausted. Event-level next-day S&P 500 return prediction is not feasible with NLP features (FinBERT or LLM). Strong EMH evidence.

→ progress: `2026-03-04-a` | plan: `2026-03-04-a` | analysis: `2026-03-04-a`

## 2026-03-03-e: Document Merge & Archive

- [x] Merged plan_v2.md useful parts into plan.md:
  - SOTA positioning tables (GNN SOTA + Selective Prediction SOTA)
  - Core Gap analysis
  - Paper narrative (elevator pitch)
  - 5 contribution points
  - Return/timing diagram (compact version)
  - 9 decision log entries not yet in plan.md
- [x] Created `archived/` folder
- [x] Moved `plan_v2.md` → `archived/plan_v2.md`
- [x] Moved `phase_d_design.md` → `archived/phase_d_design.md`
- [x] Kept `phase_f_design.md` in root (Phase 3 design spec, still needed)
- [x] Updated CLAUDE.md Rule 6: removed archived file references, added archive rule
- [x] Updated CLAUDE.md Rule 9: reflects current project state

→ plan: `2026-03-03-e` | analysis: N/A

---

## 2026-03-05-a: Phase B Parameter Analysis + Visualization Code

- [x] Analyzed 636 dynamic graph snapshots from sensitivity_analysis.csv
- [x] Identified best parameters: **w=126, t=0.6** (density 6%, std=0.064, 125 components)
- [x] Added 3 cells to `GNN测试1 colab.ipynb`:
  - Markdown: parameter selection rationale
  - Code: generate all 54 monthly snapshots as PNGs
  - Code: annotated edge count evolution + 6 regime snapshots
- [x] Updated docs/analysis.md with full 12-row parameter comparison table

→ plan: `2026-03-05-a` | analysis: `2026-03-05-a`

## 2026-03-05-b: Literature Survey — Ranking + HGT + Selective Prediction

- [x] Surveyed 10+ papers: MASTER (AAAI'24), FinMamba (2025), MDGNN (AAAI'24), THGNN (CIKM'22), HGAIT (ESWA'25), SelectiveNet (ICML'19), AUGRC (NeurIPS'24)
- [x] Confirmed DASF-Net "3-day optimal" is misleading (input aggregation, not prediction horizon)
- [x] Identified 5 key insights:
  1. Ranking target (IC/ICIR) is mainstream, not binary direction
  2. Calendar-driven is standard, not event-driven
  3. No paper combines GNN + SelectiveNet (gap!)
  4. No systematic horizon ablation exists in GNN literature
  5. Co-occurrence edges > fund-holding edges (Multi-GCGRU finding)

→ plan: `2026-03-05-b` | analysis: `2026-03-05-b`

## 2026-03-05-c: v3 Roadmap — Research Direction Pivot

- [x] Rewrote plan.md with v3 roadmap: Ranking + Dynamic HGT + Selective Prediction
- [x] 10 new decisions recorded in plan.md Decision Log
- [x] Key decisions:
  - Binary direction → Ranking (IC/ICIR/Sharpe)
  - Event-driven → Calendar-driven (predict all 502 stocks daily)
  - GraphSAGE → HGT (4 edge types: correlation, sector, mentions, co-occurrence)
  - Horizon ablation: 1d/5d/10d/21d/42d/63d
  - SelectiveNet retained as core innovation
  - w=126, t=0.6 for dynamic correlation edges

→ plan: `2026-03-05-c` | analysis: N/A

## 2026-03-05-d: v3 Full Implementation — Notebook Written

- [x] Created `v3_ranking_pipeline.ipynb` (20 cells, ~2000 lines)
- [x] Cell structure:
  - Cells 0-3: Setup, parameters, data loading
  - Cells 4-7: N1 — Calendar-driven data pipeline (price features, news mapping, multi-horizon labels, time split)
  - Cells 8-9: N2 — Graph construction (4 edge types + Jaccard audit + HeteroData builder)
  - Cell 10: Evaluation utilities (IC, ICIR, Sharpe, portfolio backtest)
  - Cell 11: N3a — Non-GNN baselines (Ridge, XGBoost, LightGBM)
  - Cells 12-14: N3b-d — HGT model + GNN ablations (5 configs) + Go/Stop gate
  - Cell 15: N4 — Horizon ablation (6 horizons × HGT + LightGBM)
  - Cells 16-18: N5 — SelectiveNet (architecture, 2-stage training, analysis + visualization)
  - Cell 19: Observations markdown
- [x] All syntax validated (ast.parse passes)
- [x] **Run on Colab** (see 2026-03-05-e)

→ plan: `2026-03-05-d` | analysis: N/A

## 2026-03-05-e: v3 First Colab Run — Baselines + GNN Ablation Results

- [x] Full pipeline ran on NVIDIA RTX PRO 6000 Blackwell (102GB VRAM)
- [x] Data pipeline (N1-N2): all correct — 501 valid tickers, 58.5% news coverage, 6 horizons
- [x] N3a Baselines: B1-B4 (Ridge×2, XGBoost, LightGBM) — best baseline IC=0.00828 (LightGBM)
- [x] N3d GNN Ablation (5 configs):
  - A1 HGT corr-only: IC=0.01023
  - A2 HGT corr+sector: IC=0.01177, Sharpe=0.994
  - **A3 HGT all 4 edges: IC=0.00432, Sharpe=-0.314 (WORST GNN!)**
  - A4 SAGE corr+sector: IC=0.01571, Sharpe=1.038
  - **A5 GAT corr+sector: IC=0.02054, Sharpe=1.011 (BEST)**
- [x] Go/Stop Gate: **GO** (Sharpe 1.038 > 0.5 threshold)
- [x] N4 Horizon Ablation: 1d HGT IC=0.00343, Sharpe=3.073 — rest drowned in warnings
- [ ] N4/N5 results **NOT visible** due to massive sklearn warnings
- [x] Fixed sklearn feature name warnings: added `warnings.filterwarnings` in Cell 11
- [x] Identified N4 bug: uses HGT (all edges) but should use GAT (corr+sector) — the best model

**Key findings**:
- corr+sector >> all 4 edges (news/cooccur edges add noise)
- GAT > SAGE > HGT for same edge config
- Ranking approach WORKS — v2 binary AUC=0.50, v3 ranking IC>0.01 with Sharpe>1.0

→ progress: `2026-03-05-e` | plan: `2026-03-05-e` | analysis: `2026-03-05-e`

## 2026-03-06-a: N4/N5 Code Updated — HGT → GAT (corr+sector)

- [x] Analyzed why GAT > HGT: (1) parameter efficiency in weak-signal regime, (2) edge-type distinction not useful, (3) news dummy nodes add noise, (4) simpler attention more robust
- [x] Cell 12: Added `get_stock_embeddings()` to `RankingGNN` class
- [x] Cell 15 (N4): `RankingHGT` → `RankingGNN(conv_type='gat')`, `train_hgt` → `train_homogeneous_gnn`, edge_types=['corr','sector']
- [x] Cell 16 (N5a): `SelectiveRankingHGT` → `SelectiveRankingGAT` (homogeneous GAT backbone)
- [x] Cell 17 (N5b): All `build_hetero_data` → `_build_homo_graph` helper, model calls use `(x, edge_index)` not `(x_dict, edge_index_dict)`
- [x] Cell 18 (N5c): Same graph construction update
- [x] All 18 code cells pass ast.parse syntax validation
- [ ] **PENDING**: Upload to Google Drive and re-run on Colab

**Note**: News data still used as stock features (772 of 781 dims). Only news graph edges (mentions, cooccur) are dropped. News features ablation deferred.

→ progress: `2026-03-06-a` | plan: `2026-03-06-a` | analysis: `2026-03-05-e`

## 2026-03-06-b: v3 Colab Run 2 — N3-N5 Complete Results

- [x] Full pipeline ran on NVIDIA A100-SXM4-40GB (42.4 GB VRAM), updated code
- [x] N3 Ablation (Run 2): SAGE IC=0.01545 best (Run 1: GAT IC=0.02054). GAT IC=0.00640 (unstable!)
- [x] **Go/Stop Gate: GO** (Sharpe 1.266 > 0.5)
- [x] **N4 Horizon Ablation** — Complete results across 6 horizons:
  - GAT 21d: **IC=0.04420, ICIR=0.374, Sharpe=1.203** ← exceeds 0.03 threshold!
  - GAT 10d: IC=0.03854 ← also exceeds 0.03
  - GAT sweet spot: 5d-21d; fails at 1d, 42d, 63d
  - LGBM improves monotonically (63d: IC=0.05207)
  - GAT vs LGBM: cross pattern (GAT wins 5d-21d, LGBM wins 42d-63d)
- [x] **N5 SelectiveNet** — Complete results:
  - Full (100%): IC=0.05595, ICIR=0.463, Sharpe=1.328, Ann_LS_net=16.48%
  - **SelectiveNet FAILED**: all coverage 5%-50% have NEGATIVE IC (-0.015 to -0.024)
  - Threshold baseline works: @20% IC=0.03070
  - Selection head learned anti-selection (selects worst predictions)
- [x] Training stability analysis: GAT IC CV=105% across runs; SAGE CV=2%
- [x] Updated docs/research_log_2026-03-06.md with full analysis

→ plan: `2026-03-06-b` | analysis: `2026-03-06-b`

---

## Directory Structure

```
GNN-Testing/
├── progress.md          # 做了什么 (this file)
├── plan.md              # 接下来做什么
├── docs/analysis.md     # 分析发现记录
├── v3_ranking_pipeline.ipynb   # NEW: v3 full pipeline
├── phase_c_model_training.ipynb
├── phase_a_data_prep.ipynb
├── GNN测试1 colab.ipynb
├── data/{reference,pilot,fullscale,dynamic_graphs}/
├── scripts/
├── plots/
├── experiments/
└── docs/{REPORT.md, gnn-llm-prediction-plan.md, 代码讲解.md}
```

---

## 2026-04-07-a: Project Review + Comprehensive Plan

- [x] 通读全部文档: progress.md, plan.md, docs/analysis.md, 所有 docs/ 文件
- [x] 阅读并分析 `docs/DynHetGNN-SP_critique_and_plan.md` (批判文档)
- [x] 与H博士讨论确定方向: 做扎实 4-6 周, 整合 plan.md + critique
- [x] 制定 6 周详细计划:
  - Week 1: 稳定性验证 (5-seed GAT + LSTM baseline)
  - Week 2: Walk-forward CV + 完整 ablation
  - Week 3: SelectiveNet 改进 (三策略) + 交易成本 + 排列检验
  - Week 4: Qwen 3.6 结构化特征 (~$26) + 论文图表
  - Week 5: Qwen 整合 + 论文初稿
  - Week 6: 论文完善
- [x] 决策: SelectiveNet 先修后报; 投稿目标 ICAIF/FinNLP workshop; Qwen ~$26 可接受
- [x] 创建 `v3_stability_experiments.ipynb` (20 cells):
  - Cells 1-10: 从 v3_ranking_pipeline.ipynb 复制的基础设施 (数据加载、图构建、评估)
  - Cell 11: LightGBM multi-seed helper
  - Cell 12: Model definitions (RankingGNN + **新增 RankingLSTM**)
  - Cell 13: **增强版训练函数** — `train_gat_with_diagnostics()` (per-epoch val IC, seed参数, 返回完整 history) + `train_lstm_ranking()` (序列模型)
  - Cell 14: Exp 1.1 — Multi-seed GAT 21d (5 seeds: 42/123/456/789/1024)
  - Cell 15: Exp 1.2 — Multi-seed LightGBM 21d (对照组)
  - Cell 16: Exp 1.3 — 训练诊断可视化 (4-panel: val IC/val loss/train loss/test IC + ensemble analysis)
  - Cell 17: Exp 1.4 — LSTM baseline + MLP baseline (GAT with empty edges)
  - Cell 18: Summary — 全部结果汇总 + 决策建议
- [x] 所有 code cells 通过 ast.parse 语法验证

→ plan: `2026-04-07-a` | analysis: N/A

*Last updated: 2026-04-07*
