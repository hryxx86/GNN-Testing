# Plan — What To Do Next

> **接下来做什么。** 按时间记录计划。每个条目与 `progress.md` 和 `docs/analysis.md` 时间对齐。

---

## 2026-02-27-a: Phase C notebook created ✅

- [x] Create experiment pipeline notebook
- [x] Full-batch training on A100

→ progress: `2026-02-27-a` | analysis: N/A

## 2026-02-27-b: Run Phase C v1 experiments ✅

- [x] Run 6 experiments on Colab A100
- [x] Analyze results → all AUC ≈ 0.50

**Decision:** Diagnostics first, then model changes.

→ progress: `2026-02-27-b` | analysis: `2026-02-27-b`

## 2026-02-27-c: Diagnostics + docs restructure ✅

- [x] Write D.1 + D.2 diagnostic cells
- [x] Restructure documentation system
- [x] Run D.1 + D.2 on Colab → see `2026-03-03-a`
- [x] Record findings in analysis.md

→ progress: `2026-02-27-c` | analysis: `2026-02-27-c`

## 2026-03-03-a: D.1 + D.2 Results — FinBERT Signal Near Zero ✅

- [x] Analyze D.1/D.2 results
- [x] Update analysis.md with findings

**Verdict**: FinBERT alone = no signal. Go/Pivot/Stop criteria approaching STOP, but baseline matrix (XGBoost + momentum) not yet tested.

→ progress: `2026-03-03-a` | analysis: `2026-03-03-a`

## 2026-03-03-b: Literature Review — NLP+GNN Stock Prediction Papers ✅

- [x] Search and compare 6 papers (THGNN, DGRCL, DASF-Net, ChatGPT-GNN, etc.)
- [x] Identify why published papers show better results than our AUC~0.50
- [x] Update analysis.md with findings

**Key insight**: Our result is NOT a bug. It matches the only large-universe paper (DGRCL: 53% on 1K+ stocks). Papers claiming strong results use 12-30 cherry-picked stocks.

-> progress: `2026-03-03-b` | analysis: `2026-03-03-b`

---

# ═══════════════════════════════════════════════════════════════
# Roadmap v3: Ranking + Dynamic HGT + Selective Prediction
# ═══════════════════════════════════════════════════════════════
#
# 核心方向变更 (2026-03-05):
# v2 用 event-driven binary direction prediction → AUC ≈ 0.50 (EMH)
# v3 换赛道: calendar-driven ranking prediction + multi-horizon ablation
# 文献基础: MASTER (AAAI'24), FinMamba (2025), MDGNN (AAAI'24)
# ═══════════════════════════════════════════════════════════════

---

## 论文定位 & 叙事 (v3)

### 前沿论文定位

#### A. 金融 GNN SOTA (Ranking Target)

| 论文 | 会议 | 做了什么 | 我们的差异化 |
|------|------|---------|-------------|
| MASTER | AAAI'24 | Cross-stock Transformer, 5d ranking, IC=0.064 (CSI300) | 无显式图构建；无新闻；无selective prediction |
| FinMamba | arXiv'25 | Mamba+动态图, 1d ranking, Sharpe=2.06 (S&P500) | 无NLP/新闻；无异构边；无selective |
| MDGNN | AAAI'24 | 3类节点+多关系+日动态, IC=0.032 (CSI300) | 仅中国市场；无selective；无horizon ablation |
| THGNN | CIKM'22 | 日动态图+Transformer+HeteroGAT, IC=4.93% | 无NLP；无新闻节点；无selective |
| ChatGPT-GNN | KDD-W'23 | LLM推断图, 3-class, F1=0.41 | 仅DOW 30；F1低；无ranking |
| HGAIT | ESWA'25 | 正/负相关异构边+反向Transformer | 无NLP；无selective |

#### B. Selective Prediction SOTA

| 论文 | 会议 | 做了什么 | 与我们的关系 |
|------|------|---------|-------------|
| SelectiveNet | ICML'19 | 3-head: prediction + selection + auxiliary | 核心参考，从未应用于金融GNN |
| AUGRC | NeurIPS'24 | 修复AURC指标缺陷 | 评估指标 |
| Sim et al. | arXiv'23 | Chart image + confidence threshold 交易 | 唯一金融selective，非GNN |

#### C. 核心 Gap

**没有任何论文**同时做到：(1) 动态异构图 (2) 新闻+价格多模态 (3) ranking prediction (4) selective prediction (5) multi-horizon ablation (6) 大规模美国市场验证。这五项中每一项都有人做过，但组合是全新的。

**额外贡献：Horizon Ablation**。没有任何 GNN 论文系统比较过 1d/5d/10d/21d/42d/63d 预测 horizon。这是一个明显的文献空白。

### 论文叙事 (v3)

> "现有金融 GNN 面临三个未解决的挑战：(1) 静态图无法捕捉市场 regime 变化，(2) binary direction prediction 在高效市场上接近随机，(3) 所有股票所有时间都预测导致噪声主导。我们提出 **DynHetGNN-SP** — 动态异构图神经网络 + selective ranking prediction 框架。通过 HGT 架构融合四种动态边类型（价格相关性、行业、新闻提及、新闻共现），用 cross-sectional ranking 替代 binary classification，并引入 volatility-calibrated selection head 决定何时交易。在 S&P 500 (502 股) 上的系统性 horizon ablation (1d-63d) 揭示了 GNN 在不同时间尺度上的预测能力谱系。"

### 论文贡献 (v3)

1. **首个 GNN + Selective Prediction 组合**: SelectiveNet 从未被引入金融 GNN
2. **动态异构图 (4 边类型)**: correlation (月度动态) + sector (静态) + news→stock + news co-occurrence
3. **系统性 Horizon Ablation**: 1d/5d/10d/21d/42d/63d — 填补文献空白
4. **Calendar-driven + News-enhanced**: 每天预测所有股票，新闻作为 optional input feature
5. **大规模验证**: S&P 500, 502股 + economic significance (long-short portfolio, transaction costs)

### Prediction Target & Timing (v3)

```
可用特征 (截止时间):                      预测目标:
├─ 新闻 FinBERT embedding: T 时刻        Cross-sectional ranking score
├─ FinBERT sentiment: T 时刻              (Spearman corr with actual d-day excess return)
├─ Price features: T-1 close (严格!)
├─ Momentum/vol: T-1 close              Label: Z-score normalized d-day forward excess return
├─ Graph edges: 截止 T-1 的历史数据       d ∈ {1, 5, 10, 21, 42, 63} (horizon ablation)
│
│  Excess return = stock_return - equal_weight_market_return
│  Z-score: per-day cross-sectional normalization
```

---

## Phase N1: Calendar-Driven Data Pipeline ✅ CODE WRITTEN

**目标**: 将 event-driven 数据转为 calendar-driven (每天 × 每只股票 = 一个预测)

**Code**: `v3_ranking_pipeline.ipynb` Cells 4-7

### N1a. Stock-Day Matrix Construction

- [x] 构建 (trading_day, ticker) 矩阵: ~1250 天 × 502 股 ≈ 627K stock-days
- [x] 每个 stock-day 的特征:
  - Price features (9-dim): 5/10/21d momentum, volatility, return mean (T-1 close)
  - News features (771-dim): 当天有新闻 → mean-pooled FinBERT 768d + 3d sentiment; 无新闻 → zero vector
  - Has-news flag (1-dim): binary indicator
  - Total: 781-dim per stock-day
- [x] 处理无新闻的 stock-day: zero vector + has_news=0 (MSGCA 2024 做法)

### N1b. Multi-Horizon Labels

- [x] 对每个 stock-day 计算 6 种 forward return:
  - `ret_1d = (close[T+1] - close[T]) / close[T]`
  - `ret_5d = (close[T+5] - close[T]) / close[T]`
  - `ret_10d, ret_21d, ret_42d, ret_63d` 同理
- [x] Excess return: `excess_d = ret_d - market_ret_d` (equal-weight S&P 500)
- [x] Cross-sectional Z-score: per-day normalization `z_i = (excess_i - mean) / std`
- [x] **严格无前瞻**: close[T] 是 T 日收盘价，features 只用 T-1

### N1c. Time Split

- [x] Train: 2021-07 → 2023-12 (~630 trading days)
- [x] Val: 2024-01 → 2024-06 (~126 trading days)
- [x] Test: 2024-07 → 2025-12 (~378 trading days)
- [ ] Walk-forward validation as Phase N6 ablation

→ progress: `2026-03-05-d` | analysis: N/A

---

## Phase N2: Dynamic HGT Graph Construction ✅ CODE WRITTEN

**目标**: 构建动态异构图，月度更新 correlation 边

**Code**: `v3_ranking_pipeline.ipynb` Cells 8-9

### N2a. 4 Edge Types

| 边类型 | 构建方式 | 动态/静态 | 数据来源 |
|--------|---------|----------|---------|
| **stock↔stock (correlation)** | Pearson \|r\|>0.6, w=126d 滑动 | 月度动态 | Phase B 价格数据 ✅ |
| **stock↔stock (sector)** | 同 GICS sector 全连接 | 静态 | Wikipedia ✅ |
| **news→stock (mentions)** | 当天新闻提及该股票 | 按日变化 | EODHD events ✅ |
| **stock↔stock (co-occurrence)** | 同一篇新闻提及两只股票 | 按日变化 | EODHD events ✅ |

### N2b. Graph Snapshots

- [x] Correlation 边: 月度滚动 (w=126, t=0.6), ~54 snapshots
- [x] Sector 边: 静态, 11 sectors, ~25K edges
- [x] News 边: 每个 trading day 一个 news→stock edge set
- [x] Co-occurrence 边: 每个 trading day, 从 events 计算同篇文章的 ticker pairs

### N2c. HGT Model Architecture

- [x] PyG HGTConv (2 layers, 4 heads)
- [x] Node types: stock (781-dim), news (771-dim)
- [x] Edge types: 4 种 (correlation, sector, news_mentions, co_occurrence)
- [x] Output: per-stock ranking score (regression head, 1-dim)
- [x] Loss: MSE on Z-score normalized forward returns
- [x] Residual connection + LayerNorm per HGT layer

### N2d. Jaccard Audit (Edge Dynamics)

- [x] 计算相邻月份 correlation 边集的 Jaccard similarity
- [x] 报告 mean/std/min Jaccard + min日期

→ progress: `2026-03-05-d` | analysis: N/A

---

## Phase N3: Baseline Matrix (Ranking Evaluation) ✅ CODE WRITTEN

**目标**: 建立 baseline 梯队，验证各组件的增量贡献

**Code**: `v3_ranking_pipeline.ipynb` Cells 10-14

### N3a. Non-GNN Baselines

| Baseline | 特征 | 用途 |
|----------|------|------|
| LR (price only) | 9-dim momentum/vol | 纯价格线性 baseline |
| LR (price + news) | 9-dim + 771-dim FinBERT | 新闻增量 |
| XGBoost (all) | 781-dim | 非线性上限 (不含图) |
| LightGBM (all) | 781-dim | Qlib benchmark 标准 |

### N3b. GNN Ablations

| Experiment | Model | 边类型 | Horizon |
|-----------|-------|--------|---------|
| GNN-A1 | HGT | correlation only | 5d (MASTER default) |
| GNN-A2 | HGT | correlation + sector | 5d |
| GNN-A3 | HGT | all 4 edge types | 5d |
| GNN-A4 | GraphSAGE | all 4 edge types | 5d |
| GNN-A5 | GAT | all 4 edge types | 5d |

### N3c. Evaluation Metrics (Ranking)

所有模型报告:
- **IC** (Spearman correlation, daily cross-sectional)
- **ICIR** (IC / std(IC), 稳定性)
- **Rank IC** (rank-based variant)
- Long-only top-30 portfolio: annualized return, Sharpe ratio
- Long-short top/bottom-30: annualized return, Sharpe, max drawdown
- **扣除 15bps round-trip transaction cost**

### N3d. Go/Stop Gate

| 判定 | 条件 | 行动 |
|------|------|------|
| **Go** | 任一模型 IC > 0.03 (daily) 或 Long-short Sharpe > 0.5 | 进入 N4+N5 |
| **Stop** | 所有模型 IC ≈ 0 | 写 negative result 论文 (ranking 也无法预测) |

→ progress: `2026-03-05-d` | analysis: `2026-03-05-e`

**2026-03-05-e Colab Results**: Go/Stop → **GO** (Sharpe 1.038 > 0.5). Best model: A5 GAT (corr+sector) IC=0.02054. Worst: A3 HGT (all 4 edges) IC=0.00432. News/cooccur edges add noise.

---

## Phase N4: Horizon Ablation ✅ COMPLETE ← 论文贡献点

**目标**: 系统比较 6 种预测 horizon，填补文献空白

**Code**: `v3_ranking_pipeline.ipynb` Cell 15

### N4 Results (Colab Run 2, 2026-03-06)

| Horizon | GAT IC | GAT ICIR | GAT Sharpe | LGBM IC | LGBM Sharpe |
|---------|--------|----------|------------|---------|-------------|
| 1d | -0.00104 | -0.013 | 2.468 | 0.00368 | 2.918 |
| 5d | 0.02334 | 0.227 | 1.568 | 0.00828 | 0.773 |
| 10d | **0.03854** | 0.320 | 1.196 | 0.01349 | 0.644 |
| **21d** | **0.04420** | **0.374** | **1.203** | 0.01513 | 0.468 |
| 42d | -0.00912 | -0.144 | 0.071 | 0.03679 | 0.668 |
| 63d | -0.00838 | -0.118 | 0.487 | 0.05207 | 1.256 |

**Key findings**:
- [x] IC vs horizon: inverted-U, **peak at 21d** (IC=0.04420)
- [x] ICIR vs horizon: same pattern, peak 21d (ICIR=0.374)
- [x] GAT > LGBM at 5d-21d; LGBM > GAT at 42d-63d (cross pattern)
- [ ] News contribution analysis (有新闻 vs 无新闻) — not yet done

→ progress: `2026-03-06-b` | analysis: `2026-03-06-b`

---

## Phase N5: Selective Prediction ✅ COMPLETE — SelectiveNet FAILED

**Code**: `v3_ranking_pipeline.ipynb` Cells 16-18

### N5 Results (Colab Run 2, 21d horizon)

| Method | IC | ICIR | Sharpe | Ann_LS_net |
|--------|-----|------|--------|------------|
| Full (100%) | **0.05595** | **0.463** | **1.328** | **16.48%** |
| Threshold @20% | 0.03070 | 0.324 | 0.724 | 5.85% |
| SelectiveNet @20% | **-0.02414** | -0.256 | -0.536 | -9.00% |

**Status**:
- [x] Threshold baseline: works (IC improves at lower coverage)
- [x] SelectiveNet: **FAILED** — negative IC at all coverage levels
- [x] Jaccard overlap: SelectiveNet selects different (worse) stocks than threshold
- [x] Coverage converged to 31% (target 20%) — lambda insufficient
- [ ] SelectiveNet improvement or replacement — needs discussion

→ progress: `2026-03-06-b` | analysis: `2026-03-06-b`

---

## Phase N6: Paper Polish ← PENDING (after Colab run)

### N6a. Walk-Forward Validation

- [ ] Fold 1: Train 2021→2023, Val early-2024, Test late-2024
- [ ] Fold 2: Train 2021→2024, Val early-2025, Test late-2025
- [ ] Permutation test: 打乱股票代码 / 打乱标签

### N6b. Figures

- [ ] Fig 1: Dynamic graph evolution (edge count + regime annotations) — Phase B+ 已完成
- [ ] Fig 2: Horizon Ablation — IC/Sharpe vs prediction horizon (核心图)
- [ ] Fig 3: Coverage-IC tradeoff curve (Selective prediction 核心图)
- [ ] Fig 4: HGT attention weights — 哪种边类型最重要
- [ ] Fig 5: Per-month coverage + VIX overlay (SelectiveNet regime-awareness)
- [ ] Fig 6: Long-short cumulative return curve (扣费后)

### N6c. Tables

- [ ] Tab 1: Full baseline matrix (all models × all horizons × IC/ICIR/Sharpe)
- [ ] Tab 2: GNN ablation (edge type ablation)
- [ ] Tab 3: Selective prediction comparison (threshold vs SelectiveNet @ multiple coverages)
- [ ] Tab 4: Transaction cost analysis (break-even cost)

→ progress: TBD | analysis: TBD

---

## 最低可发表标准 (v3) — UPDATED 2026-03-06

| 指标 | 目标 | 实际值 | 状态 |
|------|------|--------|------|
| 任一 horizon IC > 0.03 | 超越随机排名 | **GAT 21d IC=0.04420** | ✅ |
| GNN IC > LightGBM IC | 图有增量 | **21d: 0.044 vs 0.015 (2.9×)** | ✅ |
| Selective IC@20% > Full IC | Selective 有增量 | Thr @20%: 0.031 < Full 0.056 | ❌ |
| Long-short Sharpe > 0.5 (扣费后) | 经济显著 | GAT 21d **Ann_LS_net=15.11%** | ✅ |
| Horizon ablation 有明确模式 | 文献贡献 | **倒U型, peak 21d** | ✅ |

**4/5 达标。** SelectiveNet 贡献点失败，需要替代方案。

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-27 | FinBERT (not Fin-E5/voyage) | Clean ablation; upgrade later |
| 2026-02-27 | Full-batch on A100 | No torch-sparse; 80GB sufficient |
| 2026-02-27 | Diagnostics before changes | AUC ≈ 0.50 → find where signal weak first |
| 2026-03-03 | Literature confirms AUC~0.50 expected | Large-universe + FinBERT + S&P500 = near-random for binary direction |
| 2026-03-03 | LLM 选 GPT-4o-mini | 论文可引用性最强 + structured JSON 最可靠 |
| 2026-03-03 | Equal-weight market return (非 SPY) | SPY 不在 prices 文件中 |
| 2026-03-03 | Sentiment 聚合用 mean (非 max) | 简单高效 |
| 2026-03-04 | Phase 2 LLM STOP confirmed | GPT-4o-mini delta=+0.0009, skip full-scale |
| 2026-03-05 | **换赛道: binary direction → ranking** | 所有 SOTA (MASTER/FinMamba/MDGNN) 用 ranking，binary AUC=0.50 是 EMH 预期结果 |
| 2026-03-05 | **Calendar-driven 替代 event-driven** | 主流做法；无新闻用 zero vector；避免 event-level 噪声 |
| 2026-03-05 | **HGT 替代 GraphSAGE** | 异构图 attention 可区分不同边类型权重；可解释性强 |
| 2026-03-05 | **4 种边类型 (不加外部数据)** | correlation(动态)+sector(静态)+news_mentions+co_occurrence; Multi-GCGRU 发现 co-occurrence > 持股关系 |
| 2026-03-05 | **动态图参数: w=126, t=0.6** | Phase B 分析: density 6%, std=0.064 (稳定), 125 components (适中) |
| 2026-03-05 | **Horizon ablation: 1d/5d/10d/21d/42d/63d** | 无论文做过系统比较；覆盖 short-term reversal 到 classic momentum 全谱系 |
| 2026-03-05 | **不加 30d/60d**: 21d 和 42d 已覆盖 | 21d≈1月, 42d≈2月, 与 30d/60d 高度重叠；用交易日而非日历日更精确 |
| 2026-03-05 | **DASF-Net "3-day optimal" 不成立** | 指 input aggregation window 非 prediction horizon; 仅 12 股; 无 horizon ablation |
| 2026-03-05 | **Selective prediction 保留为核心创新** | GNN + SelectiveNet 组合 = 文献空白; 高风险高回报 |
| 2026-03-05 | **Supply chain/持仓边留作 future work** | 需 FactSet($$)/SEC 13F(工程量大); 先验证框架再加数据 |
| 2026-03-06 | **News/cooccur edges 有害，仅保留 corr+sector** | A3(all 4 edges) IC=0.00432 最差; A5(corr+sector) IC=0.02054 最好; news edges 加噪声 |
| 2026-03-06 | **GAT 替代 HGT 作为最佳架构** | GAT IC=0.02054 > SAGE 0.01571 > HGT 0.01177 (同edge config); 简单attention更robust |
| 2026-03-06 | **N4 必须用 GAT(corr+sector) 重跑** | 原代码用 HGT(all edges)=最差config; 已确认需修改 |
| 2026-03-06 | **N4/N5 代码已改为 GAT(corr+sector)** | Cell 12: RankingGNN+get_stock_embeddings; Cell 15: GAT+corr+sector; Cell 16: SelectiveRankingGAT; Cell 17-18: 同构图 |
| 2026-03-06 | **GAT 优于 HGT 的原因** | (1)参数效率:弱信号下少参数泛化更好 (2)edge type区分无用:corr+sector本质都是"相关" (3)news dummy节点加噪声 (4)GAT attention更稳定 |
| 2026-03-06 | **GAT 21d IC=0.04420 超过0.03门槛** | 倒U型模式: 10d-21d是GAT sweet spot; 1d和42d-63d信号消失 |
| 2026-03-06 | **SelectiveNet 失败: 选择头反向选择** | 所有coverage 5%-50%负IC; 2-stage训练可能是原因; 需要讨论替代方案 |
| 2026-03-06 | **训练稳定性是关键问题** | GAT N3 IC跨run CV=105% (0.006-0.021); SAGE最稳定CV=2%; Walk-forward CV必须做 |
| 2026-03-06 | **Full SelectiveRankingGAT (100%) IC=0.05595** | 3-head auxiliary loss提供正则化; 比单纯RankingGNN的0.04420更好 |

## Current Status (2026-03-06)

**Phase**: N3-N5 全部完成
**关键结果**:
- GAT 21d IC=0.04420 (> 0.03 ✅), ICIR=0.374, Ann_LS_net=15.11%
- Horizon inverted-U pattern: GAT peaks at 10d-21d
- SelectiveNet FAILED (negative IC at all coverages)
- Full SelectiveRankingGAT IC=0.05595 (best overall)
- 训练稳定性有问题: GAT IC CV=105%

**4/5 最低发表标准达标。**

**Next (优先级排序)**:
1. **Walk-forward CV** — 解决训练稳定性 (最重要!)
2. **多次重复实验** — GAT 21d 跑 5 次取均值±std
3. **SelectiveNet 讨论** — 改进 or 报告 negative finding or 改用 threshold
4. **论文图表** — Horizon ablation plot 已有; selective analysis plot 已有
5. **交易成本敏感性** — tc = 5/10/15/20/30 bps

**Blockers**: ~~需与H博士讨论 SelectiveNet 策略~~ ✅ 已讨论 (2026-04-07)

---

## 2026-04-07-a: 6 周完整计划确定 ← CURRENT

综合 plan.md 路线 + critique 文档，制定了 6 周扎实计划。详见 `.claude/plans/mellow-puzzling-pie.md`。

### Week 1 (当前): 稳定性验证
- [ ] Task 1.1: Multi-seed GAT 21d (5 seeds) → `v3_stability_experiments.ipynb`
- [ ] Task 1.2: Multi-seed LightGBM 21d (对照)
- [ ] Task 1.3: 训练诊断 (per-epoch val IC)
- [ ] Task 1.4: LSTM Baseline

### Week 2: Walk-Forward CV + Ablation
- [ ] Task 2.1: Walk-forward CV (3 folds)
- [ ] Task 2.2: 补齐 ablation (pure-price, GNN-no-news, MLP)
- [ ] Task 2.3: News 贡献分析

### Week 3: SelectiveNet + 经济显著性
- [ ] Task 3.1: SelectiveNet 三策略 (E2E → Aux confidence → 波动率校准)
- [ ] Task 3.2: 交易成本敏感性 (0-30 bps)
- [ ] Task 3.3: Permutation test (100 次)

### Week 4: Qwen 特征 + 图表
- [ ] Task 4.1: Qwen 3.6 结构化特征 (~10万条, ~$26)
- [ ] Task 4.2: Consistency test (500条 × 5次)
- [ ] Task 4.3: 论文图表初版

### Week 5: 整合 + 论文初稿
- [ ] Task 5.1: Qwen 特征整合管道
- [ ] Task 5.2: LLM-based 图构建 (可选)
- [ ] Task 5.3: 论文初稿

### Week 6: 论文完善
- [ ] Task 6.1-6.3: 表格 + 图表 + 定稿

### 关键决策 (2026-04-07)

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-07 | 做扎实 4-6 周, 不急于收尾 | 论文竞争力 > 速度 |
| 2026-04-07 | SelectiveNet: 先修后报 (三策略) | 成功=贡献点; 失败=negative finding, 两种都有价值 |
| 2026-04-07 | 投稿目标: ICAIF 2026 / FinNLP Workshop | 金融领域 workshop, 更注重 insight 而非 novelty |
| 2026-04-07 | Qwen 3.6 结构化特征纳入 (~$26) | 成本可接受; ablation 需要 LLM vs FinBERT 对比 |
| 2026-04-07 | Notebook 拆分为 4 个 | 单一 notebook 太大, 不利于维护和实验隔离 |
| 2026-04-07 | 论文定位: "when does graph help" 系统研究 | 降低 overclaim 风险, 更诚实; critique 建议 |

→ progress: `2026-04-07-a` | analysis: N/A

*Last updated: 2026-04-07*
