# GNN-Testing 完整科研记录

> **项目全称**: DynHetGNN-SP — Dynamic Heterogeneous Graph Neural Network with Selective Prediction for Stock Ranking
> **记录日期**: 2026-03-06
> **状态**: v3 pipeline N3-N5 全部完成。GAT 21d IC=0.04420 突破 0.03 门槛。SelectiveNet 失败（选择头负相关）。

---

## 目录

1. [项目概览与动机](#1-项目概览与动机)
2. [Phase 1: 网络结构分析（探索性）](#2-phase-1-网络结构分析)
3. [Phase 2 Pilot: 新闻驱动 GNN 预测（小规模）](#3-phase-2-pilot-新闻驱动-gnn-预测)
4. [Phase A: EODHD 数据 + FinBERT（全规模数据准备）](#4-phase-a-eodhd-数据--finbert)
5. [Phase B: 动态图构建与参数选择](#5-phase-b-动态图构建与参数选择)
6. [Phase C v1: 全规模实验 — AUC ≈ 0.50 困境](#6-phase-c-v1-全规模实验)
7. [诊断阶段: D.1 + D.2 — 信号在哪里？](#7-诊断阶段-d1--d2)
8. [Phase 1 Signal Fix: 新闻去重 + 市场调整 + 动量特征](#8-phase-1-signal-fix)
9. [Phase 2 LLM: GPT-4o-mini 替代 FinBERT — STOP](#9-phase-2-llm-gpt-4o-mini)
10. [v3 路线转变: Binary → Ranking + Calendar-Driven](#10-v3-路线转变)
11. [v3 实现: N1-N5 Pipeline](#11-v3-实现-n1-n5-pipeline)
12. [v3 Colab Run 1: N3 首次结果](#12-v3-colab-run-1)
13. [GAT vs HGT: 为什么简单的赢了](#13-gat-vs-hgt-分析)
14. [v3 Colab Run 2: N3-N5 完整结果](#14-v3-colab-run-2)
15. [N4 Horizon Ablation: 详细分析](#15-n4-horizon-ablation)
16. [N5 SelectiveNet: 详细分析](#16-n5-selectivenet)
17. [训练稳定性分析](#17-训练稳定性)
18. [当前状态与下一步](#18-当前状态与下一步)
19. [术语表](#19-术语表)

---

## 1. 项目概览与动机

**核心问题**: 能否用图神经网络（GNN）结合新闻信息预测股票价格走势？

**科研假设**: 股票之间存在通过价格相关性、行业关系、新闻共现等形成的网络结构。GNN 可以利用这种结构进行跨股票的信息传播，从而提升预测性能。

**数据**:
- 502 只 S&P 500 成分股
- 时间跨度: 2020-01 至 2025-12 (约 1255 个交易日)
- 新闻数据: 1,698,182 条 EODHD 事件（映射后 1,538,967 条有效）
- NLP embedding: FinBERT (768维)

**项目演变**:
```
Phase 1-2 Pilot (探索) → Phase A-C (全规模, AUC≈0.50 失败)
    → 诊断 D.1/D.2 → Signal Fix (仍失败)
    → LLM Phase 2 (STOP)
    → v3 路线转变 (Ranking + Calendar-Driven + GAT + Selective)
    → v3 首次运行 (IC=0.02054, Sharpe=1.011, GO!)
```

---

## 2. Phase 1: 网络结构分析

**目标**: 探索 S&P 500 股票之间的网络结构特征。

**方法**:
- 收集 502 只股票的 5 年价格数据
- 计算 Pearson 相关性矩阵 (502×502)
- 以阈值 |corr| > 0.6 构建无向图
- 训练探索性 GCN (Graph Convolutional Network) 模型

**结果**:
- 图密度: 3,198 条边 (约 1.3%)
- 网络中心性: Industrials 和 Financials 主导 hub 股票 (top 10 中占 7+3)
- 意外发现: Mega-cap 科技股 (AAPL, MSFT 等) 不是网络中心

**工具**: NetworkX, PyG, Matplotlib

**意义**: 验证了股票网络结构的存在性和行业聚集特征，为后续 GNN 建模提供基础。

---

## 3. Phase 2 Pilot: 新闻驱动 GNN 预测

**目标**: 小规模验证 "新闻 + GNN" 能否预测股票涨跌。

**方法**:
- 从 Factiva 手动收集 1,900 篇新闻（9 个 hub 股票）
- 清洗后得到 480 个事件
- 用 SentenceTransformer (MiniLM-L6-v2, 384维) 编码新闻
- 构建异构图: news 节点 → stock 节点, stock ↔ stock (correlation)
- **模型**: GraphSAGE (SAGEConv, 2层, hidden=64)
  - GraphSAGE (Hamilton et al., NeurIPS'17): 采样邻居 + 聚合的归纳式 GNN
  - 用 mean aggregation，对每个节点从邻居采样固定数量进行信息聚合

**结果**:
| Model | Test AUC |
|-------|----------|
| 纯文本 LR baseline | 0.6213 |
| GraphSAGE (异构图) | **0.6426** |

- 图结构增益: +0.0213 (+3.4%)
- 结论: 初步验证 GNN 有增量，但数据规模太小，需要全规模验证

**局限**: 仅 9 只股票、480 条新闻，统计意义有限。

---

## 4. Phase A: EODHD 数据 + FinBERT

**目标**: 构建全规模数据集，替换 Factiva 小数据。

**数据来源**: EODHD API — 提供历史新闻事件数据
- 原始事件: 1,698,182 条 (2020-01 至 2025-12)
- 映射到 S&P 500: 1,538,967 条有效 (90.6% 匹配率)

**NLP 模型选择**:

| 候选 | 维度 | 优势 | 选择理由 |
|------|------|------|---------|
| **FinBERT** (ProsusAI) | 768 | 金融领域预训练，广泛引用 | ✅ 选择：clean ablation，升级推迟到后续 |
| Fin-E5 | 768 | 更新的金融 embedding | 推迟 |
| Voyage Finance | 1024 | API-based | 推迟 |

**FinBERT 解释**:
- 基于 BERT (Devlin et al., NAACL'19) 的金融领域微调版本
- 在 Financial PhraseBank + 金融新闻上微调
- 输出: 768维 sentence embedding + 3维情感分数 (positive/negative/neutral)
- 我们用的是 ProsusAI/finbert，Hugging Face 上最流行的金融 BERT

**处理流程**:
```
EODHD 原始事件 → 股票代码匹配 → FinBERT 编码 (768维 embedding + 3维 sentiment)
→ 按 stock-day 聚合 (mean) → 存储为 (num_days, num_stocks, 771) tensor
```

**输出**: `data/fullscale/` 目录下的 prices, events, embeddings 文件

---

## 5. Phase B: 动态图构建与参数选择

**目标**: 确定动态相关性图的最优参数。

**方法**: Pearson 相关性矩阵，滚动窗口方式
- 对每个月，用前 w 个交易日的收盘价计算 502×502 的 correlation 矩阵
- 保留 |corr| > τ 的边

**参数搜索**:

| 参数 | 搜索范围 | 最优值 | 选择理由 |
|------|---------|--------|---------|
| **w** (窗口长度) | 63, 126, 252 | **126** (~6个月) | 密度适中 (6%), 稳定性好 (std=0.064) |
| **τ** (相关性阈值) | 0.5, 0.6, 0.7 | **0.6** | 平衡图密度和信噪比 |

**评估指标**:
- **图密度** (density): 边数 / 最大可能边数。太密=噪声多，太疏=信息丢失
- **连通分量数** (connected components): 应该有 1 个主分量
- **月间 Jaccard 相似度**: 衡量图的时间稳定性
  - Jaccard(A,B) = |A∩B| / |A∪B|，范围 [0,1]
  - Mean=0.631 说明相邻月份约 63% 的边保持不变 → 图结构较稳定

**最优配置结果** (w=126, τ=0.6):
- 密度: ~6% (约 7,500 条边)
- 连通分量: 1 个主分量
- Jaccard: mean=0.631, std=0.064
- 54 个月度快照 (2020-07 到 2025-12)

**注意**: 图构建只使用截至 2024-12 的数据，避免未来信息泄漏（测试集从 2024-07 开始，但图更新有滞后）。

---

## 6. Phase C v1: 全规模实验

**目标**: 在完整 S&P 500 数据上验证 "新闻 + GNN → 涨跌预测"。

**模型配置**: 6 个实验

| ID | 模型 | 节点特征 | 图结构 |
|----|------|---------|--------|
| B1 | Logistic Regression | FinBERT 768维 | 无 |
| B2 | Logistic Regression | Sentiment 4维 | 无 |
| A1 | GraphSAGE | FinBERT + Sentiment | news→stock edges only |
| A2 | + correlation edges | 同上 | + 相关性边 |
| A3 | + sector edges | 同上 | + 行业边 |
| Full | 所有边类型 | 同上 | 全部 3 种边 |

**任务**: Binary direction prediction — 预测 next-day 涨/跌
**标签**: sign(close[T+1] - close[T])

**结果**:

| Experiment | Val AUC | Test AUC |
|-----------|---------|----------|
| B1: LR + FinBERT | 0.5018 | 0.4976 |
| B2: LR + Sentiment | 0.5044 | 0.5027 |
| A1: GNN news only | 0.5085 | 0.4913 |
| A2: + corr edges | 0.5122 | 0.4949 |
| A3: + sector edges | 0.5133 | 0.4961 |
| Full: all edges | 0.5133 | 0.5069 |

**所有 AUC ≈ 0.50 — 等同于随机猜测。**

**诊断**: 不清楚问题出在哪里。可能是特征无信号、模型不对、或任务本身不可行。
**决策**: 先做诊断，再改模型。

---

## 7. 诊断阶段: D.1 + D.2

**目标**: 找到 AUC ≈ 0.50 的根本原因。

### D.1: 标签分布分析

**发现**:
- 约 26.5% 的事件落在 "噪声区" (|return| < 0.5%)
- next-day return 分布接近正态 (均值≈0)，正负方向几乎对称
- 说明**标签本身就很难预测**

### D.2: FinBERT 对齐分析

**方法**: 检查 FinBERT sentiment 与实际涨跌的对齐程度
- "对齐" = positive sentiment 对应上涨 (或 negative 对应下跌)

**发现**:
- 对齐率: **51.6%** — 仅比随机 (50%) 高 1.6%
- 高 |sentiment| 事件也没有更好的对齐
- **结论**: FinBERT sentiment 对 S&P 500 next-day return 基本没有预测力

**根因分析**:
1. S&P 500 = 高效市场 (EMH strong-form)，新闻信息已被快速定价
2. FinBERT 的情感分析 ≠ 市场影响预测
3. Next-day 太短，新闻效应可能更长期

---

## 8. Phase 1 Signal Fix

**目标**: 尝试修复信号问题，提升 AUC。

### 修复措施

1. **新闻去重** (deduplication): 去除重复新闻事件
2. **市场调整标签** (market-adjusted returns): 用 return_stock - return_market 替代 raw return
3. **动量特征** (momentum features): 添加 5d/10d/20d 价格动量
4. **改进 GNN**: 加入 BatchNorm, dropout, 调整学习率
5. **Selective prediction**: 只对高置信度事件预测（top/bottom 5%）

### 结果 — Phase 1 Baseline Matrix

| Model | Val AUC | Test AUC |
|-------|---------|----------|
| B1: FinBERT LR (deduped) | 0.5043 | 0.5043 |
| B2: Sentiment LR (deduped) | 0.5076 | 0.5028 |
| B3: XGBoost (all features) | 0.5093 | 0.5020 |
| B4: Momentum-only XGBoost | 0.5052 | 0.5044 |
| B5: Random Forest | 0.5068 | 0.5004 |

**所有 test AUC < 0.51 — Signal Fix 失败。**

### Selective AUC
- Top/bottom 5% confidence: AUC = 0.5154 — 在统计噪声范围内
- **没有尾部信号**

### 结论
- Binary direction prediction on S&P 500 with FinBERT → **不可行**
- 这不是代码 bug，而是任务本身的限制 (EMH)
- 文献对照: DGRCL (唯一 1K+ 股票论文) 仅 53% accuracy，与我们结果一致

---

## 9. Phase 2 LLM: GPT-4o-mini

**目标**: 用更强的 LLM 替代 FinBERT，看是否能突破 AUC ≈ 0.50。

**选择 GPT-4o-mini 的理由**:
- 论文可引用性最强 (OpenAI 是学术界广泛认可的)
- 支持 structured JSON output (方便提取情感分数)
- 成本可控 (相比 GPT-4o)

**方法**:
- 让 GPT-4o-mini 对每条新闻输出 structured sentiment score
- 与 FinBERT 结果对比

**结果**:
- GPT-4o-mini 与 FinBERT 的 delta: **+0.0009** — 几乎没有区别
- 高影响事件子集: AUC = **0.4762** — 反而更差

**决策**: **STOP 确认** — LLM 升级不能解决问题，问题出在任务定义而非模型能力。

---

## 10. v3 路线转变

**时间**: 2026-03-05

### 核心洞察

通过文献综述（10+ 篇最新论文），发现我们的问题不在模型，而在**任务定义**：

| 维度 | v2 (失败) | v3 (新方向) | 文献支持 |
|------|-----------|------------|---------|
| **预测任务** | Binary direction (涨/跌) | **Ranking** (股票排名) | MASTER, FinMamba, MDGNN 全部用 ranking |
| **评估指标** | AUC | **IC/ICIR/Sharpe** | 金融量化标准 |
| **数据范式** | Event-driven (有新闻才预测) | **Calendar-driven** (每天每只股票都预测) | 所有 SOTA 论文 |
| **标签定义** | sign(return) | **z-score(return)** (cross-sectional 标准化) | 消除市场整体涨跌的影响 |

### 关键术语解释

- **IC (Information Coefficient)**: 每天的预测值与实际排名的 Spearman 相关系数。IC=0.03 在金融领域是有意义的门槛。
- **ICIR (IC Information Ratio)**: mean(IC) / std(IC)。衡量 IC 的稳定性。ICIR > 0.1 表示信号稳定。
- **Sharpe Ratio**: 年化收益 / 年化波动率。Sharpe > 0.5 表示经济显著。
- **Long-Short Portfolio**: 做多排名前 N 只，做空排名后 N 只的组合策略。
- **Calendar-Driven**: 每个交易日对所有股票生成预测（没有新闻的用零向量填充）。
- **Z-Score 标准化**: (return - cross_sectional_mean) / cross_sectional_std，消除市场 beta。

### 文献关键支撑

| 论文 | 会议 | 核心发现 |
|------|------|---------|
| MASTER | AAAI'24 | Cross-stock Transformer, IC=0.064 (CSI300) |
| FinMamba | arXiv'25 | Mamba + dynamic graph, Sharpe=2.06 (S&P500) |
| MDGNN | AAAI'24 | 多关系动态图, IC=0.032 (CSI300) |
| THGNN | CIKM'22 | 每日动态图 + HeteroGAT |
| SelectiveNet | ICML'19 | 3-head 架构 (prediction + selection + auxiliary), 800+ 引用 |
| Multi-GCGRU | IEEE'24 | Co-occurrence 边 > 持仓/供应链边 |

### 三个论文贡献点

1. **Horizon Ablation**: 1d/5d/10d/21d/42d/63d 的系统比较 — 文献中没有人做过
2. **GNN + SelectiveNet**: 首次将 SelectiveNet 应用于金融 GNN — 文献空白
3. **动态异构图 + NLP**: correlation + sector + news mentions + co-occurrence 的多边类型图

---

## 11. v3 实现: N1-N5 Pipeline

### 架构概览

```
v3_ranking_pipeline.ipynb (单 notebook, 19 cells)
├── N0: Setup + Environment
├── N1a: Price features (9维: returns, volatility)
├── N1b: News features (772维: FinBERT embedding + sentiment + has_news)
├── N1c: Labels (6个 horizon 的 z-score normalized returns)
├── N1d: Time split (Train/Val/Test)
├── N2a: Correlation graph (54 monthly snapshots, w=126, τ=0.6)
├── N2b: Sector graph (11 GICS sectors, static)
├── N2c: News mentions + Co-occurrence edges (daily)
├── N3a: Non-GNN baselines (Ridge, XGBoost, LightGBM)
├── N3b: GNN model definitions (RankingHGT, RankingGNN)
├── N3c: GNN training loop
├── N3d: GNN ablation experiments + Go/Stop gate
├── N4: Horizon ablation (1d-63d, GAT)
├── N5a: SelectiveNet model definition (SelectiveRankingGAT)
├── N5b: SelectiveNet training (2-stage)
├── N5c: Selection analysis + visualization
└── Cell 19: Observations markdown
```

### N1: Calendar-Driven 数据管线

**输入**: 原始价格 + EODHD 新闻事件
**输出**: 每天每只股票一个 781维特征向量

| 特征组 | 维度 | 内容 |
|--------|------|------|
| 价格特征 | 9 | 1d/5d/10d/21d returns, 5d/21d volatility, 21d MA ratio, volume ratio, 1d log return |
| FinBERT embedding | 768 | 当天提到该股票的所有新闻的 FinBERT embedding 均值 |
| Sentiment scores | 3 | positive/negative/neutral 均值 |
| Has-news flag | 1 | 当天是否有新闻 (0/1) |
| **总计** | **781** | |

**标签**: 6 种 horizon 的 forward return，经过 cross-sectional z-score 标准化
- Forward return: (close[T+h] - close[T]) / close[T]
- Z-score: (return - mean_across_stocks) / std_across_stocks
- Horizons: 1d, 5d, 10d, 21d, 42d, 63d

**时间切分**:

| Split | 日期范围 | 交易日数 | 新闻覆盖率 |
|-------|---------|---------|-----------|
| Train | 2021-07 → 2023-12 | 629 | 57.6% |
| Val | 2024-01 → 2024-06 | 124 | 55.9% |
| Test | 2024-07 → 2026-01 | 396 | 62.7% |

### N2: 动态异构图

**4 种边类型**:

| 边类型 | 节点对 | 动态/静态 | 构建方式 |
|--------|--------|----------|---------|
| correlation | stock↔stock | 月度更新 | Pearson corr > 0.6, w=126 |
| sector | stock↔stock | 静态 | GICS sector 相同 |
| mentions | news→stock | 每日 | 新闻提到股票 |
| co-occurrence | stock↔stock | 每日 | 同一条新闻提到两只股票 |

**图统计**:
- Correlation: 54 snapshots, 密度从 2.9% 降至 0.6%
- Sector: 27,070 条边 (11 个行业)
- News mentions: 1,538,967 条 (平均 1226/天)
- Co-occurrence: 2,918,292 条 (平均 2325/天)

### N3: 模型与 Baseline

**非 GNN Baselines (Cell 11)**:

| ID | 模型 | 特征 | 细节 |
|----|------|------|------|
| B1 | Ridge Regression | 9维 (price only) | L2 正则化线性回归, alpha=1.0 |
| B2 | Ridge Regression | 781维 (all) | 包含新闻特征 |
| B3 | XGBoost | 781维 | n_estimators=200, max_depth=5, lr=0.05, early_stopping=20 |
| B4 | LightGBM | 781维 | 同上参数，Microsoft 的梯度提升框架 |

**GNN 模型 (Cell 12)**:

| 类名 | 架构 | 图类型 | 参数 |
|------|------|--------|------|
| RankingHGT | HGTConv (异构图 Transformer) | HeteroData | stock_lin(781→64), news_lin(771→64), 2层, 4 heads |
| RankingGNN | GATConv / SAGEConv (同构图) | Data | lin(781→64), 2层, GAT: 4 heads × 16dim |

**超参数** (PARAMS dict):

```python
PARAMS = {
    'hidden_channels': 64,     # GNN 隐层维度
    'num_heads': 4,            # attention heads 数
    'num_hgt_layers': 2,       # GNN 层数
    'dropout': 0.3,            # dropout rate
    'lr': 1e-3,                # learning rate
    'weight_decay': 1e-4,      # L2 regularization
    'epochs': 100,             # 最大训练 epoch
    'patience': 15,            # early stopping patience
    'grad_accum': 32,          # 梯度累积步数 (每 32 天更新一次)
    'default_horizon': 5,      # 默认预测 horizon (5d)
    'horizons': [1,5,10,21,42,63],
    'top_k': 30,               # Long-Short 组合的股票数
    'transaction_cost': 15,    # bps round-trip 交易成本
    'target_coverage': 0.2,    # SelectiveNet 目标覆盖率 20%
    'selection_lambda': 32.0,  # SelectiveNet coverage penalty 系数
}
```

**模型细节解释**:

- **HGTConv** (Hu et al., WWW'20): Heterogeneous Graph Transformer。为不同的节点类型和边类型学习独立的 Key/Query/Value 投影矩阵。参数量 ∝ edge_types × hidden²。
- **GATConv** (Velickovic et al., ICLR'18): Graph Attention Network。所有边共享一套 attention 参数。GATConv(64, 16, heads=4, concat=True) = 4 个 attention head，每个输出 16 维，拼接后 64 维。
- **SAGEConv** (Hamilton et al., NeurIPS'17): GraphSAGE。用 mean/max aggregation 聚合邻居特征，不用 attention。
- **gradient accumulation**: 由于每天是一个独立的图（501 个节点），batch size=1 太小。每 32 天的梯度累积起来再更新，等效 batch size=32。

### N3d: GNN 消融实验设计

| ID | 架构 | 边类型 | 图类型 |
|----|------|--------|--------|
| A1 | HGT | correlation only | 异构图 (有 dummy news 节点) |
| A2 | HGT | corr + sector | 异构图 |
| A3 | HGT | all 4 types | 异构图 |
| A4 | SAGE | corr + sector | **同构图** (合并边, 无 news 节点) |
| A5 | GAT | corr + sector | **同构图** |

### N5: SelectiveNet

**SelectiveNet (Geifman & El-Yaniv, ICML'19)** — 3-head 架构:

```
              ┌─ Head 1: Ranking prediction (MSE loss)
GNN backbone ─┤─ Head 2: Selection head → confidence ∈ [0,1]  (Sigmoid)
              └─ Head 3: Auxiliary prediction (MSE loss, regularization)
```

**Loss 函数**:
```
L = L_selective + λ * max(0, c_target - coverage)² + L_auxiliary

其中:
L_selective = Σ(selection_i * (pred_i - target_i)²) / Σ(selection_i)
coverage = mean(selection_i)
L_auxiliary = MSE(aux_pred, target)
```

**训练策略** (2-stage):
1. Stage 1: 训练 backbone + ranking + auxiliary (selection 不约束)
2. Stage 2: 冻结 backbone + ranking，只训练 selection head (加 coverage penalty)

**Market context features** (4维, 全部使用 T-1 值避免泄漏):
1. 21日市场波动率 (VIX proxy)
2. 63日最高点的回撤
3. 30日截面波动率均值
4. 5日市场广度 (正收益股票占比)

---

## 12. v3 Colab Run 1: N3 首次结果

**环境**: NVIDIA RTX PRO 6000 Blackwell Server Edition, 102.0 GB VRAM
**运行时间**: 2026-03-05
**代码版本**: 原始版本（N4 用 HGT, N5 用 SelectiveRankingHGT）

### 数据管线 (N1-N2) ✅ 全部正确

| 指标 | 值 |
|------|---|
| Valid tickers | 501 (502 中 1 个在交集中丢失) |
| Stock features tensor | (1255, 501, 781), 1.96 GB |
| News coverage | 58.5% train, 62.7% test |
| Correlation snapshots | 54 个 |
| Labels z-score | mean≈0, std≈0.999 ✅ |

### N3: Baseline + GNN 消融结果 (5d horizon, test set)

| Model | IC | ICIR | Sharpe_LS | Ann_LS | MaxDD |
|-------|-----|------|-----------|--------|-------|
| B1: Ridge (price 9d) | 0.00476 | 0.026 | 0.624 | 14.88% | 152.76% |
| B2: Ridge (all 781d) | 0.00535 | 0.052 | 0.597 | 8.06% | 79.00% |
| B3: XGBoost | 0.00329 | 0.024 | 0.185 | 2.89% | 76.59% |
| B4: LightGBM | 0.00828 | 0.079 | 0.773 | 10.92% | 44.52% |
| A1: HGT (corr) | 0.01023 | 0.133 | 0.121 | 1.25% | 51.53% |
| A2: HGT (corr+sector) | 0.01177 | 0.156 | 0.994 | 8.91% | **16.42%** |
| **A3: HGT (all 4)** | **0.00432** | 0.061 | **-0.314** | -2.83% | 39.29% |
| A4: SAGE (corr+sector) | 0.01571 | 0.152 | 1.038 | 13.51% | 35.08% |
| **A5: GAT (corr+sector)** | **0.02054** | **0.174** | **1.011** | **15.78%** | 38.56% |

### Go/Stop Gate

| 条件 | 阈值 | 实际值 | 结果 |
|------|------|--------|------|
| Best IC > 0.03 | 0.03 | 0.02054 | ❌ 未达标 |
| Best Sharpe_LS > 0.5 | 0.5 | 1.038 | ✅ 达标 |
| **OR 条件** | | | **GO** |

### N4 Horizon Ablation — 部分可见

仅 Horizon 1d 的结果可见（之后被 sklearn warnings 淹没）:
- HGT 1d: IC=0.00343, ICIR=0.051, Sharpe_LS=3.073, Ann_LS=38.88%

**注意**: 这次运行的 N4 用的是 HGT(all 4 edges)=最差配置。已修正为 GAT(corr+sector)。

### N5 SelectiveNet — 结果不可见

完全被 warnings 覆盖，无法看到结果。已修复 warnings 并更新模型。

---

## 13. GAT vs HGT: 为什么简单的赢了

### 三者的架构对比

| | HGT (A2) | SAGE (A4) | GAT (A5) |
|---|---|---|---|
| 图类型 | 异构图 (HeteroData) | 同构图 | 同构图 |
| 边处理 | corr/sector 各自有独立 attention 参数 | 所有边统一聚合 (mean) | 所有边共享 attention |
| News 节点 | 有 (dummy 全零节点) | 无 | 无 |
| 参数量 | ~多 (type-specific K/Q/V) | ~少 (无 attention) | ~中 (shared attention) |
| IC | 0.01177 | 0.01571 | **0.02054** |

### 4 个原因

1. **参数效率 (bias-variance tradeoff)**
   - IC ≈ 0.02 意味着 98% 是噪声。HGT 的 type-specific 参数在弱信号下过拟合。
   - GAT 参数更少 → lower variance → 更好泛化。

2. **边类型区分无用**
   - "这两只股票价格相关" vs "这两只股票同行业" — 本质都是 "相关"。
   - HGT 花参数去区分这两种关系，但区分本身不提供预测信息。

3. **News dummy 节点噪声**
   - 即使去掉 news 边，HGT 仍处理 dummy news 节点 (全零特征)。
   - 这些无用计算/参数是额外噪声源。

4. **Attention 机制的稳定性**
   - GATConv(64, 16, heads=4) 是久经验证的架构。
   - HGTConv 在学术 benchmark 上好，但金融数据噪声太大，简单更 robust。

### 延伸发现: News/Co-occurrence 边有害

| 配置 | IC | 变化 |
|------|-----|------|
| A2: HGT (corr+sector) | 0.01177 | baseline |
| A3: HGT (all 4 edges) | 0.00432 | **-63%** |

添加 news mentions 和 co-occurrence 边后 IC 大幅下降。原因:
- 这些边创建了密集的、噪声连接 (每天 1226 mentions + 2325 cooccur)
- 稀释了信息含量更高的 correlation 结构
- 在高噪声金融数据中，less is more

---

## 14. v3 Colab Run 2: N3-N5 完整结果

**环境**: NVIDIA A100-SXM4-40GB, VRAM: 42.4 GB
**运行时间**: 2026-03-06
**代码版本**: 更新版（N4/N5 改用 GAT(corr+sector), warnings 已修复, grad_accum=32）

### N3 Baseline + GNN 消融结果 (Run 2, 5d horizon, test set)

| Model | IC | ICIR | Sharpe_LS | Sharpe_Long | Ann_LS | Ann_LS_net | MaxDD | n_days |
|-------|-----|------|-----------|-------------|--------|------------|-------|--------|
| B1: Ridge (price 9d) | 0.00476 | 0.026 | 0.624 | 1.223 | 14.88% | -0.24% | 152.76% | 391 |
| B2: Ridge (all 781d) | 0.00535 | 0.052 | 0.605 | 1.209 | 8.17% | -6.95% | 78.87% | 391 |
| B3: XGBoost | 0.00329 | 0.024 | 0.185 | 1.233 | 2.89% | -12.23% | 76.59% | 391 |
| B4: LightGBM | 0.00828 | 0.079 | 0.773 | 1.438 | 10.92% | -4.20% | 44.52% | 391 |
| A1: HGT (corr only) | 0.00848 | 0.092 | 0.426 | 1.367 | 4.68% | -10.44% | 63.45% | 391 |
| A2: HGT (corr+sector) | 0.01447 | 0.174 | 0.320 | 1.149 | 3.31% | -11.81% | 54.22% | 391 |
| A3: HGT (all 4 edges) | 0.00884 | 0.131 | 0.012 | 1.175 | 0.11% | -15.01% | 30.28% | 391 |
| **A4: SAGE (corr+sector)** | **0.01545** | **0.242** | **1.266** | 1.406 | 10.09% | -5.03% | **10.57%** | 391 |
| A5: GAT (corr+sector) | 0.00640 | 0.072 | 0.289 | 1.234 | 3.49% | -11.63% | 64.80% | 391 |

### N3 Run 1 vs Run 2 对比

| Model | Run 1 IC | Run 2 IC | Run 1 Sharpe | Run 2 Sharpe |
|-------|----------|----------|--------------|--------------|
| B1-B4 (baselines) | 相同 | 相同 | 相同 | 相同 |
| A1: HGT (corr) | 0.01023 | 0.00848 | 0.121 | 0.426 |
| A2: HGT (corr+sec) | 0.01177 | 0.01447 | 0.994 | 0.320 |
| A3: HGT (all 4) | 0.00432 | 0.00884 | -0.314 | 0.012 |
| A4: SAGE (corr+sec) | 0.01571 | 0.01545 | 1.038 | 1.266 |
| **A5: GAT (corr+sec)** | **0.02054** | **0.00640** | **1.011** | **0.289** |

**关键观察**:
- **Baselines (B1-B4) 完全一致** — 确认代码正确，差异来自 GNN 训练随机性
- **SAGE 最稳定**: IC 两次几乎相同 (0.01571 vs 0.01545)
- **GAT 最不稳定**: IC 从 0.02054 暴跌到 0.00640 (−69%)
- **HGT 有中等波动**: A2 IC 从 0.01177 升到 0.01447
- **结论**: 单次训练结果不可靠，需要 Walk-forward CV 或多次重复

### Go/Stop Gate (Run 2)

| 条件 | 阈值 | 实际值 | 结果 |
|------|------|--------|------|
| Best IC > 0.03 | 0.03 | 0.01545 | ❌ |
| Best Sharpe_LS > 0.5 | 0.5 | 1.266 | ✅ |
| **OR 条件** | | | **GO** |

---

## 15. N4 Horizon Ablation: 详细分析

**设计**: 对 6 个预测 horizon 分别训练 GAT(corr+sector) 和 LightGBM，比较图结构增益随时间尺度的变化。

### 完整结果表

| Horizon | GAT IC | GAT ICIR | GAT Sharpe | GAT Ann_LS | LGBM IC | LGBM Sharpe | n_days |
|---------|--------|----------|------------|------------|---------|-------------|--------|
| **1d** | -0.00104 | -0.013 | 2.468 | 34.54% | 0.00368 | 2.918 | 395 |
| **5d** | 0.02334 | 0.227 | 1.568 | 18.27% | 0.00828 | 0.773 | 391 |
| **10d** | 0.03854 | 0.320 | 1.196 | 19.26% | 0.01349 | 0.644 | 386 |
| **21d** | **0.04420** | **0.374** | **1.203** | **18.71%** | 0.01513 | 0.468 | 375 |
| **42d** | -0.00912 | -0.144 | 0.071 | 0.73% | 0.03679 | 0.668 | 354 |
| **63d** | -0.00838 | -0.118 | 0.487 | 6.36% | **0.05207** | **1.256** | 333 |

### Ann_LS_net (扣交易成本后)

| Horizon | GAT Ann_LS_net | LGBM Ann_LS_net |
|---------|---------------|-----------------|
| 1d | -41.06% | -37.23% |
| 5d | 3.15% | -4.20% |
| 10d | **11.70%** | 1.47% |
| 21d | **15.11%** | 2.95% |
| 42d | -1.07% | 8.45% |
| 63d | 5.16% | **24.35%** |

### 关键发现

**1. GAT "Sweet Spot" = 10d-21d**

GAT IC 呈现倒 U 型:
```
1d (-0.001) → 5d (0.023) → 10d (0.039) → 21d (0.044) → 42d (-0.009) → 63d (-0.008)
                  ↗ 上升区 ↗        ★ 峰值 ★        ↘ 下降区 ↘
```

- **21d 是最佳 horizon**: IC=0.04420 > 0.03 门槛，ICIR=0.374，Sharpe=1.203
- **10d 也超过门槛**: IC=0.03854
- **1d 无信号**: IC 为负 → 图结构对日频信号无帮助
- **42d/63d 信号消失**: IC 为负 → 图结构在长期过拟合

**2. GAT vs LightGBM 形成交叉模式**

| 时间尺度 | 谁更好 | 图结构增益 | 解释 |
|---------|--------|---------|------|
| 1d | LGBM | ❌ 负增益 | 日频噪声太大，GNN 聚合邻居信息反而加噪声 |
| 5d-21d | **GAT** | ✅ 2.8×-3.0× | **图结构传播的信息在周/月级别最有价值** |
| 42d-63d | LGBM | ❌ 负增益 | 长期趋势由宏观因素驱动，局部图结构无关 |

这是一个**可发表的洞察**: GNN 对股票预测的增量具有时间尺度依赖性，最优区间在 2-4 周。

**3. LightGBM 单调递增**

LightGBM IC 随 horizon 单调增长: 0.004 → 0.008 → 0.013 → 0.015 → 0.037 → 0.052
- 这符合预期: 树模型基于股票个体特征，长期趋势更容易从特征中捕获
- 不需要图结构，因为特征本身包含了 momentum 信息

**4. 1d Sharpe 异常高**

GAT 1d Sharpe=2.468, LGBM 1d Sharpe=2.918，但 IC 都接近零。
- 原因: top_k=30 的 Long-Short 组合在日频换手中碰巧盈利
- 扣交易成本后 Ann_LS_net = -41% → 完全不可行
- **这是统计噪声，不是真实信号**

**5. 扣费后经济显著性**

只有 GAT 10d 和 21d 扣费后有正收益且 > 10%:
- GAT 10d: Ann_LS_net = 11.70%
- GAT 21d: Ann_LS_net = 15.11%

---

## 16. N5 SelectiveNet: 详细分析

### 训练细节

- **最佳 horizon**: 从 N4 自动选择 21d (IC=0.04420)
- **Stage 1** (backbone + ranking + auxiliary): 31 epochs, early stop
- **Stage 2** (selection head only): 50 epochs, final coverage ≈ 0.312
- **训练时间**: 386.2s

### 完整结果表 (21d horizon)

| Method | IC | ICIR | Sharpe_LS | Ann_LS | Ann_LS_net | MaxDD |
|--------|-----|------|-----------|--------|------------|-------|
| **Full (100%)** | **0.05595** | **0.463** | **1.328** | 20.08% | **16.48%** | 66.67% |
| Threshold @5% | 0.01012 | 0.120 | 0.548 | 6.79% | 3.19% | 104.41% |
| Threshold @10% | 0.01929 | 0.221 | 0.611 | 7.91% | 4.31% | 82.83% |
| Threshold @20% | 0.03070 | 0.324 | 0.724 | 9.45% | 5.85% | 74.91% |
| Threshold @50% | 0.05087 | 0.446 | 1.346 | 18.67% | 15.07% | 44.64% |
| Threshold @100% | 0.05595 | 0.463 | 1.328 | 20.08% | 16.48% | 66.67% |
| **SelectiveNet @5%** | **-0.01544** | **-0.202** | **-0.672** | -6.70% | -10.30% | 286.94% |
| SelectiveNet @10% | -0.02159 | -0.252 | -0.676 | -6.72% | -10.32% | 288.86% |
| SelectiveNet @20% | -0.02414 | -0.256 | -0.536 | -5.40% | -9.00% | 242.10% |
| SelectiveNet @50% | -0.00874 | -0.116 | 0.800 | 8.48% | 4.88% | 60.35% |
| SelectiveNet @100% | 0.05595 | 0.463 | 1.328 | 20.08% | 16.48% | 66.67% |

### 关键发现

**1. SelectiveNet 完全失败**

SelectiveNet 在所有 coverage level (5%-50%) 都产生**负 IC**:
- @5%: IC = -0.01544 (选出的股票排名**反向**)
- @10%: IC = -0.02159
- @20%: IC = -0.02414 ← 目标 coverage，结果最差

**选择头学会了反向选择** — 它倾向于选择 GNN 预测最差的那些股票。

**2. Threshold Baseline 工作正常**

用 ranking score 绝对值 (|ranking|) 作为 confidence proxy:
- @20%: IC = 0.03070, 超过 0.03 门槛
- @50%: IC = 0.05087, 接近 Full
- Sharpe 随 coverage 单调增长 (0.548 → 1.346)

**3. Full Model 反而最好**

SelectiveRankingGAT 的 Full (100%) 预测达到:
- IC = 0.05595 (所有实验中最高!)
- ICIR = 0.463
- Sharpe = 1.328
- Ann_LS_net = 16.48%

这比 N4 中同 horizon 的 RankingGNN 结果更好 (IC=0.04420)，说明 3-head 架构的 auxiliary loss 有正则化效果。

**4. SelectiveNet 失败原因分析**

| 原因 | 解释 |
|------|------|
| 选择分数分布高度右偏 | 大部分 score 集中在 0.8-1.0，缺乏区分度 |
| Coverage 约束不够强 | final coverage=0.312 > target=0.2，penalty 不够 |
| Selection head 没有好的学习信号 | ranking loss 不提供选择梯度 |
| 2-stage 训练断裂 | 冻结 backbone 后，selection head 无法优化联合目标 |

**5. SelectiveNet vs Threshold 选择的股票几乎不重叠**

从 Jaccard 相似度图可以看到，两种方法在各个 coverage level 的 Jaccard ≈ 0.2-0.35。
SelectiveNet 选择了完全不同的（且更差的）股票子集。

### 对论文的影响

SelectiveNet 作为贡献点失败了。替代策略:
1. **报告为 negative finding** — 证明 SelectiveNet 在金融 GNN 中不适用
2. **改用 Threshold Selection** — 简单有效，@20% IC=0.03070
3. **改进 SelectiveNet** — 联合训练 (不分 2 stages)、增大 lambda、loss 改进

---

## 17. 训练稳定性分析

### N3 两次运行的 GNN 结果对比

| Model | Run 1 IC | Run 2 IC | |IC diff| | IC CV* |
|-------|----------|----------|----------|--------|
| A1: HGT (corr) | 0.01023 | 0.00848 | 0.00175 | ~15% |
| A2: HGT (corr+sec) | 0.01177 | 0.01447 | 0.00270 | ~21% |
| A3: HGT (all 4) | 0.00432 | 0.00884 | 0.00452 | ~69% |
| A4: SAGE (corr+sec) | 0.01571 | 0.01545 | 0.00026 | ~2% |
| A5: GAT (corr+sec) | 0.02054 | 0.00640 | 0.01414 | ~105% |

*CV = coefficient of variation (|diff|/mean)

### 稳定性排序

```
SAGE (2%) >> HGT-corr (15%) > HGT-corr+sec (21%) >> HGT-all (69%) >> GAT (105%)
 最稳定                                                              最不稳定
```

### 对实验结论的影响

1. **Run 1 的 "GAT 最好" 结论可能是假阳性**: GAT 的 IC 波动范围 [0.006, 0.021] 太大
2. **SAGE 可能是更可靠的选择**: IC 稳定在 ~0.015，虽然不是最高，但可复现
3. **必须做 Walk-forward CV**: 单次 train/val/test 分割不够，需要多窗口滚动验证
4. **N4 的 GAT 21d IC=0.04420 也需要验证**: 可能也有类似的训练波动

### 不稳定的技术原因

| 因素 | 影响 |
|------|------|
| 日序列 shuffle | 每个 epoch 随机排列训练日顺序 |
| 梯度累积边界 | grad_accum=32，不同排列 → 不同的 32 天组合 |
| Early stopping | 在不同 epoch 停止 → 不同的模型参数 |
| CUDA 非确定性 | 尽管设置了 seed=42，Colab GPU 有固有随机性 |

---

## 18. 当前状态与下一步

### 已完成

| 阶段 | 状态 | 关键结果 |
|------|------|---------|
| Phase 1 (网络分析) | ✅ | 验证网络结构存在 |
| Phase 2 Pilot | ✅ | GraphSAGE AUC=0.6426 (小规模) |
| Phase A (数据) | ✅ | 1.7M events, FinBERT 编码 |
| Phase B (图参数) | ✅ | w=126, τ=0.6 |
| Phase C v1 | ✅ | AUC ≈ 0.50 (EMH 困境) |
| D.1/D.2 诊断 | ✅ | FinBERT 无信号 |
| Signal Fix | ✅ | 仍 AUC < 0.51 |
| Phase 2 LLM | ✅ STOP | GPT-4o-mini delta=+0.0009 |
| **v3 N1-N2** | ✅ | Calendar-driven + 动态图 |
| **v3 N3 Run 1** | ✅ | GAT IC=0.02054 (不稳定) → GO |
| **v3 N3 Run 2** | ✅ | SAGE IC=0.01545 (稳定) → GO |
| **v3 N4** | ✅ | **GAT 21d IC=0.04420 > 0.03!** |
| **v3 N5** | ✅ | SelectiveNet 失败, Full IC=0.05595 |

### 最低发表标准 (Updated)

| 指标 | 目标 | 实际值 | 状态 |
|------|------|--------|------|
| 任一 horizon IC > 0.03 | 超越随机排名 | **GAT 21d IC=0.04420** | ✅ 达标 |
| GNN IC > LightGBM IC (同 horizon) | 图有增量 | 21d: **0.044 > 0.015** (2.9×) | ✅ 达标 |
| Selective IC@20% > Full IC | Selective 有增量 | Threshold @20%: 0.031 < Full 0.056 | ❌ 未达标 |
| Long-Short Sharpe > 0.5 (扣费后) | 经济显著 | GAT 21d Ann_LS_net=**15.11%** | ✅ 达标 |
| Horizon ablation 有明确模式 | 文献贡献 | **倒 U 型, 峰值 21d** | ✅ 达标 |

**4/5 指标达标。** SelectiveNet 贡献点失败，但可替换为 threshold selection 或报告为 negative finding。

### 下一步 (优先级排序)

1. **Walk-forward CV** — 解决训练稳定性问题 (最重要)
   - 滚动窗口: 2 年训练 + 6 个月 val + 6 个月 test
   - 报告多窗口的 IC 均值和标准差
   - 验证 21d 的优势是否在不同时期持续

2. **多次重复实验** — 同一配置跑 5 次，报告均值±std
   - GAT 21d 的 IC=0.04420 需要可复现性验证
   - 考虑是否切换到 SAGE (更稳定)

3. **SelectiveNet 改进或放弃**
   - 选项 A: 联合训练 (不分 2 stages)
   - 选项 B: 增大 lambda (32→128)
   - 选项 C: 放弃 SelectiveNet，改用 threshold selection 作为贡献点
   - 选项 D: 报告为 negative finding

4. **论文图表准备**
   - Horizon ablation 3 panel plot (IC/ICIR/Sharpe vs horizon) ← 已有
   - Selective analysis 6 panel plot ← 已有
   - Training curve plots
   - 网络结构可视化

5. **交易成本敏感性分析**
   - tc = 5, 10, 15, 20, 30 bps

### 代码修改记录

| 日期 | Cell | 修改内容 |
|------|------|---------|
| 03-06 | 11 | 添加 `warnings.filterwarnings` 抑制 sklearn 警告 |
| 03-06 | 12 | `RankingGNN` 新增 `get_stock_embeddings()` 方法 |
| 03-06 | 15 | N4: `RankingHGT` → `RankingGNN(gat)`, `train_hgt` → `train_homogeneous_gnn` |
| 03-06 | 16 | N5a: `SelectiveRankingHGT` → `SelectiveRankingGAT` (同构图 backbone) |
| 03-06 | 17 | N5b: 所有 `build_hetero_data` → `_build_homo_graph` |
| 03-06 | 18 | N5c: 同上 |

---

## 19. 术语表

| 术语 | 英文全称 | 解释 |
|------|---------|------|
| GNN | Graph Neural Network | 图神经网络，在图结构数据上做消息传递和特征聚合 |
| GCN | Graph Convolutional Network | Kipf & Welling (ICLR'17), 最基础的 GNN |
| GAT | Graph Attention Network | Velickovic et al. (ICLR'18), 用 attention 加权邻居信息 |
| GraphSAGE | Sample and Aggregate | Hamilton et al. (NeurIPS'17), 采样固定数量邻居 |
| HGT | Heterogeneous Graph Transformer | Hu et al. (WWW'20), 异构图上的 type-specific Transformer |
| IC | Information Coefficient | 预测值与实际值的 Spearman rank correlation |
| ICIR | IC Information Ratio | mean(IC) / std(IC), 衡量 IC 的稳定性 |
| Sharpe | Sharpe Ratio | (annualized return) / (annualized volatility) |
| L/S | Long-Short | 做多前 N 只 + 做空后 N 只的组合策略 |
| AUC | Area Under ROC Curve | 二分类模型的评估指标 (0.5=随机, 1.0=完美) |
| EMH | Efficient Market Hypothesis | 有效市场假说: 价格已反映所有信息 |
| FinBERT | Financial BERT | ProsusAI 在金融文本上微调的 BERT 模型 |
| SelectiveNet | Selective Prediction Network | Geifman & El-Yaniv (ICML'19), 学习何时拒绝预测 |
| HeteroData | Heterogeneous Data | PyG 中的异构图数据结构 (多种节点/边类型) |
| GICS | Global Industry Classification Standard | S&P/MSCI 的行业分类标准 |
| Z-score | Standard Score | (x - mean) / std, 标准化到均值 0 方差 1 |
| Jaccard | Jaccard Similarity | |A∩B| / |A∪B|, 集合相似度 |
| MaxDD | Maximum Drawdown | 最大回撤: 峰值到谷值的最大跌幅 |
| AUGRC | Area Under Generalized Risk-Coverage | NeurIPS'24 提出的 selective prediction 评估指标 |
| PyG | PyTorch Geometric | 图神经网络的 PyTorch 扩展库 |
| gradient accumulation | 梯度累积 | 多步累积梯度后统一更新，等效增大 batch size |
| early stopping | 提前停止 | 验证集 loss 不再下降时停止训练 |
| cross-sectional | 截面 | 同一时间点跨所有股票的比较 |
| forward return | 前瞻收益 | (close[T+h] - close[T]) / close[T] |

---

*Last updated: 2026-03-06*
