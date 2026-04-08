# DynHetGNN-SP 项目批判、分析与行动计划

## 目录

1. [核心批判](#1-核心批判)
2. [为什么预测力弱](#2-为什么预测力弱)
3. [相关工作与竞争定位](#3-相关工作与竞争定位)
4. [Time Series + LLM 领域现状](#4-time-series--llm-领域现状)
5. [改进方案](#5-改进方案)
6. [立即可执行的任务（Qwen 3.6 API）](#6-立即可执行的任务qwen-36-api)
7. [论文重新定位建议](#7-论文重新定位建议)
8. [参考文献](#8-参考文献)

---

## 1. 核心批判

### 1.1 最根本的问题：任务本身是否可解

论文标题是"S&P 500 Stock Direction Prediction"，但从未回答一个前置问题：**S&P 500 的日频方向在统计意义上是否可预测？**

- AUC 0.64 高于 random（0.50），但需要检验：
  - 不同年份是否稳定？（如 2020 疫情极端行情贡献了大部分 AUC，而 2023 年只有 0.52，那模型只是在拟合 tail events）
  - Logistic regression baseline 的 AUC 是多少？如果是 0.61，整套框架只贡献 0.03 AUC
  - Always-predict-majority-class 的 baseline accuracy 是多少？（S&P 500 历史约 55% 交易日上涨）
- S&P 500 是全球信息效率最高的市场之一，价格快速 incorporate 信息，信噪比极低
- LLM 从新闻中提取的信号可能信噪比太低，不足以支撑可靠盈利交易（Lopez-Lira & Tang, 2024）
- 近年市场复杂度（近似熵、样本熵）增加，ML 算法预测性能在下降（Explorative study, Entropy 2022）

### 1.2 实验设计硬伤

**Walk-forward validation 只有两个 fold：**
- 两个 fold 的结果方差极大，无法得出统计可靠结论
- 如果 fold 1 AUC=0.68、fold 2 AUC=0.60，报 0.64 可能只是随机波动

**没有 transaction cost analysis：**
- Direction prediction 如果不能转化为扣除交易成本后的正收益，学术和实用价值都大打折扣

**Selective prediction 评估不完整：**
- 只报"子集上 AUC 更高"不够
- 未报 coverage 比例、coverage-AUC trade-off curve
- 未与其他 selective prediction 方法对比（confidence thresholding、MC Dropout uncertainty）
- 未排除"随机选同比例样本 AUC 也提升"的可能

### 1.3 模型设计质疑

**HGT 的必要性未被证明：**
- 缺少 HGT vs. GAT vs. GCN vs. 完全去掉 GNN 的完整 ablation

**LLM multi-field output vs. FinBERT：**
- LLM 计算成本是 FinBERT 的 10-100 倍
- LLM 输出的确定性/可复现性存疑
- 若使用 GPT-4 等闭源模型，实验不可复现

**图构建合理性：**
- Price correlation 是 backward-looking 的，market regime 一变就失效
- Correlation threshold 的选择是任意的
- LLM news edges 可能被热门新闻主导
- 图结构在不同时期可能剧烈变化，GNN 学到的是 stale 拓扑

### 1.4 Writing 和 Positioning 问题

- 贡献声称过大（overclaim）
- 缺少与 TH-GNN、ChatGPT Informed GNN、HATS 的直接对比
- 未 justify 为什么选 S&P 500（最难预测的市场）

---

## 2. 为什么预测力弱

### 2.1 任务层面：信号天花板

| 因素 | 说明 |
|------|------|
| 市场效率 | S&P 500 接近 strong-form efficiency，新信息在分钟内被价格消化 |
| 信噪比 | 在 fev-bench 上，最强模型相对 Seasonal Naive 只减少约 35-40% 误差 |
| DLinear 现象 | 单层线性模型在标准 benchmark 上打败过 transformer 架构 |
| EMH 约束 | 在有效市场中，价格反映所有可用信息，预测空间极小 |
| 适应性市场假说 | 市场效率非静态，在不同时期切换，但 S&P 500 长期处于高效率区间 |

### 2.2 GNN 层面：结构性瓶颈

| 问题 | 说明 | 文献支持 |
|------|------|----------|
| 图构建质量 | Correlation graph 几乎 dense，threshold 任意，信号密度低 | JSE 实验（Pillay & Moodley, 2021）发现 correlation inclusion 影响 negligible |
| 关系数据不一定有用 | 固定关系在特定市场条件下有用，其他条件下是噪声 | HATS（Kim et al., 2019）发现 GCN 用关系数据 Sharpe ratio 反而低于 LSTM |
| Over-smoothing | 多层 message passing 后 node representation 趋同，丢失个股异质性 | ChatGPT Informed GNN 论文指出 DOW 30 规模下复杂 GNN 可能 oversmooth |
| 图不稳定 | Correlation 在不同 regime 下剧变，GNN 学的拓扑是 stale 的 | — |
| GNN bottleneck | GNN-based stock predictors 的性能遇到难以突破的 bottleneck | ACM Computing Surveys 2025 综述 |

### 2.3 LLM 层面：可能没有增量

| 发现 | 来源 |
|------|------|
| 移除 LLM 组件或替换为基础 attention 层，性能不降反升 | Tan et al., NeurIPS 2024 |
| 预训练 LLM 不比从头训练的模型好 | Tan et al., NeurIPS 2024 |
| LLM 不能表征时序中的序列依赖关系 | Tan et al., NeurIPS 2024 |
| 在 few-shot 设定下 LLM 也没有帮助 | Tan et al., NeurIPS 2024 |
| 当 LLM 被广泛采用后，可预测性会消失（所有人看到同一信号） | Lopez-Lira & Tang, 2024 |

### 2.4 模型复杂度 vs. 信号强度的根本矛盾

**用最复杂的模型（HetGNN + LLM + selective prediction）去攻最难的问题（S&P 500 direction prediction）。** 在这个市场上，alpha 信号极其微弱且短暂，复杂模型反而容易过拟合训练期的 noise。

---

## 3. 相关工作与竞争定位

### 3.1 高度相似的工作（必须直接对比）

| 论文 | 会议/期刊 | 与 DynHetGNN-SP 的重叠 | 差异点 |
|------|-----------|----------------------|--------|
| **TH-GNN** (Xiang et al.) | CIKM 2022 | Temporal + heterogeneous GNN + 金融时序 + 美股/A股 | 最直接的竞争者，需要仔细 differentiation |
| **ChatGPT Informed GNN** (Chen et al.) | arXiv 2023 | LLM 从新闻建图 + GNN + stock movement prediction | 他们 LLM 做关系抽取建图，你做特征编码 |
| **DASF-Net** | JRFM 2025 | GNN + FinBERT + S&P 500 + 多模态 | 做价格预测而非方向分类 |
| **HATS** (Kim et al.) | arXiv 2019 | Hierarchical GAT + S&P 500 + direction prediction | 无 LLM，但 heterogeneous attention + S&P 500 设定接近 |
| **MDGNN** (Qian et al.) | AAAI 2024 | 多关系动态 GNN + 股票投资预测 | 定义了 multifacetedness 和 temporal patterns |
| **Zero-Shot Stock Graph** | EMNLP 2025 FinNLP | LLM zero-shot 股票关系图提取 + RGCN/RGAT | LLM 建图 + GNN 预测 |

### 3.2 LLM 结构化特征提取的已有工作（你想做的方向别人已经做了）

| 论文/系统 | 做法 |
|-----------|------|
| MarketSenseAI 2.0 (Fatouros et al., 2025) | RAG + LLM agents 处理 SEC filings、earnings calls、机构报告 |
| Medya et al. | 10 年 6300 家公司 earnings call transcripts，语义特征比 hard data 更能预测股价 |
| Chiang et al. (2025) | LLM 评估 earnings call Q&A 的高管透明度 |
| Alpha-GPT / AlphaAgent | LLM 生成交易信号 / 自动化 alpha 挖掘 (EMNLP 2025) |
| DNA Framework (ICAIF 2025) | GPT-o3 对 10-K 年报做 CoT Micro-Scorecard 分析，强制 JSON 输出 |
| LLM for Stock Selection (2026) | LLM 生成可执行代码将结构化金融数据转化为交易信号 |

### 3.3 你可能独特的 Contribution

- **Selective prediction / volatility-calibrated abstention**：在 stock prediction 中显式建模"什么时候不预测"。需确认无人在 stock prediction context 做过类似 calibrated abstention。
- **Conditional analysis**：在什么市场条件/regime 下 GNN 和 LLM 有增量。

---

## 4. Time Series + LLM 领域现状

### 4.1 顶会关键论文（2023-2025）

**ICLR 2024：**
- Time-LLM：reprogramming LLM，被引 1000+
- TEMPO：prompt-based GPT-2 预训练
- TEST：text prototype aligned embedding
- FITS：10K 参数频域方法，打了 LLM-based 方法的脸

**ICML 2024：**
- Position Paper: What Can LLMs Tell Us about Time Series Analysis（偏 skeptical）
- TimesFM (Google)：decoder-only 时序基础模型

**NeurIPS 2024：**
- "Are Language Models Actually Useful for Time Series Forecasting?"（最重要的批判性工作）
- UniTS：统一多任务时序模型
- From News to Forecast：event analysis + LLM + reflection

**ICLR 2025：**
- Time-MoE：billion-scale MoE 时序基础模型
- Neural Scaling Laws for TSFMs
- "Can LLMs Understand Time Series Anomalies?"

### 4.2 核心共识

1. **LLM 可能根本没用**：Tan et al. ablation 证明 LLM 组件可移除
2. **TSFM 还没到 "BERT Moment"**：轻量级监督 baseline 常常匹配 TSFM 性能
3. **泛化受限于预训练分布**
4. **Benchmark 污染严重**
5. **时序信号本身有天花板**

---

## 5. 改进方案

### 5.1 必须做的 Ablation Table

这是论文能否 survive review 的关键：

| Model | AUC (mean±std) | Coverage | Sharpe (after cost) |
|-------|---------------|----------|-------------------|
| Always-predict-majority | — | 100% | — |
| Logistic Regression + technical indicators | ? | 100% | ? |
| LSTM only (price features) | ? | 100% | ? |
| LSTM + GAT (homogeneous graph) | ? | 100% | ? |
| LSTM + HGT (heterogeneous graph) | ? | 100% | ? |
| Full model (HGT + LLM + selective) | ? | ?% | ? |
| Full model - LLM (用 FinBERT 替代) | ? | ?% | ? |
| Full model - LLM (用 Qwen structured features 替代) | ? | ?% | ? |
| Full model - selective head | ? | 100% | ? |
| Full model + LLM-constructed graph (替代 correlation graph) | ? | ?% | ? |

### 5.2 实验设计修复

| 问题 | 修复方案 |
|------|----------|
| 只有 2 个 walk-forward fold | 增加到至少 5 个，或 continuous walk-forward（每月/季度重训），报 mean ± std |
| 无统计检验 | 加 Diebold-Mariano test 或 Model Confidence Set |
| 无 transaction cost | 加 backtest：按 prediction 每天做多/做空/持有，扣单边 5bps，报 Sharpe/MaxDD/turnover |
| Selective prediction 评估不完整 | 画 coverage vs. AUC curve + random selection 对照 + confidence thresholding baseline |
| 未做 predictability analysis | 论文第一节放 permutation test / bootstrap 检验 AUC 是否显著高于 chance level |

### 5.3 图构建改进实验

| 实验 | 目的 |
|------|------|
| 只用 GICS sector/industry 静态图 vs. 你的复杂动态图 | 看动态图是否有增量 |
| LLM-constructed graph vs. correlation graph | 看语义图是否比统计图更有预测力 |
| Graph statistics analysis（不同时间窗口下的 degree、density、连通分量） | 评估图拓扑的稳定性 |
| Correlation threshold sensitivity analysis | 确认 threshold 选择的 robustness |

### 5.4 可能的论文重新定位方向

**方向一（Conditional Analysis）：** 不是证明 GNN 没用，而是证明在什么条件下 GNN 有用（低效率市场 vs. 高效率市场、低波动 vs. 高波动）。

**方向二（Selective Prediction）：** GNN only works sometimes，你的模型知道什么时候该相信它。把 selective prediction 作为主要 contribution。

**方向三（Systematic Ablation）：** 把图构建方式（correlation vs. sector vs. LLM-inferred）× 市场效率（S&P 500 vs. Russell 2000 vs. A股）× 市场 regime（低波动 vs. 高波动）做成三维实验矩阵。

---

## 6. 立即可执行的任务（Qwen 3.6 API）

### 6.1 任务一：批量生成结构化新闻特征（优先级最高）

**目的：** 建一个可用于 ablation 的结构化特征数据集。

**Prompt 设计：**

```
你是一个金融分析师。请分析以下新闻标题，输出 JSON 格式的结构化特征。
只输出 JSON，不要任何其他文字。

新闻: "{headline}"
日期: "{date}"

输出格式:
{
  "sentiment": "positive" | "negative" | "neutral",
  "affected_tickers": ["AAPL", "MSFT", ...],
  "affected_sectors": ["tech", "financials", ...],
  "sector_direction": {"tech": "positive", "financials": "negative", ...},
  "impact_magnitude": "high" | "medium" | "low",
  "impact_duration": "short_term" | "medium_term" | "long_term",
  "event_type": "earnings" | "monetary_policy" | "geopolitical" | "regulatory" | "M&A" | "market_sentiment" | "other"
}
```

**执行要点：**
- Temperature 设为 0 确保可复现
- 记录每条的 token cost 和 latency（论文报告 computational overhead 用）
- 预估工作量：若新闻 ~10 万条，按 Qwen 3.6 速度约 2-3 天

### 6.2 任务二：LLM-based Graph Construction

**目的：** 构建语义关系图，与 correlation graph 做对比实验。

**Prompt 设计：**

```
你是一个金融分析师。根据以下新闻，判断哪些 S&P 500 公司之间因为这条新闻产生了关联。
只输出 JSON。

新闻: "{headline}"
日期: "{date}"

输出格式:
{
  "edges": [
    {
      "source": "TICKER1",
      "target": "TICKER2",
      "relation": "supply_chain" | "competition" | "same_sector_impact" | "policy_impact" | "M&A" | "other",
      "sentiment_direction": "same" | "opposite"
    }
  ]
}

规则:
- 只输出你有高置信度的关系
- 如果新闻不涉及明确的公司间关系，输出 {"edges": []}
- Ticker 必须是 S&P 500 成分股的标准 ticker
```

### 6.3 任务三：Consistency / Reproducibility 测试

**目的：** 验证 LLM 输出的一致性，回应 reviewer 关于可复现性的质疑。

**方法：**
- 随机抽 500 条新闻
- 每条跑 5 次（temperature=0 跑一组，temperature=0.7 跑一组）
- 统计每个字段的 consistency rate
- 报告：sentiment 一致率、affected_tickers 的 Jaccard similarity、impact_magnitude 一致率

**论文中的呈现方式：** 在 Methodology section 加一段 "LLM Output Reliability Analysis"，报告 consistency metrics。

### 6.4 执行顺序与时间估算

```
Week 1: 任务一（结构化特征生成）+ 任务三（consistency 测试，可并行）
Week 2: 任务二（图构建）
Week 3: 将生成的数据接入 pipeline，开始跑 ablation table
Week 4: 分析结果，决定论文定位方向
```

---

## 7. 论文重新定位建议

### 7.1 降低 Claim 强度

**不要说：** "We propose a novel dynamic heterogeneous graph neural network with selective prediction that achieves superior performance on S&P 500."

**应该说：** "We systematically investigate the conditions under which graph-based models and LLM-encoded features provide incremental predictive value for stock direction prediction in highly efficient markets, and propose a selective prediction mechanism that identifies when such models should be trusted."

### 7.2 Contribution 重新排序

1. **Primary contribution：** Comprehensive ablation study revealing when and why GNN + LLM features do (or do not) improve stock direction prediction
2. **Secondary contribution：** Selective prediction mechanism with volatility calibration, evaluated with coverage-AUC trade-off analysis
3. **Tertiary contribution：** Comparison of LLM-based graph construction vs. statistical correlation graphs

### 7.3 如果 Ablation 结果是 Negative 的

这本身就是一个有价值的发表结果。参考：
- Tan et al. (NeurIPS 2024) "Are Language Models Actually Useful for Time Series Forecasting?" — 典型的 negative result paper，发在顶会
- 论文标题可以改为类似 "Do Graph Neural Networks and LLM Features Improve Stock Direction Prediction? Evidence from S&P 500"

---

## 8. 参考文献

### 顶会 Time Series + LLM

- Jin et al. Time-LLM: Time Series Forecasting by Reprogramming Large Language Models. ICLR 2024.
- Cao et al. TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting. ICLR 2024.
- Xu et al. FITS: Modeling Time Series with 10K Parameters. ICLR 2024.
- Jin et al. Position Paper: What Can Large Language Models Tell Us about Time Series Analysis. ICML 2024.
- Das et al. A Decoder-Only Foundation Model for Time-Series Forecasting (TimesFM). ICML 2024.
- Tan et al. Are Language Models Actually Useful for Time Series Forecasting? NeurIPS 2024.
- Gao et al. UniTS: Building a Unified Time Series Model. NeurIPS 2024.
- Shi et al. Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts. ICLR 2025.
- Yao et al. Towards Neural Scaling Laws for Time Series Foundation Models. ICLR 2025.
- Can LLMs Understand Time Series Anomalies? ICLR 2025.

### GNN + Stock Prediction

- Kim et al. HATS: A Hierarchical Graph Attention Network for Stock Movement Prediction. 2019.
- Xiang et al. Temporal and Heterogeneous Graph Neural Network for Financial Time Series Prediction (TH-GNN). CIKM 2022.
- Chen et al. ChatGPT Informed Graph Neural Network for Stock Movement Prediction. 2023.
- Qian et al. MDGNN: Multi-Relational Dynamic Graph Neural Network for Stock Investment Prediction. AAAI 2024.
- Patel et al. A Systematic Review on GNN-based Methods for Stock Market Forecasting. ACM Computing Surveys 2024.
- Zero-Shot Extraction of Stock Relationship Graphs with LLMs. EMNLP 2025 FinNLP Workshop.

### LLM + 金融结构化特征

- Fatouros et al. MarketSenseAI 2.0. 2025.
- Chiang et al. LLM Evaluation of Earnings Call Q&A. 2025.
- Wang et al. Alpha-GPT. 2023.
- Tang et al. AlphaAgent. EMNLP 2025 Findings.
- Kim et al. Financial Statement Analysis with Large Language Models. 2024.
- DNA Framework. ICAIF 2025.

### 市场效率与可预测性

- Lopez-Lira & Tang. Can ChatGPT Forecast Stock Price Movements? 2024.
- Fama. Efficient Capital Markets: A Review of Theory and Empirical Work. 1970.
- Complexity and ML Predictability of Stock Market Data. Entropy 2022.
- Machine Learning, Stock Market Forecasting, and Market Efficiency. IJDSA 2025.

### 时序基础模型批判

- How Foundational are Foundation Models for Time Series Forecasting? NeurIPS 2025 Workshop.
- Time Series Foundation Models: Benchmarking Challenges and Requirements. 2025.
- Against Time-Series Foundation Models. Shakoist Substack, 2025.
- It's TIME: Towards the Next Generation of Time Series Forecasting Benchmarks. 2026.

---

*文档生成日期：2026-04-07*
*基于与 Claude 的多轮讨论整理*
