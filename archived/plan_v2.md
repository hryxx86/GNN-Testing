# DynHetGNN-SP: 研究方向 & 实施计划 v2

> **Updated: 2026-02-28** — 整合审稿反馈后的修订版

## Context

S&P 500 股票方向预测 (event-level)。Phase C v1 全量预测 AUC ≈ 0.50，与随机持平。

**核心假设**: 问题不是模型太弱，而是大部分事件本质不可预测；优秀的模型应该像优秀交易员一样，只在有把握时出手——且在高波动期更加保守。

**数据**: 2021-01 → 2026-01 (5年, 502股, 1.7M事件)。Limitations: 不覆盖2008/2020系统性危机，2022熊市(-19%)作为stress test。

---

## 前沿论文定位

### A. 金融 GNN SOTA

| 论文 | 会议 | 做了什么 | 我们超越的点 |
|------|------|---------|-------------|
| THGNN | CIKM'22 | 每日动态图 + Transformer + HeteroGAT | 无NLP嵌入；无新闻节点；无selective prediction |
| DGRCL | ICAART'25 Best Paper | 动态图 + 对比学习, 53.06% acc NASDAQ | 仅价格特征；无新闻文本；无异构多边 |
| ChatGPT-GNN | KDD-W'23 | GPT推断图 → GNN + LSTM | 图是LLM猜的；静态图；仅DOW 30 |
| DASF-Net | 2025 | FinBERT情感 + 热核扩散图, 12 stocks | 回归非分类；仅12股；无动态图 |

### B. Selective Prediction SOTA

| 论文 | 会议 | 做了什么 | 与我们的关系 |
|------|------|---------|-------------|
| Geifman & El-Yaniv | NeurIPS'17 | Confidence threshold + risk-coverage理论 | 方法论基础 (F.1) |
| SelectiveNet | ICML'19 | 3-head: prediction + selection + auxiliary | 核心参考 (F.2)，我们在此基础上引入market context |
| AUGRC | NeurIPS'24 | 修复AURC指标缺陷 | 评估指标 |
| Crypto Confidence-Threshold | Applied Sci'25 | 加密货币置信度阈值, τ=0.8, coverage 12% | 唯一金融selective，但非GNN |
| Feng et al. | arXiv'25 | 双拒绝: ambiguity + novelty | 时间序列rejection参考 |

### C. 核心 Gap

Selective prediction在通用ML中已成熟 (NeurIPS/ICML级别)，但从未被引入金融GNN。唯一的金融应用是一篇crypto简单阈值论文 (非GNN)。**更重要的是，现有selective prediction方法假设数据分布平稳，未考虑金融市场的regime-dependent特性。**

---

## 选定方向：动态异构图 + Volatility-Calibrated Selective Prediction + LLM

### 论文叙事

> "现有金融GNN用静态图和全量预测，忽略了两个关键事实：(1) 股票关系随时间演化，(2) 大部分新闻事件对股价无影响。我们提出 **DynHetGNN-SP** — 一个动态异构图神经网络 + volatility-calibrated selective prediction 框架。不同于标准SelectiveNet假设数据平稳分布，我们的selection head融入市场regime信号，使模型在高波动期自动收缩覆盖范围——如同优秀交易员在崩盘时减少操作。在 S&P 500 全规模 (502股, 1.7M事件) 上，全量预测 AUC ≈ 0.50，但 selective 子集上 AUC 显著提升，且该提升跨市场环境保持稳定。"

### 论文贡献

1. **Volatility-Calibrated Selection Head**: 首次将selective prediction引入金融GNN，并提出regime-aware的selection机制（不仅是搬运SelectiveNet，而是引入金融归纳偏置）
2. **动态异构图**: 滚动窗口相关性 + 行业 + 新闻→股票，图结构逐月演化
3. **严格的诊断与消融体系**: 非GNN baseline矩阵 + selection vs confidence ranking Jaccard分析 + 动态图必要性审计 + walk-forward validation
4. **LLM ablation**: FinBERT vs GPT-5-mini 多维结构化输出
5. **大规模验证**: S&P 500, 502股, 1.7M事件 + economic significance分析

---

## 实施计划

### Phase 0: Return定义 & 数据基础 (在一切实验之前锁定)

**Return计算公式**:
```
r_i = (P_{t+1, close} - P_{t, close}) / P_{t, close}
label_i = 1 if r_i > 0 else 0
```

**Timing Diagram**:
```
事件发布 ──────────────────────────────────── 标签计算
   │                                              │
   ▼                                              ▼
Day T (任意时间)                           Day T+1 close

可用特征 (截止时间):                      预测目标:
├─ 新闻文本: T 时刻                       T+1 close vs T close
├─ FinBERT/LLM 情感: T 时刻
├─ Market context: T-1 close (严格!)
│   ├─ VIX: T-1 close
│   ├─ SPY drawdown: T-1 close
│   ├─ Realized vol 30d: 截止 T-1
│   └─ Market breadth: T-1 close
├─ Stock features: T-1 close
│   ├─ Rolling momentum 5/10/21d
│   └─ Rolling volatility 5/10/21d
└─ Graph edges: 截止 T-1 的历史数据
```

**盘后新闻处理规则**:
- 盘前/盘中新闻 (4:00-16:00 ET): P_ref = 当日 close, P_target = 次日 close
- 盘后新闻 (16:00-次日4:00 ET): P_ref = 次交易日 open, P_target = 次交易日 close
- 周末/节假日新闻: 映射到下一个交易日

**注意**: Return定义一旦锁定不可更改。如果后续需要测试不同horizon (2d, 3d)，作为sensitivity analysis单独报告。

---

### Phase D+: 增强诊断 — 建立信号水位线

**目标**: 在做任何GNN改进之前，确认 (a) 信号在哪里，(b) 非GNN方法能到什么水平，(c) LLM特征的上限。

**D.1: 标签噪声分析** (已有cells)
- 标签噪声占比: |return| < 0.5% 的事件比例
- 按 |return| 分桶的 AUC

**D.2: 非GNN Baseline 矩阵** [NEW]

| Baseline | 特征 | 目的 |
|----------|------|------|
| Random | — | 基准线 (AUC=0.50) |
| Rule-based | FinBERT sentiment > 0 → 涨 | 验证情感信号是否存在 |
| Logistic Regression | FinBERT 3-dim sentiment | 验证线性可分性 |
| XGBoost | FinBERT sentiment + price momentum (5/10/21d) | 验证非线性特征组合 |
| Phase C GNN v1 | 全pipeline | 当前水平 |

每个baseline报告: 全量AUC + Selective Top-10% AUC (按 |predicted_prob - 0.5| 排序)

**关键判断逻辑**:
- 如果 LR 全量 ≈ 0.50 且 top-10% ≈ 0.50 → 信号极弱，需先修特征
- 如果 LR 全量 ≈ 0.50 但 top-10% ≈ 0.52+ → 信号存在但稀疏，selective路线正确
- 如果 XGBoost > GNN → 图结构目前无增益，需先修图
- 如果 GNN top-10% > XGBoost top-10% → 图结构在高信号区域有价值

**D.3: LLM 小样本验证** [NEW — 原Phase L.2提前]

**重要**: 不得在最终test set上做方法选择。LLM验证使用训练期内的dev-holdout。

数据划分:
- Train: 2021-01 → 2023-09
- Dev-holdout (用于D.3 LLM验证): 2023-10 → 2023-12 (~3个月, ~7万条)
- Fold 1 test: 2024全年 (不触碰)
- Fold 2 test: 2025全年 (lockbox, 绝不触碰直到最终报告)

在 dev-holdout 的 10% 随机样本 (~7千条) 上:
- GPT-5-mini 多维结构化输出:
  ```json
  {
    "impact_level": "high/medium/low",
    "direction": "positive/negative/neutral",
    "confidence": 0.85,
    "reasoning_type": "earnings/macro/sentiment/technical"
  }
  ```
- 将4个字段编码为特征向量 (one-hot + confidence scalar)
- 对比: LLM features vs FinBERT features → LR/XGBoost 的 AUC 差异
- 预估成本: ~$3-5 (GPT-5-mini on 30K samples)

**D.4: 按条件分桶诊断**
- 按情感置信度分桶: high/medium/low → 各桶AUC
- 按行业分桶: 11 GICS sectors → 各行业AUC
- 按 |return| 幅度分桶: <0.5%, 0.5-1%, 1-2%, >2% → AUC
- 按市场regime分桶: VIX < 15 / 15-25 / > 25 → AUC

**产出**: `docs/phase_d_diagnosis.md` 包含所有诊断图表和结论

**D.0: 数据预处理 — 新闻去重** [NEW]

财经新闻存在大量转载/改写，如果不处理会夸大样本量并放大伪显著性:
- 同日同ticker的新闻做聚合: FinBERT embedding取mean，保留sentiment最强的headline
- 去重前后报告样本数变化 (预计1.7M → ~250K-500K stock-day级别)
- 这与G.4的3天窗口聚合方向一致，可在此阶段提前实施简化版 (1天窗口)

**D.5: Go/Pivot/Stop 判定标准** [NEW]

在D.1-D.4全部完成后，根据以下标准决定项目走向:

| 判定 | 条件 | 行动 |
|------|------|------|
| **Go** | 任一baseline在任一条件分桶下 AUC > 0.52 | 按计划推进 E→F |
| **Pivot** | 全量AUC ≈ 0.50，但 \|return\| > 2% 子集 AUC > 0.54 | 缩小研究范围到 high-impact events |
| **Stop** | 所有条件下所有baseline ≈ 0.50 | 重新定义问题（换target/换数据源/转向negative result论文） |

**注意**: Negative result也有发表价值 — 如果能严谨证明"在S&P 500规模上，即使用GNN+LLM+selective prediction，event-level return也不可预测"，这是对EMH的有力evidence。

---

### Phase E: 动态异构图 (Rolling Window HeteroGraph)

**已有基础**: Phase B 的 636 个 correlation 快照 (63/126/252天窗口, 0.4-0.7阈值)

**E.0: 动态图必要性审计** [NEW]

在动手改架构之前，先验证动态图是否有意义:
- 计算相邻月份边集 Jaccard similarity: J = |E_t ∩ E_{t-1}| / |E_t ∪ E_{t-1}|
- 如果 J > 0.9 (126天窗口) → 窗口太长，尝试63天窗口
- 重点可视化: 2022 Q1-Q2 (加息周期) Tech sector内部correlation边的变化
- 产出: 一张 Jaccard 随时间变化的折线图 + regime标注

**E.1: Shared weights, sequential snapshots** (先做)

| 参数 | 值 | 理由 |
|------|------|------|
| 时间粒度 | 月 (calendar month) | ~60 snapshots |
| 相关性窗口 | 126 trading days (或根据E.0调整) | Phase B sweet spot |
| 相关性阈值 | 0.6 | 保持可比性 |
| 行业边 | 静态 | GICS短期不变 |
| news→stock边 | 按月分配 | 每月只含该月事件 |
| stock节点特征 | 动态: rolling stats | 5/10/21日 return mean/std + 动量 |

所有月份共享同一组GNN权重，按时间顺序处理。

**E.2: EvolveGCN-style** (如果E.1有效再尝试)
- GNN权重通过GRU随时间演化

---

### Phase F: Volatility-Calibrated Selective Prediction — 核心创新

**方法层级 (逐级验证):**

**F.0: Random Selection Baseline** [NEW]
- 随机选X%事件的AUC → 证明selective不是simply picking easy samples

**F.1: Confidence Threshold** [Geifman, NeurIPS'17]
- GNN logit → sigmoid → p, confidence = |p - 0.5|
- 只保留 confidence > τ 的事件
- 零额外参数，用于快速验证selective有效

**F.2: Volatility-Calibrated SelectiveNet** [核心创新]

标准SelectiveNet: `selection = g(h)` 其中 h 是 GNN hidden state

**我们的改进**: `selection = g(h, m)` 其中 m 是 Market Context Vector

Market Context Vector m 包含 (全部严格使用 **T-1 close**, 无例外):
- VIX T-1 close
- SPY drawdown from 52-week high (截止T-1)
- 30日 realized volatility (截止T-1)
- Market breadth: 涨跌比例 (T-1)

**禁止使用T-0 (当日) 值**: 当日VIX与当日return存在同步性，使用T-0会引入前瞻偏差。
**Ablation**: 做T-0 vs T-1对比实验，如果T-0显著优于T-1，本身就是leak的证据。

**Selection Head 架构**:
```
h (GNN hidden) ──┐
                  ├── concat → MLP → sigmoid → select ∈ [0,1]
m (market ctx) ──┘
```

**Loss function**:
```
L = (1/n) Σ [select_i × CE(pred_i, y_i)] + λ × max(0, target_coverage - mean(select))²
```

**Training Protocol** [NEW — 回应训练目标冲突风险]:

端到端训练时，prediction head有动机在被reject的样本上"摆烂"，人为膨胀selected subset的AUC。因此:

- **Primary specification: 分阶段训练**
  1. Stage 1: 训练完整GNN (prediction head + auxiliary head)，无selection
  2. Stage 2: 冻结GNN backbone + prediction head，只训练selection head
  3. 这确保selection head是在固定的prediction质量上做选择，而非反向影响prediction

- **Secondary specification: 端到端联合训练**
  - 同时训练所有heads
  - 必须额外报告: rejected子集上prediction head的AUC (有/无selection head对比)
  - 如果端到端AUC显著高于分阶段 → 警惕训练动力学偏差，需在论文中讨论

**Per-month coverage constraint** [NEW]:
- 强制每月至少覆盖 min_coverage (5-10%) 的事件
- 防止模型在2022熊市全部abstain
- 报告每月coverage的方差作为稳定性指标

**F.3: Selection分析** [NEW]
- F.1 vs F.2 在相同coverage下的AUC差异
- F.2选出的集合 vs F.1选出的集合 Jaccard similarity
- 如果Jaccard低但F.2 AUC更高 → 证明selection head学到了非trivial的特征
- F.2选中事件的经济特征分析: 行业分布、市场环境分布、平均|return|
- **ECE (Expected Calibration Error)** plot + calibration curve

**评估指标**:
- AUGRC (NeurIPS'24)
- Selective AUC@coverage=X% (X = 5, 10, 20, 50)
- Coverage@AUC=0.55
- Precision@top-K
- Risk-Coverage curve
- ECE [NEW]
- Per-month coverage variance [NEW]

---

### Phase G: 架构改进 + 特征增强

**G.1: SAGEConv → HGT (Heterogeneous Graph Transformer)** [UPDATED: 原GAT改为HGT]
- HGT 为异构图设计，学习不同关系类型的独立attention空间
- stock↔stock (correlation)、stock↔stock (sector)、news→stock 三种元路径各有独立attention
- PyG 有 `HGTConv` 实现
- 时机: 在 E+F 验证有效之后

**G.2: 市场调整标签**
- `stock_return - SPY_return > 0` (去大盘beta)

**G.3: 丰富 stock 特征**
- 5/10/21日 rolling return mean/std + 动量

**G.4: 3天窗口聚合 (借鉴DASF-Net)**
- event-level (1.7M) → stock-day-level (~250K)
- 同一股票3天内新闻的embedding做mean/max pooling

**G.5: Momentum Spillover 边** [NEW]
- 计算过去1个月A股return对B股后续return的预测力
- 作为第四种边类型加入异构图
- 与Pearson correlation捕捉的信息正交 (lead-lag vs 同步)
- 作为ablation验证

---

### Phase L: LLM 分析 (消融实验)

**注**: L.2已在D.3中提前做小样本验证。此处是全量版本。

**L.1: LLM Event Filter**
- GPT-5-mini: "这条新闻对{ticker}股价有影响吗？high/medium/low"
- 只保留high-impact → 对比FinBERT置信度filter

**L.2: LLM 多维结构化输出** [UPDATED]
- 4字段结构化判断 (impact_level, direction, confidence, reasoning_type)
- 替代FinBERT的3-dim sentiment作为node feature
- 在test set或高信号子集上运行

**L.3: LLM 直接推理** (如果预算允许)
- GPT-5: 给定新闻+近期表现，直接预测涨跌概率
- 与GNN pipeline对比

**叙事准备** [NEW — 两个版本]:
- (a) GNN+selective > LLM-only → 图结构有独立价值
- (b) LLM特征更强但GNN+selective进一步提升 → "1+1>2"互补叙事

---

### Phase H: 论文 Figures + 最终分析

**Figures**:
- Fig 1: 动态图 vs 静态图 AUC + 图结构Jaccard随时间变化 [UPDATED]
- Fig 2: Coverage-AUC tradeoff曲线 (核心图) — 含F.0/F.1/F.2对比
- Fig 3: Diagnostic heatmap (sector × confidence → AUC)
- Fig 4: ECE calibration plot [NEW]
- Fig 5: FinBERT vs LLM 多维输出对比
- Fig 6: Per-month coverage稳定性 + VIX overlay [NEW]

**Tables**:
- Tab 1: Non-GNN baseline矩阵 (全量 vs selective) [NEW]
- Tab 2: 完整消融表 (静态vs动态 × 全量vs selective × FinBERT vs LLM)
- Tab 3: Selective prediction详细结果 (各coverage下的AUC + Jaccard分析)
- Tab 4: Walk-forward validation结果 (Fold 1 + Fold 2 的 mean±std) [NEW]
- Tab 5: Economic significance — break-even transaction cost [NEW]

**Economic Significance** [NEW]:
- Selected事件的平均|return|
- 按预测方向交易的模拟return (扣除单边10bps)
- Break-even transaction cost计算
- 不做完整backtest，但提供实际可行性的evidence

---

## Walk-Forward Validation [NEW]

**方案A (Expanding Window)**:
- Fold 1: Train 2021-01 → 2023-12, Test 2024-01 → 2024-12
- Fold 2: Train 2021-01 → 2024-12, Test 2025-01 → 2025-12

Fold 2包含Fold 1所有训练数据 + 2024全年，能学到更完整的市场周期。

所有指标报告: mean ± std across 2 folds。

**Permutation Test** [NEW]:
- 打乱股票代码 (保持时间轴不变)
- 如果selective prediction还能"提升"AUC → 说明学到了假信号
- 打乱标签的permutation test作为额外验证

---

## 执行顺序 & 优先级

```
D+ (诊断+baselines+LLM小样本) → E (动态图+Jaccard审计) → F (Volatility-Calibrated SP)
         信号水位线                  图结构验证                  核心创新
→ G (HGT+特征增强) → L (LLM全量) → H (论文figures+经济分析)
    架构榨取            消融实验         最终产出
```

**优先级逻辑**: 先确认信号存在且GNN有价值 (D+)，验证动态图必要性 (E)，再做核心方法创新 (F)，最后架构升级和ablation (G/L)。避免在错误的基础上建大厦。

---

## 关键决定记录

| 日期 | 决定 | 理由 |
|------|------|------|
| 2026-02-27 | Phase B 动态图直接复用 | 636个correlation快照已生成 |
| 2026-02-27 | Selective Prediction 主攻 SelectiveNet | 成熟方法论，首次引入金融GNN即为创新 |
| 2026-02-27 | 评估用 AUGRC (NeurIPS'24) | 学术界最新标准 |
| 2026-02-28 | 引入 Volatility-Calibrated Selection Head | 解决novelty不足问题：从pure application → methodological innovation |
| 2026-02-28 | 加入非GNN baseline矩阵 | 回应"AUC=0.50是数据问题还是模型问题"的审稿质疑 |
| 2026-02-28 | L.2提前到D阶段做小样本验证 | 排除LLM undermine GNN贡献的existential risk |
| 2026-02-28 | Walk-forward validation (方案A, expanding window) | 金融数据non-stationarity要求多fold验证 |
| 2026-02-28 | GAT → HGT (Phase G) | 更match异构图设定，PyG有现成实现 |
| 2026-02-28 | 加ECE + per-month coverage stability | 增强selective prediction可信度 |
| 2026-02-28 | 加economic significance (break-even cost) | 跨越统计显著到经济可行 |
| 2026-02-28 | LLM输出改为多维结构化 (4字段) | 比单一sentiment更丰富，且API可实现 |
| 2026-02-28 | 加momentum spillover边 (Phase G ablation) | 与Pearson正交，捕捉lead-lag效应 |
| 2026-02-28 | Return定义锁定: next-day close-to-close | 地基问题，必须在实验前固定 |
| 2026-02-28 | Market context一律T-1 close | 防止前瞻偏差，T-0 vs T-1做ablation验证 |
| 2026-02-28 | D.3 LLM验证改用dev-holdout | 避免test set泄漏（原方案在test set 10%上做方法选择） |
| 2026-02-28 | Selection Head分阶段训练为primary spec | 防止prediction head "摆烂"导致selected AUC虚高 |
| 2026-02-28 | 新闻同日同ticker聚合 | 去重防止伪显著性，与G.4方向一致 |
| 2026-02-28 | 加Go/Pivot/Stop criteria | 设硬性决策点，避免沉没成本陷阱 |
