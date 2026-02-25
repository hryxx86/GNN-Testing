# GNN + LLM 股票预测系统 — 设计文档

## Context

当前项目已完成 Phase 1（S&P 500 相关性网络分析）和 Phase 2（9 只 Hub 股票的新闻驱动预测）。GraphSAGE 仅比 Logistic Regression 基线提升 3.4% (Test AUC 0.64 vs 0.62)，原因是：
1. 异构图中**没有 stock-stock 边**，GNN 无法跨股票传播信号
2. 只有 9 只股票、480 条新闻，数据量太小
3. 使用 MiniLM (384维) 嵌入，不如现代 LLM 级模型

本次改造目标：**重构 Phase 2，将 LLM 嵌入 + 动态相关性图 + 异构 GNN 结合，扩展到 S&P 500 全规模，打造前沿 LLM+GNN 金融预测框架。**

核心创新叙事：现有研究要么用 LLM 但不用图（FinBERT-LSTM），要么用图但不用 LLM 嵌入（传统 GNN），要么用 LLM 推断图但不用真实市场数据（ChatGPT-GNN，仅 DOW 30）。本框架首次将 **LLM 级语义嵌入 + 真实价格相关性动态图 + 异构多关系结构** 结合在一起，在 S&P 500 规模上验证。

---

## Task 1: 动态图构建 & 灵敏度分析

### 目标
将 Phase 1 的静态 5 年相关性图改为滚动窗口动态图，分析网络拓扑演变规律，并将最优参数的动态图整合进 Phase 2 预测模型。

### 实现方案

#### 1.1 滚动窗口图构建
- **窗口大小**：3 个月 (63 天)、6 个月 (126 天)、12 个月 (252 天)
- **滑动步长**：1 个月 (21 个交易日)
- **阈值范围**：|corr| > {0.4, 0.5, 0.6, 0.7}
- **数据源**：现有 `sp500_5y_prices.csv` (502 股票 × 1255 天)
- **输出**：每个窗口×阈值组合生成一个 edge_index，存储为时序图序列

```python
# 伪代码：滚动窗口图序列生成
for window_size in [63, 126, 252]:
    for threshold in [0.4, 0.5, 0.6, 0.7]:
        for t in range(window_size, len(returns), 21):
            window_returns = returns[t-window_size:t]
            corr = window_returns.corr()
            edge_index = (corr.abs() > threshold).nonzero()
            graphs[(window_size, threshold, t)] = edge_index
```

#### 1.2 统计指标分析（主要分析方式）
每个窗口×阈值组合计算：
- 节点数、边数、密度
- 平均度、最大度
- 聚类系数
- 连通分量数
- Top 10 Hub 排名

汇总为：
- **Heatmap 矩阵**：行=阈值，列=窗口大小，值=平均密度/聚类系数
- **时序折线图**：Hub 排名随时间的变化（哪些股票的中心性在增加/减少）
- **行业组成演变图**：Top 10 Hub 的行业分布随时间的变化

#### 1.3 精选可视化（只对关键组合）
仅为以下 3 个组合生成完整网络图：
1. 密度最低的组合（最稀疏）
2. 密度最高的组合（最密集）
3. 预测性能最优的组合（由 Task 2 回馈确定）

每个组合生成 2 张图：Top 100 网络 + Hub 子图

#### 1.4 关键文件
- 修改：`GNN测试1 colab.ipynb` — 新增动态图构建 cells
- 新增：`build_dynamic_graphs.py` — 滚动窗口图构建脚本（可选，方便复用）
- 输出：`dynamic_graphs/` 目录，`sensitivity_analysis.csv`

---

## Task 2: LLM + GNN 新架构（Phase 2 重构）

### 整体架构

```
EODHD API → 新闻标题 ─→ FinBERT/LLM ─→ 768维嵌入 ──→ [news node]
                          └→ 情感分数 ──→ 辅助特征 ──→ [news node]

sp500_5y_prices.csv ─→ 滚动相关性 ──→ [stock↔stock 动态边]  ← 创新点1
                     └→ 技术指标 ────→ [stock node features]

sp500_sectors.csv ──→ [stock↔stock 行业边]                   ← 创新点2

         异构时序图 → GraphSAGE/GAT → 预测涨跌
```

### 2.1 数据获取：EODHD News API

**数据源**：EODHD Financial News API（学生价 $10/月）
**范围**：S&P 500 全部 ~500 只股票，2021-01-29 至 2026-01-28
**字段**：title, date, ticker, link, sentiment (EODHD 自带)
**预估数据量**：50 万 ~ 125 万条新闻

**下载脚本设计**：
```python
# 伪代码
import requests, time, pandas as pd

API_TOKEN = "your_token"
tickers = pd.read_csv("sp500_sectors.csv")["Symbol"].tolist()  # 复用现有文件

all_news = []
for ticker in tickers:
    offset = 0
    while True:
        url = f"https://eodhd.com/api/news?s={ticker}.US&from=2021-01-29&to=2026-01-28&offset={offset}&limit=50&api_token={API_TOKEN}&fmt=json"
        resp = requests.get(url).json()
        if not resp:
            break
        all_news.extend(resp)
        offset += 50
        time.sleep(0.5)  # 限速

pd.DataFrame(all_news).to_parquet("sp500_news_eodhd.parquet")
```

**输出文件**：`sp500_news_eodhd.parquet`

### 2.2 LLM 文本嵌入

**嵌入模型选择**：

| 模型 | 维度 | 金融适配 | 成本 | 推荐 |
|------|------|---------|------|------|
| `ProsusAI/finbert` | 768 | 专门金融预训练 | 免费本地 | **首选** |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | 通用但很强 | 免费本地 | 备选 |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 通用 | 免费本地 | 当前用，偏弱 |

**推荐：FinBERT**
- 专门在金融文本（10-K、财报、分析师报告）上预训练
- 768 维 vs MiniLM 的 384 维，信息容量翻倍
- 免费本地运行，无 API 成本
- 学术认可度高（金融 NLP 标准模型）

**处理流程**：
```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModel.from_pretrained("ProsusAI/finbert")

# 批量编码标题
def encode_batch(titles, batch_size=128):
    embeddings = []
    for i in range(0, len(titles), batch_size):
        batch = titles[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            emb = F.normalize(emb, dim=1)
        embeddings.append(emb.cpu().numpy())
    return np.vstack(embeddings)
```

**双特征策略**（创新点）：
- **主特征**：FinBERT 768 维嵌入（保留完整语义）
- **辅助特征**：EODHD 自带情感分数 + FinBERT 情感分类头输出（3 维：positive/negative/neutral）
- 最终 news node 特征维度：768 + 1 + 3 = **772 维**

**输出文件**：`sp500_news_emb.npy` (N × 768), `sp500_news_meta.parquet`

### 2.3 异构图构建（核心创新）

**节点类型**：
| 类型 | 数量 | 特征 |
|------|------|------|
| news | ~50万-125万 | 772维 (FinBERT嵌入 + 情感分数) |
| stock | ~500 | 动态特征（见下文） |

**边类型（3 种，创新点）**：
| 边类型 | 含义 | 来源 | 动态性 |
|--------|------|------|--------|
| news → stock (relates_to) | 新闻提及股票 | EODHD ticker 映射 | 静态 |
| stock ↔ stock (correlated_with) | 价格相关性 > 阈值 | **Task 1 滚动窗口图** | **动态** |
| stock ↔ stock (same_sector) | 同 GICS 行业 | sp500_sectors.csv | 静态 |

**Stock Node 特征**：
- GICS 行业 one-hot (11维)
- 过去 N 天收益率统计（均值、标准差、偏度）
- 对数市值（1维）
- 总维度：~16 维

**时间切分**：严格时间序列分割
- 训练：2021-01 至 2024-12（~80%）
- 验证：2025-01 至 2025-06（~10%）
- 测试：2025-07 至 2026-01（~10%）

**关键决策：动态图的时间对齐**
对于每条新闻事件（日期 t），使用截至日期 t 的最近滚动窗口相关性图作为 stock-stock 边。这确保了没有未来数据泄漏。

### 2.4 GNN 模型

**架构：Heterogeneous GraphSAGE (或 GAT)**

```python
class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden=64, heads=4):
        super().__init__()
        # 两层异构 SAGE
        self.conv1 = SAGEConv((-1, -1), hidden)
        self.conv2 = SAGEConv((-1, -1), hidden)
        # 分类头
        self.lin = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x_dict, edge_index_dict):
        x = self.conv1(x_dict, edge_index_dict).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index_dict)
        # 只对 news 节点分类
        out = self.lin(x['news'])
        return out

# 转为异构模型
model = to_hetero(HeteroGNN(), data.metadata(), aggr='mean')
```

**信息流（3 种边的作用）**：
```
第1层：
  news_A(关于AAPL) → relates_to → AAPL：AAPL 节点聚合所有相关新闻
  AAPL ↔ MSFT (correlated_with)：AAPL 和 MSFT 交换信息
  AAPL ↔ GOOG (same_sector)：同行业股票交换信息

第2层：
  AAPL → rev_relates_to → news_A：news_A 间接获取 MSFT、GOOG 的新闻信号
  结果：news_A 的预测不仅基于自身内容，还融合了相关股票的新闻和市场状态
```

**训练配置**：
- 优化器：Adam, lr=0.001, weight_decay=1e-4
- 损失：BCEWithLogitsLoss
- Dropout: 0.3
- Early stopping: patience=20, 监控 validation AUC
- 最大 epochs: 300

### 2.5 基线 & 消融实验

| 实验 | 描述 | 目的 |
|------|------|------|
| Baseline 1 | Logistic Regression + FinBERT 嵌入 | 纯文本基线 |
| Baseline 2 | Logistic Regression + EODHD 情感分数 | 传统情感基线 |
| Baseline 3 | FinBERT-LSTM (无图) | LLM+时序基线，对标现有论文 |
| **Ablation 1** | GNN 只有 news→stock 边（无 stock-stock） | 验证 stock-stock 边的价值 |
| **Ablation 2** | GNN 有 stock-stock 边但用 MiniLM 嵌入 | 验证 FinBERT vs MiniLM 的差异 |
| **Ablation 3** | GNN 有 correlated_with 边但无 same_sector 边 | 验证行业边的增量价值 |
| **Full Model** | 完整模型（FinBERT + 3种边 + 动态图） | 最终结果 |

### 2.6 评估指标
- AUC-ROC（主指标）
- Accuracy, Precision, Recall, F1
- 分行业 AUC（看哪些行业预测更准）
- 回测：基于预测构建简单多空组合，计算年化收益、Sharpe Ratio

---

## 实施顺序

### Phase A: 数据准备（先做，不依赖其他步骤）
1. EODHD 免费版测试数据质量
2. EODHD 学生版批量下载 500 股新闻 → `sp500_news_eodhd.parquet`
3. FinBERT 批量嵌入新闻标题 → `sp500_news_emb.npy`

### Phase B: Task 1 动态图（可与 Phase A 并行）
4. 实现滚动窗口图构建函数
5. 计算统计指标矩阵，生成 heatmap
6. 生成 Hub 排名时序图 + 精选网络可视化

### Phase C: Task 2 新模型
7. 构建异构图（news + stock 节点，3 种边类型）
8. 实现基线模型（LR, FinBERT-LSTM）
9. 实现 HeteroGNN 模型
10. 运行消融实验
11. 结果分析 + 可视化

---

## 关键文件清单

| 文件 | 用途 | 状态 |
|------|------|------|
| `GNN测试1 colab.ipynb` | 主 notebook | 修改 |
| `sp500_5y_prices.csv` | 价格数据 | 已有，复用 |
| `sp500_sectors.csv` | 行业数据 | 已有，复用 |
| `sp500_market_caps.csv` | 市值数据 | 已有，复用 |
| `download_news.py` | EODHD 新闻下载脚本 | 新建 |
| `embed_news.py` | FinBERT 嵌入脚本 | 新建 |
| `build_dynamic_graphs.py` | 滚动窗口图构建 | 新建 |
| `sp500_news_eodhd.parquet` | 下载的新闻数据 | 新建 |
| `sp500_news_emb.npy` | FinBERT 嵌入 | 新建 |
| `dynamic_graphs/` | 动态图序列 | 新建 |

---

## 验证方案

1. **数据验证**：检查新闻覆盖率（每只股票是否有足够新闻）、时间分布、缺失值
2. **嵌入验证**：对 FinBERT 嵌入做 t-SNE，检查是否按行业/情感聚类
3. **图结构验证**：检查异构图的连通性、degree 分布
4. **模型验证**：消融实验表格，确认每个组件都有正向贡献
5. **端到端验证**：在 Colab GPU 上完整运行 notebook，确认可复现
