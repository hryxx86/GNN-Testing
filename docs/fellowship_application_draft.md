# USC Provost's Undergraduate Research Fellowship — Project Description

## Predicting Stock Rankings with Graph Neural Networks: How Network Structure Creates Horizon-Dependent Forecasting Power

**Faculty Supervisor:** Prof. Jinchi Lv, Department of Data Sciences and Operations, Marshall School of Business

### Motivation

Stocks do not move in isolation. Price correlations, shared industry membership, and co-mentions in financial news create a network of interdependencies among equities. Graph Neural Networks (GNNs) can propagate information through such networks, potentially capturing cross-stock signals invisible to models that treat each stock independently. However, the existing literature lacks systematic evidence on *when* — at what prediction horizon — this graph structure actually helps. This project addresses that gap.

### Research Design

I constructed a full-scale prediction pipeline covering all 501 S&P 500 constituents over 1,255 trading days (2021–2026). The data infrastructure combines two sources: daily price data used to compute momentum and volatility features, and 1.7 million financial news articles encoded into 768-dimensional embeddings using FinBERT, a language model specialized for financial text. A dynamic correlation graph is rebuilt monthly using rolling 126-day return correlations (threshold 0.6), producing 54 temporal snapshots that capture evolving market regimes — for instance, correlation density surges during the 2022 bear market as stocks move in lockstep.

The prediction task is cross-sectional stock ranking: each day, the model scores all 500 stocks, and performance is measured by the Information Coefficient (IC) — the rank correlation between predicted and realized returns. I systematically compare a Graph Attention Network (GAT) against LightGBM (a non-graph baseline) across six prediction horizons: 1, 5, 10, 21, 42, and 63 trading days.

### Key Findings

The central result is a striking horizon-dependent cross-pattern between graph-based and flat-feature models. The GAT model's IC traces an inverted-U curve, peaking at the 21-day horizon (IC = 0.044, annualized net return 15.1% after transaction costs) and producing negative IC at both 1-day and 42+ day horizons. LightGBM's IC increases monotonically, peaking at 63 days. At the 10–21 day sweet spot, GAT outperforms LightGBM by 2.8–2.9 times, confirming that graph-propagated cross-stock information provides genuine, non-redundant predictive signal — but only within a specific temporal window. This finding has not been reported in the existing financial GNN literature.

Additional findings include: (1) simpler GNN architectures (GAT) outperform complex heterogeneous graph transformers in the high-noise financial regime; (2) news co-occurrence edges are harmful, degrading IC by 63%; and (3) SelectiveNet, a learned abstention mechanism with 800+ citations, fails completely when applied to financial GNNs — a novel negative result.

### Remaining Work and Timeline

Current results are preliminary: GNN training exhibits high variance across random seeds, and the observed IC advantages have not yet reached consistent statistical significance. The immediate priority is walk-forward cross-validation with multiple random seeds to determine whether the horizon-dependent pattern is a robust phenomenon or an artifact of particular training runs. If the pattern holds, I will conduct additional ablation studies (e.g., graph sparsity thresholds, alternative node feature sets) to strengthen the analysis. The fellowship period would support this rigorous validation phase, after which I plan to compile the findings into a research paper.

---

*Word count: ~400*

**Suggested figures to attach:**
1. S&P 500 Correlation Network (Top 100 by Market Cap) — visualizes the stock network structure
2. Horizon Ablation 3-Panel Plot (IC / ICIR / Sharpe vs. Prediction Horizon) — the core finding showing GAT's inverted-U vs. LightGBM's monotonic increase
