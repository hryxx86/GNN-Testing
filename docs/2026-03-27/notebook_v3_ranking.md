# Notebook Documentation: v3_ranking_pipeline.ipynb -- v3 Cross-Sectional Ranking Pipeline (N1-N5)

**Notebook**: `v3_ranking_pipeline.ipynb`
**Project**: DynHetGNN-SP -- Dynamic Heterogeneous Graph Neural Network with Selective Prediction for Stock Ranking
**Run Environment**: Google Colab (NVIDIA A100-SXM4-40GB or RTX PRO 6000 Blackwell)
**Data**: 502 S&P 500 stocks, 2020-01 to 2025-12 (~1255 trading days), 1.7M news events

---

## Table of Contents

1. [Overview and Motivation](#1-overview-and-motivation)
2. [Cell 0: Markdown Header](#2-cell-0-markdown-header)
3. [Cell 1: Setup and Environment](#3-cell-1-setup-and-environment)
4. [Cell 2: Parameters](#4-cell-2-parameters)
5. [Cell 3: Raw Data Loading](#5-cell-3-raw-data-loading)
6. [Cell 4 -- N1a: Price Features](#6-cell-4--n1a-price-features)
7. [Cell 5 -- N1b: News Features](#7-cell-5--n1b-news-features)
8. [Cell 6 -- N1c: Multi-Horizon Labels](#8-cell-6--n1c-multi-horizon-labels)
9. [Cell 7 -- N1d: Time Split](#9-cell-7--n1d-time-split)
10. [Cell 8 -- N2: Graph Construction](#10-cell-8--n2-graph-construction)
11. [Cell 9: HeteroData Builder](#11-cell-9-heterodata-builder)
12. [Cell 10: Evaluation Utilities](#12-cell-10-evaluation-utilities)
13. [Cell 11 -- N3a: Non-GNN Baselines](#13-cell-11--n3a-non-gnn-baselines)
14. [Cell 12 -- N3b: GNN Model Definitions](#14-cell-12--n3b-gnn-model-definitions)
15. [Cell 13 -- N3c: GNN Training Loop](#15-cell-13--n3c-gnn-training-loop)
16. [Cell 14 -- N3d: GNN Ablation and Go/Stop Gate](#16-cell-14--n3d-gnn-ablation-and-gostop-gate)
17. [Cell 15 -- N4: Horizon Ablation](#17-cell-15--n4-horizon-ablation)
18. [Cell 16 -- N5a: SelectiveNet Model Definition](#18-cell-16--n5a-selectivenet-model-definition)
19. [Cell 17 -- N5b: SelectiveNet Training](#19-cell-17--n5b-selectivenet-training)
20. [Cell 18 -- N5c: Selection Analysis and Visualization](#20-cell-18--n5c-selection-analysis-and-visualization)
21. [Cell 19: Observations Markdown](#21-cell-19-observations-markdown)
22. [Result Tables](#22-result-tables)
23. [Key Findings](#23-key-findings)
24. [Training Stability Analysis](#24-training-stability-analysis)

---

## 1. Overview and Motivation

This notebook implements the **v3 pipeline**, a complete pivot from the prior v2 approach. The v2 pipeline (Phases A-C) attempted binary direction prediction (up/down) using event-driven GNNs, but all models produced AUC ~ 0.50 (equivalent to random guessing) on S&P 500 stocks. The root cause was identified as a combination of Efficient Market Hypothesis (EMH) effects and a fundamentally wrong task formulation.

The v3 pivot was driven by a literature review of 10+ recent papers (MASTER AAAI'24, FinMamba arXiv'25, MDGNN AAAI'24, THGNN CIKM'22, etc.), which revealed that all state-of-the-art financial GNN papers use **ranking** rather than binary classification.

### Key changes from v2 to v3

| Dimension | v2 (Failed) | v3 (This Notebook) |
|-----------|-------------|---------------------|
| Prediction task | Binary direction (up/down) | **Cross-sectional ranking** |
| Evaluation metrics | AUC | **IC / ICIR / Sharpe ratio** |
| Data paradigm | Event-driven (predict only when news exists) | **Calendar-driven (predict all stocks every day)** |
| Label definition | sign(return) | **Z-score normalized cross-sectional return** |

### Intended contribution points

1. **Horizon ablation**: Systematic comparison of 1d/5d/10d/21d/42d/63d prediction horizons -- not done in prior literature
2. **GNN + SelectiveNet**: First application of SelectiveNet to financial GNN ranking
3. **Dynamic heterogeneous graph**: 4 edge types (correlation, sector, news mentions, co-occurrence)

### Notebook structure

The notebook contains 20 cells organized into sections N0-N5:

```
N0:  Setup + Parameters (Cells 0-2)
N1:  Data pipeline (Cells 3-7) -- price features, news features, labels, time split
N2:  Graph construction (Cells 8-9) -- 4 edge types, HeteroData builder
N3:  Baselines + GNN ablation (Cells 10-14) -- evaluation, baselines, models, training, ablation
N4:  Horizon ablation (Cell 15) -- 6 horizons with GAT and LightGBM
N5:  SelectiveNet (Cells 16-18) -- model, training, analysis
     Observations (Cell 19)
```

---

## 2. Cell 0: Markdown Header

**Type**: Markdown

This cell states the research question, hypothesis, and contribution points. It frames the notebook as investigating whether a Dynamic Heterogeneous Graph Transformer with selective prediction can outperform flat baselines on S&P 500 cross-sectional stock ranking. It also notes the predecessor (v2 event-driven binary approach) and its failure.

---

## 3. Cell 1: Setup and Environment

**Type**: Code

**Purpose**: Import libraries, set random seeds for reproducibility, detect Colab vs local environment, mount Google Drive if on Colab, install required packages (torch_geometric, lightgbm).

**Key details**:

- **Reproducibility**: Sets `SEED = 42` across Python random, NumPy, PyTorch (CPU and CUDA), and sets `torch.backends.cudnn.deterministic = True`. Note: despite these settings, CUDA operations on Colab GPUs introduce inherent non-determinism (see Section 24).
- **Environment detection**: Tries `import google.colab`; if successful, mounts Drive and changes working directory to `GNN测试`. Otherwise uses local path.
- **Directory setup**: Creates `data/fullscale`, `data/reference`, `plots`, `experiments` directories.
- **Device selection**: Uses CUDA if available, prints GPU name and VRAM.

**Libraries used**:
- PyTorch: deep learning framework
- PyTorch Geometric (PyG): graph neural network library for PyTorch
- NumPy / Pandas: numerical computation and data manipulation
- SciPy: scientific computing (used for `spearmanr` correlation and sparse matrices)
- scikit-learn: machine learning baselines (Ridge regression)
- XGBoost: Chen & Guestrin (KDD 2016) gradient boosting framework
- LightGBM (Microsoft): gradient boosting framework using histogram-based algorithms for efficient tree learning
- Matplotlib: visualization

---

## 4. Cell 2: Parameters

**Type**: Code

**Purpose**: Define all tunable hyperparameters in a single `PARAMS` dictionary for reproducibility and easy modification.

**Key parameters**:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `corr_window` | 126 | Trading days (~6 months) for rolling Pearson correlation |
| `corr_threshold` | 0.6 | Minimum absolute correlation to create an edge |
| `corr_step` | 21 | Re-estimate correlation every ~1 month |
| `horizons` | [1, 5, 10, 21, 42, 63] | Forward return horizons in trading days |
| `default_horizon` | 5 | Primary horizon for N3 baseline comparisons |
| `hidden_channels` | 64 | GNN hidden layer dimensionality |
| `num_heads` | 4 | Number of attention heads (for GAT and HGT) |
| `num_hgt_layers` | 2 | Number of GNN message-passing layers |
| `dropout` | 0.3 | Dropout rate for regularization |
| `lr` | 1e-3 | Learning rate (Adam optimizer) |
| `weight_decay` | 1e-4 | L2 regularization strength |
| `epochs` | 100 | Maximum training epochs |
| `patience` | 15 | Early stopping patience (epochs without validation improvement) |
| `grad_accum` | 32 | Gradient accumulation steps (effective batch size = 32 days) |
| `top_k` | 30 | Number of stocks in long/short legs of the portfolio |
| `transaction_cost` | 15 | Round-trip transaction cost in basis points |
| `target_coverage` | 0.2 | SelectiveNet target coverage (20% of predictions retained) |
| `selection_lambda` | 32.0 | Penalty coefficient for SelectiveNet coverage constraint |

**Gradient accumulation explanation**: Since each trading day forms a single graph with ~501 stock nodes, the effective batch size is 1 day per forward pass. This is too small for stable gradient estimation. By accumulating gradients over 32 days before performing an optimizer step, the effective batch size becomes 32, improving training stability.

**Early stopping**: Training halts when validation loss has not improved for `patience` consecutive epochs, preventing overfitting.

---

## 5. Cell 3: Raw Data Loading

**Type**: Code

**Purpose**: Load all raw data files from disk: prices, sector classifications, news events, FinBERT embeddings, and FinBERT sentiment scores.

**Data sources**:

| File | Content | Shape |
|------|---------|-------|
| `sp500_5y_prices.csv` | Daily close prices, wide format (dates x tickers) | (1255, 502) |
| `sp500_sectors.csv` | GICS sector for each ticker | 502 rows |
| `sp500_news_events.parquet` | News event metadata (date, ticker, text) | 1,698,182 rows |
| `sp500_news_emb_finbert.npy` | FinBERT 768-dim embeddings per event | (1,698,182, 768) |
| `sp500_news_sentiment_finbert.npy` | FinBERT 3-dim sentiment (pos/neg/neu) per event | (1,698,182, 3) |

**FinBERT** (ProsusAI/finbert): a BERT model (Devlin et al., NAACL 2019) fine-tuned on financial text (Financial PhraseBank + financial news). It produces 768-dimensional sentence embeddings and 3-dimensional sentiment probability scores (positive, negative, neutral).

**Processing**:
- Computes the intersection of tickers across prices, events, and sectors to get `valid_tickers` (501 stocks -- one ticker lost in the intersection).
- Creates ticker-to-ID and date-to-index mappings for efficient array-based lookups.
- Computes daily returns via `prices.pct_change()`.

---

## 6. Cell 4 -- N1a: Price Features

**Type**: Code | **Section**: N1a

**Purpose**: Compute 9 price-based features per stock per day, using strictly T-1 close prices to avoid data leakage.

**Features** (9 dimensions):

| Feature | Windows | Description |
|---------|---------|-------------|
| `ret_mean_{w}d` | 5, 10, 21 | Mean daily return over the past w days (ending T-1) |
| `ret_std_{w}d` | 5, 10, 21 | Standard deviation of daily returns (volatility proxy, ending T-1) |
| `momentum_{w}d` | 5, 10, 21 | Cumulative return = close[T-1] / close[T-1-w] - 1 |

All features use `.shift(1)` to ensure only information available at market close on day T-1 is used. The first ~21 days have NaN values due to insufficient lookback, which are filled with zeros.

**Output**: `price_features` array of shape `(num_days, num_stocks, 9)`.

---

## 7. Cell 5 -- N1b: News Features

**Type**: Code | **Section**: N1b

**Purpose**: Map event-level FinBERT embeddings and sentiment scores onto a calendar grid of shape `(num_days, num_stocks, 772)`, where each cell is the mean-pooled representation of all news events for that stock on that day.

**Calendar-driven approach**: Unlike v2's event-driven paradigm (which only produced predictions when news existed), v3 creates a feature vector for every stock on every trading day. Days without news get zero vectors, plus a `has_news = 0` flag.

**Aggregation method**: Uses a sparse matrix multiplication trick for efficient mean-pooling. For each unique (day, stock) pair with news, a sparse aggregation matrix is constructed where each row sums the corresponding event embeddings and divides by the count.

**Output features per stock-day** (772 dimensions):

| Component | Dimensions | Content |
|-----------|------------|---------|
| FinBERT embedding | 768 | Mean-pooled sentence embedding of all news mentioning this stock on this day |
| Sentiment scores | 3 | Mean positive/negative/neutral probabilities |
| Has-news flag | 1 | Binary indicator (0 or 1) |

**Final combined feature tensor**: `stock_features_np` of shape `(num_days, num_stocks, 781)` = 9 price features + 772 news features.

Also precomputes per-day event metadata (`daily_graph_data`) needed for graph construction in N2: news node features, mention edges, and co-occurrence edges for each day.

---

## 8. Cell 6 -- N1c: Multi-Horizon Labels

**Type**: Code | **Section**: N1c

**Purpose**: Compute prediction labels for 6 different forward return horizons, using cross-sectional Z-score normalization.

**Label construction process**:
1. **Forward return**: `(close[T+h] - close[T]) / close[T]` for horizon h in {1, 5, 10, 21, 42, 63} trading days
2. **Market-adjusted (excess) return**: Subtract the equal-weight market return for that day to remove market beta
3. **Z-score normalization** (cross-sectional): `(excess_return - mean_across_stocks) / std_across_stocks` for each day independently

**Z-score normalization**: `(x - mean) / std`, standardizing values to mean 0 and standard deviation 1. Applied cross-sectionally (across all stocks on the same day), this removes the overall market direction and isolates relative stock performance.

**Validity mask**: The last h trading days for each horizon have NaN forward returns (since future prices are unavailable) and are masked as invalid.

**Verification**: The code prints per-horizon statistics confirming `mean ~ 0` and `std ~ 1.0` for valid labels, verifying correct normalization.

---

## 9. Cell 7 -- N1d: Time Split

**Type**: Code | **Section**: N1d

**Purpose**: Split the data into train/validation/test periods and convert arrays to PyTorch tensors.

**Time split**:

| Split | Date Range | Trading Days | News Coverage |
|-------|------------|--------------|---------------|
| Train | 2021-07-01 to 2023-12-31 | 629 | ~57.6% |
| Val | 2024-01-01 to 2024-06-30 | 124 | ~55.9% |
| Test | 2024-07-01 onward | 396 | ~62.7% |

"News coverage" refers to the percentage of stock-days that have at least one associated news event.

**Outputs**:
- `stock_features_t`: tensor of shape `(1255, 501, 781)`, ~1.96 GB
- `labels_t`: dictionary of 6 tensors (one per horizon)
- `label_valid_t`: dictionary of 6 boolean mask tensors

---

## 10. Cell 8 -- N2: Graph Construction

**Type**: Code | **Section**: N2

**Purpose**: Construct 4 types of graph edges that define the stock relationship network.

### Edge Type 1: Correlation (monthly dynamic)

- Uses a rolling window of 126 trading days (~6 months) to compute a Pearson correlation matrix across all 501 stocks.
- Edges are created between stocks with `|correlation| > 0.6`.
- Re-estimated every 21 trading days (~1 month), producing 54 snapshots over the full period.
- Each trading day is mapped to its most recent correlation snapshot.
- **Pearson correlation**: measures the linear relationship between two stocks' daily return series; values range from -1 to +1.

Graph density decreases over time (from ~2.9% to ~0.6%), reflecting changing market structure.

### Edge Type 2: Sector (static)

- Stocks in the same GICS (Global Industry Classification Standard) sector are connected.
- **GICS**: S&P/MSCI's industry classification standard with 11 sectors (Technology, Healthcare, Financials, etc.).
- Produces 27,070 undirected edges across 11 sectors (static, same for all days).

### Edge Type 3: News Mentions (daily)

- For each day, a directed edge from a news node to a stock node is created whenever that news event mentions the stock.
- Average ~1,226 mention edges per day.
- News node features: 771 dimensions (768 FinBERT embedding + 3 sentiment scores).

### Edge Type 4: Co-occurrence (daily)

- When a single news article mentions two or more stocks, undirected edges are created between all pairs.
- Average ~2,325 co-occurrence edges per day.

**Graph statistics summary**:

| Edge Type | Dynamic/Static | Total Edges | Notes |
|-----------|---------------|-------------|-------|
| Correlation | Monthly (54 snapshots) | Varies | Density 0.6%-2.9% |
| Sector | Static | 27,070 | 11 GICS sectors |
| News mentions | Daily | 1,538,967 total | ~1,226/day average |
| Co-occurrence | Daily | 2,918,292 total | ~2,325/day average |

---

## 11. Cell 9: HeteroData Builder

**Type**: Code

**Purpose**: Define a helper function `build_hetero_data(day_idx)` that constructs a PyTorch Geometric `HeteroData` object for a single trading day.

**HeteroData** (PyTorch Geometric): a data structure for heterogeneous graphs that supports multiple node types and edge types, each with their own feature tensors and edge index arrays.

The function assembles:
- **Stock nodes**: 501 nodes, each with 781-dim features
- **News nodes**: variable count per day (days without news get 1 dummy node with zero features)
- **4 edge types**: correlation (from snapshot lookup), sector (static), mentions (from daily data), co-occurrence (from daily data)

A validation check builds a sample graph from the middle of the training period and prints edge counts per type.

---

## 12. Cell 10: Evaluation Utilities

**Type**: Code

**Purpose**: Define functions for computing ranking evaluation metrics and portfolio backtesting.

### `compute_daily_ic()`

Computes daily **IC (Information Coefficient)**: the daily cross-sectional Spearman rank correlation between predicted scores and actual returns. IC > 0.03 is considered meaningful in quantitative finance. The function iterates over test days, computing Spearman correlation between predicted and actual values for each day (requiring at least 30 valid stocks).

**Spearman rank correlation**: a non-parametric measure of rank concordance between two variables; it assesses whether higher predictions correspond to higher actual returns, regardless of the exact values.

### `compute_metrics()`

Computes a full set of ranking metrics:

| Metric | Definition | Significance Threshold |
|--------|------------|----------------------|
| **IC** (Information Coefficient) | Mean of daily Spearman correlations between predicted scores and actual returns | IC > 0.03 is meaningful |
| **ICIR** (IC Information Ratio) | mean(IC) / std(IC), measuring signal consistency | ICIR > 0.1 indicates stable signal |
| **Sharpe_LS** (Sharpe Ratio, Long-Short) | Annualized return / annualized volatility of the long-short portfolio | Sharpe > 0.5 indicates economic significance |
| **Sharpe_Long** | Sharpe ratio of the long-only leg | |
| **Ann_LS** | Annualized long-short return (gross) | |
| **Ann_LS_net** | Annualized long-short return net of transaction costs | |
| **MaxDD** (Maximum Drawdown) | Largest peak-to-trough decline in cumulative portfolio value | Lower is better |

**Long-Short portfolio**: a strategy that goes long (buys) the top-k ranked stocks and short (sells) the bottom-k ranked stocks. With `top_k = 30`, each leg holds 30 stocks. The daily long-short return is the difference between the mean return of the long leg and the mean return of the short leg.

**Sharpe Ratio**: annualized return divided by annualized volatility; measures risk-adjusted return. Sharpe > 0.5 indicates economic significance. Computed as `mean(daily_returns) / std(daily_returns) * sqrt(252)`.

**Transaction costs**: Applied as `15 basis points (bps) round-trip` per rebalance. Turnover is computed by comparing the portfolio composition between consecutive rebalance dates.

### `print_results_table()`

Formats and prints a comparison table of all model results.

---

## 13. Cell 11 -- N3a: Non-GNN Baselines

**Type**: Code | **Section**: N3a

**Purpose**: Train and evaluate non-graph baselines to establish performance floors before testing GNN models.

**Data preparation**: The `flatten_data()` function reshapes the (day, stock, features) tensor into a standard (sample, features) matrix for scikit-learn / tree models. Features are standardized using `StandardScaler` (zero mean, unit variance).

### Baseline models

| ID | Model | Features | Description |
|----|-------|----------|-------------|
| B1 | Ridge Regression | 9-dim (price only) | Ridge regression: L2-regularized linear regression (alpha=1.0) that penalizes large coefficients to reduce overfitting. Uses only the 9 price/momentum features. |
| B2 | Ridge Regression | 781-dim (all) | Same as B1 but includes all 781 features (price + news). |
| B3 | XGBoost | 781-dim (all) | XGBoost (Chen & Guestrin, KDD 2016): gradient boosting framework using second-order gradients and regularized tree learning. Config: n_estimators=200, max_depth=5, lr=0.05, early_stopping=20. |
| B4 | LightGBM | 781-dim (all) | LightGBM (Microsoft): gradient boosting framework using histogram-based algorithms for efficient tree learning, with leaf-wise growth strategy. Same hyperparameters as XGBoost. |

**Prediction method**: After training, each baseline generates predictions for every stock on every day (not just test days) by calling `predict()` on the standardized features, filling a `(num_days, num_stocks)` prediction array. Metrics are then computed on test days only.

---

## 14. Cell 12 -- N3b: GNN Model Definitions

**Type**: Code | **Section**: N3b

**Purpose**: Define two GNN architectures for the ranking task.

### RankingHGT (Heterogeneous Graph Transformer)

**HGT (Heterogeneous Graph Transformer, Hu et al., WWW 2020)**: a transformer-based GNN that learns separate Key/Query/Value projection matrices for each edge type in heterogeneous graphs. This allows type-specific attention patterns. Parameter count scales with the number of edge types.

Architecture:
- `stock_lin`: Linear(781 -> 64) projects stock features to hidden space
- `news_lin`: Linear(771 -> 64) projects news features to hidden space
- 2 HGTConv layers with 4 attention heads each
- LayerNorm + residual connections after each layer
- Ranking head: Linear(64) -> ReLU -> Dropout -> Linear(1) producing a scalar ranking score per stock

If no real news nodes exist for a day, a single dummy news node with zero features is created to maintain metadata consistency in PyG.

### RankingGNN (Homogeneous GNN)

A simpler model supporting two variants:

- **GAT (Graph Attention Network, Velickovic et al., ICLR 2018)**: applies learned attention weights to neighbor aggregation, allowing the model to focus on more informative neighbors. Configuration: `GATConv(64, 16, heads=4, concat=True)` -- 4 attention heads each outputting 16 dimensions, concatenated to 64 dimensions.

- **SAGEConv / GraphSAGE (Hamilton et al., NeurIPS 2017)**: uses mean aggregation of neighbor features without attention, followed by a learned transformation. Simpler than GAT with fewer parameters.

Both variants use:
- Linear(781 -> 64) input projection
- 2 GNN layers with LayerNorm + residual connections
- Same ranking head as RankingHGT
- Operates on homogeneous graphs (stock nodes only, merged correlation + sector edges)

The `get_stock_embeddings()` method returns intermediate node embeddings (before the ranking head) for use by SelectiveNet.

---

## 15. Cell 13 -- N3c: GNN Training Loop

**Type**: Code | **Section**: N3c

**Purpose**: Define the `train_hgt()` function for training GNN models with gradient accumulation, early stopping, and learning rate scheduling.

**Training procedure**:

1. **Per-epoch**: Shuffle training days randomly, iterate through each day
2. **Per-day**: Build the HeteroData graph, run forward pass, compute MSE loss on valid labels
3. **Gradient accumulation**: Divide loss by `grad_accum` (32), accumulate gradients. Perform optimizer step every 32 days (or at epoch end).
4. **Gradient clipping**: `clip_grad_norm_(parameters, 1.0)` prevents exploding gradients
5. **Validation**: After each epoch, evaluate on all validation days (no gradient). Compute validation loss and IC.
6. **Early stopping**: Track best validation loss; stop after `patience` (15) epochs without improvement
7. **Learning rate scheduling**: `ReduceLROnPlateau` halves the learning rate after 5 epochs without validation improvement (min LR = 1e-5)

**ReduceLROnPlateau**: a learning rate scheduler that reduces the learning rate by a factor when a monitored metric plateaus, helping the optimizer escape flat regions of the loss landscape.

A corresponding `train_homogeneous_gnn()` function exists for RankingGNN models, using merged edge indices instead of HeteroData.

---

## 16. Cell 14 -- N3d: GNN Ablation and Go/Stop Gate

**Type**: Code | **Section**: N3d

**Purpose**: Run 5 GNN ablation experiments varying architecture and edge types, then apply a Go/Stop decision gate.

### Ablation experiments

| ID | Architecture | Edge Types | Graph Type |
|----|-------------|------------|------------|
| A1 | HGT | Correlation only | Heterogeneous (with dummy news nodes) |
| A2 | HGT | Correlation + Sector | Heterogeneous |
| A3 | HGT | All 4 types | Heterogeneous |
| A4 | GraphSAGE | Correlation + Sector | **Homogeneous** (merged edges, no news nodes) |
| A5 | GAT | Correlation + Sector | **Homogeneous** |

For ablations A1 and A2, the `build_hetero_data` function is temporarily monkey-patched to zero out excluded edge types, ensuring the HGT model receives the correct graph structure while maintaining consistent metadata.

### Go/Stop Gate Logic

After all 9 models (4 baselines + 5 GNNs) are evaluated, the notebook applies an automated decision gate:

```
Best IC  = max(IC across all models)     threshold: 0.03
Best Sharpe_LS = max(Sharpe across all)  threshold: 0.5

IF (Best IC > 0.03) OR (Best Sharpe_LS > 0.5):
    >>> GO: Proceed to N4 (Horizon Ablation) + N5 (Selective Prediction)
ELSE:
    >>> STOP: ranking also unpredictable, consider negative-result paper
```

The gate uses an OR condition -- passing either threshold is sufficient to continue. This is intentionally lenient because the default 5d horizon may not be optimal (N4 explores other horizons).

Results are saved to `experiments/v3_baseline_results.csv`.

---

## 17. Cell 15 -- N4: Horizon Ablation

**Type**: Code | **Section**: N4

**Purpose**: Systematically compare prediction performance across 6 time horizons (1d, 5d, 10d, 21d, 42d, 63d) using the best GNN configuration from N3 (GAT with correlation + sector edges) and LightGBM as a non-graph baseline.

**Procedure**: For each horizon h:
1. Train a fresh GAT(corr+sector) model from scratch on labels for horizon h
2. Evaluate on the test set, computing IC, ICIR, Sharpe, and portfolio metrics
3. Train a fresh LightGBM on the same horizon for direct comparison
4. Record both results

**Visualization**: Produces a 3-panel figure:
- Panel 1: IC vs Horizon (GAT vs LightGBM)
- Panel 2: ICIR vs Horizon
- Panel 3: Sharpe vs Horizon

Results saved to `experiments/v3_horizon_ablation.csv`.

### N4 Results (from Colab Run 2)

| Horizon | GAT IC | GAT ICIR | GAT Sharpe | GAT Ann_LS | LGBM IC | LGBM Sharpe | n_days |
|---------|--------|----------|------------|------------|---------|-------------|--------|
| 1d | -0.00104 | -0.013 | 2.468 | 34.54% | 0.00368 | 2.918 | 395 |
| 5d | 0.02334 | 0.227 | 1.568 | 18.27% | 0.00828 | 0.773 | 391 |
| 10d | 0.03854 | 0.320 | 1.196 | 19.26% | 0.01349 | 0.644 | 386 |
| **21d** | **0.04420** | **0.374** | **1.203** | **18.71%** | 0.01513 | 0.468 | 375 |
| 42d | -0.00912 | -0.144 | 0.071 | 0.73% | 0.03679 | 0.668 | 354 |
| 63d | -0.00838 | -0.118 | 0.487 | 6.36% | 0.05207 | 1.256 | 333 |

### Net-of-transaction-cost returns

| Horizon | GAT Ann_LS_net | LGBM Ann_LS_net |
|---------|---------------|-----------------|
| 1d | -41.06% | -37.23% |
| 5d | 3.15% | -4.20% |
| 10d | **11.70%** | 1.47% |
| 21d | **15.11%** | 2.95% |
| 42d | -1.07% | 8.45% |
| 63d | 5.16% | **24.35%** |

---

## 18. Cell 16 -- N5a: SelectiveNet Model Definition

**Type**: Code | **Section**: N5a

**Purpose**: Define the `SelectiveRankingGAT` model -- a GAT backbone with the SelectiveNet 3-head architecture.

### SelectiveNet (Geifman & El-Yaniv, ICML 2019)

SelectiveNet: a three-head architecture that learns not only what to predict but also when to abstain from prediction. It has 800+ citations and was originally designed for image classification with a rejection option.

**Architecture**:

```
                +--> Head 1: Ranking prediction (scalar score, MSE loss)
GAT backbone ---+--> Head 2: Selection head --> confidence in [0,1] (Sigmoid)
                +--> Head 3: Auxiliary prediction (scalar score, MSE loss, regularization)
```

- **Head 1 (Ranking)**: Same as RankingGNN -- produces a ranking score per stock
- **Head 2 (Selection)**: Takes GNN embeddings concatenated with 4-dimensional market context features, outputs a selection probability via Sigmoid. Determines whether the model should "trade" on this stock.
- **Head 3 (Auxiliary)**: Independent prediction head used for regularization (prevents the backbone from collapsing when the selection head gates gradients)

**Market context features** (4 dimensions, all using T-1 values to avoid data leakage):

| Feature | Description |
|---------|-------------|
| 21-day market volatility | Standard deviation of market returns * sqrt(252), a VIX proxy |
| 63-day drawdown | (current_level - 63d_peak) / 63d_peak, measures market stress |
| 30-day cross-sectional volatility | Mean of individual stock volatilities, measures dispersion |
| 5-day market breadth | Fraction of stocks with positive 5-day returns |

### SelectiveNet Loss Function

```
L = L_selective + lambda * max(0, c_target - coverage)^2 + L_auxiliary

where:
  L_selective = sum(selection_i * (pred_i - target_i)^2) / sum(selection_i)
  coverage = mean(selection_i)
  L_auxiliary = MSE(aux_pred, target)
```

- `L_selective`: Weighted MSE where the selection head controls how much each stock contributes to the loss. Stocks with higher selection scores contribute more.
- Coverage penalty: Penalizes the model if average selection probability exceeds `c_target` (0.2), encouraging it to be selective.
- `L_auxiliary`: Standard MSE loss ensuring the backbone learns useful representations even when the selection head gates gradients.

---

## 19. Cell 17 -- N5b: SelectiveNet Training

**Type**: Code | **Section**: N5b

**Purpose**: Train the SelectiveRankingGAT model using a 2-stage procedure.

### Market context computation

Before training, this cell computes the 4-dimensional market context tensor for all trading days. Each feature uses only information available at T-1:
- 21-day rolling market volatility (annualized)
- 63-day maximum drawdown
- 30-day average cross-sectional stock volatility
- 5-day market breadth (fraction of stocks with positive returns)

### Best horizon selection

Automatically selects the horizon with the highest GAT IC from N4 results. In Colab Run 2, this was 21d (IC = 0.04420).

### Two-stage training

**Stage 1: Backbone + Ranking + Auxiliary** (no selection constraint)
- Trains the full model, but the loss is just MSE(ranking, target) + MSE(auxiliary, target)
- The selection head exists but is not constrained
- Uses the same gradient accumulation (32 days) and early stopping (patience=15) as N3
- In Colab Run 2: ran for 31 epochs before early stopping

**Stage 2: Selection Head Only** (backbone + ranking frozen)
- Freezes the GAT backbone and ranking head parameters
- Only trains the selection head with the full SelectiveNet loss including coverage penalty
- Runs for 50 fixed epochs (no early stopping)
- In Colab Run 2: final coverage ~ 0.312 (above the 0.2 target)

**Rationale for 2-stage**: Training all three heads jointly from scratch can be unstable because the selection head can collapse to trivial solutions (all ones or all zeros) early in training, disrupting backbone learning. The 2-stage approach first establishes a good backbone, then learns selection.

---

## 20. Cell 18 -- N5c: Selection Analysis and Visualization

**Type**: Code | **Section**: N5c

**Purpose**: Compare three selection strategies and produce comprehensive visualizations.

### Three selection methods

1. **Full prediction (100%)**: Use all stock predictions without any selection. Serves as the baseline.

2. **Threshold-based selection**: Use the absolute value of the ranking score (`|ranking|`) as a confidence proxy. Higher absolute scores indicate more extreme predictions, which may be more reliable. Apply percentile thresholds to retain only the top 5%, 10%, 20%, 50%, or 100% of stocks.

3. **SelectiveNet selection**: Use the learned selection head's output as confidence. Apply the same percentile thresholds.

### Evaluation

For each method and coverage level, the cell computes the full metrics (IC, ICIR, Sharpe, etc.) using `compute_metrics()`. Stocks below the selection threshold have their predictions zeroed out.

### Visualization (6-panel figure)

- IC vs Coverage (Threshold vs SelectiveNet)
- Sharpe vs Coverage
- Selection score distribution (histogram)
- Selection-ranking correlation (scatter plot)
- Daily coverage time series
- Jaccard similarity between Threshold and SelectiveNet selected sets at each coverage level

### N5 Results (from Colab Run 2, 21d horizon)

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

---

## 21. Cell 19: Observations Markdown

**Type**: Markdown

A template cell with placeholder values ("TBD") for recording observations after a Colab run. It includes tables for key metrics vs targets, and checkboxes for next steps (Walk-forward validation, attention weight analysis, paper figures, negative-result framing). This cell is meant to be filled in manually after each experiment run.

---

## 22. Result Tables

### N3 Baseline + GNN Ablation (5d horizon, test set)

Two Colab runs were performed to assess reproducibility.

**Run 1** (NVIDIA RTX PRO 6000 Blackwell, 2026-03-05):

| Model | IC | ICIR | Sharpe_LS | Ann_LS | MaxDD |
|-------|-----|------|-----------|--------|-------|
| B1: Ridge (price 9d) | 0.00476 | 0.026 | 0.624 | 14.88% | 152.76% |
| B2: Ridge (all 781d) | 0.00535 | 0.052 | 0.597 | 8.06% | 79.00% |
| B3: XGBoost | 0.00329 | 0.024 | 0.185 | 2.89% | 76.59% |
| B4: LightGBM | 0.00828 | 0.079 | 0.773 | 10.92% | 44.52% |
| A1: HGT (corr) | 0.01023 | 0.133 | 0.121 | 1.25% | 51.53% |
| A2: HGT (corr+sector) | 0.01177 | 0.156 | 0.994 | 8.91% | 16.42% |
| A3: HGT (all 4) | 0.00432 | 0.061 | -0.314 | -2.83% | 39.29% |
| A4: SAGE (corr+sector) | 0.01571 | 0.152 | 1.038 | 13.51% | 35.08% |
| **A5: GAT (corr+sector)** | **0.02054** | **0.174** | **1.011** | **15.78%** | 38.56% |

**Run 2** (NVIDIA A100-SXM4-40GB, 2026-03-06):

| Model | IC | ICIR | Sharpe_LS | Sharpe_Long | Ann_LS | Ann_LS_net | MaxDD | n_days |
|-------|-----|------|-----------|-------------|--------|------------|-------|--------|
| B1: Ridge (price 9d) | 0.00476 | 0.026 | 0.624 | 1.223 | 14.88% | -0.24% | 152.76% | 391 |
| B2: Ridge (all 781d) | 0.00535 | 0.052 | 0.605 | 1.209 | 8.17% | -6.95% | 78.87% | 391 |
| B3: XGBoost | 0.00329 | 0.024 | 0.185 | 1.233 | 2.89% | -12.23% | 76.59% | 391 |
| B4: LightGBM | 0.00828 | 0.079 | 0.773 | 1.438 | 10.92% | -4.20% | 44.52% | 391 |
| A1: HGT (corr only) | 0.00848 | 0.092 | 0.426 | 1.367 | 4.68% | -10.44% | 63.45% | 391 |
| A2: HGT (corr+sector) | 0.01447 | 0.174 | 0.320 | 1.149 | 3.31% | -11.81% | 54.22% | 391 |
| A3: HGT (all 4 edges) | 0.00884 | 0.131 | 0.012 | 1.175 | 0.11% | -15.01% | 30.28% | 391 |
| A4: SAGE (corr+sector) | 0.01545 | 0.242 | 1.266 | 1.406 | 10.09% | -5.03% | 10.57% | 391 |
| A5: GAT (corr+sector) | 0.00640 | 0.072 | 0.289 | 1.234 | 3.49% | -11.63% | 64.80% | 391 |

**Key observation**: Baselines (B1-B4) are identical across runs (deterministic), while GNN results vary significantly due to training randomness.

### Go/Stop Gate Outcomes

| Condition | Threshold | Run 1 Value | Run 1 Result | Run 2 Value | Run 2 Result |
|-----------|-----------|-------------|-------------|-------------|-------------|
| Best IC > 0.03 | 0.03 | 0.02054 | FAIL | 0.01545 | FAIL |
| Best Sharpe_LS > 0.5 | 0.5 | 1.038 | PASS | 1.266 | PASS |
| **OR condition** | | | **GO** | | **GO** |

Both runs passed the gate via the Sharpe threshold, allowing N4 and N5 to proceed.

---

## 23. Key Findings

### Finding 1: Inverted-U Horizon Pattern (GAT)

GAT IC follows an inverted-U curve as a function of prediction horizon:

```
1d (-0.001) --> 5d (0.023) --> 10d (0.039) --> 21d (0.044) --> 42d (-0.009) --> 63d (-0.008)
                   rising                   peak at 21d             falling
```

- **21d is the optimal horizon**: IC = 0.04420 exceeds the 0.03 significance threshold; ICIR = 0.374; Sharpe = 1.203
- **10d also exceeds the threshold**: IC = 0.03854
- **1d has no signal**: IC is negative. Graph-based neighbor aggregation adds noise at daily frequency where signals are dominated by microstructure noise.
- **42d-63d signal disappears**: IC is negative. At longer horizons, stock returns are driven by macroeconomic factors rather than local graph structure, and GNN overfits.

This inverted-U pattern is a **publishable insight**: GNN-based stock ranking has a time-scale-dependent advantage, optimal at 2-4 weeks.

### Finding 2: GAT vs LightGBM Cross-Pattern

GAT and LightGBM exhibit opposite behaviors across horizons:

| Time Scale | Winner | Graph Structural Benefit | Explanation |
|-----------|--------|------------------------|-------------|
| 1d | LightGBM | Negative | Daily noise dominates; GNN neighbor aggregation amplifies noise |
| 5d-21d | **GAT** | Positive (2.8x-3.0x IC) | **Graph-propagated information is most valuable at weekly/monthly scales** |
| 42d-63d | LightGBM | Negative | Long-term trends driven by macroeconomic factors; local graph structure irrelevant |

LightGBM IC increases monotonically with horizon (0.004 -> 0.052), reflecting that tree models capture individual stock momentum features that become more predictive at longer horizons. LightGBM does not use graph structure.

### Finding 3: News/Co-occurrence Edges are Harmful

Adding news mention and co-occurrence edges to HGT (A3) causes a 63% IC drop compared to correlation+sector only (A2):

| Configuration | IC |
|--------------|-----|
| A2: HGT (corr+sector) | 0.01177 |
| A3: HGT (all 4 edges) | 0.00432 |

These edge types create dense, noisy connections (~1,226 mentions + ~2,325 co-occurrences per day) that dilute the more informative correlation structure. In high-noise financial data, fewer but higher-quality edges produce better results.

### Finding 4: SelectiveNet Failure

SelectiveNet produces **negative IC** at all coverage levels below 100%:

- @5%: IC = -0.01544 (inverse prediction)
- @10%: IC = -0.02159
- @20%: IC = -0.02414 (worst, at the target coverage)
- @50%: IC = -0.00874

The selection head learned to select stocks where the GNN predictions are *worst*, effectively inverting the signal. This is evidenced by:
- Selection score distribution is heavily right-skewed (most values 0.8-1.0), lacking discrimination
- Final coverage = 0.312 > target 0.2, indicating the coverage penalty was insufficient
- Jaccard similarity between SelectiveNet and Threshold selections is only 0.2-0.35, confirming they select nearly disjoint stock sets

**Root causes of SelectiveNet failure**:
1. The 2-stage training creates a disconnect -- freezing the backbone prevents the selection head from jointly optimizing
2. The coverage penalty (lambda=32.0) is not strong enough to enforce the 0.2 target
3. The ranking loss does not provide useful gradients to the selection head about *which predictions are reliable*
4. The selection score distribution collapses, lacking the spread needed for meaningful thresholding

**Threshold baseline works**: Using `|ranking score|` as a confidence proxy, Threshold @20% achieves IC = 0.03070 (above the 0.03 threshold) and Sharpe = 0.724. This suggests prediction confidence is embedded in score magnitude, but a separate learned head fails to extract it.

### Finding 5: Full Prediction Achieves Best Results

The SelectiveRankingGAT model at 100% coverage (Full prediction, no selection) achieves the highest IC of any model in the entire notebook:
- IC = 0.05595
- ICIR = 0.463
- Sharpe = 1.328
- Ann_LS_net = 16.48%

This exceeds the N4 GAT 21d result (IC = 0.04420), suggesting that the auxiliary loss head in the SelectiveNet architecture provides a regularization benefit to the backbone, even when the selection head itself fails.

### Finding 6: 1d Sharpe Anomaly

Both GAT and LightGBM show high Sharpe ratios at 1d (2.468 and 2.918 respectively) despite near-zero IC. This is a statistical artifact: the top-30/bottom-30 long-short portfolio happens to profit from daily turnover by chance. After deducting transaction costs (15 bps round-trip per day), the annualized return collapses to approximately -41%. This is not a real signal.

---

## 24. Training Stability Analysis

Two runs of the same N3 experiment revealed significant instability in GNN training:

| Model | Run 1 IC | Run 2 IC | |IC diff| | IC CV* |
|-------|----------|----------|----------|--------|
| A1: HGT (corr) | 0.01023 | 0.00848 | 0.00175 | ~15% |
| A2: HGT (corr+sec) | 0.01177 | 0.01447 | 0.00270 | ~21% |
| A3: HGT (all 4) | 0.00432 | 0.00884 | 0.00452 | ~69% |
| A4: SAGE (corr+sec) | 0.01571 | 0.01545 | 0.00026 | ~2% |
| A5: GAT (corr+sec) | 0.02054 | 0.00640 | 0.01414 | ~105% |

*CV = coefficient of variation (|diff| / mean)

**Stability ranking**:

```
SAGE (2%) >> HGT-corr (15%) > HGT-corr+sec (21%) >> HGT-all (69%) >> GAT (105%)
  most stable                                                        least stable
```

**Sources of non-determinism**:
1. **Day sequence shuffling**: Each epoch randomly permutes training day order
2. **Gradient accumulation boundaries**: Different shuffles produce different 32-day groupings
3. **Early stopping timing**: Different runs stop at different epochs, yielding different final parameters
4. **CUDA non-determinism**: Despite setting seeds, Colab GPU operations have inherent randomness

**Implications**:
- Run 1's conclusion that "GAT is best" may be a false positive; GAT's IC range spans [0.006, 0.021]
- SAGE is more reliable (IC stable at ~0.015) despite not achieving the highest single-run IC
- The N4 result of GAT 21d IC = 0.04420 also requires verification through multiple runs
- Walk-forward cross-validation with multiple repetitions is essential before drawing firm conclusions

---

## Glossary of Technical Terms

| Term | Full Name | Explanation |
|------|-----------|-------------|
| GNN | Graph Neural Network | Neural network that operates on graph-structured data through iterative message passing between nodes |
| GAT | Graph Attention Network | Velickovic et al. (ICLR 2018): applies learned attention weights to neighbor aggregation, allowing the model to focus on more informative neighbors |
| GraphSAGE | Sample and Aggregate | Hamilton et al. (NeurIPS 2017): uses fixed-size neighbor sampling with mean/max aggregation for inductive graph learning |
| HGT | Heterogeneous Graph Transformer | Hu et al. (WWW 2020): a transformer-based GNN that learns separate attention parameters for each edge type in heterogeneous graphs |
| SelectiveNet | Selective Prediction Network | Geifman & El-Yaniv (ICML 2019): a three-head architecture that learns not only what to predict but also when to abstain from prediction |
| IC | Information Coefficient | The daily cross-sectional Spearman rank correlation between predicted scores and actual returns; IC > 0.03 is considered meaningful in finance |
| ICIR | IC Information Ratio | mean(IC) / std(IC), measuring the stability/consistency of the prediction signal |
| Sharpe Ratio | -- | Annualized return divided by annualized volatility; Sharpe > 0.5 indicates economic significance |
| Long-Short portfolio | -- | A strategy that goes long top-ranked stocks and short bottom-ranked stocks to capture relative performance |
| Z-score normalization | Standard Score | (x - mean) / std, standardizing values to mean 0 and standard deviation 1 |
| LightGBM | Light Gradient Boosting Machine | Microsoft's gradient boosting framework using histogram-based algorithms for efficient tree learning |
| XGBoost | Extreme Gradient Boosting | Chen & Guestrin (KDD 2016): gradient boosting with regularized tree learning using second-order gradients |
| Ridge Regression | -- | Linear regression with L2 penalty on coefficients to prevent overfitting |
| FinBERT | Financial BERT | ProsusAI's BERT model fine-tuned on financial text for sentiment analysis and embedding generation |
| GICS | Global Industry Classification Standard | S&P/MSCI industry classification system with 11 sectors |
| MaxDD | Maximum Drawdown | Largest peak-to-trough decline in cumulative portfolio value |
| EMH | Efficient Market Hypothesis | Theory that asset prices fully reflect all available information |
| HeteroData | Heterogeneous Data | PyTorch Geometric data structure supporting multiple node and edge types |
| Gradient accumulation | -- | Technique of summing gradients over multiple mini-batches before performing a single optimizer step, effectively increasing batch size |
| Early stopping | -- | Stopping training when validation performance stops improving, to prevent overfitting |
| Cross-sectional | -- | Comparing values across all stocks at the same point in time |
| Forward return | -- | Future price change: (close[T+h] - close[T]) / close[T] |
| Pearson correlation | -- | Measures the linear relationship between two variables; ranges from -1 to +1 |
| Spearman rank correlation | -- | Non-parametric measure of rank concordance between two variables |
| Jaccard similarity | -- | Set overlap metric: |A intersection B| / |A union B|, ranges from 0 to 1 |
| LayerNorm | Layer Normalization | Normalizes activations across features within each sample, stabilizing training |
| ReduceLROnPlateau | -- | Learning rate scheduler that reduces LR when a monitored metric stops improving |

---

*Document generated: 2026-03-27*
