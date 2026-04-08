# Notebook Documentation: GNN测试1 colab.ipynb

## Phase 1 (Network Analysis), Phase 2 Pilot (News-Driven GNN), and Phase B (Dynamic Graph Sensitivity)

**Notebook file**: `GNN测试1 colab.ipynb`
**Runtime environment**: Google Colab with GPU (CUDA)
**Total cells**: 24 (Cells 0-23)
**Data storage**: Google Drive folder `GNN测试` (symlinked to GitHub repo for scripts)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Glossary of Models, Metrics, and Techniques](#2-glossary-of-models-metrics-and-techniques)
3. [Phase 1: Network Structure Analysis (Cells 0-9)](#3-phase-1-network-structure-analysis-cells-0-9)
4. [Phase B: Dynamic Graph Construction and Sensitivity Analysis (Cells 10-15)](#4-phase-b-dynamic-graph-construction-and-sensitivity-analysis-cells-10-15)
5. [Phase 2 Pilot: News-Driven GNN Prediction (Cells 16-22)](#5-phase-2-pilot-news-driven-gnn-prediction-cells-16-22)
6. [Summary of Key Results](#6-summary-of-key-results)

---

## 1. Overview

This notebook implements the earliest exploratory stages of the DynHetGNN-SP (Dynamic Heterogeneous Graph Neural Network with Selective Prediction for Stock Ranking) research project. It spans three logical phases:

- **Phase 1** explores whether a meaningful network structure exists among S&P 500 stocks by building a static correlation graph and running a minimal GCN on it.
- **Phase B** systematically searches for optimal parameters (window length, correlation threshold) for constructing dynamic correlation graphs that evolve over time.
- **Phase 2 Pilot** tests the core hypothesis on a small scale: can a GNN that combines news text embeddings with graph structure outperform a text-only baseline for predicting next-day stock direction?

The notebook operates on 502 S&P 500 constituent stocks with approximately 5 years of price data (2020-01 through 2025-12).

---

## 2. Glossary of Models, Metrics, and Techniques

### Models

- **GNN (Graph Neural Network)**: a neural network that operates on graph-structured data by passing messages between connected nodes to learn node-level or graph-level representations.
- **GCN (Graph Convolutional Network, Kipf & Welling, ICLR 2017)**: the foundational GNN that applies spectral convolutions on graphs, aggregating features from a node's neighbors via a symmetric normalized adjacency matrix.
- **GraphSAGE (Hamilton et al., NeurIPS 2017)**: an inductive GNN that learns by sampling and aggregating features from a fixed-size neighborhood, enabling generalization to unseen nodes.
- **Logistic Regression**: a linear classifier that models the probability of a binary outcome using a sigmoid function; used here as a non-graph baseline.
- **SentenceTransformer (all-MiniLM-L6-v2)**: a lightweight sentence embedding model (384 dimensions) based on a distilled BERT architecture, trained to produce semantically meaningful sentence vectors.

### Metrics

- **AUC (Area Under ROC Curve)**: measures binary classifier quality across all decision thresholds, where 0.5 = random guessing and 1.0 = perfect classification.
- **Pearson correlation**: measures the linear relationship between two variables, ranging from -1 (perfect negative) to +1 (perfect positive), with 0 indicating no linear relationship.
- **Jaccard similarity**: measures the overlap between two sets as |A intersection B| / |A union B|, ranging from 0 (no overlap) to 1 (identical sets); used here to measure month-to-month edge stability.
- **Graph density**: the ratio of actual edges to the maximum possible edges in a graph; indicates how interconnected the network is.
- **Clustering coefficient**: measures the degree to which nodes tend to cluster together, computed as the fraction of triangles around a node relative to the number of possible triangles.
- **Connected components**: the number of maximal subgraphs in which any two vertices are connected to each other by paths; ideally 1 large component for meaningful message passing.
- **Degree (node degree)**: the number of edges connected to a node; high-degree nodes are called "hubs."

### Techniques

- **t-SNE (t-distributed Stochastic Neighbor Embedding, van der Maaten & Hinton, 2008)**: a nonlinear dimensionality reduction technique that maps high-dimensional data to 2D/3D for visualization while preserving local neighborhood structure.
- **Spring layout (Fruchterman-Reingold)**: a force-directed graph layout algorithm that positions nodes by simulating attractive forces along edges and repulsive forces between all nodes.
- **Rolling window correlation**: computing pairwise Pearson correlation using only the most recent w trading days, then sliding the window forward to capture time-varying relationships.
- **HeteroData (PyTorch Geometric)**: a data structure for heterogeneous graphs containing multiple node types and edge types, each with their own feature matrices.
- **BCEWithLogitsLoss**: binary cross-entropy loss with a built-in sigmoid activation, numerically more stable than applying sigmoid separately.
- **Early stopping**: a regularization technique that halts training when validation performance stops improving for a specified number of epochs (patience).
- **GICS (Global Industry Classification Standard)**: a standardized sector classification system that groups companies into 11 sectors (e.g., Information Technology, Financials, Industrials).

---

## 3. Phase 1: Network Structure Analysis (Cells 0-9)

### Cell 0: Environment Setup and Data Loading

**Purpose**: Set random seeds for reproducibility, mount Google Drive (on Colab) or set local working directory, and download/load 5 years of S&P 500 daily closing prices.

**Details**:
- Seeds are fixed across Python, NumPy, and PyTorch (SEED = 42) for deterministic results.
- On Colab, the working directory is set to `GNN测试` on Google Drive, and the GitHub repository is cloned/pulled for access to scripts.
- On local machines, the working directory is `/Users/heruixi/Desktop/GNN-Testing`.
- S&P 500 tickers are fetched from the Wikipedia list of S&P 500 companies. Prices are downloaded via `yfinance` with `period="5y"` and cached to `data/reference/sp500_5y_prices.csv`.
- Daily returns are computed as percentage change of closing prices.

**Output data**: `prices` DataFrame of shape approximately (1255, 502) representing ~1255 trading days and 502 stocks.

### Cell 1: Utility Function for Saving Plots

**Purpose**: Define a helper function `save_graph(filename, folder)` that saves the current matplotlib figure as a 300 DPI PNG to the specified folder.

### Cell 2: Static Correlation Graph + Exploratory GCN

**Purpose**: Build the first correlation-based graph and verify that a GCN can perform a forward pass on it.

**Graph construction**:
1. Daily returns are computed from the price CSV.
2. A 502x502 Pearson correlation matrix is calculated over the entire 5-year period.
3. Edges are created between stock pairs whose absolute correlation exceeds 0.6: `|corr| > 0.6`.
4. Self-loops are included in this initial construction (they are filtered in later phases).

**Model architecture**: A minimal 2-layer GCN:
- Layer 1: GCNConv(1, 16) followed by ReLU activation
- Layer 2: GCNConv(16, 1)
- Input: the last day's return for each stock as a 1-dimensional node feature
- This is NOT a trained predictive model; it is a sanity check that the GCN forward pass works on the constructed graph.

**Results**:
- Nodes (stocks): 502
- Edges (connections): 3,198 (approximately 1.3% density)
- The forward pass completes successfully, outputting a [502, 1] tensor.

**Significance**: This cell validates the basic pipeline of building a correlation graph and passing it through a GNN.

### Cell 3: Market Capitalization Data

**Purpose**: Fetch and cache market capitalization data for all 502 S&P 500 stocks using `yfinance`.

**Details**:
- Market caps are stored in `data/reference/sp500_market_caps.csv`.
- This data is used later for selecting the top 100 stocks by market cap for visualization.

### Cell 4: GICS Sector Data

**Purpose**: Fetch and cache the GICS sector classification for all S&P 500 stocks from Wikipedia.

**Details**:
- Sector data is stored in `data/reference/sp500_sectors.csv`.
- The 11 GICS sectors are used for color-coding network visualizations and later for constructing sector-based edges in the heterogeneous graph.

### Cell 5: Top 100 Correlation Network Visualization

**Purpose**: Visualize the correlation network restricted to the top 100 stocks by market capitalization.

**Details**:
- Edges are filtered to keep only those where both endpoints are among the top 100 stocks.
- A spring layout (Fruchterman-Reingold, seed=42, k=0.15) positions nodes in 2D.
- All nodes are colored light blue (no sector distinction yet).

**Output**: Plot saved as `plots/S&P 500 Correlation Network (Top 100 Market Cap).png`.

### Cell 6: Sector-Colored Top 100 Network

**Purpose**: Same network as Cell 5, but with nodes colored by their GICS sector to reveal industry clustering.

**Details**:
- The `tab20` colormap assigns distinct colors to each of the 11 sectors.
- A legend identifies each sector.

**Key observation**: Stocks from the same sector tend to cluster together in the network, confirming that correlation-based edges capture meaningful industry relationships.

**Output**: Plot saved as `plots/S&P 500 Top 100 Sector Colored.png`.

### Cell 7: Full 502-Stock Network Visualization

**Purpose**: Visualize the complete S&P 500 correlation network with all 502 stocks.

**Details**:
- Labels are turned off due to the high node count.
- Nodes are smaller (size=100) and semi-transparent (alpha=0.6).
- Spring layout with tighter spacing (k=0.05).

**Output**: Plot saved as `plots/sp500_all_corr.png`.

### Cell 8: Top 10 Hub Stocks Analysis

**Purpose**: Identify and visualize the 10 most connected stocks (highest degree) in the full correlation network.

**Method**:
- Node degree is computed using PyTorch Geometric's `degree()` function on the edge index.
- The top 10 nodes by degree are extracted.
- A subgraph is built containing the top 10 hubs plus all their direct neighbors.

**Results** (Top 10 Hub Stocks):

| Rank | Ticker | Connections (Degree) | Sector |
|------|--------|---------------------|--------|
| 1 | SWK | 50 | Industrials |
| 2 | ITW | 39 | Industrials |
| 3 | MAS | 36 | Industrials |
| 4 | DOV | 33 | Industrials |
| 5 | TFC | 32 | Financials |
| 6 | IR | 32 | Industrials |
| 7 | ODFL | 31 | Industrials |
| 8 | CFG | 31 | Financials |
| 9 | RF | 30 | Financials |
| 10 | WAB | 30 | Industrials |

**Key finding**: Industrials and Financials dominate the hub rankings (7 Industrials + 3 Financials in the top 10). Notably, mega-cap technology stocks (AAPL, MSFT, GOOGL, AMZN, etc.) are NOT network hubs despite their large market capitalizations. This indicates that price correlation-based centrality reflects co-movement patterns (cyclical sectors move together) rather than market dominance.

**Visualization**: Hub stocks are shown in red with larger node size (1000); their neighbors appear in blue with smaller nodes (100).

**Output**: Plot saved as `plots/top10_hubs_network.png`.

### Cell 9: t-SNE Embedding Visualization

**Purpose**: Visualize the 16-dimensional GCN embeddings (from the first GCN layer in Cell 2) in 2D using t-SNE.

**Details**:
- The output of the first GCN layer (shape: 502 x 16) is extracted via `model.conv1(x, edge_index)`.
- t-SNE reduces this to 2 dimensions with `perplexity=30`, PCA initialization.
- The resulting 2D scatter plot shows how the GCN positions stocks in its learned representation space.

**Significance**: Even with a randomly-initialized (untrained) GCN, the embedding captures some structure from the graph topology, as the correlation graph implicitly encodes sector relationships.

**Output**: Plot saved as `plots/gnn_tsne_embedding.png`.

---

## 4. Phase B: Dynamic Graph Construction and Sensitivity Analysis (Cells 10-15)

Phase B is the systematic exploration of how to construct time-varying correlation graphs. Instead of a single static graph computed over the entire 5-year period (as in Phase 1), Phase B uses rolling windows to capture how stock relationships evolve over time.

### Cell 10: Dynamic Graph Snapshot Generation

**Purpose**: Generate graph snapshots across a grid of window sizes and correlation thresholds to identify the optimal parameters.

**Method**: The cell imports the `build_dynamic_graphs` module (from `scripts/build_dynamic_graphs.py`), which implements an extensible `GraphBuilder` interface. The `PearsonGraphBuilder` is used:

1. For each (window_size, threshold) combination, the algorithm slides a window of `w` trading days across the full return series with a step size of 21 trading days (approximately 1 month).
2. At each step, a 502x502 Pearson correlation matrix is computed from the returns within the window.
3. Edges are created between stock pairs where `|correlation| > threshold`.
4. Self-loops are explicitly removed via `mask.fill_diagonal_(False)`.
5. For each snapshot, structural statistics are computed: number of edges, density, average degree, maximum degree, clustering coefficient, number of connected components, and top-10 hub stocks.

**Parameter search grid**:

| Parameter | Values Tested | Interpretation |
|-----------|--------------|----------------|
| Window size (w) | 63, 126, 252 trading days | ~3, 6, 12 months of historical data |
| Threshold (tau) | 0.4, 0.5, 0.6, 0.7 | Minimum absolute correlation for edge inclusion |
| Step size | 21 trading days | ~1 month sliding step |

**Total configurations**: 3 window sizes x 4 thresholds = 12 parameter combinations. Each combination produces multiple monthly snapshots across the full date range, resulting in a comprehensive sensitivity analysis saved to `data/dynamic_graphs/sensitivity_analysis.csv`.

### Cell 11: Sensitivity Analysis Visualizations

**Purpose**: Produce heatmaps and time series plots to compare all parameter combinations.

**Visualization 1 -- Mean Edge Density Heatmap**: Shows how density varies by threshold (rows) and window size (columns). Key observations:
- Lower thresholds produce denser graphs (more edges, more noise).
- Longer windows produce denser graphs (more time for correlations to accumulate).
- The (w=126, tau=0.6) configuration yields approximately 6% density.

**Visualization 2 -- Mean Clustering Coefficient Heatmap**: Shows how tightly clustered the network is. Higher clustering means stocks form tighter groups (triangles), which is desirable for capturing sector-level relationships.

**Visualization 3 -- Mean Connected Components Heatmap**: Shows how fragmented the graph is. Too many components means stocks are disconnected, which limits GNN message passing. Too few components (with very high density) means the graph is nearly fully connected and lacks structural information.

**Visualization 4 -- Edge Count Time Series**: Three panels (one per window size) showing how the number of edges evolves over time for each threshold. This reveals:
- Bear markets (e.g., 2022) produce **correlation surges** where more stock pairs become highly correlated (more edges), because stocks tend to fall together during crises.
- Recovery periods show correlation relaxation (fewer edges).
- Shorter windows (w=63) produce more volatile edge counts; longer windows (w=252) are smoother but less responsive to regime changes.

**Outputs**: Plots saved as `plots/heatmap_density.png`, `plots/heatmap_clustering.png`, `plots/heatmap_components.png`, `plots/timeseries_edge_count.png`.

### Cell 12: Hub Evolution and Network Visualizations

**Purpose**: Analyze how hub stocks change over time and visualize selected network snapshots.

**Part 1 -- Hub Ranking Over Time (Race Chart)**:
- Uses the mid-range parameters (w=126, tau=0.6) as reference.
- For each monthly snapshot, the top 10 stocks by degree are tracked.
- Stocks appearing in the top 10 at least 3 times across all snapshots are plotted.
- Colors distinguish Industrials (blue) and Financials (red).
- The chart reveals that hub identity is relatively stable but not static: some stocks enter or leave the top 10 depending on market regime.

**Part 2 -- Sector Composition of Top 10 Hubs**:
- An area chart showing how many of the top 10 hubs belong to each sector at each time point.
- Confirms that Industrials and Financials consistently dominate hub positions.

**Part 3 -- Comparison Network Visualizations**:
- Two contrasting snapshots are visualized: a short-window tight-threshold configuration (w=63, tau=0.7) versus a long-window loose-threshold configuration (w=252, tau=0.4), showing how parameter choices affect graph structure.

**Outputs**: Plots saved as `plots/hub_ranking_timeseries.png`, `plots/hub_sector_composition.png`, `plots/network_top100_63w_07.png`, `plots/network_top100_252w_04.png`.

### Cell 13: Phase B+ Markdown Summary (Best Parameter Selection)

**Purpose**: A markdown cell documenting the rationale for selecting w=126 and tau=0.6 as the optimal parameters.

**Selected configuration**: window = 126 trading days (~6 months), threshold = 0.6

**Rationale for selection**:

| Criterion | w=126, tau=0.6 Performance | Why It Matters |
|-----------|---------------------------|----------------|
| Density: ~6% | Each stock connects to approximately 30 peers | Balanced: not so dense that the graph becomes a noisy blob, not so sparse that information cannot propagate |
| Temporal stability (std=0.064) | 3x more stable than w=63 (std=0.095) | The graph should be stable enough for the GNN to learn consistent patterns, yet responsive enough to capture regime changes |
| Clustering coefficient: 0.453 | Industry clusters are visible | Supports sector-level message passing without collapsing into a single dense cluster |
| Connected components: ~125 | Naturally separates uncorrelated stocks | Utilities and Tech, for example, are in different components, reflecting their low correlation |
| Regime sensitivity | Edge count spikes during 2022 bear market, drops during recovery | The graph captures meaningful market dynamics |

**Comparison with rejected configurations**:

| Configuration | Issue |
|---------------|-------|
| w=63, tau=0.7 | Too sparse and volatile; graph structure changes too rapidly between months |
| w=63, tau=0.4 | Too dense; most stocks connected to most others, destroying structural information |
| w=252, tau=0.7 | Too smooth; fails to capture mid-term regime changes |
| w=252, tau=0.4 | Extremely dense; graph is nearly complete, GNN learns nothing from structure |
| w=126, tau=0.5 | Slightly too dense at ~10%; tau=0.6 provides cleaner edges |
| w=126, tau=0.7 | Slightly too sparse; some meaningful connections are lost |

**Jaccard stability analysis** (w=126, tau=0.6):
- Mean Jaccard similarity between consecutive monthly snapshots: 0.631
- Standard deviation: 0.064
- Interpretation: approximately 63% of edges persist from one month to the next, indicating that the graph is predominantly stable while still allowing meaningful evolution.

### Cell 14: Full Monthly Snapshot Generation

**Purpose**: Generate all 54 monthly network snapshots for the selected parameters (w=126, tau=0.6) and save them as individual PNG files.

**Details**:
- For each of the 54 monthly time steps, a Pearson correlation graph is built from the trailing 126 trading days.
- The graph is filtered to the top 100 stocks by market cap for visualization.
- Each snapshot is saved as a separate PNG with sector-colored nodes and structural statistics in the title (density, average degree, number of components).
- Key regime dates are flagged for inline display:
  - Pre-COVID Recovery (2021-07)
  - Rate Hike Start (2022-03)
  - Bear Market Bottom (2022-10)
  - AI Rally Start (2023-06)
  - Magnificent 7 Peak (2024-03)
  - Recent (2025-07)

**Output directory**: `plots/dynamic_snapshots_w126_t06/`
**Total snapshots**: 54 PNG files

### Cell 15: Key Regime Snapshot Display

**Purpose**: Display the edge count evolution over time with annotated market regime events, and show selected regime snapshots inline.

**Edge evolution plot with regime annotations**:
- Total edges (all 502 stocks) plotted as a solid blue line.
- Top-100 edges (scaled 5x for visibility) plotted as a dashed red line.
- Annotated events: Fed rate hike announcement (2022-01), bear market S&P -20% (2022-06), market bottom (2022-10), recovery begins (2023-01), AI rally accelerates (2023-11), Magnificent 7 dominance (2024-07).

**Key regime snapshots displayed inline** (6 critical time points):

| Period | Label | What the Graph Shows |
|--------|-------|---------------------|
| 2021-07 | Pre-Rate-Hike Baseline | Normal market conditions, moderate connectivity |
| 2022-03 | Rate Hike Begins | Correlation surge as uncertainty increases |
| 2022-10 | Bear Market (Correlation Surge) | Maximum connectivity; nearly all stocks falling together |
| 2023-06 | Recovery & AI Rally | Correlations begin to relax; tech stocks diverge |
| 2024-03 | Magnificent 7 Era | Lower overall correlation; mega-caps decouple from broader market |
| 2025-06 | Recent State | Current market structure |

**Output**: Plot saved as `plots/dynamic_edge_evolution_annotated.png`.

**Significance of Phase B**: This analysis provides empirical evidence that stock correlation networks are dynamic and regime-dependent. A static graph (as used in Phase 1) misses important structural changes. The selected parameters (w=126, tau=0.6) achieve a balance between stability and responsiveness, producing graphs suitable for GNN training.

---

## 5. Phase 2 Pilot: News-Driven GNN Prediction (Cells 16-22)

Phase 2 Pilot tests the core research hypothesis at small scale: can a heterogeneous GNN that combines news text embeddings with stock correlation structure outperform a text-only baseline for binary stock direction prediction?

### Data

- **Source**: 1,900 news articles manually collected from Factiva for 9 hub stocks (identified in Phase 1).
- **After cleaning**: 480 usable news events stored in `data/pilot/news_events.parquet`.
- **Fields per event**: `doc_id`, `ticker`, `date`, `text`, `label` (0 = down, 1 = up), `return_next` (next-day return).
- **Stocks covered**: 9 hub stocks from the Phase 1 analysis.

### Cell 16: Import and Load News Events

**Purpose**: Load the pilot news events dataset and clean text fields.

**Details**:
- News events are loaded from `data/pilot/news_events.parquet`.
- Text is cleaned by collapsing whitespace and stripping leading/trailing spaces.
- Output shape: (480, columns) where each row is one news event.

### Cell 17: Sentence Embedding Model Loading

**Purpose**: Load the SentenceTransformer model for encoding news text into dense vectors.

**Model**: `sentence-transformers/all-MiniLM-L6-v2`
- Architecture: 6-layer distilled BERT producing 384-dimensional sentence embeddings.
- This is a general-purpose sentence embedding model (not finance-specific). The later full-scale Phase A uses FinBERT instead.
- Runs on GPU (CUDA) when available.

### Cell 18: Batch Encoding of News Text

**Purpose**: Encode all 480 news articles into dense embedding vectors.

**Details**:
- Texts are encoded in batches of 64 using `model.encode()` with `torch.inference_mode()` for efficiency.
- Embeddings are normalized (`normalize_embeddings=True`) so that cosine similarity equals dot product.
- Output is cast to float16 to reduce memory.

**Result**: Embedding matrix of shape (480, 384) -- 480 news events, each represented by a 384-dimensional vector.

### Cell 19: Save Embeddings and Metadata

**Purpose**: Persist the embeddings and aligned metadata to disk.

**Files saved**:
- `data/pilot/news_events_emb.npy`: NumPy array of shape (480, 384) containing all embeddings.
- `data/pilot/news_events_emb_meta.parquet`: Parquet file with columns `doc_id`, `ticker`, `date`, `label`, `return_next`, `text_idx`.

### Cell 20: Embedding Integrity Checks

**Purpose**: Verify data integrity after saving.

**Checks performed**:
- Row count match between embeddings and metadata.
- No NaN values in embeddings.
- No NaN values in metadata columns.
- Label distribution (approximate 50/50 split between up and down).
- Number of events per ticker.

### Cell 21: Baseline Logistic Regression

**Purpose**: Establish a non-graph baseline for comparison with the GNN.

**Data split** (strict temporal, no shuffling):
- Train: first 80% of events by date
- Validation: next 10%
- Test: final 10%

**Logistic Regression configuration**:
- Solver: SAGA (efficient for medium-scale problems)
- Penalty: L2 regularization
- C = 2.0 (inverse regularization strength)
- Class weight: balanced (handles any label imbalance)
- Max iterations: 500

**Input features**: Raw 384-dimensional MiniLM embeddings (text only, no graph information).

**Evaluation**: AUC and classification report on validation and test sets.

**Result**: Test AUC = 0.6213 (as reported in the research log). This establishes the text-only performance ceiling for the pilot dataset.

**HeteroData construction** (also in Cell 21):
After the baseline, the cell builds a PyTorch Geometric `HeteroData` object for the GNN:
- **News nodes**: 480 nodes with 384-dimensional features (embeddings) and binary labels.
- **Stock nodes**: 9 nodes with one-hot identity features (placeholder; to be replaced with price features in later phases).
- **Edges (news -> stock)**: 480 edges connecting each news event to its associated stock (the `relates_to` relation).
- Train/val/test masks are applied to news nodes.

The graph is saved to `data/pilot/graph_data.pt`.

### Cell 22: GraphSAGE Heterogeneous GNN

**Purpose**: Train a heterogeneous GraphSAGE model on the news-stock graph and compare with the Logistic Regression baseline.

**Model architecture**:
- Base model: 2-layer GraphSAGE with `SAGEConv((-1, -1), 32)`
  - The `(-1, -1)` notation tells PyG to infer input dimensions automatically per node type.
  - Hidden channels: 32
  - Dropout: 0.4 between layers
  - Activation: ReLU
- Converted to heterogeneous via `to_hetero(model, data.metadata(), aggr='mean')`, which creates separate weight matrices for each (source_type, edge_type, target_type) triplet.
- Final output: Linear(32, 1) producing a single logit per news node.
- Reverse edges are added via `T.ToUndirected()` to enable bidirectional message passing (stock -> news as well as news -> stock).

**Training configuration**:
- Optimizer: Adam (lr=0.004, weight_decay=5e-4)
- Loss: BCEWithLogitsLoss (binary cross-entropy with logits)
- Early stopping: patience=15 epochs, monitoring validation AUC
- Maximum epochs: 200

**Training loop**:
1. Forward pass through the heterogeneous GNN.
2. Compute loss on training news nodes only.
3. Evaluate validation AUC and test AUC every epoch.
4. Save best model state (by validation AUC).
5. Restore best model after early stopping triggers.

**Results**:

| Model | Test AUC |
|-------|----------|
| Logistic Regression (text only) | 0.6213 |
| GraphSAGE (heterogeneous graph) | **0.6426** |

- **Graph structure benefit**: +0.0213 AUC (+3.4% relative improvement).
- The GNN successfully leverages the news-to-stock and stock-to-stock correlation edges to improve prediction beyond what text alone provides.

**Limitations acknowledged**:
- Only 9 stocks and 480 news events -- too small for statistical significance.
- The pilot uses MiniLM-L6-v2 (general-purpose) rather than FinBERT (finance-specific).
- The correlation graph is static (computed over the full period), not dynamic.
- These limitations motivate the subsequent full-scale phases (Phase A for data, Phase B for dynamic graphs, Phase C for full experiments).

### Cell 23: Empty Cell

An empty cell at the end of the notebook, likely a placeholder for future work.

---

## 6. Summary of Key Results

### Phase 1 Findings

| Finding | Detail |
|---------|--------|
| S&P 500 correlation graph | 502 nodes, 3,198 edges at |corr| > 0.6, density ~1.3% |
| Hub stocks dominated by Industrials and Financials | SWK (50 connections), ITW (39), MAS (36) lead; mega-cap tech stocks are NOT hubs |
| Sector clustering confirmed | Stocks in the same GICS sector cluster together in the correlation network |
| GCN forward pass validated | A minimal 2-layer GCN (1 -> 16 -> 1) successfully processes the graph |

### Phase B Findings

| Finding | Detail |
|---------|--------|
| Optimal window size | w = 126 trading days (~6 months): balances stability (std=0.064) with regime responsiveness |
| Optimal threshold | tau = 0.6: density ~6%, each stock connected to ~30 peers |
| Graph stability | Jaccard similarity = 0.631 between consecutive months; graph is predominantly stable |
| Regime sensitivity confirmed | Edge count spikes during 2022 bear market (correlation surge), drops during 2023-2024 recovery |
| Hub persistence | Industrials and Financials consistently dominate hub positions across market regimes |
| Total snapshots | 54 monthly graphs generated for the optimal configuration (2020-07 to 2025-12) |

### Phase 2 Pilot Findings

| Finding | Detail |
|---------|--------|
| Text-only baseline (LR) | AUC = 0.6213 on 9-stock, 480-event pilot dataset |
| GraphSAGE (heterogeneous) | AUC = 0.6426, a +3.4% improvement over text-only |
| Conclusion | Graph structure provides incremental benefit for news-driven stock prediction, but the pilot scale is too small for definitive conclusions |

### Progression to Later Phases

The results from this notebook motivated the following next steps (implemented in separate notebooks):
- **Phase A** (`phase_a_data_prep.ipynb`): Scale up from 9 stocks / 480 events to 502 stocks / 1.5M+ events using EODHD data and FinBERT embeddings.
- **Phase B parameters carried forward**: w=126, tau=0.6 became the standard configuration for all subsequent dynamic graph construction.
- **Phase C** (`phase_c_model_training.ipynb`): Full-scale experiments with 6 model configurations. Result: all AUC approximately 0.50, indicating that the pilot's positive signal did not scale to the full S&P 500 universe.
- **v3 pivot** (`v3_ranking_pipeline.ipynb`): Shifted from binary direction prediction to ranking-based prediction with calendar-driven features, ultimately achieving IC = 0.04420 with a GAT architecture.
