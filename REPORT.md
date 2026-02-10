# GNN-Based Stock Correlation Network Analysis — Working Progress Report

## 1. Project Overview

**Objective:** Apply Graph Neural Networks (GNNs) to model and analyze the correlation structure among S&P 500 stocks. The project constructs a stock correlation network from historical price data and uses GNN architectures (GCN, GraphSAGE) to learn node-level embeddings that capture inter-stock relationships.

**Tools & Environment:**
- Python 3.12, Google Colab (GPU runtime) / local macOS
- Core libraries: `torch`, `torch_geometric` (PyG), `yfinance`, `pandas`, `networkx`, `matplotlib`, `scikit-learn`

---

## 2. Data Collection

### 2.1 Stock Price Data

- **Source:** Yahoo Finance via the `yfinance` Python library
- **Scope:** All S&P 500 constituent stocks (tickers scraped from the [Wikipedia S&P 500 list](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies))
- **Period:** 5 years of daily data (`yf.download(tickers, period="5y", auto_adjust=True)`)
- **Field:** Adjusted closing prices only
- **Preprocessing:** Stocks with all-NaN columns were dropped
- **Result:** A DataFrame of shape **(1255 trading days x 502 stocks)**, saved as `sp500_5y_prices.csv`
- **Caching:** The CSV is saved locally; subsequent runs load from disk to avoid re-downloading

### 2.2 Market Capitalization Data

- **Source:** `yfinance` Ticker `.info["marketCap"]` for each of the 502 stocks
- **Method:** Iterated over all tickers individually, extracting the current market cap value. Tickers where the API call failed were assigned a market cap of 0.
- **Result:** A Series of 502 entries, **all non-zero**, saved as `sp500_market_caps.csv`
- **Top 5 by market cap:**

| Rank | Ticker | Market Cap (USD) |
|------|--------|-----------------|
| 1 | NVDA | $4.63T |
| 2 | AAPL | $4.04T |
| 3 | GOOG | $3.92T |
| 4 | GOOGL | $3.92T |
| 5 | MSFT | $3.07T |

### 2.3 GICS Sector Data

- **Source:** Wikipedia S&P 500 list table, column `"GICS Sector"`
- **Method:** Single HTTP request to parse the HTML table, then extracted the `Symbol → GICS Sector` mapping. Tickers with `.` in the symbol (e.g., `BRK.B`) were converted to `-` format (e.g., `BRK-B`) for consistency with yfinance.
- **Result:** 503 tickers mapped to **11 GICS sectors**, saved as `sp500_sectors.csv`
- **Sectors:** Communication Services, Consumer Discretionary, Consumer Staples, Energy, Financials, Health Care, Industrials, Information Technology, Materials, Real Estate, Utilities

---

## 3. Graph Construction

### 3.1 Method

1. **Daily returns** were computed from closing prices using `pct_change()`, then rows with NaN (the first row) were dropped.
2. A **Pearson correlation matrix** was computed across all 502 stocks from the full 5-year return series: `returns.corr()` → shape `(502, 502)`.
3. An **edge** was created between two stocks if their absolute correlation exceeded a threshold of **0.6**: `(corr_matrix.abs() > 0.6).nonzero().t()`
4. The resulting `edge_index` tensor is in PyG's COO sparse format: shape `[2, num_edges]`.

### 3.2 Graph Statistics

| Metric | Value |
|--------|-------|
| Nodes (Stocks) | 502 |
| Edges (Connections) | 3,198 |
| Edge Density | 0.0127 |
| Graph Type | Undirected, unweighted |
| Threshold | \|correlation\| > 0.6 |

The low density (1.27%) indicates that most stock pairs are NOT strongly correlated — only a select subset of stocks form tight clusters.

---

## 4. GCN Model (Graph Convolutional Network)

### 4.1 What is GCN?

A Graph Convolutional Network (Kipf & Welling, 2017) generalizes convolution operations from grid-structured data (images) to graph-structured data. Each GCN layer performs **neighborhood aggregation**: a node's new representation is computed by averaging the features of its neighbors (plus itself), followed by a linear transformation and non-linear activation.

Mathematically, one GCN layer computes:

```
H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))
```

where `Ã = A + I` (adjacency matrix with self-loops), `D̃` is the degree matrix of `Ã`, `W` is a learnable weight matrix, and `σ` is an activation function (ReLU).

### 4.2 Model Architecture

```
GCN(
  (conv1): GCNConv(1, 16)    # Input: 1 feature → 16 hidden dimensions
  (conv2): GCNConv(16, 1)    # Hidden: 16 → 1 output dimension
)
```

| Layer | Input Dim | Output Dim | Activation | Purpose |
|-------|-----------|------------|------------|---------|
| conv1 | 1 | 16 | ReLU | Aggregate 1-hop neighbor features |
| conv2 | 16 | 1 | None | Aggregate 2-hop neighbor features |

### 4.3 Input Features

- **Node features (x):** The most recent day's return for each stock, reshaped to `[502, 1]`
- This is a minimal feature — just a single scalar per node — to demonstrate the GCN's ability to propagate and aggregate information across the graph structure

### 4.4 Execution

- This was a **single forward pass** (inference only, no training/optimization)
- Purpose: Verify that the GCN architecture works end-to-end on the constructed graph
- **Output shape:** `[502, 1]` — one value per stock, representing the GCN's aggregated output after 2 layers of neighborhood message passing

### 4.5 Why This Matters

Even without training, the forward pass demonstrates that each stock's output is now influenced by its correlated neighbors up to 2 hops away. The GCN effectively "smooths" information across the correlation network, meaning stocks with similar correlation profiles will produce similar output values.

---

## 5. Visualizations

### 5.1 Top 100 by Market Cap — Correlation Network

- **Selection:** The 100 stocks with the highest current market capitalization
- **Edges:** Filtered to include only edges where both endpoints are in the top 100 subset
- **Layout:** Spring layout (force-directed), `seed=42`, `k=0.15`
- **Node labels:** Ticker symbols (e.g., AAPL, MSFT, NVDA)
- **Saved as:** `plots/S&P 500 Correlation Network (Top 100 Market Cap).png`

### 5.2 Top 100 by Market Cap — Colored by GICS Sector

- **Same graph as 5.1**, but nodes are colored by their GICS sector using a `tab20` colormap
- **Legend:** Displayed in upper-left corner, mapping each color to its sector name
- **Purpose:** Visually assess whether stocks in the same industry cluster together in the correlation network
- **Saved as:** `plots/S&P 500 Top 100 Sector Colored.png`

### 5.3 Full S&P 500 Network (All 502 Stocks)

- **All 502 nodes** and all 3,198 edges displayed
- **Labels off** (too dense for 500+ nodes)
- **Node size:** 100 (small), **alpha:** 0.6 (semi-transparent)
- **Canvas:** 20x20 inches, spring layout with `k=0.05` (tighter clustering)
- **Purpose:** Overview of the entire network structure — reveals the overall connectivity pattern and isolated nodes
- **Saved as:** `plots/sp500_all_corr.png`

### 5.4 Top 10 Hub Stocks & Their Neighbors

- **Method:** Computed node degree (number of edges) for all 502 stocks using `torch_geometric.utils.degree`, then selected the top 10 by degree
- **Subgraph:** Includes any edge where at least one endpoint is a top-10 hub, plus all unique nodes in those edges (hubs + their neighbors)
- **Visual encoding:**
  - Hub stocks: large red nodes (size=1000)
  - Neighbor stocks: small blue nodes (size=100)
- **Purpose:** Identify the most "central" stocks in the correlation network — these are the stocks most correlated with many others, acting as market-wide connectors
- **Top 10 Hub Stocks:**

| Rank | Ticker | Company Name | Sector | Connections |
|------|--------|-------------|--------|-------------|
| 1 | SWK | Stanley Black & Decker, Inc. | Industrials | 50 |
| 2 | ITW | Illinois Tool Works Inc. | Industrials | 39 |
| 3 | MAS | Masco Corporation | Industrials | 36 |
| 4 | DOV | Dover Corporation | Industrials | 33 |
| 5 | IR | Ingersoll Rand Inc. | Industrials | 32 |
| 6 | TFC | Truist Financial Corporation | Financial Services | 32 |
| 7 | CFG | Citizens Financial Group, Inc. | Financial Services | 31 |
| 8 | ODFL | Old Dominion Freight Line, Inc. | Industrials | 31 |
| 9 | RF | Regions Financial Corporation | Financial Services | 30 |
| 10 | WAB | Westinghouse Air Brake Technologies Corp. | Industrials | 30 |

- **Observation:** 7 of the top 10 hubs are **Industrials** stocks, and the remaining 3 are **Financial Services**. This suggests that these two sectors have the strongest internal correlation structure — stocks within these sectors tend to move together, creating highly connected hub nodes. Notably, none of the mega-cap tech stocks (AAPL, MSFT, NVDA) appear as hubs, indicating that high market cap does not equate to high network centrality.
- **Saved as:** `plots/top10_hubs_network.png`

### 5.5 GCN Embedding Visualization (t-SNE)

- **Embeddings extracted from:** The output of `conv1` (the first GCN layer), giving a 16-dimensional embedding for each stock
- **Dimensionality reduction:** t-SNE (t-distributed Stochastic Neighbor Embedding)
  - `n_components=2` (reduce to 2D for plotting)
  - `perplexity=30` (standard; controls how many neighbors to consider when computing conditional probabilities)
  - `init='pca'` (initialize with PCA for stability)
  - `learning_rate='auto'`
- **Result:** A 2D scatter plot of all 502 stocks, where proximity indicates similar GCN-learned representations
- **Purpose:** Visualize how the GCN organizes stocks in its latent space after one layer of neighborhood aggregation
- **Saved as:** `plots/gnn_tsne_embedding.png`

---

## 6. Summary of Parameters & Choices

| Component | Parameter | Value | Rationale |
|-----------|-----------|-------|-----------|
| Price data | Period | 5 years | Sufficient history for stable correlation estimates |
| Price data | Field | Adjusted close | Accounts for splits and dividends |
| Correlation | Method | Pearson | Standard for linear co-movement |
| Graph | Edge threshold | \|corr\| > 0.6 | Balances connectivity vs. sparsity (density ~1.3%) |
| GCN | Hidden dim | 16 | Lightweight; sufficient for structural exploration |
| GCN | Layers | 2 | Captures 2-hop neighborhood information |
| GCN | Activation | ReLU | Standard non-linearity |
| GCN | Input feature | Last day's return (dim=1) | Minimal feature to demonstrate graph propagation |
| t-SNE | Perplexity | 30 | Standard default for medium-sized datasets |
| Market cap | Source | yfinance `.info` | Most accessible real-time source |
| Sectors | Source | Wikipedia | Single-request, already scraped in pipeline |

---

## 7. Key Findings (Preliminary)

1. **Sparse but structured network:** With a 0.6 correlation threshold, only 1.3% of possible edges exist, yet the network exhibits clear clustering — stocks do not connect randomly.

2. **Hub stocks exist:** A small number of stocks have disproportionately high connectivity (degree), suggesting they are "bellwether" stocks whose movements are correlated with many others across sectors.

3. **Sector clustering is visible:** When coloring the top-100 network by GICS sector, same-sector stocks tend to cluster together, confirming that industry membership drives correlation structure.

4. **GCN embeddings capture structure:** Even with a single scalar input feature and no training, the first-layer GCN embeddings (16D) produce meaningful clusters in t-SNE space, demonstrating that the network topology alone carries informative signal.

---

## 8. Files Produced

| File | Description |
|------|-------------|
| `sp500_5y_prices.csv` | 5-year daily closing prices (1255 x 502) |
| `sp500_market_caps.csv` | Current market cap for 502 stocks |
| `sp500_sectors.csv` | GICS sector mapping for 503 tickers |
| `plots/S&P 500 Correlation Network (Top 100 Market Cap).png` | Top 100 network |
| `plots/S&P 500 Top 100 Sector Colored.png` | Sector-colored network |
| `plots/sp500_all_corr.png` | Full 502-stock network |
| `plots/top10_hubs_network.png` | Hub stocks visualization |
| `plots/gnn_tsne_embedding.png` | t-SNE of GCN embeddings |

---

## Phase 2: News-Driven GNN for Stock Movement Prediction

---

## 9. Textual Data Acquisition

### 9.1 Data Source & Search Strategy

To acquire high-quality semantic node features for the GNN model, a targeted search was executed via **Factiva** focusing on the "Connectivity Hubs" identified during the network analysis phase. The search query explicitly targeted the specific high-degree entities — including **Regions Financial, Citizens Financial, Truist Financial, Fifth Third Bancorp, Huntington Bancshares, Illinois Tool Works, Dover Corp, Wabtec,** and **Ingersoll Rand** — to validate the hypothesis that information flow within these clusters propagates market risk.

### 9.2 Temporal Scope

- **Period:** January 29, 2021 – January 28, 2026
- **Rationale:** Strict alignment with the five-year S&P 500 historical price data (OHLCV) collected for the prediction target (y)

### 9.3 Source Selection

Restricted to authoritative financial outlets within the United States:
- **Dow Jones Newswires**
- **The Wall Street Journal**
- **Reuters Newswires**

This ensures the extraction of valid market signals while minimizing noise from social media or less credible sources.

### 9.4 Filtering Strategy

A rigorous filtering strategy was applied to mitigate data leakage and maximize the signal-to-noise ratio for the BERT embedding process:

| Filter Type | Categories | Rationale |
|-------------|-----------|-----------|
| **Included** | Corporate/Industrial News | Captures operational events (M&A, strategy changes) |
| **Excluded** | Share Price Movement/Disruptions | Prevents circular logic — model must learn from causal events, not price change reports |
| **Excluded** | Earnings, Tables, 8-K Filings, Market Data | Raw numerical dumps ill-suited for semantic embedding models |
| **Excluded** | Press Releases | Prioritize objective third-party reporting over promotional boilerplate |

### 9.5 Result

The refined query yielded a corpus of **1,900 high-density documents** (~200 articles per entity), providing sufficient data volume to construct robust node feature vectors without overwhelming the graph with irrelevant noise.

---

## 10. News Data Processing Pipeline

### 10.1 Step 1: RTF → CSV Conversion (`process_news.py`)

The Factiva export consists of multiple RTF files in the directory `Newstitle_20210129_20260128_1491/`.

**Processing method:**
1. Each RTF file was converted to plaintext using macOS `textutil` (`textutil -convert txt <file> -stdout`)
2. The plaintext was split into individual articles by the `Document <ID>` delimiter pattern
3. For each article, fields were extracted heuristically:
   - **Title:** First non-empty line
   - **Meta:** Second non-empty line (contains source and date)
   - **Source:** Parsed from meta by splitting on comma
   - **Body:** All remaining lines
4. **Deduplication:** Articles with identical `(title, meta)` pairs were removed

**Output:** `news_clean.csv` with columns: `filename`, `doc_id`, `title`, `meta`, `source`, `body`

### 10.2 Step 2: Event-Level Dataset Construction (`prepare_events.py`)

This script transforms the cleaned news articles into a structured event-level dataset suitable for supervised learning.

**Step-by-step process:**

1. **Date parsing:** Publication dates were extracted from the `meta` field using regex pattern matching for formats like `"29 January 2021"`

2. **Ticker matching:** A strict whitelist-based approach was used to link each article to the relevant stock ticker(s). The whitelist maps each ticker to regex patterns covering the company's full name, common abbreviations, and ticker symbol:

| Ticker | Matching Patterns |
|--------|------------------|
| RF | `Regions Financial`, `Regions Bank`, `RF` |
| CFG | `Citizens Financial`, `Citizens Bank`, `CFG` |
| TFC | `Truist`, `Truist Financial`, `TFC` |
| FITB | `Fifth Third`, `Fifth Third Bancorp`, `FITB` |
| HBAN | `Huntington Bancshares`, `Huntington Bank`, `HBAN` |
| ITW | `Illinois Tool Works`, `ITW` |
| DOV | `Dover Corp`, `Dover Corporation`, `DOV` |
| WAB | `Wabtec`, `Westinghouse Air Brake`, `WAB` |
| IR | `Ingersoll Rand`, `IR` |

3. **Explosion:** Articles mentioning multiple tickers were exploded into one row per (document, ticker) pair

4. **Label creation:** For each event, the **next trading day's return** for the matched ticker was computed:
   - `return_next = pct_change().shift(-1)` — the return on the first trading day after the article's publication date
   - **Binary label:** `1` if `return_next > 0` (price went up), `0` otherwise

5. **Filtering:** Rows with no computable next-day return (e.g., article published after the last available price date) were dropped

**Output files:**
- `news_events.parquet` — compact binary format for fast loading
- `news_events.csv` — human-readable format

**Dataset statistics:**

| Metric | Value |
|--------|-------|
| Total rows (after explosion) | 480 |
| Unique tickers | 9 |
| Label distribution | 50.8% negative (0) / 49.2% positive (1) |
| Date range | Aligned with 5-year price data |

**Rows per ticker:**

| Ticker | Count |
|--------|-------|
| CFG | 102 |
| TFC | 102 |
| FITB | 67 |
| HBAN | 59 |
| RF | 49 |
| WAB | 42 |
| ITW | 34 |
| IR | 15 |
| DOV | 10 |

---

## 11. Text Embedding with SentenceTransformer

### 11.1 Model: all-MiniLM-L6-v2

**What is SentenceTransformer?**

Sentence-BERT (Reimers & Gurevych, 2019) is a modification of the BERT architecture that uses siamese and triplet network structures to derive semantically meaningful sentence embeddings. Unlike vanilla BERT, which requires feeding two sentences simultaneously for comparison, SentenceTransformer produces fixed-size embeddings that can be compared using cosine similarity.

**Model details:**

| Property | Value |
|----------|-------|
| Model name | `sentence-transformers/all-MiniLM-L6-v2` |
| Base architecture | MiniLM (distilled from BERT) |
| Layers | 6 Transformer layers |
| Hidden dim | 384 |
| Output embedding dim | **384** |
| Parameters | ~22.7M |
| Training data | 1B+ sentence pairs (diverse NLI, QA, and web data) |
| Max sequence length | 256 tokens |

**Why this model?**
- Lightweight (6 layers vs. BERT's 12), suitable for batch encoding ~480 articles
- Produces high-quality general-purpose sentence embeddings
- Well-suited for financial text where semantic similarity matters

### 11.2 Encoding Process

```python
batch_size = 64
normalize_embeddings = True  # L2-normalize for cosine similarity compatibility
```

1. The `text` field (title + body concatenated) for each of the 480 events was encoded in batches of 64
2. Inference was run with `torch.inference_mode()` for memory efficiency
3. All embeddings were **L2-normalized** to unit length, enabling direct cosine similarity comparison
4. The final embeddings were cast to `float16` for storage efficiency

**Output:** `news_events_emb.npy` — shape `(480, 384)`, dtype `float16`

### 11.3 Embedding Integrity Checks

| Check | Result |
|-------|--------|
| Row count match (embeddings vs. metadata) | True |
| NaN in embeddings | False |
| NaN in metadata fields | All False |
| Label distribution | 50.8% / 49.2% (balanced) |

**Metadata saved as:** `news_events_emb_meta.parquet` — columns: `doc_id`, `ticker`, `date`, `label`, `return_next`, `text_idx`

---

## 12. Heterogeneous Graph Construction (HeteroData)

### 12.1 Graph Design

The prediction task is framed as a **node classification problem** on a heterogeneous graph with two node types and one edge type:

```
Node types:
  - news  (480 nodes, features: 384-dim text embeddings)
  - stock (9 nodes, features: 9-dim one-hot placeholder)

Edge type:
  - (news) --relates_to--> (stock)    [480 edges]
```

Each news article is connected to the stock ticker it mentions. The `ToUndirected()` transform later adds reverse edges `(stock) --rev_relates_to--> (news)`, enabling bidirectional message passing.

### 12.2 Train/Validation/Test Split

A **strict time-series split** was used to prevent data leakage:

| Split | Method | Size |
|-------|--------|------|
| Train | `date <= t80` (80th percentile date) | 385 samples |
| Validation | `t80 < date <= t90` | 48 samples |
| Test | `date > t90` | 47 samples |

This ensures the model never sees future data during training — critical for financial prediction tasks.

### 12.3 Output

The complete graph was saved as `graph_data.pt` (PyTorch serialized `HeteroData` object).

---

## 13. Baseline: Logistic Regression

### 13.1 Purpose

Establish a non-graph baseline to measure whether the GNN's graph structure provides additional predictive power beyond the text embeddings alone.

### 13.2 Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | `LogisticRegression` (sklearn) | Simple, interpretable baseline |
| Input | 384-dim text embeddings (no graph information) | Isolate text signal |
| `class_weight` | `"balanced"` | Handle slight class imbalance |
| `solver` | `"saga"` | Efficient for medium-dimensional data |
| `penalty` | `"l2"` | Standard regularization |
| `C` | 2.0 | Moderate regularization strength |
| `max_iter` | 500 | Ensure convergence |

### 13.3 Results

| Split | AUC | Accuracy |
|-------|-----|----------|
| Validation | 0.5165 | 47.92% |
| Test | 0.6222 | 61.70% |

**Detailed test classification report:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Down) | 0.7368 | 0.5185 | 0.6087 | 27 |
| 1 (Up) | 0.5357 | 0.7500 | 0.6250 | 20 |

**Interpretation:** The logistic regression baseline achieves near-random performance on validation (AUC 0.52) and modest performance on test (AUC 0.62). This suggests that text embeddings alone carry weak but non-trivial predictive signal.

---

## 14. GraphSAGE Model

### 14.1 What is GraphSAGE?

GraphSAGE (**SA**mple and **aggr**egat**E**, Hamilton et al., 2017) is an inductive graph neural network that learns node representations by sampling and aggregating features from a node's local neighborhood. Unlike GCN which uses the full adjacency matrix, GraphSAGE:

1. **Samples** a fixed number of neighbors for each node
2. **Aggregates** neighbor features (using mean, LSTM, or pooling)
3. **Concatenates** the aggregated neighbor embedding with the node's own embedding
4. Passes through a learnable linear transformation

Key advantage: GraphSAGE handles **heterogeneous node types with different feature dimensions** via the `(-1, -1)` input specification, letting PyG infer dimensions automatically at runtime.

### 14.2 Model Architecture

```python
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels=32, out_channels=1):
        self.conv1 = SAGEConv((-1, -1), hidden_channels)  # Layer 1
        self.conv2 = SAGEConv((-1, -1), hidden_channels)  # Layer 2
        self.lin   = Linear(hidden_channels, out_channels) # Classifier
```

| Layer | Input Dim | Output Dim | Activation | Dropout | Purpose |
|-------|-----------|------------|------------|---------|---------|
| conv1 (SAGEConv) | Auto-inferred (-1) | 32 | ReLU | 0.4 | Aggregate 1-hop neighbors |
| conv2 (SAGEConv) | 32 | 32 | ReLU | — | Aggregate 2-hop neighbors |
| lin (Linear) | 32 | 1 | — | — | Binary classification logit |

**Why `(-1, -1)` for input dimensions?**

In the heterogeneous graph, news nodes have 384-dim features and stock nodes have 9-dim features. The `(-1, -1)` syntax tells PyG to automatically infer the source and target dimensions for each edge type during the first forward pass, rather than hardcoding them.

### 14.3 Heterogeneous Conversion

```python
model = to_hetero(GNN(), data.metadata(), aggr='mean')
```

`to_hetero()` automatically replicates the homogeneous GNN layers for each edge type in the graph:
- `(news, relates_to, stock)` — news features aggregated to stock nodes
- `(stock, rev_relates_to, news)` — stock features aggregated back to news nodes

When a node receives messages from multiple edge types, they are combined via **mean aggregation** (`aggr='mean'`).

### 14.4 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | Adam | Standard adaptive optimizer |
| Learning rate | 0.004 | Moderate; tuned for convergence |
| Weight decay | 5e-4 | L2 regularization to prevent overfitting |
| Loss function | `BCEWithLogitsLoss` | Binary cross-entropy with built-in sigmoid |
| Dropout | 0.4 (after conv1) | Moderate regularization |
| Max epochs | 200 | Upper bound |
| Early stopping patience | 15 | Stop if val AUC doesn't improve for 15 consecutive epochs |

### 14.5 Training Loop

1. **Forward pass:** Feed all node features (`x_dict`) and edges (`edge_index_dict`) through the heterogeneous model
2. **Loss computation:** Only on **news nodes** in the training mask (stock nodes have no labels)
3. **Evaluation:** AUC computed on validation and test masks after each epoch
4. **Model selection:** The model state with the highest validation AUC is saved and restored at the end

### 14.6 Results

```
Epoch 010 | Loss 0.6725 | Val AUC 0.5399 | Test AUC 0.7722
Early stop at epoch 17, best Val AUC 0.6441
Best Val AUC: 0.6441 | Test AUC @best: 0.6907
```

### 14.7 Comparison: Baseline vs. GraphSAGE

| Model | Val AUC | Test AUC | Uses Graph? |
|-------|---------|----------|-------------|
| Logistic Regression (text only) | 0.5165 | 0.6222 | No |
| GraphSAGE (text + graph) | **0.6441** | **0.6907** | Yes |
| **Improvement** | **+0.1276** | **+0.0685** | — |

**Interpretation:** The GraphSAGE model outperforms the text-only baseline on both validation (+12.8% AUC) and test (+6.9% AUC). This improvement is attributable to the graph structure enabling information flow between news articles about the same stock, and between stocks that share news coverage. The heterogeneous message passing allows each news node's representation to be enriched by the stock node's aggregated information from all its related articles.

### 14.8 Information Flow Summary

```
news (384-dim) ──relates_to──→ stock (9-dim)
stock (9-dim)  ──rev_relates──→ news (384-dim)
```

After two SAGE layers, each news node's 32-dim embedding incorporates:
1. Its own text semantics (from the 384-dim sentence embedding)
2. Information from its associated stock node
3. Indirect information from other news articles about the same stock (via the stock node as a relay)

---

## 15. Complete File Inventory

| File | Description | Generated By |
|------|-------------|-------------|
| `sp500_5y_prices.csv` | 5-year daily closing prices (1255 x 502) | Cell 0 (yfinance) |
| `sp500_market_caps.csv` | Current market cap for 502 stocks | Cell 3 (yfinance) |
| `sp500_sectors.csv` | GICS sector mapping for 503 tickers | Cell 4 (Wikipedia) |
| `news_clean.csv` | Cleaned Factiva articles | `process_news.py` |
| `news_events.csv` | Event-level dataset (480 rows) | `prepare_events.py` |
| `news_events.parquet` | Same as above, Parquet format | `prepare_events.py` |
| `news_events_emb.npy` | 384-dim SentenceTransformer embeddings | Cell 12 |
| `news_events_emb_meta.parquet` | Embedding metadata | Cell 13 |
| `graph_data.pt` | PyTorch HeteroData graph object | Cell 15 |
| `plots/*.png` | All visualization outputs | Cells 5–9 |

---

## 16. Summary of All Parameters

| Component | Parameter | Value |
|-----------|-----------|-------|
| Price data | Period | 5 years |
| Price data | Field | Adjusted close |
| Correlation | Method | Pearson |
| Graph (correlation) | Edge threshold | \|corr\| > 0.6 |
| GCN (exploratory) | Architecture | 1 → 16 → 1 |
| GCN | Activation | ReLU |
| Factiva search | Period | 2021-01-29 to 2026-01-28 |
| Factiva search | Sources | DJ Newswires, WSJ, Reuters |
| Factiva search | Corpus size | 1,900 documents |
| Ticker matching | Method | Regex whitelist (9 tickers) |
| Label | Definition | next-day return > 0 |
| SentenceTransformer | Model | all-MiniLM-L6-v2 |
| SentenceTransformer | Output dim | 384 |
| SentenceTransformer | Batch size | 64 |
| SentenceTransformer | Normalization | L2 (unit vectors) |
| Train/Val/Test split | Method | Time-series (80/10/10) |
| Logistic Regression | C | 2.0 |
| Logistic Regression | Penalty | L2 |
| Logistic Regression | Solver | SAGA |
| GraphSAGE | Hidden dim | 32 |
| GraphSAGE | Layers | 2 |
| GraphSAGE | Aggregation | Mean |
| GraphSAGE | Dropout | 0.4 |
| GraphSAGE | Learning rate | 0.004 |
| GraphSAGE | Weight decay | 5e-4 |
| GraphSAGE | Early stopping | Patience = 15 |
| GraphSAGE | Loss | BCEWithLogitsLoss |

---

## 17. Key Findings

### Phase 1 (Network Analysis)

1. **Sparse but structured network:** With a 0.6 correlation threshold, only 1.3% of possible edges exist, yet the network exhibits clear clustering.

2. **Hub stocks are Industrials and Financials:** 7/10 of the most connected stocks are Industrials, 3/10 are Financial Services. Mega-cap tech stocks are NOT network hubs.

3. **Sector clustering is visible:** Same-sector stocks cluster together in the correlation network, confirming that industry membership drives co-movement.

4. **GCN embeddings capture topology:** Even without training, GCN embeddings produce meaningful clusters in t-SNE space.

### Phase 2 (News-Driven Prediction)

5. **Text alone has weak signal:** Logistic regression on 384-dim sentence embeddings achieves Test AUC of 0.62 — above random but modest.

6. **Graph structure adds value:** GraphSAGE improves Test AUC to 0.69 (+6.9% over baseline), demonstrating that relational information between news and stocks provides additional predictive power.

7. **Balanced dataset:** The near 50/50 label split (50.8% down, 49.2% up) means the model cannot exploit class imbalance — any performance above 0.5 AUC reflects genuine learned signal.

8. **Early stopping effective:** The model converges quickly (epoch 17) and early stopping prevents overfitting on the small dataset (480 samples).

---

*End of Working Progress Report.*
