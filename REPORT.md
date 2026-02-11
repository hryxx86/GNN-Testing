# GNN-Based Stock Correlation Network Analysis — Progress Report

## Project Overview

This project applies Graph Neural Networks (GNNs) to model and analyze the correlation structure among S&P 500 stocks. The goal is to construct a stock correlation network from historical price data. Then, GNN architectures such as GCN and GraphSAGE are used to learn node-level embeddings that capture the relationships between stocks. The entire project was developed using Python 3.12. The computing environment includes Google Colab with GPU runtime and a local macOS machine. The core libraries used are PyTorch, PyTorch Geometric (PyG), yfinance, pandas, networkx, matplotlib, and scikit-learn.

## Phase 1: Network Analysis

### Data Collection

The first step was collecting stock price data. The data source is Yahoo Finance, accessed through the yfinance Python library. The stock universe covers all S&P 500 constituent stocks. The list of tickers was scraped from the Wikipedia S&P 500 page. We downloaded 5 years of daily data using the adjusted closing prices only (2021/01/29 to 2026/01/28). During preprocessing, any stocks with entirely missing data were dropped. The resulting dataset contains 1,255 trading days across 502 stocks. This was saved as sp500_5y_prices.csv. To avoid repeated downloads, the CSV is cached locally, and subsequent runs load directly from disk.

Next, we collected market capitalization data. For each of the 502 stocks, we called the yfinance Ticker .info["marketCap"] field individually. If the API call failed for any ticker, that ticker was assigned a market cap of 0. In the end, all 502 entries returned non-zero values. The results were saved as sp500_market_caps.csv. The top 5 stocks by market cap (as of 2026/02/09) are NVDA at $4.63 trillion, AAPL at $4.04 trillion, GOOG at $3.92 trillion, GOOGL at $3.92 trillion, and MSFT at $3.07 trillion.

We also collected GICS sector data from the Wikipedia S&P 500 table. A single HTTP request was sent to parse the HTML table and extract the mapping from stock symbol to GICS sector. Some tickers had dots in their symbols (e.g., BRK.B), which were converted to dash format (e.g., BRK-B) for consistency with yfinance. The final result maps 503 tickers to 11 GICS sectors. These sectors are Communication Services, Consumer Discretionary, Consumer Staples, Energy, Financials, Health Care, Industrials, Information Technology, Materials, Real Estate, and Utilities. The mapping was saved as sp500_sectors.csv.

### Graph Construction

With the price data in hand, we proceeded to build the correlation graph. First, daily returns were computed from closing prices using percentage change. The first row, which contains NaN values, was dropped. Then, a Pearson correlation matrix was computed across all 502 stocks using the full 5-year return series. This produced a 502 x 502 matrix. An edge was created between two stocks whenever their absolute correlation exceeded a threshold of 0.6. The resulting edge index was stored in PyG's COO sparse format.

The constructed graph has 502 nodes (stocks) and 3,198 edges (connections). The edge density is 0.0127, or about 1.27%. The graph is undirected and unweighted. The low density tells us that most stock pairs are not strongly correlated. Only a select subset of stocks form tight clusters.

### GCN Model (Exploratory)

We then built a Graph Convolutional Network (GCN) for exploratory purposes. GCN, proposed by Kipf and Welling in 2017, generalizes convolution from grid-structured data like images to graph-structured data. Each GCN layer performs neighborhood aggregation. A node's new representation is computed by averaging its neighbors' features plus its own, followed by a linear transformation and a non-linear activation.

The model architecture has two layers. The first layer (conv1) takes 1 input feature and produces 16 hidden dimensions with ReLU activation. Its purpose is to aggregate 1-hop neighbor features. The second layer (conv2) takes 16 dimensions and outputs 1 dimension with no activation. It aggregates 2-hop neighbor features. The input feature for each node is simply the most recent day's return, reshaped to a single scalar per stock. This is a minimal feature, chosen to demonstrate the GCN's ability to propagate information across the graph.

We ran a single forward pass with no training or optimization. The purpose was to verify that the GCN architecture works end-to-end on the constructed graph. The output shape is 502 x 1, meaning one value per stock. Even without training, this forward pass shows that each stock's output is influenced by its correlated neighbors up to 2 hops away. The GCN effectively smooths information across the network. Stocks with similar correlation profiles produce similar output values.

### Visualizations

Several visualizations were produced to explore the network.

The first visualization shows the correlation network of the top 100 stocks by market capitalization. Only edges where both endpoints are in the top 100 subset were included. The layout uses a spring (force-directed) algorithm with seed 42 and k=0.15. Nodes are labeled with their ticker symbols. This was saved as plots/S&P 500 Correlation Network (Top 100 Market Cap).png.

The second visualization is the same top-100 graph, but with nodes colored by GICS sector using a tab20 colormap. A legend in the upper-left corner maps each color to its sector name. The purpose is to visually check whether stocks in the same industry cluster together. This was saved as plots/S&P 500 Top 100 Sector Colored.png.

The third visualization displays the full S&P 500 network with all 502 nodes and 3,198 edges. Labels are turned off because the graph is too dense for 500+ node labels. Node size is set to 100, with alpha at 0.6 for semi-transparency. The canvas is 20x20 inches with a spring layout using k=0.05 for tighter clustering. This gives an overview of the entire network structure, revealing the overall connectivity pattern and isolated nodes. The plot was saved as plots/sp500_all_corr.png.

The fourth visualization focuses on the top 10 hub stocks and their neighbors. Node degree (number of edges) was computed for all 502 stocks using PyG's degree utility. The top 10 by degree were selected. The subgraph includes any edge where at least one endpoint is a top-10 hub, along with all connected nodes. Hub stocks are shown as large red nodes (size 1000), and neighbor stocks are shown as small blue nodes (size 100). The top 10 hubs are SWK (50 connections), ITW (39), MAS (36), DOV (33), IR (32), TFC (32), CFG (31), ODFL (31), RF (30), and WAB (30). Notably, 7 of the top 10 hubs are Industrials stocks, and the remaining 3 are Financials. This means these two sectors have the strongest internal correlation structure. Mega-cap tech stocks like AAPL, MSFT, and NVDA do not appear as hubs. High market cap does not equate to high network centrality. This plot was saved as plots/top10_hubs_network.png.

The fifth visualization is a t-SNE plot of the GCN embeddings. The embeddings were extracted from the output of conv1, the first GCN layer, giving a 16-dimensional embedding for each stock. t-SNE was then applied with n_components=2, perplexity=30, PCA initialization, and auto learning rate. The result is a 2D scatter plot of all 502 stocks. Proximity in this plot indicates similar GCN-learned representations. The purpose is to visualize how the GCN organizes stocks in its latent space after one layer of neighborhood aggregation. This was saved as plots/gnn_tsne_embedding.png.

### Phase 1 Parameters and Choices

To summarize the key parameters in Phase 1: the price data covers 5 years using adjusted close prices, which accounts for splits and dividends. The correlation method is Pearson, standard for measuring linear co-movement. The graph edge threshold is |correlation| > 0.6, chosen to balance connectivity and sparsity at about 1.3% density. The GCN uses a hidden dimension of 16, which is lightweight but sufficient for structural exploration. It has 2 layers to capture 2-hop neighborhood information. The activation function is ReLU. The input feature is the last day's return with dimension 1, chosen as a minimal feature to demonstrate graph propagation. The t-SNE perplexity is 30, a standard default. Market cap data comes from yfinance's .info field. Sector data comes from Wikipedia.

### Phase 1 Preliminary Findings

Four key findings emerged from Phase 1. First, the network is sparse but structured. With a 0.6 threshold, only 1.3% of possible edges exist, yet the network shows clear clustering. Second, hub stocks exist. A small number of stocks have disproportionately high connectivity, suggesting they are bellwether stocks correlated with many others. Third, sector clustering is visible. When coloring the top-100 network by GICS sector, same-sector stocks tend to cluster together. This confirms that industry membership drives correlation structure. Fourth, GCN embeddings capture structure. Even with a single scalar input and no training, the first-layer GCN embeddings produce meaningful clusters in t-SNE space. The network topology alone carries informative signal.

### Phase 1 Files Produced

The files produced in Phase 1 are: sp500_5y_prices.csv (5-year daily closing prices, 1255 x 502), sp500_market_caps.csv (current market cap for 502 stocks), sp500_sectors.csv (GICS sector mapping for 503 tickers), and five plot files including the top-100 network, the sector-colored network, the full 502-stock network, the hub stocks visualization, and the t-SNE embedding plot.

## Phase 2: News-Driven GNN for Stock Movement Prediction

### Textual Data Acquisition

To build semantic node features for the GNN, we conducted a targeted search via Factiva. The search focused on the connectivity hubs identified during Phase 1's network analysis. The query targeted specific high-degree entities, including Regions Financial, Citizens Financial, Truist Financial, Fifth Third Bancorp, Huntington Bancshares, Illinois Tool Works, Dover Corp, Wabtec, and Ingersoll Rand. The goal was to validate the hypothesis that information flow within these clusters propagates market risk.

The search period is January 29, 2021, to January 28, 2026. This was strictly aligned with the five-year S&P 500 price data used for the prediction target. Sources were restricted to authoritative U.S. financial outlets: Dow Jones Newswires, The Wall Street Journal, and Reuters Newswires. This ensures valid market signals while minimizing noise from social media or less credible sources.

A rigorous filtering strategy was applied. Corporate and industrial news was included to capture operational events like M&A and strategy changes. Share price movement and disruption reports were excluded to prevent circular logic, since the model must learn from causal events rather than price change reports. Earnings tables, 8-K filings, and market data were excluded because raw numerical dumps are ill-suited for semantic embedding. Press releases were also excluded to prioritize objective third-party reporting over promotional content. The refined query produced a corpus of 1,900 high-density documents, roughly 200 articles per entity.

### News Data Processing Pipeline

The Factiva export came as multiple RTF files in a directory. The first processing step, handled by process_news.py, converted these RTF files to CSV. Each RTF file was converted to plaintext using macOS textutil. The plaintext was then split into individual articles by the Document ID delimiter pattern. For each article, the title was extracted as the first non-empty line, the meta line as the second (containing source and date), the source was parsed from the meta by splitting on comma, and the body was all remaining lines. Articles with identical title-meta pairs were deduplicated. The output is news_clean.csv with columns: filename, doc_id, title, meta, source, and body.

The second processing step, handled by prepare_events.py, transformed the cleaned articles into a structured event-level dataset for supervised learning. Publication dates were extracted from the meta field using regex pattern matching for formats like "29 January 2021." Ticker matching used a strict whitelist approach. Each ticker was mapped to regex patterns covering the company's full name, common abbreviations, and ticker symbol. For example, RF matches "Regions Financial," "Regions Bank," or "RF." CFG matches "Citizens Financial," "Citizens Bank," or "CFG." Similar patterns were defined for TFC (Truist), FITB (Fifth Third), HBAN (Huntington Bancshares), ITW (Illinois Tool Works), DOV (Dover Corp), WAB (Wabtec), and IR (Ingersoll Rand).

Articles mentioning multiple tickers were exploded into one row per document-ticker pair. For each event, the next trading day's return was computed. The binary label is 1 if the next-day return is positive (price went up) and 0 otherwise. Rows with no computable next-day return were dropped. The output was saved as both news_events.parquet (compact binary format) and news_events.csv (human-readable format).

The final dataset has 480 rows covering 9 unique tickers. The label distribution is nearly balanced at 50.8% negative and 49.2% positive. The date range aligns with the 5-year price data. By ticker, CFG has 102 rows, TFC has 102, FITB has 67, HBAN has 59, RF has 49, WAB has 42, ITW has 34, IR has 15, and DOV has 10.

### Text Embedding with SentenceTransformer

To encode the news text into numerical features, we used the SentenceTransformer model all-MiniLM-L6-v2. Sentence-BERT, proposed by Reimers and Gurevych in 2019, modifies the BERT architecture using siamese and triplet network structures to produce semantically meaningful sentence embeddings. Unlike vanilla BERT, which requires feeding two sentences simultaneously, SentenceTransformer produces fixed-size embeddings that can be compared using cosine similarity.

The specific model used is sentence-transformers/all-MiniLM-L6-v2. It is based on MiniLM, a distilled version of BERT. It has 6 Transformer layers, a hidden dimension of 384, and about 22.7 million parameters. The output embedding dimension is 384. It was trained on over 1 billion sentence pairs from diverse NLI, QA, and web data. The maximum sequence length is 256 tokens. We chose this model because it is lightweight (6 layers versus BERT's 12), suitable for batch encoding about 480 articles. It produces high-quality general-purpose embeddings and is well-suited for financial text where semantic similarity matters.

The encoding was done in batches of 64. The text field for each event, consisting of the title and body concatenated, was processed with torch inference mode for memory efficiency. All embeddings were L2-normalized to unit length, enabling direct cosine similarity comparison. The final embeddings were cast to float16 for storage efficiency. The output is news_events_emb.npy with shape 480 x 384 in float16.

Integrity checks confirmed that the row count matches between embeddings and metadata, there are no NaN values in the embeddings, and no NaN values in the metadata fields. The label distribution remains 50.8% / 49.2%, confirming balance. The metadata was saved as news_events_emb_meta.parquet with columns: doc_id, ticker, date, label, return_next, and text_idx.

### Heterogeneous Graph Construction

The prediction task is framed as a node classification problem on a heterogeneous graph. The graph has two node types: news (480 nodes with 384-dimensional text embeddings as features) and stock (9 nodes with 9-dimensional one-hot placeholder features). There is one edge type: news relates_to stock, with 480 edges total. Each news article is connected to the stock ticker it mentions. The ToUndirected() transform adds reverse edges (stock rev_relates_to news), enabling bidirectional message passing.

A strict time-series split was used to prevent data leakage. The training set includes all samples with dates up to the 80th percentile, totaling 385 samples. The validation set covers dates between the 80th and 90th percentiles, with 48 samples. The test set includes dates beyond the 90th percentile, with 47 samples. This ensures the model never sees future data during training, which is critical for financial prediction. The complete graph was saved as graph_data.pt.

### Baseline: Logistic Regression

Before running the GNN, a logistic regression baseline was established. The purpose was to measure whether the GNN's graph structure provides additional predictive power beyond text embeddings alone. The model uses sklearn's LogisticRegression with 384-dimensional text embeddings as input and no graph information. Class weights were set to "balanced" to handle slight class imbalance. The solver is SAGA, which is efficient for medium-dimensional data. L2 penalty was applied with C=2.0 for moderate regularization. Max iterations were set to 500 to ensure convergence.

The baseline results show a validation AUC of 0.5165 and accuracy of 47.92%. On the test set, the AUC is 0.6213 with accuracy of 61.70%. For the detailed test classification report, Class 0 (Down) has precision 0.7368, recall 0.5185, F1 0.6087, with 27 support samples. Class 1 (Up) has precision 0.5357, recall 0.7500, F1 0.6250, with 20 support samples. The logistic regression achieves near-random performance on validation and modest performance on test. This suggests text embeddings alone carry weak but non-trivial predictive signals.

### GraphSAGE Model

The main GNN model used is GraphSAGE, proposed by Hamilton et al. in 2017. GraphSAGE stands for SAmple and aggrEgatE. It is an inductive graph neural network that learns node representations by sampling and aggregating features from a node's local neighborhood. Unlike GCN, which uses the full adjacency matrix, GraphSAGE samples a fixed number of neighbors, aggregates their features using mean, LSTM, or pooling, concatenates the aggregated neighbor embedding with the node's own embedding, and passes it through a learnable linear transformation. A key advantage is that GraphSAGE handles heterogeneous node types with different feature dimensions via the (-1, -1) input specification, letting PyG infer dimensions automatically at runtime.

The model architecture has three components. The first layer (conv1) is a SAGEConv with auto-inferred input dimensions and 32 output dimensions, followed by ReLU activation and 0.4 dropout. It aggregates 1-hop neighbors. The second layer (conv2) is another SAGEConv from 32 to 32 dimensions with ReLU activation. It aggregates 2-hop neighbors. The final layer is a linear layer from 32 to 1, producing a binary classification logit. The (-1, -1) input specification is used because in the heterogeneous graph, news nodes have 384-dimensional features while stock nodes have 9-dimensional features. This syntax tells PyG to infer the correct dimensions during the first forward pass.

The model was converted to a heterogeneous model using the to_hetero() function, which automatically replicates the GNN layers for each edge type. The two edge types are news relates_to stock (news features aggregated to stock nodes) and stock rev_relates_to news (stock features aggregated back to news nodes). When a node receives messages from multiple edge types, they are combined via mean aggregation.

The training configuration uses the Adam optimizer with a learning rate of 0.004 and weight decay of 5e-4 for L2 regularization. The loss function is BCEWithLogitsLoss, which combines binary cross-entropy with a built-in sigmoid. Dropout is 0.4 after the first convolutional layer. The maximum number of epochs is 200, with early stopping patience of 15 epochs based on validation AUC. During each epoch, a forward pass feeds all node features and edges through the model. Loss is computed only on news nodes in the training mask, since stock nodes have no labels. AUC is evaluated on validation and test masks after each epoch. The model state with the highest validation AUC is saved and restored at the end.

The results may vary depending on the hardware used (CPU vs. GPU) and the CUDA version, due to non-deterministic floating-point operations in CuBLAS. The training results show that by epoch 10, the loss was 0.6768, validation AUC was 0.4948, and test AUC was 0.6167. By epoch 20, the loss dropped to 0.6496, validation AUC was 0.4531, and test AUC was 0.7167. Early stopping triggered at epoch 21. The best validation AUC was 0.6111, and the corresponding test AUC was 0.6426.

### Baseline vs. GraphSAGE Comparison

Comparing the two models, logistic regression achieves a validation AUC of 0.5165 and test AUC of 0.6213 without using graph structure. GraphSAGE achieves a validation AUC of 0.6111 and test AUC of 0.6426, using both text and graph. The improvement is +0.0946 on validation (+18.3%) and +0.0213 on test (+3.4%). The GraphSAGE model outperforms the text-only baseline on both splits. This improvement comes from the graph structure enabling information flow between news articles about the same stock and between stocks sharing news coverage. The heterogeneous message passing allows each news node's representation to be enriched by the stock node's aggregated information from all its related articles.

### Information Flow in the Model

After two SAGE layers, each news node's 32-dimensional embedding incorporates three sources of information. First, its own text semantics from the 384-dimensional sentence embedding. Second, information from its associated stock node. Third, indirect information from other news articles about the same stock, routed through the stock node as a relay. The flow goes from news (384-dim) to stock (9-dim) via the relates_to edge, and from stock (9-dim) back to news (384-dim) via the reverse edge.

## Complete File Inventory

The complete list of files produced across both phases is as follows. sp500_5y_prices.csv contains 5-year daily closing prices (1255 x 502), generated by yfinance. sp500_market_caps.csv contains current market cap for 502 stocks, also from yfinance. sp500_sectors.csv contains the GICS sector mapping for 503 tickers, from Wikipedia. news_clean.csv contains the cleaned Factiva articles, produced by process_news.py. news_events.csv and news_events.parquet contain the event-level dataset with 480 rows, produced by prepare_events.py. news_events_emb.npy contains the 384-dimensional SentenceTransformer embeddings. news_events_emb_meta.parquet contains the embedding metadata. graph_data.pt contains the PyTorch HeteroData graph object. All visualization outputs are stored in the plots/ directory.

## Summary of All Parameters

For completeness, here is a recap of every parameter used. The price data covers 5 years of adjusted close prices. Correlation uses the Pearson method. The correlation graph uses an edge threshold of |corr| > 0.6. The exploratory GCN has architecture 1 -> 16 -> 1 with ReLU activation. The Factiva search covers January 29, 2021, to January 28, 2026, from DJ Newswires, WSJ, and Reuters, yielding 1,900 documents. Ticker matching uses a regex whitelist for 9 tickers. The label is defined as next-day return > 0. The SentenceTransformer model is all-MiniLM-L6-v2 with 384-dimensional output, batch size 64, and L2 normalization. The train/val/test split follows a time-series method at 80/10/10. Logistic regression uses C=2.0, L2 penalty, and SAGA solver. GraphSAGE uses 32 hidden dimensions, 2 layers, mean aggregation, 0.4 dropout, learning rate 0.004, weight decay 5e-4, early stopping patience of 15, and BCEWithLogitsLoss.

## Key Findings

In Phase 1, four findings emerged. The network is sparse but structured, with only 1.3% of edges existing at a 0.6 threshold, yet clear clustering is present. Hub stocks are dominated by Industrials (7 out of 10) and Financials (3 out of 10), while mega-cap tech stocks are not network hubs. Sector clustering is visible, confirming that industry membership drives co-movement. GCN embeddings capture topology, producing meaningful clusters in t-SNE space even without training.

In Phase 2, four more findings emerged. Text alone has weak signal, with logistic regression achieving a test AUC of only 0.62. Graph structure adds value, with GraphSAGE improving test AUC to 0.64, a gain of 3.4% over the baseline. The dataset is well balanced at roughly 50/50, meaning any performance above 0.5 AUC reflects genuine learned signal. Early stopping proved effective, with the model converging quickly at epoch 21 and preventing overfitting on the small dataset of 480 samples.
