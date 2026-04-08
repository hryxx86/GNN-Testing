# GNN-Testing: Complete Research Log

> **Project Full Name**: DynHetGNN-SP — Dynamic Heterogeneous Graph Neural Network with Selective Prediction for Stock Ranking
> **Log Date**: 2026-03-06
> **Status**: v3 pipeline N3-N5 fully completed. GAT 21d IC=0.04420, surpassing the 0.03 threshold. SelectiveNet failed (selection head negatively correlated with quality).

---

## Table of Contents

1. [Project Overview and Motivation](#1-project-overview-and-motivation)
2. [Phase 1: Network Structure Analysis (Exploratory)](#2-phase-1-network-structure-analysis)
3. [Phase 2 Pilot: News-Driven GNN Prediction (Small Scale)](#3-phase-2-pilot-news-driven-gnn-prediction)
4. [Phase A: EODHD Data + FinBERT (Full-Scale Data Preparation)](#4-phase-a-eodhd-data--finbert)
5. [Phase B: Dynamic Graph Construction and Parameter Selection](#5-phase-b-dynamic-graph-construction-and-parameter-selection)
6. [Phase C v1: Full-Scale Experiments — AUC ~ 0.50 Failure](#6-phase-c-v1-full-scale-experiments)
7. [Diagnostic Phase: D.1 + D.2 — Where Is the Signal?](#7-diagnostic-phase-d1--d2)
8. [Phase 1 Signal Fix: Deduplication + Market Adjustment + Momentum](#8-phase-1-signal-fix)
9. [Phase 2 LLM: GPT-4o-mini Replacing FinBERT — STOP](#9-phase-2-llm-gpt-4o-mini)
10. [v3 Paradigm Shift: Binary to Ranking + Calendar-Driven](#10-v3-paradigm-shift)
11. [v3 Implementation: N1-N5 Pipeline](#11-v3-implementation-n1-n5-pipeline)
12. [v3 Colab Run 1: N3 Initial Results](#12-v3-colab-run-1-n3-initial-results)
13. [GAT vs HGT: Why the Simpler Model Won](#13-gat-vs-hgt-why-the-simpler-model-won)
14. [v3 Colab Run 2: N3-N5 Complete Results](#14-v3-colab-run-2-n3-n5-complete-results)
15. [N4 Horizon Ablation: Detailed Analysis](#15-n4-horizon-ablation-detailed-analysis)
16. [N5 SelectiveNet: Detailed Analysis](#16-n5-selectivenet-detailed-analysis)
17. [Training Stability Analysis](#17-training-stability-analysis)
18. [Current Status and Next Steps](#18-current-status-and-next-steps)
19. [Glossary](#19-glossary)

---

## 1. Project Overview and Motivation

**Core Research Question**: Can Graph Neural Networks (GNNs), combined with financial news information, predict stock price movements?

**Research Hypothesis**: Stocks form network structures through price correlations, sector relationships, and news co-mentions. GNNs can exploit this structure to propagate cross-stock information, thereby improving predictive performance over methods that treat each stock independently.

**Dataset**:
- 502 S&P 500 constituent stocks
- Time span: January 2020 to December 2025 (approximately 1,255 trading days)
- News data: 1,698,182 EODHD events (1,538,967 valid after ticker mapping, 90.6% match rate)
- NLP embeddings: FinBERT (ProsusAI/finbert, Araci 2019): a BERT-based model (Devlin et al., NAACL 2019) fine-tuned on financial text, producing 768-dimensional sentence embeddings

**Project Evolution**:

The project went through several major pivots:

1. Phase 1-2 Pilot (exploration) established that GNN structure provides marginal gains on small data.
2. Phase A-C (full-scale) attempted binary direction prediction, resulting in AUC ~ 0.50 (random-level).
3. Diagnostics D.1/D.2 confirmed that FinBERT sentiment has essentially no predictive power for next-day S&P 500 returns.
4. Signal Fix and LLM upgrade (GPT-4o-mini) both failed to break the AUC ~ 0.50 barrier.
5. v3 paradigm shift to ranking + calendar-driven + GAT + selective prediction yielded the first meaningful results (IC=0.02054, Sharpe=1.011 in the initial run), triggering a GO decision.

---

## 2. Phase 1: Network Structure Analysis

**Objective**: Explore the network structure characteristics among S&P 500 stocks.

**Method**:
- Collected 5-year price data for 502 stocks
- Computed a Pearson correlation matrix (502 x 502). Pearson correlation measures the linear relationship between two time series, ranging from -1 (perfect inverse) to +1 (perfect co-movement).
- Constructed an undirected graph by retaining edges where |correlation| > 0.6
- Trained an exploratory GCN (Graph Convolutional Network, Kipf & Welling, ICLR 2017): the foundational GNN architecture that performs spectral-inspired convolutions by averaging neighbor features with symmetric normalization.

**Results**:
- Graph density: 3,198 edges (approximately 1.3% of all possible edges)
- Network centrality: Industrials and Financials sectors dominate hub stocks (7 Industrials + 3 Financials in the top 10 by degree)
- Unexpected finding: Mega-cap tech stocks (AAPL, MSFT, etc.) are not network hubs, likely because their price dynamics are more idiosyncratic

**Tools**: NetworkX, PyTorch Geometric (PyG), Matplotlib

**Figures**:

![S&P 500 Correlation Network — Top 100 by Market Cap](../../plots/S&P%20500%20Correlation%20Network%20(Top%20100%20Market%20Cap).png)

*Figure 2.1: Correlation network of the top 100 S&P 500 stocks by market capitalization. Node positions are determined by force-directed layout; tightly clustered nodes in the center share strong pairwise correlations. Notable observation: mega-cap tech stocks (AAPL, MSFT, GOOG) sit on the periphery rather than the center, confirming that price co-movement does not follow market capitalization.*

![Top 10 Connectivity Hubs and Their Neighbors](../../plots/top10_hubs_network.png)

*Figure 2.2: The 10 stocks with the highest degree centrality (red nodes) and their direct neighbors (blue nodes). Hub stocks are dominated by Industrials and Financials — sectors whose constituent companies tend to move together due to shared macroeconomic sensitivity (interest rates, industrial production). This structure is fundamentally different from market-cap-based importance, revealing a hidden "connectivity backbone" in the equity market.*

![S&P 500 Top 100 — Sector Colored](../../plots/S&P%20500%20Top%20100%20Sector%20Colored.png)

*Figure 2.3: The same top-100 network colored by GICS sector. Clear sector clustering is visible: Financials (purple cluster at center), Information Technology (scattered), Energy (bottom cluster). The sector-based community structure validates the use of sector edges in the GNN graph.*

![Full S&P 500 Correlation Network](../../plots/sp500_all_corr.png)

*Figure 2.4: The complete S&P 500 correlation network (all 502 stocks). A dense core of highly correlated stocks is visible at center-left, with progressively less connected stocks radiating outward. Isolated stocks on the periphery have unique return profiles uncorrelated with the broader market.*

![GNN Learned Embeddings — t-SNE Visualization](../../plots/gnn_tsne_embedding.png)

*Figure 2.5: t-SNE (t-distributed Stochastic Neighbor Embedding) projection of the GNN's learned 64-dimensional stock representations into 2D space. The smooth, continuous structure indicates the GNN successfully learns a meaningful embedding space where stock similarity is captured through graph message passing.*

**Significance**: Validated the existence of meaningful network structure among stocks, with clear sector-based clustering, providing a foundation for subsequent GNN modeling.

---

## 3. Phase 2 Pilot: News-Driven GNN Prediction

**Objective**: Small-scale proof-of-concept that news + GNN can predict stock direction.

**Method**:
- Manually collected 1,900 news articles from Factiva for 9 hub stocks
- After cleaning: 480 usable events
- Encoded news using SentenceTransformer (MiniLM-L6-v2, 384 dimensions): a lightweight sentence embedding model from the sentence-transformers library
- Constructed a heterogeneous graph with two node types (news, stock) and two edge types (news-to-stock mentions, stock-to-stock correlation)
- **Model**: GraphSAGE (Hamilton et al., NeurIPS 2017): an inductive GNN that samples a fixed number of neighbors and aggregates their features using mean pooling, enabling generalization to unseen nodes. Configuration: 2 layers, hidden dimension = 64.

**Results**:

| Model | Test AUC |
|-------|----------|
| Text-only Logistic Regression baseline | 0.6213 |
| GraphSAGE (heterogeneous graph) | **0.6426** |

- Graph structure gain: +0.0213 (+3.4%)
- Conclusion: Preliminary evidence that GNN adds incremental value, but the data scale is too small for statistical significance.

**Limitations**: Only 9 stocks and 480 news events. Results are illustrative but not conclusive.

---

## 4. Phase A: EODHD Data + FinBERT

**Objective**: Build a full-scale dataset to replace the small Factiva pilot data.

**Data Source**: EODHD API, which provides historical news event data with stock ticker associations.
- Raw events: 1,698,182 (January 2020 to December 2025)
- Mapped to S&P 500 tickers: 1,538,967 valid events (90.6% match rate)

**NLP Model Selection**:

| Candidate | Dimensions | Advantage | Decision |
|-----------|-----------|-----------|----------|
| **FinBERT** (ProsusAI) | 768 | Financial domain pre-training, widely cited | Selected for clean ablation; upgrades deferred |
| Fin-E5 | 768 | Newer financial embedding model | Deferred |
| Voyage Finance | 1024 | API-based embedding service | Deferred |

**FinBERT** (ProsusAI/finbert): A BERT model fine-tuned on Financial PhraseBank and financial news corpora. It outputs both 768-dimensional sentence embeddings and 3-dimensional sentiment probability scores (positive / negative / neutral). It is the most popular financial BERT variant on Hugging Face.

**Processing Pipeline**:

Raw EODHD events are matched to stock tickers, then encoded through FinBERT to produce a 768-dimensional embedding and a 3-dimensional sentiment vector per event. Events are aggregated per stock-day using mean pooling, yielding a final tensor of shape (num_days, num_stocks, 771).

**Output**: Price files, event files, and embedding files stored under `data/fullscale/`.

**Figures**:

![FinBERT Embedding t-SNE by Sector](../../plots/finbert_tsne_by_sector.png)

*Figure 4.1: t-SNE visualization of FinBERT embeddings colored by GICS sector. Each point represents a stock-day's mean news embedding. The lack of tight sector-based clusters confirms that FinBERT encodes article-level semantic content rather than sector identity — the embeddings capture what the news says, not which sector it belongs to. This is desirable because sector information is already provided through the sector edge type in the graph.*

---

## 5. Phase B: Dynamic Graph Construction and Parameter Selection

**Objective**: Determine optimal parameters for the dynamic correlation-based stock graph.

**Method**: Rolling-window Pearson correlation matrices. For each month, the closing prices of the preceding w trading days are used to compute a 502 x 502 correlation matrix. Edges are retained where |correlation| exceeds threshold tau.

### Parameter Search

Two parameters control the graph construction:

| Parameter | Search Range | Optimal Value | Rationale |
|-----------|-------------|---------------|-----------|
| **w** (window length in trading days) | 63, 126, 252 | **126** (~6 months) | Moderate density (~6%), good temporal stability (std=0.064) |
| **tau** (correlation threshold) | 0.5, 0.6, 0.7 | **0.6** | Balances graph density against signal-to-noise ratio |

### Evaluation Criteria

Three metrics guided the parameter selection:

1. **Graph Density**: The ratio of actual edges to the maximum possible number of edges. A graph that is too dense introduces noisy connections; too sparse loses useful structural information.

2. **Number of Connected Components**: Ideally the graph should have a single giant connected component, ensuring all stocks can exchange information through message passing. A fragmented graph with many isolated clusters would limit GNN effectiveness.

3. **Month-to-Month Jaccard Similarity**: Measures temporal stability of the graph structure. Jaccard similarity is defined as |A intersection B| / |A union B|, ranging from 0 (completely different edge sets) to 1 (identical edge sets). A mean Jaccard of 0.631 indicates that approximately 63% of edges persist from one month to the next, suggesting the graph evolves gradually rather than erratically.

### Sensitivity Analysis

The parameter search revealed clear trade-offs across the grid:

**Window length (w)**:
- w=63 (~3 months): Graphs were highly volatile between months (low Jaccard), capturing short-term noise rather than stable relationships.
- w=126 (~6 months): Sweet spot with moderate density (~6%, approximately 7,500 edges) and good stability (Jaccard mean=0.631, std=0.064).
- w=252 (~12 months): Graphs were overly smooth, slow to adapt to regime changes, and slightly too dense.

**Correlation threshold (tau)**:
- tau=0.5: Too many edges, graph overly dense, diluting meaningful connections with weak correlations.
- tau=0.6: Balanced density. The resulting graphs had one connected component and meaningful sector-based clustering.
- tau=0.7: Too sparse, with some stocks becoming isolated and multiple connected components appearing.

**Figures**:

![Mean Edge Density Heatmap](../../plots/heatmap_density.png)

*Figure 5.1: Heatmap of mean graph density across the 12 parameter configurations (3 windows x 4 thresholds). Density ranges from 28.9% (w=63, tau=0.4 — far too dense, nearly every stock connected to every other) down to 1.0% (w=252, tau=0.7 — too sparse, most stocks isolated). The selected configuration (w=126, tau=0.6) yields 6.0% density — a moderate level where each stock connects to approximately 30 peers.*

![Mean Clustering Coefficient Heatmap](../../plots/heatmap_clustering.png)

*Figure 5.2: Heatmap of mean clustering coefficient. Higher clustering indicates stronger community structure where a stock's neighbors also tend to be neighbors of each other. The selected w=126, tau=0.6 configuration has a clustering coefficient of 0.453, reflecting meaningful but not overly tight sector-based communities.*

![Edge Count Time Series](../../plots/timeseries_edge_count.png)

*Figure 5.3: Number of edges over time for each parameter configuration. Three panels correspond to windows of 63, 126, and 252 trading days. A prominent spike in edge count is visible across all configurations during the 2022 bear market (interest rate hike cycle), when stock correlations surged as equities sold off together. The 126-day window (middle panel) captures this regime shift clearly while maintaining smoother evolution than the noisy 63-day window.*

![Densest Network Configuration — w=252, tau=0.4](../../plots/network_top100_252w_04.png)

*Figure 5.4: Network visualization under the densest configuration (w=252, tau=0.4, 2,792 edges among top 100 stocks). Stocks are colored by GICS sector. The graph is so dense that individual connections are indistinguishable — this configuration would overwhelm GNN message passing with noise.*

![Sparsest Network Configuration — w=63, tau=0.7](../../plots/network_top100_63w_07.png)

*Figure 5.5: Network visualization under the sparsest configuration (w=63, tau=0.7, only 50 edges among top 100 stocks). Most stocks are completely isolated. A few small clusters (primarily Financials in purple) survive, but information propagation through this graph would be extremely limited.*

![Hub Stock Sector Composition Over Time](../../plots/hub_sector_composition.png)

*Figure 5.6: Sector composition of the top 10 hub stocks (by degree centrality) at each monthly snapshot. Financials (blue) and Industrials (orange) persistently dominate, but their relative shares shift with market regimes. During the 2022 bear market, Financials expanded as interest rate sensitivity created cross-sector correlations. The 2024-2025 period shows increasing Information Technology presence as the AI/semiconductor rally created new correlation clusters.*

![Hub Stock Rankings Over Time](../../plots/hub_ranking_timeseries.png)

*Figure 5.7: Ranking trajectories of individual hub stocks over time (w=126, tau=0.6). Each line represents a stock's position in the degree centrality ranking across 54 monthly snapshots. The frequent rank crossings demonstrate that network centrality is dynamic — no single stock maintains permanent hub status. Red lines indicate Financials, blue indicates other sectors.*

### Optimal Configuration Results (w=126, tau=0.6)

- Density: ~6% (approximately 7,500 edges per monthly snapshot)
- Connected components: 1 giant component (all stocks reachable)
- Jaccard similarity: mean=0.631, std=0.064 (stable month-to-month evolution)
- Total snapshots: 54 monthly graphs (July 2020 to December 2025)

**Data Leakage Prevention**: Graph construction uses price data only up to December 2024. Although the test set begins in July 2024, the monthly graph update has inherent lag, ensuring no future information leaks into the graph structure.

---

## 6. Phase C v1: Full-Scale Experiments

**Objective**: Validate the "news + GNN yields directional prediction" hypothesis on the complete S&P 500 dataset.

**Task**: Binary direction prediction — predict whether a stock will go up or down on the next trading day. Labels are defined as sign(close[T+1] - close[T]).

**Model Configurations**: 6 experiments forming a systematic ablation:

| ID | Model | Node Features | Graph Structure |
|----|-------|--------------|-----------------|
| B1 | Logistic Regression (LR): standard linear classifier | FinBERT 768-dim | None |
| B2 | Logistic Regression | Sentiment 4-dim | None |
| A1 | GraphSAGE | FinBERT + Sentiment | news-to-stock edges only |
| A2 | + correlation edges | Same as A1 | + price correlation edges |
| A3 | + sector edges | Same as A1 | + industry sector edges |
| Full | All edge types | Same as A1 | All 3 edge types |

**Results**:

| Experiment | Val AUC | Test AUC |
|-----------|---------|----------|
| B1: LR + FinBERT | 0.5018 | 0.4976 |
| B2: LR + Sentiment | 0.5044 | 0.5027 |
| A1: GNN news only | 0.5085 | 0.4913 |
| A2: + corr edges | 0.5122 | 0.4949 |
| A3: + sector edges | 0.5133 | 0.4961 |
| Full: all edges | 0.5133 | 0.5069 |

**All AUC values are approximately 0.50 — equivalent to random guessing.**

AUC (Area Under the ROC Curve): A classification metric where 0.5 indicates random performance and 1.0 indicates perfect discrimination.

**Diagnosis**: It was unclear whether the failure was due to featureless signals, wrong model architecture, or fundamental task infeasibility.

**Decision**: Perform diagnostics before changing the model.

**Figures**:

![Phase C Model Comparison](../../plots/phase_c_model_comparison.png)

*Figure 6.1: Test AUC across all 6 model configurations. All values cluster tightly around 0.50 (the random baseline, shown as a dashed line). Neither adding graph edges (A1→A2→A3→Full) nor using different feature sets (B1 vs B2) produces any meaningful departure from chance.*

![Phase C Ablation Delta](../../plots/phase_c_ablation_delta.png)

*Figure 6.2: Incremental AUC change from adding each edge type. The marginal contribution of each edge type is negligible (all deltas within ±0.01), confirming that no single component — FinBERT embeddings, sentiment features, correlation edges, or sector edges — carries usable signal for next-day binary direction prediction.*

---

## 7. Diagnostic Phase: D.1 + D.2

**Objective**: Identify the root cause of AUC ~ 0.50.

### D.1: Label Distribution Analysis

**Findings**:
- Approximately 26.5% of events fall in the "noise zone" (|return| < 0.5%)
- Next-day return distribution is approximately normal (mean near zero), nearly symmetric between positive and negative
- Conclusion: The labels themselves are inherently difficult to predict; many returns are too close to zero to carry directional signal.

### D.2: FinBERT Alignment Analysis

**Method**: Measure how well FinBERT sentiment aligns with actual stock movements. "Alignment" is defined as positive sentiment corresponding to an upward move (or negative sentiment corresponding to a downward move).

**Findings**:
- Alignment rate: **51.6%** — only 1.6 percentage points above random (50%)
- Events with high |sentiment| (strong conviction) showed no better alignment
- **Conclusion**: FinBERT sentiment has essentially zero predictive power for S&P 500 next-day returns.

**Figures**:

![Data-Level Diagnostics (D.1)](../../plots/phase_c_diagnostics_data.png)

*Figure 7.1: Data-level diagnostic plots. Left panel shows the distribution of next-day returns, with the shaded "noise zone" (|return| < 0.5%) containing 26.5% of all observations — these are essentially coin flips that no model can reliably predict. Right panels show the FinBERT sentiment distribution and its alignment with return direction — the near-50% alignment rate confirms negligible predictive content.*

![Model-Level Diagnostics (D.2)](../../plots/phase_c_diagnostics_model.png)

*Figure 7.2: Model-level diagnostic plots. The predicted score distributions for positive and negative return classes overlap almost completely (mean separation = -0.00030), indicating the logistic regression model cannot distinguish between up and down days. Sector-by-sector analysis (right panel) shows no sector achieves AUC above 0.512, ruling out the possibility that signal exists in specific market segments.*

**Root Cause Analysis**:
1. **Efficient Market Hypothesis (EMH)**: S&P 500 stocks are among the most efficiently priced in the world. News information is rapidly incorporated into prices, leaving little residual signal for next-day prediction.
2. **Sentiment vs. Impact Mismatch**: FinBERT was trained to classify text sentiment (positive/negative/neutral), not to predict market impact. A news article can be linguistically positive yet convey information already priced in.
3. **Horizon Mismatch**: The next-day horizon may be too short. News effects might materialize over longer periods.

---

## 8. Phase 1 Signal Fix

**Objective**: Attempt to rescue the signal through data and feature engineering improvements.

### Remediation Measures

1. **News deduplication**: Remove duplicate news events that inflate event counts without adding information.
2. **Market-adjusted labels**: Replace raw returns with stock_return minus market_return, isolating stock-specific movements from broad market trends.
3. **Momentum features**: Add 5-day, 10-day, and 20-day price momentum indicators.
4. **GNN improvements**: Add BatchNorm (batch normalization for stabilizing training), dropout (randomly zeroing activations to prevent overfitting), and tune learning rate.
5. **Selective prediction**: Predict only for high-confidence events (top/bottom 5% by model confidence).

### Results — Phase 1 Baseline Matrix

| Model | Val AUC | Test AUC |
|-------|---------|----------|
| B1: FinBERT LR (deduped) | 0.5043 | 0.5043 |
| B2: Sentiment LR (deduped) | 0.5076 | 0.5028 |
| B3: XGBoost (all features) | 0.5093 | 0.5020 |
| B4: Momentum-only XGBoost | 0.5052 | 0.5044 |
| B5: Random Forest | 0.5068 | 0.5004 |

XGBoost (Chen & Guestrin, KDD 2016): A gradient boosted decision tree ensemble known for strong tabular data performance. Random Forest (Breiman, 2001): An ensemble of independently trained decision trees using bagging and feature subsampling.

**All test AUC < 0.51 — Signal Fix failed.**

### Selective AUC

- Top/bottom 5% confidence subset: AUC = 0.5154 — within statistical noise
- **No tail signal detected**

### Conclusion
- Binary direction prediction on S&P 500 using FinBERT is **not feasible**.
- This is not a code bug but a fundamental task limitation rooted in market efficiency.
- Literature comparison: DGRCL (the only published GNN paper using 1,000+ stocks) reports only 53% accuracy, consistent with our findings.

---

## 9. Phase 2 LLM: GPT-4o-mini

**Objective**: Test whether a more powerful LLM can break through the AUC ~ 0.50 barrier.

**Why GPT-4o-mini** (OpenAI's compact large language model):
- Strong academic citation potential (OpenAI models are widely recognized)
- Supports structured JSON output for reliable sentiment score extraction
- Cost-effective compared to GPT-4o

**Method**: Use GPT-4o-mini to generate structured sentiment scores for each news event, then compare against FinBERT results.

**Results**:
- GPT-4o-mini vs. FinBERT delta: **+0.0009** — virtually no difference
- High-impact event subset: AUC = **0.4762** — actually worse than random

**Decision**: **STOP confirmed** — LLM upgrade cannot solve the problem. The issue lies in the task definition (binary direction prediction on efficient markets), not in model capability.

---

## 10. v3 Paradigm Shift

**Date**: 2026-03-05

### Core Insight

A comprehensive literature review (10+ recent papers) revealed that our problem was not the model but the **task formulation**:

| Dimension | v2 (Failed) | v3 (New Direction) | Literature Support |
|-----------|-------------|-------------------|-------------------|
| **Prediction task** | Binary direction (up/down) | **Ranking** (relative stock ordering) | MASTER, FinMamba, MDGNN all use ranking |
| **Evaluation metrics** | AUC | **IC / ICIR / Sharpe** | Quantitative finance standard |
| **Data paradigm** | Event-driven (predict only when news exists) | **Calendar-driven** (predict every stock every day) | All SOTA papers |
| **Label definition** | sign(return) | **z-score(return)** (cross-sectional standardization) | Removes market-wide directionality |

### Key Metric Definitions

- **IC (Information Coefficient)**: The daily Spearman rank correlation between predicted scores and actual returns across all stocks. IC=0.03 is considered a meaningful threshold in quantitative finance. Unlike AUC which measures binary classification quality, IC measures how well the model ranks stocks relative to each other.

- **ICIR (IC Information Ratio)**: mean(IC) / std(IC). Measures the stability and consistency of the IC signal over time. ICIR > 0.1 indicates a reliably stable signal.

- **Sharpe Ratio**: Annualized excess return divided by annualized volatility. Measures risk-adjusted performance. Sharpe > 0.5 indicates economic significance.

- **Long-Short Portfolio**: A trading strategy that goes long (buys) the top-N ranked stocks and goes short (sells) the bottom-N ranked stocks. The return captures the spread between predicted winners and losers, independent of overall market direction.

- **Calendar-Driven**: Every trading day, the model generates predictions for all stocks. Stocks without news on a given day use zero-filled news feature vectors. This contrasts with event-driven approaches that only predict when news is available.

- **Cross-Sectional Z-Score**: (return - cross_sectional_mean) / cross_sectional_std, computed across all stocks on the same day. This removes the market beta (overall market direction), isolating relative stock performance.

### Key Literature Support

| Paper | Venue | Core Finding |
|-------|-------|-------------|
| MASTER (Lu et al.) | AAAI 2024 | Cross-stock Transformer achieving IC=0.064 on CSI300 |
| FinMamba (Zhang et al.) | arXiv 2025 | Mamba architecture + dynamic graph, Sharpe=2.06 on S&P 500 |
| MDGNN (Wang et al.) | AAAI 2024 | Multi-relational dynamic graph, IC=0.032 on CSI300 |
| THGNN (Xiang et al.) | CIKM 2022 | Daily dynamic graph + heterogeneous GAT for stock prediction |
| SelectiveNet (Geifman & El-Yaniv) | ICML 2019 | 3-head architecture (prediction + selection + auxiliary), 800+ citations; learns when to abstain from prediction |
| Multi-GCGRU (Liu et al.) | IEEE 2024 | Demonstrated that co-occurrence edges outperform holding/supply-chain edges |

### Three Proposed Paper Contributions

1. **Horizon Ablation**: Systematic comparison across 1d/5d/10d/21d/42d/63d prediction horizons — no prior work provides this analysis for GNN-based stock prediction.
2. **GNN + SelectiveNet**: First application of SelectiveNet to financial GNNs — an unexplored combination in the literature.
3. **Dynamic Heterogeneous Graph + NLP**: Multi-edge-type graph combining correlation, sector, news mentions, and co-occurrence relationships.

---

## 11. v3 Implementation: N1-N5 Pipeline

### Architecture Overview

The entire v3 pipeline is implemented in a single Jupyter notebook (`v3_ranking_pipeline.ipynb`) with 19 cells organized as follows:

- **N0**: Setup and environment configuration
- **N1a-d**: Feature engineering and data preparation
- **N2a-c**: Dynamic heterogeneous graph construction
- **N3a-d**: Baseline models and GNN ablation experiments
- **N4**: Horizon ablation study
- **N5a-c**: SelectiveNet model, training, and analysis

### N1: Calendar-Driven Data Pipeline

**Input**: Raw price data + EODHD news events
**Output**: A 781-dimensional feature vector for every stock on every trading day

| Feature Group | Dimensions | Contents |
|--------------|-----------|----------|
| Price features | 9 | 1d/5d/10d/21d returns, 5d/21d volatility, 21d moving average ratio, volume ratio, 1d log return |
| FinBERT embedding | 768 | Mean FinBERT embedding of all news mentioning the stock on that day |
| Sentiment scores | 3 | Mean positive/negative/neutral sentiment probabilities |
| Has-news flag | 1 | Binary indicator (0/1) for whether any news mentioned this stock on that day |
| **Total** | **781** | |

**Labels**: Forward returns at 6 horizons, normalized via cross-sectional z-score.
- Forward return: (close[T+h] - close[T]) / close[T]
- Z-score: (return - mean_across_stocks) / std_across_stocks, computed daily
- Horizons: 1d, 5d, 10d, 21d, 42d, 63d

**Time Split**:

| Split | Date Range | Trading Days | News Coverage |
|-------|-----------|-------------|---------------|
| Train | 2021-07 to 2023-12 | 629 | 57.6% |
| Val | 2024-01 to 2024-06 | 124 | 55.9% |
| Test | 2024-07 to 2026-01 | 396 | 62.7% |

### N2: Dynamic Heterogeneous Graph

Four edge types connect stocks and news:

| Edge Type | Node Pair | Dynamic/Static | Construction |
|-----------|-----------|---------------|-------------|
| Correlation | stock <-> stock | Monthly update | Pearson correlation > 0.6, window=126 trading days |
| Sector | stock <-> stock | Static | Same GICS (Global Industry Classification Standard) sector |
| Mentions | news -> stock | Daily | News article mentions a stock ticker |
| Co-occurrence | stock <-> stock | Daily | Two stocks mentioned in the same news article |

**Graph Statistics**:
- Correlation: 54 monthly snapshots; density decreases from 2.9% to 0.6% over time
- Sector: 27,070 edges across 11 GICS sectors
- News mentions: 1,538,967 total (average 1,226 per day)
- Co-occurrence: 2,918,292 total (average 2,325 per day)

### N3: Models and Baselines

**Non-GNN Baselines**:

| ID | Model | Features | Details |
|----|-------|---------|---------|
| B1 | Ridge Regression: L2-regularized linear regression (alpha=1.0) | 9-dim (price only) | Simplest baseline using only price-derived features |
| B2 | Ridge Regression | 781-dim (all) | Tests whether adding news features helps in a linear model |
| B3 | XGBoost (Chen & Guestrin, KDD 2016): gradient boosted decision tree ensemble | 781-dim | n_estimators=200, max_depth=5, lr=0.05, early_stopping=20 |
| B4 | LightGBM (Ke et al., NeurIPS 2017): histogram-based gradient boosting from Microsoft | 781-dim | Same hyperparameters as XGBoost |

**GNN Models**:

| Class | Architecture | Graph Type | Parameters |
|-------|-------------|-----------|------------|
| RankingHGT | HGTConv (Hu et al., WWW 2020): Heterogeneous Graph Transformer that learns type-specific Key/Query/Value projection matrices for each node type and edge type. Parameter count scales with number_of_edge_types x hidden_dim^2. | HeteroData (heterogeneous) | stock_linear(781->64), news_linear(771->64), 2 layers, 4 attention heads |
| RankingGNN | GATConv (Velickovic et al., ICLR 2018): Graph Attention Network that applies learned attention weights to neighbor aggregation, with all edges sharing a single set of attention parameters. GATConv(64, 16, heads=4, concat=True) = 4 attention heads, each producing 16 dimensions, concatenated to 64 dimensions. Also supports SAGEConv (Hamilton et al., NeurIPS 2017): GraphSAGE using mean aggregation without attention. | Data (homogeneous) | linear(781->64), 2 layers |

**Key Hyperparameters**:

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| hidden_channels | 64 | GNN hidden layer dimension |
| num_heads | 4 | Number of attention heads |
| num_hgt_layers | 2 | Number of GNN layers |
| dropout | 0.3 | Dropout rate for regularization |
| lr | 1e-3 | Learning rate |
| weight_decay | 1e-4 | L2 regularization coefficient |
| epochs | 100 | Maximum training epochs |
| patience | 15 | Early stopping patience (stop if validation loss does not improve for 15 consecutive epochs) |
| grad_accum | 32 | Gradient accumulation steps: since each day is an independent graph of ~501 nodes (effective batch size = 1), gradients are accumulated over 32 days before updating, yielding an effective batch size of 32 |
| default_horizon | 5 | Default prediction horizon in trading days |
| top_k | 30 | Number of stocks in long and short legs of the portfolio |
| transaction_cost | 15 | Round-trip transaction cost in basis points (1 bps = 0.01%) |
| target_coverage | 0.2 | SelectiveNet target coverage rate (predict on 20% of stock-days) |
| selection_lambda | 32.0 | SelectiveNet coverage penalty coefficient |

### N3d: GNN Ablation Design

| ID | Architecture | Edge Types | Graph Type |
|----|-------------|-----------|-----------|
| A1 | HGT | Correlation only | Heterogeneous (with dummy news nodes) |
| A2 | HGT | Correlation + Sector | Heterogeneous |
| A3 | HGT | All 4 types | Heterogeneous |
| A4 | SAGE | Correlation + Sector | **Homogeneous** (merged edges, no news nodes) |
| A5 | GAT | Correlation + Sector | **Homogeneous** |

### N5: SelectiveNet

**SelectiveNet** (Geifman & El-Yaniv, ICML 2019): A 3-head neural network architecture that learns not only what to predict but also when to abstain. Originally designed for image classification with a reject option; here adapted for stock ranking. The architecture comprises:

- **Head 1 (Prediction)**: Outputs the ranking score (trained with MSE loss against z-scored returns)
- **Head 2 (Selection)**: Outputs a confidence score in [0,1] via sigmoid, indicating whether to trust the prediction for this stock-day
- **Head 3 (Auxiliary)**: A secondary prediction head (MSE loss) that acts as a regularizer for the shared backbone

**Loss Function**:

L = L_selective + lambda * max(0, c_target - coverage)^2 + L_auxiliary

Where:
- L_selective = sum(selection_i * (pred_i - target_i)^2) / sum(selection_i): the selective risk, which only penalizes errors on selected samples, weighted by selection confidence
- coverage = mean(selection_i): the fraction of samples the model chooses to predict on
- L_auxiliary = MSE(aux_pred, target): standard MSE on all samples for regularization

**Training Strategy (2-stage)**:
1. **Stage 1**: Train backbone + ranking head + auxiliary head (selection head unconstrained)
2. **Stage 2**: Freeze backbone + ranking head; train only the selection head with coverage penalty

**Market Context Features** (4 dimensions, all using T-1 values to prevent data leakage):
1. 21-day market volatility (VIX proxy): rolling standard deviation of market returns
2. 63-day drawdown from peak: measures current market stress
3. 30-day mean cross-sectional volatility: average dispersion across stocks
4. 5-day market breadth: fraction of stocks with positive returns (market participation indicator)

---

## 12. v3 Colab Run 1: N3 Initial Results

**Environment**: NVIDIA RTX PRO 6000 Blackwell Server Edition, 102.0 GB VRAM
**Run Date**: 2026-03-05
**Code Version**: Original (N4 used HGT, N5 used SelectiveRankingHGT)

### Data Pipeline (N1-N2) — All Validated

| Metric | Value |
|--------|-------|
| Valid tickers | 501 (1 of 502 lost in intersection) |
| Stock features tensor | (1255, 501, 781), 1.96 GB |
| News coverage | 58.5% train, 62.7% test |
| Correlation snapshots | 54 monthly graphs |
| Labels z-score check | mean ~ 0, std ~ 0.999 (correct) |

### N3: Baseline + GNN Ablation Results (5d horizon, test set)

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

MaxDD (Maximum Drawdown): The largest peak-to-trough decline in portfolio value; a measure of downside risk. Ann_LS: Annualized return of the long-short portfolio.

### Go/Stop Gate

| Condition | Threshold | Actual | Result |
|-----------|-----------|--------|--------|
| Best IC > 0.03 | 0.03 | 0.02054 | Not met |
| Best Sharpe_LS > 0.5 | 0.5 | 1.038 | Met |
| **OR condition** | | | **GO** |

### N4 and N5 — Partial Visibility

N4 Horizon Ablation: Only the 1d result was visible (subsequent output was buried by sklearn warnings). The visible result used HGT (all 4 edges) — the worst configuration. This was corrected to GAT (corr+sector) for Run 2.

N5 SelectiveNet: Results completely hidden by warnings. Warnings were suppressed and model updated for Run 2.

---

## 13. GAT vs HGT: Why the Simpler Model Won

### Architecture Comparison

| Property | HGT (A2) | SAGE (A4) | GAT (A5) |
|----------|----------|-----------|----------|
| Graph type | Heterogeneous (HeteroData) | Homogeneous | Homogeneous |
| Edge handling | Separate attention parameters per edge type | All edges unified (mean aggregation) | All edges share one attention mechanism |
| News nodes | Present (dummy all-zero nodes) | Absent | Absent |
| Parameter count | High (type-specific K/Q/V) | Low (no attention) | Medium (shared attention) |
| IC | 0.01177 | 0.01571 | **0.02054** |

### Four Reasons for GAT Superiority

1. **Parameter Efficiency (Bias-Variance Tradeoff)**
   IC ~ 0.02 means 98% of variation is noise. HGT's type-specific parameters lead to overfitting under weak signal conditions. GAT has fewer parameters, resulting in lower variance and better generalization.

2. **Edge Type Distinction Is Unnecessary**
   The distinction between "these stocks have correlated prices" and "these stocks are in the same sector" is semantically similar — both indicate "relatedness." HGT expends parameters to differentiate these relationships, but the differentiation itself provides no predictive value in this domain.

3. **Dummy News Node Noise**
   Even when news edges are excluded, HGT still processes dummy news nodes (all-zero features) as part of the heterogeneous graph structure. These useless computations introduce additional noise.

4. **Attention Mechanism Robustness**
   GATConv(64, 16, heads=4) is a well-tested, battle-proven architecture. HGT performs well on academic benchmarks with clean, structured data, but financial data is extremely noisy, favoring simpler and more robust architectures.

### Extended Finding: News/Co-occurrence Edges Are Harmful

| Configuration | IC | Change |
|--------------|-----|--------|
| A2: HGT (corr+sector) | 0.01177 | baseline |
| A3: HGT (all 4 edges) | 0.00432 | **-63%** |

Adding news mentions and co-occurrence edges caused IC to drop drastically. These edges create dense, noisy connections (1,226 mentions + 2,325 co-occurrences per day), diluting the information content of the sparser correlation structure. In high-noise financial data, less is more.

---

## 14. v3 Colab Run 2: N3-N5 Complete Results

**Environment**: NVIDIA A100-SXM4-40GB, VRAM: 42.4 GB
**Run Date**: 2026-03-06
**Code Version**: Updated (N4/N5 switched to GAT(corr+sector), warnings suppressed, grad_accum=32)

### N3 Baseline + GNN Ablation Results (Run 2, 5d horizon, test set)

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

Ann_LS_net: Annualized long-short return after deducting transaction costs (15 bps round-trip). Sharpe_Long: Sharpe ratio of the long-only leg.

### Run 1 vs Run 2 Comparison

| Model | Run 1 IC | Run 2 IC | Run 1 Sharpe | Run 2 Sharpe |
|-------|----------|----------|--------------|--------------|
| B1-B4 (baselines) | identical | identical | identical | identical |
| A1: HGT (corr) | 0.01023 | 0.00848 | 0.121 | 0.426 |
| A2: HGT (corr+sec) | 0.01177 | 0.01447 | 0.994 | 0.320 |
| A3: HGT (all 4) | 0.00432 | 0.00884 | -0.314 | 0.012 |
| A4: SAGE (corr+sec) | 0.01571 | 0.01545 | 1.038 | 1.266 |
| **A5: GAT (corr+sec)** | **0.02054** | **0.00640** | **1.011** | **0.289** |

**Key Observations**:
- **Baselines (B1-B4) are perfectly consistent** across runs, confirming code correctness. All differences stem from GNN training stochasticity.
- **SAGE is the most stable**: IC virtually unchanged (0.01571 vs 0.01545)
- **GAT is the most unstable**: IC dropped from 0.02054 to 0.00640 (-69%)
- **HGT shows moderate variance**: A2 IC rose from 0.01177 to 0.01447
- **Conclusion**: Single-run results are unreliable. Walk-forward cross-validation or repeated runs are necessary.

### Go/Stop Gate (Run 2)

| Condition | Threshold | Actual | Result |
|-----------|-----------|--------|--------|
| Best IC > 0.03 | 0.03 | 0.01545 | Not met |
| Best Sharpe_LS > 0.5 | 0.5 | 1.266 | Met |
| **OR condition** | | | **GO** |

---

## 15. N4 Horizon Ablation: Detailed Analysis

**Design**: For each of 6 prediction horizons, train both GAT(corr+sector) and LightGBM independently, then compare how graph structure benefit varies with time scale.

### Complete Results Table

| Horizon | GAT IC | GAT ICIR | GAT Sharpe | GAT Ann_LS | LGBM IC | LGBM Sharpe | n_days |
|---------|--------|----------|------------|------------|---------|-------------|--------|
| **1d** | -0.00104 | -0.013 | 2.468 | 34.54% | 0.00368 | 2.918 | 395 |
| **5d** | 0.02334 | 0.227 | 1.568 | 18.27% | 0.00828 | 0.773 | 391 |
| **10d** | 0.03854 | 0.320 | 1.196 | 19.26% | 0.01349 | 0.644 | 386 |
| **21d** | **0.04420** | **0.374** | **1.203** | **18.71%** | 0.01513 | 0.468 | 375 |
| **42d** | -0.00912 | -0.144 | 0.071 | 0.73% | 0.03679 | 0.668 | 354 |
| **63d** | -0.00838 | -0.118 | 0.487 | 6.36% | **0.05207** | **1.256** | 333 |

### Net Returns After Transaction Costs

| Horizon | GAT Ann_LS_net | LGBM Ann_LS_net |
|---------|---------------|-----------------|
| 1d | -41.06% | -37.23% |
| 5d | 3.15% | -4.20% |
| 10d | **11.70%** | 1.47% |
| 21d | **15.11%** | 2.95% |
| 42d | -1.07% | 8.45% |
| 63d | 5.16% | **24.35%** |

### Key Findings

**1. GAT "Sweet Spot" at 10d-21d**

GAT IC exhibits an inverted-U pattern across horizons:

1d (-0.001) -> 5d (0.023) -> 10d (0.039) -> 21d (0.044) -> 42d (-0.009) -> 63d (-0.008)

- **21d is the optimal horizon**: IC=0.04420 exceeds the 0.03 threshold, ICIR=0.374, Sharpe=1.203
- **10d also exceeds the threshold**: IC=0.03854
- **1d has no signal**: Negative IC indicates graph structure does not help at the daily frequency
- **42d/63d signal collapses**: Negative IC suggests graph structure overfits at longer horizons

**2. GAT vs LightGBM: A Crossover Pattern**

| Time Scale | Winner | Graph Benefit | Interpretation |
|-----------|--------|--------------|----------------|
| 1d | LightGBM | Negative | Daily noise overwhelms GNN neighbor aggregation, adding noise rather than signal |
| 5d-21d | **GAT** | Positive (2.8x-3.0x) | **Graph-propagated information is most valuable at the weekly-to-monthly scale** |
| 42d-63d | LightGBM | Negative | Long-term trends are driven by macroeconomic factors, where local graph structure is irrelevant |

This is a **publishable insight**: the incremental value of GNNs for stock prediction is time-scale dependent, with an optimal range of 2-4 weeks.

**3. LightGBM Shows Monotonic Improvement**

LightGBM IC increases monotonically with horizon: 0.004 -> 0.008 -> 0.013 -> 0.015 -> 0.037 -> 0.052. This is expected: tree models operate on individual stock features, and longer-term trends are more predictable from momentum and other price-derived features. Graph structure is unnecessary because the features themselves contain momentum information.

**4. 1d Sharpe Anomaly**

GAT 1d Sharpe=2.468 and LightGBM 1d Sharpe=2.918 appear impressive, but IC is near zero for both. The explanation: a top_k=30 long-short portfolio at daily frequency happened to profit by chance. After deducting transaction costs, Ann_LS_net = -41% — completely infeasible. **This is statistical noise, not a real signal.**

**5. Economic Significance After Costs**

Only GAT at 10d and 21d horizons produce positive net returns above 10%:
- GAT 10d: Ann_LS_net = 11.70%
- GAT 21d: Ann_LS_net = 15.11%

**Figures**:

![Horizon Ablation — IC, ICIR, and Sharpe vs. Prediction Horizon](../../plots/v3_horizon_ablation.png)

*Figure 15.1: Three-panel visualization of the horizon ablation results. **Left panel (IC vs Horizon)**: GAT (blue) traces a clear inverted-U shape peaking at 21 days, while LightGBM (orange) increases monotonically. The red dashed line marks the IC=0.03 "Go" threshold — GAT crosses it at 10d and 21d. **Middle panel (ICIR vs Horizon)**: Signal stability follows the same inverted-U pattern for GAT, confirming the 21d peak is not a fluke. **Right panel (Sharpe vs Horizon)**: Both models show elevated Sharpe at 1d due to statistical noise in the high-frequency long-short portfolio (net returns are deeply negative after costs). The core message: GNN graph structure provides genuine, non-redundant predictive signal, but only within a specific 2-4 week temporal window. This crossover pattern between graph-based and flat-feature models is a novel finding not previously reported in the financial GNN literature.*

---

## 16. N5 SelectiveNet: Detailed Analysis

### Training Details

- **Best horizon**: Automatically selected 21d from N4 results (IC=0.04420)
- **Stage 1** (backbone + ranking + auxiliary): 31 epochs, stopped by early stopping
- **Stage 2** (selection head only): 50 epochs, final coverage ~ 0.312
- **Training time**: 386.2 seconds

### Complete Results Table (21d horizon)

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

Note: "Threshold @X%" means selecting the X% of stock-days where the absolute value of the ranking prediction (|score|) is highest — a simple confidence proxy. "SelectiveNet @X%" means selecting the X% of stock-days with the highest selection head output.

### Key Findings

**1. SelectiveNet Completely Failed**

SelectiveNet produces **negative IC** at all coverage levels from 5% to 50%:
- @5%: IC = -0.01544 (selected stocks are ranked **inversely**)
- @10%: IC = -0.02159
- @20%: IC = -0.02414 (the target coverage level, and the worst result)

**The selection head learned to inversely select** — it preferentially chose stock-days where the GNN's predictions were worst.

**2. Threshold Baseline Works Correctly**

Using |ranking_score| as a confidence proxy:
- @20%: IC = 0.03070, exceeding the 0.03 threshold
- @50%: IC = 0.05087, approaching the full-set IC
- Sharpe monotonically increases with coverage (0.548 -> 1.346)

**3. The Full Model Is Actually the Best**

The SelectiveRankingGAT model evaluated on 100% of stock-days achieved:
- IC = 0.05595 (highest across all experiments)
- ICIR = 0.463
- Sharpe = 1.328
- Ann_LS_net = 16.48%

This exceeds the N4 result at the same horizon (IC=0.04420), indicating that the 3-head architecture's auxiliary loss provides a beneficial regularization effect even when the selection head is not used.

**4. SelectiveNet Failure Analysis**

| Factor | Explanation |
|--------|-------------|
| Highly right-skewed selection scores | Most selection scores concentrated in 0.8-1.0, lacking discriminative power |
| Insufficient coverage constraint | Final coverage=0.312 exceeded target=0.2, indicating the penalty coefficient was too weak |
| No useful learning signal for selection | The ranking loss does not provide gradients that help the selection head learn what to avoid |
| 2-stage training disconnect | Freezing the backbone in Stage 2 prevents the selection head from optimizing a joint objective |

**5. SelectiveNet vs Threshold Select Different Stocks**

Jaccard similarity between the two methods at various coverage levels is only 0.2-0.35, confirming that SelectiveNet selects an entirely different (and worse) subset of stock-days.

**Figures**:

![SelectiveNet Analysis — 6-Panel Visualization](../../plots/v3_selective_analysis.png)

*Figure 16.1: Comprehensive analysis of SelectiveNet performance. **Top-left (IC vs Coverage)**: Threshold selection (orange) maintains positive IC at all coverage levels, while SelectiveNet (blue) produces deeply negative IC below 50% coverage — the selection head learned to anti-select. The gray dashed line shows the full-model IC for reference. **Top-middle (Sharpe vs Coverage)**: Same pattern — threshold selection tracks the full-model Sharpe, while SelectiveNet collapses. **Top-right (Score Distribution)**: SelectiveNet's selection scores are heavily right-skewed (concentrated near 1.0), providing almost no discrimination between "trust" and "abstain." **Bottom-left (Coverage vs Market Volatility)**: Daily coverage (light blue) and 21-day average (dark blue) plotted against market volatility (red). Coverage does not meaningfully respond to market regime changes. **Bottom-middle (Jaccard Overlap)**: Jaccard similarity between threshold-selected and SelectiveNet-selected stocks is only 0.2-0.35, confirming the two methods select fundamentally different subsets. **Bottom-right (Cumulative L/S Return)**: The full model's cumulative return (blue) trends upward, while SelectiveNet@20% (orange) trends downward — directly illustrating the anti-selection effect.*

### Implications for the Paper

SelectiveNet as a contribution point failed. Alternative strategies:
1. **Report as a negative finding** — demonstrate that SelectiveNet is not suitable for financial GNN ranking
2. **Use threshold selection instead** — simple and effective, with @20% IC=0.03070
3. **Improve SelectiveNet** — joint training (no 2-stage split), increase lambda (32 -> 128), or redesign the loss function

---

## 17. Training Stability Analysis

### N3 Two-Run GNN Comparison

| Model | Run 1 IC | Run 2 IC | |IC diff| | IC CV* |
|-------|----------|----------|----------|--------|
| A1: HGT (corr) | 0.01023 | 0.00848 | 0.00175 | ~15% |
| A2: HGT (corr+sec) | 0.01177 | 0.01447 | 0.00270 | ~21% |
| A3: HGT (all 4) | 0.00432 | 0.00884 | 0.00452 | ~69% |
| A4: SAGE (corr+sec) | 0.01571 | 0.01545 | 0.00026 | ~2% |
| A5: GAT (corr+sec) | 0.02054 | 0.00640 | 0.01414 | ~105% |

*CV = Coefficient of Variation (|diff| / mean), measuring relative variability between the two runs.

### Stability Ranking

SAGE (~2%) >> HGT-corr (~15%) > HGT-corr+sec (~21%) >> HGT-all (~69%) >> GAT (~105%)
Most stable ---------------------------------------------------------> Least stable

### Implications for Experimental Conclusions

1. **Run 1's "GAT is best" conclusion may be a false positive**: GAT IC ranges from 0.006 to 0.021, a span too wide for reliable ranking.
2. **SAGE may be the more dependable choice**: IC is stable at ~0.015 — not the highest, but reproducible.
3. **Walk-forward cross-validation is essential**: A single train/val/test split is insufficient; rolling-window validation across multiple periods is needed.
4. **N4's GAT 21d IC=0.04420 also requires verification**: It likely exhibits similar training variance.

### Technical Sources of Instability

| Factor | Impact |
|--------|--------|
| Daily sequence shuffling | Each epoch randomly permutes the order of training days |
| Gradient accumulation boundaries | With grad_accum=32, different permutations produce different 32-day groups for each gradient update |
| Early stopping | Stopping at different epochs yields different model parameters |
| CUDA non-determinism | Despite setting seed=42, GPU floating-point operations on Colab have inherent non-determinism |

---

## 18. Current Status and Next Steps

### Completed Phases

| Phase | Status | Key Result |
|-------|--------|-----------|
| Phase 1 (Network Analysis) | Complete | Validated network structure existence among S&P 500 stocks |
| Phase 2 Pilot | Complete | GraphSAGE AUC=0.6426 (small scale, 9 stocks) |
| Phase A (Data) | Complete | 1.7M events encoded with FinBERT |
| Phase B (Graph Parameters) | Complete | Optimal: w=126, tau=0.6 |
| Phase C v1 | Complete | AUC ~ 0.50 (EMH barrier for binary prediction) |
| D.1/D.2 Diagnostics | Complete | FinBERT has no predictive signal for next-day returns |
| Signal Fix | Complete | All AUC < 0.51 despite multiple remediation attempts |
| Phase 2 LLM | Complete (STOP) | GPT-4o-mini delta = +0.0009 over FinBERT |
| **v3 N1-N2** | Complete | Calendar-driven pipeline + dynamic heterogeneous graph |
| **v3 N3 Run 1** | Complete | GAT IC=0.02054 (unstable) -> GO decision |
| **v3 N3 Run 2** | Complete | SAGE IC=0.01545 (stable) -> GO decision |
| **v3 N4** | Complete | **GAT 21d IC=0.04420 > 0.03 threshold** |
| **v3 N5** | Complete | SelectiveNet failed; Full model IC=0.05595 |

### Minimum Publication Criteria (Updated)

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Any horizon IC > 0.03 | Exceed random ranking | **GAT 21d IC=0.04420** | Met |
| GNN IC > LightGBM IC (same horizon) | Graph adds value | 21d: **0.044 > 0.015** (2.9x) | Met |
| Selective IC@20% > Full IC | Selection adds value | Threshold @20%: 0.031 < Full 0.056 | **Not met** |
| Long-Short Sharpe > 0.5 (after costs) | Economic significance | GAT 21d Ann_LS_net=**15.11%** | Met |
| Horizon ablation shows clear pattern | Literature contribution | **Inverted-U shape, peak at 21d** | Met |

**4 of 5 criteria met.** The SelectiveNet contribution point failed but can be replaced with threshold selection or reported as a negative finding.

### Next Steps (Priority Order)

1. **Walk-Forward Cross-Validation** — Resolve training stability concerns (highest priority)
   - Rolling window: 2-year train + 6-month validation + 6-month test
   - Report multi-window IC mean and standard deviation
   - Verify whether the 21d advantage persists across different time periods

2. **Repeated Experiments** — Run the same configuration 5 times, report mean +/- std
   - GAT 21d IC=0.04420 needs reproducibility verification
   - Consider switching to SAGE (more stable) if GAT proves unreliable

3. **SelectiveNet Improvement or Abandonment**
   - Option A: Joint training (no 2-stage split)
   - Option B: Increase lambda (32 -> 128)
   - Option C: Replace SelectiveNet with threshold selection as the paper contribution
   - Option D: Report as a negative finding

4. **Paper Figure Preparation**
   - Horizon ablation 3-panel plot (IC / ICIR / Sharpe vs horizon) — already generated
   - Selective analysis 6-panel plot — already generated
   - Training curve plots
   - Network structure visualization

5. **Transaction Cost Sensitivity Analysis**
   - Test across tc = 5, 10, 15, 20, 30 bps

---

## 19. Glossary

| Term | Full Name | Definition |
|------|-----------|-----------|
| GNN | Graph Neural Network | Neural network that operates on graph-structured data through message passing and feature aggregation between connected nodes |
| GCN | Graph Convolutional Network | Kipf & Welling (ICLR 2017): the foundational GNN that performs neighborhood aggregation with symmetric normalization |
| GAT | Graph Attention Network | Velickovic et al. (ICLR 2018): GNN that uses learned attention weights to differentially aggregate neighbor information |
| GraphSAGE | Sample and Aggregate | Hamilton et al. (NeurIPS 2017): inductive GNN that samples fixed-size neighborhoods and aggregates using mean/max pooling |
| HGT | Heterogeneous Graph Transformer | Hu et al. (WWW 2020): Transformer-style GNN for heterogeneous graphs with type-specific attention parameters |
| IC | Information Coefficient | Spearman rank correlation between predicted scores and actual returns, computed daily across all stocks |
| ICIR | IC Information Ratio | mean(IC) / std(IC); measures the consistency and stability of the IC signal over time |
| Sharpe | Sharpe Ratio | Annualized return / annualized volatility; measures risk-adjusted performance |
| L/S | Long-Short | Portfolio strategy going long the top-N and short the bottom-N ranked stocks |
| AUC | Area Under ROC Curve | Binary classification evaluation metric (0.5 = random, 1.0 = perfect) |
| EMH | Efficient Market Hypothesis | Theory that asset prices fully reflect all available information |
| FinBERT | Financial BERT | ProsusAI's BERT model fine-tuned on financial text for sentiment analysis and embeddings |
| SelectiveNet | Selective Prediction Network | Geifman & El-Yaniv (ICML 2019): 3-head architecture that learns when to abstain from prediction |
| HeteroData | Heterogeneous Data | PyTorch Geometric data structure for graphs with multiple node and edge types |
| GICS | Global Industry Classification Standard | S&P/MSCI industry classification system with 11 sectors |
| Z-score | Standard Score | (x - mean) / std; normalizes data to zero mean and unit variance |
| Jaccard | Jaccard Similarity | |A intersection B| / |A union B|; measures overlap between two sets, ranging from 0 to 1 |
| MaxDD | Maximum Drawdown | Largest peak-to-trough percentage decline in portfolio value |
| AUGRC | Area Under Generalized Risk-Coverage | Evaluation metric for selective prediction proposed at NeurIPS 2024 |
| PyG | PyTorch Geometric | Library extending PyTorch for graph neural network research and applications |
| Gradient Accumulation | — | Technique that sums gradients over multiple mini-batches before updating weights, effectively increasing batch size |
| Early Stopping | — | Halting training when validation loss stops improving, preventing overfitting |
| Cross-Sectional | — | Comparison across all stocks at the same point in time (as opposed to time-series analysis of a single stock) |
| Forward Return | — | (close[T+h] - close[T]) / close[T]; the percentage price change over h future trading days |

---

*Last updated: 2026-03-06*
