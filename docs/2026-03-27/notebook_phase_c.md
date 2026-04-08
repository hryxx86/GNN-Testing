# Notebook Documentation: phase_c_model_training.ipynb

## Phase C: Full-Scale Binary Direction Prediction, Diagnostics, and Signal Fix

---

## Table of Contents

1. [Overview and Hypothesis](#1-overview-and-hypothesis)
2. [Cell-by-Cell Walkthrough](#2-cell-by-cell-walkthrough)
   - [Cell 0: Markdown Header and Experiment Plan](#cell-0-markdown-header-and-experiment-plan)
   - [Cell 1: Environment Setup and Imports](#cell-1-environment-setup-and-imports)
   - [Cell 2: Hyperparameter Configuration (PARAMS)](#cell-2-hyperparameter-configuration-params)
   - [Cell 3: Data Loading](#cell-3-data-loading)
   - [Cell 4: Phase 1a -- News Deduplication](#cell-4-phase-1a----news-deduplication)
   - [Cell 5: Phase 1b -- Market-Adjusted Labels](#cell-5-phase-1b----market-adjusted-labels)
   - [Cell 6: Phase 1c -- Momentum and Volatility Features](#cell-6-phase-1c----momentum-and-volatility-features)
   - [Cell 7: Heterogeneous Graph Construction](#cell-7-heterogeneous-graph-construction)
   - [Cell 8: Graph Validation and Sanity Checks](#cell-8-graph-validation-and-sanity-checks)
   - [Cell 9: Baseline Models (LR + XGBoost)](#cell-9-baseline-models-lr--xgboost)
   - [Cell 10: GNN Model Definition and Training Utilities](#cell-10-gnn-model-definition-and-training-utilities)
   - [Cell 11: GNN Ablation Experiments](#cell-11-gnn-ablation-experiments)
   - [Cell 12: Selective AUC Analysis](#cell-12-selective-auc-analysis)
   - [Cell 13: OpenRouter API Key Setup](#cell-13-openrouter-api-key-setup)
   - [Cell 14: Phase 2a -- GPT-4o-mini Structured Output](#cell-14-phase-2a----gpt-4o-mini-structured-output)
   - [Cell 15: Phase 2b -- LLM vs FinBERT Feature Evaluation](#cell-15-phase-2b----llm-vs-finbert-feature-evaluation)
   - [Cell 16: Results Visualization](#cell-16-results-visualization)
   - [Cell 17: D.1 Data-Level Diagnostics](#cell-17-d1-data-level-diagnostics)
   - [Cell 18: D.2 Model Prediction Diagnostics](#cell-18-d2-model-prediction-diagnostics)
   - [Cell 19: Observations and Next Steps (Markdown)](#cell-19-observations-and-next-steps-markdown)
3. [Phase C v1 Results: All AUC Approximately 0.50](#3-phase-c-v1-results-all-auc-approximately-050)
4. [Diagnostic Analysis: D.1 and D.2](#4-diagnostic-analysis-d1-and-d2)
5. [Signal Fix Attempts and Why They Failed](#5-signal-fix-attempts-and-why-they-failed)
6. [LLM Validation (GPT-4o-mini) and the STOP Decision](#6-llm-validation-gpt-4o-mini-and-the-stop-decision)
7. [Key Terminology Reference](#7-key-terminology-reference)

---

## 1. Overview and Hypothesis

This notebook implements Phase C of the DynHetGNN-SP project: a full-scale binary direction prediction experiment on S&P 500 stocks using heterogeneous graph neural networks combined with FinBERT news embeddings. It also contains the diagnostic stages (D.1, D.2), signal fix attempts, and the LLM validation experiment that ultimately led to a STOP decision on the binary classification approach.

**Hypothesis:** A heterogeneous GNN that combines FinBERT news embeddings with stock correlation and sector graph structure will outperform text-only baselines for S&P 500 next-day movement prediction.

**Dataset:**
- 502 S&P 500 constituent stocks
- ~1.7 million news events from EODHD (2020-01 to 2025-12)
- FinBERT 768-dimensional embeddings + 3-dimensional sentiment scores
- Static correlation graph (Pearson |r| > 0.6, window=126 trading days, prices up to 2024-12)

**Task:** Binary classification -- predict whether each stock goes up or down on the next trading day.

**Prediction target (label):** `sign(close[T+1] - close[T])`, later refined to market-adjusted excess returns.

---

## 2. Cell-by-Cell Walkthrough

### Cell 0: Markdown Header and Experiment Plan

A markdown cell that documents the hypothesis, data summary, edge types, and the planned experiment matrix. It defines six experiments:

| ID | Model | Description |
|----|-------|-------------|
| B1 | LR + FinBERT embedding | Text-only baseline (768-dim) |
| B2 | LR + sentiment features | Sentiment-only baseline (3-dim) |
| A1 | GNN: news->stock only | Ablation: no stock-stock edges |
| A2 | GNN: + correlated_with | Ablation: add correlation edges |
| A3 | GNN: + same_sector | Ablation: add sector edges |
| Full | GNN: all 3 edge types | Full heterogeneous model |

---

### Cell 1: Environment Setup and Imports

Sets up the Python environment for either Google Colab or local macOS execution. Key actions:

- **Reproducibility:** Sets `SEED = 42` across Python `random`, NumPy, and PyTorch (including CUDA). Also sets `torch.backends.cudnn.deterministic = True` to ensure reproducible GPU computations.
- **Colab detection:** Mounts Google Drive at `GNN测试` folder if running on Colab; installs PyG (PyTorch Geometric). Falls back to local directory otherwise.
- **Directory creation:** Creates `data/fullscale`, `data/reference`, `plots`, and `experiments` subdirectories.
- **Device selection:** Uses CUDA GPU if available, prints GPU name and VRAM.

**Libraries used:**
- PyTorch: deep learning framework for building and training neural networks
- PyTorch Geometric (PyG): a library for graph neural network models built on PyTorch
- NumPy / Pandas: numerical computing and tabular data processing

---

### Cell 2: Hyperparameter Configuration (PARAMS)

Defines a single `PARAMS` dictionary containing all tunable values. This centralized design avoids magic numbers scattered across cells.

Key parameters:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `corr_threshold` | 0.6 | Pearson correlation cutoff for stock-stock edges |
| `corr_cutoff` | 2024-12-31 | Prevents future data leakage into graph construction |
| `train_end` | 2024-12-31 | End of training period |
| `val_end` | 2025-06-30 | End of validation period; test = everything after |
| `hidden_channels` | 64 | GNN hidden layer dimension |
| `dropout` | 0.3 | Regularization dropout rate |
| `lr` | 1e-3 | Adam optimizer learning rate |
| `weight_decay` | 1e-4 | L2 regularization strength |
| `batch_size` | 2048 | Mini-batch size for neighbor sampling |
| `num_neighbors` | [10, 5] | Per-hop neighbor sampling: 10 at hop-1, 5 at hop-2 |
| `epochs` | 100 | Maximum training epochs |
| `patience` | 15 | Early stopping patience (epochs without val improvement) |

---

### Cell 3: Data Loading

Loads all raw data files from disk:

1. **Events metadata** (`sp500_news_events.parquet`): ~1.7M rows of news events with columns including date, ticker, title, and `return_next` (next-day return).
2. **Prices** (`sp500_5y_prices.csv`): daily close prices for 502 stocks.
3. **Sectors** (`sp500_sectors.csv`): GICS sector mapping for each ticker.
4. **Market caps** (`sp500_market_caps.csv`): used later for stock node features.
5. **FinBERT embeddings** (`sp500_news_emb_finbert.npy`): 768-dim vectors, memory-mapped (`mmap_mode='r'`) to avoid loading the full ~5GB array into RAM.
6. **FinBERT sentiment** (`sp500_news_sentiment_finbert.npy`): 3-dim (positive, negative, neutral) probability scores, also memory-mapped.

Constructs `valid_tickers` as the intersection of tickers appearing in both events and prices, and builds a `ticker_to_id` mapping.

**FinBERT (ProsusAI/finbert):** A BERT-based language model fine-tuned on financial text; outputs 768-dimensional contextual embeddings and 3-dimensional sentiment probabilities (positive/negative/neutral).

---

### Cell 4: Phase 1a -- News Deduplication

**Purpose:** Multiple news articles often cover the same stock on the same day. This cell deduplicates by grouping events by `(date, ticker)` and averaging their embeddings/sentiment.

**Method:**
- Groups events by normalized day + ticker using `groupby().ngroup()`.
- Builds a sparse averaging matrix `W` of shape `(n_groups, n_events)` where `W[g, i] = 1 / |group_g|` if event `i` belongs to group `g`.
- Computes `emb_deduped = W @ emb_original` and `sent_deduped = W @ sent_original` via sparse matrix multiplication, which efficiently mean-pools embeddings within each stock-day group.
- Keeps the first metadata row per group.

**Result:** Reduces the event count from ~1.7M raw events down to deduplicated stock-day records (the exact reduction percentage is printed at runtime, typically around 50-60% reduction).

---

### Cell 5: Phase 1b -- Market-Adjusted Labels

**Purpose:** Raw next-day returns include market-wide movements (beta). On a day the entire market rises 2%, a stock rising 1.5% actually underperformed. Market-adjusted labels isolate stock-specific signal.

**Method:**
1. Computes daily equal-weight market return as the mean of all S&P 500 constituent returns.
2. Shifts by -1 to get next-day market return aligned with today's date.
3. Forward-fills to cover weekends and holidays.
4. For each event: `excess_return = stock_return_next - market_return_next`.
5. New label: `1` if `excess_return > 0`, else `0`.

**Market-adjusted return:** The difference between a stock's return and the market's return, isolating the stock-specific component.

**EMH (Efficient Market Hypothesis):** The theory that asset prices fully reflect all available information, making it difficult to consistently predict price movements.

Reports the noise zone comparison -- what fraction of events have |return| < 0.5% before and after market adjustment.

---

### Cell 6: Phase 1c -- Momentum and Volatility Features

**Purpose:** Adds 9 price-based features per event to augment the NLP features, providing the model with recent price dynamics.

**Features computed (3 windows x 3 statistics = 9 features):**

| Window | Mean Return | Volatility (Std) | Momentum (Cumulative Return) |
|--------|------------|-------------------|------------------------------|
| 5-day  | `ret_mean_5d` | `ret_std_5d` | `momentum_5d` |
| 10-day | `ret_mean_10d` | `ret_std_10d` | `momentum_10d` |
| 21-day | `ret_mean_21d` | `ret_std_21d` | `momentum_21d` |

**Critical data leakage prevention:** All rolling features use `shift(1)`, meaning they are computed using data strictly up to T-1 (the day before the event). This ensures no future information leaks into features.

**Method:**
- Computes rolling statistics on daily stock returns.
- Builds a long-format lookup table indexed by `(trading_day, ticker)`.
- Maps each event date to the nearest prior trading day (to handle weekend events).
- Merges features into the events DataFrame.
- Fills NaN values with 0.0 (events occurring before enough price history is available).

---

### Cell 7: Heterogeneous Graph Construction

**Purpose:** Builds a PyTorch Geometric `HeteroData` object representing the heterogeneous graph with two node types and three edge types.

**Node types:**

| Type | Count | Features | Dimensionality |
|------|-------|----------|----------------|
| `news` | Number of deduplicated stock-day events | FinBERT embedding (768) + sentiment (3) + momentum (9) | 780 |
| `stock` | ~501 | GICS sector one-hot (11) + log market cap (1) | 12 |

**Edge types:**

| Edge Type | Relation | Construction | Notes |
|-----------|----------|--------------|-------|
| `news -> stock` (relates_to) | News article mentions stock | 1:1 mapping from each event to its ticker | ~500K+ edges |
| `stock <-> stock` (correlated_with) | Price correlation | Pearson \|r\| > 0.6 using prices up to 2024-12-31 | Static graph; avoids future leakage |
| `stock <-> stock` (same_sector) | Same GICS sector | All pairs within same sector | ~27K undirected edges across 11 sectors |

**GICS (Global Industry Classification Standard):** A standardized classification system that assigns each company to one of 11 sectors (e.g., Technology, Healthcare, Financials).

**Labels and masks:**
- `data['news'].y`: binary labels (0/1) for each news event
- Train/val/test masks created by calendar-based time split using `train_end` and `val_end` from PARAMS
- Graph is made undirected via `T.ToUndirected()` transform, and news features are cast to float16 to save GPU memory

---

### Cell 8: Graph Validation and Sanity Checks

Prints summary statistics and runs assertions to verify graph integrity:
- Node counts and feature dimensions for each node type
- Edge counts for each edge type
- Label distribution (positive rate) per split (train/val/test)
- Verifies that feature and label dimensions match, masks cover all nodes, and edge indices are within bounds

---

### Cell 9: Baseline Models (LR + XGBoost)

**Purpose:** Establishes non-GNN baselines to measure whether graph structure adds value.

**Models:**

| ID | Model | Features | Description |
|----|-------|----------|-------------|
| B1 | SGD Logistic Regression | FinBERT 768-dim | Text-only baseline |
| B2 | SGD Logistic Regression | Sentiment 3-dim | Sentiment-only baseline |
| B3 | SGD Logistic Regression | Sentiment + Momentum 12-dim | Combined features |
| B4 | XGBoost | Sentiment + Momentum 12-dim | Tree-based ensemble |
| B5 | Random Forest | Sentiment + Momentum 12-dim | Bagging ensemble |

**Logistic Regression:** A linear model for binary classification that applies a sigmoid activation to a linear combination of input features to produce a probability; here implemented as SGDClassifier with `loss='log_loss'` for scalability.

**XGBoost (Extreme Gradient Boosting):** An optimized gradient boosting library using tree-based learners, known for strong performance on tabular data.

**Random Forest:** An ensemble method that trains multiple decision trees on random subsets of data and features, then averages their predictions.

**Training approach:**
- Uses `SGDClassifier` with `partial_fit` in mini-batches of 50,000 for memory efficiency
- Trains for 5 epochs with shuffled mini-batches
- Reports validation AUC every 2 epochs, then final val/test AUC and accuracy

---

### Cell 10: GNN Model Definition and Training Utilities

**Purpose:** Defines the GNN architecture and full-batch training procedure.

**Model: `GNN` class (2-layer GraphSAGE)**

```
Input features -> SAGEConv layer 1 -> ReLU -> Dropout(0.3) -> SAGEConv layer 2 -> ReLU -> Linear -> logit
```

**GraphSAGE (Hamilton et al., NeurIPS 2017):** A GNN that learns node representations by sampling and aggregating features from local neighborhoods using mean aggregation; it is inductive, meaning it can generalize to unseen nodes.

**SAGEConv:** The PyG implementation of GraphSAGE convolution, using `(-1, -1)` to lazily infer input dimensions.

The base homogeneous `GNN` is converted to a heterogeneous model using PyG's `to_hetero()`, which automatically creates separate parameter sets for each edge type in the graph metadata.

**`create_ablation_graph` function:** Creates a subgraph from the full HeteroData by keeping only specified edge types. This enables systematic ablation experiments.

**`run_gnn_experiment` function:** Implements full-batch training where the entire graph is loaded to GPU (feasible on A100 80GB). This avoids neighbor sampling noise and the need for the `torch-sparse` library. Training uses:
- Adam optimizer with learning rate 1e-3 and weight decay 1e-4
- `BCEWithLogitsLoss` (binary cross-entropy with logits): a numerically stable loss for binary classification
- Early stopping with patience of 15 epochs based on validation AUC
- AUC (Area Under ROC Curve): measures how well a binary classifier ranks positive vs negative examples; 0.5 = random guessing

---

### Cell 11: GNN Ablation Experiments

**Purpose:** Runs four GNN configurations to measure the contribution of each edge type.

| ID | Edge Types Included |
|----|-------------------|
| A1: news->stock only | `news -> stock` (relates_to) |
| A2: + correlation | `news -> stock` + `stock <-> stock` (correlated_with) |
| A3: + sector | `news -> stock` + `stock <-> stock` (same_sector) |
| Full: all edges | All three edge types |

Each experiment creates an ablation subgraph using `create_ablation_graph`, trains the heterogeneous GNN, and records results.

---

### Cell 12: Selective AUC Analysis

**Purpose:** Tests whether there is hidden signal in the "tails" -- i.e., whether predictions the model is most confident about are actually more accurate, even if overall AUC is near 0.50.

**Method:**
- For each model, ranks test predictions by confidence (distance from 0.5).
- Computes AUC on the top-K% most confident predictions, for K in {5%, 10%, 20%, 50%}.
- Re-trains all baselines (B1-B5) and loads GNN checkpoints for this analysis.

**Go criterion:** Full AUC > 0.52 OR Top-10% AUC > 0.54. (Neither was achieved.)

**Selective AUC:** The AUC computed only on a high-confidence subset of predictions, testing whether the model "knows what it knows."

---

### Cell 13: OpenRouter API Key Setup

A short utility cell that sets the `OPENAI_API_KEY` environment variable to an OpenRouter API key. This is required for Cell 14, which calls GPT-4o-mini via OpenRouter's API.

**OpenRouter:** An API gateway that provides unified access to multiple LLM providers (OpenAI, Anthropic, etc.) through a single endpoint.

---

### Cell 14: Phase 2a -- GPT-4o-mini Structured Output

**Purpose:** Tests whether a more capable LLM (GPT-4o-mini) can extract stronger predictive signal from news headlines than FinBERT sentiment scores.

**GPT-4o-mini (OpenAI):** A smaller, cost-efficient variant of GPT-4o; used here because it supports structured JSON output and is academically citable.

**Method:**
1. Reloads original events (with titles) from the parquet file.
2. Selects a dev-holdout period (2023-Q4, within training period, never touches val/test).
3. Samples ~7,000 events from this period.
4. For each news headline, queries GPT-4o-mini with a structured output schema requesting:
   - `impact_level`: high / medium / low
   - `direction`: positive / negative / neutral
   - `confidence`: float 0-1
   - `reasoning_type`: earnings / macro / sentiment / technical / other
5. Uses caching (`llm_dev_holdout_results.json`) to avoid redundant API calls.
6. Includes retry logic with exponential backoff for rate limits.

**Cost:** Approximately $0.15-$0.45 for 7,000 samples using GPT-4o-mini via OpenRouter.

---

### Cell 15: Phase 2b -- LLM vs FinBERT Feature Evaluation

**Purpose:** Directly compares GPT-4o-mini structured features against FinBERT features for predicting stock direction.

**Method:** 5-fold stratified cross-validation on the dev-holdout set using Logistic Regression with StandardScaler normalization.

**Stratified K-Fold Cross-Validation:** A resampling method that splits data into K folds while preserving the class distribution in each fold, providing a robust estimate of model performance.

**Feature sets compared:**

| Feature Set | Dimensions | Source |
|-------------|-----------|--------|
| FinBERT sentiment | 3 | pos/neg/neutral probabilities |
| LLM structured | 10 | impact (3-hot) + direction (1) + confidence (1) + reasoning (5-hot) |
| Combined | 13 | FinBERT sentiment + LLM structured |
| FinBERT embedding | 768 | Full contextual embedding |
| LLM + embedding | 778 | FinBERT embedding + LLM structured |

Also performs **impact-level subset analysis**: checks whether events that GPT-4o-mini labels as "high impact" have better predictability.

---

### Cell 16: Results Visualization

**Purpose:** Aggregates all experiment results into a summary table and generates two plots.

**Outputs:**
1. **Results table** saved to `experiments/phase_c_results.csv`
2. **Bar chart** (`plots/phase_c_model_comparison.png`): side-by-side val and test AUC for all models, with a horizontal dashed line at 0.5 (random baseline)
3. **Ablation delta chart** (`plots/phase_c_ablation_delta.png`): shows the incremental test AUC change from each edge type relative to the A1 baseline

---

### Cell 17: D.1 Data-Level Diagnostics

**Purpose:** Investigates the root cause of AUC near 0.50 by analyzing the data itself, independent of any model.

**Three analyses performed:**

**Analysis 1 -- Label Noise Quantification:**
- Computes the distribution of `return_next` (raw next-day returns)
- Buckets events by absolute return magnitude: <0.5%, 0.5-1%, 1-2%, 2-5%, >5%
- Finds the fraction of events in the "noise zone" (|return| < 0.5%) where the label is essentially a coin flip

**Analysis 2 -- FinBERT Sentiment Confidence vs Actual Return:**
- Classifies each event's FinBERT output as positive, negative, or neutral (based on which probability is highest)
- Groups by sentiment confidence (max of positive/negative probability)
- For positive-sentiment news: computes what fraction actually went up
- For negative-sentiment news: computes what fraction actually went down
- This measures "alignment" -- how well FinBERT sentiment predicts actual direction

**Analysis 3 -- Per-Sector Statistics:**
- Breaks down label distribution, positive rate, and return statistics by GICS sector
- Checks for sector-level patterns that might be exploitable

Additional analyses include temporal drift (monthly positive rate over time) and visualization of return distributions with bucket plots.

---

### Cell 18: D.2 Model Prediction Diagnostics

**Purpose:** Analyzes the trained LR + FinBERT model's predictions across multiple dimensions to understand where (if anywhere) the model has signal.

**Method:** Re-trains the B1 (LR + FinBERT) model and generates prediction scores for the test set, then slices performance across multiple dimensions.

**Analyses performed:**

**Analysis 5 -- Prediction Score Distribution:**
- Compares the distribution of prediction scores for actual positive vs negative labels
- Computes "mean separation" -- the difference in mean scores between the two classes (larger = more discriminative)

**Analysis 6 -- Per-Sector AUC:**
- Computes AUC separately for each GICS sector on the test set
- Identifies best and worst sectors for predictability

**Analysis 7 -- AUC by Sentiment Confidence:**
- Splits test events into confidence buckets based on FinBERT's maximum non-neutral probability
- Computes AUC within each bucket to test whether high-confidence FinBERT predictions are more accurate

**Analysis 8 -- AUC by Return Magnitude:**
- Splits by |return_next| into <0.5%, 0.5-1%, 1-2%, >2% buckets
- Tests whether large-move events are easier to predict (they should be, if the model has any signal)

**Analysis 9 -- Temporal AUC Stability:**
- Computes monthly AUC across the test period
- Checks for temporal patterns (e.g., is signal deteriorating over time?)

**Analysis 10 -- Combined Diagnostic Verdict:**
- Synthesizes all findings into a final diagnostic conclusion
- Generates summary visualization plots

---

### Cell 19: Observations and Next Steps (Markdown)

A markdown cell documenting the Phase C v1 results table, placeholders for D.1/D.2 diagnostic findings (to be filled after Colab runs), and a list of potential improvements to explore.

---

## 3. Phase C v1 Results: All AUC Approximately 0.50

The core finding of Phase C v1 is that **no model configuration achieved meaningful predictive performance**. All test AUC values are statistically indistinguishable from 0.50 (random guessing).

### Baseline Results

| Experiment | Val AUC | Test AUC |
|-----------|---------|----------|
| B1: LR + FinBERT (768-dim) | 0.5018 | 0.4976 |
| B2: LR + Sentiment (3-dim) | 0.5044 | 0.5027 |

**Logistic Regression (LR):** A linear model for binary classification using sigmoid activation; the simplest reasonable baseline for this task.

### GNN Ablation Results

| Experiment | Val AUC | Test AUC |
|-----------|---------|----------|
| A1: GNN news->stock only | 0.5085 | 0.4913 |
| A2: + correlation edges | 0.5122 | 0.4949 |
| A3: + sector edges | 0.5133 | 0.4961 |
| Full: all 3 edge types | 0.5133 | 0.5069 |

**AUC (Area Under ROC Curve):** Measures how well a binary classifier ranks positive vs negative examples; 0.5 = random guessing, 1.0 = perfect ranking.

### Key Observations

1. **FinBERT embeddings carry no signal** for next-day direction prediction on S&P 500 stocks. The 768-dimensional embedding (B1) performs identically to random.
2. **Sentiment scores are equally uninformative** (B2 test AUC = 0.5027).
3. **GNN graph structure provides marginal lift** -- the Full model's test AUC of 0.5069 is +1.6% over A1, but this is within statistical noise for this sample size.
4. **Validation AUC is consistently higher than test AUC**, suggesting mild overfitting even at these near-random performance levels.

---

## 4. Diagnostic Analysis: D.1 and D.2

After the Phase C v1 results showed AUC near 0.50, two diagnostic stages were designed to determine the root cause.

### D.1: Data-Level Diagnostics (Cell 17)

**Key findings:**

1. **Label noise is extremely high:** Approximately 26.5% of events fall in the "noise zone" where |return| < 0.5%. At this magnitude, the direction is essentially random -- microstructure noise dominates true signal.

2. **FinBERT sentiment alignment is near-random:** The alignment rate between FinBERT sentiment and actual next-day direction is only **51.6%** -- just 1.6 percentage points above the 50% random baseline. This holds even for events where FinBERT expresses high confidence (|sentiment| > 0.7).

3. **Return distribution is nearly symmetric:** The distribution of next-day returns is approximately normal with mean near zero. Positive and negative returns are almost equally likely, confirming that the binary classification label itself is extremely noisy.

### D.2: Model Prediction Diagnostics (Cell 18)

**Key findings:**

1. **Prediction score distributions nearly overlap:** The mean separation between positive-class and negative-class prediction scores is extremely small (on the order of 1e-4 to 1e-3), indicating the model cannot meaningfully distinguish the two classes.

2. **No sector shows strong AUC:** Per-sector AUC values cluster tightly around 0.50. No individual GICS sector provides a predictability pocket.

3. **High-confidence FinBERT events do not predict better:** Even filtering to events where FinBERT sentiment confidence exceeds 0.7 does not improve AUC.

4. **Large-move events are not easier:** Events with |return| > 2% do not show meaningfully higher AUC, suggesting the signal deficit is fundamental rather than being diluted by noise-zone events.

### Root Cause Analysis

The diagnostics converge on three explanations:

1. **Market efficiency (EMH):** S&P 500 stocks are among the most efficiently priced securities in the world. News information is absorbed into prices within minutes to hours, making next-day prediction from news headlines extremely difficult.

2. **FinBERT limitations:** FinBERT was trained for financial sentiment classification (positive/negative/neutral), not for market impact prediction. Sentiment and market impact are fundamentally different -- a "positive" news article about a company may already be priced in, or may have been expected and thus have no impact.

3. **Task granularity:** Binary next-day direction is the hardest possible formulation. The project later pivoted to ranking (relative performance across stocks) and longer horizons, which proved more tractable.

---

## 5. Signal Fix Attempts and Why They Failed

After the D.1/D.2 diagnostics identified the problems, the notebook implements several signal-enhancement strategies (Cells 4-6, 9, 12). These are the "Phase 1 Signal Fix" attempts.

### Fixes Applied

| Fix | Cell | Description |
|-----|------|-------------|
| News deduplication | Cell 4 | Mean-pool embeddings for same (date, ticker), reducing redundant events |
| Market-adjusted labels | Cell 5 | Label = sign(excess return) instead of sign(raw return) |
| Momentum features | Cell 6 | 9 rolling price statistics (5/10/21-day mean, std, momentum) |
| Additional baselines | Cell 9 | XGBoost, Random Forest, combined feature sets |
| Selective prediction | Cell 12 | Evaluate only on high-confidence predictions |

### Signal Fix Results (Phase 1 Baseline Matrix)

| Model | Val AUC | Test AUC |
|-------|---------|----------|
| B1: FinBERT LR (deduped, mkt-adj) | 0.5043 | 0.5043 |
| B2: Sentiment LR (deduped, mkt-adj) | 0.5076 | 0.5028 |
| B3: XGBoost (all features) | 0.5093 | 0.5020 |
| B4: Momentum-only XGBoost | 0.5052 | 0.5044 |
| B5: Random Forest | 0.5068 | 0.5004 |

**XGBoost (Extreme Gradient Boosting, Chen & Guestrin, KDD 2016):** An optimized gradient boosting library using tree-based learners with regularization; often the best performer on tabular data.

**LightGBM (Ke et al., NeurIPS 2017):** Microsoft's gradient boosting framework using histogram-based algorithms for faster training; similar to XGBoost but with different tree-building strategies.

**All test AUC values remain below 0.51.** The signal fix failed.

### Selective AUC Results

- Top/bottom 5% most confident predictions: AUC = 0.5154
- Top/bottom 10%: similarly near 0.50

**There is no tail signal.** Even the model's most confident predictions carry no useful information.

### Why the Fixes Failed

1. **Deduplication** removed redundancy but did not create signal where none existed.
2. **Market adjustment** correctly removed market beta but the remaining stock-specific signal is still too weak for binary classification.
3. **Momentum features** (which are price-based, not news-based) added minimal information. Price momentum is a well-known factor, but its predictive power for next-day binary direction on individual S&P 500 stocks is negligible.
4. **Selective prediction** confirmed that the model's confidence is not correlated with accuracy -- it does not "know what it knows."

### Literature Comparison

The DGRCL paper (the only published GNN paper tested on 1,000+ stocks) reported only 53% accuracy, consistent with the results obtained here. This suggests the finding is not a bug or implementation error but a fundamental limitation of the task definition.

---

## 6. LLM Validation (GPT-4o-mini) and the STOP Decision

### Motivation

After the signal fix failed with FinBERT features, the natural question was: Would a more capable language model produce better features? Cells 13-15 implement this test using GPT-4o-mini.

### Method (Cell 14)

- **Model:** GPT-4o-mini accessed via OpenRouter API
- **Input:** News headlines from a dev-holdout period (2023-Q4, within training data, never touches val/test)
- **Output:** Structured JSON with 4 fields: impact_level (high/medium/low), direction (positive/negative/neutral), confidence (float), reasoning_type (earnings/macro/sentiment/technical/other)
- **Sample size:** ~7,000 events
- **Features produced:** 10-dimensional vector (one-hot encoded categorical fields + continuous confidence)

### Results (Cell 15)

5-fold cross-validation AUC on dev-holdout:

| Feature Set | AUC |
|-------------|-----|
| FinBERT sentiment (3-dim) | ~0.500 |
| LLM structured (10-dim) | ~0.501 |
| Combined (13-dim) | ~0.501 |
| FinBERT embedding (768-dim) | ~0.500 |
| LLM + embedding (778-dim) | ~0.501 |

**GPT-4o-mini vs FinBERT delta: +0.0009** -- essentially no difference.

**High-impact event subset (events labeled "high" by GPT-4o-mini):** AUC = **0.4762** -- actually worse than random. The LLM's concept of "high impact" does not correspond to actual next-day price movements.

### The STOP Decision

Based on these results, the project made a definitive STOP decision on the binary direction prediction approach:

1. **The problem is not the model.** Both a specialized financial NLP model (FinBERT) and a general-purpose frontier LLM (GPT-4o-mini) produce features with zero predictive power for next-day direction on S&P 500 stocks.
2. **The problem is the task definition.** Binary direction prediction on highly efficient large-cap stocks is fundamentally intractable with news-based features.
3. **Upgrading to GPT-4o or other LLMs would not help.** The +0.0009 delta demonstrates that the bottleneck is not model capability.

This STOP decision led directly to the v3 pivot: changing from binary classification to ranking, from event-driven to calendar-driven, and from AUC to IC/ICIR/Sharpe evaluation metrics. The v3 approach is implemented in a separate notebook (`v3_ranking_pipeline.ipynb`).

---

## 7. Key Terminology Reference

| Term | One-Line Explanation |
|------|---------------------|
| **AUC (Area Under ROC Curve)** | Measures how well a binary classifier ranks positive vs negative examples; 0.5 = random guessing |
| **GraphSAGE (Hamilton et al., NeurIPS 2017)** | A GNN that learns node representations by sampling and aggregating features from local neighborhoods |
| **SAGEConv** | PyTorch Geometric implementation of GraphSAGE convolution using mean aggregation |
| **GATConv (Velickovic et al., ICLR 2018)** | Graph Attention Network layer that uses attention mechanisms to weight neighbor contributions |
| **HGTConv (Hu et al., WWW 2020)** | Heterogeneous Graph Transformer that learns type-specific attention for different node and edge types |
| **to_hetero()** | PyG utility that converts a homogeneous GNN into a heterogeneous one by replicating parameters per edge type |
| **FinBERT (ProsusAI/finbert)** | A BERT model fine-tuned on financial text for sentiment classification; outputs 768-dim embeddings + 3-dim sentiment |
| **BERT (Devlin et al., NAACL 2019)** | Bidirectional Encoder Representations from Transformers; a foundational pre-trained language model |
| **GPT-4o-mini (OpenAI)** | A cost-efficient variant of GPT-4o supporting structured JSON output |
| **XGBoost (Chen & Guestrin, KDD 2016)** | An optimized gradient boosting library using tree-based learners |
| **LightGBM (Ke et al., NeurIPS 2017)** | Microsoft's gradient boosting framework using histogram-based tree building |
| **Logistic Regression** | A linear model for binary classification using sigmoid activation |
| **Random Forest** | An ensemble of decision trees trained on random data/feature subsets, aggregated by voting |
| **SGDClassifier** | Scikit-learn's stochastic gradient descent classifier; used here with log_loss for scalable logistic regression |
| **BCEWithLogitsLoss** | Binary cross-entropy loss that applies sigmoid internally for numerical stability |
| **EMH (Efficient Market Hypothesis)** | The theory that asset prices fully reflect all available information |
| **GICS (Global Industry Classification Standard)** | A sector classification system assigning companies to 11 sectors |
| **Pearson Correlation** | A measure of linear correlation between two variables, ranging from -1 to +1 |
| **HeteroData** | PyTorch Geometric's data structure for heterogeneous graphs with multiple node and edge types |
| **Early Stopping** | A regularization technique that halts training when validation performance stops improving |
| **Dropout** | A regularization technique that randomly zeroes neuron outputs during training to prevent overfitting |
| **Market-Adjusted Return** | Stock return minus market return, isolating stock-specific performance from market-wide movements |
| **Selective Prediction** | Predicting only on high-confidence samples; tests whether the model "knows what it knows" |
| **Stratified K-Fold CV** | Cross-validation that preserves class proportions in each fold for robust AUC estimates |
| **Memory Mapping (mmap)** | Loading files on-demand from disk rather than into RAM; used here for large embedding arrays |
| **IC (Information Coefficient)** | Spearman correlation between predicted and actual rankings; used in the v3 pivot |
| **Sharpe Ratio** | Annualized return divided by annualized volatility; measures risk-adjusted performance |
| **DGRCL** | A published GNN paper tested on 1,000+ stocks, achieving only 53% accuracy -- consistent with our results |

---

*Document generated: 2026-03-27*
*Notebook: `/Users/heruixi/Desktop/GNN-Testing/phase_c_model_training.ipynb`*
*Research log reference: `/Users/heruixi/Desktop/GNN-Testing/docs/research_log_2026-03-06.md` (Sections 6-9)*
