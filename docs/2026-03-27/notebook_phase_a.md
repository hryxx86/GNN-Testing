# Notebook Documentation: phase_a_data_prep.ipynb — Phase A: Full-Scale Data Pipeline (EODHD + FinBERT)

## Table of Contents

1. [Overview](#1-overview)
2. [Cell-by-Cell Walkthrough](#2-cell-by-cell-walkthrough)
   - [Setup Cell: Environment Configuration](#setup-cell-environment-configuration)
   - [Step 1: EODHD News Download](#step-1-eodhd-news-download)
   - [Step 1b: Download Status Check](#step-1b-download-status-check)
   - [Step 2: Data Validation & Cleaning](#step-2-data-validation--cleaning)
   - [Step 2b: Deduplication & Cleaning](#step-2b-deduplication--cleaning)
   - [Step 2c: Build Event-Level Dataset](#step-2c-build-event-level-dataset)
   - [Step 3: FinBERT Setup](#step-3-finbert-setup)
   - [Step 3b: FinBERT Batch Encoding](#step-3b-finbert-batch-encoding)
   - [Step 3c: Save Embeddings & Metadata](#step-3c-save-embeddings--metadata)
   - [Step 4: Embedding Validation](#step-4-embedding-validation)
   - [Step 4b: t-SNE Visualization](#step-4b-t-sne-visualization)
3. [Data Pipeline Flow](#3-data-pipeline-flow)
4. [Key Data Statistics](#4-key-data-statistics)
5. [The Calendar-Driven vs Event-Driven Paradigm](#5-the-calendar-driven-vs-event-driven-paradigm)
6. [Output Files](#6-output-files)
7. [Technical Decisions](#7-technical-decisions)

---

## 1. Overview

This notebook implements **Phase A** of the GNN-Testing project: constructing a full-scale news event dataset for all S&P 500 stocks, encoding article titles with FinBERT, and aligning news events with next-day stock returns. It replaces the smaller Factiva-based pilot dataset (Phase 2 Pilot) with a much larger corpus sourced from the EODHD API.

**Goal**: Produce a clean, labeled dataset of (news event, stock, next-day return direction) triples with FinBERT semantic embeddings and sentiment scores, ready for downstream GNN model training in Phase C.

**Runtime environment**: Google Colab with GPU (for FinBERT inference). The notebook also supports local execution with a fallback working directory.

**Key technologies**:

- **EODHD API**: A financial data provider offering historical news events with structured metadata, including article text, associated stock symbols, publication timestamps, and pre-computed sentiment scores. It serves as the primary data source for this pipeline.
- **FinBERT** (ProsusAI/finbert): A BERT model fine-tuned on financial text (Financial PhraseBank, financial news), producing 768-dimensional sentence embeddings and 3-dimensional sentiment scores (positive/negative/neutral). It is based on BERT (Bidirectional Encoder Representations from Transformers, Devlin et al. NAACL 2019), a pre-trained language model that learns contextual word representations via masked language modeling and next-sentence prediction.
- **Sentence embedding**: A fixed-length vector representation of a text passage (here, a news article title), capturing its semantic meaning in a continuous vector space. In this notebook, the [CLS] token output from FinBERT's last hidden layer serves as the sentence embedding.

---

## 2. Cell-by-Cell Walkthrough

### Setup Cell: Environment Configuration

**Cell ID**: `setup-cell`

**Purpose**: Detect whether the notebook is running on Google Colab or locally, mount Google Drive if on Colab, set the working directory, and create required data subdirectories.

**What it does**:
1. Attempts to import `google.colab`. If successful, mounts Google Drive and sets the working directory to `/content/drive/MyDrive/GNN测试` (the shared project folder on Drive).
2. If running locally, sets the working directory to `/Users/heruixi/Desktop/GNN-Testing`.
3. Creates four subdirectories under `data/`: `reference`, `fullscale`, `pilot`, and `dynamic_graphs`.

**Why this matters**: All subsequent cells use relative paths (e.g., `data/fullscale/...`), so the working directory must be set correctly. The Colab path uses the Chinese folder name `GNN测试` (not `GNN-Testing`) because that is the actual Google Drive folder name.

---

### Step 1: EODHD News Download

**Cell ID**: `eodhd-download`

**Purpose**: Download historical news articles for all S&P 500 tickers from the EODHD API, with robust checkpointing and resume capability.

**What it does**:

1. **Configuration**:
   - Reads an API token from `data/fullscale/.api_token` (kept out of version control for security).
   - Date range: 2021-01-29 to 2026-01-28 (aligned with the project's 5-year price data window).
   - EODHD returns max 1000 articles per request; pagination is handled via the `offset` parameter.
   - Each API request costs 5 API calls (paid plan allows 100k calls/day, i.e., 20k requests/day).

2. **Resume logic**:
   - Loads the S&P 500 ticker list from `data/reference/sp500_sectors.csv` (503 tickers).
   - Checks which tickers are already present in the main output file (`sp500_news_eodhd.parquet`) and in checkpoint files (`_checkpoints/ckpt_NNNN.parquet`).
   - Only downloads data for tickers not yet covered.

3. **Download loop**:
   - For each remaining ticker, issues paginated GET requests to `https://eodhd.com/api/news?s={TICKER}.US`.
   - Extracts fields: `date`, `title`, `content`, `link`, `symbols` (semicolon-separated), `tags`, and EODHD's pre-computed sentiment (`polarity`, `neg`, `neu`, `pos`).
   - **Polarity**: A single scalar sentiment score provided by EODHD's own NLP pipeline (distinct from FinBERT's sentiment, computed later).
   - Saves a checkpoint parquet file every 10 tickers to prevent data loss from runtime interruptions.
   - Sleeps 0.3 seconds between requests to respect rate limits.

4. **Merge step**:
   - After downloading, concatenates the main output file with all checkpoint files.
   - Deduplicates by `(title, date, query_ticker)` to handle overlapping checkpoints.
   - Saves the merged result back to `data/fullscale/sp500_news_eodhd.parquet`.
   - Cleans up checkpoint files after successful merge.
   - Reports final coverage: target is all 503 S&P 500 tickers.

**Key design decision**: The checkpoint-based approach (save every 10 tickers, merge at end) avoids accumulating all data in memory and allows the download to be interrupted and resumed without losing progress. This is important because downloading 503 tickers with pagination can take several hours.

---

### Step 1b: Download Status Check

**Cell ID**: `de53a74f`

**Purpose**: A quick verification cell to check that all S&P 500 tickers have been downloaded.

**What it does**: Loads the raw parquet file, counts unique `query_ticker` values, and compares against the full ticker list from `sp500_sectors.csv`. Reports any missing tickers.

---

### Step 2: Data Validation & Cleaning

**Cell ID**: `data-validation`

**Purpose**: Load the raw downloaded data, compute summary statistics, and identify data quality issues before cleaning.

**What it does**:
1. Loads `sp500_news_eodhd.parquet` and reports row count and column names.
2. Counts missing values per column.
3. Parses the `date` column to UTC datetime; reports the date range.
4. Computes articles-per-ticker distribution (mean, median, min, max) to identify tickers with sparse coverage.
5. Flags tickers with fewer than 10 articles.
6. Reports EODHD sentiment coverage (percentage of articles with non-null `polarity`).

**Data statistics obtained**:
- Raw dataset: 1,698,182 articles across 503 tickers (per the research log).
- Sentiment coverage: the vast majority of articles have EODHD sentiment scores.

---

### Step 2b: Deduplication & Cleaning

**Cell ID**: `dedup-clean`

**Purpose**: Remove duplicates and clean the dataset for downstream use.

**What it does**:
1. **Drop nulls**: Removes rows where `title` or `date` is missing.
2. **Deduplicate by (title, date)**: The same news article often appears multiple times because it was queried via different tickers. This step keeps only the first occurrence. The `symbols` field (which lists all tickers mentioned in the article) is preserved, so ticker associations are not lost.
3. **Date filter**: Restricts to the target date range (2021-01-29 to 2026-01-28).
4. **Clean symbols**: The `symbols` field from EODHD contains entries like `AAPL;AAPL.US;MSFT`. The `clean_symbols()` function strips exchange suffixes (`.US`) and deduplicates, producing a clean semicolon-separated ticker list (e.g., `AAPL;MSFT`).
5. Saves the cleaned dataset to `data/fullscale/sp500_news_clean.parquet`.

**Why deduplication matters**: When a single news article mentions multiple S&P 500 companies, EODHD returns it for each queried ticker. Without deduplication, the same article would be encoded by FinBERT multiple times, wasting compute. The cleaned dataset stores each article once, with the `tickers_clean` field listing all associated tickers.

---

### Step 2c: Build Event-Level Dataset

**Cell ID**: `build-events`

**Purpose**: Transform the article-level dataset into an event-level dataset where each row represents a (news article, individual ticker, next-day return) triple.

**What it does**:

1. **Load cleaned news** and **price data** (`data/reference/sp500_5y_prices.csv`).
2. **Compute returns**: `pct_change()` gives daily close-to-close returns. `shift(-1)` gives the next trading day's return, which serves as the prediction label.
   - **Close-to-close return**: The percentage change from one day's closing price to the next day's closing price, i.e., `(close[T+1] - close[T]) / close[T]`.
3. **Explode**: Split the `tickers_clean` field (e.g., `AAPL;MSFT`) into separate rows, one per ticker. An article mentioning 3 tickers produces 3 event rows.
4. **Filter**: Keep only tickers that exist in the price data (i.e., are valid S&P 500 members).
5. **Return alignment**: For each event, find the next available trading day's return after the article's publication date. This handles weekends and holidays: a Friday article gets Monday's return, a holiday-eve article gets the next trading day's return.
   - The `get_next_return()` function finds the first trading day in the `next_ret` DataFrame that is strictly after the article's (timezone-stripped, normalized) date.
6. **Label creation**: `label = 1` if `return_next > 0` (price went up), `label = 0` otherwise. This is a binary direction prediction task.
7. Drops rows where no return data is available (e.g., articles near the end of the dataset where the next trading day's price is not yet known).
8. Selects output columns: `date, ticker, title, content, polarity, neg, neu, pos, tags, return_next, label`.
9. Saves to `data/fullscale/sp500_news_events.parquet`.

**Data statistics obtained** (from the research log):
- After explode and filtering: approximately 1.54 million event rows.
- After return alignment: 1,538,967 events with valid next-day returns (90.6% match rate from raw articles).
- Label distribution: approximately balanced (close to 50/50 up/down, as expected for daily returns).

**Critical design note**: The return definition (next-day close-to-close, aligned to the next trading day after the news publication date) is locked and cannot be changed. This ensures no data leakage: the label depends only on future price movement that occurs after the news is published.

---

### Step 3: FinBERT Setup

**Cell ID**: `finbert-setup`

**Purpose**: Install dependencies and load the FinBERT model for GPU inference.

**What it does**:
1. Installs `transformers` and `accelerate` via pip.
2. Loads `ProsusAI/finbert` using Hugging Face's `AutoTokenizer` and `AutoModelForSequenceClassification`.
   - **AutoTokenizer**: Hugging Face's auto-detection class that loads the correct tokenizer for a given model checkpoint. For FinBERT, this is a BERT WordPiece tokenizer.
   - **AutoModelForSequenceClassification**: Loads a transformer model with a classification head on top. For FinBERT, this includes the base BERT encoder (12 layers, 768 hidden dimensions) plus a linear classification head that outputs 3 logits (positive, negative, neutral).
3. Moves the model to GPU (`cuda`) and sets it to evaluation mode (`eval()`).

**Results obtained**:
- Device: `cuda` (GPU available on Colab).
- Model size: 109.5M parameters.
- Hidden size: 768 dimensions.

---

### Step 3b: FinBERT Batch Encoding

**Cell ID**: `finbert-encode`

**Purpose**: Encode all news article titles into 768-dimensional FinBERT embeddings and 3-dimensional sentiment probability vectors.

**What it does**:

1. **Deduplication optimization**: Many event rows share the same article title (because one article can be associated with multiple tickers after the explode step). Instead of encoding every row, the code extracts unique titles, encodes them once, and maps back to all rows via an index lookup. This saves significant GPU compute (the savings percentage is reported in the output).

2. **Batch encoding loop**:
   - Processes unique titles in batches of 128.
   - For each batch:
     - **Tokenization**: Converts titles to token IDs with padding and truncation to `max_length=128` tokens. This means titles longer than ~100 words are truncated.
     - **Forward pass** (with `torch.no_grad()` for inference-only, no gradient computation):
       - Extracts `[CLS]` embedding from the last hidden layer: `outputs.hidden_states[-1][:, 0, :]`. The `[CLS]` token is a special token prepended to every input in BERT-based models; its final hidden state serves as a summary representation of the entire input sequence.
       - **L2 normalization** (`F.normalize`): Scales each embedding to unit length (L2 norm = 1.0). This ensures that cosine similarity between embeddings equals their dot product, simplifying downstream distance computations.
       - **Sentiment probabilities**: Applies softmax to the classification head's logits, producing a 3-dimensional probability vector `[P(positive), P(negative), P(neutral)]` that sums to 1.0.
         - **Softmax**: A function that converts raw logits (unbounded real numbers) into a probability distribution by exponentiating and normalizing: `softmax(x_i) = exp(x_i) / sum(exp(x_j))`.

3. **Checkpointing**: Saves intermediate results every 5000 batches to `data/fullscale/_finbert_checkpoint.npz`. If the cell is interrupted, it can resume from the last checkpoint. Uses `float16` (half-precision) storage to reduce file size.

4. **Index mapping**: After encoding all unique titles, maps embeddings back to the full event dataset using the `title_to_idx` dictionary. Each event row gets its corresponding title's embedding and sentiment vector.

5. **Validation**: Checks for NaN values in the final arrays.

**Results obtained**:
- Unique embeddings shape: `(N_unique, 768)` where `N_unique` is the number of distinct titles.
- Final embeddings shape: `(~1.54M, 768)` after mapping back to all event rows.
- Final sentiments shape: `(~1.54M, 3)`.
- No NaN values detected.

---

### Step 3c: Save Embeddings & Metadata

**Cell ID**: `save-embeddings`

**Purpose**: Persist the computed embeddings, sentiment scores, and metadata to disk.

**What it does**:
1. Saves FinBERT embeddings as `data/fullscale/sp500_news_emb_finbert.npy` (shape: `(N_events, 768)`, dtype: float16).
2. Saves FinBERT sentiment probabilities as `data/fullscale/sp500_news_sentiment_finbert.npy` (shape: `(N_events, 3)`, dtype: float16).
3. Saves metadata as `data/fullscale/sp500_news_emb_meta.parquet` containing columns: `date, ticker, label, return_next, polarity, idx`.

**File format notes**:
- `.npy`: NumPy's native binary format for dense arrays. Fast to load, no compression overhead.
- `.parquet`: Apache Parquet columnar storage format, efficient for tabular data with mixed types. Used for metadata because it handles strings and dates well.

---

### Step 4: Embedding Validation

**Cell ID**: `embedding-validation`

**Purpose**: Verify the integrity and statistical properties of the saved embeddings and metadata.

**What it does**:
1. **Row match check**: Confirms that the embedding array, sentiment array, and metadata DataFrame all have the same number of rows.
2. **NaN check**: Verifies no NaN values exist in any output.
3. **L2 norm check**: Computes the mean L2 norm of embeddings (should be approximately 1.0 due to the normalization step).
4. **Dataset statistics**: Reports total events, unique tickers, date range, and label distribution.
5. **Coverage analysis**: Shows the top 10 and bottom 10 tickers by event count, revealing the distribution's skew (some large-cap stocks like AAPL and TSLA have far more news coverage than smaller S&P 500 members).
6. **FinBERT sentiment distribution**: Reports the percentage of events classified as positive, negative, or neutral (based on the argmax of the 3-dim sentiment vector).

---

### Step 4b: t-SNE Visualization

**Cell ID**: `tsne-validation`

**Purpose**: Visually inspect whether FinBERT embeddings capture meaningful structure by projecting them to 2D and coloring by GICS sector.

**What it does**:
1. Loads the GICS sector mapping from `data/reference/sp500_sectors.csv`.
   - **GICS** (Global Industry Classification Standard): A widely-used industry taxonomy developed by MSCI and S&P, classifying companies into 11 sectors (e.g., Technology, Healthcare, Financials).
2. Randomly samples 5000 events for computational tractability.
3. Runs t-SNE dimensionality reduction from 768 dimensions to 2 dimensions.
   - **t-SNE** (t-distributed Stochastic Neighbor Embedding, van der Maaten & Hinton 2008): A nonlinear dimensionality reduction technique that preserves local neighborhood structure. Points that are close in the high-dimensional space remain close in the 2D projection. It is used here for visualization only, not for modeling.
   - Hyperparameters: `perplexity=30` (controls the effective number of neighbors considered), `init='pca'` (initialize with PCA for reproducibility), `learning_rate='auto'`.
4. Creates a scatter plot colored by GICS sector.
5. Saves the plot to `plots/finbert_tsne_by_sector.png`.

**Purpose of this visualization**: If FinBERT embeddings capture industry-specific language patterns, news articles about companies in the same sector should cluster together. Visible sector clusters indicate that the embeddings encode meaningful domain structure. However, for the downstream task (predicting return direction), what matters more is whether the embeddings separate positive-return events from negative-return events, which is investigated in later phases.

---

## 3. Data Pipeline Flow

The complete data pipeline implemented by this notebook proceeds as follows:

```
Step 1: Raw News Acquisition
  EODHD API  ──────────────────────────────────────────────►  sp500_news_eodhd.parquet
  (paginated queries for 503 S&P 500 tickers,                 (raw: ~1.7M articles)
   2021-01-29 to 2026-01-28, with checkpointing)

Step 2: Cleaning & Event Construction
  sp500_news_eodhd.parquet
    │
    ├─ Drop null title/date
    ├─ Deduplicate by (title, date)              ──────────►  sp500_news_clean.parquet
    ├─ Filter to date range                                    (deduplicated articles)
    └─ Clean ticker symbols
                │
                ├─ Explode: 1 article × N tickers = N rows
                ├─ Filter to valid S&P 500 tickers
                ├─ Align with next trading day return  ─────►  sp500_news_events.parquet
                └─ Create binary label (up=1, down=0)          (~1.54M events)

Step 3: FinBERT Encoding
  sp500_news_events.parquet
    │
    ├─ Extract unique titles
    ├─ Batch encode with FinBERT (GPU)
    │   ├─ [CLS] embedding (768-dim, L2-normalized)
    │   └─ Sentiment probs (3-dim: pos/neg/neu via softmax)
    ├─ Map back to all event rows
    │
    ├──────────────────────────────────────────────►  sp500_news_emb_finbert.npy      (N, 768)
    ├──────────────────────────────────────────────►  sp500_news_sentiment_finbert.npy (N, 3)
    └──────────────────────────────────────────────►  sp500_news_emb_meta.parquet      (N rows)

Step 4: Validation
  - Row count consistency checks
  - L2 norm verification (~1.0)
  - NaN checks
  - t-SNE visualization by GICS sector
```

---

## 4. Key Data Statistics

| Metric | Value |
|--------|-------|
| Raw articles downloaded | ~1,698,182 |
| S&P 500 tickers queried | 503 |
| Tickers with data | 503 (100% coverage) |
| Date range | 2021-01-29 to 2026-01-28 |
| Articles after deduplication | Fewer than raw (exact count depends on run; dedup removes cross-ticker duplicates) |
| Events after explode + return alignment | ~1,538,967 (90.6% of raw mapped to valid S&P 500 tickers with returns) |
| FinBERT embedding dimensionality | 768 |
| FinBERT sentiment dimensionality | 3 (positive, negative, neutral) |
| Embedding L2 norm | ~1.0 (unit-normalized) |
| Label distribution | Approximately 50/50 (balanced, as expected for daily return directions) |
| EODHD sentiment coverage | High (vast majority of articles have non-null polarity scores) |
| Storage format for embeddings | float16 `.npy` files |
| Storage format for metadata | Parquet |

---

## 5. The Calendar-Driven vs Event-Driven Paradigm

This notebook produces an **event-driven** dataset: each row corresponds to a specific news event (article publication) paired with a specific stock. The number of events per stock per day varies; some days a stock has zero events, others have many.

This stands in contrast to a **calendar-driven** paradigm, where the dataset has a fixed grid of (trading day, stock) pairs and news features are aggregated to fill each cell. In the calendar-driven approach:
- For each trading day T and stock S, all news articles mentioning S that were published between T-1 close and T close are aggregated (e.g., by mean-pooling their embeddings) into a single feature vector.
- Days with no news for a given stock receive a zero vector or a learned default embedding.
- The resulting tensor has shape `(num_trading_days, num_stocks, feature_dim)`.

The event-driven dataset produced by this notebook is the **raw material** from which either paradigm can be constructed:
- For **event-driven modeling**: Use the events directly. Each prediction is conditioned on a specific news event.
- For **calendar-driven modeling** (used in later phases, specifically Phase C's GNN training): Aggregate events by (trading day, stock) using mean-pooling of FinBERT embeddings, producing a dense tensor. This aggregation is performed in `scripts/prepare_events.py` and the Phase C notebook, not in this notebook.

The project ultimately shifted toward a **calendar-driven approach** combined with GNN-based modeling, where the aggregated daily feature vectors serve as node features in a heterogeneous graph. The event-level dataset from this notebook provides the foundation for that aggregation.

---

## 6. Output Files

All output files are stored under `data/fullscale/`:

| File | Format | Description |
|------|--------|-------------|
| `sp500_news_eodhd.parquet` | Parquet | Raw downloaded articles from EODHD (~1.7M rows) |
| `sp500_news_clean.parquet` | Parquet | Deduplicated and date-filtered articles |
| `sp500_news_events.parquet` | Parquet | Event-level dataset: one row per (article, ticker) with next-day return label |
| `sp500_news_emb_finbert.npy` | NumPy float16 | FinBERT [CLS] embeddings, shape (N_events, 768), L2-normalized |
| `sp500_news_sentiment_finbert.npy` | NumPy float16 | FinBERT sentiment probabilities, shape (N_events, 3) |
| `sp500_news_emb_meta.parquet` | Parquet | Metadata aligned row-by-row with the .npy files (date, ticker, label, return_next, polarity) |

Additionally:
| File | Format | Description |
|------|--------|-------------|
| `plots/finbert_tsne_by_sector.png` | PNG | t-SNE scatter plot of FinBERT embeddings colored by GICS sector |

---

## 7. Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **FinBERT over Fin-E5 or Voyage Finance** | FinBERT (ProsusAI) is the most widely cited financial BERT variant, enabling clean ablation. More advanced embeddings (Fin-E5, Voyage) are deferred to a later ablation study. |
| **Encode titles only, not full article content** | Titles are concise and information-dense. Full article bodies would exceed FinBERT's 512-token context window and introduce noise. Title-only encoding is standard in financial NLP literature. |
| **L2 normalization of embeddings** | Ensures cosine similarity equals dot product, simplifying distance calculations and making embeddings comparable in magnitude. |
| **Deduplication before encoding** | A single article can appear in queries for multiple tickers. Encoding unique titles only (then mapping back) avoids redundant GPU computation, saving approximately 50%+ of FinBERT inference time. |
| **float16 storage** | Halves storage size with negligible precision loss for downstream tasks (GNN training typically operates in float32 anyway; the upcast is handled at load time). |
| **Next-day close-to-close return as label** | Locked project-wide decision. Using the next available trading day after article publication ensures no lookahead bias. |
| **Checkpoint-based download and encoding** | Both the EODHD download and FinBERT encoding support resumption from checkpoints, critical for long-running operations on Colab where runtime disconnections are common. |
| **EODHD sentiment preserved alongside FinBERT sentiment** | EODHD provides its own `polarity` score. This is kept as an auxiliary feature for potential comparison, even though FinBERT sentiment is the primary signal. |
