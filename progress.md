# Project Progress Tracker

## Project Description

GNN-based stock correlation network analysis and news-driven stock movement prediction using S&P 500 data. The project has two phases:

- **Phase 1:** Build a stock correlation graph from 5-year price data, apply GCN for exploratory embedding, and visualize network structure (sectors, hubs, t-SNE).
- **Phase 2:** Acquire financial news from Factiva for hub stocks, encode with SentenceTransformer, construct a heterogeneous graph (news + stock nodes), and train GraphSAGE to predict next-day stock movement.

**Main notebook:** `GNN测试1 colab.ipynb` (Google Colab / local)

## Directory Structure

```
GNN-Testing/
├── GNN测试1 colab.ipynb          # Main notebook
├── progress.md                    # This file
├── data/                          # All data files
│   ├── raw/                       # Raw source data
│   │   └── Newstitle_.../         # Factiva RTF exports
│   ├── sp500_5y_prices.csv        # 5-year daily prices (502 stocks)
│   ├── sp500_market_caps.csv      # Market cap data
│   ├── sp500_sectors.csv          # GICS sector mapping
│   ├── news_clean.csv             # Cleaned Factiva articles
│   ├── news_events.csv/parquet    # Event-level dataset (480 rows)
│   ├── news_events_emb.npy        # SentenceTransformer embeddings (480x384)
│   ├── news_events_emb_meta.parquet
│   ├── sensitivity_analysis.csv   # Dynamic graph statistics
│   └── graph_data.pt              # HeteroData graph object
├── scripts/                       # Python scripts
│   ├── process_news.py            # RTF -> CSV conversion
│   ├── prepare_events.py          # Event-level dataset builder
│   └── build_dynamic_graphs.py    # Rolling window graph builder
├── plots/                         # All visualization outputs
│   └── *.png (13 files)
└── docs/                          # Documentation
    ├── REPORT.md                  # Detailed progress report
    ├── 代码讲解.md                 # Code walkthrough (Chinese)
    └── gnn-llm-prediction-plan.md # Phase 3 design document
```

---

## Completed Steps

### Phase 1: Network Analysis
- [x] Download S&P 500 5-year daily closing prices via yfinance (502 stocks x 1255 days)
- [x] Compute Pearson correlation matrix and build graph (threshold |corr| > 0.6, 3198 edges)
- [x] Define and run 2-layer GCN (1 -> 16 -> 1, single forward pass, no training)
- [x] Fetch and cache market cap data from yfinance (`data/sp500_market_caps.csv`)
- [x] Fetch and cache GICS sector data from Wikipedia (`data/sp500_sectors.csv`)
- [x] Plot: Top 100 market cap correlation network
- [x] Plot: Top 100 sector-colored correlation network
- [x] Plot: Full 502-stock correlation network
- [x] Plot: Top 10 hub stocks and their neighbors
- [x] Plot: t-SNE of GCN embeddings (16D -> 2D)

### Phase 2: News-Driven Prediction
- [x] Acquire 1,900 Factiva articles for 9 hub companies (2021-01-29 to 2026-01-28)
- [x] Process RTF -> CSV with `scripts/process_news.py` (textutil + deduplication)
- [x] Build event-level dataset with `scripts/prepare_events.py` (ticker matching, next-day return labels) -> 480 rows
- [x] Encode text with SentenceTransformer (all-MiniLM-L6-v2, 384-dim embeddings)
- [x] Construct HeteroData graph (480 news nodes + 9 stock nodes, time-series 80/10/10 split)
- [x] Baseline: Logistic Regression (Val AUC 0.52, Test AUC 0.62)
- [x] Train GraphSAGE (2-layer, hidden=32, early stopping) -> Val AUC 0.61, Test AUC 0.64

### Infrastructure & Fixes
- [x] Fix Google Drive path (`GNN测试` not `GNN-Testing`)
- [x] Fix SentenceTransformer cell (removed unnecessary cache deletion)
- [x] Add reproducibility seed setting (Cell 0 + GraphSAGE cell)
- [x] Create detailed `docs/REPORT.md` covering all phases
- [x] Move Google Drive mount to Cell 0 so all cells can find cached files (prevents re-downloading)
- [x] Fix `torch.use_deterministic_algorithms` crash on Colab CUDA (set to False)
- [x] Review and correct REPORT.md (sector names, AUC values, hardware disclaimer)
- [x] Reorganize project directory structure (data/, scripts/, docs/, plots/)

---

## Current Status

**Phase 1 & Phase 2 (original) complete. Phase B (Task 1 dynamic graphs) complete. Phase A (data prep) notebook created, pending EODHD API token.**

---

## In Progress: Phase A Data Preparation

**Notebook:** `phase_a_data_prep.ipynb`

- [ ] Purchase EODHD Calendar & News API ($19.99/mo, student 50% off → ~$10/mo)
- [ ] Download S&P 500 news (500 tickers, 5 years) → `data/sp500_news_eodhd.parquet`
- [ ] Validate & clean news data → `data/sp500_news_clean.parquet`
- [ ] Build event-level dataset (explode by ticker, align next-day return) → `data/sp500_news_events.parquet`
- [ ] FinBERT 768-dim embeddings + 3-dim sentiment → `data/sp500_news_emb_finbert.npy`
- [ ] t-SNE validation of embeddings

## Next Steps (Phase C)

- [ ] Construct heterogeneous graph (3 edge types: news→stock, stock↔stock corr, stock↔stock sector)
- [ ] Implement baselines (LR + FinBERT, FinBERT-LSTM)
- [ ] Implement HeteroGNN model
- [ ] Run ablation experiments
- [ ] Results analysis + visualization

---

*Last updated: 2026-02-25*
