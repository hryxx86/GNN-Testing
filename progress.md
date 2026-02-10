# Project Progress Tracker

## Project Description

GNN-based stock correlation network analysis and news-driven stock movement prediction using S&P 500 data. The project has two phases:

- **Phase 1:** Build a stock correlation graph from 5-year price data, apply GCN for exploratory embedding, and visualize network structure (sectors, hubs, t-SNE).
- **Phase 2:** Acquire financial news from Factiva for hub stocks, encode with SentenceTransformer, construct a heterogeneous graph (news + stock nodes), and train GraphSAGE to predict next-day stock movement.

**Main notebook:** `GNN测试1 colab.ipynb` (Google Colab / local)

---

## Completed Steps

### Phase 1: Network Analysis
- [x] Download S&P 500 5-year daily closing prices via yfinance (502 stocks x 1255 days)
- [x] Compute Pearson correlation matrix and build graph (threshold |corr| > 0.6, 3198 edges)
- [x] Define and run 2-layer GCN (1 -> 16 -> 1, single forward pass, no training)
- [x] Fetch and cache market cap data from yfinance (`sp500_market_caps.csv`)
- [x] Fetch and cache GICS sector data from Wikipedia (`sp500_sectors.csv`)
- [x] Plot: Top 100 market cap correlation network
- [x] Plot: Top 100 sector-colored correlation network
- [x] Plot: Full 502-stock correlation network
- [x] Plot: Top 10 hub stocks and their neighbors
- [x] Plot: t-SNE of GCN embeddings (16D -> 2D)

### Phase 2: News-Driven Prediction
- [x] Acquire 1,900 Factiva articles for 9 hub companies (2021-01-29 to 2026-01-28)
- [x] Process RTF -> CSV with `process_news.py` (textutil + deduplication)
- [x] Build event-level dataset with `prepare_events.py` (ticker matching, next-day return labels) -> 480 rows
- [x] Encode text with SentenceTransformer (all-MiniLM-L6-v2, 384-dim embeddings)
- [x] Construct HeteroData graph (480 news nodes + 9 stock nodes, time-series 80/10/10 split)
- [x] Baseline: Logistic Regression (Val AUC 0.52, Test AUC 0.62)
- [x] Train GraphSAGE (2-layer, hidden=32, early stopping) -> Val AUC 0.64, Test AUC 0.69

### Infrastructure & Fixes
- [x] Fix Google Drive path (`GNN测试` not `GNN-Testing`)
- [x] Fix SentenceTransformer cell (removed unnecessary cache deletion)
- [x] Add reproducibility seed setting (Cell 0 + GraphSAGE cell)
- [x] Create detailed `REPORT.md` covering all phases
- [x] Move Google Drive mount to Cell 0 so all cells can find cached files (prevents re-downloading)

---

## Current Status

**All Phase 1 & Phase 2 steps are complete.** The notebook runs end-to-end with reproducible results (seed=42). GraphSAGE outperforms the text-only baseline by +6.9% Test AUC.

---

## Next Steps

- [ ] Explore model improvements (more news data, additional features, hyperparameter tuning)
- [ ] Consider adding stock-stock edges to the heterogeneous graph
- [ ] Experiment with different embedding models or fine-tuning
- [ ] Extend to more tickers beyond the 9 hub stocks

---

*Last updated: 2026-02-09*
