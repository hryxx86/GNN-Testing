# Story A — Annotated References

> Companion bibliography for the paper "When Do Graph Neural Networks Help in Cross-Sectional Stock Ranking? A Multi-Seed, Multi-Universe, Cost-Aware Study of US S&P 500." Each entry gives the venue, a DOI / arXiv link, **what the paper did**, **how they did it**, and **why we cite it** (the load-bearing connection to our methodology or claims). Citation numbers are in-text reference IDs used by `docs/storya_paper_draft_v2.md` (the confirmatory draft; the older `storya_paper_draft.md` PILOT version is archived under `archived/docs/`).

> **Editing note.** Where a venue or DOI is uncertain, the entry is marked `⚠ verify` so the next reviewer can confirm against the publisher record before submission.

---

## A. GNN-Finance Prior Art (cited in §1 and §2.1)

### [1] Feng, Chen, He, Ding, Sun & Chua (2019) — Temporal Relational Ranking for Stock Prediction

- **Venue**: *ACM Transactions on Information Systems (TOIS)*, Vol. 37, No. 2, Article 27.
- **DOI**: [10.1145/3309547](https://doi.org/10.1145/3309547)
- **arXiv**: [1809.09441](https://arxiv.org/abs/1809.09441)
- **What it did**: First end-to-end framework that frames stock prediction as a *learning-to-rank* problem over a graph of inter-stock relations, evaluated on NASDAQ and NYSE 2013–2017.
- **How it did it**: A per-stock LSTM produces a sequential embedding; a Temporal Graph Convolution then propagates information across two relation graphs — *industry classification* (~110 industries) and *first-order Wikidata company relations* (e.g., "subsidiary of"). The model is trained on a pairwise ranking loss (rather than MSE regression) and evaluated on IC, MRR, and Investment Return Ratio.
- **Why we cite**: The most-referenced GNN-for-stock-ranking baseline in the literature. Our paper revisits its single-split / single-seed evaluation regime under a 10-seed × 5-fold × cost-aware protocol (§5).

### [2] Kim, Lee, Lee, Hong, Park, Lee & Choi (2019) — HATS: A Hierarchical Graph Attention Network for Stock Movement Prediction

- **Venue**: arXiv preprint (IJCAI 2019 Workshop on Financial Technology).
- **arXiv**: [1908.07999](https://arxiv.org/abs/1908.07999)
- **What it did**: Introduces **HATS**, a hierarchical graph attention network that handles many relation types simultaneously for next-day movement-direction prediction on Korean (KOSPI) and US (S&P 500) markets.
- **How it did it**: Stocks are connected via **75 relation types** extracted from Wikidata (e.g., "owned by", "industry", "parent organization"). For each relation a separate attention head learns neighbour weights; a hierarchical relation-attention layer then aggregates across relation types into one stock embedding. A GRU on top predicts up / down direction at horizon = 1 day.
- **Why we cite**: The blueprint for our **HATS-3R-adapt** baseline (Limitation L5), which adapts HATS to 3 relation types (correlation, GICS sector, news co-occurrence) and switches the head from binary classification to 21-day cross-sectional ranking.

### [3] Sawhney, Agarwal, Wadhwa & Shah (2021) — Stock Selection via Spatiotemporal Hypergraph Attention Network: A Learning-to-Rank Approach (STHAN-SR)

- **Venue**: *Proceedings of AAAI Conference on Artificial Intelligence (AAAI 2021)*. ⚠ verify exact DOI.
- **DOI / Link**: [AAAI proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/16127)
- **What it did**: Re-frames stock selection as a learning-to-rank problem over a **hypergraph** — edges that connect more than two stocks at once (e.g., all members of a sector or industry) — combined with temporal attention.
- **How it did it**: A spatiotemporal hypergraph attention module operates on two views: a temporal hyperedge (rolling-window co-movement) and an industry hyperedge (static sector membership). Output is a listwise rank score per stock, trained with a NDCG-surrogate ranking loss; evaluated on NASDAQ and NYSE with backtested L/S returns.
- **Why we cite**: Methodological precedent for using rich relational structure for cross-sectional ranking. Our paper replicates the *conditional* advantage claim under cherry-pick defence (Hansen SPA + BH-FDR) and finds it does not survive.

---

## B. Quantitative-Finance Methodology (cited in §2.2 and §3)

### [4] Hou, Xue & Zhang (2020) — Replicating Anomalies

- **Venue**: *Review of Financial Studies (RFS)*, 33(5):2019–2133.
- **DOI**: [10.1093/rfs/hhy131](https://doi.org/10.1093/rfs/hhy131)
- **What it did**: Systematically replicates **452 cross-sectional anomalies** from the published finance literature under a uniform research protocol; finds that 65% of them fail to replicate at the 5% level.
- **How it did it**: Apply consistent NYSE-breakpoint sorts (to suppress micro-cap noise), a uniform 1967–2016 sample, value-weighted portfolios, and Newey-West standard errors on monthly L/S returns. Each anomaly is re-tested under the same Fama-French q-factor model so that pass/fail decisions are comparable.
- **Why we cite**: Foundational motivation for our multi-comparison discipline. Our §3.3 / §3.5 ledger borrows the spirit of "do not weaken the eval rules to make findings look better."

### [5] López de Prado (2018) — Advances in Financial Machine Learning

- **Publisher**: John Wiley & Sons. ISBN 978-1-119-48208-6.
- **Link**: [Wiley product page](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
- **What it did**: Comprehensive monograph codifying machine-learning practices for financial-market prediction; the single most-cited "how not to overfit your time-series ML" reference in industry.
- **How it did it**: Step-by-step chapters on triple-barrier labels, *purge-and-embargo* cross-validation, combinatorial purged CV, meta-labeling, the SADF / CUSUM filters, fractional differencing, and bet-sizing. Each method comes with Python pseudocode and a worked example.
- **Why we cite**: Our **walk-forward + 21-day purge embargo** (§3.1) and our **block-bootstrap** procedure (§3.3) follow the playbook from Chapters 7 and 12 of this book.

### [6] Jegadeesh & Titman (1993) — Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency

- **Venue**: *Journal of Finance*, 48(1):65–91.
- **DOI**: [10.1111/j.1540-6261.1993.tb04702.x](https://doi.org/10.1111/j.1540-6261.1993.tb04702.x)
- **What it did**: Seminal paper documenting the **momentum anomaly**: a strategy that buys past 3–12-month winners and shorts past 3–12-month losers earns significant abnormal returns over 1965–1989.
- **How it did it**: Sort NYSE / AMEX stocks into deciles by past J-month returns (J ∈ {3, 6, 9, 12}); form equal-weight, monthly-rebalanced L/S portfolios; track returns over K ∈ {3, 6, 9, 12} months; estimate Jensen's alpha relative to CAPM and three-factor models.
- **Why we cite**: Our **Universe-B `mom12m` feature** and our **top-decile equal-weight dollar-neutral L/S portfolio** are direct descendants of this protocol.

### [7] Yang, Liu, Zhou, Liu, Bian & Liu (2020) — Qlib: An AI-oriented Quantitative Investment Platform

- **Venue**: arXiv preprint (Microsoft Research technical report).
- **arXiv**: [2009.11189](https://arxiv.org/abs/2009.11189)
- **GitHub**: [microsoft/qlib](https://github.com/microsoft/qlib)
- **What it did**: Open-sources the platform Microsoft Research uses for end-to-end quantitative investment research; ships with the standard *Alpha158* and *Alpha360* feature handlers.
- **How it did it**: Provides modular abstractions for `DataHandler` (loading + alignment), `Feature` library (158 hand-engineered cross-sectional indicators bundled into named groups — ROC, MA, KMID, BETA, RANK, RSV, CORR, …), `Model` (LightGBM, MLP, GAT, etc.), and `Evaluator` (IC, ICIR, Sharpe). Walk-forward backtests run on H100 / V100 with one config file.
- **Why we cite**: Our **Universe C** = the project's top-15 Plan-AAA-ranked Alpha158 factor groups (51 columns); see Table T_FACTORS in §7.

---

## C. Forecast-Comparison and Multiple-Testing Statistics (cited in §3.3)

### [8] Diebold & Mariano (1995) — Comparing Predictive Accuracy

- **Venue**: *Journal of Business & Economic Statistics*, 13(3):253–263.
- **DOI**: [10.1080/07350015.1995.10524599](https://doi.org/10.1080/07350015.1995.10524599)
- **What it did**: Introduces the **Diebold-Mariano test**: a paired t-test for whether two forecasting models have equal expected loss, robust to general loss functions, non-Gaussianity, contemporaneous correlation, and serial correlation of the loss differential.
- **How it did it**: Define `d_t = L(forecast_A, actual_t) − L(forecast_B, actual_t)`; the DM statistic `mean(d) / sqrt(2π·f_d(0)/T)` is asymptotically N(0,1), where `f_d(0)` is the long-run spectral density of `d_t` at frequency zero (estimated via a HAC variance estimator).
- **Why we cite**: Foundation of our **DM/HLN pairwise tests** in §3.3.

### [9] Harvey, Leybourne & Newbold (1997) — Testing the Equality of Prediction Mean Squared Errors

- **Venue**: *International Journal of Forecasting*, 13(2):281–291.
- **DOI**: [10.1016/S0169-2070(96)00719-4](https://doi.org/10.1016/S0169-2070(96)00719-4)
- **What it did**: Provides a **small-sample correction** to the Diebold-Mariano test for the case when the sample size T is modest relative to the forecast horizon h.
- **How it did it**: Multiply the DM statistic by `√[(T + 1 − 2h + h(h−1)/T) / T]` and refer to a **Student-t(T−1)** distribution instead of N(0,1). The resulting test maintains the correct size when T is small.
- **Why we cite**: Our pooled confirmatory test span T = 749 days is still small relative to h = 21 (rebalance horizon), so this correction is essential. Applied in [compute_family1_ladder.py](../compute_family1_ladder.py) (import-reuse of `compute_e6_dm_spa.py` helpers).

### [10] Hansen (2005) — A Test for Superior Predictive Ability

- **Venue**: *Journal of Business & Economic Statistics*, 23(4):365–380.
- **DOI**: [10.1198/073500105000000063](https://doi.org/10.1198/073500105000000063)
- **What it did**: Provides a **multi-model multiple-comparison test** that controls the test's size under model-selection bias — the right tool when you want to ask "is the best of my M candidates truly better than the benchmark, after I picked the best?"
- **How it did it**: Compute `T_SPA = max over k of √T · max(0, d̄_k / ω̂_k)` where `d̄_k = mean loss differential` of candidate *k* vs benchmark and `ω̂_k = HAC long-run SE`. Approximate the null distribution by **stationary bootstrap** (Politis & Romano, [11]); report `p_consistent` (assumes poor models have zero mean), `p_lower`, and `p_upper` variants.
- **Why we cite**: Our headline cherry-pick defence (Family-1 SPA); confirmatory p_consistent = 0.2767 / 0.0774 for universes B / C (M = 9, T = 749; source `artifacts/storya_v21_family1/family1_spa.csv`), neither rejecting "no tuned arm beats tuned LightGBM".

### [11] Politis & Romano (1994) — The Stationary Bootstrap

- **Venue**: *Journal of the American Statistical Association*, 89(428):1303–1313.
- **DOI**: [10.1080/01621459.1994.10476870](https://doi.org/10.1080/01621459.1994.10476870)
- **What it did**: Introduces a bootstrap resampling scheme for stationary time series that preserves the joint dependence structure of the original data.
- **How it did it**: Resample *geometric-length blocks* of consecutive observations (block length L ~ Geom(p), where p = expected-block-length⁻¹). The geometric length makes the bootstrap series exactly stationary in expectation, in contrast to fixed-length block bootstrap.
- **Why we cite**: The bootstrap mechanism underneath Hansen SPA ([10]) and our block-bootstrap CIs (§3.3).

### [12] Newey & West (1987) — A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix

- **Venue**: *Econometrica*, 55(3):703–708.
- **DOI**: [10.2307/1913610](https://doi.org/10.2307/1913610)
- **What it did**: Provides the famous **Newey-West HAC** (Heteroskedasticity and Autocorrelation Consistent) covariance-matrix estimator — the standard SE for any time-series regression with autocorrelated residuals.
- **How it did it**: Weight the sample autocovariances with a *Bartlett kernel* (declining triangular weights) up to lag L; produces a positive-semi-definite estimate that is robust to both serial correlation (up to lag L) and heteroskedasticity of unknown form.
- **Why we cite**: The SE used inside the DM/HLN test (§3.3) and inside the Plan AAA NW-t statistics (Table ST4).

### [13] Benjamini & Hochberg (1995) — Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing

- **Venue**: *Journal of the Royal Statistical Society Series B*, 57(1):289–300.
- **DOI**: [10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)
- **What it did**: Introduces the **BH-FDR procedure**: a multiple-testing correction that controls the expected proportion of false discoveries among rejected nulls (FDR), rather than the family-wise error rate (which Bonferroni controls).
- **How it did it**: Sort p-values `p₍₁₎ ≤ … ≤ p₍ₘ₎`; find the largest `k` such that `p₍ₖ₎ ≤ k·q/m`; reject all hypotheses with p ≤ p₍ₖ₎. Controls FDR at level q under independence and many positive-dependence cases.
- **Why we cite**: Our multi-comparison family controls — Family-1 DM/HLN over the pre-registered 20-test ladder family, and Family-2 BH-FDR over the 6 fixed-capacity edge contrasts.

---

## D. Graph Neural Network Architectures (cited in §4.3)

### [14] Veličković, Cucurull, Casanova, Romero, Liò & Bengio (2018) — Graph Attention Networks

- **Venue**: *International Conference on Learning Representations (ICLR 2018)*.
- **arXiv**: [1710.10903](https://arxiv.org/abs/1710.10903)
- **What it did**: Introduces **GAT**, a graph neural network architecture that uses *self-attention* to learn per-edge importance weights for neighbour aggregation, replacing fixed Laplacian-style weights used by spectral GCNs.
- **How it did it**: For each (target i, source j) edge compute attention `a_{ij} = softmax_j(LeakyReLU(aᵀ[Wh_i ‖ Wh_j]))`; the new node representation is `h'_i = σ(Σ_j a_{ij} W h_j)`; multi-head attention concatenates K independent heads. Tested on Cora, Citeseer, PubMed, and PPI.
- **Why we cite**: Our **GAT** model in §4.3 (2 layers, hidden 64, 4 heads).

### [15] Hamilton, Ying & Leskovec (2017) — Inductive Representation Learning on Large Graphs

- **Venue**: *Neural Information Processing Systems (NeurIPS 2017)*.
- **arXiv**: [1706.02216](https://arxiv.org/abs/1706.02216)
- **What it did**: Introduces **GraphSAGE** (Graph SAmple and aggreGatE), an *inductive* GNN that learns aggregator functions over sampled fixed-size neighbourhoods — generalising to unseen nodes without retraining (unlike transductive GCNs).
- **How it did it**: Sample K neighbours per node per layer; aggregate via one of {mean, max-pool, LSTM, GCN-style}; concatenate aggregated neighbour representation with self-representation; transform via a dense layer. Tested on Reddit, PPI, and Citation networks.
- **Why we cite**: Our **SAGE-Mean** model (mean aggregator variant) in §4.3, and the backbone of the edge ablation in §5.4.

---

## E. Tooling and Statistical Software (cited in §3.3)

### [16] Sheppard et al. — `arch` Python package

- **Maintainer**: Kevin Sheppard, University of Oxford.
- **Link**: [arch.readthedocs.io](https://arch.readthedocs.io/) · [GitHub: bashtage/arch](https://github.com/bashtage/arch)
- **What it does**: Open-source Python library implementing **ARCH/GARCH** volatility models, bootstrap utilities, and (relevant to us) the **Hansen SPA** test.
- **How it does it**: The `arch.bootstrap.SPA` class implements the original Hansen ([10]) procedure with stationary bootstrap ([11]), parameter-controlled block length, and returns `p_consistent`, `p_lower`, `p_upper` variants. Vectorised over candidates.
- **Why we cite**: Implementation backbone for our SPA test reproduction; see [compute_e6_dm_spa.py:352–443](../compute_e6_dm_spa.py).

---

## F. Additional GNN-Finance Baselines (cited in §2.3 Table T8 and §4.3)

> Verified 2026-06-24 via arXiv / AAAI / ACM DL (literature-review skill). Each is included in the Table T8 related-work matrix as a single-split, point-estimate prior work without a multi-seed / SPA-FDR / cost-ladder protocol.

### [17] Cheng & Li (2021) — Modeling the Momentum Spillover Effect for Stock Prediction via Attribute-Driven Graph Attention Networks (AD-GAT)

- **Venue**: *AAAI 2021*, 35(1):55–62. [AAAI proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/16077)
- **What / how**: An **unmasked** attention mechanism infers dynamic firm relations from observed market signals (tensor-based feature extractor), modelling attribute-sensitive momentum spillovers; evaluated on three years of S&P 500.
- **Why we cite**: Representative of the **dense, unmasked learned-attention** family that our **L6** rung stands in for (§4.3); reports a single-split point estimate without multi-seed / SPA / cost-ladder.

### [18] Lin, Zhou, Liu & Bian (2021) — Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor and Optimal Transport (TRA)

- **Venue**: *KDD 2021*. **DOI**: [10.1145/3447548.3467358](https://doi.org/10.1145/3447548.3467358) · **arXiv**: [2106.12950](https://arxiv.org/abs/2106.12950)
- **What / how**: A lightweight router dispatches samples to multiple pattern-specific predictors, optimised with an Optimal-Transport assignment; runs on Qlib Alpha158/360; reports IC 0.053→0.059 over Attention-LSTM.
- **Why we cite**: A strong Qlib-based ranking baseline that, like the GNN works, reports a single-split point estimate without a cherry-pick defence or cost ladder.

### [19] Li, Liu, Shen, Wang, Chen & Huang (2024) — MASTER: Market-Guided Stock Transformer for Stock Price Forecasting

- **Venue**: *AAAI 2024*, 38(1):162–170. **arXiv**: [2312.15235](https://arxiv.org/abs/2312.15235)
- **What / how**: Alternating intra-stock and inter-stock attention with market-guided feature gating.
- **Why we cite**: Representative **dense learned-attention** model (with AD-GAT) for the L6 framing (§4.3).

### [20] Chen et al. (2025) — FinMamba: Market-Aware Graph Enhanced Multi-Level Mamba for Stock Movement Prediction

- **Venue**: preprint. **arXiv**: [2502.06707](https://arxiv.org/abs/2502.06707)
- **What / how**: Multi-level Mamba state-space model with a **dynamic graph + pruning** module; evaluated on CSI 300/500, S&P 500, NASDAQ 100.
- **Why we cite**: Representative **learned-sparse** graph architecture (pruned dynamic graph) named in Limitation L6 as outside the dense-attention family our L6 rung covers (§4.3, §7).

### [21] Schlichtkrull, Kipf, Bloem, van den Berg, Titov & Welling (2018) — Modeling Relational Data with Graph Convolutional Networks (R-GCN)

- **Venue**: *ESWC 2018*. **arXiv**: [1703.06103](https://arxiv.org/abs/1703.06103)
- **What / how**: Relation-specific weight matrices for multi-relational graph convolution.
- **Why we cite**: Methodological ancestor of our **HATS-3R-adapt** (L7) per-relation parameterisation (§4.3).

---

## Cross-reference summary

| Method / claim in paper | Reference(s) |
|---|---|
| GNN-finance prior baseline framing | [1] Feng 2019, [2] Kim 2019 (HATS), [3] Sawhney 2021 (STHAN-SR) |
| Multi-comparison / cherry-pick discipline | [4] Hou-Xue-Zhang 2020, [5] López de Prado 2018 |
| Walk-forward + purge embargo | [5] López de Prado 2018 Ch. 7 |
| 12-month momentum + top-decile L/S | [6] Jegadeesh-Titman 1993 |
| Alpha158 features (Universe C) | [7] Qlib (Yang et al. 2020) |
| Hansen SPA test | [10] Hansen 2005 (theory), [11] Politis-Romano 1994 (bootstrap), [16] `arch` (implementation) |
| Diebold-Mariano + HLN small-sample correction | [8] Diebold-Mariano 1995, [9] Harvey-Leybourne-Newbold 1997 |
| Newey-West HAC SE | [12] Newey-West 1987 |
| BH-FDR multi-comparison control | [13] Benjamini-Hochberg 1995 |
| GAT architecture | [14] Veličković 2018 |
| GraphSAGE / SAGE-Mean architecture | [15] Hamilton 2017 |

---

## To-be-added before submission (deferred)

**ADDED 2026-06-24** (verified via arXiv / AAAI / ACM DL, literature-review skill → now in §F above and Table T8): [17] AD-GAT (Cheng & Li 2021), [18] TRA (Lin et al. 2021), [19] MASTER (Li et al. 2024), [20] FinMamba (Chen et al. 2025), [21] R-GCN (Schlichtkrull et al. 2018). The Related-Work matrix (Table T8) now covers 7 prior works + this study.

Still-deferred candidates (optional, only if page budget allows):

- *Stockformer / Transformer + graph hybrid (2022)* — multiple distinct "Stockformer" papers exist; verify the exact citation before slotting into §2.1.
- *Pinheiro & Wedge 2022 — Self-Supervised Pretraining for Stock-Movement Prediction* — relevant to future-work item (self-supervised pre-training).
- *Combinatorially-Purged CV* (López de Prado Chapter 12) — already in [5]; consider deeper treatment.
- A Diebold (2015) retrospective on the DM test — useful for §3.3.
- The Romano-Wolf StepM procedure (2005) as an alternative to BH-FDR — could strengthen §3.3.
- The Romano-Wolf StepM procedure (2005) as an alternative to BH-FDR — could strengthen §3.3.

Each candidate, once verified, should follow the same six-field format above.
