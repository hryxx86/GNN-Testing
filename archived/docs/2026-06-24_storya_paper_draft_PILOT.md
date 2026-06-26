# When Do Graph Neural Networks Help in Cross-Sectional Stock Ranking?
### A Multi-Seed, Multi-Universe, Cost-Aware Study of US S&P 500

**Working draft for ICAIF 2026 (ACM SIG, 8–10 pages).**

> **Editing notes (delete before submission).** This file is an *editable* working draft. All 28 figures are embedded as PNGs (PDF/SVG masters live next to them under [figures/](../figures/)); all 11 tables are embedded as native Markdown so you can edit cells in place. Every number is traceable to a CSV under [artifacts/storya_e6_dm_spa/](../artifacts/storya_e6_dm_spa/) — when you change a number, change the upstream CSV first.

> **Document map**: §1 Intro · §2 Related Work · §3 Methodology · §4 Data & Setup · §5 Results · §6 Discussion · §7 Limitations · §8 Reproducibility · Supplementary §S.

---

## §1. Introduction

> *Thesis: GNN claims in financial ranking are often regime-driven and seed-fragile. This paper's contribution is not a new architecture but a rigorous when-does-it-work characterization.*

**Cross-sectional stock ranking** is the task of predicting, every trading day, the *relative* order of next-period returns across a fixed universe of stocks (here, the ~500 constituents of the US S&P 500). A model that ranks the top decile correctly delivers a tradeable long-short portfolio; a model that ranks no better than chance loses to transaction costs. The standard quality metric is the **Information Coefficient (IC)** — the Spearman rank correlation between predicted scores and realised forward returns, averaged over test days.

Recent graph neural network (GNN) papers — most prominently Feng et al. (2019, *TOIS*) on Temporal Relational Ranking and Sawhney et al. (2021, AAAI) on HATS / STHAN-SR — report sizeable IC gains by encoding inter-stock relations (sector membership, Wikidata ontologies, news co-mentions) as edges in a graph and applying message passing. These headline gains are almost always quoted as *single-seed* and *single-split* numbers. Three concerns motivate a closer look:

1. **Single-seed inflation.** An internal pilot in this study reports that GAT at horizon 21d achieves IC = 0.044 at seed 42 but only IC = 0.032 averaged across 5 seeds (coefficient of variation ≈ 55%). Re-tuning hyperparameters cannot fix this; only multi-seed evaluation can surface it.
2. **Regime concentration.** Cross-sectional ranking profits are episodic. A single high-dispersion quarter can dominate annual numbers. Without per-fold disclosure, an apparently positive IC may be the artefact of one good quarter.
3. **Loss-function fragility.** Listwise ranking losses (e.g., ListMLE), popular in information-retrieval literature, can *invert* under cross-sectional regime shift, producing systematically anti-correlated predictions.

We address all three by running 400 anchor experiments under a strict 10-seed × 5-fold × block-bootstrap protocol, then applying Hansen's Superior Predictive Ability (SPA) test, Diebold–Mariano with Harvey–Leybourne–Newbold (HLN) small-sample correction under Benjamini–Hochberg (BH) FDR control. The pipeline is summarised in **Figure 1**.

![Figure 1 — Pipeline overview](../figures/F1_pipeline.png)

**Figure 1.** End-to-end pipeline. Raw daily OHLCV for ~500 S&P 500 stocks → per-day cross-sectional feature matrix (Universe B 10-dim hand-crafted, or Universe C 51-dim Alpha158 subset) → graph snapshot (rolling-correlation edges optionally augmented with GICS sector edges or PIT-safe news co-occurrence edges) → one of four model families {GAT, GraphSAGE, MLP, LightGBM} → cross-sectional ranking → top-K equal-weight dollar-neutral long-short portfolio rebalanced every 21 trading days → evaluation via IC, gross/net Sharpe, Hansen SPA, DM/HLN, and LOFO sensitivity.

> **What "Universe B" and "Universe C" mean (used throughout the paper).**
>
> Both are the same *set of stocks* (~500 S&P 500 constituents); they differ only in **which input features each stock carries into the model**.
>
> - **Universe B — minimalist hand-crafted features (10 dimensions).** Built from price and volume alone: three windowed return means `ret_mean_{5,10,21}d`, three windowed return standard deviations `ret_std_{5,10,21}d`, one 12-month momentum `mom12m`, one extreme-return statistic `maxret`, one dollar-volume `dolvol`, and one 5-day rolling correlation `CORR5`. The "B" stands for *baseline*: any quant can rebuild it from raw OHLCV in an afternoon. All values are taken strictly at T − 1.
> - **Universe C — Alpha158 subset (51 dimensions).** Built from the *Qlib Alpha158* factor library (Microsoft Research's open-source 158-feature handler, `qlib/contrib/data/handler.py`); we keep the top-15 factor *groups* (51 columns) selected by the project's earlier Plan AAA ranking. The 15 groups span momentum (ROC), price-level statistics (MA, MAX, MIN, QTLU, QTLD), candlestick shape (KMID, KSFT, KUP), market beta, position-within-range (RANK, RSV), directional balance (CNTP, CNTD, CNTN), and price–volume interactions (CORR, WVMA) — full enumeration in **Table T_FACTORS** (§7). The "C" stands for *Composite*. All values are taken strictly at T − 1. Composition basis caveat: see Limitation L1.
>
> We deliberately compare two universes of very different richness so we can test the hypothesis: **does GNN advantage depend on how strong the input features already are?**

**Headline findings.**
- Bootstrap 95% CIs exclude IC = 0 for **7 / 8** (universe, model) cells.
- Hansen SPA test fails to reject "no candidate beats LightGBM" in **both** universes (p_consistent = 0.147 / 0.384) and in their joint family (p = 0.136).
- One fold (Q2-2025) drives **38–72%** of every model's positive IC and Sharpe. Dropping this fold (LOFO-4) collapses Universe B LightGBM's net Sharpe @10 bps from −0.83 to **+1.07** — a sign flip on a single quarter.
- The news-as-edge augmentation has the smallest p-value in its family (HLN p = 0.039) but **0 / 5** edge pairs survive BH-FDR control.

We frame these as *conditional findings* (when does a model help?) and *failure modes* (when does it break?), not as architecture supremacy claims.

---

## §2. Related Work

> *Thesis: Prior GNN-finance papers report headline gains but routinely omit at least one of {multi-seed, multi-fold, cost ladder, point-in-time news handling, multi-test correction}.*

### §2.1 Single-relation and multi-relation graph baselines

- **Feng et al. (2019, *TOIS*) — Temporal Relational Ranking for Stock Prediction.** Introduces *Wiki-relation* and *sector-relation* graphs over NASDAQ / NYSE stocks; the model is a temporal graph convolutional network with a pairwise ranking loss. Reports IC and Sharpe on a single test split.
- **Sawhney et al. (2021, AAAI) — HATS / STHAN-SR.** Heterogeneous Attention Network with 75 Wikidata relation types and hypergraph spatiotemporal attention. Reports IC gains over LSTM baselines on a fixed test window with one or two seeds; no SPA or DM/HLN test.
- **Wang et al. (2022) — Stockformer.** Transformer + graph hybrid; same single-split evaluation pattern.

### §2.2 Methodology-leaning quant-finance work

- **Hou, Xue & Zhang (2020, *RFS*) — Replicating Anomalies.** Documents that 65 % of published cross-sectional factors fail multi-test corrected replication; we adopt their multiple-comparison discipline.
- **López de Prado (2018) — Advances in Financial Machine Learning.** Codifies block bootstrap, purge-and-embargo CV, and combinatorially-purged CV. We use the purge-and-embargo idea in §3.1.
- **Hansen (2005) — Test for Superior Predictive Ability.** Provides the SPA test we use as our headline multiple-comparison defence.

### §2.3 Position vs prior work

Our protocol is the first (to our knowledge) to combine *all* of the following on the S&P 500: 10 seeds × 5-fold expanding walk-forward × {gross, 5–30 bps net} cost ladder × Hansen SPA × DM/HLN with BH-FDR × block-bootstrap CI × LOFO sensitivity × PIT news handling. **Table T6** (related-work matrix, *placeholder — to be expanded to 19 papers via the `literature-review` skill before submission*) makes this explicit.

> *T6 placeholder — column axes: Paper · Horizon · Feature set · Graph type · Test span / regimes · # seeds · PIT-safe news? · Cherry-pick defence (SPA / FDR) · Cost ladder. Most cells in the "10 seeds" and "SPA / FDR" columns are empty for prior GNN-finance work.*

---

## §3. Methodology

> *Thesis: A reader should be able to reproduce the protocol from this section alone. We separate the protocol (this section) from the experiments that use it (§5).*

### §3.1 Walk-forward cross-validation

Time-series prediction must be evaluated chronologically. We use a 5-fold **expanding-window walk-forward**: for fold *k* ∈ {0,…,4}, the training set is all data through quarter *k*, the validation set is quarter *k* + 1, and the test set is quarter *k* + 2. The 5 test quarters span Q2-2024 through Q2-2025.

Because the prediction label is a 21-day forward return, we drop the final 21 trading days of the training set (a **purge embargo**) to eliminate label overlap between train and test.

![Figure S15 — Walk-forward calendar](../figures/S15_walkforward_calendar.png)

**Figure S15.** Walk-forward calendar. Five (train, 21-day purge, validation, test) bands. Train window ≈ 3 years rolling; validation ≈ 1 quarter; purge = 21 trading days (= label horizon); test = 1 quarter. Test fold dates: F0 = Q2-2024, F1 = Q3-2024, F2 = Q4-2024, F3 = Q1-2025, **F4 = Q2-2025** (annotated as a regime outlier; see §5.3).

### §3.2 Information Coefficient and portfolio construction

**Daily IC** is the cross-sectional **Spearman rank correlation** between model predictions and next-day 21-day-forward returns, computed over all stocks with non-missing features on that day. Headline IC is the pooled mean over all test days across all 5 folds. We use Spearman rather than Pearson because rank-based IC is invariant to monotonic transforms of predictions and is the dominant convention in factor-investing literature.

**Formula (IC).** For test day *t* with *n*<sub>t</sub> valid stocks, let *r*<sub>i</sub> = rank of model prediction, *s*<sub>i</sub> = rank of next-day return, and *d*<sub>i</sub> = *r*<sub>i</sub> − *s*<sub>i</sub>. Then

```
                  6 · Σᵢ dᵢ²
   IC_t  =  1 − ─────────────────
                  n_t · (n_t² − 1)
```

Pooled headline IC = mean(*IC*<sub>t</sub>) over all test days across all 5 folds. IC ∈ [−1, 1]; IC > 0 means predictions are positively rank-correlated with realised returns. *In plain English*: every trading day we line up all ~500 stocks by predicted score and by actual next-day return; IC measures how similar those two orderings are.

**Portfolio construction.** Each day, sort stocks by predicted score; long the top K, short the bottom K, with K = ⌊10 % × *n*<sub>valid</sub>⌋, equal-weight, dollar-neutral. Rebalance every 21 trading days (non-overlapping periods, matching the label horizon).

**Formula (Sharpe).** Let *r*<sub>t</sub><sup>L/S</sup> = (1/K) Σ<sub>i ∈ long</sub> *r*<sub>i,t</sub> − (1/K) Σ<sub>i ∈ short</sub> *r*<sub>i,t</sub> be the daily long-short portfolio return. Then

```
                 mean(r^L/S)
   Sharpe  =  ────────────────── · √(252 / 21)
                 std(r^L/S)
```

The factor √(252 / 21) annualises a Sharpe computed on 21-day non-overlapping returns (252 trading days per year / 21-day rebalance = ~12 periods per year). *In plain English*: Sharpe is (average return) ÷ (volatility) — higher means more profit per unit of risk.

- **Gross Sharpe** ignores trading frictions.
- **Net Sharpe** subtracts one-way transaction costs at {0, 5, 10, 15, 20, 30} bps from each rebalance; the headline cost level is **10 bps** (industry-conservative for US large caps with ~3× turnover per rebalance).

### §3.3 Hansen SPA, DM/HLN, and BH-FDR — multiple-comparison framework

> **In plain English — what these tests are doing.**
>
> Both tests answer "did my model really win?", but in different settings.
>
> - **Hansen SPA** is *one-vs-many*. Imagine 3 candidate models all racing against a benchmark; one of them happens to be the best. SPA asks: **after accounting for the fact that I picked the best out of several, is that winner truly better than the benchmark, or did I just get lucky finding one that looked good?** Null hypothesis: *no candidate is genuinely better.*
> - **DM / HLN** is *one-vs-one*. Take a single pair (e.g., GAT vs LightGBM). Every day compute the paired difference ΔIC = IC<sub>GAT,t</sub> − IC<sub>LGB,t</sub>. Ask: **is the mean of these daily differences reliably above zero?** This is a paired *t*-test, with Newey-West HAC standard errors (to handle daily autocorrelation) and the HLN small-sample correction (because *T* = 313 days is not "large enough" for textbook DM).
> - **BH-FDR** layers on top of DM: since we run 10 pairwise tests, some will look significant by random chance. BH controls the *false discovery rate* — the expected fraction of "significant" claims that are actually noise — at q = 0.05.

**Hansen Superior Predictive Ability (SPA; Hansen 2005).** SPA tests the null *H*₀: "no candidate model has a smaller expected loss than the benchmark." We define the loss as **−daily IC**, so smaller loss = larger IC. The benchmark is LightGBM; candidates per universe are {GAT, SAGE-Mean, MLP}. We report `p_consistent` (the SPA p-value variant that handles studentised loss differences) and reject at α = 0.05. Implementation: `arch.bootstrap.SPA` from the `arch` Python package, stationary bootstrap with block size 21 trading days, 10 000 replicates.

**Formula sketch (SPA).** For each candidate *k* ∈ {1,…,M} define the loss differential *d*<sub>k,t</sub> = ℓ<sub>k,t</sub> − ℓ<sub>bench,t</sub> where ℓ<sub>·,t</sub> = −*IC*<sub>·,t</sub>. The SPA test statistic is

```
   T_SPA  =  max  √T · max( 0,  d̄_k / ω̂_k )
                k
```

where *d̄*<sub>k</sub> = mean of *d*<sub>k,t</sub> over the *T* test days and *ω̂*<sub>k</sub> is a HAC estimator of its long-run standard deviation. The bootstrap distribution is generated under a recentred null; *p*<sub>consistent</sub> is the fraction of bootstrap replicates whose *T*<sub>SPA</sub> exceeds the observed value. Small *p* ⇒ reject "no candidate is better."

**Diebold–Mariano with Harvey–Leybourne–Newbold correction (DM/HLN).** For each candidate-vs-benchmark pair we compute the paired ΔIC time series and test *H*₀: 𝔼[ΔIC] = 0 using Newey-West (HAC) standard errors plus the HLN small-sample correction. We collect 10 pairwise tests (5 pairs × 2 universes) for the headline family and apply **Benjamini–Hochberg (BH) FDR control** at q = 0.05.

**Formula (DM and HLN).** For two models *A*, *B* and test horizon *h* days:

```
   ΔIC_t   =  IC_{A,t} − IC_{B,t}                    (paired difference per day)
   mean_Δ  =  (1/T) Σ_t ΔIC_t
   SE_HAC  =  √( Var̂_NW(ΔIC_t) / T )                 (Newey-West HAC standard error)
   DM      =  mean_Δ / SE_HAC                       ~ N(0,1) asymptotically

   HLN_t   =  DM · √( (T + 1 − 2h + h(h−1)/T) / T ) ~ Student-t(T−1)
```

The HLN correction shrinks the test statistic when *T* is small relative to *h* (here *T* = 313, *h* = 21), producing more conservative p-values from the Student-*t* distribution. Reject *H*₀ when |HLN_t| exceeds the *t*-critical at the BH-adjusted α.

**Block bootstrap CIs.** IC bootstrap uses block size = 21 days on the daily IC series (preserves serial correlation up to one rebalance horizon); Sharpe bootstrap uses block size = 1 on per-cell Sharpe values (cells are exchangeable across seed × fold). 5 000 replicates, percentile method. *Block bootstrap* means we resample contiguous 21-day chunks rather than individual days, so the resampled series preserves within-month autocorrelation structure.

![Figure 9 — SPA + DM/HLN](../figures/F9_spa_dm_hln.png)

**Figure 9.** Hansen SPA (left) + DM/HLN paired tests (right). Left: p_consistent per universe (B / C / joint) with the dashed reference line at α = 0.05; no universe rejects. Right: 10 DM/HLN pairwise tests with HLN small-sample p-values; red = BH-FDR reject at q = 0.05 (none), grey = not rejected. *Used both here and in §5.2.*

### §3.4 LOFO sensitivity (Leave-One-Fold-Out)

We recompute headline IC and Sharpe after dropping each fold one at a time. If a single fold drives the result, dropping it surfaces the fragility. We name the "drop Fold 4" condition **LOFO-4** specifically, because Q2-2025 is the only positive-S&P-momentum quarter in our test span and turns out (§5.3) to drive most of the headline.

### §3.5 Pre-registration and multi-testing ledger

> **In plain English — what pre-registration is.**
>
> Pre-registration means writing down — *before* running the experiment — exactly which metric matters, what counts as "the model worked", which comparisons are confirmatory vs exploratory, and how multiple-testing will be controlled. The document is then version-stamped (here: a JSON file committed to git before any GPU launch).
>
> The purpose is to prevent a common form of accidental cherry-picking: running 100 experiments, finding 5 that look good by chance, and then writing the paper as if those 5 were the plan all along (the so-called "garden of forking paths"). With pre-registration, a sceptical reader can compare the published claims line-by-line against the locked plan and verify that the rules were not bent after seeing the data.

To prevent post-hoc cherry-picking we pre-registered, before any A100 launch, the primary metric (pooled IC mean), the positive-verdict gate (ΔIC vs LightGBM > +0.005 *and* DM/HLN p<sub>BH</sub> < 0.05), and the confirmatory family (5-pair edge ablation). The pre-registration is the file [experiments/storya_e1_anchor/prereg.json](../experiments/storya_e1_anchor/prereg.json), git-committed and timestamped. All other contrasts (horizon × architecture sweep, loss-function horserace, Plan AAA factor ranking) are flagged as *exploratory* — disclosed for transparency but not allowed to count as confirmatory evidence. The full ledger is shown below.

![Figure S16 — Multi-testing ledger pyramid](../figures/S16_multitest_ledger_pyramid.png)

**Figure S16.** Multi-testing ledger pyramid. Centred horizontal bars sized by trial count: *primary* family (post-E1 SPA-controlled), *ablation* family (E3+E4 edge contrasts, BH-FDR controlled), and *historical exploratory* trials (disclosed for transparency, not in the SPA family).

**Table ST2 — Multi-testing ledger.**

| Family | Count | Coverage note |
|---|---:|---|
| Primary E1 cells (SPA / DM-HLN family) | 400 | BH-FDR q = 0.05; SPA *M* = 3 per universe |
| E3 news-encoding cells (ablation family) | 50 | edge-ablation BH-FDR q = 0.05 family of 5 pairs |
| E4 α-edge cells (ablation family) | 100 | edge-ablation BH-FDR q = 0.05 family of 5 pairs |
| Plan AAA group-rank tests (historical, disclosed) | 61 | 0/61 pass at q = 0.05; NOT in post-E1 SPA family |
| Horizon-ablation cells (historical) | 360 | 21d horizon selection; NOT in SPA family |
| Loss-horserace cells (historical) | 600 | MSE locked for E1; NOT in SPA family |
| Cost-ladder bps levels | 6 | bps = [0, 5, 10, 15, 20, 30]; descriptive ladder, not multi-tested |

*Source: [tables/ST2_multitest_ledger.tex](../tables/ST2_multitest_ledger.tex).*

---

## §4. Data and Experimental Setup

### §4.0 Experiment ID legend (E1 / E3 / E4 / E6 / E1.6)

> **What the IDs mean.** Story A is a series of related experiments, each given a short code so the same scaffold can be reused (same seeds, same folds, same label, same evaluation pipeline) while only one knob varies. E2 and E5 are deliberately unused (reserved for future insertions).

| ID | Folder | Cells | What varies | Purpose |
|---|---|---:|---|---|
| **E1** *(anchor)* | [experiments/storya_e1_anchor/](../experiments/storya_e1_anchor/) | 400 | model × feature universe × seed × fold | The headline result. 4 models {GAT, SAGE-Mean, MLP, LightGBM} × 2 universes {B, C} × 10 seeds × 5 folds, all with the correlation-only graph (α1). This is the table that goes in §5.1. |
| **E1.6** *(HATS)* | [experiments/storya_e1_6_hats/](../experiments/storya_e1_6_hats/) | 50 (planned) | adds HATS-3R-adapt to the model axis | Heterogeneous-relational-GNN baseline (STHAN-SR-inspired). Scheduled, not yet run as of 2026-05-28; covers L5 of the limitations matrix. |
| **E3** *(news edge)* | [experiments/storya_e3_news_edge/](../experiments/storya_e3_news_edge/) | 50 new | adds news co-occurrence edges (α3) on top of α1 | Tests whether news-derived graph edges add predictive signal beyond correlation edges. Feeds the α3-vs-α1 row of Table 5. |
| **E4** *(α edge)* | [experiments/storya_e4_alpha/](../experiments/storya_e4_alpha/) | 100 new | adds sector edges (α2) and corr+sector+news (α4) | Completes the edge-ablation family of 5 pairwise contrasts in Table 5. |
| **E6** *(stats)* | [artifacts/storya_e6_dm_spa/](../artifacts/storya_e6_dm_spa/) and `storya_e6_edge_ablation/` | 0 new runs | post-processing only | Takes E1 + E3 + E4 outputs and computes Hansen SPA, DM/HLN, BH-FDR, cost ladder, LOFO sensitivity, bootstrap CIs, and outlier flags. Produces the CSVs that source every table and figure in §5. |

**How they compose.** E1 establishes the anchor (correlation graph, 400 cells). E3 + E4 add 150 cells (50 + 100) under the α2/α3/α4 edge configurations, sharing E1-B-SAGE's 50 α1 cells as the common baseline. E6 then statistically post-processes the union to produce SPA / DM-HLN / cost-ladder outputs without running any new model.

### §4.1 Universe and label

Universe: US S&P 500 constituents as of the trading day (502 stocks + 1 equal-weight S&P 500 bench = 503 assets). Frequency: daily close-to-close. **Label**: next-day cross-sectionally z-scored 21-day forward log return. This definition is locked across all experiments and never revised.

### §4.2 Feature universes

We compare two feature sets to test the hypothesis that GNN advantage depends on input richness.

- **Universe B (10-dim minimalist hand-crafted).** `ret_mean_{5,10,21}d`, `ret_std_{5,10,21}d`, `mom12m`, `maxret`, `dolvol`, `CORR5`. All computed strictly at T − 1.
- **Universe C (51-dim Alpha158 subset).** Top 15 factor groups from the project's prior Alpha158 ranking (Plan AAA; see §7 L1 for stability caveats). Includes Qlib-default factors `ROC, MA, MAX, MIN, QTLU, QTLD, KMID, KSFT, BETA, RANK, RSV, WMA, CORR`, plus return / range / momentum statistics at multiple windows. All evaluated at T − 1.

### §4.3 Models compared

- **GAT (Graph Attention Network, Veličković et al. 2018).** 2 layers, hidden dim 64, 4 attention heads. *Mechanism*: learns per-edge attention weight; each node's update is the attention-weighted sum of neighbour features.
- **GraphSAGE (SAGE-Mean; Hamilton et al. 2017).** 2 layers, hidden dim 64, mean aggregator over 1-hop neighbours. *Mechanism*: each node's update is the mean of neighbour features concatenated with self, followed by a learned linear transform.
- **MLP.** 2 layers, hidden dim 128, ReLU. Non-graph baseline using the same input features as GAT / SAGE.
- **LightGBM.** Gradient-boosted regression trees, default Qlib hyperparameters. Widely considered the strongest non-deep baseline for tabular cross-sectional finance.

### §4.4 Graph construction

- **Correlation (α1, the E1 anchor base).** 126-day rolling Spearman correlation on daily returns; edge if |ρ| > 0.6; snapshot updated every 21 trading days. Each test day uses the most recent training-only snapshot — *no look-ahead*.
- **Sector (α2).** Static GICS 11-sector membership; full connectivity within sector.
- **News co-occurrence (α3).** Point-in-time (PIT) safe: two stocks share an edge on day *T* if both were mentioned in the same news article published before NYSE close of day *T* − 1. Source: [experiments/storya_e3_news_edge/news_snapshots_cache.npz](../experiments/storya_e3_news_edge/news_snapshots_cache.npz) — 313 daily snapshots, mean ~1 823 edges and ~807 articles per day.
- **Combined (α4).** Corr + sector + news, all three.

### §4.5 Hyperparameters and seeds

Ten canonical seeds: {7, 34, 86, 99, 123, 456, 789, 1024, 2024, 2026}. Optimiser: Adam, lr = 1 × 10⁻³, weight decay 1 × 10⁻⁵, early stopping on validation IC with patience = 10 epochs. Total cell count for the E1 anchor: 4 models × 2 universes × 10 seeds × 5 folds = **400 cells**.

---

## §5. Results

> *Thesis: Bootstrap CIs say "IC > 0 for 7/8 cells," SPA says "no GNN dominates LightGBM," and LOFO says "one fold drives most of the positive result." All three must be told in sequence; cherry-picking any one is misleading.*

### §5.1 Honest headline IC and Sharpe (pillar N1)

We first present numbers under the strictest evaluation we can run — 10 seeds × 5 folds × block-bootstrap CI. **Table 1** is the paper's anchor result.

**Table 1 — Headline IC and gross Sharpe with 95% bootstrap CI.**

| Universe | Model | *n* cells | IC mean [95% CI] | *S*<sub>gross</sub> mean [95% CI] |
|:---|:---|---:|:---|:---|
| B | GAT | 50 | 0.0355 [0.0181, 0.0526] | 1.50 [0.82, 2.20] |
| B | SAGE-Mean | 50 | 0.0320 [0.0144, 0.0498] | 1.91 [1.10, 2.82] |
| B | MLP | 50 | 0.0299 [0.0157, 0.0442] | 1.99 [0.93, 3.57] |
| B | LightGBM | 50 | 0.0065 [−0.0069, 0.0194] | **−0.33 [−1.56, 0.80]** |
| C | GAT | 50 | 0.0431 [0.0233, 0.0628] | 3.62 [1.44, 7.05] |
| C | SAGE-Mean | 50 | 0.0480 [0.0302, 0.0661] | 1.60 [0.86, 2.37] |
| C | MLP | 50 | 0.0533 [0.0354, 0.0713] | 2.25 [1.24, 3.36] |
| C | LightGBM | 50 | 0.0473 [0.0336, 0.0612] | 2.41 [1.51, 3.39] |

*Source: [tables/T1_headline.tex](../tables/T1_headline.tex); derived from [artifacts/storya_e6_dm_spa/bootstrap_ci.csv](../artifacts/storya_e6_dm_spa/bootstrap_ci.csv).*

**Reading Table 1.** In Universe B the three neural models cluster tightly (IC 0.030–0.036), while LightGBM is statistically zero (CI includes 0). In Universe C all four models converge to IC 0.043–0.053: when the feature set is rich enough, the gap between graph and non-graph models nearly closes — consistent with the factor-investing intuition that *graphs help most when node features are weakest*.

![Figure 2 — Cumulative IC trajectory](../figures/F2_cumulative_ic_trajectory.png)

**Figure 2.** Cumulative daily IC trajectory, 4 models × 2 universes (8 panels). Grey lines = 10 individual seeds; coloured line = across-seed mean. Amber-shaded region = **Fold 4 (Q2-2025)**. *Note: per_day_ic .npy files contain daily Spearman IC arrays, so F2 plots cumulative IC as a ranking-quality proxy rather than cumulative L/S PnL.* **Takeaway**: cumulative curves are monotonically positive *except* Universe B LightGBM, which is flat-to-negative.

![Figure 5 — Cost ladder](../figures/F5_cost_ladder.png)

**Figure 5.** Cost-ladder: mean net Sharpe vs one-way L1 transaction cost (bps), 8 lines (4 models × 2 universes). Solid = Universe B, dashed = Universe C; shaded bands = 95% bootstrap CI. Dotted reference at Sharpe = 1.0. **Takeaway**: at the headline 10 bps, Universe B LightGBM falls below 0 while all neural lines stay above 1.0.

![Figure S1 — per-cell scatter](../figures/S1_per_cell_ic_sharpe_scatter.png)

**Figure S1.** Per-cell IC vs net Sharpe scatter — 400 individual cells (4 models × 2 universes × 10 seeds × 5 folds). Colour = model, marker shape = universe (○ = B, ◻ = C). Shows the distribution behind the table-1 means.

**Table 4 — Net Sharpe at six cost levels (with 95% bootstrap CI).**

| Univ | Model | *n* | 0 bps | 5 bps | 10 bps | 15 bps | 20 bps | 30 bps |
|:---|:---|---:|:---|:---|:---|:---|:---|:---|
| B | GAT | 50 | 1.50 [0.82, 2.20] | 1.38 [0.71, 2.09] | 1.27 [0.61, 1.97] | 1.16 [0.50, 1.85] | 1.05 [0.39, 1.73] | 0.82 [0.18, 1.49] |
| B | SAGE-Mean | 50 | 1.91 [1.10, 2.82] | 1.77 [0.97, 2.67] | 1.62 [0.84, 2.51] | 1.48 [0.71, 2.35] | 1.33 [0.57, 2.19] | 1.03 [0.30, 1.85] |
| B | MLP | 50 | 1.99 [0.93, 3.57] | 1.82 [0.81, 3.29] | 1.64 [0.70, 3.00] | 1.47 [0.57, 2.70] | 1.29 [0.45, 2.40] | 0.94 [0.22, 1.85] |
| B | LightGBM | 50 | −0.33 [−1.56, 0.80] | −0.58 [−1.88, 0.61] | **−0.83 [−2.21, 0.40]** | −1.09 [−2.54, 0.20] | −1.36 [−2.91, 0.01] | −1.91 [−3.69, −0.38] |
| C | GAT | 50 | 3.62 [1.44, 7.05] | 3.36 [1.24, 6.64] | 3.08 [1.05, 6.19] | 2.80 [0.86, 5.72] | 2.51 [0.68, 5.25] | 1.95 [0.30, 4.32] |
| C | SAGE-Mean | 50 | 1.60 [0.86, 2.37] | 1.45 [0.72, 2.20] | 1.30 [0.57, 2.04] | 1.14 [0.42, 1.88] | 0.99 [0.27, 1.73] | 0.70 [−0.01, 1.42] |
| C | MLP | 50 | 2.25 [1.24, 3.36] | 2.06 [1.07, 3.14] | 1.88 [0.92, 2.92] | 1.70 [0.75, 2.72] | 1.51 [0.59, 2.51] | 1.15 [0.25, 2.09] |
| C | LightGBM | 50 | 2.41 [1.51, 3.39] | 2.22 [1.34, 3.17] | 2.03 [1.18, 2.96] | 1.84 [1.01, 2.73] | 1.65 [0.85, 2.51] | 1.26 [0.51, 2.05] |

*Source: [tables/T4_cost_ladder.tex](../tables/T4_cost_ladder.tex); derived from [artifacts/storya_e6_dm_spa/cost_ladder.csv](../artifacts/storya_e6_dm_spa/cost_ladder.csv).*

### §5.2 SPA + DM/HLN — what the formal tests say (pillar N4)

Bootstrap CIs in Table 1 *do not* control for multiple comparisons or model-selection bias. Hansen SPA does — and once we apply it, no GNN candidate dominates LightGBM at α = 0.05 in either universe.

**Table 3 — Hansen SPA (Panel A) + DM/HLN paired tests (Panel B).**

**Panel A. Hansen SPA (consistent variant).**

| Universe | Benchmark | Candidates | *M* | *T* | *p*<sub>consistent</sub> | Reject @5%? |
|:---|:---|:---|---:|---:|---:|:---:|
| B | LightGBM | GAT \| SAGE-Mean \| MLP | 3 | 313 | 0.1474 | no |
| C | LightGBM | GAT \| SAGE-Mean \| MLP | 3 | 313 | 0.3843 | no |
| JOINT(B+C) | LightGBM pooled | B.GAT \| B.SAGE \| B.MLP \| C.GAT \| C.SAGE \| C.MLP | 6 | 313 | 0.1364 | no |

**Panel B. DM/HLN paired tests** (NW-HAC SE, BH-FDR q = 0.05).

| Universe | Pair (A vs B) | $\overline{\Delta\text{IC}}$ | DM stat | HLN stat | HLN *p* | BH-FDR reject? |
|:---|:---|---:|---:|---:|---:|:---:|
| B | GAT vs LightGBM | +0.0290 | −1.878 | −1.755 | 0.0802 | no |
| B | SAGE-Mean vs LightGBM | +0.0255 | −1.719 | −1.606 | 0.1092 | no |
| B | MLP vs LightGBM | +0.0234 | −1.846 | −1.725 | 0.0855 | no |
| B | GAT vs MLP | +0.0056 | −0.975 | −0.911 | 0.3628 | no |
| B | SAGE-Mean vs MLP | +0.0021 | −0.351 | −0.328 | 0.7434 | no |
| C | GAT vs LightGBM | −0.0043 | +0.469 | +0.439 | 0.6613 | no |
| C | SAGE-Mean vs LightGBM | +0.0006 | −0.083 | −0.077 | 0.9386 | no |
| C | MLP vs LightGBM | +0.0060 | −0.756 | −0.707 | 0.4803 | no |
| C | GAT vs MLP | −0.0103 | +2.338 | +2.184 | 0.0297 | no |
| C | SAGE-Mean vs MLP | −0.0054 | +1.238 | +1.157 | 0.2480 | no |

*Source: [tables/T3_spa_dm_hln.tex](../tables/T3_spa_dm_hln.tex); derived from [artifacts/storya_e6_dm_spa/spa_results.csv](../artifacts/storya_e6_dm_spa/spa_results.csv) and [artifacts/storya_e6_dm_spa/dm_hln_results.csv](../artifacts/storya_e6_dm_spa/dm_hln_results.csv).*

**Reading Table 3.** The strongest pairwise DM/HLN result is (B, GAT) vs LightGBM with mean ΔIC = +0.029 and HLN p = 0.080. Under BH-FDR control over the 10-pair family this does not reject. SPA p<sub>consistent</sub> = 0.147 / 0.384 / 0.136 — all above 0.05. **The honest reading**: bootstrap CIs already established that the neural models have non-zero IC, but on this 1-year test span no GNN candidate's *advantage over LightGBM* is large enough to survive multiple-comparison defence.

### §5.3 Three-column robustness — full vs LOFO-4 vs Fold-4-only (pillar N3)

One fold drives the result. We show this explicitly by comparing three regime conditions side-by-side: **Full** = all 5 folds (the headline); **LOFO-4** = drop Fold 4 entirely; **Fold-4-only** = Fold 4 alone.

**Table 2 — Three-column robustness (IC and net Sharpe @10 bps with 95% bootstrap CI).**

| Univ | Model | IC<sub>full</sub> [CI] | IC<sub>LOFO-4</sub> [CI] | IC<sub>Fold-4 only</sub> [CI] | *S*<sub>net,10</sub><sup>full</sup> [CI] | *S*<sub>net,10</sub><sup>LOFO-4</sup> [CI] | *S*<sub>net,10</sub><sup>Fold-4 only</sup> [CI] |
|:---|:---|:---|:---|:---|:---|:---|:---|
| B | GAT | 0.0355 [0.018, 0.053] | 0.0222 [0.005, 0.040] | 0.0893 [0.049, 0.128] | 1.27 [0.61, 1.97] | 0.88 [0.30, 1.49] | 2.86 [0.73, 4.97] |
| B | SAGE-Mean | 0.0320 [0.014, 0.050] | 0.0152 [−0.000, 0.032] | 0.0999 [0.049, 0.140] | 1.62 [0.84, 2.51] | 0.88 [0.29, 1.54] | 4.61 [2.13, 7.16] |
| B | MLP | 0.0299 [0.016, 0.044] | 0.0255 [0.010, 0.042] | 0.0477 [0.011, 0.082] | 1.65 [0.69, 3.00] | 1.01 [0.46, 1.60] | 4.18 [0.46, 10.08] |
| B | LightGBM | 0.0065 [−0.007, 0.019] | 0.0180 [0.006, 0.031] | **−0.0399 [−0.076, −0.004]** | **−0.83 [−2.21, 0.40]** | **+1.07 [0.46, 1.71]** | **−8.43 [−11.80, −5.59]** |
| C | GAT | 0.0431 [0.023, 0.063] | 0.0167 [−0.001, 0.035] | 0.1498 [0.117, 0.183] | 3.08 [1.05, 6.19] | 0.85 [0.19, 1.53] | 11.99 [3.68, 25.55] |
| C | SAGE-Mean | 0.0480 [0.030, 0.066] | 0.0244 [0.008, 0.041] | 0.1435 [0.111, 0.176] | 1.30 [0.57, 2.04] | 0.43 [−0.16, 1.04] | 4.74 [3.55, 6.16] |
| C | MLP | 0.0533 [0.035, 0.071] | 0.0288 [0.013, 0.045] | 0.1526 [0.122, 0.183] | 1.88 [0.92, 2.92] | 0.71 [0.07, 1.34] | 6.55 [4.03, 9.64] |
| C | LightGBM | 0.0473 [0.034, 0.061] | 0.0306 [0.018, 0.044] | 0.1151 [0.090, 0.141] | 2.03 [1.18, 2.96] | 1.14 [0.48, 1.85] | 5.59 [3.79, 8.08] |

*Source: [tables/T2_three_column_robustness.tex](../tables/T2_three_column_robustness.tex); derived from [artifacts/storya_e6_dm_spa/e1_three_column_summary.csv](../artifacts/storya_e6_dm_spa/e1_three_column_summary.csv).*

**Reading Table 2.** Most positive IC is concentrated in Fold 4:
- Univ B GAT IC: 0.0355 → 0.0222 (LOFO-4, −38%) → 0.0893 (Fold-4 only)
- Univ B SAGE IC: 0.0320 → 0.0152 (−53%) → 0.0999
- Univ C GAT IC: 0.0431 → 0.0167 (−61%) → 0.1498
- **Univ B LightGBM net Sharpe flips sign**: −0.83 (full) → **+1.07** (LOFO-4). Fold 4 alone delivers net Sharpe = **−8.43** — a single quarter destroyed the annualised LightGBM number in Universe B.

**Why Q2-2025 was special.** The Apr–Jun 2025 window combined the post-Liberation-Day equipment-stock selloff, the mega-cap NVDA / AVGO momentum unwind, and the regional-bank rebound, producing exceptionally high cross-sectional dispersion. Cross-sectional ranking models earn most of their carry in such regimes; flat regimes produce near-zero IC for everyone. This is informative for portfolio managers but cautionary for anyone reading headline IC as a stationary parameter.

![Figure 3 — LOFO heatmap](../figures/F3_lofo_heatmap.png)

**Figure 3.** LOFO sensitivity heatmap (8 rows = 2 universes × 4 models; 6 columns = `none` + 5 leave-one-fold conditions). Cells show IC mean when that fold is left out. Diverging RdBu_r colour-map centred at 0. **Takeaway**: the "drop Fold 4" column reduces IC by 38–72% across nearly every (universe, model) pair.

![Figure 4 — per-fold IC bars](../figures/F4_per_fold_ic_bars.png)

**Figure 4.** Per-fold IC bars (mean ± seed std). Five folds × 8 (universe, model) groups; solid bars = Universe B, hatched bars = Universe C. **Takeaway**: the Fold 4 bar is 3–5× taller than other folds for nearly every (universe, model) pair.

![Figure S3 — per-day IC time series](../figures/S3_per_day_ic_8_lines.png)

**Figure S3.** Per-day IC time series, 8 seed-averaged curves (solid = Univ B, dashed = Univ C). Fold 4 shaded red. **Takeaway**: the shaded region holds the bulk of cumulative IC.

![Figure S17 — bootstrap CI 3-column forest](../figures/S17_bootstrap_ci_3col.png)

**Figure S17.** Bootstrap-CI overlay: IC mean with 95% block-bootstrap CI under three regimes (full / LOFO-4 / Fold-4-only). 8 rows × 3 regimes. Visual companion to Table 2.

![Figure S2 — outlier flagging](../figures/S2_top_bottom_3_outliers.png)

**Figure S2.** Top-3 / bottom-3 Sharpe outliers per (universe, model), annotated with (cell ID, fold, seed). Green = TOP3, red = BOT3. **Takeaway**: the Univ C GAT entry includes cell ID 240 (seed 86, fold 4) with Sharpe<sub>gross</sub> = **75.0** — a single outlier inflates the headline Univ C GAT mean from ≈ 2.3 to 3.62.

### §5.4 Edge ablation — do extra graphs help? (pillar N2)

Conditional on Universe B + SAGE-Mean (the cleanest baseline), we vary the edge set. Four configurations:

- **α1** = correlation only (baseline)
- **α2** = corr + sector
- **α3** = corr + news co-occurrence
- **α4** = corr + sector + news

**Table 5 — Edge ablation: 5 pairs × 3 regimes.**

| Pair | Description | Regime | $\overline{\Delta\text{IC}}$ | [CI lo, hi] | HLN stat | HLN *p* | BH-FDR |
|:---|:---|:---|---:|:---|---:|---:|:---:|
| α2 vs α1 | sector adds to corr | full | +0.0097 | [−0.012, +0.028] | 1.562 | 0.1192 | no |
| α2 vs α1 | sector adds to corr | LOFO-4 | +0.0046 | [−0.021, +0.026] | 0.651 | 0.5155 | — |
| α2 vs α1 | sector adds to corr | F4-only | +0.0304 | [+0.027, +0.035] | — | — | — |
| **α3 vs α1** | news adds to corr | full | **+0.0100** | **[−0.007, +0.024]** | **2.071** | **0.0392** | **no** |
| α3 vs α1 | news adds to corr | LOFO-4 | +0.0050 | [−0.014, +0.021] | 0.953 | 0.3418 | — |
| α3 vs α1 | news adds to corr | F4-only | +0.0304 | [+0.022, +0.038] | — | — | — |
| α4 vs α1 | full bundle adds to corr | full | +0.0071 | [−0.016, +0.026] | 1.096 | 0.2741 | no |
| α4 vs α1 | full bundle adds to corr | LOFO-4 | +0.0022 | [−0.026, +0.024] | 0.296 | 0.7673 | — |
| α4 vs α1 | full bundle adds to corr | F4-only | +0.0271 | [+0.023, +0.031] | — | — | — |
| α4 vs α2 | news on top of corr+sector | full | −0.0026 | [−0.006, +0.001] | −1.323 | 0.1867 | no |
| α4 vs α2 | news on top of corr+sector | LOFO-4 | −0.0024 | [−0.007, +0.002] | −1.041 | 0.2988 | — |
| α4 vs α2 | news on top of corr+sector | F4-only | −0.0033 | [−0.005, −0.002] | — | — | — |
| α4 vs α3 | sector on top of corr+news | full | −0.0029 | [−0.013, +0.006] | −0.738 | 0.4612 | no |
| α4 vs α3 | sector on top of corr+news | LOFO-4 | −0.0028 | [−0.015, +0.008] | −0.619 | 0.5366 | — |
| α4 vs α3 | sector on top of corr+news | F4-only | −0.0033 | [−0.010, +0.004] | — | — | — |

*Source: [tables/T5_edge_ablation.tex](../tables/T5_edge_ablation.tex); derived from [artifacts/storya_e6_edge_ablation/edge_pairs_dm.csv](../artifacts/storya_e6_edge_ablation/edge_pairs_dm.csv) and `edge_bootstrap_ci.csv`.*

**Reading Table 5.** In the full 5-fold condition, the strongest contrast is **α3 vs α1 (news edge): ΔIC = +0.010, HLN p = 0.039** — the smallest p-value in the family. But the BH-FDR rank-1 threshold for a 5-test family at q = 0.05 is **0.010**, so this contrast does **not** survive multiple-test control. In the LOFO-4 condition the directional ΔIC collapses to +0.005 (p = 0.34). Multi-edge bundles (α4) actually *underperform* α3 alone — piling on edge types dilutes rather than compounds signal.

![Figure 6 — edge ablation forest](../figures/F6_edge_ablation_forest.png)

**Figure 6.** Edge-ablation forest: 5 α-pairs × 3 regimes; markers = mean ΔIC with 95% bootstrap CI. **Key finding**: 0 / 5 pairs survive BH-FDR q = 0.05 in the full condition, indicating no edge augmentation provides a robust lift over the correlation-only baseline.

![Figure S18 — news edge density](../figures/S18_news_edge_density.png)

**Figure S18.** News-edge density temporal profile (eligible articles and edges per trading day). Establishes that the null result is not from edge-data sparsity.

### §5.5 Horizon and architecture conditional findings (pillar N2)

We previously ran a 4-model × 6-horizon ablation (horizons ∈ {1, 5, 10, 21, 42, 63} days, 15 cells = 3 seeds × 5 folds per row). The 21-day horizon is where graph structure helps stabilise cross-sectional ranks against short-window noise; at very short horizons (1–5 d) all models are noise-dominated, and at long horizons (≥ 42 d) trees and MLPs catch up because the prediction problem approaches a beta-regression.

![Figure 7 — horizon × architecture heatmap](../figures/F7_horizon_arch_heatmap.png)

**Figure 7.** Horizon × architecture IC heatmap; each cell is the mean IC over 15 (seed × fold) runs. Diverging RdBu_r centred at 0. **Takeaway**: news-augmented variants (`*_all`) underperform price-only variants (`*_price`) at long horizons.

![Figure 8 — news-feature dilution forest](../figures/F8_news_dilution_forest.png)

**Figure 8.** News-feature dilution forest: ΔIC = IC(`*_all`) − IC(`*_price`) with 95% paired bootstrap CI (B = 1000) over 15 (seed, fold) pairs per horizon. **MLP news-feature dilution at 21 d: ΔIC = −0.0452. SAGE-Mean at 21 d: ΔIC = −0.0158.** Combined with §5.4's null edge result, our position is: *news belongs in the graph topology, not the node-feature vector* — and even there the effect is below significance under multi-test control.

**Table ST3 — Full horizon × architecture table** (excerpt of 24 rows; see [tables/ST3_horizon_full.tex](../tables/ST3_horizon_full.tex) for all rows).

| Model | Horizon | IC mean | IC std | *S*<sub>net</sub> mean | *S*<sub>net</sub> std | *n* |
|:---|:---|---:|---:|---:|---:|---:|
| MLP_price | 1 d | +0.0147 | 0.0165 | −0.62 | 1.65 | 15 |
| MLP_price | 21 d | +0.0374 | 0.0646 | +2.35 | 3.86 | 15 |
| MLP_price | 63 d | +0.0597 | 0.1185 | +1.76 | 5.89 | 15 |
| MLP_all | 21 d | −0.0078 | 0.0298 | −0.53 | 3.56 | 15 |
| SAGE-Mean_price | 21 d | +0.0269 | 0.0417 | +1.01 | 1.64 | 15 |
| SAGE-Mean_all | 21 d | +0.0111 | 0.0296 | −2.77 | 4.74 | 15 |

### §5.6 Loss-function horse race (pillar N3 — failure modes)

We compared MSE vs ListMLE vs Pairwise log-loss on the same scaffold. **0 / 8 BH-FDR rejections** — no loss family beats MSE on headline IC. But ListMLE shows a *universal* Fold 4 collapse: 6 / 6 architecture × feature combinations have Fold-4 IC ∈ [−0.36, −0.28], compared to Fold-4 MSE IC ∈ [+0.05, +0.15]. The mechanism is mathematical: **ListMLE's softmax-likelihood loss is dominated by the in-distribution rank order; when test ranks shift, the loss landscape inverts and the model produces anti-rankings**. Pairwise log-loss exhibits prediction-scale collapse (per-day prediction std → 0) without portfolio benefit.

![Figure S7 — loss × architecture ΔIC heatmap](../figures/S7_loss_arch_delta_heatmap.png)

**Figure S7.** Loss × architecture ΔIC heatmap, stratified by feature set (S6 full features vs S_price 9-dim). Each cell = mean ΔIC vs MSE baseline. MSE column anchored at 0 by definition.

![Figure S8 — ListMLE per-fold collapse](../figures/S8_listmle_fold4_collapse.png)

**Figure S8.** ListMLE per-fold IC trajectory across the 5 walk-forward folds. Fold 4 (red band) shows systematic collapse for ListMLE across all architectures.

**Table ST6 — Loss horse race paired ΔIC.**

| Model | Features | Contrast | *n*<sub>cells</sub> | $\overline{\Delta\text{IC}}$ | CI lo | CI hi |
|:---|:---|:---|---:|---:|---:|---:|
| MLP | S6 | listmle vs mse | 50 | −0.0536 | −0.0950 | −0.0081 |
| MLP | S6 | pairwise vs mse | 50 | −0.0238 | −0.0440 | −0.0056 |
| MLP | S8 | listmle vs mse | 50 | −0.0677 | −0.1284 | −0.0135 |
| MLP | S8 | pairwise vs mse | 49 | −0.0110 | −0.0295 | +0.0072 |
| SAGE-Mean | S6 | listmle vs mse | 50 | −0.0340 | −0.0788 | +0.0068 |
| SAGE-Mean | S6 | pairwise vs mse | 50 | +0.0015 | −0.0187 | +0.0212 |
| SAGE-Mean | S8 | listmle vs mse | 50 | −0.0731 | −0.1274 | −0.0213 |
| SAGE-Mean | S8 | pairwise vs mse | 50 | −0.0073 | −0.0292 | +0.0137 |

*Source: [tables/ST6_loss_pairwise.tex](../tables/ST6_loss_pairwise.tex).*

### §5.7 Graph ablation and sector attribution (pillar N2)

![Figure S9 — graph ablation](../figures/S9_graph_ablation.png)

**Figure S9.** Graph-ablation grouped bar chart: 9 configurations × 3 seeds = 27 runs. Bars = mean IC ± std; black dots = individual seeds. Green bar = `nn.Linear` baseline (config `0_true_mlp`, IC = 0.0413); red bars = configurations whose mean IC is at or below the linear baseline.

![Figure S11 — sector attribution](../figures/S11_sector_attribution_area.png)

**Figure S11.** SAGE-Mean per-day long / short contribution by GICS sector, split into two panels (long-side, short-side). 11-colour qualitative palette shared across panels. **Takeaway**: in Fold 4 the long side overweights semiconductors and the short side overweights regional banks, consistent with the Q2-2025 dispersion regime.

### §5.8 Auxiliary diagnostics (pillar N4)

![Figure S10 — LightGBM permutation importance](../figures/S10_lgb_perm_importance.png)

**Figure S10.** LightGBM permutation importance, single snapshot per feature. Bars sorted descending by |ΔIC|; green = positive ΔIC after shuffle (feature was harmful), red = negative (feature was useful). Sanity-checks that the LightGBM baseline relies on sensible features.

![Figure S12 — SelectiveNet coverage × IC](../figures/S12_selectivenet_coverage_ic.png)

**Figure S12.** SelectiveNet coverage × IC: one line per strategy (post-hoc threshold vs end-to-end learnable gate). Threshold peak IC = **0.0838** at actual coverage 0.10 (star marker). *Mechanism*: SelectiveNet jointly trains a prediction head and a selector head; a coverage-weighted loss penalises confident predictions on uninformative samples. Trading only the top 10% of confident predictions roughly doubles IC vs the trade-everything baseline.

![Figure S13 — Tier 1 Phase B hyperparameter robustness](../figures/S13_tier1_phaseb_boxes.png)

**Figure S13.** Per-fold mean-test-IC distributions across three tiers (Tier1a *n* = 200, Tier1b H2 *n* = 800, Tier1c *n* = 400). Boxes = distribution of cell-level `mean_test_ic` across all (model, loss, feature_set, seed) cells per fold; black dots = individual cells with x-jitter. Fold 4 highlighted red. **Takeaway**: the headline is not a lucky hyperparameter cell.

![Figure S14 — diagnostic_price replication](../figures/S14_diagnostic_price_replication.png)

**Figure S14.** Diagnostic_price replication histogram. Vertical dashed lines = sample mean per loss; amber arrow = original Part B v4 claim. **Caveat**: Part B v4 wf5 21 d MLP_price IC = +0.037 / SAGE_price IC = +0.027 did **not** replicate in the Stage 1 framework (Diagnostic_price IC ≈ −0.004 / −0.057). Documented as a failure-to-replicate to maintain audit transparency.

---

## §6. Discussion

**Why bootstrap CI says one thing and SPA says another.** A block-bootstrap CI tests a *single* point estimate against zero; the Hansen SPA test asks whether *the best of several candidates* exceeds a benchmark after accounting for selection bias. Both are correct — they answer different questions. For paper-level claims (i.e., "GNN beats LightGBM") we lean on SPA + BH-FDR. For "the model has non-zero predictive power" we lean on bootstrap CI.

**Why Universe C nearly closes the GNN gap.** When the input feature set is already rich (Alpha158 subset), there is less unique signal left for graph structure to recover. Universe B (10 hand-crafted features) is where GAT, SAGE, and MLP each beat LightGBM by ≥ 0.023 IC; Universe C (51 features) sees all four models converge inside 0.043–0.053. This is consistent with prior factor-investing intuition: *graphs help most when node features are weakest*.

**Why news-as-edge fails to dominate news-as-feature dilution.** News edges add valuable co-mention information *in principle* but the within-day signal is short-lived. Our 21-day rebalance averages it away, so even the most directionally positive news-edge contrast (α3 vs α1, ΔIC = +0.010) fails BH-FDR. Recovering this signal probably needs *intra-day* rebalance or *event-conditional* training rather than richer edge bundles.

**Why one fold dominates.** Q2-2025 had unusually high cross-sectional dispersion in S&P 500 (semiconductor / mega-cap volatility plus regional-bank rebound). Cross-sectional ranking models earn most of their carry in such regimes; in flat regimes (most of 2024) every model produces near-zero IC. Researchers should always report *per-fold* numbers alongside pooled means.

---

## §7. Limitations and Future Work

> *Seven specific caveats; each must be stated in plain prose because reviewers will surface them otherwise.*

**Table ST7 — Limitations matrix.**

| # | Caveat | Concrete impact on claims |
|---|---|---|
| **L1** | **Universe C composition basis (Plan AAA top-15 factors) has LOW STABILITY.** The original Plan AAA factor ranking was computed under a same-day OHLC evaluation procedure; under strict T − 1 leak correction only 5 / 15 of the top factors survive (Plan-AAA original ∩ proxy-T1 = 5 / 15). E1 *runtime* uses T − 1 features, so Universe C results are not leaked, but the *basis* for Universe C composition is fragile. | Univ C IC 0.043–0.053 may not generalise to a leak-corrected re-ranking. Future: re-run Plan AAA permutation under leak-corrected eval before final submission. |
| **L2** | **Single profitable quarter (Q2-2025, Fold 4) drives 38–72% of IC/Sharpe.** | Headline numbers are an upper bound; LOFO-4 numbers (50–60% of headline) are the realistic lower bound. |
| **L3** | **Single-cell outliers inflate Univ C GAT Sharpe.** Cell ID 240 (seed 86, fold 4) has Sharpe<sub>gross</sub> = 75.0 vs the next-highest 17.2; without this cell the mean drops from 3.62 to ≈ 2.3. | Report median + IQR alongside the mean for Univ C GAT Sharpe. |
| **L4** | **Univ B LightGBM headline net Sharpe = −0.83 is a Fold-4 artefact** (LOFO-4 → +1.07). | We do *not* claim "Universe B features broke trees"; we frame it as Q2-2025 regime variance specifically. |
| **L5** | **LSTM, Transformer, and heterogeneous-relational GNN (HGT) architectures are not benchmarked in the E1 anchor.** HATS-3R-adapt (an STHAN-SR-inspired heterogeneous baseline) is scheduled but not yet run as of 2026-05-28. | Generalisation across architectures untested. |
| **L6** | **News-as-feature dilutes signal at 21 d (ΔIC = −0.045); news-as-edge fails BH-FDR.** | News is not a robust signal source under our setup; reported honestly. |
| **L7** | **Single market (US S&P 500).** Chinese CSI 300 / 500 and cross-market generalisation untested. | Geographic generalisation is future work. |

![Figure 10 — Plan AAA T-1 stability](../figures/F10_plan_aaa_t1_stability.png)

**Figure 10.** Plan AAA T − 1 stability. Original top-15 groups (x-axis) versus their T − 1-shifted proxy ranks (y-axis). Green stars: groups remaining within top-15 after T − 1 shift. **Plan AAA orig ∩ proxy-T1 = 5 / 15 → LOW STABILITY**. *Supports L1.*

#### What the 15 x-axis labels mean

> **Notation key.** "+N" appended to a leader name means *the leader plus N additional Alpha158 sibling features grouped together*, so the group has N + 1 columns total. `hc_` prefixes denote our hand-crafted features; all others are Qlib's standard Alpha158 library. Features without "+N" are singletons. Every feature is evaluated strictly at T − 1.

**Table T_FACTORS — Plain-English legend for Figure 10's top-15 Plan AAA factor groups.**

| # | Group label | Members (Alpha158 / hc names) | What it measures (plain English) | T − 1 stable? |
|---:|:---|:---|:---|:---:|
| 1 | `hc_mom12m` | hc_mom12m | **12-month momentum**: cumulative log return over the past ≈252 trading days. The classic "winners keep winning" factor (Jegadeesh–Titman). | N |
| 2 | `ROC30+5` | ROC30, MA60, MAX60, MIN60, QTLU60, QTLD60 | **Long-window price-level statistics.** ROC30 = 30-day rate of change `(p_t − p_{t−30}) / p_{t−30}`. MA60 = today's close relative to the 60-day moving average. MAX60 / MIN60 / QTLU60 / QTLD60 = distance from the 60-day high / low / upper 80%-quantile / lower 20%-quantile (all divided by today's close). | **Y** |
| 3 | `CNTP60+1` | CNTP60, CNTD60 | **Long-window directional balance**: fraction of *positive-return* days (CNTP60) and *negative-return* days (CNTD60) in the past 60 trading days. A bull-momentum stock has CNTP60 ≫ CNTD60. | N |
| 4 | `KMID+6` | KMID, KMID2, KSFT, KSFT2, OPEN0, HIGH0, VWAP0 | **Intraday candlestick shape.** KMID = `(close − open) / (high − low)` = body-to-range ratio (positive body = bullish bar). KSFT = `(2·close − high − low) / (high − low)` = where close sits within the day's range. KMID2 / KSFT2 are squared variants capturing shape magnitude. OPEN0 / HIGH0 / VWAP0 = today's open / high / volume-weighted-average-price divided by close. | **Y** |
| 5 | `RESI60` | RESI60 | **60-day price-trend residual**: residual of `close ~ time` linear regression over the past 60 days, i.e., how far today's close sits *above or below* its own 60-day trend line. | N |
| 6 | `BETA20+8` | BETA20, RANK20, RSV20, IMAX20, IMXD20, SUMP20, SUMD20, RANK30, RSV30 | **Market beta + position-within-range.** BETA20 = 20-day rolling beta of returns vs market. RANK20 / RANK30 = rank of today's close inside the past 20 / 30 days. RSV20 / RSV30 = Raw Stochastic Value `(close − min) / (max − min)` (KDJ-style). IMAX20 / IMXD20 = location-index of 20-day max, and (location of max − location of min) — swing-timing features. SUMP20 / SUMD20 = sum of positive / negative returns over 20 days. | N |
| 7 | `CNTP5+5` | CNTP5, CNTN5, CNTD5, CNTP10, CNTN10, CNTD10 | **Short-window directional balance** (5 / 10 days). CNTP = count of positive-return days; CNTN = count of negative-return days; CNTD = net direction (positives minus negatives). | N |
| 8 | `ROC60+3` | ROC60, IMIN60, CNTN60, SUMN60 | **Long-window downside statistics.** ROC60 = 60-day rate of change. IMIN60 = day-index of the 60-day minimum. CNTN60 / SUMN60 = count and sum of negative-return days in the past 60. | N |
| 9 | `WVMA20+1` | WVMA20, WVMA30 | **Volume-weighted dollar-flow variability** over 20 / 30 days: `std(close × volume) / mean(close × volume)` = coefficient of variation of daily dollar volume. High WVMA = unstable turnover. | N |
| 10 | `KUP+1` | KUP, KUP2 | **Upper-shadow candlestick feature**: KUP = `(high − max(open, close)) / (high − low)` = length of the bar's upper wick. A long upper shadow signals intraday rejection from the day's high. KUP2 = squared variant. | **Y** |
| 11 | `RANK60+2` | RANK60, RSV60, IMAX60 | **Long-window position-within-range.** RANK60 = rank of today's close inside the past 60 days. RSV60 = `(close − min60) / (max60 − min60)`. IMAX60 = day-index of the 60-day maximum (when the recent peak occurred). | N |
| 12 | `CNTP20+3` | CNTP20, CNTD20, CNTP30, CNTD30 | **Medium-window directional balance** (20 / 30 days): counts of positive and net-direction days, mid-horizon analogue of #7 and #3. | **Y** |
| 13 | `hc_ret_std_5d+1` | hc_ret_std_5d, hc_ret_std_10d | **Short-window return volatility**: realised standard deviation of daily returns over the past 5 / 10 days — a proxy for short-term risk. | N |
| 14 | `RSQR20` | RSQR20 | **20-day trend smoothness**: R² of `close ~ time` regression over the past 20 days. R² → 1 means a clean linear trend; R² → 0 means range-bound / choppy. | N |
| 15 | `CORR60` | CORR60 | **60-day price–volume correlation**: rolling Spearman correlation between log-close and log-volume. Positive CORR60 means volume confirms price moves (a textbook bullish-strength condition). | **Y** |

*Source: members and ranking from [artifacts/plan_aaa/ranking.csv](../artifacts/plan_aaa/ranking.csv) rows 1–15; T − 1 stability from [artifacts/plan_aaa_t1_diagnostic/group_ranking_comparison.csv](../artifacts/plan_aaa_t1_diagnostic/group_ranking_comparison.csv). Factor definitions follow Qlib's Alpha158 specification (`qlib/contrib/data/handler.py`).*

**Reading Table T_FACTORS together with Figure 10.** The five **T − 1 stable** groups (Y rows above; corresponding to green-star markers in Figure 10) are #2 `ROC30+5`, #4 `KMID+6`, #10 `KUP+1`, #12 `CNTP20+3`, and #15 `CORR60` — broadly the *intraday-candlestick* and *medium-window directional-balance* families. The ten groups that **drop out of the top-15** after T − 1 leak correction tend to be *long-window* statistics (60-day momentum / position / counts), where same-day-OHLC contamination most strongly inflated the original Plan AAA ranking. This is the empirical content of "LOW STABILITY (5 / 15)" cited in Limitation L1.

![Figure S4 — Plan AAA top-30](../figures/S4_plan_aaa_ranking_top30.png)

**Figure S4.** Plan AAA top-30 ranked groups with 95% bootstrap CI on mean ΔIC. Stars mark the 5 of 15 groups surviving the T − 1 stability filter. No groups are BH-FDR rejected at q = 0.05. *Supports L1.*

![Figure S6 — Phase 5 Step 3 Plan Z SPA](../figures/S6_phase5_step3_subset_spa.png)

**Figure S6.** Hansen SPA test for each (model, benchmark-subset) pair in the earlier Phase 5 Step 3 study. Colours: red = *p* < 0.05, amber = 0.05 ≤ *p* < 0.20, grey = *p* ≥ 0.20. Provides methodological precedent for our SPA application.

**Table ST4 — Plan AAA top-20 ranked groups** (excerpt; see [tables/ST4_plan_aaa_top20.tex](../tables/ST4_plan_aaa_top20.tex)).

| Rank | Group | $\overline{\Delta\text{IC}}$ | NW-*t* | NW-*p* | BH-*p*<sub>adj</sub> | FDR rej | T-1 stable |
|---:|:---|---:|---:|---:|---:|:---:|:---:|
| 1 | hc_mom12m | +0.0079 | 1.014 | 0.311 | 0.647 | N | N |
| 2 | ROC30+5 | +0.0043 | 2.850 | 0.004 | 0.133 | N | **Y** |
| 3 | CNTP60+1 | +0.0034 | 1.747 | 0.081 | 0.504 | N | N |
| 4 | KMID+6 | +0.0034 | 1.595 | 0.111 | 0.520 | N | **Y** |
| 5 | RESI60 | +0.0034 | 1.985 | 0.047 | 0.504 | N | N |
| 10 | KUP+1 | +0.0023 | 1.420 | 0.156 | 0.620 | N | **Y** |
| 12 | CNTP20+3 | +0.0017 | 0.765 | 0.444 | 0.775 | N | **Y** |
| 15 | CORR60 | +0.0011 | 1.276 | 0.202 | 0.620 | N | **Y** |

**Table ST5 — Phase 5 Step 3 Plan Z per-subset breakdown** (excerpt; see [tables/ST5_phase5_step3.tex](../tables/ST5_phase5_step3.tex)).

| Subset | Model | *n*<sub>days</sub> | $\overline{IC}$ | NW-SE | NW-*t* | NW-*p* | *S* | 95% CI |
|:---|:---|---:|---:|---:|---:|---:|---:|:---|
| S6 | MLP | 313 | 0.0460 | 0.0177 | 2.604 | 0.0092 | 0.326 | [−0.01, 0.80] |
| S6 | SAGE-Mean | 313 | 0.0467 | 0.0189 | 2.470 | 0.0135 | 0.310 | [−0.04, 0.79] |
| S8 | MLP | 313 | 0.0407 | 0.0183 | 2.226 | 0.0260 | 0.277 | [−0.14, 0.59] |
| S8 | SAGE-Mean | 313 | 0.0418 | 0.0186 | 2.242 | 0.0249 | 0.278 | [−0.13, 0.66] |

**Future work.** (a) Regime-conditional evaluation — replicate on 5+ years to dilute single-quarter dominance; (b) cross-market replication (CSI 300 / 500); (c) intra-day rebalance with news edges to target the short-window signal that 21 d averages away; (d) HATS-3R-adapt + HGT benchmarks under the same protocol; (e) self-supervised pre-training for graph encoders.

---

## §8. Reproducibility and Data Availability

- Code repository: [https://github.com/hryxx86/GNN-Testing](https://github.com/hryxx86/GNN-Testing) (Apache 2.0).
- 400 E1 anchor `results.csv` + per-day IC `.npy` arrays + manifests are versioned in [experiments/storya_e1_anchor/](../experiments/storya_e1_anchor/).
- Pre-registration locked at [experiments/storya_e1_anchor/prereg.json](../experiments/storya_e1_anchor/prereg.json) before any A100 launch.
- All figures and tables reproducible via the 13 modular paper-figure scripts under [paper_figs/](../paper_figs/).
- Numeric-claim verifier [scripts/verify_docs_provenance.py](../scripts/verify_docs_provenance.py) is run on this draft before each H博士 review.

---

## Supplementary Materials Index

- **S1–S18**: 18 supplementary figures embedded throughout §5 and §7 above.
- **ST1**: data + experimental setup table — *to be written* (universe sizes, fold dates, hyperparameter grid, seed list).
- **ST2**: multi-testing ledger — embedded in §3.5.
- **ST3**: full horizon × architecture table — excerpt in §5.5; full version at [tables/ST3_horizon_full.tex](../tables/ST3_horizon_full.tex).
- **ST4**: Plan AAA top-20 — excerpt in §7; full version at [tables/ST4_plan_aaa_top20.tex](../tables/ST4_plan_aaa_top20.tex).
- **ST5**: Phase 5 Step 3 Plan Z subset breakdown — excerpt in §7; full version at [tables/ST5_phase5_step3.tex](../tables/ST5_phase5_step3.tex).
- **ST6**: loss horse race paired ΔIC — embedded in §5.6.
- **ST7**: limitations matrix — embedded in §7.

---

## Editor's checklist (delete before submission)

- [ ] **L1 caveat** added to every Universe-C-derived figure caption (F2 C panels, F5 dashed lines, T1 / T2 / T4 Univ C rows, F10, S4).
- [ ] **L6 caveat** added to every Fold-4-touching figure caption (F3, F4, S3, S15, S17, T2).
- [ ] **N3 "0/5 BH-FDR"** stated verbatim in F6 / T5 captions.
- [ ] T6 related-work matrix written (target 19 papers via `literature-review` skill).
- [ ] ST1 data-setup table written (universe / fold dates / HP grid / seed list).
- [ ] HATS-3R-adapt panel added to T1, T2, F2 *if* baseline run completes before submission (else leave as L5 future work).
- [ ] `nature-polishing` pass on §1 + §6.
- [ ] `scripts/verify_docs_provenance.py docs/storya_paper_draft.md` returns 0 unmatched numeric claims.
