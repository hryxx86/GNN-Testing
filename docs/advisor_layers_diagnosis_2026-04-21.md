# Three-Layer Diagnostic Report: GNN-Testing Project

> **Framework**: Data/Label → Loss/Objective → Architecture
> A bug in a lower layer cannot be fixed by a more sophisticated layer above it.
> Prepared for advisor meeting, 2026-04-21.

---

## Executive Summary

Under the three-layer priority framework for ML research (Data → Loss → Architecture), this project's **main disease lies in Layer 2**: the production pipeline uses `F.mse_loss` on raw next-day returns, which — given a label distribution with mean ≈ 0, median ≈ 0.055%, std ≈ 2.35%, and 26.5% of events in the `|r| < 0.5%` noise zone — induces a **near-constant "lying flat" predictor**. This is consistent with all downstream symptoms (|IC| < 0.05 across architectures, SelectiveNet reverse-selecting worst predictions, normalization × regime ±0.2 IC swings, and the statistical indistinguishability between a 3-feature PC probe and 158-feature Alpha158 under Hansen SPA).

**Implication**: architecture comparisons (SAGE vs GAT vs HGT vs Transformer) conducted under a compressed, lying-flat loss are not informative about true architectural capacity. The recommended next step is to migrate the main pipeline from MSE to a ranking loss (ListNet already implemented in an archived script) and **re-run** Layer 3 comparisons afterwards.

**Key headline (verified 2026-04-21 against AAAI'24 SOTA)**: our MLP / SAGE on Alpha158 + S&P500 reaches an IC point estimate of **≈ 0.041–0.042**. StockMixer (AAAI'24) reports an IC point estimate of **0.041** on the same market (S&P500, 3 seeds). StockMixer's GNN baselines report IC point estimates of GAT 0.034, RGCN 0.028, HGNN variants 0.036–0.037. **No joint statistical test has been run across the two studies** (different data splits / re-implementations / no access to StockMixer's per-day IC series). The above are therefore reported-point-estimate comparisons only; we make **no** parity, matching, equivalence, superiority, or non-inferiority claim relative to StockMixer or its baselines, in either direction. The apparent gap to HIST (IC = 0.131) and MASTER (IC = 0.064) is entirely a **market + feature-set artefact**: those papers use Chinese CSI300, and HIST specifically uses Alpha360 (not Alpha158). See §"Literature SOTA Benchmarking" below for the cross-paper grouped comparison. **Hansen SPA with S8 as benchmark fails to reject the one-sided null of no-alternative-superiority**: MLP T_SPA = 0.270, p_consistent = 0.5506; SAGE-Mean T_SPA = 1.231, p_consistent = 0.5509 (per `experiments/step3_plan_z/hansen_spa_results.csv`). **Interpretation caveat**: SPA is a one-sided superiority test — failure to reject means "no subset, including S6, demonstrates statistically significant superior IC over Alpha158 at α = 0.05"; it is **not** an equivalence test and does **not** positively establish "S6 = S8". A rigorous equivalence claim would require TOST (two one-sided tests) against a pre-specified margin δ, which we have not conducted (Codex Round 3 Q3 constraint, see `docs/analysis.md:1844`).

---

## Layer 1 — Data / Label

### 1.1 Current Implementation

| Dimension | Implementation | File / Source |
|---|---|---|
| Stock universe | 501 S&P 500 constituents (survivorship bias acknowledged, not corrected) | `data/reference/` |
| Horizon | 5 years, daily, **1255 trading days** | yfinance |
| Price data | OHLCV; cross-validated against EODHD at **corr = 0.99982** | `download_ohlcv_yf.py` |
| Price features (3 parallel sets) | • 9-dim compact (mean / momentum / std × 5/10/21d) <br> • 14-dim Phase 5 (+ mom12m, dolvol, CORR5, maxret, RSV5) <br> • **158-dim Alpha158** (faithful qlib replication: 9 KBAR + 4 PRICE + 145 ROLLING) | `build_alpha158_features.py` |
| Text features | EODHD 1.38M news events, FinBERT 768-dim embedding + 4-dim sentiment (status: empirically shown harmful) | Phase A |
| Graph structure | Dynamic graph: **126-day window, correlation threshold 0.6**; 54 monthly snapshots + static industry edges (27,070) + news co-occurrence (2,325 / day) | v3 pipeline |
| Label | **Next-day close-to-close return** (cross-sectional, horizon ∈ {1, 5, 10, 21, 42, 63} d) | Locked per `CLAUDE.md` Rule 8 |
| Normalization | Per-fold train-only p1/p99 winsorize → daily cross-sectional z-score (no temporal leakage) | |

### 1.2 Rationale

- **Alpha158 as engineered baseline**: Codex Round 7 required a strong engineered-factor baseline to pre-empt reviewer challenges that "3 features beat everything" is a fluke. Alpha158 directly benchmarks against qlib.
- **9-dim compact set**: PCA shows effective rank is only 3 (PC1 momentum / PC2 volatility / PC3 horizon spread account for 89.7% cumulative variance); the 3-dim PC probe is a principled parsimony probe.
- **Label locked to close-to-close**: avoids tick-data noise and overnight gap contamination; consistent across horizons for multi-horizon ablation.
- **Dynamic graph (126d, 0.6)**: Pareto optimum selected from a 3×4 sensitivity grid (density 6%, stability std = 0.064).

### 1.3 Known Issues

| Issue | Evidence | Severity |
|---|---|---|
| **Fold 4 regime drift** | Q2-2025 tariff shock: daily vol 1.81% (vs 0.65-0.87% in Fold 0-3); signed pairwise correlation 0.496 (2.3×); 97% of pairs positively correlated. 4-test leakage framework rules out look-ahead, but `ret_std_21d` train→test scale drifts **+33.8%** | Honestly reported variance, not a bug |
| **FinBERT text is harmful** | MLP price-only IC = 0.026 vs all-features IC = 0.004 (**7× gap**); high-impact subset AUC = 0.476 (worse than random); replacing FinBERT with Qwen/GPT-4o yields Δ = +0.0009 | Route already cancelled |
| **SEC Lazy Prices harmful** | SAGE price IC = 0.034 → + SEC = 0.013 (**−61%**). `log1p_days_since_filing` (scale 0-7) dominates first Linear layer gradient (> 0.8), drowning the ranking signal | Gate 1 STOP |
| **Effective rank only 3** | 9 features concentrate 89.7% of variance in top 3 PCs; `ret_mean_{5,10,21}d` ≡ `momentum_{5,10,21}d` (corr = 1.00) | Redundant, not fatal |
| **Survivorship bias** | 501 tickers = current membership; no point-in-time reconstruction | Unfixed, common in literature |
| **21d horizon-feature artifact not verified** | 21d is the IC peak (inverted-U), but `momentum_21d` / `ret_std_21d` may induce label-feature leakage | **plan.md P0 pending** |

### 1.4 Improvement Options

- **P0**: Drop 21d-horizon features, re-run horizon ablation to rule out artifact.
- **P1**: Rebuild universe with point-in-time Russell 3000 / CRSP membership to address survivorship bias (high engineering cost).
- **P2**: Abandon title-level FinBERT in favour of analyst reports / earnings call transcripts — current titles average ~15 words, so the 768-dim embedding suffers the curse of dimensionality in a weak-signal regime.

---

## Layer 2 — Loss / Objective (Primary Bottleneck)

### 2.1 Current Implementation

**The production pipeline is 100% MSE**:

```python
# run_walkforward_5fold.py:422
loss = F.mse_loss(pred[mask], target[mask])
```

- All production training (walk-forward 5-fold, architecture comparison, horizon ablation, Phase 5, Gate 1) uses **MSE on raw next-day returns**.
- ListNet exists only in the archived `run_ranking_loss.py` (results in `experiments/ranking_loss_results.csv`); it was **not** adopted into the main pipeline.
- Evaluation metric is **Rank IC** (cross-sectional Spearman) — **the training loss is not aligned with the evaluation metric**.

### 2.2 Historical Rationale

- Inertia from v2: MSE is the default for regression.
- Simple, differentiable, numerically stable; no softmax-temperature sensitivity to batch size.
- The interaction between MSE optimum and this specific label distribution was not anticipated.

### 2.3 The "MSE Induces Lying Flat" Diagnosis

**Mechanism**: The next-day return distribution has mean ≈ 0.077%, median ≈ 0.055%, std ≈ 2.35%, and **26.5% of events fall in the `|r| < 0.5%` noise zone**. For a zero-mean, heavy-tailed symmetric target, the MSE-optimal predictor is driven towards a **near-constant output at zero**. The model learns "lie flat", not "rank".

**Evidence chain**:

1. **MLP price-only IC = 0.026, |IC| < 0.05 across all architectures** — prediction variance is minimal, outputs effectively collapse toward a constant.
2. **Mean separation on LR baseline = −0.00030** (exactly zero class separation).
3. **SelectiveNet reverse-selects the worst predictions** (coverage 20%: IC = −0.015 vs threshold baseline IC = +0.031) — the model has no meaningful confidence signal to select on because predictions themselves are degenerate.
4. **Normalization × Regime interaction** (Diag 1b, 14/15 cells same sign across SAGE / MLP / NoGraph): Fold 3 norm ΔIC = −0.105 (catastrophe), Fold 4 +0.211 (rescue). The mechanism is **input-scale saturation at the first Linear layer**, a classical MSE symptom (sensitive to tail outliers, compressed in the middle).
5. **Wilcoxon p = 0.60 masks ±0.2 swings** — consistent mean IC hides prediction-distribution degeneracy across regimes.

### 2.4 Improvement Options

| Option | Mechanism | Cost | Expected benefit |
|---|---|---|---|
| **ListNet top-1** (already implemented, τ = 0.2) | Softmax over cross-sectional returns per day, CE on rank distribution | Low (un-archive) | Directly aligned with Rank IC metric |
| ListMLE | Plackett–Luce likelihood, full permutation | Medium | Better top-k behaviour |
| Margin ranking / RankNet | Pairwise comparison | Medium | More robust to outliers |
| ApproxNDCG / NeuralNDCG | Differentiable NDCG surrogate | Medium-high | Directly optimises top-k portfolio |
| IC loss | Differentiable Pearson / Spearman surrogate | Low | Most target-aligned; gradients unstable |
| Sharpe loss | End-to-end differentiable Sharpe (DeepLOB-style) | High | Optimises portfolio metric directly |
| Huber + rank regulariser | Retain MSE form, add ranking regulariser | Low | Conservative upgrade |

**Recommendation**: Run a **ListNet main-pipeline replacement experiment** (2–3 days). If IC jumps 2×+, Layer 2 refactoring is justified. MASTER / FinMamba / MDGNN / THGNN all use ranking losses — MSE on raw returns is not the field-standard choice.

---

## Layer 3 — Architecture

### 3.1 Current Implementation

Six architectures systematically compared (152 runs, `arch_comparison_results.csv`):

| Model | Mean IC | CV (%) | Comment |
|---|---|---|---|
| **SAGE-Sum** | 0.04766 ± 0.00237 | **5.0%** | Most stable |
| SAGE-Mean | 0.03525 | 62.0% | Stable |
| GAT | 0.03215 | 55.1% | Seed 1024 complete failure (IC = 0.002) |
| Transformer | 0.02448 | 91.8% | Least stable |
| HGT (corr + sector) | 0.01177 | — | Medium |
| HGT (all 4 edges) | 0.00432 | very poor | **news / co-occurrence edges harmful** |

Shared structural choices:

- **2-layer message passing + ranking head (2-layer MLP)**.
- Full-batch training on Colab A100 / RTX Pro 6000.
- Input dim ∈ {9, 14, 158}; hidden 64/128; dropout 0.2–0.5; Adam + ReduceLROnPlateau.

### 3.2 Rationale

- **SAGE**: simplest message passing; most robust in published benchmarks (Hamilton 2017).
- **GAT**: adds attention to test whether edge weighting matters.
- **HGT**: heterogeneous node types (stock / news / sector), matching the project's original heterogeneous graph design.
- **Transformer**: architectural family used by MASTER (AAAI'24); chosen here for architectural comparability, not as a performance-parity claim with MASTER.
- **MLP / NoGraph baseline**: graph-free control for ablation.

### 3.3 Known Issues

1. **HGT hurt by extra edge types**: news / co-occurrence edges drop IC from 0.012 → 0.004, consistent with Layer 1's NLP-is-harmful finding.
2. **GAT / Transformer are seed-sensitive**: CV 55–92%, unsuitable for a weak-signal financial regime (parameter-variance trade-off).
3. **MLP price-only point-estimate IC is arithmetically higher than SAGE in 3 of 5 folds** of the walk-forward (`wf5_results.csv`); graph gains are small in magnitude and negative in at least one fold (Fold 0 ablation). This is a per-fold point-estimate count, not a significance-tested "beat" claim.
4. **Meta-issue (most important)**: **architecture comparison under a lying-flat loss is not informative**. Different architectures' "lying-flat degree" converges; real capacity differences are compressed by MSE. This directly motivates Layer 2 refactoring before further Layer 3 work.
5. **SPA with S8 as benchmark fails to reject at α = 0.05** (MLP p_c = 0.5506, SAGE-Mean p_c = 0.5509 — authoritative source: `experiments/step3_plan_z/hansen_spa_results.csv`): no subset (including S6) demonstrates statistically significant superior IC over the 158-feature Alpha158 library. This is a **non-superiority** result, not an equivalence proof; Layer 1's information ceiling is low, further reducing any room for architectural sophistication to shine. An equivalence claim ("S6 = S8") would require TOST with a pre-specified margin.

### 3.4 Improvement Options

- **Short term**: **Do not add new architectures**. Comparing Layer 3 under a broken Layer 2 wastes compute.
- **Medium term**: **After** migrating to a ranking loss, re-run SAGE vs GAT vs Transformer — the ranking may flip.
- **Long term**: If GNNs still fail to beat MLP under ranking loss, honestly concede that "cross-stock message passing offers limited gain for S&P 500 ranking" (consistent with Findings #10 / #11) and report as a workshop-level negative result.

---

## Literature SOTA Benchmarking (verified 2026-04-21)

**Important framing rule**: IC numbers from different papers are only comparable when **market**, **feature set**, and **test-period regime** match. The table below groups results accordingly.

### Group 1 — CSI300 + Alpha158 (same feature set as ours, Chinese market)

#### MASTER (AAAI 2024), Alpha158, 5 seeds
Train: Q1-2008 – Q1-2020; Val: Q2-2020; Test: Q3-2020 – Q4-2022 (10 quarters)

| Model | IC | ICIR | RankIC | RankICIR | Ann. Ret | IR |
|---|---|---|---|---|---|---|
| XGBoost | 0.051±0.001 | 0.37 | 0.050±0.001 | 0.36 | 0.23 | 1.9 |
| LSTM | 0.049±0.001 | 0.41 | 0.051±0.002 | 0.41 | 0.20 | 2.0 |
| GRU | 0.052±0.004 | 0.35 | 0.052±0.005 | 0.34 | 0.19 | 1.5 |
| Transformer | 0.047±0.007 | 0.39 | 0.051±0.002 | 0.42 | 0.22 | 2.0 |
| GAT | 0.054±0.002 | 0.36 | 0.041±0.002 | 0.25 | 0.19 | 1.3 |
| DTML | 0.049±0.006 | 0.33 | 0.052±0.005 | 0.33 | 0.21 | 1.7 |
| **MASTER** | **0.064±0.006** | **0.42** | **0.076±0.005** | **0.49** | **0.27** | **2.4** |

MASTER on CSI800 (same setup): IC = 0.052±0.006.

#### qlib official benchmarks, CSI300, Alpha158, 20 seeds

| Model | IC | ICIR | RankIC | RankICIR |
|---|---|---|---|---|
| Transformer | 0.0264 | 0.205 | 0.0407 | 0.327 |
| GRU | 0.0315 | 0.245 | 0.0428 | 0.344 |
| LSTM | 0.0318 | 0.237 | 0.0435 | 0.339 |
| ALSTM | 0.0362 | 0.279 | 0.0463 | 0.366 |
| GATs | 0.0349 | 0.251 | 0.0462 | 0.356 |
| MLP | 0.0376 | 0.285 | 0.0429 | 0.322 |
| SFM | 0.0379 | 0.296 | 0.0464 | 0.382 |
| Linear | 0.0397 | 0.300 | 0.0472 | 0.353 |
| **TRA** | **0.0440** | **0.354** | **0.0540** | **0.445** |
| LightGBM | 0.0448 | 0.366 | 0.0469 | 0.388 |
| CatBoost | 0.0481 | 0.337 | 0.0454 | 0.331 |
| XGBoost | 0.0498 | 0.378 | 0.0505 | 0.413 |
| **DoubleEnsemble** | **0.0521** | **0.422** | **0.0502** | **0.412** |

The gap between qlib-reported baselines (GAT ≈ 0.035, Transformer ≈ 0.026) and MASTER-reported baselines (GAT ≈ 0.054, Transformer ≈ 0.047) is almost entirely explained by the test-period regime (2020-Q3 through 2022-Q4 in MASTER is the high-volatility post-COVID period). **Test-period choice dominates IC magnitude**.

### Group 2 — CSI300 + **Alpha360** (different feature set; not comparable to us)

#### HIST (CIKM'22), Alpha360, 10 seeds, MSE loss
Train: 2007-01-01 – 2014-12-31; Val: 2015-2016; Test: 2017-01-01 – 2020-12-31

| Method | CSI300 IC | Rank IC | Prec@3 | @5 | @10 | @30 |
|---|---|---|---|---|---|---|
| MLP | 0.082 | 0.079 | 57.21 | 57.10 | 56.75 | 55.56 |
| LSTM | 0.104 | 0.098 | 59.51 | 57.30 | 58.40 | 56.98 |
| GRU | 0.113 | 0.108 | 59.95 | 59.28 | 58.59 | 57.43 |
| Transformer | 0.106 | 0.104 | 60.76 | 60.06 | 59.48 | 57.71 |
| GATs | 0.111 | 0.105 | 60.49 | 59.96 | 59.02 | 57.41 |
| ALSTM | 0.115 | 0.109 | 59.51 | 58.92 | 58.92 | 57.47 |
| ALSTM+TRA | 0.119 | 0.112 | 60.45 | 59.52 | 59.16 | 58.24 |
| **HIST** | **0.131** | **0.126** | **61.60** | **61.08** | **60.51** | **58.79** |

**HIST IC = 0.131 is NOT comparable to MASTER IC = 0.064 or to our IC = 0.041.** Alpha360 contains 60 days × 6 raw OHLCV fields = 360-dimensional time-series input — a fundamentally richer feature set than the 158 aggregated factors of Alpha158. Do not cross-compare.

### Group 3 — **S&P500** (the only directly comparable group for us)

#### StockMixer (AAAI 2024), 16-day lookback raw features, 3 seeds
S&P500: 474 stocks, 2016-01-04 – 2022-05-25, Train 1006 / Val 253 / Test 352 days

| Method | IC | RIC | Prec@N | Sharpe |
|---|---|---|---|---|
| Linear | 0.016 | 0.156 | 0.520 | 0.674 |
| LSTM | 0.031 | 0.186 | 0.531 | 1.332 |
| ALSTM | 0.029 | 0.181 | 0.532 | 1.298 |
| RGCN | 0.028 | 0.175 | 0.528 | 1.359 |
| GAT | 0.034 | 0.191 | 0.541 | 1.484 |
| RSR-I | 0.033 | 0.200 | 0.542 | 1.437 |
| STHAN-SR (HGNN) | 0.037 | 0.227 | 0.549 | 1.533 |
| ESTIMATE (HGNN) | 0.036 | 0.241 | **0.553** | 1.547 |
| **StockMixer** | **0.041** | **0.262** | 0.551 | **1.586** |
| **Ours (MLP / SAGE on Alpha158)** | **0.041 / 0.042** | — | — | — |

### Cross-market calibration

For the same model family, moving from CSI300 → S&P500 roughly halves reported IC:

| Model family | CSI300 IC | S&P500 IC | Ratio |
|---|---|---|---|
| LSTM | ≈ 0.049 | 0.031 | 0.63× |
| GAT | ≈ 0.054 | 0.034 | 0.63× |
| SOTA (MASTER vs StockMixer) | 0.064 | 0.041 | 0.64× |

**S&P500 SOTA ≈ CSI300 SOTA × 0.6** — reflecting EMH-strong large-cap efficiency plus no price-limit / T+1 market frictions.

### Our positioning (revised)

| Comparison | Our IC (point est.) | Reference IC (reported) | Arithmetic note |
|---|---|---|---|
| Ours vs StockMixer on S&P500 | 0.041–0.042 | 0.041 | Reported point estimates coincide numerically; **no joint statistical test run** (different data splits / re-implementations) |
| Ours vs StockMixer's GNN baselines | 0.041 | 0.028–0.037 | Our reported point estimate is arithmetically higher than each listed baseline's reported point estimate; **no joint statistical test run** |
| Ours vs MASTER on CSI300 | 0.041 | 0.064 | Different market + regime; not directly comparable |
| Ours vs HIST on CSI300 Alpha360 | 0.041 | 0.131 | Different feature set (Alpha360 vs Alpha158); not directly comparable |
| Sample size (seeds) | 3 | StockMixer 3, MASTER 5 | Same order of magnitude as the cited papers' seed counts |

**Claims table reading rule (2026-04-21-c)**: all cells above are *arithmetic comparisons of reported point-estimate IC values* (ours vs other authors' self-reported numbers). They are **not** parity / matching / equivalence / beating claims in any statistical sense, and they are not significance-tested. We have no access to StockMixer or MASTER per-day IC series, so a paired or SPA test across studies is not feasible.

Directly quotable from StockMixer (AAAI'24): "slight performance degradation is observed on NYSE with most stocks (1737), which may indicate that **insufficient inductive bias gradually come into force in dealing with larger candidate pools**" — cited here for its large-universe qualitative observation; it is **not** offered as independent validation of any quantitative parity claim.

---

## Prior Art on Feature Parsimony (verified 2026-04-21)

**Originality verdict: moderate, not high.** The "few features suffice" direction is already established in the asset-pricing literature. The specific combination — Alpha158 benchmark + PC-representative compact set + Hansen SPA + deep GNN + daily S&P 500 — is new, but the directional claim is not.

### Must-cite prior art

| Paper | Venue | Core claim | Threat level |
|---|---|---|---|
| **Gu, Kelly & Xiu (2020), "Empirical Asset Pricing via Machine Learning"** | RFS | Across 11 ML methods, **momentum / liquidity / volatility** are the dominant signals | **High** — names the exact 3 feature families we extract as PC1/PC2/PC3. A finance reviewer will say "known." |
| **Messmer & Audrino (2022), "The Lasso and the Factor Zoo"** | Forecasting | A very sparse model with **3–5 principal components** matches a **10–30 characteristic** model | **Highest** — single closest prior art. We must show our result is stronger (lower-dim, NN-based, SPA-tested, daily-horizon, larger engineered baseline). |
| **Feng, Giglio & Xiu (2020), "Taming the Factor Zoo"** | JF | Double-Selection Lasso: most new factors redundant given existing | Medium — supports redundancy but on factor returns, not daily stock ranking |
| **Green, Hand & Zhang (2017)** | RFS | Of 94 characteristics, only **2** survive post-2003 in non-microcaps | Medium — establishes characteristic-dimension collapse as an empirical fact |
| **Harvey, Liu & Zhu (2016), "…and the Cross-Section…"** | RFS | t > 3 cutoff with multiple testing; only 9 of 313 variables survive | Medium — methodological precedent for our SPA usage |
| **Kelly, Pruitt & Su (2019), IPCA** | JFE | 5–6 latent factors price the cross-section | Low — monthly panel, not daily ranking |
| **Lettau & Pelger (2020), RP-PCA** | RFS | 5–6 factors suffice for covariance + mean fit | Low — portfolio-level |
| **DeMiguel, Martin-Utrera, Nogales & Uppal (2020)** | RFS | 6 characteristics jointly significant (15 under transaction costs) | Medium — supports parsimony |
| **StockMixer (AAAI'24), Fang et al.** | AAAI | Reports a simple MLP with IC = 0.041 on S&P500, higher in point-estimate than their own GNN / Transformer baselines (their reported table, no joint test across their architectures shown here) | Low — our reported point estimate (0.041–0.042) numerically coincides with theirs in the same market and at a similar seed count; **no joint statistical test across the two studies is run**, so this is cited as a cross-study point-estimate coincidence, not as an independent parity result |

### Defensible contribution framing

The paper should **not** be pitched as "3 features ≈ 158 features" (directionally known — reviewers will reject as duplicative). Instead pitch:

> "**Hansen SPA + BH-FDR as a principled statistical protocol for factor-library redundancy in deep stock-ranking pipelines.** We show that the 158-factor Alpha158 library of qlib **does not deliver statistically significant IC improvement over** a compact 3-feature PCA-representative set of economically-grounded momentum / volatility / horizon-spread basics, on daily S&P 500 ranking with a graph neural network — at the point-estimate IC level of 0.041, which coincides numerically with the IC reported by AAAI'24 StockMixer on the same market. We do **not** claim parity, matching, or equivalence with StockMixer (no joint statistical test across studies); we report the coincidence as context for the reader. (Note: the paper submission should add a TOST equivalence test with pre-specified margin δ and a reverse-direction SPA to convert this non-superiority finding into a positive equivalence claim about S6 vs S8.)"

Lean on the **statistical test** (SPA + FDR is the new methodological contribution), not on the parsimony claim itself.

### Residual threats

1. **Gu-Kelly-Xiu already named the three feature families.** Our PC probe (momentum / volatility / mean-return) is literally their ML-consensus signals. A strict reviewer will say the feature choice is unoriginal — we must defend via "their result is at monthly horizon with linear models; ours is daily with GNN + SPA."
2. **Messmer-Audrino already showed 3–5 PCs ≈ 10–30 characteristics.** Our 3 ≈ 158 is quantitatively sharper but directionally identical. We must show that (a) SPA is stricter than their test, (b) 158 is a larger engineered library than their 30, (c) daily ranking is a harder regime than their setup.
3. **No survey on "parsimony in financial ML" exists** — small gap to fill in discussion, but insufficient for a standalone contribution.

**Full paper list and annotations available in the 2026-04-21 prior-art search result.**

---

## Recommendations for the Advisor Meeting

### Three-Layer Priorities

1. **Layer 1 — Data / Label is basically healthy**:
   - Alpha158 engineered baseline in place.
   - Label locked; Fold 4 regime stress honestly reported.
   - Residual: survivorship bias unfixed; 21d horizon-feature artifact check pending.
   - Text route empirically falsified (FinBERT / SEC both harmful) — this is itself a publishable negative result.

2. **Layer 2 — Loss is the primary bottleneck (priority discussion point)**:
   - Main pipeline is 100% MSE; under mean ≈ 0, heavy-tailed symmetric label distribution, it induces near-constant predictions.
   - SOTA literature (MASTER / FinMamba / MDGNN / THGNN) uniformly uses ranking losses.
   - **Proposed action**: within 1 week, promote ListNet to the main pipeline and re-run walk-forward 5-fold against the MSE baseline.
   - If IC jumps meaningfully, all prior architecture rankings must be re-run (results may invert).

3. **Layer 3 — Architecture is on hold**:
   - SAGE-Sum is already a CV = 5% stable operator under MSE.
   - Real architectural differences can only be revealed after Layer 2 is fixed.
   - **Avoid writing an architecture-comparison paper on top of a lying-flat loss**.

### Publication Strategy

- **Plan A (strongly recommended)**: Parsimony paper (Finding #1, ICAIF / FinNLP). **Independent of the Layer 2 issue** — the S6 "does not outperform S8" finding is a loss-invariant Layer 1 result; drafting can start immediately. Context only: our reported-IC point estimate (0.041–0.042) coincides numerically with StockMixer's reported 0.041 at the same market and similar seed count, but with no joint statistical test across studies — this is a **point-estimate coincidence in reported numbers**, not a SOTA-match / parity / equivalence claim in either direction. **Note**: to claim true equivalence with S8 (not just non-superiority), a TOST with pre-specified margin δ + reverse-direction SPA (S6 as benchmark) should be added before submission.
- **Plan B**: Horizon ablation paper (Finding #2); requires the 21d artifact check first.
- **Plan C**: Negative results on text features (Findings #7 / #11 / #12); workshop tier.
- Results from the Layer 2 refactoring belong to a subsequent paper and do not block the current submission.

---

## Evidence Appendix (key file pointers)

- MSE loss call site: `run_walkforward_5fold.py:422`
- Archived ranking-loss implementation: `archived/scripts/run_ranking_loss.py` (ListNet at τ = 0.2)
- Label distribution diagnostics: `docs/analysis.md` §2026-03-03-a (D.1 label noise)
- Normalization × regime interaction: `experiments/diag1_normalization_results.csv`, `experiments/diag1b_replication_results.csv`
- Architecture comparison: `experiments/arch_comparison_results.csv`
- Hansen SPA S6 vs S8: `experiments/hansen_spa_results.csv`
- Fold 4 diagnostic framework: `analyze_fold4_leakage.py`, `experiments/fold4_zdrift_summary.csv`
- Full project overview: `docs/project_findings_overview_2026-04-20.md`
