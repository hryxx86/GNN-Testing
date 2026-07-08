# Literature Benchmark: Our Paper vs. 15 Most-Relevant Recent Top-Venue Papers

> Date: 2026-07-03. Extraction: 3 parallel research agents (per-paper source fetches:
> ar5iv/AAAI-OJS/ACM-DL/Springer/publisher pages), synthesis and evaluation by Claude,
> our-paper facts from paper/main.tex (post-M14, commit 499fd75). Section 3 = Group A
> (direct competitors) + Group C (critical); Section 4 = Group B (finance rigor);
> Section 5 = forensic comparison matrix; Section 6 = evaluation of our paper.

## 1. Scope & selection (PRISMA-lite)

- **Search date**: 2026-07-03. Sources: arXiv, Semantic Scholar via web search, ACM DL,
  publisher pages, plus our own `paper/references.bib` (allowed per H博士 instruction).
- **Inclusion**: (a) GNN/deep cross-sectional stock ranking at top CS venue (AAAI/KDD/CIKM/
  TOIS/ICLR/NeurIPS), 2019+; or (b) evaluation-rigor empirical finance at top journal
  (RFS/JF/MS), 2020+; or (c) critical benchmark/systematic survey at top journal, 2024+.
- **Exclusion log**: HATS (arXiv-only, no venue); FinMamba 2025 (arXiv-only);
  CI-STHPAN & ECHO-GL (AAAI'24, same-cluster redundancy with MASTER/StockMixer/MDGNN);
  Leippold et al. JFE'22 (redundant with GKX for the protocol axis); Kim 2019 excluded
  with HATS. FinRL/RL-trading strand excluded (different task).
- **Final 15**: Group A (direct competitors, 8): RSR TOIS'19, STHAN-SR AAAI'21, AD-GAT
  AAAI'21, TRA KDD'21, THGNN CIKM'22, MASTER AAAI'24, StockMixer AAAI'24, MDGNN AAAI'24.
  Group B (rigor, 5): GKX RFS'20, HXZ RFS'20, JKP JF'23, KMZ JF'24, ACM MS'23.
  Group C (critical, 2): Patel et al. CSUR'24 survey, Prata et al. AIR'24 LOB benchmark.

## 2. Our paper's protocol profile (comparison baseline)

All facts from `paper/main.tex` (post-M14, commit 499fd75):

| Axis | Our paper |
|---|---|
| Research question | When, if at all, does the graph help in cross-sectional S&P 500 ranking? (conditional attribution, not architecture ranking) |
| Data | S&P 500 fixed survivor snapshot, 501 names, 2021–2026 daily; label = 21d fwd c-t-c market-excess z-scored; features strictly T-1; 1-day execution lag (main.tex:166-172) |
| Universes | 2 feature universes: B leak-free price-volume; C leak-selected Alpha158 (leak disclosed as L1) |
| Models | Tuned ladder L0–L7: LightGBM→MLP→corr-GAT→news→sector→combined→dense-attention (MASTER-family)→R-GCN/HATS-style; each independently tuned, 30 Optuna trials equal budget (main.tex:174) |
| Splits | 12-fold expanding walk-forward, quarterly test 2023Q1–2025Q4, T=749 pooled days; 21d purge (main.tex:129) |
| Seeds | 10 canonical seeds everywhere; 2160 main cells; seed-mean estimand + dispersion caveat (main.tex:159,174) |
| Significance | Hansen SPA (consistent, stationary bootstrap, block 21d) global; 20 pre-reg DM/HLN local contrasts (main.tex:145-147) |
| Mult-testing | BH-FDR q=0.05 pooled 20-test family (+ per-universe identical); post-hoc: pooled-26 BH, BY(arbitrary dependence) 7/11 survive (main.tex:147,248) |
| Costs | Turnover-scaled net Sharpe at c∈{0,5,10,15,20,30}bps, headline 10bps; per-arm per-fold L1 turnover (main.tex:140) |
| Pre-registration | Frozen hparams (md5 59ddd0a2) before confirmatory run; 2 confirmatory families only; exploratory disclosed & excluded (main.tex:162) |
| Power | MDE=2.8×SE_block per contrast; "underpowered" made explicit (0/6 Family-2 all under MDE) (main.tex:155) |
| Positive control | Pre-confirmatory planted-signal: GAT recovers 82%, SAGE 91% of achievable IC 0.047, MLP≈0 (main.tex:180) |
| Graph-off ablation | L2−L1 = feature-matched no-graph contrast IS the headline (ΔIC=−0.0133 B / −0.0119 C, BH-reject both; M14 3× budget: B survives p=0.002, C drops p=0.059) (main.tex:114,174) |
| Capacity control | Family-2 fixed-operating-point edge attribution (0/6 BH, 6/6 underpowered); tuned-ladder contrasts labeled capacity-confounded (main.tex:153) |
| Leakage audit | T-1 features; PIT news (publication timestamp); Universe-C selection leakage quantified (5/15 groups survive T-1 re-rank); survivorship L8 quantified: 14.8% names / 8.2% stock-days / 8.1% look-ahead / 16.3% two-sided (main.tex:176,370,378) |
| Stability | C/L5s collapse disclosed (27.5% cells undefined IC), not repaired (main.tex:224) |
| Robustness | LOFO 0/12, LOSO 0/10, NW lag21, BY, M14 trials sweep, zero-fill/exclusion sensitivity |
| Results | SPA non-reject (p=0.277 B / 0.077 C); strongest signal = graph penalty L2−L1<0; MLP strongest positive (suggestive); news edge harms tuned but not fixed-op; cost crosswalk 10bps |
| Conclusion type | Conditional findings + failure modes, explicitly NOT an architecture ranking |

## 3. Per-paper summaries — Group A (direct competitors)

### 3.1 GNN core five (agent 1 extraction, verified sources: ar5iv/AAAI PDFs)

## RSR (ACM TOIS 2019)
- **Full citation**: Fuli Feng, Xiangnan He, Xiang Wang, Cheng Luo, Yiqun Liu, Tat-Seng Chua. "Temporal Relational Ranking for Stock Prediction." ACM Transactions on Information Systems 37(2), 2019. arXiv:1809.09441.
- **Research question**: (a) Is formulating stock prediction as a *ranking* task more useful than regression/classification? (b) Do stock relations (sector-industry, Wikidata) enhance neural sequence models?
- **Data**: NASDAQ (1,026 stocks) and NYSE (1,737 stocks); 01/02/2013–12/08/2017; daily; label = 1-day return ratio; ranking task (pointwise + pairwise).
- **Graph construction**: Sector-industry relations (112/130 types) + Wikidata first/second-order corporate relations (42/32 types). Static binary relation encodings; edge weights recomputed daily by Temporal Graph Convolution. Look-ahead concerns: Wikidata single snapshot applied unchanged across train and 2017 test; universe restricted to stocks with near-complete full-sample histories (survivorship-style filter); neither discussed.
- **Method**: LSTM sequential embedding (1-day return + 5/10/20/30d MAs) → Temporal Graph Convolution (explicit/implicit variants) → FC prediction. Loss: pointwise regression + pairwise max-margin ranking.
- **Evaluation protocol**: single chronological split — train 2013–2015 (756d), val 2016 (252d), test 2017 (237d); no walk-forward; 5 repeated runs mean; metrics MSE/MRR/IRR (top-1 daily buy-hold-sell); NO significance tests; NO multiple-testing correction; costs explicitly ignored ("transaction costs ... are quite cheap"); top-1 daily strategy implies ~100% daily turnover, unreported.
- **Headline results**: RSR_I: NASDAQ IRR 1.19 (Wiki) vs Rank_LSTM 0.68; NYSE 1.06 (industry) vs 0.56.
- **Conclusions claimed**: Ranking formulation superior; relations enhance prediction via TGC; relation usefulness market-dependent.
- **Stated limitations**: Single bullish test year (2017); top-1 selection "highly risky"; metric inconsistency (best IRR ≠ best MSE/MRR); no costs/short/risk management.
- **Ablations**: Rank_LSTM = no-graph variant. On NASDAQ, no-graph Rank_LSTM (0.68) BEATS several graph variants (GCN 0.24, RSR_E industry 0.20, RSR_I industry 0.23); only Wiki RSR_I (1.19) exceeds it — graph benefit strongly relation- and market-dependent, not uniform.

## STHAN-SR (AAAI 2021)
- **Full citation**: Ramit Sawhney, Shivam Agarwal, Arnav Wadhwa, Tyler Derr, Rajiv Ratn Shah. "Stock Selection via Spatiotemporal Hypergraph Attention Network: A Learning to Rank Approach." AAAI 2021, 35(1), 497–504.
- **Research question**: Do learning-to-rank profit optimization + higher-order hypergraph relations improve stock selection over SOTA neural forecasters?
- **Data**: NASDAQ (1,026), NYSE (1,737), TSE (95); 2013–2017 / 2015–2020; daily; 1-day return ratio; top-5 traded.
- **Graph construction**: Static hypergraph — industry hyperedges (GICS) + Wikidata corporate hyperedges; 862/1,595/84 hyperedges. Look-ahead: Wikidata mined at collection time, held fixed through test (incl. TSE test to 08/2020); not discussed.
- **Method**: LSTM → temporal attention + Hawkes-process attention → 2 spatial hypergraph conv layers with hypergraph attention (K=4 heads). Loss: pointwise + pairwise ranking.
- **Evaluation protocol**: single chronological split per market (3 markets, 1 test window each); 5 runs mean, no variance table; metrics Sharpe (top-5), IRR, NDCG@5; Wilcoxon signed-rank p<0.01 vs 2 reference models; NO multiple-testing correction; NO costs; turnover unreported.
- **Headline results**: NASDAQ SR 1.42 vs RSR-I 1.34; NYSE 1.12 vs 0.95; TSE 1.19 vs iRDPG 1.10.
- **Conclusions claimed**: Ranking + hypergraph higher-order relations significantly outperform SOTA across 3 markets; hypergraphs > pairwise graphs.
- **Stated limitations**: Static hypergraph (future work: time-evolving, news/social data). Costs not discussed.
- **Ablations**: Graph removal: LSTM-only SR 0.95 vs full 1.42; Hawkes+LSTM (no graph) 1.06; HG-conv w/o attention 0.93 (plain hypergraph conv "does not lead to significant improvements"); hyperedge deletion monotonically degrades NDCG@5.

## AD-GAT (AAAI 2021)
- **Full citation**: Rui Cheng, Qing Li. "Modeling the Momentum Spillover Effect for Stock Prediction via Attribute-Driven Graph Attention Networks." AAAI 2021, 35(1), 55–62.
- **Research question**: Model momentum spillover via latent time-varying firm relations inferred from market signals + attribute-sensitive gating?
- **Data**: S&P 500 filtered to 198 stocks (no missing data + ≥100 news articles over full period — survivorship/coverage filter using future info); Feb 2011–Nov 2013 (700 days); daily; binary movement label (close>open); 5 technical + 6 Loughran–McDonald news sentiment features.
- **Graph construction**: No predefined graph — unmasked all-pairs attention infers latent relation strength each timestamp (fully dynamic); attribute-mattered (AM) gated aggregation. Comparison-only: 5 static Capital IQ relations. Look-ahead: full-period universe filters; Capital IQ single snapshot.
- **Method**: Firm-specific bilinear tensor fusion (technical × textual) → GRU(360) → unmasked attention (6 heads) + AM aggregator → softmax. Loss: cross-entropy.
- **Evaluation protocol**: single split 560/70/70 days (ONE 70-day test window); 30 inits, top-5 SELECTED BY VALIDATION, mean of the 5 reported (selection protocol!); metrics DA/AUC only (no backtest); t-tests p<0.05 (variant unspecified); NO multiple-testing correction; NO costs (no trading sim).
- **Headline results**: DA 0.5647 / AUC 0.5894 vs TGC 0.531/0.532.
- **Conclusions claimed**: Latent inferred relations beat predefined relations; spillover is attribute-sensitive.
- **Stated limitations**: Only generalization to other domains "yet to be explored". No discussion of short single test window, cost-free eval, or run-selection protocol.
- **Ablations**: Aggregator × relation grid; tensor-fusion ablation. No pure no-graph ablation within framework (LSTM/GRU/eLSTM DA≈0.515–0.520 = implicit reference).

## THGNN (CIKM 2022)
- **Full citation**: Sheng Xiang, Dawei Cheng, Chencheng Shang, Ying Zhang, Yuqi Liang. "Temporal and Heterogeneous Graph Neural Network for Financial Time Series Prediction." CIKM 2022. arXiv:2305.08740.
- **Research question**: Do dynamically learned price-correlation graphs + temporal/heterogeneous attention beat manual/NLP static relations?
- **Data**: S&P 500 + CSI 300; prices 2016–2021; test = calendar 2020; label = binary top-100/bottom-100 next-day return classification.
- **Graph construction**: Regenerated daily from trailing 20-day correlation, |ρ|≥0.6, positive/negative edge types (NOTE: near-identical to our α1 graph: trailing Spearman 126d, |ρ|>0.6). No obvious graph leakage.
- **Method**: Transformer encoder (8h, d128) → temporal graph attention (4h) over pos/neg neighbors → heterogeneous fusion → MLP. Loss: BCE on labeled nodes.
- **Evaluation protocol**: single test year (2020 = COVID year) per market; re-trained daily; 5 repeated tests mean, no std/CI; metrics ACC + portfolio (ARR/AVol/MDD/ASR/Calmar/IR) from daily top-k; NO significance tests; NO multiple-testing correction; costs explicitly ignored; turnover unreported.
- **Headline results**: S&P: ACC 0.579 vs AD-GAT 0.564; ARR 0.665 vs 0.535; ASR 1.421 vs 1.170. Production deployment at EMoney Inc. claimed.
- **Conclusions claimed**: Dynamic correlation graphs + temporal/hetero attention beat sequence-only and static-relation GNNs; real-world utility.
- **Stated limitations**: "Room for improvement" on relation graphs; no discussion of costs or single-COVID-year test.
- **Ablations**: -noenc 0.548, -notemp 0.539, -nohete 0.553 vs full 0.579; all ablations RETAIN the graph — no pure no-graph variant of the framework (standalone Transformer only as external baseline).

## MDGNN (AAAI 2024)
- **Full citation**: Hao Qian, Hongting Zhou, Qian Zhao, et al. "MDGNN: Multi-Relational Dynamic Graph Neural Network for Comprehensive and Dynamic Stock Investment Prediction." AAAI 2024. arXiv:2402.06633.
- **Research question**: Does a multi-relational dynamic heterogeneous graph (stocks/banks/industries, daily snapshots) + temporal Transformer beat sequential and single-relation/static graph methods?
- **Data**: CSI 100 + CSI 300 (China); Jan 2020–Feb 2023; daily; label = next-day benchmark-adjusted return (regression, IC/Precision@30 evaluated).
- **Graph construction**: Daily heterogeneous snapshots; 3 meta-path families (Stock–Stock sector/ownership/co-holding; Stock–Bank–Stock; Stock–Industry–Industry–Stock); 42-dim features; 10-day window. Look-ahead: PIT correctness of ownership/bank data NOT documented.
- **Method**: Intra-day hierarchical multi-relational attention → inter-day Transformer with ALiBi + causal masking → sigmoid linear head. Exact loss form not reported.
- **Evaluation protocol**: 7 rolling folds (6mo train, last month val, 6mo test frozen); seeds NOT reported; stds in tables of unclear provenance; metrics IC/IR/CR/Precision@30; NO significance tests; NO multiple-testing correction; NO costs.
- **Headline results**: CSI300 IC 0.0322 vs HTGNN 0.0192; CSI100 IC 0.0123 vs 0.0118 (+4% only — gains concentrate on bigger graph).
- **Conclusions claimed**: Multi-relational + dynamic jointly necessary; bank relations > industry relations; benefits grow with graph size.
- **Stated limitations**: Essentially none stated; single market, costs, seed robustness not discussed.
- **Ablations**: w/o meta-path IC 0.0216 (largest drop); relation-subset ladder 0.0217→0.0264→0.0283→0.0322. No complete no-graph variant (MLP/LSTM/Transformer only external).

### 3.2 Recent architectures + TRA (agent 2 extraction)

## MASTER (AAAI 2024)
- **Full citation**: Tong Li, Zhaoyang Liu, Yanyan Shen, Xue Wang, Haokun Chen, Sen Huang. "MASTER: Market-Guided Stock Transformer for Stock Price Forecasting." AAAI 2024. arXiv:2312.15235.
- **Research question**: Model momentary (same-day) + cross-time stock correlations jointly; use market status for automatic feature gating.
- **Data**: CSI300 + CSI800 (China); 2008–2022; daily; label = cross-sectionally z-scored 5-day return; lookback 8 days.
- **Method**: Market-guided gating → intra-stock transformer (4h) → inter-stock attention per timestep (2h) → temporal attention → linear head. Loss: MSE.
- **Evaluation protocol**: SINGLE chronological split (train 2008–2020Q1, val 2020Q2, test 2020Q3–2022Q4); 5 seeds with std; metrics IC/RankIC/ICIR/RankICIR + AR/IR (top-30 daily); t-test p<0.01 starred; NO multiple-testing correction; NO costs; turnover unreported.
- **Headline results**: CSI300 IC 0.064 vs DTML 0.049 (+31%); claims avg +13% ranking, +47% portfolio metrics.
- **Conclusions claimed**: Market-guided gating + decoupled attention beats SOTA; attention maps reveal realistic correlations.
- **Stated limitations**: No formal limitations section; data-hungriness acknowledged.
- **Ablations**: Aggregation swaps (bi-LSTM 0.058, naive joint attention 0.041); gating helps across temperatures. No graph-off/new-market generalization test.

## StockMixer (AAAI 2024)
- **Full citation**: Jinyong Fan, Yanyan Shen. "StockMixer: A Simple Yet Strong MLP-Based Architecture for Stock Price Forecasting." AAAI 2024, 8389–8397.
- **Research question**: Can a simple MLP-based architecture match/beat complex RNN/GNN/Transformer hybrids?
- **Data**: NASDAQ (1,026) + NYSE (1,737) 2013–2017 (Feng et al. datasets) + S&P500 (474) 2016–2022; daily; 1-day return; 16d lookback.
- **Method**: Indicator mixing + causally-masked time mixing with multi-scale patching + stock mixing via N→m market-state compression (m=20/25/8) — "self-learnable hypergraph analogue, no prior relation data". Loss: MSE + 0.1·pairwise ranking hinge.
- **Evaluation protocol**: SINGLE chronological split per dataset; 3 repetitions (variance NOT reported); metrics IC/RankIC/prec@10/Sharpe; t-test p<0.01 (caption); NO multiple-testing correction; NO costs.
- **Headline results**: NASDAQ IC 0.043 vs STHAN-SR 0.039; S&P500 SR 1.586 best; claimed +7.6%/+10.8% rank metrics, +10.9% risk-adjusted.
- **Conclusions claimed**: Lightweight MLP beats hybrid SOTA on most metrics at lower cost; time mixing most important; market-state compression more robust than full message passing.
- **Stated limitations**: Degrades on largest pool (NYSE 1,737) — insufficient inductive bias for large candidate pools.
- **Ablations**: w/o time mixing IC 0.043→0.018; w/o stock mixing 0.037; LSTM+stock-mixing 0.041 competitive w/o prior knowledge. No new-data generalization test.

## TRA (KDD 2021)
- **Full citation**: Hengxu Lin, Dong Zhou, Weiqing Liu, Jiang Bian. "Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor and Optimal Transport." KDD 2021. arXiv:2106.12950.
- **Research question**: Discover/distinguish multiple co-existing trading patterns without explicit pattern labels.
- **Data**: CSI800 (China); 2007–2020; daily; 16 cross-sectionally ranked features; label = percentile of next-month cross-sectional return.
- **Method**: K linear predictors + router (latent rep + temporal error memory, Gumbel-softmax); optimal transport (Sinkhorn) for balanced assignment. Loss: prediction + λ·CE(OT assignment, router).
- **Evaluation protocol**: SINGLE chronological split with purge gaps (train→2016, val→2018, test 2018-09→2020-06); 5 seeds with std; metrics MSE/MAE/IC/ICIR/AR/AVol/Sharpe/MDD; NO formal significance tests; NO multiple-testing correction; NO costs; turnover unreported.
- **Headline results**: ALSTM+TRA IC 0.059 vs ALSTM 0.053; NOTE ALSTM+TRA Sharpe 0.885 < plain ALSTM 0.897 (rank metric gain ≠ portfolio gain).
- **Conclusions claimed**: First explicit multi-pattern design; lightweight plug-in; OT essential.
- **Stated limitations**: Periodic retraining assumed; online new-pattern setting future work.
- **Ablations**: Router inputs, OT removal (collapse), K sweep (plateau ≥5).

### 3.3 Group C (critical benchmarks/survey)

## LOB benchmark (Artificial Intelligence Review 2024)
- **Full citation**: Matteo Prata, Giuseppe Masi, Leonardo Berti, et al. "LOB-based deep learning models for stock price trend prediction: a benchmark study." Artificial Intelligence Review (2024). arXiv:2308.01915.
- **Research question**: Do LOB deep models reproduce claimed performance (robustness) and transfer to unseen stocks/periods (generalizability)?
- **Data**: FI-2010 (5 Finnish stocks, 10 days) + LOB-2021/2022 (6 US NASDAQ stocks, 2 weeks each); ternary trend label.
- **Method (benchmark design)**: Re-implemented 15 models (only 6/15 SOTA had published code!); released LOBCAST open framework.
- **Evaluation protocol**: single splits; 5 runs mean±std; ~50% of HP-search runs diverged; F1 primary; NO formal tests; costs: only a supplementary trading sim ("profitability far from guaranteed").
- **Headline results**: 10/15 models underperform claimed numbers (TRANSLOB claimed 87.3% → reproduced 59.4%; ATNBoF claimed 67.1% → 40.9±7.7). Universal drop on new data: best model BINCTABL 82.6% → 61.2 (−21.4pp) → 59.2 (−23.4pp higher-vol period). Ensembles don't save it (METALOB 82.2→55.9/53.2).
- **Conclusions claimed**: Low robustness + poor generalizability; extreme sensitivity to HPs/init/stock/regime; practical applicability questionable.
- **Stated limitations**: Non-exhaustive HP search; short new datasets; licensing limits release.
- **Robustness findings**: model RANKING preserved across datasets but absolute F1 falls 10–23pp; per-stock variance large; volatility regime matters.

## GNN survey (ACM Computing Surveys 2024)
- **Full citation**: Manali Patel, Krupa Jariwala, Chiranjoy Chattopadhyay. "A Systematic Review on Graph Neural Network-based Methods for Stock Market Forecasting." ACM Computing Surveys 57(2), Art. 34, 1–38, 2024.
- **Scope**: 2016–2023, ~40 graph-based models tabulated (≈20 classification / 9 regression / 8 recommendation); most common datasets S&P500 (~12), CSI, NASDAQ/NYSE, Nikkei/TOPIX.
- **Framework**: Graph construction (statistical / corporate / textual) + temporal encoder + relational module taxonomy.
- **Key meta-finding FOR US**: the survey does NOT quantify evaluation-practice prevalence — no counts of significance testing, costs, seeds, or code availability across reviewed papers; itself evidence of the field's rigor gap. Qualitative conclusions: multi-relation > single-relation; learned/dynamic graphs are the direction; costs "rarely mentioned"; calls for "more robust evaluation frameworks addressing real-world constraints such as transaction costs".
- **Stated limitations of reviewed methods**: Pearson graphs miss non-linear co-movement; news graphs short-horizon only; deep GCN over-smoothing.

### 4.1 Group B — evaluation-rigor empirical finance (agent 3 extraction)

## GKX (RFS 2020)
- **Full citation**: Shihao Gu, Bryan Kelly, Dacheng Xiu. "Empirical Asset Pricing via Machine Learning." RFS 33(5), 2223–2273, 2020.
- **Research question**: Can ML improve OOS prediction of individual stock returns; which model families/features drive gains?
- **Data**: CRSP US equities, monthly, 1957–2016 (60y); ~30,000 stocks, >6,200/month; 94 characteristics + 74 industry dummies + 8 macro predictors → 920 covariates.
- **Method**: 13 models — penalized/dim-reduced linear (OLS±H, PLS, PCR, ENet, GLM) + RF, GBRT, NN1–NN5 (seed-ensembled).
- **Evaluation protocol**: recursive — 18y train / 12y rolling val / 30y OOS test (1987–2016), annual refits, NO cross-validation (temporal order preserved); R²_oos vs ZERO benchmark (not historical mean — mean inflates ~3pp); modified DM test on cross-sectional avg errors + Newey–West; NO multiple-testing correction; costs NOT modeled (VW noted less cost-sensitive); VW+EW both, top/bottom-1000 size breakouts, micro-cap-free appendix.
- **Headline results**: best NN3 R²_oos=0.40%/month (OLS all-features −3.46%); depth beyond 3 layers does NOT help; VW NN4 long-short Sharpe 1.35 vs OLS-3 0.61; dominant predictors: momentum/liquidity/volatility.
- **Conclusions claimed**: ML improves risk-premium measurement via nonlinear interactions; "shallow beats deep" in low-SNR small-data settings.
- **Stated limitations**: fixed small architectures ("lower bound"); no cost analysis; opacity.
- **Relevance to us**: canonical protocol — recursive scheme, zero-benchmark R², DM tests, VW/large-cap breakouts. Also: their "shallow beats deep" is the RFS-scale analogue of our MLP>GAT.

## HXZ (RFS 2020)
- **Full citation**: Kewei Hou, Chen Xue, Lu Zhang. "Replicating Anomalies." RFS 33(5), 2019–2133, 2020. [Already cited by our paper]
- **Research question**: Do 452 published anomalies replicate under a common protocol mitigating micro-caps and multiple testing?
- **Data**: CRSP+Compustat, 1967–2016 (600 months), 452 anomalies; delisting returns incorporated.
- **Method**: uniform replication — NYSE breakpoints + VW deciles primary; EW/all-exchange + Fama–MacBeth alternatives.
- **Evaluation protocol**: full-sample; HAC t-stats; single hurdle |t|≥1.96, multiple-testing hurdle |t|≥2.78; costs not modeled (micro-cap exclusion motivated by them); micro-caps = 60.7% of names but 3.2% of cap.
- **Headline results**: 65% of 452 fail |t|≥1.96 (VW+NYSE); 82.1% fail at t≥2.78; trading-frictions category 96% fail; even replicated anomalies shrink vs original claims.
- **Conclusions claimed**: most anomalies fail modern standards; micro-cap overweighting + p-hacking; markets more efficient than believed.
- **Relevance to us**: evaluation-design choices flip majorities of findings; EW small-cap-driven gains presumed fragile. (Our 65%/82% citation verified accurate.)

## JKP (JF 2023)
- **Full citation**: Theis I. Jensen, Bryan T. Kelly, Lasse H. Pedersen. "Is There a Replication Crisis in Finance?" JF 78(5), 2465–2518, 2023.
- **Research question**: Do factors replicate under theory-consistent test objects (CAPM alpha), Bayesian joint multiple-testing treatment, and time/geographic OOS?
- **Data**: 153 factors × 93 countries (jkpfactors.com); US from 1926, global from 1986, through 2020.
- **Method**: capped-VW terciles; hierarchical (empirical-Bayes) joint alpha model with zero-alpha prior + shrinkage across correlated factors/themes/regions; frequentist BY-FDR comparison.
- **Evaluation protocol**: full sample + post-publication + 93-country OOS + real-time 1990–2020 exercise; replication rate / posterior FDR / OOS IR / tangency weights.
- **Headline results**: replication-rate sequence 35% (HXZ-style) → 55.6% (their construction) → 82.4% (CAPM alpha) → 75.6% (BY) → 82.4% (Bayesian); posterior FDR 0.1%; factors rescued by Bayes vs frequentist MT earn OOS IR 0.93–1.10 (t>5); "factor zoo" ≈ 13 themes.
- **Conclusions claimed**: NO replication crisis; joint modeling of correlated signals strengthens rather than weakens evidence.
- **Relevance to us**: the MT framework choice can move headline conclusions by ~45pp; hierarchical shrinkage is the principled next step beyond BH/BY when testing correlated contrasts. Our conservative BH/BY choice means our surviving results are robust to the stricter direction.

## KMZ (JF 2024)
- **Full citation**: Bryan T. Kelly, Semyon Malamud, Kangying Zhou. "The Virtue of Complexity in Return Prediction." JF 79(1), 459–503, 2024.
- **Research question**: Should forecasters use P>T overparameterized models? Theory + market-timing empirics.
- **Data**: US market index, monthly, 1926–2020; 15 Goyal–Welch predictors → Random Fourier Features up to P=12,000.
- **Method**: random matrix theory for ridge/ridgeless; RFF two-layer nets; rolling T=12/60/120-month windows; 1,000 random draws.
- **Evaluation protocol**: rolling; R²_oos + timing-strategy Sharpe/alpha/IR t-stats; NO multiple-testing correction; NO costs; single asset.
- **Headline results**: OOS timing Sharpe gain ≈0.47/yr (t≈3) at high complexity WHILE R²_oos stays substantially negative (e.g. −3.8% with Sharpe 0.46) — "use the largest model you can compute" (with shrinkage).
- **Conclusions claimed**: simple models understate predictability; R²_oos incomplete/misleading for economic value.
- **Stated limitations**: linear high-dim theory; single risky asset; Sharpe-criterion-specific shrinkage.
- **Relevance to us**: strongest published counterweight to "simple beats complex" — BUT: (i) it's market timing, not cross-sectional ranking; (ii) it endorses our dual metric reporting (IC + net Sharpe); (iii) our GAT arms (hidden 32–64) are nowhere near the P>T ridge regime, so our claim must stay bounded to "tuned operating points of standard GNN architectures" — which it is (main.tex:365).

## ACM (MS 2023)
- **Full citation**: Doron Avramov, Si Cheng, Lior Metzker. "Machine Learning vs. Economic Restrictions: Evidence from Stock Return Predictability." Management Science 69(5), 2587–2619, 2023.
- **Research question**: Does ML stock-return predictability survive economic restrictions — micro-cap/distress exclusion, limits-to-arbitrage states, turnover and costs?
- **Data**: US stocks, OOS 1987–2017, monthly; CRSP/Compustat + credit ratings.
- **Method**: re-implements GKX-NN3, Chen–Pelger–Zhu GAN, IPCA, conditional autoencoder; common economic-restrictions gauntlet (explicitly NOT a horse race).
- **Evaluation protocol**: original recursive schemes; VW+EW; NW(4) t-stats on FF6-adjusted L-S returns; NO multiple-testing correction; costs CENTRAL — turnover per method + break-even one-way costs vs Novy-Marx–Velikov estimates.
- **Headline results**: EW→VW cuts profits ~48%; VW FF6-adj returns fall 66%/71%/48% ex-microcaps and 78%/69%/94% ex-distressed; NO deep method keeps significant VW FF6-adj return at 5% after excluding distressed firms; profits long-leg-driven and high-VIX-concentrated; turnover 87–168%/month; break-even costs 0.26–0.36% ex-microcaps (marginal). Positives: ML keeps predicting in recent years; economically structured models (IPCA/CA) more robust.
- **Conclusions claimed**: headline DL performance does not clear standard economic restrictions; genuine information exists but is hard to monetize.
- **Relevance to us**: the implementability gauntlet — external validation at RFS/MS scale of exactly our cost-layer thesis; their "economically structured > flexible DL under restrictions" rhymes with our LightGBM/MLP-vs-GAT result.

## 4. Per-paper summaries — Group B (rigor)
[TO FILL: agent 3 & 2 results]

## 5. Comparison matrix (evaluation-protocol forensics)

| Paper | Splits | #Seeds | Sig. tests | Mult-test ctrl | Costs | Graph-off ablation | Leakage/survivorship handling |
|---|---|---|---|---|---|---|---|
| RSR (TOIS'19) | single (test=2017) | 5 runs | none | none | no (dismissed) | YES — no-graph wins several configs | Wikidata snapshot leak; survivorship filter undisclosed |
| STHAN-SR (AAAI'21) | single ×3 markets | 5 runs | Wilcoxon p<.01 vs 2 refs | none | no | YES — LSTM-only SR 0.95 vs 1.42 | static Wikidata through test |
| AD-GAT (AAAI'21) | single (70d test!) | top-5 of 30 by val (selection!) | t-test p<.05 | none | no (no backtest) | no (external baselines only) | full-period universe filter |
| TRA (KDD'21) | single w/ purge gaps | 5 + std | none | none | no | n/a (not graph) | purge gaps — best practice in group A |
| THGNN (CIKM'22) | single (2020 COVID yr) ×2 mkts | 5 runs | none | none | no (dismissed) | no (all ablations retain graph) | trailing-only correlation graph (clean) |
| MASTER (AAAI'24) | single (10q test) | 5 + std | t-test p<.01 | none | no | no | not discussed |
| StockMixer (AAAI'24) | single ×3 datasets | 3 runs | t-test p<.01 | none | no | partial (mixing ablations) | reuses Feng datasets (inherits filters) |
| MDGNN (AAAI'24) | 7 rolling folds ✓ | not reported | none | none | no | no | PIT of ownership/bank data undocumented |
| LOB bench (AIR'24) | single ×3 datasets | 5 + std | none | none | suppl. sim only | n/a | n/a (benchmark) |
| CSUR survey ('24) | n/a | n/a | no prevalence stats | n/a | "rarely mentioned" | n/a | n/a |
| GKX (RFS'20) | recursive 30y OOS, annual refits | seed-ensembled NNs | modified DM + NW | none | not modeled (turnover reported) | n/a | VW/EW + size breakouts; delistings |
| HXZ (RFS'20) | full-sample replication | n/a | HAC t-tests | t≥2.78 hurdle | motivates micro-cap exclusion | n/a | micro-cap design = the paper's core |
| JKP (JF'23) | full + post-pub + 93-country OOS | n/a | CAPM-alpha t + Bayesian | BY-FDR + hierarchical Bayes | not modeled (capped VW for tradability) | n/a | capped VW; delistings |
| KMZ (JF'24) | rolling 12/60/120mo; 1,000 draws | 1,000 random-feature draws | t-stats on Sharpe/alpha/IR | none | not modeled | n/a | single asset (n/a) |
| ACM (MS'23) | recursive, common OOS 1987–2017 | n/a | NW(4) t on FF6 alphas | none | CENTRAL: turnover + break-even costs | n/a | VW/EW, ex-microcap/ex-distressed screens |
| **Ours (ICAIF'26 sub)** | **12-fold expanding WF, 21d purge** | **10 + LOSO 0/10** | **SPA + 20 DM/HLN pre-reg** | **BH q=.05 + BY + pooled-26** | **6-level bps grid + per-arm turnover** | **headline contrast L2−L1, feature-matched** | **T-1 features, PIT news, L1 leak quantified, L8 survivorship quantified 4-way** |

### 5.1 Findings available before Group B fill-in

1. **Splits**: 7/8 Group-A papers use a single chronological split (MDGNN alone: 7 rolling folds). None uses expanding multi-fold walk-forward with purge except TRA's gap trick. Ours: 12 folds + 21d purge.
2. **Seeds**: mode = 5, one paper unreported, StockMixer only 3; AD-GAT reports mean of top-5-of-30 selected on validation — a run-selection protocol that inflates reported test numbers. None does leave-one-seed-out. Ours: 10 seeds, LOSO, dispersion CIs.
3. **Significance/multiplicity**: 4/8 report NO hypothesis test at all; the other 4 use uncorrected t/Wilcoxon vs 1-2 references. **0/10 Group A+C papers apply any multiple-testing correction** (while comparing 8-15 models × 3-6 metrics × 2-3 markets). Ours: SPA selection-adjusted + BH/BY.
4. **Costs**: 0/8 model transaction costs; RSR and THGNN explicitly dismiss them; the field's daily top-k rotation strategies imply near-100% daily turnover — cost-fragile by construction. Ours: 0–30bps grid, per-arm turnover, net-Sharpe crosswalk.
5. **Graph-off evidence already mixed IN the literature**: RSR's own ablation has the no-graph Rank_LSTM beating GCN and both industry-relation RSR variants on NASDAQ; STHAN-SR's hypergraph conv w/o attention (0.93) loses to LSTM-only (0.95). The seminal papers' internal evidence is consistent with our L2−L1<0 finding — it was just never the headline.
6. **StockMixer convergence**: AAAI'24 SOTA is an MLP-family model beating GNN hybrids — independent architectural convergence with our "non-graph MLP strongest" result, but still evaluated under single-split/3-seed/no-cost protocol.
7. **LOB benchmark validates premise**: 10/15 SOTA claims don't reproduce; universal 10–23pp collapse on new data; ~50% HP runs diverge. Field-level instability directly analogous to our C/L5s collapse disclosure and seed-dispersion caveat.
8. **THGNN comparability**: its graph (trailing 20d corr, |ρ|≥0.6) is nearly our α1 (126d Spearman, |ρ|>0.6) — the arm our headline penalizes. Our result directly stresses the exact edge-construction the CIKM'22 SOTA relies on.
9. **The CSUR survey itself lacks rigor meta-statistics** — it catalogues architectures but never counts who tests significance/costs/seeds; our "reporting differences" table fills a hole the survey leaves open.

## 6. Evaluation of our paper

### 6.1 Where our protocol dominates (evidence-backed, from the matrix)

Against **Group A** (the 8 direct competitors), our paper dominates on every forensic axis:

| Axis | Group A (8 papers) | Ours |
|---|---|---|
| Split scheme | 7/8 single chronological split (MDGNN: 7 rolling folds; TRA: purge gaps only) | 12-fold expanding walk-forward + 21d purge |
| Seeds | mode 5, min 3, one unreported; AD-GAT reports mean of top-5-of-30 selected on validation | 10 + LOSO 0/10 + dispersion CIs |
| Significance tests | 4/8 none; 4/8 uncorrected t/Wilcoxon vs 1–2 references | SPA (selection-adjusted) + 20 pre-registered DM/HLN |
| Multiple-testing control | **0/8** | BH q=.05 + BY + pooled-26 sensitivity |
| Transaction costs | **0/8** (RSR & THGNN explicitly dismiss) | 0–30bps grid + per-arm/fold turnover + net Sharpe |
| Pre-registration | 0/8 | frozen hparams md5 + 2 confirmatory families |
| Power analysis | 0/8 (0/15 incl. Groups B–C) | per-contrast MDE |
| Positive control | 0/8 (0/15) | planted-signal pipeline control |
| Survivorship handling | RSR/AD-GAT have UNdisclosed full-sample filters | quantified: 14.8% names / 8.2% stock-days / 8.1% look-ahead / 16.3% two-sided (L8) |

Two Group-A defects our audit newly surfaced (useful for rebuttals):
- **Look-ahead in graph construction**: RSR and STHAN-SR apply a single Wikidata relation snapshot unchanged through their test periods; MDGNN's ownership/bank relations have undocumented point-in-time status. Our news graph is strictly PIT (publication timestamp); our correlation graph trailing-only.
- **Selective run reporting**: AD-GAT trains 30 inits and reports the mean of the top 5 selected on validation — an upward-biased estimate of deployment performance. Our seed-mean estimand + LOSO is the antithesis.

Against **Group B**, we import their standards into GNN territory: DM tests (GKX), multiple-testing hurdle (HXZ), FDR-family control (JKP's frequentist arm), cost/implementability gauntlet (ACM). On axes of pre-registration, MDE power reporting, and a planted-signal positive control, we exceed all five (none of the 15 has any of the three).

**The paper's positioning claim survives the audit.** main.tex:124 claims first S&P 500 stock-ranking study combining ten seeds + 12-fold WF + tuned ladder + two pre-registered families + gross/net evaluation + PIT news + stability-failure disclosure. Nothing in the 15 (or the wider search sweep) contradicts it; the CSUR survey — the field's own systematic review — does not even tabulate significance-testing/cost/seed prevalence, confirming the rigor gap is real and unmeasured before us.

### 6.2 Convergent external evidence for our headline findings

1. **"MLP beats GAT" is corroborated at AAAI'24**: StockMixer — an MLP-family model — beats RSR-I/STHAN-SR/GAT hybrids on their own datasets. Its stated mechanism (market-state compression more robust than full message passing) is architecture-level convergence with our L2−L1<0.
2. **The seminal papers' own ablations already contained the negative signal**: RSR's no-graph Rank_LSTM beats GCN and both industry-relation RSR variants on NASDAQ (0.68 vs 0.24/0.20/0.23); STHAN-SR's hypergraph-conv-without-attention (SR 0.93) loses to LSTM-only (0.95). The "graph helps" headline was config-conditional from the start; our contribution is measuring it under controlled inference.
3. **"Shallow beats deep" at RFS scale**: GKX find depth beyond 3 layers does not help and regularized-simple is a hard baseline — the same phenomenology as our LightGBM/MLP results, in a 60-year panel.
4. **Cost-layer thesis confirmed at MS scale**: ACM show EW→VW cuts DL profits ~48%, and no deep method survives FF6 + distress screens at 5%; turnover 87–168%/month. Our cost crosswalk (news-edge contrast dies in net Sharpe) is the GNN-specific instance of the same law.
5. **Reproducibility collapse is general**: LOB benchmark — 10/15 SOTA claims don't reproduce; 10–23pp universal drop on new data; ~50% of HP runs diverge. Our C/L5s collapse disclosure and seed-dispersion caveats are the honest-reporting analogue.
6. **THGNN (CIKM'22 SOTA) relies on nearly our exact α1 edge type** (trailing corr, |ρ|≥0.6): our headline stresses the edge construction the SOTA literature actually uses — not a strawman.

### 6.3 Where the literature challenges us (residual exposures, ranked)

1. **Sample scale (biggest exposure)**: GKX = 30-year OOS × >6,200 stocks; JKP = 93 countries; ACM = 31 years. Ours = 12 test quarters (2023Q1–2025Q4), 501 stocks, one market, one horizon. Disclosed (L7 + regime-concentration L2), and partially inherent: a tuned-ladder × 10-seed × 12-fold design (2,160 cells) cannot also be 30 years at equal compute. But a finance-literate reviewer will press. Defense: our estimand is architecture attribution under controlled inference, not risk-premium measurement; the design trades breadth for internal validity.
2. **EW-only portfolio layer**: HXZ/ACM show EW inflates and VW deflates DL profits. Our decile portfolios are equal-weight (disclosed in L5). Partial insulation: S&P 500-only universe contains no micro-caps (smallest look-ahead addition $6.6B — main.tex L8), so HXZ's micro-cap mechanism has limited bite. Still, a VW sensitivity would materially strengthen the cost layer. **Recommended as reviewer-response ammunition or camera-ready addition.**
3. **KMZ complexity counterpoint**: P>T ridge-regime models with heavy shrinkage beat simple models in market timing even at negative R²_oos. Our GAT arms (hidden 32–64, ≤2 layers) are nowhere near that regime, so our claims must remain bounded to "tuned operating points of standard GNN architectures" — which the Discussion already does (main.tex:365 "not graph architectures in general"). Citing KMZ would pre-empt the sophisticated version of reviewer I-02.
4. **JKP's Bayesian alternative**: hierarchical shrinkage over correlated contrasts is more powerful than BH/BY and can *rescue* borderline findings. Our choice is the conservative direction (BY keeps the headline; C L1−L0 drops) — defensible, but "why not joint Bayesian?" is an anticipatable question. Answer: pre-registered frequentist families; Bayesian joint model noted as future work.
5. **Zero-benchmark R²_oos absent**: GKX's canonical metric is R²_oos vs zero. We report IC only (+ net Sharpe). Not a flaw for a ranking estimand — but one sentence mapping IC to the GKX metric convention would ease finance readers in.

### 6.4 Actionable recommendations (H博士 decision required; anon build is at exactly 8pp)

Priority-ordered; each costs ~2–4 lines (text + bib) that must be found elsewhere:
- **R1 (recommended)**: Cite GKX in the §2 methodology-oriented paragraph (currently HXZ/LdP/Hansen/Politis/NW — GKX is the missing canonical anchor, and its "shallow beats deep" directly supports our result). 1 clause + 1 bib entry.
- **R2 (recommended)**: Cite ACM (MS'23) in the cost-layer paragraph (§2 or §5.4): "consistent with evidence that ML stock-return profits concentrate in hard-to-arbitrage segments and shrink under costs". 1 clause + 1 bib entry.
- **R3 (optional)**: Cite KMZ in Discussion as the complexity counterpoint our bounded claim does not contradict. Pre-empts sophisticated I-02 escalation.
- **R4 (optional)**: Cite THGNN at the α1 graph definition — documents that our stressed edge type is the one recent SOTA uses.
- **R5 (no page cost; rebuttal file)**: Keep this audit's Group-A forensic table as reviewer-response material: single-split prevalence 7/8, MT correction 0/8, costs 0/8, AD-GAT selection reporting, RSR/STHAN-SR Wikidata look-ahead. Directly substantiates our related-work sentence if challenged.
- **R6 (future work / camera-ready)**: VW-decile sensitivity of the net-Sharpe layer (recomputation on existing outputs; no new training).

### 6.5 Verdict

**Our paper's evaluation protocol is the strictest among all 15 benchmarked papers.** It strictly dominates the 8 direct GNN/DL competitors on every audited axis; it transfers Group-B (RFS/JF/MS) inferential standards into the GNN stock-ranking niche, and on pre-registration + power reporting + positive control it exceeds even Group B. Its headline negative finding is independently corroborated by four strands (StockMixer's architecture-level convergence, the seminal papers' own ablations, GKX's shallow-beats-deep, ACM's economic-restrictions gauntlet). The genre — controlled negative/conditional results with failure-mode disclosure — is exactly what the field's own critical literature (LOB benchmark, CSUR survey) says is missing.

Residual exposures are scale (5y/1 market/1 horizon), the EW-only portfolio layer, and the absent complexity-regime coverage — all disclosed as limitations, none undermining the bounded claims. Publication risk is not the protocol; it is a reviewer preferring breadth over internal validity. R1/R2 citations + R5 rebuttal table are the cheapest hedges.
