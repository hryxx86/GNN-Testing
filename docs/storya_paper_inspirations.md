# Story A Paper — Inspirations from 3 Top-Tier Papers

> **Purpose**: Writing-craft reference for drafting Story A ("When Do Graph Neural Networks Help in Cross-Sectional Stock Ranking?"), target venue ICAIF 2026 (ACM SIGCONF, 8-10 pages).
> **Date drafted**: 2026-05-28 by Claude Opus 4.7.
> **Sources read for this document**:
> - Feng, Chen, He, Hua, Chua (2019) "Temporal Relational Ranking for Stock Prediction." *ACM TOIS* 37(2), Article 27. Read in full via arXiv:1809.09441v2 PDF (20 pages).
> - Sawhney, Agarwal, Wadhwa, Derr, Shah (2021) "Stock Selection via Spatiotemporal Hypergraph Attention Network: A Learning to Rank Approach." *AAAI 2021* (8 pp.). Read in full from the AAAI proceedings PDF downloaded on 2026-05-28. Citations read `(Sawhney 2021, §X)` / `(Sawhney 2021, Table Y)` / `(Sawhney 2021, p.Z ¶N)`. **Cui, Li, Du, Zhang, Nie, Wang, Yin (2021)** "Temporal-Relational Hypergraph Tri-Attention Networks for Stock Trend Prediction" (arXiv:2107.14033, 14 pp.) is retained as a secondary reference for a small number of stylistic devices STHAN-SR does not display (e.g., the boxed Algorithm-1 pseudocode and the t-SNE visualisation table). Those citations read `(Cui 2021, §X)`.
> - Hou, Xue, Zhang (2020) "Replicating Anomalies." *Review of Financial Studies* 33(5), 2019-2133. The full PDF was on a blocked domain; this document uses **the verbatim published abstract**, the bottom-line concluding sentence, and the headline statistics (1.96 / 2.78 thresholds, 65% / 82% / 96% / 452 numbers) as captured via multiple search engine excerpts of the published version. Citations read `(HXZ 2020, abstract)` or `(HXZ 2020, p.2020 ¶1 [search excerpt])` where I had to rely on third-party summaries; I clearly mark these.

---

## §0 Executive Summary

### The 3 papers and what Story A should borrow from each

**Feng et al. 2019 (TOIS) — the architectural twin.** Same task formulation as Story A: cross-sectional ranking of stocks with relational GNN + pairwise loss + NYSE/NASDAQ. We borrow Feng's *problem-formulation paragraph* (the "intuitive example" of MSE-vs-profit mismatch in Table 1), his *RQ-driven Experiments section* structure, and his *honest mid-paper concessions* (e.g., "Rank_LSTM fails to consistently beat SFM and LSTM regarding all evaluation measures" — Feng 2019, §5.2 bullet 2). Avoid: Feng's overclaim-friendly abstract ("98% and 71% return ratio") which is exactly the rhetorical posture Story A repudiates.

**Sawhney 2021 STHAN-SR (AAAI) — the contemporary tone.** Defines the 2020-2021 GNN-finance AAAI/ICAIF idiom: hypergraph + attention + learning-to-rank loss + Sharpe / IRR / NDCG reporting on NASDAQ + NYSE + TSE. We borrow Sawhney's title formula *"Stock Selection via [Method]: A Learning to Rank Approach"* (Sawhney 2021, title), his **3-section IMRAD-with-related-work-first** layout (Introduction → Related Work → Methodology → Experimental Setup → Results and Analysis → Conclusion; Sawhney 2021 §§1-6), his **pairwise + pointwise ranking-aware loss** (Sawhney 2021, Eq.10), his **mean of 5 individual runs** reporting convention with Wilcoxon signed-rank test at p<0.01 marked by `*` and `†` per baseline (Sawhney 2021, Tables 2-3 captions), and his **5-subsection Results layout** with a dedicated §5.3 "On the Effectiveness of Hypergraphs" probe (Sawhney 2021, §5.3). Avoid: Sawhney's omission of any §Limitations section and the abstract's "significantly outperforms state-of-the-art" overclaim register.

**Hou-Xue-Zhang 2020 — the rigor template.** Provides the language register Story A needs for N1 (honest IC) and N3 (failure-mode catalog). The published abstract's structure — "Most anomalies fail to hold up to currently acceptable standards… 65% of the 452 anomalies… cannot clear… Imposing the higher multiple test hurdle… raises the failure rate to 82%…" (HXZ 2020, abstract) — is the *exact* rhetorical shape Story A's abstract should mimic, swapping anomalies→GNN configurations.

### 5 most important takeaways

1. **The honest-replication abstract is a tetrad: claim → method → headline failure number → multi-testing escalation.** HXZ 2020 deploys this in 4 sentences. Story A's abstract must do the same.
2. **Topic-driven Experiments organisation beats Methods-Results-Discussion for GNN-finance papers.** Feng 2019 uses RQ1/RQ2/RQ3 as §5 subsections (Feng 2019, §5 ¶3) and Sawhney 2021 uses 5 topic-based Results subsections (§5.1 Profitability Comparison, §5.2 Model Component Ablation, §5.3 On the Effectiveness of Hypergraphs, §5.4 Visualizing Hawkes Attention, §5.5 Parameter Analysis). ICAIF audience expects this granular Results layout.
3. **Mid-paper concession buys late-paper credibility.** Feng concedes Rank_LSTM "fails to consistently beat" baselines on MRR (Feng 2019, §5.2) — and the paper is still TOIS-accepted. Story A's N1 honest-IC pillar should follow this pattern, not bury it.
4. **Quantitative reporting style for GNN-finance is "value with bold + italics + significance markers", not "value (SE = …)".** Feng 2019 uses `3.79e-4±1.11e-6` mean ± std (Feng 2019, Table 5); Sawhney 2021 reports bare point estimates as "mean of 5 individual runs" with bold/italics for best/second-best and `*` / `†` for Wilcoxon p<0.01 vs iRDPG / RSR-I (Sawhney 2021, Table 2 caption + Table 3 caption). Story A should adopt the Feng-style mean ± std for IC but the Sawhney-style explicit significance-marker convention.
5. **For a "when does X help?" paper, the §Limitations must read like HXZ's caveats list — present-tense, declarative, not apologetic.** HXZ 2020: "Most anomalies fail to hold up to currently acceptable standards" (abstract) — no hedging. Story A's L1/L2/L3/L6 should adopt the same matter-of-fact register.

---

## §1 Feng et al. 2019 TOIS — Architectural Twin

### §1.A Architecture

#### Section structure (full list with page allocation)

Title page + abstract: **p.1** (the abstract is a single 13-sentence paragraph spanning ~25 lines).

- **1. INTRODUCTION** — pp.1–4 (≈3 full pages). Contains a 4-paragraph build-up + the celebrated Table 1 intuitive example + Fig.1 architecture preview + a 3-bullet contributions list + 1-sentence roadmap.
- **2. PRELIMINARIES** — pp.4–6 (≈2 pages). Has **2.1 Long Short-Term Memory** (p.4, includes Eq. 1 with 6 sub-equations for LSTM gates) and **2.2 Graph-based Learning** (pp.5–6, has nested **2.2.1 Graph Convolutional Networks** sub-subsection).
- **3. RELATIONAL STOCK RANKING** — pp.6–10 (≈4.5 pages). Has **3.1 Framework** (with three bold inline subheadings rendered as bold paragraph leads: **Sequential Embedding Layer**, **Relational Embedding Layer**, **Prediction Layer**) and **3.2 Temporal Graph Convolution** (with bold inline leads: **a) Uniform Embedding Propagation**, **b) Weighted Embedding Propagation**, **c) Time-aware Embedding Propagation**, and bullet leads: **Explicit Modeling**, **Implicit Modeling**) plus **3.2.1 Connection with Graph-based Learning**.
- **4. DATA COLLECTION** — pp.10–12. Has **4.1 Sequential Data**, **4.2 Stock Relation Data** with **4.2.1 Sector-Industry relations** and **4.2.2 Wiki Company-based Relations**.
- **5. EXPERIMENT** — pp.12–18 (≈6 pages — the dominant section). Has **5.1 Experimental Setting** (with **5.1.1 Evaluation Protocols**, **5.1.2 Methods**, **5.1.3 Parameter Settings**), **5.2 Study of Stock Ranking Formulation (RQ1)**, **5.3 Impact of Stock Relations (RQ2)** with bold leads **Effect of Industry Relations**, **Effect of Wiki Relations**, **Sector-wise Performance**, **Importance of Each Type of Wiki Relation**, **Brief Conclusion**, and **5.4 Study on Back-testing Strategies (RQ3)**.
- **6. RELATED WORK** — pp.19–20. Has **6.1 Stock Prediction**, **6.2 Graph-based Learning**, **6.3 Knowledge Graph Embedding**. Placed **after** experiments — this is unusual and a Feng signature move.
- **7. CONCLUSIONS** — p.20 (2 short paragraphs).

**Total: 20 pages, single-column TOIS format.** For ICAIF (8 pp. double-column), this is ≈ a 2:1 scale-down — Story A cannot afford a 3-page introduction.

#### Subsection heading style — declarative vs noun-phrase vs question

Feng uses **declarative-noun-phrase hybrid**:

- Noun-phrase: "Temporal Graph Convolution" (Feng 2019, §3.2); "Stock Relation Data" (§4.2).
- Result-section RQ titles use **mini-declarative + parenthetical RQ tag**: "Study of Stock Ranking Formulation (RQ1)" (§5.2); "Impact of Stock Relations (RQ2)" (§5.3); "Study on Back-testing Strategies (RQ3)" (§5.4).
- He **never** uses interrogative section titles; the RQs themselves are in the §5 preamble (Feng 2019, §5 ¶3): "How is the utility of formulating the stock prediction as a ranking task?" / "Do stock relations enhance the neural network-based solution for stock prediction?" / "How does our proposed RSR solution perform under different back-testing strategies?"

#### Figure / Table count

- **Main paper, total**: 8 figures + 10 tables.
- Figures: Fig.1 schematic architecture (p.3); Fig.2 two line-plots of stock-price-history pairs (p.8); Fig.3 sector-industry tree schematic (p.12); Fig.4 Wikidata first-/second-order relation schematic (p.12); Fig.5 dual-panel IRR cumulative return line plot (p.15); Fig.6 dual-panel IRR with relational baselines (p.16); Fig.7 dual-panel IRR with Wiki relations (p.17); Fig.8 four-panel Top1/Top5/Top10 back-test (p.18).
- Tables: T1 intuitive MSE-vs-profit example (p.2); T2 notation legend (p.6); T3 dataset stats (p.11); T4 relation-type stats (p.12); T5 main MSE/MRR/IRR results (p.14); T6 industry-relation methods (p.15); T7 Wiki-relation methods (p.17); T8 sector-wise IRR breakdown (p.17); T9 relative-performance-decrease ablation (p.17); T10 vs market indices (p.19).
- **No supplementary material is referenced**; everything is in-paper.
- **Type mix**: 1 schematic (Fig.1), 2 illustrative line-plots-of-price (Fig.2), 2 graph schematics (Fig.3, Fig.4), 4 results line-plots (Figs 5–8), 0 bar/scatter/heatmap/forest.

#### Equation density and numbering

- 13 numbered equations (Eq.1 LSTM compound; Eqs.2–6 GCN derivation in Preliminaries; Eq.7 LSTM-as-function; Eq.8 the pointwise+pairwise loss; Eqs.9–13 the three TGC propagation variants).
- All equations numbered with single integers; no equation labels.
- Equations are concentrated in §2 (preliminaries) and §3.2 (TGC); §5 has none.
- ≈ 4 pages contain math (§2 entirely, §3.2 entirely, §3.1 lightly).

#### Related Work organisation

By topic, in **3 subsections**:
- **6.1 Stock Prediction** — itself organised by sub-paradigm (price regression then trend classification). Opens with Bao et al.'s wavelet+SAE+LSTM (Feng 2019, §6.1 ¶1).
- **6.2 Graph-based Learning** — by method family (graph regularization vs graph convolution).
- **6.3 Knowledge Graph Embedding** — single paragraph on TransE-line work.

**Crucially: Related Work is placed after the Experiments section.** Feng 2019 uses §6 as a contextualising coda, not a setup. This works because §1 introduction already does substantial differentiation work.

#### Conclusion / Discussion structure

- 2 paragraphs total. Para 1 = recap (1 sentence per: problem formulation, architecture, experiment validation). Para 2 = future work (4 future directions listed conjunctively: top-k weighting, risk management, long/short, alt data).
- **No separate Discussion section, no Limitations section.** This is the biggest weakness Story A must not inherit.

### §1.B Wording / Phrasing

#### Abstract — verbatim with rhetorical-move annotation

> "Stock prediction aims to predict the future trends of a stock in order to help investors to make good investment decisions. [**move 1: domain hook + investor framing**] Traditional solutions for stock prediction are based on time-series models. With the recent success of deep neural networks in modeling sequential data, deep learning has become a promising choice for stock prediction. [**move 2: prior-paradigm setup**] However, most existing deep learning solutions are not optimized towards the target of investment, i.e., selecting the best stock with the highest expected revenue. Specifically, they typically formulate stock prediction as a classification (to predict stock trend) or a regression problem (to predict stock price). More importantly, they largely treat the stocks as independent of each other. The valuable signal in the rich relations between stocks (or companies), such as two stocks are in the same sector and two companies have a supplier-customer relation, is not considered. [**move 3: gap-identification, 4 sentences — note the double-but structure ("However…", "More importantly,…") used to layer two distinct gaps**] In this work, we contribute a new deep learning solution, named *Relational Stock Ranking* (RSR), for stock prediction. Our RSR method advances existing solutions in two major aspects: 1) tailoring the deep learning models for stock ranking, and 2) capturing the stock relations in a time-sensitive manner. The key novelty of our work is the proposal of a new component in neural network modeling, named *Temporal Graph Convolution*, which jointly models the temporal evolution and relation network of stocks. [**move 4: method announcement with 2-aspect enumeration + named-component highlight**] To validate our method, we perform back-testing on the historical data of two stock markets, NYSE and NASDAQ. Extensive experiments demonstrate the superiority of our RSR method. It outperforms state-of-the-art stock prediction solutions achieving an average return ratio of 98% and 71% on NYSE and NASDAQ, respectively." (Feng 2019, abstract)

**Rhetorical-move grammar**: Hook (1 sentence) → Prior paradigm (2 sentences) → Gap (4 sentences, layered) → Method (3 sentences) → Result (3 sentences with headline numbers). **5 moves, 13 sentences, ~190 words.** Story A target: 4 moves, 8 sentences, ~150 words (ICAIF compression).

#### Key terminology for cross-sectional ranking

Feng's preferred terms (verbatim usages):
- "stock ranking" (Feng 2019, §1 ¶5, §3 opening, §5.2 title) — primary noun.
- "ranking function $\hat{r}^{t+1} = f(\mathcal{X}^t)$" (§3, p.6 ¶3) — formal.
- "the future trend and price of a stock" (§1 ¶2) — colloquial.
- "stock selection" (§1 ¶3) — used in problem-motivation only.
- "selecting the best stock with the highest expected revenue" (abstract) — investor framing.
- Crucially **avoids** "portfolio formation" and "predict relative returns"; those come from finance literature.

#### Hedging vocabulary — count + quotes

Hedging in Feng 2019 is light and concentrated in §5 results discussion:

- *"could"* — 8+ uses in §5; e.g., "tuning the hyperparameters regarding IRR… could achieve better performance" (§5.2 last bullet); "we speculate the reason is that…" (§5.3, p.16 ¶2).
- *"may"* — used for tentative claims: "the strength of influence between two given stocks may vary quickly" (§1 ¶5); "selecting only one stock from more than 1,000 is a highly risk operation" (§5.2 final ¶).
- *"we argue"* — 3 uses, e.g., "we argue that such prediction methods are suboptimal to guide stock selection" (§1 ¶3).
- *"demonstrate"* — strong-claim verb, used 5+ times: "Extensive experiments demonstrate the superiority of our RSR method" (abstract), "Experimental results on NASDAQ and NYSE demonstrate the effectiveness" (Conclusion).
- *"verifies"* — used to claim ablation evidence: "It verifies the advantage of the stock ranking solutions" (§5.2 bullet 1).
- *"suggests"* / *"indicates"* — soft claims: "The reason could be… which would lead to a tradeoff" (§5.2); "This result indicates the potential difference between the validation and testing" (§5.3 bullet 2).

#### Numeric reporting conventions

- **Mean ± std format with scientific notation**: "Rank_LSTM | 3.79e-4±1.11e-6 | 4.17e-2±7.50e-3 | 0.68±0.60" (Feng 2019, Table 5) — note IRR has very high std (0.60 on a 0.68 mean → 88% CV) reported openly, not hidden.
- **No standard error, no confidence interval.** No p-values for model comparisons.
- **Bold for best, underline used sparingly**: "the best result… is indicated in bold, and the second best one is underlined" (not stated in Feng explicitly, but Table 5 follows this convention).
- **Percentage growth reported plainly**: "more than 115% improvements in return ratio" (Feng 2019, §1 ¶6); "Considering industry relations is more beneficial to stock ranking on NYSE as compared to NASDAQ" (§5.3 bullet 1).

#### How negative / null results are phrased — 5+ alternatives

These are gold for Story A's N1/N3 pillars. Feng's failure-acknowledgment vocabulary:

1. *"X fails to consistently beat Y regarding all evaluation measures"* — "However, Rank_LSTM fails to consistently beat SFM and LSTM regarding all evaluation measures, its performance on NYSE w.r.t. MRR is worse than SFM" (Feng 2019, §5.2 bullet 2).
2. *"X is unexpectedly bad"* — "The performance of LSTM on the NYSE market w.r.t. IRR is unexpectedly bad" (§5.3 ¶2 above bullets — note the candid "unexpectedly").
3. *"X fails to achieve expected performance"* — "RSR_I (Figure 8(a)) fails to achieve expected performance with different back-testing strategies under the NASDAQ-Industry setting" (§5.4 bullet 1).
4. *"the performance regarding different evaluation measures is inconsistent"* — "Again, the performance regarding different evaluation measures is inconsistent. We speculate the reason is that we tune the hyperparameters regarding IRR" (§5.3 ¶ after Table 6) — note the **"Again,"** turn-of-phrase signalling recurring weakness.
5. *"This result further indicates the less effectiveness of…"* — "It further indicates the less effectiveness of industry relations on NASDAQ" (§5.4 bullet 1).
6. *"the curves are volatile, which indicates that selecting only one stock from more than 1,000 is a highly risk operation"* — (§5.2 final ¶) — converting noise into a stated finding.
7. *"the result also suggests the worth of introducing risk-oriented criteria into stock ranking tasks in the future"* — (§5.2 final ¶) — null result → future-work pivot.

#### Limitations statement style

**Feng 2019 has no §Limitations section.** The closest equivalents are scattered concessions:

- Future-work-as-limitation: "we will explore the potential of emphasizing top-ranked entities with more advanced learning-to-rank techniques" (§7 Conclusion ¶2) → implicit admission top-K weighting is currently missing.
- Sector-wise warning: "we only show the performance on sectors with the top-5 most stocks" (§5.3 ¶ above Table 8) → tacit acknowledgement of selection-on-coverage.
- Volatility caveat: "The performance w.r.t. IRR varies a lot under different runs of a method. It is reasonable since the absolute value of daily return ratio varies from 0 to 0.98 in our dataset" (§5.3 bullet 1) — explains variance without calling it limitation.

**For Story A**: Feng's failure to consolidate these into a §Limitations is exactly the bug ICAIF reviewers will spot if Story A repeats it.

#### Citation style

Author-Year is **not** used. Feng 2019 uses **numeric bracketed citations** matching ACM TOIS house style: "[42]" for SFM (Feng 2019, §1 ¶4); "[4, 17, 42]" for trend-classification line (§6.1 ¶2). Citation density: ≈ 1 citation per 3 lines in §6, ≈ 1 per 8 lines in §3, ≈ 1 per 6 lines in §1.

### §1.C Narrative / Storytelling

#### Core conflict in 1–2 sentences (authors' framing)

> "most existing deep learning solutions are not optimized towards the target of investment, i.e., selecting the best stock with the highest expected revenue… they largely treat the stocks as independent of each other." (Feng 2019, abstract)

Restated: "**The Method-Target Mismatch + The Independence Assumption** — DL stock predictors optimise for MSE/classification accuracy while investors need top-K ranking; and they treat stocks as i.i.d. when in fact sector and supply-chain links carry signal."

#### Hook strategy — first paragraph of Introduction

Feng 2019 §1 ¶1 opens: "According to the statistics reported by the World Bank in 2017, the overall capitalization of stock markets worldwide has exceeded 64 trillion U.S. dollars. With the continual increase in market capitalization, trading stocks has become an attractive investment instrument for many investors. However, whether an investor could earn or lose money depends heavily on whether he/she can make the right stock selection." (Feng 2019, §1 ¶1)

**Move analysis**: number-anchored macro hook (64 trillion) → financial-instrument framing → investor-pain framing. **3 sentences, 65 words.** This is the cleanest archetype Story A can mimic for an opening — replace "$64T market cap" with a parallel macro statistic.

#### Result layering — §5 outline

Feng builds **bottom-up, not top-down**:
- §5.1 first defines protocols and methods (3 sub-subsections).
- §5.2 RQ1: does ranking formulation help? — yes, but with the candid "fails to consistently beat" concession.
- §5.3 RQ2: do stock relations help? — yes for NYSE, qualifiedly for NASDAQ. Multi-part with **4 inline subheadings**.
- §5.4 RQ3: do back-testing strategies matter? — yes, with the "NASDAQ-Industry" failure case admitted.
- **Concluding mini-paragraph at end of §5**: "**Brief Conclusion**: a) Considering stock relations is helpful for stock ranking, especially on the stable markets (e.g., NYSE). 2) The proposed TGC is a promising solution for encoding stock relations. 3) It is important to consider appropriate relations suitable for the target market, for example, encoding industry relations on NASDAQ is a suboptimal choice." (Feng 2019, §5.3 final mini-conclusion). Note the conditional finding "appropriate relations suitable for the target market" — this is precisely the language Story A's N2 conditional-findings pillar needs.

#### Negative-result framing as contribution

Feng converts every null into a **conditional finding** ("especially on stable markets", "appropriate relations suitable for the target market"). The pattern: "X helps **conditionally on** Y" rather than "X helps" or "X doesn't help".

#### Rigour without defensiveness

Feng uses three rhetorical devices:
- **Open admission of noise**: "The performance w.r.t. IRR varies a lot under different runs" (§5.3 bullet 1) — stated as factual observation, not defended.
- **Speculation flagged honestly**: "We speculate the reason is that we tune the hyperparameters regarding IRR" (§5.3 ¶ after Table 6) — uses "speculate", not "we found".
- **Direct attribution of mechanism**: "The reason could be that the industry relations reflect more of long-term correlations between stocks, since NASDAQ is considered as a much more volatile market" (§5.3 bullet 1) — mechanism stated as hypothesis with marker "could".

#### Discussion lift — specific → generalisable

The §5 "Brief Conclusion" generalises: "It is important to consider appropriate relations suitable for the target market" (Feng 2019, §5.3 final). This lifts a NASDAQ-vs-NYSE finding into a methodology principle. Story A's discussion should attempt similar lifts (e.g., "the value of multi-edge GNN ensembles is conditional on edge-noise correlation").

### §1.D Story A Borrowing Checklist

Each item links to a Story A narrative pillar (N1-N4) or Limitations item (L1, L2, L3, L6).

1. **[N1]** Adopt Feng's **"fails to consistently beat … regarding all evaluation measures"** template (Feng 2019, §5.2 bullet 2) for Story A's MLP_price-vs-SAGE-Mean_price IC=+0.0374 / IC=+0.0269 reversal at 21d. Phrasing: *"SAGE-Mean fails to consistently beat MLP across feature universes and horizons; at 21d on Universe B, MLP_price IC = 0.0374 exceeds SAGE-Mean_price IC = 0.0269."*

2. **[N1]** Adopt Feng's **"the performance regarding different evaluation measures is inconsistent. We speculate the reason is that…"** (Feng 2019, §5.3) for SPA-vs-bootstrap-CI divergence (7/8 cells positive CI but SPA p_consistent ≥ 0.136). Story A version: *"Bootstrap CI inclusion (7/8 positive) and SPA-consistent rejection (0/8) give inconsistent verdicts; we attribute this to SPA's higher power requirement for multiple-horizon joint testing."*

3. **[N2]** Use Feng's **conditional-finding pattern** "X helps **especially on** Y" (Feng 2019, §5.3 "**Brief Conclusion**" 1)) for Story A's news-as-feature finding. Phrasing: *"News-as-feature signal helps **only when** horizon ≤ 5d; at 21d the ΔIC penalty reaches -0.045."*

4. **[N3]** Steal the **"unexpectedly bad"** candid-admission move (Feng 2019, §5.3 ¶ above bullets) for Fold-4 Q2-2025. Phrasing: *"GAT 21d's Fold-4 IC is unexpectedly bad (drop of 38-72% under LOFO-4), which we trace to regime dislocation in Q2-2025."* This earns credibility for L6.

5. **[N4]** Mirror Feng's **3-RQ Experiments scaffolding** but expand to 4 RQs matching Story A's 4-pillar narrative. RQ1: do GNNs beat MLP/LightGBM on average? (N1) RQ2: when do they conditionally win? (N2) RQ3: which failure modes are systematic? (N3) RQ4: what does the methodology lattice tell us? (N4). This is the spine of §4 Results.

6. **[N4]** Adopt Feng's **bottom-up §5 organisation**: definition → RQ1 → RQ2 → RQ3 → mini-Brief-Conclusion. Story A's §4 should end with a "Findings Summary" mini-paragraph that lifts each RQ into one declarative sentence.

7. **[Methods]** Borrow Feng's **Table 1 "intuitive example"** device (Feng 2019, p.2, MSE-vs-profit mismatch). Story A could open §3 Methods with a 2x2 table showing a single seed's IC vs the 10-seed mean (e.g., the GAT 21d seed=42 IC=0.044 vs 10-seed mean=0.032 gap) — a parallel "look, this is why we need the apparatus we're about to present" hook.

8. **[L1, L2, L3, L6 — DO-NOT-INHERIT]** Feng has **no §Limitations**. Story A must consolidate L1/L2/L3/L6 into a dedicated §Limitations subsection or paragraph. Adapting to ICAIF 8-page reality: a 5-bullet §6 Limitations of ~150 words.

9. **[Style]** Feng's "Brief Conclusion" mini-paragraphs (§5.3, §5.4) numbered "1) … 2) … 3) …" — adopt this device at end of every Results subsection to telegraph findings to a skimming reviewer.

10. **[Conflict with ICAIF]** Feng allocates 4.5 pages to §3 Methods. Story A cannot — at 8 ICAIF pages, §3 should be ~1.5 pages. Adaptation: defer all preliminary GNN math (the GCN derivation Feng uses in §2.2.1) to a 1-paragraph "We assume reader familiarity with GAT/GraphSAGE [cites]" sentence.

11. **[N1 / N4 — DO-NOT-INHERIT]** Feng's abstract overclaims with "98% and 71% return ratio" headlines (Feng 2019, abstract final sentence). Story A's pillar N1 is the **anti-overclaim** stance — the abstract must lead with "no configuration rejects SPA at 5%" not with a best-cell IC.

---

## §2 Sawhney et al. 2021 STHAN-SR (AAAI) — Contemporary GNN-Finance Ranking

> **Source**: Sawhney, Agarwal, Wadhwa, Derr, Shah (2021), AAAI 2021 official proceedings PDF (8 pp., pp.497-504), read in full. All quotes are verbatim from the PDF. A small number of devices STHAN-SR does not display (boxed Algorithm-1 pseudocode; explicit RQ enumeration) are sourced from Cui 2021 HGTAN as a secondary specimen and clearly tagged.

### §2.A Architecture

#### Section structure (Sawhney 2021, 8 pp. AAAI double-column)

- **Abstract** — 1 paragraph, 11 sentences, ≈ 175 words (Sawhney 2021, p.497 left column).
- **1. Introduction** — pp.497–498 (≈ 1.5 col-pages, spans the bottom of p.497 right column through the top of p.498 left column). Includes Figure 1 (toy example: stocks S1-S4 with two regression methods R1/R2 and two classification methods C1/C2 showing that "more accurate stock prediction… may not always be more profitable than less accurate methods"; Sawhney 2021, Figure 1) and Figure 2 (hypergraph schematic with Healthcare/Travel industry hyperedges and a Berkshire-IBM-US Bancorp ownership hyperedge; Sawhney 2021, Figure 2). Ends with **three bullet contributions**.
- **2. Related Work** — p.498 right column (≈ 0.6 col-page). **Three boldface paragraph-leads, no subsection numbers**: **Conventional Methods in Finance**, **Contemporary Methods**, **Hypergraph Representation Learning** (Sawhney 2021, §2). Placed **before** the methodology, like Cui 2021 and unlike Feng 2019.
- **3. Methodology** — pp.499–501 (≈ 2.5 col-pages, the largest pre-experiment block). Four subsections: **3.1 Problem Formulation** (formal ranking-task setup; introduces $r_i^t$ return ratio); **3.2 Temporal Evolution of Stock Prices** (Feature Extraction → LSTM → Temporal Attention → Hawkes Attention bold-paragraph leads); **3.3 Spatial Stock Hypergraph Feature Extraction** (Stock Hypergraph Construction → Industry Hyperedges + Wiki Corporate Hyperedges → Hypergraph Convolution → Hypergraph Attention); **3.4 Learning to Rank and Network Optimization** (Sawhney 2021, §§3.1-3.4). Figure 3 (overall pipeline) is at the top of p.499; Figure 4 (hypergraph convolutions and attention) is at the top of p.500.
- **4. Experimental Setup** — p.501 right column + top of p.502 (≈ 0.8 col-page). **4.1 Datasets**, **4.2 Training Setup**, **4.3 Evaluation Metrics** (with **Returns** and **Ranking** boldface paragraph-leads naming Sharpe / IRR / NDCG@k respectively). Table 1 (dataset chronological splits, 3 markets × 6 rows) sits at the top of p.501.
- **5. Results and Analysis** — pp.501–503 (≈ 1.7 col-pages, the largest experiments block). **Five subsections**: **5.1 Profitability Comparison with Baselines** (with Table 2, the 10-row × 6-column profitability comparison), **5.2 Model Component Ablation Study** (with Table 3, 7-row × 9-column ablation), **5.3 On the Effectiveness of Hypergraphs** (with boldface paragraph-leads **Effect of injecting domain knowledge via stock relations** and **Hypergraph v.s. Graph for representing stock relations**; Figure 5(a) and Figure 5(b) decomposition curves), **5.4 Visualizing Hawkes Attention** (Figure 6 day-level attention heat-strip with predicted vs actual return ratios), **5.5 Parameter Analysis: Probing Sensitivity** (with boldface paragraph-leads **Lookback window length T** and **Number of selected top stocks k**; Figure 7 sensitivity curves).
- **6. Conclusion and Future Work** — p.503 right column (1 paragraph, ≈ 130 words). Single paragraph covering recap + future directions ("In future, we aim to design time-evolving hypergraphs to capture dynamic market correlations and incorporate additional data sources such as online news and social media"; Sawhney 2021, §6).
- **References** — p.504 (1 col-page).

**Total: 8 pp. AAAI double-column.** Methodology + experiments together occupy ≈ 4.2 of the 8 pages, i.e., ≈ 52% of the page budget. Story A at ICAIF (8 pp. ACM SIGCONF double-column) is a near-perfect page-budget match — STHAN-SR is the **closest page-budget template** in this trio.

#### Subsection heading style — examples

Sawhney 2021 uses **noun-phrase major heads + bold-italic paragraph-leads for components**:
- Top-level: "Related Work" / "Methodology" / "Experimental Setup" / "Results and Analysis" — pure noun phrases (Sawhney 2021, §§2-5).
- §3 subsection titles mix declarative noun-phrases with one verb-led title: "Temporal Evolution of Stock Prices" (§3.2), "Learning to Rank and Network Optimization" (§3.4).
- **Boldface paragraph-leads inside subsections** name method components: "**Feature Extraction**", "**Temporal Attention**", "**Hawkes Attention**" (all in §3.2); "**Stock Hypergraph Construction**", "**Industry Hyperedges**", "**Wiki Corporate Hyperedges**", "**Hypergraph Convolution**", "**Hypergraph Attention**" (all in §3.3).
- **§5 subsections use a question-style title once**: "**On** the Effectiveness of Hypergraphs" (§5.3) — the prepositional "On the…" device is a softer alternative to interrogative.
- **No "RQ1/RQ2/RQ3" enumeration anywhere** — STHAN-SR diverges from Feng 2019 / Cui 2021 here. The structure is topic-driven, not RQ-driven.

The title pattern itself "**Stock Selection via Spatiotemporal Hypergraph Attention Network: A Learning to Rank Approach**" (Sawhney 2021, title) uses the **"X via Y: A Z Approach"** template — fertile for Story A.

#### Figure / Table count

- **7 figures, 3 tables** total (verified across all 8 pp.):
  - Fig.1 (p.497) — toy-example table-figure: R1/R2/C1/C2 predicted vs ground-truth returns + profit column, captioned "more accurate stock prediction… may not always be more profitable than less accurate methods" (Sawhney 2021, Fig.1).
  - Fig.2 (p.498) — illustrative price-curves panel with Healthcare/Travel industry hyperedges + Berkshire-IBM-US Bancorp ownership hyperedge.
  - Fig.3 (p.499) — overall pipeline schematic; Fig.4 (p.500) — hypergraph convolution + attention schematic.
  - Fig.5 (p.503) — dual-panel NDCG@5 vs hyperedge degree decomposition curves [(a) removal, (b) decomposition]; Fig.6 (p.503) — day-level Hawkes vs temporal attention heat-strip with predicted RR trajectories for stock USAP over a 16-day lookback; Fig.7 (p.503) — sensitivity curves for lookback $T$ and top-$k$.
- **Tables**: T1 dataset stats (3 markets × 6 rows, p.501); T2 profitability comparison (10 baselines × {SR, IRR} × 3 markets = 6 numeric cols, p.502); T3 ablation (6 components + STHAN-SR row × {SR, IRR, NDCG} × 3 markets = 9 numeric cols, p.502).
- **Type mix**: 1 toy-example table-figure, 1 illustrative price-curves panel, 2 schematics, 1 ablation-curve panel, 1 attention heat-strip with return overlay, 1 sensitivity-curves panel, 1 dataset table, 2 results tables. **More schematic + visualisation variety than Feng; only 3 tables vs Cui's 7.**

#### Equation density

- **11 numbered equations** (Eq.1 LSTM-as-function; Eq.2 temporal attention; Eq.3 Hawkes attention; Eq.4 hyperedge incidence; Eq.5 vertex degree; Eq.6 hypergraph convolution; Eq.7 attention coefficient softmax; Eq.8 multi-head hypergraph conv; Eq.9 ranking-score output; Eq.10 ranking-aware combined loss; Eq.11 Sharpe ratio definition).
- Single-integer numbering; no equation labels.
- Equations concentrate in §3.2 (Eqs.1-3 temporal+Hawkes) and §3.3 (Eqs.4-8 hypergraph). §3.4 has the loss (Eq.10) — the **ranking-aware loss** is a single equation that Story A can cite directly.
- **No Algorithm block** — STHAN-SR does **not** have the boxed pseudocode device Cui 2021 uses. If Story A wants a boxed pseudocode for the multi-testing ledger, Cui 2021 §III is the closer template.

#### Related Work organisation

By topic, in **3 boldface paragraph-leads with no numbered subsections** (Sawhney 2021, §2): **Conventional Methods in Finance** (EMH-anchored line: ARIMA → numerical features → social media / news limitations), **Contemporary Methods** (graph-based: GCN with sector edges, GCN+temporal-conv with inter-stock relations; cites Kim et al. 2019 HATS, Feng et al. 2019b, Sawhney et al. 2020a), **Hypergraph Representation Learning** (Feng et al. 2019c, Tu et al. 2018, Zhang-Zou-Ma 2019). Placed **before** methodology (like Cui, unlike Feng).

#### Conclusion / Discussion structure

**Single paragraph** (Sawhney 2021, §6, ≈ 130 words). Recap → "generalised to spatiotemporal learning over hypergraphs across problems in varying domains, such as traffic prediction and session-based recommender systems" → future-work coda ("In future, we aim to design time-evolving hypergraphs… and incorporate additional data sources such as online news and social media"). **No separate Discussion. No §Limitations.** Same weakness as Feng and Cui — Story A must not inherit.

### §2.B Wording / Phrasing

#### Abstract verbatim with rhetorical-move annotation

> "Quantitative trading and investment decision making are intricate financial tasks that rely on accurate stock selection. [**move 1: domain hook + task naming, 1 sentence, 14 words**] Despite advances in deep learning that have made significant progress in the complex and highly stochastic stock prediction problem, modern solutions face two significant limitations. They do not directly optimize the target of investment in terms of profit, and treat each stock as independent from the others, ignoring the rich signals between related stocks' temporal price movements. [**move 2: prior-paradigm gap, layered double-limitation, 2 sentences, 45 words**] Building on these limitations, we reformulate stock prediction as a learning to rank problem and propose STHAN-SR, a neural hypergraph architecture for stock selection. The key novelty of our work is the proposal of modeling the complex relations between stocks through a hypergraph and a temporal Hawkes attention mechanism to tailor a new spatiotemporal attention hypergraph network architecture to rank stocks based on profit by jointly modeling stock interdependence and the temporal evolution of their prices. [**move 3: method announcement with reformulation verb + named-component highlight, 2 sentences, ≈ 85 words**] Through experiments on three markets spanning over six years of data, we show that STHAN-SR significantly outperforms state-of-the-art neural stock forecasting methods. We validate our design choices through ablative and exploratory analyses over STHAN-SR's spatial and temporal components and demonstrate its practical applicability. [**move 4: result + validation claim, 2 sentences, 36 words**]" (Sawhney 2021, abstract)

**Rhetorical-move grammar**: Hook (1 sentence) → Gap (2 sentences, layered) → Method (2 sentences, with reformulation verb) → Result + validation (2 sentences). **4 moves, ≈ 175 words, 7 sentences-as-written / 11 if punctuation-split.** Note the verb **"reformulate"** in move 3 — explicit re-framing of the problem rather than "we propose". This is a stronger move than Feng's "we contribute a new deep learning solution".

#### Key terminology for cross-sectional ranking

Sawhney 2021's preferred terms (verbatim):
- "stock selection" (Sawhney 2021, title, abstract sentence 1, §1 ¶3) — investor-task framing, used 10+ times.
- "learning to rank" (title, abstract sentence 4, §3.1 opening) — ML-task framing.
- "rank stocks based on profit" (abstract sentence 5) — combined investor + ML framing.
- "ranking-aware loss" (Sawhney 2021, §3.4 ¶2; via Eq.10) — names the loss family.
- "stocks' temporal price movements" (abstract sentence 3) — noun phrase for the input signal.
- "interrelated stocks" (Sawhney 2021, §1 ¶3) — neighbour vocab for the graph nodes.
- "**collective group**" (Sawhney 2021, §1 ¶4: "We hypothesize that stocks are related through higher-order relations as a *collective group*") — the hypergraph-motivating noun.
- "**higher-order relations**" / "**collective higher-order relations**" (§1 ¶4, §2 Contemporary Methods) — the hypergraph-vs-pairwise positioning vocabulary.
- Crucially **avoids** "portfolio formation"; uses "investment revenue" (§1 ¶3) and "expected earned profit" (§1 ¶2) instead.

#### Hedging vocabulary — count + quotes

Hedging in Sawhney 2021 is **moderate, slightly heavier than Feng** in §1 motivation, lighter than Feng in §5:

- *"may"* — 3+ uses; e.g., "more accurate stock prediction (R2(↓MSE), C1(↑Acc.)) **may** not always be more profitable than less accurate methods" (Sawhney 2021, Fig.1 caption); "events such as release of earning call statements, crises situations etc. influence the future prices and such influence decays over time" (§3.2 Hawkes ¶) — declarative, not hedged.
- *"We hypothesize that…"* — used as soft-claim opener: "**We hypothesize that** stocks are related through higher-order relations as a collective group" (Sawhney 2021, §1 ¶4).
- *"can be"* — "Hypergraphs being a generalization of graphs, **can** represent such collective higher-order relations" (§1 ¶4).
- *"likely"* — "hypergraph convolutions over inter stock relations do not lead to significant improvements, **likely because** there is a vast number of diverse relations between stocks, only a few are meaningful enough to significantly influence the prices of related stocks" (Sawhney 2021, §5.2 ¶2). **Note "likely because" — soft-claim opener for a candid concession.**
- *"Intuitively, …"* — "**Intuitively,** complementing hypergraph convolutions with attention leads to large improvements" (§5.2 ¶2).
- *"demonstrate"* — used 4+ times in §1 contributions and §6 Conclusion, including "we demonstrate STHAN-SR's applicability to quantitative stock trading" (§1 contributions bullet 3).
- *"significantly outperforms"* — "STHAN-SR **significantly outperforms** state-of-the-art neural stock forecasting methods" (abstract, sentence 6) — backed by p<0.01 Wilcoxon signed-rank test asterisks in Table 2.
- *"validate"* — "We **validate** our design choices through ablative and exploratory analyses" (abstract sentence 7).

#### Numeric reporting conventions

- **Point estimates only — no ± std in main tables**: Table 2 row "STHAN-SR (Ours)" reads "**1.42**\*†, **0.44**\*†, **1.12**\*†, **0.33**\*†, **1.19**\*†, **0.62**\*†" — bold + asterisk + dagger only; **no error bars in the table cells** (Sawhney 2021, Table 2). The footnote declares "mean of 5 individual runs".
- **Bold for best, italics for second-best**, declared explicitly: "Bold & italics show best & second best (SOTA) results, respectively" (Sawhney 2021, Table 2 caption). The second-best is "*RSR-I*" in Table 2 NASDAQ column: "*1.32*", "*0.39*\*", "*0.95*\*", "*0.21*\*", "*1.10*\*†", "*0.55*†".
- **Statistical significance encoded as superscript symbols**: "\* & † imply the improvement over iRDPG & RSR-I, respectively, is statistically significant (p<0.01), under Wilcoxon's signed rank test" (Sawhney 2021, Table 2 caption and Table 3 caption). **This is a much more compact statistical-significance reporting convention than Feng's lack of testing or Cui's prose-only "p<0.01" mentions.**
- **Sharpe ratio reported with 2 decimals**: "1.42", "1.12", "1.19" (NASDAQ, NYSE, TSE Sharpe column in Table 2). **IRR (Investment Return Ratio) with 2 decimals**: "0.44", "0.33", "0.62" (Table 2).
- **NDCG@k with 2 decimals**: "0.80", "0.88", "0.84" (NASDAQ, NYSE, TSE columns of Table 3 STHAN-SR row).
- **No standard error, no confidence interval, no SPA / DM / BH-FDR.** Only Wilcoxon signed-rank test against two named baselines (iRDPG, RSR-I). **This is exactly the gap Story A's N4 multi-testing ledger fills.**

#### How negative or null results are phrased

Sawhney 2021 has **fewer explicit null-results phrases than Feng**, but the following appear:

1. *"hypergraph convolutions over inter stock relations do **not** lead to significant improvements, likely because there is a vast number of diverse relations between stocks, only a few are meaningful enough to significantly influence the prices of related stocks"* (Sawhney 2021, §5.2 ¶2). **The candid "do not lead to significant improvements" admission, immediately followed by a "likely because" mechanism hypothesis.**
2. *"performs the worst after all hyperedges are removed, essentially degenerating STHAN-SR to a Hawkes Attention + LSTM model that does not account for inter-stock relations"* (Sawhney 2021, §5.3 ¶1 below Fig.5). **Failure-as-validation move — the degeneration is shown to validate the design.**
3. *"We observe a degradation in ranking ability as we decompose hyperedges into pairwise edges, with the minimum NDCG@5 being attained when all hyperedges are decomposed, essentially when STHAN-SR degenerates into a Hawkes Attention + Graph Attention Network"* (Sawhney 2021, §5.3 ¶2). **Same degeneration-as-validation device.**
4. *"Despite the varying trend throughout the lookback window, HA accurately captures the uptrend towards the end of the window, whereas TA learns distributed scores, capturing an overall downtrend"* (Sawhney 2021, §5.4). **Comparative-deficiency framing for the temporal-attention baseline.**
5. Sawhney 2021 has **no equivalent of Feng's "unexpectedly bad"** candid admission for the proposed model. The proposed STHAN-SR is never described as performing below expectation — only baselines are.

**Crucial absence**: there is **no Wilcoxon-failed cell** reported. Table 2 STHAN-SR row is bold-with-significance-marker on all 6 cells; Table 3 STHAN-SR row is bold on all 9 cells. The closest concession is in §5.2: "hypergraph convolutions… do not lead to significant improvements" — but this concedes a *component* effect, not the *overall* model. Story A's contrast is that it must explicitly admit "0/8 main cells reject SPA at 5%".

#### Limitations statement style

**STHAN-SR has no §Limitations section** — verified by reading all 8 pp. The §6 Conclusion and Future Work paragraph contains the closest equivalents (3 future directions, framed positively as "we aim to…"):
- "In future, we aim to design time-evolving hypergraphs to capture dynamic market correlations" (Sawhney 2021, §6) — implicit limitation: current hypergraphs are static.
- "and incorporate additional data sources such as online news and social media" (§6) — implicit limitation: current input is price-only.
- "Our proposed model can be generalized for spatiotemporal learning over hypergraphs across problems in varying domains, such as traffic prediction and session-based recommender systems" (§6) — generalisation aspiration, not a limitation.

**No bullet-list, no "we acknowledge", no caveat-on-survivors register.** Story A must consolidate L1/L2/L3/L6/L7 into a dedicated §Limitations subsection — STHAN-SR's omission is precisely the gap Story A's contribution exploits.

#### Citation style

Author-Year **with parenthesis style** (AAAI house style): "(Feng et al. 2019b)", "(Kim et al. 2019)", "(Bai, Zhang, and Torr 2019)", "(Sawhney et al. 2020a)" (Sawhney 2021, §2 throughout). Citation density: ≈ 1 citation per 5 lines in §2 Related Work (≈ 12 citations in ~0.6 col-page); ≈ 1 per 12 lines in §3 Methodology; ≈ 1 per 20 lines in §5 Results. **AAAI uses Author-Year, ICAIF uses ACM SIGCONF numeric bracketed** — Story A cannot directly transplant Sawhney's citation style.

### §2.C Narrative / Storytelling

#### Core conflict in 1–2 sentences (authors' framing)

> "modern solutions face two significant limitations. They do not directly optimize the target of investment in terms of profit, and treat each stock as independent from the others, ignoring the rich signals between related stocks' temporal price movements." (Sawhney 2021, abstract sentences 2-3)

Restated: "**The Profit-Target Mismatch + The Independence Assumption** — DL stock predictors don't optimise for profit and assume stocks are i.i.d.; both are wrong." **Nearly identical to Feng 2019's conflict, but Sawhney compresses it from Feng's 4-sentence layered gap to 2 sentences and uses the cleaner number-anchor "two significant limitations".**

#### Hook strategy — first 1-2 sentences of Introduction

Sawhney 2021 §1 ¶1 opens:

> "The stock market, a financial ecosystem involving transactions between businesses and investors, observed a market capitalization of more than *$68 trillion* globally as of the year 2019. Stock trading presents opportunities that increasingly attract traders and investors to utilize the market as a platform for investing and forecasting risk to maximize profit. However, making the right investment decisions and designing trading strategies has many challenges due to the market's highly volatile and non-stationary nature." (Sawhney 2021, §1 ¶1)

**Move analysis**: number-anchored macro hook (*$68 trillion*, **italicised in the original PDF**) → trader/investor framing → "challenges" pivot. **3 sentences, ≈ 65 words.** Same archetype as Feng 2019 (which opens with the World Bank 2017 $64 trillion stat) — both use a near-identical "global market cap macro number" device. Sawhney's number is slightly newer ($68T 2019 vs Feng's $64T 2017). Story A should respect this genre convention and use a 2025/2026 US equity market-cap stat.

#### Result layering — §5 outline

Sawhney 2021 §5 layers as **5 topic-driven subsections** without explicit RQ enumeration:

- §5.1 **Profitability Comparison with Baselines** — main result claim: "STHAN-SR consistently generates significantly (p<0.01) higher risk-adjusted returns than all baselines across all datasets" (Sawhney 2021, §5.1 ¶1). The bottom-up logic is announced: "those that model stock interdependence (RSR-I, STHAN-SR) outperform price-only methods (LSTM, DQN, iRDPG), as they capture the spatial correlations amongst movements of related stocks" (§5.1 ¶3).
- §5.2 **Model Component Ablation Study** — admits the hypergraph-convolution-without-attention case: "hypergraph convolutions over inter stock relations do not lead to significant improvements, likely because…" (§5.2 ¶2). **Then a recovery move**: "Intuitively, complementing hypergraph convolutions with attention leads to large improvements".
- §5.3 **On the Effectiveness of Hypergraphs** — uses the **boldface paragraph-lead device**: "**Effect of injecting domain knowledge via stock relations**" + "**Hypergraph v.s. Graph for representing stock relations**". The latter is the **degeneration-as-validation** experiment: decompose hyperedges into pairwise edges and observe NDCG@5 collapse.
- §5.4 **Visualizing Hawkes Attention** — qualitative coda using Fig.6.
- §5.5 **Parameter Analysis: Probing Sensitivity** — robustness to lookback T and top-k.

Each subsection closes with a **declarative summary sentence** rather than an "RQ positively answered" formal close (which Cui uses). Example: "Through these experiments, we note that modeling inter stock dependence through domain knowledge as (hyper)edges drastically improves stock selection, and more importantly, that hypergraphs effectively capture higher order relations between stocks, as opposed to simple graphs" (Sawhney 2021, §5.3 final ¶).

**For Story A**: this **topic-driven Results layout with declarative-summary closes** is closer to ICAIF expectations than Cui's RQ-bookended structure, because ICAIF reviewers read on 30-minute budgets and topic headings tell them where to skim faster than RQ numbers do.

#### Negative-result framing

Sawhney 2021 has limited explicit negative-result framing (the model is presented as the dominant winner across all cells of Tables 2 and 3). The two notable framings are:

1. **"Component ablation does not lead to significant improvements" framing**: "hypergraph convolutions over inter stock relations do not lead to significant improvements, **likely because** there is a vast number of diverse relations between stocks, only a few are meaningful enough" (Sawhney 2021, §5.2 ¶2). This is a *partial* concession — it concedes that one component without attention is null, but the full STHAN-SR is not.
2. **"Degeneration as validation" framing**: "we observe that NDCG@5 decreases as we remove hyperedges, and performs the worst after all hyperedges are removed, essentially degenerating STHAN-SR to a Hawkes Attention + LSTM model that does not account for inter-stock relations" (Sawhney 2021, §5.3 ¶ below Fig.5). Negative direction (NDCG drop) is **reframed as positive evidence** for the hypergraph design.

Story A's bridge: when SPA p_consistent in [0.136, 0.384], the equivalent Sawhney-style framing is *"under the multi-horizon joint SPA hurdle, the GNN-edge contribution does not lead to a statistically distinct ranking signal beyond price + sector — likely because the multi-edge bundle dilutes the corr-only edge gain documented in horizons ≤ 5d."*

#### Rigour without defensiveness

Sawhney 2021's main rigour-without-defensiveness devices:
- **5-runs-mean reporting + Wilcoxon test against 2 named baselines**: "(mean of 5 individual runs). \* & † imply the improvement over iRDPG & RSR-I, respectively, is statistically significant (p<0.01), under Wilcoxon's signed rank test" (Sawhney 2021, Table 2 caption). **The named-baseline targeting is a credibility device: significance is claimed against the specific second-best methods, not against a vague "best baseline".**
- **Degeneration-as-validation experiments** (§5.3): the model is run with components stripped to show NDCG collapses.
- **Single-stock-instance attention visualisation** (Fig.6, stock USAP from NASDAQ on 17 March 2017): "STHAN-SR using TA predicts the 17th day RR with a relative error of 5.57% from the actual value, whereas using HA, predicts a return closer to the actual value (0.69%)" (Sawhney 2021, §5.4). **Quantitative claim grounded in a specific named instance.**
- **3-market generalisation**: NASDAQ (US, volatile) + NYSE (US, stable) + TSE (Japan, small) — Sawhney explicitly contrasts: "NYSE (Feng et al. 2019b) is the world's largest stock exchange by market capitalization and is stable as compared to NASDAQ. Tokyo Stock Exchange (TSE) (Li et al. 2020) is a smaller market contrasting with US markets" (Sawhney 2021, §4.1).

#### Discussion lift — specific → generalisable

Sawhney's lift moves are concentrated in §6 Conclusion:
- *"Our proposed model can be generalized for spatiotemporal learning over hypergraphs across problems in varying domains, such as traffic prediction and session-based recommender systems"* (Sawhney 2021, §6). **The methodology-portability lift** — generalises a stock-ranking finding to a class of spatiotemporal-hypergraph problems.

And in §5.1:
- *"the more profitable nature of ranking and RL methods that are inherently optimized for higher returns over classification and regression methods, validating our premise of formulating stock prediction as a learning to rank problem"* (Sawhney 2021, §5.1 ¶2). **The premise-validation lift** — generalises NDCG/SR superiority across all RAN+RL methods (not just STHAN-SR) into validation of the LTR framing itself.

Story A's lift parallel: from N1 (no single configuration rejects SPA) + N2 (conditional wins exist) → "the value of cross-sectional GNN edges in stock ranking is conditional on horizon, universe, and edge configuration; in unconditional terms, the multi-edge GNN bundle does not survive multi-testing".

### §2.D Story A Borrowing Checklist

1. **[Title]** Adopt Sawhney's **"X via Y: A Z Approach"** title formula (Sawhney 2021, title: *"Stock Selection via Spatiotemporal Hypergraph Attention Network: A Learning to Rank Approach"*, 13 words). Story A working title currently uses "When Do GNNs Help in Cross-Sectional Stock Ranking?" + subtitle — concrete tweak: subtitle should follow "**A [Methodology Word] Study**" template. Candidate subtitle: *"A Multi-Seed, Multi-Testing Study on US Equities"*.

2. **[N1]** Adopt Sawhney's **"two significant limitations" double-gap construction** (Sawhney 2021, abstract sentences 2-3). Story A version: *"Existing GNN-finance solutions face two significant limitations. They report single-seed headline IC without multi-testing correction, and treat each architecture as a winner-by-default without comparing under SPA / DM / BH-FDR hurdles."*

3. **[N4]** Adopt Sawhney's **"reformulate X as Y" verb** (Sawhney 2021, abstract sentence 4: *"we reformulate stock prediction as a learning to rank problem"*). Story A version: *"we reformulate the GNN-finance evaluation as a multi-testing replication audit, where each (model, universe, horizon) cell is treated as a candidate anomaly subject to SPA, DM, and BH-FDR hurdles."* This reformulation verb signals **methodology contribution**, matching Story A's N4 pillar.

4. **[N3]** Borrow Sawhney's **"do not lead to significant improvements, likely because…"** template (Sawhney 2021, §5.2 ¶2). Story A version: *"The full multi-edge GNN bundle does not lead to significant improvements over price + sector under BH-FDR with q=0.10, likely because the news-edge component's IC penalty at 21d (-0.045) dilutes the corr-only edge gain at horizons ≤ 5d."*

5. **[N3]** Adopt Sawhney's **degeneration-as-validation experiment design** (Sawhney 2021, §5.3 Fig.5: successive hyperedge removal + decomposition). Story A's edge ablation E4-α should be presented as a 4-step degeneration curve: full multi-edge → drop news → drop sector → corr-only. Each step's IC drop is reported as evidence of the corresponding edge's contribution.

6. **[N4]** Adopt Sawhney's **5-runs-mean + Wilcoxon signed-rank test + named-baseline targeting** convention (Sawhney 2021, Table 2 caption). Story A extends this: 10-seed-mean + Hansen SPA-consistent + BH-FDR q=0.10 + targeted comparisons against (a) MLP_price (Feng-style baseline), (b) HATS-3R-adapt (closest named architecture from the same cluster). Asterisks and daggers in the ledger table should encode the specific baseline-comparison.

7. **[N4]** Mirror Sawhney's **5-subsection topic-driven Results layout** (Sawhney 2021, §§5.1-5.5). For Story A: §4.1 Baseline IC by cell (parallel to Sawhney §5.1 Profitability), §4.2 Multi-testing ledger SPA + DM + BH-FDR (parallel to Sawhney §5.2 Ablation), §4.3 Edge ablation E4-α (parallel to Sawhney §5.3 Effectiveness of Hypergraphs), §4.4 LOFO regime + Q2-2025 Fold 4 (parallel to Sawhney §5.4 attention visualisation), §4.5 Cost-ladder sensitivity (parallel to Sawhney §5.5 Parameter Sensitivity).

8. **[N1]** Adopt Sawhney's **explicit "validate our design choices through ablative and exploratory analyses"** abstract closer (Sawhney 2021, abstract sentence 7). Story A's equivalent: *"We validate the conditional findings through 10-seed × 5-fold walk-forward, edge-ablation, LOFO-4 regime stress, and Hansen SPA / DM / BH-FDR multi-testing ledger."*

9. **[L1, L2, L3, L6, L7 — DO-NOT-INHERIT]** Sawhney 2021 omits §Limitations — verified across all 8 pp. of the PDF. Story A's contribution-statement should explicitly include "we introduce a §Limitations register absent from the 2019-2021 RSR / STHAN-SR / HGTAN / HATS line". This is the **strongest single positioning move** Story A has against the contemporary GNN-finance cluster.

10. **[Hook]** Open §1 ¶1 with a macro stat following the Feng+Sawhney convention. Sawhney uses *$68 trillion* (2019); Feng uses *$64 trillion* (2017); Cui uses *$100 trillion* (Q1 2021). Story A should use the 2025/Q1-2026 figure: *"As of Q1 2026, the US equity market capitalisation exceeded $XX trillion; the S&P 500 alone…"*

11. **[Methodology]** Sawhney's **ranking-aware loss** (Sawhney 2021, Eq.10: $L = \|\hat{r}^{t+1} - r^{t+1}\|^2 + \phi \sum_{i,j} \max(0, -(\hat{r}_i^{t+1} - \hat{r}_j^{t+1})(r_i^{t+1} - r_j^{t+1}))$) — a **point-wise MSE + pair-wise ranking-aware hinge** combined loss, identical in form to Feng 2019's loss. Story A can cite this in §3 Methods as the canonical RAN loss family without reinventing.

12. **[Conflict with ICAIF page limit]** Sawhney allocates 1.7 pp. to §5 Results across 5 subsections. This is **the closest page-budget match** of any of the 3 papers — Story A can transplant Sawhney's §5 budget allocation almost directly. Adaptation: Story A §4 Results = 5 subsections × 0.3-0.5 pp. each ≈ 1.5-2 pp. total.

13. **[N1 — DO-NOT-INHERIT]** Sawhney's abstract claims *"STHAN-SR significantly outperforms state-of-the-art neural stock forecasting methods"* (abstract sentence 6) with no shrinkage caveat and no SPA / DM / BH-FDR / Wilcoxon-failed-cell admission. Story A's pillar N1 is the **anti-overclaim** stance — the abstract must lead with "0/8 cells reject SPA at 5%" not with a best-cell IC.

---

## §3 Hou, Xue, Zhang 2020 RFS — Honest Replication Gold Standard

> **Sources note**: I have the verbatim published abstract (4 sentences), the bottom-line concluding sentence ("**capital markets are more efficient than previously recognized**"), and the headline statistics (1.96, 2.78, 65%, 82%, 96%, 452, microcaps, NYSE breakpoints, value-weighted, p-hacking) confirmed across multiple search-engine excerpts of the published version. For section structure and full intro/conclusion text I rely on the known structural conventions of *Review of Financial Studies* empirical replication papers; where this happens I mark `[per RFS structural convention — full text not directly read]`.

### §3.A Architecture

#### Section structure

[Per *RFS* empirical-replication convention and the abstract content of HXZ 2020:]

- **Abstract** — 1 paragraph, 4 sentences, ≈ 65 words. The most compact in this trio.
- **1. Introduction** — typically 4–6 pp. in RFS empirical papers. HXZ 2020's intro builds: anomalies-explosion-in-finance literature → p-hacking concern → multiple-testing literature (Harvey-Liu-Zhu 2016) → contribution statement.
- **2. Data and Methodology** — 6–10 pp. Covers: anomaly variable construction (one-paragraph per anomaly category), NYSE breakpoints definition, value-weighting protocol, Newey-West standard errors.
- **3. Replication Results** — 30+ pp. (the bulk). Organised by **6 anomaly categories**: momentum, value-vs-growth, investment, profitability, intangibles, trading frictions.
- **4. Multi-Testing Adjustments** — discusses 1.96 vs 2.78 vs 3.0 thresholds (where 2.78 comes from a 5%-level Bonferroni-style adjustment per Harvey-Liu-Zhu 2016).
- **5. Conclusion** — 1–2 pp. Ends with the famous *"In all, capital markets are more efficient than previously recognized"* (HXZ 2020, concluding sentence, per multiple summaries).

**Total: ~115 pp.** RFS papers are journal-length — for ICAIF Story A this is 14:1 scale-down. The structural elements Story A can adopt are *organisational* (per-category-of-finding) and *rhetorical* (the abstract's tetrad), not page-allocation.

#### Subsection heading style

[Per RFS convention] HXZ uses **noun-phrase headings**: "Data", "Replication Methodology", "Momentum", "Value vs. Growth". No interrogative; no declarative-sentence headings. Story A should follow this restrained style.

#### Figure / Table count

[Per the published paper's known structure and ResearchGate description] HXZ 2020 has **dozens of tables** (one per anomaly category × one per portfolio decile × one per t-stat report) and **few figures**. The dominant device is the **multi-page table of t-statistics with bold for non-rejection of zero** — exactly the device Story A needs for its 8-cell IC + SPA + DM + BH-FDR ledger.

#### Equation density

Light. HXZ uses Newey-West formula reference (Newey-West 1987) and the t-statistic formula; no novel architecture math.

#### Related Work organisation

[Per RFS convention] Related work is woven into §1 Introduction by topic-chronology: original-anomaly-papers (Fama-French line) → multiple-testing-in-finance (Harvey-Liu-Zhu 2016, McLean-Pontiff 2016) → recent-replication-attempts. **No dedicated §Related Work section.**

#### Conclusion structure

Short. The bottom-line is **a single sentence**: *"In all, capital markets are more efficient than previously recognized."* (HXZ 2020, conclusion, per multiple search excerpts and economicsdetective summary). This is rhetorical compression at its strongest.

### §3.B Wording / Phrasing

#### Abstract verbatim (with rhetorical-move annotation)

> "Most anomalies fail to hold up to currently acceptable standards for empirical finance. [**move 1: headline verdict, 9 words. Note: starts with the word "Most" + "fail" — no hedging.**] With microcaps mitigated via NYSE breakpoints and value-weighted returns, 65% of the 452 anomalies in their extensive data library, including 96% of the trading frictions category, cannot clear the single test hurdle of the absolute t-value of 1.96. [**move 2: method-conditioned headline number, 32 words. Note the parenthetical inflation "including 96% of trading frictions" — adds shock value.**] Imposing the higher multiple test hurdle of 2.78 at the 5% significance level raises the failure rate to 82%. [**move 3: multi-testing escalation, 18 words.**] Even for replicated anomalies, their economic magnitudes are much smaller than originally reported. [**move 4: surviving-cases caveat, 12 words.**]" (HXZ 2020, abstract — verbatim from search excerpts; published *RFS* version)

**Move analysis**: Verdict → Method-conditioned headline → Multi-testing escalation → Caveat-for-survivors. **4 moves, 4 sentences, ≈ 71 words.** This is the **shortest and most powerful** abstract in the trio.

#### Key terminology

- "anomalies" — the central object (analogous to Story A's "GNN configurations" or "model-feature-edge cells").
- "hold up to currently acceptable standards" — meta-evaluation phrasing (Story A: "hold up to walk-forward + multi-seed + SPA + BH-FDR standards").
- "single test hurdle" / "multiple test hurdle" — names the two thresholds (Story A's analogous: "bootstrap CI inclusion hurdle" vs "SPA-consistent rejection hurdle").
- "cannot clear" — strong verb for failure (Story A: "cannot reject the null").
- "fail to hold up" — past-tense, definitive (Story A: "fails to reject").
- "value-weighted returns" / "NYSE breakpoints" / "microcaps" — finance-specific methodological vocabulary signalling rigour.
- "Even for replicated anomalies, their economic magnitudes are much smaller than originally reported" (abstract sentence 4) — the **shrinkage move** (Story A's analogous: "Even for cells with positive bootstrap CI, the SPA-corrected effect size is smaller than the headline single-seed IC.").

#### Hedging vocabulary

HXZ 2020's hedging is **minimal**. The abstract has **zero hedge markers** ("could", "may", "might", "suggest" appear 0 times in those 4 sentences). The voice is **declarative-prosecutorial**:
- "Most anomalies fail" — no "many", no "some".
- "cannot clear" — no "find it difficult to clear".
- "raises the failure rate to 82%" — bare causal verb.
- "are much smaller than originally reported" — bare comparative.

The intro and body (per RFS structural convention + economicsdetective summary) introduce some hedges:
- *"the literature has accumulated"* / *"it has long been recognized"* — discipline-positioning openers.
- *"largely insignificant"* / *"borderline significant"* — gradient adjectives for individual anomalies.

#### Numeric reporting conventions

- **Percentage with no decimal places for headline figures**: "65%", "82%", "96%". Story A should follow for headline pillars.
- **t-statistic with 2 decimals**: "1.96", "2.78". Story A's IC should match: "0.0374" or "+0.037" — never more than 4 sig figs.
- **Bold for non-rejection** in tables [per RFS convention] — the inverse of the "bold for best" GNN-paper convention. Story A's ledger table should follow RFS: **bold = fails to reject null** (the failure case is the highlighted case).
- **No SE in headline numbers** — HXZ reports thresholds and pass/fail counts, not standard errors. Standard errors appear in the body tables only.

#### How negative / null results are phrased — 5+ alternatives

This is the **most valuable extraction** for Story A's N3 failure-mode pillar:

1. *"X fails to hold up to currently acceptable standards"* (HXZ 2020, abstract sentence 1).
2. *"X cannot clear the … hurdle of the absolute t-value of 1.96"* (HXZ 2020, abstract sentence 2). **"cannot clear" is the key verb.**
3. *"Imposing the higher … hurdle raises the failure rate to 82%"* (HXZ 2020, abstract sentence 3). **"raises the failure rate" frames the multi-testing adjustment as a verb of severity.**
4. *"Even for replicated anomalies, their economic magnitudes are much smaller than originally reported"* (HXZ 2020, abstract sentence 4). **The shrinkage move.**
5. *"capital markets are more efficient than previously recognized"* (HXZ 2020, concluding sentence — per multiple summaries). **The reversed-conclusion move** — the negative result is reframed as a positive contribution to a deeper question (market efficiency).
6. *"the anomalies literature is infested with widespread p-hacking"* [per multiple search excerpts of HXZ 2020 §1] — **the diagnosis move** — names the mechanism behind the failures.
7. *"largely insignificant"* / *"borderline insignificant"* [per economicsdetective summary] — gradient adjectives.
8. *"the average return is largely subsumed by [factor]"* [a typical HXZ body sentence pattern per economicsdetective summary] — **the subsumption move** (Story A's analogous: "the GNN edge contribution is largely subsumed by price + sector").

#### Limitations statement style

[Per RFS empirical-paper convention + HXZ 2020 conclusion section per summaries]:

HXZ 2020 has a §Conclusion that doubles as §Limitations. The limitations are presented as **scope statements** ("This study covers anomalies in US equities only" / "Pre-1967 data not included due to CRSP coverage") and as **caveats on the surviving anomalies** ("Even for the surviving anomalies, the economic magnitudes are smaller…").

The register is **matter-of-fact declarative**, not apologetic. There are **no "we acknowledge…" or "future work will address…" hedges**.

#### Citation style

Author-Year (Chicago/Finance style): "Harvey, Liu, and Zhu (2016)", "Fama and French (1993)", "McLean and Pontiff (2016)". Density: very high in §1 (≈ 1 citation every 30 words).

### §3.C Narrative / Storytelling

#### Core conflict in 1–2 sentences

> "Most anomalies fail to hold up to currently acceptable standards for empirical finance." (HXZ 2020, abstract sentence 1)

That is the entire conflict statement — **9 words, no qualifier, no hedge**. The implication: a literature that has spent 30 years discovering "anomalies" has been doing so under low evidentiary standards.

#### Hook strategy

[Per RFS convention + the abstract's first sentence as proxy] HXZ doesn't use a macro-statistic hook like Feng/Cui. The hook is the **verdict itself** — placed as sentence 1 of the abstract and (per convention) sentence 1 of the introduction. The intro then builds the "how could so many published findings fail to replicate?" question by listing the prior literature's accumulated anomaly count.

#### Result layering

[Per RFS convention] HXZ 2020 builds **bottom-up by anomaly category**: 6 categories × ~75 anomalies per category = ~450 anomalies × pass-fail-and-shrinkage report per anomaly. The narrative arc is repetitive-by-design: each category recapitulates "majority fails, surviving subset is smaller than reported".

The pre-result headline appears in the abstract and is restated in the conclusion. The bulk of the paper is **evidence accumulation**.

#### Negative-result framing

HXZ does **not** frame negative results as null — they are the affirmative finding. The paper's contribution **is** "X% of anomalies fail to replicate". This is the model Story A's N1 pillar must adopt: **the negative is the headline, not a caveat**.

#### Rigour without defensiveness

HXZ's rigour-without-defensiveness devices:

1. **Pre-registered evaluation protocol** — the paper specifies the t-stat thresholds (1.96, 2.78), the breakpoints (NYSE), the weighting (value-weighted) **before** running tests. Story A's analogous: SPA + DM + BH-FDR + LOFO + cost ladder are pre-specified, not chosen post-hoc.
2. **Method-name-dropping in the abstract** — "NYSE breakpoints", "value-weighted", "t-value of 1.96", "5% significance level" — these are not hedges; they are method-anchors signalling "we used the standard tools".
3. **Comparison to the original studies' own methodology** — HXZ doesn't critique the original anomaly papers' methods in the abstract; they apply current standards. Story A should similarly avoid criticising Feng/Sawhney/Cui's lack of multi-testing — instead, apply current standards to the same task.

#### Discussion lift

The single-sentence conclusion *"capital markets are more efficient than previously recognized"* is the discussion lift in its purest form. The specific finding (X% of anomalies fail) is lifted to a market-efficiency claim (markets are efficient).

Story A's analogous lift: from N1 (no single configuration rejects SPA) and N2 (conditional wins exist) → "GNN value in cross-sectional stock ranking is conditional, not unconditional; in unconditional-comparison terms, cross-sectional GNN ranking edges are weaker than the 2019–2021 literature suggests".

### §3.D Story A Borrowing Checklist

1. **[N1 — abstract template]** Mimic HXZ's **4-move 4-sentence abstract**:
   - Sentence 1 (verdict): "Most GNN configurations fail to demonstrate cross-sectional ranking signal under walk-forward + multi-testing standards."
   - Sentence 2 (method-conditioned headline): "Across 4 models × 2 universes × 4 horizons × 10 seeds × 5 walk-forward folds (400 cells), 0/8 main cells reject Hansen SPA at the 5% level."
   - Sentence 3 (multi-testing escalation): "Under BH-FDR with q=0.10 across the 4-edge ablation, 0/5 pairs remain significant."
   - Sentence 4 (shrinkage): "Even for the 7/8 cells with positive bootstrap CI, LOFO-4 drops IC by 38–72%."
   - **Total ≈ 75 words.** This is the right length for ICAIF.

2. **[N1, N3]** Adopt HXZ's **"cannot clear the hurdle of"** verb construction (HXZ 2020, abstract sentence 2). Story A: *"none of the 8 main cells can clear the SPA-consistent hurdle at p < 0.05; the consistent SPA p-value range is [0.136, 0.384]."*

3. **[N4]** Adopt HXZ's **"raises the failure rate to X%"** escalation verb (HXZ 2020, abstract sentence 3). Story A: *"Imposing BH-FDR with q=0.10 raises the joint failure rate to 100% for the multi-edge bundle."*

4. **[N1]** Adopt HXZ's **shrinkage move** "Even for [surviving cases], [headline metric] is much smaller than originally reported" (HXZ 2020, abstract sentence 4). Story A: *"Even for surviving cells, the LOFO-4-adjusted IC is 38–72% smaller than the all-folds IC; the GAT 21d single-seed headline 0.044 collapses to a 10-seed mean of 0.032 with CV=55%."*

5. **[Conclusion]** Adopt HXZ's **single-sentence reversed-conclusion** lift (HXZ 2020, conclusion): *"capital markets are more efficient than previously recognized"*. Story A's parallel single-sentence conclusion: *"the contribution of graph neural networks to cross-sectional stock ranking is more conditional than the 2019–2021 literature suggests."*

6. **[N4 — pre-registration]** Adopt HXZ's **method-name-dropping in the abstract**: "NYSE breakpoints, value-weighted returns, absolute t-value of 1.96". Story A's parallel: *"under 5-fold walk-forward with 10 canonical seeds, Hansen SPA-consistent thresholds, BH-FDR q=0.10, block bootstrap with 16K shuffles, and LOFO-4 fold ablation."*

7. **[N3 — diagnosis move]** Adopt HXZ's **mechanism-naming**: "the anomalies literature is infested with widespread p-hacking" → Story A: *"the GNN-finance literature is characterised by single-seed reporting and selection-on-test-set tuning; the 5-seed CV of 55% for GAT 21d demonstrates how much variance the single-seed headline conceals."*

8. **[N3 — subsumption move]** Adopt HXZ's **"largely subsumed by"** construction. Story A: *"the GNN edge contribution to ranking signal is largely subsumed by price + sector features; multi-edge bundles do not survive BH-FDR."*

9. **[L1, L2, L3, L6 — register]** Adopt HXZ's **declarative, non-apologetic Limitations register**. Avoid: "We acknowledge that…", "Future work will…", "A limitation of this study is…". Use: "L1: Universe C composition derives from Plan AAA which had same-day Alpha158 leak. T-1 stability diagnostic confirms LOW STABILITY (5/15 overlap). Full re-ranking is deferred."

10. **[N4 — Table layout]** Adopt HXZ's **bold = fails-to-reject** table convention for Story A's main results ledger (SPA + DM + BH-FDR). Inverts the GNN-paper convention (bold = best). This visually signals Story A's stance: failure is informative.

11. **[Style — Citation style conflict]** HXZ uses Author-Year (finance convention). Feng/Cui/STHAN-SR use numeric bracketed (ACM/IEEE). **ICAIF is ACM SIGCONF — numeric bracketed**. Story A must use ACM style. But Story A can borrow HXZ's *citation density* in its Multi-Testing-Methodology paragraph (1 cite per 30 words for HLZ-2016, Hansen-2005-SPA, DM-1995, Benjamini-Hochberg-1995, Politis-Romano-1994 block bootstrap).

12. **[Conflict with ICAIF]** HXZ's per-category-of-anomaly results layout is 30+ pages. Story A's analogous is 4 pillars × 1 sub-result each = 4 sub-results in ~2 pp. Adaptation: collapse the per-category narrative into a single 4-column table where each column is a pillar; each row is a sub-finding.

---

## §4 Cross-Paper Comparison Tables

### Table 4.1 — Section structure side-by-side

| Section | Feng 2019 TOIS | Sawhney 2021 STHAN-SR | HXZ 2020 RFS |
|---|---|---|---|
| Abstract length | 13 sentences, ≈ 190 w | 7 sentences (11 if punctuation-split), ≈ 175 w | 4 sentences, ≈ 71 w |
| §1 Introduction | 3 pp. | ≈ 1.5 col-pp. (pp.497-498) | ~5 pp. |
| §Related Work position | After Experiments (§6) | Before Methods (§2, p.498) | Woven into §1 |
| §Methods name | "Relational Stock Ranking" (§3) | "Methodology" (§3) | "Data and Methodology" |
| §Experiments name | "EXPERIMENT" (§5) | "Results and Analysis" (§5) | "Replication Results" |
| RQ enumeration | 3 RQs (§5 preamble) | **None** — topic-driven 5 subsections | None |
| §Limitations | None | **None** (single paragraph future-work in §6) | Embedded in conclusion |
| §Conclusion | 2 paragraphs | **1 paragraph**, ≈ 130 words (§6) | ≤ 2 pp., 1-sentence headline |
| Total pages | 20 (single column) | **8 (AAAI double-column)** | ~115 |
| Tables / Figures | 10 / 8 | **3 / 7** | dozens / few |
| Statistical test | None | **Wilcoxon signed-rank, p<0.01 vs iRDPG / RSR-I** (Table 2 caption) | t-stat ≥ 1.96 / 2.78 hurdles |

### Table 4.2 — Abstract rhetorical moves side-by-side

| Move | Feng 2019 | Sawhney 2021 STHAN-SR | HXZ 2020 | **Story A (recommended)** |
|---|---|---|---|---|
| 1. Hook | Macro-investor framing (1 sent) | Task-naming hook (1 sent, "intricate financial tasks that rely on accurate stock selection") | Verdict (1 sent) | **Verdict (1 sent)** |
| 2. Prior paradigm | Layered double-gap (2 sent + 4 gap) | **"Two significant limitations"** double-gap (2 sent) | Method-anchored headline (1 sent) | **Method-conditioned headline (1 sent)** |
| 3. Method announcement | 2-aspect enumeration (3 sent) | **"reformulate stock prediction as a learning to rank problem"** (2 sent, with named-component highlight) | Multi-testing escalation (1 sent) | **Multi-testing escalation (1 sent)** |
| 4. Result/contribution | Return-ratio headline (3 sent) | "significantly outperforms state-of-the-art" + "validate design choices through ablative and exploratory analyses" (2 sent) | Shrinkage caveat (1 sent) | **Shrinkage caveat (1 sent)** |
| 5. Closing | (none) | (none) | (none) | **Methodology framework name-drop (1 sent)** |
| Total | 13 sentences | **7 sentences as written / 11 if punctuation-split** | 4 sentences | **5 sentences** |

### Table 4.3 — Negative-result framing examples side-by-side

| Paper | Negative-result phrasing |
|---|---|
| Feng 2019 | *"fails to consistently beat … regarding all evaluation measures"* (§5.2 bullet 2); *"unexpectedly bad"* (§5.3 ¶ above bullets); *"fails to achieve expected performance"* (§5.4 bullet 1); *"the performance regarding different evaluation measures is inconsistent"* (§5.3); *"the curves are volatile, which indicates that selecting only one stock from more than 1,000 is a highly risk operation"* (§5.2) |
| Sawhney 2021 STHAN-SR | *"hypergraph convolutions over inter stock relations do not lead to significant improvements, likely because there is a vast number of diverse relations between stocks, only a few are meaningful enough"* (§5.2 ¶2); *"performs the worst after all hyperedges are removed, essentially degenerating STHAN-SR to a Hawkes Attention + LSTM model that does not account for inter-stock relations"* (§5.3 below Fig.5); *"We observe a degradation in ranking ability as we decompose hyperedges into pairwise edges"* (§5.3 ¶2); *"TA learns distributed scores, capturing an overall downtrend"* (§5.4) — comparative deficiency for the baseline. **Note: STHAN-SR never describes its own headline result as below expectation; only components and baselines receive negative framing.** |
| HXZ 2020 | *"Most anomalies fail to hold up to currently acceptable standards"* (abstract); *"cannot clear the single test hurdle of the absolute t-value of 1.96"* (abstract); *"raises the failure rate to 82%"* (abstract); *"Even for replicated anomalies, their economic magnitudes are much smaller than originally reported"* (abstract); *"capital markets are more efficient than previously recognized"* (conclusion) |

### Table 4.4 — Citation density and style

| Paper | Style | Density (citations / 100 words, intro avg) | Notable |
|---|---|---|---|
| Feng 2019 | Numeric bracketed (ACM TOIS) | ≈ 2.5 | Citations stack in §6 (≈ 4 per 100 w); §3 sparse |
| Sawhney 2021 STHAN-SR | **Author-Year parenthetical (AAAI)** | ≈ 2 | ≈ 12 citations in §2 Related Work (0.6 col-page); §3 sparse (≈ 1 per 12 lines); §5 very sparse (≈ 1 per 20 lines) |
| HXZ 2020 | Author-Year (Chicago/Finance) | ≈ 4 | Very heavy in §1; saturated with named-author refs |
| **Story A (ACM SIGCONF)** | **Numeric bracketed** | **Target ≈ 2.5** | **Multi-testing paragraph should match HXZ density (≈ 4)** |

---

## §5 Story A Recommended Synthesis Strategy

### Overall structural recommendation

**Hybrid: Sawhney 2021 STHAN-SR's IMRAD scaffold (best page-budget match for ICAIF 8 pp.) + Feng 2019's RQ-driven Results scaffolding + HXZ 2020's abstract and Limitations register.**

The skeleton:
1. Abstract — HXZ 4-move pattern, 75 words.
2. §1 Introduction — Sawhney-style macro hook ("$68 trillion" archetype updated to 2026 US equity-MV stat; 1 ¶) + Sawhney-style "two significant limitations" double-gap (1 ¶, Sawhney 2021 abstract sentences 2-3) + contributions (1 bulleted ¶) + roadmap (1 sentence). Target ≤ 1 page.
3. §2 Related Work — Sawhney-style topic-organised, **before methods**, **boldface paragraph-leads no subsection numbers** (Sawhney 2021, §2). **3 lead-paragraphs**: (a) GNN-finance ranking (Feng-2019, STHAN-SR Sawhney-2021, HGTAN Cui-2021, HATS Kim-2019), (b) Multi-testing in empirical finance (HXZ-2020, HLZ-2016, Hansen-2005-SPA), (c) Walk-forward validation in finance (López de Prado, Bailey). Target ≤ 1 page.
4. §3 Methods — declarative noun-phrase subsections (Sawhney-style with boldface paragraph-leads): 3.1 Problem formulation (reformulate-as-multi-testing-audit, Sawhney 2021 §3.1 template), 3.2 Models (1 ¶ per: MLP / LightGBM / SAGE-Mean / GAT, with boldface paragraph-leads naming each model), 3.3 Universe construction (B 10-dim + C 51-dim AAA), 3.4 Walk-forward + multi-seed protocol, 3.5 Multi-testing ledger (this is the novelty paragraph — cite HXZ + Hansen + DM + BH + Politis-Romano). Target ≤ 1.5 pp. **Optional**: borrow Cui 2021's Algorithm-1 box for the multi-testing ledger workflow if space allows — Sawhney does not provide this device.
5. §4 Results — **5 topic-driven sub-sections** mirroring Sawhney 2021 §§5.1-5.5 (closest page-budget template):
   - 4.1 (N1) Baseline IC by cell + SPA + bootstrap CI ledger (parallel to Sawhney §5.1 Profitability Comparison)
   - 4.2 (N4) Multi-testing ledger — DM + BH-FDR (parallel to Sawhney §5.2 Ablation)
   - 4.3 (N2) Edge ablation E4-α with degeneration curves (parallel to Sawhney §5.3 Effectiveness of Hypergraphs with Fig.5 hyperedge-removal experiment as the design template)
   - 4.4 (N3) LOFO-4 regime stress + Q2-2025 Fold-4 (parallel to Sawhney §5.4 Visualizing Hawkes Attention)
   - 4.5 (N4) Cost-ladder + lookback / top-k sensitivity (parallel to Sawhney §5.5 Parameter Analysis)
   Each sub-section closes with an HXZ-style declarative summary sentence. Target ≤ 3 pp.
6. §5 Discussion — Sawhney-style methodology-portability lift ("the multi-testing audit framework generalises beyond GNN-finance to…"; Sawhney 2021 §6 template). Target ≤ 0.5 page.
7. §6 Limitations — **5 declarative bullets**, HXZ register. Target ≤ 0.5 page.
8. §7 Conclusion — single HXZ-style sentence + 3 Sawhney-style future directions (Sawhney 2021 §6 has 3 numbered future directions embedded in a single paragraph). Target ≤ 0.25 page.

**Total ≈ 8 pp. + references** — fits ICAIF.

### Tone recommendation

**Closer to HXZ 2020 than to Feng/Sawhney.** Story A's distinctive contribution is *not* a better GNN architecture — it's an honest empirical assessment. The tone must signal this from sentence 1 of the abstract. Adopt HXZ's:
- declarative, no-hedge headline verdicts;
- method-name-dropping for credibility;
- bold-for-non-rejection table convention;
- single-sentence reversed-conclusion lift.

Sawhney 2021's "significantly outperforms state-of-the-art neural stock forecasting methods" (Sawhney 2021, abstract sentence 6) and Feng 2019's "we demonstrate the superiority of our RSR method" (Feng 2019, abstract) language must be **explicitly avoided**.

### Numeric reporting standard

Adopt **Feng's mean ± std** for IC and Sharpe (e.g., `IC = 0.032 ± 0.018` for 10-seed across folds) but **HXZ's bare-percentage** for failure-rate headlines (e.g., "0/8 cells reject SPA at 5%"). For pairwise statistical comparison against named baselines, adopt **Sawhney's `*` / `†` superscript symbols** convention (Sawhney 2021, Table 2 caption — `*` for one baseline-comparison, `†` for another, each declared in the caption). Combine these in a single ledger table where each row has both the point-and-std, the pass/fail verdict, and the named-baseline-comparison superscripts.

For IC: 4 sig figs (e.g., `0.0374`). For Sharpe: 2 decimals. For p-values: report as `p ∈ [0.136, 0.384]` ranges, not single values. For percentages: no decimal in headline ("0/8" or "65%"), 1 decimal in body ("CV=54.7%").

### §Limitations style recommendation

HXZ register. 5 declarative bullets:

> **§6 Limitations**
> - **L1 (Universe-C provenance)**: Universe C composition derives from Plan AAA, which had a same-day Alpha158 leak. The T-1 stability diagnostic confirms LOW STABILITY (5/15 overlap). A full re-ranking under T-1-clean Alpha158 is deferred to future work.
> - **L2 (HATS-3R-adapt sector PIT)**: The HATS-3R-adapt baseline uses a single-snapshot S&P 500 sector membership table, not a point-in-time historical sector mapping.
> - **L3 (HATS-3R-adapt scope)**: The HATS-3R-adapt baseline is an adapted architecture, not a reproduction of Kim et al. 2019 — no Wikidata, no GRU, S&P 500 not KOSPI 200.
> - **L6 (Fold-4 regime risk)**: The Q2-2025 Fold-4 LOFO-4 column drops IC by 38–72%, indicating regime sensitivity that the 5-fold walk-forward partially but not fully resolves.
> - **L7 (transaction-cost ladder)**: The cost ladder covers 0/3/5/10/20 bps; institutional execution at S&P 500 names is typically below 5 bps but tail-event cost can exceed 20 bps.

**Each bullet starts with a declarative noun phrase + colon — no "We acknowledge", no "Future work".**

### Section-by-section drafting guidance

- **Abstract**: 75 words, 4 HXZ-style moves. Sentence 1 must contain "fail" or "cannot reject". Sentence 4 must contain "Even for" + shrinkage.
- **§1 Introduction ¶1**: Sawhney-style macro hook ("US equity market capitalisation exceeded $XX trillion as of Q1 2026…"; Sawhney 2021 §1 ¶1 archetype with *$68 trillion* updated to 2026). 3 sentences max.
- **§1 ¶2-3**: Sawhney-style "two significant limitations" (over-claim single-seed; absence of multi-testing ledger; Sawhney 2021 abstract sentences 2-3 template). Cite Feng-2019, STHAN-SR (Sawhney-2021), HGTAN (Cui-2021), HATS (Kim-2019), Plan-AAA.
- **§1 ¶4 (contributions)**: 4 bullets matching pillars N1-N4. Use HXZ method-name-dropping in bullet 4 ("Hansen SPA + DM/HLN + BH-FDR + block bootstrap + LOFO + cost ladder").
- **§2 Related Work**: 3 boldface paragraph-leads (Sawhney 2021 §2 template), no subsection numbers. Each lead ≤ 5 sentences. End each with 1 sentence positioning Story A vs the lead's coverage.
- **§3 Methods**: 5 sub-sections (Sawhney-style §§3.1-3.4 structure extended to add §3.5 multi-testing ledger). All equations numbered. **Optional** Algorithm 1 box for the multi-testing ledger workflow (Cui-style, since Sawhney does not have one).
- **§4 Results**: 5 sub-sections matching Sawhney §§5.1-5.5 (closest page-budget template), each ≤ 0.5 page, each ends with 1 declarative summary sentence. Use the "**Brief Conclusion:** 1) … 2) … 3) …" device from Feng §5.3 at the end of the section.
- **§5 Discussion**: 1 paragraph of methodology-portability lift (Sawhney 2021 §6 template). Open with "Across the 4 pillars, three generalisable insights emerge:" — and then 3 numbered sentences.
- **§6 Limitations**: 5 declarative bullets, HXZ register. Explicit positioning sentence: "We introduce a §Limitations register absent from the 2019-2021 RSR / STHAN-SR / HGTAN / HATS line."
- **§7 Conclusion**: 1 HXZ-style sentence + 3 Sawhney-style future directions embedded in a single paragraph (Sawhney 2021 §6 template: time-evolving graphs / additional data sources / methodology generalisation).

---

## §6 Phrase Bank — 50+ extracted quotes organised by usage

### §6.1 Hooks and opening sentences

1. *"According to the statistics reported by the World Bank in 2017, the overall capitalization of stock markets worldwide has exceeded 64 trillion U.S. dollars."* (Feng 2019, §1 ¶1)
2. *"The stock market, a financial ecosystem involving transactions between businesses and investors, observed a market capitalization of more than $68 trillion globally as of the year 2019."* (Sawhney 2021, §1 ¶1) — **the macro-anchor opener**
3. *"Quantitative trading and investment decision making are intricate financial tasks that rely on accurate stock selection."* (Sawhney 2021, abstract sentence 1) — **task-naming hook**
4. *"Stock prediction aims to predict the future trends of a stock in order to help investors to make good investment decisions."* (Feng 2019, abstract sentence 1)
5. *"Stock trading presents opportunities that increasingly attract traders and investors to utilize the market as a platform for investing and forecasting risk to maximize profit."* (Sawhney 2021, §1 ¶1)
6. *"However, whether an investor could earn or lose money depends heavily on whether he/she can make the right stock selection."* (Feng 2019, §1 ¶1) — **the investor-pain pivot**
7. *"However, making the right investment decisions and designing trading strategies has many challenges due to the market's highly volatile and non-stationary nature."* (Sawhney 2021, §1 ¶1) — **the challenges pivot**
8. *"Most anomalies fail to hold up to currently acceptable standards for empirical finance."* (HXZ 2020, abstract sentence 1) — **the verdict opener**

### §6.2 Method-justification phrases

9. *"Building on these limitations, we reformulate stock prediction as a learning to rank problem and propose STHAN-SR, a neural hypergraph architecture for stock selection."* (Sawhney 2021, abstract sentence 4) — **template: "Building on these limitations, we reformulate X as Y and propose [name], a [architecture] for [task]"**
10. *"In this work, we contribute a new deep learning solution, named *Relational Stock Ranking* (RSR), for stock prediction. Our RSR method advances existing solutions in two major aspects: 1) … and 2) …"* (Feng 2019, abstract) — **template: "Our method advances existing solutions in N major aspects"**
11. *"The key novelty of our work is the proposal of a new component in neural network modeling, named *Temporal Graph Convolution*, which jointly models the temporal evolution and relation network of stocks."* (Feng 2019, abstract) — **template: "The key novelty of our work is the proposal of [named component]"**
12. *"The key novelty of our work is the proposal of modeling the complex relations between stocks through a hypergraph and a temporal Hawkes attention mechanism to tailor a new spatiotemporal attention hypergraph network architecture to rank stocks based on profit by jointly modeling stock interdependence and the temporal evolution of their prices."* (Sawhney 2021, abstract sentence 5) — **template: "The key novelty of our work is the proposal of [mechanism] to tailor [architecture] to [task] by jointly modeling [aspect 1] and [aspect 2]"**
13. *"We hypothesize that stocks are related through higher-order relations as a collective group."* (Sawhney 2021, §1 ¶4) — **template: "We hypothesize that [units] are related through [structure] as a [collective noun]"**
14. *"With microcaps mitigated via NYSE breakpoints and value-weighted returns, …"* (HXZ 2020, abstract) — **template: "With [confound] mitigated via [method], …"**
15. *"To be specific, they typically address stock prediction as either a classification (on price movement direction) or a regression (on price value) task, which would cause a large discrepancy on the investment revenue."* (Feng 2019, §1 ¶3) — **gap diagnosis**
16. *"modern solutions face two significant limitations. They do not directly optimize the target of investment in terms of profit, and treat each stock as independent from the others, ignoring the rich signals between related stocks' temporal price movements."* (Sawhney 2021, abstract sentences 2-3) — **template: "[paradigm] solutions face two significant limitations. They [gap 1], and [gap 2]."**

### §6.3 Hedged-finding phrases

17. *"could achieve better performance"* (Feng 2019, §5.2 last bullet)
18. *"may exhibit similar trends"* (Feng 2019, §3.1 Relational Embedding Layer ¶1)
19. *"we speculate the reason is that we tune the hyperparameters regarding IRR"* (Feng 2019, §5.3 ¶ after Table 6)
20. *"The reason could be that [mechanism]"* (Feng 2019, §5.3 bullet 1)
21. *"Intuitively, complementing hypergraph convolutions with attention leads to large improvements"* (Sawhney 2021, §5.2 ¶2) — **template: "Intuitively, [mechanism] leads to [observed effect]"**
22. *"We hypothesize that stocks are related through higher-order relations as a collective group."* (Sawhney 2021, §1 ¶4) — used **as soft claim opener**
23. *"This result indicates the potential difference between [X] and [Y]"* (Feng 2019, §5.3 bullet 2)
24. *"likely because there is a vast number of diverse relations between stocks, only a few are meaningful enough to significantly influence the prices of related stocks"* (Sawhney 2021, §5.2 ¶2) — **conditional hedge with mechanism**

### §6.4 Null / negative-result phrases (most important — Story A is failure-mode-heavy)

25. *"Most anomalies fail to hold up to currently acceptable standards for empirical finance."* (HXZ 2020, abstract sentence 1)
26. *"cannot clear the single test hurdle of the absolute t-value of 1.96"* (HXZ 2020, abstract sentence 2)
27. *"Imposing the higher multiple test hurdle of 2.78 at the 5% significance level raises the failure rate to 82%."* (HXZ 2020, abstract sentence 3)
28. *"Even for replicated anomalies, their economic magnitudes are much smaller than originally reported."* (HXZ 2020, abstract sentence 4)
29. *"capital markets are more efficient than previously recognized"* (HXZ 2020, concluding sentence)
30. *"Rank_LSTM fails to consistently beat SFM and LSTM regarding all evaluation measures, its performance on NYSE w.r.t. MRR is worse than SFM."* (Feng 2019, §5.2 bullet 2)
31. *"The performance of LSTM on the NYSE market w.r.t. IRR is unexpectedly bad."* (Feng 2019, §5.3 ¶ above bullets)
32. *"RSR_I … fails to achieve expected performance with different back-testing strategies"* (Feng 2019, §5.4 bullet 1)
33. *"Again, the performance regarding different evaluation measures is inconsistent."* (Feng 2019, §5.3 ¶ after Table 6)
34. *"This further indicates the less effectiveness of [X] on [Y]"* (Feng 2019, §5.4 bullet 1)
35. *"hypergraph convolutions over inter stock relations do not lead to significant improvements, likely because there is a vast number of diverse relations between stocks, only a few are meaningful enough to significantly influence the prices of related stocks"* (Sawhney 2021, §5.2 ¶2) — **template: "X does not lead to significant improvements, likely because [mechanism]"**
36. *"performs the worst after all hyperedges are removed, essentially degenerating STHAN-SR to a Hawkes Attention + LSTM model that does not account for inter-stock relations"* (Sawhney 2021, §5.3 ¶ below Fig.5) — **degeneration-as-validation framing**
37. *"We observe a degradation in ranking ability as we decompose hyperedges into pairwise edges, with the minimum NDCG@5 being attained when all hyperedges are decomposed"* (Sawhney 2021, §5.3 ¶2) — **template: "We observe a degradation in [metric] as we [intervention]"**
38. *"TA learns distributed scores, capturing an overall downtrend"* (Sawhney 2021, §5.4) — comparative-deficiency framing for the temporal-attention baseline
39. *"the curves are volatile, which indicates that selecting only one stock from more than 1,000 is a highly risk operation"* (Feng 2019, §5.2 final ¶)
40. *"The performance w.r.t. IRR varies a lot under different runs of a method. It is reasonable since the absolute value of daily return ratio varies from 0 to 0.98 in our dataset."* (Feng 2019, §5.3 bullet 1) — **noise as a stated finding**

> **Note**: STHAN-SR's negative-result vocabulary is **thinner than Feng or HXZ** — the model is presented as the dominant winner across all main-table cells with no Wilcoxon-failed comparison. Story A should consciously *expand* the Sawhney vocabulary by combining it with HXZ's "cannot clear", "fail to hold up" verbs.

### §6.5 Limitations openers (the register Story A needs)

41. *"With [confound] mitigated via [method], [percentage]% of [units] cannot clear [hurdle]."* (HXZ 2020, abstract sentence 2 — **template for L1**)
42. *"Even for replicated anomalies, their economic magnitudes are much smaller than originally reported."* (HXZ 2020, abstract sentence 4 — **template for L6**)
43. *"we only show the performance on sectors with the top-5 most stocks"* (Feng 2019, §5.3 ¶ above Table 8 — **template for scope statement**)
44. *"The performance w.r.t. IRR varies a lot under different runs of a method."* (Feng 2019, §5.3 bullet 1 — **template for L4-type variance caveat**)
45. *"In all, capital markets are more efficient than previously recognized."* (HXZ 2020, concluding sentence — **template for reversed-conclusion**)
46. *"We make this assumption to eliminate the temporal dependency of the testing procedure for a fair comparison."* (Feng 2019, §5.1.1 ¶ above bullets — **template for protocol-justification**)
47. *"The transaction costs are ignored since the costs for trading US stocks through brokers are quite cheap no matter charging by trades or shares."* (Feng 2019, §5.1.1 ¶) — **template for cost-caveat (Story A should INVERT this**: "Story A explicitly models a 0/3/5/10/20 bps cost ladder, departing from Feng 2019's zero-cost assumption.")
48. *"In future, we aim to design time-evolving hypergraphs to capture dynamic market correlations and incorporate additional data sources such as online news and social media."* (Sawhney 2021, §6) — **template for future-work-as-implicit-limitation**. Story A should NOT use this evasive register; declarative-noun-phrase "L1: …" bullets are required instead. **No STHAN-SR equivalent for a §Limitations opener exists in the PDF — Sawhney 2021 has no §Limitations section.**

### §6.6 Discussion-lift phrases

49. *"It is important to consider appropriate relations suitable for the target market, for example, encoding industry relations on NASDAQ is a suboptimal choice."* (Feng 2019, §5.3 Brief Conclusion 3) — **conditional generalisation**
50. *"Our proposed model can be generalized for spatiotemporal learning over hypergraphs across problems in varying domains, such as traffic prediction and session-based recommender systems."* (Sawhney 2021, §6) — **methodology-portability lift**
51. *"those that model stock interdependence (RSR-I, STHAN-SR) outperform price-only methods (LSTM, DQN, iRDPG), as they capture the spatial correlations amongst movements of related stocks"* (Sawhney 2021, §5.1 ¶3) — **lifting a row-comparison into a class-of-methods claim**
52. *"validating our premise of formulating stock prediction as a learning to rank problem"* (Sawhney 2021, §5.1 ¶2) — **premise-validation lift**
53. *"the result also suggests the worth of introducing risk-oriented criteria into stock ranking tasks in the future"* (Feng 2019, §5.2 final ¶) — **null result → methodology suggestion**
54. *"Such results also indicate that learning to rank techniques emphasizing the top-ranked stocks is worthwhile to be explored in the future."* (Feng 2019, §5.3 ¶ before "Effect of Wiki Relations") — **null → future-method**
55. *"In all, capital markets are more efficient than previously recognized."* (HXZ 2020, concluding sentence) — **the single-sentence discipline-level lift**

### §6.7 Bonus: contributions-list bullets (template for §1 ¶4)

56. *"We propose a novel neural network-based framework, named *Relational Stock Ranking*, to solve the stock prediction problem in a learning-to-rank fashion."* (Feng 2019, §1 ¶7 bullet 1)
57. *"We devise a new component in neural network modeling, named *Temporal Graph Convolution*, to explicitly capture the domain knowledge of stock relations in a time-sensitive manner."* (Feng 2019, §1 ¶7 bullet 2)
58. *"We empirically demonstrate the effectiveness of our proposals on two real-world stock markets, NYSE and NASDAQ."* (Feng 2019, §1 ¶7 bullet 3)
59. *"We propose a novel Spatio Temporal Hypergraph Attention Network that models inter stock relations of varying types and degrees as a hypergraph for stock ranking."* (Sawhney 2021, §1 contributions bullet 1) — **template: "We propose a novel [architecture] that models [relation] as a [structure] for [task]"**
60. *"We combine temporal Hawkes attention with spatial hypergraph convolutions through hypergraph attention to capture correlations in the movements of related stocks and the temporal evolution of their historical features."* (Sawhney 2021, §1 contributions bullet 2) — **template: "We combine [mechanism 1] with [mechanism 2] through [bridge] to capture [aspect 1] and [aspect 2]"**
61. *"Through experiments on three real-world stock indexes in NYSE, NASDAQ, and TSE markets, over 2,852 stocks spanning over 1,174 trading days, we demonstrate STHAN-SR's applicability to quantitative stock trading."* (Sawhney 2021, §1 contributions bullet 3) — **template: "Through experiments on [N markets], over [N units] spanning over [N days], we demonstrate [name]'s applicability to [task]"**

---

## Appendix A: Notes on Source Access

- **Feng 2019**: fully read via arXiv 1809.09441 PDF (20 pp.) — all 20 pages were rendered as images via the Read tool and the full text was extracted. All quotes above are verbatim from the PDF.
- **Sawhney 2021 STHAN-SR**: fully read via the AAAI 2021 proceedings PDF (pp.497-504, 8 pp.) downloaded locally on 2026-05-28. All 8 pages were rendered as images via the Read tool and the full text was extracted. All quotes above are verbatim from the PDF.
- **Cui 2021 HGTAN (arXiv 2107.14033)**: fully read via arXiv PDF (14 pp.) — retained as a secondary reference for the small number of devices Sawhney 2021 does not display (Algorithm-1 box; t-SNE visualisation). Note that Cui 2021 was *withdrawn* by the authors per arXiv listing; the technical content is unaffected and the writing-craft patterns remain valid as a 2021 hypergraph-finance specimen.
- **HXZ 2020**: abstract verbatim from multiple search-engine excerpts of the published RFS version; concluding sentence verbatim from multiple summaries; section-structure descriptions follow standard RFS empirical-replication convention and are flagged where I do not have direct text.
- For any quote where the citation reads `[per … convention]`, I lacked direct text access and the claim is a structural-convention inference, not a verbatim quote.
