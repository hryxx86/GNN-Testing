# Analysis Log

> **分析发现记录。** 每次分析的结果和观察。每个条目与 `progress.md` 和 `plan.md` 时间对齐。

---

## 2026-02-27-b: Phase C v1 — Why AUC ≈ 0.50?

→ progress: `2026-02-27-b` | plan: `2026-02-27-b`

### Context
6 experiments on 1.7M news events, 502 S&P 500 stocks. Data quality verified.

### Results

| Experiment | Val AUC | Test AUC |
|-----------|---------|----------|
| B1: LR + FinBERT (768-dim) | 0.5018 | 0.4976 |
| B2: LR + Sentiment (4-dim) | 0.5044 | 0.5027 |
| A1: GNN news→stock only | 0.5085 | 0.4913 |
| A2: + correlation edges | 0.5122 | 0.4949 |
| A3: + sector edges | 0.5133 | 0.4961 |
| Full: all 3 edge types | 0.5133 | 0.5069 |

### Observations
1. FinBERT embeddings → zero predictive power for raw next-day returns
2. Sentiment scores → also near-random
3. Graph structure → marginal +1.6% (Full vs A1), but on zero-signal features
4. Val > Test consistently → slight temporal shift or overfitting

### Hypotheses
- Label noise: `return > 0` is coin flip for |return| < 0.5%
- Market beta confound: most stocks follow SPY direction
- Title too short (~15 words) for 768-dim embeddings
- No event quality filtering: 1.7M includes low-relevance news

---

## 2026-02-27-c: Diagnostic Cells D.1 + D.2 — COMPLETED

→ progress: `2026-02-27-c` | plan: `2026-02-27-c`

*(Cells written, awaiting run — see 2026-03-03-a for results)*

---

## 2026-03-03-a: D.1 + D.2 Diagnostic Results — FinBERT Signal Near Zero

→ progress: `2026-03-03-a` | plan: `2026-03-03-a`

### D.1: Data-Level Diagnostics

**1. Label Noise**
- |return| < 0.5%: **26.5%** of events (near-random noise zone)
- |return| < 1.0%: **48.0%** of events
- Return mean=0.077%, std=2.35%, median=0.055%
- Pos rate by |return| bucket:

| Bucket | Count | Pos Rate |
|--------|-------|----------|
| <0.5% | 445K | 0.494 (coin flip) |
| 0.5-1% | 365K | 0.508 |
| 1-2% | 456K | 0.532 |
| 2-5% | 358K | 0.522 |
| >5% | 69K | 0.525 |

**2. Sentiment-Direction Alignment**
- Positive news (>0.7 conf): alignment = **51.6%** (barely above random)
- Negative news (>0.7 conf): alignment = **48.9%** (slightly anti-predictive)
- FinBERT sentiment has near-zero predictive power at all confidence levels

**3. Per-Sector**
- IT dominates: 420K events (24.8%)
- All sectors have pos_rate close to 50%
- Sector distribution is imbalanced but not broken

**4. Temporal Stability**
- 2022 Q2: pos_rate = 45.7% (bear market), mean_return = -0.37%
- 2023 Q4: pos_rate = 55.9% (rally), mean_return = +0.21%
- 2025 Q3: 262K events (anomalous volume spike)
- Clear regime shifts — static model assumption is problematic

### D.2: Model Prediction Diagnostics (LR + FinBERT, Test Set)

**Overall LR Test AUC: 0.4976** (below random)

**5. Prediction Score Distribution**
- Mean separation between pos/neg labels: **-0.00030** (essentially zero)
- LR cannot separate the two classes at all

**6. Per-Sector AUC**

| Sector | AUC | N Events |
|--------|-----|----------|
| Utilities | 0.512 | 9K |
| Health Care | 0.505 | 35K |
| Communication Services | 0.503 | 65K |
| Real Estate | 0.501 | 6K |
| Financials | 0.500 | 59K |
| Consumer Staples | 0.497 | 34K |
| Energy | 0.497 | 13K |
| Information Technology | 0.497 | 175K |
| Industrials | 0.495 | 38K |
| Consumer Discretionary | 0.492 | 57K |
| Materials | 0.485 | 8K |

- Best: Utilities 0.512 (only 9K events — likely noise)
- No sector exceeds 0.52 with meaningful sample size

**7. AUC by Sentiment Confidence**

| Confidence | AUC | N Events |
|------------|-----|----------|
| <0.3 | 0.497 | 243K |
| 0.3-0.5 | 0.502 | 39K |
| 0.5-0.7 | 0.497 | 45K |
| >0.7 | 0.496 | 174K |

- **No improvement from high confidence.** High-confidence FinBERT is equally useless.

**8. AUC by Return Magnitude**

| |Return| Bucket | AUC | N Events |
|----------------|-----|----------|
| <0.5% | 0.504 | 142K |
| 0.5-1% | 0.496 | 110K |
| 1-2% | 0.493 | 134K |
| 2-5% | 0.494 | 97K |
| >5% | 0.496 | 17K |

- **No improvement for large moves.** Even for |return| > 5%, AUC = 0.496.

### Key Conclusion

**The original hypothesis is REJECTED.** Neither high-confidence sentiment nor large-return events show improved AUC. FinBERT title-level sentiment has zero predictive power for next-day S&P 500 returns at event level, across ALL conditions tested.

### Go/Pivot/Stop Assessment (per plan_v2.md criteria)

| Criterion | Threshold | Actual | Met? |
|-----------|-----------|--------|------|
| **Go** | Any baseline in any bucket AUC > 0.52 | Max = 0.512 (Utilities, 9K events) | **Borderline NO** |
| **Pivot** | |return| > 2% subset AUC > 0.54 | AUC = 0.494-0.496 | **NO** |
| **Stop** | All conditions all baselines ~ 0.50 | Yes | **YES** |

**However**: This is only LR + FinBERT. We have NOT yet tested XGBoost with momentum features, nor the full D+ baseline matrix. The Stop criteria says "all baselines" — we need to complete D.2 baseline matrix before making the final call.

---

## 2026-03-03-b: Literature Review — Why Published Papers Report Good Results and We Don't

-> progress: `2026-03-03-b` | plan: `2026-03-03-b`

### Papers Reviewed

| Paper | Venue | Stocks | Market | News? | Prediction Target | Best Metric |
|-------|-------|--------|--------|-------|-------------------|-------------|
| THGNN | CIKM 2022 | ~300-500 | China CSI 300/500, US S&P 500 | NO (price only) | Return ranking | ARR -0.015 (CSI300), +0.048 (CSI500) |
| DGRCL | ICAART 2025 | 1,026 (NASDAQ) / 1,737 (NYSE) | US | NO (price+volume) | Next-day binary | 53.06% acc (NASDAQ), 54.07% (NYSE) |
| DASF-Net | JRFM 2025 | **12** | US S&P 500 subset (4 sectors) | YES (FinBERT) | Price regression (MSE) | 91.6% MSE reduction vs baselines |
| ChatGPT-GNN | KDD WS 2023 | **30** (DOW 30) | US | YES (ChatGPT on headlines) | 3-class (up/down/neutral, +/-1%) | F1=0.41 (weighted) |
| Kengmegni 2024 | SSRN | S&P 500 | US | YES (FinBERT) | Short-term return | Sentiment = no robust predictive power |
| Sentiment-Size Nexus 2025 | JBA | Large/mid/small cap | India+Asia | Yes (Doc2Vec+SVM) | Index-level | Strong for large/mid-cap indices (NOT individual stocks) |

### Critical Findings

**1. Stock Universe Size is THE Biggest Confound**

Papers reporting strong results almost always use tiny, cherry-picked universes:
- ChatGPT-GNN: 30 stocks (DOW 30) — most covered, most liquid
- DASF-Net: 12 stocks from 4 sectors — extreme cherry-picking
- Our experiment: 502 S&P 500 stocks — 17x to 42x larger universe

With 12-30 stocks, random variation can produce seemingly meaningful AUC/accuracy. With 502 stocks, noise averages out and the true (near-zero) signal is revealed.

**2. Most GNN Papers Do NOT Use News At All**

- THGNN: Price-only, no NLP. The authors explicitly say NLP relation extraction is unreliable.
- DGRCL: Price+volume only, dynamic graphs from DTW on volume volatility.
- These papers show GNN graph structure adds marginal value (53-54% acc vs 50-51% baselines) even without any text.
- Our GNN experiments (A1-Full) also show marginal graph structure benefit (~0.5-1.5%), consistent with these papers.

**3. Papers That Use NLP Report Modest Results**

- ChatGPT-GNN on DOW 30: Weighted F1 = 0.41 on 3-class task. Random baseline for 3-class is ~0.33, so actual lift is modest.
- DASF-Net: Reports MSE reduction (regression), NOT classification accuracy. 91.6% MSE reduction sounds impressive but this is price regression on 12 stocks, not direction prediction on 500.

**4. DGRCL's "53% Accuracy" Is Consistent With Our Results**

DGRCL tests on 1,026 NASDAQ stocks (large universe) and gets 53.06% accuracy. This is only ~3% above random. Our AUC ~ 0.50-0.51 on 502 stocks is in the same regime. The difference is they use sophisticated dynamic graph construction (DTW + Zipf thresholding) and contrastive learning, whereas we use simpler static correlation graphs.

**5. Efficient Market Hypothesis: Large-Cap S&P 500 is Maximally Efficient**

Multiple 2024-2025 papers confirm:
- Sentiment scores lack robust predictive power for large-cap stocks (Kengmegni 2024)
- Market behavior is anticipatory: forward-looking implied sentiment captures ~45-50% of return variation, leaving almost nothing for news-reactive models
- Only ~20% of US large-cap active funds beat index (vs ~38% small-cap) — small-cap is less efficient
- News content is predominantly neutral/objective — only a small fraction carries sentiment signal

**6. Signal-to-Noise Problem in Aggregation**

Our setup: 1.7M events -> average ~6.7 events/stock/day -> aggregated to stock-day features.
- Most news is neutral noise that dilutes any signal
- FinBERT on short titles (~15 words) produces shallow sentiment
- Daily aggregation of multiple conflicting sentiments cancels out
- DASF-Net found optimal 3-day aggregation window, suggesting single-day is too noisy

**7. Label Definition Matters Enormously**

- Our label: `return > 0` (raw next-day return) — 26.5% of events have |return| < 0.5% (coin flip zone)
- ChatGPT-GNN: uses +/-1% threshold for up/down, neutral in between — filters out noise zone
- THGNN: uses return ranking (relative performance), not absolute direction
- Market-adjusted returns (`stock - SPY`) would remove beta confound

### Implications for Our Experiment

Our AUC ~ 0.50 is NOT a bug. It is the expected result given:
1. Large universe (502 stocks) on the world's most efficient market (S&P 500)
2. Raw next-day returns as labels (noisy, beta-dominated)
3. FinBERT on short titles (shallow signal)
4. EODHD news (lower quality than Reuters/Bloomberg/Factiva)
5. Event-level prediction (no temporal aggregation)

### What Would Actually Improve Results (from literature)

1. **Market-adjusted labels** (stock - SPY return) — removes market beta
2. **Threshold labels** (+/-1% or return ranking) — eliminates coin-flip zone
3. **Multi-day returns** (3-5 day cumulative) — allows sentiment to propagate
4. **Smaller universe** (30-50 stocks) — concentrates signal, more news per stock
5. **Higher-quality text** (full articles from Reuters/Bloomberg, not just EODHD titles)
6. **LLM over FinBERT** (GPT-4 prompt-based analysis captures nuance better)
7. **Dynamic graph construction** (DTW/correlation-based daily re-estimation)

---

## 2026-03-03-f: Phase 1 Preprocessing Results — Colab Run

→ progress: `2026-03-03-f` | plan: `2026-03-03-f`

### 1a. News Deduplication

- 1,698,182 events → **437,194 stock-days** (3.88:1 compression)
- Average ~3.9 news events merged per stock-day
- Within expected range (250K-500K)

### 1b. Market-Adjusted Labels

| Metric | Raw | Market-adjusted | Change |
|--------|-----|-----------------|--------|
| Pos rate | 0.5164 | 0.4925 | -2.4pp (more balanced) |
| Noise zone (|ret|<0.5%) | 27.6% | 23.0% | -4.6pp (17% relative reduction) |
| Coverage | — | 437,194/437,194 | 100.0% |

**Observations**:
1. Raw label had bullish bias (S&P 500 long-term uptrend); market-adjusted is near-balanced
2. Noise zone reduction is modest (23% still in coin-flip zone) — stock-specific micro-volatility remains a major noise source beyond market beta
3. Pos rate 0.4925 (slightly < 0.50) suggests equal-weight mean is pulled by large-cap stocks, making average stock slightly underperform

### 1c. Momentum/Volatility Features

- 9 features (3 windows × 3 stats): all built correctly
- Lookup table: 622,513 rows
- Coverage: 434,833/437,194 = **99.5%** (0.5% filled with 0)
- Missing records likely from stocks with insufficient early trading history

### Pipeline Summary

```
Input:  1,698,182 events
Output: 437,194 stock-days
  ├─ Embeddings: (437194, 768)
  ├─ Sentiment:  (437194, 3)
  ├─ Momentum:   (437194, 9)
  └─ Labels: market-adjusted (label_raw preserved for ablation)
Processing time: 40.9s
```

### Assessment

Data pipeline is working correctly. Key question remains: **do these fixes push any baseline AUC past the 0.52 Go threshold?** The noise zone reduction from 27.6% → 23.0% is helpful but not dramatic — the signal test depends on the 1d baseline matrix results.

---

## 2026-03-03-g: Phase 1d Baseline Matrix Results — Signal Still Zero

→ progress: `2026-03-03-g` | plan: `2026-03-03-g`

### Baseline Results (market-adjusted labels, deduped data, 437K stock-days)

| Baseline | Val AUC | Test AUC | Notes |
|----------|---------|----------|-------|
| B1: LR + FinBERT (768-dim) | 0.4988 | 0.4993 | Random — same as Phase C |
| B2: LR + Sentiment (3-dim) | 0.5001 | 0.5031 | Random |
| B3: LR + Sent+Momentum | 0.5182 | 0.4965 | **Overfitting** (val >> test) |
| B4: LR + Momentum only | 0.5178 | 0.4987 | **Overfitting** (val >> test) |
| B5: XGBoost (Sent+Mom) | 0.5034 | 0.5046 | Best test, still random |

### Key Findings

1. **FinBERT still zero signal** even with market-adjusted labels and dedup
2. **Momentum features overfit**: Val AUC ~0.518 but test AUC drops below random (~0.497). Classic regime-change overfitting — momentum patterns in validation period don't persist to test period
3. **XGBoost (0.5046)** is the most "honest" result due to built-in regularization, but still far below 0.52 Go threshold
4. All test AUCs in range [0.4965, 0.5046] — no baseline exceeds 0.51

### Go/Stop Assessment

| Criterion | Threshold | Actual | Met? |
|-----------|-----------|--------|------|
| **Go** | Any test AUC > 0.52 | Max = 0.5046 | **NO** |
| **Pivot** | XGBoost+momentum > 0.52 | 0.5046 | **NO** |
| **Stop** | All baselines ≈ 0.50 after signal fix | Yes | **YES** |

### Interpretation

After three signal fixes (dedup, market-adjusted labels, 9 momentum features), event-level next-day excess return direction remains unpredictable on S&P 500 at full scale. This is consistent with:
- Efficient Market Hypothesis for large-cap US equities
- DGRCL (53% acc on 1K+ stocks — same regime)
- Kengmegni 2024 (FinBERT sentiment = no robust predictive power for S&P 500)

### Missing Data Points (before final Stop decision)

1. ~~Selective Top-10%/5% AUC not yet computed~~ → Done, see 2026-03-03-h
2. ~~GNN v2 on new data not yet run~~ → Done, see 2026-03-03-h
3. Phase 2 LLM features not yet tested

---

## 2026-03-03-h: Selective AUC + GNN v2 — Final Stop Confirmation

→ progress: `2026-03-03-h` | plan: `2026-03-03-h`

### Selective AUC Results (all methods, market-adjusted + deduped data)

| Method | Full | @50% | @20% | @10% | @5% |
|--------|------|------|------|------|-----|
| B1: LR+FinBERT | 0.5000 | 0.5014 | 0.5000 | 0.5071 | 0.5154 |
| B2: LR+Sent | 0.5010 | 0.5024 | 0.4955 | 0.4890 | 0.4773 |
| B3: LR+Sent+Mom | 0.5008 | 0.4962 | 0.4861 | 0.4921 | 0.4866 |
| B4: LR+Mom | 0.4991 | 0.4927 | 0.4951 | 0.4898 | 0.4928 |
| B5: XGB | 0.5045 | 0.5043 | 0.5021 | 0.5041 | 0.4988 |
| GNN Full (780-dim) | 0.5002 | 0.4998 | 0.4994 | 0.5026 | 0.5023 |

### Go/Stop Final Assessment

| Criterion | Threshold | Actual | Met? |
|-----------|-----------|--------|------|
| Full AUC > 0.52 | 0.52 | 0.5045 (XGB) | **NO** |
| Top-10% AUC > 0.54 | 0.54 | 0.5071 (B1) | **NO** |
| Top-5% AUC | — | 0.5154 (B1, ~2K samples, within noise) | **NO** |

### Key Findings

1. **GNN v2 with 780-dim features**: Test AUC = 0.5002 — graph structure adds zero value even with momentum features
2. **No tail signal**: Selective prediction at 5% coverage yields max AUC = 0.5154 (noise-level for ~2K samples; 95% CI ≈ ±0.02)
3. **Momentum features hurt selective AUC**: B3/B4 @20%/@10% < 0.50 — model is "most confident" on its worst predictions
4. **XGBoost most stable**: flat ~0.50 across all coverages — regularization prevents overfitting but confirms no signal

### Conclusion

**STOP condition confirmed.** After:
- News deduplication (1.7M → 437K)
- Market-adjusted labels (noise zone 27.6% → 23.0%)
- 9 momentum/volatility features
- 5 baseline methods + GNN
- Selective prediction at 4 coverage levels

Event-level next-day excess return direction is unpredictable on S&P 500 at full scale. Consistent with EMH for large-cap US equities.

### Remaining option: Phase 2 LLM features (~$0.45) — different signal dimension (impact prediction vs sentiment)

---

## 2026-03-04-a: Phase 2 LLM Results — GPT-4o-mini Also Has Zero Signal

→ progress: `2026-03-04-a` | plan: `2026-03-04-a`

### Context

GPT-4o-mini structured output on 7K dev-holdout events (2023-Q4). Testing whether LLM impact prediction provides signal beyond FinBERT sentiment. Run via OpenRouter API, 0 errors.

### LLM Output Distributions

| Field | Distribution |
|-------|-------------|
| Impact | medium 49.5%, low 37.6%, high 13.0% |
| Direction | neutral 44.3%, positive 37.1%, negative 18.6% |
| Reasoning | sentiment 58.1%, other 21.2%, earnings 10.2%, macro 10.2%, technical 0.2% |
| Avg confidence | 0.661 |

**Observation**: LLM classifies 58% of reasoning as "sentiment" — it's largely doing the same thing FinBERT does, just with more overhead.

### 5-Fold CV AUC Comparison

| Feature Set | AUC | ±std |
|-------------|-----|------|
| FinBERT sentiment (3d) | 0.5025 | 0.0131 |
| LLM structured (10d) | 0.5034 | 0.0137 |
| Combined (13d) | 0.5019 | 0.0097 |
| FinBERT embedding (768d) | 0.5112 | 0.0139 |
| LLM + embedding (778d) | 0.5102 | 0.0128 |

**LLM vs FinBERT delta: +0.0009** — within noise, no signal.

### Impact-Level Subset Analysis

| Impact | N (%) | Pos Rate | AUC |
|--------|-------|----------|-----|
| High | 908 (13.0%) | 0.537 | **0.4762** ± 0.0446 |
| Medium | 3,462 (49.5%) | 0.557 | 0.5090 ± 0.0110 |
| Low | 2,630 (37.6%) | 0.543 | 0.5034 ± 0.0078 |

**Critical finding**: High-impact events have WORSE-than-random AUC (0.4762). The LLM's "high impact" classification is anti-predictive. This suggests the market prices in high-impact news fastest — by the time the LLM labels it "high impact," the move has already happened.

### LLM Direction Prediction

| Metric | Value |
|--------|-------|
| Non-neutral predictions | 3,902 / 7,000 (55.7%) |
| Direction accuracy | 0.5208 (random = 0.50) |
| High-impact + directional | n=898, accuracy = **0.4989** (random) |

The LLM cannot predict return direction, even for events it considers high-impact and directional.

### Go/Stop Assessment

| Criterion | Threshold | Actual | Met? |
|-----------|-----------|--------|------|
| LLM delta > 0.02 AUC | 0.02 | 0.0009 | **NO** |
| High-impact AUC > 0.54 | 0.54 | 0.4762 | **NO** |
| Direction accuracy > 0.55 | 0.55 | 0.5208 | **NO** |

**STOP confirmed. Skip full-scale LLM run ($19 saved).**

### Cumulative Conclusion — All Avenues Exhausted

| Phase | What we tried | Result |
|-------|--------------|--------|
| Phase C v1 | Raw FinBERT + GNN (6 configs) | All AUC ≈ 0.50 |
| Phase 1a | News deduplication (1.7M → 437K) | No improvement |
| Phase 1b | Market-adjusted labels | Noise zone 27.6% → 23.0%, no AUC lift |
| Phase 1c | 9 momentum/volatility features | Overfits on val, no test improvement |
| Phase 1d | 5 baselines + GNN v2 (780-dim) | Max test AUC = 0.5046 (XGB) |
| Phase 1e | Selective AUC @5%/10%/20%/50% | Max = 0.5154 (noise for 2K samples) |
| **Phase 2** | **GPT-4o-mini structured output** | **AUC = 0.5034, delta = +0.0009** |

**Final verdict**: Event-level next-day excess return direction on S&P 500 is unpredictable with NLP features (FinBERT or GPT-4o-mini), momentum features, GNN graph structure, or any combination thereof. This constitutes strong empirical evidence for the Efficient Market Hypothesis in large-cap US equities.

### Path Forward

The remaining viable path is a **negative result paper** (EMH evidence) with the extensive experimental record as the contribution. Discussion with H博士 needed on paper framing.

---

## 2026-03-05-a: Phase B Dynamic Graph Parameter Analysis — Best Config Identified

→ progress: `2026-03-05-a` | plan: `2026-03-05-a`

### Context

Analyzed 636 dynamic graph snapshots from Phase B sensitivity analysis across 3 window sizes (63/126/252 trading days) × 4 thresholds (0.4/0.5/0.6/0.7), step=21d, covering 2021-04 to 2026-01.

### Full Parameter Comparison

| Window | Threshold | Mean Edges | Mean Density | Avg Degree | Components | Clustering | Density Std (stability) |
|--------|-----------|-----------|-------------|-----------|-----------|-----------|------------------------|
| 63 | 0.4 | 72,720 | 28.9% | 144.9 | 7 | 0.628 | 0.196 |
| 63 | 0.5 | 41,768 | 16.6% | 83.2 | 21 | 0.565 | 0.155 |
| 63 | 0.6 | 20,155 | 8.0% | 40.2 | 79 | 0.471 | 0.095 |
| 63 | 0.7 | 7,206 | 2.9% | 14.4 | 197 | 0.333 | 0.041 |
| **126** | 0.4 | 68,868 | 27.4% | 137.2 | 12 | 0.652 | 0.192 |
| **126** | 0.5 | 36,592 | 14.5% | 72.9 | 44 | 0.570 | 0.134 |
| **126** | **0.6** | **15,177** | **6.0%** | **30.2** | **125** | **0.453** | **0.064** |
| **126** | 0.7 | 4,244 | 1.7% | 8.5 | 239 | 0.307 | 0.017 |
| 252 | 0.4 | 67,489 | 26.8% | 134.4 | 16 | 0.676 | 0.164 |
| 252 | 0.5 | 30,950 | 12.3% | 61.7 | 64 | 0.571 | 0.098 |
| 252 | 0.6 | 10,291 | 4.1% | 20.5 | 141 | 0.455 | 0.037 |
| 252 | 0.7 | 2,509 | 1.0% | 5.0 | 260 | 0.293 | 0.007 |

### Best Parameter: **window=126, threshold=0.6**

**Selected rationale (5 criteria):**

1. **Density 6.0%** — Moderate: each stock connects to ~30 peers. Not so dense that noise drowns signal (cf. thr=0.4 at 27%), not so sparse that GNN message passing fails (cf. thr=0.7 at 1.7%).

2. **125 connected components** — Acceptable fragmentation: ~half of 502 stocks participate in connected subgraphs. Stocks that are genuinely uncorrelated (e.g., Utilities vs Tech) naturally separate.

3. **Clustering coefficient 0.453** — Balanced: industry clusters are visible but the graph isn't a single dense blob. Supports GNN's ability to learn sector-level patterns.

4. **Temporal stability std=0.064** — Key differentiator: 3× more stable than window=63/thr=0.6 (std=0.095), yet responsive enough to capture regime changes (2022 rate hikes visible in edge count time series). Window=252 is more stable (std=0.037) but too sluggish to react to market shifts.

5. **Regime sensitivity** — Edge count time series shows clear spikes during 2022 Q1-Q2 (correlation surge during sell-off) and recovery in 2023-2024 (market differentiation). This dynamic behavior is exactly what we want for regime-aware prediction.

### Why NOT Other Parameters

| Rejected | Reason |
|----------|--------|
| thr=0.4 (any window) | ~27% density, avg degree 135-145. Graph says "everything correlates with everything" — dilutes meaningful relationships |
| thr=0.7 (any window) | 197-260 components. Majority of stocks are isolated islands — GNN degenerates to MLP |
| window=63 | Density std 2-3× higher than window=126. Too noisy — graph structure changes more from noise than from real market regime shifts |
| window=252 | Too sluggish. 12-month window smooths over regime transitions (2022 bear → 2023 recovery blurred). Density std suspiciously low (0.037) = not capturing real dynamics |
| thr=0.5 (window=126) | 14.5% density, 44 components. Viable alternative but edges still ~2.4× more than thr=0.6. More edges = more noise when correlation ≠ causation |

---

## 2026-03-05-b: Literature Survey — Ranking + Dynamic HGT + Selective Prediction

→ progress: `2026-03-05-b` | plan: `2026-03-05-b`

### Context

Comprehensive literature survey to support v3 research direction pivot. Surveyed 10+ recent papers on GNN stock prediction, ranking targets, and selective prediction.

### Key Papers & Findings

| Paper | Venue | Key Finding for Our Work |
|-------|-------|------------------------|
| MASTER | AAAI'24 | Cross-stock Transformer, 5d ranking, IC=0.064 (CSI300). No graph structure. |
| FinMamba | arXiv'25 | Mamba + dynamic graph, 1d ranking, Sharpe=2.06 (S&P500). No NLP, no heterogeneous edges. |
| MDGNN | AAAI'24 | 3 node types + multi-relation + daily dynamic, IC=0.032 (CSI300). Chinese market only. |
| THGNN | CIKM'22 | Daily dynamic graph + HeteroGAT, IC=4.93%. No NLP, no news nodes. |
| HGAIT | ESWA'25 | Positive/negative correlation heterogeneous edges + inverse Transformer. No NLP. |
| SelectiveNet | ICML'19 | 3-head architecture (pred+selection+aux). Never applied to financial GNN. |
| AUGRC | NeurIPS'24 | Fixes AURC metric for selective prediction evaluation. |
| Sim et al. | arXiv'23 | Chart images + confidence threshold trading. Only financial selective pred paper, non-GNN. |
| Multi-GCGRU | IEEE'24 | Co-occurrence edges outperform fund-holding and supply-chain edges. Supports our edge choice. |
| QuantBench | 2025 | Comprehensive benchmark comparing 20+ stock prediction methods. |

### Critical Insights

**1. DASF-Net "3-Day Optimal" is Misleading**
- The "3-day" in DASF-Net refers to input sentiment aggregation window, NOT prediction horizon
- Only tested on 12 cherry-picked stocks from 4 sectors
- No actual horizon ablation was performed
- Our planned 1d/5d/10d/21d/42d/63d ablation fills a genuine literature gap

**2. Ranking Targets are the Standard**
- Every major paper (MASTER, FinMamba, MDGNN, THGNN) uses ranking/IC evaluation
- Binary direction prediction is NOT how SOTA is measured
- IC > 0.03 is a meaningful threshold; MASTER achieves 0.064 on CSI300
- Our v2 used binary direction → explains why AUC ≈ 0.50 was inevitable

**3. Calendar-Driven is Mainstream**
- All ranking papers predict every stock every day (calendar-driven)
- Event-driven (predict only when news arrives) is NOT standard
- Days without news: use zero vector for news features (MSGCA 2024 approach)
- This change resolves our small-sample-per-day problem

**4. SelectiveNet + GNN = Uncharted Territory**
- SelectiveNet (ICML'19) has 800+ citations but zero financial GNN applications
- Only one financial selective prediction paper exists (chart images, non-GNN)
- This is a clear, publishable gap: first to combine GNN + SelectiveNet for stock prediction
- High risk but high novelty reward

**5. Edge Type Selection**
- Multi-GCGRU (IEEE'24) found: co-occurrence > fund-holding > supply-chain for edge effectiveness
- Supports our 4-edge design: correlation (dynamic) + sector (static) + news mentions + co-occurrence
- No need for external data (FactSet supply chain, SEC 13F holdings) at this stage

### Implications for v3 Design

The literature strongly supports our v3 pivot:
- Ranking prediction is the right task (not binary direction)
- Calendar-driven is the right paradigm (not event-driven)
- HGT is appropriate for heterogeneous multi-edge graphs
- Horizon ablation is a genuine contribution (no paper has done this)
- Selective prediction is the main novelty (no prior work in GNN+finance)

---

## 2026-03-05-e: v3 First Colab Run — Ranking Works, Graph Structure Validates

→ progress: `2026-03-05-e` | plan: `2026-03-05-e`

### Context

First full run of `v3_ranking_pipeline.ipynb` on NVIDIA RTX PRO 6000 Blackwell (102GB VRAM). Calendar-driven ranking prediction on 501 S&P 500 stocks, 5d default horizon, z-score normalized labels.

### Data Pipeline (N1-N2) — All Correct

| Metric | Value |
|--------|-------|
| Valid tickers | 501 (of 502) |
| Total events mapped | 1,538,967 / 1,698,182 |
| News coverage (stock-days) | 58.5% (train), 55.9% (val), 62.7% (test) |
| Time split | Train: 629d, Val: 124d, Test: 396d |
| Correlation snapshots | 54 (density: 2.9%→0.6%) |
| Co-occurrence edges | 2,918,292 total (2325/day) |
| Jaccard stability | Mean=0.631, Std=0.124 |

### N3: Baseline + GNN Ablation Results (5d horizon, test set)

| Model | IC | ICIR | Sharpe_LS | Ann_LS | MaxDD |
|-------|-----|------|-----------|--------|-------|
| B1: Ridge (price 9d) | 0.00476 | 0.026 | 0.624 | 14.88% | 152.76% |
| B2: Ridge (all 781d) | 0.00535 | 0.052 | 0.597 | 8.06% | 79.00% |
| B3: XGBoost | 0.00329 | 0.024 | 0.185 | 2.89% | 76.59% |
| B4: LightGBM | 0.00828 | 0.079 | 0.773 | 10.92% | 44.52% |
| A1: HGT (corr) | 0.01023 | 0.133 | 0.121 | 1.25% | 51.53% |
| A2: HGT (corr+sector) | 0.01177 | 0.156 | 0.994 | 8.91% | 16.42% |
| **A3: HGT (all 4)** | **0.00432** | 0.061 | **-0.314** | -2.83% | 39.29% |
| A4: SAGE (corr+sector) | 0.01571 | 0.152 | 1.038 | 13.51% | 35.08% |
| **A5: GAT (corr+sector)** | **0.02054** | **0.174** | **1.011** | **15.78%** | 38.56% |

**Go/Stop**: Best IC=0.02054 (< 0.03), Best Sharpe=1.038 (> 0.5) → **GO**

### N4: Horizon Ablation — Partially Visible

Only 1d horizon result visible (rest drowned in sklearn warnings):
- HGT 1d: IC=0.00343, ICIR=0.051, Sharpe_LS=3.073, Ann_LS=38.88%

**N5 SelectiveNet**: Not visible in output due to warnings.

### Key Observations

**1. News/co-occurrence edges ADD NOISE, not signal**
- A3 (all 4 edges) is the WORST GNN: IC=0.00432, Sharpe=-0.314
- A2 (corr+sector only) is much better: IC=0.01177, Sharpe=0.994
- This is consistent across architectures: adding news edges hurts ALL models
- Possible reason: news mentions create dense, noisy connections that dilute the informative correlation structure

**2. GAT > SAGE > HGT (same edge configuration)**
- GAT IC=0.02054 vs SAGE IC=0.01571 vs HGT IC=0.01177
- GAT's simpler attention may be more robust than HGT's more complex type-specific attention
- SAGE and GAT both use homogeneous graph (corr+sector merged), while HGT uses heterogeneous
- The heterogeneous distinction between corr and sector edges may not be useful

**3. Graph structure provides genuine signal over baselines**
- Best GNN IC=0.02054 vs Best baseline IC=0.00828 (LightGBM): **2.5× improvement**
- Graph adds +0.01226 IC over flat features — substantial for financial prediction
- This validates the core thesis: stock correlation structure carries predictive information

**4. Ranking approach succeeds where binary direction failed**
- v2 binary direction: AUC=0.50 across ALL models (random)
- v3 ranking: IC>0.01, Sharpe>1.0 for best models
- Confirms literature guidance: ranking is the right task for large-universe stock prediction

**5. A2 HGT (corr+sector) has remarkably low MaxDD (16.42%)**
- Much lower than all other models (35-153% MaxDD)
- The sector edges may provide diversification that reduces drawdowns
- Worth investigating further for risk-adjusted metrics

**6. N4 uses wrong model configuration**
- Code uses HGT with all 4 edges (A3 config = worst GNN)
- Should use GAT with corr+sector (A5 config = best GNN)
- Must fix before next Colab run

### Implications

1. **Drop news/co-occurrence edges**: They hurt. Use only corr+sector edges going forward
2. **GAT replaces HGT**: Simpler architecture, better performance, lower risk
3. **N4 horizon ablation needs re-run**: With GAT (corr+sector), not HGT (all edges)
4. **N5 SelectiveNet needs re-run**: After fixing warnings + N4 model
5. **Ranking approach validated**: Proceed with paper narrative around ranking + selective prediction

---

## 2026-03-06-b: v3 Colab Run 2 — N3-N5 Complete (GAT, Updated Code)

→ progress: `2026-03-06-b` | plan: `2026-03-06-b`

### Context

Second run of `v3_ranking_pipeline.ipynb` on NVIDIA A100-SXM4-40GB (42.4 GB VRAM). Updated code: N4/N5 use GAT(corr+sector), sklearn warnings suppressed, grad_accum=32.

### N3 Run 2 vs Run 1 Comparison

| Model | Run 1 IC | Run 2 IC | Δ |
|-------|----------|----------|---|
| B1-B4 (baselines) | identical | identical | 0 |
| A1: HGT (corr) | 0.01023 | 0.00848 | -0.00175 |
| A2: HGT (corr+sec) | 0.01177 | 0.01447 | +0.00270 |
| A3: HGT (all 4) | 0.00432 | 0.00884 | +0.00452 |
| A4: SAGE (corr+sec) | 0.01571 | 0.01545 | -0.00026 |
| **A5: GAT (corr+sec)** | **0.02054** | **0.00640** | **-0.01414** |

**Critical finding**: GAT IC dropped 69% across runs. SAGE is most stable (CV=2%).

### N4 Horizon Ablation — Full Results

| Horizon | GAT IC | GAT ICIR | GAT Sharpe | LGBM IC | LGBM Sharpe |
|---------|--------|----------|------------|---------|-------------|
| 1d | -0.00104 | -0.013 | 2.468 | 0.00368 | 2.918 |
| 5d | 0.02334 | 0.227 | 1.568 | 0.00828 | 0.773 |
| 10d | **0.03854** | **0.320** | 1.196 | 0.01349 | 0.644 |
| **21d** | **0.04420** | **0.374** | **1.203** | 0.01513 | 0.468 |
| 42d | -0.00912 | -0.144 | 0.071 | 0.03679 | 0.668 |
| 63d | -0.00838 | -0.118 | 0.487 | 0.05207 | 1.256 |

**Key findings**:
1. **GAT 21d IC=0.04420 > 0.03 threshold** — first time exceeding Go criterion for IC
2. **Inverted-U pattern**: GAT peaks at 10d-21d, fails at 1d and 42d-63d
3. **LGBM monotonic**: IC increases with horizon (1d:0.004 → 63d:0.052)
4. **Cross pattern**: GAT > LGBM at 5d-21d (graph structure helps), LGBM > GAT at 42d-63d (individual features dominate)
5. **1d Sharpe anomaly**: High Sharpe (2.5-2.9) but IC ≈ 0 and Ann_LS_net = -41% → pure noise amplified by daily rebalancing

### N5 SelectiveNet — Complete Results (21d horizon)

| Method | IC | ICIR | Sharpe_LS | Ann_LS_net | MaxDD |
|--------|-----|------|-----------|------------|-------|
| Full (100%) | **0.05595** | **0.463** | **1.328** | **16.48%** | 66.67% |
| Threshold @20% | 0.03070 | 0.324 | 0.724 | 5.85% | 74.91% |
| Threshold @50% | 0.05087 | 0.446 | 1.346 | 15.07% | 44.64% |
| **SelectiveNet @5%** | **-0.01544** | -0.202 | -0.672 | -10.30% | 286.94% |
| SelectiveNet @20% | -0.02414 | -0.256 | -0.536 | -9.00% | 242.10% |
| SelectiveNet @50% | -0.00874 | -0.116 | 0.800 | 4.88% | 60.35% |

**Key findings**:
1. **SelectiveNet FAILED**: Negative IC at all coverage levels (5%-50%)
2. **Selection head is anti-correlated**: It selects the stocks where GNN predictions are WORST
3. **Threshold baseline works**: |ranking| > percentile is a valid confidence proxy
4. **Full model (100%) is best**: IC=0.05595, SelectiveRankingGAT's auxiliary loss provides regularization benefit
5. **Selection score distribution**: Heavily right-skewed (0.8-1.0), lacks discrimination
6. **Coverage converged to ~31%** (target was 20%) — lambda=32 insufficient

### Publication Metrics Assessment (Updated)

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Any horizon IC > 0.03 | > 0.03 | **GAT 21d IC=0.04420** | ✅ |
| GNN > LGBM (same horizon) | GNN wins | **21d: 0.044 vs 0.015 (2.9×)** | ✅ |
| Selective > Full | Selective wins | Threshold @20%: 0.031 < Full 0.056 | ❌ |
| Sharpe > 0.5 (net) | > 0.5 | GAT 21d Ann_LS_net=15.11% | ✅ |
| Horizon pattern | Clear trend | **Inverted-U, peak 21d** | ✅ |

**4/5 metrics met.** SelectiveNet contribution point failed.

### Implications

1. **Horizon ablation is THE key contribution** — inverted-U pattern is novel and publishable
2. **SelectiveNet needs rethink** — either report as negative finding or try alternative approaches
3. **Training stability is a concern** — GAT IC varies 0.006-0.021 across runs; Walk-forward CV essential
4. **SAGE may be more reliable than GAT** for production use (stable IC, good Sharpe)
5. **Full SelectiveRankingGAT model (100%) gives best IC=0.05595** — auxiliary loss helps

---
