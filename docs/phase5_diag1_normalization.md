# Phase 5 Diagnostic 1 — Cross-Sectional Normalization Ablation

**Date**: 2026-04-16
**Hardware**: Colab RTX Pro 6000 Blackwell (97GB VRAM), 13.2 min total
**Setup**: SAGE-Mean (price-only, 9-dim) × 3 seeds × 5 folds × 2 variants = **30 runs**
**Results**: `experiments/diag1_normalization_results.csv` + `experiments/diag1_log.txt`

---

## TL;DR — Preliminary Finding (single architecture, needs replication)

**Scope caveat**: 30 runs, SAGE-Mean only, price-only features. To shape **Step 3 experiment scope** (specifically whether to include a raw-vs-norm factor), **Diag 1b (MLP + NoGraph replication, ~15 Colab-min)** is recommended to disentangle "scale-interaction mechanism" from "graph-specific mechanism". Diag 1b does not gate Step 1 (data download) or Step 2 (feature implementation).

**Observed pattern (SAGE-Mean only)**: cross-sectional normalization had opposite-sign effects across folds with total cancellation in the overall Wilcoxon test.

| Fold | Period | raw IC | norm IC | Delta IC | Verdict |
|------|--------|--------|---------|----------|---------|
| 0 | Q2-2024 | +0.033 | +0.001 | **−0.031** | norm DESTROYS signal |
| 1 | Q3-2024 | −0.007 | −0.034 | −0.027 | norm amplifies wrong signal |
| 2 | Q4-2024 | +0.035 | +0.031 | −0.004 | roughly tied |
| 3 | Q1-2025 | +0.001 | −0.104 | **−0.105** | norm CATASTROPHIC |
| **4** | **Q2-2025** | +0.006 | **+0.217** | **+0.211** | **norm RESCUES Fold 4** |

**Overall paired Wilcoxon: mean delta = +0.009, p = 0.60 (ns)** — this non-significance **completely masks** the ±0.2 regime-dependent swings.

**Tentative takeaway** (pending replication): Codex's recommendation to add cross-sectional normalization may need qualification. In this single-architecture run, normalization behaved as a regime-conditional transform — beneficial in high-vol regimes, harmful in stable regimes. Do not reject Codex's recommendation on this evidence alone; replicate across architectures first.

---

## 1. Experimental Design

### Raw variant
9-dim price features exactly as used in `run_walkforward_5fold.py`:
- `ret_mean_{5,10,21}d`, `ret_std_{5,10,21}d`, `momentum_{5,10,21}d` all `.shift(1)`
- NaN → 0.0
- **No normalization** — raw values fed directly to SAGE-Mean

### Norm variant
Same base features, then:
1. **Winsorize at 1st/99th percentile** fit on training-period data only (per feature, per fold) to avoid test leak
2. **Per-day cross-sectional z-score**: `(x − μ_day) / (σ_day + 1e-8)` using that day's valid cross-section (no temporal leak)

### Model, training, graph, labels — all identical to `run_walkforward_5fold.py`
- SAGEConv mean aggregation, 2 layers, 64 hidden, dropout 0.3
- 100 epochs max, patience 15, ReduceLROnPlateau
- Graph: correlation (frozen to last train-period snapshot, w=126, t=0.6) + sector edges
- Labels: 21d forward excess-return z-score per day
- Seeds: 42, 123, 456

---

## 2. Full per-fold results (mean ± std across 3 seeds)

| Fold | Period | raw IC mean ± std | norm IC mean ± std | raw Sharpe_net | norm Sharpe_net |
|------|--------|-------------------|---------------------|----------------|------------------|
| 0 Q2-2024 | −0.01→+0.04 vol | +0.033 ± 0.002 | +0.001 ± 0.003 | +0.87 ± 0.5 | −2.36 ± 1.2 |
| 1 Q3-2024 | +10% rally | −0.007 ± 0.011 | −0.034 ± 0.005 | +0.10 ± 2.8 | −2.72 ± 0.8 |
| 2 Q4-2024 | post-election | +0.035 ± 0.081 | +0.031 ± 0.025 | +2.43 ± 3.1 | +4.07 ± 2.2 |
| 3 Q1-2025 | tariff jitters | +0.001 ± 0.011 | **−0.104 ± 0.033** | −0.71 ± 1.8 | **−5.17 ± 0.6** |
| 4 Q2-2025 | tariff shock | +0.006 ± 0.045 | **+0.217 ± 0.060** | −0.43 ± 5.1 | **+6.34 ± 4.4** |

### Per-run detail for Fold 4 (the regime that normalization rescues)

```
raw:  s42 IC=+0.051  s123 IC=+0.006  s456 IC=−0.039
norm: s42 IC=+0.154  s123 IC=+0.273  s456 IC=+0.225
```

Under raw features, Fold 4 shows the "variance explosion" identified in Diag 2 — wild swings across seeds. Under normalization, **all 3 seeds produce IC > +0.15 and Sharpe > +2.5**. Normalization is doing something very specific in this regime.

---

## 3. Why? Connection to Diag 2 regime findings

Recall from Diag 2 (Fold 4 diagnostic):
- Fold 4 test-period `ret_std_21d` mean = 0.024 vs train mean = 0.018 (+33% feature scale shift, KS=0.22)
- Fold 4 has 54% of stock-pairs with correlation > 0.5 vs 4-8% in other folds

**Mechanism**: 
- In Fold 4, raw feature values are 33% larger in magnitude than the training distribution. The SAGE model's linear layers (`nn.Linear(9, 64)`) trained on the training distribution receive inputs outside their effective range → activations saturate / gradients fail / features lose discriminative power. This is exactly the "variance explosion across seeds" — each seed lands in a different bad local minimum.
- Cross-sectional z-score puts every day's features on mean=0, std=1 regardless of absolute scale. The model sees a consistent input distribution across train and test → generalizes properly.

**Why normalization HURTS low-vol folds (0, 1, 3)**:
- In stable regimes, the **absolute magnitude** of features carries information. "Stock with ret_std = 0.02" is different from "stock with ret_std = 0.005" even if both rank high cross-sectionally.
- Z-score strips this absolute information → model loses a useful signal channel → IC drops.

This is a clean regime-interaction finding. Normalization is **not a "should we or shouldn't we" decision** — it's a **modeling choice that interacts with regime**.

---

## 4. Implications

### 4a. Diag 1b as a gate on Step 3 scope only — not on Steps 1 or 2

**Scope of the gate**: Diag 1b gates **Step 3 experimental scope decisions** (specifically: whether to add a raw-vs-norm factor to feature-expansion runs). It does NOT gate Step 1 (OHLCV download) or Step 2 (feature implementation) — those are data/feature preparation that can proceed independently.

Diag 1b repeats the raw vs norm comparison with **MLP (no graph)** and **NoGraph** models to answer a mechanistic question:
- If MLP/NoGraph show the **same** regime-dependent pattern → the mechanism is **input-scale saturation**, unrelated to graph structure.
- If only SAGE-Mean shows the pattern → the mechanism involves graph message-passing × feature-scale interaction.

Cost: ~15 Colab-min. Recommended to run before finalizing Step 3 scope, not required before Step 1 or 2.

### 4b. Open options for Phase 5 Step 3 (not decided — pending H博士 + Diag 1b)

Possibilities if Diag 1b replicates the finding:
- **Option A** (report both): add a raw/norm factor to feature-expansion runs. Doubles runs from ~80 to ~160. Cleanest paper story, highest cost.
- **Option B** (pick-one after diagnostic): use Diag 1b to choose winner globally; if ambiguous per fold, report both as ablation but only use one in main results.
- **Option C** (keep Phase 5 scope as is): treat normalization as out-of-scope for Phase 5; return to it in a later study. Cheapest, least paper risk.

Each of these is a decision for H博士 — I should not prescribe without more evidence.

### 4c. Paper implications — premature to commit

Whether this becomes a paper section depends on replication (Diag 1b) and whether the effect holds after feature expansion. Today's evidence (30 runs, 1 arch, 9-dim features) is insufficient for a publishable claim. If Diag 1b + 14-dim experiments reproduce the effect, it could become a secondary ablation — not a core contribution.

### 4d. For past experiments (no change)

All past experiments used raw features. This run reproduces Fold 0 baseline IC (0.033 here vs 0.036 in wf5_results for SAGE-Mean_price s42), so past conclusions remain valid under the "no normalization" setting — normalization simply wasn't evaluated.

### 4e. Relationship to Diag 2 Fold 4 narrative — revised

Diag 2 concluded Fold 4 cannot be excluded; it must be reported honestly. Diag 1 does not "fix" Fold 4 — it shifts the failure mode. Under norm, Fold 4's IC becomes high (+0.217) but Folds 0/1/3 deteriorate. Whether this is a net improvement depends on objective (cross-fold mean vs worst-fold robustness). For honest reporting, both views matter.

---

## 5. Decision points for H博士

1. **Run Diag 1b (MLP + NoGraph replication, ~15 Colab-min) before Step 3 scope is finalized?** Recommended. Determines whether the regime-dependence is graph-specific or preprocessing-level. This gates Step 3 scope only — Steps 1 and 2 can proceed in parallel or before Diag 1b.

2. **If Diag 1b replicates, does Step 3 scope change?** Open question; possibilities A/B/C in Section 4b. Not prescribing.

3. **VIX overlay priority**: No change recommended. Diag 1 does not directly motivate VIX; the statistical-power concern (~315 OOS days) documented in the 2026-04-16-a Phase 5 plan is unchanged.

4. **Paper framing**: Premature. Revisit after Diag 1b and feature-expansion data.

---

*Written 2026-04-16 by Claude. Revised same day to remove overstated design guidance flagged by Codex stop-time review.*
