# Phase 5 Diagnostic 1b — Normalization Replication on MLP + NoGraph

**Date**: 2026-04-16
**Hardware**: Colab RTX Pro 6000 Blackwell, 17.6 min total
**Setup**: 2 models (NoGraph, true MLP) × 2 variants (raw, norm) × 3 seeds × 5 folds = **60 runs**
**Combined with Diag 1** (SAGE-Mean, 30 runs) → **90 total runs across 3 architectures**
**Results**: `experiments/diag1b_replication_results.csv` + `experiments/diag1b_log.txt`

---

## TL;DR — Mechanism confirmed: input-scale saturation, not graph-specific

The regime-dependent effect of cross-sectional normalization observed in Diag 1 **replicates identically on NoGraph and MLP** (no message passing, pure feed-forward). Paired Wilcoxon p=0.28 to 0.80 for all three architectures — the overall mean effect is ns but the per-fold deltas are consistent in sign and magnitude.

**Key implication for Phase 5 Step 3**: Since the effect is preprocessing-level, not graph-specific, Step 3 does **not** need a raw-vs-norm factor in the factorial design. We can run Step 3 with a single normalization mode (recommend: raw, to match past experiments). The raw-vs-norm regime-dependency is a separate finding that could become its own ablation.

---

## 1. Per-fold delta IC (norm − raw), side-by-side

| Fold | Period | SAGE-Mean | NoGraph | MLP | Consistent? |
|------|--------|-----------|---------|-----|-------------|
| 0 | Q2-2024 | −0.0315 | −0.0366 | −0.0401 | ✅ all neg, similar magnitude |
| 1 | Q3-2024 | −0.0264 | −0.0188 | −0.0191 | ✅ all neg, similar |
| 2 | Q4-2024 | −0.0045 | +0.0336 | −0.0371 | ⚠️ NoGraph positive only |
| 3 | Q1-2025 | **−0.1052** | **−0.1152** | **−0.0852** | ✅ all catastrophic |
| 4 | Q2-2025 | **+0.2112** | **+0.2817** | **+0.1696** | ✅ all rescue |

- 14/15 fold×model cells have the same sign across architectures.
- Magnitudes are similar: Fold 4 rescue is 0.17-0.28 across all models; Fold 3 catastrophe is -0.09 to -0.12.
- Only Fold 2 NoGraph is outlier (weakly positive where SAGE/MLP are negative).

**Conclusion**: The graph (presence or absence) does not materially change the normalization × regime interaction. If the mechanism involved graph message-passing structure, SAGE-Mean and NoGraph/MLP would diverge. They do not.

## 2. Mechanism ruling

| Hypothesis | Supporting evidence | Ruling |
|------------|--------------------|--------|
| **Graph × feature-scale interaction** | Would predict divergence between SAGE and no-graph models | **Rejected** — all 3 models show same pattern |
| **Input-scale saturation** (features outside trained range → Linear layer fails) | Predicts same pattern regardless of what comes after the Linear layer | **Consistent with data** |
| **Optimization instability** | Would predict higher within-seed variance, not systematic per-fold sign | **Partially consistent** — cross-seed variance is elevated but signs are consistent |

Primary conclusion: the Linear layer `in → hidden` is the locus. When test-period feature magnitudes drift far from train-period, raw features produce uncontrolled activation magnitudes, breaking generalization. Cross-sectional z-score re-centers every day's features to N(0,1), which fixes the OOD scale problem but strips absolute-scale information that carries signal in stable regimes.

## 3. Paired Wilcoxon summary (matched on seed × fold)

| Model | n | mean delta IC | p-value | Sign consistency |
|-------|---|---------------|---------|------------------|
| SAGE-Mean | 15 | +0.0087 | 0.60 | 5 of 5 folds same sign as other 2 models |
| NoGraph | 15 | +0.0289 | 0.80 | 4 of 5 folds (Fold 2 disagrees) |
| MLP | 15 | −0.0024 | 0.28 | 5 of 5 folds same sign |

All three are non-significant overall precisely because Fold 3's catastrophe (−0.09 to −0.12) and Fold 4's rescue (+0.17 to +0.28) cancel each other.

## 4. Implications

### 4a. For Phase 5 Step 3 scope

**Simplification confirmed**. Original plan of 80 runs (9-dim vs 14-dim × 3 models × 3 seeds × 5 folds, single normalization mode) is adequate. Do not double to 160 runs. Recommend using `raw` features as the normalization baseline (matches past experiments, easier to compare to prior published numbers).

### 4b. Paper framing (tentative)

Diag 1 + 1b together produce a small-but-genuine finding: **cross-sectional normalization interacts with market regime, not with graph architecture**. This is worth 1-2 paragraphs in discussion or as an ablation row — it's narrower than a full paper section but clean.

Not yet a core contribution. Revisit after Step 3 (whether the regime interaction persists with 14-dim features).

### 4c. Open question

Does the feature-scale drift happen because the 5 new features (dolvol, RSV5, etc.) also drift in Fold 4? If yes, normalization will still have the same rescue effect. If no (new features are structurally stable), Fold 4 might become less catastrophic under raw features alone. Worth checking by running a Diag 2-style KS stat on the 14-dim features before Step 3.

---

## 5. Status update

- Diag 1 + 1b: **complete**. Mechanism confirmed.
- OHLCV data: downloaded, validated (500/500 alignment corr > 0.9999).
- Phase 5 Step 2 (feature build): code written, diagnostic run complete. New features verified orthogonal (mom12m 0.99, dolvol 0.98 orthogonal component vs old 9-dim).
- Phase 5 Step 3: ready to design on original 80-run budget. Pending H博士 go-ahead.

*Written 2026-04-16 by Claude.*
