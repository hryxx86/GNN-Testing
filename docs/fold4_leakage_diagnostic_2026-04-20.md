# Fold 4 Leakage Diagnostic (2026-04-20)

- Fold 4 train_days: 0–965 (n=966)
- Fold 4 test_days: 1047–1108 (n=62)
- Scope: Test 1/2 metrics computed on **all stock nodes** (SAGE-Mean message passing consumes full-graph features; label_valid only enters via scaler fit domain matching Part C and is already baked into IC from `part_c_s8_daily_ic.csv`).
- Scheme G scaler fit: train_days × label_valid stocks (matches `fit_feature_scaler` used in Part C).
- Raw features bit-exact verified against existing clipped .npy.

## Test 1: Per-feature Fold 4 tail concentration

Top-10 features by `|Δ_top|+|Δ_bot|`:

| feature   |   delta_top |   delta_bot |   abs_delta_sum |   n_test_obs |
|:----------|------------:|------------:|----------------:|-------------:|
| MIN60     |      0.0001 |      0.0068 |          0.0069 |        30938 |
| QTLD60    |      0.0035 |      0.0024 |          0.0060 |        31000 |
| MA60      |      0.0026 |      0.0018 |          0.0044 |        31000 |
| MIN30     |     -0.0001 |      0.0040 |          0.0041 |        30938 |
| KLEN      |      0.0038 |     -0.0002 |          0.0039 |        30938 |
| RESI60    |      0.0029 |      0.0006 |          0.0036 |        30973 |
| STD5      |      0.0034 |      0.0000 |          0.0034 |        31000 |
| CNTN20    |     -0.0032 |      0.0000 |          0.0032 |        31000 |
| MA30      |      0.0018 |      0.0013 |          0.0032 |        31000 |
| QTLD30    |      0.0018 |      0.0014 |          0.0032 |        31000 |

- # features with `abs_delta_sum` > 0.05: 0
- max abs_delta_sum: 0.0069

## Test 2: Z-score shift

### (a) Per-feature shift magnitude on Fold 4 (all stock nodes)

- median `delta_mean`: 0.0002
- max `delta_mean`: 0.0120
- median `delta_std`: 0.0003
- max `delta_std`: 0.0141

### (b) Cross-sectional Spearman ρ distribution

- per-feature median ρ: min=1.0000, median=1.0000, max=1.0000
- per-feature 5th-pct ρ: min=0.9975, median=1.0000
- # features with median ρ < 0.95: 0
- # features with 5th-pct ρ < 0.98: 0

### (c) Top-5 drift features (by 1 − ρ_median)

| feature   |   rho_median |   rho_p5 |   delta_mean |   delta_std |
|:----------|-------------:|---------:|-------------:|------------:|
| MIN60     |       1.0000 |   0.9999 |       0.0099 |      0.0082 |
| QTLD60    |       1.0000 |   0.9989 |       0.0015 |      0.0022 |
| MA60      |       1.0000 |   0.9998 |       0.0002 |      0.0004 |
| BETA60    |       1.0000 |   1.0000 |       0.0004 |      0.0010 |
| MIN30     |       1.0000 |   0.9997 |       0.0028 |      0.0006 |

### (d) Z-drift ↔ IC Spearman correlation (per model, n=62)

Canonical domain pairing: MLP ↔ z_drift_lv (no message passing, only label_valid stocks feed into IC); SAGE-Mean ↔ z_drift_all (graph message passing ingests all node features). Both pairings reported for completeness.

- **MLP × z_drift_lv**: ρ = +0.508, p=0.000, n=62 **[[canonical]]**
- **MLP × z_drift_all**: ρ = +0.508, p=0.000, n=62
- **SAGE-Mean × z_drift_lv**: ρ = +0.413, p=0.001, n=62
- **SAGE-Mean × z_drift_all**: ρ = +0.413, p=0.001, n=62 **[[canonical]]**

**Canonical-domain check (stop-hook fix)**: `z_drift_lv` and `z_drift_all` agree to 3 decimals on Fold 4 (label_valid mask filters very few stocks per day), so the MLP headline ρ = +0.508 and SAGE headline ρ = +0.413 hold under the correct per-model domain.

## Summary

**Mixed signal** (per plan decision rule → Path A):

| Component | Reading | Threshold | Verdict |
|---|---|---|---|
| Test 1 tail displacement | max |Δ|=0.0069 | <0.05 | negative |
| Test 2(a) z-shift magnitude | max=0.014 std | tiny | negative |
| Test 2(b) rank preservation | min 5th-pct ρ=0.9975 | >0.98 | negative |
| **Test 2(d) canonical MLP × z_drift_lv** | ρ=+0.508, p<0.001, n=62 | — | **strong positive** |
| **Test 2(d) canonical SAGE × z_drift_all** | ρ=+0.413, p=0.001, n=62 | — | **strong positive** |

## Interpretation (Codex touchpoint-3 review)

1. **Correlation ≠ causation**. A regime confounder (Q2-2025 market event → both abnormal feature values *and* elevated cross-sectional dispersion → higher attainable Spearman IC) is fully compatible with the data.
2. **Magnitude disconnect argues against direct causal leakage**. Max z_drift ≈ 0.009 std-units produces ρ=0.5 IC variation: the model would need extreme local sensitivity and many rank flips, but `ρ>0.9975` shows rank structure is essentially preserved. Mechanistically, this favors a **regime / third-variable** explanation over leakage.
3. **Temporal pattern** (both z_drift and IC peak days 1049-1057, decay together) is **compatible with either mechanism**:
   - Leakage: global clip bounds "anticipate" the early-Fold-4 regime, giving model a clean signal until regime normalizes
   - Regime: Q2-2025 event causes both unusual feature distributions and elevated dispersion, both decaying over weeks
4. **Serial dependence caveat**: n=62 consecutive days; nominal p-values likely optimistic.

## Conclusion

The pre-committed decision rule treats mixed signals as indicative → **Path A** (rebuild with per-fold winsorization and rerun Part C). Scientific weight of evidence does not *prove* meaningful leakage (input perturbations too small, ranks preserved), but does not *exonerate* either. Path A is the only clean adjudication of the causal chain without retraining assumptions.

If Path A S8 Fold 4 IC remains ≈ +0.22 → regime hypothesis validated, write Path B narrative (parsimony).
If Path A S8 Fold 4 IC drops to ≈ 0 → leakage confirmed, original "compact beats library" narrative holds.
