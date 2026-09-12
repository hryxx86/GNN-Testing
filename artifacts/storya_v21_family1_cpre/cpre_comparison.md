# C-pre feature re-selection sensitivity (POST-HOC, PRE-EVALUATION selection) — L1 (MLP) − L0 (LightGBM)

_CPRE = 48 columns = union of all members of the top-15 Plan-AAA groups ranked by single-feature |IC| on the tuning-train window 2021-07-01..2022-05-31 (label end ≤ 2022-06-30; τ = 0.50 coverage; docs/c_pre_plan_2026-09-11.md §3) — scoring and grouping inputs are bounded by 2022-06-30, the protocol was chosen retrospectively. 240 cells; frozen_hparams md5 a8fdfb8f9cefb24dcaf827a5905d6aad; integrity PASS=True. Raw (unadjusted, nominal) HLN p; no BH family; not confirmatory._

_INPUT (primary): `experiments/storya_v21_main12_cpre` — device mps (devices seen ['mps'], 1 invocation(s)), platform macOS-26.3.1-arm64-arm-64bit, results.csv md5 3cf9f526dbc105f279702c30deb52b1c, git_rev 46ca6e38966f6304fefc55dd67da710bf5d78a55, source_clean True, post-hoc code identity None; family-dir stats from the same results.csv: True; analysis mode {'n_boot': 5000, 'strict': True, 'smoke': False, 'out_dir': 'artifacts/storya_v21_family1_cpre'}._

| universe | role | ΔIC (L1−L0, seed-averaged daily) | 95% block-boot CI (on the 10-seed average) | HLN p (NW auto lag) | HLN p (lag 21 = horizon) | IC L0 [CI] | IC L1 [CI] | MDE (≈2.8×SE, approx. nominal) | per-seed same sign | LOSO flips |
|---|---|---|---|---|---|---|---|---|---|---|
| CPRE | post-hoc sensitivity, pre-evaluation selection (no BH) | -0.0024 | [-0.0256, +0.0178] | 0.786 | 0.847 | 0.0057 [-0.0230, 0.0376] | 0.0033 [-0.0181, 0.0245] | 0.0313 | 5/10 | 1/10 |
| C | confirmatory (BH over 20-test family) | +0.0148 | [-0.0004, +0.0304] | 0.011 | 0.063 | 0.0195 [-0.0070, 0.0481] | 0.0343 [0.0044, 0.0669] | 0.0220 | 10/10 | 0/10 |
| B | confirmatory (BH over 20-test family) | +0.0143 | [-0.0051, +0.0341] | 0.052 | 0.181 | 0.0228 [0.0024, 0.0429] | 0.0371 [0.0118, 0.0630] | 0.0275 | 10/10 | 0/10 |
| C5 | post-hoc sensitivity (no BH) | +0.0134 | [+0.0008, +0.0283] | 0.008 | 0.054 | 0.0203 [-0.0037, 0.0472] | 0.0337 [0.0021, 0.0704] | 0.0197 | 10/10 | 0/10 |

(source: family1_{dm_hln,ic_ci,mde}.csv in artifacts/storya_v21_family1_cpre for CPRE, artifacts/storya_v21_family1 for C, artifacts/storya_v21_family1 for B, artifacts/storya_v21_family1_c5 for C5; cpre_seed_robustness.csv for k/10, m/10)

_Reading notes (computed from this run; closeout EXPL-STAT-01/02/03/05/06/10 checks). (i) The headline HLN p uses the Newey-West AUTO lag — an implementation default, NOT a protocol-specified choice; the label overlaps 21 days, so the horizon-matched lag-21 p is reported alongside. Nominal p < 0.05 at the auto lag: ['C', 'C5']; at lag 21: none. (ii) Percentile-CI boundary check — CPRE: CI includes 0 (|ΔIC| 0.0024 vs 1.96×SE_block 0.0219); C: CI includes 0 (|ΔIC| 0.0148 vs 1.96×SE_block 0.0154); B: CI includes 0 (|ΔIC| 0.0143 vs 1.96×SE_block 0.0192); C5: CI excludes 0 although 1.96×SE_block (0.0138) exceeds |ΔIC| (0.0134) — a boundary case (percentile asymmetry), not a robust rejection. (iii) 5% verdicts per universe — CPRE: CI includes 0, auto-lag p 0.786, lag-21 p 0.847; C: CI includes 0, auto-lag p 0.011, lag-21 p 0.063, BH reject; B: CI includes 0, auto-lag p 0.052, lag-21 p 0.181, BH no-reject; C5: CI excludes 0, auto-lag p 0.008, lag-21 p 0.054. |ΔIC| below its own ≈2.8×SE MDE: ['CPRE', 'C', 'B', 'C5']. (iv) Smallest nominal p: C5 (0.008); smallest |ΔIC|: CPRE (SE_block CPRE 0.0112, C 0.0079, B 0.0098, C5 0.0070). (v) k/10 = 5/10 is a seed/initialisation stability check on the SAME data (not independent replication); LOSO flips = 1. (vi) The CI is for the seed-averaged ensemble (day-to-day variance only; per-seed ΔIC in CPRE spans -0.0314…+0.0173). Per-arm IC levels are conditional on the pre-evaluation selection rule and on a retrospectively chosen protocol; nominal, not confirmatory out-of-sample performance figures._

## Fold concentration — pooled statistics EXCLUDING fold 9 (largest single-fold contribution in ['C', 'C5']; CPRE (rank 12; largest = fold 11), B (rank 2; largest = fold 7))

| universe | fold ΔIC (excluded fold) | share of pooled ΔIC | ΔIC ex-fold | 95% block-boot CI | HLN p | HLN p (lag 21) | MDE (≈2.8×SE) | T |
|---|---|---|---|---|---|---|---|---|
| CPRE | -0.1386 | n/a (pooled ΔIC -0.0024 within one SE of 0; contribution rank 12/12) | +0.0099 | [-0.0069, +0.0265] | 0.198 | 0.348 | 0.0241 | 687 |
| C | +0.0785 | 44% (rank 1/12) | +0.0090 | [-0.0052, +0.0228] | 0.125 | 0.248 | 0.0203 | 687 |
| B | +0.0529 | 31% (rank 2/12) | +0.0108 | [-0.0089, +0.0300] | 0.143 | 0.312 | 0.0283 | 687 |
| C5 | +0.0862 | 53% (rank 1/12) | +0.0069 | [-0.0032, +0.0179] | 0.132 | 0.231 | 0.0153 | 687 |

(source: cpre_ex_fold.csv; share = n_days(fold) × fold ΔIC / (T × pooled ΔIC): CPRE n/a (pooled ΔIC -0.0024 within one SE of 0; contribution rank 12/12), C 44% (rank 1/12), B 31% (rank 2/12), C5 53% (rank 1/12). A share near or above one half means the pooled contrast is not evenly persistent across quarters; the ratio is not meaningful when the pooled contrast is within one SE of zero, and its denominator carries the same uncertainty as the headline. The ex-fold series joins the retained observations across the removed quarter, so the HAC window and the 21-day blocks straddle one artificial seam — the ex-fold row is a diagnostic; the full-period row is primary)

## Paired daily contrast (seed-averaged daily ΔIC, same test days; conditional contrast — absolute change only)

| contrast | mean paired diff | 95% CI | HLN p | HLN p (lag 21) | SE_block | MDE (≈2.8×SE) | T |
|---|---|---|---|---|---|---|---|
| (L1-L0)_C - (L1-L0)_CPRE | +0.0172 | [-0.0111, +0.0494] | 0.102 | 0.271 | 0.0155 | 0.0433 | 749 |
| (L1-L0)_B - (L1-L0)_CPRE | +0.0167 | [-0.0133, +0.0509] | 0.131 | 0.309 | 0.0160 | 0.0447 | 749 |
| (L1-L0)_C5 - (L1-L0)_CPRE | +0.0158 | [-0.0095, +0.0481] | 0.131 | 0.292 | 0.0148 | 0.0414 | 749 |

_(L1-L0)_C - (L1-L0)_CPRE: CI includes 0; paired SE 0.0155 → paired MDE 0.0433 > |C contrast| 0.0148. Read the absolute point estimate with its interval; equivalence is not tested and a non-rejection is not equivalence (paired MDE exceeds the comparator contrast: underpowered comparison)._

_(L1-L0)_B - (L1-L0)_CPRE: CI includes 0; paired SE 0.0160 → paired MDE 0.0447 > |B contrast| 0.0143. Read the absolute point estimate with its interval; equivalence is not tested and a non-rejection is not equivalence (paired MDE exceeds the comparator contrast: underpowered comparison)._

_(L1-L0)_C5 - (L1-L0)_CPRE: CI includes 0; paired SE 0.0148 → paired MDE 0.0414 > |C5 contrast| 0.0134. Read the absolute point estimate with its interval; equivalence is not tested and a non-rejection is not equivalence (paired MDE exceeds the comparator contrast: underpowered comparison)._

(source: cpre_paired_contrast.csv; positive = the comparator's L1−L0 exceeds the CPRE one. Conditional contrast — not an identified leakage-inflation effect (column sets differ and both arms were re-tuned); no proportional (halved/doubled) inference)

_Data-derived checks (EXPL-CODE-04): |ΔIC| vs its own MDE — CPRE: BELOW, C: BELOW, B: BELOW, C5: BELOW._

## Tuned winners (30 trials, top-5 × 3 tuning seeds) and MLP capacity at the actual input width

| universe | arm | model | n_inputs | winner params | val-IC (3-seed, SELECTION metric only) | MLP #params |
|---|---|---|---|---|---|---|
| CPRE | L0 | LightGBM | 48 | `{'num_leaves': 15, 'learning_rate': 0.010520856370501335, 'min_data_in_leaf': 10, 'lambda_l1': 2.3110698302219116e-08, 'lambda_l2': 0.3394454877627604}` | 0.0311 | — |
| CPRE | L1 | MLP | 48 | `{'lr': 0.001990231214211617, 'weight_decay': 0.0002549552933302671, 'dropout': 0.2, 'hidden_channels': 128, 'num_layers': 1}` | 0.0135 | 31361 |
| C | L0 | LightGBM | 51 | `{'num_leaves': 15, 'learning_rate': 0.027924746980950325, 'min_data_in_leaf': 100, 'lambda_l1': 1.1945711070427778e-07, 'lambda_l2': 0.04279834445098042}` | 0.0735 | — |
| C | L1 | MLP | 51 | `{'lr': 0.0019531137875202621, 'weight_decay': 0.00024029526489767312, 'dropout': 0.1, 'hidden_channels': 128, 'num_layers': 1}` | 0.0597 | 31745 |
| C5 | L0 | LightGBM | 20 | `{'num_leaves': 63, 'learning_rate': 0.01288916458910432, 'min_data_in_leaf': 100, 'lambda_l1': 0.000465635505528644, 'lambda_l2': 0.015550600226247952}` | -0.0121 | — |
| C5 | L1 | MLP | 20 | `{'lr': 0.009172900184875115, 'weight_decay': 6.558234573242414e-05, 'dropout': 0.3, 'hidden_channels': 32, 'num_layers': 1}` | -0.0448 | 2337 |

(source: experiments/storya_v21_tune/frozen_hparams_cpre.json, artifacts/storya_v21_tune/frozen_hparams.json, experiments/storya_v21_tune/frozen_hparams_c5.json; param count via run_storya_e1_anchor.make_nn_model at n_inputs)

_DISCLOSURE (TP2-B B-04 / TP3 R-A-04; values computed from the tune JSONs): L0: 0/5 finalists with negative 2022H2 val-IC (range +0.0302…+0.0311) — the 3 tuning seeds give identical val-IC for this arm (deterministic), so the 3-seed average carries no initialisation information here; L1: 0/5 finalists with negative 2022H2 val-IC (range +0.0014…+0.0135) — vs C L0 winner val-IC +0.0735, C L1 winner val-IC +0.0597; CPRE MLP 31,361 params vs C MLP 31,745 (ratio 1.0×). The frozen HPs are protocol-consistent; the contrast (or its similarity to C) is not attributed to feature restriction/re-selection or capacity alone._
