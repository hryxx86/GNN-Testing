# C5 feature-subset sensitivity (POST-HOC, TEST-INFORMED selection) — L1 (MLP) − L0 (LightGBM)

_C5 = 20 columns = Plan-AAA permutation top-15 ∩ single-feature-IC proxy top-15 (proxy top-15 identical with/without the T-1 shift; both selectors scored inside the 12-fold test period — see docs/c5_rerun_brief_2026-09-10.md §9.9). 240 cells; frozen_hparams md5 cdb4d92314b0d43d3287ea6d403d840d; integrity PASS=True. Raw (unadjusted) HLN p; no BH family; not confirmatory._

| universe | role | ΔIC (L1−L0) | 95% block-boot CI | HLN p | HLN p (lag 21) | IC L0 [CI] | IC L1 [CI] | MDE (≈2.8×SE, approx. nominal) | per-seed same sign | LOSO flips |
|---|---|---|---|---|---|---|---|---|---|---|
| C5 | post-hoc sensitivity (no BH) | +0.0134 | [+0.0008, +0.0283] | 0.008 | 0.054 | 0.0203 [-0.0037, 0.0472] | 0.0337 [0.0021, 0.0704] | 0.0197 | 10/10 | 0/10 |
| C | confirmatory (BH over 20-test family) | +0.0148 | [-0.0004, +0.0304] | 0.011 | 0.063 | 0.0195 [-0.0070, 0.0481] | 0.0343 [0.0044, 0.0669] | 0.0220 | 10/10 | 0/10 |
| B | confirmatory (BH over 20-test family) | +0.0143 | [-0.0051, +0.0341] | 0.052 | 0.181 | 0.0228 [0.0024, 0.0429] | 0.0371 [0.0118, 0.0630] | 0.0275 | 10/10 | 0/10 |

(source: family1_{dm_hln,ic_ci,mde}.csv in artifacts/storya_v21_family1_c5 for C5 and artifacts/storya_v21_family1 for C/B; c5_seed_robustness.csv for k/10, m/10)

## Paired daily contrast (seed-averaged daily ΔIC, same test days; conditional subset contrast)

| contrast | mean paired diff | 95% CI | HLN p | HLN p (lag 21) | T |
|---|---|---|---|---|---|
| (L1-L0)_C - (L1-L0)_C5 | +0.0013 | [-0.0159, +0.0189] | 0.838 | 0.882 | 749 |
| (L1-L0)_B - (L1-L0)_C5 | +0.0008 | [-0.0186, +0.0200] | 0.915 | 0.938 | 749 |

(source: c5_paired_contrast.csv; positive = the confirmatory universe's L1−L0 exceeds the C5 one. Conditional subset contrast — feature restriction + re-tuning; NOT an identified leakage-inflation effect: C5's columns were selected with evaluation-period outcomes, brief §9.9)

## Tuned winners (30 trials, top-5 × 3 tuning seeds) and MLP capacity at the actual input width

| universe | arm | model | n_inputs | winner params | val-IC (3-seed, SELECTION metric only) | MLP #params |
|---|---|---|---|---|---|---|
| C5 | L0 | LightGBM | 20 | `{'num_leaves': 63, 'learning_rate': 0.01288916458910432, 'min_data_in_leaf': 100, 'lambda_l1': 0.000465635505528644, 'lambda_l2': 0.015550600226247952}` | -0.0121 | — |
| C5 | L1 | MLP | 20 | `{'lr': 0.009172900184875115, 'weight_decay': 6.558234573242414e-05, 'dropout': 0.3, 'hidden_channels': 32, 'num_layers': 1}` | -0.0448 | 2337 |
| C | L0 | LightGBM | 51 | `{'num_leaves': 15, 'learning_rate': 0.027924746980950325, 'min_data_in_leaf': 100, 'lambda_l1': 1.1945711070427778e-07, 'lambda_l2': 0.04279834445098042}` | 0.0735 | — |
| C | L1 | MLP | 51 | `{'lr': 0.0019531137875202621, 'weight_decay': 0.00024029526489767312, 'dropout': 0.1, 'hidden_channels': 128, 'num_layers': 1}` | 0.0597 | 31745 |

(source: experiments/storya_v21_tune/frozen_hparams_c5.json + artifacts/storya_v21_tune/frozen_hparams.json; param count via run_storya_e1_anchor.make_nn_model at n_inputs)
