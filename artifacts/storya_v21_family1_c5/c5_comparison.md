# C5 feature-subset sensitivity (POST-HOC, TEST-INFORMED selection) — L1 (MLP) − L0 (LightGBM)

_C5 = 20 columns = Plan-AAA permutation top-15 ∩ single-feature-IC proxy top-15 (proxy top-15 identical with/without the T-1 shift; both selectors scored inside the 12-fold test period, using NN-based permutation importance — see docs/c5_rerun_brief_2026-09-10.md §9.9). 240 cells; frozen_hparams md5 cdb4d92314b0d43d3287ea6d403d840d; integrity PASS=True. Raw (unadjusted, nominal) HLN p; no BH family; not confirmatory._

_INPUT (primary): `experiments/storya_v21_main12_c5_t4` — device cuda (devices seen ['cuda'], 2 invocation(s)), platform Linux-6.6.122+-x86_64-with-glibc2.39, results.csv md5 eaa8af1de0e50168b439cb9eb0d7eee7, git_rev None, source_clean None, post-hoc code identity {'all_modules_match': True, 'commit': '9008dbe', 'verified_at': '2026-09-11 02:51:56'}; family-dir stats from the same results.csv: True; analysis mode {'n_boot': 5000, 'strict': True, 'smoke': False, 'out_dir': 'artifacts/storya_v21_family1_c5'}._

| universe | role | ΔIC (L1−L0, seed-averaged daily) | 95% block-boot CI (on the 10-seed average) | HLN p (NW auto lag) | HLN p (lag 21 = horizon) | IC L0 [CI] | IC L1 [CI] | MDE (≈2.8×SE, approx. nominal) | per-seed same sign | LOSO flips |
|---|---|---|---|---|---|---|---|---|---|---|
| C5 | post-hoc sensitivity (no BH) | +0.0134 | [+0.0008, +0.0283] | 0.008 | 0.054 | 0.0203 [-0.0037, 0.0472] | 0.0337 [0.0021, 0.0704] | 0.0197 | 10/10 | 0/10 |
| C | confirmatory (BH over 20-test family) | +0.0148 | [-0.0004, +0.0304] | 0.011 | 0.063 | 0.0195 [-0.0070, 0.0481] | 0.0343 [0.0044, 0.0669] | 0.0220 | 10/10 | 0/10 |
| B | confirmatory (BH over 20-test family) | +0.0143 | [-0.0051, +0.0341] | 0.052 | 0.181 | 0.0228 [0.0024, 0.0429] | 0.0371 [0.0118, 0.0630] | 0.0275 | 10/10 | 0/10 |

(source: family1_{dm_hln,ic_ci,mde}.csv in artifacts/storya_v21_family1_c5 for C5, artifacts/storya_v21_family1 for C, artifacts/storya_v21_family1 for B; c5_seed_robustness.csv for k/10, m/10)

_Reading notes (computed from this run; closeout EXPL-STAT-01/02/03/05/06/10 checks). (i) The headline HLN p uses the Newey-West AUTO lag — an implementation default, NOT a protocol-specified choice; the label overlaps 21 days, so the horizon-matched lag-21 p is reported alongside. Nominal p < 0.05 at the auto lag: ['C5', 'C']; at lag 21: none. (ii) C5: the percentile CI excludes 0 although 1.96×SE_block (0.0138) exceeds |ΔIC| (0.0134) — a boundary case (percentile asymmetry), not a robust rejection. (iii) 5% verdicts per universe — C5: CI excludes 0, auto-lag p 0.008, lag-21 p 0.054; C: CI includes 0, auto-lag p 0.011, lag-21 p 0.063, BH reject; B: CI includes 0, auto-lag p 0.052, lag-21 p 0.181, BH no-reject. |ΔIC| below its own ≈2.8×SE MDE: ['C5', 'C', 'B']. (iv) The smallest nominal p (C5, 0.008) accompanies the smallest |ΔIC| — the ordering is variance-driven (SE_block C5 0.0070, C 0.0079, B 0.0098), not a larger effect. (v) k/10 = 10/10 is a seed/initialisation stability check on the SAME data (not independent replication); m = 0 LOSO flips is implied by k = n. (vi) The CI is for the seed-averaged ensemble (day-to-day variance only; per-seed ΔIC in C5 spans +0.0029…+0.0249). Per-arm IC levels are conditional on the test-informed selection and are not out-of-sample performance figures._

## Fold concentration — pooled statistics EXCLUDING fold 9 (largest single-fold contribution in ['C5', 'C']; B (rank 2; largest = fold 7))

| universe | fold ΔIC (excluded fold) | share of pooled ΔIC | ΔIC ex-fold | 95% block-boot CI | HLN p | HLN p (lag 21) | MDE (≈2.8×SE) | T |
|---|---|---|---|---|---|---|---|---|
| C5 | +0.0862 | 53% | +0.0069 | [-0.0032, +0.0179] | 0.132 | 0.231 | 0.0153 | 687 |
| C | +0.0785 | 44% | +0.0090 | [-0.0052, +0.0228] | 0.125 | 0.248 | 0.0203 | 687 |
| B | +0.0529 | 31% | +0.0108 | [-0.0089, +0.0300] | 0.143 | 0.312 | 0.0283 | 687 |

(source: c5_ex_fold.csv; share = n_days(fold) × fold ΔIC / (T × pooled ΔIC): C5 53%, C 44%, B 31%. A share near or above one half means the pooled contrast is not evenly persistent across quarters)

## Paired daily contrast (seed-averaged daily ΔIC, same test days; conditional contrast — absolute change only)

| contrast | mean paired diff | 95% CI | HLN p | HLN p (lag 21) | SE_block | MDE (≈2.8×SE) | T |
|---|---|---|---|---|---|---|---|
| (L1-L0)_C - (L1-L0)_C5 | +0.0013 | [-0.0159, +0.0189] | 0.838 | 0.882 | 0.0088 | 0.0245 | 749 |
| (L1-L0)_B - (L1-L0)_C5 | +0.0008 | [-0.0186, +0.0200] | 0.915 | 0.938 | 0.0098 | 0.0276 | 749 |

_(L1-L0)_C - (L1-L0)_C5: CI includes 0; paired SE 0.0088 → paired MDE 0.0245 > |C contrast| 0.0148. Read the absolute point estimate with its interval; equivalence is not tested and a non-rejection is not equivalence (paired MDE exceeds the comparator contrast: underpowered comparison)._

_(L1-L0)_B - (L1-L0)_C5: CI includes 0; paired SE 0.0098 → paired MDE 0.0276 > |B contrast| 0.0143. Read the absolute point estimate with its interval; equivalence is not tested and a non-rejection is not equivalence (paired MDE exceeds the comparator contrast: underpowered comparison)._

(source: c5_paired_contrast.csv; positive = the comparator's L1−L0 exceeds the C5 one. Conditional contrast — not an identified leakage-inflation effect (C5 selection is test-informed); no proportional (halved/doubled) inference)

_Data-derived checks (EXPL-CODE-04): |ΔIC| vs its own MDE — C5: BELOW, C: BELOW, B: BELOW._

## Tuned winners (30 trials, top-5 × 3 tuning seeds) and MLP capacity at the actual input width

| universe | arm | model | n_inputs | winner params | val-IC (3-seed, SELECTION metric only) | MLP #params |
|---|---|---|---|---|---|---|
| C5 | L0 | LightGBM | 20 | `{'num_leaves': 63, 'learning_rate': 0.01288916458910432, 'min_data_in_leaf': 100, 'lambda_l1': 0.000465635505528644, 'lambda_l2': 0.015550600226247952}` | -0.0121 | — |
| C5 | L1 | MLP | 20 | `{'lr': 0.009172900184875115, 'weight_decay': 6.558234573242414e-05, 'dropout': 0.3, 'hidden_channels': 32, 'num_layers': 1}` | -0.0448 | 2337 |
| C | L0 | LightGBM | 51 | `{'num_leaves': 15, 'learning_rate': 0.027924746980950325, 'min_data_in_leaf': 100, 'lambda_l1': 1.1945711070427778e-07, 'lambda_l2': 0.04279834445098042}` | 0.0735 | — |
| C | L1 | MLP | 51 | `{'lr': 0.0019531137875202621, 'weight_decay': 0.00024029526489767312, 'dropout': 0.1, 'hidden_channels': 128, 'num_layers': 1}` | 0.0597 | 31745 |

(source: experiments/storya_v21_tune/frozen_hparams_c5.json + artifacts/storya_v21_tune/frozen_hparams.json; param count via run_storya_e1_anchor.make_nn_model at n_inputs)

_DISCLOSURE (TP2-B B-04 / TP3 R-A-04; values computed from the tune JSONs): L0: 5/5 finalists with negative 2022H2 val-IC (range -0.0122…-0.0121); L1: 5/5 finalists with negative 2022H2 val-IC (range -0.0453…-0.0448) — vs C L0 winner val-IC +0.0735, C L1 winner val-IC +0.0597; C5 MLP 2,337 params vs C MLP 31,745 (ratio 13.6×). The frozen HPs are protocol-consistent but not a validated optimum where the finalists are negative; the contrast (or its similarity to C) is not attributed to feature restriction/re-selection or capacity alone._

## Device replication — primary vs replicate result directories (same frozen HPs, same code)

| arm   |   n_cells |   corr_cell_IC |   mean_IC_primary |   mean_IC_replicate |   mean_abs_diff |   max_abs_diff |   n_identical |   wall_primary_s |   wall_replicate_s |
|:------|----------:|---------------:|------------------:|--------------------:|----------------:|---------------:|--------------:|-----------------:|-------------------:|
| L0    |       120 |         1      |           0.02026 |             0.02026 |         0       |        0       |           120 |              1.3 |                0.6 |
| L1    |       120 |         0.9511 |           0.03368 |             0.03341 |         0.01768 |        0.10272 |             0 |             29.4 |               51.2 |

_experiments/storya_v21_main12_c5_t4 (primary) vs experiments/storya_v21_main12_c5 (replicate): pooled ΔIC L1−L0 +0.01343 vs +0.01318 (same sign; absolute gap 0.00025). Per arm — L0: cell-IC corr 1.000, mean |diff| 0.0000, max |diff| 0.0000, 120/120 identical; L1: cell-IC corr 0.951, mean |diff| 0.0177, max |diff| 0.1027, 0/120 identical. These are descriptive replication statistics; whether the replicate's inference agrees is read from the replicate's own family1/analysis outputs, not from this table (source: c5_device_replication.csv)._
