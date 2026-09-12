# Split Methodology Discussion - 2026-04-29

Grounding read: `docs/analysis.md` entries `2026-04-27-a` and `2026-04-27-b`; `run_step3_plan_z_part_a.py` fold, feature, label, manifest, graph, scaler, and train-loop code. The code-level facts below refer to that file: `HORIZON = 21`, `TRAIN_START = '2021-01-29'`, quarterly `FOLDS`, rolling price features with `.shift(1)`, 21-day forward labels with `prices.shift(-HORIZON) / prices - 1`, train-only scaling, and a frozen SAGE correlation graph selected at `snaps[train_days.max()]`.

## Q1 - Expanding vs Rolling Train

**Literature position.** Lopez de Prado's AFML recommendation is not "always use expanding" or "always use rolling"; the hard requirement is that samples are defined by information intervals and evaluated with purging plus embargo when labels overlap (Lopez de Prado, 2018, *Advances in Financial Machine Learning*, ch. 7). Chapter 12's CPCV is about obtaining a distribution of backtest paths and reducing selection bias, not about proving a fixed optimal training-window length. Bailey et al. (2015) make the stronger point that repeated backtest selection can create high apparent performance even from weak signals, so changing the train window after seeing fold-4 must be treated as a model-selection event, not as a neutral preprocessing choice ([Bailey, Borwein, Lopez de Prado, Zhu, 2015](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253)).

The stock-GNN literature supports time-aware relational modeling and ranking, but it does not rescue naive temporal validation. Feng et al.'s Relational Stock Ranking frames stock prediction as a ranking problem and uses temporal graph convolution because stock relations are time-sensitive ([Feng et al., 2019 / arXiv 1809.09441](https://ideas.repec.org/p/arx/papers/1809.09441.html)). HATS similarly reports that relation quality matters and uses rolling train/evaluation/test phases rather than random splits ([Kim et al., 2019 / arXiv 1908.07999](https://ideas.repec.org/p/arx/papers/1908.07999.html)). The implication for this project is: use chronological, purged, deployable evaluation; treat window length as a robustness axis.

**Recommendation for this setup.** Keep expanding train as the paper primary baseline, and add 2-year rolling as the high-priority robustness test. Do not make 1-year rolling the primary design.

Why:

- Current code is not actually training from 2020 in this file: `TRAIN_START = '2021-01-29'`. The existing manifest after embargo has `n_train = 714, 775, 838, 902, 966` across folds, with fold-4 training ending `2024-11-29`.
- A 1-year rolling window gives about 252 trading days. With a 21-day horizon, labels are heavily overlapping, so the number of independent temporal regimes is far smaller than 252. For SAGE-Mean with train-only scaling and a 126-day graph window, 1-year rolling is too close to a short-regime fit. It may look attractive on fold-4 but is likely to underfit stable regimes and overreact to recent factor rotations.
- A 2-year rolling window gives about 504 training days. This is the shortest window I would defend: it excludes old 2021/early-2022 regimes in later folds, preserves two annual cycles, leaves enough observations for early stopping, and still supports the 126-day correlation graph.
- A 3-year rolling window is also defensible, but in fold 0 it is nearly identical to expanding because the post-embargo train set has only 714 days. It is less informative for testing the "old regime contamination" hypothesis.

Hypothesis: fold-4 weakness may reflect stale-regime contamination from 2021/2022 rather than generic overfitting. The correct test is expanding vs 2-year rolling with identical test days, seeds, hparams, graph construction, feature scaler, and label. If rolling improves only fold-4 and hurts folds 0-3, call it regime-conditional adaptation, not a globally superior split.

**Code-level change.** Add an explicit rolling fold list or a `rolling_train_days` mode. For exact 2-year post-embargo train sets under the current calendar, use these `train_start` dates and change manifest construction to read `cfg.get('train_start', TRAIN_START)`:

```python
FOLDS_ROLL2Y = [
    dict(id=0, train_start='2021-11-29', train_end='2023-12-31', val_end='2024-03-31', test_end='2024-06-30'),
    dict(id=1, train_start='2022-02-25', train_end='2024-03-31', val_end='2024-06-30', test_end='2024-09-30'),
    dict(id=2, train_start='2022-05-26', train_end='2024-06-30', val_end='2024-09-30', test_end='2024-12-31'),
    dict(id=3, train_start='2022-08-29', train_end='2024-09-30', val_end='2024-12-31', test_end='2025-03-31'),
    dict(id=4, train_start='2022-11-29', train_end='2024-12-31', val_end='2025-03-31', test_end='2025-06-30'),
]
```

In `build_fold_manifest`, change:

```python
ts = pd.Timestamp(TRAIN_START)
```

to:

```python
ts = pd.Timestamp(cfg.get('train_start', TRAIN_START))
```

An even safer implementation is to crop after the existing tail embargo:

```python
ROLLING_TRAIN_DAYS = 504
tr_days = tr_days[-ROLLING_TRAIN_DAYS:]
```

That avoids off-by-one calendar errors and guarantees identical post-embargo train length.

## Q2 - CPCV vs Walk-Forward

**Literature position.** CPCV in AFML ch. 12 is designed to generate many purged backtest paths from contiguous time groups. With `N` groups and `k` test groups per split, the number of train/test splits is `C(N, k)`, while the number of recombined out-of-sample paths is:

```text
phi(N, k) = k / N * C(N, k) = C(N - 1, k - 1)
```

Bailey et al.'s PBO framework estimates how often the in-sample winner degrades out-of-sample under combinatorially symmetric splits. This is directly relevant when choosing among many losses, feature sets, graph rules, and train windows. It is less critical when the experiment is a two-arm comparison with fixed hparams and locked labels.

**Recommendation for this setup.** Do not implement full CPCV before the expanding-vs-2y rolling experiment. The implementation cost is not justified for the immediate question. Use walk-forward as the deployable primary design and CPCV later only if the paper needs a model-selection/PBO appendix.

If CPCV is added later, use half-year groups:

- `N = 10`, `k = 2`: 10 half-year groups over roughly 5 years; 1-year test sets; `C(10,2) = 45` splits; `phi = 9` OOS paths.
- `N = 8`, `k = 2`: lower-cost version; `28` splits; `phi = 7` paths.
- Avoid quarterly `N = 20`, `k = 2` for this M4 budget: `190` splits and `phi = 19` paths. The path count is attractive, but training 190 purged SAGE models per seed/model setting is not.

The current walk-forward setup has 5 quarterly test folds. If fold-level IC standard deviation is about `0.04`, the naive standard error of the 5-fold mean is `0.04 / sqrt(5) = 0.018`. CPCV `N=10,k=2` gives 9 paths, so the naive path-mean width would be `0.04 / sqrt(9) = 0.013`. In practice those CPCV paths are correlated because they reuse periods and training samples, so the effective width is closer to a modest improvement, not a 2x revolution. CPCV's main benefit here is estimating selection fragility/PBO, not merely shrinking the confidence interval.

One caution: standard CPCV trains on all non-test groups, which can include calendar periods after an earlier test group. That is valid for PBO/model-selection analysis with purging, but it is not a deployable "what would I have known then?" backtest. For the paper's main performance claim, walk-forward remains the cleaner design.

**Code-level change.** If implemented, CPCV groups must be contiguous date blocks. For each split:

```text
test_groups = selected k contiguous-group IDs
train_groups = all other groups
purge any train sample t whose label interval [t, t + HORIZON] intersects a test interval
embargo HORIZON trading days after each test interval before allowing train samples
freeze graph and fit scaler using only surviving train samples
```

Do not reuse the current single-boundary manifest logic unchanged; CPCV needs interval-overlap purging against multiple test blocks.

## Q3 - Val/Test Window Size

**Literature/statistical position.** Cross-sectional IC uses 500 stocks per day, but expected IC is a time-series estimand. The 500-stock cross-section makes each daily Spearman correlation estimable; it does not create 500 independent days. Under a null rank correlation with `N=500`, the per-day correlation noise scale is approximately `1 / sqrt(N - 1) = 0.045`, close to the stated `sd ~= 0.04`. With a 21-day forward return label, adjacent daily ICs are strongly overlapping; a conservative effective sample size is:

```text
n_eff ~= n_raw / HORIZON
```

For 80% power to detect mean `IC = 0.03` with two-sided `alpha = 0.05` and daily `sigma = 0.04`:

```text
n_eff >= ((z_0.975 + z_0.80) * sigma / delta)^2
      = ((1.96 + 0.84) * 0.04 / 0.03)^2
      ~= 14 independent observations
```

With 21-day overlapping labels, that is about:

```text
n_raw ~= 14 * 21 = 294 trading days
```

Power implications:

- 1 quarter, `n_raw ~= 63`: `n_eff ~= 3`, power only about 25%.
- 6 months, `n_raw ~= 126`: `n_eff ~= 6`, power about 45%.
- 1 year, `n_raw ~= 252`: `n_eff ~= 12`, power about 74%.
- Current 5-fold aggregate, `63+64+64+60+62 = 313`: `n_eff ~= 14.9`, power about 83%.

**Recommendation for this setup.** Do not interpret any single quarterly fold IC as statistically reliable. A quarterly fold is useful as a regime diagnostic, not as a stand-alone performance estimate. Keep the 5 quarterly walk-forward tests for regime visibility, but make the primary statistical estimate the paired aggregate over all 313 test days with 21-day HAC/Newey-West or moving-block bootstrap inference.

Do not switch the main design to 1-year test windows right now. You would gain per-fold stability but lose fold count and regime localization. For this dataset, the right balance is:

1. Primary: all 5 quarterly test folds aggregated, with block/HAC inference.
2. Mandatory sensitivity: fold 0-3 aggregate and fold 4 separately.
3. Optional table: non-overlapping 6-month test aggregation for reader intuition.

**Code-level change.** Add an analysis-layer estimator, not necessarily new FOLDS:

```text
daily paired diff d_t = IC_rolling(t, seed) - IC_expanding(t, seed)
estimate mean(d_t)
standard error: Newey-West lag = HORIZON, or moving-block bootstrap block length >= HORIZON
cluster/sensitivity: report fold-cluster bootstrap over the 5 fold means
```

If you do create wider test windows, do not overlap test periods in the primary table. Overlapping 6-month windows increase apparent sample size without adding independent outcomes.

## Q4 - Purge + Embargo Correctness

**Literature position.** AFML ch. 7's purge/embargo logic is based on information intervals. A training observation must be removed if its label interval overlaps the validation/test interval. Embargo then protects against residual leakage due to serial dependence and overlapping labels. The correct buffer is driven by forward-looking information, not by the mere existence of a backward-looking feature window.

**Current code facts.**

- Labels: `fwd_ret = prices.shift(-HORIZON) / prices - 1`, so label for day `t` uses `close[t + 21]`.
- Manifest: the last 21 trading days are removed from train and validation. Assertions enforce `train_days.max() + HORIZON < val_days.min()` and `val_days.max() + HORIZON < test_days.min()`.
- Price features: `returns.rolling(w).mean().shift(1)` and `.std().shift(1)` for `w in {5,10,21}`. Feature at day `t` uses returns ending at `t-1`.
- Graph: snapshots use `returns.iloc[t_end - corr_window:t_end]` with `corr_window = 126`; SAGE freezes `frozen_si = snaps[train_days.max()]`.

**(a) Feature leakage formula.** For a feature `x_j(t)` with source interval:

```text
[t - L_j^-, t + L_j^+]
```

the required feature-side tail purge before a validation/test boundary is:

```text
feature_purge_j = max(0, L_j^+)
```

For the current shifted rolling features:

```text
x(t) uses [t - w, t - 1]
L_j^+ = -1
feature_purge_j = 0
```

Therefore the 21-day label embargo is sufficient for `ret_mean_21d` and `ret_std_21d`; you do not need an additional 21-day buffer just because the feature has a 21-day backward lookback. A larger feature buffer would be required only for centered, forward-filled-from-future, globally normalized, or otherwise non-causal features. The precomputed `sp500_5y_phase5_features.npy` must be audited separately because no boundary embargo fixes global winsorization/scaling leakage.

**(b) Graph snapshot leakage formula.** For a graph snapshot with window length `W` and exclusive end index `e`, built as:

```text
returns[e - W : e]
```

the latest return index used is `e - 1`. The leak-free frozen graph condition is:

```text
e - 1 <= max_train_feature_date_idx
```

Under the current conservative design, use:

```text
e - 1 <= train_days.max()
```

This is stronger than necessary if the graph is purely unsupervised from known past returns, but it is clean and reviewer-defensible. The snapshot lookback `W = 126` does not require a 126-day embargo; it requires the snapshot end to be on the allowed side of the boundary. If a dynamic test-time graph is allowed to update during validation/test using past validation/test returns, label leakage is still avoidable, but the experiment becomes transductive/adaptive and should not be compared to the frozen-graph result without being named as a separate design.

**(c) Label leakage formula.** For a forward close-to-close return label:

```text
y(t) = close[t + H] / close[t] - 1
```

a training sample before a validation boundary at `b` is allowed only if:

```text
t + H < b
```

Equivalently:

```text
max_train_t <= b - H - 1
tail_purge_train = H
```

The same formula applies to validation before test. The current manifest implements this correctly for the quarterly folds.

**Zero-leakage verification protocol.**

1. Export raw split boundaries plus post-embargo `train_days`, `val_days`, `test_days` for every fold and split type. Assert `max(train_days) + HORIZON < min(val_days)` and `max(val_days) + HORIZON < min(test_days)`.
2. Add feature provenance metadata: for each feature, store `(min_lag, max_lag, fitted_on)`. Assert `max_lag <= 0` for all features used at day `t`; for this code, shifted rolling features should record source `[t-w, t-1]`.
3. Audit precomputed features. For `sp500_5y_phase5_features.npy`, verify that any winsorization, standardization, rolling rank, or sector normalization is computed causally or per-fold train-only. If not provable, rebuild those features per fold.
4. Log graph provenance per fold: `frozen_si`, `snap_end = snap_points[frozen_si]`, `snap_window = [snap_end - corr_window, snap_end)`. Assert `snap_end - 1 <= train_days.max()`. This also catches the early-fold edge case where `snaps[di]` can map to snapshot 0 before the first `snap_point`.
5. Add a sentinel test: perturb prices/features strictly after `min(val_days)` and prove that train features, train labels, train scaler, and frozen train graph are bitwise unchanged.

## Q5 - Sector-Adjusted vs Absolute Return Labels

**Literature position.** A universe-level cross-sectional z-score label asks the model to rank total forward returns. A sector-adjusted residual label asks the model to rank alpha after removing coarse sector factors. In factor-model language:

```text
r_i,t:t+H = sector_beta_i' f_t + alpha_i,t + epsilon_i,t
```

The current label rewards both sector rotation and within-sector selection. A sector residual label mostly rewards within-sector selection. Novy-Marx's anomaly-performance paper is a useful warning here: conditioning and factor timing can look predictable in-sample for spurious reasons, so claims about alpha should avoid accidentally measuring broad factor/sector bets ([Novy-Marx, 2014](https://www.nber.org/papers/w18063)). Feng et al.'s RSR supports ranking as the right formulation for stock selection, but it does not imply that the target should be raw total return if the economic objective is sector-neutral alpha.

**Recommendation for this setup.** Because the label is locked, do not change the primary training target in the current experiment. Keep the universe z-scored 21-day return label as the registered primary label. But add sector-adjusted IC as a mandatory secondary evaluation metric before making any alpha claim.

For a future paper iteration, I would use the sector-adjusted residual as the primary label if the intended portfolio is dollar-neutral and sector-neutral. It will be more regime-stable and closer to "stock selection skill." If the intended portfolio is allowed to make sector bets, the absolute universe label is valid, but then the paper must say it predicts cross-sectional total return ranks, not pure alpha.

The practical advantage of adding sector-adjusted evaluation now is that it diagnoses whether SAGE-Mean is using sector/correlation edges to capture stable within-sector relative value or simply loading on sector rotations that happen to dominate fold-4.

**Code-level change.** Add a secondary label/evaluation path:

```python
fwd_ret = prices.shift(-HORIZON) / prices - 1
sector_mean = fwd_ret.groupby(sector_by_ticker, axis=1).transform('mean')
sector_resid = fwd_ret - sector_mean
sector_resid_z = sector_resid.sub(sector_resid.mean(axis=1), axis=0).div(sector_resid.std(axis=1), axis=0)
```

Then report:

```text
IC_abs = Spearman(pred, universe_z_forward_return)
IC_sector_resid = Spearman(pred, sector_resid_z)
```

Do not select hparams on `IC_sector_resid` if the primary label remains absolute; use it as an out-of-sample diagnostic.

## Q6 - Practical Recommendation Under 10h M4 Budget

**Literature/statistical position.** The split-window comparison is a paired experiment. Bailey et al.'s PBO warning applies because choosing rolling after seeing fold-4 is a backtest-selection risk. The right control is not another unpaired mean table; it is same test days, same seeds, same hyperparams, and paired inference on daily IC differences.

**Minimum viable experiment.** Run exactly this:

```text
model: SAGE-Mean only
loss: current locked/default loss for this project
features: locked feature set
labels: locked universe z-score 21d label
train regimes: expanding vs 2-year rolling
folds: same 5 quarterly test folds
seeds: same 10 seeds if available; otherwise at least same 3 seeds as a pilot, then promote to 10
hparams: identical, no retuning
graph: same corr_window=126, corr_step=21, threshold=0.6, frozen train graph
scaler: fit train-only separately within each fold/regime
```

The diagnostic run in `docs/analysis.md` reports 200 cells in 439 minutes on M4 MPS. A 100-cell comparison, `2 train regimes * 5 folds * 10 seeds * 1 model * 1 loss`, should be within the 10-hour budget unless the selected feature set is much heavier than the diagnostic setup.

**Statistical controls.**

- Same seeds and same test days; never compare rolling fold means to expanding fold means with different day coverage.
- No hparam retuning for rolling. If rolling gets its own tuned hparams, that is a second experiment and must be reported as such.
- Paired daily test:

```text
d_{fold, seed, day} = IC_2yrolling - IC_expanding
```

Report mean `d`, 95% CI, and p-value using Newey-West lag 21 or moving-block bootstrap with block length at least 21. Also report fold-cluster bootstrap because there are only 5 regimes.

**Fold-4 confound handling.** Fold-4 making rolling look better is not automatically "wrong"; a rolling window is supposed to adapt to regime change. The confound is interpretive: if all improvement comes from Q2 2025 and rolling hurts 2024 folds, the result is not "2-year rolling is better"; it is "2-year rolling helped in the fold-4 stress regime."

Pre-specify this reporting rule:

1. Primary result: all 5 folds, because Q2 2025 is valid out-of-sample and excluding it would be post-hoc cherry-picking.
2. Stability subset: folds 0-3 aggregate.
3. Stress subset: fold 4 alone.
4. Decision rule: call 2-year rolling preferable only if it improves all-5 aggregate and is not materially negative on folds 0-3. If the sign is positive only in fold 4, report it as regime-conditional.

**Code-level change.** Generate two manifests with identical `val_days` and `test_days` and different `train_days`. Save both manifests before training:

```text
fold_manifest_expanding.json
fold_manifest_roll2y.json
```

For every fold, assert:

```text
expanding.test_days == rolling.test_days
expanding.val_days == rolling.val_days
rolling.train_days is a suffix/cropped subset of expanding.train_days
max(rolling.train_days) + HORIZON < min(rolling.val_days)
```

## Action List

**Tier 1 - Mandatory fixes before any new experiment**

1. Generalize the manifest for rolling without changing test coverage. Specific fix: either support `cfg.get('train_start', TRAIN_START)` or crop `tr_days = tr_days[-504:]` after the existing 21-day tail embargo. Formula to preserve: `max(train_days) + HORIZON < min(val_days)` and `max(val_days) + HORIZON < min(test_days)`.
2. Add graph provenance assertions. Specific fix after `frozen_si = snaps[train_days.max()]`: compute `snap_end = snap_points[frozen_si]` and assert `snap_end - 1 <= train_days.max()`. Log `[snap_end - corr_window, snap_end)` into the fold manifest.
3. Add feature provenance/audit for all precomputed features. Specific formula: each feature must declare `source_max_idx(t) <= t` for after-close prediction, and shifted rolling price features should satisfy `source_max_idx(t) = t - 1`. Any global scaler/winsorizer must be replaced with train-only per-fold fitting.
4. Save and compare expanding vs rolling manifests before training. Specific fix: assert identical `val_days` and `test_days`; assert rolling train days are a subset/suffix of expanding train days for the same fold.

**Tier 2 - High-ROI improvements**

1. Run the 100-cell paired experiment: SAGE-Mean, fixed hparams, expanding vs 2-year rolling, 5 folds, 10 shared seeds.
2. Analyze paired daily IC differences with Newey-West lag 21 or moving-block bootstrap block length `>= 21`; report fold-cluster bootstrap as a sensitivity.
3. Report three split views: all folds primary, folds 0-3 stability subset, fold 4 stress subset. Use the decision rule that rolling must not materially hurt folds 0-3 to be called generally preferable.
4. Add sector-adjusted IC as a secondary metric while keeping the locked absolute label. This distinguishes total-return ranking from alpha ranking.
5. Aggregate quarterly folds for statistical conclusions; stop treating individual 63-day fold ICs as reliable performance estimates.

**Tier 3 - Nice-to-have**

1. CPCV appendix only after the main comparison is stable. Recommended design: `N=10` half-year groups, `k=2`, `45` purged splits, `phi=9` OOS paths.
2. Hybrid weighting after 2-year rolling: exponential time decay with half-life 252 or 504 trading days, applied as sample weights without changing test folds.
3. 1-year rolling as a stress test only. It is not a defensible primary window for this 21-day horizon SAGE setup.
4. Dynamic causal graph variant, clearly labeled as adaptive/transductive, with snapshot condition `snapshot_end <= inference_day` and no use of future labels.
