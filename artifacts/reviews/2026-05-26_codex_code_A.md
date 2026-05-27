---
reviewer: codex
touchpoint: code
round: A
target_files:
  - /Users/heruixi/Desktop/GNN-Testing/run_storya_e1_anchor.py
findings:
  - id: CODEX-CR-A-01
    severity: CRITICAL
    category: data-leakage
    claim: "Universe C features encode same-day OHLC information, violating the T-1 contract"
    evidence: "`build_alpha158_features.py` evaluates qlib expressions that reference `$close`, `$open`, `$high`, `$low`, `$vwap` at the CURRENT day (no .shift(1) anywhere in the alpha158 pipeline). Examples from lines 209-217 (KBAR): `($close-$open)/$open`, `($high-$low)/$open`. Lines 222-225 (PRICE): `$open/$close`, `$high/$close`, `$low/$close`, `$vwap/$close`. Lines 244-251 (ROLLING): `Mean($close>Ref($close,1), d)`, `Sum(Greater($close-Ref($close,1), 0), d)`. Result: `data/reference/sp500_5y_alpha158_features_raw.npy` row t contains expressions evaluated with prices[t]. `run_storya_e1_anchor.py` `build_universe_C` (lines 312-318) loads this npy directly and assigns it to feature_date t, but the label at feature_date t is `prices.shift(-21)/prices - 1`, which requires the feature to be known by T-1 close (Rule 8 universal invariant + plan §1.8 contract). Universe C therefore leaks same-day OHLC into the 21d-ahead prediction. KBAR/PRICE leakage is direct (close/open ratio at day t); ROLLING leakage is via the rolling window endpoint being day t."
    suggested_fix: "Apply `np.roll(arr, 1, axis=0); arr[0]=0` to the alpha158 slice inside `build_universe_C` (matches the news-feature T-1 lag pattern in `archived/scripts/run_horizon_ablation.py:169-171`). Add assertion documenting that the source npy is NOT T-1 lagged and the runtime shift is the corrective. Phase5 features (mom12m, maxret, dolvol, CORR5) DO have .shift(1) baked in per `build_phase5_features.py:78-96` and need no further shift."
    status: FIXED
    resolution_notes: "FIXED 2026-05-26 night, H博士 option A. Applied np.roll(a158_slice, shift=1, axis=0) in build_universe_C immediately after the column slice; row 0 zeroed. Two runtime assertions added: (1) `a158_slice[1] == a158_slice_raw[0]` confirms the shift moved row t→t+1; (2) `a158_slice[0] == 0` confirms zeroing. Verified by Claude in fresh Python: max|C[1,:,:48] - raw[0]| = 0.0 (exact match); max|C[1,:,:48] - raw[1]| = 3.58e+10 (confirms shift was applied, not a no-op). Plan AAA limitation explicitly added to plan §1.9 'Honest caveats' #5 — Plan AAA ranking results are flagged but NOT re-run (BH-FDR 0/61 is directionally robust to leak; Universe C composition based on Plan AAA top-15 acknowledged as leak-influenced but the E1 results themselves are leak-free)."
  - id: CODEX-CR-A-02
    severity: MAJOR
    category: correctness
    claim: "LightGBM callback uses fragile attribute access + bare except mask silent training failures"
    evidence: "Lines ~459-485 of run_storya_e1_anchor.py: `_ValICCallback.__call__` raises `lgb.callback.EarlyStopException(it, [(env.model.boost_round, ...)])` — `env.model.boost_round` may not exist on newer LightGBM versions (modern API uses `current_iteration()`). The outer `try: model.fit(...) except Exception: pass` after callback installation silently swallows ALL exceptions; if fit fails for any reason (memory, malformed data, attribute error), a partial booster is used for prediction and the cell is recorded as 'completed' with garbage predictions."
    suggested_fix: "1) Replace `EarlyStopException` raise with a flag set on `cb` (return True from callback to stop). 2) Replace bare `except Exception: pass` with `except lgb.callback.EarlyStopException:` (and ANY other exception re-raised). 3) Remove `env.model.boost_round` reference — pass iteration directly."
    status: FIXED
    resolution_notes: "FIXED 2026-05-26 night. Refactored train_lightgbm() to use lgb.train() + feval(val_IC) + lgb.early_stopping callback (official LightGBM 4.x API; no env.model attribute introspection). Removed bare `except Exception: pass`. **Bonus discovery during testing**: on macOS M4, importing lightgbm + torch in the same process segfaults (SIGSEGV exit 139) due to libomp / libiomp5 OpenMP runtime conflict. Reproduced bare-Python test confirms; OMP_NUM_THREADS=1 env var BEFORE numpy/torch import is the documented fix. Added platform-conditional os.environ['OMP_NUM_THREADS']='1' at top of script (Darwin-only, no perf impact on Colab A100 Linux). Verified end-to-end: LightGBM cell ran successfully (IC=+0.020, Sharpe_gross=1.41, wall=0.5s) on Universe B fold 0 seed 86."
  - id: CODEX-CR-A-03
    severity: CONCERN
    category: statistics
    claim: "D-05 L1 turnover convention is 2x existing horizon_ablation baseline; column names do not encode the convention"
    evidence: "run_storya_e1_anchor.py `compute_cost_ladder_sharpe` uses turnover_L1 = 2x oneside per plan §1.4(d) D-05 fix. Output columns are Sharpe_net_{0,5,10,15,20,30}bps. The existing experiments/horizon_ablation_results.csv uses oneside convention with column Sharpe_net. Cross-comparison without conversion will produce wrong claims. Plan §1.4(d) acknowledges this; prereg has the locked formula. But the results.csv consumer (downstream compute_e6_dm_spa.py and analyze_storya_results.py) needs to know the convention."
    suggested_fix: "Add a `cost_convention` field to results.csv (string='L1_norm_one_way') and a `turnover_definition` field. Also add a header comment in the .csv via a separate `_meta.json` next to results.csv with the convention spec."
    status: FIXED
    resolution_notes: "FIXED. Added cost_convention='L1_one_way' column to results.csv (verified in smoke output, e.g. row 'L1_one_way' appears for both LightGBM and MLP test cells). Also write_run_meta_json() writes experiments/storya_e1_anchor/_meta.json at startup with cost_ladder.convention + turnover_definition + relation_to_archived + cost_formula + annualization explanation. Downstream consumers (compute_e6_dm_spa.py, analyze_storya_results.py) can read this metadata to detect which convention is in the CSV."
  - id: CODEX-CR-A-04
    severity: CONCERN
    category: reproducibility
    claim: "CSV append uses per-row os.path.exists check for header; multi-session run would duplicate headers"
    evidence: "append_results / append_manifest do `df.to_csv(path, mode='a', header=not os.path.exists(path))`. If two Colab sessions run concurrently and both check os.path.exists at the same instant, both will write headers and produce corrupted CSV."
    suggested_fix: "At script startup, if file does not exist, initialize it with the header row (empty body). Then all subsequent appends use header=False unconditionally."
    status: FIXED
    resolution_notes: "FIXED. Added init_csv_files() called once at startup; defines RESULTS_COLUMNS (23 cols incl. cost_convention) + MANIFEST_COLUMNS (10 cols) constants; pre-writes headers if files don't exist. append_results() and append_manifest() now use header=False unconditionally with explicit `columns=RESULTS_COLUMNS` / `columns=MANIFEST_COLUMNS` parameter to lock column ordering across concurrent appends. Verified smoke output: both results.csv and manifest.csv carry headers as expected even after a single-cell run."
  - id: CODEX-CR-A-05
    severity: CONCERN
    category: reproducibility
    claim: "set_seed lacks torch.backends.cudnn.deterministic / benchmark settings present in archived baseline"
    evidence: "archived/scripts/run_horizon_ablation.py:33-34 sets `torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False` at module load. run_storya_e1_anchor.py set_seed does not. On CUDA (Colab A100), some kernels remain non-deterministic across runs of the same seed."
    suggested_fix: "In set_seed (or at module load), set torch.backends.cudnn.deterministic=True and torch.backends.cudnn.benchmark=False."
    status: FIXED
    resolution_notes: "FIXED. Added torch.backends.cudnn.deterministic=True and torch.backends.cudnn.benchmark=False at module top (immediately after torch import). Verified at runtime via direct attribute inspection: deterministic=True, benchmark=False."
summary:
  critical: 1
  major: 1
  concern: 3
  fixed_before_reply: 5
overall_verdict: PROCEED-WITH-FIXES
---

# Code Review Body

Codex Round A returned 1 CRITICAL + 1 MAJOR + 3 CONCERN. All 5 findings INDEPENDENTLY VERIFIED by Claude reading cited code. **NO FIXES APPLIED YET — awaiting H博士 disposition on CR-A-01 because it has scope implications beyond E1** (the Alpha158 artifact T-1 leak likely affects Plan AAA and any other prior experiment that loaded `sp500_5y_alpha158_features_raw.npy` directly without re-shifting).

**Correction trail (2026-05-26 night)**: Initial draft of this review file prematurely marked all 5 findings as `status: FIXED` (Rule 9 integrity violation — fixes were only designed, not applied). Reverted to `status: OPEN` + `fixed_before_reply: 0` + `overall_verdict: BLOCK-EXECUTION` to honestly report state to H博士 before any code change. After H博士 confirmed option A, all 5 fixes were then actually applied to `run_storya_e1_anchor.py` (plus a bonus macOS OpenMP fix discovered during smoke testing), and end-to-end verified with a LightGBM cell (IC=+0.020, 0.5s) AND an MLP cell (IC=+0.011, 76s). Only NOW (post-verification) are statuses flipped back to FIXED with concrete evidence in each resolution_notes. This matches the Round C → D → E lesson: never mark FIXED on intent; only on evidence.

## CR-A-01 detail (CRITICAL data-leakage)

The root cause is that `build_alpha158_features.py` evaluates qlib expressions like `($close-$open)/$open` directly on day t's price data, without applying `.shift(1)`. This is the qlib-standard convention because in qlib the temporal alignment is handled at the LABEL side (labels are also computed using the same prices), but our pipeline uses LABEL = `prices.shift(-21)/prices - 1` which DOES use day-t close as the price base, so the day-t feature already encodes the same close value used to compute the 21d-ahead label denominator.

The fix is to apply np.roll along the time axis after slicing alpha158 columns in `build_universe_C`. This matches the pattern used for news features in `archived/scripts/run_horizon_ablation.py:169-171` (post-load T-1 shift for already-aggregated features).

Phase5 features used in BOTH Universe B and Universe C's `hc_mom12m` DO have .shift(1) baked in per `build_phase5_features.py:78-96`, so no additional shift is needed for those.

## CR-A-02 detail (MAJOR LightGBM fragility)

The original callback used `env.model.boost_round` which may not exist on modern LightGBM. The bare `except Exception: pass` after `model.fit()` masks ANY failure, allowing a partial model to silently produce garbage predictions.

Fix: switched to the official `lgb.early_stopping` callback with a custom `feval` that computes val IC. This is the LightGBM-documented pattern and works across versions.

## CR-A-03 / 04 / 05 (CONCERNS)

All three addressed:
- CR-A-03: cost convention metadata written to results.csv + _meta.json
- CR-A-04: CSV files pre-initialized at startup
- CR-A-05: cudnn deterministic flags set

## Next action

Claude re-runs startup sanity tests + smoke verification, then triggers Codex Touchpoint 2 Round B to verify all 5 fixes hold under independent scrutiny (lesson from Round C → D protocol: never mark FIXED on intent alone).
