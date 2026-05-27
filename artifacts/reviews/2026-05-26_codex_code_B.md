---
reviewer: codex
touchpoint: code
round: B
target_files:
  - /Users/heruixi/Desktop/GNN-Testing/run_storya_e1_anchor.py
findings:
  - id: CODEX-CR-A-01-ROUND-B
    severity: CRITICAL
    category: data-leakage
    claim: "Round A CR-A-01 Alpha158 T-1 leak fix re-verified"
    evidence: |
      run_storya_e1_anchor.py:125: 'alpha158_npy': 'data/reference/sp500_5y_alpha158_features_raw.npy',
      run_storya_e1_anchor.py:364: alpha158_arr = np.load(PATHS['alpha158_npy'])  # (T, N, 158)
      run_storya_e1_anchor.py:370: a158_slice_raw = alpha158_arr[:, :, a158_cols].astype(np.float32)  # (T, N, 48), SAME-DAY
      run_storya_e1_anchor.py:373: a158_slice = np.roll(a158_slice_raw, shift=1, axis=0)
      run_storya_e1_anchor.py:374: a158_slice[0] = 0.0
      run_storya_e1_anchor.py:377-380: assert np.array_equal(a158_slice[1], a158_slice_raw[0]); assert np.all(a158_slice[0] == 0.0)
    status: FIXED
    resolution_notes: |
      Verified np.roll uses shift=1 on axis=0, then zeroes row index 0. The two assertions would fail for an unshifted/no-op slice: row 1 must equal raw row 0, and row 0 must be all zeros. The raw npy load is direct at line 364 and the raw slice at line 370 is not pre-shifted; the only np.roll near the alpha158 load site is the required runtime shift at line 373.
  - id: CODEX-CR-A-02-ROUND-B
    severity: MAJOR
    category: correctness
    claim: "Round A CR-A-02 LightGBM fragility fix re-verified"
    evidence: |
      run_storya_e1_anchor.py:44-45: if platform.system() == 'Darwin': os.environ['OMP_NUM_THREADS'] = '1'
      run_storya_e1_anchor.py:61: import numpy as np
      run_storya_e1_anchor.py:63: import torch
      run_storya_e1_anchor.py:685-696: def val_ic_feval(...); return ('val_IC', float(...), True)
      run_storya_e1_anchor.py:713-725: booster = lgb.train(..., feval=val_ic_feval, callbacks=[lgb.early_stopping(...), ...])
      run_storya_e1_anchor.py:727: best_iter = booster.best_iteration if booster.best_iteration else booster.current_iteration()
    status: FIXED
    resolution_notes: |
      Verified train_lightgbm uses lgb.train, not lgb.LGBMRegressor.fit. The custom feval returns the LightGBM 4.x tuple order (name, value, higher_is_better). best_iteration is read from the Booster, not model.best_iteration_. OMP_NUM_THREADS is set before numpy and torch imports and only inside if platform.system() == 'Darwin', so Linux does not receive that environment override.
  - id: CODEX-CR-A-03-ROUND-B
    severity: CONCERN
    category: statistics
    claim: "Round A CR-A-03 cost convention metadata fix re-verified with exact-key mismatch"
    evidence: |
      run_storya_e1_anchor.py:846-851: RESULTS_COLUMNS ends with 'cost_convention'
      run_storya_e1_anchor.py:855: COST_CONVENTION = 'L1_one_way'
      run_storya_e1_anchor.py:880-884: cost_ladder contains 'convention': COST_CONVENTION, 'turnover_definition': ..., and 'turnover_relation_to_archived': ...
      run_storya_e1_anchor.py:1118: 'cost_convention': COST_CONVENTION
      AST check of cost_ladder keys: ['levels_bps', 'convention', 'turnover_definition', 'turnover_relation_to_archived', 'cost_formula', 'annualization', 'note']; exact key 'relation_to_archived' is absent.
    status: FIXED
    resolution_notes: |
      FIXED 2026-05-26 night post Round B by Claude (1-line rename in write_run_meta_json).
      Independently verified via Python read:
        cost_ladder keys: ['levels_bps', 'convention', 'turnover_definition', 'relation_to_archived', 'cost_formula', 'annualization', 'note']
        relation_to_archived value: 'L1 = 2 * one_side (used in archived/scripts/run_horizon_ablation.py:370)'
      Old key 'turnover_relation_to_archived' confirmed absent. H博士 chose option B (skip Round C verification of this 1-char cosmetic rename; accept self-verification + proceed to smoke benchmark) per progress.md 2026-05-26-k.
  - id: CODEX-CR-A-04-ROUND-B
    severity: CONCERN
    category: reproducibility
    claim: "Round A CR-A-04 CSV header race fix re-verified"
    evidence: |
      run_storya_e1_anchor.py:858-865: init_csv_files() pre-initializes RESULTS_CSV and MANIFEST_CSV with pd.DataFrame(columns=...).to_csv(..., index=False)
      run_storya_e1_anchor.py:962: init_csv_files() is called once during startup before feature construction and before the cell loop
      run_storya_e1_anchor.py:911-914: append_manifest builds pd.DataFrame([row], columns=MANIFEST_COLUMNS) and writes header=False
      run_storya_e1_anchor.py:917-920: append_results builds pd.DataFrame([row], columns=RESULTS_COLUMNS) and writes header=False
    status: FIXED
    resolution_notes: |
      Verified startup header initialization for both CSVs and unconditional header=False in both appenders. The append functions pass explicit columns= parameters and do not call os.path.exists. Separate AST checks found the manifest row dict exactly matches MANIFEST_COLUMNS, and the results row dict exactly matches RESULTS_COLUMNS with no extra or missing keys.
  - id: CODEX-CR-A-05-ROUND-B
    severity: CONCERN
    category: reproducibility
    claim: "Round A CR-A-05 cudnn deterministic fix re-verified"
    evidence: |
      run_storya_e1_anchor.py:163-165: module-level torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
      run_storya_e1_anchor.py:168-173: set_seed only sets random, numpy, torch, and cuda seeds; it does not contain the cudnn backend flags
      rg found torch.backends.cudnn only on lines 164-165.
    status: FIXED
    resolution_notes: |
      Verified both cudnn flags are set at module load level, outside set_seed and outside any other function.
summary:
  critical: 0
  major: 0
  concern: 0
  fixed_before_reply: 0
overall_verdict: BLOCK-EXECUTION
---

## CODEX-CR-A-01-ROUND-B

Status: FIXED.

`build_universe_C()` now loads `data/reference/sp500_5y_alpha158_features_raw.npy` directly at line 364, slices same-day raw features at line 370, then applies the T-1 runtime correction at line 373:

```python
a158_slice = np.roll(a158_slice_raw, shift=1, axis=0)
a158_slice[0] = 0.0
```

The roll is on axis 0, the time axis, and the zeroing targets row index 0. Lines 377-380 assert both required invariants: shifted row 1 equals raw row 0, and shifted row 0 is all zeros. Search of the alpha158 load site found no pre-shift before this roll, so there is no double-shift in `run_storya_e1_anchor.py`.

## CODEX-CR-A-02-ROUND-B

Status: FIXED.

`train_lightgbm()` uses `lgb.train()` at lines 713-725. The `val_ic_feval` callback returns `('val_IC', float(...), True)` at line 696, which is the LightGBM 4.x order `(name, value, higher_is_better)`. The prediction iteration uses `booster.best_iteration` with fallback to `booster.current_iteration()` at line 727.

The macOS OpenMP block is syntactically conditional: lines 44-45 set `OMP_NUM_THREADS=1` only when `platform.system() == 'Darwin'`. This occurs before `numpy` at line 61 and `torch` at line 63, and therefore does not set `OMP_NUM_THREADS` on Linux.

## CODEX-CR-A-03-ROUND-B

Status: STILL-OPEN.

The CSV portion is fixed. `RESULTS_COLUMNS` ends with `cost_convention` at lines 846-851, `COST_CONVENTION = 'L1_one_way'` is defined at line 855, and result rows populate `'cost_convention': COST_CONVENTION` at line 1118.

The metadata portion is not exact. `write_run_meta_json()` writes:

```python
'convention': COST_CONVENTION,
'turnover_definition': 'L1-norm: turnover_L1 = sum_i|p_i(t) - p_i(t-1)|; at full L-S rotation = 4',
'turnover_relation_to_archived': 'L1 = 2 * one_side (used in archived/scripts/run_horizon_ablation.py:370)',
```

The verification table asked for a key named `relation_to_archived`. The implementation writes `turnover_relation_to_archived` instead. Because the requested key is absent, CR-A-03 remains still open under the exact verification contract.

## CODEX-CR-A-04-ROUND-B

Status: FIXED.

`init_csv_files()` pre-initializes both CSV files with headers at lines 858-865. `main()` calls it once at line 962 before data loading, feature construction, and the cell loop. The appenders now enforce column order and always append without headers:

```python
df = pd.DataFrame([row], columns=MANIFEST_COLUMNS)
df.to_csv(path, mode='a', header=False, index=False)

df = pd.DataFrame([row], columns=RESULTS_COLUMNS)
df.to_csv(path, mode='a', header=False, index=False)
```

AST verification found the result row keys exactly match the 23 `RESULTS_COLUMNS`, including `Sharpe_net_0bps`, `Sharpe_net_5bps`, `Sharpe_net_10bps`, `Sharpe_net_15bps`, `Sharpe_net_20bps`, and `Sharpe_net_30bps`. The manifest row keys exactly match the 10 `MANIFEST_COLUMNS`.

## CODEX-CR-A-05-ROUND-B

Status: FIXED.

The cudnn flags are module-level statements at lines 163-165:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

They are not inside `set_seed()` or another function. `set_seed()` at lines 168-173 only sets Python, NumPy, PyTorch, and CUDA seeds.

## Round B New Bug Scan

No new Round B findings were opened. The np.roll axis, feval tuple order, result row schema, manifest row schema, macOS-only OMP conditional, and alpha158 double-shift checks all passed. The only blocker is the still-open CR-A-03 exact metadata key mismatch.

Overall verdict: BLOCK-EXECUTION, because the verdict rules require BLOCK-EXECUTION for any Round A finding that remains STILL-OPEN.
