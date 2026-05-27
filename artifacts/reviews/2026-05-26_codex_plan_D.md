---
reviewer: codex
touchpoint: plan
round: D
target_plan: /Users/heruixi/.claude/plans/handoff-session-ranking-swirling-lemur.md
target_files:
  - /Users/heruixi/.claude/plans/handoff-session-ranking-swirling-lemur.md
  - /Users/heruixi/Desktop/GNN-Testing/experiments/storya_multiseed/prereg.json
  - /Users/heruixi/Desktop/GNN-Testing/artifacts/reviews/2026-05-26_codex_plan_C.md
  - /Users/heruixi/Desktop/GNN-Testing/experiments/storya_e3_news_edge/news_edge_source_schema.md
findings:
  - id: CODEX-D-01
    severity: CRITICAL
    category: correctness
    claim: "Plan §1.1 cell_id formula is non-injective AND has wrong range claim — universe_idx*100 + model_idx*25 + fold_idx*5 + seed_idx gives max=204 not 399, with collisions like (fold=0,seed=5)==(fold=1,seed=0)==5."
    evidence: "plan line 246-250 `cell_id = universe_idx*100 + model_idx*25 + fold_idx*5 + seed_idx` with universe_idx ∈ {0,1}, model_idx ∈ {0..3}, fold_idx ∈ {0..4}, seed_idx ∈ {0..9}. Math: max = 1*100 + 3*25 + 4*5 + 9 = 204. Collision: (0,0,0,5) = 5 = (0,0,1,0). prereg.json:167 has CORRECT formula `universe_idx*200 + model_idx*50 + fold_idx*10 + seed_idx` (max = 399, injective)."
    suggested_fix: "Plan §1.1 must use prereg.json:167 formula. Add startup assertion: `assert len(set(cell_ids)) == 400, 'cell_id collision detected'`"
    status: OPEN
    resolution_notes: null
  - id: CODEX-D-02
    severity: MAJOR
    category: data-leakage
    claim: "C-02 fix is too weak — the Python assertion in plan §1.8 collapses to a trivial check (train_end_idx < val_end_idx) that cannot detect 21d label window overlap."
    evidence: "plan §1.8 Python code: `assert train_feature_end + 21 < val_end_idx - 21 + 21` simplifies to `train_feature_end + 21 < val_end_idx`, then `train_end_idx - 21 + 21 < val_end_idx` = `train_end_idx < val_end_idx` (always true by chronological fold structure). Does NOT catch a genuine leak. Also: prereg.json:121-125 still has the v3 between-fold embargo formula that was supposed to be removed."
    suggested_fix: "Use real label-feature gap: `assert max(train_label_end_idx) + embargo < min(val_feature_idx)` and similarly val→test. Use trading-day index arrays explicitly. Delete prereg between-fold embargo entry."
    status: OPEN
    resolution_notes: null
  - id: CODEX-D-03
    severity: MAJOR
    category: data-leakage
    claim: "C-03 schema doc timezone logic is wrong for US stock prediction — uses UTC midnight as end_of_day(t-1) cutoff, but US market close is NYSE 16:00 ET (~21:00 UTC), leaving a 2-3 hour after-hours window where news could leak."
    evidence: "experiments/storya_e3_news_edge/news_edge_source_schema.md:101-109 defines `t_minus_1_end = pd.Timestamp(prediction_date_t, tz='UTC') - pd.Timedelta(seconds=1)` = UTC 23:59:59 of (t-1 calendar date in UTC). NYSE closes at 16:00 ET = 21:00 UTC (winter) / 20:00 UTC (summer). Articles published between NYSE close and UTC midnight (~2-3 hours of after-hours) would be incorrectly admitted as 'PIT-safe' by current logic but are NOT available to a real trader making predictions at NYSE close."
    suggested_fix: "Define cutoff using NYSE calendar: `nyse_close_t_minus_1_ET = pd.Timestamp(f'{date_t_minus_1} 16:00:00', tz='America/New_York'); cutoff_utc = nyse_close_t_minus_1_ET.tz_convert('UTC')`. Update schema doc + plan §1.2 + runtime assertion to use this."
    status: OPEN
    resolution_notes: null
  - id: CODEX-D-04
    severity: MAJOR
    category: statistics
    claim: "SPA M=40 is inconsistent with C-04 seed aggregation rule — if seeds are pre-averaged for DM/HLN per (model, fold, date), then SPA candidates collapse to M=4 (just models), not M=40. Plan §1.4 has both M=40 statement and seed-aggregation logic."
    evidence: "plan §1.4 line ~394 says `M = 4 models × 10 seeds (Universe B)` (M=40); plan §1.4 line ~397 says `v3 seed aggregation ... averaged per (model, date, fold) BEFORE being included in the SPA loss matrix`. These contradict: if pre-averaged, only 4 series exist per universe (one per model). E4-α §1.3 also does not specify seed aggregation for edge-vs-edge DM."
    suggested_fix: "Pick one: (a) SPA M=4 per universe (model-level, consistent with C-04 — minimal cherry-pick risk because only 4 candidates); OR (b) SPA M=40 (seed-level candidates, treats each (model,seed) as separate strategy — different statistical interpretation). Plan + prereg must agree. Also pre-register seed aggregation for E4-α DM."
    status: OPEN
    resolution_notes: null
  - id: CODEX-D-05
    severity: MAJOR
    category: correctness
    claim: "Cost ladder turnover formula gives ~4 not ~2 for full L-S rebalance, and N=15 effective sample is wrong (actual is 17 per existing horizon_ablation_results.csv)."
    evidence: "plan §1.4 line ~498-505 says `turnover ≈ 2 (entire portfolio recycled)` but formula `sum_i |pos_i(t) - pos_i(t-1)|` for full L-S rebalance: old positions ±1/K → 0 contributes sum |old| = 2; new positions 0 → ±1/K contributes sum |new| = 2; total = 4. Also: plan claims N=15 (5 folds × 3 periods) but `experiments/horizon_ablation_results.csv` n_periods column shows 3/4/4/3/3 = 17 across the 5 folds."
    suggested_fix: "Pick turnover convention (full = 2 per leg vs full = 4 round-trip); document which. Update N to 17 in plan + prereg + bootstrap CI spec. Good news: sqrt(252/21) annualization matches existing `run_horizon_ablation.py:379-386` ppy=252/horizon, so historical Sharpe_net values are still comparable."
    status: OPEN
    resolution_notes: null
  - id: CODEX-C-01-ROUND-D
    severity: CRITICAL
    category: reproducibility
    claim: "Plan still has residual v2 text after Option B+ — line 494 mentions 8-test family, line 616 mentions compute_dsr.py, line 1035 has 500-cell smoke, line 1072 has 5-cell smoke."
    evidence: "Independent re-grep of plan file shows v2 text NOT fully cleaned at the locations Round C flagged. C-01 Round D status: NEW-CONCERN (cleanup incomplete + new D-01 formula error)."
    suggested_fix: "Final pass over plan to clean ALL residual v2 references — `grep -n '5 models\\|LSTM\\|compute_dsr\\|M=50\\|M=100\\|1450\\|500 cells'` and address each match."
    status: STILL-OPEN
    resolution_notes: "Round C marked C-01 as FIXED but Round D verification found residual text at lines 494/616/1035/1072. Plus D-01 cell_id formula error is a NEW correctness issue on top of cleanup incompleteness."
  - id: CODEX-C-02-ROUND-D
    severity: MAJOR
    category: data-leakage
    claim: "C-02 fix mechanism applied (plan §1.8 rewritten) but assertion code is too weak; prereg between-fold embargo formula not deleted."
    evidence: "See D-02 above."
    suggested_fix: "See D-02 above."
    status: STILL-OPEN
    resolution_notes: "Round C marked C-02 FIXED but Round D verification found assertion code collapses to trivial check; prereg still has old between-fold formula."
  - id: CODEX-C-03-ROUND-D
    severity: MAJOR
    category: data-leakage
    claim: "C-03 schema doc is implementable in concept but UTC timezone cutoff is wrong for US stock prediction (NYSE close at 21:00 UTC, not midnight)."
    evidence: "See D-03 above."
    suggested_fix: "See D-03 above."
    status: NEW-CONCERN
    resolution_notes: "Round C marked C-03 FIXED but Round D verification found timezone bug. Codex notes: `news_articles_raw.parquet` file does not exist (dtype audit skipped); existing audit was on `sp500_news_events.parquet` which Claude already verified."
  - id: CODEX-C-04-ROUND-D
    severity: MAJOR
    category: statistics
    claim: "C-04 seed aggregation for E1 DM/HLN is correctly fixed; SPA M and E4-α DM seed aggregation are NOT aligned."
    evidence: "See D-04 above."
    suggested_fix: "See D-04 above."
    status: NEW-CONCERN
    resolution_notes: "Round C C-04 fix applies only to E1 DM/HLN. SPA M-value inconsistency and E4-α aggregation rule are new gaps discovered in Round D."
  - id: CODEX-C-05-ROUND-D
    severity: MAJOR
    category: correctness
    claim: "C-05 fix structure applied (equal-weight + bps + annualization) but turnover formula numerical claim is wrong (~4 not ~2) and N is wrong (17 not 15)."
    evidence: "See D-05 above."
    suggested_fix: "See D-05 above."
    status: STILL-OPEN
    resolution_notes: "Round C marked C-05 FIXED but Round D found numerical errors in turnover constant and N. Good: sqrt(252/21) annualization confirmed correct (matches existing horizon_ablation.py)."
  - id: CODEX-C-06-ROUND-D
    severity: MAJOR
    category: prior-art
    claim: "C-06 deferral to paper writing phase is acceptable for v3 EXECUTION."
    evidence: "Plan §1.9 has appropriate caveats about novelty being conditional on matrix verification."
    suggested_fix: "Maintain deferral; address before paper submission."
    status: VERIFIED-FIXED
    resolution_notes: "Round D confirms: deferral to paper writing phase is sound for v3 execution; novelty claim in §1.9 is appropriately caveated."
  - id: CODEX-C-07-ROUND-D
    severity: CONCERN
    category: statistics
    claim: "C-07 MDE rule is operational — prereg line 211 pre-registers threshold |CI upper| < 0.005 and inconclusive framing."
    evidence: "Confirmed by independent read of prereg.json line 211."
    suggested_fix: "None needed."
    status: VERIFIED-FIXED
    resolution_notes: "Round D verified prereg has operational MDE pre-commitment at correct threshold."
  - id: CODEX-C-08-ROUND-D
    severity: CONCERN
    category: reproducibility
    claim: "C-08 B/C HP-transfer caveat accepted as paper §Limitations — no plan/prereg change needed."
    evidence: "Plan and prereg both lock identical hparams for B and C; framing as 'fixed-default robustness' is honest and traceable."
    suggested_fix: "Add to paper §Limitations during writing phase."
    status: VERIFIED-FIXED
    resolution_notes: "Round D confirms execution not blocked; paper writing must include §Limitations note."
summary:
  critical: 1
  major: 4
  concern: 0
  round_c_dispositions_verified: 8
  round_c_verified_fixed: 3
  round_c_still_open: 2
  round_c_new_concern: 3
overall_verdict: BLOCK-EXECUTION
---

# Round D Review Body

## Overall Assessment

Round D verifies Option B+ fixes claimed in Round C dispositions. Of 8 Round C findings:
- **3 confirmed FIXED**: C-06 (deferral sound), C-07 (MDE pre-registered correctly), C-08 (paper §Limitations sufficient)
- **2 STILL-OPEN**: C-02 (assertion too weak + prereg residual), C-05 (turnover formula + N math wrong)
- **3 NEW-CONCERN**: C-01 (cleanup incomplete + new D-01 formula error), C-03 (timezone bug), C-04 (SPA M / E4-α aggregation not aligned)

PLUS 5 new D-series findings, 1 CRITICAL and 4 MAJOR. Verdict remains BLOCK-EXECUTION.

The Option B+ approach was correct in spirit — fix the actionable Round C findings in one pass. But the fixes were **applied too quickly without independent verification**: the cell_id formula has a math error, the assertion code collapses to trivially true, the turnover constant and N are wrong, the timezone assumption is wrong, and the SPA M is inconsistent with the C-04 fix.

## Key Findings Detail

### CODEX-D-01 (CRITICAL): cell_id formula non-injective

This is a Python-level bug that would silently corrupt checkpoint/resume logic in `run_storya_e1_anchor.py`. Two different cells would write to the same manifest row.

**Plan formula** (`universe_idx*100 + model_idx*25 + fold_idx*5 + seed_idx`):
- universe ∈ {0,1}, model ∈ {0..3}, fold ∈ {0..4}, seed ∈ {0..9}
- Max value = 1×100 + 3×25 + 4×5 + 9 = 100 + 75 + 20 + 9 = **204**
- Plan claims "range 0..399 globally unique" — both range and uniqueness false

**Prereg formula** (`universe_idx*200 + model_idx*50 + fold_idx*10 + seed_idx`):
- Max value = 1×200 + 3×50 + 4×10 + 9 = 200 + 150 + 40 + 9 = **399** ✓
- Radix encoding: each axis multiplier matches product of lower-axis cardinalities (200 = 4×5×10, 50 = 5×10, 10 = 10), guaranteeing injectivity

Plan must adopt prereg formula and add startup assertion `assert len(set(cell_ids)) == 400`.

### CODEX-D-03 (MAJOR): Timezone bug in C-03 schema

US stock prediction at trading day t uses information available at NYSE close of trading day t-1 (~16:00 ET = 21:00 UTC winter / 20:00 UTC summer). Articles published between NYSE close and UTC midnight of the next day are **after** the prediction moment in real trading but **before** the cutoff in the current schema doc (which uses UTC midnight - 1 second).

This is a real ~2-3 hour leak window. Fix:
```python
import exchange_calendars as xcals
nyse = xcals.get_calendar('XNYS')
nyse_close_et = nyse.session_close(date_t_minus_1_session)  # tz-aware ET
cutoff_utc = nyse_close_et.tz_convert('UTC')
eligible = edge_source_df[edge_source_df['publication_timestamp'] <= cutoff_utc]
```

### CODEX-D-04 (MAJOR): SPA M inconsistency

If C-04 mandates "average IC across seeds per (model, date, fold) BEFORE forming the DM series" for pairwise tests, then for SPA the candidate family is also seed-averaged → M = 4 (per universe), not 40. Two reasonable resolutions:
- **Option A**: SPA M = 4, treats each model as one strategy. Minimal cherry-pick burden but small family.
- **Option B**: SPA M = 40, treats each (model, seed) as separate strategy. Larger cherry-pick burden but more conservative.

Both are defensible; the plan must pick one and apply consistently. Currently the plan has both M=40 statement AND seed-aggregation logic, which is a contradiction.

### CODEX-D-05 (MAJOR): Turnover + N math errors

Full L-S rebalance with no overlap from prior period (which is the case for 21d non-overlapping):
- Old positions ±1/K → 0: contributes `sum |old - 0| = sum |old| = 2` (top K of 1/K + bottom K of 1/K)
- New positions 0 → ±1/K: contributes `sum |new - 0| = 2`
- Total turnover = **4**, not 2

For N effective periods, the existing `horizon_ablation_results.csv` `n_periods` column shows actual count per fold: 3 + 4 + 4 + 3 + 3 = **17**, not 15. (Different folds have slightly different test window lengths.)

Good news: `sqrt(252/21)` annualization matches `archived/scripts/run_horizon_ablation.py:379-386` `ppy = 252 / horizon`, so historical Sharpe_net values in `horizon_ablation_results.csv` ARE directly comparable to new v3 numbers (no rebuild needed for baseline).

## Path to PROCEED-WITH-FIXES (Round E)

Tightly-scoped fix list, all in plan + prereg + schema doc (no new experiments needed):

1. **D-01 + C-01 residual**: Replace plan §1.1 cell_id formula with prereg formula; add startup assertion; final grep pass over plan to clean residual `5 models / LSTM / compute_dsr / M=50 / 1450` text at lines 494, 616, 1035, 1072.
2. **D-02 + C-02**: Rewrite §1.8 assertion code to use real label-feature index gap; delete prereg lines 121-125 (old between-fold embargo).
3. **D-03 + C-03**: Update schema doc and plan §1.2 cutoff to use NYSE session_close converted to UTC; update runtime assertion accordingly.
4. **D-04 + C-04**: Pick SPA M=4 or M=40 explicitly; pre-register E4-α DM seed aggregation rule (same as E1 DM).
5. **D-05 + C-05**: Update turnover constant (≈4) and N (=17) in plan + prereg + bootstrap CI spec. Note that sqrt(252/21) annualization is OK as-is.

Estimated total fix time: ~1-1.5 hours. After fixes, trigger Round E to verify.

Recommendation: **Do NOT** trigger Round E until all 5 fixes are applied with explicit code/line citation per fix. Round C → D revealed that surface-level "FIXED" claims need independent verification.
