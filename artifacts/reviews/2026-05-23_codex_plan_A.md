---
reviewer: codex
touchpoint: plan
round: A
target_plan: docs/plan_aaa_v0_2026-05-23.md
findings:
  - id: CODEX-A-01
    severity: MAJOR
    category: data-leakage
    claim: "The plan states the correct Alpha158 source but lacks an execution-level provenance check for the known globally winsorized _features.npy leakage bug."
    evidence: "Plan v0 §2.4 says to load _raw.npy but contains no runtime assertion. The 2026-05-22-a Option B bug demonstrated that intent-only documentation is insufficient when contaminated and clean files coexist."
    suggested_fix: "At runtime, assert the Alpha158 input path resolves to _raw.npy; save per-fold train winsor bounds and scaler parameters to artifacts/plan_aaa/audit/; fail the run if _features.npy is referenced anywhere in Plan AAA code."
    status: FIXED
    resolution_notes: "Accepted; Plan v1 §2.4 adds runtime assertion spec + audit log artifact. Verified by reading run_step3_plan_z_part_a.py:81-138 — load_data_and_features() does not currently encode this assertion, so Plan AAA must add it."

  - id: CODEX-A-02
    severity: MAJOR
    category: data-leakage
    claim: "Hand-curated feature timing is underspecified, leaving a forward-looking price leakage risk."
    evidence: "Plan v0 §2 says hc features are 'computed on-the-fly from prices' but does not define as-of dates, return windows, label horizon alignment, or whether close_t is available at prediction time."
    suggested_fix: "Add a formula table for all 10 hc features specifying X_t inputs, label y_t horizon, inclusive/exclusive window endpoints, and prediction-time availability."
    status: FIXED
    resolution_notes: "Accepted; Plan v1 §2.5 adds explicit formula table with as-of dates. Formulas verified against existing analyze_loss_horserace.py:706-734 (load_s6_features) and load_s_price_features, which already use shift(1) to avoid same-day leakage."

  - id: CODEX-A-03
    severity: MAJOR
    category: correctness
    claim: "The grouped permutation operation is ambiguous and could break within-group covariance if columns are shuffled independently."
    evidence: "Plan v0 §3 Step 3 pseudo-code does not state explicitly whether one shared row permutation is applied to all columns in the group."
    suggested_fix: "Define ΔIC using one shared stock-level permutation index per test day and group, applied jointly to every column in that group, with labels, graph edges, dates, masks, and all non-group features unchanged."
    status: FIXED
    resolution_notes: "Accepted. Verified Plan Z++ Part A code at run_step3_plan_z_part_a.py:494-497: `x[:, grp_indices_t] = x[perm][:, grp_indices_t]` — `x[perm]` indexes ROWS by perm first (full panel row permutation), then `[:, grp_indices_t]` selects only group columns, then assigns back. All columns in group get the SAME row permutation. Plan v1 §3.3 makes this explicit + adds runtime assertion."

  - id: CODEX-A-04
    severity: MAJOR
    category: statistics
    claim: "The planned NW-HAC test is not valid as written because the unit of analysis is a correlated model/fold/seed panel, not a single independent time series."
    evidence: "Plan v0 §3 Step 4 proposes 'mean Δ-IC across 30 cells + NW-HAC t-stat'; this collapses the date dimension first, which destroys the time-series structure NW-HAC requires."
    suggested_fix: "Store daily paired deltas ΔIC_{date,cell,group}; collapse CELLS first to get a daily mean series per group, then apply NW-HAC over DATES. Do not run HAC on cell-level means."
    status: FIXED
    resolution_notes: "Accepted; critical statistical fix. Plan v1 §3.4 changes aggregation order: per date d, mean ΔIC across cells → daily series per group → NW-HAC over date dimension. Aligns with how part_a's compute_grouped_permutation_ic outputs daily-IC arrays."

  - id: CODEX-A-05
    severity: MAJOR
    category: correctness
    claim: "The model-cell count is internally inconsistent."
    evidence: "Plan v0 §3 Step 2 says '3 seeds × 5 folds = 30 cells per architecture, 60 total'; the arithmetic is wrong (3×5=15 per arch, 30 total)."
    suggested_fix: "Explicitly enumerate: arch_count × seeds × folds = N total cells; use N consistently in §3.4 denominators, §4 compute, §6 output filenames."
    status: FIXED
    resolution_notes: "Accepted; arithmetic correction. Plan v1: 2 archs × 3 seeds × 5 folds = 30 total cells (15 per arch). All §3.4, §4, §6 updated."

  - id: CODEX-A-06
    severity: MAJOR
    category: data-leakage
    claim: "Using fold-0 train data to define groups for every fold can introduce distribution bias for other folds."
    evidence: "Plan v0 §3 Step 1 defines one Spearman matrix on fold-0 train slice. Verified: fold_manifest_expanding.json fold-0 train=[0,713]; fold-1 train extends to day 774, fold-2 to 837, etc. (expanding). Fold-k (k>0) train includes data not in fold-0 train, so groups may not reflect their feature dependency structure."
    suggested_fix: "Either (a) build fold-specific groups using each fold's training slice (groups not globally comparable, requires rank-stability reporting); or (b) use a strictly pre-experiment calibration window (e.g., 252 days at start of dataset, before any test fold). Plan must commit to (a) or (b) before clustering."
    status: FIXED
    resolution_notes: "Accepted with modification. Plan v1 §3.1 commits to option (b): use first 252 trading days (~2021-01-29 to ~2022-01-29) as the strictly pre-experiment calibration window. Rationale: simpler interpretation (single global ranking), no overlap with any test fold's evaluation period (earliest test starts day 796), and feature correlation structure is reasonably stable within 1 year. Trade-off documented: 252-day window has less data than fold-k train slices, so group structure may differ from Part A's fold-0 (714 days). Sensitivity check at fold-0 train slice (Part A protocol) reported in supplementary table for comparability."

  - id: CODEX-A-07
    severity: CONCERN
    category: statistics
    claim: "The multiple-testing family and resampling details are not fully pre-specified."
    evidence: "Plan v0 §3 Step 4 mentions NW-HAC + bootstrap CI but §8 Q6 still asks 'paired Wilcoxon vs NW-HAC + BH-FDR?'"
    suggested_fix: "Predeclare: primary family = all K discovered groups; FDR method = BH at q=0.05; HAC lag = floor(4 × (T/100)^(2/9)) (Newey-West 1994); bootstrap = stationary block bootstrap over dates with block mean = 21d (matching analyze_tier1_phase_a.py)."
    status: FIXED
    resolution_notes: "Accepted; Plan v1 §3.4 pre-specifies all 4 items."

  - id: CODEX-A-08
    severity: CONCERN
    category: reproducibility
    claim: "Permutation randomness is not reproducibly specified."
    evidence: "Plan v0 §3 Step 3 calls rng.permutation but does not define seed schedule or persist permutation indices."
    suggested_fix: "Define seed_for(cell_id, group_label, date_index) = hash deterministically; save the seed table OR the actual permutation indices used to artifacts/plan_aaa/audit/."
    status: FIXED
    resolution_notes: "Accepted; Plan v1 §3.3 specifies seed_for(seed, group_label, date_idx) = SeedSequence([seed, hash(group_label) % 2^32, date_idx]).generate_state(1)[0] and persists the (cell, group, date)→seed table as artifact. **2026-05-23 stop-hook second-pass PATCH: original v1 fix used Python builtin hash(group_label) which is randomized per process unless PYTHONHASHSEED=0 → non-reproducible. Patched v1 to use stable integer group_id (0..K-1) assigned in groups_168.json instead. Seed = SeedSequence([int(seed), int(group_id), int(date)]).** New §5 hedge #12 + groups_168.json schema in §6 updated. **2026-05-23 stop-hook 3rd-pass PATCH: original 2nd-pass patch claimed permutation 'byte-for-byte reproducible' downstream — overclaim. Permutation INDEX arrays are bit-exact; downstream permuted_preds and IC values are subject to MPS fp32 non-determinism (~1e-5 ulp). Narrowed claim in §3.3 Reproducibility scope; added §5 hedge #13 + environment.json artifact in §6. **2026-05-23 stop-hook 4th-pass PATCH: 3rd-pass language about audit (\"perm indices saved to permutations.parquet\") implied full perm capture but actually only first-10 indices stored. Clarified audit trail scope in §3.3: rng_seed is primary truth source (deterministically regenerates full perm), perm_first10 is spot-check sanity hash, full 501-element perms NOT stored (would be GB-scale). Replay protocol: load record → regenerate full perm from seed → verify perm[:10] == stored perm_first10. **2026-05-23 stop-hook 5th-pass PATCH: 4th-pass code+doc still inconsistent — code used SeedSequence(...).generate_state(1)[0] returning uint32 (4 bytes = 32-bit entropy), but doc claimed '64-bit rng_seed (8 bytes per record)'. 32 bits + 84K records = nontrivial birthday collision. Fix: pass SeedSequence directly to default_rng (full 128-bit PCG64 entropy via SHA256 mixing); audit truth source switched from stored rng_seed to stored INPUT TRIPLE (cell_seed, group_id, date); permutations.parquet schema column rng_seed REMOVED, replaced by cell_seed; replay protocol regenerates SeedSequence from triple, no uint32 intermediate. **2026-05-23 stop-hook 6th-pass PATCH: 5th-pass triple used cell_seed (86/123/456) which aliases across cells with same seed_idx (e.g., SAGE/fold0/seed86 and MLP/fold0/seed86 share triple → audit uniqueness assertion fails, replay can't distinguish cells). Fix: define cell_id = arch_idx*15 + fold_idx*3 + seed_idx (range 0..29, globally unique). Triple becomes (cell_id, group_id, date), globally unique across all 30 × K × N_dates records. permutations.parquet schema updated: cell_seed column → cell_id (replay primary) + cell_seed_value (decorative). Uniqueness assertion at audit write time now actually holds and is enforceable."

  - id: CODEX-A-09
    severity: CONCERN
    category: other
    claim: "The 6-8 hour compute estimate may be low because the stated inference arithmetic alone is about 4 hours."
    evidence: "30 cells × 40 groups × 200 test days × 30ms = 7,200s = 2h (Codex says 4h assuming 60 cells; with corrected 30 cells it's ~2h)."
    suggested_fix: "Benchmark ms/day/model in smoke test, then report projected training time, baseline inference time, permutation inference time, and total for K=30/40/50."
    status: FIXED
    resolution_notes: "Accepted; Plan v1 §4 adds smoke-test calibration step + revised estimate. With 30 cells (corrected per A-05): training ~3h, baseline inference ~30min, permutation ~2-3h, analysis ~30min = ~6-7h M4. Smoke test to confirm before full run."

  - id: CODEX-A-10
    severity: CONCERN
    category: correctness
    claim: "The execution protocol lacks a negative-control sanity check for the permutation pipeline."
    evidence: "Plan v0 §9 lists smoke test + full run, but no identity-permutation or null-feature-group control."
    suggested_fix: "Add smoke-only control: (1) identity permutation expected ΔIC ≈ 0; (2) randomly constructed noise group expected ΔIC ≈ 0."
    status: FIXED
    resolution_notes: "Accepted; Plan v1 §9 step 4a adds two negative controls to smoke test."

  - id: CODEX-A-11
    severity: CONCERN
    category: reproducibility
    claim: "The plan does not pre-specify how to handle failed or non-converged 168-dim training cells."
    evidence: "Plan v0 §8 Q8 leaves convergence handling open."
    suggested_fix: "Define before running: convergence = train loss decreases >1% from epoch 1 to early-stop; if >20% cells fail, halt + report (no auto-exclude); any retry uses same seed + hparams (no tuning)."
    status: FIXED
    resolution_notes: "Accepted; Plan v1 §3.2 adds convergence criteria + halt-rule."

  - id: CODEX-A-12
    severity: CONCERN
    category: prior-art
    claim: "The paper integration should avoid implying methodological novelty without positioning against existing feature-attribution literature."
    evidence: "Plan v0 §7.2 adds 'Universe Sensitivity Analysis' but doesn't cite Strobl et al. 2008 (conditional permutation importance) or Lundberg & Lee 2017 (SHAP)."
    suggested_fix: "Frame Plan AAA as 'domain-specific production-model sensitivity analysis adapting grouped cross-sectional permutation importance' with explicit Strobl 2008 + Lundberg & Lee 2017 citations."
    status: FIXED
    resolution_notes: "Accepted; Plan v1 §7.2 adds prior-art framing + citations."

summary:
  critical: 0
  major: 6
  concern: 6
  fixed_before_reply: 12
overall_verdict: BLOCK-EXECUTION
post_review_status: All 6 MAJOR + 6 CONCERN findings accepted and incorporated into Plan AAA v1 (docs/plan_aaa_v1_2026-05-23.md). Round B review recommended before execution.
---

# Codex Round A Review — Plan AAA v0

## Summary Verdict

**BLOCK-EXECUTION** — 0 CRITICAL + 6 MAJOR + 6 CONCERN findings.

Codex 判定 Plan AAA v0 不能直接执行，但 logic 是 sound 的、p-hacking hedges (§5) 是 thorough 的。所有 12 个 finding 都是 fixable 修复后可以重新提交 Round B。

## Per-finding Verification + Disposition

All 6 MAJOR + 6 CONCERN findings ACCEPTED. Verification performed:

- **A-01** (`_raw.npy` audit): Verified `run_step3_plan_z_part_a.py:81-138` (`load_data_and_features`) does NOT currently encode raw-file assertion. Plan v1 adds it.
- **A-02** (hc feature formulas): Verified existing `analyze_loss_horserace.py:706-734` uses `shift(1)` for momentum features (no same-day leakage). Plan v1 adds explicit formula table to make assumptions auditable.
- **A-03** (shared row perm): Verified `run_step3_plan_z_part_a.py:494-497` does `x[:, grp_indices_t] = x[perm][:, grp_indices_t]` — `x[perm]` row-permutes the full panel first, then `[:, grp_indices_t]` selects group columns. **All columns in group share the same row permutation**. Plan v1 makes this explicit + adds runtime assertion.
- **A-04** (NW-HAC on panel): Confirmed: Plan v0 ambiguous about whether dates or cells get collapsed first. Statistical methodology fix: cells first → daily series per group → NW-HAC over dates.
- **A-05** (cell count): 2 archs × 3 seeds × 5 folds = 30 total (15 per arch); Plan v0 incorrectly said 60. Arithmetic fix.
- **A-06** (fold-0 clustering bias): Verified `fold_manifest_expanding.json` — fold-0 train = [0, 713], fold-1 train extends to 774, etc. Expanding window means fold-k (k>0) train includes data not in fold-0 train. Plan v1 commits to **option (b)**: use first 252 trading days (~2021-01 to ~2022-01) as strictly pre-experiment calibration window for clustering. No overlap with any test fold (earliest test starts day 796).
- **A-07** to **A-12**: Pre-commitment + reproducibility + prior-art fixes accepted as specified.

## Round B Trigger

Plan AAA v1 (docs/plan_aaa_v1_2026-05-23.md) is ready for Codex Round B review before execution. H博士 decides Round B vs proceed-to-execution.
