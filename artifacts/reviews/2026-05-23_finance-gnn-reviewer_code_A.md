---
reviewer: finance-gnn-reviewer
touchpoint: code
round: A
target_files:
  - run_plan_aaa_168_ranking.py:1-1106
target_plan: docs/plan_aaa_v1_2026-05-23.md
fallback_reason: "Codex CLI hit rate limit 2026-05-23"
findings:
  - id: FINGNN-CODE-A-01
    severity: MAJOR
    category: data-leakage
    claim: "Per-fold p1/p99 winsorization promised in Plan v1 §2.4 is not implemented in either smoke or full mode; only the per-fold scaler is applied. The `_winsor_p1_p99` helper exists but is only invoked inside `compute_groups` (clustering), never in the training path."
    evidence: "run_plan_aaa_168_ranking.py:661 comment 'Per-fold winsor + scaler (inherit pattern)' followed by only `pa.fit_feature_scaler(...)` at 662 and `pa.apply_scaler(...)` at 663 — no winsor call. Same omission at 944-945 in full mode. Plan v1 §2.4: 'Per-fold train-only winsor at p1/p99 + per-fold scaler (Plan Z++ Phase 0 protocol).' Note also that Plan Z++ Part A (the referenced parent script) likewise contains no winsor — grep'winsor' in run_step3_plan_z_part_a.py returns 0 hits. So 'inherit pattern' inherits a no-op."
    suggested_fix: "Either (a) implement per-fold winsor fit on train_days and apply to all splits, matching Plan v1 §2.4 promise; or (b) edit Plan v1 §2.4 to remove the winsor language and rely on the scaler + nan_to_num path that part_a actually uses. The 158-dim Alpha158 panel has ROC5 max ~2.4 (raw signature) and could carry similarly heavy tails on other columns, so leaving winsor out introduces a discrepancy with the stated protocol but is unlikely to invalidate the Δ-IC ranking (permutation is rank-based)."
    status: FIXED
    resolution_notes: "Chose option (a): imported canonical `per_fold_winsorize` from `run_tier1_phase_a.py:132-151` (Plan Z++ Phase 0 helper, p1/p99, train-only fit). Applied in smoke mode (run_plan_aaa_168_ranking.py:670), full mode (~952), and smoke 4b noise control (~727). Winsor bounds also stored in `audit/per_fold_scaler.json`. Smoke v4 confirms: train loss decrease 4.5% (was 6% pre-winsor, both > 1% threshold); val IC best improved +0.068 → +0.105; baseline IC stable +0.0457 → +0.0455; 4c real-perm ΔIC dropped +0.00375 → +0.00155 (winsor reduces tail influence so permutation perturbation is smaller in magnitude, sign unchanged)."
  - id: FINGNN-CODE-A-02
    severity: MAJOR
    category: correctness
    claim: "Alpha158 ticker alignment is asserted only on the cardinality (N_a == N_hc), not on the actual ticker ordering. The script assumes the Alpha158 build script used the identical `sorted(prices ∩ events ∩ sectors)` intersection but never verifies it at runtime."
    evidence: "run_plan_aaa_168_ranking.py:138-146 — only `T_hc == T_a` and `N_hc == N_a` are checked. The comment at 132-134 claims 'verified offline 2026-05-23' but there is no live assert against an Alpha158 ticker-order manifest. If the Alpha158 build sorted by something else (e.g. by listing order, by market cap snapshot, by file row order), each Alpha158 column would be paired with the wrong stock's hc features, silently scrambling every prediction without any test triggering. This is a single-line fix (load Alpha158 ticker list from metadata, assert == valid_tickers) and the cost of catching it at runtime vs. discovering it in paper review is asymmetric."
    suggested_fix: "Extend `ALPHA158_META` (sp500_5y_alpha158_features_meta.json) to include the ticker order it was built with, then assert at load time: `assert alpha_meta['ticker_order'] == valid_tickers`. If the meta file doesn't expose ticker order, add it in the build script and rerun. Until then, downgrade only with a documented hash comparison."
    status: FIXED
    resolution_notes: "Two-layer fix at run_plan_aaa_168_ranking.py:148-201. Layer A: replicate Alpha158 build script's intersection logic (`sorted(set(prices.cols) ∩ events ∩ sectors)`) and assert equality with part_a's valid_tickers — guarantees logical consistency across scripts. Layer B: numeric KMID time-series Pearson correlation spot-check on 3 tickers (idx 1, 100, 300 = AAPL, CNC, MDLZ at 501 valid tickers), require ρ > 0.9 (point-comparison alone is fragile to ex-dividend OHLC adjustments — yfinance adj_open vs EODHD close drift ~0.1% normal, ~1% on ex-div dates). Smoke v4: 3/3 tickers ρ = 1.0000. An initial point-based variant (smoke v3) failed on MDLZ 2024-04-05 due to dividend adjustment; the time-series correlation form is robust to single-date corporate actions while still catching scrambled column orderings (those would collapse ρ → ~0)."
  - id: FINGNN-CODE-A-03
    severity: MAJOR
    category: statistics
    claim: "NW-HAC inline implementation at lines 291-312 is correct for the long-run-variance formula but uses a hard floor `long_run_var = max(long_run_var, 1e-12)` that can mask negative finite-sample HAC estimates without warning. In low-T weak-signal contexts (this project — daily ΔIC per group with ~70 dates per fold), Bartlett autocovariance sums can legitimately go very small or even negative for finite samples; silently clamping to 1e-12 will produce ultra-large t-statistics (mean/sqrt(1e-12 / n)) and synthetic 'significance'."
    evidence: "run_plan_aaa_168_ranking.py:308 `long_run_var = max(long_run_var, 1e-12)`. In the aggregate path 526-560, T is the number of dates with finite daily-mean ΔIC per group, pooled across folds (≈ 350 days per Plan v1 §3.4), with `lag = nw_auto_lag(T) ≈ floor(4*(350/100)^(2/9)) ≈ 4`. Sample size is OK, but the clamp has no print/warn and no `nan` fallback. Compare to `analyze_tier1_phase_a.py:69-91` — same code, same clamp — so this is inherited risk, but worth flagging because Plan AAA fans this across K≈61 groups and one clamped p-value can poison the BH-FDR adjustment."
    suggested_fix: "Replace `max(long_run_var, 1e-12)` with `if long_run_var <= 0: return mean, np.nan, np.nan, np.nan` so a degenerate HAC estimate surfaces as a missing p-value rather than a false-positive t. Then BH-FDR (line 604) already maps missing p to 1.0. A single explicit `n_hac_degenerate` counter dumped to the audit JSON would also let the analysis layer surface how many groups hit this branch."
    status: FIXED
    resolution_notes: "Adopted reviewer's exact suggestion at run_plan_aaa_168_ranking.py:309-315 — non-positive long-run-variance now returns (mean, nan, nan, nan) instead of clamping. BH-FDR at line 621 already maps NaN p → 1.0 so degenerate groups do not poison the FDR step. Added `n_hac_degenerate` counter to `summary` dict in `aggregate_delta_ic` so the analysis layer can surface this branch's hit count in ranking.json."
  - id: FINGNN-CODE-A-04
    severity: MAJOR
    category: correctness
    claim: "`compute_groups` performs winsorization (p1/p99) globally over the entire calibration slice (across all 252 days × all stocks in one stack), but feeds `spearmanr(X, axis=0)` over those rows. Two separable issues: (a) cross-day pooling of cross-sectional rows means correlation reflects a mixture of cross-sectional and time-series covariance — defensible only if features have stationary means, which is unlikely for 158-dim Alpha158 over a year; (b) the singleton-Bonus path 'if isinstance(rho, float)' (line 234) only triggers when `X` has exactly 2 columns, which won't happen here, so the line is dead but harmless."
    evidence: "run_plan_aaa_168_ranking.py:219-249. `rows = [features_np[d][label_valid_np[d]] for d in day_indices ...]` → `X = np.vstack(rows)` produces ≈ 252 × N_valid rows of 168 columns. `spearmanr(X, axis=0)` then computes a single 168x168 rank-correlation over this pooled panel. Plan v1 §3.1 is silent on cross-day vs. day-by-day Spearman, but Plan Z++ Part A (referenced parent) used the same pooled approach — so this is inherited methodology. The concern is whether the resulting cluster structure faithfully captures cross-sectional comovement at any single date, which is what the permutation step disturbs."
    suggested_fix: "Two options. (a) Lower-cost: document this is a pooled-panel Spearman in the cluster artifact's JSON header so reviewers know what it estimates; ARI vs. fold-0 (already produced at 906-916) provides empirical robustness evidence. (b) Higher-cost: average per-day Spearman matrices across calibration days (median Spearman per pair), which more directly estimates the cross-sectional comovement at typical dates. If ARI(calib, fold-0) < 0.85 (the existing concern flag at 912-913), revisit. Defer the substantive fix unless ARI fires."
    status: FIXED
    resolution_notes: "Adopted option (a). Added `'pooled_panel': True` and `'pooled_panel_note'` field to `compute_groups` output dict (run_plan_aaa_168_ranking.py:255-260) — the note explicitly describes the pooled (n_days × n_valid_stocks) × n_features Spearman semantic, references Plan Z++ Part A inheritance, and points to ARI vs. fold-0 as the empirical robustness gate. Option (b) (per-day Spearman median) deferred behind the existing ARI < 0.85 trigger at line 906-918."
  - id: FINGNN-CODE-A-05
    severity: CONCERN
    category: correctness
    claim: "`_groups_to_labels` (lines 280-286) assumes both groupings cover the SAME 168 features (same `num_features`). This holds in Plan AAA (cluster_calib and cluster_fold0 both run on the 168-dim panel) but the function does not assert this, and if the fold-0 cluster encountered an additional zero-variance column it would still pass."
    evidence: "run_plan_aaa_168_ranking.py:280-286 + caller at 906-909 passes `len(feature_names)` for both. Fine as written but fragile — if a future change ever feeds groupings with different `num_features`, the ARI would be silently computed on differently-aligned label arrays."
    suggested_fix: "Optional belt-and-suspenders assert in `_groups_to_labels`: `assert max(g['members_idx'] for g in groups_obj['groups'] for i in g['members_idx']) < num_features`. Defer if H博士 thinks the surface is small."
    status: OPEN
  - id: FINGNN-CODE-A-06
    severity: CONCERN
    category: reproducibility
    claim: "Smoke mode noise control (4b) trains on a 173-dim tensor (168 + 5 noise) but reuses the production `train_one_with_telemetry` with the same HPARAMS. Convergence check has no chance to be meaningful at 5 epochs and the soft threshold |Δ| < 0.01 may be hit by random noise rather than by genuine 'no-effect' behaviour. This is fine as a smoke check, but the noise-control PASS reported in the smoke report shouldn't be cited in the paper as evidence of methodological correctness."
    evidence: "run_plan_aaa_168_ranking.py:716-748. At 5 epochs the model is essentially the initialization; the soft threshold is documented, but the smoke_report JSON still records `pass_noise_soft: true/false` which a downstream analysis or paper claim might over-interpret."
    suggested_fix: "Add a comment in smoke_report.json (or rename the field `pass_noise_soft_smoke_only`) clarifying that this passes/fails a smoke-grade soft threshold, not a publication-grade negative control. Plan AAA does not need this elevated — full-mode results stand on the ranking and the BH-FDR pass."
    status: OPEN
  - id: FINGNN-CODE-A-07
    severity: CONCERN
    category: reproducibility
    claim: "`set_seed` is called inside `train_one_with_telemetry` (368), but `train_days[np.random.permutation(...)]` at line 404 uses NumPy's global RNG. The global seed was set by `pa.set_seed(seed)` and is thus deterministic given the same seed value — but two cells with the same `seed_idx` (different fold or arch) will see the same train-day shuffle pattern. This is by design in part_a and inherited, but worth flagging because it means train-time randomness is NOT independent across the 30 cells; only across the 3 seeds within a (fold, arch)."
    evidence: "run_plan_aaa_168_ranking.py:368 + 404. `pa.set_seed(seed)` sets `np.random.seed(s)`. Subsequent `np.random.permutation` at 404 uses that global state. The shuffle sequence is therefore a deterministic function of seed only, not of (fold, arch). For the 30-cell run this gives identical train-day orderings within each fold across SAGE-Mean and MLP at the same seed_idx. Not a leakage; not a bug; but reduces effective seed variance and is worth declaring in the convergence audit."
    suggested_fix: "Document in convergence.json that train-day shuffle within a fold is a function of seed only. No action required. If H博士 wants true independence across (arch, fold, seed_idx), thread a fresh `np.random.default_rng(cell_id)` through `train_one_with_telemetry` and use `rng.permutation(...)` instead of the global state."
    status: OPEN
  - id: FINGNN-CODE-A-08
    severity: CONCERN
    category: correctness
    claim: "`paired.merge(..., validate='many_to_one')` at line 571 assumes baseline_df has exactly one row per (cell_id, arch, fold_idx, seed_idx, cell_seed_value, day_idx). If a baseline cell ran but produced NaN IC (mask < 30 stocks on a date), `daily_ic` still emits a NaN row per the part_a convention, so the merge key is intact and `many_to_one` holds. Verified by reading `pa.daily_ic`. Confirming no action needed; flagging only because this guarantee depends on `pa.daily_ic` emitting one row per test day regardless of validity."
    evidence: "run_plan_aaa_168_ranking.py:568-572 + run_step3_plan_z_part_a.py:446-459. `daily_ic` allocates `np.full(len(days), np.nan)` upfront and writes only when mask passes, so output length is always `len(days)`. Plan AAA writes one baseline row per (cell, day) at 985-991. Merge invariant holds."
    suggested_fix: "No change; this is a positive verification."
    status: REJECTED
  - id: FINGNN-CODE-A-09
    severity: CONCERN
    category: correctness
    claim: "The runtime non-group assert at lines 540-544 runs per (group, day) inside `torch.no_grad()`. On a 168-dim panel with K≈61 groups × ~70 test days = ~4,300 asserts per cell × 30 cells = ~130k asserts. Each is a tensor equality over a (N_stocks, 168-|group|) slice. Overhead is real (likely several seconds total) but bounded and the safety it buys for a one-time paper run is worth it. Flagging only because if a future maintainer time-profiles and removes this 'for performance', the safety guarantee disappears."
    evidence: "run_plan_aaa_168_ranking.py:540-544. Not a bug; flagging the maintenance hazard."
    suggested_fix: "If the assert is ever removed, replace with a once-per-(arch,fold,seed) smoke assertion at the first test day to retain coverage at lower cost."
    status: OPEN
  - id: FINGNN-CODE-A-10
    severity: CONCERN
    category: statistics
    claim: "`block_bootstrap_mean_ci` (lines 320-337) uses fixed-length blocks of 21 days, not stationary bootstrap with geometrically-distributed block lengths as the term 'stationary-style' in the docstring suggests. This is a fixed-block bootstrap (Künsch-style), not Politis-Romano stationary."
    evidence: "run_plan_aaa_168_ranking.py:323 docstring 'Stationary-style block bootstrap' + 332-334 implementation: `starts = rng.integers(...)` + concatenate fixed `s[s_:s_+block_len]`. Politis-Romano stationary bootstrap samples block length L_b ~ Geometric(1/block_len) per block. The Künsch fixed-block variant is also valid and is what `analyze_tier1_phase_a.py:112-136` uses, so this is inherited. Just don't call it 'stationary'."
    suggested_fix: "Rename docstring to 'Fixed-length block (Künsch 1989) bootstrap CI' or implement geometric block lengths. The CI estimate quality is similar for our setting; the naming is the issue."
    status: FIXED
    resolution_notes: "Renamed docstring at run_plan_aaa_168_ranking.py:344-350 to 'Fixed-length block (Künsch 1989) bootstrap CI' with explicit note that Plan v1 §3.4 used the term 'stationary' but the implementation matches analyze_tier1_phase_a.py's Künsch fixed-block variant. Plan v1 wording in §3.4 also needs an update — flagged for the analysis-writeup phase."
summary:
  critical: 0
  major: 4
  concern: 6
  fixed_after_smoke_v4: 5  # A-01, A-02, A-03, A-04, A-10
  rejected: 1              # A-08 self-rejected by reviewer as positive verification
  remaining_open_concern: 4  # A-05, A-06, A-07, A-09 deferred to analysis writeup
overall_verdict: PROCEED-WITH-FIXES
post_review_status: "All 4 MAJOR + A-10 CONCERN FIXED in script (verified by smoke v4 2026-05-25). 4 remaining CONCERNs (A-05/06/07/09) are documentation/robustness items deferred to analysis writeup, not execution-blocking. Plan AAA full mode is now unblocked from Touchpoint 2 perspective."
---

# Review body

I read `run_plan_aaa_168_ranking.py` (1106 lines) in full plus the relevant slices of `run_step3_plan_z_part_a.py`, `analyze_tier1_phase_a.py`, and `docs/plan_aaa_v1_2026-05-23.md` §2.4, §3.1-§3.4. Smoke test results, the cell_id schema, the SeedSequence direct-pass, the shared-row permutation semantic, the audit uniqueness assertion, the convergence telemetry, and the BH-FDR + NW-HAC aggregation order all match Plan v1's specifications and the post-stop-hook refinements. The code is much tighter than the v0 baseline I would have reviewed; most of what's left is real but not blocking.

**No CRITICAL findings.** I went looking for cell_id collision, audit triple non-uniqueness, SeedSequence truncation, non-group corruption during permutation, fold-leakage in scaler fitting, T-0 leakage in hc features, and graph-snapshot lookahead. None of these are present. The cell_id arithmetic (arch_idx*15 + fold_idx*3 + seed_idx, range 0..29) is mechanically verified — at lines 676 and 952 the formula is identical, arch_idx multiplier (15) = N_FOLDS*N_SEEDS, fold_idx multiplier (3) = N_SEEDS, all of these match the constants at lines 62-64 (`SEEDS` of length 3, `N_FOLDS=5`, `ARCHS` of length 2). The non-group assert at lines 540-544 plus the audit-triple uniqueness assert at line 1028 make this much more defensible than the v0 design.

**The MAJOR cluster — items 01-04** — is where a NeurIPS/ICAIF reviewer would land. (01) The winsor language in Plan v1 §2.4 is not honored in code. This is the most likely paper-defense issue: a reviewer will read "per-fold train-only winsor at p1/p99 + per-fold scaler" in the methodology section and then check that the implementation does it. Decide whether to implement or to delete the language from the plan; the discrepancy itself is the risk. (02) Ticker ordering between the Alpha158 panel and the hc panel is asserted only by cardinality. A "verified offline" comment is not a runtime check. Plumb a ticker-order list through the metadata and assert at load — a one-day fix that closes a silent-corruption class. (03) The NW-HAC clamp to 1e-12 will fabricate t-statistics for groups whose finite-sample long-run-variance is non-positive, and these false positives then propagate through BH-FDR. This is inherited from `analyze_tier1_phase_a.py:87` but Plan AAA fans it across K≈61 groups, increasing exposure. The fix (return NaN instead of clamping) is single-line and protects the headline ranking. (04) Pooled-panel Spearman for clustering is a methodological choice with inherited precedent (part_a), and the ARI fold-0 check at 906-918 acts as an empirical sanity gate — but the JSON output should make the pooling explicit so a reviewer doesn't misread the artifact.

**Items 05-10 (CONCERN)** are documentation / robustness / wording — none of them block execution. Item 06 (smoke 4b noise control) and item 10 (bootstrap naming) are presentation issues; item 07 (train-day shuffle determinism within seed_idx) is a known design choice in part_a that's worth declaring; items 05, 08, 09 are forward-looking maintenance hazards.

What I did NOT find but checked for explicitly:
- T-0 leakage in hc features: confirmed via `prices.pct_change()` + `rolling().shift(1)` pattern at part_a:101-110. No same-day return enters the panel.
- Graph train-only invariant: enforced by `pa.assert_graph_train_only` at line 382, with the same fold-bounded `frozen_si = snaps[int(train_days.max())]` semantic that Plan Z++ Part A already passed.
- Per-fold scaler scope: `pa.fit_feature_scaler` at 944 uses only `train_days` (plus `label_valid_np[d]`), no val/test rows enter. Cleanly train-only.
- SeedSequence uint32 truncation: gone. Line 530 passes `SeedSequence([int(cell_id), gid, d_int])` straight to `default_rng`. Full 128-bit PCG64 state via SHA256 mixing. The audit schema correctly omits `rng_seed` (per stop-hook 5th pass).
- Audit uniqueness: line 1028-1032 asserts `len({(cell_id, group_id, date)}) == len(rows)` at write time. Smoke produced 63 rows / 63 unique triples → schema correct.
- BH-FDR direction: lines 340-353 sort ascending p, multiply by n/i, then cumulative-min from right — standard step-up. Compared against `analyze_tier1_phase_a.py:139-153`, line-for-line identical. Correct.

**Bottom line.** PROCEED-WITH-FIXES. Fix (01) by deciding plan-vs-code authority on the winsor language; fix (02) by adding the ticker-order assertion to the Alpha158 metadata; fix (03) by returning NaN instead of clamping in NW-HAC. Item (04) can be deferred behind the existing ARI gate at line 912-918 — only revisit if `concern_triggered` fires. The remaining CONCERNs can be addressed inline with the analysis writeup. I would not block the Colab launch on the CONCERN items; I would block it on items 01, 02, 03.
