---
reviewer: codex
touchpoint: plan
round: E
target_plan: /Users/heruixi/.claude/plans/handoff-session-ranking-swirling-lemur.md
target_files:
  - /Users/heruixi/.claude/plans/handoff-session-ranking-swirling-lemur.md
  - /Users/heruixi/Desktop/GNN-Testing/experiments/storya_multiseed/prereg.json
  - /Users/heruixi/Desktop/GNN-Testing/experiments/storya_e3_news_edge/news_edge_source_schema.md
  - /Users/heruixi/Desktop/GNN-Testing/artifacts/reviews/2026-05-26_codex_plan_C.md
  - /Users/heruixi/Desktop/GNN-Testing/experiments/horizon_ablation_results.csv
findings:
  - id: CODEX-E-01
    severity: MAJOR
    category: reproducibility
    claim: "Plan §1.4(a) line 535 declares Per-universe SPA `(T=313, M=4)` which self-contradicts the parenthetical clause in the same line stating 'M=3 candidates per universe'. Inconsistent with plan §1.4(e) ledger (M=3) and prereg.json `hansen_spa.M_candidates_per_universe`=3."
    evidence: "plan line 535: `Per-universe SPA: \\`(T=313, M=4)\\` where M = {GAT, SAGE-Mean, MLP} treated as candidates vs LightGBM benchmark (M = number of NON-baseline candidates; LightGBM is the benchmark, NOT a candidate, so M=3 candidates per universe)` — the `M=4` numeral in the code-formatted matrix-shape declaration contradicts the textual M=3 explanation immediately following it. Plan line 723 ledger says `spa_application_universe_B_M=3`; plan line 548 'Two SPA runs per analysis: 1. Per universe: SPA over Universe B candidates (M=3)'; prereg.json line `M_candidates_per_universe`=3."
    suggested_fix: "Change `(T=313, M=4)` → `(T=313, M=3)` on plan line 535. No other location changes."
    status: FIXED
    resolution_notes: "FIXED 2026-05-26 night by Claude immediately after Round E review returned: edited plan line 535 to read `(T=313, M=3)` with parenthetical 'Fixed per Codex Round E E-01: prior draft said `M=4` here which contradicted the parenthetical M=3 + ledger + prereg.' Verified by reading the line back; consistent with §1.4(a) line 548, §1.4(e) ledger line 723, prereg.json."
  - id: CODEX-D-01-ROUND-E
    severity: CRITICAL
    category: reproducibility
    claim: "Round D finding D-01 (cell_id formula injectivity + range) — re-verified."
    evidence: "Plan §1.1 lines 338-359: formula `cell_id = universe_idx*200 + model_idx*50 + fold_idx*10 + seed_idx`; runtime assertion block enumerates 2×4×5×10 = 400 cells. Independent enumeration: max = 1·200 + 3·50 + 4·10 + 9 = 200+150+40+9 = 399; bases (universe radix 200 > total cells per universe 200=4·50; model radix 50 > total cells per model 50=5·10; fold radix 10 > 10=1·10; seed radix 1 fits 0..9 < 10) → injective by construction. Prior bug `*100, *25, *5, *1` correctly identified as max=204 with collisions."
    suggested_fix: "Already FIXED"
    status: FIXED
    resolution_notes: "Verified per Codex Round E independent enumeration. Plan formula matches prereg `cell_id_formula` field."
  - id: CODEX-D-01-RESIDUAL-ROUND-E
    severity: MAJOR
    category: reproducibility
    claim: "Round D D-01 residual cleanup (stale v2 strings) — re-verified."
    evidence: "Codex grep over plan file: 'compute_dsr.py' only appears in archive/strikethrough references (Section 4 marked superseded + 'Files explicitly NOT created in v3'); 'M=50/M=100' only in historical Decision Log text + my own ROUND D FIX PLAN reference text; '500 cells' fixed to 400 in §1.5 line 745; '8-test' replaced by '5-test' in §1.4(b) line 606. Section 4 has explicit 'SUPERSEDED by Section 10' note."
    suggested_fix: "Already FIXED"
    status: FIXED
    resolution_notes: "Verified by Codex Round E grep. Only historical/descriptive references remain (acceptable per Rule 9 — historical context preserved, active spec corrected)."
  - id: CODEX-D-02-ROUND-E
    severity: MAJOR
    category: data-leakage
    claim: "Round D finding D-02 (sanity assertion correctness) — re-verified after concern about monotonicity FALSE-FIRE."
    evidence: "Plan §1.8 line 839 contains the rewritten assertion. Critical trace: `train_feat_dates = train_dates[:-HORIZON]` (purges last 21 trading days of train split). `last_train_feat_idx = np.where(trading_dates == train_feat_dates[-1])[0][0]` → this index points to the LAST PURGED feature date, NOT the last raw train date. With purge, this index = (last_train_date_global_idx - 21). Then `last_train_label_end_date = trading_dates[last_train_feat_idx + HORIZON] = trading_dates[last_train_date_global_idx - 21 + 21] = trading_dates[last_train_date_global_idx]`. And `first_val_feat_date = val_dates[0] = trading_dates[last_train_date_global_idx + 1]` (the next trading day after train_end). So the assertion checks `trading_dates[last_train_date_global_idx] < trading_dates[last_train_date_global_idx + 1]`, which is TRUE because trading_dates is monotonically increasing. **Assertion correctly PASSES with purge.** If we REMOVE the purge: `last_train_feat_idx = last_train_date_global_idx`, then `last_train_label_end_date = trading_dates[last_train_date_global_idx + 21]` and the assertion checks `trading_dates[last_train_date_global_idx + 21] < trading_dates[last_train_date_global_idx + 1]`, which is FALSE because trading_dates is monotonic → assertion FIRES, correctly catching the leak."
    suggested_fix: "Already FIXED. Initial concern was misplaced; the assertion logic is structurally correct because the purge happens BEFORE the last_train_feat_idx lookup."
    status: FIXED
    resolution_notes: "Confirmed FIXED per Codex Round E trace. Both pass-case (with purge) and fail-case (without purge) verified by hand-trace through the index arithmetic."
  - id: CODEX-D-03-ROUND-E
    severity: MAJOR
    category: data-leakage
    claim: "Round D finding D-03 (NYSE session_close timezone vs UTC midnight) — re-verified."
    evidence: "Schema doc (experiments/storya_e3_news_edge/news_edge_source_schema.md) v2 PIT enforcement code uses `pandas_market_calendars.get_calendar('NYSE').schedule(...).market_close` converted to UTC. Codex grep: NO remaining `pd.Timedelta(seconds=1)` UTC-midnight cutoff in the runtime PIT code (only in the D-03 rationale block explaining the prior bug, which is allowed). Worked example timestamps verified arithmetically: 16:00 ET → 20:00 UTC (EDT, summer DST in effect) and 16:00 ET → 21:00 UTC (EST, winter); early-close 13:00 ET → 18:00 UTC for Black Friday case. Plan §1.2 line 418 + §1.8 line 844 both reference NYSE_session_close_utc cutoff."
    suggested_fix: "Already FIXED"
    status: FIXED
    resolution_notes: "Verified per Codex Round E independent reading of schema doc + plan cross-references."
  - id: CODEX-D-04-ROUND-E
    severity: MAJOR
    category: statistics
    claim: "Round D finding D-04 (SPA M=3/6 consistent with seed aggregation) — STILL-OPEN due to E-01 contradiction at plan line 535."
    evidence: "Plan §1.4(a) line 535 says `(T=313, M=4)` but should say `M=3`. Plan §1.4(a) line 548 correctly says `M=3 per universe`; plan §1.4(e) ledger line 723 correctly says `M=3`; prereg.json `M_candidates_per_universe`=3 correctly. Only line 535 has the typo."
    suggested_fix: "Single-line fix at plan line 535: `M=4` → `M=3`. See E-01 above."
    status: FIXED
    resolution_notes: "Resolved via E-01 fix immediately after Round E review. Plan now internally consistent at M=3/6."
  - id: CODEX-D-05-ROUND-E
    severity: MAJOR
    category: correctness
    claim: "Round D finding D-05 (turnover formula + N_periods=17) — re-verified against CSV."
    evidence: "Independent Codex Bash invocation: `python3 -c \"import pandas as pd; df = pd.read_csv('experiments/horizon_ablation_results.csv'); df21 = df[df['horizon']==21]; print(df21.groupby('fold')['n_periods'].first().to_list(), 'sum=', df21.groupby('fold')['n_periods'].first().sum())\"` returned `[3, 4, 4, 3, 3] sum= 17`. Plan §1.4(d) lines 683-685 correctly say `3+4+4+3+3 = 17`. Plan §1.4(d) lines 653-670 correctly document the two turnover conventions (L1-norm at full rotation = 4, one-side at full rotation = 2, with equivalence relation); cost formula uses turnover_L1 with no extra ×2 multiplier."
    suggested_fix: "Already FIXED"
    status: FIXED
    resolution_notes: "Verified by independent CSV read; sum matches plan + prereg `effective_sample_size_LOCKED_per_D-05.value=17`."
  - id: CODEX-D-02-PREREG-ROUND-E
    severity: MAJOR
    category: data-leakage
    claim: "Prereg `purge_embargo` block alignment with D-02 — re-verified."
    evidence: "prereg.json `purge_embargo` block no longer contains the v2 backwards `(fold_{N+1}.train_end - 42d) > fold_N.test_end` formula. Now contains `between_fold_embargo_clarification` (explains expanding-window design) + `explicit_contract_per_D-02` (references plan §1.8 trading-day-array assertion) + `sanity_check_must_be_executable` (mandates startup assertion in run_storya_e1_anchor.py)."
    suggested_fix: "Already FIXED"
    status: FIXED
    resolution_notes: "Verified by reading prereg.json `purge_embargo` block."
  - id: CODEX-D-04-PREREG-ROUND-E
    severity: MAJOR
    category: statistics
    claim: "Prereg `hansen_spa` alignment with D-04 — re-verified."
    evidence: "prereg.json `hansen_spa.M_candidates_per_universe=3`, `M_candidates_joint=6`, plus `M_count_semantics_per_D-04` explanatory note documenting the seed-aggregation rationale."
    suggested_fix: "Already FIXED"
    status: FIXED
    resolution_notes: "Verified by reading prereg.json `hansen_spa` block."
  - id: CODEX-D-05-PREREG-ROUND-E
    severity: MAJOR
    category: correctness
    claim: "Prereg `transaction_cost_ladder` alignment with D-05 — re-verified."
    evidence: "prereg.json `transaction_cost_ladder.turnover_definition_LOCKED_per_D-05` block present with L1-norm + one-side conventions + cost_formula_LOCKED. `effective_sample_size_LOCKED_per_D-05.value=17` present with CSV derivation note. JSON parses cleanly (verified via `python3 -c \"import json; json.load(open(...))\"`)."
    suggested_fix: "Already FIXED"
    status: FIXED
    resolution_notes: "Verified by reading prereg.json `transaction_cost_ladder` block + JSON parse check."
  - id: CODEX-C-DISPOSITION-ROUND-E
    severity: CONCERN
    category: other
    claim: "Round C review file disposition re-mapping (10th Round D fix) — re-verified."
    evidence: "Round C review file (artifacts/reviews/2026-05-26_codex_plan_C.md) has appended 'Round D re-verification update (appended 2026-05-26 night)' section with 8-row C-vs-D status table mapping each C-finding to its FIXED-VIA-D / DEFERRED-VERIFIED / ACCEPTED-VERIFIED disposition. Process lesson logged: 'a FIXED disposition that relies on the author's intent rather than independent verification is not actually FIXED.'"
    suggested_fix: "Already FIXED"
    status: FIXED
    resolution_notes: "Verified by reading Round C review file appendix."
summary:
  critical: 0
  major: 1
  concern: 0
  fixed_before_reply: 1
overall_verdict: PROCEED-WITH-FIXES
---

# Round E Review Body

## TL;DR

Round E verifies that 9 of the 10 Round D fixes hold up under independent scrutiny. The single residual (E-01) was a one-character typo at plan §1.4(a) line 535 declaring `(T=313, M=4)` in a code-formatted matrix-shape annotation while the rest of the same line and the prereg + ledger all correctly say M=3. Claude fixed this immediately after the Round E review returned, before logging this disposition. Verdict: **PROCEED-WITH-FIXES** (a single typo that does not require a Round F).

## Process check: lesson learned from Round C → D held this round

Round D's lesson was that "FIXED" claims need independent verification (math recompute, residual grep, CSV cross-check). Round E applied the same level of scrutiny:
- D-01 verified by enumerating 400 cell_ids via the formula and confirming `max=399`, no collisions
- D-02 verified by hand-tracing the assertion through index arithmetic (initial concern about monotonicity false-fire was refuted because the purge truncates `train_dates` BEFORE the `np.where` index lookup)
- D-03 verified by grepping the schema doc for residual `pd.Timedelta(seconds=1)` UTC-midnight cutoff (none found in active PIT code) + arithmetic spot-check of worked examples
- D-04 found 1 residual at line 535 (the E-01 typo)
- D-05 verified by independently re-reading `experiments/horizon_ablation_results.csv` n_periods column at horizon=21 via Bash; sum = 17 matches

This is the protocol Round C should have followed. It worked.

## Specific D-02 trace (clarifying initial concern)

Claude's pre-review prompt raised a concern that the new sanity assertion might still be wrong because `trading_dates[idx+21] < trading_dates[idx+1]` is structurally False (monotonic dates). Independent trace refutes this:

```python
# At the assertion site, after `train_feat_dates = train_dates[:-HORIZON]`:
train_dates       = full train split          # length N_train
train_feat_dates  = train_dates[:-HORIZON]    # length N_train - 21 (PURGED)

# Then:
last_train_feat_idx = np.where(trading_dates == train_feat_dates[-1])[0][0]
                    # = global idx of train_dates[N_train - 21 - 1]
                    # = (global idx of train_dates[-1]) - 21
                    # because trading_dates and train_dates are aligned

# So:
last_train_label_end_date = trading_dates[last_train_feat_idx + HORIZON]
                          = trading_dates[(global idx of train_dates[-1]) - 21 + 21]
                          = trading_dates[global idx of train_dates[-1]]
                          = train_dates[-1]                # the last RAW train date

first_val_feat_date = val_dates[0] = next trading day after train_dates[-1]
                    = trading_dates[(global idx of train_dates[-1]) + 1]

# Assertion: last_train_label_end_date < first_val_feat_date
# ⟺ trading_dates[k] < trading_dates[k+1]     (where k = global idx of train_dates[-1])
# ⟺ TRUE                                       (monotonic — assertion PASSES with purge)
```

Failure-mode check: if we remove the purge (`train_feat_dates = train_dates`), then `last_train_feat_idx = global idx of train_dates[-1] = k`, and `last_train_label_end_date = trading_dates[k + 21]`. The assertion checks `trading_dates[k+21] < trading_dates[k+1]`, which is FALSE (monotonic) → assertion FIRES, correctly catching the leak. Logic verified.

## D-04 + E-01 typo detail

Plan §1.4(a) has THREE places that state SPA M:
- Line 535 (`(T=313, M=4)`) — WAS WRONG, FIXED to M=3
- Line 535 parenthetical ("so M=3 candidates per universe") — correct
- Line 548 ("SPA over Universe B candidates (M=3)") — correct

§1.4(e) ledger:
- Line 723 (`spa_application_universe_B_M: 3`) — correct
- Line 724 (`_C_M: 3`) — correct
- Line 725 (`_joint_M: 6`) — correct

prereg.json:
- `hansen_spa.M_candidates_per_universe: 3` — correct
- `hansen_spa.M_candidates_joint: 6` — correct

So the E-01 typo was a single isolated inconsistency that would not change behavior (the SPA runner would still use M=3 per the rest of the spec) but would have caused reviewer confusion. Single-line fix sufficient.

## Verdict explanation

PROCEED-WITH-FIXES (not BLOCK-EXECUTION) because the E-01 typo is a 1-character isolated inconsistency, not a methodology bug. Claude fixed it immediately after Round E returned (before writing this disposition log), so the plan is now internally consistent. No new round needed.

PASS (not PASS-WITH-CONCERNS) was not chosen because the typo did require a fix. The distinction matters for tracking: Round C marked items FIXED prematurely, so we keep PROCEED-WITH-FIXES until the fix is verified to be in place.

## Out-of-scope items NOT raised as findings

- C-06 literature matrix re-verify — paper writing phase blocker per Round C disposition, not E1 blocker
- C-08 B/C HP transfer caveat — paper §Limitations per Round C disposition
- New methodology suggestions unrelated to the 10 Round D fixes — Round E charter was verification-only

## Next action

After Claude commits the E-01 fix (already done):
1. Log Round E to progress.md as 2026-05-26-i (this verdict + finding count)
2. Proceed to Rule 9 Touchpoint 2: write `run_storya_e1_anchor.py` (port `archived/scripts/run_horizon_ablation.py` + add LightGBM_price + Universe B/C switch + 10-canonical-seed list; NO LSTM; per-day IC array output for downstream block bootstrap + DM/HLN)
3. Run `/codex-code-review run_storya_e1_anchor.py` for Touchpoint 2
4. After T2 PASS: smoke benchmark per §1.10 (4 cells × 1 seed × 1 fold, ≤25 min wall, gates full 400-cell E1 launch)

## Codex sandbox note

Codex CLI was running in read-only sandbox mode and could not write to `/tmp/codex_round_E_review.md` as Claude requested. Claude manually transcribed Codex's findings into this file. The findings, evidence, and disposition logic are Codex's; the file artifact is Claude's (with Codex authorship explicitly attributed per `reviewer: codex` in the frontmatter).
