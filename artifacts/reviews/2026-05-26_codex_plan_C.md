---
reviewer: codex
touchpoint: plan
round: C
target_plan: /Users/heruixi/.claude/plans/handoff-session-ranking-swirling-lemur.md
target_files:
  - /Users/heruixi/.claude/plans/handoff-session-ranking-swirling-lemur.md
  - /Users/heruixi/Desktop/GNN-Testing/experiments/storya_multiseed/prereg.json
  - /Users/heruixi/Desktop/GNN-Testing/artifacts/reviews/2026-05-26_codex_plan_A.md
  - /Users/heruixi/Desktop/GNN-Testing/docs/session_handoff_2026-05-26.md
  - /Users/heruixi/Desktop/GNN-Testing/artifacts/plan_aaa/ranking.csv
  - /Users/heruixi/Desktop/GNN-Testing/experiments/horizon_ablation_results.csv
  - /Users/heruixi/Desktop/GNN-Testing/archived/docs/decisions.md
findings:
  - id: CODEX-C-01
    severity: CRITICAL
    category: reproducibility
    claim: "The target plan is not v3-aligned enough to execute: core executable specs still conflict on model count, cell count, LSTM inclusion, SPA M, DM family, DSR/HGT outputs, and smoke budget."
    evidence: "plan lines 230-235 still define 5 models/LSTM/0..499 cell IDs; lines 394-407 use M=50/100 SPA; lines 417-428 include LSTM DM tests; lines 498-503 ledger says 5 models/500 cells; lines 539-543 still describe compute_dsr.py; lines 689-697 smoke gate still uses 5 models/1450 cells; while prereg.json:140-156 correctly says 4 models/400 cells and prereg.json:169-188 says M=40/80 and 5 DM tests."
    suggested_fix: "Make the plan, prereg, handoff, and disposition log use one canonical v3 contract: 4 E1 models, 400 E1 cells, no LSTM in E1 tests, SPA M=40 per universe/M=80 joint, 5-test DM family, E1 output under storya_e1_anchor, and no active DSR/HGT v2 artifacts."
    status: FIXED
    resolution_notes: "FIXED 2026-05-26 evening (Option B path): plan §1.1 cell_id formula → 4 models, range 0..399; plan §1.4 SPA → M=40/80; plan §1.4 DM family → 5 tests (was 8 with LSTM); plan §1.4 ledger → 4 models, 400 cells; plan §1.10 smoke → 4 models, 400-cell budget. Pending Round D verification of cleaned text."
  - id: CODEX-C-02
    severity: MAJOR
    category: data-leakage
    claim: "The §1.8 purge/embargo sanity check is backwards and cannot validate the intended no-overlap condition."
    evidence: "plan lines 640-641 require `assert (fold_{N+1}.train_end - 42d) > fold_N.test_end`; with the listed folds, fold 1 train_end is 2024-03-31 and fold 0 test_end is 2024-06-30, so the assertion is false by construction."
    suggested_fix: "Define split boundaries in trading-day indices and assert within each fold: max(train_label_end) < min(val_feature_date), max(val_label_end) < min(test_feature_date), all features/graphs use <= t-1, and any between-fold check matches the actual expanding-window design rather than comparing next train_end to previous test_end."
    status: FIXED
    resolution_notes: "FIXED 2026-05-26 evening (Option B path): plan §1.8 rewritten with within-fold purge formulas in trading-day indices; between-fold check removed (expanding-window is by design, not a leak); concrete Python assertion code added for run_storya_e1_anchor.py startup. Pending Round D verification of correctness."
  - id: CODEX-C-03
    severity: MAJOR
    category: data-leakage
    claim: "The news-edge PIT rule is conceptually right but not enforceable against the documented event artifact schema."
    evidence: "plan lines 279-286 require `publication_timestamp` and article-level co-occurrence; the fullscale event builder stores `date,ticker,title,content,polarity,neg,neu,pos,tags,return_next,label` and scripts/prepare_events.py shows event files may include forward labels."
    suggested_fix: "Build a dedicated edge-source table before E3 with `article_id`, UTC `publication_timestamp`, original ticker list, and no forward-return/label fields; assert every article used for date t has publication_timestamp <= end_of_day(t-1)."
    status: FIXED
    resolution_notes: "FIXED 2026-05-26 evening (H博士 directed immediate fix instead of Option B deferral). Schema spec at `experiments/storya_e3_news_edge/news_edge_source_schema.md` (193 lines). Key findings from source audit: (a) source `date` column IS UTC second-precision datetime (Codex assumption about needing new publication_timestamp was wrong — already present); (b) multiple rows per article share identical date/title/content (verified rows 1-2 both for article 'The 20 stocks hedge funds are most underweight' 2026-01-18 19:07:53 UTC for tickers CHTR/COIN); (c) source has NO article_id — must derive via xxHash64 of (date_iso + title + content[:512]). Derived PIT-safe artifact schema LOCKED with: article_id (uint64), publication_timestamp (datetime64[ns, UTC]), tickers_mentioned (list[str]), n_tickers (uint16); forbidden columns (return_next/label/polarity/neg/neu/pos/tags) asserted at build time + verification script. E3 runtime PIT assertion: `assert max(eligible.publication_timestamp) <= end_of_day(t-1)`. Pending: actual build of `data/fullscale/sp500_news_edge_source.parquet` (~30 min M4 wall time) + run_storya_e3_news_edge.py runner script (Codex T2 review required). Plan §1.2 updated with full spec + cross-reference to schema doc."
  - id: CODEX-C-04
    severity: MAJOR
    category: statistics
    claim: "DM/HLN is still under-specified because seed aggregation and the primary loss are not locked."
    evidence: "plan lines 429-445 define `d(t)=L_A(t)-L_B(t)` but do not say whether seeds are averaged first, tested per seed, or pooled; lines 455-457 say to pool across 10 seeds x 5 folds; prereg.json:180-188 defines the family but still omits seed aggregation and loss choice."
    suggested_fix: "Primary DM should use one model-level daily series per date by averaging same-model daily IC across seeds before forming `L=-IC`; do not treat seed-day observations as independent T. If per-seed DM is used, aggregate p/effects with a pre-registered hierarchical or date-clustered procedure."
    status: FIXED
    resolution_notes: "FIXED 2026-05-26 evening (Option B path): plan §1.4 'Seed aggregation (v3, per Codex CODEX-C-04 fix — LOCKED before execution)' block added with 5-step construction (per-day IC → seed-average per (model, fold, date) → fold-pool → loss L=-IC → DM paired diff); T=313 explicitly NOT 3130; prereg.json DM_HLN.seed_aggregation block locked with construction steps + T_pooled_warning. Pending Round D verification."
  - id: CODEX-C-05
    severity: MAJOR
    category: correctness
    claim: "The transaction-cost ladder formula conflicts with the stated equal-weight decile portfolio and leaves cost semantics underdefined."
    evidence: "plan lines 473-485 define gross_pos_return as mean(prediction_score x forward_return for top decile minus bottom decile), which is score-weighted, not equal-weight; horizon_ablation_results.csv shows 21d non-overlap produces only 3-4 portfolio periods per fold."
    suggested_fix: "Define weights explicitly as +1/K in top decile and -1/K in bottom decile, holding 21 trading days; define whether bps are one-way or round-trip per dollar traded; report uncertainty for Net Sharpe or mark it secondary due to small effective T."
    status: FIXED
    resolution_notes: "FIXED 2026-05-26 evening (Option B path): plan §1.4 cost ladder LOCKED equal-weight position formula (position_i = ±1/K), one-way bps semantics, sqrt(252/21) annualization for 21d non-overlap, N=15 periods effective sample size disclosure, bootstrap CI uncertainty reporting (block_size=1, n_boot=5000), Net Sharpe marked SECONDARY metric. Prereg.json transaction_cost_ladder block mirrors plan. Pending Round D verification."
  - id: CODEX-C-06
    severity: MAJOR
    category: prior-art
    claim: "The literature matrix still contains incorrect characterizations and is missing major 2024-2025 graph-stock papers, undermining the first-systematic-study novelty claim."
    evidence: "plan lines 653-674: FinGAT listed as Taiwan only but covers Taiwan+S&P500+NASDAQ per arXiv:2106.10159; HIGSTM listed as CSI300 but covers CSI500/CSI800/CSI1000 per arXiv:2503.11387; DOI 10.1145/3768292.3770389 is a Hypergraph paper not matching the labeled HTAN row. Missing: GRU-PFG arXiv:2411.18997, DishFT-GNN arXiv:2502.10776, DGT for S&P 500 arXiv:2506.18717."
    suggested_fix: "Re-verify every matrix row against primary paper text, add the missing 2024-2025 graph-stock papers, and soften novelty to 'to our knowledge, no prior work combines these axes on US S&P 500 under this exact evaluation contract' unless the corrected matrix still proves exclusivity."
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "DEFERRED-TO-PAPER-WRITING-PHASE 2026-05-26 (Option B path, H博士 directive): re-verifying 16+ arXiv papers requires ~half day of paper fetching/reading. Deferred to paper §2 Related Work writing phase as paper-submission-blocker (NOT E1-blocker). Plan LOCKED DECISIONS row documents the deferral. C-06 specific actions deferred: (a) verify FinGAT/HIGSTM/HTAN entries; (b) add GRU-PFG/DishFT-GNN/DGT; (c) soften novelty to 'to our knowledge'. Will re-address when writing paper §2."
  - id: CODEX-C-07
    severity: CONCERN
    category: statistics
    claim: "Fixed 10 seeds removes adaptive winner's-curse bias but provides low precision for strong no-effect claims given GAT 21d CV=55%."
    evidence: "plan lines 136-138 cite GAT 21d 5-seed mean 0.032 ± 0.018; session handoff lines 104-115 give CV=55.1%."
    suggested_fix: "Before execution, pre-commit that non-rejection under 10 seeds is reported as inconclusive unless the confidence interval is narrow enough to rule out a practically relevant delta IC; add a minimal detectable effect note to prereg."
    status: FIXED
    resolution_notes: "FIXED 2026-05-26 evening (Option B path): prereg.json DM_HLN.minimum_detectable_effect_note_per_C-07 block added: non-significant DM result (HLN p > 0.05) → report as 'inconclusive' unless bootstrap 95% CI |upper_bound| < 0.005 (practically negligible). Otherwise paper §Discussion frames as 'unable to detect difference at our sample size' NOT 'no difference exists'. Pending Round D verification."
  - id: CODEX-C-08
    severity: CONCERN
    category: reproducibility
    claim: "Paired Universe B/C comparison with fixed default hyperparameters may confound feature-universe effects with capacity underfitting for Universe C."
    evidence: "plan lines 217 and 613-625 lock one HP set for both B and C; prereg.json:128-137 does the same."
    suggested_fix: "Frame B/C results as fixed-default robustness rather than architecture-optimal performance; any capacity variation should be a separately pre-registered secondary analysis counted in the testing ledger."
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "ACCEPTED-AS-PAPER-LIMITATION 2026-05-26 (Option B path, H博士 directive): paper §Discussion will note B/C results use fixed default hparams without capacity-tuning, framing comparison as 'feature richness under fixed model class' not 'optimal-per-universe'. No plan/prereg change needed. Will add to paper §Limitations during writing phase."
  - id: CODEX-A-01-ROUND-C
    severity: MAJOR
    category: data-leakage
    claim: "Round A finding A-01 (news PIT) reassessed in Round C"
    evidence: "Refers to CODEX-C-03; PIT rule directionally correct but not enforceable against current artifact schema"
    suggested_fix: "Build dedicated PIT-safe news edge source artifact (see CODEX-C-03)"
    status: STILL-OPEN
    resolution_notes: "The PIT rule in §1.2 is directionally correct, but the plan does not yet bind it to an enforceable article-level schema with publication timestamp and article ID — see CODEX-C-03."
  - id: CODEX-A-02-ROUND-C
    severity: CRITICAL
    category: statistics
    claim: "Round A finding A-02 (adaptive extension bias) reassessed in Round C"
    evidence: "v3 fixes 10 seeds with no observation-based extension"
    suggested_fix: "Already FIXED"
    status: FIXED
    resolution_notes: "Fixed canonical 10 seeds with no observation-based extension removes the adaptive winner's-curse mechanism. Precision remains a concern (CODEX-C-07) but that is not the original A-02 bias."
  - id: CODEX-A-03-ROUND-C
    severity: CRITICAL
    category: statistics
    claim: "Round A finding A-03 (PBO seed splits) reassessed in Round C"
    evidence: "PBO dropped; Hansen SPA replaces"
    suggested_fix: "Already FIXED"
    status: FIXED
    resolution_notes: "PBO is dropped rather than relabeled. SPA is an appropriate replacement for the global data-snooping/cherry-pick question if the candidate family is corrected to M=40/80 (CODEX-C-01)."
  - id: CODEX-A-04-ROUND-C
    severity: MAJOR
    category: statistics
    claim: "Round A finding A-04 (DSR formula underspecified) reassessed in Round C"
    evidence: "DSR dropped"
    suggested_fix: "Already FIXED"
    status: FIXED
    resolution_notes: "DSR is dropped, so its formula/trial-count underspecification is moot. SPA is defensible for academic multi-model superior-predictive-ability testing."
  - id: CODEX-A-05-ROUND-C
    severity: MAJOR
    category: data-leakage
    claim: "Round A finding A-05 (purge/embargo) reassessed in Round C"
    evidence: "§1.8 v3 contract added but sanity check formula is backwards"
    suggested_fix: "Rewrite sanity check per CODEX-C-02"
    status: STILL-OPEN
    resolution_notes: "The temporal contract is expanded, but the proposed between-fold sanity check is wrong by construction and must be rewritten — see CODEX-C-02."
  - id: CODEX-A-06-ROUND-C
    severity: MAJOR
    category: prior-art
    claim: "Round A finding A-06 (literature matrix) reassessed in Round C"
    evidence: "§1.9 expanded to 16 papers but spot checks found incorrect market entries + missing papers"
    suggested_fix: "Re-verify matrix; add missing papers per CODEX-C-06"
    status: STILL-OPEN
    resolution_notes: "The matrix expanded to 16 papers but spot checks found incorrect market/title details and missing 2024-2025 graph-stock papers — see CODEX-C-06."
  - id: CODEX-A-07-ROUND-C
    severity: MAJOR
    category: reproducibility
    claim: "Round A finding A-07 (LSTM/Mamba hparam pre-reg) reassessed in Round C"
    evidence: "LSTM dropped from E1; LightGBM Qlib defaults pre-registered"
    suggested_fix: "Already FIXED; clean LSTM references per CODEX-C-01"
    status: FIXED
    resolution_notes: "LSTM removed from E1 and LightGBM defaults are pre-registered. Defensible as a no-search fixed-default baseline, subject to cleaning stale LSTM references under CODEX-C-01."
  - id: CODEX-A-08-ROUND-C
    severity: MAJOR
    category: correctness
    claim: "Round A finding A-08 (Mamba narrative) reassessed in Round C"
    evidence: "Mamba reclassified OPTIONAL E5 with pre-registered ablation matrix"
    suggested_fix: "Already ACCEPTED-AS-CONCERN; do not promote Mamba to core without reopening"
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "Mamba is optional E5 with an ablation matrix if run; GO/SKIP gate is time-budget based rather than performance-based, so this remains appropriate."
  - id: CODEX-A-09-ROUND-C
    severity: MAJOR
    category: statistics
    claim: "Round A finding A-09 (cross-pick nulls) reassessed in Round C"
    evidence: "DM/HLN + SPA in v3; but seed aggregation underdefined"
    suggested_fix: "Pre-register seed aggregation per CODEX-C-04"
    status: STILL-OPEN
    resolution_notes: "DM/HLN + SPA is the right direction, but seed aggregation, loss definition, and model-family dimensions are still inconsistent — see CODEX-C-04."
  - id: CODEX-A-10-ROUND-C
    severity: CONCERN
    category: correctness
    claim: "Round A finding A-10 (Mamba underpowered) reassessed in Round C"
    evidence: "Mamba E5 OPTIONAL"
    suggested_fix: "Already ACCEPTED-AS-CONCERN"
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "Optional exploratory E5 is appropriate. Do not promote Mamba to a core positive anchor without reopening this finding."
  - id: CODEX-A-11-ROUND-C
    severity: CONCERN
    category: reproducibility
    claim: "Round A finding A-11 (compute optimistic) reassessed in Round C"
    evidence: "Cells reduced 1450→400; smoke gating retained"
    suggested_fix: "Already FIXED; clean 1450/500-cell text per CODEX-C-01"
    status: FIXED
    resolution_notes: "v3 materially reduces E1 compute and retains smoke-benchmark gating/checkpointing. Stale 1450/500-cell text must be cleaned under CODEX-C-01 but the design is fixed."
summary:
  critical: 1
  major: 5
  concern: 2
  round_a_disposed: 11
  round_c_findings_status_after_option_B_plus_C03: "6 FIXED (C-01, C-02, C-03, C-04, C-05, C-07) + 2 ACCEPTED-AS-CONCERN (C-06 paper-writing, C-08 paper-limitations) — all OPEN findings disposed"
overall_verdict: BLOCK-EXECUTION
verdict_status_after_option_B_plus_C03: "Option B+ fixes applied 2026-05-26 evening (H博士 extended Option B with immediate C-03 fix). Ready for Round D verification. v3 plan + prereg now internally consistent (C-01 cleaned), sanity check correct (C-02), news PIT artifact schema spec'd (C-03), seed aggregation locked (C-04), cost ladder locked (C-05), MDE pre-registered (C-07). Only C-06 (literature matrix re-verify) deferred to paper writing phase and C-08 (B/C HP transfer) to §Limitations during writing — both NOT v3 blockers."
---

# Round C Review Body

## Overall Assessment

The v3 concept is substantially stronger than Round A. Five Round A findings are genuinely fixed (A-02, A-03, A-04, A-07, A-11), two are accepted-as-concern with appropriate OPTIONAL framing (A-08, A-10), and four remain open (A-01, A-05, A-06, A-09). However, the target plan document itself has not been fully updated to reflect v3 — it still contains v2-era text, counts, model lists, and method references that conflict with the canonical prereg.json. That document-level inconsistency is the CRITICAL blocker (CODEX-C-01). The other five MAJOR findings are substantive methodology issues.

---

## CODEX-C-01 (CRITICAL): Plan Document Not Internally Consistent with v3

The prereg.json is cleaner than the plan. Specific conflicts found:

- Plan line 230-235: E1 loop still iterates over `[GAT, SAGE-Mean, MLP, LSTM, LightGBM]` (5 models, cell IDs 0-499)
- Plan line 394-407: SPA section specifies M=50 per universe, M=100 joint
- Plan line 417-428: DM family includes LSTM vs LightGBM test
- Plan lines 498-503: testing ledger credits 5 model × 10 seed × 5 fold × 2 universe = 500 cells
- Plan lines 539-543: still references `compute_dsr.py` as an active output
- Plan lines 689-697: smoke benchmark still validates 5 models × 1450 cells
- prereg.json:140-156 correctly states 4 models, 400 cells, M=40/80 SPA

The plan is currently a v2→v3 partial rewrite. No engineer can execute E1 from the plan document without resolving these conflicts. The fix is a targeted edit pass over the plan to align it with prereg.json as the ground truth.

---

## CODEX-C-02 (MAJOR): Sanity Check Formula is Backwards

The v3 §1.8 purge/embargo formula is:

```
assert (fold_{N+1}.train_end_effective - 42d) > fold_N.test_end
```

In the listed expanding-window fold structure, fold 0 test_end is 2024-06-30 and fold 1 train_end is 2024-03-31. The left side evaluates to approximately 2024-02-19, which is before the right side 2024-06-30. The assertion fails on the example data, meaning either the folds are listed in the wrong order, the formula operands are swapped, or the expanding-window structure is different from the description. This is not a labeling issue; the direction is wrong.

The intended check should verify that no training label touches test features. In an expanding walk-forward where each fold's test period follows its train period, the correct within-fold check is:

```
max(train_label_effective_end) + gap < min(test_feature_date)
max(val_label_effective_end) + gap < min(test_feature_date)
```

And the cross-fold check (if needed) is:

```
fold_N.test_end + embargo_gap < fold_{N+1}.val_start
```

---

## CODEX-C-03 (MAJOR): News PIT Not Enforceable Against Actual Artifact

The fullscale events artifact at `data/fullscale/sp500_news_events.parquet` was built to contain `date,ticker,...,return_next,label` fields, confirmed from scripts/prepare_events.py. The E3 news-edge builder needs a new, clean artifact derived from this source that:

1. Strips all forward-looking fields (return_next, label)
2. Retains article-level granularity (deduplicated by article_id or content hash, not by date-ticker)
3. Exposes UTC publication_timestamp for PIT gating
4. Is partitioned by (article_id, tickers_mentioned, publication_date) rather than (date, ticker, aggregate_sentiment)

Until this artifact exists, the E3 PIT rule cannot be verified at execution time, regardless of how clearly §1.2 describes it.

---

## CODEX-C-04 (MAJOR): DM/HLN Seed Aggregation Underdefined

The plan describes `d(t) = L_A(t) - L_B(t)` where L is daily IC-based loss, but does not say:
- Whether seeds contribute independent daily observations (inflating T by 10x)
- Whether the 10 seed series are averaged to one model series before DM
- Whether per-seed DM p-values are aggregated by minimum, median, or Simes/Fisher

If 10 seeds are pooled as independent rows, T becomes 3130 (not 313), and the HLN correction assumes a longer effective test series than actually exists. The correct approach is to average IC across seeds per date and model, giving a 313-day series per (model, universe) pair, and run DM on that. This must be pre-registered before execution.

---

## CODEX-C-05 (MAJOR): Cost Ladder Formula Mismatch

Plan line 478 formula is `gross_pos_return(t) = mean(score_i * r_i for i in top_decile) - mean(score_j * r_j for j in bottom_decile)`, which is a score-weighted return, not equal-weight. Plan line 473 says "equal-weight." These are inconsistent. Choose one, pre-register it, and define bps units (one-way vs. round-trip). At 21d non-overlapping, each fold has roughly 3 periods in the test window, so Net Sharpe estimates are highly imprecise and should be labeled secondary with uncertainty bounds (e.g., bootstrap CI over periods).

---

## CODEX-C-06 (MAJOR): Literature Matrix Inaccuracies

Spot-checked 8 papers against primary sources:

| Plan Row | Plan Claim | Actual |
|----------|-----------|--------|
| FinGAT | Market: Taiwan | arXiv:2106.10159 tests Taiwan, S&P 500, and NASDAQ |
| HIGSTM | Market: CSI300 | arXiv:2503.11387 reports CSI500, CSI800, CSI1000 |
| HTAN row | DOI 10.1145/3768292.3770389 | Resolves to a Hypergraph paper, not HTAN |
| MASTER | Market: US | Confirmed correct |
| MDGNN | Market: US | Confirmed correct |

Missing 2024-2025 papers:
- GRU-PFG (arXiv:2411.18997): US S&P 500 graph + sentiment
- DishFT-GNN (arXiv:2502.10776): US market, dynamic graph
- DGT (arXiv:2506.18717): S&P 500, dynamic graph transformer

These omissions weaken the novelty claim. The claim "first systematic conditional study on US S&P 500 with 7 axes controlled" requires a verified negative sweep of the corrected matrix.

---

## CODEX-C-07 (CONCERN): 10 Seeds Precision for No-Effect Claims

GAT 21d history shows CV=55%, meaning 10 seeds gives SE ≈ 0.018/√10 ≈ 0.0057. A 95% CI for a single model is roughly ±0.011 around the mean. If the headline claim is "GNN does not significantly beat MLP at 21d," then non-rejection from DM should be accompanied by a CI showing the upper bound on ΔIC is below a practically relevant threshold (e.g., 0.005). Otherwise the result reads as "we couldn't detect a difference," which is weaker than "the difference is ruled out." Pre-register the interpretation rule before execution.

---

## CODEX-C-08 (CONCERN): Universe B/C HP Transfer

The plan and prereg lock identical default hyperparameters for B (10-dim) and C (51-dim). This is reproducible and defensible as a scope choice, but it means any capacity-sensitive model (LightGBM with num_leaves=31 may underfit 51 features) will show B/C differences that conflate feature richness with model capacity. The plan should explicitly note this in the B/C comparison section and frame results accordingly.

---

## Round A Dispositions Summary (Round C reassessment)

| Finding | v3 Disposition Claimed | Round C Verdict |
|---------|---------------|----------------------|
| A-01 (news PIT) | claimed fixed via §1.2 | STILL-OPEN (enforcement missing) |
| A-02 (adaptive extension) | fixed: 10 fixed seeds | FIXED |
| A-03 (PBO seed splits) | fixed: PBO dropped, SPA replaces | FIXED |
| A-04 (DSR formula) | fixed: DSR dropped, SPA replaces | FIXED |
| A-05 (purge/embargo) | claimed fixed via §1.8 | STILL-OPEN (sanity check wrong) |
| A-06 (literature matrix) | claimed fixed via expanded 16-paper matrix | STILL-OPEN (inaccuracies found) |
| A-07 (LSTM hparam) | partially resolved: LSTM dropped, LGB pre-reg added | FIXED |
| A-08 (Mamba ablation) | Mamba → optional E5 | ACCEPTED-AS-CONCERN |
| A-09 (cross-pick nulls) | DM+HLN + SPA | STILL-OPEN (seed aggregation underdefined) |
| A-10 (Mamba power) | Mamba → optional E5 | ACCEPTED-AS-CONCERN |
| A-11 (compute) | cells reduced 1450→400, smoke gating retained | FIXED |

---

## Path to PROCEED-WITH-FIXES

The plan can reach PROCEED-WITH-FIXES with the following targeted changes, none of which require redesigning the experiment:

1. **CODEX-C-01**: Edit plan document to remove all stale v2 text (5-model loops, LSTM DM tests, M=50/100 SPA, 500-cell count, DSR references, 1450-cell smoke gate). Prereg.json is already correct; align plan to it.
2. **CODEX-C-02**: Fix the between-fold sanity check formula or replace it with a within-fold label-to-feature-gap assertion in trading-day units.
3. **CODEX-C-03**: Pre-register that E3 requires a new news edge artifact built from the raw parquet with forward fields stripped, article_id added, and UTC publication_timestamp exposed.
4. **CODEX-C-04**: Add one paragraph to prereg.json specifying seed aggregation: average IC across seeds per (model, date, fold) before forming the daily DM series; use L=-IC.
5. **CODEX-C-05**: Choose score-weighted or equal-weight, add bps definition (one-way vs. round-trip), and mark Net Sharpe secondary with uncertainty note.
6. **CODEX-C-06**: Re-verify all 16 matrix rows against primary papers, correct FinGAT and HIGSTM market entries, resolve HTAN DOI, add 3 missing 2024-2025 papers, soften novelty language.

---

## Round D re-verification update (appended 2026-05-26 night)

Codex Round D was triggered after Option B+ fixes for C-01/C-02/C-03/C-04/C-05/C-07. **Round D independent verification revealed that only 3/8 Round C dispositions actually held** (C-06 DEFERRED → still valid deferral; C-07 FIXED → verified; C-08 ACCEPTED → verified). The remaining 5 (C-01/C-02/C-03/C-04/C-05) were marked FIXED in Round C based on intent rather than verification; Round D surfaced concrete residual bugs in each:

| Round C finding | Round C status (claimed) | Round D verification result | Re-mapped status |
|----------------|--------------------------|------------------------------|-------------------|
| C-01 (stale v2 text) | FIXED | NEW-CONCERN — cell_id formula math wrong (D-01: `*100,*25,*5` gives max=204 with collisions, not 399); 8-test family text residual; `compute_dsr.py` still referenced; "500 cells" still in §1.5 | **FIXED-VIA-D-01** (plan §1.1 formula corrected to `*200,*50,*10`; residuals cleaned 2026-05-26 night) |
| C-02 (sanity check backwards) | FIXED | STILL-OPEN — replacement assertion `train_feature_end + 21 < val_end_idx - 21 + 21` algebraically simplifies to `train_end_idx < val_end_idx` (trivially true; D-02) | **FIXED-VIA-D-02** (plan §1.8 + prereg purge_embargo rewritten with explicit label_end vs feature_start comparison on trading-day arrays) |
| C-03 (PIT artifact) | FIXED | NEW-CONCERN — UTC midnight cutoff leaves 3-4h after-hours leak window (NYSE closes 20-21 UTC; D-03) | **FIXED-VIA-D-03** (schema doc + plan §1.2/§1.8 + prereg use NYSE session_close via pandas_market_calendars; DST worked examples added) |
| C-04 (DM seed aggregation) | FIXED | NEW-CONCERN — SPA M=40/80 spec contradicts seed-aggregation rule (D-04) | **FIXED-VIA-D-04** (SPA M=3 per universe, M=6 joint — applies seed aggregation consistently with DM/HLN) |
| C-05 (cost ladder) | FIXED | STILL-OPEN — turnover "≈2 at full rotation" wrong (L1-norm gives 4; D-05); N_periods "5×3=15" wrong (actual 3+4+4+3+3=17 from horizon_ablation_results.csv) | **FIXED-VIA-D-05** (turnover_L1 formula + cost coefficient clarified; N=17 verified against CSV) |
| C-06 (literature matrix) | DEFERRED | Confirmed deferral OK | **DEFERRED-VERIFIED** (paper-writing-phase blocker, not E1 blocker) |
| C-07 (MDE pre-commit) | FIXED | Confirmed FIXED — prereg block present and locked | **FIXED-VERIFIED** |
| C-08 (B/C HP transfer) | ACCEPTED | Confirmed acceptance OK as §Limitations | **ACCEPTED-VERIFIED** |

**Process lesson (logged for future touchpoints)**: a "FIXED" disposition that relies on the author's intent rather than independent verification (formula recomputation, residual grep, cross-data check) is not actually FIXED. Going forward, every CRITICAL/MAJOR fix must include an explicit Verification step in its disposition note (matching `progress.md` entry 2026-05-26-h "Round D fix landing").

Full Round D review: [`artifacts/reviews/2026-05-26_codex_plan_D.md`](2026-05-26_codex_plan_D.md). Round E queued after the 10 Round D fixes (5 D-series + 5 C-series re-mappings) commit to plan + prereg + schema doc.
