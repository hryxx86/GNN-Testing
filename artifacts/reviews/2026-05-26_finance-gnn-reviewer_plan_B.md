---
reviewer: finance-gnn-reviewer
touchpoint: plan
round: B
target_plan: /Users/heruixi/.claude/plans/handoff-session-ranking-swirling-lemur.md
prior_round_review: artifacts/reviews/2026-05-26_codex_plan_A.md
fallback_reason: "Codex CLI returned available: false; finance-gnn-reviewer fallback per CLAUDE.md Rule 9"
findings:
  - id: CODEX-A-01
    severity: CRITICAL
    category: data-leakage
    claim_round_a: "News co-occurrence edge not point-in-time; can front-run prediction timestamp."
    v2_disposition_text: "§1.2 rewrites edge spec: articles with publication_timestamp ≤ end_of_day(t-1), lookback [t-6, t-1], explicit ban on articles whose date falls inside [t, t+21], reuse np.roll(news_emb, 1, axis=0) T-1 pattern from run_week1_FIXEDNOTRUN.py:168-171."
    verification_status: FIXED
    verification_notes: "§1.2 enumerates the temporal contract correctly. Three correct elements: (i) eligibility by publication_timestamp ≤ end_of_day(t-1); (ii) lookback window [t-6, t-1] excludes day t; (iii) explicit exclusion of articles inside the [t, t+21] label window. The reuse of np.roll(news_emb, 1, axis=0) was verified at run_week1_FIXEDNOTRUN.py:160-199 (T-1 shift pattern is the same family). Two concerns remain (not blockers, fold into FINGNN-B-02): (a) news_events.parquet `date` column is normalized via dt.tz_localize(None).dt.normalize() per run_horizon_ablation.py:100 — confirm in §1.2 implementation that the comparator uses publication TIMESTAMP not just `date` (intra-day articles after market close on day t-1 are problematic if market data is end-of-day t-1); (b) §1.2 says symmetric, no self-loops, but does not say what happens on days with zero articles (empty graph → fall back to correlation-only?)."
  - id: CODEX-A-02
    severity: CRITICAL
    category: statistics
    claim_round_a: "Adaptive 30→100 seed extension conditions on observed pilot mean → winner's-curse inflation."
    v2_disposition_text: "Adaptive extension DROPPED. Pre-committed fixed sizes: SAGE-Mean=100, GAT=100, MLP/LSTM/Mamba-SAGE=30. Sizes written into experiments/storya_multiseed/prereg.json before any cell runs. prereg.json v2 explicitly forbids observation-based extension and any post-hoc seed adjustment."
    verification_status: FIXED
    verification_notes: "prereg.json v2 is well-structured: fixed_sample_sizes_per_model committed, what_is_FORBIDDEN_under_this_v2_prereg lists 8 prohibited actions including 'Observation-based extension' and 'Reporting top-K-by-test-IC as headline'. The selection-bias fix is correct. However, it introduces a new wrinkle (see FINGNN-B-01): the v2 plan §1.1 states '30-seed models: first 30 from same pool', creating SEED OVERLAP between 30-seed and 100-seed models. For per-cell paired ΔIC tests (§1.4(c)) that pair GAT_s86_f0 with MLP_s86_f0, this is actually desirable (paired test requires same seed). But for unpaired analyses or for pooling across (seed,fold), the 30-seed MLP cells are NOT independent of the 30-seed SUBSET of GAT cells — the seed-driven variance components are shared. This must be acknowledged in the variance estimator for §1.4(c). See FINGNN-B-01."
  - id: CODEX-A-03
    severity: CRITICAL
    category: statistics
    claim_round_a: "PBO procedure splits seeds not time observations — not Bailey-Borwein-Lopez de Prado CSCV."
    v2_disposition_text: "PBO DROPPED entirely. F2 framework (DSR at selected-headline level + block-bootstrap CI on per-day IC + BH-FDR over 4 paired ΔIC comparisons against MLP) replaces it."
    verification_status: FIXED
    verification_notes: "Dropping PBO eliminates the misapplication. The F2 framework covers two of three things CSCV provides: (1) cross-validation-style robustness (via 5-fold walk-forward, already there) and (2) selection-bias correction (via DSR's N_trials). What F2 does NOT directly replicate is CSCV's specific 'probability of backtest overfit' as a scalar diagnostic of in-sample vs out-of-sample rank stability across symmetric time splits. This is a non-trivial gap — see FINGNN-B-03. However, given (a) the plan's existing walk-forward design with non-overlapping test periods Q2-2024 through Q2-2025, and (b) the DSR's N_trials accounting captures the selection-bias dimension PBO was supposed to provide, dropping PBO is defensible. The literature matrix §1.9 still claims '✅' for 'Overfit diagnostic' for our paper — verify this remains tenable post-PBO-drop (DSR alone is publishable as overfit diagnostic in JPM-style finance venues)."
  - id: CODEX-A-04
    severity: MAJOR
    category: statistics
    claim_round_a: "DSR formula and trial count underspecified; per-cell DSR not the BLPdP DSR object."
    v2_disposition_text: "§1.4(a) computes DSR ONLY at selected headline level (not per cell). Full BLPdP formula written out with skew/kurt/N_trials. σ_SR_estimator = sqrt((1 - skew × SR + (kurt - 1)/4 × SR²) / (T - 1)). N_trials counted explicitly via §1.4(d) multi-testing ledger."
    verification_status: STILL-OPEN
    verification_notes: "Two real issues with the v2 formula. (1) σ_SR_estimator KURTOSIS CONVENTION AMBIGUITY: Bailey-Lopez de Prado 2014 eq. (9) writes (γ_4 - 1)/4 where γ_4 is the FOURTH STANDARDIZED MOMENT (a.k.a. raw kurtosis; for a normal it is 3, so the term reduces to (3-1)/4 = 0.5, contributing 0.5·SR²). If `kurt` in the v2 plan means EXCESS kurtosis (the scipy.stats.kurtosis default), then for a normal it is 0, giving (0-1)/4 = -0.25·SR², which is wrong sign. The plan must explicitly state convention: either kurt = γ_4 (full fourth standardized moment, normal=3) or excess kurt (normal=0) with the formula adjusted to (excess_kurt + 2)/4. This will silently bias DSR if a coder uses scipy.stats.kurtosis (excess by default) with the plan's literal formula. (2) E[max_SR_null] formula: 'μ_SR + σ_SR × [(1 - γ) × Z^{-1}(1 - 1/N_trials) + γ × Z^{-1}(1 - 1/(N_trials × e))]' matches BLPdP exactly — this part is correct. (3) ledger has 'hyperparameter_trials_per_model: 1' — this contradicts the historical reality that Phase 5 / Plan AAA explored many HP configurations even if the v2 plan freezes them. The ledger should distinguish 'current-experiment HP trials = 1' from 'historical HP trials informing the claim = M' where M is the count of distinct HP configs ever tried for the headline model. Without this, N_trials is undercount and DSR is liberal. RECOMMEND: H博士 decide whether kurtosis convention is raw or excess; document in §1.4(a) explicitly; and audit historical HP trial count from Phase 5 / Plan AAA logs to populate ledger correctly."
  - id: CODEX-A-05
    severity: MAJOR
    category: data-leakage
    claim_round_a: "Walk-forward does not specify purge/embargo for overlapping 21d labels or preprocessing boundaries."
    v2_disposition_text: "§1.8 'Temporal contract' table added: T-1 features via prices.shift(1); 21d labels via prices.shift(-21); C1 purge (last 21 train+val days) at run_horizon_ablation.py:316; per-fold winsorization+scaling; correlation graph last-snapshot ≤ end_of_day(t-1). Explicit text: 'fold N+1 train_end > fold N test_end is NOT required because fold N+1 doesn't see fold N's test labels'."
    verification_status: STILL-OPEN
    verification_notes: "VERIFIED at code level: run_horizon_ablation.py:316-318 does implement train_days = train_days[:-horizon] (and val_days[:-horizon]), so within-fold purge is real. HOWEVER, the v2 reasoning about adjacent-fold embargo is WRONG and matches exactly the concern listed in the user's verification spot #4. Per WALK_FORWARD_FOLDS (file lines 72-83): fold 0 test_end = 2024-06-30; fold 1 train_end = 2024-03-31. So fold 1 training INCLUDES dates 2024-04-01 through 2024-03-31 (none) — actually wait: fold 1 train_end = fold 0 val_end = 2024-03-31, so fold 1 train data is [TRAIN_START, 2024-03-31]. Fold 0's test labels for the LAST test day around 2024-06-30 require prices through 2024-06-30 + 21d ≈ 2024-07-21. Fold 1's training data ends 2024-03-31 — before fold 0's test starts. So actually fold 1 training does NOT overlap fold 0's test-label window. The reasoning in §1.8 'fold N+1 train_end > fold N test_end is NOT required' is therefore stating something correct for THIS PARTICULAR fold layout: each fold's train_end = previous fold's val_end, which is BEFORE the previous fold's test_end. But the rationale given is sloppy. The CORRECT rationale is: 'Adjacent folds use chronologically ordered splits where fold N+1's train_end equals fold N's val_end, both of which are strictly before fold N's test_start; fold N's test labels reach to test_end + 21d, which is also before fold N+2's train_start.' VERIFY by computing fold N+1 val_end vs fold N test_end + 21d. Fold 0 test_end = 2024-06-30, +21bd ≈ 2024-07-30. Fold 1 val_end = 2024-06-30. Fold 1 val starts at 2024-04-01 (after fold 1 train_end). FOLD 1 VAL OVERLAPS FOLD 0 TEST PERIOD (both contain Q2-2024 dates). This is a real overlap: fold 1's val period [2024-04-01, 2024-06-30] equals fold 0's test period. Since features at date t use T-1 prices, fold 1 val features overlap fold 0 test features. Whether this is a leak depends on what 'val' is used for in fold 1: model selection (early stopping) only, not training. Since model parameters are not updated from val, and val features at date t use t-1 prices which are the same data fold 0 already saw in TEST, the question is: does ranking on val days in fold 1 'leak' future info? Answer: No, because we are choosing the best epoch for fold 1, not training on val data. BUT: if winsorization or scaling is per-fold and fit on fold 1's TRAIN (which ends 2024-03-31), then fold 1 val/test scaling is consistent. This is fine. NET ASSESSMENT: the fold layout is acceptable IF and ONLY IF (a) features are strictly T-1 (verified at lines 178-182), (b) per-fold winsor/scaler is fit only on train (claimed in §1.8), and (c) cross-fold feature caching does not use a single global scaler. Recommend: verify in run_storya_multiseed.py code that StandardScaler is instantiated PER FOLD inside the fold loop, not cached across folds. Mark this finding STILL-OPEN pending code review at Touchpoint 2; the PLAN is correct in intent but the rationale text in §1.8 is misleading and risks downstream coder misimplementation."
  - id: CODEX-A-06
    severity: MAJOR
    category: prior-art
    claim_round_a: "Novelty claim too strong without formal related-work table on horizon/feature/relation/regime/seed/PIT/overfit axes."
    v2_disposition_text: "§1.9 literature matrix added with 9 rows (HATS, RSR, FinGAT, MASTER, MDGNN, SAMBA, FinMamba, OmniGNN, When Alpha Breaks). Caveat: 'preliminary; must expand with 5+ more recent papers before submission.'"
    verification_status: FIXED
    verification_notes: "Matrix is the right structure. Three matrix entries warrant verification before submission (not blockers for v2 plan approval): (1) HATS '❌ (5d only)' — HATS arXiv:1908.07999 uses 1d, 7d, 20d, 30d returns per their original Table 4; horizon IS varied. Update to 'partial (4 fixed horizons, no monotone analysis)'. (2) FinMamba '❌ (1d only)' — verify against arXiv:2502.06707 directly; if FinMamba evaluates multi-horizon this entry is misleading. (3) OmniGNN '✅ (COVID shock)' — arXiv:2510.10775 — verify this is genuine regime evaluation with a held-out shock period vs in-sample regime mention. Otherwise the table structure is sound, and the caveat about expanding the survey is appropriate. The 'first systematic conditional study where ALL controlled simultaneously' framing in §1.9 is defensible IF the table is genuinely accurate. The 2024-2025 GNN-finance survey gap (5+ more papers) needs to be filled BEFORE paper §2 is finalized — papers to add at minimum: HIGSTM (arXiv:2503.11387, in Codex's Round A sources), THGNN, MDGNN ✓ already there, GraphMixer/StockMixer (AAAI'24), and any 2026 ICAIF/AAAI/KDD GNN-finance work."
  - id: CODEX-A-07
    severity: MAJOR
    category: reproducibility
    claim_round_a: "New LSTM and Mamba-SAGE baselines added without pre-registered hyperparameter protocol."
    v2_disposition_text: "§1.7 locks HP grid per model. All 5 models use hidden=64, layers=2, dropout=0.3, lr=1e-3, wd=1e-4, epochs=100, patience=15. 'Source for these values: copy from archived/scripts/run_horizon_ablation.py PARAMS dict (lines 54-64)... production defaults.' No HP search; deviations count in multi-testing ledger."
    verification_status: STILL-OPEN
    verification_notes: "VERIFIED at run_horizon_ablation.py:54-64 — PARAMS dict does contain hidden_channels=64, num_layers=2, dropout=0.3, lr=1e-3, weight_decay=1e-4, epochs=100, patience=15. So GAT, SAGE-Mean, MLP HP values match production defaults — that part is honest. HOWEVER: LSTM_price and Mamba-SAGE_price are NEW to codebase per §1.7 and prereg.json. There are NO production defaults for them. Using hidden=64, layers=2, dropout=0.3 by analogy is a choice, not a documented prior. The justification 'production defaults' in §1.7 is therefore correct for GAT/SAGE/MLP but FALSE for LSTM and Mamba-SAGE. The honest framing: 'for LSTM and Mamba-SAGE, we adopt the same hyperparameters as GNN models without per-architecture tuning, by deliberate choice to avoid HP search inflation; this may handicap their absolute performance.' Also, for LSTM 'layers=2 (bi-LSTM)' the bidirectional flag doubles parameter count, breaking the §1.7 'parameter counts within 2× across models' capacity-match claim. RECOMMEND: H博士 either (a) make bi-LSTM unidirectional to capacity-match, or (b) add explicit param-count table to §1.7 showing actual counts and noting Bi-LSTM is 2× the capacity (and what that implies for interpretation). Mamba HP: 'Mamba d_model=13, expand=2' is technically a Mamba-specific choice — verify these are SAMBA paper defaults (arXiv:2410.03707) or standard mamba-ssm library defaults; document source."
  - id: CODEX-A-08
    severity: MAJOR
    category: correctness
    claim_round_a: "Mamba-SAGE insurance can contradict story narrative unless graph contribution isolated via ablations."
    v2_disposition_text: "§1.5 pre-registers 5-cell ablation matrix A1-A5 (Mamba-only, SAGE-only on 13 features, Mamba-SAGE main, Mamba+identity graph, Mamba+shuffled graph) with outcome-to-claim mapping pre-committed."
    verification_status: FIXED
    verification_notes: "Ablation matrix is well-designed. Pre-committed outcome map (A3 > A1 AND A3 > A2 → narrative intact; A3 ≈ A1 → temporal-dominant; A3 ≈ A4 OR A3 ≈ A5 → strongly anti-graph) is exactly the right structure for falsifiable conditional claims. One gap: capacity-match — §1.5 says 'pad with extra Linear layers in A2 to match Mamba's ~5K params' but does not specify the parameter counts. A2 (SAGE-only) without Mamba is naturally smaller; padding with Linear adds capacity but in different layer roles, which is not a clean capacity-match. Mark as fine-to-proceed; rec is to log final param counts in results.csv per cell so reviewers can verify the match post hoc."
  - id: CODEX-A-09
    severity: MAJOR
    category: statistics
    claim_round_a: "Cherry-pick detection lacks defined primary nulls, paired effect sizes, multi-testing family."
    v2_disposition_text: "§1.4(c) defines H₀_k: E[ΔIC_t(model_k, MLP_price)] = 0 for k ∈ {GAT, SAGE-Mean, LSTM, Mamba-SAGE}. Family = 4 paired comparisons against MLP. NW-HAC SE on per-day ΔIC series with auto lag = floor(4×(T/100)^(2/9)). Two-sided p-value. BH-FDR at q=0.05 over family-of-4. §1.4(d) multi-testing ledger with explicit trial counts (model selection=5, seed=100/30, fold=5, horizon historical=6, edge variants=2+4, prior groups=61, prior subsets=7)."
    verification_status: STILL-OPEN
    verification_notes: "Primary null is correct and pairing against MLP is appropriate. Family-of-4 with BH-FDR is sensible. THREE OPEN ISSUES: (1) MIXED SEED COUNTS in paired test — see FINGNN-B-01. The text 'ΔIC_t = mean across (seed, fold) of IC_t(model_k) − IC_t(MLP_price) at each test date t' is ambiguous when model_k has 100 seeds and MLP has 30. Three valid interpretations: (a) inner pairing on first 30 seeds only (uses 30×5×T cells per model) — clean paired test; (b) outer averaging: mean over GAT's 100 seeds minus mean over MLP's 30 seeds — UNPAIRED at the seed level, which violates 'paired' in the test name and inflates the variance estimate; (c) outer averaging then NW-HAC on the difference of per-day means — closer to (b). The plan needs to LOCK option (a) explicitly. (2) NW-HAC AUTO LAG: Newey-West 1994 auto lag floor(4×(T/100)^(2/9)) for T≈63d (one fold's test) gives floor(4×(0.63)^(2/9)) = floor(4×0.905) = 3. This is too short for 21d overlapping-label autocorrelation. Need lag ≥ 21 to handle the actual autocorrelation in IC series. RECOMMEND: override auto-lag with max(auto, 21) for 21d-horizon IC series. (3) The BH-FDR family of 4 is the in-paper comparison family. The ledger §1.4(d) counts historical trials separately (good). But the BH-FDR q=0.05 over family-of-4 is conditional on having pre-committed exactly these 4 models. If H博士 later 'reports SAGE-Mean as headline' and 'GAT-21d as failure-mode anchor' then the family is two separate claim families and FDR control needs to be applied per-family or holistically — document this disambiguation in §1.4(c)."
  - id: CODEX-A-10
    severity: CONCERN
    category: correctness
    claim_round_a: "Vanilla Mamba on 21×13 is outside SAMBA/FinMamba validated regime; may be underpowered baseline."
    v2_disposition_text: "§1.5 marked EXPLORATORY; A-08 ablation matrix addresses interpretation; §Limitations to include caveat 'Mamba on 21d×13 input is outside SAMBA's validated regime; results are preliminary.'"
    verification_status: STILL-OPEN
    verification_notes: "The 'mark as exploratory' framing addresses Codex's correctness concern. HOWEVER: A-10's full intent included 'Mamba-only, GRU/LSTM/TCN, and capacity-matched baselines' — i.e., a temporal-encoder horse race. §1.5's A1-A5 matrix has A1=Mamba-only and A2=SAGE-only, BUT NO LSTM/GRU/TCN capacity-matched temporal-encoder baselines. The plan claims §1.1's LSTM_price 30-seed run covers this, but §1.1 LSTM_price uses 9-dim flat price features (not 21×13 sequence input), so it does NOT capacity-match Mamba's temporal-sequence encoder role. RECOMMEND: §1.5 should ADD an A6 = LSTM (or GRU) with same 21×13 input as Mamba, same param count, same training; this is the true Mamba baseline. Without it, a Mamba-SAGE win cannot be attributed to Mamba over a simpler RNN. See FINGNN-B-04."
  - id: CODEX-A-11
    severity: CONCERN
    category: reproducibility
    claim_round_a: "Compute estimate ~1.6 min/cell unverified; needs measured smoke before committing budget."
    v2_disposition_text: "§1.10 smoke benchmark added as non-bypassable gating milestone. Protocol: 5 cells × 1 seed × 1 fold; measure wall_time + GPU mem; output smoke_benchmark.csv. Decision gate: <30min → full budget; >60min → drop models from §1.1 (NO reverting to F1 adaptive design)."
    verification_status: FIXED
    verification_notes: "Smoke gate is appropriate. One detail: §1.10 says 'drop a model from §1.1 to fit budget' if smoke is slow — this would be a post-commitment change. Pre-commit the drop ORDER NOW: e.g., 'if budget exceeded, drop in priority order: (1) Mamba-SAGE, (2) LSTM_price, (3) reduce SAGE-Mean to 50 seeds.' Otherwise the post-smoke decision risks being outcome-influenced if smoke also reveals which models look strong on the 1 seed × 1 fold result."
  - id: FINGNN-B-01
    severity: MAJOR
    category: statistics
    claim: "Mixed seed counts (100 GAT/SAGE-Mean vs 30 MLP) + seed-pool subset overlap make §1.4(c) paired ΔIC ambiguous. Plan does not specify whether paired test uses (a) all 100 GAT vs all 30 MLP (unequal precision, breaks 'paired' semantics), or (b) GAT_first30 ∩ MLP_first30 (clean pairing, wastes 70 GAT seeds)."
    evidence: "§1.4(c): 'ΔIC_t = mean across (seed, fold) of IC_t(model_k) − IC_t(MLP_price) at each test date t' — ambiguous for k = GAT or SAGE-Mean which have 100 seeds vs MLP's 30."
    suggested_fix: "Lock option (a) for primary test: paired ΔIC uses the first-30 seed subset for ALL models (since 30-seed pool = first 30 of 100-seed pool per §1.1, this is naturally paired at the seed level). Use the remaining 70 GAT/SAGE-Mean seeds for headline mean ± std + bootstrap CI in Table 1 (not paired tests). Document this split explicitly in §1.4(c)."
    status: OPEN
  - id: FINGNN-B-02
    severity: MAJOR
    category: statistics
    claim: "Block-bootstrap terminology inconsistency. §1.4(b) says 'stationary block bootstrap, block_len=21' but the cited implementation in run_plan_aaa_168_ranking.py:404 is explicitly Künsch 1989 FIXED-LENGTH block bootstrap (per code comment: 'NB: Plan v1 §3.4 mentioned stationary block bootstrap; this implementation matches analyze_tier1_phase_a.py's Künsch fixed-block variant'). These are DIFFERENT methods: stationary block bootstrap (Politis-Romano 1994) draws geometric block lengths with mean = b; Künsch fixed-length uses fixed b."
    evidence: "Plan §1.4(b): 'Stationary block bootstrap: block_len=21'; run_plan_aaa_168_ranking.py:404 docstring: 'Fixed-length block (Künsch 1989) bootstrap CI'."
    suggested_fix: "Either (a) change plan text to 'fixed-length block bootstrap (Künsch 1989), block_len=21' to match the cited implementation, OR (b) implement true stationary bootstrap (e.g., arch.bootstrap.StationaryBootstrap from `arch` package) and update the plan accordingly. For 21d-overlapping IC autocorrelation, stationary bootstrap is theoretically slightly preferred (random block lengths avoid edge artifacts), but Künsch is widely accepted and matches existing code; the only requirement is consistency between plan text and implementation citation."
    status: OPEN
  - id: FINGNN-B-03
    severity: CONCERN
    category: statistics
    claim: "DSR kurtosis convention ambiguity. §1.4(a) formula σ_SR_estimator = sqrt((1 - skew × SR + (kurt - 1)/4 × SR²) / (T - 1)) silently assumes raw 4th-moment kurtosis (γ_4, normal=3), but scipy.stats.kurtosis default is EXCESS kurtosis (normal=0). A naïve implementation using scipy default → (0-1)/4 = -0.25 contribution from kurtosis for a normal-distributed return, which is the WRONG SIGN."
    evidence: "Plan §1.4(a): 'σ_SR_estimator = sqrt((1 - skew × SR + (kurt - 1)/4 × SR²) / (T - 1)) — skew, kurt = skewness, excess-kurtosis of SELECTED strategy's daily returns'."
    suggested_fix: "The plan TEXT says 'excess-kurtosis' AND uses (kurt - 1)/4. These are inconsistent with Bailey-Lopez de Prado 2014 eq. 9. Either (a) keep 'excess-kurtosis' label and change formula to (excess_kurt + 2)/4, OR (b) keep formula as (kurt - 1)/4 and change label to 'raw 4th-moment kurtosis γ_4 (scipy.stats.kurtosis(..., fisher=False))'. Both are correct; pick one and lock it in compute_dsr.py with an assertion."
    status: OPEN
  - id: FINGNN-B-04
    severity: CONCERN
    category: correctness
    claim: "§1.5 Mamba ablation lacks LSTM/GRU/TCN capacity-matched temporal-encoder baseline. Without an RNN/CNN baseline on the SAME 21×13 sequence input, a Mamba-SAGE win cannot be attributed to Mamba over simpler temporal encoders. §1.1 LSTM_price uses 9-dim flat features, not 21×13 sequences, so it does not substitute."
    evidence: "Plan §1.5 ablation matrix A1-A5: 'Vanilla Mamba (T=21, D=13)' for A1/A3/A4/A5; A2 is 'SAGE-only on 13 features... none (use last-step features directly)'. No LSTM/GRU/TCN on (21,13) input."
    suggested_fix: "Add A6: GRU (or LSTM) with input (21, 13) → hidden 64, capacity-matched to Mamba (~5K params), same training pipeline. Even just one such baseline answers Codex A-10's intent. Pre-commit outcome map: 'A3 > A6 → Mamba beats RNN at temporal encoding'; 'A3 ≈ A6 → conditional benefit is from any temporal encoder, not specifically Mamba'. Adds 30×5×1 = 150 cells, +~8h A100."
    status: OPEN
  - id: FINGNN-B-05
    severity: CONCERN
    category: data-leakage
    claim: "Per-fold scaler instantiation is claimed in §1.8 but cross-fold scaler caching is a common bug. Plan does not require explicit unit test that scaler.mean_/scale_ differs across folds."
    evidence: "Plan §1.8: 'Per-fold StandardScaler fit on train, applied to val + test (no cross-fold contamination)'. No verification protocol."
    suggested_fix: "Add to §6.1 smoke test verification: 'Assert StandardScaler.mean_ differs across at least 2 folds with > 1e-6 L2 distance; if equal, scaler is being cached across folds — abort.'"
    status: OPEN
  - id: FINGNN-B-06
    severity: CONCERN
    category: statistics
    claim: "NW-HAC auto-lag with T≈63 (one fold's test days) gives lag = floor(4×(63/100)^(2/9)) = 3. With 21d overlapping forward-return labels, daily IC series has autocorrelation up to lag 21. Auto-lag of 3 will severely underestimate the long-run variance, inflating significance."
    evidence: "Plan §1.4(c): 'NW-HAC SE on the per-day ΔIC series (auto lag per Newey-West 1994: floor(4×(T/100)^(2/9)))'."
    suggested_fix: "Override auto-lag with max(auto, 21) for 21d-horizon experiments. Document choice in §1.4(c). Alternatively, pool across folds (T ≈ 315) which gives auto-lag = 5 — still too short. Lock manual lag = 21 (matches horizon and bootstrap block_len)."
    status: OPEN
summary:
  round_a_findings_fixed: 6
  round_a_findings_still_open: 5
  round_a_findings_rejected: 0
  new_findings_b_critical: 0
  new_findings_b_major: 2
  new_findings_b_concern: 4
overall_verdict: PROCEED-WITH-FIXES
---

# Round B Plan Review (fallback: finance-gnn-reviewer)

## Top-line

v2 plan addresses all 11 Round A findings substantively — no findings rejected, no gaslighting, the Section 9 disposition log is honest. Six are clean fixes; five carry residual issues that need clarification before code is written; six new B-tier findings (2 MAJOR, 4 CONCERN) surface from a deeper read.

**No CRITICAL findings remain.** The plan is structurally sound. The remaining gaps are mostly specification ambiguities (kurtosis convention, paired-test seed mismatch, NW lag, scaler verification) that need to be locked in §1.4(c)/§1.4(a)/§1.8 before run_storya_multiseed.py is written, not blockers for moving from plan to code.

## Reasoning on the high-stakes verifications

**Adjacent-fold leakage (A-05)**: I traced the dates manually. Fold 0 test = Q2-2024 (Apr-Jun); fold 0 test labels need prices through ~2024-07-30 (21bd after 2024-06-30). Fold 1 train ends 2024-03-31 (BEFORE fold 0 test starts), so fold 1 training cannot see fold 0 test labels. Fold 1 VAL = Q2-2024, which is the SAME calendar period as fold 0 test, but fold 1 val is only used for early stopping (no parameter updates). This is acceptable IF and ONLY IF (i) features are strictly T-1, (ii) per-fold StandardScaler is instantiated inside the fold loop. Both are claimed in §1.8 but the SCALER lifecycle has not been verified by reading the porting target code. The rationale in §1.8 itself ("fold N+1 train_end > fold N test_end is NOT required because fold N+1 doesn't see fold N's test labels") is technically true under this layout but misleading — the safer phrasing would describe the actual non-overlap of fold N test-label window vs fold N+2 train start. STILL-OPEN because the plan's rationale text is sloppy and risks future coders breaking the assumption.

**Paired ΔIC under mixed seed counts (FINGNN-B-01)**: This is the most important new finding. The verification spot from the prompt was on-target. The 30-seed-from-100-pool design creates seed overlap (good for pairing) but unequal n (bad for pairing). The cleanest fix: paired ΔIC uses GAT_first30 vs MLP_first30 (clean paired at seed×fold×day), and separately report 100-seed mean ± CI for GAT/SAGE-Mean in Table 1 as the headline distribution. This needs to be locked into §1.4(c).

**Bootstrap terminology (FINGNN-B-02)**: Real bug. The plan text says "stationary block bootstrap" but the cited implementation is Künsch fixed-length, per the code's own comment. Either fix the plan text or change the implementation. Reviewers WILL catch this.

**DSR kurtosis convention (FINGNN-B-03)**: The plan literally writes "kurt = excess-kurtosis" AND uses formula "(kurt - 1)/4". These two are mutually inconsistent with Bailey-Lopez de Prado 2014. If a coder writes scipy.stats.kurtosis(returns) (default fisher=True → excess) and plugs into the plan's formula verbatim, the kurtosis correction term is WRONG SIGN for any near-normal return series. Easy fix: pick one convention and assert it in code.

**NW-HAC lag (FINGNN-B-06)**: Auto-lag formula gives 3 for T=63, well below the 21-day overlap autocorrelation. Override to max(auto, 21).

## Bottom line for next milestone

The plan is ready to proceed to Touchpoint 2 (code review on run_storya_multiseed.py) AFTER the following items are locked in §1.4(a), §1.4(c), §1.4(b), and §1.8:

1. **(MAJOR)** Lock paired-test seed protocol: paired ΔIC uses first-30 subset; 100-seed distribution reported separately (FINGNN-B-01).
2. **(MAJOR)** Lock bootstrap terminology: either change "stationary" → "fixed-length block (Künsch 1989)" in §1.4(b) to match the existing implementation, OR commit to implementing true stationary bootstrap in compute_dsr.py (FINGNN-B-02).
3. **(CONCERN)** Lock DSR kurtosis convention with assertion (FINGNN-B-03).
4. **(CONCERN)** Lock NW-HAC lag at max(auto, 21) (FINGNN-B-06).

Items A-04 (DSR formula skew/kurt + ledger), A-05 (fold rationale text), A-07 (LSTM/Mamba HP justification), A-09 (paired test ambiguity), A-10 (LSTM-on-21x13 baseline) overlap with these and the new findings — addressing 1-4 above closes most of them.

Items §1.10 smoke gate, §1.7 HP lock for GNN/MLP models (production defaults verified), §1.5 ablation matrix, §1.2 point-in-time news edge are all in good shape.

H博士 decision points before plan v3 (or amendments-in-place):
- Lock kurtosis convention (raw γ_4 vs excess γ_4 - 3)
- Lock bootstrap variant (Künsch fixed vs Politis-Romano stationary)
- Lock paired-test seed scheme (first-30 paired vs full-N unpaired)
- Decide whether to add A6 LSTM/GRU baseline to §1.5 (+150 cells, +8h A100)
- Decide whether to pre-commit drop priority for §1.10 budget-overrun fallback

None of these require a v3 rewrite — they can be amendments to v2 §1.4/§1.5/§1.10 with a one-line entry in Section 7 Decision Log. After amendments, proceed to write run_storya_multiseed.py and trigger Touchpoint 2.
