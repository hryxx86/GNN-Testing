---
reviewer: codex
touchpoint: plan
round: A
target_plan: /Users/heruixi/.claude/plans/handoff-session-ranking-swirling-lemur.md
findings:
  - id: CODEX-A-01
    severity: CRITICAL
    category: data-leakage
    claim: "The news co-occurrence edge is not specified as point-in-time and can front-run the prediction timestamp."
    evidence: "§1.2: \"Build edge: for each trading day t, edge weight (i, j) = count of articles in past 5d mentioning both ticker i and j\""
    suggested_fix: "Define the exact portfolio-formation timestamp and include only articles with publication timestamps strictly before it, preferably lagged one full trading day; rebuild the edge store with point-in-time ticker membership, deduplication, and an explicit ban on articles inside the 21d label window."
    status: OPEN
    resolution_notes: "v3 plan still must add explicit PIT spec to E3 news-as-edge experiment (publication timestamp ≤ T-1, exclude articles in 21d label window). Pending Codex Round C verification."
  - id: CODEX-A-02
    severity: CRITICAL
    category: statistics
    claim: "The adaptive 30-to-100 seed extension rule selects on observed positive pilot performance and biases the final model estimates."
    evidence: "§1.1: \"extend to 100 seeds (Phase B) iff ALL true: (a) 30-seed mean IC > 0.020 ... (b) 30-seed CV > 30%\""
    suggested_fix: "Use a fixed seed count for all primary models, or make extension depend only on blinded variance/compute rather than observed mean IC; if a two-stage design is retained, report pilot and extension estimates separately and use a selection-adjusted estimator/test."
    status: FIXED
    resolution_notes: "RESOLVED-BY-V3-METHODOLOGY-CHANGE (H博士 directive 2026-05-26 evening): v3 plan drops adaptive extension entirely. All 5 models use fixed canonical 10 seeds [86,123,456,789,1024,2024,7,34,99,2026]. No conditioning on observed pilot means. Winner's-curse inflation source eliminated by construction. Disposition source: H博士 directive, NOT Codex Round B confirmation. Round C must verify v3 design indeed eliminates this concern."
  - id: CODEX-A-03
    severity: CRITICAL
    category: statistics
    claim: "The proposed PBO procedure is not Bailey et al. CSCV because it splits random seeds rather than time observations."
    evidence: "§1.4(b): \"Split 100 seeds into 2 halves K times\" and \"rank by val IC on half A, check if top performers remain top on half B\""
    suggested_fix: "Construct an N-configuration by T-date performance matrix and run CSCV over even time blocks, selecting the best configuration in-sample and ranking it out-of-sample; use seeds only as configurations if a trained seed is selectable, not as the resampling axis."
    status: FIXED
    resolution_notes: "RESOLVED-BY-V3-METHODOLOGY-CHANGE (H博士 directive 2026-05-26 evening): v3 plan drops PBO entirely. Cherry-pick defense now via Hansen SPA (Hansen 2005) which uses bootstrap over per-period loss series of all M candidates against benchmark — this is the canonical multi-comparison test that Codex's suggested CSCV redesign was approximating. Disposition source: H博士 directive. Round C must verify SPA implementation handles the candidate family correctly."
  - id: CODEX-A-04
    severity: MAJOR
    category: statistics
    claim: "The DSR formula and trial count are underspecified and do not match the Bailey-Lopez de Prado DSR object."
    evidence: "§1.4(a): \"ONC on per-cell Sharpe correlation matrix → N_eff\" and \"DSR per cell = Φ((SR_cell − E[max_SR_null]) × √(T−1) / σ_corrected)\""
    suggested_fix: "Define the selected strategy/claim-level Sharpe series, T, skewness, kurtosis, cross-trial Sharpe variance, and total number of attempted independent trials; compute DSR at the selected headline level, not per model-seed-fold cell."
    status: FIXED
    resolution_notes: "RESOLVED-BY-V3-METHODOLOGY-CHANGE (H博士 directive 2026-05-26 evening): v3 plan drops DSR entirely. Cherry-pick defense delegated to Hansen SPA (academic standard, bootstrap-based, no parametric Sharpe distribution assumption). DSR underspecification concern moot. H博士 explicit decision: \"DSR 学术界不查 (HATS/RSR/FinGAT/MASTER/MDGNN/FinMamba 全部不做), industry standard only. Hansen SPA is the academic-standard cherry-pick defense\". Disposition source: H博士 directive."
  - id: CODEX-A-05
    severity: MAJOR
    category: data-leakage
    claim: "The walk-forward design does not specify a purge/embargo for overlapping 21d labels or preprocessing boundaries."
    evidence: "§1.1: \"Folds: existing 5-fold (WALK_FORWARD_FOLDS)\" and \"Horizon: 21d\""
    suggested_fix: "Write exact train/validation/test date boundaries and impose at least a 21-trading-day embargo between adjacent splits; document that all rolling features, correlation graphs, winsorization, scaling, and label construction are fit only on information available up to each prediction date."
    status: OPEN
    resolution_notes: "v3 plan still must add explicit 21-day embargo spec between fold boundaries + temporal contract for rolling features/correlation graphs/winsor/scaling/labels. Pending E1 spec update + Round C verification. archived/scripts/run_horizon_ablation.py:316 currently does 'C1 FIX: Purge last HORIZON days (adaptive to current horizon)' which partially addresses this — but plan must document the contract explicitly."
  - id: CODEX-A-06
    severity: MAJOR
    category: prior-art
    claim: "The novelty claim is too strong without a formal related-work table separating relation-type, horizon, feature-universe, and regime-conditioning studies."
    evidence: "§1.4: \"None of the 8 surveyed GNN-finance papers (HATS, RSR, FinGAT, MASTER, MDGNN, FinMamba, OmniGNN, THGNN) include DSR / PBO / SPA / CSCV / pre-registration.\""
    suggested_fix: "Add a literature matrix covering HATS, RSR, FinGAT, MASTER, SAMBA, FinMamba, OmniGNN, When Alpha Breaks, and 2024-2025 surveys; frame novelty as a point-in-time conditional evaluation by horizon and feature universe only if that exact gap survives verification."
    status: OPEN
    resolution_notes: "v3 plan still must add literature matrix with axes (horizon × feature_universe × graph_relation × regime × PIT × seed_count × overfit_diagnostic) populated by 8-12 published papers. Novelty framing adjusted: v3 drops 'DSR/PBO gap' as novelty (since DSR/PBO themselves dropped); shifts to 'conditional analysis under strict point-in-time multi-seed evaluation' as the claim, which needs literature matrix to verify exact gap. Pending plan section + Round C."
  - id: CODEX-A-07
    severity: MAJOR
    category: reproducibility
    claim: "The plan adds new LSTM and Mamba-SAGE baselines without a pre-registered hyperparameter/search protocol."
    evidence: "§1.1: \"Models: GAT_price, SAGE-Mean_price, MLP_price, LSTM_price, Mamba-SAGE prefix\" and §1.5: \"Vanilla Mamba (mamba-ssm) as per-stock temporal encoder\""
    suggested_fix: "Pre-register architectures, parameter budgets, optimizer settings, early-stopping rules, and validation-only hyperparameter grids for every model; count all tried hyperparameter settings in the multiple-testing ledger."
    status: OPEN
    resolution_notes: "v3 plan still must add pre-registration table for LSTM_price (new to codebase) + LightGBM_price (new to E1 model list). Mamba pre-reg deferred since E5 is OPTIONAL in v3. Pending plan §1.1 spec update + Round C verification."
  - id: CODEX-A-08
    severity: MAJOR
    category: correctness
    claim: "Mamba-SAGE as an insurance positive anchor can contradict the Story A narrative unless the graph contribution is isolated."
    evidence: "§1.5: \"Mamba-SAGE prefix (INSURANCE — Story B element)\" and \"If Mamba prefix improves over MLP/SAGE, paper has positive anchor.\""
    suggested_fix: "Pre-register ablations: Mamba-only, SAGE-only with the same 13 features, Mamba-SAGE, Mamba plus identity graph, and Mamba plus shuffled/permuted graph; define in advance how each outcome maps to the paper claim."
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "v3 plan reclassifies Mamba-SAGE as OPTIONAL (skip if 8-week deadline tight). If E5 runs, v2 pre-registered A1-A5 ablation matrix (Mamba-only, SAGE-only, Mamba-SAGE, Mamba+identity graph, Mamba+shuffled graph) carries forward unchanged. If E5 skipped, finding is moot. Disposition: ACCEPTED-AS-CONCERN with conditional plan."
  - id: CODEX-A-09
    severity: MAJOR
    category: statistics
    claim: "The cross-pick detection logic does not define the primary null hypotheses, paired effect sizes, or the full multiple-testing family."
    evidence: "§1.4: \"If DSR > 0.95 AND PBO < 0.5 AND bootstrap CI excludes 0 → headline real, paper Claim 1\""
    suggested_fix: "Define primary tests as paired date-level differences, e.g. ΔIC(GNN, MLP) and ΔSharpe(GNN, MLP), with block-bootstrap or permutation CIs; maintain a ledger of all model, seed, fold, hyperparameter, edge, horizon, and prior exploratory trials and adjust claims accordingly."
    status: FIXED
    resolution_notes: "RESOLVED-BY-V3-METHODOLOGY-CHANGE (H博士 directive 2026-05-26 evening): v3 plan defines primary tests as standard DM (Diebold-Mariano 1995) + HLN (Harvey-Leybourne-Newbold 1997) small-sample correction on per-day paired ΔIC(GNN, MLP) and ΔSharpe(GNN, MLP), with block-bootstrap CI. Multi-comparison family covered by Hansen SPA over all candidate (model, seed) pairs vs LightGBM benchmark. Trial ledger (model × seed × fold × universe) explicit in pre-registration. Disposition source: H博士 directive. Round C must verify DM/HLN/SPA primary-test specification."
  - id: CODEX-A-10
    severity: CONCERN
    category: correctness
    claim: "The proposed vanilla Mamba setup is outside the validated regime implied by recent stock-Mamba papers and may be an underpowered or confounded baseline."
    evidence: "§1.5: \"Input: (501, 21, 13) — 21d lookback × 13 features\" and \"Vanilla Mamba (mamba-ssm) as per-stock temporal encoder\""
    suggested_fix: "Treat Mamba-SAGE as exploratory unless Mamba-only, GRU/LSTM/TCN, and capacity-matched baselines are included; do not claim SAMBA/FinMamba support without matching the data regime and architecture."
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "v3 plan accepts this concern: Mamba-SAGE is OPTIONAL exploratory experiment (E5), not core. If E5 runs, paper claim is conditional ('Mamba-SAGE shows X under these specific 13-dim 21d settings'), not SAMBA/FinMamba SOTA replication. If E5 skipped, moot. Disposition source: H博士 directive."
  - id: CODEX-A-11
    severity: CONCERN
    category: reproducibility
    claim: "The compute plan is optimistic for 750-2500 training cells plus new model implementation under Colab-style runtime limits."
    evidence: "§1.1: \"Compute: Phase A ~20h A100; Phase B 0-50h A100\" and Context: \"8-9 week realistic timeline.\""
    suggested_fix: "Run a timed 1-seed x 1-fold smoke benchmark for each model before committing the design; add checkpoint/resume, per-cell runtime estimates, package-install risk for mamba-ssm, and a reduced fixed-N fallback."
    status: FIXED
    resolution_notes: "RESOLVED-BY-V3-METHODOLOGY-CHANGE (H博士 directive 2026-05-26 evening): v3 plan reduces total cells from 1450 → 500 (5 models × 10 seeds × 5 folds × 2 universes for E1; ~350 additional cells across E3 news-edge + E4 HGT 21d + optional E5 Mamba). Revised A100 estimate: ~50-60h total across E1-E5. Smoke benchmark per §1.10 still required before full launch. mamba-ssm package install risk avoided if E5 skipped. Checkpoint/resume still must be added per Codex's structural recommendation. Disposition source: H博士 directive (cell-count reduction) + Codex recommendation (smoke + checkpoint still required)."
summary:
  critical: 3
  major: 6
  concern: 2
  fixed_before_reply: 0
overall_verdict: BLOCK-EXECUTION
---

# Rule 9 Plan Review - Story A

Verdict: BLOCK-EXECUTION until the adaptive sampling, PBO/DSR definitions, news edge point-in-time rules, and 21d split embargo are corrected. The story is plausible after fixes, but the current plan would produce difficult-to-defend inference even if the code ran cleanly.

## Q1. Adaptive 30-to-100 Extension Rule

The rule is not just a compute-saving device; it changes the estimand. Let the 30-seed pilot mean be \(\hat{\mu}_{30} \sim N(\mu, \sigma^2/30)\). If extension occurs when \(\hat{\mu}_{30} > c\), then:

\[
E[\hat{\mu}_{30} \mid \hat{\mu}_{30} > c] =
\mu + \frac{\sigma}{\sqrt{30}}
\frac{\phi(\alpha)}{1-\Phi(\alpha)}, \quad
\alpha = \frac{c-\mu}{\sigma/\sqrt{30}}.
\]

The positive term is the winner's-curse inflation from truncating on a favorable pilot. If the final 100-seed estimate includes the same selected pilot plus 70 independent extension seeds, then:

\[
E[\hat{\mu}_{100} \mid \text{extended}] =
\mu + 0.3\frac{\sigma}{\sqrt{30}}
\frac{\phi(\alpha)}{1-\Phi(\alpha)}.
\]

So the final estimate is still upward biased. The CV condition does not fix this; with mean IC forced above 0.020, requiring CV > 30% preferentially extends high-variance "promising but uncertain" models and leaves low/negative pilot outcomes at 30 seeds. Relaxing to 0.015 or tightening above 0.020 only moves the truncation point. The fix is to avoid conditioning extension on the observed mean for confirmatory results. Use fixed N across primary models, or base extension only on blinded variance/runtime.

## Q2. DSR With Adaptive Sample Sizes

Bailey and López de Prado's Deflated Sharpe Ratio corrects an observed Sharpe for selection across multiple trials and non-normal returns; it needs the selected strategy's return series length, skewness, kurtosis, cross-trial Sharpe variance, and number of independent trials. The plan's "DSR per cell" is not aligned with that object. A model-seed-fold cell is a training replicate, not necessarily a selectable trading strategy, and fold-specific Sharpe values may have different dates and T.

ONC-based \(N_{\mathrm{eff}}\) can be a reasonable estimator only after defining a common matrix of attempted strategy return series over the same OOS dates. Adaptive 30 vs 100 seed counts make this worse: the number of attempted trials differs by model because the pilot result determined whether more seeds existed. If the paper selects a model after seeing all results, DSR must account for all attempted models, seeds, hyperparameters, edge variants, and prior exploratory trials that informed the claim, not merely "5 models x 100 seeds x 5 folds."

The plan's expected maximum formula is also incomplete relative to the 2014 paper. The Bailey-Lopez de Prado implementation computes \(E[\max SR]\) as \(\mu_{SR} + \sigma_{SR}[(1-\gamma)Z^{-1}(1-1/N)+\gamma Z^{-1}(1-1/(Ne))]\), and the DSR denominator includes the selected return distribution's skewness and kurtosis. The plan's `sigma_corrected` placeholder must be made explicit before execution.

## Q3. PBO With Validation IC In A Walk-Forward Setup

The current PBO description should not be used. Bailey, Borwein, López de Prado, and Zhu define PBO through in-sample selection and out-of-sample ranking across time-split performance observations. CSCV splits the observation axis into symmetric time blocks, not the random-seed axis.

For this project, define one candidate configuration as the unit that could have been selected before test evaluation: e.g. model class plus hyperparameters plus seed if seeds are selectable, or model class plus hyperparameters if seed-averaging is pre-committed. For each candidate, collect a date-level validation/backtest performance series. Split dates into even contiguous blocks, select the top candidate on the in-sample block set, and rank that same candidate on the complementary out-of-sample block set. With 5 walk-forward folds, folds alone are too few and odd-numbered for classical CSCV; use date blocks inside the validation/OOS periods or redesign the fold structure.

The plan must also decide whether val IC is per `(model, seed)`, per `(model, seed, fold)`, or a date-level series aggregated across folds. There is no canonical val IC under the current text.

## Q4. Mamba On 21d x 13 Features

This should be treated as exploratory. SAMBA, FinMamba, and HIGSTM justify Mamba-style sequence modeling through richer temporal/graph structures, market-aware modules, multi-level temporal recall, or adaptive graph construction. The proposed `21 x 13` vanilla Mamba input is short and low-dimensional; the linear-time long-sequence advantage is unlikely to matter, and the model may underperform or win for capacity/optimization reasons unrelated to the graph.

Minimum ablations: Mamba-only, Mamba-SAGE, LSTM/GRU/TCN with matched parameter budget, SAGE on the same 13 features, identity graph, and shuffled graph. Without these, a Mamba-SAGE win is not interpretable.

## Q5. Narrative Consistency If Mamba-SAGE Helps

The narrative survives only if the paper's claim is "GNN helpfulness is conditional," not "GNNs do not help." If Mamba-SAGE improves while GAT/SAGE do not, the most likely interpretations are:

- temporal encoder strength is the main effect;
- graph propagation helps only after a better per-stock encoder;
- the 13-feature universe, not the graph, explains the gain;
- the hybrid model is an architecture search result and should be counted in multiple testing.

The word "insurance" is a warning sign. Pre-register the narrative mapping before running it. For example: if Mamba-only equals Mamba-SAGE, the conclusion is no graph benefit; if Mamba-SAGE beats Mamba-only and shuffled graph, the conclusion is conditional graph benefit under a stronger temporal encoder.

## Q6. Prior-Art Threat

UNCERTAIN - requires literature verification before submission. I did not find, from targeted search and known papers, an exact prior paper that performs a PIT-safe conditional study of GNN helpfulness by horizon x feature universe with DSR/PBO/pre-registration. That is a possible contribution.

But the novelty is not safe as currently phrased. HATS studies the effect of relation types on stock prediction; RSR explicitly targets relational stock ranking; FinGAT learns latent stock/sector interactions for top-K recommendation; MASTER studies dynamic feature effectiveness and cross-time stock correlations; SAMBA, FinMamba, and HIGSTM are Mamba/graph or market-aware temporal stock models; OmniGNN focuses on multi-relational GNN robustness during macro shocks; When Alpha Breaks studies horizon/regime failure for cross-sectional rankers. The contribution should be narrowed to the exact missing combination: strict point-in-time cross-sectional ranking, controlled horizon/feature-universe interaction, paired baselines, and overfit diagnostics.

Also, the surveyed-paper claim should not stop at "0/8 include DSR/PBO." That is useful, but it is a methodology claim, not the novelty of "When Do GNNs Help." Add a related-work table with axes: horizon varied, feature universe varied, graph relation varied, regime evaluated, seed count, PIT evaluation, and overfit diagnostic.

## Q7. Compute Budget And Implementation Risk

Phase A is 5 models x 30 seeds x 5 folds = 750 training cells. Full extension is up to 2500 cells before news-edge, HGT, Mamba ablations, HATS reproduction, and hyperparameter search. A 20-hour A100 estimate for Phase A implies about 1.6 minutes per training cell including data loading, graph construction, validation, logging, and failure recovery. That is not credible until measured, especially with LSTM and Mamba-SAGE not yet established in the codebase.

Colab Pro-style runtime limits make this a scheduling and reproducibility risk. Add checkpoint/resume, per-cell manifests, deterministic seed replay, package pinning for `mamba-ssm`, and a fixed-N fallback. The smoke tests in §6.1 should be promoted from verification to a gating milestone with measured wall-clock estimates before any large run.

## Additional Methodology Notes

Data leakage: Price features, correlation graphs, news edges, winsorization, and scaling need one written temporal contract. For a 21d label, features at date \(t\) must use data available no later than the formation timestamp, and labels must cover only \(t+1\) through \(t+21\) or the explicitly chosen equivalent. Adjacent folds require purge/embargo because 21d forward returns overlap. If regime/HMM analysis is reintroduced, use filtered state probabilities only; smoothed HMM states would look into the future.

Statistics: DSR/PBO do not replace the primary model-comparison test. The core null should be paired and concrete, e.g. \(H_0: E[\mathrm{IC}_{\mathrm{GNN},t}-\mathrm{IC}_{\mathrm{MLP},t}] = 0\). Report paired ΔIC, ΔSharpe, turnover/cost-adjusted metrics, and block-bootstrap CIs. Use the overfit diagnostics as secondary evidence about selection bias, not as the sole gate for "headline real."

Reproducibility: The fixed seed list is good, but it is not enough. The plan needs exact data split boundaries, package versions, model hyperparameters, early-stopping criteria, validation metrics, and a manifest format that lets every row of `phase_a_results.csv` be traced to git commit, seed, fold, model config, data snapshot, and runtime.

## Sources Checked

- Bailey and López de Prado, "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality," Journal of Portfolio Management 2014 / SSRN: https://ssrn.com/abstract=2460551
- Bailey, Borwein, López de Prado, and Zhu, "The Probability of Backtest Overfitting," Journal of Computational Finance, DOI 10.21314/JCF.2016.322: https://scholarworks.wmich.edu/math_pubs/42/
- HATS, arXiv:1908.07999: https://arxiv.org/abs/1908.07999
- RSR, "Temporal Relational Ranking for Stock Prediction," arXiv:1809.09441: https://arxiv.org/abs/1809.09441
- FinGAT, arXiv:2106.10159: https://arxiv.org/abs/2106.10159
- MASTER, arXiv:2312.15235: https://arxiv.org/abs/2312.15235
- SAMBA, "Mamba Meets Financial Markets," arXiv:2410.03707: https://arxiv.org/abs/2410.03707
- FinMamba, arXiv:2502.06707: https://arxiv.org/abs/2502.06707
- HIGSTM, arXiv:2503.11387: https://arxiv.org/abs/2503.11387
- OmniGNN, "Structure Over Signal," arXiv:2510.10775: https://arxiv.org/abs/2510.10775
- "When Alpha Breaks," arXiv:2603.13252: https://arxiv.org/abs/2603.13252
- Das et al., "Integrating sentiment analysis with graph neural networks for enhanced stock prediction: A comprehensive survey," Decision Analytics Journal 2024: https://doi.org/10.1016/j.dajour.2024.100417
- Lin and Marques, "Stock market prediction using artificial intelligence: A systematic review of systematic reviews," Social Sciences & Humanities Open 2024: https://doi.org/10.1016/j.ssaho.2024.100864

---

## v3 Disposition Update (2026-05-26 evening, H博士 directive)

> **IMPORTANT — DISPOSITION SOURCE**: The disposition statuses for the 11 findings above (modified in YAML frontmatter on 2026-05-26 evening) reflect **H博士 strategy-simplification directive**, NOT a Codex Round B confirmation. Codex Round B was NEVER run on v2; instead, after H博士 evening discussion, the plan moved directly to v3 with different methodology (drop adaptive 100-seed + DSR + PBO; adopt fixed canonical 10 seeds + Hansen SPA + DM/HLN + transaction cost ladder). **Codex Round C must be triggered to verify these v3 dispositions** per Rule 9 cross-round diffing schema (`.claude/rules/docs.md` §6).

### Summary of v3 dispositions

| Finding | Severity | v3 disposition | Mechanism |
|---------|----------|----------------|-----------|
| A-01 | CRITICAL | **OPEN** | News edge PIT spec still required in v3 E3 |
| A-02 | CRITICAL | **FIXED** | v3 fixed 10 seeds, no adaptive extension |
| A-03 | CRITICAL | **FIXED** | v3 drops PBO; SPA replaces |
| A-04 | MAJOR | **FIXED** | v3 drops DSR; SPA replaces |
| A-05 | MAJOR | **OPEN** | 21d purge/embargo still required in v3 E1 |
| A-06 | MAJOR | **OPEN** | Literature matrix still required in v3 |
| A-07 | MAJOR | **OPEN** | LSTM/LightGBM hparam pre-reg still required in v3 |
| A-08 | MAJOR | **ACCEPTED-AS-CONCERN** | Mamba reclassified OPTIONAL in v3 |
| A-09 | MAJOR | **FIXED** | v3 primary tests now standard DM+HLN+SPA paired |
| A-10 | CONCERN | **ACCEPTED-AS-CONCERN** | Mamba reclassified OPTIONAL in v3 |
| A-11 | CONCERN | **FIXED** | v3 cells reduced 1450 → 500; smoke benchmark gated |

**Net resolution**: 5 FIXED (by v3 methodology change) + 4 OPEN (need explicit fixes in v3 plan sections) + 2 ACCEPTED-AS-CONCERN (Mamba OPTIONAL).

### v3 design changes (summary)

| Item | v2 (Codex Round A response, BLOCK-EXECUTION) | v3 (H博士 simplification) |
|------|----------------------------------------------|---------------------------|
| Seeds per model | 100 with adaptive 30→100 extension rule | Fixed canonical 10 `[86,123,456,789,1024,2024,7,34,99,2026]` |
| Multi-comparison defense | DSR (Lopez de Prado 2014) + PBO (Bailey-Borwein-Lopez de Prado-Zhu 2017) | **Hansen SPA** (Hansen 2005) |
| Pairwise model comparison | Implicit via DSR/PBO | **DM (Diebold-Mariano 1995) + HLN (Harvey-Leybourne-Newbold 1997)** small-sample correction |
| Cost analysis | Single Sharpe_net | **Transaction cost ladder** {0, 5, 10, 15, 20, 30 bps} + Net Sharpe per level |
| Feature universe | Universe B (10-dim hc) only | **Universe B (10-dim hc) + Universe C (51-dim Plan AAA top-15 members) paired anchor** |
| Mamba-SAGE | "insurance" 30 seeds in core plan | **OPTIONAL** (E5, skip if 8-week tight) |
| HATS reproduction | STRETCH | **STRETCH** (unchanged) |
| Total cells E1 | 1450 (100×SAGE + 100×GAT + 30×3 others) × 5 folds | **500** (5 models × 10 seeds × 5 folds × 2 universes) |
| A100 estimate | 30-130h | **~50-60h total** across E1-E5 |
| Timeline | 8-9 weeks | **8 weeks** |
| Cherry-pick concern (H博士 original "find 10 lucky seeds of 100") | DSR + PBO + adaptive | **None: v3 explicitly does not cherry-pick; reports full distribution; SPA catches multi-comparison inflation** |

### Why H博士 directed the simplification

Excerpt from 2026-05-26 evening discussion (reconstructed from session memory):

1. **H博士 question**: "DSR 是标准做法吗，还是说别人不做"
2. **Claude honesty**: DSR is industry standard (Lopez de Prado followers, JFDS), NOT academic GNN-finance standard (0/8 surveyed GNN-finance papers use DSR)
3. **H博士 question**: "我们没做但应该加的是什么"
4. **Claude answer**: Diebold-Mariano + Modified DM (HLN) for pairwise model comparison; transaction cost ladder + net Sharpe; Hansen SPA for multi-model cherry-pick defense (academic equivalent of DSR, bootstrap-based, used in econometrics forecasting/trading-rule literature with 5000+ Hansen 2005 citations)
5. **H博士 question**: "最小必加和 Hansen SPA 是否可以检测出 cherry-pick"
6. **Claude correction**: DM/HLN/cost/Net-Sharpe do NOT catch cherry-pick (they assume pre-selected model); **Hansen SPA DOES** (bootstrap-based, designed for "best of M models" test, same mechanism as DSR but academic-standard rather than industry)
7. **H博士 question**: "假如说我总共就跑了 10 seeds, SPA 是怎么做的"
8. **Claude answer**: SPA works with M ≥ 2; with 10 seeds × 5 models = 50 candidates, SPA still gives valid multi-comparison p-value; just lower power than M=500. Implementation: `arch.bootstrap.SPA`, ~5 lines, ~30-60s compute
9. **H博士 directive**: lock as **10 seeds + SPA + DM + cost ladder + Universe B/C paired**; drop DSR, PBO, adaptive, Romano-Wolf, Full MCS, PIMP B=100+ as overkill

### What this disposition update does NOT do (honesty constraint per Rule 9)

- **Does NOT confirm Codex Round B verdict**: Round B was never run on v2. v3 supersedes v2 before Round B trigger.
- **Does NOT replace Round C requirement**: v3 plan with all sections aligned (including A-01/A-05/A-06/A-07 explicit fixes) MUST be reviewed by Codex Round C before E1 launch per Rule 9.
- **Does NOT remove the A-01/A-05/A-06/A-07 explicit-fix requirements**: these still need plan section updates before Round C.
- **Does NOT close findings that v3 only partially addresses**: A-08 and A-10 are accepted-as-concern conditional on Mamba being OPTIONAL; if E5 is later promoted to core, those findings re-open.

### Next Codex review trigger

After plan v3 Sections 1.1-1.6 + 4-8 + 9 are aligned + prereg v3 written + A-01/A-05/A-06/A-07 explicit fixes added → trigger `/codex-plan-review .claude/plans/handoff-session-ranking-swirling-lemur.md` with prompt mentioning:
- "This is v3 plan; supersedes v2 adaptive 100-seed + DSR/PBO design"
- "v3 directly responds to Round A findings with new methodology (see Section 9 disposition log + this review file's v3 disposition update)"
- "Round C scope: (1) verify v3 design indeed resolves A-02/A-03/A-04/A-09/A-11; (2) verify v3's explicit fixes for A-01/A-05/A-06/A-07; (3) verify A-08/A-10 ACCEPTED-AS-CONCERN classification is appropriate given Mamba OPTIONAL status"
