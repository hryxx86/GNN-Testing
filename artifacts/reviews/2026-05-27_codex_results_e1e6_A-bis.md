---
reviewer: codex
touchpoint: results
round: A-bis
target_files:
  - artifacts/storya_e6_dm_spa/spa_results.csv
  - artifacts/storya_e6_dm_spa/dm_hln_results.csv
  - artifacts/storya_e6_dm_spa/bootstrap_ci.csv
  - artifacts/storya_e6_dm_spa/cost_ladder.csv
  - artifacts/storya_e6_dm_spa/multiple_testing_ledger.json
  - artifacts/storya_e6_dm_spa/summary.md
  - /Users/heruixi/Library/CloudStorage/GoogleDrive-hryxx86@gmail.com/我的云端硬盘/GNN测试/experiments/storya_e1_anchor/results.csv
findings:
  - id: CODEX-RR-E1E6-A-bis-01
    severity: OK
    category: statistics
    claim: "SPA/DM vs Bootstrap tension — statistically compatible if framed correctly"
    evidence: "Bootstrap CI excludes 0 for B neural (GAT 0.036 [0.018, 0.053]); SPA B vs LGB p=0.147. Different nulls (IC>0 vs IC>LGB); both can hold."
    fix: "Paper must keep distinction explicit: bootstrap supports IC above zero; SPA/DM do not establish benchmark dominance."
    status: NOTED
  - id: CODEX-RR-E1E6-A-bis-02
    severity: MAJOR
    category: results-interpretation
    claim: "Fold 4 uniformity (4 models ≈ 0.14 IC on Univ C) cannot be attributed to (a) regime vs (b) leak vs (c) label autocorr from E6 aggregates alone"
    evidence: "bootstrap_ci.csv only reports n_cells=50 aggregates; no fold-by-model or label-autocorr rows."
    fix: "Add fold-by-model table for Fold 4; add diagnostic separating feature provenance from label persistence; do NOT state Fold 4 proves tradeable regime from aggregates."
    status: PENDING
  - id: CODEX-RR-E1E6-A-bis-03
    severity: MAJOR
    category: results-interpretation
    claim: "Univ C 4-model convergence supports feature richness conditional on no shared leak"
    evidence: "GAT 0.043 / SAGE 0.048 / MLP 0.053 / LGB 0.047 — narrow ~0.010 range. Pairwise BH-FDR all fail. Shared leak (Plan AAA alpha158) would also produce this pattern."
    fix: "Phrase as 'all model families perform similarly on rich features' + explicit leak/provenance qualification; leak magnitude not estimable from these files."
    status: PENDING
  - id: CODEX-RR-E1E6-A-bis-04
    severity: CONCERN
    category: results-interpretation
    claim: "Univ C GAT Net Sharpe 3.08 unstable: std 9.94, CI [1.05, 6.19], not even top IC"
    evidence: "cost_ladder.csv: Sharpe_net_10bps mean 3.08, std 9.94, CI [1.05, 6.19]. bootstrap_ci.csv: GAT IC 0.043 < MLP 0.053. Likely Fold 4-driven."
    fix: "Do NOT headline as robust; present as unstable economic sensitivity; require LOFO/cell decomposition before attributing to persistent alpha."
    status: PENDING
  - id: CODEX-RR-E1E6-A-bis-05
    severity: CONCERN
    category: results-interpretation
    claim: "LightGBM Univ B 'failure' claim too strong — IC indistinguishable from 0, not confirmed negative"
    evidence: "B LGB IC 0.006 [-0.007, 0.019], gross Sharpe -0.33 [-1.56, 0.80], Net Sharpe @10bps -0.83 [-2.21, 0.40]. Only 30bps Sharpe is clearly negative."
    fix: "Qualify as 'Universe B economic underperformance after costs', NOT 'statistical IC failure'."
    status: PENDING
  - id: CODEX-RR-E1E6-A-bis-06
    severity: MAJOR
    category: reproducibility
    claim: "multi_testing_ledger.json does not enumerate historical exploratory families"
    evidence: "Ledger declares 400 E1 cells + SPA M=3/6 + BH q=0.05; mentions 'horizon ablation, Plan AAA, phase5 step3 subsets' qualitatively but does not enumerate counts."
    fix: "Enumerate omitted exploratory families with counts; state SPA controls only declared post-E1 confirmatory family, NOT broader research path."
    status: PENDING
  - id: CODEX-RR-E1E6-A-bis-07
    severity: OK
    category: statistics
    claim: "SPA bootstrap stability OK"
    evidence: "reps=10000, MCSE ~0.003-0.005, all SPA p-values 20+ MCSE above 0.05 boundary."
    status: NOTED
summary:
  critical: 0
  major: 3
  concern: 2
  fixed_before_reply: 0
  ok: 2
overall_verdict: MIXED/PROCEED-WITH-FIXES
---

# Codex Touchpoint 3 Round A-bis (REAL Codex retry)

Round A was finance-gnn-reviewer fallback (codex-rescue auto-fallback). H博士 requested
real Codex retry for independent confirmation. Two reviewers now CONVERGE on substantive
recommendations, with real Codex downgrading 2 severities (CRITICAL→OK on Check 1;
MAJOR→CONCERN on Check 4 and 5).

**Convergent recommendations across both reviewers**:
1. SPA non-rejection ≠ equivalence — paper §Results must separate bootstrap (absolute IC>0)
   and SPA/DM (paired benchmark dominance) framing
2. Fold 4 uniform 4-model strength needs fold-by-model table + leak/regime/autocorr diagnostic
3. Universe C 4-model convergence: feature-richness thesis CONDITIONAL on no shared leak;
   Plan AAA alpha158 provenance must be acknowledged in §Limitations
4. Univ C GAT Sharpe 3.08 must be reported as unstable secondary metric; LOFO/per-cell
   decomposition required before headlining
5. LGB Univ B "failure" should be scoped to economic/cost-aware, not statistical IC failure
6. multi_testing_ledger.json must enumerate historical exploratory families with counts
   to make SPA scope explicit ("post-E1 confirmatory only, not full research history")

No BLOCK-EXECUTION issue. Core story is usable; paper-level risks concentrated in Checks 2
and 3 (Universe C / Fold 4 interpretation can be read as either feature richness or shared
inflation unless explicitly qualified).
