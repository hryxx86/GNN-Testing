# artifacts/storya_v21_cost/ — Cost-口径 (gross/net) crosswalk

Net-of-cost economic-sensitivity layer for the D-RERUN-12F confirmatory tuned ladder. **DESCRIPTIVE
only — IC remains the sole confirmatory metric** (no BH-FDR family on Sharpe). Produced by
`compute_cost_confirmatory.py` from the per-cell `Sharpe_net_*bps` columns already stored in the
confirmatory `results.csv` (no re-run). Net口径 headline = 10bps (L1-one-way:
net = gross − turnover_L1 × bps/10000). See `docs/analysis.md` 2026-06-21-a §5.

## 当前内容

- `cost_headline_crosswalk.csv` — one row per pre-registered pair: gross ΔIC + BH (copied verbatim
  from `../storya_v21_family1/family1_dm_hln.csv`) next to net ΔSharpe@10bps + CI + LOFO stability +
  `cost_sensitive` flag (gross-IC vs net-Sharpe@10bps sign disagreement).
- `cost_pairwise_dsharpe.csv` — the 20-test family fold-level ΔSharpe at {0,10,30}bps + block-bootstrap
  CI + LOFO min/max + sign-stability + drop-Q2-2025.
- `cost_pairwise_folddeltas.csv` — the 12 per-fold ΔSharpe values per pair × cost (provenance for any
  per-fold statement; added per Codex T3 CODEX-A-07).
- `cost_ladder_by_arm.csv` — per (universe, arm, cost_bps) net Sharpe mean + median + bootstrap CI +
  drop-Q2-2025 + turnover + max-|gross Sharpe| cell (heavy-tail flag); 0–30bps ladder.
- `cost_fc_dsharpe.csv` — 6 Family-2 FC contrasts' net ΔSharpe (descriptive).
- `cost_ledger.json` — metadata: framing, cost convention, EXCLUDE policy, cost-sensitive pairs.
- `cost_summary.md` — human-readable summary (cost-sensitive findings + crosswalk + per-arm ladder).

## 关键文件速查

- **The deliverable**: `cost_headline_crosswalk.csv` — every headline claim's gross/net dual口径.
- **The one cost-sensitive finding**: C L3-L2 (IC "news hurts" does NOT reproduce at net; net CI
  straddles 0) — see crosswalk + `cost_pairwise_folddeltas.csv` (6 pos / 6 neg folds).
- Reviewers: `../reviews/2026-06-21_codex_code_cost_A.md` (T2), `../reviews/2026-06-21_codex_results_cost_A.md` (T3).

## 变更日志

- 2026-06-21: dir created; cost-口径 crosswalk on confirmatory tuned ladder（→ progress: 2026-06-21-b）
