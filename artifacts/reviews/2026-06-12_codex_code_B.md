---
reviewer: codex
touchpoint: code
round: B
target_files:
  - run_storya_v21_main12.py
target_plan: docs/protocol_v2_freeze.md
findings:
  - id: CODEX-B-01
    severity: CONCERN
    category: correctness
    claim: "train_gnn_per_day_edges has a silent ei=None skip for missing edge days; under current imported builders every used day is populated (no active bias), but the runner lacks an explicit coverage assert (E3 runner has one), so the apples-to-apples invariant for confirmatory pairs L3−L2/L5−L2 is implicit, not checked."
    evidence: "run_storya_v21_main12.py train_gnn_per_day_edges (ei=per_day_edges.get(d); if ei is None: continue) in train/val/test loops; build_fold_edges built fold_edges without coverage assertion; imported union_edges_per_day (run_storya_e3_news_edge.py:284-306) and union_edges_per_day_e4 (run_storya_e4_alpha.py:182-202) populate every requested day; the E3 runner explicitly guards this (run_storya_e3_news_edge.py:623-631)."
    suggested_fix: "After build_fold_edges, assert every int(d) in used_days exists in fold_edges[ec] for ec in {'corr_news','corr_sector_news'}; raise on missing. Makes the ei=None branch unreachable for confirmatory cells."
    status: FIXED
    resolution_notes: "Added coverage assert inside build_fold_edges: for each dynamic edge_config, missing = used_set - set(out[ec].keys()); assert not missing. Verified (no training): fold0 (399 used days) and fold11 (1089 used days) both report corr_news/corr_sector_news 0 days missing → COVERAGE OK; assert did NOT false-trigger. Verified by Claude (actually executed)."
summary:
  critical: 0
  major: 0
  concern: 1
  fixed_before_reply: 1
overall_verdict: PASS-WITH-CONCERNS
---

# Codex Review — Code (Touchpoint 2, Round B) — run_storya_v21_main12.py edge-arm increment

Rule 9 Touchpoint 2 Round B: correctness review of the edge-DAG ladder arms (L3/L4/L5/L5s) added to the v2.1 12-fold main-axis runner since Round A (`2026-06-12_codex_code_A.md`).

## Verdict: PASS-WITH-CONCERNS (0 CRITICAL + 0 MAJOR + 1 CONCERN — FIXED before reply)

## Codex confirmations (clean)

- **Apples-to-apples**: `train_gnn_per_day_edges` matches the imported `train_nn` on seed, model construction, Adam, ReduceLROnPlateau, grad_accum=32, clip=1.0, MSE, patience=15, best_state restore, and the same train/val/test masks — only model_name and the per-day edge lookup differ. Confirmatory pairs L3−L2 / L5−L2 are valid.
- **No edge-less-day bias** under current code: no-news days fall back to corr-only (L3) or corr+sector-only (L5/L5s) via the imported union builders — never None.
- **L4 routing correct**: `corr_sector` = static `corr_frozen ∪ sector` (`union_static_edges`) and uses the frozen-snapshot `train_nn` path (not per-day) — correct, since corr (frozen per fold) ∪ sector (static) is static within a fold.
- **C1 assert b sufficient**: fires inside the imported `build_per_day_news_edges` at construction (`assert window_max <= cutoff_np`, cutoff = `nyse_session_close_utc(t-1)`).
- **F import present** (the earlier F-NameError is fixed); static AST parse passed.

## Finding (CONCERN, FIXED + re-verified by Claude)

See frontmatter CODEX-B-01: added an explicit per-fold edge-coverage assert so the per-day trainer's `ei is None` skip is unreachable for confirmatory cells (defensive invariant mirroring the E3 runner's guard). Verified coverage holds at fold0 (399 used days) and fold11 (1089 used days): 0 days missing for both dynamic edge configs.

## Smoke (Claude, actually executed) — all 4 edge arms exit 0 (fold0/B/seed86)

| arm | edge_config | path | wall | IC | cell_id |
|---|---|---|---|---|---|
| L3  | corr_news (per-day)        | train_gnn_per_day_edges | 168s | -0.035 | 0360 |
| L4  | corr_sector (static)       | train_nn frozen          | 253s | -0.036 | 0480 |
| L5  | corr_sector_news (per-day) | train_gnn_per_day_edges | 281s | -0.030 | 0600 |
| L5s | corr_sector_news (per-day) | train_gnn_per_day_edges | 163s | -0.017 | 1080 |

C1 assert b confirmed firing at news construction (max PIT-eligible ts ≤ cutoffs). cell_id injective preserved.

## Out of scope (not in this file): L7 HATS (separate runner), Optuna tuning harness (not built), L6 complete-graph cost (known, not a bug).
