---
reviewer: codex
touchpoint: code
round: A
date: 2026-06-13
target_files:
  - run_storya_e3_news_edge.py:119-136    # cell_id_e3 assert generalized to len(WALK_FORWARD_FOLDS)
  - run_storya_e4_alpha.py:117-140        # cell_id_e4 stride 50→N_FOLDS*10 + assert + backfill
  - compute_e6_dm_spa.py:64-68,665-676    # N_FOLDS data-driven in main(); e1_n_cells dynamic
  - analyze_e1_lofo.py:25-46,181-225      # paths→local; FOLDS data-driven; §5 full→all folds; N_FOLDS propagation
note: "Reviews the propagation of the 5→12 fold window extension into 4 sibling/downstream scripts.
       Distinct from 2026-06-12_codex_code_anchorwindow_A.md (the anchor runner itself) and
       2026-06-12_codex_code_A.md (the v2.1 ladder runner, set aside)."
findings:
  - id: CODEX-T2-01
    severity: CONCERN
    category: data_leakage
    claim: "E3/E4 reuse an existing news snapshot cache (news_snapshots_cache.npz) without validating provenance; the runtime PIT assertion only fires during BUILD, not on cache LOAD, so a cache produced by a hypothetical pre-PIT-safe builder would be loaded unchecked for the new folds 5-11."
    evidence:
      - "run_storya_e3_news_edge.py:562 `if os.path.exists(args.news_cache):` → :573 loads snapshots, skipping build-time PIT assertion"
      - "run_storya_e4_alpha.py:341/346 same load path (shares E3 cache via NEWS_SNAPSHOT_CACHE line 103)"
    suggested_fix: "Codex suggested a metadata/version validation schema (PIT version, date hash, ticker hash, source mtime, lookback params). REJECTED as defensive over-engineering per Rule 9 主线聚焦."
    status: RESOLVED-BY-REBUILD
    resolution_notes: >
      Provenance independently verified: nyse_session_close_utc (the PIT cutoff, Codex D-03 fix) was
      INTRODUCED in commit 039eb36 (2026-05-27 00:30) — the ONLY commit touching this file, so there
      is no pre-PIT-safe version that could have produced a leaky cache. The cache file mtime is
      2026-05-27 22:24 (22h AFTER the builder commit) → necessarily built by the current PIT-safe
      builder. Coverage independently confirmed earlier: cache keys 1..1254 cover the full panel,
      0 missing snapshot-days across new folds 5-11, probe snapshots in 2023+2025 windows non-empty.
      DISPOSITION: rather than write Codex's heavy metadata-validation code, the stale cache is DELETED
      before the 12-fold launch and rebuilt fresh — the rebuild's per-day runtime PIT assertion
      (run_storya_e3_news_edge.py:258) validates every snapshot, closing the load-path concern with
      zero new code. E3 runs first (rebuilds), E4 runs after (loads fresh cache); sequential on single
      MPS device, no build race.
  - id: CODEX-T2-02
    severity: CONCERN
    category: hardcoded_assumption
    claim: "E3 _meta.json writes 'baseline_cells_reused_from: ... 50 cells', false for the 12-fold run (should be 120). Stale 50-cell hardcode in provenance metadata; does not affect training/results."
    evidence:
      - "run_storya_e3_news_edge.py:471 hardcoded string '... SAGE-Mean — 50 cells'"
    suggested_fix: "Derive as len(WALK_FORWARD_FOLDS) * len(CANONICAL_SEEDS) or drop the count."
    status: FIXED
    resolution_notes: >
      run_storya_e3_news_edge.py:471 changed to f-string
      f'... {len(WALK_FORWARD_FOLDS) * len(CANONICAL_SEEDS)} cells' = 120 for the 12-fold run.
      py_compile clean.
verified_PASS:
  - "cell_id injectivity + back-compat + backfill (E3 fold*10+seed injective→119; E4 stride-120 alias-free; resume tuple-keyed; E4 backfill 0 mismatch on real CSV)"
  - "compute_e6 N_FOLDS global mutation timing (set in main() before aggregate_e1(); 3 loaders read module global at call time; contiguity assert)"
  - "lofo N_FOLDS propagation (_e6.N_FOLDS = N_FOLDS before collect_per_day_ic_matrix(); function reads module global via lookup not closure capture, so `from X import func` binding does not defeat it)"
  - "DM/HLN/SPA T + NW-lag computed dynamically; no hardcoded 313 in execution path; block_size = horizon"
  - "no residual 400/313/50-cell assumption in EXECUTION path of the 4 files (only the now-fixed metadata string)"
summary:
  critical: 0
  major: 0
  concern: 2
  fixed_before_reply: 1
  resolved_by_rebuild: 1
overall_verdict: PASS_WITH_CONCERNS → cleared for launch (both concerns dispositioned)
---

# Codex Code Review — 12-fold propagation into E3/E4/e6_dm_spa/lofo, Round A

Rule 9 Touchpoint 2 review of the propagation of the 5→12 fold window extension into 4 scripts
(E3 news-edge runner, E4 alpha runner, e6 DM/SPA stats, lofo diagnostics). Verdict
**PASS_WITH_CONCERNS** — 0 CRITICAL, 0 MAJOR, 2 CONCERN.

Both CONCERNs dispositioned before launch:
- **T2-01** (news cache load-path has no provenance check): provenance independently verified safe
  (cache postdates the only/PIT-safe builder commit by 22h; full-panel coverage confirmed). Resolved
  the *load-path* concern the lightweight way — delete + rebuild so the runtime PIT assertion
  re-validates every snapshot — instead of Codex's heavy metadata-schema suggestion (rejected per
  Rule 9 anti-defensive-code).
- **T2-02** (stale "50 cells" metadata string): FIXED → dynamic f-string (120 for 12 folds).

Codex independently PASSed the 5 substantive correctness items (cell_id injectivity/backfill,
N_FOLDS global-mutation timing in e6, the lofo→imported-loader N_FOLDS propagation, dynamic DM/HLN/SPA
T, and no residual 5-fold execution-path assumptions). Cleared to rebuild the news cache, smoke
fold-5, and launch the E3→E4 12-fold re-run (+210 cells).
