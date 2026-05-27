---
reviewer: codex
touchpoint: code
round: A
target_files:
  - /Users/heruixi/Desktop/GNN-Testing/run_storya_e3_news_edge.py
findings:
  - id: CODEX-CR-E3RUN-A-01
    severity: MAJOR
    category: correctness
    claim: "n_news_articles_avg is hard-coded to 0 placeholder, not actually populated"
    evidence: "run_storya_e3_news_edge.py:593-608 + line 659: docstring claims stats include 'avg #articles per test day', but only avg_news_edges is computed; line 659 writes 'n_news_articles_avg': 0 with comment 'placeholder; refined-stat not tracked per cell'."
    suggested_fix: "Track eligible-article counts during build (return alongside snapshots), average over test_days per fold, write that value into every results row."
    status: FIXED
    resolution_notes: |
      FIXED. build_per_day_news_edges now returns (snapshots, article_counts) tuple.
      Article counts cached in news_snapshots_cache.npz as '__article_counts__' sidecar (int32 array, length=num_days, sentinel -1 for not-computed dates).
      Per fold: avg_news_articles = mean(article_counts[d] for d in test_days).
      Written to results.csv n_news_articles_avg column.
      Backward-compat: old caches without __article_counts__ fall through with article_counts={}, defaulting to 0 (forces cache rebuild for fresh runs).
  - id: CODEX-CR-E3RUN-A-02
    severity: MAJOR
    category: reproducibility
    claim: "Resume skip uses cell_id integer; stale manifests with old cell_id formula could skip wrong (fold, seed)"
    evidence: "load_manifest_done returns set of cell_id ints (lines 431-437); skip check is 'if cid in done' (lines 570-572). If cell_id formula ever changes, a stale manifest entry could collide with a new (fold, seed) cell."
    suggested_fix: "Resume skip should use canonical (fold, seed) tuple from the manifest row, not the formula-derived cell_id."
    status: FIXED
    resolution_notes: |
      FIXED. load_manifest_done now returns set[tuple[int, int]] of (fold, seed) pairs.
      Skip check changed to `if (fold_idx, seed) in done`.
      cell_id is still computed and logged for debug/output, but is no longer the resume identity.
      Same fix propagated to run_storya_e4_alpha.py with (edge_config, fold, seed) tuple identity.
  - id: CODEX-CR-E3RUN-A-03
    severity: CONCERN
    category: correctness
    claim: "If per-day edge tensor is missing in training loop, predictions silently remain zero for that day"
    evidence: "train_sage_per_day_edges line ~393 initializes preds to zeros; missing edge tensor leads to `continue` at lines ~397-399, leaving preds[d] = 0."
    suggested_fix: "Pre-validate every used (train/val/test) day has an edge tensor, or raise on missing key."
    status: FIXED
    resolution_notes: |
      FIXED via pre-validation in main() per-fold setup. After union_edges_per_day(),
      explicitly check that every day in concat(train_days, val_days, test_days) is a
      key in per_day_edges_cpu. If any day is missing, raise RuntimeError with first 5 missing
      day indices listed. This catches the invariant break at fold-setup time, not at
      training time, preventing silent zero predictions.
summary:
  critical: 0
  major: 2
  concern: 1
  fixed_before_reply: 3
overall_verdict: PROCEED-WITH-FIXES
---

# Codex Round A Review — run_storya_e3_news_edge.py

**Verdict: PROCEED-WITH-FIXES** (now FIXED: all 3 findings addressed).

PIT contract is INTACT — Codex explicitly verified the NYSE session_close cutoff, eligibility filter strictness, runtime PIT-violation assertion, and the worked example (article at 2024-05-31 21:00 UTC properly EXCLUDED for prediction_date 2024-06-03 with cutoff 20:00 UTC). Edge union dedupe + torch.long coercion correct. Cell ID injectivity check correct. Module import from E1 anchor confirmed safe (main() guarded).

## Fixes applied 2026-05-27

1. **CR-E3RUN-A-01 (MAJOR)**: `n_news_articles_avg` now actually tracks eligible-article counts (was hardcoded 0). `build_per_day_news_edges` returns a 2-tuple; cache `.npz` carries an `__article_counts__` sidecar; per-fold cache populates `avg_news_articles` from test_days; row writes real value.

2. **CR-E3RUN-A-02 (MAJOR)**: Resume identity switched from cell_id (formula-dependent) to `(fold, seed)` tuple (canonical). Same fix propagated to E4-α with `(edge_config, fold, seed)`.

3. **CR-E3RUN-A-03 (CONCERN)**: Pre-validation at per-fold setup raises if any (train/val/test) day lacks an edge tensor. Catches invariant break at setup time, not silently at training time.

Re-smoke after fixes (forced cache rebuild) PASSED — see progress.md 2026-05-27 entry.
