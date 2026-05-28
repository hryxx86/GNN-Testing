---
reviewer: codex
touchpoint: code
round: A
target_files:
  - run_storya_e1_6_hats.py:1-1001
target_plan: /Users/heruixi/.claude/plans/hats-baseline-reproduction-delightful-lighthouse.md
findings:
  - id: CODEX-A-01
    severity: CONCERN
    category: data-leakage
    claim: "Cached news snapshots bypass the E3 PIT runtime assertion path."
    evidence: "run_storya_e1_6_hats.py:762-774 uses `if os.path.exists(args.news_cache)` to load `np.load(...)`; `build_per_day_news_edges(...)` is called only in the `else` branch. The PIT assertion itself is in run_storya_e3_news_edge.py:252-255: `assert window_max <= cutoff_np`."
    suggested_fix: "Before the 50-cell launch, force a rebuild of the news cache with the current E3 builder or add a required cache metadata/version/source-hash check and reject metadata-less caches. The rebuilt path must execute `build_per_day_news_edges()` at least once so the PIT assertion fires."
    status: FIXED
    resolution_notes: "Claude independently verified (Rule 9 诚信要求 #5) by reading run_storya_e1_6_hats.py:762-786 + run_storya_e3_news_edge.py:240-258. Confirmed: cache load branch at HATS L762-772 skips the assertion that lives at E3 L253-255. ACTION TAKEN: (1) Renamed cache `news_snapshots_cache.npz` -> `.npz.bak_pre_hats_rebuild_2026_05_27`. (2) Re-ran `python run_storya_e1_6_hats.py --smoke` (resume=ON → cell 400 already done, planned=0; but cache rebuild still triggered as part of data prep). (3) PIT assertion fired for all 1254 daily snapshots — built in 6.9s, max PIT-eligible ts = 2026-01-26 23:57:06+00:00 (before the latest fold's test_end cutoff). No PIT VIOLATION raised. (4) Verified bit-equivalence: old cache vs rebuilt cache → 1255 keys both, 0 differing keys (numpy array_equal check). Old backup deleted after verification. Rebuilt cache now lives at experiments/storya_e3_news_edge/news_snapshots_cache.npz with mtime 2026-05-27 22:24. Future cache rebuilds are cheap (~7s); no need to add metadata/version check for this run. If E3 builder code changes later, project convention will require fresh cache rebuild before next launch."
summary:
  critical: 0
  major: 0
  concern: 1
  fixed_before_reply: 0
  fixed_after_reply: 1
overall_verdict: PASS-WITH-CONCERNS
post_disposition_verdict: PASS
---

# Review body

Data leakage: one concern. The news PIT builder is correct when executed: `build_per_day_news_edges()` filters articles to `ts <= cutoff_np` and asserts `window_max <= cutoff_np` at `run_storya_e3_news_edge.py:245-255`. HATS does not always execute that path because the default cache branch at `run_storya_e1_6_hats.py:762-772` loads edge arrays directly; the builder is called only on cache miss at `run_storya_e1_6_hats.py:774-776`. The 1-cell smoke did not trigger this because the cache file existed and the run took the load branch, so this is not a crash issue; it is an auditability/PIT-cache concern.

No issues found in frozen correlation, train-only preprocessing, purge handling, or label indexing. HATS uses `create_fold_masks(...)` at `run_storya_e1_6_hats.py:824`, then fits winsor/scaler on `train_days` only at `run_storya_e1_6_hats.py:829-830`, freezes correlation with `get_frozen_snapshot_idx(train_days[-1], ...)` at `run_storya_e1_6_hats.py:832`, and builds edges only for `train_days`, `val_days`, and `test_days` at `run_storya_e1_6_hats.py:840-843`.

No issues found in logic bugs. The relation loop uses per-layer/per-relation `GATConv` and `LayerNorm` stacks at `run_storya_e1_6_hats.py:245-255`; alpha is softmaxed across relation dimension at `run_storya_e1_6_hats.py:257-260`. Gradient accumulation matches E1, including leftover scaling at `run_storya_e1_6_hats.py:381-390`. Validation alpha stats are reset each epoch and skipped with invalid val days at `run_storya_e1_6_hats.py:397-425`. The edge builder returns `[corr, sector, news_day]` per day at `run_storya_e1_6_hats.py:293-301`. Cell IDs are asserted in `[400,449]` at `run_storya_e1_6_hats.py:152-163`.

No issues found in reproducibility. `set_seed(seed)` is called before model creation at `run_storya_e1_6_hats.py:324-333`; cuDNN deterministic mode is set at `run_storya_e1_6_hats.py:71-72`; shuffling uses seeded NumPy at `run_storya_e1_6_hats.py:367`.

No issues found in numeric stability or PyG GATConv contract. `squeeze(-1)` preserves `(N,R)` logits at `run_storya_e1_6_hats.py:258`; `last_alpha.detach()` avoids retaining the graph at `run_storya_e1_6_hats.py:264`; alpha division is guarded at `run_storya_e1_6_hats.py:423-425`; `hidden % heads` is asserted at `run_storya_e1_6_hats.py:200-201`.

No issues found in E6 schema prep for alpha columns. HATS writes the alpha diagnostic columns in `RESULTS_COLUMNS` at `run_storya_e1_6_hats.py:129-138`; pandas concat will tolerate E1 missing those columns by filling NaN, and current downstream code reads only required common metric columns.

No issues found in resume identity. `load_manifest_done()` returns `(fold, seed)` pairs at `run_storya_e1_6_hats.py:519-528`. The smoke row is a real full-configuration cell and will be skipped by a later resume run; that is only stale if code changes after smoke, in which case cid 400 should be rerun or the smoke row removed.

---

**Verdict: PASS-WITH-CONCERNS**

The sole finding is CODEX-A-01 (CONCERN): the cached news snapshot path bypasses the E3 PIT runtime assertion. All other focus areas — frozen correlation, train-only scaler fit, purge handling, label indexing, HATS3RAdapt forward pass, grad accumulation, alpha aggregation, cell-id injectivity, seed reproducibility, numeric stability, GATConv contract, E6 column schema, and resume manifest — are clean.

Before the 50-cell Colab launch, H博士 should decide: delete the news cache and force a full rebuild (which re-runs the PIT assertion), or accept CODEX-A-01 as ACCEPTED-AS-CONCERN given that E3 already passed its own smoke test with the same cache file.
