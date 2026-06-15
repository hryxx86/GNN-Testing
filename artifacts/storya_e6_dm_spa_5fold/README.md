# storya_e6_dm_spa_5fold/ — FROZEN 5-fold E6 snapshot

These are the **5-fold** E6 outputs (E1 anchor, 400 cells, `n_cells=50` per universe×model),
extracted verbatim from git commit `5bef3b9` — the last commit before the 2026-06 12-fold
window extension. They are the source-of-record for `docs/analysis.md` entries
**2026-05-27-a** and **2026-05-27-c**.

## Why this exists

`compute_e6_dm_spa.py` writes to `artifacts/storya_e6_dm_spa/` **in place**. The 2026-06
12-fold rerun (`docs/analysis.md` 2026-06-13-a) overwrote every file in that live dir with
12-fold numbers. The same path therefore could not simultaneously back the 5-fold citations.
The pre-overwrite 5-fold files are frozen here so the 5-fold-era citations stay resolvable.

| Quantity | 5-fold (this dir) | 12-fold (`../storya_e6_dm_spa/`) |
|---|---|---|
| Univ-C GAT IC | 0.043 | 0.018 |
| SPA p_consistent B / C / joint | 0.147 / 0.384 / 0.136 | 0.295 / 0.338 / 0.466 |
| n_cells per universe×model | 50 (5 folds × 10 seeds) | 100 (… × 12 folds, seed-avg T=749) |

## Rules

- **Frozen.** Do NOT regenerate into this dir. New `compute_e6_dm_spa.py` runs write to the
  live `../storya_e6_dm_spa/` dir.
- 12-fold entries (analysis.md 2026-06-13-a, 2026-06-15-a) correctly cite the **live** dir.
- 5-fold entries (2026-05-27-a/c) cite **this** snapshot.

Created 2026-06-15 — see `progress.md` 2026-06-15-c.
