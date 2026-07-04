---
handoff_date: 2026-07-03
last_completed: "2026-07-03-b: Codex TP3 results review of M14 — PROCEED-WITH-FIXES (0 CRIT / 2 MAJOR / 3 CONCERN), all 5 findings verified + fixed"
in_flight:
  - id: m14-paper-edit
    file: paper/main.tex
    status: "DRAFTED + Codex-TP3-approved wording; NOT yet applied. Awaiting H博士 sign-off, then apply + recompile-verify anonymous 8pp + commit. Two sentence-level edits (exact before→after in body §A below)."
    blockers: ["H博士 sign-off on the two drafted edits"]
open_questions:
  - "M14 edit: keep abstract/§1 confirmatory 'both universes' (pre-registered 30-trial truth) and add M14 as a §4/§6 sensitivity — agreed approach, just needs apply. Any wording change before applying?"
file_state:
  modified_since_last_commit:
    - progress.md          # +2026-07-03-a (M14 verdict) +2026-07-03-b (Codex TP3); +parallel 2026-07-02-b
    - docs/analysis.md     # +2026-07-03-a (M14, corrected per TP3); +parallel 2026-07-02-a
    - plan.md              # M14 plan entry 2026-06-30-a
    - paper/main.tex       # PARALLEL session (2026-07-02 T0/T1 reviewer-anticipation) — NOT this session
    - paper/README.md      # parallel + this-session author/arXiv notes
  new_files:
    - experiments/storya_v21_main12_m14_retune/          # M14 L2@90 eval, 240 cells
    - experiments/storya_v21_main12_m14_merged/          # symlink merge (confirmatory + L2@90) for the analyzer
    - artifacts/storya_v21_tune/frozen_hparams_m14.json  # L2 B+C @90 merged (confirmatory frozen_hparams.json UNTOUCHED)
    - artifacts/storya_v21_family1_m14/                  # recomputed DM-HLN with L2@90
    - experiments/storya_v21_tune/{B,C}_L2.json.30trial.bak  # confirmatory 30-trial backups
    - artifacts/reviews/2026-07-03_codex_results_A.md    # Codex TP3
    - docs/session_handoff_2026-07-03.md                 # this file
rule9_status:
  touchpoint_1_plan: PASSED     # M14 plan approved 2026-06-30 (primary=B pre-committed)
  touchpoint_2_code: N/A        # M14 reused existing scripts, no new code
  touchpoint_3_results: PASSED  # Codex 2026-07-03_codex_results_A.md, PROCEED-WITH-FIXES, all 5 fixed
next_actions:
  - "Apply the two drafted M14 edits to paper/main.tex (§4 line ~174, §6 line ~365 — see body §A)"
  - "Recompile: confirm the ANONYMOUS ICAIF build stays 8pp (add `anonymous` to \\documentclass); non-anon arXiv build ~9pp is fine"
  - "Commit the M14 milestone (experiment artifacts + docs + paper edit) once 8pp confirmed"
  - "Optional: remaining reviewer-anticipation majors — check which the parallel 2026-07-02 session already closed (I-01/07/15/34/35/etc.); leftover set is a text-only pass"
---

# Session Handoff — 2026-07-03

## Where we are (one paragraph)
The Codex-compact 8pp ICAIF paper passed a full paperjury ultracode review (committed
`a5da503`, arXiv non-anon author block = Tracy He + Jinchi Lv). This session then ran **M14**
(GAT trials-sensitivity sweep) to answer reviewer concern I-02 ("the correlation-GAT is just
under-tuned at the equal 30-trial budget"). **Verdict: the leak-free headline is ROBUST to a
3× search budget; Universe-C's significance is search-sensitive.** Codex TP3 reviewed the
result (PROCEED-WITH-FIXES) and caught one number mis-citation (my A-05, fixed) plus wording
over-claims (fixed). **All experiment results are recorded and number-verified; the only
pending item is applying the two drafted, Codex-approved paper sentences (§A below).**

## M14 verdict (recorded in docs/analysis.md 2026-07-03-a)
| L2−L1 | confirmatory (L2@30) | M14 (L2@90, 3× budget) |
|---|---|---|
| Universe B (leak-free, PRIMARY) | ΔIC=−0.0133, p=4.0e-4, BH-reject | **ΔIC=−0.0090, p=0.0021, BH-reject** ← survives |
| Universe C (leak-selected) | ΔIC=−0.0119, p=6.9e-6, BH-reject | **ΔIC=−0.0069, p=0.059, BH-NOT-reject** ← search-sensitive |

Sources: `artifacts/storya_v21_family1_m14/family1_dm_hln.csv` (M14) vs
`artifacts/storya_v21_family1/family1_dm_hln.csv` (confirmatory). Pooled test IC and val-IC
in analysis.md 2026-07-03-a, all re-verified against source 2026-07-03 (0 discrepancies).
Interpretation (Codex-corrected): M14 addresses the *specific* equal-30/categorical-density
objection in the primary leak-free universe; it does NOT prove exhaustive tuning fairness
(density-matching is a TPE heuristic). Gap narrows ~30–40% but does not close; C direction
stays negative, near-nominal. M14 is an **independent** 90-trial retune (TPE RNG restarts
per process — NOT a superset of the confirmatory 30).

## §A — The two drafted M14 paper edits (THE TODO — apply after sign-off)

**Edit 1 — §4 Data and Setup (main.tex ~line 174), replace the last sentence:**
- BEFORE: `A trials-sensitivity sweep is left for future work.`
- AFTER:  `A trials-sensitivity sweep probes it: re-tuning the correlation-GAT (L2) at $3\times$ the budget (90 trials, matching the MLP's search density) leaves the leak-free Universe-B penalty significant (L2$-$L1${}=-0.0090$, HLN $p=0.002$, BH-reject) but drops the Universe-C penalty below significance ($-0.0069$, $p=0.059$, sign unchanged); the gap narrows without closing, so the primary leak-free result is not an equal-budget artifact.`

**Edit 2 — §6 Discussion (main.tex ~line 365), replace the mid-sentence clause:**
- BEFORE: `...reduces IC relative to the MLP, though at a fixed 30-trial budget an under-searched GAT space remains an alternative reading (\S\ref{sec:spa}).`
- AFTER:  `...reduces IC relative to the MLP; a $3\times$-budget trials sweep (\S\ref{sec:data}) narrows but does not close this gap in the leak-free universe, so it is not merely an equal-budget artifact, even if exhaustive tuning fairness is not established.`

Notes: keep abstract/§1 confirmatory "BH-significant in both universes" (accurate at the
pre-registered 30-trial budget); M14 is the sensitivity qualifier. After applying, recompile
and confirm the anonymous ICAIF build is still 8pp (may be +2–3 lines → trim if it spills).

## Broader paper state (not M14)
- `paper/main.tex` = single canonical source. Non-anon (arXiv) build ~9pp; anonymous ICAIF
  build 8pp (add `anonymous` to `\documentclass` line 9). Authors: Tracy He (USC Viterbi,
  Financial Engineering) + Jinchi Lv (USC Marshall, DSO). Preprint `\acmConference`.
- Paperjury ledger: `paper/.paper-review/LEDGER.json`. A parallel 2026-07-02 session applied
  T0×21 + T1×5 reviewer-anticipation fixes (Codex TP3 `2026-07-02_codex_results_A.md`) — e.g.
  I-01 Table-1 caption, I-07 pooled BH + BY, I-34 Family-2→edge-attribution, I-35 HXZ 65/82%,
  I-42/44/46/47/49. Cross-check which of the original 8 leftover majors remain.
- M14 supersedes the 2026-07-02-b "T2/trials-sweep skipped" note (H博士 re-directed to run it).
