---
handoff_date: 2026-09-11
last_completed: "2026-09-11-g: C-pre plan frozen (docs/c_pre_plan_2026-09-11.md) + Codex TP1 Round A PROCEED-WITH-FIXES applied; 2026-09-11-f: plan_aaa_t1_stability figure redrawn; commits pushed to origin/main"
in_flight: []
open_questions:
  - "C-pre GO / NO-GO: docs/c_pre_plan_2026-09-11.md §8 D1-D6 (frozen defaults; Codex TP1-A agreed on all six); if GO → TP2 (+ Codex Round B) → commit selector → select → freeze UNIVERSE_CPRE_NAMES → tune → 240 cells (Mac, or T4 if a hostname is given) → stats → TP3; Mac ≈ 3 h compute"
  - "Paper L1 / appendix rewording (main.tex:290, :998, :1012) AND the figure caption (figure redrawn 2026-09-11-f; main.tex untouched here): 'only 5 of 15 groups survive strict T-1 re-ranking' is a misstatement (proxy top-15 identical with/without the shift; 5/15 = permutation top-15 ∩ proxy top-15) and the 'definitive check' promise cannot be discharged by C5 (its selection is test-informed) — H博士 to rewrite before inserting any C5 paragraph"
  - "C-pre (pre-test selector, brief §9.10): only needed if the paper wants a quantitative statement about leakage in C (TP3); requires freezing the coverage rule for hc_mom12m warm-up (Codex TP1-B B-01), the group-score definition, and grouping reuse vs re-clustering; then its own TP1"
  - "Push: DONE 2026-09-11 (C5 commits + report + figure/C-pre-plan commit are on origin/main)"
  - "Optional L2 layer on C5 (T4 idle, deps installed): recommended NOT to run (weak bearing on the paper's claim)"
file_state:
  modified_since_last_commit:
    - analyze_e1_lofo.py            # pre-existing (earlier session), not touched here
    - paper_figs/fig_family2.py     # pre-existing, not touched here
    - figures/family2_edge_causal.pdf / .png   # pre-existing, not touched here
    - (many D / ?? entries = earlier-session archived/ reorganisation, untouched)
  new_files:
    - analyze_c5_sensitivity.py
    - docs/c5_rerun_brief_2026-09-10.md
    - docs/session_handoff_2026-09-11.md
    - experiments/storya_v21_main12_c5/ (Mac replicate, 240 cells) + experiments/storya_v21_main12_c5_t4/ (T4 primary, 240 cells)
    - artifacts/storya_v21_family1_c5/ (primary stats) + artifacts/storya_v21_family1_c5_mac/ (replicate stats)
    - artifacts/storya_v21_tune/{C5_L0,C5_L1,frozen_hparams_c5}.json + c5_tune_archive_md5.json + studies_c5/
    - artifacts/reviews/2026-09-10_codex_plan_A.md, _plan_B.md, _code_A.md, 2026-09-11_finance-gnn-reviewer_code_B.md, _results_A.md
rule9_status:
  touchpoint_1_plan: PASSED        # Codex Round A BLOCK-EXECUTION → re-scoped → Round B PROCEED-WITH-FIXES (closures applied)
  touchpoint_2_code: PASSED        # Codex Round A PROCEED-WITH-FIXES (5/5 fixed) → finance-gnn-reviewer Round B fallback (Codex usage limit) PROCEED-WITH-FIXES (4/4 fixed)
  touchpoint_3_results: PASSED     # finance-gnn-reviewer Round A fallback (Codex usage limit) PROCEED-WITH-FIXES (4 M fixed, 3 Cn accepted)
  cpre_touchpoint_1_plan: PASSED-ROUND-A   # Codex 2026-09-11 PROCEED-WITH-FIXES (0 C / 2 M / 2 Cn, all applied); Round B deferred to TP2 after H博士 approval
  closeout_audit: PASSED           # 4 Explore agents; artifacts/reviews/2026-09-11_explore-*_closeout.md; progress.md 2026-09-11-e
next_actions:
  - "H博士: C-pre go/no-go (recommended: GO) and confirm D1-D6; paper rewording + new figure caption before 9/25"
  - "If C-pre approved: freeze coverage/group-score/grouping rules in brief §9.10 → /codex-plan-review → implement selector → tune L0/L1 → 240 cells (T4) → sensitivity stats + paired contrast"
  - "Insert the C5 sensitivity paragraph into the paper appendix ONLY with the permitted wording in docs/analysis.md 2026-09-11-a"
---

# Session Handoff — 2026-09-11

## What happened

H博士's brief (`docs/c5_rerun_brief_2026-09-10.md`) asked for the "definitive check" promised in the paper's Limitation L1: re-tune and re-evaluate L0 (LightGBM) and L1 (MLP) on the 20 columns of the 5 Plan-AAA factor groups that "survive the T-1 re-rank" (C5), under the frozen confirmatory protocol.

Codex Touchpoint 1 blocked the plan as specified and both of its core objections were verified by Claude: (i) the T-1 proxy ranking scored the last 313 valid label days of the panel (2024-09-27..2025-12-26) and Plan AAA itself scored the 5-fold test quarters (2024-04-01..2025-06-30) — both inside the 12-fold test period, so the 5-group selection is test-informed, not leak-free; (ii) the proxy top-15 is identical with and without the T-1 shift, so "5/15" is the intersection of two importance measures, not a leakage-removal effect. The run was therefore re-scoped (Codex option 1) as an explicitly **test-informed feature-subset sensitivity**, executed as requested, and the paper-side rewording plus the pre-test selector alternative ("C-pre") were escalated to H博士.

Execution: L0/L1 tuned on C5 (Mac, 30 trials each; both arms' winners had negative 2022H2 val-IC), 240 cells run on the Colab T4 (primary, pre-declared after H博士 said the GPU was idle) and on the Mac (replicate). Statistics via `compute_family1_ladder.py --sensitivity` + `analyze_c5_sensitivity.py`. Codex hit its usage limit mid TP2-B; TP2-B and TP3 were taken by `finance-gnn-reviewer` (fallback, recorded).

## Headline (T4 primary; docs/analysis.md 2026-09-11-a has the full permitted/forbidden wording)

C5 L1−L0 = +0.0134, 21d block-bootstrap 95% CI [+0.0008, +0.0283] (boundary exclusion: 1.96×SE = 0.0138 > ΔIC), HLN p 0.008 at the implementation-default NW auto lag / 0.054 at lag 21 (under lag 21 none of C5/C/B reaches 0.05), 10/10 seeds same sign (so 0 LOSO flips by construction); vs C +0.0148 and B +0.0143. Paired C − C5 = +0.0013 [−0.016, +0.019] (paired MDE ≈ 0.025 > the C effect → equivalence not established). About half of the contrast comes from 2025Q2 (ex-fold-9: +0.0069, p 0.13), as in C and B. |ΔIC| < MDE in all three. Mac replicate: +0.0132 [+0.0002, +0.0279].

## Where things are

- Code: `run_storya_e1_anchor.py` (build_universe_C5), `run_storya_v21_main12.py` (explicit `--universe C5`, cell_id 2400–3599, provenance), `run_storya_v21_tune.py`, `run_v21_tune_launcher.py` (subset merge), `compute_family1_ladder.py` (`--universes/--arms/--sensitivity`), `analyze_c5_sensitivity.py` (new).
- Results/stats: see file_state above. T4 code identity: `experiments/storya_v21_main12_c5_t4/_code_identity_t4.json` (== commit 9008dbe).
- Reviews: `artifacts/reviews/2026-09-10_*` (Codex TP1 A/B, TP2 A) and `2026-09-11_finance-gnn-reviewer_*` (TP2 B, TP3 A).
- Colab T4 runtime `degrees-competitions-medical-earrings` had deps installed and the C5 code scp'd (not git-pulled; origin/main still at eb8314e).
