---
handoff_date: 2026-09-12
last_completed: "2026-09-12-c: C-pre production run done (240 cells, integrity PASS) + Codex TP3 PROCEED-WITH-FIXES applied; docs/analysis.md 2026-09-12-a written (ΔIC −0.0024 [−0.0256, +0.0178]; paired vs C/B/C5 all include 0); 4-agent closeout run"
in_flight: []
open_questions:
  - "Paper (before 9/25): apply Codex TP3 A-04 — correct main.tex:290/:998/:1012 (the five-group overlap is a ranking-method disagreement, not T-1 leak removal), acknowledge the completed L0/L1 pre-evaluation re-selection sensitivity with the PERMITTED sentences of docs/analysis.md 2026-09-12-a, keep the qualification on C's original results, state that graph/edge re-selection is unresolved; drop the halved/doubled sentence from the C5 paragraph; new caption for the redrawn plan_aaa_t1_stability figure"
  - "Push: the C-pre commits (b969a62 .. the closeout commit) are local only — H博士 to confirm"
  - "Not recommended: L2 layer on C-pre, C5h, per-arm paired level tests (only needed to claim a per-arm decline)"
file_state:
  modified_since_last_commit:
    - analyze_e1_lofo.py            # pre-existing (earlier session), not touched
    - paper_figs/fig_family2.py     # pre-existing, not touched
    - figures/family2_edge_causal.pdf / .png   # pre-existing, not touched
    - (many D / ?? entries = earlier-session archived/ reorganisation, untouched)
  new_files:
    - artifacts/reviews/2026-09-12_codex_code_A.md
    - artifacts/reviews/2026-09-12_codex_results_A.md
    - artifacts/storya_v21_tune/CPRE_L0.json
    - artifacts/storya_v21_tune/CPRE_L1.json
    - artifacts/storya_v21_tune/cpre_tune_archive_md5.json
    - artifacts/storya_v21_tune/frozen_hparams_cpre.json
    - run_step3_plan_z_part_a.py
    - run_storya_cpre_select.py
    - artifacts/storya_cpre_select/ (results / stats / selector archive)
    - artifacts/storya_v21_family1_cpre/ (results / stats / selector archive)
    - artifacts/storya_v21_tune/ (results / stats / selector archive)
    - experiments/storya_v21_main12_cpre/ (results / stats / selector archive)
  rewritten_files:
    - analyze_c5_sensitivity.py     # universe-parametrised (C5 numeric outputs byte-identical)
    - paper_figs/fig_plan_aaa_t1.py  # retitled figure (measure disagreement, not T-1 leak correction)
    - run_storya_e1_anchor.py / run_storya_v21_main12.py / run_storya_v21_tune.py   # CPRE universe wiring
rule9_status:
  touchpoint_1_plan: PASSED        # C-pre plan: Codex Round A 2026-09-11 PROCEED-WITH-FIXES (applied) -> Round B 2026-09-12 all FIXED
  touchpoint_2_code: PASSED        # Codex Round A 2026-09-12 PASS-WITH-CONCERNS (1 Cn fixed before the run)
  touchpoint_3_results: PASSED     # Codex Round A 2026-09-12 PROCEED-WITH-FIXES (1 M wording + 3 Cn; applied / accepted)
  closeout_audit: PASSED           # 4 Explore agents 2026-09-12 (0 CRITICAL, 1 MAJOR + 20 CONCERN, all fixed); artifacts/reviews/2026-09-12_explore-*_closeout.md; progress 2026-09-12-d
next_actions:
  - "H博士: paper L1 / appendix edits per open_questions[0]; confirm push"
  - "Nothing else is queued for the experiment side; the C5 and C-pre sensitivity chains are complete"
---

# Session Handoff — 2026-09-12

## What happened

H博士 approved the C-pre plan ("go"). The frozen selector (`run_storya_cpre_select.py`, docs/c_pre_plan_2026-09-11.md §3) was committed, then run on clean source: single-feature |IC| on the tuning-train window 2021-07-01..2022-05-31 (231 dates, label end ≤ 2022-06-30), τ = 0.50 coverage (`hc_mom12m` unscored, 85/231), Plan-AAA 61 groups reused, top-15 groups → **48 columns** (5 hc + 43 Alpha158; 22 shared with C, 4 with C5). L0/L1 were re-tuned (30 trials; all finalists' three-seed val-IC positive) and 240 cells run on the Mac (git 46ca6e3, integrity PASS). Codex reviewed the plan (A/B), the code (A) and the results (A); every finding was applied or accepted; the closeout audit ran with four Explore agents.

## Headline (docs/analysis.md 2026-09-12-a has the permitted / forbidden wording)

On C-pre the seed-averaged daily MLP−LightGBM contrast is −0.0024 (95% block-bootstrap CI [−0.0256, +0.0178]; HLN p 0.786 auto lag / 0.847 lag 21; own MDE 0.0313; 5/10 seeds same sign). The positive pooled point estimate seen in C (+0.0148), C5 (+0.0134) and B (+0.0143) is not reproduced, but the three paired comparator-minus-C-pre intervals (+0.017, +0.017, +0.016) all contain zero (paired MDE ≈ 0.041–0.045). Both arms' pooled IC point estimates are low on C-pre (L0 0.0057, L1 0.0033; intervals include 0) — descriptive only. The contrast is sensitive to 2025Q2 (L0 0.204 vs L1 0.066; ex-fold-9 +0.0099 [−0.0069, +0.0265]); the full-period result is primary. C-pre does not estimate leakage inflation in C and does not convert C's results into clean confirmatory evidence; B remains the leak-free anchor.

## Where things are

- Plan: `docs/c_pre_plan_2026-09-11.md`. Selector archive: `artifacts/storya_cpre_select/`. Run: `experiments/storya_v21_main12_cpre/`. Stats: `artifacts/storya_v21_family1_cpre/` (`cpre_comparison.md` is the one-page view). Tune archive: `artifacts/storya_v21_tune/{{CPRE_L0,CPRE_L1,frozen_hparams_cpre,cpre_tune_archive_md5}}.json`.
- Reviews: `artifacts/reviews/2026-09-11_codex_plan_A.md`, `2026-09-12_codex_code_A.md`, `2026-09-12_codex_results_A.md`, `2026-09-12_explore-*_closeout.md`.
- The 2026-09-11 handoff (C5) is frozen as of its own date.
