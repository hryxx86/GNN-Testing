---
handoff_date: 2026-06-21
last_completed: "2026-06-21-a: D-RERUN-12F confirmatory analysis COMPLETE — Family-1 + Family-2 ran (full n_boot=5000), Touchpoint 2 + 3 PASSED, analysis.md headline written."
in_flight:
  - id: paper-figs-regen
    file: paper_figs/
    status: "NOT started. The 27 paper figures + 10 LaTeX tables (built 2026-05-28) were on the PILOT / earlier numbers — they must be regenerated on the CONFIRMATORY tuned-ladder + FC artifacts (artifacts/storya_v21_family1/ + storya_v21_family2_fc/). Figure-script architecture is the 13 modular paper_figs/fig_*.py (see plan Decision Log 2026-05-27)."
    blockers: []
  - id: paper-results-rewrite
    file: docs/storya_paper_draft.md
    status: "NOT started. §Results/§Discussion need the confirmatory TWO-FAMILY framing + the T3 narrative-discipline (local ladder evidence, not global/causal 'graphs hurt'; Family-2 0/6 underpowered honestly framed; C/L5s 27.5% stability finding)."
    blockers: ["paper-figs-regen (figures referenced in §Results)"]
open_questions:
  - "Paper figures: regenerate ALL 27 on confirmatory numbers, or only the subset whose data changed (the tuned-ladder SPA/DM/FC figs change; the prior-work figs e.g. horizon ablation / loss horserace do not)?"
  - "How prominently to feature the C/L5s 27.5% constant-collapse as a 'tuning selected an unstable config' finding — its own §, or folded into the 'smoothing hurts ranking' mechanism para?"
  - "Family-1 SPA C p_consistent=0.0774 is marginal (just above 0.05). Headline wording: 'no arm confirmed to beat LightGBM' (current) vs surfacing the C-universe near-miss — keep conservative per T3 R-A-05?"
file_state:
  modified_uncommitted:
    - "progress.md, plan.md, docs/analysis.md (2026-06-21-a entries — commit at closeout)"
    - "compute_family1_ladder.py, compute_fc_edge_causal.py (new analyzers, post-T2-fixes)"
  new_files:
    - "compute_family1_ladder.py, compute_fc_edge_causal.py"
    - "artifacts/storya_v21_family1/ (9 outputs: spa/dm_hln/ic_ci/mde/lofo/stability/cl5s_robustness .csv + ledger.json + summary.md)"
    - "artifacts/storya_v21_family2_fc/ (family2_fc_causal.csv + ledger.json + summary.md)"
    - "artifacts/reviews/2026-06-20_codex_code_A.md, artifacts/reviews/2026-06-21_codex_results_A.md"
    - "experiments/storya_v21_main12_tuned/ (2160-cell merged main table), experiments/_rerun_colab_staging/ (heavy/L7/FC pulled from Drive)"
    - "docs/session_handoff_2026-06-21.md (this file)"
rule9_status:
  touchpoint_1_plan: PASSED       # FC two-family v2 (2026-06-17_codex_plan_A)
  touchpoint_2_code: PASSED       # analyzers (2026-06-20_codex_code_A) — 1 MAJOR + 1 CONCERN, both fixed
  touchpoint_3_results: PASSED    # PASS-WITH-CONCERNS (2026-06-21_codex_results_A) — 1 MAJOR + 4 CONCERN, all narrative-discipline, addressed in analysis.md framing
next_actions:
  - "Decide figure-regen scope (open_question 1), then regenerate the confirmatory paper figs/tables on artifacts/storya_v21_family1 + family2_fc."
  - "Rewrite paper §Results/§Discussion with the two-family confirmatory framing + T3 narrative discipline."
  - "Commit this session's work (progress/plan/analysis 2026-06-21-a + the two analyzers + artifacts)."
---

# Session Handoff — 2026-06-21 (D-RERUN-12F confirmatory analysis COMPLETE)

## TL;DR
The whole D-RERUN-12F arc is done: data generated (2160 main + 240 L7 + 720 FC, 0 fail/0 dup, survived 3 Colab recycles), merged, and the **two confirmatory analyzers ran at full `n_boot=5000`** with **Touchpoint 2 (code) + Touchpoint 3 (results) both PASSED**. The paper headline is written into `docs/analysis.md` 2026-06-21-a. What remains is **paper-side**: regenerate the confirmatory figures/tables and rewrite §Results with the disciplined framing.

## The confirmatory result (all numbers source-cited in analysis.md 2026-06-21-a)
- **Family-1 (predictive)**: **no tuned arm reliably beats tuned LightGBM** — Hansen SPA does not reject in either universe (B p_consistent=0.277, C=0.077; source `artifacts/storya_v21_family1/family1_spa.csv`). Local DM-HLN ladder evidence (source `family1_dm_hln.csv`): in Univ C the tuned MLP beats tuned LightGBM (L1−L0=+0.015) but the tuned corr-GAT loses to that MLP (L2−L1=−0.012) and the tuned news-edge arm loses to corr-GAT (L3−L2=−0.012). **Read LOCALLY** (within this tuned ladder), NOT as global "graphs hurt" — C L4/L5/L6 recover above L2.
- **Family-2 (causal, fixed-capacity edge)**: **0/6 contrasts survive BH-FDR; 6/6 underpowered** (source `family2_fc_causal.csv`). Edge effects are "directionally positive but not family-significant"; the B sign reversal (matched +ve vs tuned −ve) is the capacity-confound evidence.
- **C/L5s stability finding**: the tuned C/L5s SAGE-Mean config degenerates to a **constant predictor in 27.5%** of C cells (source `family1_stability.csv`) — verified converged-to-constant (best_val_loss≈0.998), NOT a crash. Primary treatment = EXCLUDE (H博士 2026-06-21 LOCKED); conclusion-invariant under 3 treatments (source `family1_cl5s_robustness.csv`); never re-tuned.

## Rule 9 trail
- T2 code review: Codex (after a transparently-logged premature finance-gnn fallback — Codex was not broken, just slow at `gpt-5.5 xhigh`; see progress 2026-06-21-a). Full: `artifacts/reviews/2026-06-20_codex_code_A.md`.
- T3 results review: Codex, **PASS-WITH-CONCERNS** (0 CRIT). The 1 MAJOR + 4 CONCERN are all narrative-discipline (don't promote local ladder signals into global/causal claims; label FC CIs as descriptive vs BH as confirmatory; keep the two-family hierarchy clear); all addressed in the analysis.md framing. Full: `artifacts/reviews/2026-06-21_codex_results_A.md`.

## Integrity gates cleared this round
- **Ghost-dimension audit (full)**: all 12 tuned dims (6 NN + 6 LGB) + 6 HATS dims have live consumption points (`run_storya_e1_anchor.py:483-586/728-747`, `run_storya_e1_6_hats.py:328-335`) → the confirmatory runs used the §4 tuned HPs, none silently defaulted.
- **n_eff label** (CODEX-A-02): `family1_mde.csv` now reports `T_days` + `n_eff_blocks` separately.
- **L5s isolation**: not in any DM pair; enters only SPA M=9 → C/L5s degeneracy touches no pairwise and no Family-2 contrast.

## What's NOT done (the next session's job)
1. **Figures/tables on confirmatory numbers** — `paper_figs/` were on pilot/earlier data; regenerate the SPA/DM/FC ones on `artifacts/storya_v21_family1` + `storya_v21_family2_fc` (decide scope per open_question 1).
2. **Paper §Results/§Discussion** — write the two-family confirmatory story with the T3 discipline (local-not-global, underpowered-not-null, two separate families).
3. **Commit** — this session left analyzers + artifacts + the three docs uncommitted.

## Reading red lines (unchanged)
- §4 tuning val-IC is a SELECTION metric (2022H2, optimistic) — never a finding.
- Two separate confirmatory families: Family-1 = predictive/model-selection; Family-2 = causal edge. Their primaries (SPA/DM for F1; matched-ΔIC for F2) do not interchange.
- C/L5s never re-tuned (equal-budget symmetry); the 27.5% collapse is a finding, reported, not hidden.
