---
handoff_date: 2026-05-01
last_completed: "2026-04-29-c: Plan Z++ v2 complete; all 8 Codex Round B fixes (B-01 through B-08) applied; awaiting H博士 final approval gate"
in_flight:
  - id: plan-zpp-final-approval
    file: /Users/heruixi/.claude/plans/plan-zpp-unified-2026-04-29.md
    status: "v2 patched per Codex Round B PROCEED-WITH-FIXES (5 MAJOR + 3 CONCERN); all fixes inline; awaiting H博士 sign-off on direct-execute vs Codex Round C verify"
    blockers:
      - "H博士 final approval (direct execute OR Round C verify first)"
      - "If approved: Phase 0 (Tier 0 audits + manifest + sentinel test, ~10h dev, 0 compute) starts immediately"
  - id: stage1-paper-writing-paused
    file: docs/analysis.md
    status: "Stage 1 + diag verdict (Scenario B with fold-4 caveat) is final; paper draft skeleton paused pending Plan Z++ Tier 1 outcomes for supplementary"
    blockers:
      - "Plan Z++ Tier 1 results (or null finding) needed to finalize Story C+ supplementary section"
open_questions:
  - "Plan Z++ v2 direct execute, or run Codex Round C verify first?"
  - "If Plan Z++ Tier 1.E regime forensic FAILS Gate 1.E primary (degradation_share < 0.50): Tier 2.A skipped, paper Story C+ stays as plain Scenario B verdict (no regime mechanism)"
  - "If Plan Z++ Tier 1.B Huber/Tukey clears Gate 1.B: 5-seed → 10-seed expansion authorized; budget impact ~6h additional M4"
file_state:
  modified_since_last_commit:
    - run_loss_horserace.py (resume bug fixes v1-v9 across 9 Codex review passes; 8 corruption modes A–H caught + cleanup persisted via safe_to_csv + mode-specific fallback paths + foreign-file quarantine)
    - analyze_loss_horserace.py (statsmodels API fix line 247; cluster_bootstrap_delta function; D-03 direction check; D-04/D-05/D-06 verdict text rewrites)
    - progress.md (2026-04-27-a Stage 1 horse race verdict; 2026-04-27-b Diagnostic_price + ListMLE fold-4 collapse; resume bug v1-v9 patch log)
    - plan.md (2026-04-27-a Path A/B/C decision point; 2026-04-27-b Path A Story C+ recommendation)
    - docs/analysis.md (2026-04-27-a Stage 1 8-contrast tables + Scenario B verdict; 2026-04-27-b Q1+Q2 verification + ListMLE fold-4 universal collapse mechanism + Story C+ outline; both with full provenance)
    - README.md (changelog entries 2026-04-27)
    - ~/.claude/plugins/marketplaces/openai-codex/plugins/codex/{hooks/hooks.json, scripts/stop-review-gate-hook.mjs} (restored to upstream after H博士 fixed gpt-5.5 issue)
    - ~/.claude/plugins/cache/openai-codex/codex/1.0.3/{hooks/hooks.json, scripts/stop-review-gate-hook.mjs} (mirror of marketplaces)
  new_files:
    - analyze_seed_diagnostic.py (Phase 1 seed diagnostic, descriptive only)
    - run_local_stage1_segmented.sh (M4 12h+1h segmented runner)
    - artifacts/reviews/2026-04-27_codex_results_D.md (Round D results review, PROCEED-WITH-FIXES, 5 MAJOR + 3 CONCERN, all fixed except D-01 which was ACCEPTED-AS-DISCLOSURE)
    - artifacts/reviews/2026-04-29_codex_discussion_A_data_length_regime.md (xhigh effort discussion A)
    - artifacts/reviews/2026-04-29_codex_discussion_B_split_methodology.md (xhigh effort discussion B)
    - artifacts/reviews/2026-04-29_codex_discussion_C_loss_noise.md (xhigh effort discussion C)
    - artifacts/reviews/2026-04-29_codex_plan_zpp_round_B.md (Round B verdict on Plan Z++)
    - experiments/loss_horserace/results_diagnostic_price.csv (200-cell diagnostic, mse + listmle × {MLP,SAGE} × S_price × 5 folds × 10 seeds)
    - experiments/loss_horserace/preds_diagnostic_price/ (200 .npy prediction files)
    - experiments/loss_horserace/cluster_bootstrap_ic.csv (Round D D-02 sensitivity)
    - experiments/loss_horserace/cluster_bootstrap_pred_cs_std.csv (Round D D-02 sensitivity)
    - experiments/loss_horserace/seed_diagnostic/ (Phase 1 seed analysis: per-seed × per-config IC heatmap, winrates, fold-4 panel)
    - /Users/heruixi/.claude/plans/seed-level-diagnostic-2026-04-29.md (superseded by Plan Z++)
    - /Users/heruixi/.claude/plans/plan-zpp-unified-2026-04-29.md (PRIMARY ACTION ITEM)
rule9_status:
  touchpoint_1_plan_seed_diagnostic_round_a: BLOCK-EXECUTION (Codex caught critical permutation-test + selection-bias-direction errors; plan superseded by Plan Z++)
  touchpoint_1_plan_zpp_3_discussions: PASSED (informal exploratory discussions, no formal verdict)
  touchpoint_1_plan_zpp_round_b: PROCEED-WITH-FIXES (5 MAJOR + 3 CONCERN; all 8 fixes applied to v2)
  touchpoint_2_code_resume_bug_v1_to_v9: PASSED (9 Codex review passes; v9 final state covers 8 corruption modes A–H + Drive-safe persistence + mode-specific fallback paths + foreign-file quarantine + cleanup persisted via safe_to_csv)
  touchpoint_3_results_round_d: PASSED (Codex Round D results review 2026-04-27, PROCEED-WITH-FIXES 5 MAJOR + 3 CONCERN, all fixes applied; D-01 ACCEPTED-AS-DISCLOSURE per H博士 Option A)
next_actions:
  - "H博士 sign-off on Plan Z++ v2: direct execute (recommended) OR Codex Round C verify first"
  - "If sign-off: Phase 0 starts immediately — audit data/reference/sp500_5y_phase5_features.npy for global winsorization/scaling, generalize manifest for rolling support, add graph provenance assertions, add behavioral leakage sentinel test"
  - "Phase 0 deliverables: artifacts/audits/phase5_features_audit.md + sentinel_leakage_test.md + fold_manifest_expanding.json + fold_manifest_roll2y.json"
  - "Phase A (after Phase 0 PASS): 1.B robust pointwise sweep (300 cells, ~8h M4) + 1.D hparam sweep (120 cells, ~3h M4) in parallel"
  - "Phase B: 1.A 100-cell window ablation + 1.C anchored RankNet pilot in parallel"
  - "Phase C: 1.E regime forensic + 2.C sector-adjusted IC (analysis only)"
  - "Phase D (conditional): Tier 2 experiments triggered by Phase A/B/C gate outcomes"
key_context_notes:
  - "Stage 1 preregistered verdict (Scenario B) is LOCKED. Plan Z++ work is supplementary only."
  - "Compute budget: ~30h M4 MPS before paper deadline. Phase 0-C uses ~17-21h compute + ~26h dev."
  - "Codex hooks (stop-review-gate) are restored and working; H博士 fixed gpt-5.5 issue 2026-04-27. If hooks fail again, see progress.md 2026-04-27-a §Codex Stop hook saga for context."
  - "H博士 communication style: Chinese conversation; English code/technical reports per CLAUDE.md Rule 3"
  - "Per CLAUDE.md Rule 9: every plan/code/results checkpoint goes through Codex; this Plan Z++ has been through Round A (3 discussions) + Round B (formal review)"
critical_paths:
  plan_v2: /Users/heruixi/.claude/plans/plan-zpp-unified-2026-04-29.md
  round_b_review: artifacts/reviews/2026-04-29_codex_plan_zpp_round_B.md
  source_discussions:
    - artifacts/reviews/2026-04-29_codex_discussion_A_data_length_regime.md
    - artifacts/reviews/2026-04-29_codex_discussion_B_split_methodology.md
    - artifacts/reviews/2026-04-29_codex_discussion_C_loss_noise.md
  current_results:
    - experiments/loss_horserace/results.csv (Stage 1 600 cells)
    - experiments/loss_horserace/results_diagnostic_price.csv (Diag 200 cells)
    - experiments/loss_horserace/per_cell_stats.csv (Stage 1 8-contrast verdict table)
    - experiments/loss_horserace/seed_diagnostic/ (Phase 1 seed analysis)
---

# Session Handoff — 2026-05-01

## Quick Resume Protocol for new session

1. Read this manifest first (frontmatter)
2. Read `progress.md` 2026-04-27-a + 2026-04-27-b for full Stage 1 + Diagnostic context
3. Read `plan.md` 2026-04-27-a + 2026-04-27-b for Path A Story C+ direction
4. Read `docs/analysis.md` 2026-04-27-a + 2026-04-27-b for full statistical results with provenance
5. Read `/Users/heruixi/.claude/plans/plan-zpp-unified-2026-04-29.md` (the primary action item — Plan Z++ v2 with all 8 Round B fixes)
6. Read `artifacts/reviews/2026-04-29_codex_plan_zpp_round_B.md` for the 8 fixes context
7. Confirm with H博士: direct execute Plan Z++ Phase 0 OR Codex Round C verify first

## Current state in 1 paragraph

The preregistered Stage 1 horse race (600 cells, mse/listmle/pairwise × {MLP, SAGE-Mean} × {S6, S8} × 5 folds × 10 seeds) and 200-cell S_price diagnostic are complete. **Verdict locked: Scenario B with fold-4 caveat** — 0/8 co-primary rejection (ΔIC + ΔSharpe gates) of ranking losses vs MSE; ListMLE shows architecture-and-feature-independent fold-4 catastrophic collapse (6/6 combos, IC ∈ [-0.36, -0.28]); pairwise shows scale collapse on 4/4 SAGE contracts under cluster bootstrap. Phase 1 seed diagnostic showed no high-IC seed in current 10-seed sample (mse seed 99 best at +0.015 mean IC across 30 fold-config combos; listmle has near-zero seed differentiation, range 0.006). Plan Z++ v2 is a comprehensive supplementary experiment plan synthesized from 3 Codex discussions (A: data length / regime; B: split methodology; C: loss + noise reduction), reviewed by Codex Round B (PROCEED-WITH-FIXES), and patched with all 8 fixes (B-01 through B-08). Awaiting H博士 final approval to execute Phase 0 (Tier 0 mandatory pre-experiment audits and fixes, ~10h dev, 0 compute).

## What the new session should do FIRST

After reading the files above, ask H博士:

> "H博士，session 已恢复。Plan Z++ v2 已 patch 完 Codex Round B 的 8 个 fixes，等您 sign-off。两个选择：(a) **direct execute** → 立即开始 Phase 0 (Tier 0 audits + manifest + sentinel test, ~10h dev, 0 compute); (b) **Codex Round C verify** → 再过一次 Codex 5-10 min 确认 fixes 正确再 execute。我推荐 (a)。您选哪个？"

DO NOT modify Plan Z++ further without H博士 explicit instruction. The plan is at the approval gate.

## What NOT to do without H博士 OK

- Don't restart Stage 1 / diagnostic experiments (they are locked)
- Don't change preregistered verdict text
- Don't begin Phase 0 / Phase A / etc. until sign-off
- Don't add new losses / experiments beyond what's in Plan Z++ v2 Tier 1/2
- Don't commit / push (per CLAUDE.md Rule 9 — only on H博士 instruction)
