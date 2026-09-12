---
handoff_date: 2026-05-20
last_completed: "2026-05-20-a: 10-seed expansion + finalize.v3 — Tier 1.D verdict FLIPS to NULL; 0/36 BH-FDR cumulative; 7 nulls + 3 mechanism findings; tri-doc updated"
in_flight:
  - id: paper-v3-rewrite
    file: docs/paper_draft_2026-05-18_v2.md (to become paper_draft_2026-05-XX_v3.md)
    status: "deferred to next session; v2 contains 5-seed numbers + Tier 1.D 'marginal positive' framing; needs substantial rewrite per 2026-05-20-a verdict changes"
    blockers:
      - "Substantial rewrite needed: all Tier 1.B/1.C/1.D/1.A numeric tables → 10-seed; abstract numbers; §1.2 contributions drop 'marginal regularization positive'; §4.5/§4.7 Tier 1.D rewrite; §6.3 practitioner recommendation rewrite"
  - id: codex-touchpoint-final
    file: artifacts/reviews/(forthcoming)
    status: "Codex Touchpoint 2 (code) + Touchpoint 3 (results) on final 10-seed analyses + paper v3 not yet run; Codex quota was unavailable across 2026-05-13 / 14 / 18 attempts"
    blockers:
      - "Codex quota status (last seen quota-exhausted; check at start of next session)"
      - "If Codex still unavailable, claude-self-review fallback per 2026-05-02 / 2026-05-13 precedent is appropriate"
open_questions:
  - "Paper title: 3 alternatives in v2 §Title section; pick one for v3"
  - "Stage 1 integration: combined 36-contrast paper or separate references? v2 references Stage 1 as separately preregistered"
  - "Sharpe with raw fwd_ret (vs z-score proxy): ~5 min compute on existing preds; switch for headline table?"
  - "Anchored RankNet σ-guard 'mechanism failure' framing: is it strong enough as a standalone contribution, or supplementary?"
file_state:
  modified_since_last_commit:
    - progress.md (2026-05-18-a + 2026-05-19-a + 2026-05-20-a entries appended)
    - plan.md (Decision Log row + 2026-05-20-a section appended)
    - docs/analysis.md (2026-05-18-a + 2026-05-20-a entries appended, supersedes any in-flight 2026-05-18 5-seed numbers)
    - run_tier1_phase_a.py (SEEDS_5 + SEEDS_3 → 10 seeds)
    - run_tier1a_phase_b.py (SEEDS_5 → 10 seeds)
    - run_tier1c_phase_b.py (SEEDS_5 → 10 seeds)
    - run_tier1b_h2_phase_b.py (SEEDS_5 → 10 seeds)
    - analyze_tier1_phase_a.py (SEEDS_5 + seeds_3 → 10 seeds)
    - analyze_phase_b_finalize.py (SEEDS_5 → 10 seeds)
    - analyze_tier2c_sector_ic.py (5 → 10 seed lists, 4 EXPERIMENTS rows)
    - analyze_tier1e_regime_forensic.py (seeds_tier1 → 10 seeds)
    - artifacts/tier1_phase_a/results.csv (1,204 rows now: 4 smoke + 800 tier1b + 400 tier1d)
    - artifacts/tier1a_phase_b/results.csv (200 rows)
    - artifacts/tier1c_phase_b/results.csv (400 rows)
    - artifacts/tier1b_h2_phase_b/results.csv (800 rows)
    - artifacts/tier1_phase_a/stat_per_cell.csv + stat_tier1d.csv (10-seed values)
    - artifacts/phase_b_finalize/stat_tier1b_h2.csv + stat_tier1a.csv + stat_tier1c.csv (10-seed values)
    - artifacts/phase_b_finalize/ic_sector_resid_per_cell.csv (Tier 2.C, 10-seed for Phase A/B + 10-seed for Stage 1)
    - artifacts/phase_b_finalize/tier1e_regime_forensic.csv (Tier 1.E, 10-seed for Phase A/B targets)
  new_files:
    - run_tier1a_phase_b.py (Tier 1.A runner, written 2026-05-13)
    - run_tier1c_phase_b.py (Tier 1.C runner)
    - run_tier1b_h2_phase_b.py (Tier 1.B h2 runner)
    - analyze_phase_b_finalize.py (Phase B unified statistical analysis)
    - analyze_tier2c_sector_ic.py (Plan §2.C IC_sector_resid)
    - analyze_tier1e_regime_forensic.py (Plan §1.E regime stratification)
    - artifacts/tier1a_phase_b/preds/*.npy (200 cells)
    - artifacts/tier1c_phase_b/preds/*.npy (400 cells)
    - artifacts/tier1b_h2_phase_b/preds/*.npy (800 cells)
    - artifacts/phase_b_finalize/ (new dir with 3 stat CSVs + Tier 2.C + Tier 1.E outputs + stat_report.md + logs)
    - artifacts/reviews/2026-05-13_claude-self-review_code_phase_a5.md
    - artifacts/reviews/2026-05-13_claude-self-review_results_phase_a5.md
    - artifacts/tier1_phase_a/preds/tier1b_*.npy (400 new at 10-seed) + tier1d_*.npy (280 new at 10-seed)
    - docs/paper_draft_2026-05-13.md (v0 paper draft, 5-seed)
    - docs/paper_draft_2026-05-14_v1.md (v1 paper, 5-seed)
    - docs/paper_draft_2026-05-18_v2.md (v2 paper, 5-seed, adds Tier 2.C + 1.E; OUTDATED Tier 1.D framing — needs v3)
rule9_status:
  touchpoint_1_plan_zpp_v2: PASSED (Codex Round B PROCEED-WITH-FIXES, all 8 fixes applied 2026-04-29)
  touchpoint_2_phase_a5_code: claude-self-review fallback (2026-05-13_code_phase_a5.md PASS-WITH-CONCERNS, 0 CRITICAL + 1 MAJOR + 4 CONCERN + 5 PASS); Codex pass deferred
  touchpoint_3_phase_a5_results: claude-self-review fallback (2026-05-13_results_phase_a5.md PASS-WITH-CONCERNS); Codex pass deferred
  touchpoint_2_phase_b_code: PENDING (Codex unavailable since 2026-05-13)
  touchpoint_3_phase_b_results: PENDING; Codex stop-time review 2026-05-06 corrected one Tier 1.D framing issue (h0 → h2 baseline correction)
  touchpoint_2_3_10seed_finalize: PENDING (this session 2026-05-20)
next_actions:
  - "Verify Codex quota at session start (try `/codex-code-review` or check claude-code-guide); resets at known PT times"
  - "Paper v3 rewrite: 10-seed numbers + Tier 1.D verdict revocation. Estimate 2-4h of writing. Start from docs/paper_draft_2026-05-18_v2.md as base."
  - "Codex Touchpoint 2 (code review) on the 3 new run scripts + 4 analyze scripts that were patched"
  - "Codex Touchpoint 3 (results review) on the 10-seed cumulative null + Tier 1.D flip + Tier 1.E rejected hypothesis"
  - "If H博士 directs publication: pick title, pick Stage 1 integration framing, polish abstract, generate figures (3 tables in v2 already; consider σ_fold vs mean_IC scatter for Tier 1.D, fold-4 bar chart, sentinel diagram)"
  - "Sharpe with raw fwd_ret recompute (~5 min) if H博士 wants headline-grade Sharpe"
key_context_notes:
  - "H博士 communication: Chinese conversation, English code/reports (Rule 3)"
  - "H博士 autonomous-work-mode memory note: long autonomous sessions allowed, maximize output, record everything"
  - "10-seed expansion was H博士 directive 2026-05-18 — overrides Plan §1.B Gate 5→10 conditional expansion rule"
  - "Tier 1.D 5-seed → 10-seed verdict flip is the MOST IMPORTANT new finding this session: marginal positive → FULL NULL; demonstrates seed-expansion as necessary robustness check"
  - "All 10-seed numbers in tri-doc supersede earlier 5-seed numbers; paper v2 inherits 5-seed numbers and is OUTDATED"
  - "Stage 1 always had 10 seeds (separately preregistered horse race); Phase A/B all expanded to 10 seeds this session"
  - "Codex Touchpoints all failed due to quota; claude-self-review fallback used per 2026-05-02 precedent (H博士 explicit authorization '不能用就自己检查')"
  - "Total Phase A+B cells: 2,604; total cells including Stage 1 (600) = 3,204"
  - "Compute: ~43h M4 wall clock for the 4 10-seed expansion chains, mostly overnight autonomous"
critical_paths:
  paper_v2_outdated: docs/paper_draft_2026-05-18_v2.md
  phase_b_finalize_report: artifacts/phase_b_finalize/stat_report.md (5-seed numbers; needs update for v3)
  stat_csvs_10seed:
    - artifacts/tier1_phase_a/stat_per_cell.csv (Tier 1.B Adam)
    - artifacts/tier1_phase_a/stat_tier1d.csv (Tier 1.D, the FLIPPED verdict)
    - artifacts/phase_b_finalize/stat_tier1b_h2.csv
    - artifacts/phase_b_finalize/stat_tier1a.csv
    - artifacts/phase_b_finalize/stat_tier1c.csv
    - artifacts/phase_b_finalize/ic_sector_resid_per_cell.csv
    - artifacts/phase_b_finalize/tier1e_regime_forensic.csv
  preds_dirs:
    - experiments/loss_horserace/preds (Stage 1, 600 cells)
    - artifacts/tier1_phase_a/preds (Tier 1.B Adam + 1.D + smoke, 1,204 cells)
    - artifacts/tier1a_phase_b/preds (200)
    - artifacts/tier1c_phase_b/preds (400)
    - artifacts/tier1b_h2_phase_b/preds (800)
  fold_manifests:
    - data/reference/fold_manifest_expanding.json (5 folds)
    - data/reference/fold_manifest_roll2y.json (Tier 1.A only)
  pre_registration: /Users/heruixi/.claude/plans/plan-zpp-unified-2026-04-29.md
---

# Session Handoff — 2026-05-20

## Quick Resume Protocol for new session

1. Read this manifest first (frontmatter above)
2. Read `progress.md` 2026-05-18-a + 2026-05-19-a + **2026-05-20-a** for full Tier 2.C + Tier 1.E + 10-seed expansion + Tier 1.D flip context
3. Read `plan.md` 2026-05-20-a Decision Log for binding verdict updates
4. Read `docs/analysis.md` 2026-05-20-a for full statistical detail of 10-seed results
5. Read `artifacts/tier1_phase_a/stat_tier1d.csv` (the FLIPPED verdict, 8 rows × 9 cols)
6. Compare `docs/paper_draft_2026-05-18_v2.md` (v2, 5-seed, OUTDATED) against the 10-seed verdicts; identify all tables/numbers/claims to update for v3.

## Current state in 1 paragraph

The 10-seed expansion for all Phase A/B experiments (per H博士 2026-05-18 directive) completed across 4 background chain-launches over ~43h M4 wall clock (2026-05-18 04:44 PT → 2026-05-20 00:38 PT). +1,380 new cells; 2,604 total Phase A/B cells. 10-seed re-analysis exposed a **major flip in Tier 1.D**: the 5-seed "marginal positive regularization" verdict (h2 NW p=0.059) was a 5-seed selection artifact — at 10-seed ALL 4 hparam configs are Score-NEGATIVE with all NW p > 0.5 vs the 10-seed baseline. Other verdicts are stable or strengthened: Tier 1.B Adam fold-4 NW-significant negative went 8/12 → 10/12; Tier 1.A ListMLE fold-4 attenuation went p=0.009 → p<0.001; Tier 1.C Gate 1.C is still 0/4 (σ-guard universal failure robust); Tier 1.E ListMLE primary gate is still 0/4 (regime hypothesis stably rejected). Cumulative: **0/36 BH-FDR rejections + 7 nulls + 3 mechanism findings + 1 regime-conditional finding**. Tri-doc updated (progress + plan + analysis 2026-05-20-a entries). Paper v3 deferred to next session for substantial rewrite. Codex Touchpoint 2/3 deferred due to ongoing quota issues; claude-self-review fallback remains the established backup per H博士 precedent.

## What the new session should do FIRST

1. **Read this handoff manifest in full** (frontmatter + this prose).
2. **Verify Codex quota status** — if available, run final Touchpoint 2 + 3 on the 10-seed code/results before paper polish. If unavailable, plan a comprehensive claude-self-review at session end.
3. **Ask H博士** about paper v3 priorities:
   - Title choice (3 alternatives in v2 §Title)
   - Stage 1 integration (combined 36-contrast paper or current separately-referenced framing)
   - Whether to recompute Sharpe with raw fwd_ret (5 min) before drafting v3
4. **Start paper v3 rewrite** based on `docs/paper_draft_2026-05-18_v2.md`:
   - Abstract numbers
   - §1.2 Contributions: drop "marginal regularization positive" contribution; replace with "robustness check via seed-expansion exposed 5-seed artifact"
   - §4.5 + §4.7 Tier 1.D sections: rewrite from "marginal positive" to "FULL NULL"
   - §6.3 Tier 1.D practitioner recommendation: replace
   - All Tier 1.B / 1.C / 1.A / 1.D numeric tables → 10-seed values
   - §5 Mechanism: counts updated (Tier 1.B Adam fold-4 8/12 → 10/12; Tier 1.B-h2 fold-4 11/12 → 9/12)

## What NOT to do without H博士 OK

- Don't re-launch any experiments (10-seed is final; 20-seed not authorized)
- Don't push or commit (per CLAUDE.md Rule 9)
- Don't add new experiments or losses beyond what's in Plan Z++ v2 Tier 1/2 (Tier 2.A/B/D remained conditional-skipped per registered protocol)
- Don't merge paper v2 into v3 — write a separate v3 file, keep v2 for archival
- Don't claim Tier 1.D supports regularization (the 5-seed verdict is REVOKED)
- Don't combine 5-seed and 10-seed numbers in any single table (use one or the other; current paper v2 has 5-seed; v3 should be pure 10-seed)
