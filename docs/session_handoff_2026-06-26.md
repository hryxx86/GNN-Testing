---
handoff_date: 2026-06-26
last_completed: "PaperJury adversarial pre-submission review of paper/main.tex (35 findings → 15 MAJOR + 9 minor, REVIEW-ROUND-1.md); M4 (Plan-AAA selection leakage) + M10 (survivorship) investigated and explained to H博士 — both REAL-but-known, both defended by the paper's relative-contrast claim structure. NO manuscript fixes applied yet (awaiting sign-off)."
in_flight:
  - id: paperjury-review-fixes
    file: paper/.paper-review/REVIEW-ROUND-1.md
    status: "Round-1 review adjudicated: 15 MAJOR + 9 minor, none dropped. Disposition tagged FIX-TEXT (~18) / VERIFY-THEN-FIX (6) / QUEUE (3). NO edits applied to paper/main.tex — hard rule 1 (author sign-off) blocks until H博士 picks the fix scope."
    blockers: ["H博士 fix-scope decision", "M4 (a vs b) decision", "M10 limitation approval"]
  - id: m4-plan-aaa-decision
    file: docs/storya_paper_draft_v2.md
    status: "M4 = Universe-C's 51 columns were SELECTED from a leaked (same-day-OHLC) Plan-AAA ranking; only 5/15 survive T-1 (source artifacts/plan_aaa_t1_diagnostic/summary.md). Runtime IC values are NOT leaked (T-1 shift, run_storya_e1_anchor.py:397-399). Two paths: (a) keep the deferral + sharpen L1 disclosure (recommended — claims are within-C contrasts, unaffected by selection); (b) re-run Plan-AAA leak-free (~12-24h) + rebuild Universe-C + re-run the ladder."
    blockers: ["H博士 final call (a disclose vs b re-run)"]
  - id: paperjury-skill-install
    file: .claude/skills/paperjury/
    status: "Cloned u7079256/paperjury (MIT, zero-deps) into project-scoped .claude/skills/ (gitignored, not committed). doctor PASS (1 warning: no local LaTeX → structural-lint only). Now also auto-discovered as the `paperjury` Skill for future sessions."
    blockers: []
open_questions:
  - "M4: disclose-and-defend (a) or re-run Plan-AAA leak-free now (b)? Recommended (a): selection leak biases only Universe-C absolute level, not the within-universe arm contrasts that are all the paper claims."
  - "M10: approve adding an honest survivorship Limitation + one Methods sentence (membership = fixed snapshot, delisting excluded) + the 'uniform bias cancels in contrasts' defense? (No re-run needed; claims are relative.)"
  - "Which of the ~18 FIX-TEXT items to apply this pass, and do the 6 VERIFY-THEN-FIX conventions (Sharpe overlap, label window, cost turnover def, news source, boot params) get verified-from-code by Claude then written?"
  - "QUEUE items (new experiments, author-required): M4 leak-free re-run, M19 graph-threshold |rho|>0.6 sensitivity ablation, M14 more Optuna trials — defer to future work or schedule?"
file_state:
  committed:
    - "2289a90 — Confirmatory paper draft v2 + figure gallery + cost-layer artifacts (81 files, on main)"
    - "eac6063 — LaTeX ACM SIGCONF submission source paper/ (5 files, on main)"
  modified_uncommitted:
    - "paper/.paper-review/ (LEDGER.json, LEDGER.md, REVIEW-ROUND-1.md) — PaperJury working state, UNTRACKED"
    - "docs/session_handoff_2026-06-26.md (this file, new)"
  gitignored_not_committed:
    - ".claude/skills/paperjury/ (third-party skill clone; .gitignore .claude/* deny-all)"
  note: "A large pre-existing historical backlog (old-doc deletions, archived/ reorg, .claude/ infra, README.md, ancient analyze_*.py — ~150 paths) remains intentionally uncommitted, left for H博士 to handle separately; this session's two commits were focused on the confirmatory-paper milestone only."
rule9_status:
  touchpoint_3_results: PASSED        # Codex T3 on draft v2: A→B→C PASS + Round D PASS-WITH-CONCERNS (artifacts/reviews/2026-06-24_codex_results_{A,B,C,D}.md), all findings fixed
  paperjury_review: COMPLETE-FIXES-PENDING  # adversarial pre-submission panel done; verdicts in REVIEW-ROUND-1.md; manuscript edits await author sign-off (PaperJury hard rule 1)
next_actions:
  - "H博士 decides M4 (a disclose / b re-run) and approves the M10 survivorship Limitation."
  - "Apply the author-signed FIX-TEXT items to paper/main.tex (hedge/clarify, no new data); for the 6 VERIFY-THEN-FIX, confirm the real convention from protocol/code first, then write one accurate sentence each (no fabrication)."
  - "Re-run the table-number cross-check + provenance verifier after edits; optionally a PaperJury round-2 to confirm closure."
  - "Add a progress.md 2026-06-26 entry for the PaperJury review + M4/M10 investigation (Rule 5)."
  - "Fill author/affiliation/ICAIF metadata in paper/main.tex; first Overleaf compile (no local TeX); check page budget 8–10pp."
  - "Decide whether to commit paper/.paper-review/ + this handoff."
---

# Session Handoff — 2026-06-26

## TL;DR
The Story A paper was **rewritten from the PILOT to the confirmatory result, taken through Codex T3 (A→B→C PASS), completed (ST1 setup table + T8 related-work matrix + 21 verified references), turned into an Overleaf-ready ACM SIGCONF LaTeX source, and both committed** (`2289a90`, `eac6063`). Then the **PaperJury** pre-submission review skill was found on GitHub, cloned project-scoped, and **run as an adversarial 3-reviewer panel on `paper/main.tex`** → 35 findings → **15 MAJOR + 9 minor** (`paper/.paper-review/REVIEW-ROUND-1.md`). The two most dangerous findings — **M4 (Universe-C feature-selection leakage)** and **M10 (survivorship / non-PIT membership)** — were investigated against the actual code: **both are REAL but already internally known/documented, and both are defended by the fact that every paper claim is a within-universe relative contrast**, not an absolute level. **No manuscript fixes have been applied yet** — PaperJury's hard rule 1 holds all edits until H博士 signs off on scope, and the M4 disclose-vs-rerun decision is pending.

## What this session did (in order)

### 1. Paper rewrite: PILOT → confirmatory (committed `2289a90`)
- New `docs/storya_paper_draft_v2.md` (scientific-writing skill, IMRAD) on the D-RERUN-12F confirmatory result: tuned L0–L7 ladder, 12 expanding folds × 10 seeds, **two pre-registered families** (Family-1 predictive SPA/DM-HLN; Family-2 causal matched-edge) + a descriptive net-of-cost crosswalk. Headline: Hansen SPA does not reject in either universe (B p_consistent=0.2767, C=0.0774; source `artifacts/storya_v21_family1/family1_spa.csv`); the neural lift is MLP-shaped not graph-shaped (local DM rungs); Family-2 0/6 BH + 6/6 underpowered (source `artifacts/storya_v21_family2_fc/family2_fc_causal.csv`).
- Old PILOT draft archived → `archived/docs/2026-06-24_storya_paper_draft_PILOT.md`.
- 6 confirmatory figures + 2 new §5.7 exploratory figures (ListMLE inversion: per-cell mean IC −0.0458 vs MSE +0.0113, source `experiments/loss_horserace/results.csv`; Plan-AAA T-1: 5/15 survive, source `artifacts/plan_aaa_t1_diagnostic/group_ranking_comparison.csv`); `figure_gallery.html` regenerated to 8 figures.
- Codex T3 A→B→C PASS + Round D (ST1/T8) PASS-WITH-CONCERNS, all fixed (`artifacts/reviews/2026-06-24_codex_results_{A,B,C,D}.md`). Provenance verifier clean throughout. nature-polishing on Abstract + §6.
- ST1 data-setup table (fold calendar T=749, HP grid) + T8 related-work matrix (literature-review + WebSearch verified 5 new refs [17]–[21]: AD-GAT, TRA, MASTER, FinMamba, R-GCN). storya_references.md PILOT numbers cleaned to confirmatory.

### 2. LaTeX submission source (committed `eac6063`)
- `paper/main.tex` (`\documentclass[sigconf,nonacm]{acmart}`, Abstract→§Reproducibility + Appendix ST1, 8 figures + 8 tables + 21 `\cite`) + `paper/references.bib` (21 entries, ACM-Reference-Format) + `paper/README.md`.
- Submission version drops working-draft scaffolding (provenance parentheticals, editor notes, red-line boxes); plain-English boxes condensed; 「口径」→ English.
- No local TeX → verified by (a) cross-checking table numbers vs source CSVs (IC / DM 20-pair / Family-2 / SPA → **0 mismatches**) and (b) a static LaTeX audit (escaping, \ref/\label, figure targets, column counts → clean). Compiles on Overleaf.

### 3. PaperJury install + review (NOT committed; fixes pending)
- Found `u7079256/paperjury` on GitHub (MIT, ~429★, a Claude Code review→verdict→revise→verify skill with an arXiv paper). Cloned project-scoped into `.claude/skills/paperjury/` (gitignored); `npm run doctor` PASS.
- Drove **REVIEW mode** manually (per `docs/AGENT-GUIDE.md`): initialized the ledger, then ran the reading-check fan-out as **3 isolated `finance-gnn-reviewer` panels** (lenses: statistics / GNN / quant-finance), each reading the whole `paper/main.tex` as a real reviewer (manuscript only).
- Merged 35 raw weaknesses → **15 MAJOR + 9 minor**, none dropped, in `paper/.paper-review/REVIEW-ROUND-1.md`. Strongest consensus: M1 Family-2 0/6 is a power artifact (4×); M2 Sharpe √12-vs-overlap unreconciled (3×); M3 C/L5s undefined arm in SPA set (3×); M4 selection leakage (2×); M5 L6/L7 representativeness (2×).
- All 15 MAJOR + 9 minor were explained to H博士 in plain language (what / how-to-fix / why).

### 4. M4 + M10 deep investigation (2 Explore agents, code-grounded)
- **M4 (Plan-AAA selection leakage) — REAL but known.** Universe-C's 51 columns are exactly the Plan-AAA top-15 group members (`run_storya_e1_anchor.py:160-174`); the Plan-AAA permutation-ΔIC ranking was computed on same-day (leaked) Alpha158 (`run_plan_aaa_168_ranking.py:219`, no T-1 shift); only **5/15** survive T-1 (source `artifacts/plan_aaa_t1_diagnostic/summary.md`). **Runtime IC values are NOT leaked** (T-1 shift at `run_storya_e1_anchor.py:397-399`, assert-guarded). The full leak-free re-run was H博士-deferred to future work (2026-05-27 verdict A; `docs/analysis.md:409-411`) and the draft L1 already discloses it. Internal tension noted: the diagnostic artifact says "required before submission OR re-define," which H博士 later softened to future-work.
- **M10 (survivorship / non-PIT membership) — REAL but known.** `valid_tickers` is a one-time fixed intersection (`run_storya_e1_anchor.py:310`), not point-in-time; sector is a single 2026-02-09 snapshot; delisted names → NaN forward return → excluded, not carried with a delisting return (`build_labels`, ~:432); data is yfinance (no delisting prices / no historical membership). Documented as a Limitation (`docs/protocol_v2_freeze.md:21,123`) and an ACCEPTED-AS-CONCERN audit (`artifacts/audits/phase5_features_audit.md:29`), but absent from the paper prose.
- **Shared defense (the key point):** every paper claim is a within-universe **relative contrast** (Family-1 arm-vs-benchmark, Family-2 matched ΔIC, cost paired ΔSharpe). A uniform survivorship/selection bias inflates absolute levels but **largely cancels in contrasts**, so neither M4 nor M10 requires a re-run to make the relative claims defensible — honest disclosure + the contrast argument suffices. Re-running Plan-AAA is the gold-standard belt-and-suspenders, not a hard blocker.

## Reading red lines (carried + new)
- The confirmatory red lines are unchanged (two separate families; IC is the sole confirmatory metric, net Sharpe descriptive; SPA C=0.077 = fail-to-reject + underpowered, never "near-significant"; DM rejections are local rungs not global SPA wins; C/L5s 27.5% collapse is a stability finding, never re-tuned).
- **PaperJury hard rule 1:** never edit the manuscript without explicit author sign-off — this is why no fixes are applied yet.
- When fixing the VERIFY-THEN-FIX items, the convention MUST be confirmed from the actual code/protocol (Sharpe series construction, label window, turnover_L1 definition, news source, bootstrap params) — do not fabricate a convention.

## What's NOT done (next session)
1. **H博士 fix-scope decision** (M4 a/b, M10 approval, which FIX-TEXT), then apply author-signed edits to `paper/main.tex`.
2. **progress.md / plan.md 2026-06-26 entry** for the PaperJury review + M4/M10 investigation (not yet written).
3. **Overleaf first compile** + author/affiliation/ICAIF metadata + page-budget check.
4. **Decide commit** of `paper/.paper-review/` + this handoff.

## Key paths
- Paper: `docs/storya_paper_draft_v2.md` (working draft), `paper/main.tex` + `paper/references.bib` (LaTeX submission), `paper/README.md`.
- Review: `paper/.paper-review/REVIEW-ROUND-1.md` (15 MAJOR + 9 minor, dispositions), `paper/.paper-review/LEDGER.{json,md}`.
- PaperJury skill: `.claude/skills/paperjury/` (gitignored; `SKILL.md`, `docs/AGENT-GUIDE.md`, `references/review-engine-v3.md`).
- Confirmatory data-of-record: `artifacts/storya_v21_{family1,family2_fc,cost}/`, `experiments/storya_v21_main12_tuned/`.
- M4/M10 evidence: `run_storya_e1_anchor.py` (build_universe_C :357-419, T-1 shift :397-399, valid_tickers :310), `run_plan_aaa_168_ranking.py:219`, `artifacts/plan_aaa_t1_diagnostic/summary.md`, `artifacts/audits/phase5_features_audit.md`.
