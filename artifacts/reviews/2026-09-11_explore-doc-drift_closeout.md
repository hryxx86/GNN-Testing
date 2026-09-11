<!-- Rule 9 session-closeout audit 4/4 (Explore agent, independent context), 2026-09-11 ~03:30 local. Scope = docs touched in
eb8314e..a903c5e + §7-implied dependencies. Statuses filled in by Claude after applying the fixes (uncommitted README edits
and the fixes below are committed together in the closeout commit). -->
---
reviewer: explore-doc-drift
touchpoint: closeout
round: closeout
target_files:
  - progress.md:1-97
  - plan.md:9-37, 2108-2112, 2119
  - docs/analysis.md:7-39
  - docs/c5_rerun_brief_2026-09-10.md
  - docs/README.md
  - experiments/README.md
  - artifacts/README.md
  - README.md
  - .gitignore:56-70
findings:
  - id: EXPL-DOC-01
    severity: CRITICAL
    category: other
    claim: "New subdir experiments/storya_v21_main12_c5_t4/ — the PRE-DECLARED PRIMARY result directory — is not indexed in experiments/README.md; the only C5 entry points at storya_v21_main12_c5/ (the Mac replicate) without saying so."
    evidence: "experiments/README.md:24-26; `ls -d experiments/storya_v21_main12_c5*` returns both; plan.md:2110 Decision Log row locks T4 = primary."
    suggested_fix: "Index storya_v21_main12_c5_t4/ as PRIMARY, relabel storya_v21_main12_c5/ as the Mac replicate, append a 2026-09-11 变更日志 line."
    status: FIXED
    resolution_notes: "experiments/README.md: both dirs indexed (PRIMARY T4 with _code_identity_t4.json; Mac 设备复现), 变更日志 2026-09-11 line added (→ progress 2026-09-11-b/-c)."
  - id: EXPL-DOC-02
    severity: CRITICAL
    category: other
    claim: "New subdir artifacts/storya_v21_family1_c5_mac/ (Mac replicate statistics) is not indexed in artifacts/README.md."
    evidence: "artifacts/README.md:21 indexes only storya_v21_family1_c5/; the _mac dir is cited as a source in docs/analysis.md:33."
    suggested_fix: "Add storya_v21_family1_c5_mac/ to the 子目录 list and a 2026-09-11 变更日志 line."
    status: FIXED
    resolution_notes: "artifacts/README.md: _c5 labelled PRIMARY (T4), _c5_mac added as Mac replicate; 变更日志 2026-09-11 line; header as-of bumped; 关键文件速查 +3 C5 rows (also closes EXPL-DOC-11)."
  - id: EXPL-DOC-03
    severity: MAJOR
    category: other
    claim: "plan.md 2026-09-10-a tri-doc line still reads `analysis: PENDING` although docs/analysis.md 2026-09-11-a exists."
    evidence: "plan.md:25 vs docs/analysis.md:9."
    suggested_fix: "plan.md:25 → analysis: 2026-09-11-a."
    status: FIXED
    resolution_notes: "plan.md 2026-09-10-a line → `analysis: 2026-09-11-a | README: experiments/README.md + artifacts/README.md 2026-09-10`."
  - id: EXPL-DOC-04
    severity: MAJOR
    category: other
    claim: "progress.md 2026-09-11-c tri-doc line still reads `analysis: PENDING（TP3 后）`; TP3 completed and docs/analysis.md 2026-09-11-a exists."
    evidence: "progress.md:31; `grep -n '## 2026-09-11-a' docs/analysis.md` → line 7."
    suggested_fix: "progress.md:31 → analysis: 2026-09-11-a."
    status: FIXED
    resolution_notes: "progress.md 2026-09-11-c line → `analysis: 2026-09-11-a | README: …`."
  - id: EXPL-DOC-05
    severity: CONCERN
    category: other
    claim: "§7 'Folder structure change' + 'New experiment script' rows fired, so tri-doc lines must name the README update; none of the new progress/plan entries carries a README field."
    evidence: "progress.md:20,31,40,49,64,77,87 and plan.md:11,25 — no `| README:` field; precedent plan.md:39, progress.md:106."
    suggested_fix: "Append `| README: …` to progress 2026-09-11-c/-d and plan 2026-09-11-a."
    status: FIXED
    resolution_notes: "README fields appended to progress 2026-09-11-c (experiments/artifacts README 2026-09-10) and 2026-09-11-d (README.md + docs/README.md + experiments/README.md + artifacts/README.md 2026-09-11), and to plan 2026-09-10-a / 2026-09-11-a."
  - id: EXPL-DOC-06
    severity: CONCERN
    category: reproducibility
    claim: "Root README.md and docs/README.md updates for the new root script / subdirs exist only as UNCOMMITTED working-tree edits at the audited tip a903c5e."
    evidence: "`git diff --name-only eb8314e..a903c5e -- README.md` → empty; `git status --porcelain README.md` → M."
    suggested_fix: "Commit README.md + docs/README.md with the C5 commits."
    status: FIXED
    resolution_notes: "Included in the closeout commit together with all closeout fixes (git status re-checked after the commit)."
  - id: EXPL-DOC-07
    severity: CONCERN
    category: statistics
    claim: "progress.md 2026-09-11-d quotes the ex-fold-9 CI as [−0.0029, +0.0182] (reviewer's 2000-rep pre-fix recompute) while the regenerated artifact says [−0.00316, +0.01788]; no source attribution. Lesser: B−C5 paired diff +0.0009 vs +0.0008 rounding."
    evidence: "progress.md:14 vs artifacts/storya_v21_family1_c5/c5_ex_fold.csv vs docs/analysis.md:31."
    suggested_fix: "Update progress.md:14 to the artifact values with a (source: …) citation."
    status: FIXED
    resolution_notes: "progress.md 2026-09-11-d now reads [−0.0032, +0.0179] with `(source: artifacts/storya_v21_family1_c5/c5_ex_fold.csv)`; B−C5 paired diff harmonised to +0.0008 in 2026-09-11-c."
  - id: EXPL-DOC-08
    severity: CONCERN
    category: other
    claim: "§8 audience leak: docs/analysis.md 2026-09-11-a carries rule-flavored MUST / MUST-NOT prose (必须披露 / 不得 / 禁止的解读)."
    evidence: "docs/analysis.md:15, :35; .claude/rules/docs.md §8 anti-pattern."
    suggested_fix: "Re-voice to descriptive narrative; keep the substance."
    status: FIXED
    resolution_notes: "Re-voiced: '披露：…', '本条不把…归因', '本文不把它们当作样本外表现引用', '本条采用的表述（TP3 许可清单）' / '本条不采用、论文也不应采用的表述（TP3 清单）'. Substance unchanged."
  - id: EXPL-DOC-09
    severity: CONCERN
    category: other
    claim: "§8 audience leak: docs/c5_rerun_brief_2026-09-10.md is addressed to the AI and written as imperatives."
    evidence: "docs/c5_rerun_brief_2026-09-10.md:1, :19, :34, :96."
    suggested_fix: "State the §8 exemption in its header (verbatim H博士 brief, audience = executing AI) or move §9 directives to .claude/rules/ or plan.md."
    status: FIXED
    resolution_notes: "Header now states the explicit §8 exemption: §0–§8 = H博士's brief kept verbatim (audience = executing AI; task record, not a rules file); §9 = implementation/deviation ledger."
  - id: EXPL-DOC-10
    severity: CONCERN
    category: reproducibility
    claim: "Brief §9.6 still says the 240 cells run on the Mac; never amended after the T4-primary decision, contradicting the committed primary."
    evidence: "docs/c5_rerun_brief_2026-09-10.md:92 vs plan.md:2110 / progress.md 2026-09-11-b."
    suggested_fix: "Add §9.11 superseding §9.6."
    status: FIXED
    resolution_notes: "§9.11 added (T4 = primary, pre-declared; Mac = replicate; tuning stays on Mac; code identity via _code_identity_t4.json)."
  - id: EXPL-DOC-11
    severity: CONCERN
    category: other
    claim: "artifacts/README.md header (as of 2026-04-20) and 关键文件速查 stale relative to its own 2026-09-10 content."
    evidence: "artifacts/README.md:7, :28-31."
    suggested_fix: "Bump as-of date; add C5 stat dirs to 关键文件速查."
    status: FIXED
    resolution_notes: "as of 2026-09-11; 关键文件速查 +3 rows (c5_comparison.md, c5_run_integrity.json, frozen_hparams_c5.json)."
  - id: EXPL-DOC-12
    severity: CONCERN
    category: other
    claim: "§7 'Experiment results produced' names experiments/<dir>/README.md if new; neither C5 run dir has one (repo precedent weak: confirmatory dirs lack one too)."
    evidence: "`ls experiments/*/README.md` → 5 of 34 subdirs."
    suggested_fix: "Either add a short README per C5 dir or accept the parent-index substitute (requires EXPL-DOC-01 fixed)."
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "Parent-index substitute adopted (consistent with the confirmatory storya_v21_main12_tuned* precedent) now that EXPL-DOC-01 is fixed; the per-dir _universe_c5.json / _run_provenance.json / _code_identity_t4.json serve as machine-readable per-dir descriptors."
summary:
  critical: 2
  major: 2
  concern: 8
  fixed_before_reply: 0
overall_verdict: PROCEED-WITH-FIXES
---

## Checks that PASSED (agent)

- Cross-reference targets: every ID named in the new entries resolves (progress 2026-09-10-a/-b/-c, 2026-09-11-a/-b/-c/-d; plan 2026-09-10-a, 2026-09-11-a; analysis 2026-09-11-a); the only unresolvable tokens were the two literal PENDINGs (now fixed).
- Review artifacts: all five exist with conformant §6 frontmatter.
- Relative-time grep: 5 hits total, 0 inside the new entries.
- §7 script→progress coupling: all nine modified/new .py files have progress coverage (2026-09-10-a; 2026-09-10-c for the 46b3b8c housekeeping). Decision Log gained four rows; footer bumped with the prior version preserved.
- Numeric provenance in docs/analysis.md 2026-09-11-a: 6/6 spot-checks have an in-paragraph (source: …) within 5 lines and every cited path exists; the ex-fold and paired rows match the CSVs exactly.
- CLAUDE.md not modified this session.
- .gitignore whitelist block glob-correct for primary and replicate dirs, with the naming mapping commented.

## Files/sections read (agent)

Full: docs/c5_rerun_brief_2026-09-10.md, docs/README.md, experiments/README.md, artifacts/README.md, README.md, .claude/rules/docs.md §§1–8, .gitignore diff. Sections only: progress.md:1-97, plan.md:9-37 + :2108-2112 + :2119, docs/analysis.md:7-39. Grep/tail only: artifacts/reviews/2026-09-11_* frontmatter, c5_ex_fold.csv, c5_paired_contrast.csv, git log/status/diff.

Out of scope (noted only): pre-existing uncommitted working-tree changes from earlier sessions (analyze_e1_lofo.py, paper_figs/fig_family2.py, figures/family2_edge_causal.*, ~20 deleted/moved archived/ and docs/ files, untracked .claude/ tree).
