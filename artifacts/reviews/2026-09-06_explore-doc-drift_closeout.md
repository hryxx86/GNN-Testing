---
reviewer: explore-doc-drift
touchpoint: closeout
round: closeout
target_files:
  - progress.md:1-120
  - plan.md:1-120
  - paper/README.md
  - paper/iclr2027/README.md
  - paper/iclr2027/main.tex
findings:
  - id: EXPD-CO-01
    severity: MAJOR
    category: other
    claim: "All six deadline records collapse ICLR 2027 to a single '截稿 9/18' with no abstract-vs-full distinction; per iclr.cc CFP 9/18 = abstract, 9/25 = full paper. plan.md actively discarded the correct full-paper date ('替代此前 ~9/24 估计')."
    evidence: "plan.md:13,17; progress.md:9,16; paper/README.md:3; paper/iclr2027/README.md:3; contradicts legacy plan.md:47 / progress.md:158 two-deadline records"
    suggested_fix: "Record both dates everywhere; split plan step 3 into abstract (9/18) and full (9/25)."
    status: FIXED
    resolution_notes: "Fixed same session: both dates recorded in plan.md 2026-09-06-a, progress.md 2026-09-06-a, both READMEs."
  - id: EXPD-CO-02
    severity: MAJOR
    category: other
    claim: "Relative-time leak in post-2026-09-01 entry: progress.md 2026-09-06-a quotes '今天写完' whose referent (2026-09-01) differs from entry date (2026-09-06)."
    evidence: "progress.md:9"
    suggested_fix: "Anchor referent inside the quote: 今天（= 2026-09-01）."
    status: FIXED
    resolution_notes: "Fixed same session."
  - id: EXPD-CO-03
    severity: CONCERN
    category: other
    claim: "R6 skip rationale imprecise: recorded as '无数据' but data/reference/sp500_market_caps.csv exists (dateless end-of-sample snapshot); real blocker is PIT look-ahead if used per-day; analysis.md:51 still says R6 '现有输出可重算'."
    evidence: "progress.md:11; plan.md:13; docs/analysis.md:51; data/reference/sp500_market_caps.csv (503 rows, dateless); analyze_m_scout_step1.py:41"
    suggested_fix: "Reword rationale to dateless-snapshot/PIT-leak constraint; qualify analysis.md:51."
    status: FIXED
    resolution_notes: "Verified by Claude (read analysis.md:51, CSV head, m_scout PIT note) then reworded in progress.md/plan.md. analysis.md:51 is a pre-2026-09 legacy entry — left as historical record; constraint documented in progress.md 2026-09-06-b."
  - id: EXPD-CO-04
    severity: CONCERN
    category: other
    claim: "Tri-doc cross-ref lines omit README coupling required by docs.md §1 Exception for the §7 folder-structure row."
    evidence: "progress.md:18; plan.md:11"
    suggested_fix: "Extend lines with README update date."
    status: FIXED
    resolution_notes: "Fixed same session."
  - id: EXPD-CO-05
    severity: CONCERN
    category: reproducibility
    claim: "paper/iclr2027/main.tex+pdf carry uncommitted post-audit edits, so README '校验状态' certifies the committed artifact, not the working tree. Agent re-verified equivalence: numeric-token diff empty, 23pp, statements on p10, 30 bib entries — prose-only delta."
    evidence: "git status M paper/iclr2027/main.tex (+20/-10); mtimes"
    suggested_fix: "Commit working tree; note prose-only zero-numeric-delta edit in README changelog."
    status: FIXED
    resolution_notes: "README changelog line added; committed at closeout end."
summary:
  critical: 0
  major: 2
  concern: 3
  fixed_before_reply: 0
overall_verdict: PROCEED-WITH-FIXES
---

# Doc Drift Audit — closeout (agent full notes)

§7 coupling intact (no CRITICAL): paper/iclr2027/README covers all 8 files; parent README announces subdir; progress 2026-09-06-a exists and describes reality apart from EXPD-CO-03. Cross-ref targets resolve. Relative-time grep: all other hits are pre-2026-09-01 legacy (noted, not flagged). Accuracy spot-checks (a),(b),(d) corroborated. §8 audience check clean. Pre-existing nit (not counted): both READMEs lack 关键文件速查 section.
