---
reviewer: explore-statistics
touchpoint: closeout
round: closeout
target_files: [paper/iclr2027/main.tex]
findings:
  - {id: EXPS-CO-01, severity: MAJOR, category: statistics, claim: "SS4 M14 sentence reports only the supportive half of the trials sensitivity; C's loss of BH significance (p 6.9e-6 -> 0.059) lived only in appendices", status: FIXED, resolution_notes: "Main-text sentence restored to two-sided form with p=0.059, sign unchanged."}
  - {id: EXPS-CO-02, severity: CONCERN, category: statistics, claim: "SS5.2 regime paragraph over-broadened LOFO/LOSO coverage from the four checked contrasts to all BH-rejected ladder contrasts", status: FIXED, resolution_notes: "Enumeration (L1-L0, L2-L1, L3-L2, L5-L3) reinserted."}
  - {id: EXPS-CO-03, severity: CONCERN, category: statistics, claim: "L8 'largely shielded' upgrade (same as EXPL-CO-01)", status: FIXED, resolution_notes: "See EXPL-CO-01."}
  - {id: EXPS-CO-04, severity: CONCERN, category: statistics, claim: "BY sentence names 3 of 4 drops so 7-of-11 does not tally (inherited from acmart)", status: FIXED, resolution_notes: "C L4-L2 added; label attribution restricted to the first two."}
  - {id: EXPS-CO-05, severity: CONCERN, category: statistics, claim: "Discussion drops CI-vs-SPA disambiguation sentence", status: ACCEPTED-AS-CONCERN, resolution_notes: "Decision recorded: rely on the retained SS5.1 guard ('A non-zero per-arm IC is not the same as beating the benchmark'); page budget. Revisit if a reviewer conflates the two."}
summary: {critical: 0, major: 1, concern: 4, fixed_before_reply: 0}
overall_verdict: PASS-WITH-CONCERNS
---
# Statistics closeout audit — findings + resolutions
Verified clean: method formulas byte-identical (NW bandwidth L=6 recomputed; HLN factor 0.973; BY c(20)=3.598 recomputed); all Appendix C tables match paper_eval_robustness.csv and analysis.md cell-for-cell; SPA p-values consistent across 4 appearances; no orphaned number dependencies after compression. Full agent output preserved in session workflow transcripts; this archive records findings + resolutions.
