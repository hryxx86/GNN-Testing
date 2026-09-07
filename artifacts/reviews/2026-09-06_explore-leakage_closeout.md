---
reviewer: explore-leakage
touchpoint: closeout
round: closeout
target_files: [paper/iclr2027/main.tex]
findings:
  - {id: EXPL-CO-01, severity: MAJOR, category: data-leakage, claim: "Main-text L8 'paired differences largely shielded' overstates shielding — source and own Appendix D deny it for the look-ahead side", status: FIXED, resolution_notes: "L8 rewritten to two-sided form: pairing reduces mechanical imbalance but does not make estimand PIT and does not remove look-ahead sampling bias."}
  - {id: EXPL-CO-02, severity: CONCERN, category: data-leakage, claim: "News-graph parameter-free claim retained but its three substantiating facts (provider ticker tags / multi-ticker co-mention / daily cutoff only) deleted, not moved", status: FIXED, resolution_notes: "New Appendix A subsection 'News-graph construction' with the acmart sentence verbatim + pointer added in SS4."}
  - {id: EXPL-CO-03, severity: CONCERN, category: data-leakage, claim: "Dropped the exonerating 'magnitude-matched Universe-B contrast does not survive BH' clause — overstates leak attribution", status: FIXED, resolution_notes: "Clause restored in SS5.2 with +0.0143."}
  - {id: EXPL-CO-04, severity: CONCERN, category: data-leakage, claim: "SS4 M14 summary omits that Universe-C falls below significance", status: FIXED, resolution_notes: "Same fix as EXPS-CO-01."}
  - {id: EXPL-CO-05, severity: CONCERN, category: data-leakage, claim: "'within-universe' scope qualifier dropped from L1 exposure claim (and Appendix D)", status: FIXED, resolution_notes: "Qualifier reinserted in both places."}
  - {id: EXPL-CO-06, severity: CONCERN, category: prior-art, claim: "THGNN graph-construction claim stated as property, not as-reported", status: FIXED, resolution_notes: "Hedged to 'a construction that, as described, uses no future or full-sample information'."}
summary: {critical: 0, major: 1, concern: 5, fixed_before_reply: 0}
overall_verdict: PROCEED-WITH-FIXES
---
# Leakage-claims closeout audit — findings + resolutions
Temporal-alignment claims verified clean (T-1/purge/trailing-graph/PIT-news all verbatim vs source). Universe-C leak disclosure consistent at all sites. Appendix B other-paper claims accurate vs lit_benchmark. Appendix D survivorship numbers exact. All six findings fixed same session; 9pp limit re-verified after fixes. Full agent output preserved in session workflow transcripts; this archive records findings + resolutions.
