---
review_id: CODEX-RR-EDGE-A-01
date: 2026-05-27
reviewer: codex
touchpoint: 3
round: A
target: E3+E4 edge ablation (E6 post-process)
verdict: PASS-WITH-CONCERNS
findings:
  - id: CODEX-RR-EDGE-A-01
    severity: CONCERN
    category: results_interpretation
    title: "Full-sample negative inference is valid but needs directional caveat"
    body: >-
      The 0/5 BH-FDR rejection result is legitimate evidence against a general edge-augmentation benefit in the pre-registered full condition. The raw HLN p=0.039 for alpha3 versus alpha1 is not ignorable, but under the Benjamini-Hochberg step-up rule (Benjamini and Hochberg, 1995), with m=5 and q=0.05, the rank-1 threshold is (1/5)*0.05=0.010, so p=0.039 fails to reject. The paper should not describe this as "no signal"; it should describe it as a directional trend that does not survive the pre-specified multiplicity control.
    recommendation: >-
      Keep the pre-registered Template 3 negative narrative, but add one sentence noting that alpha3 versus alpha1 had the smallest raw p-value and remains only directional after BH-FDR.
    status: OPEN
  - id: CODEX-RR-EDGE-A-02
    severity: MAJOR
    category: results_interpretation
    title: "Fold-4 IC claim requires temporal label and bounded scope"
    body: >-
      The fold-4-only positive IC intervals can be described as a Q2-2025 regime-specific signal only if fold 4 is indeed Q2-2025 in the paper's fold calendar. If that mapping is not shown, the text should say "fold-4 regime" rather than "Q2-2025." Even if fold 4 is Q2-2025, T=62 days and N=10 cells per arm support a diagnostic table but not a general statement that edge augmentation works in that regime.
    recommendation: >-
      Verify fold-4 calendar mapping before labeling as Q2-2025. Cap language to "localized fold-4 regime signal" regardless.
    status: OPEN
  - id: CODEX-RR-EDGE-A-03
    severity: CONCERN
    category: results_interpretation
    title: "Fold-4 alpha4 vs alpha2 Sharpe negative interval lacks cell-level support"
    body: >-
      The alpha4 versus alpha2 fold-4 Sharpe interval [-0.005,-0.002] should not be stated as "news on top of sector hurts" without determining whether it is driven by one or two outlier cells. Cell-level Sharpe differences or leave-one-cell diagnostics would be needed for that determination. The defensible claim is narrower: in the fold-4 Sharpe diagnostic, corr+sector+news underperformed corr+sector.
    recommendation: >-
      Describe as "fold-4 Sharpe diagnostic instability" rather than a directional news-harm claim. Add to Limitations.
    status: OPEN
  - id: CODEX-RR-EDGE-A-04
    severity: INFO
    category: statistical
    title: "BH-FDR application is correct"
    body: >-
      Under the Benjamini-Hochberg (1995) step-up rule with m=5, q=0.05, rank-1 critical value = (1/5)*0.05 = 0.010. The smallest raw HLN p=0.039 > 0.010, so 0/5 rejections is correct. This is also stricter than Bonferroni (0.05/5=0.010 for rank-1), so the claim is robust.
    recommendation: No action required.
    status: ADDRESSED
  - id: CODEX-RR-EDGE-A-05
    severity: MAJOR
    category: statistical
    title: "Fold-4 Sharpe bootstrap N=10 is too small for reliable percentile CI"
    body: >-
      Block_size=1 reduces to i.i.d. bootstrap over N=10 cells. Percentile bootstrap coverage is asymptotic (Efron 1979; Efron and Tibshirani 1993) and unreliable at N=10 due to skew, leverage points, and shared-fold dependence. The negative alpha4 vs alpha2 Sharpe interval should not be described as statistically significant.
    recommendation: >-
      Remove "statistically significant" language for fold-4 Sharpe intervals. Describe as descriptive diagnostics in both Results and Limitations.
    status: OPEN
  - id: CODEX-RR-EDGE-A-06
    severity: CONCERN
    category: statistical
    title: "Global SPA is not required for the edge ablation conclusion"
    body: >-
      The five-pair BH-FDR family is the pre-registered confirmatory inference unit for E3/E4. Global SPA or Reality Check (White 2000; Hansen 2005) would only be needed for claims selected across the full ~1571-cell ledger. The local negative conclusion does not require global adjustment, but any exploratory claims drawn from the broader project do.
    recommendation: >-
      Clarify in Methods that BH-FDR applies to the five edge-ablation contrasts only. Any cross-experiment claims should be labeled exploratory or subject to separate data-snooping adjustment.
    status: OPEN
---

## Executive Summary

**Verdict: PASS-WITH-CONCERNS**

The pre-registered full-sample result supports the Template 3 negative finding: no edge-augmentation contrast survives BH-FDR across the five-pair family. The main paper risk is language: fold 4 shows a real regime diagnostic signal in IC, but the T=62, N=10 setting and LOFO-4 collapse mean it should be presented as localized evidence, not as a reversal of the full-sample conclusion. Two MAJOR findings require text changes before submission; no reanalysis is needed.

---

## Area 1 — Results Interpretation

### Finding CODEX-RR-EDGE-A-01 [CONCERN] Full-sample negative inference

**Assessment**: The 0/5 BH-FDR result is legitimate evidence against edge augmentation generality within the pre-registered five-pair family. Valid.

**Caveat required**: The smallest raw HLN p=0.039 (alpha3 vs alpha1) is below the unadjusted 0.05 level and aligns with the point estimate (alpha3 IC=0.041 vs baseline 0.032). The paper should not say "no signal"; it should say the directional trend does not survive pre-specified multiplicity control. This keeps Template 3 intact while being honest.

**Action**: In the §Results paragraph, after stating 0/5 BH-FDR rejections, add one parenthetical noting the smallest raw p and the BH rank-1 threshold.

---

### Finding CODEX-RR-EDGE-A-02 [MAJOR] Fold-4 IC claim scope

**Assessment**: The fold-4-only positive IC intervals (all excluding zero) are a legitimate regime-diagnostic signal within Fold 4. However, two conditions apply:

1. The "Q2-2025" label requires the fold calendar to map fold 4 to that period. If this is not shown in Table 5 or a footnote, the text must say "fold-4 regime" rather than "Q2-2025."
2. N=62 days × N=10 cells is adequate for a diagnostic table with bootstrap intervals, but not for a claim that "edge augmentation works in this regime" at a generalization level.

**Action**: Verify fold-4 dates. Cap all fold-4 language to "localized fold-4 regime signal" or "fold-4 diagnostic."

---

### Finding CODEX-RR-EDGE-A-03 [CONCERN] Fold-4 Sharpe negative interval epistemic status

**Assessment**: The alpha4 vs alpha2 Sharpe interval [-0.005,-0.002] is the only Sharpe interval that excludes zero. Describing this as "news on top of sector hurts" is too strong without cell-level diagnostics (e.g., leave-one-cell). Given N=10 cells and block_size=1 bootstrap (see Area 2), this interval has weak nominal coverage guarantees.

**Appropriate epistemic status**: "In the fold-4 Sharpe diagnostic, corr+sector+news (alpha4) underperformed corr+sector (alpha2), suggesting instability when news edges are layered onto sector edges in this fold. This finding is exploratory and should not be generalized."

**Action**: Add to Limitations. Do not use as a headline finding.

---

## Area 2 — Statistical Validity

### Finding CODEX-RR-EDGE-A-04 [INFO] BH-FDR is correctly applied

Under Benjamini and Hochberg (1995) step-up rule:
- Sort p-values: p_(1) ≤ p_(2) ≤ ... ≤ p_(m)
- Reject up to k = max{i : p_(i) ≤ (i/m) × q}
- m=5, q=0.05, rank-1 critical value = (1/5) × 0.05 = 0.010

Smallest raw HLN p=0.039 > 0.010 → 0 rejections. This is also stricter than Bonferroni (0.05/5=0.010 for rank-1 comparison), confirming robustness. No error in BH-FDR application.

---

### Finding CODEX-RR-EDGE-A-05 [MAJOR] Fold-4 Sharpe bootstrap N=10

**Problem**: Block_size=1 reduces to i.i.d. bootstrap over N=10 cells. Percentile bootstrap coverage is asymptotic (Efron 1979; Efron and Tibshirani 1993, §13). At N=10, the empirical distribution has only 10 distinct values, making percentile CI sensitive to skew and leverage points. Additionally, cells within the same fold share factor exposure, so exchangeability assumptions may not hold.

**Consequence**: The alpha4 vs alpha2 Sharpe interval [-0.005,-0.002] cannot be described as "statistically significant" or treated as a formal confidence interval. It is a descriptive diagnostic only.

**Action**: (1) Remove "statistically significant" or any confidence-language from fold-4 Sharpe claims. (2) Add to Limitations that fold-4 Sharpe bootstrap is descriptive due to N=10.

---

### Finding CODEX-RR-EDGE-A-06 [CONCERN] Global SPA and trial ledger

**Assessment**: The five-pair BH-FDR family is the pre-registered confirmatory inference unit. Under this framing, E3/E4 edge-ablation inference is self-contained and does not require global SPA (White 2000 Reality Check; Hansen 2005 SPA test) to be valid as a negative finding.

**When global SPA would apply**: If the paper selects any claim from across the ~1571-cell project ledger (e.g., "the best configuration we tested beats the baseline"), that claim is subject to data-snooping and should be labeled exploratory or receive a separate SPA-style adjustment.

**Action**: Add one sentence to Methods clarifying that BH-FDR covers only the five pre-registered edge-ablation contrasts. Cross-experiment exploratory claims should be labeled accordingly.

---

## Area 3 — Paper Language Recommendations

### 3(a) Proposed §Results Paragraph (3 sentences max)

> "Across the pre-registered edge-ablation family on Univ B, edge augmentation did not produce a statistically reliable full-sample gain over the E1 SAGE baseline: none of the five DM/HLN contrasts survived BH-FDR control at q=0.05 (smallest raw HLN p=0.039 for corr+news_cooccur versus baseline; rank-1 BH threshold=0.010), consistent with the pre-registered Template 3 interpretation that multi-relation graph construction added overhead without a generalizable ranking benefit, though the corr+sector and corr+news_cooccur point estimates were directionally higher in mean IC (0.041 versus 0.032). Regime diagnostics required by the study design show a localized fold-4 IC signal—edge-variant bootstrap intervals excluding zero versus baseline—but LOFO-4 (leaving fold 4 out) collapses the full-sample delta IC to +0.002 to +0.005 with all HLN p>0.30, confirming the fold-4 signal is regime-concentrated rather than general. Fold-4 Sharpe diagnostics are treated as descriptive only, given the small cell count per arm (N=10)."

---

### 3(b) Proposed §Limitations Additions

Beyond plan §1.9 items 1-7, add the following:

**L-8**: "The fold-4 regime diagnostics are based on a single 62-day fold and ten cells per arm. Positive fold-4 IC bootstrap intervals should be interpreted as localized regime concentration evidence rather than full-sample robustness."

**L-9**: "The fold-4 Sharpe per-cell bootstrap uses block_size=1 (i.i.d. resampling over N=10 cells). Percentile bootstrap coverage is unreliable at N=10; fold-4 Sharpe intervals, including the negative alpha4-alpha2 interval, are descriptive diagnostics and should not be interpreted as formal confidence statements."

**L-10**: "The five-pair BH-FDR family controls multiplicity for the pre-registered edge-ablation contrasts only. Any claim selected from the broader project trial ledger (~1571 cells across E1, E3, E4, and historical exploration) would require separate data-snooping adjustment and should be treated as exploratory."

---

### 3(c) Universe B vs Universe C sufficiency

**Assessment**: Univ B alone is sufficient for paper completeness given the LOFO-4 + Fold-4 framing, which provides a two-level regime stability analysis. The negative full-sample finding plus the localized fold-4 diagnostic constitute a complete inferential story. Univ C extension is appropriately deferred.

**Recommended text**: Add one sentence to the Limitations or Future Work section: "Edge ablation results are reported for Univ B (large-cap); generalizability to broader universes, including Univ C, remains future work."

---

## Final Verdict

**PASS-WITH-CONCERNS**

The E3+E4 edge ablation results support paper submission with the Template 3 negative finding as the primary conclusion. Two MAJOR text changes are required:

1. (CODEX-RR-EDGE-A-02) Verify fold-4 calendar label and cap fold-4 language to "localized regime signal."
2. (CODEX-RR-EDGE-A-05) Remove significance language from fold-4 Sharpe intervals; describe as descriptive diagnostics.

Three CONCERN items (directional caveat for p=0.039, Sharpe negative interval epistemic status, global SPA scope) require text additions but do not block submission.

No reanalysis, hyperparameter changes, or Universe C extension is recommended. The results are paper-ready after the above text edits.
