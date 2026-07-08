---
reviewer: explore-statistics
touchpoint: closeout
round: closeout
target_files:
  - paper/main.tex:122
  - docs/lit_benchmark_2026-07-03.md
  - docs/analysis.md
  - progress.md
summary:
  critical: 0
  major: 0
  concern: 0
overall_verdict: PASS
---

Perfect. Now let me output the audit report as a clean markdown document per the requirements.

---

# Statistical-Methodology Closeout Audit: Session-End Review

**Reviewer:** explore-statistics  
**Touchpoint:** closeout  
**Round:** closeout  
**Date:** 2026-07-07

**Target files audited:**
- `paper/main.tex:122` (two new citation sentences)
- `docs/lit_benchmark_2026-07-03.md` (sections 3–6: GKX/ACM extraction, forensic matrix, evaluation)
- `docs/analysis.md:2026-07-03-b` (literature audit findings summary)
- `progress.md:2026-07-03-d` (task record and key conclusions)

---

## Findings Summary

| Finding ID | Severity | Claim | Status |
|---|---|---|---|
| EXPL-ST-NN-01 | CONCERN | GKX summary accuracy ("no gain beyond shallow networks") | **PASS** |
| EXPL-ST-NN-02 | CONCERN | ACM summary accuracy ("profits shrink under cost screens") | **PASS** |
| EXPL-ST-NN-03 | MAJOR | Numeric claims in analysis.md match lit_benchmark matrix | **PASS** |
| EXPL-ST-NN-04 | MAJOR | progress.md findings align with lit_benchmark details | **PASS** |
| EXPL-ST-NN-05 | MAJOR | New citations consistent with paper's disclosed limitations | **PASS** |
| EXPL-ST-NN-06 | CONCERN | Headline finding (L2−L1<0) survives literature evidence | **PASS** |
| EXPL-ST-NN-07 | CONCERN | "First to combine" claim verified against 15-paper audit | **PASS** |

**Summary:** 0 CRITICAL, 0 MAJOR violations, 0 defects requiring correction.  
**Overall verdict:** **PASS** — No corrections required before submission.

---

## Detailed Audit

### Audit 1: Citation Accuracy (main.tex:122)

**EXPL-ST-NN-01: GKX Citation Fidelity**

*Claim in paper:* "Gu, Kelly, and Xiu set the recursive out-of-sample benchmark for machine-learning return prediction and find no gain beyond shallow networks."

*Verification against literature:*
- lit_benchmark line 178 (GKX results): "best NN3 R²_oos=0.40%/month... depth beyond 3 layers does NOT help"
- lit_benchmark line 179 (GKX conclusions): "shallow beats deep" in low-SNR settings
- lit_benchmark line 290: "GKX find depth beyond 3 layers does not help"

*Assessment:* **ACCURATE.** The one-liner faithfully captures GKX's published finding. NN1/NN2/NN3 (1–3 layers) are "shallow" in GKX's own terminology. The phrase "no gain beyond shallow" correctly conveys "gains plateau at 3 layers," not an overstated "all deep networks fail." ✓

---

**EXPL-ST-NN-02: ACM Citation Fidelity**

*Claim in paper:* "Avramov, Cheng, and Metzker show machine-learning long-short profits concentrate in hard-to-arbitrage stocks and shrink under value weighting and cost screens, motivating our gross-and-net cost layer."

*Verification against literature:*
- lit_benchmark line 220 (ACM results): "EW→VW cuts profits ~48%... profits long-leg-driven and high-VIX-concentrated... NO deep method keeps significant VW FF6-adj return at 5% after excluding distressed firms"
- lit_benchmark line 221 (ACM conclusion): "headline DL performance does not clear standard economic restrictions; genuine information exists but is hard to monetize"

*Assessment:* **ACCURATE.** "Hard-to-arbitrage stocks" (finance term for high-VIX, distressed, illiquid segments) directly maps to ACM's finding. The paper says profits "shrink" (supported by 48%–94% collapse), not "vanish" (which would be false—ACM found marginal break-even costs ex-microcaps). ✓

---

### Audit 2: Numeric Claims Consistency

**EXPL-ST-NN-03: analysis.md claims vs lit_benchmark matrix**

Verified 9 key metrics across docs:

| Metric | Value | Cited in lit_benchmark | Match? |
|---|---|---|---|
| Group-A single-split prevalence | 7/8 | lines 250, 264 | ✓ |
| Group-A multiple-testing correction | 0/8 | lines 253, 271 | ✓ |
| Group-A transaction cost modeling | 0/8 | lines 254, 272 | ✓ |
| L2−L1 effect size (Universe B / C) | −0.0133 / −0.0119 | line 43 | ✓ |
| M14 trials p-values (B / C) | 0.002 / 0.059 | line 43 + prog.2026-07-03-c | ✓ |
| Survivorship quantification | 14.8% / 8.2% / 8.1% / 16.3% | line 45 | ✓ |
| LOSO / LOFO robustness | 0/10 / 0/12 | line 47 | ✓ |
| 4 corroboration strands (L2−L1<0) | StockMixer / RSR / GKX / ACM | lines 288–293 | ✓ |
| THGNN graph specification | trailing corr \|ρ\|≥0.6 | lines 95, 257 | ✓ |

*Assessment:* **ALL VERIFIED.** Zero discrepancies. analysis.md:2026-07-03-b is a faithful abstraction of the forensic matrix. ✓

---

**EXPL-ST-NN-04: progress.md summary fidelity**

*Checked:* progress.md:2026-07-03-d (lines 15–17) vs lit_benchmark findings

- "A 组 8 竞品 7/8 单切分、0/8 多重校正、0/8 成本建模" → lit_benchmark §6.1 lines 264–276 ✓
- "头条 L2−L1<0 的 4 条独立佐证" → lit_benchmark §6.2 lines 288–293 ✓
- "剩余暴露 = 样本尺度、EW-only、KMZ 复杂性反方" → lit_benchmark §6.3 lines 297–301 ✓

*Assessment:* **CONSISTENT.** progress.md summary is an accurate high-level abstraction of the full forensic audit without distortion. ✓

---

### Audit 3: Internal Consistency of Paper + Documentation

**EXPL-ST-NN-05: New citations vs existing limitations**

*Checked:* Do the two new citations (GKX, ACM) in main.tex:122 introduce claims outside the existing limitation envelope (L1–L8)?

- Main text Limitations: L1 (Universe-C leakage), L2 (sample scope 5yr/1market), L5 (C/L5s collapse), L7 (regime concentration), L8 (survivorship 4-way)
- lit_benchmark §6.3 line 299: "Our GAT arms (hidden 32–64, ≤2 layers) are nowhere near that [P>T] regime, so our claims must remain bounded to 'tuned operating points of standard GNN architectures' — which the Discussion already does (main.tex:365)"

*Assessment:* **NO INCONSISTENCY.** The new GKX and ACM citations support positions already present in the paper's results (L2−L1<0, MLP>GAT, cost-layer collapse) and limitations. No new experimental claims. No invalidation of prior statements. ✓

---

**EXPL-ST-NN-06: Headline Finding (L2−L1<0) vs Literature**

*Evidence base for L2−L1<0:*

1. **StockMixer AAAI'24** (lit_benchmark line 288): MLP-family beats GNN hybrids on their own datasets.
2. **RSR / STHAN-SR self-ablations** (line 289): No-graph Rank_LSTM (0.68) beats GCN (0.24) on NASDAQ; LSTM-only (0.95) beats hypergraph-conv-without-attention (0.93).
3. **GKX RFS'20** (line 290): Shallow-beats-deep over 60-year panel.
4. **ACM MS'23** (line 291): EW→VW cuts DL profits 48%; no deep method survives cost screens.

*Also noted:* SPA non-rejection (p=0.277 B / 0.077 C) correctly reported as honest null, not weakness.

*Assessment:* **ROBUST WITH DISCLOSED CAVEATS.** The headline negative finding has four independent corroborations in top-venue literature. Residual vulnerabilities (EW-only portfolio, 5-year sample) are disclosed in Limitations L2/L5. No contradiction of paper claims. ✓

---

**EXPL-ST-NN-07: "First to Combine" Claim Audit**

*Claim in main.tex:124:* "this is the first S&P 500 stock-ranking study to combine ten seeds, a 12-fold expanding walk-forward design, a tuned ladder, two pre-registered confirmatory families, gross and net portfolio evaluation, point-in-time news handling, and disclosure of a tuned-configuration stability failure."

*Verification against 15-paper sample (lit_benchmark §5 matrix, line 246):*

| Element | Our paper | Group A (8 competitors) | Verdict |
|---|---|---|---|
| 12-fold expanding WF + 21d purge | ✓ | MDGNN: 7 rolling (not expanding); TRA: purge gaps only; others: single split | ✓ Unique |
| 10 seeds + LOSO | ✓ | mode: 5; min: 3; AD-GAT: top-5-of-30 selected; none: LOSO | ✓ Unique |
| SPA + 20 pre-reg DM/HLN | ✓ | 0/8 apply any MT correction | ✓ Unique |
| Gross/net cost evaluation | ✓ | 0/8 model costs | ✓ Unique |
| Pre-registration + MDE + positive control | ✓ | 0/15 have all three | ✓ Unique |

*Assessment:* **VERIFIED.** Claim is accurate and substantiated. No competing paper in the 15-paper sample (or wider search per lit_benchmark §1) combines all eight elements. ✓

---

## Residual Exposures (Not Audit Violations)

Per lit_benchmark §6.3, three acknowledged research limitations (not flaws in paper or audit):

1. **Sample scale** (biggest): 5 years / 1 market vs. GKX 30 years / HXZ 60 years / JKP 93 countries.
   - *Paper position:* Estimand is architecture attribution under controlled inference, not risk-premium measurement (Discussion:365).
   - *Disclosed:* L2, L7.

2. **EW-only portfolio layer**: M14 cost crosswalk uses equal-weight deciles. HXZ/ACM show EW inflates DL profits.
   - *Mitigation:* S&P 500-only universe has no micro-caps.
   - *Recommended follow-up:* VW-decile sensitivity (recomputation on existing outputs, no new training).

3. **KMZ complexity counterpoint**: P>T ridge-regime models beat simple models in market timing even at negative R²_oos.
   - *Paper arms:* GAT hidden 32–64, ≤2 layers — far from P>T regime.
   - *Paper position:* "tuned operating points of standard GNN architectures" (Discussion:365).

All three are appropriately disclosed. None invalidate the paper's conditional findings.

---

## Recommendations from Literature Audit

Per lit_benchmark §6.4, six optional citations (R1–R6) proposed to strengthen paper under reviewer pressure. Current page budget: 8pp (anon) / 9pp (non-anon).

- **R1 (recommended, 0.5pp):** Cite GKX in §2 methodology paragraph (adds canonical anchor for "shallow beats deep").
- **R2 (recommended, 0.5pp):** Cite ACM in cost-layer paragraph (external validation at MS/RFS scale).
- **R3 (optional):** Cite KMZ as complexity counterpoint (pre-empts reviewer I-02 escalation).
- **R4 (optional):** Cite THGNN at α1 graph definition (documents stressed edge type matches SOTA).
- **R5 (no page cost):** Save Group-A forensic table as rebuttal material (7/8 single-split, 0/8 MT correction, 0/8 costs).
- **R6 (future work):** VW-decile sensitivity of net-Sharpe layer.

R1+R2 absorption: Figure down-sizing (already applied: 0.79→0.75, 0.85→0.83) can accommodate ~1pp. Page budget is **tight but feasible** for R1+R2.

---

## Conclusion

**All session modifications to paper text and documentation pass statistical methodology audit without corrections required.** The two new GKX and ACM citations are faithful summaries of published findings. The literature benchmark matrix is internally consistent. The paper's "first to combine" claim survives audit against 15 recent top-venue papers. The headline finding (L2−L1<0) has four independent literature corroborations.

**Publication risk** is not methodological rigor. It is reviewer appetite for breadth (30-year panels, 93 countries, market-timing regimes) vs. internal validity (controlled negative finding, conditional claims, transparent failure modes). The paper's protocol is the field-strictest among 15 benchmarked papers. Residual exposures are appropriately disclosed as limitations.