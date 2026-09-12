---
reviewer: finance-gnn-reviewer
touchpoint: results
round: B
target_files:
  - paper/main.tex
panels: [statistics, gnn, quant-finance]
findings:
  - id: FINGNN-R2-qf-01
    severity: MAJOR
    category: survivorship
    claim: "L8 part (ii) 'liquid large-cap framing holds' conflates liquidity-benignity with estimand-benignity; the 8.2% omitted stock-days include SIVB/SBNY/FRC (2023 bank failures, FDIC receivership) — the test span's highest-dispersion event, structurally absent."
    evidence: "main.tex L8 part (ii) closing clause; m10_universe_gap.csv codes SIVB/SBNY/FRC as receivership in the 87 absent names."
    suggested_fix: "Scope 'framing holds' to liquidity only; re-surface the distress-name caveat adjacent in part (ii)."
    status: FIXED
    resolution_notes: "L8 part (ii) ending rewritten: 'a statement about liquidity only, not about distress: the survivorship side (i) still omits the 2023 regional-bank failures (SVB, Signature, First Republic), so arm behaviour on the test span's single highest-dispersion event is unmeasured.'"
  - id: FINGNN-R2-gnn-02
    severity: CONCERN
    category: internal-consistency
    claim: "M6 corrected abstract/intro to 'operating point' but prose 'capacity-matched arms answer the scientific one' (Discussion) + 'at fixed capacity' re-assert the retired claim."
    evidence: "main.tex Discussion §6."
    suggested_fix: "Downgrade prose to 'operating-point-matched'; gloss the FC↔operating-point relation."
    status: FIXED
    resolution_notes: "Discussion: 'at fixed capacity' → 'at the matched L2 operating point'; 'capacity-matched arms answer the scientific one' → 'operating-point-matched arms answer the scientific one (subject to the FC-underfit caveat of §3.2)'. FC acronym + abstract/novelty/title term-of-art retained (glossed at §3.2)."
  - id: FINGNN-R2-qf-02
    severity: CONCERN
    category: reproducibility
    claim: "L8 audit numbers have no in-text provenance pointer + omit the artifact's 'Wikipedia-reconstructed, not S&P DJI/CRSP' caveat."
    evidence: "main.tex L8; caveat present in m10_universe_gap.md but not the manuscript."
    suggested_fix: "Add a half-sentence: constituent history is Wikipedia-reconstructed (not official), figures approximate."
    status: FIXED
    resolution_notes: "L8 part (i) intro now: 'a Wikipedia-reconstructed constituent audit over the window (public index-change records, not the official S&P DJI/CRSP history, so the figures below are approximate)'."
  - id: FINGNN-R2-stat-03
    severity: CONCERN
    category: statistics
    claim: "tab:dm reports ΔIC but not p; CSV has DM_p_normal/HLN_p_t/HLN_p_t_lag21 differing materially; paper does not state which p-column drove BH rejections (HLN_p_t)."
    evidence: "compute_family1_ladder.py:240 bh_fdr(df['HLN_p_t']); main.tex §Family-1 did not name the column."
    suggested_fix: "State BH applied to HLN-corrected t p-values (HLN_p_t)."
    status: FIXED
    resolution_notes: "§Family-1: 'BH-FDR at q=0.05 (applied to the HLN-corrected t p-values) over the pooled 20-test family'. (Full per-contrast p-column in tab:dm deferred — page budget; transparency clause suffices.)"
  - id: FINGNN-R2-stat-01
    severity: CONCERN
    category: internal-consistency
    claim: "M7 seed-averaging disclosure (PARTIAL closure) mitigates but does not fully reconcile the §1 cross-seed-dispersion thesis with the §3.2 inference-unit choice."
    evidence: "§1 concern #1 vs §3.2 inference-unit note (far apart)."
    suggested_fix: "Add a forward-pointing clause in §1 noting dispersion is surfaced descriptively while the confirmatory estimand is the seed-mean."
    status: FIXED
    resolution_notes: "§1 concern #1 now: 'We surface it descriptively through the per-arm multi-seed CIs, while the confirmatory tests take the seed-mean signal as their estimand (§3.2) — a deployment-precision caveat, not a quantity the family-wise inference itself absorbs.' This upgrades M7 PARTIAL → CLOSED."
  - id: FINGNN-R2-stat-02
    severity: CONCERN
    category: statistics
    claim: "The '80% MDE' invoked for the vs-LightGBM/SPA underpower statement is a per-contrast z-test MDE used as a proxy for the SPA max-statistic's power, not a formal SPA power calc."
    evidence: "main.tex §spa / abstract."
    suggested_fix: "One clause clarifying the MDE is a conservative per-contrast proxy for the SPA max-statistic."
    status: DEFERRED
    resolution_notes: "Optional clarity addition; adds length to an 11pp paper. Conclusion ('underpowered') is correct. Deferred to author / camera-ready."
  - id: FINGNN-R2-stat-04
    severity: CONCERN
    category: red-line-defense-in-depth
    claim: "Abstract reports C p=0.077 without the 'lower bound 0.055 ≥ 5%' qualifier (only at §spa); red-line NOT violated (no 'near-significant' anywhere) but defense-in-depth."
    status: DEFERRED
    resolution_notes: "Abstract already guards with 'underpowered ... fail-to-reject'. Optional bracket addition deferred (page budget)."
  - id: FINGNN-R2-gnn-01
    severity: CONCERN
    category: contribution-framing
    claim: "L6/L7 fidelity disclaimers + Family-2 hedges do not undersell the contribution (headline rests on faithful L1/L2 rungs) but the paper never states this, so hedges read as weakness."
    suggested_fix: "Optional sentence: L6/L7 corroborating, Family-2 confound-removing; headline rests on faithful L1/L2."
    status: DEFERRED
    resolution_notes: "Additive clarity, not a correctness fix; adds length. Deferred to author."
  - id: FINGNN-R2-gnn-04
    severity: CONCERN
    category: contribution-framing
    claim: "Family-2's stacked hedges (0/6 underpowered + possibly-underfit) read as 'proves nothing'; never states its role is confound-removal for the news-edge claim (not MLP>graph support)."
    suggested_fix: "Optional sentence clarifying Family-2's confound-removal role; its underpowered null is non-fatal to the headline."
    status: DEFERRED
    resolution_notes: "Additive; deferred to author. Pre-registration locks Family-2 as confirmatory (md5 freeze) — only its role can be clarified, not re-labelled."
  - id: FINGNN-R2-qf-03
    severity: CONCERN
    category: framing
    claim: "Subtitle 'of the US S&P 500' is framing-adjacent to a PIT investable claim; body scopes it correctly (§3.1 conditional estimand). No contradiction."
    status: DEFERRED
    resolution_notes: "Body scoping sufficient for a methods/scaffold paper. Optional '(survivor-snapshot)' subtitle tag deferred (style call)."
round1_closure:
  statistics: "M1 CLOSED, M2 CLOSED, M3 CLOSED, M7 PARTIAL→CLOSED (stat-01 fix), M8 CLOSED, M9 CLOSED, m16 CLOSED, m17 CLOSED, m18 CLOSED"
  gnn: "M5 CLOSED, M6 CLOSED (gnn-02 prose tightened), M13 CLOSED, M14 CLOSED, m20 CLOSED (StockMixer/MDGNN real+correct)"
  quant_finance: "M4 CLOSED, M10 CLOSED (qf-01 benignity scoped), M11 CLOSED, M12 CLOSED, M15 CLOSED, m21 CLOSED, m22 CLOSED, m23 CLOSED, m24 CLOSED"
red_lines: "ALL HELD across all 3 panels — no 'near-significant'; Family-2 0/6 underpowered-not-harmless; Universe-C positive suggestive; IC sole confirmatory, net Sharpe descriptive; no 'cancels' for survivorship; local DM ≠ global SPA; mechanism = candidate not proven."
numeric_audit: "All 3 panels independently cross-checked load-bearing numbers vs source CSVs (SPA bracket, 20 DM ΔIC+BH flags, Family-2 6 matched ΔIC, C/L5s exclude robustness, cost level CIs, L8 14.8/8.2/8.1/16.3% + $6.6B/$5B/54-cap-change, fold days=749) — ZERO numeric/sign errors introduced by Round-1 edits."
summary:
  critical: 0
  major: 1
  concern: 9
  fixed: 5
  deferred: 5
overall_verdict: PASS-WITH-CONCERNS
---

# PaperJury Round-2 — 3 isolated panels (statistics / GNN / quant-finance) on the post-Round-1 paper/main.tex

Re-ran the Round-1 protocol (3 isolated `finance-gnn-reviewer` panels, manuscript-only) on the edited paper to (A) verify Round-1 closure and (B) hunt for new issues from the edits.

**Result: all 3 panels PASS-WITH-CONCERNS, 0 CRITICAL, 1 MAJOR.** Round-1's 24 findings all confirmed closed (M7 was PARTIAL → closed by the stat-01 §1 fix). All four red lines hold across all panels; every load-bearing number independently re-verified against source CSVs with zero errors introduced.

**5 fixes applied this round** (1 MAJOR + 4 CONCERN): qf-01 (L8 benignity scoped to liquidity + distress caveat re-surfaced), gnn-02 (Discussion "capacity-matched"→"operating-point-matched"), qf-02 (Wikipedia-reconstructed source caveat in L8), stat-03 (BH on HLN-corrected t p stated), stat-01 (§1 forward-pointer → M7 fully closed).

**5 CONCERNs deferred** (optional clarity, add length to an 11pp paper, no correctness impact): stat-02 (SPA-MDE-is-a-proxy clause), stat-04 (abstract p-bracket defense-in-depth), gnn-01 (L6/L7 corroborating-not-load-bearing sentence), gnn-04 (Family-2 confound-removal-role sentence), qf-03 (subtitle survivor-snapshot tag).

Post-fix: brace Δ=0, end-to-end tectonic compile clean (0 errors, ~11pp PDF).

→ progress: 2026-06-27-a | plan: 2026-06-26 (Decision Log) | analysis: 2026-06-26-a
