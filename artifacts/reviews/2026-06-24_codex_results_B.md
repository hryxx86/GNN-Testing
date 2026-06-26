---
reviewer: codex
touchpoint: results
round: B
target_files:
  - docs/storya_paper_draft_v2.md
  - artifacts/storya_v21_family1/family1_spa.csv
  - artifacts/storya_v21_family1/family1_mde.csv
  - artifacts/storya_v21_family1/family1_dm_hln.csv
  - artifacts/storya_v21_family1/family1_cl5s_robustness.csv
  - artifacts/storya_v21_family2_fc/family2_fc_causal.csv
  - artifacts/storya_v21_cost/cost_headline_crosswalk.csv
  - artifacts/storya_v21_cost/cost_ladder_by_arm.csv
findings:
  - id: CODEX-A-01
    severity: MAJOR
    category: statistics
    claim: "SPA C p=0.077 is now paired with fail-to-reject and underpowered/MDE language in the prominent occurrences."
    evidence: "Draft lines 21, 43, and 181; family1_spa.csv line 3 p_consistent=0.0774, reject_h0_at_5pct=False; family1_mde.csv line 12 C/L1-L0 mean_delta_IC=0.01477 vs MDE_2p8xSE=0.02201."
    suggested_fix: null
    status: FIXED
    resolution_notes: "Abstract and headline findings now state fail-to-reject/underpowered; the headline cites the C L1-L0 observed gap below MDE."
  - id: CODEX-A-02
    severity: MAJOR
    category: interpretation
    claim: "Local rung wording was improved in the abstract, headline findings, Table 3, and graph discussion, but old news-edge shorthand remains in §6."
    evidence: "Draft line 274 says 'news hurts ranking' and line 276 says 'news edges hurt ranking'; family1_dm_hln.csv line 17 only supports local C L3-L2 tuned ΔIC=-0.012306 with BH_FDR_reject_family=True, while family2_fc_causal.csv line 5 has C/news matched_delta_IC=+0.00114, BH_FDR_reject=False, underpowered_vs_effect=True, same_sign_matched_vs_tuned=False, and cost_headline_crosswalk.csv line 17 has net_dSharpe_10bps=+0.0821 with CI [-0.7713,+0.8868]."
    suggested_fix: "Replace the remaining §6 shorthand with local-rung wording."
    status: STILL-OPEN
    resolution_notes: "Partially fixed: abstract, §1 headline, Table 3, and the graph paragraph are localized; §6 still contains broad news-hurts phrasing."
  - id: CODEX-A-03
    severity: CONCERN
    category: correctness
    claim: "C/L5s robustness prose now distinguishes the two computed SPA p-values from the zero-skill treatment with no SPA p."
    evidence: "Draft line 256; family1_cl5s_robustness.csv line 2 exclude C_SPA_p_consistent=0.0774, line 3 zerofill=0.0795, line 4 zeroskill_cell blank."
    suggested_fix: null
    status: FIXED
    resolution_notes: "The zero-skill-cell treatment is now described as having an IC value but no reported SPA p in the CSV."
  - id: CODEX-B-01
    severity: MAJOR
    category: correctness
    claim: "§5.5 incorrectly says Universe-C matched and tuned edge effects agree in sign and magnitude."
    evidence: "Draft line 252 says Universe-C matched and tuned ΔIC agree; family2_fc_causal.csv line 5 for C/news has matched_delta_IC=+0.00114, tuned_delta_IC=-0.01231, same_sign_matched_vs_tuned=False."
    suggested_fix: "Say that only C sector and C sector+news agree with tuned signs; C news is opposite-signed and non-significant."
    status: OPEN
    resolution_notes: null
summary:
  critical: 0
  major: 2
  concern: 0
  fixed_before_reply: 0
overall_verdict: PROCEED-WITH-FIXES
---

**Discussion**

CODEX-A-01: FIXED. Draft lines 21 and 43 now pair Universe C SPA p with fail-to-reject/underpowered language, and line 43 adds the MDE example. Source cells match: `family1_spa.csv` line 3 has `p_consistent=0.0774`, `reject_h0_at_5pct=False`; `family1_mde.csv` line 12 has C/L1-L0 `mean_delta_IC=0.01477`, `MDE_2p8xSE=0.02201`.

CODEX-A-02: STILL-OPEN, partially fixed. Abstract, §1 headline, Table 3, and the graph paragraph are now mostly localized. But §6 still says “news hurts ranking” at draft line 274 and “news edges hurt ranking” at line 276. The CSV support is narrower: `family1_dm_hln.csv` line 17 supports only the local tuned C L3-L2 IC rung; `family2_fc_causal.csv` line 5 has C/news small positive, non-BH, underpowered, and opposite-signed; `cost_headline_crosswalk.csv` line 17 has net +0.0821 with CI crossing zero.

CODEX-A-03: FIXED. Draft line 256 correctly says SPA p is computed only for exclude and zero-fill. `family1_cl5s_robustness.csv` line 2 has exclude `0.0774`, line 3 zerofill `0.0795`, and line 4 zeroskill_cell has a blank SPA p cell.

CODEX-B-01: OPEN. Draft line 252 says Universe-C matched and tuned ΔIC agree in sign and magnitude, but `family2_fc_causal.csv` line 5 contradicts this for C/news: matched `+0.00114`, tuned `-0.01231`, `same_sign_matched_vs_tuned=False`. This needs a prose correction only.

`python scripts/verify_docs_provenance.py docs/storya_paper_draft_v2.md` passed.

overall_verdict: PROCEED-WITH-FIXES
diff --git a/artifacts/reviews/2026-06-24_codex_results_B.md b/artifacts/reviews/2026-06-24_codex_results_B.md
new file mode 100644
index 0000000000000000000000000000000000000000..bb7398fd98f13a40dbdf48d143aa2901366e45d6
--- /dev/null
+++ b/artifacts/reviews/2026-06-24_codex_results_B.md
@@ -0,0 +1,67 @@
+---
+reviewer: codex
+touchpoint: results
+round: B
+target_files:
+  - docs/storya_paper_draft_v2.md
+  - artifacts/storya_v21_family1/family1_spa.csv
+  - artifacts/storya_v21_family1/family1_mde.csv
+  - artifacts/storya_v21_family1/family1_dm_hln.csv
+  - artifacts/storya_v21_family1/family1_cl5s_robustness.csv
+  - artifacts/storya_v21_family2_fc/family2_fc_causal.csv
+  - artifacts/storya_v21_cost/cost_headline_crosswalk.csv
+  - artifacts/storya_v21_cost/cost_ladder_by_arm.csv
+findings:
+  - id: CODEX-A-01
+    severity: MAJOR
+    category: statistics
+    claim: "SPA C p=0.077 is now paired with fail-to-reject and underpowered/MDE language in the prominent occurrences."
+    evidence: "Draft lines 21, 43, and 181; family1_spa.csv line 3 p_consistent=0.0774, reject_h0_at_5pct=False; family1_mde.csv line 12 C/L1-L0 mean_delta_IC=0.01477 vs MDE_2p8xSE=0.02201."
+    suggested_fix: null
+    status: FIXED
+    resolution_notes: "Abstract and headline findings now state fail-to-reject/underpowered; the headline cites the C L1-L0 observed gap below MDE."
+  - id: CODEX-A-02
+    severity: MAJOR
+    category: interpretation
+    claim: "Local rung wording was improved in the abstract, headline findings, Table 3, and graph discussion, but old news-edge shorthand remains in §6."
+    evidence: "Draft line 274 says 'news hurts ranking' and line 276 says 'news edges hurt ranking'; family1_dm_hln.csv line 17 only supports local C L3-L2 tuned ΔIC=-0.012306 with BH_FDR_reject_family=True, while family2_fc_causal.csv line 5 has C/news matched_delta_IC=+0.00114, BH_FDR_reject=False, underpowered_vs_effect=True, same_sign_matched_vs_tuned=False, and cost_headline_crosswalk.csv line 17 has net_dSharpe_10bps=+0.0821 with CI [-0.7713,+0.8868]."
+    suggested_fix: "Replace the remaining §6 shorthand with local-rung wording, e.g. 'the tuned C L3-L2 news-edge rung underperforms corr-GAT on IC, but this does not reproduce in net economics or fixed-capacity Family-2.'"
+    status: STILL-OPEN
+    resolution_notes: "Partially fixed: abstract, §1 headline, Table 3, and the graph paragraph are localized; §6 still contains broad news-hurts phrasing."
+  - id: CODEX-A-03
+    severity: CONCERN
+    category: correctness
+    claim: "C/L5s robustness prose now distinguishes the two computed SPA p-values from the zero-skill treatment with no SPA p."
+    evidence: "Draft line 256 says SPA is stable only across the two treatments for which it is computed; family1_cl5s_robustness.csv line 2 exclude C_SPA_p_consistent=0.0774, line 3 zerofill C_SPA_p_consistent=0.0795, line 4 zeroskill_cell C_SPA_p_consistent is blank."
+    suggested_fix: null
+    status: FIXED
+    resolution_notes: "The zero-skill-cell treatment is now described as having an IC value but no reported SPA p in the CSV."
+  - id: CODEX-B-01
+    severity: MAJOR
+    category: correctness
+    claim: "§5.5 incorrectly says Universe-C matched and tuned edge effects agree in sign and magnitude."
+    evidence: "Draft line 252 says 'the Universe-C matched and tuned ΔIC agree in sign and magnitude'; family2_fc_causal.csv line 5 for C/news has matched_delta_IC=+0.00114, tuned_delta_IC=-0.01231, same_sign_matched_vs_tuned=False, BH_FDR_reject=False, underpowered_vs_effect=True."
+    suggested_fix: "Revise the sentence to say that in Universe C the sector and sector+news rows agree with tuned signs, but the news row is opposite-signed and non-significant; the family-level causal conclusion remains no BH-surviving, underpowered edge effects."
+    status: OPEN
+    resolution_notes: null
+summary:
+  critical: 0
+  major: 2
+  concern: 0
+  fixed_before_reply: 0
+overall_verdict: PROCEED-WITH-FIXES
+---
+
+**Discussion**
+
+CODEX-A-01: FIXED. The abstract now pairs the SPA fail-to-reject with the underpowered qualifier at draft line 21. The headline finding at draft line 43 adds the concrete MDE comparison: Universe C L1-L0 observed +0.0148 vs MDE 0.0220. The source cells match: `family1_spa.csv` line 3 has Universe C `p_consistent=0.0774` and `reject_h0_at_5pct=False`; `family1_mde.csv` line 12 has `mean_delta_IC=0.01477` and `MDE_2p8xSE=0.02201`. Draft line 181 also remains correctly bounded as fail-to-reject, not a near-miss.
+
+CODEX-A-02: STILL-OPEN, partially fixed. The abstract and §1 headline now localize the load-bearing claims to Universe-C rungs, and Table 3 now uses `Local rung (Universe C, on IC)` at draft line 220 with neutral row labels at lines 222-224. The graph paragraph at draft line 272 is bounded and explicitly rejects a general "graphs hurt ranking" claim. However §6 still retains the old news-edge shorthand: draft line 274 says "news hurts ranking" and draft line 276 says "news edges hurt ranking." The source support is narrower: `family1_dm_hln.csv` line 17 supports only the local tuned C L3-L2 IC rung (`mean_delta_IC=-0.012306...`, BH true); `family2_fc_causal.csv` line 5 has the C/news fixed-capacity contrast small, positive, non-BH, and underpowered; and `cost_headline_crosswalk.csv` line 17 has C L3-L2 net Sharpe +0.0821 with CI [-0.7713,+0.8868].
+
+CODEX-A-03: FIXED. Draft line 256 now says SPA p is stable only across the two treatments for which SPA is computed, then explicitly says zero-skill-cell has no SPA p reported. The CSV cells match: `family1_cl5s_robustness.csv` line 2 exclude `C_SPA_p_consistent=0.0774`, line 3 zerofill `0.0795`, line 4 zeroskill_cell blank.
+
+CODEX-B-01: OPEN. Draft line 252 says the Universe-C matched and tuned edge effects agree in sign and magnitude. That is not true for the C/news row: `family2_fc_causal.csv` line 5 has `matched_delta_IC=+0.00114`, `tuned_delta_IC=-0.01231`, `same_sign_matched_vs_tuned=False`, `BH_FDR_reject=False`, and `underpowered_vs_effect=True`. Only the C sector and C sector+news rows have same-sign matched/tuned values. This is a correctness issue in the Family-2 reading paragraph, not a request for new analysis.
+
+`python scripts/verify_docs_provenance.py docs/storya_paper_draft_v2.md` passed.
+
+overall_verdict: PROCEED-WITH-FIXES

2026-06-24T09:36:35.247810Z ERROR codex_core::session: failed to record rollout items: thread 019ef8f9-2d30-71c1-8b02-3a181df007a2 not found
tokens used
118,411
---
reviewer: codex
touchpoint: results
round: B
target_files:
  - docs/storya_paper_draft_v2.md
  - artifacts/storya_v21_family1/family1_spa.csv
  - artifacts/storya_v21_family1/family1_mde.csv
  - artifacts/storya_v21_family1/family1_dm_hln.csv
  - artifacts/storya_v21_family1/family1_cl5s_robustness.csv
  - artifacts/storya_v21_family2_fc/family2_fc_causal.csv
  - artifacts/storya_v21_cost/cost_headline_crosswalk.csv
  - artifacts/storya_v21_cost/cost_ladder_by_arm.csv
findings:
  - id: CODEX-A-01
    severity: MAJOR
    category: statistics
    claim: "SPA C p=0.077 is now paired with fail-to-reject and underpowered/MDE language in the prominent occurrences."
    evidence: "Draft lines 21, 43, and 181; family1_spa.csv line 3 p_consistent=0.0774, reject_h0_at_5pct=False; family1_mde.csv line 12 C/L1-L0 mean_delta_IC=0.01477 vs MDE_2p8xSE=0.02201."
    suggested_fix: null
    status: FIXED
    resolution_notes: "Abstract and headline findings now state fail-to-reject/underpowered; the headline cites the C L1-L0 observed gap below MDE."
  - id: CODEX-A-02
    severity: MAJOR
    category: interpretation
    claim: "Local rung wording was improved in the abstract, headline findings, Table 3, and graph discussion, but old news-edge shorthand remains in §6."
    evidence: "Draft line 274 says 'news hurts ranking' and line 276 says 'news edges hurt ranking'; family1_dm_hln.csv line 17 only supports local C L3-L2 tuned ΔIC=-0.012306 with BH_FDR_reject_family=True, while family2_fc_causal.csv line 5 has C/news matched_delta_IC=+0.00114, BH_FDR_reject=False, underpowered_vs_effect=True, same_sign_matched_vs_tuned=False, and cost_headline_crosswalk.csv line 17 has net_dSharpe_10bps=+0.0821 with CI [-0.7713,+0.8868]."
    suggested_fix: "Replace the remaining §6 shorthand with local-rung wording."
    status: STILL-OPEN
    resolution_notes: "Partially fixed: abstract, §1 headline, Table 3, and the graph paragraph are localized; §6 still contains broad news-hurts phrasing."
  - id: CODEX-A-03
    severity: CONCERN
    category: correctness
    claim: "C/L5s robustness prose now distinguishes the two computed SPA p-values from the zero-skill treatment with no SPA p."
    evidence: "Draft line 256; family1_cl5s_robustness.csv line 2 exclude C_SPA_p_consistent=0.0774, line 3 zerofill=0.0795, line 4 zeroskill_cell blank."
    suggested_fix: null
    status: FIXED
    resolution_notes: "The zero-skill-cell treatment is now described as having an IC value but no reported SPA p in the CSV."
  - id: CODEX-B-01
    severity: MAJOR
    category: correctness
    claim: "§5.5 incorrectly says Universe-C matched and tuned edge effects agree in sign and magnitude."
    evidence: "Draft line 252 says Universe-C matched and tuned ΔIC agree; family2_fc_causal.csv line 5 for C/news has matched_delta_IC=+0.00114, tuned_delta_IC=-0.01231, same_sign_matched_vs_tuned=False."
    suggested_fix: "Say that only C sector and C sector+news agree with tuned signs; C news is opposite-signed and non-significant."
    status: OPEN
    resolution_notes: null
summary:
  critical: 0
  major: 2
  concern: 0
  fixed_before_reply: 0
overall_verdict: PROCEED-WITH-FIXES
---

**Discussion**

CODEX-A-01: FIXED. Draft lines 21 and 43 now pair Universe C SPA p with fail-to-reject/underpowered language, and line 43 adds the MDE example. Source cells match: `family1_spa.csv` line 3 has `p_consistent=0.0774`, `reject_h0_at_5pct=False`; `family1_mde.csv` line 12 has C/L1-L0 `mean_delta_IC=0.01477`, `MDE_2p8xSE=0.02201`.

CODEX-A-02: STILL-OPEN, partially fixed. Abstract, §1 headline, Table 3, and the graph paragraph are now mostly localized. But §6 still says “news hurts ranking” at draft line 274 and “news edges hurt ranking” at line 276. The CSV support is narrower: `family1_dm_hln.csv` line 17 supports only the local tuned C L3-L2 IC rung; `family2_fc_causal.csv` line 5 has C/news small positive, non-BH, underpowered, and opposite-signed; `cost_headline_crosswalk.csv` line 17 has net +0.0821 with CI crossing zero.

CODEX-A-03: FIXED. Draft line 256 correctly says SPA p is computed only for exclude and zero-fill. `family1_cl5s_robustness.csv` line 2 has exclude `0.0774`, line 3 zerofill `0.0795`, and line 4 zeroskill_cell has a blank SPA p cell.

CODEX-B-01: OPEN. Draft line 252 says Universe-C matched and tuned ΔIC agree in sign and magnitude, but `family2_fc_causal.csv` line 5 contradicts this for C/news: matched `+0.00114`, tuned `-0.01231`, `same_sign_matched_vs_tuned=False`. This needs a prose correction only.

`python scripts/verify_docs_provenance.py docs/storya_paper_draft_v2.md` passed.

overall_verdict: PROCEED-WITH-FIXES
