---
reviewer: codex
touchpoint: results
round: A
target_files:
  - docs/storya_paper_draft_v2.md
findings:
  - id: CODEX-A-01
    severity: MAJOR
    category: statistics
    claim: "SPA C p=0.077 is not always paired with the underpowered/MDE qualifier."
    evidence: "Draft lines 21 and 43 state SPA B/C p-values without MDE; family1_spa.csv row C p_consistent=0.0774 fail-to-reject; family1_mde.csv row C L1-L0 mean_delta_IC=0.01477 vs MDE_2p8xSE=0.02201."
    suggested_fix: "In the abstract/headline occurrence, add that the vs-LightGBM comparison is underpowered/MDE exceeds observed gaps; keep line 181 wording."
    status: OPEN
    resolution_notes: null
  - id: CODEX-A-02
    severity: MAJOR
    category: interpretation
    claim: "Local rung results are sometimes compressed into global shorthand: 'news edges hurt', 'graph adds nothing', and 'neural beats trees'."
    evidence: "Draft lines 21, 43, 223-224, 272, 274, 276; family1_dm_hln.csv rows C L1-L0/L2-L1/L3-L2 are local BH rungs, while family2_fc_causal.csv rows B/C news are +0.00153/+0.00114, BH_FDR_reject=False, underpowered=True; cost_headline_crosswalk.csv row C L3-L2 net=+0.0821 CI [-0.7713,0.8868]."
    suggested_fix: "Replace shorthand with local wording: 'Universe-C L1-L0 local DM rung', 'C L2-L1 corr-GAT underperforms MLP', and 'tuned L3 news-edge rung underperforms corr-GAT on IC'; avoid standalone 'news hurts' or 'graph adds nothing'."
    status: OPEN
    resolution_notes: null
  - id: CODEX-A-03
    severity: CONCERN
    category: correctness
    claim: "C/L5s robustness prose implies all three treatments have SPA p-values."
    evidence: "Draft line 256 says SPA p stays 0.077-0.080 under {exclude, zero-fill, zero-skill}; family1_cl5s_robustness.csv rows exclude=0.0774, zerofill=0.0795, zeroskill_cell has blank C_SPA_p_consistent."
    suggested_fix: "Say: 'exclude and zero-fill give C-SPA p=0.0774 and 0.0795; zeroskill_cell gives C/L5s mean IC=0.00021 but no SPA p is reported in the CSV.'"
    status: OPEN
    resolution_notes: null
summary:
  critical: 0
  major: 2
  concern: 1
  fixed_before_reply: 0
overall_verdict: PROCEED-WITH-FIXES
---

**Findings**

CODEX-A-01: The SPA numeric values are correct, but the red-line qualifier is missing in prominent prose. `family1_spa.csv` has Universe B `p_consistent=0.2767`, Universe C `p_consistent=0.0774`, `M=9`, `T=749`, both `reject_h0_at_5pct=False`. Draft line 181 handles this correctly. Draft lines 21 and 43 do not pair the C p-value with the underpowered/MDE qualifier. The relevant MDE evidence is `family1_mde.csv`: C `L1-L0` observed `+0.01477` vs `MDE_2p8xSE=0.02201`; B `L1-L0` observed `+0.01428` vs `MDE=0.02749`.

CODEX-A-02: The draft mostly guards local-vs-global interpretation, but several high-visibility phrases still violate the discipline. The phrases “news edges hurt,” “graph adds nothing,” and “neural beats trees” should be rewritten as local rung statements. Source evidence: `family1_dm_hln.csv` supports local C rungs only: C `L1-L0=+0.014765`, BH true; C `L2-L1=-0.011925`, BH true; C `L3-L2=-0.012306`, BH true. Family-2 does not support causal news harm: `family2_fc_causal.csv` has B news `matched_delta_IC=+0.00153`, C news `+0.00114`, both BH false and underpowered. Cost also does not support economic news harm: `cost_headline_crosswalk.csv` C `L3-L2` net `+0.0821`, CI `[-0.7713,+0.8868]`.

CODEX-A-03: The C/L5s stability framing is otherwise correct: `family1_stability.csv` row C/L5s gives 25 full + 8 partial collapses out of 120, collapse rate `0.275`, and the draft correctly says EXCLUDE/undefined not measured zero/no retune. The one provenance mismatch is the robustness p-value sentence. `family1_cl5s_robustness.csv` reports SPA p only for `exclude=0.0774` and `zerofill=0.0795`; `zeroskill_cell` has no SPA p cell.

**Numbers Verified**

The key headline numbers match the source CSVs: Table 1 IC CIs match `family1_ic_ci.csv` after rounding; Table 2 DM/HLN values match `family1_dm_hln.csv`; Table 4 Family-2 values match `family2_fc_causal.csv`; Table 3 cost values match `cost_headline_crosswalk.csv`; and C L0/L1 net Sharpe @10 bps matches `cost_ladder_by_arm.csv` (`-0.2191`, `+0.9542`). Protocol claims also match: `docs/protocol_v2_freeze.md` specifies the 20-test DM family, SPA `M=9`, and `MDE ~= 2.8 × SE`.

`python scripts/verify_docs_provenance.py docs/storya_paper_draft_v2.md` passed; direct execution was blocked by file permissions, but the Python invocation returned clean.
2026-06-24T09:28:22.143877Z ERROR codex_core::session: failed to record rollout items: thread 019ef8f2-7240-7242-82e0-92ba2c681c05 not found
tokens used
128,343
---
