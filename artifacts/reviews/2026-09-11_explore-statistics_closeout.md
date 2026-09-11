<!-- Rule 9 session-closeout audit 2/4 (Explore agent, independent context), 2026-09-11 ~03:35 local. Scope = this session's
statistics code + reporting. Statuses filled in by Claude after personally verifying each claim (protocol §6 grep: no HAC lag
spec; B LOFO max = fold 7 +0.05735; fold-9 shares 53/44/31%; 1.96×0.00704 = 0.0138 > 0.01343) and applying the fixes. -->
---
reviewer: explore-statistics
touchpoint: closeout
round: closeout
target_files:
  - compute_family1_ladder.py:212-256, 304-382, 389-436, 516-645, 652-744
  - analyze_c5_sensitivity.py:55-497
  - compute_e6_dm_spa.py:229-260, 686-700
  - docs/analysis.md:7-37
  - artifacts/storya_v21_family1_c5/c5_comparison.md
  - artifacts/storya_v21_family1_c5/family1_ledger.json
findings:
  - id: EXPL-STAT-01
    severity: MAJOR
    category: statistics
    claim: "The headline HLN p uses the Newey-West AUTO lag (L=6 at T=749) while the label is a 21-day overlapping forward return (ΔIC ≈ MA(20)); the h=21 HLN factor is applied on top of a variance truncated at lag 6. Outcome-determining, and the docs call the auto-lag 'frozen' although protocol §6 never specifies a HAC lag."
    evidence: "compute_e6_dm_spa.py:238 vs :257; recomputed ACF(ΔIC) lag6 = 0.323, HAC SE(L=6)=0.00491 → t=2.733 (matches), HAC SE(L=21)=0.00676 → t=1.986, block-boot SE 0.00704; C5 p 0.0080 → 0.0537; C 0.0109 → 0.0632; B 0.0524 → 0.1812. docs/protocol_v2_freeze.md §6 specifies no HAC lag; docs/analysis.md:25 said '冻结的 NW auto-lag ≈6'."
    suggested_fix: "Record the HAC-lag policy in the ledger; drop 'frozen' for the auto lag; state that under lag 21 none of C5/C/B is nominally significant."
    status: FIXED
    resolution_notes: "Verified (grep of docs/protocol_v2_freeze.md for lag/Newey/HAC/bandwidth: no hits). Sensitivity ledger now carries hln_hac_lag policy text; c5_comparison.md reading note (i) and docs/analysis.md 读法 (i) state the auto lag is an implementation default (not protocol) and that under lag 21 none of C5/C/B reaches nominal 0.05; '冻结的' removed."
  - id: EXPL-STAT-02
    severity: MAJOR
    category: statistics
    claim: "'lag-21 and the 21d block bootstrap agree' is false at the decision threshold for C5: the percentile CI excludes 0 while lag-21 HLN p = 0.0537; SEs agree (0.00676 vs 0.00704), verdicts do not; the CI exclusion is boundary-driven (1.96×SE = 0.0138 > ΔIC 0.01343; percentile asymmetry)."
    evidence: "c5_comparison.md:15, docs/analysis.md:25; family1_mde.csv C5 delta_ci_lo=0.00075 vs family1_dm_hln.csv HLN_p_t_lag21=0.05372."
    suggested_fix: "Reword: SEs agree, 5% verdicts differ marginally for C5; the CI's exclusion of zero is a boundary case, not a robust rejection."
    status: FIXED
    resolution_notes: "Verified 1.96×0.00704 = 0.0138 > 0.01343. Reading note (ii) in c5_comparison.md and 读法 (ii) in docs/analysis.md now say exactly this; the headline permitted sentence adds '边界排零'."
  - id: EXPL-STAT-03
    severity: CONCERN
    category: statistics
    claim: "'marginal, underpowered detections' labels B a detection although B is a non-detection on every criterion (CI includes 0, p 0.052, BH not rejected)."
    evidence: "c5_comparison.md:15; docs/analysis.md:25; family1_mde.csv B ci_excludes_0=False."
    suggested_fix: "Split: C5 rejects only under auto-lag/percentile-CI reading; C BH-significant at auto lag but CI includes 0; B non-detection; all three below the 80%-power threshold."
    status: FIXED
    resolution_notes: "Reading note (iii) in both places uses the split sentence."
  - id: EXPL-STAT-04
    severity: CONCERN
    category: statistics
    claim: "Multiplicity ledger records n_tests_total = 1 while the entry publishes ≥14 nominal p-values (3 universes × 2 lags, 3 ex-fold × 2, 2 paired × 2), the headline being the smallest."
    evidence: "family1_ledger.json sensitivity_scope.n_tests_total: 1; docs/analysis.md quotes 14 distinct p."
    suggested_fix: "Add a tests_reported inventory so 'no BH family opened' is paired with an explicit count."
    status: FIXED
    resolution_notes: "analyze_c5_sensitivity.py now writes c5_tests_reported.json (16 nominal p-values inventoried per run dir; headline flagged as the smallest); docs/analysis.md cites it."
  - id: EXPL-STAT-05
    severity: CONCERN
    category: statistics
    claim: "k/10 and LOSO m/10 are arithmetically dependent (k = n forces m = 0); k/10 is initialisation stability on shared data, not independent replication."
    evidence: "analyze_c5_sensitivity.py:160-164; all rows 10/10 & 0/10."
    suggested_fix: "Note m=0 is implied by k=n; label k/10 as seed/initialisation stability."
    status: FIXED
    resolution_notes: "Reading note (v) in c5_comparison.md and 读法 (v) in docs/analysis.md; headline sentence says 'LOSO 0 翻转为必然'."
  - id: EXPL-STAT-06
    severity: CONCERN
    category: statistics
    claim: "C5's smallest raw p accompanies the smallest point estimate — ordering is variance-driven (SE 0.00704 vs 0.00786 / 0.00982); unstated, it invites the banned 'subset preserved/strengthened the contrast' reading."
    evidence: "docs/analysis.md:21-23 table; c5_comparison.csv SE_block."
    suggested_fix: "Add one clause after the table."
    status: FIXED
    resolution_notes: "Reading note (iv) in both places."
  - id: EXPL-STAT-07
    severity: CONCERN
    category: correctness
    claim: "HLN_stat sign convention differs between family1 (loss difference, −2.658 for +ΔIC) and the paired/ex-fold code (IC difference, +0.2047); p unaffected."
    evidence: "compute_family1_ladder.py:227 vs analyze_c5_sensitivity.py:191, :231."
    suggested_fix: "Negate d or rename the column and state the convention."
    status: FIXED
    resolution_notes: "Renamed to HLN_stat_on_IC_diff in c5_paired_contrast.csv and c5_ex_fold.csv with an inline convention comment; p-values unchanged."
  - id: EXPL-STAT-08
    severity: CONCERN
    category: statistics
    claim: "Ex-fold framing does not hold for B: B's largest single-fold contribution is fold 7 (+0.05735), not fold 9 (+0.05286); fold-9 shares are 53.1% (C5), 44.0% (C), 30.6% (B), so 'exactly as in C and B' overstates B."
    evidence: "artifacts/storya_v21_family1/family1_lofo.csv B rows; shares recomputed."
    suggested_fix: "State shares per universe; fold 9 dominant in C5 and C, second-largest in B."
    status: FIXED
    resolution_notes: "Verified (B fold 7 = +0.05735 > fold 9 +0.05286; shares 0.531/0.440/0.307). c5_ex_fold.csv now carries excluded_fold_share; the md section header/footer and docs/analysis.md state 53%/44%/31% and B's fold 7."
  - id: EXPL-STAT-09
    severity: CONCERN
    category: statistics
    claim: "family1_summary.md MDE header equates 'ci_excludes_0' (α=0.05 rejection) with 'detected at this design' (80%-power MDE); they differ by 43%, exactly the C5 case."
    evidence: "compute_family1_ladder.py:634 → family1_summary.md."
    suggested_fix: "Reword the header."
    status: FIXED
    resolution_notes: "Header reworded (generated-text string only; confirmatory numbers untouched; the confirmatory summary is not regenerated)."
  - id: EXPL-STAT-10
    severity: CONCERN
    category: statistics
    claim: "The headline CI's estimand (seed-averaged ensemble; day-to-day variance only) is not labelled in docs/analysis.md; per-seed ΔIC range exceeds the CI half-width."
    evidence: "compute_family1_ladder.py:152-156, :312-334; docs/analysis.md table header."
    suggested_fix: "Label the table header and note the conditioning."
    status: FIXED
    resolution_notes: "Table headers in c5_comparison.md and docs/analysis.md now read 'ΔIC L1−L0 (seed-averaged daily)' / 'CI (on the 10-seed average)'; reading note (vi)."
summary:
  critical: 0
  major: 2
  concern: 8
  fixed_before_reply: 0
overall_verdict: PASS-WITH-CONCERNS
---

## Verified clean (agent)

- Default branch of compute_family1_ladder.py behaviourally identical to the confirmatory run (defaults reconstruct the old hard-coded lists; apply_bh True; ref_arms None → L2; cl5s_robustness runs; C L1−L0 p = 0.010852 reproduced to 15 digits).
- Pre-registration discipline: LADDER_PAIRS/EDGE_PAIRS untouched; pairs only intersected with arms present; ledger records pairs_tested [L1-L0], BH NOT APPLIED, SPA NOT RUN.
- Raw-p honesty stated in ledger, summary title, comparison header/CSV role column, analysis.md and its forbidden list.
- SE/MDE construction consistent across family1, paired contrast and ex-fold (same StationaryBootstrap(21, d, seed=86), same n_boot, same MDE_FACTOR).
- Estimand coherence: seed_pooled equals the seed-averaged daily mean here because no cell collapsed (0.01343 == 0.01343).
- Ex-fold concatenation order correct (fold 9 at offset 562 reproduces excluded_fold_delta_IC exactly); folds contiguous quarters.
- Paired-contrast alignment guard real (per-cell calendar check); ρ(ΔIC_C, ΔIC_C5) = 0.298, ρ(ΔIC_B, ΔIC_C5) = 0.168 → paired SE exceeds marginal SEs; 'underpowered non-rejection' wording is correct.
- Device confounding disclosed.

## Numbers spot-checked by the agent: 17 checks, 15 exact, 2 mismatches (both → EXPL-STAT-08, now fixed).

## Notes not raised as findings: compute_family1_ladder imports two_sided_power/_Z_975/_Z_80 unused (MDE_FACTOR 2.8 vs 2.8016, 0.06%); sensitivity run_pairwise adds a bh_applied column absent from the BH branch (schemas differ by one column — documented behaviour).
