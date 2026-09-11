<!-- Rule 9 Touchpoint 3, Round A — FALLBACK reviewer (CLAUDE.md Rule 9 Fallback): Codex CLI at its usage limit until
04:40 local (2026-09-11). finance-gnn-reviewer completed ~02:57 local. Target = the C5 sensitivity statistics
(T4 primary) in artifacts/storya_v21_family1_c5/ + the Mac replicate. Statuses filled in by Claude after personally
re-deriving each numeric claim (ex-fold-9 statistics, tuning finalists, Plan-AAA architectures). -->
---
reviewer: finance-gnn-reviewer
touchpoint: results
round: A
target_files:
  - artifacts/storya_v21_family1_c5/c5_comparison.md
  - artifacts/storya_v21_family1_c5/c5_paired_contrast.csv
  - artifacts/storya_v21_family1_c5/c5_run_integrity.json
  - artifacts/storya_v21_family1_c5/c5_device_replication.md
findings:
  - id: FINGNN-R-A-01
    severity: MAJOR
    category: statistics
    claim: "About half of the C5 contrast is one quarter (fold 9 = 2025Q2); the 'persistence' claim must carry this."
    evidence: "Recomputed from experiments/storya_v21_main12_c5_t4/per_day_ic: fold-9 seed-avg ΔIC = +0.086 (L1 IC 0.27, L0 0.17); excluding fold 9: C5 ΔIC = +0.0069, 21d-block CI [-0.0029, +0.0182], HLN p = 0.13 (auto) / 0.23 (lag 21). Same pattern in C (ex-9 +0.0090, p=0.12) and B (ex-9 +0.0108, p=0.14); family1_lofo.csv row fold 9: lofo_mean_delta 0.00681."
    suggested_fix: "Add an ex-fold-9 row (ΔIC, CI, both p) to c5_comparison.md and state in docs/analysis.md + paper that the C5 contrast inherits the 2025Q2 concentration of C/B; do not describe the effect as evenly persistent."
    status: FIXED
    resolution_notes: "Verified by Claude (recomputed: C5 fold-9 ΔIC +0.0862, ex-9 +0.0069 [−0.0029, +0.0182] p=0.132/0.231; C +0.0090 p=0.125; B +0.0108 p=0.143). analyze_c5_sensitivity.py --ex-fold 9 now emits c5_ex_fold.csv + a 'Fold concentration' table in c5_comparison.md; wording adopted in docs/analysis.md 2026-09-11-a."
  - id: FINGNN-R-A-02
    severity: MAJOR
    category: statistics
    claim: "p = 0.008 is the optimistic (NW auto-lag 6) figure; lag-21 HAC and the 21d block bootstrap agree on ~0.05, and the observed effect is below the design MDE."
    evidence: "Seed-avg ΔIC autocorrelation lags 1..21 = 0.62, 0.54, 0.46, 0.40, 0.40, 0.32, 0.28, 0.24, 0.21, 0.17, ... 0.01 (recomputed); auto lag 6 truncates at 0.28-0.32. family1_dm_hln.csv: HLN_p_t 0.0080 vs HLN_p_t_lag21 0.0537; family1_mde.csv: SE_block 0.00704 (z≈1.9), MDE 0.0197 > observed 0.0134; CI lower bound +0.00075 (5000 reps) / +0.00043 (my 2000-rep recompute)."
    suggested_fix: "Lead with the CI; always quote both HAC lags (as done for C in the paper); state explicitly that observed ΔIC < MDE for C5, C and B, i.e. these are underpowered marginal detections. Remove 'nominal p=0.008' as a stand-alone figure."
    status: FIXED
    resolution_notes: "c5_comparison.md now carries an explicit reading note (CI first, both HAC lags, |ΔIC| < MDE in C5/C/B = marginal underpowered detections); docs/analysis.md entry quotes CI + both lags together and never p=0.008 alone."
  - id: FINGNN-R-A-03
    severity: MAJOR
    category: statistics
    claim: "The paired C-minus-C5 interval (±0.017) is wider than the contrast itself; 'did not materially change the contrast' is an equivalence claim the data cannot support."
    evidence: "c5_paired_contrast.csv: +0.0013 [-0.0159, +0.0189]. Given C = +0.0148, the interval is compatible with a C5 contrast anywhere in [-0.004, +0.031] (zero to double)."
    suggested_fix: "Replace with: point estimate essentially unchanged; the paired difference is not distinguishable from zero but its interval does not exclude a halving or a doubling of the contrast; equivalence is not established."
    status: FIXED
    resolution_notes: "Paired rows now include SE_block + MDE (TP2-B B-03); wording 'underpowered non-rejection; does not exclude a halving or a doubling; equivalence not established' adopted in c5_comparison.md and docs/analysis.md. The draft sentence 'did not materially change the contrast' was dropped."
  - id: FINGNN-R-A-04
    severity: MAJOR
    category: other
    claim: "Tuning on C5 selected among configurations indistinguishable from noise; the equal-budget protocol is procedurally intact but substantively uninformed, and this must be disclosed alongside the capacity change."
    evidence: "experiments/storya_v21_tune/C5_L1.json top_table: all 5 finalists hidden 32 / dropout 0.3 / 1 layer, mean val-IC -0.0448..-0.0453 (spread 0.0005), per-tuning-seed -0.006/-0.045/-0.084; C5_L0.json: all finalists num_leaves 63 / min_data 100, val-IC -0.0121..-0.0122. Confirmatory C winners had val-IC +0.0735 / +0.0597. MLP params 2337 (C5) vs 31745 (C) (c5_comparison.md:27,29)."
    suggested_fix: "Disclose in docs/analysis.md and the paper appendix: negative tuning-window val-IC for every finalist of both arms; HPs are protocol-consistent but not a validated optimum; the sign flip vs C's val-IC is itself consistent with the subset's signal being concentrated in its (test-period) selection window; the L1 that 'persists' is a 14x smaller model, so no capacity attribution."
    status: FIXED
    resolution_notes: "Verified from the two top_tables. Disclosure added to c5_comparison.md (tuned-winners section) and docs/analysis.md; no capacity attribution anywhere."
  - id: FINGNN-R-A-05
    severity: CONCERN
    category: data-leakage
    claim: "The Plan-AAA selector is model-class-informed (permutation ΔIC under SAGE-Mean and MLP on test quarters), so the direction of L1-L0 may itself be selection-favored in both C and C5; C5 cannot speak to this."
    evidence: "artifacts/plan_aaa/baseline_ic_per_cell.csv arch ∈ {SAGE-Mean, MLP}, fold_idx 0..4 (5-fold test quarters 2024-04..2025-06 per brief §9.9)."
    suggested_fix: "Add to the limitation text: selection used NN-based importance, so within-C contrasts involving NN arms are not selection-neutral; only B (and a pre-test selector) are."
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "Verified (arch column = {SAGE-Mean, MLP}). Sentence added to docs/analysis.md limitations and flagged for the paper L1 rewrite (H博士)."
  - id: FINGNN-R-A-06
    severity: CONCERN
    category: reproducibility
    claim: "T4-vs-Mac L1 divergence is expected backend nondeterminism amplified by early stopping on a flat validation curve; immaterial for pooled inference but the range should be disclosed."
    evidence: "Top |Δ| cells are all fold 9 with different epochs_run (s1024: 16 vs 22; s99: 26 vs 16; s34: 16 vs 25), best_val_loss 0.997-1.001 (val MSE ≈ label variance). Cell-IC SD 0.086 vs mean |Δ| 0.018, corr 0.951; pooled ΔIC 0.01343 vs 0.01318; HLN p 0.0080 vs 0.0125; paired C-C5 +0.0013 (T4) vs +0.0016 (Mac)."
    suggested_fix: "Keep T4 as the pre-declared primary (progress 2026-09-11-b); report the Mac replicate's ΔIC / p as a range in analysis.md; note the C(Mac)-C5(T4) device confound in the paired contrast is bounded by that replicate."
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "T4 kept as primary; docs/analysis.md reports the Mac replicate as a range (ΔIC +0.0132 [+0.0002, +0.0279], p 0.013/0.067; paired +0.0016) and notes the device confound bound."
  - id: FINGNN-R-A-07
    severity: CONCERN
    category: other
    claim: "The paper still asserts the retracted framing ('only 5 of 15 groups survive strict T-1 re-ranking'; 'definitive check'); C5 numbers cannot be inserted next to it without self-contradiction."
    evidence: "paper/iclr2027/main.tex:290, :998; paper/main.tex:352 (TP1 CODEX-A-02 flagged; still present)."
    suggested_fix: "Paper-side edit before any C5 paragraph: describe C5 as 'Plan-AAA top-15 ∩ single-feature-IC proxy top-15 (identical with/without the T-1 shift)' and withdraw the 'definitive check' promise. H博士 decision."
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "Paper-side action; escalated to H博士 (progress 2026-09-10-b, plan 2026-09-11 update). No paper edits made in this session."
summary:
  critical: 0
  major: 4
  concern: 3
  fixed_before_reply: 0
overall_verdict: PROCEED-WITH-FIXES
---

# Review body — C5 results (Touchpoint 3, Round A, fallback reviewer)

## 1. Credibility of the computation: PASS

Independently recomputed from `experiments/storya_v21_main12_c5_t4/`: seed-averaged daily ΔIC mean = +0.013433 (matches `family1_dm_hln.csv`), per-seed pooled ΔIC matches `c5_seed_robustness_per_seed.json` to 5 dp, 10/10 same sign, all 240 `.npy` files at full frozen calendar length (749/arm), no NaN, per-cell `.npy` mean vs `results.csv` IC_mean max |diff| 5e-7, cell_id 2400–2639 disjoint from confirmatory. Provenance gate is genuinely enforced (`analyze_c5_sensitivity.py:91-99`). No broken cell, no degenerate seed, no misalignment. The confirmatory C row reproduces `artifacts/storya_v21_family1/family1_dm_hln.csv` exactly.

## 2. Statistical method

- HLN on seed-averaged daily ΔIC with 21d stationary bootstrap is the frozen protocol and is applied correctly (`compute_e6_dm_spa.py:229-262, 286-300`). The problem is emphasis, not method: with a 21-day overlapping label the auto NW lag (6) truncates while autocorrelation is still ~0.3 (A-02). Lag-21 p (0.054) and the bootstrap (z≈1.9) agree with each other; the paper must quote both, exactly as it does for C (0.011 / 0.063).
- MDE label "≈2.8×SE, approximate nominal" is honest. What is missing is the consequence: observed ΔIC < MDE in all three universes — say it.
- Paired contrast: pairing by common test day is the right dependence structure; seed pairing is inert (different input width/HPs → different draws) and harmless. The interval is simply too wide to support anything beyond "point estimate similar" (A-03).
- Raw p is appropriate for a single pre-specified post-hoc contrast; but neither HLN nor the bootstrap accounts for the test-informed selection step, so "nominal" must accompany every p. If C5h and/or C-pre are later run, report all sensitivity contrasts together.
- Device divergence is expected (A-06); early stopping on a val loss of ≈0.998 (label is z-scored, so the model explains <0.3% of val variance) is inherently backend-sensitive. Not a concern for the pooled inference.

## 3. Interpretation — proposed wording assessed

The implementer's sentence is defensible in its second half (test-informed, does not resolve the leakage limitation, B remains anchor) and not defensible in its first half: "persists ... essentially the same point estimate", "nominal p=0.008" alone, and "did not materially change the contrast" over-read the data (A-01/A-02/A-03). It also omits the tuning disclosure (A-04).

**Permitted wording (docs/analysis.md and paper):**
- "On the 20-column subset selected by the intersection of two test-informed importance rankings, the MLP−LightGBM point estimate is +0.0134 (95% 21d-block CI [+0.0008, +0.0283]; nominal HLN p = 0.008 at the frozen NW auto-lag, 0.054 at HAC lag = horizon; 10/10 seeds same sign; 0 LOSO flips), versus +0.0148 in C and +0.0143 in B."
- "The paired change C − C5 is +0.0013 [−0.016, +0.019]: the point estimate is essentially unchanged, but the interval does not distinguish an unchanged contrast from a halved or doubled one; equivalence is not established."
- "Observed ΔIC is below the design's approximate MDE (0.020) in C5, C and B; these are marginal, underpowered detections."
- "Roughly half of the pooled contrast in C5 (as in C and B) comes from 2025Q2 (fold 9); excluding it, C5 ΔIC = +0.007 [−0.003, +0.018]."
- "Both arms' tuning-window validation IC on C5 was negative (L0 −0.012, L1 −0.045); the frozen HPs are protocol-consistent but not a validated optimum, and the C5 MLP has 2,337 parameters vs 31,745 in C."
- "C5's selection used evaluation-period labels and NN-based permutation importance; this sensitivity does not resolve, bound, or estimate feature-selection leakage in C. B remains the leak-free feature-basis anchor."
- "post-hoc", "sensitivity", "nominal", "unadjusted".

**Forbidden wording:**
- "leak-free re-selection", "survives T−1 re-ranking", "definitive check", "confirms/validates C".
- "the advantage persists" without the fold-9 and MDE qualifiers; "robust to leakage".
- "did not materially change", "unchanged", "equivalent", "no effect of restriction".
- "p = 0.008" without the lag-21 figure and the CI.
- "re-tuned to its optimum" / any capacity attribution.
- Quoting C5 per-arm IC levels (0.020 / 0.034) as out-of-sample performance.
- "the L1−L0 contrast is unaffected by selection" (A-05).

## 4. What a reviewer will still raise; C5h / C-pre

A finance/ML reviewer will say: (i) selection on test labels with NN-based importance → the subset and the direction of the contrast are both post-selection; (ii) negative tuning val-IC + positive test IC is the signature of that selection; (iii) one quarter carries half the effect; (iv) effect < MDE. None of these are fixed by C5h (adds 3 hc columns; no bearing on the claim — **not needed**). C-pre is the only follow-up that can support any statement about leakage magnitude in C; it is **not required** if the paper keeps the current claim ("suggestive; B is the anchor; leakage not quantified") with the L1 sentences corrected (A-07). If H博士 wants the paper to say anything quantitative about leakage, C-pre with a pre-registered paired contrast is the route; deadline arithmetic (≈3 h compute + TP1/TP3) is feasible but is H博士's call.

## Genuine gaps
- `docs/analysis.md` entry for C5 not yet written (progress 2026-09-11-c marks it PENDING) — must include A-01..A-04 disclosures.
- No ex-fold-9 row in `c5_comparison.md`.
- Paper L1 / Appendix sentences (A-07) unchanged.

## Bottom line
The numbers are computed correctly and reproduce independently; the run is clean. Proceed to the analysis.md entry, but only with wording that (1) leads with the CI and both HAC lags and notes effect < MDE, (2) drops the equivalence reading of the paired contrast, (3) discloses fold-9 concentration and the negative tuning val-IC / capacity change, and (4) is preceded by the paper-side retraction of the "5 survive T−1" / "definitive check" framing. C5h is unnecessary; C-pre is optional and only matters if the paper wants to quantify leakage.
