# Plan — Fixed-Capacity (FC) Edge-Causal Arm + Two-Family Pre-Registration (2026-06-17, v2)

> **v2 = post-Codex-Touchpoint-1** (`artifacts/reviews/2026-06-17_codex_plan_A.md`, BLOCK-EXECUTION,
> 2 CRIT + 5 MAJOR + 2 CONCERN, ALL accepted). v1's fatal flaw: it called matched-ΔIC "primary causal"
> yet gave it no inference. v2 fixes that with a two-family pre-registered hierarchy.
>
> Sits ON TOP of the locked D-RERUN-12F (tuned 2160-cell rerun under §4 frozen_hparams). **This doc +
> H博士 approval = the PRE-REGISTRATION; it must be locked BEFORE the 720 FC cells run** (CODEX-A-02).

## Context / problem

§4 equal-budget tuning tuned each arm INDEPENDENTLY (correct defense vs "you crippled the baseline").
Side effect: arms differ in CAPACITY (hidden/layers), so naive edge-ablation ΔIC (L3/L4/L5 − L2) and
graph-vs-no-graph (L2 − L1) mix the edge/graph effect with a capacity difference.

**Verified capacity matrix** (source `artifacts/storya_v21_tune/frozen_hparams.json`, hid/layers):

| contrast | B | C |
|---|---|---|
| L2−L1 (GAT vs MLP) | 64/2 vs 128/3 → CONFOUNDED | 128/1 vs 128/1 → clean |
| L3−L2 (+news) | 32/2 vs 64/2 → CONFOUNDED | 32/2 vs 128/1 → CONFOUNDED (L3 outlier) |
| L4−L2 (+sector) | 32/2 vs 64/2 → CONFOUNDED | 128/1 vs 128/1 → clean |
| L5−L2 (+sec+news) | 32/2 vs 64/2 → CONFOUNDED | 128/1 vs 128/1 → clean |
| L6−L2 (complete vs corr) | 32/2 vs 64/2 → CONFOUNDED | 64/1 vs 128/1 → CONFOUNDED |
| L2s−L2 (SAGE vs GAT) | 128/1 vs 64/2 → CONFOUNDED | 128/1 vs 128/1 → clean |

Univ B confounded across the board; in Univ C only L3−L2 and L6−L2 are confounded.

## Two-family pre-registered hierarchy (LOCKED before FC execution)

**Family 1 — PREDICTIVE / model-selection (unchanged, the locked confirmatory).**
Tuned 2160-cell ladder; DM-HLN pairwise + BH-FDR q=0.05 + Hansen SPA M=9. Answers *"does any TUNED
arm beat TUNED LightGBM"* (best-vs-best). Capacity confound is NOT a defect here — best-vs-best is the
correct question for deployable model ranking.

**Family 2 — CAUSAL edge attribution (NEW; the FC arm, carries its OWN inference).**
Estimand = the **local pure-edge effect at the frozen L2 operating point** (CODEX-A-04: labelled
local-to-L2-architecture, NOT global edge superiority). Per universe, FREEZE the COMPLETE L2 (corr-GAT)
winner HP vector (CODEX-A-03 — all HPs, not just architecture):
- B: hid 64, layers 2, heads 2, dropout 0.3, lr 2e-4, wd 1e-4
- C: hid 128, layers 1, heads 4, dropout 0.1, lr 5e-4, wd 3.4e-5

Vary ONLY the edge set. Confirmatory FC contrasts = **{L3fc corr+news, L4fc corr+sector, L5fc
corr+sec+news} − L2, × {B,C} = 6 contrasts** (CODEX-A-06: L6 EXCLUDED from the family; CODEX-A-07:
L2−L1 EXCLUDED). **Inference**: paired **fold-level seed-averaged** ΔIC (CODEX-A-05: effective n ≈ 12
fold blocks, NOT 120 cells) → **block bootstrap over the 12 folds** for the CI → **BH-FDR q=0.05 over
the 6 contrasts**. Report the matched-ΔIC point + CI + same-sign vs the tuned-ΔIC.

**Reporting hierarchy (locked, no post-hoc primary-switching, CODEX-A-02/AF-03):**
- Family-1 (tuned) governs **deployable model ranking / "does graph help predictively"**.
- Family-2 (FC) governs **causal "do news/sector edges add, model held fixed"**.
- tuned-ΔIC for the edge arms is a **DESCRIPTIVE complement** (best-config edge effect), never a
  post-hoc-selectable causal primary.

## Cells / construction

- FC = 3 new configs (L3fc/L4fc/L5fc) × 2 univ × 12 folds × 10 seeds = **720 cells** →
  `experiments/storya_v21_main12_fc/`.
- **Baseline = the FROZEN tuned-ladder L2 per-day predictions, REUSED** for the paired deltas
  (CODEX-A-08); rerun L2 only as a reproducibility AUDIT (report max |Δpred| / |ΔIC|), not as the
  primary baseline.
- Import-only reuse of the T2-validated 12-fold frozen-snapshot graph + winsor/standardize-train-only
  machinery; the ONLY new design surface is the HP-freeze injection + the FC output dir → Touchpoint 2.

## Honest power expectation (CODEX-A-05 / Q5)

Effective n ≈ 12 fold blocks → MDE@80% ≈ 2.8·sd/√12 ≈ 0.008–0.032 (sd 0.01–0.04). Edge ΔIC ~0.005–
0.016 (pilot/MDE) → ~0.005 UNDERPOWERED, ~0.016 detectable only at low paired-difference sd. Several
FC contrasts will likely be **"directional but not reliable."** The FC arm's value is a **defensible
pre-registered causal frame + reviewer defense**, NOT a guarantee of resolving the edge question.
State this limitation in the paper.

## Deferred / out of FC scope (explicit, to prevent scope creep)

- **L6−L2** (Cn1 dense complete-graph vs sparse corr): topology/density stress, different mechanism →
  if an FC-style matched L6 check is run, it is a SEPARATELY-labelled EXPLORATORY analysis with its own
  multiplicity, NOT in the 6-contrast edge family.
- **L2−L1** (graph-vs-no-graph): covered by Family-1 (best-vs-best); an optional hidden/layers-matched
  MLP-vs-GAT study is deferred (MLP arch doesn't map to GAT heads/message-passing — handle as a
  separate non-graph baseline, not inside FC).

## Sequencing (after H博士 approval = pre-registration lock)

1. main12 (+ l7_hats) read `frozen_hparams.json`, inject per-arm HPs → **Touchpoint 2**. Acceptance
   loop: verify the lambda-fixed `train_lightgbm` actually lands B-L0 λ2=0.47 (the (a)-change closure).
2. tuned 2160-cell rerun → `experiments/storya_v21_main12_tuned/` (don't overwrite pilot).
3. FC arm (720 cells, full L2-vector freeze) → `experiments/storya_v21_main12_fc/`.
4. Family-1 §2a (DM-HLN+BH-FDR+SPA M=9) on the tuned table + Family-2 FC inference (paired fold-level
   seed-avg ΔIC + block bootstrap + BH-FDR/6) → **Touchpoint 3** → analysis.md.

→ progress: 2026-06-17-a | plan: Decision Log 2026-06-17 (FC two-family) | analysis: N/A
