---
reviewer: codex
touchpoint: plan
round: A
target_plan: docs/idea_expansions_2026-07-28.md
findings:
  - id: CODEX-A-01
    severity: CRITICAL
    category: statistics
    claim: "A: Reusing old cells as variance-decomposition anchors can confound design-choice variance with run-vintage and protocol variance."
    evidence: "The plan pools 2160 confirmatory cells, horizon ablations, architecture comparisons, M14 trials sweeps, and loss horse-race runs; these were not generated as one balanced factorial design and may differ in code/config versions, seed reuse, tuning state, feature-basis leakage exposure, and partial axis coverage."
    suggested_fix: "Make a provenance manifest per cell; include run vintage/protocol as a blocking factor; rerun a minimal balanced calibration grid under one frozen pipeline; use legacy anchors only as descriptive or as priors after sensitivity shows variance shares are stable."
    status: FIXED
    resolution_notes: "ACCEPTED in full. Claude verification: the 360 horizon-ablation + 150 arch-comparison + loss horse-race runs are April-2026 Step-0 era (old 5-fold protocol, pre-D-RERUN-12F untuned/old HPs) while the 2160 confirmatory cells are v2.1-frozen 12-fold — pooling them repeats the exact mixed-vintage flaw we ourselves rejected in the D-RERUN-12F decision (plan.md Decision Log 2026-06-12). Sketch amended (idea_expansions §Rev-1): per-cell provenance manifest; vintage as blocking factor; only single-vintage v2.1 cells enter the primary decomposition; new cells under one frozen pipeline; legacy April-era runs demoted to priors/descriptive + sensitivity. Binding version lands in the formal A plan doc."
  - id: CODEX-J-01
    severity: CRITICAL
    category: data-leakage
    claim: "J: The CSI300 data and execution layer is not yet strong enough for confirmatory claims."
    evidence: "The sketch relies on Qlib CN daily with near-PIT membership, then promises a one-week QA gate for membership, adjustment factors, delisting handling, T+1 labels, and limit-lock tradability; any one of these can introduce look-ahead or survivor/tradability bias."
    suggested_fix: "Promote the QA gate to a formal kill gate: independently verify PIT constituents, adjusted prices, delistings/suspensions, index membership dates, and formation-time tradability masks against AkShare/TuShare or another source before model work begins."
    status: FIXED
    resolution_notes: "ACCEPTED. Sketch amended (§Rev-2): QA spike promoted to formal kill gate with pre-declared pass/fail criteria (constituent-date cross-source agreement threshold, adjusted-price reconciliation tolerance, suspension/delisting completeness check, formation-time tradability mask audit) — zero model work before gate passes; gate report saved as artifact."
  - id: CODEX-K-01
    severity: CRITICAL
    category: data-leakage
    claim: "K: The screening-to-confirmation split leaks unless the 21-day forward-label boundary is purged and all edge-selection choices are frozen before confirmation."
    evidence: "The plan selects on folds 1-8 and confirms on folds 9-12, but 21d forward close-to-close labels from the end of fold 8 can overlap fold 9 outcomes; threshold and stability-selection choices could also be tuned after seeing confirmation behavior."
    suggested_fix: "Add at least a 21-trading-day embargo between selection and confirmation; make the fully nested walk-forward design primary; freeze score definitions, thresholds, graph density, and model HPs using selection data only."
    status: FIXED
    resolution_notes: "ACCEPTED. Claude verification: correct — paper 1 applies a 21d purge at every train/test boundary (main.tex:129); the selection/confirmation boundary needs the identical purge. Sketch amended (§Rev-3): 21-trading-day embargo; fully nested walk-forward promoted to PRIMARY design (folds-1-8/9-12 split demoted to illustration); all selection hyperparameters (score defs, thresholds, density, HPs) frozen on selection data with md5 manifest before confirmation, mirroring the frozen_hparams discipline."
  - id: CODEX-M-01
    severity: MAJOR
    category: statistics
    claim: "M: The dead-versus-alive branch rule is underdefined and can misclassify an alive-but-underpowered peer signal as dead."
    evidence: "The rule branches on whether modern implied IC exceeds the Step-3 MDE, but the mapping from anomaly test statistics to rank IC, the uncertainty band around implied IC, and the Step-3 MDE inputs are not specified."
    suggested_fix: "Define the IC mapping before Step 1, compute Step-3 MDE from frozen design assumptions, and branch using a confidence rule such as upper CI below practical detectability for dead or lower CI above MDE for alive; run a minimal Step-3 calibration regardless."
    status: FIXED
    resolution_notes: "ACCEPTED with a simplification that strengthens it: no t-stat→IC mapping needed — the peer-momentum signal is itself a per-stock characteristic, so its daily rank IC vs 21d forward excess return is DIRECTLY measurable on the panel with a block-bootstrap CI. Sketch amended (§Rev-4): three-way pre-registered rule — dead if IC upper CI < detectability floor; alive if lower CI > Step-3 MDE (computed from frozen design assumptions before Step 1); middle → bounded-diagnostic path. Minimal Step-3 calibration runs regardless of branch."
  - id: CODEX-M-02
    severity: MAJOR
    category: prior-art
    claim: "M: The dead-anomaly branch is not novel enough unless the GNN detectability bridge is the primary contribution."
    evidence: "McLean-Pontiff and Chen-Welch already establish post-publication anomaly decay and a low modern non-micro-cap effect-size prior; a fresh large-cap decay curve alone is incremental."
    suggested_fix: "Frame the paper around graph-borne signal detectability and harvest failure: modern peer-momentum IC, measured GNN detection threshold, and whether exact anomaly edges/features are recoverable under the paper-1 protocol."
    status: FIXED
    resolution_notes: "ACCEPTED. Sketch amended (§Rev-5): primary contribution re-declared as the decay↔detectability bridge (measured modern graph-borne IC vs measured GNN detection threshold under the paper-1 protocol); the decay curve is supporting evidence, not the headline. Consistent with Claude's original delta list but now binding."
  - id: CODEX-J-02
    severity: MAJOR
    category: statistics
    claim: "J: The 2020-2025 CSI300 window can let COVID-era graph-density regimes dominate the pooled inference."
    evidence: "The plan treats 24 quarterly folds as a power bonus, while Wade arXiv:2605.19278 reports crisis correlation-graph density exploding during COVID; 21d overlapping labels also reduce the effective independent sample."
    suggested_fix: "Pre-register leave-2020-out, leave-one-year-out, and 2023-2025 matched-window analyses; cluster or block inference by quarter/year; report whether SPA and DM/HLN conclusions survive outside crisis regimes."
    status: FIXED
    resolution_notes: "ACCEPTED. Sketch amended (§Rev-6): pre-registered leave-2020-out + leave-one-year-out + 2023–2025 matched-window sensitivity family; block/cluster inference by quarter retained (21d-block bootstrap as in paper 1); '2× power bonus' framing withdrawn — effective sample discussion added."
  - id: CODEX-A-02
    severity: MAJOR
    category: statistics
    claim: "A: Fractional-factorial sampling plus adaptive seed extension can make variance-share estimates non-identifiable or non-ignorable."
    evidence: "The design has eight axes, selected 2-way interactions, expensive 3-seed subsets, and adaptive extension; if missing cells depend on observed variance or performance, REML/Bayesian variance shares can be biased."
    suggested_fix: "Pre-register the D-efficient design matrix, estimands, missingness rule, and adaptive-extension trigger; run simulation recovery checks showing the planned grid can recover known variance shares and interactions."
    status: FIXED
    resolution_notes: "ACCEPTED. Sketch amended (§Rev-7): full design matrix pre-registered before any run; adaptive-extension triggers defined on pre-committed, outcome-independent criteria (per the project's existing adaptive-seed pre-commit discipline); simulation recovery study (known variance shares injected → recovered) required before real-data analysis — also upgrades the methods contribution."
  - id: CODEX-K-02
    severity: MAJOR
    category: statistics
    claim: "K: Shah-Samworth false-selection control is not automatically valid under dependent financial panels."
    evidence: "The proposed stability selection operates across folds, seeds, and subsamples over about 250K ordered pairs, but returns, labels, sectors, and overlapping horizons are serially and cross-sectionally dependent."
    suggested_fix: "Use blocked complementary-pair subsampling, sector/degree strata, and permutation or planted-null calibration; report any PFER bound as conditional on assumptions rather than unconditional false-selection control."
    status: FIXED
    resolution_notes: "ACCEPTED. Sketch amended (§Rev-8): blocked (temporal) complementary-pair subsampling; sector/degree stratification; planted-null + permutation calibration of the realized false-selection rate; PFER bound reported as conditional-on-assumptions with the empirical calibration as the operative guarantee."
  - id: CODEX-J-03
    severity: MAJOR
    category: design-gaps
    claim: "J: Cross-market conditional comparisons are confounded by different sample windows and universe construction."
    evidence: "The CSI design is PIT CSI300 over 2020Q1-2025Q4, while paper 1 is a fixed S&P 500 survivor snapshot over 2021-2026 with confirmatory folds in 2023-2025."
    suggested_fix: "Separate replication from market-structure claims: add matched-window analyses, and if possible a PIT US robustness panel; otherwise state that US-versus-CN differences mix market, calendar, and survivorship effects."
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "PARTIALLY ACCEPTED. Matched-window (2023Q1–2025Q4) analyses pre-registered as the primary cross-market comparison layer (§Rev-6 covers this window); replication claims and market-structure claims will be separated in the writeup. The PIT US robustness panel is deferred — it is the L8 rebuild, cost-prohibitive within this paper's budget; the market/calendar/universe confound will be stated explicitly as a limitation instead."
  - id: CODEX-M-03
    severity: MAJOR
    category: reproducibility
    claim: "M: The yfinance survivor fallback cannot support a publishable dead-anomaly or decay claim."
    evidence: "The plan itself notes WRDS is preferred for PIT membership and delisting returns; paper 1 already lists survivor snapshot as limitation L8, and decay estimates are sensitive to delistings and historical membership."
    suggested_fix: "Make CRSP/Compustat or equivalent PIT data a hard requirement for the dead-anomaly branch; without it, restrict M to a bounded 2021-2026 mechanism/detectability analysis."
    status: FIXED
    resolution_notes: "ACCEPTED. Sketch amended (§Rev-9): CRSP/Compustat via WRDS is a HARD requirement for the dead-anomaly/decay branch; without WRDS access M is scoped down to the bounded 2021–2026 detectability analysis (which the paper-1 panel supports). WRDS access check is therefore the first action of the M scout."
  - id: CODEX-J-04
    severity: CONCERN
    category: reproducibility
    claim: "J: MASTER-family replication risk is under-scoped."
    evidence: "The sketch says official code and frozen settings, but CSI SOTA papers often differ in horizon, loss, universe filters, and Qlib conventions; failure to reproduce the reference metric would weaken the home-turf confrontation."
    suggested_fix: "Add a reproduction smoke test against the published MASTER/Qlib-style metric before the equal-budget protocol; if it fails, label the arm as MASTER-family rather than MASTER replication."
    status: FIXED
    resolution_notes: "ACCEPTED. Sketch amended (§Rev-10): reproduction smoke test against the published Qlib-convention metric added as a pre-step; the arm is labeled 'MASTER-family (L6-class)' unless the smoke test reproduces the reference metric within tolerance."
  - id: CODEX-K-03
    severity: CONCERN
    category: design-gaps
    claim: "K: A density-matched random graph is too weak as the sole null for selected sparse edges."
    evidence: "A selected lead-lag graph can differ from random graphs in degree concentration, directionality, sector locality, and size structure, so a density-only null may attribute generic graph topology to signal."
    suggested_fix: "Use degree-, direction-, sector-, and size-stratified random controls, plus signed/weighted ablations where applicable."
    status: FIXED
    resolution_notes: "ACCEPTED. Sketch amended (§Rev-11): null family expanded to degree-/direction-/sector-/size-stratified rewired controls (configuration-model style) plus signed/weighted ablations."
summary:
  critical: 3
  major: 7
  concern: 2
  fixed_before_reply: 0
overall_verdict: BLOCK-EXECUTION
---

# Review body (Codex, verbatim)

**A Assessment**

High novelty only if the estimand is genuinely nonstandard error for deep cross-sectional ranking, not a repackaging of paper-1 robustness. The current design is statistically fragile because the reused cells were not born from one factorial experiment. Treating them as exchangeable observations in REML/Bayes variance decomposition risks estimating "what changed across project history" rather than "what design choices matter."

Feasibility is medium, but only after a clean design matrix and provenance layer exist. The minimum acceptable version is a frozen-pipeline calibration grid plus sensitivity excluding all legacy anchors. Universe C also needs explicit leakage-status handling, because its feature-basis selection caveat can inflate the universe axis.

**J Assessment**

This is the strongest full-paper spine if the data gate passes. It has a clean audience: CSI300 is where many GNN-stock claims live, and the conditional GW layer directly answers the "when" question that regime-gate papers leave untested.

The main risk is not compute; it is market-data validity. PIT membership, T+1 alignment, adjusted prices, suspensions, limit-up/limit-down, and delisting handling must be solved before any confirmatory run. Also, do not sell 24 folds as simply double power. COVID can be the treatment here because correlation graphs behave very differently in crises, and 21d labels are heavily overlapping.

**K Assessment**

K is the most intellectually interesting and the riskiest. The sparse-subgraph hypothesis is a real rescue path for graph value, and the planted sparse control is a strong idea. But the current inference chain is not yet credible: selection, stability control, and confirmation must be fully separated, purged, and calibrated under dependence.

This should not be paper 2. It belongs after the planted-signal machinery is hardened and after M/J reveal whether there is a live graph-borne signal worth rescuing.

**M Assessment**

M is the best cheap scout and the most natural mechanism sequel to paper 1. Its publishable contribution cannot be "peer momentum decayed"; that is too close to McLean-Pontiff and Chen-Welch. The contribution has to be the bridge from measured modern graph-borne signal strength to GNN detectability and harvest failure.

The WRDS gate is decisive. With PIT/delist data, M can become a credible mechanism paper. With yfinance survivor history, it should be demoted to a bounded diagnostic or a section supporting J/paper 1.

**Sequencing**

Recommended order: M Step-1 scout and WRDS check immediately; J data QA gate in parallel; if J passes, make J the next full confirmatory paper. M-full proceeds only if PIT data and the branch rule are fixed. A should wait until there is a clean factorial/provenance plan. K should remain last or be absorbed as a sparse-edge arm after the signal-detectability machinery is proven.

---

# Claude disposition summary (Round A processing, 2026-07-28)

All 12 findings ACCEPTED (11 FIXED in the design sketches via `docs/idea_expansions_2026-07-28.md` §Rev-1..11; 1 ACCEPTED-AS-CONCERN: CODEX-J-03's PIT-US panel deferred with explicit confound disclosure). No findings REJECTED. Verification notes recorded per finding above; the CODEX-A-01 verification confirmed the April-2026 Step-0 runs (old 5-fold protocol) are a different vintage from the v2.1-frozen 2160 cells — Codex's core objection stands on our own project history.

BLOCK-EXECUTION disposition: no execution was pending (sketches are PENDING H博士 selection). The block converts to: the formal plan doc for whichever idea H博士 selects must incorporate the corresponding §Rev fixes and then return for Codex Round B before implementation.
