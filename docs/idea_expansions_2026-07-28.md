# Idea Expansions: A / J / K / M — Full Design Sketches

> Date: 2026-07-28. Expansion of the four candidates H博士 flagged from
> `docs/lit_scan_2026-07-26.md` §4/§4b. These are DESIGN SKETCHES, not
> committed plans — the selected one(s) get a formal plan doc + Rule 9
> Touchpoint 1 before any implementation. Compute figures are estimates
> extrapolated from D-RERUN-12F actuals (~3.5–4.5 A100·days for 2160 cells
> + tuning), not measured.

---

## A. Nonstandard Errors in Deep Cross-Sectional Ranking

**RQ.** How much of the variation in reported performance (rank IC, net
Sharpe, turnover) of deep cross-sectional ranking pipelines is attributable
to each design choice — architecture (graph on/off nested), loss, label
horizon, label transform, tuning budget, seed, feature universe, split
protocol — and how large is the nonstandard error (dispersion across
defensible designs) relative to the sampling standard error?

**Anchors.** Menkveld et al. (JF 2024) NSE concept; Chen–Hanauer–Kalsbach
SSRN 5031755 (1,056 models, 7 choices, NSE=1.59×SE — monthly, classic ML,
no loss/seed/graph axes); Lalwani et al. (NSE 1.99); foil arXiv:2603.16886
(seed=0.01% of variance in MSE forecasting) vs our GAT-21d 5-seed CV=55%.

**Design axes** (8): architecture {LightGBM, MLP, corr-GAT, dense-attn};
loss {MSE, IC-surrogate, pairwise, ListMLE}; horizon {1d, 5d, 21d};
label transform {raw excess, z-scored, cross-sectional rank}; tuning budget
{10, 30, 90 trials}; seed {10 canonical}; universe {B, C}; split
{expanding-12, sliding-14}. Existing cells reused as grid anchors:
2160 confirmatory + 360 horizon ablation + 150 arch comparison + M14
trials sweep + loss horse-race runs. New cells via pre-registered
fractional-factorial / D-efficient sampling (NOT full factorial); 3-seed
subsets on expensive arms with pre-committed adaptive extension.

**Statistics.** Crossed random-effects (REML) and/or Bayesian hierarchical
variance decomposition → variance share per axis + chosen 2-way
interactions (arch×loss, arch×horizon, arch×universe only, pre-registered);
NSE/SE ratio for deep ranking; sign-flip share of design space for the
GNN−MLP contrast (links to paper 1); turnover as co-primary outcome
(design choices can move economic behavior at flat IC — Poggio-lab
optimizer result arXiv:2603.02620 as motivation).

**Headline candidates.** (i) variance ranking of axes (expected: seed+loss
+horizon ≫ graph); (ii) NSE/SE for deep ranking vs CHK's 1.59; (iii) "X%
of defensible designs flip the sign of the graph effect"; (iv) seed-domain
resolution vs 2603.16886.

**Compute.** ~800–1200 new cells ≈ 3–6 A100·days. **Data:** all existing.
**Venue.** ICLR 2028 / NeurIPS (D&B possible with grid release) / KDD.
**Risks.** Salami-slicing perception (mitigate: new estimand + 5 new axes
+ new outcomes + NSE literature anchor); unbalanced-grid identification
(Bayesian priors + sensitivity); compute creep (pre-registered cap);
CHK-team concurrency (they are asset-pricing/monthly — window exists).
**Absorbs** N (seed-ensemble vs architecture, computable from saved
predictions) and H (graph-construction axes) if desired.

---

## J. Cross-Market Confirmatory Replication (CSI300 home turf)

**RQ.** Do the graph penalty (L2−L1<0) and the SPA non-rejection replicate
on CSI300 — the benchmark where nearly all GNN-stock SOTA claims are
produced (ACT, USTGCN, StockMamba, MASTER, StockMixer)? Does conditional
(GW) analysis reveal market-structure-dependent graph value?

**Why home turf.** Mutually incompatible reported ICs (0.069 vs 0.154 on
the same universe); THGNN's graph ≈ our α1 (trailing corr) and THGNN is a
Chinese-market model; MASTER (AAAI'24, CSI) fits our L6 dense-attention
family → direct equal-budget confrontation of a flagship SOTA claim under
confirmatory statistics, which nobody has done.

**Design.** Data: Qlib CN daily (free; near-PIT membership — improves on
our L8), 1-week data-QA spike FIRST (cross-check vs AkShare/TuShare;
adjustment factors; delisting handling). Universe: CSI300 PIT. Label:
mirror paper 1 (21d fwd c-t-c excess vs index, z-scored) with CN execution
conventions (T+1; standard Qlib Ref(close,-2)/Ref(close,-1) lag structure;
limit-locked stocks excluded from tradable set per CN-literature
convention — all pre-registered). Test window: 2020Q1–2025Q4 = 24
expanding quarterly folds (double paper 1's inference sample — power
bonus). Arms (scoped ladder, 5): L0 LightGBM / L1 MLP / L2 corr-GAT (α1)
/ L5s SW-industry sector / L6 MASTER-family. 10 seeds; 30-trial Optuna
frozen pre-confirmatory; SPA + DM/HLN + BH-FDR + MDE; CN cost model
(commission ~2.5bp + stamp 5bp sell + slippage; pre-registered grid);
Family-2 fixed-op edge attribution (corr vs sector vs shuffled); planted-
signal positive control ported.

**Conditional layer (absorbs B).** GW conditional predictive ability on
daily loss differentials; pre-registered conditioning set: T-1 index vol,
cross-sectional dispersion, corr-graph density, turnover regime, limit-hit
fraction (CN-specific); FDR over conditioning family; run the same tests
on the US panel → cross-market comparison of conditional structure.

**Outcome branches (all publishable).** (i) penalty replicates →
universality; (ii) graph helps on CSI300 → first confirmed "when" +
microstructure mechanism hypotheses (T+1, ±10% limits, ~80% retail
turnover, higher average pairwise corr); (iii) MASTER-family fails to beat
equal-budget LightGBM under SPA → treadmill inflation quantified on home
turf (near-certain to be informative).

**Compute.** ~720 cells + tuning ≈ 2–3 A100·days. **Venue.** KDD 2027 /
ICLR 2028 / ICAIF 2027. **Risks.** Qlib data quality (QA spike gate);
label/execution conventions under T+1 (pre-register); MASTER
reimplementation fidelity (official code, frozen); "just another market"
(mitigated by conditional layer + home-turf confrontation framing).

---

## K. The Sparse-Subgraph Hypothesis

**RQ.** Paper 1: the dense correlation graph hurts at the tuned operating
point. The learned-sparse trend (FinMamba, STN-TGAT, GAPNet) implicitly
bets that a sparse edge subset helps — never tested with error control.
Formally: does a sparse edge set S exist such that message passing over S
improves ranking, and can S be identified stably with false-selection
control? Theory link: arXiv:2604.17166 (capacity = search space for sparse
structure); Lo–MacKinlay lead-lag (documented lead-lag is sparse and
directed, large→small, within-industry).

**Design (3 stages).**
1. *Screening (CPU-cheap).* Per-fold, per candidate edge (i,j): does
   lagged r_i improve prediction of j's 21d excess return beyond j's own
   features (regularized per-fold regression over ~250K ordered pairs)?
   Produces per-fold edge scores = an estimated lead-lag matrix.
2. *Stability selection.* Complementary-pairs stability selection
   (Shah–Samworth) across folds × seeds × subsamples → selected edge set
   with bounded expected false selections. Edge selection strictly
   walk-forward (train-window data only) to avoid selection leakage.
   Alternative (robustness): learnable L0/concrete mask over adjacency at
   the frozen GAT operating point.
3. *Confirmatory.* Pre-registered contrasts: GAT(selected sparse graph) vs
   GAT(density-matched random graph) vs MLP; selection on folds 1–8,
   confirmation on folds 9–12 (plus fully nested walk-forward variant);
   DM/HLN + BH + MDE. Positive control via planted-sparse-signal (inject
   signal on a KNOWN sparse subgraph; verify the pipeline recovers those
   exact edges — reuses/extends idea-I machinery).

**Secondary deliverables (non-null regardless of branch).** Edge-set
persistence (Jaccard across folds) → edge-decay curve, connecting graph
value to the alpha-decay literature; alignment of selected edges with
sector/size/lead-lag priors.

**Compute.** Screening route: ~240 confirmatory cells ≈ 1–1.5 A100·days
(+CPU weeks); mask route ~3–4 A100·days. **Venue.** ICLR/NeurIPS main.
**Risks.** Highest-variance idea: selection may not transfer
(non-stationary edges) → falls back to the decay-curve finding (weaker
venue); power at sparse scale (MDE upfront + planted-sparse control);
run AFTER I-machinery hardened, or absorb into J/M as an arm.

---

## M. The Peer-Momentum Paradox (mechanism paper)

**RQ.** Linked-firm momentum is a documented anomaly (Moskowitz–Grinblatt
1999 industry momentum; Cohen–Frazzini 2008 economic links; Lo–MacKinlay
lead-lag) and is the motivating citation of the GNN-stock genre (RSR/
THGNN cite economic-links work). Paper 1 shows the graph hurts. Paradox:
either (D) the anomaly is dead in the modern large-cap sample —
post-publication decay (McLean–Pontiff; Chen–Welch 7bp/month base rate) —
so the genre's motivation is stale; or (H) it is alive and GNNs fail to
harvest it — a localizable representation/optimization failure.

**Design (3 steps).**
1. *Anomaly audit (1–2 weeks, CPU).* Classical tests on our panel AND
   extended history (2000–2026; prefer CRSP/Compustat via USC WRDS access
   if available → proper PIT + delisting returns; else yfinance survivor
   panel with disclosed bounds): (a) industry momentum (VW 6-1); (b) peer
   momentum: FM regressions of r_{i,t+1} on lagged connected-stock returns
   (sector peers / corr top-k / Wikidata links à la RSR), controlling own
   momentum, reversal, size, beta, NW-HAC; (c) Lo–MacKinlay weekly
   cross-autocorrelations. Deliverable: effect-size decay curve by
   subperiod (pre-2010 / 2010–2020 / 2021–2026).
2. *Branch adjudication (pre-registered rule).* Branch by whether the
   modern implied IC of the peer signal exceeds the MDE of the Step-3
   designs. Dead → headline: "the graph-borne signal that motivates GNN
   stock models decayed before the models arrived"; quantify implied
   achievable IC vs detectability. Alive → Step 3.
3. *Harvest-failure localization (only if alive; ~1 A100·day).* Handicap
   ladder: (i) hand-crafted peer-momentum feature added to MLP (is
   explicit featurization sufficient?); (ii) GAT with edges = the exact
   anomaly links (directed industry/economic links) vs corr edges (edge-
   mismatch hypothesis); (iii) planted signal calibrated at the anomaly's
   MEASURED modern strength → is peer momentum below the GAT detection
   threshold at N=501, T≈750? If yes, the paradox dissolves
   quantitatively into a power statement (unifies with idea I and paper
   1's positive control: GAT recovers planted IC≈0.047; measured modern
   peer-momentum implied IC likely ≪ that).

**Novelty vs McLean–Pontiff/Chen–Welch.** They document decay; our delta =
(a) modern 2021–2026 large-cap estimates of specifically graph-borne
anomalies, (b) the quantitative bridge decay ↔ GNN detectability
thresholds, (c) the indictment of the GNN genre's motivation citations.

**Compute.** Steps 1–2 CPU/local; Step 3 ≈ 1 A100·day. **Venue.**
ICLR/ICML "understanding" paper, or ICAIF; natural mechanism-sequel to
paper 1. **Risks.** Middle outcome (t≈1.5, neither branch clean) —
mitigate: decay curve is the primary deliverable regardless + pre-
registered branch rule; survivorship in extended history (WRDS if
available; industry momentum VW is less exposed); check WRDS access early
(it changes data quality tier).

---

## Program arc (thesis-shaped)

Paper 1 (does the graph help? → no, at this operating point) → **M** (why
not? → signal decay / detectability) → **J** (where else? → cross-market +
conditional structure) → **A** (what dominates instead? → design-choice
variance) → **K** (is any graph value rescuable? → sparse substructure).
Shared instruments: planted-signal machinery (I) powers M-step-3 and
K-stage-3; GW conditional layer (B) lives inside J; N/H live inside A.

**Sequencing (proposed, pre-decision).** Aug–Sep: ICLR rewrite is the main
line; M-step-1 scout (1–2 wks, CPU) + WRDS access check can run in gaps.
Oct (post-submission): commit paper-2 spine (J vs M-full, informed by the
scout) → formal plan doc → Codex Touchpoint 1. A launches after paper-2
confirmatory runs are in flight (shares Colab efficiently). K last / or
absorbed.

---

## Touchpoint-1 Round A revisions (Codex review 2026-07-28; full review + per-finding dispositions in `artifacts/reviews/2026-07-28_codex_plan_A.md`)

Verdict: 3 CRITICAL + 7 MAJOR + 2 CONCERN, BLOCK-EXECUTION → all 12 findings
accepted (11 fixed here, 1 accepted-as-concern). Binding versions land in the
formal plan doc of whichever idea is selected; that doc returns for Round B.

- **Rev-1 (A, CODEX-A-01 CRITICAL).** Legacy runs are NOT poolable as-is:
  the April-2026 Step-0 runs (360 horizon + 150 arch + loss horse race) are
  old-5-fold/old-HP vintage vs the v2.1-frozen 2160 cells. Primary
  decomposition uses single-vintage v2.1 cells + new cells under one frozen
  pipeline only; per-cell provenance manifest; vintage as blocking factor;
  April-era runs demoted to priors/descriptive + sensitivity.
- **Rev-2 (J, CODEX-J-01 CRITICAL).** CN data QA spike promoted to a formal
  kill gate with pre-declared pass/fail criteria (cross-source constituent-
  date agreement, adjusted-price reconciliation tolerance, suspension/
  delisting completeness, formation-time tradability masks). Zero model work
  before the gate passes; gate report saved as artifact.
- **Rev-3 (K, CODEX-K-01 CRITICAL).** 21-trading-day embargo at the
  selection/confirmation boundary; fully nested walk-forward is PRIMARY;
  all selection-stage choices (score defs, thresholds, density, HPs) frozen
  on selection data with md5 manifest before any confirmation run.
- **Rev-4 (M, CODEX-M-01).** Branch rule concretized: peer momentum is a
  characteristic → measure its daily rank IC vs 21d forward excess return
  directly (no t→IC mapping), block-bootstrap CI; dead if upper CI <
  detectability floor, alive if lower CI > Step-3 MDE (MDE computed from
  frozen design assumptions BEFORE Step 1), else bounded-diagnostic path;
  minimal Step-3 calibration runs regardless.
- **Rev-5 (M, CODEX-M-02).** Primary contribution re-declared: the
  decay↔detectability bridge (measured modern graph-borne IC vs measured
  GNN detection threshold under the paper-1 protocol). Decay curve =
  supporting evidence, not headline.
- **Rev-6 (J, CODEX-J-02/J-03).** Pre-registered sensitivity family:
  leave-2020-out, leave-one-year-out, 2023Q1–2025Q4 matched-window (also
  the primary cross-market comparison layer); block/cluster inference by
  quarter; "2× power" framing withdrawn (overlapping 21d labels reduce
  effective sample). Replication claims separated from market-structure
  claims; US-vs-CN differences disclosed as mixing market/calendar/universe
  effects (PIT-US panel deferred = accepted-as-concern).
- **Rev-7 (A, CODEX-A-02).** Design matrix pre-registered before any run;
  adaptive-extension triggers outcome-independent and pre-committed;
  simulation recovery study (inject known variance shares → recover)
  required before real-data analysis.
- **Rev-8 (K, CODEX-K-02).** Blocked temporal complementary-pair
  subsampling; sector/degree strata; planted-null + permutation calibration
  of realized false-selection rate; PFER bound reported as conditional.
- **Rev-9 (M, CODEX-M-03).** WRDS/CRSP = HARD requirement for the
  dead-anomaly/decay branch; without it M scopes down to the bounded
  2021–2026 detectability analysis. WRDS check = first scout action.
- **Rev-10 (J, CODEX-J-04).** MASTER reproduction smoke test vs published
  Qlib-convention metric added as pre-step; arm labeled "MASTER-family
  (L6-class)" unless reproduction within tolerance.
- **Rev-11 (K, CODEX-K-03).** Null family expanded: degree-/direction-/
  sector-/size-stratified rewired controls (configuration-model style) +
  signed/weighted ablations; density-only null demoted.

Codex's comparative verdict matches ours with one modification: J = strongest
paper-2 spine IF the data gate passes; M scout + WRDS check first; **A must
wait for a clean factorial/provenance plan (not merely "after ICLR")**; K
last or absorbed as an arm.

---

## Touchpoint-1 Round B revisions (Codex gpt-5.6-sol, 2026-07-28; full review in `artifacts/reviews/2026-07-28_codex_plan_B.md`)

Round B verdict: 0 CRITICAL + 8 MAJOR + 1 CONCERN, **PROCEED-WITH-FIXES**.
Round A diff: 8 FIXED + 4 PARTIALLY-FIXED (residues discharged below).
Sequencing unchanged. All 9 new findings accepted (8 fixed, 1 as-concern).

- **Rev-12 (A, CODEX-B-01; discharges A-02 residue).** The formal A plan must
  publish the explicit fractional design matrix with rank/aliasing
  diagnostics, minimum-cells-per-component floor, per-cell compute costing,
  and pre-committed stopping rules. Full cross (~17,280 runs) explicitly
  disclaimed as not the design.
- **Rev-13 (A, CODEX-B-02).** Primary estimand = FINITE-DESIGN dispersion
  (NSE over the pre-registered enumerated design universe, per Menkveld/CHK
  convention) with date/fold-block-resampled uncertainty; REML/Bayesian
  variance shares demoted to complementary/sensitivity.
- **Rev-14 (J, CODEX-B-03).** CN corr-edge spec inherits paper-1 α1 verbatim
  (126d trailing, |ρ|≥0.6, T-1 adjusted-close returns, min-history rule);
  deviations pre-registered; per-fold formation-time edge manifests archived
  with md5 before confirmatory runs.
- **Rev-15 (J, CODEX-B-04).** Complete GW contrast universe (conditioning
  vars × arm-contrasts × markets × windows) enumerated and pre-registered as
  ONE family; hierarchical/BH-FDR across the full cell count; 21d-block /
  quarter-clustered covariance; no re-testing outside the registered universe.
- **Rev-16 (J, CODEX-B-05, as-concern).** Positioning pre-registered:
  confirmatory cross-market falsification on the incumbent benchmark (the
  crowdedness IS the point); local SOTA reproduction scoped to the MASTER
  smoke test; exclusions pre-registered.
- **Rev-17 (K, CODEX-B-06).** Confirmatory arms expanded with selected-edge
  NON-GNN baselines (peer-aggregate features → LightGBM/MLP; linear edge
  aggregation). Mechanism claim requires GAT(selected) > feature-matched
  non-GNN selected-edge baseline, not merely nulls/MLP.
- **Rev-18 (K, CODEX-B-07).** Screening on sector/factor-residualized
  returns (primary) with liquidity/asynchrony controls; raw-return screening
  demoted to comparison; robustness across residualization specs
  pre-registered.
- **Rev-19 (M, CODEX-B-08).** PIT peer definitions primary (historical
  sector classifications via WRDS/Compustat when available + trailing-corr
  peers); Wikidata only with dated relations + documented vintage, else
  descriptive-only. (Self-note: current-snapshot Wikidata over 2000–2026 is
  the exact look-ahead we criticized in RSR/STHAN-SR — caught by Round B.)
- **Rev-20 (M, CODEX-B-09; discharges M-02 residue).** ONE primary
  characteristic pre-registered: PIT sector-peer momentum on the primary
  modern window drives the branch rule; all other definitions secondary
  under FDR within a declared family. Ex ante estimand fixed: primary-
  characteristic modern rank IC vs frozen GNN detection threshold.
- **Residues.** J-03: PIT-US deferral stands (disclosed confound). K-02:
  PFER conditionality stands as disclosed; planted-null calibration is the
  operative guarantee.

**Gate to implementation**: the selected idea's formal plan doc carries
§Rev-1..20 as binding; Round C only if the plan deviates from them.

→ progress: 2026-07-28-a, 2026-07-28-b, 2026-07-28-e | plan: 2026-07-26-a (amended) | analysis: N/A
