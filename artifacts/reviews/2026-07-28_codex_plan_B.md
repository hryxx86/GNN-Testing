---
reviewer: codex
touchpoint: plan
round: B
model: gpt-5.6-sol (xhigh)  # Round A was gpt-5.5 xhigh; model switched per H博士 2026-07-28
target_plan: docs/idea_expansions_2026-07-28.md
round_a_diff:
  - {id: CODEX-A-01, status: FIXED, reasoning: "Rev-1 removes legacy April-2026 cells from the primary decomposition and keeps vintage only as provenance/sensitivity."}
  - {id: CODEX-A-02, status: PARTIALLY-FIXED, reasoning: "Outcome-independent triggers and recovery simulation help, but the supplied revision still lacks the actual estimable design/aliasing plan."}
  - {id: CODEX-J-01, status: FIXED, reasoning: "Rev-2 makes CN data QA a formal kill gate with pass/fail criteria and no model work before passage."}
  - {id: CODEX-J-02, status: FIXED, reasoning: "Rev-6 adds leave-2020/year checks plus 21d block/quarter inference, addressing regime dominance and label overlap."}
  - {id: CODEX-J-03, status: PARTIALLY-FIXED, reasoning: "Matched-window comparison fixes the calendar-window issue, but PIT-US deferral leaves universe/market construction confounded."}
  - {id: CODEX-J-04, status: FIXED, reasoning: "Rev-10 prevents exact-MASTER overclaiming via a reproduction smoke test and L6-class relabeling if tolerance fails."}
  - {id: CODEX-K-01, status: FIXED, reasoning: "Rev-3 adds a 21d embargo, nested walk-forward primary design, and frozen selection manifests."}
  - {id: CODEX-K-02, status: PARTIALLY-FIXED, reasoning: "Blocked CPSS and calibration reduce the risk, but PFER validity remains explicitly conditional under dependent panel data."}
  - {id: CODEX-K-03, status: FIXED, reasoning: "Rev-11 replaces the weak density-only null with degree/direction/sector/size rewires and signed/weighted ablations."}
  - {id: CODEX-M-01, status: FIXED, reasoning: "Rev-4 defines the branch rule using direct daily rank IC, block-bootstrap CIs, and pre-Step-1 MDE thresholds."}
  - {id: CODEX-M-02, status: PARTIALLY-FIXED, reasoning: "Rev-5 improves positioning, but the decay-detectability bridge still needs an ex ante estimand versus anomaly-decay prior art."}
  - {id: CODEX-M-03, status: FIXED, reasoning: "Rev-9 makes WRDS/CRSP mandatory for the decay/dead branch and limits any fallback to a bounded 2021-2026 detectability analysis."}
findings:
  - id: CODEX-B-01
    severity: MAJOR
    category: feasibility
    claim: "A's revised 8-axis design is not executable as written."
    evidence: "The axes imply 4x4x3x3x3x10x2x2 = 17,280 cells before Optuna trials and 12/14-fold training, yet compute is listed as 3-6 A100-days."
    suggested_fix: "Use a pre-registered fractional/block design with rank/aliasing diagnostics, minimum cells per component, and compute-backed stopping rules."
    status: FIXED
    resolution_notes: "ACCEPTED (subsumes the A-02 PARTIALLY-FIXED residue). Sketch amended (§Rev-12): the formal A plan MUST publish the explicit fractional design matrix with rank/aliasing diagnostics, minimum-cells-per-component floor, per-cell compute costing, and pre-committed stopping rules; the full-cross enumeration is explicitly disclaimed as not the design."
  - id: CODEX-B-02
    severity: MAJOR
    category: statistics
    claim: "A's variance-component estimand is not yet justified as a population claim."
    evidence: "Architectures, losses, horizons, labels, universes, and splits are fixed researcher-chosen levels sharing the same panels; crossed REML/Bayes random effects may imply unjustified exchangeability."
    suggested_fix: "Declare a finite-design estimand or justify exchangeability, and use date/fold-block uncertainty with model-based variance shares as sensitivity."
    status: FIXED
    resolution_notes: "ACCEPTED — technically the sharpest new finding. Sketch amended (§Rev-13): primary estimand re-declared as FINITE-DESIGN dispersion (NSE over the pre-registered enumerated design universe, matching the Menkveld/CHK convention) with date/fold-block-resampled uncertainty; REML/Bayesian variance shares demoted to complementary decomposition/sensitivity, not the headline claim."
  - id: CODEX-B-03
    severity: MAJOR
    category: data-leakage
    claim: "J still has a graph-construction leakage and reproducibility opening."
    evidence: "The corr-GAT arm is defined as trailing corr |rho|>=0.6, but window length, cadence, adjustment basis, min-history rules, and T-1 edge snapshots are unspecified."
    suggested_fix: "Freeze and archive per-fold edge manifests built only from information available at formation time."
    status: FIXED
    resolution_notes: "ACCEPTED. Sketch amended (§Rev-14): CN corr-edge spec inherits paper-1 α1 verbatim (126d trailing window, |ρ|≥0.6, T-1 close-based adjusted returns, min-history rule) with any CN deviation pre-registered; per-fold edge manifests (formation-time information only) archived with md5 before confirmatory runs."
  - id: CODEX-B-04
    severity: MAJOR
    category: statistics
    claim: "J's conditional predictive ability layer can overstate discoveries."
    evidence: "GW tests span conditioning variables, arms, markets, and windows, while FDR is described only over the conditioning family."
    suggested_fix: "Predefine the full GW contrast universe and apply 21d-block or quarter-clustered inference with hierarchical/FDR control across all tested cells."
    status: FIXED
    resolution_notes: "ACCEPTED. Sketch amended (§Rev-15): the complete GW contrast universe (conditioning vars × arm-contrasts × markets × windows) is enumerated and pre-registered as ONE family; hierarchical FDR (or BH over the full cell count) + 21d-block/quarter-clustered covariance; no per-slice re-testing outside the registered universe."
  - id: CODEX-B-05
    severity: CONCERN
    category: prior-art
    claim: "J's novelty is exposed to China-GNN prior art."
    evidence: "Qlib CSI300, MASTER-family, and THGNN-like correlation edges are already central settings in CN stock-GNN papers."
    suggested_fix: "Position J as confirmatory cross-market falsification, add direct local SOTA reproductions where feasible, or pre-register exclusions."
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "The crowdedness IS the point — J's claim is confirmatory falsification on the incumbent benchmark, not a new architecture. Sketch amended (§Rev-16): positioning statement pre-registered (confirmatory cross-market falsification; local SOTA reproduction scoped to the MASTER smoke test; exclusions pre-registered). No further action."
  - id: CODEX-B-06
    severity: MAJOR
    category: design-gaps
    claim: "K does not yet identify the sparse-subgraph mechanism."
    evidence: "GAT(selected graph) versus rewired nulls and MLP cannot distinguish message passing from simpler selected-peer or lead-lag features."
    suggested_fix: "Add selected-edge non-GNN baselines such as peer-feature LightGBM/MLP, graph-regularized regression, and linear edge aggregation."
    status: FIXED
    resolution_notes: "ACCEPTED — matches paper 1's own feature-matched-contrast philosophy. Sketch amended (§Rev-17): confirmatory arm set expanded with selected-edge NON-GNN baselines (peer-aggregate features into LightGBM/MLP; linear edge aggregation); the mechanism claim requires GAT(selected) to beat the feature-matched non-GNN selected-edge baseline, not merely nulls/MLP."
  - id: CODEX-B-07
    severity: MAJOR
    category: statistics
    claim: "K's screening target can select common-factor and microstructure artifacts."
    evidence: "Lagged r_i predicting j's 21d excess return beyond j's own features can reflect sector shocks, nonsynchronous trading, size/liquidity, or residual market autocorrelation."
    suggested_fix: "Screen on factor/sector-residualized returns with liquidity/asynchrony controls and require pre-registered robustness across residualized specifications."
    status: FIXED
    resolution_notes: "ACCEPTED. Sketch amended (§Rev-18): screening runs on sector/factor-residualized returns (primary) with liquidity/asynchrony controls; raw-return screening demoted to comparison; robustness across residualization specs pre-registered."
  - id: CODEX-B-08
    severity: MAJOR
    category: data-leakage
    claim: "M's historical peer-link audit has a point-in-time data risk."
    evidence: "Step 1 uses Wikidata links over 2000-2026, but no PIT relationship timestamping or historical classification protocol is specified."
    suggested_fix: "Make PIT sector and correlation peers primary; use Wikidata only with dated relationships and documented data vintage, otherwise demote it to descriptive evidence."
    status: FIXED
    resolution_notes: "ACCEPTED — and noted with humility: a current-snapshot Wikidata graph applied over 2000–2026 is exactly the look-ahead we criticized in RSR/STHAN-SR in docs/lit_benchmark_2026-07-03.md. Sketch amended (§Rev-19): PIT sector (historical classifications via WRDS/Compustat when available) + trailing-correlation peers are the PRIMARY peer definitions; Wikidata used only with dated relations + documented vintage, else descriptive-only."
  - id: CODEX-B-09
    severity: MAJOR
    category: statistics
    claim: "M's alive/dead branch can still be selected across multiple anomaly definitions."
    evidence: "The audit includes sector peers, corr top-k, Wikidata links, FM regressions, and Lo-MacKinlay tests, but Rev-4 does not name one primary peer definition or family-level branch rule."
    suggested_fix: "Pre-register one primary peer-momentum characteristic and modern window; handle all variants with closed testing, FDR, or a worst-case branch rule."
    status: FIXED
    resolution_notes: "ACCEPTED. Sketch amended (§Rev-20): ONE primary characteristic pre-registered — PIT sector-peer momentum (the Moskowitz–Grinblatt-adjacent classical form) on the primary modern window — drives the branch rule; all other definitions (corr top-k, Wikidata, Lo–MacKinlay) are secondary under FDR within a declared family; also discharges the M-02 PARTIALLY-FIXED residue by fixing the ex ante estimand: primary-characteristic modern rank IC vs frozen GNN detection threshold."
summary:
  critical: 0
  major: 8
  concern: 1
  fixed_before_reply: 0
overall_verdict: PROCEED-WITH-FIXES
---

# Review body (Codex gpt-5.6-sol, verbatim)

**Per-Idea Assessment**

A is cleaner after Rev-1/Rev-7, but remains the most exposed to execution drift. Its novelty over NSE/model-selection prior art depends on a sharply defined deep-ranking estimand: loss/label/horizon/graph interactions and GNN-vs-MLP sign flips, not just a larger variance audit. Feasibility is the main blocker.

J is still the best candidate for the paper-2 spine if the CN data gate and MASTER-family smoke test pass. The revisions materially improved leakage control and inference, but corr-edge construction and the GW conditional layer must be frozen more tightly. Prior-art risk is real because Qlib CSI300 and MASTER/THGNN-style setups are crowded.

K improved on leakage and null controls, but its identification remains weakest. As written, it may rediscover known lead-lag/common-factor structure and then show that a selected graph is useful, without proving that sparse GNN message passing is the mechanism. Prior-art collision with Lo-MacKinlay-style lead-lag and network Granger work is high.

M remains the cheapest high-value scout. Rev-4/5/9 make the branch logic much more defensible, but the plan must lock a PIT peer definition and family-level branch rule before looking at results. The decay-to-detectability bridge is promising only if it is specified as a graph-borne IC versus frozen GNN detection-threshold estimand.

**Priority**

The Round A sequencing still stands.

1. Run M scout first, with WRDS/CRSP availability as the first check, but freeze PIT peer definitions and multiplicity rules before the anomaly audit.
2. Run J data QA and MASTER smoke test in parallel; if both pass, J should be the paper-2 spine.
3. Hold A until the factorial/fractional design and finite-design estimand are explicit and compute-backed.
4. Keep K last unless it adds simple selected-edge baselines and residualized screening, because otherwise its mechanism claim is too easy to attack.

---

# Claude disposition summary (Round B processing, 2026-07-28)

All 9 new findings ACCEPTED (8 FIXED via §Rev-12..20 in the design sketches; 1 ACCEPTED-AS-CONCERN: B-05 crowdedness reframed as the confirmatory-falsification point). The 4 PARTIALLY-FIXED Round A residues are discharged as follows: A-02 → Rev-12 (explicit design/aliasing plan required in formal plan); J-03 → unchanged (PIT-US deferral stands with disclosure); K-02 → conditional-PFER limitation stands as disclosed, empirical planted-null calibration is the operative guarantee; M-02 → Rev-20 fixes the ex ante estimand.

PROCEED-WITH-FIXES disposition: sketches may advance to the formal plan doc for whichever idea H博士 selects; the formal plan must carry §Rev-1..20 as binding and returns for Round C only if it deviates from them. Cross-model note: 5.6-sol's Round B found no CRITICALs in the revised designs and confirmed Round A's (5.5) sequencing — the two models' verdicts are consistent, with 5.6-sol adding finer statistical-estimand and identification findings.
