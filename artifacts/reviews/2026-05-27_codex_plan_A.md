---
reviewer: codex
touchpoint: plan
round: A
target_plan: /Users/heruixi/.claude/plans/hats-baseline-reproduction-delightful-lighthouse.md
target_files: []
findings:
  - id: CODEX-A-01
    severity: CRITICAL
    category: data-leakage
    claim: "Sector edges are not PIT-audited"
    evidence: "data/reference/sp500_sectors.csv:1 has only `Symbol,GICS Sector`; run_storya_e1_anchor.py:277-281 loads that static file with no as-of date; run_step3_plan_z_part_a.py:159-166 fully connects those static groups."
    suggested_fix: "Before launch, document and assert sector provenance: either use a sector map known as of each fold's prediction period, or freeze a documented pre-test as-of snapshot and record it in HATS metadata. Abort if the sector file lacks source_date/as_of_date provenance."
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "Verified: sp500_sectors.csv has only Symbol,GICS Sector columns; file mtime 2026-02-09. H博士 2026-05-27 directive: project-level §Limitations treatment, same as Plan AAA Alpha158 same-day OHLC leak. Reasoning: (1) issue affects E1/E3/E4-α (all already completed); fixing only HATS is inconsistent. (2) SP500 sector reclassifications are infrequent (~5-15/year, 1-3% of universe) — leakage magnitude bounded. Action: prereg.json and Story A §Limitations document sector file fetch_date=2026-02-09 explicitly; HATS still uses 3 relations (corr+sector+news)."
  - id: CODEX-A-02
    severity: MAJOR
    category: statistics
    claim: "Joint SPA family size is wrong"
    evidence: "Derivation: HATS is Universe B only, so joint candidates become B.{GAT,SAGE-Mean,MLP,HATS}=4 plus C.{GAT,SAGE-Mean,MLP}=3, total M=7, not 6. compute_e6_dm_spa.py:394-427 builds joint SPA from all non-baseline candidates and compute_e6_dm_spa.py:582-584 hardcodes joint_M=6."
    suggested_fix: "Set prereg.json and the E6 ledger to joint_M=7 when HATS is B-only, or explicitly exclude HATS from joint SPA and justify that joint SPA remains E1-only."
    status: FIXED
    resolution_notes: "H博士 2026-05-27: HATS EXCLUDED from joint SPA. Rationale: HATS is Story A supplement (Template 1 framing), not main panel. Per-universe B SPA expands to M=4 (current 3 + HATS); joint SPA stays M=6 (E1-only). compute_e6_dm_spa.py:582-584 ledger keeps spa_application_joint_M=6 unchanged; --include-hats-csv flag will inject HATS only into per-universe B SPA and per-universe B DM pairs."
  - id: CODEX-A-03
    severity: MAJOR
    category: statistics
    claim: "HATS pairing can silently truncate dates"
    evidence: "compute_e6_dm_spa.py:367-372 truncates SPA candidate/benchmark arrays to min length; compute_e6_dm_spa.py:454-457 truncates DM pairs to min length. compute_e6_edge_ablation.py:156-164 correctly hard-errors on fold length mismatch."
    suggested_fix: "For HATS, align per-day IC by explicit (fold, date) keys and hard-error on any length/calendar mismatch before SPA, DM/HLN, and bootstrap. Do not use min-length truncation."
    status: FIXED
    resolution_notes: "Verified L367-372 and L454-457 do silent min-length truncate. Plan updated: --include-hats-csv flag must port compute_e6_edge_ablation.py:156-164 hard-error pattern (RuntimeError on any (fold, date) calendar mismatch) into compute_e6_dm_spa.py SPA + DM + bootstrap paths. Listed as a required additive change in plan §E6 integration."
  - id: CODEX-A-04
    severity: MAJOR
    category: reproducibility
    claim: "HATS cell_id namespace collides"
    evidence: "run_storya_e1_anchor.py:197-215 defines E1 cell_id range [0,399]. The HATS plan defines cell_id_hats=f*10+s_idx, range [0,49], which overlaps E1 cells 0-49 when concatenated."
    suggested_fix: "Use `cell_id_hats = 400 + fold_idx*10 + seed_idx` and add a startup injectivity assertion over E1 plus HATS ids, or add an experiment_id column and stop treating cell_id as globally unique."
    status: FIXED
    resolution_notes: "Verified collision: E1 universe_idx=0 (B) × model_idx=0 (GAT) × fold=0 × seed=0 yields cell_id=0; HATS f*10+s_idx with f=0,s=0 also yields 0. Plan updated: cell_id_hats = 400 + fold_idx*10 + seed_idx, range [400, 449]. Plus startup injectivity assertion when concatenating with E1 results.csv inside compute_e6_dm_spa.py (--include-hats-csv path)."
  - id: CODEX-A-05
    severity: MAJOR
    category: prior-art
    claim: "The module is not Kim HATS"
    evidence: "Kim et al. 2019 arXiv:1908.07999 Section 3.2, PDF lines 316-343 and 370-405, defines state attention and relation attention using relation embeddings and concatenated node/relation summaries. The plan uses per-relation PyG GATConv plus `Linear(64,1)(h_stack)` relation scoring."
    suggested_fix: "Either implement Kim's two-level attention equations with relation embeddings, or rename the model to `HATS-style relational GAT` and make the adaptation explicit in prereg.json and paper text."
    status: FIXED
    resolution_notes: "H博士 2026-05-27: rename to 'HATS-style three-relation adaptation'. results.csv `model` column value = 'HATS-3R-adapt'. prereg.json and paper text MUST state: 'we adapt the HATS hierarchical-attention mechanism (Kim et al. 2019) to 3 substitute relations (correlation/sector/news); GRU encoder omitted; relation-attention scoring uses Linear(64,1) instead of Kim §3.2 relation-embedding concatenation.' This is an inspired-by adaptation, not a literal HATS reproduction. Template-1 claim scope restricted accordingly."
  - id: CODEX-A-06
    severity: MAJOR
    category: prior-art
    claim: "Skipping GRU blocks reproduction claims"
    evidence: "Kim et al. 2019 arXiv:1908.07999 PDF lines 259-268 and 615-618 state that LSTM/GRU feature extraction modules are used before relational modeling; the plan's Option A feeds Universe B 10-dim handcrafted features directly."
    suggested_fix: "Pre-register the run as an adaptation of the HATS relational module, not a literal HATS reproduction. Restrict any Template-1 claim to this adapted strict-evaluation setting."
    status: FIXED
    resolution_notes: "Same disposition as A-05: HATS renamed to 'HATS-3R-adapt'. Template-1 claim narrowed to 'a HATS-style three-relation attention module on strict 21-day cross-sectional ranking features [does/does not] beat locked baselines' — explicitly NOT 'Kim's published HATS architecture fails under reproduction'. prereg.json claim_scope field added."
  - id: CODEX-A-07
    severity: MAJOR
    category: prior-art
    claim: "Relation set substitution changes the estimand"
    evidence: "Kim et al. 2019 arXiv:1908.07999 PDF lines 574-594 describes Wikidata corporate relations and 75 relation/meta-path types. The plan uses only correlation, sector, and PIT news co-occurrence."
    suggested_fix: "Rename the baseline as a HATS-inspired three-relation adaptation, or add a separate relation-substitution limitation that prevents claiming failure of Kim's Wikidata HATS."
    status: FIXED
    resolution_notes: "Same disposition as A-05/A-06: rename + explicit scope. prereg.json and paper text declare: 'Wikidata corporate relations (75 types) substituted by 3 readily-available relations in this project; resulting model does not test Kim's Wikidata HATS performance.'"
  - id: CODEX-A-08
    severity: MAJOR
    category: statistics
    claim: "Decision rules have no thresholds"
    evidence: "compute_e6_dm_spa.py:392 and 437 operationalize SPA rejection as p_consistent < 0.05, but the HATS plan only says outperforms, underperforms, or attention collapses with no delta IC, Sharpe, SPA, DM/HLN, LOFO-4, or tie threshold."
    suggested_fix: "Before execution, define primary and secondary gates: model pair(s), minimum delta IC or equivalence margin, BH-adjusted DM/HLN p-value, SPA p-value role, Sharpe role, and required LOFO-4 sign/size behavior."
    status: FIXED
    resolution_notes: "H博士 2026-05-27: Codex 3-gate thresholds locked in prereg.json. POSITIVE: ΔIC_HATS_vs_GAT > +0.005 AND BH-HLN p < 0.05 in full condition AND LOFO-4 sign preserved (ΔIC > 0). NEGATIVE: ΔIC < -0.005 OR (BH-HLN p > 0.20 in full AND LOFO-4 ΔIC ≤ 0). TIE: |ΔIC| ≤ 0.005 AND BH-HLN p > 0.05 (neither rejected nor strongly negative). Primary comparator = GAT (Universe B). Primary metric = IC. Secondary metric = Sharpe_net_10bps for narrative; not a gating condition."
  - id: CODEX-A-09
    severity: MAJOR
    category: statistics
    claim: "LOFO-4 reporting is missing"
    evidence: "analyze_e1_lofo.py:167-169 adds full/LOFO-4/Fold-4-only paper columns; compute_e6_edge_ablation.py:7-18 and 90-94 define full, lofo4, and fold4_only regimes for edge ablations. The HATS plan has no LOFO-4 column or decision rule."
    suggested_fix: "Add HATS full 5-fold, LOFO-4, and Fold-4-only summaries for IC and Sharpe, and require LOFO-4 in the positive/negative/tie verdict."
    status: FIXED
    resolution_notes: "Verified analyze_e1_lofo.py:167-169 + compute_e6_edge_ablation.py:7-18 use the three-condition pattern. Plan updated: HATS post-process produces full 5-fold + LOFO-4 + Fold-4-only IC + Sharpe_net_10bps with block-bootstrap CIs per condition, matching E1/E3/E4 reporting. LOFO-4 sign preservation is now part of decision rule per A-08 disposition (POSITIVE branch requires it)."
  - id: CODEX-A-10
    severity: CONCERN
    category: reproducibility
    claim: "Compute estimate is unsupported"
    evidence: "run_storya_e1_anchor.py:1164-1169 estimates E1 at about 6 min/cell; run_storya_e1_anchor.py:468-495 uses two GAT layers for E1 GAT, while HATS would call three GATConv relations per layer, six GATConv calls total, plus E3-style per-day edge movement at run_storya_e3_news_edge.py:319-331."
    suggested_fix: "Make 13 min/cell provisional until a measured HATS smoke is recorded. Update wall-time budget from the 1-cell smoke, including GPU memory and per-day edge transfer overhead."
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "Plan updated: 13 min/cell labeled PROVISIONAL. Mandatory 1-cell A100 smoke benchmark BEFORE 50-cell launch, recording peak memory, average edge counts per relation per day, epoch wall time, CPU-GPU transfer overhead. Total budget locked from smoke measurement, not pre-estimate."
  - id: CODEX-A-11
    severity: CONCERN
    category: correctness
    claim: "Uniform-alpha control is missing"
    evidence: "Kim et al. 2019 arXiv:1908.07999 PDF lines 304-312 and 370-420 make selective relation aggregation central. The HATS plan includes a num_relations=1 alpha check but no uniform relation-weight baseline."
    suggested_fix: "Add a frozen uniform-alpha control or state that attention-specific claims are deferred. If attention collapse is a decision branch, the uniform-alpha control should be run on the same 50 cells or a predeclared diagnostic subset."
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "Plan updated: prereg.json explicitly states 'NO attention-specific claims are made without uniform-α control'. Attention collapse is a DIAGNOSTIC (logged α statistics per epoch), not an interpretive claim. If results merit a uniform-α extension, it becomes a separate experiment (~50 additional cells) added after seeing the primary 50-cell result; this extension rule is pre-committed in prereg.json."
summary:
  critical: 1
  major: 8
  concern: 2
  fixed_before_reply: 0
  fixed_after_reply: 8
  accepted_as_concern: 3
  rejected: 0
overall_verdict: BLOCK-EXECUTION
post_disposition_verdict: PROCEED-WITH-FIXES
---

# Review body

## 1. Data Leakage

News PIT is the strongest part of the plan if it reuses E3 exactly. The contract is explicit: the E3 runner defines eligible articles as publication_timestamp at or before NYSE session_close(t-1) in UTC, with DST handling, and asserts the max selected timestamp does not exceed the cutoff (run_storya_e3_news_edge.py:27-30, 161-181, 237-255). The schema also strips forward fields from the edge source, including return_next and label (experiments/storya_e3_news_edge/news_edge_source_schema.md:44-49; scripts/build_news_edge_source.py:53-56, 199-215). That is acceptable.

Winsorization and scaling are acceptable only if HATS calls the E1 helpers directly. E1 computes p1/p99 bounds on train_days only and clips all days (run_storya_e1_anchor.py:522-536), then fits StandardScaler on train_days only (run_storya_e1_anchor.py:539-546). The HATS plan states the same policy, so this is correct if implemented through those helpers or an exact equivalent.

Frozen correlation alignment is also correct if the existing assertion is reused. E1 builds correlation windows ending at snapshot_points using returns.iloc[t_end-window:t_end] (run_storya_e1_anchor.py:417-436) and selects the last snapshot at or before the purged train end (run_storya_e1_anchor.py:439-447). The older helper asserts snapshot end is no later than max(train_days) (run_step3_plan_z_part_a.py:171-188). The HATS smoke protocol mentions this assertion, which is necessary.

Sector edges are the blocker. The static sector file has only Symbol and GICS Sector, with no as-of/effective date (data/reference/sp500_sectors.csv:1). E1 loads this static file directly (run_storya_e1_anchor.py:277-281), and the sector builder fully connects every member inside each sector (run_step3_plan_z_part_a.py:159-166). If the file is current-sector metadata, it can encode post-test-period reclassifications or membership knowledge. Before HATS uses sector edges as one of three relations, the plan needs an as-of provenance assertion or an explicit freeze date.

## 2. Statistical Methodology

SPA family math must be fixed before execution. The plan says per-universe M goes 3 to 4, which is correct for Universe B. But HATS is Universe B only, so the joint family is not 6 and not 8. It is 7 if HATS participates in joint SPA: four B candidates and three C candidates. The existing E6 joint SPA code constructs candidates across expected universes and non-baseline models (compute_e6_dm_spa.py:394-427), and the ledger currently hardcodes joint_M=6 (compute_e6_dm_spa.py:582-584). The plan's M_joint_new=6 is wrong.

DM/HLN with T=313 is consistent with E6 precedent. The current E6 code uses seed-aggregated daily IC, not seed-day pseudo-replication, and documents T=313 with HLN small-sample correction (compute_e6_dm_spa.py:14-25, 239-250). With h=21 and T=313, the HLN multiplier is about 0.935, so p-values are materially wider than uncorrected DM. For LOFO-4, T drops to about 251; for Fold-4-only, T is about 62 and edge ablation correctly reports no HLN p-value for that condition (compute_e6_edge_ablation.py:211-218).

Block bootstrap block_size=21 is consistent with the 21-day forward label and E6 convention (compute_e6_dm_spa.py:21-22, 69-73, 277-292). Keep it for HATS IC CIs. For Sharpe/cell-level summaries, E6 uses block_size=1 over cell values (compute_e6_dm_spa.py:519-521, 543-547), and edge ablation follows the same pattern (compute_e6_edge_ablation.py:232-234).

The paired comparison implementation needs a hard alignment contract. HATS uses per-day news edges, so all comparisons must still share the same fold-date test calendar. compute_e6_dm_spa.py currently truncates unequal arrays to min length in SPA and DM (lines 367-372 and 454-457). That is not acceptable for HATS. Edge ablation fixed this by hard-erroring on length mismatch (compute_e6_edge_ablation.py:156-164); HATS should use the same rule or explicit date-key joins.

## 3. Apples-to-Apples Confound

Option A is fine as an engineering choice, but it is not a literal HATS reproduction. Kim et al. use a feature extraction module before relational modeling; the paper describes LSTM/GRU feature extraction and a 50-day lookback setup before HATS is applied (arXiv:1908.07999 PDF lines 259-268 and 615-618, https://arxiv.org/pdf/1908.07999). Feeding Universe B's 10 handcrafted features directly into relational attention changes the representation distribution and likely changes how relation softmax behaves.

That means the Template-1 conclusion must be narrowed. If this run underperforms GAT, the defensible statement is "a HATS-style three-relation attention module on strict 21-day cross-sectional ranking features did not beat locked baselines." It is not evidence that Kim's published HATS architecture fails under reproduction, because the temporal encoder and relation source are both changed.

## 4. Hierarchical Attention Spec

The proposed architecture diverges from Kim's HATS in three substantive ways.

First, Kim's state attention uses relation embeddings and node-pair context to aggregate neighbors inside each relation; relation attention then uses the summarized relation vector, current node representation, and relation embedding (arXiv PDF lines 316-343 and 370-405). The plan instead uses independent PyG GATConv modules per relation and scores each relation output with Linear(64,1). Shared relation scoring is not itself the problem; the problem is that the planned scorer drops the relation embedding and current-node context Kim uses.

Second, two stacked GAT layers imply two-hop propagation. That is fair for E1 hyperparameter comparability, but it is an adaptation. The original HATS section defines a relational modeling module around one hierarchy of state and relation attention, not "two PyG GAT layers per relation" as the core specification.

Third, the original individual-stock task is classification with a softmax and cross-entropy over up/neutral/down labels (arXiv PDF lines 428-452). Replacing that with regression on 21-day z-scored forward returns is acceptable for Story A's ranking target, but it further reinforces that this is a task-adapted baseline, not a faithful reproduction.

## 5. Relation Set Substitution Validity

The relation set is not comparable to Kim's. Kim et al. build corporate relational data from Wikidata and report 75 direct relation/meta-path types (arXiv PDF lines 574-594). The HATS plan uses three relations: frozen correlation, static GICS-11 sector, and PIT news co-occurrence. Those are useful Story A relations, but they are not Wikidata corporate relation types.

So the model should be named accordingly. "HATS baseline reproduction" overstates the claim. "HATS-style three-relation attention baseline" or "HATS-inspired relational-attention adaptation" is accurate. If the paper uses this as Template-1 replication-failure evidence, it must say the failed object is the adapted strict-evaluation baseline, not Kim's original data/model pair.

## 6. Compute Estimate

The 13 min/cell A100 estimate is plausible but not yet honest enough to lock. E1's own smoke summary estimates full E1 as 400 cells times about 6 minutes per cell (run_storya_e1_anchor.py:1164-1169). E1 GAT uses two GAT layers (run_storya_e1_anchor.py:468-495). HATS as planned uses three GATConv calls per layer and two layers, so six GATConv calls per forward pass before relation attention overhead. It also needs per-day edge switching for the news relation, similar to the E3 loop that pre-moves day-specific edges and switches by day (run_storya_e3_news_edge.py:319-331).

Do not present 11-13 hours as locked. Present it as provisional and require the one-cell smoke to update the estimate before launching 50 cells. The smoke gate should record peak memory, average edge counts by relation, epoch time, and whether CPU-GPU edge transfers are visible.

## 7. Decision Rule Rigor

The current decision rules are not operational. "Outperforms GAT," "underperforms GAT," and "attention collapses" are labels, not tests. Existing E6 has concrete thresholds for SPA rejection (p_consistent < 0.05 at compute_e6_dm_spa.py:392 and 437) and BH-FDR q=0.05 for DM/HLN families (compute_e6_dm_spa.py:477-482; compute_e6_edge_ablation.py:250-263). HATS needs the same level of specificity before results exist.

Minimum preregistration should define: primary comparator, primary metric, minimum meaningful delta, adjusted p-value gate, and LOFO-4 requirement. Example: positive validation only if HATS minus GAT has mean delta IC above a predeclared margin, BH-adjusted HLN p<0.05 in the full condition, no sign reversal under LOFO-4, and no material Sharpe degradation at 10 bps. A tie rule should also be explicit, for example an absolute delta IC margin and non-rejection. The exact numbers are a scientific choice, but they must be chosen before execution.

## 8. Cell ID Namespace Collision

The plan's statement that HATS has a separate namespace is false. E1 cell_id is universe_idx*200 + model_idx*50 + fold_idx*10 + seed_idx, range [0,399] (run_storya_e1_anchor.py:197-215). HATS cell_id_hats=f*10+s_idx, range [0,49], collides with E1 Universe B GAT fold/seed ids immediately.

This can break resume logic, joins, deduplication, and audit trails if any downstream script assumes cell_id is unique. The fix is simple: start HATS at 400, or add experiment_id and enforce uniqueness on (experiment_id, cell_id). Since the plan says results.csv schema is identical to E1, the safer fix is an offset: 400-449.

## 9. Missing Ablations

The degenerate single-relation sanity check is present in the smoke protocol: num_relations=1 should force alpha=1. That is enough as a shape/math sanity check.

The missing control is uniform-alpha relation aggregation. Kim's contribution is selective aggregation across relation types (arXiv PDF lines 304-312 and 370-420). If HATS beats or loses to GAT, without a uniform-alpha control it is hard to tell whether relation attention mattered or whether the effect came from adding three edge sets and extra parameters. This does not have to block the 50-cell baseline if the paper will not claim attention-specific benefit, but it is required for any attention-collapse interpretation stronger than a diagnostic plot.

## 10. LOFO-4 Integration

LOFO-4 is mandatory for Story A consistency. E1 has already added full, LOFO-4, and Fold-4-only paper columns (analyze_e1_lofo.py:167-169). Edge ablation defines full, lofo4, and fold4_only conditions and explicitly uses them in the post-process (compute_e6_edge_ablation.py:7-18, 90-94). The HATS plan omits this.

Given Fold 4 is the known Q2-2025 regime outlier, a HATS result without LOFO-4 cannot be interpreted next to E1, E3, or E4. HATS reporting needs at least IC and Sharpe_net_10bps for full 5-fold, LOFO-4, and Fold-4-only, plus decision-rule language saying how LOFO-4 affects positive, negative, and tie verdicts.

Bottom line: block execution until the sector provenance, SPA family math, paired date alignment, cell_id offset, HATS naming/prior-art scope, decision thresholds, and LOFO-4 reporting are fixed. The news PIT, train-only winsor/scaler, and frozen-correlation contracts are otherwise reusable and do not need redesign.
