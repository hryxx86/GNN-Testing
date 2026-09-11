<!-- Rule 9 Touchpoint 1, Round B (cross-round diffing on the re-scoped plan). Reviewer: codex CLI 0.153.4 @ gpt-6-astra xhigh, read-only sandbox, invoked 2026-09-11 00:00 local from the main shell (codex exec ... < /dev/null). Prompt = Round A artifact with dispositions + the revised brief (scratchpad tp1b/prompt.txt). Post-review closures by Claude are appended below. -->
---
reviewer: codex
touchpoint: plan
round: B
target_plan: docs/c5_rerun_brief_2026-09-10.md
round_a_findings_status:
  - id: CODEX-A-01
    status: STILL-OPEN
    note: "The scientific execution objection is resolved by the explicitly test-informed scope in §9.9. Closure is incomplete because analyze_c5_sensitivity.py:146,230 still emits inflation and leak-free C5 wording. This residual reporting issue does not reinstate the original leakage-based execution block."
  - id: CODEX-A-02
    status: FIXED
    note: "Section 9.9 correctly defines the intersection and acknowledges identical proxy top-15 sets before and after shifting. I independently verified both sets and the five-group intersection. Paper corrections remain explicitly pending H博士; they are not prerequisites for calculating this sensitivity."
  - id: CODEX-A-03
    status: STILL-OPEN
    note: "The paired estimand, reversed sign convention, both HAC lags and paired bootstrap are implemented correctly under complete coverage. However, positional loading, within-arm truncation and a pooled-length assertion do not enforce common dates or ten contributing seeds; failed paired assertions are caught and reporting continues."
  - id: CODEX-A-04
    status: FIXED
    note: "The comparison writer includes paired-difference intervals, both HAC specifications and the approximate nominal MDE label; paired contrasts also report both lags. This closes the requested reporting specification, subject to A-03 alignment."
  - id: CODEX-A-05
    status: FIXED
    note: "For the revised predictive-sensitivity scope, the requested mitigations are specified: tuned architectures and actual-width MLP parameter counts, correct explanation of unscored hc groups, and the 23-column C5h definition in §9.9. These changes remain jointly varied and do not identify a leakage effect."
  - id: CODEX-A-06
    status: STILL-OPEN
    note: "The new manifest materially improves provenance, but records only HEAD despite the current modified/untracked implementation, omits price/label and ticker-universe input hashes, records evaluation rather than tuning execution versions, and overwrites provenance on subsequent invocations. Complete study backup is specified but not yet produced."
findings:
  - id: CODEX-B-01
    severity: CONCERN
    category: statistics
    claim: "C-pre needs a specified warm-up and missing-IC policy: its proposed window provides substantially different scoring coverage for at least one candidate."
    evidence: "The purged selection window contains 231 feature dates. In data/reference/sp500_5y_phase5_features.npy, mom12m is identically zero across stocks on 146 of them and nonconstant on only 85 dates, 2022-01-28..2022-05-31. This follows build_phase5_features.py:78's 252-session lookback. Section 9.10 does not specify coverage requirements or how undefined IC affects feature and group scores."
    suggested_fix: "Before approving C-pre, freeze a coverage policy: common eligible dates, a declared minimum-coverage rule with feature-specific dates, or additional pre-cutoff history. Do not treat undefined IC as observed zero. Archive each feature's scoring dates/count, exclusion reason and the group aggregation rule."
    status: OPEN
    resolution_notes: null
summary:
  critical: 0
  major: 0
  concern: 1
  fixed_before_reply: 0
overall_verdict: PROCEED-WITH-FIXES
---

**C5 is scientifically defensible as the explicitly labelled, test-informed subset sensitivity in §9.9.** It does not need a clean selector or completed paper edits before computation. The remaining fixes concern faithful reporting, paired-data integrity and preservation of the actual run inputs. They do not require redesigning the experiment.

The summary counts above cover new Round-B findings; the unresolved Round-A items are tracked separately.

**Round-A closure and execution conditions.** I accept the revised purpose, the exact 20-column subset, equal-budget re-tuning, Mac execution, exclusion from the confirmatory families, and reporting nominal unadjusted inference. “FIXED” here means the requested remedy is incorporated for that purpose; it does not mean selection bias or the joint pipeline changes have disappeared.

Three closures remain incomplete:

1. **A-01: propagate §9.9 into generated results.** The report header correctly says “TEST-INFORMED,” but its generated footer still calls the comparator “the leak-free C5 one.” The paired CSV note also labels a positive difference “inflation.” These are executable output strings, not merely preserved historical wording: see [paired CSV note](/Users/heruixi/Desktop/GNN-Testing/analyze_c5_sensitivity.py:146) and [report footer](/Users/heruixi/Desktop/GNN-Testing/analyze_c5_sensitivity.py:230). Replace them with “positive = C’s L1−L0 exceeds C5’s,” and state that this is a change under feature restriction and re-tuning. The p values and intervals condition on the selected subset and do not account for its test-informed selection. This is a substantive interpretation fix, but it need not delay model fitting.

2. **A-03: enforce the pairing assumption.** The implemented contrast
   \[
   g_t=\frac1{10}\sum_s(IC^C_{L1,t,s}-IC^C_{L0,t,s})
       -\frac1{10}\sum_s(IC^{C5}_{L1,t,s}-IC^{C5}_{L0,t,s})
   \]
   and bootstrap of that paired series are appropriate. The changed sign convention is harmless and documented. However, [the IC helper](/Users/heruixi/Desktop/GNN-Testing/run_storya_e1_anchor.py:827) drops undefined days, [the loader](/Users/heruixi/Desktop/GNN-Testing/compute_family1_ladder.py:139) places retained values into consecutive positions, and [the analyzer](/Users/heruixi/Desktop/GNN-Testing/analyze_c5_sensitivity.py:120) truncates within-universe series before checking pooled lengths. One seed can lose an interior date while another preserves the full length; the aggregate can still have 749 positions with incorrect pairing and fewer than ten contributors. Moreover, [assertion failures are caught](/Users/heruixi/Desktop/GNN-Testing/analyze_c5_sensitivity.py:289), allowing a nominally successful report without the required contrast.

   For this run, the smallest sufficient fix is to enforce complete expected **per-fold calendar coverage for every arm and seed** before aggregation. Alternatively, retain date identifiers and join explicitly. Production analysis should stop on an alignment failure. Validate this path before the full evaluation so omitted dates do not require reconstructing predictions afterward.

3. **A-06: freeze the actual execution state.** The [new provenance writer](/Users/heruixi/Desktop/GNN-Testing/run_storya_v21_main12.py:696) is useful, but HEAD alone does not identify the currently modified implementation; the analyzer and plan are also untracked. Archive a source snapshot or committed revision containing the reviewed changes, plus hashes of prices and the inputs determining ticker membership. Record tuning device/software metadata alongside evaluation metadata, and preserve invocation history on resume instead of overwriting it. Keep the proposed complete SQLite-study backups and finalist records. These records should start before tuning; they cannot reliably be reconstructed afterward.

The other dispositions are supported. I verified the identical proxy top-15 sets and the five-group intersection, the implemented interval/lag/MDE reporting, and the parameter-count construction. The operative C5h specification is in **§9.9**, despite the Round-A disposition’s reference to §9.11. Its joint changes remain descriptive; a later C5h run would not isolate selection bias.

Historical comparison inputs also check out: all **480 B/C L0/L1 arrays** match their recorded lengths, every arm/seed totals **749 days**, and all **240 merged C arrays** are byte-identical to their Mac-source counterparts. The historical frozen-hyperparameter MD5 matches `59ddd0a29c810c8f2cc6142a3559c681`. These checks support the historical side of the pairing; they do not establish future C5 coverage.

**C-pre’s proposed temporal boundary is clean, conditional on applying it to every selection input.** Reconstructing the actual calendar and 21-session purge gives:

| Stage | Retained feature dates | Latest label endpoint |
|---|---|---|
| Selection / tuning training | 2021-07-01–2022-05-31 | 2022-06-30 |
| Tuning validation | 2022-07-01–2022-11-30 | 2022-12-30 |

These follow the local price calendar and [purge implementation](/Users/heruixi/Desktop/GNN-Testing/run_storya_e1_anchor.py:574). Using the training segment for feature selection is legitimate; a separate selection holdout is not required merely because those observations subsequently train the models.

Checking **group-definition provenance is exactly right**, and the available artifacts resolve much of that question positively. The [61-group artifact](/Users/heruixi/Desktop/GNN-Testing/artifacts/plan_aaa/groups_168.json:1065) records calibration dates **2021-01-29–2022-01-27**. I verified that all 61 member lists match `ranking.csv`. Clustering uses a [label-validity mask](/Users/heruixi/Desktop/GNN-Testing/run_plan_aaa_168_ranking.py:290); its latest associated label endpoint is **2022-02-28**, also before the proposed selection cutoff. Thus these memberships need not be discarded merely because the same CSV also contains test-informed importance rankings.

Before approving C-pre, H博士 should settle:

- **Coverage:** address CODEX-B-01. The 231-day nominal window supplies only 85 nonconstant days for `hc_mom12m`; candidate scores must have a declared treatment of that difference.
- **Exact selector:** freeze whether the group score is the mean across members of each member’s absolute *time-mean* IC, rather than the time-mean of absolute daily IC. Score all intended hc members, specify missing-member handling, and break ties independently of the original test-ranked CSV order.
- **Grouping and scope:** choose reuse of the verified pre-cutoff memberships or re-clustering shifted features within the cutoff. Either can respect the temporal boundary. Use the full candidate universe without intersecting the result with test-selected C, accept the resulting membership/width, and retain the separate TP1 review.

C-pre could therefore support “re-selection using pre-test information,” while remaining a retrospective analysis designed after the paper’s test results were known. It would not identify how much leakage inflated C’s advantage or provide untouched confirmatory evidence.

This review was read-only. I checked source syntax, local artifacts, array coverage and calendar boundaries; I did not train models, rerun bootstrap inference or modify files.

---

## Claude closure of the STILL-OPEN items (2026-09-11, before launch)

- **A-01 (wording residue)** → FIXED after this review: analyze_c5_sensitivity.py docstring/header/footer/paired-CSV note now say "test-informed feature-subset sensitivity", "conditional subset contrast (feature restriction + re-tuning)", "NOT an identified leakage-inflation effect"; a grep for leak-free / inflat on the script leaves only the negated phrases. Same fix independently requested by TP2 Round A CODEX-A-04.
- **A-03 (pairing enforcement)** → FIXED after this review (TP2 Round A CODEX-A-02 fix): run_integrity now requires EVERY C5 cell's per-day .npy length == the frozen calendar's per-fold day count (taken from the confirmatory L0 cells, sum 749), so any partial collapse fails integrity (cells_not_full_calendar_length), and the paired-contrast assertion is re-raised outside smoke mode (production stops on misalignment). Fixture test: full synthetic C5 dir (= confirmatory C L0/L1 renamed) → PASS and paired diff exactly 0; one interior observation removed from one seed/fold → FAIL.
- **A-06 (execution state)** → FIXED after this review: reviewed code is COMMITTED before the run so git_rev identifies it; _run_provenance.json additionally records blob sha + git status of every repo module imported by the process (source_clean flag), md5 of prices/sectors (ticker universe + labels), n_stocks/n_days, the invocation (argv/arms/seeds/folds/resume), and is APPENDED per invocation (resume history kept); the tune JSONs now carry an execution block (device/platform/python/torch/lightgbm/optuna/numpy/git_rev/study_db/timestamp); the two Optuna sqlite studies are copied to artifacts/storya_v21_tune/ with md5 after tuning.
- **B-01 (C-pre coverage policy)** → ACCEPTED-AS-CONCERN, folded into the §9.10 proposal's open decisions for H博士 (coverage rule for hc_mom12m 252-session warm-up: 85/231 nonconstant dates; exact group score; grouping reuse vs re-clustering; full candidate universe without intersecting C). Not executed.
