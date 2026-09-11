<!-- Rule 9 Touchpoint 1, Round A. Reviewer: codex CLI 0.153.4 @ gpt-6-astra xhigh, read-only sandbox, invoked 2026-09-10 23:44 local from the main shell as: codex exec --sandbox read-only -o <out.md> "<prompt>" < /dev/null (the codex:codex-rescue subagent could not initialize the in-process app-server inside its sandbox: "Operation not permitted"; the direct main-shell call worked). Prompt = brief docs/c5_rerun_brief_2026-09-10.md verbatim + focus list (scratchpad tp1/prompt.txt). Statuses/resolution_notes filled in by Claude after personally verifying each finding (see brief §9.9). -->
---
reviewer: codex
touchpoint: plan
round: A
target_plan: docs/c5_rerun_brief_2026-09-10.md
findings:
  - id: CODEX-A-01
    severity: CRITICAL
    category: data-leakage
    claim: "C5 has T-1 runtime features but is selected using test-period labels; it is not a leak-free re-selection."
    evidence: "analyze_plan_aaa_t1_diagnostic.py:86-99,128-129 selects and scores the last 313 valid label days. Reconstructing this rule on the referenced local panel gives 2024-09-27 through 2025-12-26, all inside the main test period."
    suggested_fix: "Before execution, either redefine this run as an explicitly test-informed subset ablation, retaining the selection-leakage limitation, or replace the selector with a pre-test selection procedure over the original candidate universe. A clean selector must use only labels whose endpoints precede its selection cutoff; its resulting groups need not be these five."
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "Verified by Claude: analyze_plan_aaa_t1_diagnostic.py:86-99 scores the last 313 valid label days = 2024-09-27..2025-12-26 (reconstructed on the local panel); Plan AAA itself scored the 5-fold test quarters 2024-04-01..2025-06-30 (data/reference/fold_manifest_expanding.json). Both inside the 12-fold test period. Disposition = Codex option (1): run C5 as an explicitly TEST-INFORMED feature-subset sensitivity (H博士 explicitly requested these numbers), withdraw the 'definitive check' framing in all outputs; paper L1 wording change escalated to H博士; option (2) pre-test selector written up as a proposal (brief §9.10) awaiting H博士 approval + its own TP1."
  - id: CODEX-A-02
    severity: MAJOR
    category: correctness
    claim: "The diagnostic does not establish that these five groups survived removal of leakage: changing the importance method produces the same five-group overlap even without the T-1 shift."
    evidence: "artifacts/plan_aaa_t1_diagnostic/summary.md:21-23 reports original versus proxy-raw = 5/15, original versus proxy-T1 = 5/15, and proxy-raw versus proxy-T1 = 15/15. progress.md:6515 explicitly calls the diagnostic inconclusive for permutation-ranking stability."
    suggested_fix: "Describe C5 as the intersection of the original top-15 and the proxy top-15. Remove claims that the other groups failed because leakage was removed, and replace the paper's definitive-check promise. Establishing lag-induced ranking changes requires holding the ranking method and evaluation window fixed."
    status: FIXED
    resolution_notes: "Verified by Claude on group_ranking_comparison.csv: set(proxy_rank_raw<=15) == set(proxy_rank_t1<=15); orig∩raw == orig∩t1 == the same 5 groups. All C5 outputs/docs now describe C5 as 'Plan-AAA top-15 ∩ single-feature-IC proxy top-15 (proxy top-15 identical with/without T-1 shift)'; the paper sentence 'only 5 of 15 survive strict T-1 re-ranking' (main.tex:290/:998/:1012) flagged to H博士 as a misstatement to correct."
  - id: CODEX-A-03
    severity: MAJOR
    category: statistics
    claim: "Separate C5, C, and B estimates and MDEs do not directly estimate the change in the L1-L0 advantage."
    evidence: "docs/c5_rerun_brief_2026-09-10.md:53-57 requests within-universe results and a comparison row, but no paired cross-universe contrast. Shared test days make covariance between the contrasts material."
    suggested_fix: "Add the daily paired contrast g_t = delta_IC_C5,t minus delta_IC_C,t, averaging the ten matched seeds first. Report its mean, nominal two-sided HLN p, and paired stationary-bootstrap 95% CI using common dates and resampling indices. Describe it as a change under feature restriction and re-tuning, not an identified leakage-inflation effect."
    status: FIXED
    resolution_notes: "analyze_c5_sensitivity.py::paired_contrast — seeds averaged first, g_t = ΔIC_C,t − ΔIC_C5,t on the common 749 days (strict equal-length assert), HLN p (auto lag + lag 21) and 21d stationary block-bootstrap 95% CI (5000) on the paired series; positive = C contrast larger than C5; interpreted as change under feature restriction + re-tuning, not identified leakage inflation; outside the frozen confirmatory pair list."
  - id: CODEX-A-04
    severity: MAJOR
    category: statistics
    claim: "The requested headline p values omit an existing, consequential dependence sensitivity, while the deliverables omit explicit L1-L0 confidence intervals."
    evidence: "artifacts/storya_v21_family1/family1_dm_hln.csv:12 gives C L1-L0 p=0.010852 with automatic HAC lag 6 but p=0.063187 with lag 21. family1_mde.csv:12 gives its 21-day bootstrap CI [-0.00036, 0.03041]. The plan requests per-arm CIs but only delta, p, and MDE for L1-L0."
    suggested_fix: "Report the L1-L0 paired-difference CI and both existing HAC specifications for C5, C, and B, and for the new cross-universe contrast. Preserve the frozen headline specification while explaining any disagreement with lag-21 inference. Label 2.8 times bootstrap SE as an approximate nominal MDE, not an exact power calculation for the automatic-lag HLN test."
    status: FIXED
    resolution_notes: "c5_comparison.md reports for C5/C/B: L1-L0 point estimate, 21d bootstrap CI, HLN p at the frozen automatic lag AND at lag 21, per-arm CIs, and MDE labelled '≈2.8×SE, approximate nominal'; the paired contrast reports both HAC lags too."
  - id: CODEX-A-05
    severity: CONCERN
    category: correctness
    claim: "C5 versus C changes feature information, MLP capacity, hyperparameter selection, and the inclusion of three runtime leak-unaffected hc columns simultaneously."
    evidence: "docs/c5_rerun_brief_2026-09-10.md:15-16,23 specifies 20 versus 51 columns, hc exclusion, and re-tuning. run_storya_e1_anchor.py:535 makes the input-layer parameter count depend on the feature dimension. The diagnostic assigns unscored hc groups bottom ranks at analyze_plan_aaa_t1_diagnostic.py:159-173,192-193."
    suggested_fix: "Keep equal-budget re-tuning for the predictive sensitivity, but limit attribution accordingly. Report selected architectures and parameter counts. Explain that hc exclusion follows the literal five-group definition and is not evidence that hc groups failed a leakage audit. Specify any C5h or fixed-hyperparameter follow-up before inspecting results if it is intended to support attribution."
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "Tuned winners + MLP parameter count at the actual input width (20 vs 51, via make_nn_model) reported in c5_tuned_hparams.csv / c5_comparison.md; hc exclusion documented as the literal five-group definition (hc groups are UNSCORED in the proxy, not failed). C5h (C5 + 3 hc columns) pre-specified in brief §9.9 (A-05 bullet) BEFORE results; runs only on H博士 request."
  - id: CODEX-A-06
    severity: CONCERN
    category: reproducibility
    claim: "The frozen-hyperparameter MD5 gate alone does not establish the selection, data, or execution provenance needed for the historical C comparison."
    evidence: "run_storya_v21_main12.py:651-669 checks hyperparameter-file hash and mode. analyze_plan_aaa_t1_diagnostic.py:86-99 selects a moving panel tail, and its summary acknowledges possible calendar drift. The historical macC results CSV has no device field."
    suggested_fix: "Retain the MD5 gate and additionally archive the ordered feature list, selector/input hashes and exact selection dates, code revision or source snapshot, evaluation calendar, complete tuning-study records, and actual device/software versions. Prefer the proposed Mac execution and document backend parity for both tuning and evaluation."
    status: FIXED
    resolution_notes: "run_storya_v21_main12.py C5 branch writes _run_provenance.json: git rev, platform, python/torch/lightgbm/numpy/pandas versions, device, ordered feature list, selector-input md5s (group_ranking_comparison.csv, ranking.csv, alpha158 meta+npy), reconstructed selection windows, tuning window, 12-fold calendar, seeds, frozen md5. Mac execution kept for backend parity with the confirmatory C/L0,C/L1 cells."
summary:
  critical: 1
  major: 3
  concern: 2
  fixed_before_reply: 4
overall_verdict: BLOCK-EXECUTION  # Round A verdict as issued; dispositions above; Round B requested on the re-scoped plan (brief §9.9/§9.10)
---

**The plan should not launch under its current scientific specification.** The blocking issue is the use of test labels to define C5. The proposed computation remains useful as an explicitly test-informed subset ablation, but cannot discharge the paper’s promise of a definitive leak-free check. This review was read-only; no fixes or experiments were performed.

**Leakage and selection provenance.** The runtime construction passes the narrow temporal check: `build_universe_C5` selects named columns from C after its T−1 shift, and downstream winsorization and standardization fit on training days only. These operations introduce no additional temporal leakage. See [feature construction](/Users/heruixi/Desktop/GNN-Testing/run_storya_e1_anchor.py:408) and [preprocessing](/Users/heruixi/Desktop/GNN-Testing/run_storya_e1_anchor.py:592).

The selector fails the separate information-boundary check. I reconstructed its valid-label mask and trailing-window rule without running the diagnostic. On the referenced local panel:

- The 313 selected dates are **2024-09-27–2025-12-26**.
- All 313 are among the **749 valid main-test dates**.
- The last selected label uses prices through **2026-01-28**.

The script’s comment suggesting Q2-2024–Q2-2025 does not match this reconstruction. Because the diagnostic did not preserve its exact dated selection manifest, these are dates reconstructed from the current inputs; the script itself nevertheless explicitly targets test-period outcomes. Freezing this subset now, shifting its features, or re-tuning before 2023 cannot remove the information already embedded in its membership.

The tuning boundary itself **passes**. Applying the actual purge gives training feature dates through **2022-05-31**, with labels ending **2022-06-30**, and validation feature dates through **2022-11-30**, with labels ending **2022-12-30**. Thus the tuning labels stay before the 2023 test period.

There are two defensible revisions:

1. Keep the exact 20 columns and run a **post-hoc, test-informed feature-subset sensitivity**. Retain the selection-leakage limitation and withdraw “definitive check.”
2. Construct a separate subset using pre-test information. Prefer selection before the fixed tuning-validation window, with purged label endpoints. Start from the original candidate universe and pre-test group definitions; intersecting a new ranking with the already test-selected C does not erase inherited selection. Accept that the result may contain different groups and a different number of columns.

Even the second revision remains post-hoc in the paper’s research history. Independent future evaluation would provide stronger validation. Selection must be included in the evaluation boundary; ordinary evaluation after outcome-informed selection can be biased. [Cawley and Talbot](https://www.jmlr.org/papers/v11/cawley10a.html)

**The five-group premise needs correction.** I verified the five groups and their 20 member columns. However, the actual criterion is:

\[
\{\text{original rank}\le15\}\cap\{\text{proxy T−1 rank}\le15\}.
\]

`proxy_rank_t1 <= 15` alone returns **15 groups**, not five. More importantly, the proxy-raw and proxy-T1 top-15 sets are identical. The same five original groups overlap both sets. Consequently, the diagnostic demonstrates disagreement between importance methods; it does not demonstrate that removing same-day information eliminated ten groups. This does not negate the original same-day feature error, but it prevents attributing this particular subset change to that error.

**Statistics should answer the paired question directly.** Define

\[
\bar d^u_t=\frac1{10}\sum_s
\left(IC^u_{L1,t,s}-IC^u_{L0,t,s}\right),\qquad
g_t=\bar d^{C5}_t-\bar d^C_t.
\]

Report the mean of \(g_t\), its nominal HLN p, and its 95% stationary-bootstrap interval. Resample the already paired \(g_t\) series, or jointly resample all constituent series using identical date indices: expected block length 21, 5,000 replicates. A negative mean indicates a smaller MLP advantage under C5. Its negative can be reported as the observed reduction.

This directly estimates the **change between fitted pipelines**. It does not isolate the amount caused by leakage: selection method, feature content, capacity, and re-tuning also change. The additional contrast belongs in the sensitivity analyzer, outside the frozen confirmatory pair list.

Side-by-side estimates and MDEs are useful context, but neither different significance labels nor overlapping marginal intervals test the cross-universe difference. [Gelman and Stern](https://doi.org/10.1198/000313006X152649)

The existing artifacts already show why the full uncertainty information matters:

| Universe | L1−L0 | HLN p, lag 6 | HLN p, lag 21 | 21-day bootstrap 95% CI | MDE |
|---|---:|---:|---:|---:|---:|
| B | +0.01428 | 0.05236 | 0.18121 | [−0.00507, +0.03414] | 0.02749 |
| C | +0.01477 | 0.01085 | 0.06319 | [−0.00036, +0.03041] | 0.02201 |

Sources: [HLN results](/Users/heruixi/Desktop/GNN-Testing/artifacts/storya_v21_family1/family1_dm_hln.csv:2) and [paired intervals/MDE](/Users/heruixi/Desktop/GNN-Testing/artifacts/storya_v21_family1/family1_mde.csv:2).

Preserve the frozen automatic-lag result, but show the existing lag-21 sensitivity alongside it. The HLN horizon correction does not itself set the HAC bandwidth to 21. The bootstrap intervals and approximate MDE describe a different uncertainty calculation from the lag-6 headline test.

**Raw p without a new BH family is acceptable for descriptive post-hoc reporting.** State that these are unadjusted, nominal p values, and avoid discovery or confirmatory language. With the current test-selected C5, neither HLN nor an ordinary bootstrap accounts for the selection step; nominal p values and intervals cannot certify unbiased out-of-sample evidence. Adding BH would not repair this problem. Report all attempted sensitivities, including any later C5h or L2 run.

**The proposed seed estimand is appropriate under common coverage.** When every arm and seed contributes the same dates,

\[
D_s=\frac{\sum_f n_f\,IC_{L1,s,f}}{\sum_f n_f}
-\frac{\sum_f n_f\,IC_{L0,s,f}}{\sum_f n_f}
\]

equals the pooled daily difference for that seed, and averaging \(D_s\) reproduces the headline daily estimand.

I checked matching within-fold array lengths across all L0/L1 seeds for existing B and C, totaling 749 days. Both reproduce **10/10 sign agreement and 0/10 LOSO flips**; CSV rounding changes their pooled means by less than \(2\times10^{-8}\).

For C5, preserve actual date alignment. The existing daily-IC helper omits undefined days, so equal array lengths alone would not establish alignment if different dates were omitted. Define whether \(k\) counts positive effects or agreement with the pooled sign: ten negative effects also produce 10/10 agreement. LOSO should mean a flip of the pooled effect’s sign, not a change in significance. These are optimization-stability diagnostics over shared market data, not ten independent market replications.

**Equal-budget re-tuning and the implementation deviations are mostly sound.** Re-tuning both arms with the frozen 30-trial search and five finalists × three seeds is the appropriate treatment for a comparison of separately tuned predictive pipelines. Equal trial counts do not imply equal optimization difficulty, but that is not a reason to reuse C’s winners selectively.

Reducing 51 inputs to 20 removes \(31h\) input-layer weights at fixed hidden width \(h\), and re-tuning may also change width and depth. Report those changes. Parameter matching is not required for this predictive sensitivity; a fixed-hyperparameter supplement could address a narrower question but would still not identify leakage inflation.

Excluding the three `hc_` columns matches the literal five-group definition. Their bottom proxy ranks mean **unscored**, not unimportant or invalid after lagging. Their removal is therefore an additional information change. If that distinction matters to the paper’s attribution, specify C5h before inspecting C5 rather than deciding whether it is useful from the result.

The separate sensitivity universe, explicit sensitivity statistics mode, and two-study merge preserve the confirmatory defaults appropriately. The proposed ID allocation is disjoint: C5 reserves **2400–3599**, with this L0/L1 run using **2400–2639**.

The Mac choice is preferable for historical comparability. The cited directory contains 120 C/L0 and 120 C/L1 cells, and L1 averages **60.95 seconds/cell**; the tuning launcher also assigns L0/L1 to Mac. I verified that the historical frozen file matches its recorded MD5, `59ddd0a29c810c8f2cc6142a3559c681`. Record actual backend and software versions for C5; filenames and timing alone do not establish execution parity.

**Paper wording should follow the evidence and interval, including unfavorable outcomes.** For the present test-informed subset, use wording such as:

| Outcome | Defensible interpretation |
|---|---|
| C5 effect approximately equals C | “The MLP advantage persisted with a similar point estimate in the test-informed 20-feature subset. The paired change was … [CI …]. This sensitivity does not resolve feature-selection bias.” |
| Smaller, same sign | “The estimated advantage decreased from … to …; the paired reduction was … [CI …]. The reduction reflects feature restriction and re-tuning and cannot be attributed solely to leakage.” |
| Negative point estimate | “The estimated model ordering reversed on the subset; uncertainty was … .” Claim evidence of reversal only if the interval supports it. |
| Interval includes zero | “The subset analysis did not establish a positive MLP advantage; its interval remains compatible with … .” Do not equate non-rejection with no effect. |

A near-zero paired change with a wide interval does not establish equivalence. A significance change does not establish attenuation. Under every outcome, B remains the leak-free feature-basis anchor with its stated uncertainty, and an L0/L1-only experiment provides no new direct evidence about when GNNs help.