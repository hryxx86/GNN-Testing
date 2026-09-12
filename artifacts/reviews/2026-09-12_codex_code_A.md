<!-- Rule 9 Touchpoint 2, Round A, for the C-pre implementation (plus Touchpoint 1 Round B on docs/c_pre_plan_2026-09-11.md).
Reviewer = Codex CLI (gpt-6-astra, xhigh), `codex exec --sandbox read-only`, 2026-09-12 02:09–02:17 local (≈7.5 min, no
usage-limit interruption). Prompt archived in the session scratchpad (codex/cpre_code_A_prompt.md). Reviewed tree = HEAD
e80c6ac (all reviewed files committed). Status/resolution_notes filled in by Claude after reading the cited lines and fixing. -->
---
reviewer: codex
touchpoint: code
round: A
target_files:
  - run_storya_cpre_select.py
  - run_storya_e1_anchor.py
  - run_storya_v21_main12.py
  - run_storya_v21_tune.py
  - analyze_c5_sensitivity.py
target_plan: docs/c_pre_plan_2026-09-11.md
plan_round_B:
  - id: CODEX-A-01
    status: FIXED
    note: "docs/c_pre_plan_2026-09-11.md:24-28 specifies exhaustive positive/negative/zero-containing CI branches, each series' own MDE, and disclosure of CI/HLN disagreement. Lines 16 and 28 bound the clean-basis claim."
  - id: CODEX-A-02
    status: FIXED
    note: "docs/c_pre_plan_2026-09-11.md:27-30 requires absolute paired changes, both HAC lags and paired-series MDE; forbids proportional inference and non-rejection-as-equivalence. Line 82 requires the analyzer replacement."
  - id: CODEX-A-03
    status: FIXED
    note: "docs/c_pre_plan_2026-09-11.md:71 explicitly disclaims architecture neutrality and distinguishes the tuning metric from independent validation. Lines 53-54 accurately limit the tau table to admitting the short-history feature."
  - id: CODEX-A-04
    status: FIXED
    note: "docs/c_pre_plan_2026-09-11.md:63 requires eligible-date indices, resolved-axis identifiers, determining-input hashes, committed-source identity and downstream selection hashes; line 97 requires commit-before-run."
findings:
  - id: CODEX-A-01
    severity: CONCERN
    category: statistics
    claim: "The optional device-replication report still emits an unconditional causal explanation and an unsupported claim that pooled inference is insensitive to device differences."
    evidence: "analyze_c5_sensitivity.py:580-583 always attributes divergence to backend nondeterminism amplified by early stopping and declares inference insensitive. device_replication() at lines 352-372 computes descriptive differences and pooled means, without testing either claim. An in-memory counterexample with primary delta=-0.030 and replicate delta=+0.030 still emitted the same assertion."
    suggested_fix: "Remove the fixed attribution and inference claim; report the computed replication statistics descriptively. Any claim of unchanged inference must be based on separately computed inferential results."
    status: FIXED
    resolution_notes: "Verified (the sentence was literal). write_md's device note now reports only computed statistics: pooled ΔIC per dir with same/DIFFERENT sign and the absolute gap, per-arm cell-IC corr / mean|diff| / max|diff| / n identical, and states that agreement of inference is read from the replicate's own family1/analysis outputs. The published C5 artifacts were regenerated with it (numbers unchanged)."
summary:
  critical: 0
  major: 0
  concern: 1
  fixed_before_reply: 0
overall_verdict: PASS-WITH-CONCERNS
---

# Codex review body (verbatim)

**All four Touchpoint-1 findings are resolved in the current plan. The implementation has no identified execution-blocking defect.** The remaining concern affects the optional device-replication narrative, not the declared Mac production run.

I reviewed the current files against `HEAD` (`e80c6ac`); all seven reviewed Python files match their committed versions. Verification was read-only: I executed extracted function bodies in memory, recomputed selection statistics, inspected arrays and hashes, and exercised reporting with synthetic inputs. I did not repeat model training or the complete CLI pipeline.

The selector's temporal boundary holds, conditional on the documented fixed stock panel:

- `selection_window()` uses the purged tuning-train mask. Independent reconstruction gives **231 dates, 2021-07-01 through 2022-05-31**, with the last 21-session label ending **2022-06-30** (run_storya_cpre_select.py:138; run_storya_e1_anchor.py:662).
- Alpha158 is rolled once and row zero is cleared. Its raw artifact is saved **before** the original build's full-panel winsorization, so that later operation does not contaminate selection (build_alpha158_features.py:378).
- hc rolling features use `.shift(1)`; phase5 momentum and other features use the documented historical shifts. Momentum's warm-up values are all zero across stocks and are rejected as constant. I reproduced **167 × 231 eligible dates and momentum's 85 dates**, including exact agreement with every archived eligible-date list.
- The 61 groups partition all 168 candidates. Calibration ends 2022-01-27; its label-validity dependence reaches 2022-02-28, still before cutoff. The whole-panel news membership determines the fixed ticker axis; it does not supply selection scores.

**Selection and feature construction reproduce the archive.** Recomputed feature-score numeric fields match exactly. The top-15 groups, all-member union, ordering, τ variants, and 48-column MD5 all match. τ=0.75 reproduces the frozen ranking; τ=0 admits momentum and displaces RESI60.

The selector rounds feature means and group scores before sorting. I separately ranked using unrounded values: **the entire primary ranking is unchanged**, not merely its top-15 membership. Thus this does not alter the frozen selection.

The CPRE builder produces a finite `(1255, 501, 48)` tensor **elementwise identical** to the selector's selected columns after the declared fill convention. All **22 shared C columns are also bitwise identical**. Name slicing, column order, row-zero handling and the list/MD5 gate work as intended (run_storya_e1_anchor.py:494).

The archived input hashes, ticker/date-axis hashes and news-membership hash match current inputs. All four recorded selector-source files match both their archived hashes and commit `044dd09`. The smoke tuning outputs and main12 provenance reference the current `selection.json` MD5, `0a853e6909a15319b888bb8a1218b2eb`.

Pipeline wiring checks passed:

- Enumerating all **4,800 IDs** confirms disjoint B/C, C5 and CPRE blocks. CPRE L0/L1 occupy `[3600,3839]` within the reserved `[3600,4799]`.
- Frozen injection resolves `CPRE_L0` and `CPRE_L1`; the smoke provenance passes the applied-key/parameter gate.
- `both` remains B,C. The diff leaves confirmatory feature construction, training logic, frozen hyperparameters and statistical-family definitions unchanged.
- Launcher subset merging excludes smoke studies. Family1's existing generic sensitivity path supports CPRE with nominal HLN tests, bootstrap CI/MDE, and no BH/SPA/L7.

The analyzer's principal calculations and gates are consistent with the plan. The CPRE smoke fails completeness at **2/240**, while passing its 48-name, block, calendar-length and frozen-parameter checks. Strict pairing checks both universes fold by fold. I independently inspected every L0/L1 comparator array: **C, B and C5 each have all 240 finite, full-calendar cells over the same 749 days**.

The comparison uses each series' own MDE and both HAC specifications; paired differences have the required comparator-minus-CPRE orientation and no halving/doubling inference. The tests inventory is computed: production with three paired comparators and `--ex-fold 9` reports **22 nominal p-values**. The device prose identified above remains the exception to the "all conclusions computed" requirement.

**The C5 numeric-regression claim is verified against `6acd834`.** Nine CSVs are byte-identical. In `c5_ex_fold.csv`, all existing field values are unchanged and only the two descriptive columns were added. In `c5_paired_contrast.csv`, all fields except `note` are unchanged.

**Yes—the production run may start:** 30-trial CPRE L0/L1 tuning → frozen subset merge → 240 Mac cells → family1 `--sensitivity` → analyzer, including `--ex-fold 9`. Retain the frozen 48-column selection. Correct the remaining device-replication prose before using that optional report.

---

## Claude disposition (2026-09-12, before the production run)

- Plan Round B: all four TP1-A findings confirmed FIXED by Codex → Touchpoint 1 closed.
- TP2-A: 0 CRITICAL / 0 MAJOR / 1 CONCERN (device-replication prose) → FIXED before the production run (descriptive statistics only). Codex's independent checks (window, eligibility counts, group partition, hashes, 4,800-id enumeration, builder bitwise identity, C5 regression) agree with Claude's own.
- Smoke directories deleted; production pipeline (Mac primary) launched after committing the fix.
