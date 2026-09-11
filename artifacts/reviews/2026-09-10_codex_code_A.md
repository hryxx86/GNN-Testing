<!-- Rule 9 Touchpoint 2, Round A. Reviewer: codex CLI 0.153.4 @ gpt-6-astra xhigh, read-only sandbox, invoked 2026-09-10 23:52 local from the main shell (codex exec ... < /dev/null). Prompt = unified diff of the 5 modified scripts + full analyze_c5_sensitivity.py + design constraints + smoke evidence (scratchpad tp2/prompt.txt). Statuses/resolution_notes by Claude after personally verifying each finding (fixture tests in scratchpad; see progress.md 2026-09-10-c). -->
---
reviewer: codex
touchpoint: code
round: A
target_files:
  - run_storya_e1_anchor.py
  - run_storya_v21_main12.py
  - run_storya_v21_tune.py
  - run_v21_tune_launcher.py
  - compute_family1_ladder.py
  - analyze_c5_sensitivity.py
findings:
  - id: CODEX-A-01
    severity: MAJOR
    category: reproducibility
    claim: "Default launcher merge changes frozen_hparams.json bytes and MD5 despite identical study contents, violating the default-behavior constraint and invalidating existing resume provenance."
    evidence: "run_v21_tune_launcher.py:161 unconditionally adds universes and arms; run_storya_v21_main12.py:653 hashes raw file bytes and :658 rejects a changed MD5."
    suggested_fix: "Preserve the original serialization for the default B,C merge. Emit the additional scope metadata only for non-default merges."
    status: FIXED
    resolution_notes: "run_v21_tune_launcher.merge: scope keys (universes/arms) are emitted ONLY for non-default merges; verified default merge output is byte-identical to HEAD merge output on the same 20 study JSONs (md5 2d49f67a… both). Note: the historical artifacts/storya_v21_tune/frozen_hparams.json (59ddd0a2…) differs from a fresh HEAD merge only because B_L2/C_L2.json were replaced by the 90-trial M14 versions on 2026-07-02, not because of this change."
  - id: CODEX-A-02
    severity: MAJOR
    category: statistics
    claim: "A partially degenerate C5 cell can pass integrity and strict 749-day pairing while silently pairing different dates."
    evidence: "analyze_c5_sensitivity.py:65 validates length against the already-filtered n_test_days; :120 pools compressed arrays and :131 checks only total length. run_storya_e1_anchor.py:837 drops undefined dates; compute_family1_ladder.py:144 packs remaining observations into leading positions."
    suggested_fix: "Before C5 aggregation, validate every canonical seed/fold/arm against the frozen eligible-day sequence. Reject shortened arrays when dates are unavailable, or retain date keys and align explicitly. Propagate alignment failures outside smoke mode. Use an available reference or the frozen calendar for C5 degeneracy reporting."
    status: FIXED
    resolution_notes: "analyze_c5_sensitivity.run_integrity: per-cell .npy length must equal the frozen calendar per-fold day count (from confirmatory L0 cells; sum 749) for all 240 cells → cells_not_full_calendar_length must be empty for PASS; paired-contrast AssertionError re-raised outside --smoke; compute_family1_ladder.degeneracy_report takes ref_arms (sensitivity mode = arms present; default L2 unchanged). Fixture: full synthetic C5 (=C L0/L1) PASS + paired diff 0; one interior obs removed → FAIL."
  - id: CODEX-A-03
    severity: MAJOR
    category: reproducibility
    claim: "The C5 analysis integrity gate reports PASS with missing frozen/provenance files or an incorrect provenance mode, allowing an unverified or untuned run into the comparison."
    evidence: "analyze_c5_sensitivity.py:229 converts a missing frozen file to None; :57 substitutes an empty provenance object; :90 accepts an unavailable MD5 check. provenance_mode is recorded at :82 but never required."
    suggested_fix: "Outside smoke mode, require the frozen file and provenance, complete 2/2 C5 studies, mode TUNED per-arm, matching MD5, and matching applied C5_L0/C5_L1 parameters."
    status: FIXED
    resolution_notes: "run_integrity(strict=True outside --smoke) requires: frozen file present + complete + n_studies==expected==2; provenance present with mode == TUNED per-arm, frozen_md5 == md5(frozen file), applied == {C5_L0, C5_L1} with src==key and params == frozen winner_params. Fixture: wrong mode → FAIL; missing provenance → FAIL; missing frozen → FAIL."
  - id: CODEX-A-04
    severity: MAJOR
    category: data-leakage
    claim: "C5 remains selected using evaluation-period outcomes; the generated leak-free re-selection and leakage-inflation interpretation is unsupported."
    evidence: "analyze_plan_aaa_t1_diagnostic.py:98 selects the last 313 valid label days; :129 scores T-1 features against those labels; :193 ranks that score. run_storya_e1_anchor.py:182 defines C5 from this ranking. analyze_c5_sensitivity.py:146 describes the resulting contrast as inflation and :197 calls C5 leak-free."
    suggested_fix: "Keep the specified C5 subset as a post-hoc sensitivity, explicitly disclose evaluation-informed selection, and describe C-minus-C5 as a conditional subset contrast. Remove claims that this isolates leakage inflation or establishes selection-independent performance. Such claims require a separate pre-evaluation selection or untouched holdout design."
    status: FIXED
    resolution_notes: "All generated strings reworded: test-informed feature-subset sensitivity; conditional subset contrast — feature restriction + re-tuning; NOT an identified leakage-inflation effect; C5 columns selected with evaluation-period outcomes (brief §9.9). Scientific disposition per TP1 Round A/B (option 1)."
  - id: CODEX-A-05
    severity: CONCERN
    category: reproducibility
    claim: "Sensitivity ledger fields contradict the actual statistical execution: they declare ten tests, a BH scope, and SPA configuration when only L1-L0 is tested and BH/SPA are skipped."
    evidence: "compute_family1_ladder.py:519 retains full-family counts and :525 retains BH settings; :528 records SPA as configured. The sensitivity summary at :557 also reports L7 KEPT despite skipping its contingency."
    suggested_fix: "In sensitivity mode, record the actual restricted pairs and test count, explicitly mark BH and SPA as unapplied, and mark L7 contingency as skipped. Preserve default ledger and summary serialization."
    status: FIXED
    resolution_notes: "compute_family1_ladder.write_ledger(sensitivity=True) now records the ACTUAL scope: pairs_tested [L1-L0], n_tests_total = 1×universes, bh_fdr NOT APPLIED, spa NOT RUN, l7_contingency SKIPPED; summary header states the same. Default (confirmatory) ledger/summary serialization untouched."
summary:
  critical: 0
  major: 4
  concern: 1
  fixed_before_reply: 5
overall_verdict: PROCEED-WITH-FIXES  # all 5 fixed + self-verified (fixture tests); Round B requested
---

H博士，the C5 feature and training paths satisfy the requested temporal contract. The remaining issues concern reproducibility, incomplete-day handling, and the interpretation of the resulting sensitivity.

1. **CODEX-A-01 — Default merge changes provenance.** Running the HEAD and modified merge functions against identical study inputs, with writes captured in memory, produced identical 20-study dictionaries but different MD5s: `2d49f67a…` versus `f05a4461…`. Removing the added `universes` and `arms` fields restored the exact original bytes. This is a functional regression because main12 deliberately rejects a changed frozen-file hash.

2. **CODEX-A-02 — Equal pooled lengths do not establish date alignment.** I constructed an in-memory C5 fixture from the complete C L0/L1 results, then removed one interior observation from L1/seed 86/fold 0 and updated its `n_test_days`, matching the runner’s behavior when Spearman IC is undefined. Integrity still returned `PASS=True`; strict pairing accepted `T=749`. The compressed seed array shifted subsequent observations, changing **52 daily contrast values**. Other seeds preserved the fold’s maximum length, hiding the missing date from the pooled-length check.

   The stability report also classified all 120 L1 cells as normal: its reference remains L2, which is absent from C5. Moreover, alignment assertions are caught and converted to “skipped” messages even outside smoke mode (`analyze_c5_sensitivity.py:254`), allowing successful completion without the requested paired result.

3. **CODEX-A-03 — Provenance absence passes the production gate.** Against a complete 240-cell fixture, each of these independently returned `PASS=True`: missing frozen file and provenance; present frozen file but missing provenance; and matching MD5 with mode `FC fixed-arm=L1`. The runner’s existing provenance check works when invoked with frozen hyperparameters, but the analysis does not establish that its inputs passed that check.

4. **CODEX-A-04 — Selection still consumes test outcomes.** The diagnostic chooses groups using absolute feature IC against evaluation-period labels. On the current source panel, its last-313-day rule selects **2024-09-27 through 2025-12-26**, inside the evaluation period. The saved diagnostic lacks exact historical date keys, so that range is a reproduction of the current source rule, not a claim about its historical execution dates.

   T−1 shifting establishes feature availability. It does not remove outcome-informed feature selection. The ranking artifact also shows **identical raw-proxy and T−1-proxy top-15 sets**; the 5/15 overlap compares the original permutation ranking with a different proxy. Consequently, C5 remains a useful post-hoc subset sensitivity, but its contrast cannot identify leakage inflation. The non-confirmatory designation should remain, with this selection dependence disclosed explicitly.

5. **CODEX-A-05 — Sensitivity metadata overstates executed procedures.** The captured sensitivity ledger reported ten tests, `bh_fdr_q=0.05`, the full-family BH scope, and SPA `M=1`, although execution correctly produced only L1−L0 and skipped BH/SPA. This affects the audit record, rather than the computed statistics.

The following checks passed:

| Constraint | Verification |
|---|---|
| C5 composition and timing | Actual tensors had shapes `(1255, 501, 51)` and `(1255, 501, 20)`. Every C5 column exactly equaled its named C column; row zero was zero. Groups and members matched both ranking artifacts. |
| Preprocessing and windows | Shared train-only preprocessing and frozen tuning/evaluation splits remain intact. Fold-0 winsorized and standardized C5 columns also matched C exactly. |
| Defaults and cell IDs | `both` remains B,C. Exhaustive enumeration confirmed unchanged `[0,2399]` and disjoint C5 `[2400,3599]`; smoke IDs were 2510 and 2630. |
| Merge completeness | Added C5 study fixtures were ignored by default merge. Explicit C5 merge produced complete 2/2 studies, accepted by main12’s completeness function. |
| Family-1 preservation | All seven default CSVs and the ledger matched HEAD and existing artifacts byte-for-byte. Summary matched after excluding its generation timestamp. |
| Statistical reuse | C-only sensitivity reproduced the confirmatory HLN, CI, and MDE rows. C seed robustness reproduced `+0.01477`, `10/10`, and zero LOSO flips. The paired subtraction direction is correctly C minus C5. |

No files were modified. Verification used existing data and memory-only output capture; model training was not rerun.