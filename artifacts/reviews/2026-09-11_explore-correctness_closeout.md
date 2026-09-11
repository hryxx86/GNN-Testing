<!-- Rule 9 session-closeout audit 3/4 (Explore agent, independent context), 2026-09-11 ~03:40 local. Scope = this session's
code diff eb8314e..a903c5e + housekeeping 46b3b8c. Statuses filled in by Claude after reading each cited location and
applying the fixes (negative tests run for the new guards; artifacts regenerated). -->
---
reviewer: explore-correctness
touchpoint: closeout
round: closeout
target_files:
  - analyze_c5_sensitivity.py
  - compute_family1_ladder.py
  - run_storya_v21_main12.py
  - run_storya_v21_tune.py
  - run_v21_tune_launcher.py
  - run_storya_e1_anchor.py
  - compute_e6_dm_spa.py
  - run_storya_e3_news_edge.py
  - run_storya_e4_alpha.py
findings:
  - id: EXPL-CODE-01
    severity: MAJOR
    category: correctness
    claim: "compute_family1_ladder.py --sensitivity writes into --output-dir whose default is the CONFIRMATORY artifacts/storya_v21_family1; omitting --output-dir silently overwrites the confirmatory Family-1 artifacts with 1-row C5 stats and exits 0."
    evidence: "compute_family1_ladder.py:657 default vs :704-738 sensitivity writes."
    suggested_fix: "Refuse (explicit raise) when --sensitivity targets the confirmatory default dir."
    status: FIXED
    resolution_notes: "main() now raises SystemExit if --sensitivity and output_dir resolves to artifacts/storya_v21_family1; negative test: `--sensitivity` without --output-dir → refusal, exit 1."
  - id: EXPL-CODE-02
    severity: MAJOR
    category: reproducibility
    claim: "analyze_c5_sensitivity.py takes cell-level inputs from --c5-main-dir and headline stats from --c5-family-dir with no check that both describe the same run; two divergent stat/result dirs now exist (T4 vs Mac) and the two defaults could be mixed silently."
    evidence: "analyze_c5_sensitivity.py:402-411, :466-468; family1_ledger.json inputs.results_csv_md5 eaa8af1d (T4) vs 1d7445bf (Mac)."
    suggested_fix: "Fail closed unless the family-dir ledger's inputs.results_csv_md5 equals the md5 run_integrity computes for --c5-main-dir."
    status: FIXED
    resolution_notes: "main() loads <family_dir>/family1_ledger.json and raises SystemExit on md5 mismatch (non-smoke); integrity json records family_dir_matches_main_dir; negative test (Mac main dir + T4 family dir) → refusal."
  - id: EXPL-CODE-03
    severity: MAJOR
    category: correctness
    claim: "Degraded modes (--conf-only, --smoke: n_boot=200, strict=False) write the SAME filenames into the published --c5-family-dir; no smoke/n_boot marker in the md."
    evidence: "analyze_c5_sensitivity.py:418-420, :427-430, :439-440, :469/:489."
    suggested_fix: "Namespace degraded output and stamp n_boot/strict."
    status: FIXED
    resolution_notes: "Degraded modes default to <family_dir>_smoke / _confonly and refuse to write into the published dir; integrity json + md header carry analysis_mode {n_boot, strict, smoke, out_dir}. Smoke re-run lands in family1_c5_smoke_smoke/."
  - id: EXPL-CODE-04
    severity: MAJOR
    category: correctness
    claim: "write_md emits fixed interpretive verdicts (|ΔIC| below MDE; paired interval wider than the contrast; fold-9 share; negative finalist val-IC; ≈14× params) independent of the computed numbers — true today, wrong on any future re-run."
    evidence: "analyze_c5_sensitivity.py:348-351, :360-361, :369-371, :384-387."
    suggested_fix: "Derive each verdict from the computed rows."
    status: FIXED
    resolution_notes: "MDE verdict per universe computed ('BELOW'/'ABOVE'); paired note derived from ci_excludes_0 and paired MDE vs |C contrast|; fold share computed per universe (excluded_fold_share) with per-universe wording; tuning disclosure computed from the finalist tables (negatives/total, ranges), C winners, and the param ratio. The long reading note remains editorial text tied to the 2026-09-11 run and is labelled as such."
  - id: EXPL-CODE-05
    severity: CONCERN
    category: correctness
    claim: "family1_dm_hln.csv columns differ between modes (sensitivity adds bh_applied, writes BH columns as None → NaN, bool(NaN) True); asymmetric flag."
    evidence: "compute_family1_ladder.py:248-253."
    suggested_fix: "Always emit bh_applied; explicit sentinel."
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "Not changed: emitting bh_applied in the confirmatory branch would alter the confirmatory CSV schema on re-run (byte-identity constraint). The only reader (_row) uses pd.isna; documented in the ledger ('bh_fdr: NOT APPLIED') and here."
  - id: EXPL-CODE-06
    severity: CONCERN
    category: correctness
    claim: "paired_contrast builds a per-row dynamic key mean_delta_{other} → ragged CSV columns."
    evidence: "analyze_c5_sensitivity.py:200; c5_paired_contrast.csv."
    suggested_fix: "Fixed keys."
    status: FIXED
    resolution_notes: "Keys are now other_universe / mean_delta_other / mean_delta_c5."
  - id: EXPL-CODE-07
    severity: CONCERN
    category: reproducibility
    claim: "_run_provenance.json readers take [-1] but the T4 file is not chronological (correction entry stamped in local time); readers surface one entry only, hiding a multi-device resume."
    evidence: "analyze_c5_sensitivity.py:92-95; compute_family1_ladder.py:521-524; T4 _run_provenance.json (2 entries)."
    suggested_fix: "Select explicitly; emit n_invocations + distinct devices."
    status: FIXED
    resolution_notes: "run_integrity prefers an explicit CORRECTION entry, else the last; records n_invocations, devices_seen and multi_device_warning; md header shows devices seen."
  - id: EXPL-CODE-08
    severity: CONCERN
    category: reproducibility
    claim: "run_integrity reads the code-identity file with a commit-specific key and stamps the literal '9008dbe'."
    evidence: "analyze_c5_sensitivity.py:106-107; _code_identity_t4.json keys."
    suggested_fix: "Generic keys."
    status: FIXED
    resolution_notes: "_code_identity_t4.json now also carries generic 'commit' / 'all_modules_match'; the reader uses generic keys with a fallback to the suffixed legacy keys and no literal commit."
  - id: EXPL-CODE-09
    severity: CONCERN
    category: correctness
    claim: "Strict gate checks n_features == 20 but never compares feature_names to anchor.UNIVERSE_C5_NAMES."
    evidence: "analyze_c5_sensitivity.py:133, :143-148."
    suggested_fix: "Add the name comparison to PASS."
    status: FIXED
    resolution_notes: "feature_names_match_UNIVERSE_C5_NAMES added to the integrity json and to the strict PASS conjunction (True on both runs)."
  - id: EXPL-CODE-10
    severity: CONCERN
    category: correctness
    claim: "An empty requested (universe, arm) scope silently yields empty CSVs, exit 0, and a ledger n_tests_total computed from the requested scope."
    evidence: "compute_family1_ladder.py:220-226, :543, :670-672."
    suggested_fix: "Fail closed on empty scope; n_tests_total from rows produced."
    status: FIXED
    resolution_notes: "Sensitivity mode raises SystemExit if any requested (universe, arm) has an empty series; n_tests_total now = len(dm_df) (1 for C5)."
  - id: EXPL-CODE-11
    severity: CONCERN
    category: reproducibility
    claim: "Tune execution block runs git rev-parse against CWD (Colab → Drive folder → None; foreign repo → wrong rev)."
    evidence: "run_storya_v21_tune.py:322-332."
    suggested_fix: "Mirror main12 (git -C code dir, git_error, md5 fallback)."
    status: FIXED
    resolution_notes: "tune execution block now uses git -C <code dir>, records git_error and md5 of tune/anchor/main12 modules; smoke-tested."
  - id: EXPL-CODE-12
    severity: CONCERN
    category: reproducibility
    claim: "main12 provenance keys git status on code-dir-relative paths while git status prints toplevel-relative paths; vacuous source_clean if the code ever lives in a subdirectory."
    evidence: "run_storya_v21_main12.py:709-720."
    suggested_fix: "Resolve --show-toplevel and key on toplevel-relative paths."
    status: FIXED
    resolution_notes: "Implemented (toplevel via git -C repo rev-parse --show-toplevel; status/hash-object keyed on toplevel-relative paths; toplevel_path recorded per module); smoke-tested — source_clean now reflects the actual uncommitted state at run time."
  - id: EXPL-CODE-13
    severity: CONCERN
    category: correctness
    claim: "n_test_days_total_per_arm reports the first seed column only, mislabelled as a per-arm total."
    evidence: "analyze_c5_sensitivity.py:128-129."
    suggested_fix: "Rename and emit min/max across seeds."
    status: FIXED
    resolution_notes: "Replaced by n_test_days_per_arm_per_seed_min_max (each [749, 749])."
  - id: EXPL-CODE-14
    severity: CONCERN
    category: correctness
    claim: "ex_fold_stats does not validate --ex-fold; an out-of-range fold would produce mislabelled full-sample rows then crash in write_md."
    evidence: "analyze_c5_sensitivity.py:220-239, :358."
    suggested_fix: "Raise on out-of-range / missing fold."
    status: FIXED
    resolution_notes: "ValueError raised for ex_fold outside 0..11 or missing per-day data."
  - id: EXPL-CODE-15
    severity: CONCERN
    category: other
    claim: "Section banner says '(L1-L0)_C5 minus (L1-L0)_C' while code/docstring/outputs compute C − C5."
    evidence: "analyze_c5_sensitivity.py:174 vs :191."
    suggested_fix: "Fix the banner."
    status: FIXED
    resolution_notes: "Banner corrected."
summary:
  critical: 0
  major: 4
  concern: 11
  fixed_before_reply: 0
overall_verdict: PASS-WITH-CONCERNS
---

## Verified correct by the agent (no findings)

- cell_id space re-enumerated with the real 10-arm ARM_ORDER: injective, [0,3599], confirmatory block exactly [0,2399], C5 block [2400,3599]; the run occupies [2400,2639]; `--universe both` still B,C.
- C5 = pure column selection of C (all 20 names in UNIVERSE_C_ALPHA158_NAMES, no hc_; winsorize/standardize per column); the selection reproduces `plan_aaa_orig_rank<=15 ∩ proxy_rank_t1<=15` exactly with member lists verbatim from ranking.csv.
- Headline numbers reproduce from the .npy files with plain numpy (C5 0.01343271, C 0.01476515, B 0.01428195); fold lengths identical across universes (sum 749); ex-fold-9 reproduces exactly.
- Default-path byte-identity: merge() default → no scope keys, key order unchanged (frozen_hparams.json still 59ddd0a2… on disk); degeneracy_report(ref_arms=None) → old L2 path; filtered lists equal the full lists; frozen_hparams* glob skip strictly safer.
- tune execution block cannot break the JSON dump; study_db matches run_study's study_name incl. _smoke suffix.
- hparam_report monkeypatch restores NN_HPARAMS in finally; 2337 vs 31745 params back the ≈14× claim.
- e4 cell_id stride change does not corrupt resume (manifest keyed on (edge_config, fold, seed)); all rows match the new formula.
- All nine files compile; no files were modified by the agent.
