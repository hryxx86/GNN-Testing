<!-- Rule 9 session-closeout audit 3/4 (Explore agent, independent context), 2026-09-12. Scope = the C-pre session diff
(b969a62..HEAD). The agent worked strictly read-only and verified the C5 regression in-process rather than by file diff.
Statuses filled in by Claude after applying each fix and regenerating the artifacts. -->
---
reviewer: explore-correctness
touchpoint: closeout
round: closeout
target_files:
  - run_storya_cpre_select.py
  - run_storya_e1_anchor.py:78,197-221,494-554
  - run_storya_v21_main12.py:129,182-214,692-790
  - run_storya_v21_tune.py:184-190,341-346
  - analyze_c5_sensitivity.py
  - paper_figs/fig_plan_aaa_t1.py
  - .gitignore:68-78
findings:
  - id: EXPL-CODE-01
    severity: MAJOR
    category: correctness
    claim: "excluded_fold_share is guarded only in the markdown. The published CSV and the stdout line emit the raw ratio, which for CPRE is +4.7441 for a fold whose contribution is the MOST NEGATIVE of the twelve - the sign is flipped by the near-zero negative denominator, so the CSV row is quantitatively and directionally misleading."
    evidence: "analyze_c5_sensitivity.py:338 share = (n_ex * fold_delta) / ((T + n_ex) * pooled_all); both negative -> positive ratio. Published in cpre_ex_fold.csv as 4.7441 with no column marking it undefined; the stdout line formatted it as 474%; a None share would raise TypeError there."
    suggested_fix: "Compute the guard once in ex_fold_stats, put the flag in the returned row and the CSV, and have write_md and stdout read the flag."
    status: FIXED
    resolution_notes: "ex_fold_stats now bootstraps the all-fold pooled SE, sets excluded_fold_share only when |pooled| >= that SE, and additionally emits excluded_fold_share_raw, excluded_fold_share_is_meaningful, pooled_delta_IC_all_folds, pooled_SE_block_all_folds and n_folds_present. write_md and the stdout line read the flag (no recomputation, no TypeError path). Artifacts regenerated for CPRE and C5; every pre-existing numeric field is unchanged."
  - id: EXPL-CODE-02
    severity: CONCERN
    category: reproducibility
    claim: "source_clean can be reported True for a source module that is not in the repository at all: git status --porcelain -- <path> prints nothing for an IGNORED file, so an ignored module maps to git_status == '' and counts as clean."
    evidence: "run_storya_cpre_select.py:90-94 and run_storya_v21_main12.py:755-762; verified empirically on an ignored artifacts path. Not hit this session only because run_step3_plan_z_part_a.py sat at repo root and surfaced as '??'."
    suggested_fix: "Probe tracked-ness explicitly (git ls-files) and mark untracked/ignored paths so source_clean is False."
    status: FIXED
    resolution_notes: "Both git-identity blocks now run git ls-files over the same path list and stamp '!!untracked-or-ignored' for anything git does not track, which fails the source_clean conjunction."
  - id: EXPL-CODE-03
    severity: CONCERN
    category: correctness
    claim: "np.nan_to_num(x, 0.0) binds 0.0 to copy, so the call fills in place; at run_storya_e1_anchor.py:546 the argument is a view into part_a's tensor, and +-inf maps to +-1.79e308 rather than 0."
    evidence: "numpy signature (x, copy=True, nan=0.0, posinf=None, neginf=None); new call sites at run_storya_e1_anchor.py:541-543, :546, :549."
    suggested_fix: "Use the keyword form at the new call sites."
    status: FIXED
    resolution_notes: "All five new call sites use nan=0.0, posinf=0.0, neginf=0.0 (copy=True at the verification assert). Values unchanged on the current artifacts (inputs are NaN- and inf-free), so nothing was regenerated for this. Same issue as EXPL-LEAK-05."
  - id: EXPL-CODE-04
    severity: CONCERN
    category: reproducibility
    claim: "For --universe CPRE the C5 comparator row in <p>_tuned_hparams.csv is pinned to the default C5 frozen file and a hardcoded n_inputs=20, ignoring the --c5-* overrides; and the markdown source line never cites that file."
    evidence: "analyze_c5_sensitivity.py:706 extra_hp = [('C5', SPECS['C5']['frozen'], 20)]; :411 hardcodes 51 for C; :563 emits a source line naming only frozen_target and the confirmatory file."
    suggested_fix: "Add --c5-frozen, derive widths from the anchor name lists, and build the source line from the actual specs."
    status: FIXED
    resolution_notes: "Added --c5-frozen; the C width comes from len(UNIVERSE_C_ALPHA158_NAMES)+len(UNIVERSE_C_EXTRA_NAMES) and the C5 width from len(UNIVERSE_C5_NAMES); hparam_report collects the frozen paths it actually read and write_md prints them in the source line. Regenerated markdowns now name all three frozen files."
  - id: EXPL-CODE-05
    severity: CONCERN
    category: correctness
    claim: "ex_fold_stats performs no frozen-calendar assertion and is internally inconsistent about truncation: per-fold deltas truncate to min(len(L1), len(L0)) but the excluded fold is re-derived from the untruncated arrays; with --no-paired --ex-fold N no per-fold alignment check runs for the comparator universes."
    evidence: "analyze_c5_sensitivity.py:320-321 vs :335; ex_fold_stats took no calendar argument; the rank denominator hardcoded N_FOLDS while the rank is computed over present folds."
    suggested_fix: "Pass the frozen calendar and assert per fold on both arms; reuse the truncated series; report len(contrib) as the denominator."
    status: FIXED
    resolution_notes: "ex_fold_stats takes the frozen calendar (passed for every universe outside smoke mode) and asserts len(L1) == len(L0) == calendar[f] per fold; the excluded fold reuses the truncated series; n_folds_present is emitted and used as the rank denominator in the markdown and stdout."
  - id: EXPL-CODE-06
    severity: CONCERN
    category: other
    claim: "The cell_id docstrings still state the pre-CPRE invariant (3 universes, range [0, 3599], universe < 3, sensitivity block == C5) inside the two functions changed this session."
    evidence: "run_storya_v21_main12.py:183-185 and :191-193 against UNIVERSE_IDX at :129 and the runtime output over 4800 ids."
    suggested_fix: "Restate both docstrings in terms of len(UNIVERSE_IDX) and per-universe blocks; injectivity needs arm_idx < 10."
    status: FIXED
    resolution_notes: "Both docstrings rewritten; assert_cell_id_injective still passes over the 4800-id space with the three blocks reported."
summary:
  critical: 0
  major: 1
  concern: 5
  fixed_before_reply: 0
overall_verdict: PASS-WITH-CONCERNS
---

## Verified correct (agent)

- C5 regression: recomputing with the rewritten analyzer against the published C5 dirs gives DataFrame-equality for c5_ex_fold.csv, c5_paired_contrast.csv (including the note strings at that time) and c5_seed_robustness.csv, and run_integrity PASS. Against 6acd834 every pre-existing value is byte-identical; the deltas are additive columns/keys plus reworded prose.
- Confirmatory byte-invariance: empty diff for artifacts/storya_v21_family1/, experiments/storya_v21_main12_tuned/, frozen_hparams.json, compute_family1_ladder.py, compute_e6_dm_spa.py, analyze_paper_eval_robustness.py. ALL_UNIVERSES is still [B, C]; --universe both cannot reach a sensitivity universe; the launcher's default merge still suppresses the scope keys.
- All in-scope files compile; assert_cell_id_injective passes over 4x10x12x10 with confirmatory [0,2399] and blocks C5 [2400,3599], CPRE [3600,4799] pairwise disjoint, and the generalised assert is strictly stronger than the old one.
- Window asserts are tight, not off-by-one: 231 train days, all_dates[train_days[-1] + 21] == 2022-06-30 == train_end, next label would end 2022-07-01, val_days[0] == 2022-07-01.
- Roll axis, row-0 zeroing and the raw/rolled equality assert are correct and not defeated by NaN (the array is NaN-free); name slicing maps index to name consistently.
- Column order is group rank then groups_168 member order with dedupe; the ranking sort is total and deterministic; unranked groups cannot be selected.
- The md5 gate is non-circular and closed; the full hash chain (selection.json -> tune JSONs -> run provenance -> family ledger -> analyzer) matches on disk.
- Smoke artifacts are excluded from merges by three independent mechanisms; the selector refuses to write smoke output into the published dir.
- The C5 legacy alias behaves as documented in both directions and cannot be reinterpreted mid-run; the degraded-mode out-dir guard refuses the published dir.
- Numeric edge cases checked on the real panel: the Alpha158 array has no NaN or inf; the high-zero-rate columns are genuine values; hc_mom12m's 63% constant cross-sections are what the non-constant test removes, giving exactly 85/231; Spearman guards are in place; contribution ranks are correct; the tests-inventory minimum has no collision risk.
- Reproducibility: toplevel-relative status keys, porcelain offset, hash-object zip failing closed, archived source identity at 044dd09, input md5 coverage, the CPRE-only execution keys, append-style provenance with the CORRECTION preference, and untouched seeding.
- Output schema: explicit columns everywhere, prefixes from SPECS, BH column degraded to None rather than fabricated False, integrity records n_features_expected and makes the name match a PASS term.
- paper_figs/fig_plan_aaa_t1.py: the invariant assert holds on the current CSV and every count in the title/legend is derived from the data.
- .gitignore additions are strictly additive whitelists scoped to the new C-pre paths.
