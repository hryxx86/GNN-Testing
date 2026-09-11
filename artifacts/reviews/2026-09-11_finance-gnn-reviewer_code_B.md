<!-- Rule 9 Touchpoint 2, Round B — FALLBACK reviewer (CLAUDE.md Rule 9 Fallback): Codex CLI hit its ChatGPT usage limit
mid-review on 2026-09-11 00:16 ("try again at 4:40 AM"); a first finance-gnn-reviewer attempt was cut off by the Claude
monthly spend limit (HTTP 429); this second attempt completed 2026-09-11 ~02:55 local. Cross-round diffing of
artifacts/reviews/2026-09-10_codex_code_A.md. Statuses of the new findings filled in by Claude after personally verifying
each (see resolution_notes; fixes committed after this file). -->
---
reviewer: finance-gnn-reviewer
touchpoint: code
round: B
target_files:
  - run_storya_e1_anchor.py
  - run_storya_v21_main12.py
  - run_storya_v21_tune.py
  - run_v21_tune_launcher.py
  - compute_family1_ladder.py
  - analyze_c5_sensitivity.py
round_a_findings_status:
  - id: CODEX-A-01
    status: FIXED
    note: "Ran eb8314e merge() and current merge() on identical 20 study JSONs (scratchpad copies, OUT_DIR patched): both md5 2d49f67ab31a1876927b355a29632bd2, byte-identical. Scope keys emitted only when default_scope is False (run_v21_tune_launcher.py:618,625). Subset merge C5×{L0,L1} reproduces production frozen_hparams_c5.json md5 cdb4d923… exactly."
  - id: CODEX-A-02
    status: FIXED
    note: "run_integrity requires every C5 cell .npy length == confirmatory-L0 per-fold calendar count (analyze_c5_sensitivity.py:78-88,125); a full-count cell cannot have dropped a date, so positional pooling is alignment-safe. Paired AssertionError re-raised outside --smoke (:337-339). degeneracy_report ref_arms=arms in sensitivity mode (compute_family1_ladder.py:389-401). Independently verified: all 240 C5 cells (Mac and T4) and all 480 confirmatory B/C L0/L1 cells have full calendar length [62,62,63,63,61,63,64,64,60,62,64,61]=749."
  - id: CODEX-A-03
    status: FIXED
    note: "Strict gate (analyze_c5_sensitivity.py:91-99,127) requires provenance mode 'TUNED per-arm', frozen md5 match, applied=={C5_L0,C5_L1} with src==key and params==frozen winner_params, frozen complete 2/2. Production c5_run_integrity.json: strict=true, provenance_gate_ok=true; I re-derived the md5 (cdb4d923…) and matched applied params to frozen winners."
  - id: CODEX-A-04
    status: FIXED
    note: "All generated report/provenance strings say test-informed / conditional subset contrast / NOT an identified leakage-inflation effect (analyze_c5_sensitivity.py:14-19,183-184,246-251,268-270; run_storya_v21_main12.py:715,733-738). Residual: run_storya_e1_anchor.py:78 and :442 docstrings still say 'leak-free re-selection' — code comments only, not emitted; recommend aligning wording."
  - id: CODEX-A-05
    status: FIXED
    note: "Production family1_ledger.json (C5): pairs_tested [L1-L0], n_tests_total 1, bh_fdr NOT APPLIED, spa NOT RUN, l7_contingency SKIPPED. Summary header states the same. C-only sensitivity run reproduces confirmatory family1_dm_hln/ic_ci/mde rows exactly (see body)."
findings:
  - id: FINGNN-B-01
    severity: MAJOR
    category: reproducibility
    claim: "The Colab T4 (designated PRIMARY) C5 run's provenance records source_clean=true while git_rev is null and zero module hashes were captured; the code identity of the primary run is unverifiable from its artifacts, and the C5 commit is not on origin/main so the Colab code cannot have come from a bootstrap clone."
    evidence: "experiments/storya_v21_main12_c5_t4/_run_provenance.json: git_rev=null, imported_repo_modules={'error': \"git rev-parse HEAD ... exit status 128\"}, source_clean=true. run_storya_v21_main12.py:718-719 — all(... for v in src_state.values() if isinstance(v, dict)) is vacuously True when src_state={'error':...}. Root cause: run_storya_e1_anchor.py:226 setup_workdir() chdirs to /content/drive/MyDrive/GNN测试 (not a git repo, CLAUDE.md Rule 7) and :702-705 filters modules by cwd, so on Colab mods is empty and git fails. git ls-remote origin main = eb8314e; git merge-base --is-ancestor 9008dbe origin/main = NO."
    suggested_fix: "Code: set source_clean to False/None when git fails; hash imported module files with hashlib (no git dependency); use git -C dirname(anchor.__file__). Reporting now: either designate the Mac run (self-certified 9008dbe, module blob shas 6e23f486/2696a633 == current worktree) as primary, or reconstruct and record the T4 code identity (md5 of the .py files that were copied to Colab) in progress.md, and cite the device-replication agreement (L0 120/120 bit-identical, L1 cell corr 0.951, ΔIC 0.01343 vs 0.01318) as the cross-check."
    status: FIXED
    resolution_notes: "Verified by Claude (the vacuous-truth path is real). Code: run_storya_v21_main12.py C5 branch now anchors the repo at dirname(anchor.__file__) (not cwd), records a hashlib md5 for every imported repo module unconditionally, uses `git -C repo`, and sets source_clean = None (with git_error) when git is unavailable. Reporting: T4 code identity reconstructed post hoc by md5sum ON THE COLAB VM of all 7 imported repo modules + frozen file vs the content of commit 9008dbe → all match (experiments/storya_v21_main12_c5_t4/_code_identity_t4.json); c5_run_integrity.json 'inputs' cites it; device replication cited. T4 stays primary (pre-declared)."
  - id: FINGNN-B-02
    severity: MAJOR
    category: reproducibility
    claim: "Two C5 result directories exist with different L1 numbers, and the primary analysis artifacts do not record which one they were computed from; the primary was produced with a NON-default --c5-main-dir that appears nowhere in the outputs, and the device-replication table has no generator script in the repo."
    evidence: "analyze_c5_sensitivity.py:286 default --c5-main-dir = experiments/storya_v21_main12_c5 (Mac, mps). artifacts/storya_v21_family1_c5/c5_seed_robustness.csv C5 ΔIC=0.01343 / HLN p=0.00802 matches ONLY the T4 dir (I recomputed: Mac 0.01318 / p 0.01252; T4 0.01343). run_integrity output (:100-122) and write_ledger sensitivity branch (compute_family1_ladder.py:513-536) carry no input path, results.csv md5, or device. grep for c5_device_replication across *.py: no generator."
    suggested_fix: "Add an inputs block (c5_main_dir, conf_main_dir, results.csv/manifest.csv md5, device + git_rev from _run_provenance.json) to c5_run_integrity.json and the sensitivity ledger; state 'primary = T4 dir' in c5_comparison.md header; commit the device-replication generator."
    status: FIXED
    resolution_notes: "c5_run_integrity.json now carries 'inputs' {c5_main_dir, conf_results_csv, results/manifest md5, device, platform, git_rev, source_clean, post-hoc code identity}; compute_family1_ladder.write_ledger (sensitivity branch only) carries 'inputs' {main_dir, results/manifest md5, device, platform, git_rev, source_clean}; c5_comparison.md header states the INPUT dir/device/md5; default --c5-main-dir switched to the pre-declared primary (T4); device replication is now generated by analyze_c5_sensitivity.py --replicate-main-dir (function device_replication) — regenerated artifacts committed."
  - id: FINGNN-B-03
    severity: CONCERN
    category: statistics
    claim: "The paired contrast (ΔIC_C − ΔIC_C5) reports a CI but no SE/MDE; its implied MDE (~0.025) exceeds the entire C L1−L0 effect (0.0148), so the null result cannot be read as 'C5 ≈ C' — only as 'not distinguishable at this power'."
    evidence: "c5_paired_contrast.csv: mean 0.00133, CI [−0.01586, +0.01890] → SE≈0.0089, 2.8×SE≈0.025. analyze_c5_sensitivity.py:174-182 computes CI only; run_ci_and_mde (compute_family1_ladder.py:331-343) already has the SE/MDE machinery."
    suggested_fix: "Add SE_block and MDE_2p8xSE to the paired-contrast row and phrase the finding as an underpowered non-rejection, not equivalence (a TOST bound would be the honest alternative)."
    status: FIXED
    resolution_notes: "paired_contrast now reports SE_block + MDE_2p8xSE (same StationaryBootstrap construction as family1); c5_comparison.md paired section states 'underpowered non-rejection; does not exclude a halving or a doubling; equivalence not established'. Same point raised by TP3 R-A-03 — wording adopted in docs/analysis.md."
  - id: FINGNN-B-04
    severity: CONCERN
    category: statistics
    claim: "C5's tuned winners were selected on NEGATIVE validation IC (L0 −0.012; L1 −0.045 with tuning-seed ICs −0.006/−0.045/−0.084) whereas C's winners had +0.074/+0.060; the C5 're-tuning' therefore carried no positive selection signal and moved capacity in opposite directions (MLP hidden 128→32, LightGBM leaves 15→63). The comparison table shows the val-IC but does not flag this asymmetry."
    evidence: "experiments/storya_v21_tune/C5_L1.json top_table (all 5 finalists mean_val_ic_3seed ≈ −0.045); artifacts/storya_v21_family1_c5/c5_tuned_hparams.csv rows 0-3."
    suggested_fix: "One sentence in c5_comparison.md / appendix: the C5 contrast is conditional on an HP selection that was effectively uninformative on 2022H2; do not attribute the ΔIC difference (or its absence) to feature restriction alone."
    status: FIXED
    resolution_notes: "Verified from C5_L1.json top_table (all finalists hidden 32 / 1 layer / dropout 0.3, val-IC −0.0448…−0.0453) and C5_L0.json (−0.0121…−0.0122). Disclosure sentence added to c5_comparison.md (tuned-winners section) and to docs/analysis.md 2026-09-11-a."
summary:
  critical: 0
  major: 2
  concern: 2
  fixed_before_reply: 0
overall_verdict: PROCEED-WITH-FIXES
---

## What I actually ran

1. `git diff eb8314e` on the five scripts (full read) plus full read of `analyze_c5_sensitivity.py`.
2. Launcher byte-identity: loaded `git show eb8314e:run_v21_tune_launcher.py` and the current file as modules with `OUT_DIR` pointed at scratchpad copies of the study JSONs; default merges byte-identical (md5 `2d49f67a…` both); C5 subset merge reproduces production md5 `cdb4d923…`.
3. `compute_family1_ladder.py --main-dir experiments/storya_v21_main12_tuned --output-dir <scratchpad> --universes C --arms L0,L1 --sensitivity`: the C L1−L0 rows of `family1_dm_hln.csv`, `family1_ic_ci.csv`, `family1_mde.csv` are identical to `artifacts/storya_v21_family1/` (all non-BH columns, `DataFrame.equals` True).
4. Per-cell `.npy` length vs `n_test_days` vs calendar for C5 Mac, C5 T4, and confirmatory B/C L0/L1: zero mismatches, zero short cells; cell_id ranges [2400,2639] and [0,2399], all unique.
5. Sign convention: `d = dc − d5` (`analyze_c5_sensitivity.py:171`); production mean_paired_diff +0.00133 = 0.01477 − 0.01343. Positive = C larger than C5, as specified.
6. Estimand: `seed_pooled` (`analyze_paper_eval_robustness.py:43-47`) is the n_test_days-weighted mean of fold IC_mean per seed, matching the docstring; k/10 and LOSO m/10 computed on per-seed ΔIC. MLP param count 2337 checks by hand (20→32, 32→32, LN, 32→16→1).
7. Provenance: Mac run `_run_provenance.json` blob shas for `run_storya_e1_anchor.py`/`run_storya_v21_main12.py` equal current `git hash-object` output and the worktree is clean against 9008dbe. T4 run: see B-01.

## Notes on items I checked and did not flag

- `compute_family1_ladder` default branches: `degeneracy_report` with `ref_arms=None` reduces to the prior L2 loop; `write_summary` default title/headers reconstruct the same strings; `cl5s_robustness` still runs when `('C','L5s')` is in `agg`. Combined with the Round-A byte-for-byte check, default outputs are preserved.
- `_delta_series` truncates L0/L1 to min length silently even in strict mode, and the confirmatory side is not gated by `run_integrity`. Empirically moot (item 4 above), so not a finding.
- `main12` `--universe both` still equals `['B','C']` (`run_storya_v21_main12.py:625`); `assert_cell_id_injective` verifies the confirmatory block is exactly [0,2399].
- `_meta.json` in the C5 dirs lists `arms_implemented` for the full runner, not the C5 scope; `_run_provenance.json` carries the actual invocation, so this is not a mislabel.

## Bottom line

The statistical path (integrity gate, paired contrast, seed-level estimands, sensitivity-mode ladder) is correct and reproduces the confirmatory numbers where it should. What is not yet defensible is the chain of custody for the primary numbers: the T4 run's own provenance falsely certifies a clean source with no code identity captured (B-01), and the primary analysis artifacts do not say which of the two result directories they came from (B-02). Both are fixable without retraining: decide primary vs replicate explicitly, record the input dir and T4 code identity, and add the paired-contrast MDE (B-03) plus the negative-val-IC caveat (B-04) before any of these numbers go into the appendix.

## Claude closure (2026-09-11)

- A-04 residual (anchor docstrings :78/:442 "leak-free re-selection") → reworded to "test-informed feature-subset".
- B-01..B-04 → FIXED as per resolution_notes; the regenerated primary artifacts (`artifacts/storya_v21_family1_c5/`) and the Mac replicate (`_c5_mac/`) were produced with the fixed analyzer; T4 remains the pre-declared primary.
