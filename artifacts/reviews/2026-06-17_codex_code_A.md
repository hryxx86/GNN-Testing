---
reviewer: codex
touchpoint: code
round: A
target_files:
  - run_storya_v21_main12.py
  - run_storya_v21_l7_hats.py
findings:
  - id: CODEX-A-01
    severity: MAJOR
    category: reproducibility
    claim: "--frozen-hparams/--fc-fix-arm can run against the DEFAULT pilot output dir (--out-dir optional), so resume can silently skip pilot-completed cells or mix tuned/FC rows into the pilot manifest."
    evidence: "main12 L127-130 default pilot paths; --out-dir reassign is optional; load_manifest_done reads that path → identical cell_ids → skips pilot cells / appends tuned rows to the pilot results.csv. Same in l7_hats."
    suggested_fix: "Fail closed: when --frozen-hparams is set, REQUIRE a non-default --out-dir (refuse to write tuned/FC into the pilot dir)."
    status: FIXED
    resolution_notes: "FIXED both runners: capture _pilot_out = OUT_DIR before reassign; if args.frozen_hparams and OUT_DIR == _pilot_out → raise SystemExit (covers both 'no --out-dir' and '--out-dir == pilot'). Verified: `--frozen-hparams X` without --out-dir now aborts with a clear message; with --out-dir experiments/storya_v21_main12_tuned it proceeds."
  - id: CODEX-A-02
    severity: CONCERN
    category: reproducibility
    claim: "Provenance not trustworthy: main12 overwrites _frozen_hp_provenance.json per subset invocation; L7 writes no provenance."
    evidence: "main12 builds `applied` only for current universes_run/arms_run and opens with 'w' (a subset run clobbers a fuller record); l7_hats only prints + patches, no provenance writer."
    suggested_fix: "Write provenance in both; include frozen md5 + selected cells; MERGE with existing (validate mode/md5) instead of clobbering."
    status: FIXED
    resolution_notes: "FIXED. Both runners now: compute frozen_md5; if _frozen_hp_provenance.json exists, READ + validate (mode + md5 must match, else abort 'refuse to mix HP modes in one dir') + MERGE the applied dict (subset runs union, never clobber); else create. L7 now writes provenance too. md5 recorded for the exact frozen file used."
  - id: CODEX-A-03
    severity: CONCERN
    category: correctness
    claim: "New fail-closed gates use `assert` → python -O disables the frozen-completeness check + FC same-model guard."
    evidence: "main12 load_frozen_hparams completeness assert; --fc-fix-arm requires/model-compat asserts; L7 imports the same load_frozen_hparams."
    suggested_fix: "Replace these asserts with explicit `if ...: raise SystemExit/ValueError`."
    status: FIXED
    resolution_notes: "FIXED: load_frozen_hparams completeness + the FC require-frozen + FC same-model guard are now explicit `if ...: raise SystemExit(...)` (survive python -O). load_frozen_hparams is shared (L7 imports it), so the completeness gate covers both."
summary:
  critical: 0
  major: 1
  concern: 2
  fixed_before_reply: 0
overall_verdict: PROCEED-WITH-FIXES
---

# Codex Code Review — Touchpoint 2, Round A — D-RERUN-12F frozen-HP injection

Verdict **PROCEED-WITH-FIXES** (0 CRIT / 1 MAJOR / 2 CONCERN). All FIXED.

**Codex confirmed the injection MECHANICS are sound** (the core correctness): the monkeypatch reaches
every training path with the correct per-arm value given the loop order — main12 injects before the
seed loop for each (universe,fold,arm) and the call paths read anchor globals at call time
(train_gnn_per_day_edges function-local re-import; anchor.train_nn/train_lightgbm module-global); L7
patches hats.HATS_HPARAMS once per universe and train_hats reads it at call time. No stale-binding or
wrong-arm-HP hazard. Claude's self-verification (no-flag identity, B-L0 λ2 land, per-univ distinct,
FC L2-fix, L7 structure-key preservation) all stand.

The 3 findings were about **output-dir / provenance ISOLATION**, not the HP binding. All fixed:
- A-01 (MAJOR): frozen/FC now REQUIRE a non-pilot --out-dir (fail closed) — the must-fix before launch.
- A-02 (CONCERN): provenance now merges + validates (md5 + mode) in both runners; L7 writes it too.
- A-03 (CONCERN): gates converted assert → raise (survive python -O).

Cleared for the tuned rerun + FC arm AFTER the fixes (verified). Actual launch = separate Colab job + H博士.
