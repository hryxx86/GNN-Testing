---
reviewer: codex
touchpoint: results
round: D
overall_verdict: PASS-WITH-CONCERNS
findings:
  - id: CODEX-D-01
    status: OPEN
    severity: CONCERN
    location: docs/storya_paper_draft_v2.md:368
    finding: >
      ST1c counts and seeds match, but "0 duplicate after merge" is only
      block-local. Raw merge of main + L7 + FC has 3120 rows, 2400 unique
      cell_id values, and 720 duplicate cell_ids between tuned L3/L4/L5 and FC
      L3/L4/L5.
    fix: >
      Bound duplicate-free as within-block/shard merge, or define uniqueness as
      (family/block, cell_id).
checks:
  ST1a: "PASS: fold max n_test_days = [62,62,63,63,61,63,64,64,60,62,64,61], sum 749."
  ST1b: "PASS: neural and LGB grids match protocol §4."
  ST1c: "COUNTS PASS: 2160 main + 240 L7 + 720 FC = 3120; canonical 10 seeds present."
  T8: "PASS: novelty is bounded; caveat frames ✗ as not reported, not impossible."
---
