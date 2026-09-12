---
reviewer: finance-gnn-reviewer
touchpoint: code
round: A
fallback_reason: "Codex CLI hit rate limit (error response; Rule 9 Fallback clause triggered). Session date 2026-04-23."
target_files:
  - scripts/verify_docs_provenance.py
findings:
  - id: FINGNN-A-01
    severity: MAJOR
    category: correctness
    claim: "TEST_STAT_RE over-matches single-letter statistic names (t, F, Z) in prose, driving false-positive violations that will force spurious citations or push authors to weaken the rule."
    evidence: "verify_docs_provenance.py:42 — TEST_STAT_RE alternation included [tFZ]. Claude smoke-tested: a file with 'at t=0 the gradient stabilizes', 'Z = 1.96 is the CI threshold', 'F = 5 pipeline stages' produced 3 unintended violations."
    suggested_fix: "Drop the bare `[tFZ]` alternation. Keep T_SPA / NW_t / studentized_t — these are the real statistical-claim forms in our corpus and cover the 2026-04-21-c failure mode."
    status: FIXED
    resolution_notes: "Applied: TEST_STAT_RE narrowed to `T_SPA | NW_t | studentized_t` only (verify_docs_provenance.py:42). Smoke-tested 2026-04-23: 3→0 false positives on prose scenarios; T_SPA = 1.231 still correctly flagged as un-cited."
  - id: FINGNN-A-02
    severity: MAJOR
    category: correctness
    claim: "CITATION_PATH_RE accepts bare `.py` filenames anywhere in the ±5-line window, which lets unrelated script mentions (e.g. 'Imported from run_losses.py') silence a real numeric claim a few lines away."
    evidence: "verify_docs_provenance.py:76 — extension list included `py` and `ipynb`. Claude smoke-tested: 'Imported from run_losses.py ... Sharpe = 0.54' produced 0 violations (Sharpe silenced by .py mention). Ironically, this is the symmetric of the 2026-04-21-c attribution bug."
    suggested_fix: "Remove `py` and `ipynb` from CITATION_PATH_RE. Optionally allow them via CITATION_PER_RE with `per|from|see|cf.` prefix."
    status: FIXED
    resolution_notes: "Applied. CITATION_PATH_RE restricted to data artifacts only (csv/json/parquet/npy/tsv/pkl/yaml/yml/feather). Went further than reviewer suggested: also removed .py and .ipynb from CITATION_PER_RE because 'Imported from run_losses.py' matches `from X.py` but is an import description, not an attribution. Smoke-tested 2026-04-23: 'Sharpe = 0.54' with only .py mention nearby now correctly fires; genuine data citations ('per experiments/...csv') still pass."
  - id: FINGNN-A-03
    severity: CONCERN
    category: correctness
    claim: "iter_non_code_lines does not recognize indented fenced code blocks (up to 3 leading spaces per CommonMark), so numeric examples inside list-nested code fences will be scanned as prose."
    evidence: "verify_docs_provenance.py:99 — CODE_FENCE_RE anchored at column 0."
    suggested_fix: "Relax to `^\\s{0,3}```` per CommonMark spec."
    status: FIXED
    resolution_notes: "Applied (verify_docs_provenance.py:99). Documented the remaining limitation (fences with 4+ leading spaces inside deeply-nested list items still scanned as prose) in .claude/rules/docs.md §4 under 'Known limitations of the verifier'."
  - id: FINGNN-A-04
    severity: CONCERN
    category: correctness
    claim: "Markdown table rows like `| Sharpe | 0.54 |` are silently skipped because NAMED_METRIC_RE requires `:` or `=` between label and value. This is exactly the shape of our advisor result tables, where the 2026-04-21-c T_SPA error would also plausibly live."
    evidence: "verify_docs_provenance.py:37 — `\\s*[:=]\\s*` separator class. Tested: '| IC | 0.046 |' returns None."
    suggested_fix: "Document the limitation; MVP-appropriate heuristic (≥3 numeric cells in a row → require citation within ±10 lines of table header) is deferred."
    status: DEFERRED-DOCUMENTED
    resolution_notes: "Not fixed in code; documented in .claude/rules/docs.md §4 under 'Known limitations of the verifier'. Rule 9 reviewers of advisor docs must manually verify table provenance. This is explicitly framed as 'necessary but not sufficient': verifier pass + manual table review is the full gate. Revisit if table false-negatives occur in practice."
  - id: FINGNN-A-05
    severity: CONCERN
    category: correctness
    claim: "Non-capturing 'Rank IC' and 'Mean IC' prefixes match the IC token but the captured substring drops the qualifier, so violation messages are less actionable."
    evidence: "verify_docs_provenance.py:58-67 — find_numeric_claims returns m.group(0)."
    suggested_fix: "Widen NAMED_METRIC_RE to include optional `(Rank|Mean|Median)\\s+` prefix."
    status: REJECTED
    resolution_notes: "Cosmetic finding (violation-message readability only; does not affect flag correctness or exit code). Per CLAUDE.md Rule 9 主线聚焦: 不在风格上纠缠. Rejected."
summary:
  critical: 0
  major: 2
  concern: 3
  fixed_before_reply: 3
overall_verdict: PROCEED-WITH-FIXES
---

# Review body

The script is small, well-structured, and correctly scoped to its stated purpose. Exit-code contract (0 / 1 / 2) matches CLI convention; dataclass `Violation` is appropriately frozen; `iter_non_code_lines` correctly handles single-frontmatter blocks and does not confuse later `---` dividers with frontmatter closers (verified).

## Disposition after fixes (written 2026-04-23 by Claude)

- **FINGNN-A-01 (MAJOR)**: FIXED. Dropped bare `[tFZ]` alternation. Verified by smoke test that prose mentions no longer trigger false positives; T_SPA still triggers as expected.
- **FINGNN-A-02 (MAJOR)**: FIXED. Removed `.py` / `.ipynb` from CITATION_PATH_RE AND from CITATION_PER_RE (went further than reviewer suggested — see resolution_notes). Citations must be to data artifacts only. Verified.
- **FINGNN-A-03 (CONCERN)**: FIXED. Relaxed CODE_FENCE_RE per CommonMark.
- **FINGNN-A-04 (CONCERN)**: DEFERRED-DOCUMENTED. Not a code fix; documented the table blind-spot explicitly in `.claude/rules/docs.md` §4 under "Known limitations of the verifier," framing the verifier as necessary-but-not-sufficient when advisor docs contain tables. Manual table-provenance review by the Rule 9 human reviewer remains required.
- **FINGNN-A-05 (CONCERN)**: REJECTED. Cosmetic (violation-message readability only). Per Rule 9 主线聚焦 clause.

## Fallback attestation

Codex CLI returned a rate-limit error (empty content → Rule 9 Fallback trigger per CLAUDE.md). `finance-gnn-reviewer` executed this touchpoint with equal review-weight per Rule 9 Fallback clause.

## Verdict after fixes

PASS-WITH-CONCERNS (originally PROCEED-WITH-FIXES; both MAJORs now FIXED, 1 CONCERN documented, 1 CONCERN deferred, 1 rejected per correctness-focus clause). Script is ready for use under the documented limitations.
