# Neat-Freak Skill Integration Plan

**Date**: 2026-04-27
**Version**: v3 (final — H博士 directive 2026-04-27: delete skill instead of suppress; do not mirror LLM-Finance-Benchmark)
**Author**: Claude (Opus 4.7)
**Touchpoint**: Rule 9 §1 (Plan review pre-execution)
**Target reviewer**: Codex (Round A complete — see `artifacts/reviews/2026-04-27_codex_plan_A.md`)
**Fallback**: finance-gnn-reviewer (not invoked; Codex returned within 5 min)
**Round B re-review**: not invoked (v3 strictly simplifies v2 — removes risk surface CODEX-A-04 cared about, no new mechanism added)

## Changelog from v1 → v2 (Codex Round A fixes)

- **CODEX-A-01 fix** (Phase A-0 added): amend `docs.md` §1 N/A clause to point at §7
- **CODEX-A-02 fix** (Phase A-3 rewritten): Agent 4 scope restricted from "full repo doc tree" to git-diff modified files + §7-implied dependencies + tail-N + cheap full-tree grep
- **CODEX-A-03 fix** (Phase B-2 rewritten): Rule 11 → **Rule 6.5**, placed between Rule 6 (archived) and Rule 7 (key paths); CLAUDE.md header line 3 updated to enumerate Rule 6.5
- **CODEX-A-04 fix** (Phase C reordered): /tmp scratch verification BEFORE live repo; live repo verification gated on git-clean preflight + abort-and-restore protocol

## Changelog from v2 → v3 (H博士 directive 2026-04-27)

H博士 reviewed v2 + Codex review and elected to **delete the user-level skill** instead of suppressing it. This collapses the entire Phase B mechanism (deny + Rule 6.5 cognitive layer) into a single `rm -rf` and removes the live-test risk surface CODEX-A-04 flagged.

- **Phase B replaced**: instead of `permissions.deny: ["Skill(neat-freak)"]` + CLAUDE.md Rule 6.5 + LLM-Finance-Benchmark mirror, Phase B now simply deletes `~/.claude/skills/neat-freak/`. Skill becomes globally inaccessible (not just suppressed in this project).
- **Phase B-3 (LLM-Finance-Benchmark mirror) cancelled**: H博士 confirmed not mirroring after reviewing LLM-Finance-Benchmark/CLAUDE.md. Reasons: that project uses Tri-Doc + Rule 5.5 wide-scope README (different from GNN's Quad-Doc + narrow-scope), no `.claude/rules/` infrastructure (would require expanding the 11 KB CLAUDE.md), Phase = "Project initialization" so doc-drift risk is currently low, and after skill deletion no suppression is needed anyway.
- **Phase C simplified**: C-1 fixture test, C-2 live-repo /sync test, C-3 LLM verification, C-4 negative control all REMOVED — they only existed to verify suppression worked, which is moot after delete. Only C-5 (`/session-closeout` Agent 4 dry-test) and C-6 (relative-time grep baseline) retained.
- **Phase A unchanged**: A-0 through A-3 still execute as v2 (the 4 absorbed ideas remain valuable independent of skill suppression).
- **CODEX-A-03 disposition**: the Rule 6.5 placement fix becomes moot because Rule 6.5 is no longer added. CODEX-A-03 finding stays FIXED-and-superseded — its concern (placement of a permanent skill-suppression rule) doesn't apply once the rule itself is dropped.
- **CODEX-A-01, A-02, A-04 dispositions**: A-01 + A-02 still apply (Phase A unchanged). A-04 fully resolved by deletion (no live test of suppression needed).

---

## 1. Context

### What prompted this

H博士 installed the `neat-freak` skill (KKKKhazix/khazix-skills) at user level (`~/.claude/skills/neat-freak/`). The skill performs end-of-session knowledge cleanup: enumerate all docs/memory, identify drift between code and documentation, and **actively edit/delete** to reconcile.

Reviewing the SKILL.md (`~/.claude/skills/neat-freak/SKILL.md`) and references (`agent-paths.md`, `sync-matrix.md`) revealed a **design conflict**:

- neat-freak L87: "**删除优于保留**：完成的临时计划、推翻的决策、过期的上下文，删掉"
- neat-freak L88: "**合并优于追加**：新信息是对旧信息的更新，改旧条目，不要再加一条"
- neat-freak references/sync-matrix.md: "已完成的待办 → 删除——知识库不是历史档案"

These collide head-on with this project's:

- **Quad-Doc system** (`progress.md` is *intentionally* an append-only log of dated entries `## YYYY-MM-DD-x`).
- **`plan.md` Decision Log** (intentionally retains overruled decisions).
- **`archived/` read-only convention** (Rule 6).
- **`MEMORY.md` schema** (4 types with `**Why:** / **How to apply:**` blocks; neat-freak's `agent-paths.md` only knows `name/description/type`).

If neat-freak is allowed to fire on `/sync`, `/neat`, "整理文档", "同步一下", "梳理一下", "收尾", or "这个阶段做完了" inside this project, it will silently delete `progress.md` historical entries, flatten MEMORY.md schema, and touch `archived/` — destroying load-bearing history.

### What neat-freak gets right that we lack

After full read of `SKILL.md` + `references/`, four ideas are **genuinely better than what we have today**:

1. **Change-impact matrix** (`references/sync-matrix.md`): explicit "code change type → which doc files MUST update" lookup table. Our `.claude/rules/docs.md` currently lacks this — Rule 5 only describes update *triggers* per doc class (`progress.md` updates after every task, etc.) but doesn't enumerate which docs must update *together* when X happens.

2. **Forced inventory**: SKILL.md §First Step demands `ls` + `find -name "*.md"` + per-file labeling ("evaluated / needs change / no change") before any judgment. Our `/session-closeout` audits *code modifications* but doesn't audit doc drift.

3. **Self-check grep**: `grep -E "今天|昨天|刚刚|最近|recently|today|yesterday"` to zero out relative time references. We have *zero* automated drift checks for docs.

4. **Three-audiences principle** (SKILL.md §Key Concepts): explicit separation
   - agent memory ↔ self across sessions
   - `CLAUDE.md` ↔ AI in this project
   - `docs/` + `README.md` ↔ humans / future AI / external readers

   Our project follows this implicitly but never wrote it down → risk of drift over time (e.g. CLAUDE.md absorbing user-facing tutorials).

### Why "absorb + disable" instead of "use as-is" or "ban entirely"

- **Use as-is** = data loss in `progress.md` / Decision Log / `MEMORY.md` / `archived/`. Unacceptable.
- **Ban entirely** = miss the 4 genuine improvements above. Wasteful.
- **Absorb the 4 ideas into our existing rule system + disable the user-level skill in this project** = best of both worlds and matches Rule 2 (no substitutes — adapt to our exact constraints).

---

## 2. Scope

This plan covers **two projects**, both `/Users/heruixi/Desktop/GNN-Testing` and `/Users/heruixi/Desktop/LLM-Finance-Benchmark`. Both use the same Quad-Doc style (LLM-Finance-Benchmark has `progress.md` 129 KB, `plan.md` 53 KB, `CLAUDE.md` 11 KB — confirmed similar architecture, though detailed compatibility check is deferred to step P-2 below).

The plan does **not** modify `~/.claude/skills/neat-freak/` (user-level skill kept intact for other projects).

---

## 3. Plan Steps

### Phase A — Absorb the 4 ideas into `.claude/rules/`

#### A-0. Amend `.claude/rules/docs.md` §1 N/A clause (per CODEX-A-01)

**Why**: §7 (added in A-1) tightens the N/A rule for change types listed in the Sync Matrix. §1 currently says "`N/A` is allowed when the doc type doesn't apply" — too broad once §7 exists. Without an explicit pointer, AI may use §1's loose N/A to bypass §7's mandatory couplings.

**Edit**: replace existing line 31 of `.claude/rules/docs.md`:

```markdown
`N/A` is allowed when the doc type doesn't apply (e.g. a progress entry with no new analysis).
```

with:

```markdown
`N/A` is allowed when the doc type doesn't apply (e.g. a progress entry with no new analysis).
**Exception**: if §7 Sync Matrix lists a coupling that includes the change type at hand, `N/A` is **not allowed** for any of the files §7 lists for that row — they all must update together. The tri-doc cross-reference line then names the actual entry IDs (or, if the file is a README, the date of the README update).
```

This must be done **before** A-1 lands so §7 already has the pointer back from §1.

#### A-1. Add `.claude/rules/docs.md` §7: Sync Matrix (project-specific)

Append a new section to `/Users/heruixi/Desktop/GNN-Testing/.claude/rules/docs.md` after current §6.

Draft content:

```markdown
## 7. Change-Impact Sync Matrix (project-specific)

**Problem this solves**: knowing which doc trigger applies per Quad-Doc table (§1) is necessary but not sufficient — a single code change can require coordinated updates across 2–4 files. Without an explicit lookup table, the "tri-doc cross-reference" rule (§1) gets satisfied with `N/A` even when a real coupling exists. This matrix lists the cross-doc couplings that have actually mattered in this project's history.

| Change type | Files that MUST update together |
|---|---|
| New experiment script (`run_*.py`, `analyze_*.py`, `build_*.py`) | `progress.md` (entry per script) + `<folder>/README.md` (if folder gains structure) + `plan.md` (if it executes a planned phase, mark phase status) |
| Experiment results produced (IC / Sharpe / stat tests in `experiments/<dir>/`) | `progress.md` (entry) + `docs/analysis.md` (findings) + Codex Results Review file in `artifacts/reviews/` (Rule 9 Touchpoint 3) + `experiments/<dir>/README.md` if new |
| Loss / feature / architecture decision (model code change with design intent) | `progress.md` (what changed) + `plan.md` Decision Log (rationale row) + Codex Code Review file in `artifacts/reviews/` (Rule 9 Touchpoint 2) |
| New feature module added to `experiments/` or model dir | `progress.md` + folder `README.md` + (if it's a paper-figure-producing change) `docs/analysis.md` |
| New / renamed dataset path | All `*.py` config blocks + `README.md` references + `plan.md` if data plan changes |
| Phase milestone (Phase 5 → next phase, etc.) | `plan.md` (decision in Decision Log + new phase entry) + `progress.md` (milestone entry) + new `docs/session_handoff_<date>.md` per §5 manifest |
| Numeric advisor doc (`docs/advisor_*`, `docs/REPORT*`, `docs/project_findings_*`) | Per §4: every numeric claim cited inline; `python scripts/verify_docs_provenance.py` must pass before send-to-H博士 |
| Folder structure change (new subdir, mass archival, folder repurpose) | The folder's `README.md` per §2 + `progress.md` (the structural change itself) + (only if the parent folder's `README.md` lists subdirs) parent README |
| Rule changes (`.claude/rules/*.md` or `CLAUDE.md` itself) | The rule file + `progress.md` entry + (if external behavior visible) `plan.md` Decision Log row |

**How to use**: when a change spans more than one row, all listed files for all matched rows must update together. Cross-reference with §1 tri-doc line — `N/A` is allowed *only* when this matrix doesn't list a coupling for the change type.

**Not covered by this matrix** (intentional gaps): one-off ad-hoc files (per §2 README narrow scope); `archived/` (read-only per Rule 6); `~/.claude/CLAUDE.md` (per CLAUDE.md Rule 2 — global config never touched without explicit cross-project rationale).
```

#### A-2. Add `.claude/rules/docs.md` §8: Three Audiences

Append after §7. Draft content:

```markdown
## 8. Three Audiences (knowledge layer separation)

**Problem this solves**: over time, `CLAUDE.md` tends to absorb tutorial content, `docs/` tends to absorb reminders-to-self, and `MEMORY.md` tends to absorb project state. The result is three half-redundant copies that drift from each other. This rule fixes the audience for each layer.

| Layer | Path | Audience | Owns | Does NOT own |
|---|---|---|---|---|
| Agent memory | `~/.claude/projects/-Users-heruixi-Desktop-GNN-Testing/memory/*.md` (4-type schema: user / feedback / project / reference; `Why:` + `How to apply:` for feedback / project) | Future Claude (this project's auto memory) | Personal preferences, non-obvious project facts, cross-project references, locked thresholds | Anything externally visible; tutorial content; verbatim docs |
| Project AI rules | `CLAUDE.md` + `.claude/rules/*.md` + `.claude/commands/*.md` | AI working in this project (any model, any session) | Conventions, red lines, env vars, slash command protocols, path-scoped rules | "I remember last time…" (memory); how-to-use tutorials (docs); quoted external API specs (docs) |
| Human / external | `docs/` + `README.md` (project root + per-folder) | H博士 / future advisors / paper reviewers / external readers | Architecture, findings (`docs/analysis.md`), reports, READMEs as folder indices | Reminders-to-self ("we should check X next time"); rule-flavored "must" / "MUST NOT" (those go in `.claude/rules/`) |

**Operational consequence**: the same fact, when written in two layers, must be **rephrased to the audience**, not copy-pasted. If "we lock next-day c-t-c return" appears in `CLAUDE.md`, the corresponding `docs/analysis.md` mention reads "Returns are next-day close-to-close (rationale: …)" — narrative for the human, not a rule citation.

**Anti-pattern**: a `MEMORY.md` entry that says "see CLAUDE.md Rule 8" is fine. A `docs/analysis.md` paragraph that says "see CLAUDE.md Rule 8" is **not** fine — analysis docs are for external readers who don't read internal rule files. Inline the substance.
```

#### A-3. Extend `/session-closeout` with Agent 4: Doc Drift Audit (scope-restricted per CODEX-A-02)

**Sustainability constraint** (Codex Round A finding): full doc-tree scope is unsustainable. Verified file sizes 2026-04-27:

| File | Lines | Bytes |
|---|---|---|
| `progress.md` | 2,582 | 168 KB |
| `plan.md` | 1,623 | 92 KB |
| `docs/` total | — | 348 KB |

A naive "read every doc fully" closeout would push 600+ KB through Agent 4's context per session. Instead, scope is bounded by what *actually changed* this session.

Modify `/Users/heruixi/Desktop/GNN-Testing/.claude/commands/session-closeout.md` step 2 to spawn a 4th agent in parallel:

```markdown
**Agent 4 — Doc Drift Audit** (added per .claude/rules/docs.md §7-§8 absorption from neat-freak skill; scope-restricted per Codex Round A CODEX-A-02)

**Scope rules** (apply IN ORDER, do not exceed):
1. **Modified-files set**: run `git diff --name-only HEAD` and `git status --short`. Read ONLY these files in full.
2. **§7-implied dependencies**: for each modified `run_*.py` / `analyze_*.py` / `build_*.py` / `experiments/<dir>/*` change, look up `.claude/rules/docs.md` §7 row → identify the docs that MUST update together. For each such doc, read the **last 100 lines** (`tail -100`) — sufficient to verify the latest entry got added.
3. **Cross-reference target check**: for any new entry ID referenced in #2's tail-100 reads (e.g. `→ progress: 2026-04-27-x`), verify the cited entry exists. Use `grep -n "## 2026-04-27-x" progress.md` — single grep, not full read.
4. **Relative-time grep is cheap and stays full-tree**: `grep -nE "今天|昨天|刚刚|最近|上周|today|yesterday|recently|last week" progress.md plan.md docs/*.md` is a single-pass grep, fine to run on full tree (~600 KB grep is sub-second).
5. **Audience leak check (§8)**: scoped to modified-files only (#1). Don't scan unchanged docs.
6. **README narrow-scope (§2) check**: only fires if git diff shows new subdirs / mass archival / file renames / folder repurpose. Else skip.

**Findings categories**:
- CRITICAL = §7 matrix coupling violated (e.g. modified `run_*.py` but no progress.md entry)
- MAJOR = relative-time leak; tri-doc cross-ref missing or pointing at non-existent ID
- CONCERN = §8 audience leak; §2 README missed for structural change

Report per `.claude/rules/docs.md` §6.

**What Agent 4 deliberately does NOT do**: read all of progress.md / plan.md / docs/ fully; audit unchanged docs; flag style/wording issues. Per session-closeout.md original rule "Do not audit unchanged files — waste of effort."
```

Step 2 of session-closeout.md becomes "spawn 4 agents IN PARALLEL (single message, 4 tool calls)". Step 3 aggregation table gains a "Doc Drift" row.

### Phase B (v3) — Delete the user-level skill

**v3 simplification**: H博士 elected to delete the skill outright (2026-04-27 directive) instead of suppressing it via `permissions.deny` + Rule 6.5 cognitive layer. This collapses Phase B-1, B-2, B-3 into a single `rm -rf` and removes the live-test risk surface CODEX-A-04 cared about.

#### B (v3). Delete `~/.claude/skills/neat-freak/`

```bash
rm -rf ~/.claude/skills/neat-freak
ls -la ~/.claude/skills/  # verify empty (other skills, if any, remain)
```

After deletion: skill is globally inaccessible (not just suppressed in this project). All trigger phrases (`/sync`, `/neat`, `整理文档`, `同步一下`, `梳理一下`, etc.) become no-ops because Claude Code cannot load a skill that does not exist on disk.

**Implication for `.claude/settings.local.json`**: no `permissions.deny` entry needed. Original GNN-Testing `settings.local.json` (allow-list only) stays unchanged.

**Implication for `CLAUDE.md`**: no Rule 6.5 needed. Project's existing rule structure (1-10) stays unchanged.

**Implication for LLM-Finance-Benchmark**: see B' below — H博士 confirmed not mirroring.

---

#### B' (v3). LLM-Finance-Benchmark — NOT mirroring (H博士 directive 2026-04-27)

After reading `/Users/heruixi/Desktop/LLM-Finance-Benchmark/CLAUDE.md`, decision is to NOT mirror Phase A (or any Phase B equivalent). Reasons:

- LLM-Finance-Benchmark uses **Tri-Doc** (Rule 5: progress / plan / analysis) plus a **separate Rule 5.5 wide-scope README convention** (every semantic folder + nested issuer/family subfolder must have a README; updates triggered on any folder content change). This contradicts GNN-Testing's Quad-Doc + narrow-scope-README philosophy. §7 Sync Matrix (designed for Quad-Doc) does not transplant cleanly.
- LLM-Finance-Benchmark has no `.claude/rules/` infrastructure — would require expanding the 11 KB CLAUDE.md by ~30% to add equivalent §7 / §8 content inline.
- LLM-Finance-Benchmark's current Phase 10 = "Project initialization" (12-week summer 2026 timeline). Doc-drift risk is currently low because the project is small and recently bootstrapped.
- Skill is deleted globally, so no neat-freak suppression mechanism is needed in either project. Only the *positive* absorption (§7 / §8 / Agent 4) was at issue, and H博士 elected not to apply it to LLM-Finance-Benchmark.

If LLM-Finance-Benchmark later grows to a state where doc drift becomes a concern, a tailored mirror can be written then — but it needs a separate plan, not a copy of this one.

---

#### B-OBSOLETE (v2 contents kept for audit trail; do not execute)

The following v2 sections were **superseded by v3** and are retained here only for Rule 9 audit traceability:

- ~~B-1. Add `permissions.deny: ["Skill(neat-freak)"]` to GNN-Testing settings.local.json~~ — superseded; skill deleted instead.
- ~~B-2. Insert `CLAUDE.md` Rule 6.5: Skill Compatibility Filter (per CODEX-A-03 placement fix)~~ — superseded; no rule needed when skill is gone.
- ~~B-3. Mirror to LLM-Finance-Benchmark after P-2 compatibility check~~ — cancelled per H博士 directive 2026-04-27.

(Strikethrough indicates superseded content. Audit trail intact: Codex Round A findings A-01/A-02 still applied (Phase A), A-03 became moot (Rule 6.5 not added), A-04 fully resolved (no live-test of suppression needed).)

---

#### B-DELETED-DETAIL (original v2 §B-2 wording, archived for reference)

**Why Rule 6.5, not Rule 11**: Codex Round A flagged that appending after Rule 10 (a dated project-state snapshot) makes a permanent skill-suppression rule visually subordinate to dated context. The rule is fundamentally about doc/archive policy — it belongs adjacent to Rule 5 (Quad-Doc) and Rule 6 (archived). Rule 6.5 sits naturally between Rule 6 (archived/) and Rule 7 (Key Paths).

**Two-part edit**:

**Part 1**: amend `CLAUDE.md` line 3 (header summary):

```markdown
> Loaded every session. Rules 1-4, 6.5, 7, 8-invariants, 9, 10 are universal and inline here.
```

(adds `6.5` to the universal-rule enumeration)

**Part 2**: insert after current Rule 6 ("Unimplemented Plans") and before current Rule 7 ("Key Paths"):

```markdown
## Rule 6.5: Skill Compatibility Filter

User-level skills installed at `~/.claude/skills/` may have document-management semantics incompatible with this project's Quad-Doc append-only model, `archived/` read-only convention, and MEMORY.md schema. The following skill is currently filtered:

### `neat-freak` — DISABLED in this project

Source: `~/.claude/skills/neat-freak/SKILL.md`. Conflict surface:
- SKILL.md L87 "删除优于保留" / L88 "合并优于追加" vs `progress.md` append-only dated entries (`## YYYY-MM-DD-x`)
- SKILL.md L91 "面向读者：5 分钟看完" vs `plan.md` Decision Log retention of overruled decisions
- `references/sync-matrix.md` "已完成的待办 → 删除" vs Quad-Doc historical entries are load-bearing
- `references/agent-paths.md` MEMORY frontmatter schema (`name/description/type` only) vs this project's 4-type schema with mandatory `**Why:** / **How to apply:**` blocks for feedback/project memories
- SKILL.md §Step 3 "用删除命令清理废弃文件" vs Rule 6 `archived/` read-only

**Trigger handling**: if user says `/sync`, `/neat`, `整理文档`, `同步一下`, `梳理一下`, `更新记忆`, `收尾`, `这个阶段做完了`, `新人能直接上手`, `tidy up`, `clean up docs`, `update memory`, or any other neat-freak trigger phrase, **DO NOT invoke the skill**. Instead:

1. Reply: "本项目禁用 neat-freak（CLAUDE.md Rule 6.5）。本项目同步用 `.claude/rules/docs.md` §7 Sync Matrix + §8 Three Audiences + `/session-closeout` Agent 4 Doc Drift Audit。"
2. Offer to run `/session-closeout` if the user wants the equivalent end-of-session check.

**Why both `permissions.deny` (in `.claude/settings.local.json`) AND this rule**: official Claude Code docs are silent on whether `permissions.deny: Skill(name)` blocks explicit user-typed `/neat` invocation vs only auto-invocation (`code.claude.com/docs/en/permissions.md` confirms deny exists; behavior on user-typed invocation undocumented). This rule covers the gap by binding Claude (the executor) at the cognitive layer.

**Adding more skills to the filter**: append a new subsection here naming the skill, the SKILL.md path, the conflict surface, the trigger phrases, and the substitute mechanism within this project.

**Scope**: project-specific. Filtered skills remain active in projects that don't have this rule.
```

#### B-3. Mirror to LLM-Finance-Benchmark

**Pre-step P-2 (BEFORE B-3 implementation)**: read `/Users/heruixi/Desktop/LLM-Finance-Benchmark/CLAUDE.md` end-to-end + spot-check `progress.md` / `plan.md` to confirm:

- It uses the same Quad-Doc append-only style (date-keyed entries `## YYYY-MM-DD-x`, tri-doc cross-references).
- It has an `archived/` directory or equivalent retention convention.
- Its `CLAUDE.md` does not already define a "Rule N" that conflicts with the proposed Rule 11.

If LLM-Finance-Benchmark has materially different conventions, **stop B-3 and ask H博士** rather than blindly mirroring. Per Rule 2.

If P-2 confirms compatibility:
- Create `/Users/heruixi/Desktop/LLM-Finance-Benchmark/.claude/settings.local.json` with `{"permissions": {"deny": ["Skill(neat-freak)"]}}`.
- Append the equivalent Rule 11 to `LLM-Finance-Benchmark/CLAUDE.md`, with the wording adapted to that project's rule numbering and any project-specific path differences.

### Phase C (v3) — Verification (massively simplified after skill deletion)

**v3 simplification**: original C-1 through C-4 all existed to verify that the *suppression* mechanism worked (deny rule + Rule 6.5 cognitive layer). After v3 deletes the skill, suppression is moot — the skill cannot fire because it does not exist on disk. Only C-5 (verify the new `/session-closeout` Agent 4 behaves) and C-6 (baseline relative-time grep) remain.

C-1 ~ C-4 (v2 contents kept below for audit trail) **NOT executed in v3**.

#### C-1. Disposable-fixture verification (NO live data at risk)

```bash
# Create scratch dir mimicking minimal Claude Code project structure.
mkdir -p /tmp/neat-freak-suppress-test/.claude
cat > /tmp/neat-freak-suppress-test/.claude/settings.local.json <<'EOF'
{"permissions": {"deny": ["Skill(neat-freak)"]}}
EOF
cat > /tmp/neat-freak-suppress-test/CLAUDE.md <<'EOF'
# Test Project
## Rule 6.5: Skill Compatibility Filter
neat-freak DISABLED. If user says /sync /neat etc., refuse and reply "本项目禁用 neat-freak (test fixture)".
EOF
echo "# progress.md (test)" > /tmp/neat-freak-suppress-test/progress.md
echo "# plan.md (test)" > /tmp/neat-freak-suppress-test/plan.md
```

Then `cd /tmp/neat-freak-suppress-test` in a new Claude Code session and test:

1. **Auto-invocation gate**: ask Claude something memory-related ("整理一下我们刚才聊的"). Expected: Claude refuses per Rule 6.5; no skill fires; `progress.md` / `plan.md` unmodified.
2. **Explicit slash gate** (the docs-silent gap): type `/neat`. Expected: refuse. If skill fires anyway, document the failure mode and **escalate to H博士 before B-3 mirror**.
3. **Trigger-phrase gate**: type `/sync`, `/neat`, `整理文档`, `收尾`. Expected: all refused.

Pass criteria: all 3 gates produce refusal AND `git diff` (or simple file diff in /tmp) shows zero changes to `progress.md` / `plan.md` / `CLAUDE.md`.

**If any gate fails**: do NOT proceed to C-2. Report failure mode to H博士 and discuss whether to (i) add a hook layer, (ii) escalate to "ban entirely" path, or (iii) use a more aggressive cognitive-layer instruction.

#### C-2. Live-repo verification (gated on C-1 pass + git-clean preflight)

**Preflight (mandatory)**:
```bash
cd /Users/heruixi/Desktop/GNN-Testing
git status --short  # must be clean (no uncommitted changes) OR clean except for the plan v2 itself
git rev-parse HEAD  # record commit hash for restore
```

If `git status` is not clean (excluding the plan file itself), commit pending work or stash before running C-2. Do NOT run C-2 with uncommitted changes.

**Test**:
1. Open new Claude Code session in `GNN-Testing`. Type `/sync`. Expected: refuse per Rule 6.5.
2. Type `整理文档`. Expected: refuse.
3. Immediately after each test prompt: `git status --short` from a separate terminal. If output shows ANY modification to `progress.md`, `plan.md`, `archived/`, `MEMORY.md`, or `docs/*.md`, **abort and restore** with `git checkout -- <files>` or `git reset --hard <preflight-hash>`. Document the failure as a CRITICAL.

#### C-3. Cross-project verification (LLM-Finance-Benchmark)

After P-2 + B-3 done. Same protocol as C-2 (preflight + abort-restore) on `/Users/heruixi/Desktop/LLM-Finance-Benchmark`.

#### C-4. Negative control — confirm scope is project-local

Open new Claude Code session in unrelated dir (e.g. `~/Desktop` or `/tmp/scratch-other`). Type `/sync`. Expected: neat-freak fires normally — confirms the disable is project-scoped, not user-scoped.

#### C-5. `/session-closeout` Agent 4 sanity test

Run `/session-closeout` after some trivial git change (e.g. add a comment to a file). Expected: 4 Explore agents launched in parallel; Agent 4's scope (per CODEX-A-02 fix) reads only the modified file + tail-100 of relevant docs + cheap full-tree grep — not full doc tree. Verify Agent 4's output references only the changed file's required couplings per §7.

#### C-6. Baseline doc-drift grep

`grep -nE "今天|昨天|最近|recently" progress.md plan.md docs/*.md`. Document baseline (legacy matches in archived plan entries OK). Going forward, new entries (post-plan-execution) must produce zero new matches.

---

## 4. Sequencing

H博士 has three options on order:
- (a) Phase A first, then B (absorb before disable)
- (b) Phase B first, then A (disable before absorb)
- (c) A and B together (single PR-style commit)

**Claude's recommendation: (c)**. Rationale: A and B are mutually reinforcing — the Rule 11 wording in B-2 references `.claude/rules/docs.md` §7-§8 written in A-1/A-2. Doing them together avoids a transient state where Rule 11 points to non-existent sections, or §7-§8 exist without the corresponding suppression mechanism.

If (c) is too large a single change, fall back to (a) — never (b) — because (b) leaves a window where the user-level skill is suppressed but the replacement is not yet documented.

---

## 5. Risks and open questions for Codex review

These are the points the reviewer should pressure-test:

1. **Is "absorb + disable" actually better than "ban entirely"?** Specifically, is `.claude/rules/docs.md` §7 Sync Matrix (project-specific) really an *addition* over what `.claude/rules/docs.md` §1 already describes via Quad-Doc update triggers? Or is it redundant?
2. **`permissions.deny: Skill(name)` actually works as documented?** Official docs (`code.claude.com/docs/en/skills.md`) confirm `Skill()` syntax exists, but do not confirm it blocks user-typed slash invocations. Is the cognitive-layer Rule 11 sufficient backup, or is there a real chance the skill fires anyway?
3. **Is Rule 11 placement after Rule 10 correct?** `CLAUDE.md` currently caps at Rule 10 ("Current Project State"). Adding Rule 11 at the end means it loads after the project-state snapshot. Should it be earlier (e.g. Rule 5.5) where doc-related rules cluster?
4. **`/session-closeout` Agent 4 scope**: full repo doc tree may be too broad — large `progress.md` (already 100+ entries in this project) means the doc-drift audit reads megabytes of markdown. Should scope be limited to "files touched this session + any doc that *should* have been touched per §7 sync matrix"?
5. **Tri-doc cross-reference + sync matrix interaction**: §7 says "all listed files for all matched rows must update together". §1 tri-doc reference says `N/A` is allowed. Are these two rules in tension? Does §7 effectively forbid `N/A` in cases the matrix lists a coupling, and if so should §1 be amended to point at §7?
6. **LLM-Finance-Benchmark P-2 check**: Claude has not actually read that project's `CLAUDE.md` yet. Is the P-2 gate sufficient, or should the plan require explicit H博士 confirmation before any LLM-Finance-Benchmark file is touched?
7. **What this plan deliberately does not do**: it does not put a `SKILL.md` shadow in `.claude/skills/neat-freak/` (would not work — official docs confirm user-level overrides project-level for skills); it does not modify the user-level `~/.claude/skills/neat-freak/`; it does not add an `.claude/settings.json` (committed-to-git) variant; it does not add a hook to detect neat-freak triggers. Are any of these omissions wrong?

---

## 6. Files to be modified (concrete list)

### Phase A
- `/Users/heruixi/Desktop/GNN-Testing/.claude/rules/docs.md` — append §7 + §8
- `/Users/heruixi/Desktop/GNN-Testing/.claude/commands/session-closeout.md` — extend step 2 with Agent 4

### Phase B (v3)
- `~/.claude/skills/neat-freak/` — DELETED (`rm -rf`); user-level skill removed globally. NOT a project file edit.
- `/Users/heruixi/Desktop/GNN-Testing/.claude/settings.local.json` — UNCHANGED (no deny entry needed)
- `/Users/heruixi/Desktop/GNN-Testing/CLAUDE.md` — UNCHANGED (no Rule 6.5 added)

### Phase B' (v3, LLM-Finance-Benchmark)
- NOT MIRRORED. No files in `/Users/heruixi/Desktop/LLM-Finance-Benchmark/` modified.

### Phase A doc-trail (Rule 5)
- `/Users/heruixi/Desktop/GNN-Testing/progress.md` — entry `## 2026-04-27-x: Absorbed neat-freak ideas (Sync Matrix §7 + Three Audiences §8 + Doc Drift Audit) + Rule 6.5 skill filter`
- `/Users/heruixi/Desktop/GNN-Testing/plan.md` — Decision Log row: "2026-04-27 | neat-freak: absorb 4 ideas (§7 §8 Agent 4) + Rule 6.5 disable | Conflict with append-only Quad-Doc, but 4 ideas fill real gaps; deny + cognitive-layer dual-lock per Codex Round A CODEX-A-04"

### Codex Round A audit trail
- `artifacts/reviews/2026-04-27_codex_plan_A.md` — full review with 4 findings (all FIXED), Claude verification log, Codex's review body verbatim. Reference from `progress.md` entry per Rule 9.

---

## 7. What this plan does NOT do (v3 update)

- ~~Does not delete `~/.claude/skills/neat-freak/`~~ — **UPDATED in v3**: plan now DOES delete the skill globally per H博士 directive 2026-04-27.
- Does not modify other user-level config (`~/.claude/CLAUDE.md`, `~/.claude/settings.json`).
- Does not add hooks (`.claude/settings.json` `hooks` field).
- Does not pre-emptively rewrite any existing `progress.md` / `plan.md` entries.
- Does not run `/session-closeout` itself as part of this plan execution (closeout fires after implementation, not during).
- Does not modify any file under `/Users/heruixi/Desktop/LLM-Finance-Benchmark/`.
- Does not modify GNN-Testing `CLAUDE.md` or `.claude/settings.local.json` (Phase B v3 removed those edits).
