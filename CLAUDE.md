# CLAUDE.md — GNN-Testing Project Rules

> This file is loaded every session. All rules are mandatory.

---

## Rule 1: Addressing

Every response MUST start with **"H博士，"** then the actual answer.

---

## Rule 2: No Guessing, No Substitutes

- **Unknown things or new plans**: MUST ask H博士 for details first. Never make assumptions or decisions autonomously.
- **Code implementation**: MUST NOT use substitute libraries, workarounds, or placeholder code to "get it working." If the intended approach doesn't work, stop and ask.
- **Architecture changes**: Propose and explain impact BEFORE making changes. H博士 must approve.

---

## Rule 3: Communication

- **Language**: Chinese for conversation, English for code and technical reports.
- **Transparency**: Explain the impact of any change BEFORE executing it.
- **Goal-oriented**: Experimental results must be good — outcomes over process.

---

## Rule 4: Session Start Protocol

Every session, BEFORE doing anything else:

1. Read `progress.md` — understand what was done
2. Read `plan.md` — understand what's next
3. Read `docs/analysis.md` — understand current findings
4. Orient and summarize current state to H博士

---

## Rule 5: Tri-Doc System (MANDATORY)

Three documents must stay **time-aligned**. Each uses the same date-based entry IDs.

| File | Purpose | Update When |
|------|---------|-------------|
| `progress.md` | 做了什么 (past) | After every completed task |
| `plan.md` | 接下来做什么 (future) | Start + end of each session |
| `docs/analysis.md` | 分析发现 (findings) | After every analysis/experiment |

### Entry Format

```
## YYYY-MM-DD-x: Short Title
```

Where `x` = a, b, c... for multiple entries on the same day.

### Cross-References

Each entry includes a reference line pointing to the corresponding entries:

```
→ progress: `2026-02-27-b` | plan: `2026-02-27-b` | analysis: `2026-02-27-b`
```

If no corresponding entry exists, use `N/A`.

### Update Rules

1. **Start of session**: Read all three docs to orient
2. **After completing a task**: Add entry to `progress.md`, mark done in `plan.md`
3. **After running analysis**: Add entry to `docs/analysis.md` with results + observations
4. **End of session**: Update `plan.md` with next steps
5. **All three docs must share the same date-based entry IDs**

### plan.md Decision Log

`plan.md` includes a Decision Log table at the bottom:

```
| Date | Decision | Rationale |
```

Record every non-trivial decision (model choice, architecture change, data processing choice).

---

## Rule 6: Unimplemented Plans — Discuss Before Acting

The following documents describe **planned but unimplemented** work. They require discussion with H博士 before any implementation:

| File | Content | Status |
|------|---------|--------|
| `phase_f_design.md` | Phase F volatility-calibrated SelectiveNet design | Needs discussion (Phase 3) |

**If any of these plans are feasible and approved**, integrate the details into `plan.md` with proper entries.

### Archived Documents

Superseded documents are moved to `archived/`. These are kept for reference only — their useful content has already been merged into `plan.md`.

| File | Original Content | Archived On |
|------|-----------------|-------------|
| `archived/plan_v2.md` | DynHetGNN-SP full research plan v2 | 2026-03-03 |
| `archived/phase_d_design.md` | Phase D+ diagnostics notebook design | 2026-03-03 |

**Rule**: When a planning/design document is superseded, merge its useful parts into `plan.md`, then move the original to `archived/`.

---

## Rule 7: Key Paths

- Google Drive folder: `GNN测试` (NOT `GNN-Testing`)
- Local project: `/Users/heruixi/Desktop/GNN-Testing`
- Conda Python: `/Users/heruixi/anaconda3/bin/python`

---

## Rule 8: Technical Conventions

- **NotebookEdit gotcha**: `insert` mode places new cell AFTER the specified `cell_id`. To put a cell at the beginning, merge into Cell 0.
- **Notebooks**: Run on Google Colab (A100 80GB for GNN training).
- **Data leakage**: All market context features use strictly T-1 close values. No exceptions.
- **Return definition**: Next-day close-to-close, locked and cannot be changed.

---

## Rule 9: Current Project State (as of 2026-02-27)

- Phase 1, 2 Pilot, A, B: DONE
- Phase C v1: DONE (all AUC ~ 0.50)
- Diagnostic cells D.1 + D.2: Written, PENDING Colab run
- Phase 1 (signal fix): Code written, pending Colab run
- Phase 2 (LLM validation), 3 (selective prediction), 4 (paper): Planned in `plan.md`

### Key Technical Decisions Already Made

| Decision | Rationale |
|----------|-----------|
| FinBERT (not Fin-E5/voyage) | Clean ablation; upgrade deferred |
| Full-batch training on A100 80GB | Avoids torch-sparse |
| Static correlation graph (prices up to 2024-12) | No future leakage |
| Diagnostics before model changes | AUC ~ 0.50 → find signal first |
