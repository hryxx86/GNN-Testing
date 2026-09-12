---
handoff_date: 2026-04-20
schema_version: 1  # per .claude/rules/docs.md §5 (added retroactively 2026-04-22)
last_completed: "2026-04-20-a: Phase 5 Step 3 Part A/B/C complete; Alpha158 S8 baseline trained; Module 4 Hansen SPA + BH-FDR analysis done"
in_flight:
  - id: narrative-path-decision
    file: docs/analysis.md
    status: "awaiting H博士 decision between Path A (rebuild Alpha158 per-fold winsorization, ~2.5h) vs Path B (accept current S6 non-superiority narrative)"
    blockers: ["H博士 decision on Path A vs Path B"]
  - id: fold4-anomaly-diagnosis
    file: build_alpha158_features.py
    status: "S8 Fold-4 IC +0.220 outlier — winsorization leakage artifact vs real Q2-2025 regime, undetermined"
    blockers: ["decided in Path A branch; deferred in Path B branch"]
open_questions:
  - "Path A vs Path B? (determines scope of the paper narrative)"
  - "Fold-4 is leakage artifact or real regime? (only answerable under Path A)"
  - "If Path B: run reverse SPA (S8 as candidate, S6 as benchmark) to strengthen the non-superiority framing, and/or TOST for equivalence claim?"
file_state:
  new_this_session:
    - build_alpha158_features.py
    - run_step3_plan_z_part_c.py
    - data/reference/sp500_5y_alpha158_features.npy
    - data/reference/sp500_5y_alpha158_features_meta.json
    - experiments/step3_plan_z/part_c_s8_daily_ic.csv
    - artifacts/step3_plan_z/part_c_meta.json
  modified_this_session:
    - analyze_step3_plan_z.py  # Hansen SPA + BH-FDR logic
    - docs/analysis.md
rule9_status:
  touchpoint_1_plan: PASSED  # Rounds 1-3 agents ad372bb181, a0bf2209f8, a80e980969; Plan Z++ consensus reached
  touchpoint_2_code: PASSED  # Rounds 4-6 on Modules 1/2/3: 6 CRITICAL + 1 MAJOR total, all fixed
  touchpoint_3_results: PASSED  # Round 7 agent a4c569fc07: 1 CRITICAL + 4 MAJOR addressed; Alpha158 baseline added per recommendation
  closeout_audit: NOT-RUN  # this handoff predates the /session-closeout slash command (2026-04-22)
next_actions:
  - "Ask H博士: Path A vs Path B?"
  - "If Path A: patch build_alpha158_features.py for per-fold winsorization → rerun Part C → rerun Module 4"
  - "If Path B: write paper Section 4 with the S6 parsimony framing; pre-submit run reverse SPA + TOST to strengthen claim"
  - "Post-2026-04-21: errata propagation fixed (T_SPA=0.23 → 1.231; SPA interpretation: non-superiority, not equivalence); re-verify advisor docs before sending"
---

# Session Handoff — 2026-04-20

> 新窗口开始前必读。Phase 5 Step 3 + Part C 完整状态。
>
> **Note (added 2026-04-22)**: frontmatter schema retrofitted per `.claude/rules/docs.md` §5 as template reference. Also note: after this handoff was written, errata were corrected in 2026-04-21-c (T_SPA for SAGE vs S8 was misread — correct value 1.231 not 0.23; SPA "equivalence" phrasing was wrong — correct framing is "non-superiority"). The tables below have been updated to the corrected values.

---

## 当前状态：所有训练完成，等待 narrative 决策

- ✅ Part A (30 runs, 52 min): permutation ranking, mom12m 第 1
- ✅ Part B (210 runs, ~4h): 7 subsets × 2 models × 5 folds × 3 seeds
- ✅ Part C (30 runs, 51 min): S8 Alpha158 158-feat 外部 baseline
- ✅ Module 4 analysis: 完整 Hansen SPA + BH-FDR + 图表

**无运行中进程**。

---

## 关键结果（详见 progress `2026-04-20-a`）

### 主要 subset IC + NW t-stat

| Subset | # feat | MLP IC (p) | SAGE IC (p) |
|--------|--------|-----------|-------------|
| S1 full (Plan Z 10) | 10 | +0.023 (0.20) | +0.016 (0.46) |
| **S6 PC probe** | **3** | **+0.046 (0.009)** ✅ | **+0.047 (0.014)** ✅ |
| S7 wf5 9-dim | 9 | -0.006 (0.68) | **-0.048 (0.036)** ⚠️ |
| **S8 Alpha158** | **158** | **+0.041 (0.026)** ✅ | **+0.042 (0.025)** ✅ |

### Hansen SPA (primary p_consistent)

| Model vs Benchmark | T_SPA | p_c | 结论 |
|------|-------|-----|------|
| MLP vs S1 | 3.24 | 0.053 | 边际 |
| MLP vs S7 | 3.24 | 0.038 | ✅ |
| **MLP vs S8** | 0.27 | **0.551** | **不拒绝** |
| SAGE vs S1 | 2.69 | 0.117 | ❌ |
| SAGE vs S7 | 4.82 | 0.006 | ✅ |
| **SAGE vs S8** | 1.23 | **0.551** | **不拒绝** (corrected 2026-04-21-c; prior 0.23/0.590 was a misread of the S6-pair t-stat 0.225, not the benchmark T_SPA) |

**关键**：S8 加入 SPA 后 S6 无法显著 beat S8 — 原 narrative 需修订。

---

## 🔴 待决策：S8 Fold 4 异常

```
         S6 MLP   S6 SAGE    S8 MLP    S8 SAGE
Fold 0   +0.027   +0.022    -0.008    +0.034
Fold 1   +0.027   +0.031    -0.020    -0.041
Fold 2   +0.102   +0.111    +0.026    +0.024
Fold 3   -0.031   -0.030    -0.019    -0.021
Fold 4   +0.101   +0.096    +0.226    +0.214  ← S8 异常
```

S8 去 Fold 4 IC ≈ 0；S6 去 Fold 4 IC ≈ +0.03。

**解释待定**：
- A. Winsorization leakage artifact（`build_alpha158_features.py` 用全样本 1/99 percentile，含 test 数据设阈值）
- B. 真实 Q2-2025 regime 效应

---

## 两条 paper path

### Path A：验 leakage（~2.5h）

1. 修 `build_alpha158_features.py` 改 per-fold winsorization（每 fold train-only p1/p99）
2. 重跑 Part C (~1h)
3. 重跑 analyze (~1min)
4. 若 S8 IC 降到 ≈0 → "S6 PC-probe beats Alpha158 library" ✅ narrative 站
5. 若 S8 IC 仍 ≈0.04 → 证实 B (真效应)

### Path B：接受现状（~0h）

按 "S6 parsimony 优势" 写：
- S6 3 feat 在 Hansen SPA (S8 为 benchmark) 下未以 α=0.05 显著**胜过** S8 158 feat（one-sided non-superiority 方向 candidate > benchmark；反向 SPA 和 TOST 都未跑）
- S6 trains 3× faster, 50× fewer inputs, interpretable
- Paper contribution (submission 前 pending 反向 SPA + TOST): "under Hansen SPA with Alpha158 as benchmark, a compact economically-grounded feature set does not demonstrate statistically superior rank-IC at α = 0.05, at a fraction of the complexity"
- 较弱但诚实，risks rebuttal: "为什么选 3 feat 不是更多?" + "只证 non-superiority, 未证 equivalence"

---

## 项目文件

### 本轮新增
- `build_alpha158_features.py` — qlib Alpha158DL faithful reproduction
- `run_step3_plan_z_part_c.py` — S8 training runner
- `data/reference/sp500_5y_alpha158_features.npy` — 1255×501×158 float32, ~400MB
- `data/reference/sp500_5y_alpha158_features_meta.json` + `sp500_5y_alpha158_qa.csv`
- `experiments/step3_plan_z/part_c_s8_daily_ic.csv`
- `artifacts/step3_plan_z/part_c_meta.json`

### S8 gate 3 层完整性 check（在 `analyze_step3_plan_z.py`）
1. `part_c_meta.json` 存在
2. Triple set identity (30 (model,fold,seed) exact match fold_manifest)
3. day_idx set identity (matches fold_manifest test_days, 无重复 无漂移)

---

## Codex 讨论历史（Rule 9 all 3 touchpoints）

| Round | Agent ID | 范围 | 关键输出 |
|-------|----------|------|---------|
| 1-3 | ad372bb181, a0bf2209f8, a80e980969 | Plan 设计 | Plan Z++ 共识 |
| 4 | a886429f68 | Module 1 subsets | 2 CRITICAL 修 |
| 5 | ae897eb628 | Module 2 Part A | 2 CRITICAL + 1 MAJOR 修 + SHA-256 RNG |
| 6 | a49cf14a80 | Module 3 Part B | 2 CRITICAL 修（列顺序, preflight） |
| 7 | a4c569fc07 | **Results** | 1 CRITICAL + 4 MAJOR; 推荐 Alpha158 baseline |
| 8 (implied) | — | Alpha158 设计 | Level I 选定 |

---

## 下一个 conversation 建议开头

读 `progress.md 2026-04-20-a` + 本文件，然后问 H博士：

> "H博士，Phase 5 Step 3 完整跑完含 Alpha158 S8。关键发现 S6 在 Hansen SPA 下未以 α=0.05 显著胜过 S8 (non-superiority; 不等同于 equivalence — 严格主张需 TOST)，Fold 4 异常待判定。**Path A (重 build 验 leakage, ~2.5h) 还是 Path B (接受现状写论文)?**"

---

*Written 2026-04-20 by Claude for session handoff*
