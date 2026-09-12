# 下一篇论文 Ideation 工作流 — 文件索引（专门管理）

> **用途**：2026-07-26 起"下一篇论文 idea"工作流的**唯一入口**。本工作流跨
> 多个 session 的全部产出文件在此登记；后续新增文件（正式 plan doc、数据闸门
> 报告、Round B review 等）**必须**在本文件补登记。
> **创建**：2026-07-28，应 H博士 指令（"这个聊天里的所有文件，都写在一个新的
> 文档，专门管理"）。
> **当前状态**：PENDING H博士 总决策（见 §4）。

---

## 1. 本工作流新建的文件（3 个）

| # | 文件 | 角色 | 内容摘要 | 状态 |
|---|------|------|---------|------|
| 1 | `docs/lit_scan_2026-07-26.md` | 文献综述 + idea 池 | 近半年（2026-01→07）金融 ML 六切面扫描（6 并行 agent，~70 篇，arXiv 逐篇核验）：§1 六切面综述、§2 跨切面模式、§3 空白清单 G1–G9、§4 第一批候选 A–F、§4b 第二批候选 G–N、§5 多轮自辩记录、§6 状态 | 完成，只读参考 |
| 2 | `docs/idea_expansions_2026-07-28.md` | A/J/K/M 完整设计稿 | 四个被 H博士 点名候选的设计草案（RQ/文献锚/设计轴/统计机器/结局分支/算力/venue/风险）+ 论文弧线（paper 1 → M → J → A → K）+ 排期建议 + **Touchpoint-1 Round A 修正 §Rev-1..11**（已并入 Codex 全部修正） | 完成，随 Round A 修正更新；正式 plan doc 的直接母本 |
| 3 | `artifacts/reviews/2026-07-28_codex_plan_A.md` | Rule 9 Touchpoint 1 Round A 评审存档（gpt-5.5 xhigh） | Codex 评审：3 CRITICAL + 7 MAJOR + 2 CONCERN，verdict BLOCK-EXECUTION；YAML frontmatter 含逐条 status/resolution_notes（11 FIXED + 1 ACCEPTED-AS-CONCERN，0 REJECTED）+ Codex 原文 + Claude 处置总结 | 完成，已被 Round B diff |
| 3b | `artifacts/reviews/2026-07-28_codex_plan_B.md` | Round B 评审存档（**gpt-5.6-sol** xhigh，模型切换后首评） | 0 CRITICAL + 8 MAJOR + 1 CONCERN，**PROCEED-WITH-FIXES**；round_a_diff（8 FIXED + 4 PARTIALLY-FIXED）+ 新发现 CODEX-B-01..09 全处置（8 FIXED via §Rev-12..20 + 1 as-concern） | 完成；正式 plan doc 以 §Rev-1..20 为约束，偏离才 Round C |

## 2. 本工作流修改的既有文件（4 个）

| # | 文件 | 修改条目 | 内容 |
|---|------|---------|------|
| 4 | `progress.md` | `2026-07-26-a` | 文献扫描 + 多轮自辩完成记录 |
| | | `2026-07-28-a` | A/J/K/M 设计展开记录 |
| | | `2026-07-28-b` | Codex Touchpoint 1 Round A 记录（3C+7M+2Cn，处置结果） |
| 5 | `docs/analysis.md` | `2026-07-26-a` | 扫描核心发现（7 条）+ idea 候选与自辩裁决 |
| 6 | `plan.md` | `2026-07-26-a` | 候选 A/B 决策项挂起 + 第二批 G–N 增补（第 4 条）+ 2026-07-28 设计稿指针增补 |
| 7 | `docs/README.md` | 2026-07-28 行 | 本索引 + lit_scan + idea_expansions 挂载到 docs/ 索引 |

## 3. 文件阅读顺序（新 session 快速恢复）

1. **本文件**（工作流全景 + 当前状态）
2. `docs/idea_expansions_2026-07-28.md`（四候选设计 + §Rev 修正 = 最新技术状态）
3. `artifacts/reviews/2026-07-28_codex_plan_A.md`（评审细节，写正式 plan / Round B 时必读）
4. `docs/lit_scan_2026-07-26.md`（文献底座，按需查阅）
5. progress/plan/analysis 对应条目（项目状态对齐）

## 4. 当前决策状态（as of 2026-07-28）

- **已完成**：文献扫描 → 两批 idea（A–F, G–N）→ 多轮自辩 → H博士 点名 A/J/K/M 展开 → Codex Round A（gpt-5.5，3C+7M+2Cn，12 条全处置 §Rev-1..11）→ Codex 模型升级 gpt-5.6-sol（CLI 0.116→0.145，Rule 9 运行时条款）→ **Round B（gpt-5.6-sol，0C+8M+1Cn，PROCEED-WITH-FIXES，§Rev-12..20 落地）**；两模型排序结论一致
- **等 H博士 批准**：
  1. M 第 1 步侦察（异象审计，1–2 周 CPU）+ **WRDS/CRSP 权限核查**（M 死亡分支的硬前提，CODEX-M-03）
  2. J 的 CN 数据 **kill gate**（正式通过/失败标准，CODEX-J-01）并行执行
- **此后流程**：胜出者正式 plan doc（含 §Rev 修正 + 预注册件）→ Codex Round B → 实施
- **约束**：8–9 月主线 = ICLR 2027 改稿（~9 月下旬截稿），本工作流只用间隙时间与 CPU/零算力任务

## 5. 执行层文件（2026-07-28-f 批准后新增）

| # | 文件 | 角色 | 状态 |
|---|------|------|------|
| 6 | `docs/prereg_m_scout_2026-07-28.md` | M 侦察预注册（看数据前冻结：主特征/分支规则/次要族/多重校正） | 冻结 |
| 7 | `docs/j_cn_data_gate_criteria_2026-07-28.md` | J 数据闸门标准（5 项检查 + 阈值，看数据前冻结） | 冻结 |
| 8 | `analyze_m_scout_step1.py` + `experiments/m_scout_step1/` | M 侦察脚本与 confirmatory 产出 | **完成**：T2 3 MAJOR 修复 → --full 跑完 → 分支裁决 **DEAD (bounded)** → T3 PASS-WITH-CONCERNS → analysis.md 2026-07-28-a |
| 9 | `~/.qlib/qlib_data/cn_data/`（项目外路径） | Qlib CN 日频 bundle | **陈旧（止于 2020-09-25）**——闸门拦截，见 progress 2026-07-28-h；替代源探测中（baostock K线可用，成分端点不稳） |
| 10 | `analyze_cn_data_gate.py` + `experiments/cn_data_gate/` | J 数据闸门脚本（C1–C5）+ smoke 证据 | 实现完成；full gate 待数据源定案 + T2 |
| 11 | `artifacts/reviews/2026-07-28_codex_code_mscout_A.md` | M 脚本 T2 评审（0C+3M，全修） | 完成 |
| 12 | `artifacts/reviews/2026-07-28_codex_results_mscout_A.md` | M 结果 T3 评审（0C+0M+4Cn，PASS-WITH-CONCERNS，含许可/禁止措辞清单） | 完成 |

## 6. 预留登记位（未来文件，产生即补登）

| 预期文件 | 触发条件 |
|---------|---------|
| `docs/plan_<idea>_<date>.md`（正式 plan doc，携带 §Rev-1..20） | M 侦察 + J 闸门结果出来、H博士 定骨架后 |
| `artifacts/audits/cn_data_gate_<date>.md`（J 数据闸门报告） | 闸门脚本跑完后 |
| J 闸门脚本（`analyze_cn_data_gate.py` 类） | CN bundle 下载完成后 |
| `artifacts/reviews/<date>_codex_code_*.md`（M/J 脚本 T2 评审） | 各脚本实现完成后 |
| WRDS 权限核查结论（记入 progress.md 条目） | **H博士 行动项**：USC 邮箱申请 WRDS 账号 |

→ progress: 2026-07-28-c | plan: 2026-07-26-a | analysis: N/A
