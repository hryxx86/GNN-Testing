# docs/ — 活跃文档

> 当前阶段的分析报告、诊断记录、session handoff。tri-doc 的 `analysis.md` 在此。

---

## 当前内容 (as of 2026-09-11)

### Tri-Doc
- `analysis.md` — 分析发现总表（Quad-Doc Rule 5 强制更新）

### C5 sensitivity（2026-09-10）
- `c5_rerun_brief_2026-09-10.md` — H博士 C5 任务简报（§0–§8 原文）+ 实施注记 §9（偏离清单、TP1 处置 §9.9、C-pre 提案 §9.10）（→ progress: 2026-09-10-a/-b）
- `c_pre_report_2026-09-12.md` — **C-pre 完整报告**（一句话结论、选择器与两条限定、运行与调参、结果三表、许可/禁用读法、论文改法、评审链、复现命令、遗留）（→ progress: 2026-09-12-d）
- `c_pre_plan_2026-09-11.md` — **C-pre 冻结方案**（pre-evaluation 特征重选：选择窗 / 覆盖规则 / 组分数 / 分组 / 候选范围 / 下游协议 / 预设措辞 / 代码改动；取代 `c5_rerun_brief` §9.10；Codex TP1-A PROCEED-WITH-FIXES 已落实；**待 H博士 批准**）（→ progress: 2026-09-11-g）
- `c5_sensitivity_report_2026-09-11.md` — **C5 完整报告**（摘要、定性问题、设计、结果表、解读边界、论文改法 + 英文附录段落草稿、后续选项、评审链、产物与复现命令）（→ progress: 2026-09-11-e）

### 文献对照
- `lit_benchmark_2026-07-03.md` — 15 篇顶会/顶刊文献对照评估（逐篇总结 + 法证对照矩阵 + 我方论文评估；→ progress: 2026-07-03-d）
- `lit_scan_2026-07-26.md` — 近半年（2026-01→07）金融 ML 六切面扫描（~70 篇 + 空白 G1–G9 + idea 候选 A–N；→ progress: 2026-07-26-a）

### 下一篇论文 Ideation 工作流（2026-07-26 起）
- `next_paper_ideation_index.md` — **工作流唯一入口/文件索引（专门管理，新 session 从这里进）**
- `idea_expansions_2026-07-28.md` — A/J/K/M 四候选完整设计稿 + Codex Round A 修正 §Rev-1..11（→ progress: 2026-07-28-a/-b）

### 项目概览
- `project_findings_overview_2026-04-20.md` — 最新项目总结
- `advisor_presentation_2026-04-21.md` — 导师汇报（中英混排，2026-04-21 新增）
- `advisor_presentation_2026-04-21_en.md` — 导师汇报（全英文，paper-ready）

### Phase 5 诊断报告
- `phase5_diag_fold4.md` — Fold 4 异常诊断
- `phase5_diag1_normalization.md` — 归一化诊断 (diag1)
- `phase5_diag1b_replication.md` — 复现性诊断 (diag1b)
- `phase5_diag_feature_importance.md` — 特征重要性诊断
- `fold4_leakage_diagnostic_2026-04-20.md` — Fold 4 泄露专项诊断

### Session Handoffs
- `session_handoff_2026-04-15.md` … `session_handoff_2026-07-03.md`（历史，按日期排列）
- `session_handoff_2026-07-07.md`
- `session_handoff_2026-09-12.md` ← 最新（C5 + C-pre sensitivity 完成；待 H博士：论文改口径 / push）
- `session_handoff_2026-09-11.md` — C5 session 交接（冻结于其日期）

---

## 关键文件速查

| 文件 | 用途 | 产出于 | 状态 |
|------|------|-------|------|
| `analysis.md` | Tri-doc 分析记录（必读） | 持续更新 | active |
| `session_handoff_2026-09-12.md` | 最新 session 交接（新窗口必读；历史 handoff 按日期排列） | 2026-09-12 | active |
| `project_findings_overview_2026-04-20.md` | 项目总纲 | 2026-04-20 | active |

---

## 相关上下游

- 归档文档 → `archived/docs/`（旧 REPORT、session handoff、literature review）
- 产出的诊断基于 → `experiments/diag_phase5_*.csv`

---

## 变更日志

- **2026-04-20**: 新增 README；补充 fold4_leakage_diagnostic、project_findings_overview（→ progress: 2026-04-20-d）
- **2026-04-21**: 新增 `advisor_presentation_2026-04-21.md` + `_en.md`（导师汇报中英双版；图见 `plots/advisor/`）（→ progress: 2026-04-21-a）
- **2026-07-28**: 新增下一篇论文 ideation 工作流三件套：`next_paper_ideation_index.md`（专门管理索引）+ `lit_scan_2026-07-26.md` + `idea_expansions_2026-07-28.md`；评审存档在 `artifacts/reviews/2026-07-28_codex_plan_A.md`（→ progress: 2026-07-28-c）
- **2026-09-11**: 新增 `c5_rerun_brief_2026-09-10.md`（C5 任务简报 + 实施注记 + C-pre 提案）与 `session_handoff_2026-09-11.md`（→ progress: 2026-09-10-a, 2026-09-11-d）
- **2026-09-11**: 新增 `c_pre_plan_2026-09-11.md`（C-pre 冻结方案，Codex TP1-A 通过，待 H博士 批准）（→ progress: 2026-09-11-g）
- **2026-09-12**: 新增 `session_handoff_2026-09-12.md`（C-pre 完成后的交接；09-11 的 handoff 冻结）（→ progress: 2026-09-12-d）
- **2026-09-12**: 新增 `c_pre_report_2026-09-12.md`（C-pre 完整报告）（→ progress: 2026-09-12-d）
