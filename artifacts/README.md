# artifacts/ — 中间产物

> 运行日志、fold 元数据、scaler 状态等 **非最终结果** 的中间产物。

---

## 当前内容 (as of 2026-09-12)

### 根级日志 (.log, 4)
- `alpha158_rebuild_save_raw.log` — Alpha158 重建
- `fold4_leakage_diag.log` — Fold 4 泄露诊断
- `part_c_perfold_run.log` — Step 3 Plan Z Part C perfold 运行
- `analyze_perfold.log` — Perfold 分析

### 子目录
- [`step3_plan_z/`](step3_plan_z/) — Step 3 Plan Z 元数据 JSON + CSV
- [`storya_v21_family1/`](storya_v21_family1/) — D-RERUN-12F Family-1 (predictive) confirmatory: SPA / DM-HLN / IC CI / MDE / LOFO / stability / C-L5s robustness
- [`storya_v21_family2_fc/`](storya_v21_family2_fc/) — D-RERUN-12F Family-2 (causal FC edge) confirmatory: matched-ΔIC + BH-FDR/6
- [`storya_v21_cost/`](storya_v21_cost/) — Cost-口径 (gross/net) crosswalk on the confirmatory ladder (descriptive; IC stays confirmatory)
- [`reviews/`](reviews/) — Rule 9 reviewer outputs (Codex / finance-gnn-reviewer / closeout), §6 schema
- [`storya_v21_family1_c5/`](storya_v21_family1_c5/) — **C5 post-hoc test-informed subset sensitivity, PRIMARY（T4 运行）**（NOT confirmatory）: family1 machinery in `--sensitivity` mode (raw HLN p, no BH/SPA) + `c5_*` seed robustness / paired contrast (SE+MDE) / ex-fold-9 / device replication / comparison table / integrity（2026-09-10/11）
- [`storya_v21_family1_c5_mac/`](storya_v21_family1_c5_mac/) — 同一套统计在 **Mac 设备复现**目录上的结果（对照用，2026-09-11）
- [`storya_cpre_select/`](storya_cpre_select/) — **C-pre 选择器归档**（2026-09-12）：`selection.json`（冻结列集 48 列 + md5、规则、选择窗、输入 md5、ticker/日期轴、选择器源码 git rev）、`feature_scores.csv`、`eligible_dates.json`、`group_scores.csv`、`selector_robustness.csv`、`summary.md`
- [`storya_v21_family1_cpre/`](storya_v21_family1_cpre/) — **C-pre post-hoc pre-evaluation re-selection sensitivity（Mac 主运行）**（NOT confirmatory）：family1 `--sensitivity` 统计 + `cpre_*`（integrity / seed / paired vs C,B,C5 / ex-fold / 调参披露 / tests 清单）
- [`storya_v21_tune/`](storya_v21_tune/) — frozen_hparams 源记录：confirmatory 20 winners + `frozen_hparams.json`；2026-09-10 起加 C5 两臂 winners + `frozen_hparams_c5.json` + `studies_c5/` sqlite 归档 + `c5_tune_archive_md5.json`；2026-09-12 起加 C-pre 两臂 winners（`CPRE_{L0,L1}.json`）+ `frozen_hparams_cpre.json` + `cpre_tune_archive_md5.json`（sqlite 在 `studies_cpre/`，未入库）

---

## 关键文件速查

| 文件 | 用途 | 产出于 | 状态 |
|------|------|-------|------|
| `fold4_leakage_diag.log` | Fold 4 专项诊断日志 | 2026-04-20 | active |
| `part_c_perfold_run.log` | Part C perfold 最新运行 | 2026-04-20 | active |
| `storya_v21_family1_c5/c5_comparison.md` | C5 sensitivity 并列表（主，T4）+ paired/ex-fold/调参披露 | 2026-09-11 | active |
| `storya_v21_family1_c5/c5_run_integrity.json` | C5 主运行完整性 + inputs（目录/md5/设备/代码身份） | 2026-09-11 | active |
| `storya_v21_tune/frozen_hparams_c5.json` | C5 两臂冻结超参（md5 cdb4d923） | 2026-09-11 | active |
| `storya_cpre_select/selection.json` | C-pre 冻结列集（48 列，md5 0ac94cf6）+ 选择规则/窗口/输入 md5/源码身份 | 2026-09-12 | active |
| `storya_v21_family1_cpre/cpre_comparison.md` | C-pre sensitivity 并列表（CPRE / C / B / C5）+ paired/ex-fold/调参披露 | 2026-09-12 | active |

---

## 相关上下游

- 最终结果 CSV → `experiments/step3_plan_z/`
- 对应脚本（已归档 2026-05-21）→ `archived/scripts/2026-05-21/run_step3_plan_z_part_c_perfold.py`, `archived/scripts/2026-05-21/analyze_fold4_leakage.py`, `archived/scripts/2026-05-21/cleanup_and_rebuild_features.py`

---

## 变更日志

- **2026-04-20**: 新增 README（→ progress: 2026-04-20-d）
- **2026-06-21**: 子目录索引补 D-RERUN-12F confirmatory 产出（storya_v21_family1 / family2_fc / cost / reviews）（→ progress: 2026-06-21-b）
- **2026-09-10**: 新增 `storya_v21_family1_c5/`（C5 sensitivity 统计产物）+ `storya_v21_tune/` C5 归档（→ progress: 2026-09-10-c）
- **2026-09-11**: 新增 `storya_v21_family1_c5_mac/`（Mac 复现统计）；`storya_v21_family1_c5/` 定为主（T4）；关键文件速查补 C5 三行（→ progress: 2026-09-11-c/-d）
- **2026-09-12**: 新增 `storya_cpre_select/`（C-pre 选择器归档）+ `storya_v21_family1_cpre/`（C-pre 统计产物）+ `storya_v21_tune/` C-pre 归档；关键文件速查补 2 行（→ progress: 2026-09-12-a）
