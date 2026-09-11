# experiments/ — 实验结果

> 主实验结果文件夹（100+ files）。按 CLAUDE.md Rule 5 narrow 策略：**仅索引论文级主表 + 子目录**；其余（per-seed 预测、训练日志、11 个 Phase 5 诊断 CSV、扇区归因 CSV、置换 IC 数组 .npy 等）查 `progress.md` 对应日期或 `ls`。

---

## 子目录

- [`horizon_preds/`](horizon_preds/) — 60× 预测 .npy（MLP/SAGE-Mean × 2 variants × 3 seeds × 5 folds）
- [`step3_plan_z/`](step3_plan_z/) — Step 3 Plan Z 统计检验
- [`qwen_cache/`](qwen_cache/) — Qwen LLM 特征缓存
- `stale_pre_fix/` — empty（保留位）

### Sanity-Check Suite（2026-06-11,管线证伪 E0–E4;详见 `docs/analysis.md` 2026-06-11-a）
- [`sanity_e0_wiring/`](sanity_e0_wiring/) — E0 接线 + provenance canary（`wiring_check.csv`,14/14 PASS）
- [`sanity_e1_oracle/`](sanity_e1_oracle/) — E1 return-corr oracle（**上界诊断,泄露**）
- [`sanity_e1b_label_oracle/`](sanity_e1b_label_oracle/) — E1b label-sim oracle（**支持诊断,泄露**）
- [`sanity_e2_shuffled/`](sanity_e2_shuffled/) — E2 degree-preserving shuffled（负控）
- [`sanity_e3_planted/`](sanity_e3_planted/) — **E3 planted-signal recovery（决定性必要控制:GNN 恢复 82-91%、MLP≈0）**
- [`sanity_e4_diagnostics/`](sanity_e4_diagnostics/) — E4 零训练图诊断（density/degree/AUC,进论文）
- [`sanity_summary/`](sanity_summary/) — `verdicts.json` + `sanity_summary.md`（总 verdict = RESULT A,管线无罪）
- `sanity_{fullrun,smoke,resmoke}_*.log` — 运行日志（不索引,按需查）

### C5 sensitivity（2026-09-10，post-hoc、test-informed；详见 `docs/c5_rerun_brief_2026-09-10.md` §9.9）
- [`storya_v21_main12_c5/`](storya_v21_main12_c5/) — L0/L1 × 10 seeds × 12 folds = 240 cell（cell_id 2400–2639）；`results.csv` / `manifest.csv` / `per_day_ic/` / `_universe_c5.json`（20 列）/ `_frozen_hp_provenance.json` / `_run_provenance.json`（git 白名单提交）
- `storya_v21_tune/C5_{L0,L1}.json` + `frozen_hparams_c5.json` — C5 两臂 30-trial 调参冠军与冻结文件（副本+md5 在 `artifacts/storya_v21_tune/`）

---

## 论文级主表

| File | Use | Status |
|------|-----|--------|
| `wf5_results.csv` | v4 walk-forward 主表（90 runs） | active (paper main) |
| `horizon_ablation_results.csv` | Horizon 消融（360 runs） | active (paper) |
| `arch_comparison_results.csv` | 架构消融（150 runs） | active (paper) |
| `graph_ablation_results.csv` | 图结构消融 | active (paper) |
| `ranking_loss_results.csv` | Ranking loss 消融 | active (paper) |
| `permutation_v2_results.csv` | 置换检验（16K） | active (paper) |
| `selectivenet_results.csv` | SelectiveNet 覆盖-IC | active (paper) |
| `comprehensive_metrics.csv` | Week 3 综合指标 | active |
| `gate1_results.csv` | SEC Gate 1 → STOP | decision made |

---

## 其他文件（不在此索引，按需查找）

- **Phase 5 Step 0-3 诊断 CSV** (15+) → `docs/phase5_diag_*.md` 或 `ls experiments/diag_*.csv`
- **置换检验 IC 数组 .npy** (15+) → 论文级汇总在 `permutation_v2_results.csv`；raw IC 分布用 `ls experiments/perm_*.npy`
- **Per-seed/per-fold 预测 .npy** → 命名规则 `preds_<model>_<variant>_s<seed>.npy` / `diag_preds_*.npy`
- **训练日志 .txt** (30+) → 查 `progress.md` 对应日期条目（每次运行都有 log 记录）
- **扇区归因 CSV** → `diag_sector_{attribution,composition,ic}_*.csv`，分析见 `docs/analysis.md`

---

## 相关上下游

- 脚本 → 根目录 `run_*.py` + `v3_*.ipynb`，归档脚本 `archived/scripts/`
- 可视化 → `plots/paper_*.png`
- 诊断文档 → `docs/phase5_diag_*.md`, `docs/fold4_leakage_diagnostic_2026-04-20.md`
- 旧版 results → `archived/stale_results/`

---

## 变更日志

- **2026-06-11**: 新增 7 个 Sanity-Check 子目录（`sanity_e0..e4_*` + `sanity_summary/`）到子目录索引；run 日志不索引（→ progress: 2026-06-11-a）
- **2026-04-20**: 新增 README。严格遵循 Rule 5 narrow：只索引论文级主表（9 个 CSV）+ 子目录；诊断 CSV/日志/per-seed NPY 不在此登记（→ progress: 2026-04-20-d, 2026-04-20-e）
- **2026-09-10**: 新增 `storya_v21_main12_c5/`（C5 test-informed subset sensitivity，240 cell）+ `storya_v21_tune/C5_*.json`/`frozen_hparams_c5.json`（→ progress: 2026-09-10-c）
