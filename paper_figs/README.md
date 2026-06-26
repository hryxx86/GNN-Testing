# paper_figs/ — Story A 论文 figure 生产 pipeline

> **作用**: 生产 Story A 论文（ICAIF 2026 ACM SIGCONF target）的所有 figure 与 LaTeX table。所有脚本 read-only 读取 `experiments/` 与 `artifacts/` 下的源 CSV / npy，输出到 `figures/` 与 `tables/`。
>
> **设计原则**: 模块化（每个实验家族一个 fig_*.py），共享 rcparams 与 helper，source provenance 通过 SOURCE_CONTRACT 头部 + md5 校验，每个数值化输出可追溯到源 CSV。
>
> **Rule 9 状态**: T2 self-review PASS（Round A+B+C, 2026-05-28），见 `artifacts/reviews/2026-05-28_claude-self-review_code_{A,C}.md`。

## 当前内容

### 共享基础设施

| 文件 | 作用 |
|------|------|
| `rcparams_storya.py` | 共享 rcparams、`setup(format)`、`save(fig, name)`、`PALETTE`、`model_color()`、`write_tex_table()`。ACM SIGCONF 兼容尺寸（single_col=3.33in / full_width=7.0in）。Times serif fallback DejaVu Serif。颜色 colorblind-safe (ColorBrewer Set2)。底部含 smoke test 入口 (`python paper_figs/rcparams_storya.py`)，输出 `figures/test_rcparams.{pdf,png}` |

### Phase 6.2 模块（8 个，先验实验，本地 CSV 输入）

| 文件 | 产出 figures | 产出 tables | 源 CSV |
|------|------|------|------|
| `fig_horizon_ablation.py` | F7, F8 | ST3 | experiments/horizon_ablation_results.csv (360×12) |
| `fig_plan_aaa.py` | F10, S4 (S5 skipped) | ST4 | artifacts/plan_aaa/ranking.csv + artifacts/plan_aaa_t1_diagnostic/group_ranking_comparison.csv |
| `fig_phase5_step3.py` | S6 | ST5 | experiments/step3_plan_z/{hansen_spa_results, part_b_summary}.csv |
| `fig_loss_horserace.py` | S7, S8, S14 | ST6 | experiments/loss_horserace/{results, paired_delta_ic, results_diagnostic_price}.csv |
| `fig_graph_ablation.py` | S9 | — | experiments/graph_ablation_results.csv |
| `fig_phase5_diagnostics.py` | S10, S11 | — | experiments/diag_phase5_permutation_importance_lgb.csv + diag_sector_attribution_sage_mean.csv |
| `fig_selectivenet.py` | S12 | — | experiments/selectivenet_results.csv |
| `fig_tier1_phaseb.py` | S13 | — | artifacts/tier1{a,b_h2,c}_phase_b/results.csv |

### Phase 6.3 模块（5 个 + 2 helpers，Story A 主实验输入）

| 文件 | 产出 figures | 产出 tables | 源 |
|------|------|------|------|
| `fig_e1_anchor.py` | F2, F3, F4, S1, S2, S3, S17 | T1, T2 | experiments/storya_e1_anchor/{results.csv, per_day_ic/*.npy} + artifacts/storya_e6_dm_spa/{bootstrap_ci, lofo_diagnostic, per_fold_table, per_cell_distribution, e1_three_column_summary}.csv |
| `_fig_e1_anchor_perday.py` | (helper) F2, S3 + npy loader | — | per_day_ic/*.npy（402 个 float32 数组）|
| `_fig_e1_anchor_tables.py` | — | T1, T2 + caption writer | bootstrap_ci.csv + e1_three_column_summary.csv |
| `fig_e6_statistical.py` | F9, S16 | T3, ST2 | artifacts/storya_e6_dm_spa/{spa_results, dm_hln_results, multiple_testing_ledger.json} |
| `fig_e6_cost_ladder.py` | F5 | T4 | artifacts/storya_e6_dm_spa/cost_ladder.csv |
| `fig_edge_ablation.py` | F6, S18 | T5 | artifacts/storya_e6_edge_ablation/{edge_bootstrap_ci, edge_pairs_dm}.csv + experiments/storya_e3_news_edge/news_snapshots_cache.npz |
| `fig_walkforward_calendar.py` | S15 | — | experiments/storya_e1_anchor/results.csv (用 fold + test_period 列) |

### Schematic（独立架构图）

| 文件 | 产出 | 工具 |
|------|------|------|
| `fig_pipeline_schematic.py` | F1 pipeline 架构图 (PDF + SVG) | matplotlib + FancyBboxPatch；非数据 figure，输入为 plan 文档常量；ACM SIGCONF 7.0×3.5 in |

### 工具与缓存（不在 paper 引用，但保留在仓库）

| 文件/目录 | 作用 | 保留原因 |
|---|---|---|
| `_inspect.py` | 36 行 throwaway 脚本，遍历 21 个源 CSV 打印 schema + head（首次 Phase 6.2 agent 沙盒受限时用主 session 主动跑的）| 保留：(a) 复用价值 — 后续新 CSV 加入时可一行修改后重跑；(b) 文档作用 — 直接显示了 Phase 6.2 / 6.3 所有源 CSV 的 schema 真值；不要在 production 调用 |
| `__pycache__/` | Python bytecode 缓存 | 保留：本地开发缓存。建议在仓库根 `.gitignore` 中过滤；不影响其他人 clone 后运行（自动重生）|

## 关键文件速查

- **入口（共享前置）**: `rcparams_storya.py` — 所有 fig_*.py 必须 import 这个模块
- **smoke test**: `python paper_figs/rcparams_storya.py` 单独跑会渲染 dummy bar chart 到 `figures/test_rcparams.{pdf,png}` 验证 rcparams 工作
- **批量重跑全部 figures**:
  ```bash
  cd /Users/heruixi/Desktop/GNN-Testing
  PY=/opt/homebrew/Caskroom/miniforge/base/envs/gnn/bin/python
  for f in paper_figs/fig_*.py; do echo "=== $f ==="; $PY "$f"; done
  ```

## 调用契约（每个 fig_*.py 共有）

每个 `fig_*.py` 模块必须遵守的契约：

1. **SOURCE_CONTRACT 头部** — 文件顶部 docstring 内含的注释块，列出 `inputs[]`（path + columns + md5 + n_rows）和 `outputs[]`（path + headline_values）。md5 是写脚本时源 CSV 的快照，便于检测 CSV 在脚本之后被静默修改。
2. **`from paper_figs.rcparams_storya import …`** — 共享 rcparams 必须经此路径，不允许重定义。
3. **`sys.path.insert` shim** — 顶部 3 行（已统一），让 `python paper_figs/fig_X.py` 与 `python -m paper_figs.fig_X` 两种方式都可调用。
4. **无副作用** — 不允许写到 `experiments/`、`artifacts/`、源 CSV 任何路径。只写 `figures/`、`tables/`。
5. **`def main()` + `if __name__ == "__main__": main()`** — 统一入口。
6. **强制 caveat 文本入 caption** —
   - L1 (Universe C LOW STABILITY) 必须 verbatim 出现在 F5/T4/F10/S4 captions
   - L6 (Fold 4 regime) 必须 verbatim 出现在 F3/F4/S3/S15/S17/T2 captions
   - N3 (0/5 pairs survive BH-FDR) 必须 verbatim 出现在 F6/T5 captions

## 变更日志

- 2026-05-28: 文件夹创建，16 个 .py 模块写完（rcparams + 8 Phase 6.2 + 5 Phase 6.3 + 2 split helpers + F1 schematic）+ 1 个 throwaway `_inspect.py`。所有模块通过 Rule 9 T2 self-review，verdict PASS（→ progress: 2026-05-28-b/d/e/f）。
- 2026-06-23: 6 个 confirmatory 脚本（`fig_headline_ic` / `fig_f9_confirmatory` / `fig_regime` / `fig_cost` / `fig_family2` / `fig_pipeline`）+ `build_gallery.py`；`rcparams_storya.py` 改全局 sans-Arial；旧 Phase 6.2/6.3 脚本（pilot 数据）保留备查但不进 confirmatory 论文（→ progress: 2026-06-23-a/b）。
- 2026-06-24: 新增 2 个 §5.7 exploratory 脚本 `fig_loss_inversion.py`（ListMLE 翻转，`experiments/loss_horserace/results.csv`）+ `fig_plan_aaa_t1.py`（Plan-AAA T-1 5/15 存活，`artifacts/plan_aaa_t1_diagnostic/group_ranking_comparison.csv`），sans 风格、嵌入 confirmatory 草稿 v2 §5.7（→ progress: 2026-06-24-a）。
