# figures/ — Story A 论文图

> **作用**: Story A 论文（ICAIF 2026, ACM SIGCONF target）所有 figure 的矢量 PDF + PNG 预览。

## 状态 (2026-06-23: 已清空旧批，重建中)

**2026-05-28 的整批旧图（27 figs + smoke test）已于 2026-06-23 全部删除**——它们建立在 PILOT / 旧数据上（untuned 4-model anchor、5 折、`storya_e1_anchor` + `storya_e6_dm_spa`），已被 D-RERUN-12F **confirmatory** 结果（tuned L0–L7 梯子、12 折、两族 + cost 口径）取代。整套图正用 `nature-figure` skill（数据图）+ `scientific-schematics` skill（流程图/GNN 示意）**按新质量标准重画**。

> ⚠️ `figures/` 未被 git 跟踪；删除不可用 git 恢复，但任一旧图可由 `paper_figs/*.py` 对旧数据重跑再生（脚本全部保留）。

**全局图表字体标准（H博士 2026-06-23 锁定）**: **无衬线 Arial**（数学体 DejaVu Sans），写进 `paper_figs/rcparams_storya.py`。理由：小字号更清晰 + ML 顶会惯例（图无衬线、ACM 正文衬线是标准搭配）。各图脚本可加 `--font serif` 出衬线备选（带 `_serif` 后缀），canonical 文件一律 sans。

> 📖 **图库（中英文详解）**: [`figure_gallery.html`](figure_gallery.html) — 浏览器打开，逐图中英文对照描述（怎么看 / 结论 / 数据源）。新图按文件顶部模板追加。

## 当前内容（6 张 confirmatory 图，2026-06-23 重建，已过三方 QA）

| 文件 | 节 | 内容 | 脚本 | 源 |
|---|---|---|---|---|
| `headline_ic_ladder` | §5.1 | 每臂 IC forest（L0–L7×2池）+ LightGBM 基准线 | `fig_headline_ic.py` | family1_ic_ci.csv |
| `F9_spa_dm_confirmatory` | §5.2 | 左 Hansen SPA（B0.277/C0.077 皆不拒绝）+ 右 20 对 DM/HLN BH-FDR 森林 | `fig_f9_confirmatory.py` | family1_spa/dm_hln/mde.csv |
| `regime_perfold_ic` | §5.3 | 逐季 IC 热图（臂×12 季）+ 逐季均值 IC 柱 | `fig_regime.py` | storya_v21_main12_tuned/results.csv |
| `cost_gross_net` | §5.4 | 成本阶梯（Univ C）+ gross/net 一致性散点 | `fig_cost.py` | storya_v21_cost/*.csv |
| `family2_edge_causal` | §5.5 | FC 因果森林 matched vs tuned + ±MDE 区（0/6 BH，6/6 欠功效） | `fig_family2.py` | family2_fc_causal.csv |
| `pipeline_confirmatory` | §Methods | 5 阶段流程 + 梯子 zoom + GNN 消息传递示意 | `fig_pipeline.py` | 概念图（无数据） |

**三方 QA**（2026-06-23）：Codex 代码/数据正确性（3 fix + 3 PASS）+ nature-figure QA 清单（字号/编码/可追溯）+ 我亲眼看图（信息量/直观/美观/无重叠/本科生可读）。详见 progress 2026-06-23-b。各图均有 `--font serif` 备选（带 `_serif` 后缀）。

> **caption 待办**：每张数据图的 LaTeX caption 必须带 ML-stats 块（seeds=10 / folds=12 / metric / CI 定义 / baseline=L0）——QA 清单要求，写 §Results 时补。

## Exploratory 图（2026-06-24 选择性保留，§5.7，sans 风格，明标非 confirmatory）

| 文件 | 节 | 内容 | 脚本 | 源 |
|---|---|---|---|---|
| `loss_listmle_inversion` | §5.7 图7 | loss 家族 mean IC（ListMLE 翻转 −0.0458 vs MSE +0.0113） | `fig_loss_inversion.py` | experiments/loss_horserace/results.csv |
| `plan_aaa_t1_stability` | §5.7 图8 | Plan-AAA permutation rank vs 单特征 abs-IC proxy rank（5/15 一致；proxy top-15 有无 T−1 shift 相同 → 度量分歧、非泄漏修正；hc 组 unscored；L1） | `fig_plan_aaa_t1.py` | artifacts/plan_aaa_t1_diagnostic/group_ranking_comparison.csv |

## 待定（已决定去留）

- 选择性保留（H博士 2026-06-24）：仅 loss-horserace（ListMLE 翻转）+ Plan-AAA T-1（Univ-C 基础 caveat）重建为 §5.7 exploratory；其余先验图（horizon / graph-ablation / step3 / selectivenet / tier1）**不重建、不进 confirmatory 论文**。旧 `fig_horizon_ablation.py` 等脚本保留备查。

## 变更日志

- **2026-05-28**: 初版 27 figs（pilot 数据）
- **2026-06-23**: 清空全部旧图，仅保留新 `F9_spa_dm_confirmatory`；改用 nature-figure / scientific-schematics 在 confirmatory 数据上重建（→ progress: 2026-06-23-a）
- **2026-06-23**: 重建 6 张 confirmatory 图（§5.1–§5.5 + 流程图），过三方 QA（→ progress: 2026-06-23-b）
- **2026-06-24**: 新增 2 张 §5.7 exploratory 图（`loss_listmle_inversion`、`plan_aaa_t1_stability`），嵌入 confirmatory 草稿 v2（→ progress: 2026-06-24-a）；`build_gallery.py` 加 `.tag.expl` 灰色 tag + 2 entries，重生 `figure_gallery.html` = **8 张图**（6 confirmatory + 2 exploratory，base64 内嵌）（→ progress: 2026-06-24-b）
- **2026-09-11**: 重画 `plan_aaa_t1_stability`（旧标题 "only 5/15 … after T−1 leak correction" 为误表述：proxy top-15 集合有无 T−1 shift 完全相同，5/15 = Plan-AAA permutation top-15 ∩ 单特征 IC proxy top-15；两个纯 hc 组标为 proxy unscored；脚本内断言该不变式）+ 同步 `paper/iclr2027/figures/` 副本 + gallery 文案重写（→ progress: 2026-09-11-f）
