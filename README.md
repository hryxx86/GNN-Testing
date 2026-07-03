# GNN-Testing — S&P 500 股票预测 GNN 研究项目

> 基于 GNN + NLP 的美股排名预测研究。当前处于 Phase 5（截面归一化 + 新特征工程）。

---

## 项目入口（新 session 必读）

**按优先级阅读**：
1. `CLAUDE.md` — 项目规则（每 session 强制加载）
2. `progress.md` — 已完成工作
3. `plan.md` — 接下来计划
4. `docs/analysis.md` — 分析发现
5. `docs/session_handoff_2026-04-20.md` — 最新 session 交接
6. 本次任务相关文件夹的 `README.md`（Quad-Doc Rule 5）

---

## 文件夹索引

| 文件夹 | 作用 | 说明 |
|--------|------|------|
| [`archived/`](archived/) | 归档 | 已废弃的脚本/notebook/结果/旧文档 |
| [`artifacts/`](artifacts/) | 中间产物 | 运行日志、元数据 JSON |
| [`data/`](data/) | 数据集 | OHLCV、新闻 embedding、SEC 特征、alpha158 |
| [`docs/`](docs/) | 活跃文档 | analysis、诊断报告、session handoff |
| [`experiments/`](experiments/) | 实验结果 | CSV/NPY 结果、训练日志、预测文件 |
| [`plots/`](plots/) | 可视化 | 43 个 PNG：网络图、诊断图、论文图 |
| [`scripts/`](scripts/) | 数据准备脚本 | 图构建、事件处理、新闻处理 |
| [`utils/`](utils/) | 工具模块 | grouping、stats_tests、plan_z_subsets |
| [`.claude/rules/`](.claude/rules/) | Path-scoped 规则（2026-04-23 新增） | 按 paths 触发加载，补充 CLAUDE.md 通用部分 |
| [`.claude/commands/`](.claude/commands/) | Rule 9 slash commands（2026-04-23 新增） | `/codex-plan-review`, `/codex-code-review`, `/codex-results-review`, `/session-closeout`, `/verify-docs-provenance` |
| [`.claude/agents/`](.claude/agents/) | Subagent 定义 | `finance-gnn-reviewer.md` — Codex fallback reviewer |
| [`artifacts/reviews/`](artifacts/reviews/) | Rule 9 review artifacts（2026-04-23 新增） | 每个 Codex/finance-gnn-reviewer/Explore review 一份 YAML+body |

---

## 根目录脚本/Notebook（核心列表 as of 2026-05-21）

> **注**：下方显式计数为 2026-05-21 快照，**早于** Story A 流水线（`run_storya_e1_anchor.py`、`run_storya_e3_news_edge.py`、`run_storya_e4_alpha.py`、`run_storya_e1_6_hats.py`、`compute_e6_dm_spa.py`、`compute_e6_edge_ablation.py`、`analyze_e1_lofo.py`）与 2026-06-11 Sanity-Check Suite——这些另见各自专节，根目录脚本总数已增。

### Active Notebooks (.ipynb, 4)
| 文件 | 用途 | 状态 |
|------|------|------|
| `v3_ranking_pipeline.ipynb` | v3 排名预测主 pipeline | active |
| `v3_stability_experiments.ipynb` | 稳定性实验 | active |
| `v3_week2_experiments.ipynb` | Week 2 实验 | active |
| `v3_week3_diagnostics.ipynb` | Week 3 诊断 | active |

### Tier 1 / Stage 1 实验 Runners (.py, 5)
- `run_tier1_phase_a.py` — Tier 1.B (4 losses × 2 models × 2 feats × 5 folds × 10 seeds) + Tier 1.D (hparam sweep)
- `run_tier1a_phase_b.py` — Tier 1.A rolling 2y vs expanding window pilot
- `run_tier1b_h2_phase_b.py` — Tier 1.B 在 h2 (AdamW lr=5e-4) 复跑
- `run_tier1c_phase_b.py` — Tier 1.C anchored RankNet 对照
- `run_loss_horserace.py` — Stage 1 loss horse race (4 losses × 2 models × 3 features, 10 seeds)

### Active Analyzers (.py, 5)
- `analyze_tier1_phase_a.py` — **CANONICAL stat engine**: NW-HAC + fold-cluster bootstrap + BH-FDR (Tier 1.B + 1.D)
- `analyze_phase_b_finalize.py` — Phase B 综合：Tier 1.A/1.C/1.B-h2 统计（复用 canonical engine）
- `analyze_tier1e_regime_forensic.py` — Tier 1.E regime gate (lagged high-stress)
- `analyze_tier2c_sector_ic.py` — Tier 2.C IC_sector_resid 跨 6 tiers
- `analyze_loss_horserace.py` — Stage 1 mixed-effects + Studentized block bootstrap + BH-FDR

### 数据 + 特征构建 (.py, 3)
- `download_ohlcv_yf.py` — yfinance OHLCV pipeline（取代失效 EODHD）
- `build_alpha158_features.py` — Alpha158 158-feature 集合（S8 baseline）
- `build_phase5_features.py` — Phase 5 5-feature 集合（mom12m, maxret, dolvol, CORR5, RSV5）

### Shared Library (.py, 1)
- `run_step3_plan_z_part_a.py` — **不可归档**：被 5 个 active script `import` 作为 data loading + model definitions + utilities 共享模块（4 个 Tier 1 runner + `run_loss_horserace.py`）。注：另有 4 个引用 part_a 的脚本（part_b/c/c_perfold + smoke_test_part_a）已归档；它们的 import 在 archived 位置不会触发，仅作历史记录。

### Sanity-Check Suite (.py, 3, 2026-06-11)
管线证伪套件（E0–E4），发布 Story A null 前falsify"null 是破管线伪影"。import-only 复用 `run_storya_e1_anchor.py`，零改动 anchor。详见 `docs/analysis.md` 2026-06-11-a。
- `run_sanity.py` — E0 wiring/provenance canary + E1/E1b/E2/E3 runner（`--experiment/--graph_type/--smoke/--resume`）
- `analyze_sanity.py` — E4 零训练诊断 + verdict 发射 + `experiments/sanity_summary/`（复用 `compute_e6_dm_spa.hln_test`+`bh_fdr`）
- `sanity_common.py` — 4 个 edge/signal builder（oracle / label-oracle / shuffled / planted）+ verdict 阈值逻辑（shared lib）

### 敏感文件
- `openrouter_key.txt` — API key（不 commit）

### 归档脚本（18 files）
- `archived/scripts/2026-05-21/` — Phase 5 legacy (7) + Plan Z++ completed (6) + one-off fixes (5)
- 详细分类 + 归档原因见 `archived/scripts/2026-05-21/README.md`

---

## 关键路径（来自 CLAUDE.md Rule 7）

- Conda env: `gnn` — `/opt/homebrew/Caskroom/miniforge/base/envs/gnn/bin/python`
- GitHub: `https://github.com/hryxx86/GNN-Testing`
- Google Drive folder: `GNN测试`

---

## 变更日志

- **2026-07-02**: 新增 `analyze_paper_eval_robustness.py`（投稿前评估 4 项零重跑核查：per-seed 符号 / LOSO / 26 检验合池 BH / BY 敏感性 → `artifacts/audits/paper_eval_robustness.csv`）；评估报告 `docs/paper_evaluation_2026-07-02.md`（→ progress: 2026-07-02-a）
- **2026-06-26**: 新增 `analyze_m10_universe_gap.py`（M10 survivorship 审计：Wikipedia 重构 PIT 成分股 vs 固定 universe → `artifacts/audits/m10_universe_gap.{csv,md}`）；PaperJury Round-1 全部处置（→ progress: 2026-06-26-a）
- **2026-06-11**: 新增 Sanity-Check Suite 专节（`run_sanity.py`/`analyze_sanity.py`/`sanity_common.py`）；标注根脚本计数为 pre-Story-A 快照（→ progress: 2026-06-10-c, 2026-06-11-a）
- **2026-04-20**: 建立 24 个文件夹 README 体系，CLAUDE.md 升级为 Quad-Doc（→ progress: 2026-04-20-d）
- **2026-04-27**: 新增 `run_local_stage1_segmented.sh`（本地 12h+1h segmented runner）；patched `analyze_loss_horserace.py:247` (sm.stats.norm → scipy.stats.norm fix for statsmodels 0.14 API)（→ progress: 2026-04-27-a）
- **2026-05-21**: 根目录 .py 32 → 14（18 files 归档到 `archived/scripts/2026-05-21/`），重写根脚本索引（→ progress: 2026-05-21-a）
