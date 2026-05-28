# Plan — What To Do Next

> **接下来做什么。** 按时间记录计划。每个条目与 `progress.md` 和 `docs/analysis.md` 时间对齐。
>
> **历史脚本路径说明（2026-05-21）**：本文件 2026-05-20 之前的条目中引用的根目录脚本（如 `run_walkforward_5fold.py`, `run_diag1*.py`, `run_gate1_experiment.py`, `run_phase5_step3_feature_expansion.py`, `diagnostic_phase5_*.py`, `run_step3_plan_z_part_{b,c,c_perfold}.py`, `smoke_test_part_a.py`, `analyze_step3_plan_z.py`, `analyze_fold4_leakage.py`, `analyze_seed_diagnostic.py`, `refetch_zts.py`, `cleanup_and_rebuild_features.py`, `run_figures_tables.py`, `make_advisor_figures.py`）均已在 2026-05-21-a 归档到 `archived/scripts/2026-05-21/`。历史条目中的路径不再修改（保留为时间点记录）；如需查阅文件请去归档目录，分类索引见 `archived/scripts/2026-05-21/README.md`。**Active 脚本路径**见 root `README.md` 根目录脚本节。

---

## 2026-05-26-a: Phase Milestone — Pivot from Plan AAA to Story A Paper ⏳ START

### What's now decided

- **Paper direction**: Story A "When Do GNNs Help in Cross-Sectional Stock Ranking" — conditional-findings paper using ~80% existing data + targeted new experiments
- **Plan AAA wording fixes (A-01/A-02 from 2026-05-23-d)**: DEFERRED — Plan AAA becomes a §X feature ablation within Story A, not standalone paper
- **Approved plan**: `/Users/heruixi/.claude/plans/handoff-session-ranking-swirling-lemur.md` (Story A §1-§8 + Mamba/Regime/Sector archive §3)
- **Target venue**: ICAIF 2026 (8-week realistic timeline; stretch +2 weeks if HATS reproduction included)

### Story A core experiments (in order of execution)

1. **§1.1 Multi-seed model comparison (CORE)** — adaptive 30→100 seed × 5 models × 5 folds × 21d
   - Models: GAT_price / SAGE-Mean_price / MLP_price / LSTM_price / Mamba-SAGE
   - Phase A (30 seeds guaranteed): ~20h A100
   - Phase B (70 more seeds, per-model conditional): 0-50h A100
   - Pre-committed extension rule in `experiments/storya_multiseed/prereg.json` BEFORE Phase A runs
2. **§1.2 News-as-edge co-occurrence** — test "news-as-feature hurts, news-as-edge helps" at 21d
3. **§1.3 HGT 21d rerun** — test if HGT IC=0.003 at 1d was horizon artifact
4. **§1.4 Cherry-pick detection suite** — DSR + PBO + bootstrap CI (no surveyed GNN-finance paper has these)
5. **§1.5 Mamba-SAGE prefix** — positive anchor / insurance
6. **§1.6 HATS baseline reproduction** — STRETCH only

### Mamba+Regime+Sector deferred to post-Story-A

Discussion archived in plan §3 of `handoff-session-ranking-swirling-lemur.md`. Reactivate if Story A succeeds and time permits.

### Negative-result framing (publication insurance)

4 templates identified with precedent (Hou-Xue-Zhang 2020 RFS, Chordia-Goyal-Saretto 2020 RFS, Lopez de Prado):
- T1: Strict eval reveals overstated claims
- T2: When-X-helps-when-X-hurts conditional findings
- T3: Failure-mode diagnose + mitigate
- T4: Methodology framework

### Immediate next actions (this session)

1. ✅ Update progress.md / plan.md / MEMORY.md (Rule 5/7 phase milestone)
2. ✅ Create `docs/session_handoff_2026-05-26.md` per §5 manifest
3. ⏳ Trigger Codex Touchpoint 1 plan review on the approved plan file (Rule 9 mandatory before any code)
4. ⏳ Write `experiments/storya_multiseed/prereg.json` (pre-commit extension rule for §1.1 Phase A)
5. ⏳ Copy `archived/scripts/run_horizon_ablation.py` → `run_storya_multiseed.py`, change SEEDS to 30-seed pool

→ progress: 2026-05-26-a | plan: 2026-05-26-a | analysis: N/A (pivot decision)

---

## 2026-04-27-b: Diagnostic_price 完成 → Path A (paper writing, Story C+) 推荐

### What's now decided (post-diagnostic)

- ✅ Diagnostic_price 200 cells in 7.3h, fully synced to Drive
- ✅ **Q1 答案**: Part B 高 IC (+0.037 MLP_price, +0.027 SAGE_price) **不可在 Stage 1 框架下复现** — 同 9-dim feature set 跑出 -0.004 / -0.057。说明 Part B 数字是 specific-setup artifact (code path + fold timing + model spec 综合), 不是 feature-set 决定。
- ✅ **Q2 答案**: Part B + Stage 1 都是 test IC（不是 val IC）；Stage 0 pilot 同时记录 val + test，但 winner selection 用 val_ic 是 bias source。
- ✅ **Major discovery (paper-strength)**: **ListMLE 系统性 fold-4 catastrophic collapse**，6/6 (architecture × feature) 组合 fold-4 IC ∈ [-0.36, -0.28]，σ_fold ≈ 0.18 (3× MSE)。Architecture-independent + Feature-independent + 10/10 seed-stable。

### Recommended path: A (paper writing) with Story C+

**Story C+ paper outline**:
- **Negative primary**: 600-cell horse race finds 0/8 co-primary rejection (ListMLE/Pairwise vs MSE)
- **Positive mechanism 1**: ListMLE fold-4 catastrophic collapse (universal across arch × feat) → likelihood ranking surrogate fails on regime shifts
- **Positive mechanism 2**: Pairwise scale collapse (4/4 contrasts under cluster bootstrap)
- **Practitioner warning**: val-IC pilot selection mis-predicts test-IC for ListMLE (Stage 0 val=+0.118 vs test=-0.045 on fold-2 pilot)
- **Robustness**: feature set independent (verified via S_price diagnostic), architecture independent (MLP + SAGE-Mean both)

**Target venue**: ICAIF 2026 (deadline mid-summer) or Workshop@NeurIPS/ICML (negative-result-friendly tracks)

### Path B/C considered and rejected

- **Path B (Stage 2 SPA)**: less valuable — no clear winner to test for supremacy; SPA framework is for "is X best vs all others" not "is anything better than baseline". Skip.
- **Path C (Stage 1.5b: full horse race +pairwise on S_price, +100 cells)**: limited value-add — 9-dim feature already shows same conclusion qualitatively. Pairwise on S_price would only confirm scale-collapse pattern (already 4/4 in Stage 1). **Skip unless paper reviewer specifically asks.**

### Next deliverables

- [ ] Patch resume bug in `run_loss_horserace.py` (Codex stop-review flagged: `.npy`-only check ignores partial CSV; not active issue but latent risk for future re-runs)
- [ ] Update `docs/analysis.md` 2026-04-27-b entry with full diagnostic + ListMLE fold-4 collapse evidence (after H博士 sign-off on Story C+ direction)
- [ ] Begin paper draft skeleton (4-6 page workshop format): intro / preregistration / methods / horse race results / ListMLE fold-4 mechanism / pairwise scale collapse / practitioner implications / discussion

### Decision needed from H博士

1. Approve Story C+ direction (Path A paper writing)?
2. Skip Stage 1.5b? (recommend skip)
3. Patch resume bug now or after paper draft skeleton?

→ progress: 2026-04-27-b | analysis: 2026-04-27-b (pending)

---

## 2026-04-27-a: Stage 1 horse race 完成 → 方向决策点 ⏳

**State**: Stage 1 全 600 cells 跑完（Colab 225 + 本地 M4 MPS 375），analyze 跑通，verdict = **Scenario B**（IC + Sharpe 主端点全部 0/8 reject；只有 SAGE × pairwise 的 pred_cs_std scale collapse 显著 2/8）。Codex Round D review 在跑（pending）。

详细数字 + verification 见 `progress.md` 2026-04-27-a。

### 当前阻塞

- **等 Codex Round D review** (`artifacts/reviews/2026-04-27_codex_results_D.md`)
  - 重点关注：fold-4 LOFO 反转（MSE 优势是 fold-4-driven）；scale collapse claim 是否 2/8 就够；mixed-effects 规格是否过/欠估 SE；prior-art framing
- **等 H博士 决策**：根据 Round D 结论，下一步应走以下哪条路径

### 三条候选路径（pending H博士 decision）

**Path A: 直接进入 paper writing（推荐 if Round D PASS / PASS-WITH-CONCERNS）**
- 故事：在 US 500-stock × 10-year × {S6, Alpha158} × {MLP, SAGE-Mean} 设置下，listwise / pairwise ranking losses 不显著优于 MSE 在 IC + Sharpe 上；scale collapse 在 SAGE × pairwise 高度可重复。
- 写作目标：ICAIF / FinNLP short paper（4–6 页）or workshop-track。
- 关键 risk：fold-4 LOFO 反转 → MSE 优势其实薄弱 → "non-superiority" 而非 "MSE 更好" 是更安全的 framing。
- Time: 1-2 周写作 + Codex 1 轮 paper-draft review。

**Path B: Stage 2 Hansen SPA re-run on winning loss（plan 预注册）**
- Plan 提到 `run_stage2_spa_rerun.py` 是 post-Stage 1 deliverable，但脚本未实现。
- 目的：用 SPA framework 给 MSE 做 supremacy claim, 而非 paired comparisons。
- 但**逻辑前提**是有"明显 winner"——Scenario B 没明显 winner（mse 仅 numerically 占优），跑 SPA 可能价值有限。Round D 应判断是否还值得跑。

**Path C: 从 Round D findings 判断需要 supplementary analysis 再做决策**
- 例：如果 Codex 提 "fold-4 driven 太严重，必须做 fold-by-fold breakdown 才能下结论"
- 例：如果 Codex 提 "mixed-effects 规格欠估了 SE，要换更保守的 GEE 或 cluster bootstrap"
- 需要 H博士 sign-off 是否值得做。

### 还要做的固定项（不依赖 path 选择）

- [x] progress.md 2026-04-27-a 已写
- [x] Drive sync 完成（600 preds + 9 analyze CSV + results.csv）
- [ ] `docs/analysis.md` 2026-04-27-a entry：8-contrast table + verdict + fold-4 caveat + scale collapse interpretation。等 Round D review 后写（避免重写）。
- [ ] README.md 变更日志条目：新增 `analyze_loss_horserace.py:247` patch + `run_local_stage1_segmented.sh`

→ progress: 2026-04-27-a | analysis: 2026-04-27-a (pending)

---

## 2026-04-23-a: Infrastructure overhaul (meta-project, not research) ✅

Drawn from the Claude Certified Architect — Foundations exam guide patterns applicable to our workflow. See `/Users/heruixi/.claude/plans/soft-jumping-aurora.md` for the selection rationale.

- [x] H1: CLAUDE.md → path-scoped `.claude/rules/*.md`
- [x] H2: Rule 9 touchpoints encoded as `.claude/commands/*.md` slash commands
- [x] H3: Structured reviewer YAML frontmatter schema in `.claude/rules/docs.md` §6
- [x] H4: Doc provenance rule (`.claude/rules/docs.md` §4) + `scripts/verify_docs_provenance.py` + `/verify-docs-provenance` command
- [x] H5: Session handoff manifest schema (`.claude/rules/docs.md` §5); retrofitted `docs/session_handoff_2026-04-20.md` as template
- [x] Rule 9 Touchpoint 2 on new script: Codex rate-limited → `finance-gnn-reviewer` fallback; 2 MAJOR + 3 CONCERN findings processed; all MAJORs FIXED
- [x] All smoke tests pass; verifier known limitations (tables, deeply-indented fences) documented in `.claude/rules/docs.md` §4

**Verdict**: infrastructure shift, zero research disruption. Stage 0 pilot (loss horse race) is still queued and unaffected.

→ progress: 2026-04-23-a | analysis: N/A

---

## 2026-04-21-a: Advisor-facing visualization package ✅

- [x] Design: single-file `make_advisor_figures.py` (CSV-driven, no training) +
  two bilingual `docs/advisor_presentation_2026-04-21{,_en}.md` files + 12
  figures in `plots/advisor/`
- [x] Covered all 12 findings (Priority S+A+B) with What/How/Figure/Analysis/
  Takeaway structure + Glossary + code-file appendix
- [x] Audit pass: 3 parallel Explore agents fact-checked; 4 numeric/wording
  issues fixed (Findings 6, 7, 8, 11 — details in progress.md 2026-04-21-a)
- [x] Rule-9 Codex code review: 2 BUGs fixed (fig_01 Panel-b SPA misuse; fig_06 inconsistent E2E/Vol-Cal aggregation). Both figures regenerated, both mds resynced.

→ progress: `2026-04-21-a` | analysis: N/A

**Decision**: Keep `make_advisor_figures.py` separate from `run_figures_tables.py`
to decouple advisor deck from paper pipeline and avoid hard-coded-number
reproducibility debt in the existing paper script.

---

## 2026-02-27-a: Phase C notebook created ✅

- [x] Create experiment pipeline notebook
- [x] Full-batch training on A100

→ progress: `2026-02-27-a` | analysis: N/A

## 2026-02-27-b: Run Phase C v1 experiments ✅

- [x] Run 6 experiments on Colab A100
- [x] Analyze results → all AUC ≈ 0.50

**Decision:** Diagnostics first, then model changes.

→ progress: `2026-02-27-b` | analysis: `2026-02-27-b`

## 2026-02-27-c: Diagnostics + docs restructure ✅

- [x] Write D.1 + D.2 diagnostic cells
- [x] Restructure documentation system
- [x] Run D.1 + D.2 on Colab → see `2026-03-03-a`
- [x] Record findings in analysis.md

→ progress: `2026-02-27-c` | analysis: `2026-02-27-c`

## 2026-03-03-a: D.1 + D.2 Results — FinBERT Signal Near Zero ✅

- [x] Analyze D.1/D.2 results
- [x] Update analysis.md with findings

**Verdict**: FinBERT alone = no signal. Go/Pivot/Stop criteria approaching STOP, but baseline matrix (XGBoost + momentum) not yet tested.

→ progress: `2026-03-03-a` | analysis: `2026-03-03-a`

## 2026-03-03-b: Literature Review — NLP+GNN Stock Prediction Papers ✅

- [x] Search and compare 6 papers (THGNN, DGRCL, DASF-Net, ChatGPT-GNN, etc.)
- [x] Identify why published papers show better results than our AUC~0.50
- [x] Update analysis.md with findings

**Key insight**: Our result is NOT a bug. It matches the only large-universe paper (DGRCL: 53% on 1K+ stocks). Papers claiming strong results use 12-30 cherry-picked stocks.

-> progress: `2026-03-03-b` | analysis: `2026-03-03-b`

---

# ═══════════════════════════════════════════════════════════════
# Roadmap v3: Ranking + Dynamic HGT + Selective Prediction
# ═══════════════════════════════════════════════════════════════
#
# 核心方向变更 (2026-03-05):
# v2 用 event-driven binary direction prediction → AUC ≈ 0.50 (EMH)
# v3 换赛道: calendar-driven ranking prediction + multi-horizon ablation
# 文献基础: MASTER (AAAI'24), FinMamba (2025), MDGNN (AAAI'24)
# ═══════════════════════════════════════════════════════════════

---

## 论文定位 & 叙事 (v3)

### 前沿论文定位

#### A. 金融 GNN SOTA (Ranking Target)

| 论文 | 会议 | 做了什么 | 我们的差异化 |
|------|------|---------|-------------|
| MASTER | AAAI'24 | Cross-stock Transformer, 5d ranking, IC=0.064 (CSI300) | 无显式图构建；无新闻；无selective prediction |
| FinMamba | arXiv'25 | Mamba+动态图, 1d ranking, Sharpe=2.06 (S&P500) | 无NLP/新闻；无异构边；无selective |
| MDGNN | AAAI'24 | 3类节点+多关系+日动态, IC=0.032 (CSI300) | 仅中国市场；无selective；无horizon ablation |
| THGNN | CIKM'22 | 日动态图+Transformer+HeteroGAT, IC=4.93% | 无NLP；无新闻节点；无selective |
| ChatGPT-GNN | KDD-W'23 | LLM推断图, 3-class, F1=0.41 | 仅DOW 30；F1低；无ranking |
| HGAIT | ESWA'25 | 正/负相关异构边+反向Transformer | 无NLP；无selective |

#### B. Selective Prediction SOTA

| 论文 | 会议 | 做了什么 | 与我们的关系 |
|------|------|---------|-------------|
| SelectiveNet | ICML'19 | 3-head: prediction + selection + auxiliary | 核心参考，从未应用于金融GNN |
| AUGRC | NeurIPS'24 | 修复AURC指标缺陷 | 评估指标 |
| Sim et al. | arXiv'23 | Chart image + confidence threshold 交易 | 唯一金融selective，非GNN |

#### C. 核心 Gap

**没有任何论文**同时做到：(1) 动态异构图 (2) 新闻+价格多模态 (3) ranking prediction (4) selective prediction (5) multi-horizon ablation (6) 大规模美国市场验证。这五项中每一项都有人做过，但组合是全新的。

**额外贡献：Horizon Ablation**。没有任何 GNN 论文系统比较过 1d/5d/10d/21d/42d/63d 预测 horizon。这是一个明显的文献空白。

### 论文叙事 (v3)

> "现有金融 GNN 面临三个未解决的挑战：(1) 静态图无法捕捉市场 regime 变化，(2) binary direction prediction 在高效市场上接近随机，(3) 所有股票所有时间都预测导致噪声主导。我们提出 **DynHetGNN-SP** — 动态异构图神经网络 + selective ranking prediction 框架。通过 HGT 架构融合四种动态边类型（价格相关性、行业、新闻提及、新闻共现），用 cross-sectional ranking 替代 binary classification，并引入 volatility-calibrated selection head 决定何时交易。在 S&P 500 (502 股) 上的系统性 horizon ablation (1d-63d) 揭示了 GNN 在不同时间尺度上的预测能力谱系。"

### 论文贡献 (v3)

1. **首个 GNN + Selective Prediction 组合**: SelectiveNet 从未被引入金融 GNN
2. **动态异构图 (4 边类型)**: correlation (月度动态) + sector (静态) + news→stock + news co-occurrence
3. **系统性 Horizon Ablation**: 1d/5d/10d/21d/42d/63d — 填补文献空白
4. **Calendar-driven + News-enhanced**: 每天预测所有股票，新闻作为 optional input feature
5. **大规模验证**: S&P 500, 502股 + economic significance (long-short portfolio, transaction costs)

### Prediction Target & Timing (v3)

```
可用特征 (截止时间):                      预测目标:
├─ 新闻 FinBERT embedding: T 时刻        Cross-sectional ranking score
├─ FinBERT sentiment: T 时刻              (Spearman corr with actual d-day excess return)
├─ Price features: T-1 close (严格!)
├─ Momentum/vol: T-1 close              Label: Z-score normalized d-day forward excess return
├─ Graph edges: 截止 T-1 的历史数据       d ∈ {1, 5, 10, 21, 42, 63} (horizon ablation)
│
│  Excess return = stock_return - equal_weight_market_return
│  Z-score: per-day cross-sectional normalization
```

---

## Phase N1: Calendar-Driven Data Pipeline ✅ CODE WRITTEN

**目标**: 将 event-driven 数据转为 calendar-driven (每天 × 每只股票 = 一个预测)

**Code**: `v3_ranking_pipeline.ipynb` Cells 4-7

### N1a. Stock-Day Matrix Construction

- [x] 构建 (trading_day, ticker) 矩阵: ~1250 天 × 502 股 ≈ 627K stock-days
- [x] 每个 stock-day 的特征:
  - Price features (9-dim): 5/10/21d momentum, volatility, return mean (T-1 close)
  - News features (771-dim): 当天有新闻 → mean-pooled FinBERT 768d + 3d sentiment; 无新闻 → zero vector
  - Has-news flag (1-dim): binary indicator
  - Total: 781-dim per stock-day
- [x] 处理无新闻的 stock-day: zero vector + has_news=0 (MSGCA 2024 做法)

### N1b. Multi-Horizon Labels

- [x] 对每个 stock-day 计算 6 种 forward return:
  - `ret_1d = (close[T+1] - close[T]) / close[T]`
  - `ret_5d = (close[T+5] - close[T]) / close[T]`
  - `ret_10d, ret_21d, ret_42d, ret_63d` 同理
- [x] Excess return: `excess_d = ret_d - market_ret_d` (equal-weight S&P 500)
- [x] Cross-sectional Z-score: per-day normalization `z_i = (excess_i - mean) / std`
- [x] **严格无前瞻**: close[T] 是 T 日收盘价，features 只用 T-1

### N1c. Time Split

- [x] Train: 2021-07 → 2023-12 (~630 trading days)
- [x] Val: 2024-01 → 2024-06 (~126 trading days)
- [x] Test: 2024-07 → 2025-12 (~378 trading days)
- [ ] Walk-forward validation as Phase N6 ablation

→ progress: `2026-03-05-d` | analysis: N/A

---

## Phase N2: Dynamic HGT Graph Construction ✅ CODE WRITTEN

**目标**: 构建动态异构图，月度更新 correlation 边

**Code**: `v3_ranking_pipeline.ipynb` Cells 8-9

### N2a. 4 Edge Types

| 边类型 | 构建方式 | 动态/静态 | 数据来源 |
|--------|---------|----------|---------|
| **stock↔stock (correlation)** | Pearson \|r\|>0.6, w=126d 滑动 | 月度动态 | Phase B 价格数据 ✅ |
| **stock↔stock (sector)** | 同 GICS sector 全连接 | 静态 | Wikipedia ✅ |
| **news→stock (mentions)** | 当天新闻提及该股票 | 按日变化 | EODHD events ✅ |
| **stock↔stock (co-occurrence)** | 同一篇新闻提及两只股票 | 按日变化 | EODHD events ✅ |

### N2b. Graph Snapshots

- [x] Correlation 边: 月度滚动 (w=126, t=0.6), ~54 snapshots
- [x] Sector 边: 静态, 11 sectors, ~25K edges
- [x] News 边: 每个 trading day 一个 news→stock edge set
- [x] Co-occurrence 边: 每个 trading day, 从 events 计算同篇文章的 ticker pairs

### N2c. HGT Model Architecture

- [x] PyG HGTConv (2 layers, 4 heads)
- [x] Node types: stock (781-dim), news (771-dim)
- [x] Edge types: 4 种 (correlation, sector, news_mentions, co_occurrence)
- [x] Output: per-stock ranking score (regression head, 1-dim)
- [x] Loss: MSE on Z-score normalized forward returns
- [x] Residual connection + LayerNorm per HGT layer

### N2d. Jaccard Audit (Edge Dynamics)

- [x] 计算相邻月份 correlation 边集的 Jaccard similarity
- [x] 报告 mean/std/min Jaccard + min日期

→ progress: `2026-03-05-d` | analysis: N/A

---

## Phase N3: Baseline Matrix (Ranking Evaluation) ✅ CODE WRITTEN

**目标**: 建立 baseline 梯队，验证各组件的增量贡献

**Code**: `v3_ranking_pipeline.ipynb` Cells 10-14

### N3a. Non-GNN Baselines

| Baseline | 特征 | 用途 |
|----------|------|------|
| LR (price only) | 9-dim momentum/vol | 纯价格线性 baseline |
| LR (price + news) | 9-dim + 771-dim FinBERT | 新闻增量 |
| XGBoost (all) | 781-dim | 非线性上限 (不含图) |
| LightGBM (all) | 781-dim | Qlib benchmark 标准 |

### N3b. GNN Ablations

| Experiment | Model | 边类型 | Horizon |
|-----------|-------|--------|---------|
| GNN-A1 | HGT | correlation only | 5d (MASTER default) |
| GNN-A2 | HGT | correlation + sector | 5d |
| GNN-A3 | HGT | all 4 edge types | 5d |
| GNN-A4 | GraphSAGE | all 4 edge types | 5d |
| GNN-A5 | GAT | all 4 edge types | 5d |

### N3c. Evaluation Metrics (Ranking)

所有模型报告:
- **IC** (Spearman correlation, daily cross-sectional)
- **ICIR** (IC / std(IC), 稳定性)
- **Rank IC** (rank-based variant)
- Long-only top-30 portfolio: annualized return, Sharpe ratio
- Long-short top/bottom-30: annualized return, Sharpe, max drawdown
- **扣除 15bps round-trip transaction cost**

### N3d. Go/Stop Gate

| 判定 | 条件 | 行动 |
|------|------|------|
| **Go** | 任一模型 IC > 0.03 (daily) 或 Long-short Sharpe > 0.5 | 进入 N4+N5 |
| **Stop** | 所有模型 IC ≈ 0 | 写 negative result 论文 (ranking 也无法预测) |

→ progress: `2026-03-05-d` | analysis: `2026-03-05-e`

**2026-03-05-e Colab Results**: Go/Stop → **GO** (Sharpe 1.038 > 0.5). Best model: A5 GAT (corr+sector) IC=0.02054. Worst: A3 HGT (all 4 edges) IC=0.00432. News/cooccur edges add noise.

---

## Phase N4: Horizon Ablation ✅ COMPLETE ← 论文贡献点

**目标**: 系统比较 6 种预测 horizon，填补文献空白

**Code**: `v3_ranking_pipeline.ipynb` Cell 15

### N4 Results (Colab Run 2, 2026-03-06)

| Horizon | GAT IC | GAT ICIR | GAT Sharpe | LGBM IC | LGBM Sharpe |
|---------|--------|----------|------------|---------|-------------|
| 1d | -0.00104 | -0.013 | 2.468 | 0.00368 | 2.918 |
| 5d | 0.02334 | 0.227 | 1.568 | 0.00828 | 0.773 |
| 10d | **0.03854** | 0.320 | 1.196 | 0.01349 | 0.644 |
| **21d** | **0.04420** | **0.374** | **1.203** | 0.01513 | 0.468 |
| 42d | -0.00912 | -0.144 | 0.071 | 0.03679 | 0.668 |
| 63d | -0.00838 | -0.118 | 0.487 | 0.05207 | 1.256 |

**Key findings**:
- [x] IC vs horizon: inverted-U, **peak at 21d** (IC=0.04420)
- [x] ICIR vs horizon: same pattern, peak 21d (ICIR=0.374)
- [x] GAT > LGBM at 5d-21d; LGBM > GAT at 42d-63d (cross pattern)
- [ ] News contribution analysis (有新闻 vs 无新闻) — not yet done

→ progress: `2026-03-06-b` | analysis: `2026-03-06-b`

---

## Phase N5: Selective Prediction ✅ COMPLETE — SelectiveNet FAILED

**Code**: `v3_ranking_pipeline.ipynb` Cells 16-18

### N5 Results (Colab Run 2, 21d horizon)

| Method | IC | ICIR | Sharpe | Ann_LS_net |
|--------|-----|------|--------|------------|
| Full (100%) | **0.05595** | **0.463** | **1.328** | **16.48%** |
| Threshold @20% | 0.03070 | 0.324 | 0.724 | 5.85% |
| SelectiveNet @20% | **-0.02414** | -0.256 | -0.536 | -9.00% |

**Status**:
- [x] Threshold baseline: works (IC improves at lower coverage)
- [x] SelectiveNet: **FAILED** — negative IC at all coverage levels
- [x] Jaccard overlap: SelectiveNet selects different (worse) stocks than threshold
- [x] Coverage converged to 31% (target 20%) — lambda insufficient
- [ ] SelectiveNet improvement or replacement — needs discussion

→ progress: `2026-03-06-b` | analysis: `2026-03-06-b`

---

## Phase N6: Paper Polish ← PENDING (after Colab run)

### N6a. Walk-Forward Validation

- [ ] Fold 1: Train 2021→2023, Val early-2024, Test late-2024
- [ ] Fold 2: Train 2021→2024, Val early-2025, Test late-2025
- [ ] Permutation test: 打乱股票代码 / 打乱标签

### N6b. Figures

- [ ] Fig 1: Dynamic graph evolution (edge count + regime annotations) — Phase B+ 已完成
- [ ] Fig 2: Horizon Ablation — IC/Sharpe vs prediction horizon (核心图)
- [ ] Fig 3: Coverage-IC tradeoff curve (Selective prediction 核心图)
- [ ] Fig 4: HGT attention weights — 哪种边类型最重要
- [ ] Fig 5: Per-month coverage + VIX overlay (SelectiveNet regime-awareness)
- [ ] Fig 6: Long-short cumulative return curve (扣费后)

### N6c. Tables

- [ ] Tab 1: Full baseline matrix (all models × all horizons × IC/ICIR/Sharpe)
- [ ] Tab 2: GNN ablation (edge type ablation)
- [ ] Tab 3: Selective prediction comparison (threshold vs SelectiveNet @ multiple coverages)
- [ ] Tab 4: Transaction cost analysis (break-even cost)

→ progress: TBD | analysis: TBD

---

## 最低可发表标准 (v3) — UPDATED 2026-03-06

| 指标 | 目标 | 实际值 | 状态 |
|------|------|--------|------|
| 任一 horizon IC > 0.03 | 超越随机排名 | **GAT 21d IC=0.04420** | ✅ |
| GNN IC > LightGBM IC | 图有增量 | **21d: 0.044 vs 0.015 (2.9×)** | ✅ |
| Selective IC@20% > Full IC | Selective 有增量 | Thr @20%: 0.031 < Full 0.056 | ❌ |
| Long-short Sharpe > 0.5 (扣费后) | 经济显著 | GAT 21d **Ann_LS_net=15.11%** | ✅ |
| Horizon ablation 有明确模式 | 文献贡献 | **倒U型, peak 21d** | ✅ |

**4/5 达标。** SelectiveNet 贡献点失败，需要替代方案。

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-27 | FinBERT (not Fin-E5/voyage) | Clean ablation; upgrade later |
| 2026-02-27 | Full-batch on A100 | No torch-sparse; 80GB sufficient |
| 2026-02-27 | Diagnostics before changes | AUC ≈ 0.50 → find where signal weak first |
| 2026-03-03 | Literature confirms AUC~0.50 expected | Large-universe + FinBERT + S&P500 = near-random for binary direction |
| 2026-03-03 | LLM 选 GPT-4o-mini | 论文可引用性最强 + structured JSON 最可靠 |
| 2026-03-03 | Equal-weight market return (非 SPY) | SPY 不在 prices 文件中 |
| 2026-03-03 | Sentiment 聚合用 mean (非 max) | 简单高效 |
| 2026-03-04 | Phase 2 LLM STOP confirmed | GPT-4o-mini delta=+0.0009, skip full-scale |
| 2026-03-05 | **换赛道: binary direction → ranking** | 所有 SOTA (MASTER/FinMamba/MDGNN) 用 ranking，binary AUC=0.50 是 EMH 预期结果 |
| 2026-03-05 | **Calendar-driven 替代 event-driven** | 主流做法；无新闻用 zero vector；避免 event-level 噪声 |
| 2026-03-05 | **HGT 替代 GraphSAGE** | 异构图 attention 可区分不同边类型权重；可解释性强 |
| 2026-03-05 | **4 种边类型 (不加外部数据)** | correlation(动态)+sector(静态)+news_mentions+co_occurrence; Multi-GCGRU 发现 co-occurrence > 持股关系 |
| 2026-03-05 | **动态图参数: w=126, t=0.6** | Phase B 分析: density 6%, std=0.064 (稳定), 125 components (适中) |
| 2026-03-05 | **Horizon ablation: 1d/5d/10d/21d/42d/63d** | 无论文做过系统比较；覆盖 short-term reversal 到 classic momentum 全谱系 |
| 2026-03-05 | **不加 30d/60d**: 21d 和 42d 已覆盖 | 21d≈1月, 42d≈2月, 与 30d/60d 高度重叠；用交易日而非日历日更精确 |
| 2026-03-05 | **DASF-Net "3-day optimal" 不成立** | 指 input aggregation window 非 prediction horizon; 仅 12 股; 无 horizon ablation |
| 2026-03-05 | **Selective prediction 保留为核心创新** | GNN + SelectiveNet 组合 = 文献空白; 高风险高回报 |
| 2026-03-05 | **Supply chain/持仓边留作 future work** | 需 FactSet($$)/SEC 13F(工程量大); 先验证框架再加数据 |
| 2026-03-06 | **News/cooccur edges 有害，仅保留 corr+sector** | A3(all 4 edges) IC=0.00432 最差; A5(corr+sector) IC=0.02054 最好; news edges 加噪声 |
| 2026-03-06 | **GAT 替代 HGT 作为最佳架构** | GAT IC=0.02054 > SAGE 0.01571 > HGT 0.01177 (同edge config); 简单attention更robust |
| 2026-03-06 | **N4 必须用 GAT(corr+sector) 重跑** | 原代码用 HGT(all edges)=最差config; 已确认需修改 |
| 2026-03-06 | **N4/N5 代码已改为 GAT(corr+sector)** | Cell 12: RankingGNN+get_stock_embeddings; Cell 15: GAT+corr+sector; Cell 16: SelectiveRankingGAT; Cell 17-18: 同构图 |
| 2026-03-06 | **GAT 优于 HGT 的原因** | (1)参数效率:弱信号下少参数泛化更好 (2)edge type区分无用:corr+sector本质都是"相关" (3)news dummy节点加噪声 (4)GAT attention更稳定 |
| 2026-03-06 | **GAT 21d IC=0.04420 超过0.03门槛** | 倒U型模式: 10d-21d是GAT sweet spot; 1d和42d-63d信号消失 |
| 2026-03-06 | **SelectiveNet 失败: 选择头反向选择** | 所有coverage 5%-50%负IC; 2-stage训练可能是原因; 需要讨论替代方案 |
| 2026-03-06 | **训练稳定性是关键问题** | GAT N3 IC跨run CV=105% (0.006-0.021); SAGE最稳定CV=2%; Walk-forward CV必须做 |
| 2026-03-06 | **Full SelectiveRankingGAT (100%) IC=0.05595** | 3-head auxiliary loss提供正则化; 比单纯RankingGNN的0.04420更好 |

## Current Status (2026-03-06)

**Phase**: N3-N5 全部完成
**关键结果**:
- GAT 21d IC=0.04420 (> 0.03 ✅), ICIR=0.374, Ann_LS_net=15.11%
- Horizon inverted-U pattern: GAT peaks at 10d-21d
- SelectiveNet FAILED (negative IC at all coverages)
- Full SelectiveRankingGAT IC=0.05595 (best overall)
- 训练稳定性有问题: GAT IC CV=105%

**4/5 最低发表标准达标。**

**Next (优先级排序)**:
1. **Walk-forward CV** — 解决训练稳定性 (最重要!)
2. **多次重复实验** — GAT 21d 跑 5 次取均值±std
3. **SelectiveNet 讨论** — 改进 or 报告 negative finding or 改用 threshold
4. **论文图表** — Horizon ablation plot 已有; selective analysis plot 已有
5. **交易成本敏感性** — tc = 5/10/15/20/30 bps

**Blockers**: ~~需与H博士讨论 SelectiveNet 策略~~ ✅ 已讨论 (2026-04-07)

---

## 2026-04-07-a: 6 周完整计划确定 ← CURRENT

综合 plan.md 路线 + critique 文档，制定了 6 周扎实计划。详见 `.claude/plans/mellow-puzzling-pie.md`。

### Week 1: 稳定性验证 ✅ COMPLETE (2026-04-08)
- [x] Task 1.1: Multi-seed GAT 21d — IC=0.032±0.018 (CV=55%), 3/5 > 0.03
- [x] Task 1.2: Multi-seed LightGBM 21d — IC=0.014±0.002 (CV=12.7%), 极稳定
- [x] Task 1.3: 训练诊断 — 跳过 (需要完整 GAT run 数据, 后续补)
- [x] Task 1.4: LSTM=0.023, MLP=0.023 (序列建模无增量; 图结构+40% IC)

### Week 1.5: Architecture Ablation ✅ COMPLETE (2026-04-08)
- [x] SAGE-Sum 21d × 5 seeds: IC=0.04766±0.00237 (**CV=5.0%**, 最稳定!)
- [x] TransformerConv 21d × 5 seeds: IC=0.02448±0.02248 (CV=91.8%, 最不稳定)
- [x] 关键发现: SAGE-Sum > SAGE-Mean > GAT > Transformer (by single-seed IC)
- [x] 关键发现: Sum 聚合解决了 SAGE-Mean 的 CV=62% 不稳定问题

### Week 2: Walk-Forward CV + Ablation ✅ COMPLETE (2026-04-09)
- [x] Task 2.1: Walk-forward CV — SAGE-Mean IC=0.045, SAGE-Sum IC=0.048 (both PASS >0.03)
- [x] Task 2.2: Feature ablation — price-only SAGE/SAGE-Sum/MLP/LGB done
- [x] Task 2.3: News contribution — no-news IC=0.059 >> with-news IC=0.008 (FinBERT hurts!)
- **意外发现**: Price-only features 在 Fold 0 上 IC 更高; FinBERT embedding 可能是噪声
- **SAGE-Sum Sharpe 异常**: IC 高但 Sharpe 负值, 需要调查

### Week 3: Diagnostics + SelectiveNet + 经济显著性 ✅ COMPLETE (2026-04-10)
- [x] Task 3.0: IC-Sharpe 诊断 — root cause = sum aggregation sector concentration
- [x] Task 3.0b: Non-overlapping 21d rebalancing + turnover-based TC
- [x] Task 3.0c: Comprehensive re-evaluation (9 models, all cached predictions)
- [x] Task 3.1: SelectiveNet 三策略 — Threshold wins (IC=0.082@10%), E2E fails, Vol-Cal mixed
- [x] Task 3.2: 交易成本敏感性 (0-30 bps) — SAGE-Mean @30bps Sharpe=1.94
- [x] Task 3.3: Permutation test (1000 shuffles) — p<0.001 ALL models
- [x] Task 3.4: 论文图表初版 — 7 figures generated

### Week 4: Qwen 特征 + 图表
- [ ] Task 4.1: Qwen 3.6 结构化特征 (~10万条, ~$26)
- [ ] Task 4.2: Consistency test (500条 × 5次)
- [ ] Task 4.3: 论文图表初版

### Week 5: 整合 + 论文初稿
- [ ] Task 5.1: Qwen 特征整合管道
- [ ] Task 5.2: LLM-based 图构建 (可选)
- [ ] Task 5.3: 论文初稿

### Week 6: 论文完善
- [ ] Task 6.1-6.3: 表格 + 图表 + 定稿

### 关键决策 (2026-04-07)

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-07 | 做扎实 4-6 周, 不急于收尾 | 论文竞争力 > 速度 |
| 2026-04-07 | SelectiveNet: 先修后报 (三策略) | 成功=贡献点; 失败=negative finding, 两种都有价值 |
| 2026-04-07 | 投稿目标: ICAIF 2026 / FinNLP Workshop | 金融领域 workshop, 更注重 insight 而非 novelty |
| 2026-04-07 | Qwen 3.6 结构化特征纳入 (~$26) | 成本可接受; ablation 需要 LLM vs FinBERT 对比 |
| 2026-04-07 | Notebook 拆分为 4 个 | 单一 notebook 太大, 不利于维护和实验隔离 |
| 2026-04-07 | 论文定位: "when does graph help" 系统研究 | 降低 overclaim 风险, 更诚实; critique 建议 |
| 2026-04-08 | GAT 信号真实但不稳定 (CV=55%) | 5-seed mean IC=0.032>0.03, 但 2/5 seeds 低于阈值 |
| 2026-04-08 | 图结构贡献确认: GAT>MLP +40% IC | 跨 seed 稳定的增量, 不是运气 |
| 2026-04-08 | LSTM 可从论文删除 | LSTM≈MLP, 序列建模对已有 rolling 特征无增量 |
| 2026-04-08 | SAGE multi-seed: IC=0.035±0.022, CV=62% | SAGE 也不稳定! 但 Sharpe 更高 (1.21 vs 0.84), 更鲁棒 |
| 2026-04-08 | SAGE Ensemble IC=0.052, Sharpe=1.344 | 5-seed 平均非常强, 超过任何单次运行 |
| 2026-04-08 | 建议: SAGE 为主模型 + ensemble | 更高 Sharpe, 更鲁棒初始化, 论文同时报两者对比 |
| 2026-04-08 | SAGE-Sum 是最稳定架构 (CV=5%) | 5/5 seeds 全部 >0.044; Mean 的 CV=62% 问题被 Sum 聚合解决 |
| 2026-04-08 | Transformer 极不稳定 (CV=91.8%) | IC 从 -0.008 到 0.053; 但 ensemble IC=0.053 最强 |
| 2026-04-08 | Week 2 同时跑 SAGE-Mean + SAGE-Sum 对比 | H博士要求: 时间不是问题, 两者都跑 |
| 2026-04-08 | 验证标准 LOCKED, 不允许修改 | H博士明确要求: IC>0.03, Sharpe>0.5 等阈值不可调 |
| 2026-04-08 | 代码必须模块化、可调用、可维护 | H博士明确要求: 后续 4-6 周需要反复迭代 |
| 2026-04-09 | Walk-Forward 验证通过 (SAGE-Mean 0.045, SAGE-Sum 0.048) | 两种架构跨 3 个时期均 > 0.03 阈值 |
| 2026-04-09 | FinBERT embedding 对预测可能有害 | News IC=0.008 << No-news IC=0.059; LGB price-only > LGB all |
| 2026-04-09 | SAGE-Sum Sharpe 异常 (IC高但Sharpe低/负) 需调查 | price-only: IC=0.069 但 Sharpe=-0.94; 可能是预测分布问题 |
| 2026-04-10 | **SAGE-Sum Sharpe 异常 root cause: sector concentration** | Sum aggregation 放大 sector 度数效应, LONG HHI=0.877, 只含 3/11 sectors; Mean 的 HHI=0.214 |
| 2026-04-10 | **SAGE-Mean 是论文主模型** | Non-overlap Sharpe=2.18, 所有 TC 水平下盈利, portfolio 跨 11 sectors 分散 |
| 2026-04-10 | **SAGE-Sum IC 高但经济无用** | IC=0.063 是真实的 ranking quality, 但 portfolio 集中导致亏损 — 论文作为 aggregation finding 报告 |
| 2026-04-10 | **Non-overlapping 21d 调仓为论文标准** | H博士确认更真实; SAGE-Mean Sharpe 1.17→2.18; 约 6 独立观测/fold |
| 2026-04-10 | **TC 3.2 已完成** | 0-30 bps sensitivity done; SAGE-Mean @30bps Sharpe=1.94; break-even > 30bps |
| 2026-04-10 | **FinBERT 保留作 ablation finding, Week 4 用 Qwen 验证** | H博士确认 |
| 2026-04-10 | **Permutation test: p<0.001 ALL models** | 1000 shuffles; real IC 0.032-0.037 vs shuffled 0.000±0.004; signal is real |
| 2026-04-10 | **SelectiveNet: Threshold > E2E > Vol-Cal** | |pred| 作为 confidence → IC=0.082@10% cov; E2E degrades IC; learned selection 在弱信号下失败 |
| 2026-04-10 | **Threshold 作为论文主要 selective method** | 简单有效, 不需要额外训练; E2E 和 Vol-Cal 作为 negative finding |
| 2026-04-10 | **Price-only Sharpe >> all-features Sharpe** | Non-overlap: price SAGE=2.18, all SAGE ens=1.27; 论文应同时报告两种 |
| 2026-04-10 | **MLP price-only Sharpe=2.59 (单 seed)** | 高于 SAGE, 但只有 1 seed 在 Fold 0; 需要 walk-forward 验证才能下结论 |
| 2026-04-10 | **Week 3 全部完成** | SelectiveNet + Permutation + TC + 图表 + 诊断 全部 done |

→ progress: `2026-04-10-e` | analysis: `2026-04-10-d`

---

## 2026-04-13-a: Codex Review Fix + True MLP Baseline ← RUNNING

### Background
Codex rescue agent 审查了 C1/C2/C3 修复，发现 5 个额外问题（2 Critical, 2 High, 1 Medium）。
所有修复已实现，v3 实验正在运行 (90 runs: 6 models × 3 seeds × 5 folds)。

### What's Running
- PID 71233, log: `experiments/wf5_v3_log.txt`
- 6 models: SAGE-Mean, NoGraph (renamed from old "MLP"), True MLP — each × {all, price}
- 3 seeds × 5 folds = 15 runs per model = 90 total
- C4 FIX: test evaluation purge (last 21 days removed from eval)
- Enhanced summary: median IC, Wilcoxon paired tests, 3-way comparison

### After Experiment Completes
- [ ] Analyze 3-way results: SAGE vs NoGraph vs True MLP
- [ ] Update analysis.md with new findings
- [ ] Determine if SAGE > MLP conclusion holds with true MLP baseline
- [ ] Reassess paper positioning (Option A/B/C from adversarial review)
- [ ] Decide which adversarial review P0 experiments to run next

### Decision Log Additions

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-12 | C1/C2/C3 data leakage fixes | Codex audit discovered 3 sources; all fixed |
| 2026-04-12 | Results reversal: SAGE > MLP (post-fix) | Pre-fix MLP advantage was leakage artifact |
| 2026-04-12 | ListNet improves IC +38% | But Wilcoxon p=0.72, not significant |
| 2026-04-13 | "MLP" renamed to "NoGraph" | Was actually empty-edge SAGEConv, not true MLP |
| 2026-04-13 | True RankingMLP class added | nn.Linear layers, same depth/width as GNN for fair comparison |
| 2026-04-13 | All models now run 3 seeds | Previous MLP had only seed=42 — unfair comparison |
| 2026-04-13 | C4 FIX: test eval purge | Test labels of last 21d overflow quarter boundary |
| 2026-04-13 | Old CSV archived (wf5_results_v2_BEFORE_TRUE_MLP.csv) | Prevent resume contamination with wrong model type |

→ progress: `2026-04-13-a` | analysis: pending

---

## 2026-04-15-a/e: SEC 10-K/10-Q Text Features — Gate 1 STOP

### 已完成
- [x] 文献调研 (40+ 论文, `archived/docs/literature_review.md`)
- [x] 确认 research gap: 10-K/10-Q + GNN 无人做过
- [x] 三层方案 + Go/Stop Gates 设计 (`archived/plans/plan_sec_text_features.md`)
- [x] SEC filing 数据收集 (11,988 records, 500 tickers, 2019-2025)
- [x] 4 轮 section 提取修复 (Item 1A 95.4%, Item 7 93-95%)
- [x] Layer 1 Lazy Prices TF-IDF 计算 (same-type, pre-test fit, median fill)
- [x] Claude + Codex 双重验证通过

### Gate 1 实验结果 ✅ STOP (2026-04-15)
- [x] Gate 1 实验: price+lazy vs price-only (Fold 0, 21 runs)
- [x] Go/Stop 决策: **STOP** — SEC L1 对 NN 有害 (SAGE -61%, MLP -34%)
- [x] 单特征 ablation: `days_since_filing` 是毒源 (IC: +0.037→-0.004)
- [x] LGB 唯一不恶化 (+17%), 但增量不足以继续
- [ ] ~~Layer 2: FinBERT sentence sentiment~~ — CANCELLED (Gate 1 STOP)
- [ ] ~~Layer 3: Qwen structured output~~ — CANCELLED (Gate 1 STOP)

### Decision Log Additions

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-15 | Same-type pairs only for Lazy Prices | Cross-type (10-K↔10-Q) sim=0.56 是结构差异不是信号; 原始论文也只比同类型 |
| 2026-04-15 | TF-IDF fit on pre-test filings only | 防止 test period 词频信息泄露; 2-line fix 但消除 reviewer objection |
| 2026-04-15 | Median fill (0.88) for missing data | 0.5 让缺失数据看起来像"变化最大"; median 代表"平均水平" |
| 2026-04-15 | SEC text features 独立于 pipeline 修复 | Layer 1 特征已计算好, 等 pipeline 稳定后直接跑实验 |
| 2026-04-15 | **Gate 1 STOP: SEC L1 Lazy Prices 无效** | SAGE IC -61%, MLP -34%, LGB +17%; days_since_filing 是毒源 (scale 0-7 >> price ±0.5); 跨模型一致恶化 |
| 2026-04-15 | Layer 2/3 CANCELLED | Gate 1 STOP; 同样的 carry-forward 稀疏结构会影响所有 SEC 特征 |

→ progress: `2026-04-15-a` `2026-04-15-b` `2026-04-15-c` | analysis: `2026-04-15-c`

---

## 2026-04-15-d: Pre-fix 实验重跑 — 脚本已创建, 待 Colab 执行

### 背景
Experiments 1-4 (horizon ablation, architecture comparison, walk-forward, permutation test) 在 C1/C2/C3 修复前运行, 结果不可信。已与 Codex 讨论并创建 3 个重跑脚本。

### 执行计划 (Colab 并行)

| 脚本 | Runs | 预估时间 | Colab Instance |
|------|------|---------|----------------|
| `run_horizon_ablation.py` | 360 | ~15-20h | Instance 1 |
| `run_arch_comparison.py` | 150 | ~8-10h | Instance 2 |
| `run_permutation_v2.py` | N/A | ~10min | After Instance 1 |

### 产出文件

| 文件 | 内容 |
|------|------|
| `experiments/horizon_ablation_results.csv` | 360 行: model × horizon × seed × fold |
| `experiments/arch_comparison_results.csv` | 150 行: model × seed × fold |
| `experiments/horizon_preds/*.npy` | 21d 预测缓存 (for permutation) |
| `experiments/permutation_v2_results.csv` | 排列检验结果 |

### 待执行
- [ ] 上传 2 个脚本到 Google Drive
- [ ] Colab Instance 1: run_horizon_ablation.py
- [ ] Colab Instance 2: run_arch_comparison.py
- [ ] 完成后: run_permutation_v2.py
- [ ] 分析结果, 更新 analysis.md
- [ ] 更新论文图表

### Decision Log Additions

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-15 | Horizon ablation 两套特征集完整跑 | Price-only 是更强 setting, 论文图表需完整曲线 |
| 2026-04-15 | 架构比较仅 21d | 21d 是选定 horizon, 不需要 900 run 全矩阵 |
| 2026-04-15 | SAGE-Sum 保留在架构比较中 | IC-Sharpe 分离的教学案例, reviewer 会问为什么没有 |
| 2026-04-15 | 统计检验: Wilcoxon 主 + DM 辅 | Codex 建议; DM 用 HAC 方差, 处理 IC 序列相关性 |
| 2026-04-15 | C1 purge 自适应 horizon | 63d purge 多于 21d, 但保证无 label 前瞻泄露 |

→ progress: `2026-04-15-d` | analysis: N/A

---

## 2026-04-16-a: Phase 5 — Feature Expansion + Rolling Graph + VIX Overlay

> Codex 已审核本方案。核心结论：特征族缺失是瓶颈，OHLCV 数据需下载，排期上 pending reruns 先完成。

### 背景

当前 9 维特征（ret_mean, ret_std, momentum × [5,10,21]d）全部来自 adjusted close。
文献（Alpha158 LightGBM importance + Gu-Kelly-Xiu 2020 NN importance）显示我们缺少高价值特征族。
SEC 10-K/10-Q 文本特征 Gate 1 STOP（2026-04-15）。宏观指标作为 GNN 节点特征无效（同一天全股票相同值，message passing 无法利用）。

### 排期：Pending Reruns ✅ COMPLETE (2026-04-16)

**Step 0 已完成**: horizon ablation (360 runs) + arch comparison (150 runs) + permutation v2 (16 models × 1000 shuffles)。9 维基线结果确认，可进入特征扩展。

```
Phase 5 排期:
Step 0: 完成 pending reruns (horizon + arch + permutation) ✅ DONE
Step 1: 下载 OHLCV 数据 ← NEXT
Step 2: 实现 5 个新特征 + 截面归一化
Step 3: 特征扩展实验 (SAGE-Mean 14 dims vs 9 dims)
Step 4: Rolling graph 实验
Step 5: VIX regime overlay 实验
```

---

### Priority 1: Alpha158/GKX Feature Expansion (9→14 dims)

#### 数据现状与缺口

| 数据 | 现有？ | 需要？ | 来源 |
|------|--------|--------|------|
| Adjusted Close | ✅ `sp500_5y_prices.csv` | — | EODHD |
| Volume | ❌ | dolvol, CORR5 | EODHD API (需下载) |
| Adjusted OHLC | ❌ | RSV5 | EODHD `splitadjusted` API (需验证) |

#### 5 个新特征（按 Codex 优先级排序）

| # | 特征 | 公式 | 所需数据 | Codex 评级 |
|---|------|------|---------|-----------|
| 1 | **mom12m** | `prices.shift(22) / prices.shift(252) - 1` | close only | 最高优先 |
| 2 | **maxret** | `returns.rolling(21).max().shift(1)` | close only | 高 |
| 3 | **dolvol** | `log(mean(close * volume, 63d)).shift(1)` | close + volume | 高 |
| 4 | **CORR5** | `Corr(close, log(volume+1), 5).shift(1)` | close + volume | 中高 |
| 5 | **RSV5** | `(close-Min(low,5))/(Max(high,5)-Min(low,5)).shift(1)` | adj OHLC | 中（有复权风险） |

**Fallback**: 若 EODHD 不提供复权 OHLC → 放弃 RSV5 → 9+4=13 dims。
**Codex 警告**: EODHD 的 open/high/low 可能未复权，混用会在拆股附近产生伪极值。

#### 截面归一化（Codex 强烈建议）

```python
# 每日截面 robust z-score
for feat in feature_names:
    daily_vals = features[:, :, feat_idx]
    # Winsorize at 1st/99th percentile
    lo, hi = np.percentile(daily_vals[~np.isnan(daily_vals)], [1, 99])
    clipped = np.clip(daily_vals, lo, hi)
    # Z-score across stocks per day
    mu = np.nanmean(clipped, axis=1, keepdims=True)
    sigma = np.nanstd(clipped, axis=1, keepdims=True) + 1e-8
    features[:, :, feat_idx] = (clipped - mu) / sigma
```

#### NaN 处理规则

| 特征 | NaN 条件 | 处理 |
|------|---------|------|
| mom12m | 不满 252 天历史 | 保持 NaN → 截面归一化后填 0 |
| maxret | 不满 21 天 | 同上 |
| dolvol | 不满 63 天 | 同上 |
| CORR5 | 不满 5 天完整窗口 | 同上 |
| RSV5 | 分母=0（5天价格平坦） | 设为 NaN |

#### 实验设计

**Step 1 — 筛选（快速）**: 1 seed × 5 folds, 仅 SAGE-Mean
- 9-dim baseline vs 14-dim (或 13-dim if no OHLC)
- 逐个消融：每次加 1 个特征，观察 IC 变化

**Step 2 — 确认（如 Step 1 有效）**: 3 seeds × 5 folds, SAGE-Mean + MLP + LGB
- 完整 factorial: {9-dim, 14-dim} × {SAGE, MLP, LGB} × 3 seeds × 5 folds

**预估 runs**: 筛选 ~35 runs + 确认 ~45 runs = ~80 runs total
**预估时间**: ~8-12h Colab GPU

#### Go/Stop Gate

- **Go**: 14-dim SAGE IC ≥ 9-dim SAGE IC (无害即继续)
- **Stop**: 14-dim IC < 9-dim IC - 0.005 (显著恶化)

---

### Priority 2: Rolling Graph Update (测试期不冻结)

#### 当前问题

测试期使用训练期最后一个相关性快照，到测试期末可能 ~3 个月陈旧。
Codex 建议：主方案改为滚动更新，冻结图作为 robustness check。

#### 实现

```python
# 当前（冻结）:
if FROZEN_CORR_SI is not None:
    edge_lists.append(corr_snapshots[FROZEN_CORR_SI])

# 改为（滚动，仅用 t-1 数据）:
snap = corr_day_to_snapshot.get(day_idx, 0)
edge_lists.append(corr_snapshots[snap])
# 注: snapshot_points 已确保每个 snapshot 只用 <= t_end 的数据
```

**泄露检查**: 每个 snapshot 用的是 `returns.iloc[t_end - corr_w : t_end]`，其中 `t_end <= day_idx`。模型权重仍在 fold 内冻结。无泄露。

**Codex 提醒的隐蔽风险**: 确保不会在一个 batch 中为整个测试块构建图时意外使用后期观测值。需逐日验证 `corr_day_to_snapshot` 映射。

#### 实验设计

- 3 seeds × 5 folds, SAGE-Mean only
- Frozen vs Rolling 对比（paired，相同 seed/fold）
- 报告两个版本的 IC, ICIR, Sharpe

**预估 runs**: 15 runs (rolling) + 对比已有的 frozen 结果
**预估时间**: ~3-5h Colab GPU

---

### Priority 3: VIX Regime Overlay (探索性)

#### 设计

```python
# 最简单的线性交互
# 在 validation set 上 fit Ridge:
#   final_score = w1 * gnn_score + w2 * gnn_score * vix_zscore
# 在 test set 上 evaluate

from sklearn.linear_model import Ridge
vix_daily = pd.read_csv('data/reference/vix_daily.csv')  # 从 Yahoo/FRED 下载
vix_zscore = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
```

#### 数据获取

VIX: Yahoo Finance `^VIX` daily close, 或 FRED `VIXCLS`。免费，日频，2020-2025 完整覆盖。

#### Codex 警告

- ~63 个测试日/fold，统计功效弱。合并 5 fold 约 315 天，可探测 partial R²≈0.025
- **只作为探索性结果，不作为核心结论**
- 若各 fold 效果不一致 → 报告 null result

#### 实验设计

- 用 Priority 1/2 的最佳 SAGE-Mean 预测作为 GNN score
- Ridge fit on validation, evaluate on test
- 对比：GNN only vs GNN × VIX interaction

**预估时间**: ~2-3h (主要 CPU，不需 GPU)

---

### 总时间线

| 步骤 | 工作 | 预估时间 | 依赖 |
|------|------|---------|------|
| Step 0 | Pending reruns (已有脚本) | 15-20h GPU | Colab 执行 |
| Step 1 | 下载 OHLCV + VIX 数据 | 2-3h | EODHD API key |
| Step 2 | 实现新特征 + 归一化 | 3-4h | Step 1 |
| Step 3 | P1 特征扩展实验 | 8-12h GPU | Step 0 + Step 2 |
| Step 4 | P2 Rolling graph | 3-5h GPU | Step 3 (可并行) |
| Step 5 | P3 VIX overlay | 2-3h CPU | Step 3 |
| Step 6 | 分析 + 更新 tri-doc | 2-3h | All above |
| **总计** | | **~35-50h** (其中 GPU ~30h) | |

**关键路径**: Step 0 → Step 1+2 (并行) → Step 3 → Step 4+5 (并行) → Step 6

---

### Decision Log Additions

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-16 | **Phase 5 三优先级确定**: (1) 特征扩展 (2) 滚动图 (3) VIX overlay | Codex + Claude 讨论; 特征族缺失是主要瓶颈 |
| 2026-04-16 | **SEC 文本→GNN 方向放弃** | Gate 1 STOP + 理论分析: 公司自身基本面特征无跨股票溢出性, GNN message passing 无法利用 |
| 2026-04-16 | **宏观指标不作 GNN 节点特征** | 同一天全股票相同值→message passing 后不变→等价于偏置项; Codex 确认文献无先例 |
| 2026-04-16 | **宏观数据用法: 图结构调节 > 仓位管理 > 节点特征** | Codex 推荐; FinMamba 和 ScienceDirect 2025 论文支持 |
| 2026-04-16 | **国会议员/13F 数据暂不推进** | 45-90天披露延迟+大盘股信号弱+季度频率与21d rebalance不匹配 |
| 2026-04-16 | **Pending reruns 先于特征扩展** | Codex 建议: 保持干净9维基线→再叠加新特征, 论文叙事更清晰 |
| 2026-04-16 | **截面归一化 (robust z-score) 必须加入** | Codex 强烈建议; Qlib 论文普遍使用; 对 volume/dolvol 等偏斜特征尤其重要 |
| 2026-04-16 | **RSV5 有复权风险, 需验证 EODHD adjusted OHLC** | 若不可用则放弃 RSV5, 改为 9+4=13 dims |
| 2026-04-16 | **VIX overlay 仅作探索性结果** | 315天 OOS 统计功效弱; 各 fold 效果不一致→报 null result |
| 2026-04-16 | **论文定位不变: "when does graph help" 系统研究** | 特征扩展和图改进是自然延伸, 不改变核心叙事 |

| 2026-04-16 | **Step 0 reruns 全部完成** | 360+150+16K shuffles; 9维基线确认, 可进入特征扩展 |
| 2026-04-16 | **"倒 U 型" 不成立** | 5-fold WF 下 peak 移到 63d, 但被 Fold 4 扭曲; 21d 最可靠 (唯一 bootstrap CI 排除 0) |
| 2026-04-16 | **Price-only 下架构无显著差异** | 5 种架构 Wilcoxon 全部 ns; GNN 的贡献不在架构选择 |
| 2026-04-16 | **Permutation v2 确认信号真实** | Per-day cross-sectional shuffle; price models 全 p<0.001; SAGE_all p=0.002, MLP_all p=1.0 |
| 2026-04-16 | **Fold 4 (Q2-2025) 系统性异常** | 所有模型极端表现; 论文必须报告 fold-by-fold + wide CI |

→ progress: `2026-04-16-b` | analysis: `2026-04-16-b`

---

## 2026-04-16-c: Phase 5 Step 0.5 诊断完成 → Plan 调整

> H博士 选 Option B (诊断先行)。三件套全部完成, **Diag 1 是 bombshell**。

### 核心发现
1. **Fold 4 = 市场 regime 压力测试**, 不是 bug: 54.3% 股票对 corr>0.5, 波动率 2× 其他期
2. **9 维特征实际 effective rank ≈ 3** (eigendecomp 实跑): momentum/vol/horizon-spread 三因子, 前 3 PC 解释 89.7%
3. **截面归一化 regime-dependent**: Fold 4 IC +0.006→+0.217 (拯救); Folds 0/1/3 −0.03 到 −0.10 (摧毁); 总体 Wilcoxon p=0.60 (ns) 完全掩盖交互效应

### 诊断产出

- [x] Step 0.5 — Diag 2/3/1 全部完成 (2026-04-16-c)

### 诊断结论 (事实层, 非决策)

1. **Fold 4 = 市场 regime 压力**: Q2-2025 54.3% 股票对 corr>0.5; 波动率 2× 其他 fold
2. **9 维特征 effective rank ≈ 3** (eigendecomp 实跑, 前 3 PC=89.7%): momentum/vol/horizon-spread 三因子
3. **SAGE-Mean 归一化 regime-dependent** (30 runs, 单架构): Fold 4 +0.21, Folds 0/1/3 -0.03 到 -0.10; 总体 p=0.60 ns
4. **Codex 指出 2 处计算缺陷已修复**: signed corr 替代 |corr|, 实跑 SVD 替代推断

### 待 H博士 决策 (诊断只提供证据, 不越权改 Phase 5 plan)

A. **Diag 1b — MLP + NoGraph 复现归一化实验 (~15 Colab-min)**: 建议运行, 仅 gate **Step 3 scope 决策** (不阻塞 Step 1 或 Step 2)。
   - 若 MLP/NoGraph 出现同样 regime-dependency → 机制是 input-scale saturation, 与 graph 无关
   - 若仅 SAGE-Mean 出现 → graph × normalization 交互
   - Diag 1b 结果仅影响 Step 3 是否加 raw-vs-norm 因子; Step 1 (OHLCV 下载) 和 Step 2 (新特征实现) 可独立推进

B. **Phase 5 Step 1 (OHLCV 下载) 是否立即启动?** 原计划未变, Diag 结果与此无关, 可立即启动。

C. **Step 3 scope 是否改变?** 原计划 ~80 runs (14-dim 对 9-dim, 单一 norm)。可能改为 raw+norm 并跑 (~160 runs) 或保持原计划。**此决策需要 Diag 1b 证据 + H博士 批准, 不自动推进。**

D. **新特征 (mom12m, dolvol, CORR5, RSV5, maxret) 的优先级判断**: Diag 3 提供了 PC 分析, 但各特征的真实 PC 载荷要等加入后实测。现在仅有"基于数据源的正交性假设", **不是已验证优先级**。

E. **VIX overlay (原 P3) 是否升级**: **不建议**。Diag 1 是 SAGE-only 单架构结果, 不直接证明 VIX 有效; 且 2026-04-16-a plan 里 "~315 OOS 天统计功效弱" 的事实未变。VIX 保持 P3 exploratory。

F. **论文是否加"normalization effects" section**: **暂不建议**。30 runs 单架构证据不足, 等 Diag 1b + 特征扩展结果再定。

### Decision Log Additions (仅记录事实与 H博士 已批准决策)

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-16 | Option B (诊断先行) | H博士 选择; 避免盲加特征的沉没成本 |
| 2026-04-16 | Diag 2/3/1 全部完成 | 本地 35s + Colab 13.2 min |
| 2026-04-16 | **特征 effective rank = 3 (eigendecomp 实跑)** | 事实性发现, 取代早先"5-6"推断 |
| 2026-04-16 | Codex 审查 2 轮, 计算 + 设计建议均已订正 | Round 1: signed corr, SVD 实跑; Round 2: 撤回越权设计建议 |
| 2026-04-16 | Diag 1b 列为 Step 3 scope 决策前的建议性 checkpoint (不阻塞 Step 1/2, 待 H博士 确认) | 机制确认成本低 (~15 min), 避免 Step 3 错误 scoping; 与 OHLCV 下载和特征实现无关 |

### 未作出的决策 (明确记录, 避免未来误读)

- ❌ **未决定** Codex "必加归一化" 结论是否推翻 — 需 Diag 1b 证据
- ❌ **未决定** Step 3 runs 是否翻倍 — 需 Diag 1b + H博士 批准
- ❌ **未决定** VIX overlay 升级优先级 — 无直接证据
- ❌ **未决定** 各新特征 ROI 排序 — 待实测 PC 载荷
- ❌ **未决定** 论文是否加 normalization section — 证据不足

→ progress: `2026-04-16-c` | analysis: `2026-04-16-c`

---

## 2026-04-16-d: Diag 1b + Step 1 + Step 2 完成, Step 3 待决策

### 完成项 (事实)
1. **Diag 1b 机制确认**: 归一化 regime-dependency 在 NoGraph/MLP 上复现, **不是 graph-specific**; Step 3 scope 保持原 80 runs 计划即可
2. **Step 1 OHLCV 下载**: yfinance, 500/500 对齐 > 0.9999 (AXON 单补), 0.7 min
3. **Step 2 新特征构建**: 5 特征 shape (1255,501,5), NaN rate 1.6-21%, 14 维 effective rank 从 3 升至 7
4. **新特征 PC 载荷实测, 颠覆先前假设**: mom12m 和 dolvol 高正交 (~0.99/0.98 orthogonal), maxret/RSV5 实际 mostly 冗余

### Step 3 实验方案 (待 H博士 批准)

基于 Diag 1b + PC 载荷实测, 建议:

**Scope**: 原计划 80 runs 不变, 不加 raw/norm 因子
- `{9-dim, 14-dim} × {SAGE, MLP, LGB} × 3 seeds × 5 folds` = 90 runs (含 LGB)
- 归一化模式: **raw** (与过去实验一致, 可直接对比先前 IC=0.032-0.048 baseline)
- Feature set: 老 9 维 + 新 5 维 = 14 维 (不做特征选择, 让模型自己学)

**预估时间 (Colab RTX Pro 6000)**:
- SAGE/MLP: ~20s/run × 60 runs (2 模型 × 10 configs) = ~20 min
- LGB: ~30s/run × 30 runs = ~15 min
- 总计 ~35-45 min

**Go/Stop gate** (per 2026-04-16-a plan, 未修改):
- **Go**: 14-dim SAGE IC ≥ 9-dim SAGE IC (无害即继续)
- **Stop**: 14-dim IC < 9-dim IC - 0.005 (显著恶化)

### Step 4/5 待决策 (保持原计划)
- Step 4 Rolling graph: 不变
- Step 5 VIX overlay: 保持 P3 exploratory, 不升级

### Decision Log Additions

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-16 | Step 1 改用 yfinance (放弃 EODHD) | EODHD 订阅异常, yfinance 500/500 alignment>0.9999 足够精度 |
| 2026-04-16 | **Diag 1b 确认 归一化 regime-dependency 是 input-scale 机制** | NoGraph/MLP 与 SAGE 同 pattern (14/15 cell 同号); message passing 排除 |
| 2026-04-16 | **Step 3 scope 保持 80-runs 原计划** | Diag 1b 结论→不需 raw/norm 因子交叉 |
| 2026-04-16 | **新特征 PC 载荷推翻我先前假设**: mom12m 高正交, RSV5 反而冗余 | 实测优于推断; mom12m 长 horizon 贡献新 PC 维度 |
| 2026-04-16 | **14 维 effective rank ≈ 7** (9 维是 3) | 新特征让信息维度近乎翻倍, 主要来自 mom12m + dolvol |

### 未作出的决策 (累积自 2026-04-16-c, 仍待)
- ❌ 论文是否加 "normalization effects" section (暂不建议, 2 段 discussion 合适)
- ❌ 是否删 RSV5 (实测 mostly 冗余, 但留着不伤害)
- ❌ VIX overlay 升级 (不建议)
- ❌ Full Alpha158 (暂不考虑)

→ progress: `2026-04-16-d` | analysis: `2026-04-16-d`

---

## 2026-04-17-a: Statistical Rigor Checklist — Missing Significance Tests

> **触发**: 盘点项目当前统计检验覆盖, 识别投稿 ICAIF 2026 / FinNLP 的标配缺口。
> **不改变**: Phase 5 Step 3 排期、特征扩展决策。此为独立 checklist, 实验代码冻结后一次性补齐。

### 当前已实现（保留）
| 检测 | 位置 |
|---|---|
| Paired Wilcoxon signed-rank (跨 seed×fold 模型对比) | `run_walkforward_5fold.py:626`, `run_gate1_experiment.py:670`, `run_phase5_step3_feature_expansion.py:387`, `run_diag1_normalization.py:416`, `run_diag1b_replication.py:395` |
| Permutation test (per-day cross-sectional shuffle, 16K) | `archived/scripts/run_permutation_v2.py` |
| Win rate (seed×fold 配对胜率) | `run_walkforward_5fold.py:646` |
| Spearman IC (主评估指标) | 所有实验脚本 |
| Kurtosis / Skew (特征/标签厚尾诊断) | `diagnostic_phase5_step0.py:153` |
| KS 2-sample test (train/test 分布漂移) | `diagnostic_phase5_step0.py:185` |

### 缺口（待补）

#### Must-Have（论文投稿必备）
- [ ] **IC t-stat with Newey-West HAC standard errors**
  - 对每个模型的 daily IC 序列计算 `t = mean(IC) / NW_SE`, lag=5 或 10
  - 处理 IC 序列自相关；金融顶刊标配，替代朴素 t = mean/std
  - 影响：Table 1 主结果表每个模型加一列 t-stat + p-value
- [ ] **Sharpe ratio bootstrap 95% CI**
  - Stationary bootstrap (Politis-Romano), block size ≈ 5-10 天
  - 10K resamples per model
  - 影响：Sharpe 数值后面加 `[lo, hi]` 置信区间, 避免点估计误导
- [ ] **Holm-Bonferroni 多重比较校正**
  - 当前 6 模型两两 Wilcoxon = 15 次检验, 未校正可能假阳性
  - 对所有 pairwise Wilcoxon p-value 做 Holm 调整, 报 `p_adj`
  - 影响：Section 4 architecture comparison 所有 p 值改为调整后

#### Nice-to-Have
- [ ] **Diebold-Mariano 检验**（两模型预测误差配对 HAC 方差）
  - 金融圈比 Wilcoxon 更标准; 处理 loss 序列自相关
  - 作为 Wilcoxon 的稳健性对照
- [ ] **Jarque-Bera / Anderson-Darling 正态性检验**
  - 对 daily IC 序列做正态性检验
  - 在 method section 一句话引用: "IC 序列非正态 (JB p<0.001), 故主检验用 Wilcoxon + bootstrap"

### 实现计划（1-2 天一次性）

1. ✅ **`utils/stats_tests.py` 已写完** (2026-04-17, **not yet run**):
   ```python
   newey_west_tstat(series, lag=None) -> (mean, nw_se, t, p)
   sharpe_bootstrap_ci(returns, n_boot=10000, block_mean=5, annualize=None) -> dict
   holm_bonferroni(pvals: dict) -> dict
   dm_test_classical(loss1, loss2, horizon=1, harvey_correction=True) -> (dm, p)
   dm_test_hac(loss1, loss2, horizon, lag=None) -> (dm, p)
   jarque_bera(series) -> (jb, p, skew, excess_kurt)
   ic_summary_row(daily_ic) -> dict  # one-stop
   ```
   - NW lag 默认 `floor(4 * (n/100)^(2/9))` (Newey-West 1994)
   - Stationary bootstrap (Politis-Romano 1994), 默认 block_mean=5 (daily IC)
   - **DM 两种变体（Codex 审查后拆分）**:
     - `dm_test_classical`: 矩形核, 截断在 `horizon-1`, Harvey-Leybourne-Newbold (1997) 校正 + Student-t(n-1); 短/中 horizon 用
     - `dm_test_hac`: Bartlett 核 (Newey-West), **`horizon` 为必填参数**, HAC bandwidth 自动 floored 在 `horizon-1` (Codex 复审: NW 自动规则对 n=60–300 只返 3–4, 会严重 under-truncate 21d ahead 的 structural autocorrelation), 无 Harvey 校正, N(0,1); 长 horizon 用 (如 21d ahead)
   - 所有函数纯函数, 返回 Python float, 可直接写 CSV
2. **待 H博士 批准后** backfill 到现有结果:
   - `experiments/wf5_results.csv` → 生成 `wf5_results_with_stats.csv`
   - `experiments/step3_feature_expansion_results.csv` 同理
   - Phase 5 Step 4/5 输出 CSV 直接调用新函数
3. 更新 `run_figures_tables.py`, 图表/Table 自动带置信区间和调整后 p 值

### 时机

**建议**: Phase 5 Step 3 实验完成后启动（不阻塞 Step 3）。
- Step 3 仍用现有 Wilcoxon 做 quick check
- 所有 Phase 5 结果冻结后, 一次性 backfill + 重生成图表
- 避免中途换检验方法导致数字版本混乱

### Go/Stop Gate（无 — 这是工程补全, 不是实验）

### Decision Log Additions

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-17 | 补齐 IC t+NW, Sharpe bootstrap CI, Holm-Bonferroni 三件套 | ICAIF/FinNLP 审稿标配; 当前仅 Wilcoxon 不足以支撑"显著性"叙事 |
| 2026-04-17 | DM + JB 列为 nice-to-have, 不阻塞主线 | 主检验已经有 Wilcoxon + permutation; DM 作稳健性对照, JB 仅用于 method justification |
| 2026-04-17 | 实现为 `utils/stats_tests.py` 模块, 非 notebook | 所有实验脚本调用; 避免重复代码 |
| 2026-04-17 | Backfill 时机: Phase 5 Step 3 完成后 | 避免中途换方法导致版本混乱 |
| 2026-04-17 | **`utils/stats_tests.py` 已写完, 未跑** | H博士 指示先写不跑; 等批准后 backfill 到 wf5 + step3 结果 |
| 2026-04-17 | **DM 拆为 classical + HAC 两版本** | Codex 审查发现原单一 `dm_test` 混淆了"HAC 截断带宽"与"预测 horizon"; H博士 决定两个都保留 |
| 2026-04-18 | **`dm_test_hac` `horizon` 改为必填, lag 自动 floor 在 h-1** | Codex 二次审查: NW 自动带宽对 n=60–300 只给 3–4, 对 21d ahead 会严重 under-truncate, 低估 LRV 导致假阳性 |

→ progress: TBD | analysis: TBD

---

## 2026-04-18-a: Phase 5 Step 3 最终方案 — Feature Pruning + Grouped Permutation (Plan Z++)

> **触发**: Plan X (incremental add 9→11→13) 与 Plan Y (guided backward from LGB) 均被 Codex 2 轮批判性评估否决。最终方案为 Plan Z++，已 Codex 共识。
>
> **叙事转向**: 从 "when does graph help on price features" → **"compact economically-grounded feature set beats redundant technical libraries for GNN stock ranking under regime shift"**。

### 先置状态

| 项目 | 状态 |
|------|------|
| ZTS yfinance 重抓 | ✅ 1255/1255, ret_corr=0.9998 |
| yfinance ohlcv parquet ZTS 污染列 (6 列 stringified tuple) | ⏳ 小清理, 执行前处理 |
| `sp500_5y_phase5_features.npy` ZTS 行 | ⏳ 需重跑 `build_phase5_features.py` 消除 ZTS dummy 0 |

### 方法论核心

1. **先语义折叠 3 对 duplicate**: `momentum_{5,10,21}d` 删除, 保留 `ret_mean_{5,10,21}d` (词典序 tie-break, 非 data-driven)。13 features → **10 features**。
2. **Complete-link 分组** (training-fold Spearman, 阈值 |ρ|≥0.6):
   - G_vol = {ret_std_5d, ret_std_10d, ret_std_21d} (pairwise 0.68-0.84, 过阈值)
   - G_ret_5d, G_ret_10d, G_ret_21d — 单独 (pairwise 0.47-0.69, `5d↔21d` 的 0.47 未过阈值)
   - G_maxret, G_mom12m, G_dolvol, G_corr5 — 单独 (max corr 与现有 G 均 < 0.6)
   - **最终 6-8 groups** (待 training-fold Spearman 实测确认; 全量 Pearson 估计给出 8 groups)
3. **Grouped permutation importance at inference** (non-retrain): 训练好的 SAGE + MLP on 10-feature full set, 推理时 shuffle 每组, 测 ΔIC。这是 ranking 主方法。
4. **Intra-group 成员选择**: domain reasoning + preregistered outcome-blind rule (不能用 outcome data 选 winner)。
5. **7 subsets retrain**:
   - S1: full 10 (all groups)
   - S2: top-4 groups (by grouped ΔIC ranking)
   - S3: top-3 groups
   - S4: top-2 groups
   - S5: top-1 group
   - S6: PC-representative 3D 非嵌套探针 (一个 feature from PC1 momentum + 一个 PC2 vol + 一个 PC3 horizon-spread)
   - S7: 原 9-dim (duplicate-kept) baseline — 与 Step 0 arch_comparison 直接可比
6. **Statistical validation**:
   - **主**: Hansen SPA (2005), daily IC differentials, 21-day block bootstrap, n_boot=10,000, 对比 S1 baseline
   - **副**: BH-FDR adjusted pairwise (exploratory only)

### 运行预算

| 部分 | runs | 时间 (M4 MPS) |
|------|------|----------|
| Part A: 10-dim full SAGE + MLP × 3 seeds × 5 folds (for permutation ranking) | 30 | ~3h |
| Part B: 2 models × 7 subsets × 3 seeds × 5 folds | 210 | ~20-22h |
| **合计** | **240** | **~24h** |

→ 本地 M4 可一夜+白天跑完, **不需要 Colab**, 除非 H博士 要求并行对比。

### 成果物

- `experiments/step3_plan_z/permutation_group_ranking.csv` — Part A 输出
- `experiments/step3_plan_z/subset_sweep_results.csv` — Part B 210 runs
- `experiments/step3_plan_z/hansen_spa_test.csv` — 主检验结果
- `plots/step3_ic_vs_dim.png` — IC-vs-group-count 曲线
- `plots/step3_permutation_ranking.png` — Part A 可视化

### Go/Stop Gate

- **Go-to-next**: IC vs dim 曲线有清晰非单调峰值 OR top-k < full 胜率 > 60% (fold×seed 级别) → 写论文
- **Null**: 曲线单调 (即全 10-dim 最好) → 回退到 simpler narrative (feature set justification), 不发表 "pruning" 主张
- **Stop**: 全 subset 都 null vs random → 回到 Phase 4 rethink

### Decision Log Additions

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-18 | Plan Z++ 采纳 (240 runs, feature pruning narrative) | Codex 2 轮辩论共识: inference-time grouped permutation 避开 LGB→NN transfer 陷阱; 7 subsets = 5 nested + 2 controls |
| 2026-04-18 | 3 对 momentum/ret_mean duplicate 语义折叠 (保留 ret_mean_Nd) | 非 data-driven 决定, 避 outcome-selection bias |
| 2026-04-18 | Complete-link \|ρ\|≥0.6 on training folds (非 single-link) | Codex: single-link 有链式风险, complete-link 更严 |
| 2026-04-18 | maxret 单独成组 (不进 G_vol) | 与 ret_std_5d 相关 0.52 < 0.6 阈值, 未过 complete-link |
| 2026-04-18 | Hansen SPA (主) + BH-FDR (exploratory) | Codex: SPA 是金融 model-search-under-dependence 的标准, Bonferroni 过保守 |
| 2026-04-18 | PC-representative 3D 非嵌套探针 (一个, 不加更多) | 防 preregistration 瓦解 + multiplicity 膨胀 |
| 2026-04-18 | M4 本地跑, 不需 Colab | 24h 可一夜+白天完成; Colab 会话漂移风险不值得 |

→ progress: `2026-04-18-a` | analysis: TBD (Step 3 跑完后)

---

*Last updated: 2026-04-18 (Plan Z++ 敲定, ZTS 重抓完成)*

---

## 2026-04-19-a: Phase 5 Step 3 Plan Z++ 完成 → 论文 narrative 锁定

> Step 3 完整执行完毕（241 runs, 5h M4 MPS）。详情见 progress `2026-04-19-a`。

### 核心结论（未决 → 已决）

| 议题 | 结论 |
|------|------|
| **论文定位** (Round 2 留的 defer) | **"Parsimonious economically-grounded features beat redundant technical libraries for GNN ranking under regime shift"** — 基于 S6 (IC +0.046) vs S7 (SAGE IC −0.048) 的跨符号差 0.095 IC 差异, 配合 Hansen SPA p_c=0.002 |
| Step 4/5 是否推进 | **暂停**：Step 3 已提供足够 paper-quality 主实验；先写 analysis/figures/tables，后补 SelectiveNet |
| 要不要做 Sensitivity 完整版 | **跳过**：便宜版（per-fold ranking Spearman）已揭示 per-fold importance 不稳定，写入 Limitations 即可 |

### 下一步 Phase 5 Step 4 候选（待 H博士 决策优先级）

- **A. 论文图表打磨 + 初稿**：Step 3 结果直接写方法+结果部分
- **B. 不需训练的 P0 分析**（R5/R6/R8/S7，adversarial review 遗留）：用 cached preds 补充 sector-neutral + coverage-Sharpe-turnover + multiple testing
- **C. SelectiveNet 叠加**（Step 5）：在 S6 best subset 上加 SelectiveNet threshold variant
- **D. 论文的 Related Work + Intro 撰写**
- **E. Rule 9 触发点 3: Codex 分析 review**（results 送审）

### Decision Log Additions

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-19 | Step 3 完成, 论文定位锁定为 "parsimonious features > redundant" | 4 个 Hansen SPA 中 3 个在 α=0.05 拒绝 null; S6 双模型 p<0.02 显著 |
| 2026-04-19 | S6 PC probe (mom12m+ret_mean_10d+ret_std_10d) 定为主力 subset | MLP IC=+0.046 (p=0.009), SAGE IC=+0.047 (p=0.014), 为聚合最优 (非 per-fold 最优, 详见 analysis.md Cross-fold stability) |
| 2026-04-19 | S7 9-dim baseline 显著负 (SAGE IC=-0.048) | 冗余 momentum_Nd 重复特征主动添噪, 支持去冗余决策 |
| 2026-04-19 | Per-fold ranking instability 写入 Limitations, 不做完整 sensitivity 重跑 | 便宜版已足够 disclose; 完整版额外 +210 runs 不值 |
| 2026-04-19 | Phase 5 Step 4 暂定为 "论文图表+初稿"(A) 或 "Codex 结果 review"(E), 待 H博士 选 | 主实验证据已足, paper-writing 优先级 vs 补充分析优先级待决 |
| 2026-04-19 | 叙事转向 (Codex R7 CRITICAL Q5): "time-unstable ranking-based pruning vs. more-generalizable compact PC-representative subsets" | S2-S5 降为 exploratory, S6 上主角位, Part A 用作"排名不稳"证据 |
| 2026-04-19 | 不声索 economic alpha, 只讲 predictive patterns | Codex R7 MAJOR Q1: Sharpe bootstrap CI 跨零, 无法支持经济显著 |
| 2026-04-19 | 防御性表述: "PC-representative design 更稳定", 不说 "optimal subset" | Codex R7 MAJOR Q2: S6 受 Diag 3 启发, 非完全独立确认 |
| 2026-04-19 | SAGE vs S1 p=0.076 诚实披露, 不声索"双架构均 0.05 显著" | Codex R7 MINOR Q3 |
| 2026-04-19 | S7 claims 限制在本研究 design space 内 | Codex R7 MAJOR Q4: S7 自参照, 非外部基线 |
| 2026-04-19 | 论文表述改为 "frozen analysis pipeline with preregistered subset construction rules", 不说 full preregistration | Codex R7 MAJOR Q6: 软预注册 |
| 2026-04-19 | Phase 5 Step 4 = Alpha158 外部 baseline (S8) | Codex R7 Q7 ONE 推荐: 直接解决 S7 自参照问题, 锚定 compact-vs-library |

→ progress: `2026-04-19-a` | analysis: `2026-04-19-a`

*Last updated: 2026-04-19 (Step 3 完, 进入 paper-writing 阶段)*

---

## 2026-04-20-a: Phase 5 Step 3 Part C 完成 → 叙事风险

> Alpha158 S8 外部 baseline 跑完。结果**挑战原 narrative**：以 S8 为 benchmark 的 Hansen SPA 不拒绝 non-superiority 零假设 (MLP p_c=0.551, SAGE p_c=0.551 per CSV；**非** equivalence 证明)。Fold 4 异常 +0.22 可能 leakage artifact。（2026-04-21-c 更正：此处早期版本写 p_c=0.55/0.59 + "统计无差异"，数字和解释双错——见 progress.md 2026-04-21-c。）

### 结果重要发现

- S8 Alpha158 158-feat: MLP IC=+0.041 (p=0.026), SAGE IC=+0.042 (p=0.025) — **本身显著 > 0**
- S6 vs S8 差距仅 0.005 IC, 统计上不可区分
- S8 Fold 4 IC=+0.22 (vs Fold 0-3 ≈ 0) — 可能 winsorization leakage 引入

### 论文 narrative 两条路

| 路径 | 前提 | Narrative |
|------|------|----------|
| A. 重 build 验 leakage | Fold 4 高 IC 是 winsorization 全样本 p1/p99 泄漏 → 修复后 S8 ≈ 0 | "Compact PC-probe (S6) beats Alpha158 library" 站得住 |
| B. 接受现状 | Fold 4 +0.22 是真 regime 效应 | "S6 在 Hansen SPA (S8 为 benchmark) 下未以 α=0.05 显著**胜过** S8 (non-superiority 方向：candidate > benchmark)，S6 效率 50× 高" (parsimony argument 较弱；严格 equivalence 需 TOST + 反向 SPA) |

### 决定待 H博士

1. **A** 重 build 用 per-fold training-only winsorization (~1.5h 代价) + 重跑 Part C (~1h)
2. **B** 接受现状按 "S6 non-superiority vs S8 under Hansen SPA" 写论文（原措辞 "两条路都有效" 属 equivalence overclaim，2026-04-21-c 更正）
3. 其他

### 下一阶段 (Step 4, 待 H博士 path 选择后启动)

- [ ] 论文 Methods + Results section 初稿
- [ ] Figures refine (IC vs dim 曲线加 S8, per-fold heatmap 8×2)
- [ ] Related Work 起草
- [ ] Codex Round 8 analysis review (if path A, 补发; path B, 现有 review 足够)

### Decision Log Additions

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-20 | Alpha158 158-feat S8 跑完, S6 vs S8 Hansen SPA p_c=0.55/0.59 不拒绝 | 30 runs × 2 models × 5 folds × 3 seeds on faithful qlib Alpha158DL default config |
| 2026-04-20 | Winsorization 策略: 全样本 1/99 (pending per-fold verification) | VMA/VSTD 除以 volume=0 flat days 爆 1e15, 必须 clip; 全样本 vs per-fold trade off 待验 |
| 2026-04-20 | **Narrative 风险 logged**: S6 non-superiority vs S8 under Hansen SPA (non-equivalence — SPA 是单侧 superiority 检验，严格 equivalence 需 TOST) | 原 "compact beats library" 断言需改写 or verify via leakage check |
| 2026-04-20 | VWAP proxy: (H+L+C)/3 | yfinance 无 tick 数据, 主流 substitute |

→ progress: `2026-04-20-a` | analysis: `2026-04-20-a`

*Last updated: 2026-04-20 (Part C 完, narrative 风险 logged, 待 H博士 path 决策)*

---

## 2026-04-20-b: Fold 4 Leakage Diagnostic 完成 → 推荐 Path A

> 诊断跑完，**mixed signal**：tail/rank 指标均 negative（leakage 量级极小），但 z_drift↔IC 强正相关 (MLP ρ=+0.51, SAGE ρ=+0.41, both p<0.001)。pre-committed rule → Path A。科学解读倾向 regime confounder，但不形式排除 leakage，只有 retraining 能定论。

### 当前 H博士 待决策

**推荐 Path A**：
- ~1-1.5h 重跑 Part C with per-fold train-only winsorization
- 输出：S8' IC（Scheme T 训练）vs 原 S8 IC（Scheme G 训练）
- 判据：
  - 若 S8' Fold 4 IC 降到 ≈0 → leakage 确认，原 "S6 compact beats S8 library" narrative 站得住
  - 若 S8' Fold 4 IC 仍 ≈+0.2 → regime 确认，接受 parsimony narrative

**备选**：
- Path B 直接接受：跳过重跑，按 "S6 在 SPA(S8) 下未显著**胜过** S8 (one-sided non-superiority)" parsimony argument 写论文；较弱但诚实（严格 "S6 = S8" 需 TOST + 反向 SPA）
- 诊断加码：用市场 dispersion proxy 做 partial correlation，看 z_drift↔IC 是否在控制 regime 后消失（~30 min, 但非 decisive）

### 下一阶段 (pending H博士 批准)

- [ ] **Path A**：改 `build_alpha158_features.py` 加 `--per-fold-winsor` 或写独立脚本直接在 `run_step3_plan_z_part_c.py` 的 scaler 管道前做 per-fold train-only clip → 重跑 Part C (30 runs) → 重算 SPA + BH-FDR
- [ ] 结果出来后 Codex Round 8 review
- [ ] 根据 S8' Fold 4 IC 走 narrative A 或 B

### Decision Log Additions

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-20 | Fold 4 leakage diagnostic 4-round plan + 2-round code review + 1-round results review 执行完毕 | Rule 9 严格三个触发点 |
| 2026-04-20 | Diagnostic verdict: mixed signal (tail/rank negative, correlation positive) | Test 1 max abs_delta_sum=0.007 (<0.05); ρ>0.9975; MLP ρ(z,IC)=+0.51, SAGE=+0.41, both p<0.001 |
| 2026-04-20 | **Path A 推荐** (仅等 H博士 批准) | Pre-committed rule: mixed signals → Path A; retraining 是唯一 decisive test |
| 2026-04-20 | Codex 科学解读：magnitude disconnect (0.009 std 扰动 → ρ=0.5 IC 变化) 机制上支持 regime confounder 而非 leakage | 但 rank preservation 很强 不能形式排除 input-level leakage 有微妙非线性通道 |

→ progress: `2026-04-20-b` | analysis: `2026-04-20-b`

*Last updated: 2026-04-20 (Fold 4 诊断完成, Path A 推荐, 待 H博士 批准)*

---

## 2026-04-20-c: Path A 完成 → Parsimony narrative 锁定

> S8_pf (per-fold train-only winsor) 重跑完毕。Fold 4 IC 几乎不变 (MLP +0.226→+0.223; SAGE +0.214→+0.270) → **Fold 4 不是 leakage artifact**。S6 vs S8_pf 在 MLP (p_BH=0.769) 和 SAGE (p_BH=0.938) 下均不可区分 → **原 "compact beats library" narrative 被 falsified**。改写 parsimony narrative。

### 下一阶段：Paper writing

- [ ] **Methods section**: 完整记录 walk-forward protocol, Plan Z 特征定义, S6 PC probe 选择, Alpha158 外部 baseline（S8 with global winsor + S8_pf with per-fold winsor 都作为 baseline 讨论，披露 MLP aggregate leakage）
- [ ] **Results section**:
  - Table 1: IC summary by subset × model (include S8, S8_pf)
  - Table 2: Hansen SPA (benchmarks S1/S7/S8/S8_pf × 2 models)
  - Table 3: pairwise BH-FDR (focus S6 vs S8_pf)
  - Figure 1: IC vs subset with NW errorbars
  - Figure 2: per-fold heatmap 9 subsets × 2 models
  - Figure 3: z_drift ↔ IC 诊断图（Fold 4 regime 证据）
- [ ] **Related Work**: Alpha158 / qlib context, parsimony in factor investing
- [ ] **Limitations**:
  - MLP aggregate leakage in original S8 (disclosed, addressed by S8_pf)
  - Fold 4 regime effect — Q2-2025 specific, generalization uncertain
  - 5-fold walk-forward on 5-year S&P 500 sample, not cross-market validation
- [ ] **Codex Round 9 review**: 论文初稿 overall review

### Decision Log Additions

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-20 | Path A 完成. Fold 4 +0.22 确认为 **regime**（非 leakage，per-fold winsor 保留效应） | S8 MLP Fold 4 = +0.226 → S8_pf = +0.223 (unchanged); SAGE +0.214 → +0.270 (increased) |
| 2026-04-20 | **"S6 compact beats S8 library" narrative 放弃** | S6 vs S8_pf on both models: MLP p_BH=0.769, SAGE p_BH=0.938 — 无法拒绝无差异 |
| 2026-04-20 | **新 narrative: Parsimony argument (weak form)** — S6 (3 feat) 在 Hansen SPA (S8_pf 为 benchmark) 下不显著胜过 S8_pf (non-superiority, 非 equivalence), 50× fewer features, 3× faster training. 原写 "≈ S8_pf in IC / statistically indistinguishable" 于 2026-04-21-c 更正——SPA 是单侧 superiority 检验，不支持 equivalence；严格 equivalence 需 TOST + 预设边际 δ | Hansen SPA (S8_pf): MLP p_c=0.075, SAGE p_c=0.700. Operationally superior on features/time/interpretability. |
| 2026-04-20 | S8 MLP 小 aggregate leakage (p_BH=0.037) disclosed in Limitations | Codex Round 3 要求诚实披露；范围仅限 Alpha158 global winsor 路径，不污染 Plan Z (Part A/B) |
| 2026-04-20 | 措辞降温：不说 "regime confirmed" / "equivalence evidence" | Codex Round 3 Q1/Q3: leak-free rerun 只能"consistent with regime, not proof"; "failed to reject null" 比 "evidence for equivalence" 更准确 |
| 2026-04-20 | Part B 不重跑（Plan Z 特征无 winsor 入库，path clean） | Codex Q4 verify |
| 2026-04-20 | Phase 5 Step 3 Plan Z++ 实验阶段 **完整终结** | 主实验 + 外部 baseline + leak-free 验证 + SPA/BH-FDR 完成 |

→ progress: `2026-04-20-c` | analysis: `2026-04-20-c`

*Last updated: 2026-04-20 (Path A 完成，parsimony narrative 锁定，进入 paper writing)*

---

## 2026-04-20-d: README 体系建立 + CLAUDE.md Quad-Doc 升级 ✅

> 以文件夹 README 作为项目索引；session 结束强制维护（Quad-Doc）。

### 已完成

- [x] 24 个文件夹 README.md（Tier 1: 8 + Tier 2: 9 + Tier 3 archived: 7）
- [x] CLAUDE.md Rule 4 新增"读相关 README"步骤
- [x] CLAUDE.md Rule 5 升级为 **Quad-Doc**（`<folder>/README.md` 纳入强制更新）
- [x] CLAUDE.md Rule 10 日期更新 + README 体系条目

### 下一步

- [ ] Codex Review（触发点 1，流程变更）
- [ ] 恢复 paper writing 主线（2026-04-20-c 未完成项继续）

### Decision Log Additions

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-20 | 建立 24 个文件夹 README 索引 | H博士 反馈文件查找困难；简洁索引 + 变更日志机制 |
| 2026-04-20 | CLAUDE.md 升级为 Quad-Doc（Rule 5 表格加 4 行） | Session 结束必须同步 progress + plan + analysis + README |
| 2026-04-20 | README 中文为主 + 英文文件名/术语 | 与 Rule 3（中文会话）一致，文件名保留英文便于 grep |
| 2026-04-20 | `archived/` 每个子目录独立 README | H博士 选择；便于未来查找归档内容 |

→ progress: `2026-04-20-d` | analysis: N/A

*Last updated: 2026-04-20 (README 体系 + Quad-Doc 完成)*

---

## 2026-04-21-b: Loss layer 先于 Architecture — Pred-scale collapse 诊断与修复

> **H博士 提出 3 层诊断框架** (Data / Label → Loss / Objective → Architecture): 第 2 层未修好时比较架构 = 在躺平状态下比架构, 不同架构的躺平程度相近, 真实差异被 loss 压掉。当前诊断: 所有 baseline _price 模型 **CS std capture 仅 13-23% of √R²** theoretical max (SAGE-Sum_price 0.23%), MSE loss 在 R²≈4% 数据上天然 shrink pred variance, IC 看不见但 portfolio-level 受损。**所以 Phase 5 既往架构比较 (SAGE vs MLP 在 S6 上仅差 +0.001) 可能都是在躺平底部做的, loss 修复后架构差异才会 surface。**

### 三层诊断框架 (本条目的 driving rationale)

| Layer | 状态 | 证据 |
|---|---|---|
| **1. Data / Label** | ✅ 已修 | 2026-04-21-a 清理 cross-sectionally standardized 21d forward returns; 无 rf 死代码; labels 数值不变但语义诚实 |
| **2. Loss / Objective** | 🔴 **有病, 待修** | MSE 在 R²≈4% 上诱导 variance shrinkage; SAGE-Mean_price capture 仅 **23% of √R²** (benchmark=0.20, actual=0.046); Paper-level 问题 |
| **3. Architecture** | ⚠️ **比较结果暂停解读** | 在 Layer 2 未修时 SAGE vs MLP 差距仅 +0.001, 可能被 loss 压掉; 需 Layer 2 修好后重评 |

**核心后果**: Plan Z Part B 的 SAGE vs MLP 结论 (S6 SAGE +0.047 vs MLP +0.046) 在 loss 修复后可能爆开或翻转。**Paper 不应 claim "graph marginal"** until loss layer 清洁后重做。

---

### Session 2026-04-21 已查证事实

| Claim | 状态 | 证据 |
|---|---|---|
| Pred-scale collapse 存在 | ✅ 已证 | SAGE-Mean_price CS std=0.046 / 0.20 = **23%**; MLP 14%; SAGE-Sum_price **0.23%** |
| Regime-dependent scale | ✅ 已证 | 5/6 模型 F1-vs-F2 KS p<0.001; MK trend p<0.001 |
| SAGE-Sum_price ≠ MSE shrinkage | ✅ 已证 | KS p=0.143, MK p=0.668 (不响应 regime), 其他模型全部 p<0.001 |
| **Ranking loss 改 Sharpe** | ❌ **无证据** | Paired bootstrap MLP ΔSharpe=−0.35 (p=0.58), NoGraph +2.61 (p=0.11); SAGE-Mean MSE **0 runs 从未跑过**; 之前 +0.78 claim 是 aggregate 非 paired |
| Ranking loss 改 pred_CS_std | ⚠️ **未测** | 旧 ranking_loss runs 没保存 preds, 需重跑 |

**Push 采纳记录** (2026-04-21 对话):
- Capture benchmark 从 label_std=1.0 → √R² ≈ 0.20 (R²=4%) — 所有模型都在躺平区, _all 只是 less bad
- Framing: "误判" → "evaluation framework 不完整" (paper-level methodology 贡献)
- SAGE-Sum 分诊: aggregator pathology vs MSE shrinkage 是两种病
- Paper insight: "invisible to IC" → **"downstream damage underestimated by scale-invariant metric"** (Spearman 对 near-ties 敏感, IC 会降, 敏感度不匹配 downstream utility)

---

### P0-A: Complete ListNet ablation + pred 保存 + 配对显著性

**脚本**: `run_ranking_loss_v2.py` (新, 基于 archived/scripts/run_ranking_loss.py + fixes)

**修复 vs 原版**:
- [ ] **补 SAGE-Mean_price MSE** (原版 0 runs)
- [ ] **补 seed 42** 对 MLP/NoGraph MSE (原版只有 123, 456)
- [ ] **保存 test predictions** (`.npy`, day-aligned): `experiments/ranking_loss_v2/preds_{model}_{loss}_s{seed}_f{fold}.npy`
- [ ] **每 run 额外记录** `pred_cs_std_mean`, `pred_cs_std_median` 到 results.csv
- [ ] **统一使用 2026-04-21-a 清理后的 label 构造** (直接 CS z-score, 无冗余 excess)
- [ ] Graph/sector/hparam 与 `run_walkforward_5fold.py` 完全对齐 (apples-to-apples)

**设计矩阵**:
```
3 models × 2 losses × 5 folds × 3 seeds = 90 runs
  Models:   SAGE-Mean_price, MLP_price, NoGraph_price  (price-only 9-dim)
  Losses:   MSE (当前 Plan Z 用的), ListNet (τ=0.2)
  Folds:    5 (Q2-2024 ~ Q2-2025, 同 wf5)
  Seeds:    42, 123, 456
  Hparams:  hidden=64, layers=2, dropout=0.3, lr=1e-3, epochs=100, patience=15
```

**Estimated runtime**: Colab A100 ~90 min, T4 ~150 min

**产出**:
- `experiments/ranking_loss_v2/results.csv` (90 行, 13 列含 pred_cs_std_*)
- `experiments/ranking_loss_v2/preds_*.npy` (90 files, 每 (n_test_days, 501) float32)
- `experiments/ranking_loss_v2/log.txt`

**分析**: `analyze_ranking_loss_v2.py`

配对 (model, fold, seed) 三个 hypothesis tests, 每个 n=45 per loss pair per model:
1. **ΔIC**: Wilcoxon + NW HAC-paired t
2. **ΔSharpe_net**: Paired bootstrap (B=10,000) + Jobson-Korkie ratio test + NW HAC t
3. **Δpred_cs_std**: Wilcoxon paired

BH-FDR 跨 3 tests × 3 models = 9 比较做多重校正。

**Pre-committed 判决规则** (防 p-hacking):

| 场景 | 触发条件 | 后续行动 |
|---|---|---|
| **A (best)** | ΔSharpe BH p ≤ 0.05 **AND** Δpred_cs_std BH p ≤ 0.05, 方向正 | 确认 ranking loss useful → 启动 P0-B λ-scan |
| **B (diagnostic)** | 仅 Δpred_cs_std 显著, ΔSharpe ns | Paper section 改为 "scale 改变但 portfolio 利益 marginal"; 不启动 P0-B |
| **C (null)** | 三 tests 全 ns | Drop ranking-loss claim; 保留 "scale collapse as diagnostic finding" 作 methodology 贡献 |

---

### P0-B: λ-scan Mixed Loss (conditional on P0-A Scenario A)

**脚本**: `run_lambda_scan.py` (新)

**损失**:
```python
L = λ · MSE_loss(pred, label) + (1 − λ) · ListNet_loss(pred, label, τ=0.2)
λ ∈ {0.0, 0.25, 0.5, 0.75, 1.0}
```

**矩阵**: 5 λ × 3 models × 5 folds × 3 seeds = **225 runs** (~4h Colab A100)

**产出**: IC / Sharpe / pred_cs_std 三条曲线 vs λ, 每 model 一 facet; 定位最优 λ*。

**判决行**:
- λ* ∈ [0, 0.5): ranking-dominant → P2 Vol-Cal Head magnitude calibration 需重设计
- λ* ∈ [0.5, 1.0]: MSE-dominant → 现设计可用

---

### P0-C (新): **架构再评估** (conditional on P0-A Scenario A 或 B)

> **这是 H博士 三层框架的直接后果**, 加入本条目。

**动机**: 如果 loss 修复后 pred scale 展开, Phase 5 Part B 的 SAGE vs MLP 在 S6 上仅差 +0.001 的结论**可能重大变化**。需要在最佳 loss 下**重跑关键架构比较**, 否则 paper architecture section 基于错误 baseline。

**脚本**: `run_arch_comparison_v2.py` (基于 archived/scripts/run_arch_comparison.py)

**设计**:
```
5 架构 × 2 特征集 × 5 folds × 3 seeds × best-loss = 150 runs
  架构:     SAGE-Mean, SAGE-Sum, GAT, Transformer, MLP (没图)
  特征集:   S6 (3-dim PC probe), S1 (10-dim full Plan Z)
  Best loss: 由 P0-A/P0-B 决定 (scenario A: λ*; scenario B: MSE)
```

**Estimated runtime**: Colab A100 ~3h

**判决**:

| 原 (MSE, 躺平下) | 新 (best loss) | 含义 |
|---|---|---|
| SAGE 比 MLP +0.001 | Δ 不变 | 图价值真的低, paper claim "marginal graph" 成立 |
| SAGE 比 MLP +0.001 | Δ ≥ +0.01 | **之前的 "graph marginal" 结论是 loss artifact**, paper architecture section 重写 |
| SAGE 比 MLP +0.001 | Δ 翻负 | Graph harmful, 保留但换新方向 |

**Paper 影响**: P0-C 结果**直接决定 paper architecture section 怎么写**。现在的 [10] SAGE vs GAT 比较, [1] S6 on SAGE vs MLP 差距解读, 都 conditional on 这个结果。

---

### P1: SAGE-Sum Hidden Representation Diagnostic (Architecture 层独立问题)

**脚本**: `diag_sage_sum_hidden.py` (新, <200 行)

**目标**: 证明 SAGE-Sum_price 的 0.23% capture 是 **aggregator degenerate fixed point**, 与 loss 层病分开。

**设计** (独立于 P0, 可并行):

- 1 fold (Fold 0) × 2 seeds (42, 123) × 2 架构 (SAGE-Mean, SAGE-Sum) = 4 runs
- 每次训练完, forward 一次 test batch, hook 最后一层 encoder hidden state `h` (num_stocks × 64)
- 记录:

```python
diagnostics = {
    'h_std_per_dim':       h.std(dim=0).cpu().numpy(),       # (64,) 每维 scale
    'h_std_per_dim_mean':  float(h.std(dim=0).mean()),
    'h_std_per_dim_min':   float(h.std(dim=0).min()),         # dead dim
    'h_abs_max':           float(h.abs().max()),               # explode
    'matrix_rank':         int(torch.linalg.matrix_rank(h, tol=1e-4)),   # rank collapse
    'frac_near_zero':      float((h.abs() < 1e-6).float().mean()),
    'active_dim_frac':     float((h.std(dim=0) > 1e-3).float().mean()),
}
```

**Estimated runtime**: Colab T4 ~10 min

**产出**: `experiments/sage_sum_hidden_diag.json`

**判决**: SAGE-Sum 的 `h_std_per_dim_mean` < 1/10 × SAGE-Mean 的对应值 → 确认 encoder 层 pathology, paper Limitations 将 SAGE-Sum 放独立子节 (非 MSE shrinkage 同段)。

---

### P2: Vol-Cal Selection Head 重设计 (deferred to post-P0-B)

**前置**: 等 P0-B λ-scan 的 λ* 位置确定。

**如果 λ* → 0** (ranking-dominant): Vol-Cal Head 原设计假设 "MSE-regularized pred magnitude carries calibration signal", 在 ranking loss 下 pred scale 语义变了 (只保 rank 信息), Head 的 selection mechanism 需重设计。

**如果 λ* 混合**: 现设计大概率可用。

具体方案待 P0-B 出来后起草。

---

### 启动顺序与时间线

| Step | Action | Duration | Blocker |
|---|---|---|---|
| 1 | 写 `run_ranking_loss_v2.py` + `analyze_ranking_loss_v2.py` | 本地 ~1h | — |
| 2 | 写 `diag_sage_sum_hidden.py` | 本地 ~0.5h | — |
| 3 | **Codex Round 10: Code Review** (P0-A + P1 脚本) | ~20 min | Step 1-2 完成 |
| 4 | Colab 启动: P0-A (90-150 min) **并行** P1 (10 min) | ~2.5h 挂机 | Step 3 通过 |
| 5 | 分析 P0-A, 读 commit line 触发 A/B/C | 本地 ~0.5h | Step 4 完成 |
| 6 | **Codex Round 11: Results Review** (P0-A) | ~20 min | Step 5 完成 |
| 7 | 若 Scenario A: 写 `run_lambda_scan.py`, review, 启动 P0-B (~4h) | Colab ~5h | Step 6 判决 |
| 8 | 写 `run_arch_comparison_v2.py`, 用 P0-B 出的 best-loss, 启动 P0-C (~3h) | Colab ~4h | Step 7 完成 (scenarios A/B 都启动) |
| 9 | 整合: paper architecture section 重评估 | 本地 ~2h | Step 8 完成 |
| 10 | 若 λ-scan 指向 ranking-dominant: P2 Vol-Cal Head 重设计 | TBD | Step 7 判决 |

**总时长预估**: Scenario A 路径 ~12-15 h 工时 (3-4 Colab sessions); Scenario B/C 路径 ~4-6 h。

---

### Paper 影响 (三层框架下的 scope 改写)

**重点**: P0-C 的加入**将架构比较从既有结论升级为 conditional claim**:

> "We defer architecture-level comparisons (SAGE vs MLP vs HGT) until after loss-layer remediation, as prediction-scale collapse under MSE loss in low-R² regimes compresses architectural differences below measurement threshold. Our preliminary ranking-loss results [Table X] show [SAGE/MLP diff Δ] compared to [Δ=0.001 under MSE], confirming that architecture comparisons require a calibrated loss baseline."

这是一段**强方法论 claim**, 比当前 paper 的 "S6 SAGE ≈ MLP, graph is marginal" 更 nuanced 也更正确。

### Paper 影响预测 (三 scenarios)

| Scenario | 概率 | Paper impact |
|---|---|---|
| A (P0-A positive, P0-B 有 λ*) | 中 | 新 section 5.4 "Loss Layer Matters: Evaluation Protocol Before Architecture"; 重写所有架构比较; P2 Vol-Cal Head 重设计 |
| B (只 pred_cs_std 显著) | 中 | Section 5.4 作 diagnostic finding; 不重做架构比较 (但注明局限) |
| C (全 ns) | 低 | 保留 "scale collapse as diagnostic + MSE in low-R² limitation"; 原架构结论不变 |

---

### Decision Log Additions

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-21 | **采纳 H博士 3 层诊断框架** (Data → Loss → Architecture) | Layer 2 未修时 Layer 3 比较被压缩; 整个 paper 优先级重排 |
| 2026-04-21 | Capture benchmark 修正: √R² (≈0.20) 而非 label_std | Var decomp: predictable var = R² → optimal MSE predictor pred std = √R² ≈ 0.17-0.22 |
| 2026-04-21 | "ListNet 改 Sharpe +0.78" claim **正式撤回** | Paired bootstrap MLP p=0.58, NoGraph p=0.11; +0.78 是 aggregate 非 paired; SAGE MSE 根本没跑 |
| 2026-04-21 | SAGE-Sum_price 分诊为 **aggregator pathology**, 不是 MSE shrinkage | F1-F2 KS p=0.143 (不响应 regime), 其他模型全部 p<0.001 响应 |
| 2026-04-21 | Paper framing: "invisible to IC" → "**downstream damage underestimated by scale-invariant metric**" | Spearman 对 near-ties 敏感, IC 会降, 敏感度不匹配 downstream utility |
| 2026-04-21 | 新增 `pred_cs_std` 为 P0 evaluation metric | Multi-metric protocol 是 paper methodology contribution |
| 2026-04-21 | **P0-A**: 补齐 ListNet v2 + 存 preds + Jobson-Korkie | 现有 ranking_loss.csv 数据空洞无法做 honest claim |
| 2026-04-21 | **P0-C 新增**: 架构再评估 (conditional on P0-A A/B) | 三层框架直接后果: loss 修复后架构差异可能爆开 |
| 2026-04-21 | **P0-B conditional**: λ-scan 仅 Scenario A 启动 | 避免 sunk cost; P0-A 已可决定 paper 保留/drop |
| 2026-04-21 | **P1 并行**: SAGE-Sum hidden diag | Architecture 层独立问题, 不 block P0, 便宜 ~10 min |
| 2026-04-21 | P2 Vol-Cal Head 推迟到 λ-scan 完成后 | magnitude assumption 取决于 λ* |
| 2026-04-27 | neat-freak skill: absorb 4 ideas (§7 Sync Matrix + §8 Three Audiences + `/session-closeout` Agent 4) into `.claude/rules/docs.md` | Skill's "delete completed / merge duplicates" model conflicts with Quad-Doc append-only progress.md and Decision Log; but its change-impact matrix, forced inventory, relative-time grep, and three-audiences principle fill genuine gaps. Codex Round A (PROCEED-WITH-FIXES, 0C/3M/1Cn, all FIXED) confirmed strategy. Full plan: `docs/neat_freak_integration_plan_2026-04-27.md`. Codex review: `artifacts/reviews/2026-04-27_codex_plan_A.md`. |
| 2026-05-01 | neat-freak skill: DELETED globally (`rm -rf ~/.claude/skills/neat-freak/`) instead of suppressed via deny+Rule 6.5 | H博士 directive 2026-04-27: deletion is simpler and removes CODEX-A-04 live-test risk surface. No `permissions.deny` entry, no Rule 6.5, no LLM-Finance-Benchmark mirror. Skill remains available on GitHub (KKKKhazix/khazix-skills) if ever needed in a different project. |
| 2026-05-01 | LLM-Finance-Benchmark: NOT mirroring Phase A (§7/§8/Agent 4) | Project uses Tri-Doc + Rule 5.5 wide-scope README, not Quad-Doc + narrow scope; no `.claude/rules/` infra; currently in Phase init (low doc-drift risk); skill deletion makes mirror-for-suppression-purposes moot. If LLM-Finance-Benchmark grows enough to need it, write a separate tailored plan. |

→ progress: 2026-04-27-a (Codex Round A); 2026-05-01-a (Phase A+B execution) | analysis: N/A

---

## 2026-05-02-a: Plan Z++ Phase 0 — Tier 0 audits + manifest + sentinel ALL DONE

### What's now decided (post-Phase 0)

- ✅ **Step 0.1 audit**: phase5 features clean; **alpha158 has CRITICAL global p1/p99 winsorization leakage** (build_alpha158_features.py:389-396 across full panel). Stage 1 results carry latent leakage but verdict locked. raw `_raw.npy` already saved → Tier 1 must use `winsorize_per_fold_train(raw, train_days)` helper instead of post-winsor file.
- ✅ **Step 0.2 manifests**: `data/reference/fold_manifest_expanding.json` reproduces existing `artifacts/step3_plan_z/fold_manifest.json` exactly on day-index sets. New `data/reference/fold_manifest_roll2y.json` with 504-train-day windows passes all 4 cross-manifest assertions (test/val coverage match, rolling ⊆ expanding train, rolling embargo).
- ✅ **Step 0.3 graph provenance**: `assert_graph_train_only()` helper added to `run_step3_plan_z_part_a.py`; `build_fold_manifest()` extended to optionally stamp `graph_snap_end` + `graph_snap_window` per fold; `train_one()` runtime guard added. All 10 fold-split combos (5 expanding + 5 rolling) clear assertion offline (gap = 19-21 days between snap_end and train_max).
- ✅ **Step 0.4 sentinel test**: per-fold-winsor pipeline 10/10 PASS (bitwise unchanged on train artifacts under price/feature perturbation); legacy global-winsor control 10/10 FAIL as expected (200K-400K elements differ per fold).

### What this changes for Phase A

Phase A (Tier 1.B robust pointwise + Tier 1.D hparam sweep) **must** load `sp500_5y_alpha158_features_raw.npy` and use the per-fold winsor helper. Cannot load the post-winsor `sp500_5y_alpha158_features.npy` directly.

### Decision Log Additions

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-02 | Plan Z++ Tier 1 must use raw alpha158 + per-fold winsor | Step 0.1 audit found CRITICAL global p1/p99 winsorization in `build_alpha158_features.py:389-396`; sentinel control empirically confirms 200K-400K train cells contaminated per fold. Stage 1 results stay locked but Tier 1 onward switches to leakage-free per-fold pipeline. raw `_raw.npy` already exists, so this is a runtime helper change, not a feature rebuild. |
| 2026-05-02 | New manifests live at `data/reference/fold_manifest_{expanding,roll2y}.json` (graph provenance stamped per fold) | Plan Z++ §0.2 + §0.3. Old `artifacts/step3_plan_z/fold_manifest.json` retained for Stage 1 reproducibility; future Tier 1 scripts redirect to `data/reference/`. New manifests carry `graph_snap_end` + `graph_snap_window` so any Tier 1 script can do offline graph-train-only checks without rebuilding correlation snapshots. |
| 2026-05-02 | Sentinel test required as gate for any future feature/loss/manifest change | Plan Z++ §0.5 (B-07). Empirically catches 100% of legacy leakage on the 10-cell control matrix. Re-run if any of: alpha158 raw rebuilt, manifest spec changes, new winsor/scaler logic added, new graph snapshot logic. |

→ progress: 2026-05-02-a | analysis: 2026-05-02-a

*Last updated: 2026-05-02 (Plan Z++ Phase 0 complete; Tier 0 audits + manifest + sentinel ALL PASS; Phase A authorized to start on per-fold-winsor pipeline)*

---

## 2026-05-06-a: Plan Z++ Phase A complete — Tier 1.B null replication + Tier 1.D MARGINAL at registered Score gate (CORRECTED 2026-05-06 per Codex stop-time review)

> **Correction note**: original 2026-05-06-a entry framed Tier 1.D as "POSITIVE / 3 of 4 hparams significantly beat baseline" with h0 as the new baseline. Codex stop-time review 2026-05-06-b flagged this as a registered-gate violation: Plan §1.D explicitly mandates "Score = mean_IC − 0.35·σ_fold − 0.05·𝟙[min_fold_IC < −0.10]; NOT raw mean IC alone (避免 Stage 0 ListMLE-style val-overfit)". Tier 1.D verdict downgraded to MARGINALLY SUPPORTED; new baseline corrected from h0 to h2. See `progress.md` 2026-05-06-b for full violation log.

### Locked verdicts (from `docs/analysis.md` 2026-05-06-a + `artifacts/tier1_phase_a/stat_report.md`, both with correction notes)

- **Tier 1.B (robust pointwise sweep, 400 cells)**: 0/12 BH-FDR rejections at α=0.05; min BH-adj p = 0.830; 11/12 ΔIC < 0; 1/12 positive (Huber × SAGE-Mean × S8 = +0.0084) but NW p = 0.42 (not significant)
- **Fold-4 stress diagnostic**: 8/12 contrasts statistically significantly NEGATIVE on fold-4 (NW p < 0.05) — robust losses **harm** fold-4 directional accuracy
- **Tier 1.D (hparam regularization, 120 cells, CORRECTED interpretation)**: pre-registered Score gate selects **h2 (AdamW, lr=5e-4, wd=1e-3, patience=5)** as winner with Score = +0.0007. h2's NW-HAC ΔIC vs Tier 1.B baseline is **p = 0.059 — marginal, NOT significant at α=0.05**. h0 is Score-second (+0.0006). h1 and h3 are Score-NEGATIVE (each Score = −0.0027) despite NW p < 0.005 — they have higher mean_IC but high σ_fold, exactly the val-overfit pattern Plan §1.D filters out.

### What's now decided

- Plan §1.B Hypothesis "70% label noise dominant; bounded gradient losses help" is **REJECTED** with statistical confirmation
- Plan §1.D Hypothesis "30% overfitting addressable via regularization" is **MARGINALLY SUPPORTED at the pre-registered Score gate** but does NOT reach α=0.05 — winner h2 has NW p = 0.059 vs baseline. The "3/4 significant" claim from raw mean_IC perspective is post-hoc and NOT pre-registered.
- New best-practices hparam baseline: **h2 = AdamW + lr=5e-4 + wd=1e-3 + patience=5** (NOT h0 — corrected per Codex stop-time review). Replaces Stage 1's Adam + lr=1e-3 + wd=1e-4 + patience=10.
- **Novel paper-grade negative finding** (unchanged): robust pointwise losses are statistically significantly worse than MSE during regime shifts (fold-4 Q2-2025), opposite to heavy-tail-noise intuition. Mechanism: bounded-influence suppresses gradient signal from extreme observations that ARE the directional signal under stress.

### Phase B direction options (awaiting H博士 sign-off; baseline references CORRECTED to h2)

1. **(a) Trigger Codex/finance-gnn-reviewer Touchpoint 2 + 3 on Phase A.5** before any further work — code review (`run_tier1_phase_a.py`, `analyze_tier1_phase_a.py`) + results review (3 statistical verdicts including the corrected Tier 1.D marginal verdict). Recommended for paper-defensibility.
2. **(b) Write paper draft with current results** — Story C+ now has 2 null findings (Stage 1 + Tier 1.B) + 1 novel mechanism (fold-4 robust-loss harm) + 1 marginal-only constructive observation (Tier 1.D h2 marginal at Score gate). 4-6 page workshop format feasible. Paper framing for Tier 1.D needs care: report Score-gated marginal verdict, NOT the post-hoc "3/4 significant" view.
3. **(c) Run Tier 1.A (rolling 2y vs expanding) with h2 baseline** — 100-cell pilot (~4-5h M4) per Plan Z++ Round B B-05 fix. Tests data-length / regime-shift-attenuation hypothesis on the registered Score-winner baseline.
4. **(d) Run Tier 1.C (anchored RankNet) with h2 baseline** — 100-cell pilot (~3-5h M4). Tests structural (not pointwise-robustness) ranking loss hypothesis. Tier 1.B null does NOT predict Tier 1.C null.
5. **(e) Re-run Tier 1.B with h2 baseline** — NOT recommended; even h2's marginal Score-gate effect (+0.0007) << −0.04 to −0.20 fold-4 penalty robust losses show.

### Decision Log Additions

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-06 | Tier 1.B robust pointwise hypothesis REJECTED | 0/12 BH-FDR rejections + 8/12 fold-4 NW-significant negative; replication on leakage-free pipeline confirms Stage 1 Scenario B verdict and adds novel stress-period mechanism finding |
| 2026-05-06 | New best-practices baseline = **h2** hparam (AdamW, lr=5e-4, **wd=1e-3**, patience=5) — CORRECTED from h0 per Codex stop-time review 2026-05-06-b | Plan §1.D pre-registered Score gate (Score = mean_IC − 0.35·σ_fold − 0.05·𝟙[min<−0.10]) selects h2 as winner with Score=+0.0007 (h0 second at +0.0006). h2 NW-HAC p=0.059 vs Tier 1.B baseline — marginal, not significant at α=0.05. The earlier "h0 as new baseline" decision violated Plan's explicit "NOT raw mean IC alone" rule (h0/h1/h3 won by mean_IC but h1/h3 are Score-NEGATIVE due to high σ_fold). h2 is the correct locked baseline for Tier 1.A and Tier 1.C. |
| 2026-05-06 | Re-running Tier 1.B with the new h2 baseline NOT authorized (originally written as "h0 baseline" before 2026-05-06-b correction) | h2's marginal Score-gate effect (+0.0007) is much smaller than −0.04 to −0.20 fold-4 penalty robust losses show; highly unlikely robust losses recover even with the registered Score-winner baseline |

→ progress: 2026-05-06-a + 2026-05-06-b (Codex correction) | analysis: 2026-05-06-a (with correction note)

*Last updated: 2026-05-06 (Phase A complete; Tier 1.B strong null + fold-4 novel mechanism + Tier 1.D MARGINALLY SUPPORTED at Score gate (CORRECTED from "POSITIVE" per Codex stop-time review); new baseline h2 not h0; awaiting H博士 sign-off on Phase B direction (a/b/c/d/e))*

---

## 2026-05-14-a: Phase B (a)(b)(c)(d)(e) + Finalize ALL COMPLETE — 0/28 BH-FDR cumulative null, paper v1 drafted

### Phase B execution summary

H博士 directive 2026-05-13: "按 abcde 的顺序一个一个做". All 5 substeps + finalize complete by 2026-05-14 13:30 PT.

| Step | Outcome | Compute |
|---|---|---|
| (a) Touchpoint 2+3 self-reviews | PASS-WITH-CONCERNS each (Codex unavailable, fallback) | 0h |
| (b) Paper draft v0 (Story C+) | 3,810 words, provenance-clean | 0h |
| (c) Tier 1.A rolling vs expanding | 100 cells; ListMLE fold-4 attenuation NW-significant (+0.092, p=0.009) but FAILS Plan §1.A general-preference gate | ~2h M4 |
| (d) Tier 1.C anchored RankNet | 200 cells; 0/4 BH-FDR; **0/4 Gate 1.C (σ-guard fails universally)** | ~4.5h M4 |
| (e) Tier 1.B-h2 | 400 cells; **0/12 BH-FDR; 12/12 ΔIC negative; 11/12 fold-4 NW-significant negative** | ~7.5h M4 |
| Finalize stat + paper v1 | 0h compute (analysis on cached preds); paper v1 ~3,905 words, 6 contributions, provenance-clean | 0h |

Total Phase B compute: ~14h M4 (mostly autonomous overnight).

### Locked verdicts (binding)

1. **Tier 1.B-h2: 0/12 BH-FDR rejections; ALL 12 ΔIC NEGATIVE**. Min BH-adj p = 0.582 (more conclusive than Tier 1.B Adam baseline). Cross-baseline robustness check: regularization helps MSE itself but does NOT let robust losses recover.
2. **Tier 1.B-h2 fold-4: 11/12 NW p < 0.05 in NEGATIVE direction** (vs 8/12 at Adam). The bounded-influence-harm-under-regime-stress mechanism is hparam-agnostic.
3. **Tier 1.A: regime-conditional ListMLE attenuation** (+0.092 fold-4 IC, NW p=0.009) but FAILS Plan §1.A "generally preferable" gate (folds 0-3 marginally negative for both losses).
4. **Tier 1.C: 0/4 Gate 1.C — σ-guard mechanism fails universally**. Median pred_cs_std = 0.022-0.036 vs target floor 0.05. Paper-grade negative mechanism finding on σ_penalty=0.05 inadequacy.
5. **Cumulative across Tier 1**: **0/28 BH-FDR rejections** at α=0.05. MSE is empirically hard to beat across three distinct alternative-loss families.

### Paper draft v1 status

- File: `docs/paper_draft_2026-05-14_v1.md` (~3,905 words, 6 contributions, 6-page workshop format)
- Title (working): "MSE Is Hard to Beat: A 28-Contrast Preregistered Horse Race of Loss Functions for Cross-Sectional Stock Ranking, with a Novel Regime-Stress Mechanism"
- Provenance verifier: PASS
- Status: ready for H博士 review + final Codex pass (when quota available)

### Decision Log Additions

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-13 | Phase B sequence a→b→c→d→e (per H博士) | Comprehensive coverage of Story C+ findings; (e) explicitly NOT recommended but completed per directive — confirms cross-baseline null robustness |
| 2026-05-14 | Tier 1.A verdict: regime-conditional ListMLE attenuation (NW p=0.009 fold-4) but FAILS general-preference gate (folds 0-3 ΔIC negative) | Plan §1.A pre-registered "rolling generally preferable" requires both all-folds improvement AND folds 0-3 not materially negative. Folds 0-3 fail. Report as regime-conditional partial attenuation. |
| 2026-05-14 | Tier 1.C verdict: 0/4 Gate 1.C — σ-guard mechanism fails universally | Plan §1.C Gate 1.C condition C3 (median pred_cs_std ≥ 0.05) fails on all 4 cells (0.022-0.036). Explicit σ_penalty=0.05 is empirically insufficient to prevent scale collapse in pairwise log-loss. Paper-grade negative finding on the σ-guard mechanism itself, with implications beyond our specific implementation. |
| 2026-05-14 | Cumulative null finding LOCKED: 0/28 BH-FDR rejections across Tier 1.B Adam + Tier 1.B-h2 + Tier 1.C | Three distinct alternative-loss families (robust pointwise × 2 baselines, anchored Bradley-Terry pairwise) all fail to beat MSE on leakage-free panel. Strongest published null in cross-sectional equity loss literature to our knowledge. |
| 2026-05-14 | Paper draft v1 status: ready for H博士 review; pending final Codex pass | v1 covers all 4 nulls + Tier 1.B-h2 cross-baseline robustness + Tier 1.A regime-conditional + Tier 1.C σ-guard failure + fold-4 mechanism + Tier 1.D marginal-regularization observation. ~3,905 words, provenance-clean. |

### Open items / next actions

1. **H博士 decision on title** (2 options in draft)
2. **Stage 1 integration decision** (combined paper vs separate references)
3. **Sharpe with raw fwd_ret** (vs z-score proxy): ~5 min compute on existing preds
4. **IC_sector_resid (Plan §2.C)** secondary metric: ~2h dev, ~1 min compute
5. **Final Codex Touchpoint 2 + 3 on paper v1**: pending Codex quota reset

→ progress: 2026-05-14-c | analysis: 2026-05-14-a

*Last updated: 2026-05-14 13:35 PT (Phase B all 5 substeps + finalize stat + paper v1 complete; 0/28 BH-FDR cumulative null + 11/12 fold-4 NW-significant robust-loss harm at h2 + Tier 1.A regime-conditional + Tier 1.C σ-guard failure; awaiting H博士 review of paper v1)*

---

## 2026-05-20-a: 10-seed expansion + Tier 1.D verdict revoked

### What changed

H博士 2026-05-18 directive ("我们每一轮实验都是跑了十个 seeds 对吧 / 没有的都要补") triggered 10-seed expansion for all Phase A/B experiments (only Stage 1 had been 10 seeds originally). +1,380 new cells across 4 experiments in ~43h M4 autonomous compute. Total Phase A/B cell count: 2,604.

10-seed re-analysis exposed a **5-seed selection artifact** in Tier 1.D:
- 5-seed: h2 (AdamW lr=5e-4 wd=1e-3 patience=5) was registered Score winner +0.0007, marginal NW p=0.059 vs Tier 1.B baseline
- 10-seed: ALL 4 hparam configs are Score-NEGATIVE; ALL NW p > 0.5 vs baseline → **FULL NULL**

Tier 1.D's positive finding was driven by the 3 matched seeds [86, 123, 456] being a lower-than-average baseline subset. At 10 seeds the baseline catches up to the regularized variants.

### Other 10-seed verdict changes

- Tier 1.B Adam fold-4 NW-sig negative: 8/12 → **10/12** (mechanism stronger)
- Tier 1.B-h2 fold-4 NW-sig negative: 11/12 → **9/12** (slight weakening due to seed averaging on SAGE-S8 cells)
- Tier 1.A ListMLE fold-4 attenuation NW p: 0.009 → **<0.001** (stronger)
- Tier 1.B Adam + h2 + Tier 1.C primary BH-FDR: **0/28 stable**
- Tier 1.C Gate 1.C: **0/4 stable** (σ-guard universal failure robust)
- Tier 1.E ListMLE primary gate: **0/4 stable** (Stage 1 already 10 seeds)

### Cumulative narrative now (Story C+ v3)

**0/36 BH-FDR rejections (Stage 1 ranking 8 + Tier 1.B Adam 12 + Tier 1.B-h2 12 + Tier 1.C 4) + 7 nulls (8 with Tier 1.D revoked positive) + 3 mechanism findings**:
1. Fold-4 robust-loss harm (10/12 NW-sig at Adam, 9/12 at h2)
2. σ-guard mechanism failure (0/4 Gate 1.C)
3. Lagged-cs-dispersion does NOT explain ListMLE collapse (0/4 Plan §1.E primary gate)

Plus 1 regime-conditional finding: ListMLE rolling-2y fold-4 attenuation NW p<0.001 (but FAILS Plan §1.A general-preference gate).

### Paper v3 deferred

Paper v2 (`docs/paper_draft_2026-05-18_v2.md`) requires substantial rewrite for 10-seed numbers and Tier 1.D verdict revision. Deferred to next session.

### Decision Log Additions

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-18 | 补 Tier 2.C IC_sector_resid + Tier 1.E regime forensic (Plan-listed missing items, 0 compute) | Plan §2.C "every Tier 1 + 2 cell gets BOTH IC_abs and IC_sector_resid" + Plan §1.E pre-registered regime test. Both analysis-only on existing preds. Tier 1.E REJECTS regime hypothesis for ListMLE collapse (0/4 primary gate). |
| 2026-05-18 | 10-seed expansion for all Phase A/B (H博士 directive) | Phase A/B used 5 seeds (Tier 1.D used 3); only Stage 1 had 10 seeds. Plan §"Hard Constraints" Gate 1.B "5→10 expansion authorized only if PASS gate" was overridden by H博士 — comprehensive 10-seed coverage requested. |
| 2026-05-20 | Tier 1.D verdict REVOKED from "marginal positive" to "FULL NULL" | 10-seed re-analysis exposed 5-seed selection artifact. h2 NW p flipped from 0.059 marginal → 0.997 NULL. All 4 hparam configs Score-NEGATIVE at 10-seed. Paper v3 narrative dropped "marginal regularization positive" contribution; added "robustness check via seed-expansion exposed artifact" methodological finding. |
| 2026-05-20 | Cumulative Story C+ headline = 0/36 BH-FDR + 7 nulls + 3 mechanism findings | Combined Stage 1 (10 seeds, 0/8) + Tier 1.B Adam (10 seeds, 0/12) + Tier 1.B-h2 (10 seeds, 0/12) + Tier 1.C (10 seeds, 0/4). 7 nulls = Stage 1 ranking + Tier 1.B Adam robust + Tier 1.B-h2 robust + Tier 1.C anchored + Tier 1.A general-preference + Tier 1.D regularization + Tier 1.E regime hypothesis. 3 mechanism findings = fold-4 robust-loss harm + σ-guard failure + lagged-dispersion not the ListMLE mechanism. |
| 2026-05-26 | **Pivot to Story A "When Do GNNs Help" paper** | Plan AAA / GAT 21d numbers don't support clean SOTA story; honest conditional analysis is publishable; uses 80% of existing data; target ICAIF 2026 |
| 2026-05-26 | **Adaptive 30 → 100 seed design with pre-committed extension rule** | Avoids "run until 10 good ones" selection bias; tests feasibility first; pre-registered rule (mean IC > 0.020 AND CV > 30%) prevents post-hoc rationalization; honest mean ± std reporting |
| 2026-05-26 | **News-as-feature DROPPED for 21d models, switch to news-as-edge co-occurrence** | ΔIC = −0.045 empirically (source: experiments/horizon_ablation_results.csv MLP_price vs MLP_all); 768-d FinBERT dilutes 9-d price signal; news-as-edge follows FRI ICAIF 2024 evidence |
| 2026-05-26 | **DSR + PBO + bootstrap CI suite implemented as cherry-pick detection** | Template 4 methodology contribution; 0/8 surveyed GNN-finance papers (HATS, RSR, FinGAT, MASTER, MDGNN, FinMamba, OmniGNN, THGNN) do this; positions our paper as methodologically ahead of subfield |
| 2026-05-26 | **Mamba+regime+sector deferred to post-Story-A** | Discussion archived in plan §3; reactivate only if Story A succeeds and time permits; novelty potential preserved without blocking current paper |
| 2026-05-26 | **HATS baseline reproduction = STRETCH not core** | 1.5-2 week cost vs 8-week budget; only needed for Template 1 "replication-failure" framing supplement; Story A core arguments hold without it |
| 2026-05-27 | **Plan AAA T-1 stability — verdict A: accept inconclusive + §Limitations qualifier; full Plan AAA permutation re-run deferred to paper §Future Work** | T-1 diagnostic (`analyze_plan_aaa_t1_diagnostic.py`) shows proxy single-feature IC stable under T-1 shift (proxy-raw ∩ proxy-T1 = 15/15; group IC drops ≤0.007 absolute), but proxy ≠ permutation Δ-IC by construction (only 5/15 match Plan AAA's ranking). Cannot rule out group-ordering change without a full ~12-24h M4 re-run; H博士 chose to acknowledge in §Limitations Item 5+7 rather than redo. E1's runtime T-1 shift in `build_universe_C` keeps the 400 E1 results leak-free. Source: artifacts/plan_aaa_t1_diagnostic/summary.md. |
| 2026-05-27 | **HATS baseline reproduction promoted from STRETCH to GO** | Story A 4-element narrative element (1) "honest baseline under strict eval" needs a representative published-GNN comparator that we re-ran ourselves; reduces ~30-40% desk-reject risk on "results not believable" axis. ~1-1.5 week effort accepted. plan §1.6 status: STRETCH → SCHEDULED. |
| 2026-05-27 | **Paper-writing strategy: selective rather than exhaustive pre-emption** | Per H博士 directive: paper §Results / §Limitations should NOT self-correct every Codex Touchpoint 3 concern. Exhaustive pre-emption signals defensiveness, removes reviewer-Q&A "contribution" opportunity, reduces narrative cleanness. Concrete split documented in docs/analysis.md 2026-05-27-c Q4 (MUST surface / MAY surface / SHOULD reserve for reviewer Q&A). All Codex findings remain INTERNAL ACK as full boundary references; selective application happens at writing phase. |
| 2026-05-27 | **Multi-testing ledger expanded: historical exploratory family counts enumerated** | Plan AAA (61 group tests), horizon ablation (360 cells), loss horserace (600 cells) now explicitly listed in `artifacts/storya_e6_dm_spa/multiple_testing_ledger.json` `historical_exploratory_trials` block. SPA confirmatory scope clarified: SPA controls post-E1 confirmatory family ONLY; exploratory trials are pre-experiment, disclosed but not entered into SPA family per Codex CODEX-RR-E1E6-A-bis-06 PENDING-fix resolution. |
| 2026-05-27 | **E3/E4 edge ablation = analytical complete; 0/5 BH-FDR rejected in full + LOFO-4 collapse + Fold-4-only diagnostic** | Headline edge-ablation finding LOCKED: in full 5-fold condition, no edge augmentation (α2 sector, α3 news, α4 both) survives BH-FDR q=0.05 over the 5-pair family (smallest raw HLN p=0.039 < rank-1 threshold 0.010); LOFO-4 collapses all 5 ΔIC magnitudes toward zero; Fold-4-only (Q2-2025) shows ΔIC CI excluding zero for all 3 augmented configs but N=10 cells per arm limits to diagnostic-only. Source: artifacts/storya_e6_edge_ablation/edge_pairs_dm.csv + edge_bootstrap_ci.csv + Codex T3 results review 2026-05-27_codex_results_e3e4edge_A.md PASS-WITH-CONCERNS. |

→ progress: 2026-05-27-c | analysis: 2026-05-27-c

*Last updated: 2026-05-27 (Story A v3 confirmatory experiments COMPLETE: E1 400 cells + E3 50 + E4-α 100; E6 SPA/DM/LOFO/edge_ablation all post-processed; 8 Codex Touchpoints across plan/code/results; 5 new Decision Log rows for 2026-05-27 closure)*
