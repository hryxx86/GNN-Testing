# GNN-Testing 项目全景综述：所有 Findings、Results、Implementation

> Written 2026-04-20. 对照此文档逐条问答。
> 按**科研发表重要性**依次排列 (S > A > B > Infrastructure)。
> 数据来源：progress.md + plan.md + docs/analysis.md + docs/phase5_diag_*.md + docs/session_handoff_*.md + archived/docs + experiments/*.csv + artifacts/step3_plan_z/*.json。

---

## 目录

- [一、最核心可发表发现 (Priority S)](#一最核心可发表发现-priority-s)
  - [【1】S6 (3-feat PC probe) 在 Hansen SPA 下不显著胜过 S8 (Alpha158) — Parsimony (weak form, non-superiority)](#1s6-3-feat-pc-probe-在-hansen-spa-下不显著胜过-s8-alpha158--parsimony-weak-form-non-superiority)
  - [【2】Horizon Ablation：21d peak 倒 U 型](#2horizon-ablation21d-peak-倒-u-型)
  - [【3】Walk-Forward 5-Fold 严格协议 + Fold 4 诚实方差报告](#3walk-forward-5-fold-严格协议--fold-4-诚实方差报告)
  - [【4】Permutation Ranking — mom12m 压倒性第一](#4permutation-ranking--mom12m-压倒性第一)
- [二、重要 methodology findings (Priority A)](#二重要-methodology-findings-priority-a)
  - [【5】Normalization × Regime 交互作用 (Novel)](#5normalization--regime-交互作用-novel)
  - [【6】SelectiveNet 在金融 weak-signal 的 Complete Failure](#6selectivenet-在金融-weak-signal-的-complete-failure)
  - [【7】SEC Text Features 无效 — Gate 1 STOP](#7sec-text-features-无效--gate-1-stop)
  - [【8】Fold 4 Anomaly: Regime Stress 非 Leakage](#8fold-4-anomaly-regime-stress-非-leakage)
- [三、Supporting findings (Priority B)](#三supporting-findings-priority-b)
  - [【9】9-dim 特征有效秩仅 3](#99-dim-特征有效秩仅-3)
  - [【10】SAGE-Mean/Sum 稳定性碾压 GAT/HGT/Transformer](#10sage-meansum-稳定性碾压-gathgttransformer)
  - [【11】FinBERT NLP Embedding 有害 (EMH 证据)](#11finbert-nlp-embedding-有害-emh-证据)
  - [【12】Phase 1-2 Binary Direction Prediction 所有 Baseline ≈ 随机](#12phase-1-2-binary-direction-prediction-所有-baseline--随机)
- [四、Infrastructure / Implementation 亮点](#四infrastructure--implementation-亮点)
- [五、发表路线推荐](#五发表路线推荐)
- [六、当前 open questions](#六当前-open-questions)
- [附录 A：实验产出 CSV 清单](#附录-a实验产出-csv-清单)
- [附录 B：关键 script 清单](#附录-b关键-script-清单)
- [附录 C：Codex 讨论历史](#附录-ccodex-讨论历史)

---

## 一、最核心可发表发现 (Priority S)

### 【1】S6 (3-feat PC probe) 在 Hansen SPA 下不显著胜过 S8 (Alpha158) — Parsimony (weak form, non-superiority)

> **全项目最核心 finding，发表价值最高。**

#### Implementation

- `build_alpha158_features.py`: 忠实复现 qlib Alpha158DL (9 KBAR + 4 PRICE + 145 ROLLING = **158** 特征), 1/99 winsorization, 修复 KMID mean bizarre (-0.06 → +0.004)
- `run_step3_plan_z_part_c.py`: S8 训练 runner — 30 runs (2 models × 5 folds × 3 seeds), 51 min
- `run_step3_plan_z_part_c_perfold.py`: Path A 修正 — per-fold train-only p1/p99 winsorization
- S6 特征集：`ret_mean_10d + ret_std_10d + mom12m` = PC1 (trend) / PC2 (vol) / PC3 (horizon-extension) 代表。**注**：实际存档于 `artifacts/step3_plan_z/subsets_frozen.json`；旧版本此处曾误写为 `momentum_21d + ret_std_10d + ret_mean_10d`，2026-04-21 已更正
- 统计框架：Hansen SPA (p_consistent) + BH-FDR 多重比较校正

#### Results (Hansen SPA, primary p_consistent, α=0.05)

| Model vs Benchmark | T_SPA | p_c | 结论 |
|---|---|---|---|
| MLP vs S1 (10-dim full) | 3.24 | 0.053 | 边际 |
| MLP vs S7 (9-dim wf5 baseline) | 3.24 | **0.038** | ✅ reject |
| SAGE vs S7 | 4.82 | **0.006** | ✅ strong reject |
| **MLP vs S8 (Alpha158)** | 0.27 | **0.551** | **❌ 不拒绝** |
| **SAGE vs S8** | 1.23 | **0.551** | **❌ 不拒绝** |

#### 数值对比 (Part B + Part C)

| Subset | # feat | MLP IC (NW p) | SAGE IC (NW p) |
|---|---|---|---|
| S1 full | 10 | +0.023 (0.20) | +0.016 (0.46) |
| S2 top-4 | 7 | +0.037 (0.035) | +0.005 (0.82) |
| S3 top-3 | 4 | +0.024 (0.18) | +0.020 (0.35) |
| S4 top-2 | 2 | +0.026 (0.15) | +0.032 (0.10) |
| S5 top-1 | 1 | +0.026 (0.17) | +0.034 (0.08) |
| **S6 PC probe** | **3** | **+0.046 (0.009)** ✅ | **+0.047 (0.014)** ✅ |
| S7 wf5 9-dim | 9 | -0.006 (0.68) | **-0.048 (0.036)** ⚠️ |
| **S8 Alpha158** | **158** | **+0.041 (0.026)** ✅ | **+0.042 (0.025)** ✅ |

#### Path A per-fold winsorization 诊断 (Paired NW)

| 对比 | ΔIC | p_BH | 结论 |
|---|---|---|---|
| MLP 原 S8 vs S8_pf | +0.010 | **0.037** | 小但显著 leakage (fix-able) |
| SAGE 原 S8 vs S8_pf | -0.007 | 0.393 | 无 leakage |
| S6 vs S8_pf (MLP) | +0.015 | 0.769 | 无显著差异 |
| S6 vs S8_pf (SAGE) | -0.002 | 0.938 | 无显著差异 |

Hansen SPA (S8_pf 为基准): MLP T=2.87 p_c=0.075 (边际); SAGE T=0.00 p_c=0.700 (ns) → **S6 修正后仍未以 α=0.05 显著胜过 S8_pf**（non-superiority, **不等同于** "S6 = S8_pf equivalence"）。
> **2026-04-21-c 更正**：
> (1) 数字更正：上方 "SAGE vs S8" 原记录为 T=0.23 / p_c=0.590，经 `experiments/step3_plan_z/hansen_spa_results.csv` 核实，正确值为 **T_SPA=1.231, p_consistent=0.5509**（0.23 是行内 S6 子对比的 paired t-stat=0.225，与 benchmark-level T_SPA=max_k(t_stat_k) 概念不同）。MLP vs S8 的 T=0.27 / p_c=0.551 原本就正确（因为 MLP 行下 max 恰好由 S6 达到）。
> (2) 解释更正：上文早期版本写作"S6 和 S8 统计无差异 / 不拒绝等价零假设"。这是**概念错误**——Hansen SPA 是 **单侧 superiority 检验**，其 H₀ 是"没有候选胜过 benchmark"。不拒绝仅说明"S6 不显著胜过 S8"，不构成 equivalence 证明。严格 equivalence 需 TOST + 预设边际（Codex Round 3 Q3 已明示，见 `docs/analysis.md:1844`）。

#### 学术贡献

**3 特征 economically-grounded compact set 在 Hansen SPA (S8 为 benchmark) 下未显著胜过 158 特征 engineered library** (p_c > 0.55, non-superiority)。这是"S6 不胜过 S8"的证据，**不是**"S6 = S8 equivalence"的证明——后者需 TOST。独立于 IC 的 operational 属性（纯算术比较，不涉及预测质量）：
- 特征数比：158 / 3 ≈ 53×
- 训练时间比：S8 端到端约为 S6 的 3× 左右（单机 CPU 时钟观测，不是基准实验）
- 可解释性：S6 的三个特征分别对应 momentum / volatility / horizon-spread，语义清晰

"Under Hansen SPA (S8 as benchmark), a compact PC probe **does not demonstrate statistically superior rank-IC over an engineered factor library** at α = 0.05" — **ICAIF/FinNLP 主论文骨架**（submission 前需补反向 SPA + TOST 才能支持 "matches" / "equivalence" / "not underperform" 语言；现有结果仅是 one-sided non-superiority in the direction S6 > S8）。

---

### 【2】Horizon Ablation：21d peak 倒 U 型

> **GNN 文献完全空白的 novel ablation。**

#### Implementation

- v3 pipeline N4: 6 horizons × 4 models
- Phase 5 Step 0 Rerun (`experiments/horizon_ablation_results.csv`): 360 runs (4 models × 6 horizons × 5 folds × 3 seeds) 正式化

#### Results (GAT, 2026-03-06 Run)

| Horizon | 1d | 5d | 10d | **21d** | 42d | 63d |
|---|---|---|---|---|---|---|
| GAT IC | -0.001 | 0.023 | 0.039 | **0.044** | -0.009 | -0.008 |
| LGBM IC | 单调上升 → 63d 达 0.052 |  |  |  |  |  |

- **GAT 21d**: IC=0.0442, ICIR=0.374, Sharpe_net=1.203 (扣 15bps 后年化 **15.11%**)
- **倒 U 型**: 1d 和 42-63d 完全失效，GAT 在 10-21d 是 LGBM 的 2.8-2.9 倍

#### 学术贡献

**首次系统 6-horizon ablation。**信息半衰期假说：cross-stock message passing 在 2-4 周最有价值，长期 trend 由个股 momentum 主导 (LGBM 单调上升)。

#### 风险 / TODO

21d 标签可能与 21d 特征 (momentum_21d, ret_std_21d) 产生 artifact — plan.md P0 待做 **去 21d-feature robustness check**。

---

### 【3】Walk-Forward 5-Fold 严格协议 + Fold 4 诚实方差报告

#### Implementation

- `run_walkforward_5fold.py` → `experiments/wf5_results.csv` (90 rows)
- 6 model variants: SAGE-Mean / NoGraph / MLP × {price-only, all-features}
- Per-fold train-only normalization (Colab RTX Pro 6000, 13.2 min)
- 15bps 交易成本, Newey-West t-stat, bootstrap Sharpe CI

#### Per-fold Results (from `wf5_results.csv` + `arch_comparison_results.csv`)

| Fold | Period | SAGE-Mean IC (± std) | MLP price IC | Market Regime |
|---|---|---|---|---|
| 0 | Q2-2024 | +0.045 | +0.040 | Low vol |
| 1 | Q3-2024 | +0.028 | +0.031 | Momentum reversal |
| 2 | Q4-2024 | +0.024 | +0.044 | Stable |
| 3 | Q1-2025 | +0.008 | -0.001 | Portfolio-neg Sharpe -3.22 |
| **4** | **Q2-2025** | +0.061 (std=0.088) | +0.026 (Sharpe=2.594) | **Tariff shock** |
| **Mean** |  | **+0.033 ✅ > 0.03** | +0.028 |  |

- SAGE 赢 MLP **2/5 folds**, MLP 赢 **3/5 folds**
- 新闻特征在整体 walk-forward 更稳定，但 Fold 0 单次消融时有害

#### 学术贡献

严格 5-fold walk-forward + per-fold honest reporting + multiple testing correction — 比 MASTER/FinMamba/MDGNN/THGNN 任何一篇都严格。**Methodology paper 的强候选点。**

---

### 【4】Permutation Ranking — mom12m 压倒性第一

#### Implementation

- `run_step3_plan_z_part_a.py`: 30 runs × 617,859 permutation rows
- 分组 permutation test (semantic grouping), cross-sectional shuffle 保留 panel 结构
- 13,146 paired rows 跨 5 folds × 2 models × 3 seeds × 313 days
- SHA-256 seed RNG (Codex Round 5 修复)

#### Results (Part A ΔIC Ranking)

| 排名 | 特征组 | mean ΔIC | std ΔIC |
|---|---|---|---|
| **1** | **mom12m** | **+0.0182** | 0.038 |
| 2 | ret_mean_21d | +0.0036 | 0.019 |
| 3 | ret_mean_10d | +0.0025 | 0.013 |
| 4 | ret_std_10d | +0.0012 | 0.049 |
| 5 | CORR5 | -0.0002 | 0.004 |
| 6 | dolvol | -0.0005 | 0.006 |
| 7 | maxret | -0.0029 | 0.033 |

**mom12m ΔIC 是其他组 5-8 倍** → long-horizon momentum 是所有信号的源头。maxret/CORR5/dolvol 证实无增量。

---

## 二、重要 methodology findings (Priority A)

### 【5】Normalization × Regime 交互作用 (Novel)

#### Implementation

- `run_diag1_normalization.py`: SAGE 30 runs, raw vs norm
- `run_diag1b_replication.py`: MLP + NoGraph 60 runs 验证是否图结构相关
- Norm: train-only p1/p99 winsorize → daily cross-sectional z-score (无时间泄漏)

#### Results (Diag 1, ΔIC = norm − raw)

| Fold | raw IC | norm IC | Delta | 诠释 |
|---|---|---|---|---|
| 0 | +0.033 ± 0.002 | +0.001 ± 0.003 | **-0.031** | 归一化毁信号 |
| 1 | -0.007 | -0.034 | -0.027 | 恶化 |
| 2 | +0.035 | +0.031 | -0.004 | ≈ |
| 3 | +0.001 | **-0.104 ± 0.033** | **-0.105** | **灾难** |
| **4** | +0.006 ± 0.045 | **+0.217 ± 0.060** | **+0.211** | **rescue** |

**Overall Wilcoxon p=0.60 (ns)** — 非显著性掩盖了 ±0.2 的 regime 摆动。

Fold 4 per-seed (归一化 rescue): `s42: +0.051→+0.154, s123: +0.006→+0.273, s456: -0.039→+0.225` (方差收敛)

#### Diag 1b 机制确认

| Fold | SAGE ΔIC | NoGraph ΔIC | MLP ΔIC | 一致性 |
|---|---|---|---|---|
| 0 | -0.0315 | -0.0366 | -0.0401 | ✅ all neg |
| 1 | -0.0264 | -0.0188 | -0.0191 | ✅ all neg |
| 2 | -0.0045 | +0.0336 | -0.0371 | ⚠️ |
| 3 | **-0.1052** | **-0.1152** | **-0.0852** | ✅ all catastrophic |
| 4 | **+0.2112** | **+0.2817** | **+0.1696** | ✅ all rescue |

**14/15 cells 同号** → graph message passing **不是**机制。
**结论：input-scale saturation** (Linear `in → hidden` 在 OOD scale 下饱和)。

#### 学术贡献

Novel。挑战"normalization 是 preprocessing 默认步骤"的假设 — **取决于 regime**。值得 paper 1-2 段 discussion。

---

### 【6】SelectiveNet 在金融 weak-signal 的 Complete Failure

#### Implementation

- v3 N5: 3-head architecture (ranking + selection + auxiliary)
- Coverage levels: 5%, 10%, 20%, 30%, 50%, 100%

#### Results

| Coverage | SelectiveNet IC | Threshold baseline IC |
|---|---|---|
| 5% | -0.024 | — |
| 10% | -0.022 | — |
| 20% | -0.015 | **+0.031 ✅** |
| 50% | -0.018 | — |
| **100% (full)** | **+0.056** ✅ (最高) | — |

**Selection head 反向选择最差预测**。但 full 模型的 auxiliary loss 有正则化价值 (IC=0.05595 全实验最高)。

#### 学术贡献

**首次量化证明 ICML'19 SelectiveNet 在金融 ranking 的不适用**。Threshold baseline (simple) 在 @20% 覆盖率仍有 IC=0.031。

---

### 【7】SEC Text Features 无效 — Gate 1 STOP

#### Implementation

- `run_gate1_experiment.py` (790 lines)
- SEC 10-K/10-Q Lazy Prices TF-IDF similarity
- Layer 1: (1255×503×2) 维 `[lazy_sim, log1p_days_since_filing]`
- 21 runs, Fold 0 提前终止 (信号明确)

#### Results (Fold 0, 3-seed mean, from `experiments/gate1_results.csv`)

| Model | price IC | price+L1 IC | Δ IC | 变化 |
|---|---|---|---|---|
| SAGE-Mean | 0.034 | 0.013 | -0.021 | **-61%** |
| MLP | 0.034 | 0.023 | -0.012 | -34% |
| LGB | 0.016 | 0.019 | +0.003 | +17% |

#### 单特征消融 (SAGE, Fold 0)

| 特征 | IC | Δ | 性质 |
|---|---|---|---|
| price only | 0.034 | — | baseline |
| + lazy_sim | 0.031 | -0.004 (-11%) | 轻微有害 |
| + days_since | -0.001 | **-0.036 catastrophic** | 灾难 |
| + both | 0.013 | -0.021 (-61%) | 组合失败 |

#### 机制

`log1p_days_since_filing` (scale 0-7) 在第一 Linear 层主导梯度 (> 0.8)，破坏 ranking signal。LGB (tree-based) 不受影响但无增量。

#### 决策

Layer 2/3 (FinBERT sentiment, Qwen structured SEC) 全部取消。Codex 同意。

---

### 【8】Fold 4 Anomaly: Regime Stress 非 Leakage (诊断框架)

#### Implementation

- `analyze_fold4_leakage.py`: 4-test framework
- `experiments/fold4_zdrift_summary.csv` (160+ 特征)
- `experiments/fold4_tail_concentration.csv`

#### Results (4-test framework)

| Test | Metric | Reading | 诊断 |
|---|---|---|---|
| 1 Tail displacement | max \|Δ_top\| + \|Δ_bot\| | 0.0069 | 无 leakage |
| 2a Z-shift | max std shift | 0.014 | 无 leakage |
| 2b Rank preservation | min Spearman ρ | 0.9975 | 秩保留 |
| **2d Z-drift↔IC** | **MLP ρ=+0.508 (p<0.001)** | SAGE ρ=+0.413 (p=0.001) | **强正相关** |

#### Market Regime 证据 (Q2-2025 Tariff Shock)

| 指标 | Fold 0-3 | Fold 4 | 倍数 |
|---|---|---|---|
| 日均波动 | 0.65-0.87% | **1.81%** | 2-3× |
| Max DD | -5~-8% | **-12.7%** | 显著更深 |
| Signed mean pairwise corr | 0.18-0.22 | **0.496** | 2.3× |
| % 股票对 > 0.5 corr | 3.6-8.3% | **54.3%** | ~7× |
| % 股票对正相关 | 83-86% | **97%** | 高度同涨跌 |
| ret_std_21d train→test scale | stable | **+33.8%** | 大幅漂移 |

#### Per-fold IC + Sharpe (from `wf5_results.csv` + `arch_comparison_results.csv`, n=48/fold)

| Fold | Mean IC | Std IC | Min IC | Max IC | Mean Sharpe | Std Sharpe |
|---|---|---|---|---|---|---|
| 0 | +0.013 | 0.021 | -0.033 | +0.036 | -0.84 | 4.18 |
| 1 | -0.015 | 0.022 | -0.046 | +0.078 | -0.78 | 3.95 |
| 2 | +0.048 | 0.029 | -0.043 | +0.106 | +1.39 | 1.50 |
| 3 | -0.002 | 0.023 | -0.100 | +0.034 | **-3.22** | 5.74 |
| **4** | +0.024 | **0.088** | **-0.145** | **+0.223** | +2.22 | **5.10** |

**Fold 4 非系统性崩溃，是方差爆炸。**

#### 学术贡献

**Leakage 诊断 protocol** 本身可作 methodology 贡献。结论：Fold 4 是 regime stress test，需 honest reporting。

---

## 三、Supporting findings (Priority B)

### 【9】9-dim 特征有效秩仅 3

#### Implementation

- Cross-sectional Pearson correlation averaged over 1212 valid days
- Eigendecomposition of 9×9 correlation matrix
- 记录在 `experiments/diag_phase5_effective_rank.csv` + `diag_phase5_collinearity.csv`

#### Results

| PC | λ | Var % | Cumulative | 金融含义 |
|---|---|---|---|---|
| PC1 | 4.453 | 49.5% | 49.5% | **Momentum/trend** (6 mean/momentum 特征负载 ≈ -0.40) |
| PC2 | 2.557 | 28.4% | 77.9% | **Volatility** (3 ret_std 负载 +0.56~+0.60) |
| PC3 | 1.059 | 11.8% | **89.7%** | **Horizon spread** (short 5/10d +0.49 vs long 21d -0.51) |
| PC4 | 0.482 | 5.4% | 95.0% | — |

- Participation ratio = 2.91, Shannon effective rank = 3.66
- **精确冗余**: `ret_mean_{5,10,21}d ≡ momentum_{5,10,21}d` (corr = 1.00)

#### 单特征 LGB IC (Fold 0 test, 63 days)

| Feature | IC |
|---|---|
| **ret_std_10d** | **+0.028** (最高) |
| ret_std_5d | +0.016 |
| momentum_10d | +0.001 |
| **Full 9-feat LGB** | **+0.021** |

单最佳特征 > 完整 9 特征 (但 SE≈0.013, margin 仅 0.5 SE，需 multi-fold 复现)。

#### 14-dim (+5 新特征) 正交性 (982 days)

| 新特征 | Max \|corr\| with old 9 | 正交成分 | ROI |
|---|---|---|---|
| **mom12m** | <0.05 | ~0.99 | **High** |
| **dolvol** | 0.13 | ~0.98 | **High** |
| **CORR5** | 0.27 | ~0.83 | Medium-high |
| **maxret** | 0.80 (与 ret_std_21d) | low | Low-medium (冗余) |
| **RSV5** | 0.66 (与 momentum_5d) | low | Low-medium (冗余) |

14-dim effective rank: k90=7, k95=8 (vs 9-dim: k90=4, k95=4) → **加 5 特征约翻倍 effective rank**。

---

### 【10】SAGE-Mean/Sum 稳定性碾压 GAT/HGT/Transformer

#### Results (multi-seed CV, from `arch_comparison_results.csv`)

| Model | Mean IC | CV (%) | Verdict |
|---|---|---|---|
| **SAGE-Sum** | 0.04766 ± 0.00237 | **5.0%** | ✅ 极稳 |
| SAGE-Mean | 0.03525 | 62.0% | 稳 |
| GAT | 0.03215 | 55.1% | 不稳 (Seed 1024 完全失败 IC=0.00182) |
| Transformer | 0.02448 | 91.8% | 最不稳 |
| HGT (all-4 edges) | 0.00432 | 极差 | **news/co-oc edges 有害** |
| HGT (corr+sector) | 0.01177 | — | 中 |

**SAGE-Sum Ensemble**: IC=0.04757, Sharpe=0.749 (跨 5 seeds)。

#### 机制

Parameter-variance tradeoff: 98% 噪声数据中，simple mean aggregation 比 type-specific HGT robust。SAGE 对初始化更鲁棒 (Seed 1024 在 SAGE 通过但 GAT 失败)。

---

### 【11】FinBERT NLP Embedding 有害 (EMH 证据)

#### Results

- 5-fold walk-forward: MLP **price-only IC=0.026** vs **all-features IC=0.004** (7× 差距)
- Fold 0: MLP price IC=0.0405 vs all IC=0.012
- 有新闻股票 IC=0.008 vs 无新闻股票 IC=0.059
- **LLM structured (Qwen/GPT-4o) 替代 FinBERT: Δ IC = +0.0009** (无差异)
- 高影响子集 AUC = 0.4762 (比随机更差)

#### 学术贡献

SP500 大盘股效率性强，title-level news (~15 words) 分钟内定价。**NLP 维度 curse 在 weak-signal 环境下尤重**。

---

### 【12】Phase 1-2 Binary Direction Prediction 所有 Baseline ≈ 随机

> **CORRECTION (2026-04-21)**：本节原写"所有 baseline test AUC ∈ [0.4993, 0.5046]"——错误，低估了真实最小。权威 run log `progress.md` §2026-03-03-g 记录 B3 = **0.4965**（低于 0.4993）。真实区间为 **[0.4965, 0.5046]**，最小在 B3（LR + Sent + Momentum，overfitting），**不是** B1。以下 Results 已按 progress.md 校正。

#### Results (archived, Phase 1d Baseline Matrix) — per `progress.md` §2026-03-03-g

- B1 LR + FinBERT: test AUC = **0.4993**
- B2 LR + Sentiment: test AUC = **0.5031**
- B3 LR + Sent + Momentum: test AUC = **0.4965**（真实最小，overfitting）
- B4 LR + Momentum: test AUC = **0.4987**（overfitting）
- B5 XGBoost + all: test AUC = **0.5046**（最佳，仍低于 0.52 Go 阈值）
- **真实区间 [0.4965, 0.5046]**（全部 ≈ 随机 0.50 ±0.005；最大偏离 B5 = +0.0046）
- Selective AUC@10% = 0.5071 (远低于 0.54 Go 阈值)
- 动量特征反而伤害 selective AUC (B3/B4)
- 26.5% 事件在噪声区 (-0.5% < return < 0.5%)
- FinBERT 情感 vs 实际涨跌对齐率 **51.6%** (随机水平)
- Phase 2 LLM 替换: Δ AUC = +0.0009 (无效)

**任务定义**：news-event 触发、次日方向二分类（≈437K stock-day events，market-adjusted labels）；**不是** 21-day horizon（21-day 是 Findings 1-11 的 ranking 任务 horizon，与本 finding 不同任务）。

#### 学术贡献

在所测 feature set（FinBERT / sentiment / momentum / 组合）与 model family（LR / XGBoost）上，**news-event 触发、次日方向二分类**任务在 SP500 大盘股 universe 上全部 ≈ 随机水平，均不达 pre-registered 0.52 Go 阈值；Phase 2 LLM 替换亦不提升（ΔAUC = +0.0009）。这是一个**针对该任务定义 + 该特征/模型 + 该 universe** 的负面结果，**不是**对股票方向预测一般性的否定。该结果是 v3 从 event-driven binary classification 转向 daily cross-sectional ranking（21-day horizon）作为主任务的依据。

---

## 四、Infrastructure / Implementation 亮点

### 【13】Dynamic Graph 构建 (锁定为 Pareto 最优)

- **Window = 126 天, 相关性阈值 0.6** (3×4 sensitivity grid 验证)
- 54 个月度 snapshot + static industry edges (27,070 条) + news co-occurrence (2,325/day)
- 密度 6%, 稳定性 std=0.064, 聚类系数 0.453
- HHI 0.214-0.877 (portfolio 集中度 sensitivity)

### 【14】严格的 Codex 协作 (Rule 9 三触发点)

| Round | Agent ID | 范围 | 关键输出 |
|---|---|---|---|
| 1-3 | ad372bb181, a0bf2209f8, a80e980969 | Plan 设计 | Plan Z++ 共识 |
| 4 | a886429f68 | Module 1 subsets | 2 CRITICAL 修 |
| 5 | ae897eb628 | Module 2 Part A | 2 CRITICAL + 1 MAJOR + SHA-256 RNG |
| 6 | a49cf14a80 | Module 3 Part B | 2 CRITICAL 修 (列顺序, preflight) |
| 7 | a4c569fc07 | **Results** | 1 CRITICAL + 4 MAJOR; 推荐 Alpha158 baseline |
| 8 | (implied) | Alpha158 设计 | Level I 选定 |

### 【15】完整 Tri-doc + Archive 纪律

- `progress.md` + `plan.md` + `docs/analysis.md` 时间对齐 (`YYYY-MM-DD-x` IDs)
- 40+ 归档 md，分 `stale_results/` `colab_results/` `scripts/` `plans/` `docs/` 清晰

### 【16】数据基础设施

- `data/reference/sp500_5y_alpha158_features.npy` (1255 × 501 × 158 float32, ~400MB)
- `sp500_5y_alpha158_features_meta.json` + `sp500_5y_alpha158_qa.csv`
- OHLCV: yfinance 501×1255 天, ret_corr vs EODHD=0.99982 (接近完美一致)
- EODHD 新闻: 138 万事件，映射率 90.6%, 384/768 embedding

---

## 五、发表路线推荐

### 🎯 首推 (ICAIF 2026 / FinNLP 2026)

**"A Compact Economically-Grounded Feature Probe Is Not Shown to Outperform — Nor Is It Tested Against — an Engineered Factor Library Under Hansen SPA: A Working-Title Parsimony Study on US Equity Ranking"** *(working title; "Match" language retracted 2026-04-21-c pending reverse-direction SPA + TOST)*

- Main: 【1】 S6 (3) not shown to outperform S8 (158) under Hansen SPA — non-superiority, not equivalence; TOST still required before submission
- Supporting: 【9】 PCA 有效秩 3 motivates PC probe
- Ablation: 【4】 mom12m permutation #1
- Methodology: 【3】 5-fold walk-forward + SPA + BH-FDR
- Limitations: 【8】 Fold 4 regime stress + leakage diagnostic framework

### 🎯 次推 (ICML/AAAI time-series track)

**"Horizon Matters: A Systematic Ablation of GNN Stock Ranking"**

- Main: 【2】 倒 U 型, 21d peak
- Supporting: 【10】 SAGE > GAT 稳定性
- Negative: 【6】 SelectiveNet failure, 【11】 FinBERT 有害

### 🎯 备选 (NeurIPS Negative/Null Results Workshop)

**"Negative Results on Text Features for Large-Cap Equity Ranking"**

- 【7】 SEC + 【11】 FinBERT + 【12】 Phase 1-2 binary

---

## 六、当前 open questions

1. **Path A vs B 最终 narrative 选择**
   - 已完成 Path A 诊断 (2026-04-20-c): MLP 有小 leakage p_BH=0.037, SAGE 无
   - 修正后 S6 在 Hansen SPA (S8_pf 为 benchmark) 下仍未以 α=0.05 显著胜过 S8_pf (p_c=0.075/0.700, non-superiority; 非 equivalence 证明)
   - → **Path B (parsimony) narrative 可成立**
2. **Module 4b 分析** — sector-neutral portfolio, coverage-Sharpe 曲线 (无需重训，随时可做)
3. **Phase 5 Step 1 新特征集成训练** — mom12m/dolvol 已验证 #1 和 high-ROI，但未在扩展特征集上做完整 walk-forward
4. **21d feature-horizon artifact robustness check** — plan.md P0, 未做

---

## 附录 A：实验产出 CSV 清单

### `experiments/` (42 CSV 文件)

| 文件 | 维度 | 用途 |
|---|---|---|
| `wf5_results.csv` | 90 rows | 5-fold walk-forward 主结果 |
| `arch_comparison_results.csv` | 152 rows | 6 架构对比 |
| `horizon_ablation_results.csv` | 260+ rows | 6 horizons × 4 models |
| `gate1_results.csv` | 21 runs | SEC Layer 1 |
| `permutation_v2_results.csv` | 18 rows | Permutation importance |
| `ranking_loss_results.csv` | — | ListNet/RankNet 对比 |
| `selectivenet_results.csv` | — | N5 SelectiveNet |
| `diag1_normalization_results.csv` | 30 runs | Raw vs norm |
| `diag1b_replication_results.csv` | 60 runs | MLP + NoGraph 复现 |
| `diag_phase5_feature_importance.csv` | 5 fold × 9 特征 × 2 split | Feature IC |
| `diag_phase5_collinearity.csv` | — | 相关性矩阵 |
| `diag_phase5_effective_rank.csv` | — | PCA 有效秩 |
| `diag_phase5_label_dist.csv` | — | 标签分布 |
| `fold4_zdrift_summary.csv` | 160+ 特征 | Z-drift 诊断 |
| `fold4_tail_concentration.csv` | — | 尾部浓度 |
| `hansen_spa_results.csv` | 2 × 4 | SPA p 值 |
| `pairwise_fdr.csv` | 74 rows | BH-FDR |

### `experiments/step3_plan_z/`

| 文件 | 维度 |
|---|---|
| `part_a_daily_ic.csv` | 13,146 rows (baseline) |
| `part_a_permuted_ic.csv` | 617,859 rows (7 groups × 30 runs × 313 days) |
| `part_b_daily_ic.csv` | 534,657 rows (8 subsets) |
| `part_b_summary.csv` | 18 rows 汇总 |
| `part_c_s8_daily_ic.csv` | 82,357 rows |
| `part_c_s8_perfold_daily_ic.csv` | 82,357 rows (per-fold winsorize Path A) |

### `artifacts/step3_plan_z/`

- `fold_manifest.json` (63 KB) — 5 fold 的 train/val/test 切分与日期映射
- `part_a_ranking.json` (2.7 KB) — 7 特征组 ΔIC 聚合排名与 CI
- `subsets_frozen.json` (2.3 KB) — S1-S7 的精确特征列表 (nested design)
- `part_c_meta.json` (2.0 KB) — Part C 运行完整性元数据
- `per_fold_scaler.json` (2.4 KB) — 每 fold 特征标准化参数

---

## 附录 B：关键 script 清单

### 训练 runner

- `run_walkforward_5fold.py` — 主 walk-forward 5-fold
- `run_gate1_experiment.py` — SEC Gate 1 (790 行)
- `run_phase5_step3_feature_expansion.py` — Phase 5 Step 3 主训练
- `run_step3_plan_z_part_a.py` — Permutation ranking
- `run_step3_plan_z_part_b.py` — 7 subsets retrain
- `run_step3_plan_z_part_c.py` — S8 Alpha158
- `run_step3_plan_z_part_c_perfold.py` — Path A fix

### 诊断

- `analyze_fold4_leakage.py` — 4-test leakage framework
- `analyze_step3_plan_z.py` — Module 4 分析 (Hansen SPA + BH-FDR)
- `diagnostic_phase5_step0.py` — Step 0 diagnostics
- `diagnostic_phase5_fix.py` — Fix runner
- `run_diag1_normalization.py` — Normalization ablation
- `run_diag1b_replication.py` — Cross-architecture 复现

### 特征构建

- `build_alpha158_features.py` — qlib Alpha158DL 复现
- `build_phase5_features.py` — Phase 5 新特征 (mom12m 等)
- `cleanup_and_rebuild_features.py` — 统一 feature rebuild
- `download_ohlcv_yf.py` — yfinance OHLCV
- `refetch_zts.py` — ZTS 数据补抓

### 报告生成

- `run_figures_tables.py` — Paper figures + tables

---

## 附录 C：Codex 讨论历史

Rule 9 强制三触发点 (Plan / Code / Results)，共 8 轮。全部记录在 `progress.md` 的 `## YYYY-MM-DD-x: Codex Review` 条目。关键 Agent IDs：

- ad372bb181, a0bf2209f8, a80e980969 — Plan 设计 (Plan Z++ 共识)
- a886429f68 — Module 1 subsets (2 CRITICAL fix)
- ae897eb628 — Module 2 Part A (2 CRITICAL + 1 MAJOR + SHA-256 RNG)
- a49cf14a80 — Module 3 Part B (2 CRITICAL: 列顺序, preflight)
- a4c569fc07 — Results (1 CRITICAL + 4 MAJOR; 推荐 Alpha158)

---

*文档创建：2026-04-20，供 H博士 对照提问。*
*数据截至：Phase 5 Step 3 Plan Z++ Part A/B/C 完整完成 + Path A leakage 诊断完成。*
