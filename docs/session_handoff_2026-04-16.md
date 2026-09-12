# Session Handoff — 2026-04-16

> 新窗口开始前必读。当前 session 的完整状态。

---

## 当前状态：无后台进程

Step 0 所有实验已完成，无运行中进程。

---

## 本轮 Session 做了什么

### Step 0: Pending Reruns 全部完成

三个修复后的大规模实验全部在本地 M4 MPS 上完成：

#### 1. Horizon Ablation (360 runs) ✅
- 文件: `experiments/horizon_ablation_results.csv`
- 预测缓存: `experiments/horizon_preds/*.npy` (60 files)
- **核心发现: "倒 U 型" 消失**
  - 原版 (GAT, single fold, pre-fix): peak at 21d
  - 现版 (SAGE-Mean, 5-fold WF, post-fix): peak at 63d, 但 63d 被 Fold 4 (Q2-2025) 严重扭曲
  - 21d 是最可靠 horizon: Bootstrap 95% CI [+0.006, +0.048], 唯一排除 0
  - MLP > SAGE at all price horizons, 但均不显著 (Wilcoxon p > 0.05)
  - SAGE > MLP at 21d all features (p=0.02)

#### 2. Architecture Comparison (150 runs) ✅
- 文件: `experiments/arch_comparison_results.csv`
- **Price features: 5 种架构 IC 无显著差异**
  - SAGE-Sum (0.039) > MLP (0.037) > Transformer (0.027) > SAGE-Mean (0.026) > GAT (0.022)
  - 所有 pairwise Wilcoxon vs MLP: ns
- **All features: SAGE-Mean (0.011) 领先, MLP (-0.008) 无信号**
  - SAGE vs MLP p=0.107 (接近但未达显著)

#### 3. Permutation Test v2 (16 models × 1000 shuffles) ✅
- 文件: `experiments/permutation_v2_results.csv`, `plots/paper_permutation_test_v2.png`
- Per-day cross-sectional shuffle (正确方法)
- **Price models 全部 p=0.000** (信号真实)
- **SAGE-Mean_all p=0.002** (弱信号但真实)
- **MLP_all p=1.000** (无信号 → graph 正则化价值确认)

### Codex 讨论 (2 轮，Phase 5 方案批判性评估)

#### 达成共识：

| 议题 | 结论 |
|------|------|
| **大运行分拆** | 不混合三个变更。Run A (9-dim + normalization) 先跑，验证后再 Run B (14-dim) |
| **ListNet** | 踢出大运行。待主结果稳定后单独评估 |
| **倒 U 型** | 贡献重新框架为 "21d 最可靠, 长 horizon 受 fold-specific 市场环境驱动" |
| **Fold 4** | 5-fold 为主要结果, 含/不含 F4 作为敏感性分析 |
| **截面归一化** | 应用于所有特征 (新旧), 需重跑 9-dim baseline (在 Run A 中) |
| **MLP 控制组** | 每个 SAGE run 必须有匹配的 MLP run |
| **特征分阶段** | 先 11-dim (mom12m+maxret, 只需 close), 再加 volume 类 (dolvol+CORR5) |
| **RSV5** | 排除在主实验外, 除非验证 EODHD adjusted OHLC 正确 |
| **Pre-specified tests** | 等最终特征集确定后锁定; 以 SAGE_best_rolling 为锚点定义 3 个对比 |

---

## Phase 5 修订后的执行计划

```
Step 0: Pending reruns ✅ DONE
Step 1: 实现截面归一化 + 2 个 close-only 新特征 (mom12m, maxret)
Step 2: Run A — {9-dim, 11-dim} × {SAGE, MLP} × {frozen, rolling} × 5 folds × 3 seeds
         = 4 feature-graph combos × 2 models × 5 folds × 3 seeds = 120 runs
Step 3: 分析 Run A → 决定是否继续 Run B
Step 4: (如 Go) 下载 Volume 数据, 实现 dolvol + CORR5 → 13-dim
Step 5: Run B — 13-dim × {SAGE, MLP} × {frozen, rolling} × 5 folds × 3 seeds = 60 runs
Step 6: 锁定最终特征集 + 图配置, pre-register test family
Step 7: 不需重新训练的分析 (R5/R6/R8/S7)
Step 8: 论文图表 + 初稿
```

### 不需训练的待做分析 (用现有 cached predictions)

| 分析 | 来源 | 用哪些 cached preds |
|------|------|-------------------|
| R5: Sector-neutral portfolio | Adversarial review | horizon_preds/*.npy |
| R6: Coverage-Sharpe-Turnover 曲线 | Adversarial review | selectivenet cached |
| R8: Multiple testing (Bonferroni/BH) | Adversarial review | permutation_v2_results.csv |
| S7: Random selection baseline | Adversarial review | horizon_preds/*.npy |

这些可以在任何时候做, 不阻塞 Step 1-5。

---

## 关键文件

| 文件 | 内容 |
|------|------|
| `experiments/horizon_ablation_results.csv` | 360 runs (修复版) |
| `experiments/arch_comparison_results.csv` | 150 runs (修复版) |
| `experiments/permutation_v2_results.csv` | 16 models × 1000 shuffles |
| `experiments/horizon_preds/*.npy` | 21d 预测缓存 (60 files) |
| `experiments/wf5_results.csv` | v4 5-fold WF (90 runs) |
| `plots/paper_permutation_test_v2.png` | Permutation 图表 |
| `docs/adversarial_review_2026-04-12.md` | 对抗性审查 + 待做实验清单 |
| `docs/decisions.md` | 所有技术决策 |

---

## 遗留问题（需 H博士 决策）

1. **截面归一化细节**: Winsorize percentile (1/99 还是 5/95)? 填充策略 (NaN → 0 after z-score)?
2. **EODHD Volume 数据**: 需要 API 调用下载, 是否有 quota 限制?
3. **RSV5**: 是否值得验证 adjusted OHLC? 还是直接放弃, 做 13-dim?
4. **Colab 用不用**: Run A (120 runs) 在 M4 上约 12-15h, Colab T4 约 6-8h
5. **论文定位微调**: "When does graph help" 仍然成立, 但需要更强调 reliability + conditions

---

## 项目综合状态

| 阶段 | 状态 |
|------|------|
| v4 5-fold walk-forward (90 runs) | ✅ DONE |
| True MLP baseline | ✅ DONE |
| Data leakage C1-C3 fix | ✅ DONE |
| SEC Layer 1 Gate 1 | ✅ STOP |
| **Step 0 reruns (horizon+arch+perm)** | ✅ **DONE** |
| Adversarial review P0 analyses (R5/R6/R8/S7) | ⏳ PENDING (no retraining needed) |
| **Phase 5 Step 1: 新特征 + 归一化** | ⏳ **NEXT** |
| Phase 5 Run A: normalized baseline | ⏳ PENDING |
| 论文图表 + 初稿 | ⏳ PENDING |

---

*Written: 2026-04-16*
