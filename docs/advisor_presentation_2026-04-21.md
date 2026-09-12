# GNN-Testing — Findings 可视化汇报（导师版）

> 2026-04-21 编写。来自 SP500 日频股票排名项目的 12 条实证发现。每节包含
> (1) 是什么、(2) 怎么实现、(3) 图说了什么、(4) 克制的结论。数字全部从
> **Appendix B** 列出的 CSV/JSON 产出里直接提取，未人工修饰。术语的中文解释
> 集中在 **Appendix A**，代码文件与数据产物清单集中在 **Appendix B**。

**实验公共设定**

- Universe：SP500 成分股 501 只，2020-01 至 2025-12 的日频 OHLCV（1,255 个交易日）。
- 目标：次日 cross-sectional return rank（主实验 21-day horizon，except noted）。
- 特征：9-dim price/volatility probe（主力）+ 158-dim Alpha158 库（对照）。
- 评估：5-fold walk-forward + per-fold train-only normalization；日度 IC 用
  Newey-West (NW) 校正 t 值；多模型比较用 Hansen SPA + BH-FDR；Sharpe 扣 15 bps
  round-trip 交易成本。
- Seeds：每个配置 3 粒（42 / 123 / 456），seed 内可复现。

---

## Finding 1 — 3-Feature "PC Probe" 在 Hansen SPA 下**不显著胜过** 158-Feature Alpha158

**是什么**。Hansen SPA 检验（以 S8 Alpha158 为 benchmark）**未能拒绝"任何候选子集都不胜过 benchmark"的零假设**：即手工 3 特征 PC probe（S6 = ret_mean_10d + ret_std_10d + mom12m，分别代表 PC1 trend / PC2 vol / PC3 horizon-extension）与完整 qlib Alpha158 库（S8 = 158 engineered features）的日度 IC 均在 0.041–0.047 区间（313 个测试日覆盖 5 个 walk-forward fold），S6 未以 α = 0.05 显著胜过 S8（MLP: T_SPA = 0.270, p_consistent = 0.5506；SAGE-Mean: T_SPA = 1.231, p_consistent = 0.5509）。**注意**：这是 **non-superiority** 结果，不等同于"S6 = S8 equivalence"——Hansen SPA 是单侧 superiority 检验，不检验等价性；严格的 equivalence 主张需要 TOST（two one-sided tests）加预设边际 δ（Codex Round 3 Q3 明示，见 `docs/analysis.md:1844`）。

**怎么实现**。S8 在 `build_alpha158_features.py` 中忠实复现 qlib Alpha158DL
operator 集合（9 K-bar + 4 price + 145 rolling，并施加 1/99 winsorization）。
训练由 `run_step3_plan_z_part_c.py` 驱动（30 runs = 2 models × 5 folds × 3
seeds）。`run_step3_plan_z_part_c_perfold.py` 是 "Path A" 的严格 per-fold
train-only winsorization 复跑，用于排除残余泄露。聚合 + SPA 计算在
`analyze_step3_plan_z.py`。

**图**。
![Finding 1](../plots/advisor/fig_01_s6_vs_s8.png)

**图说**。Panel (a) 绘制 9 个 subsets × 2 个模型的 seed 平均日度 IC + NW 95%
CI。S6（3 特征）与 S8（Alpha158）在 MLP 与 SAGE-Mean 下柱高几乎相等，两者
均显著（星号：S6 MLP p=0.009 **；S8 MLP p=0.026 *）。Panel (b) 绘制 S6 专属
的 studentized paired t-statistic（S6 − benchmark，取自 Hansen SPA 的每
alternative `t_stats` 字典）。4 条柱全部在 |t| < 1.96 内（two-sided α=0.05，
asymptotic normality 下），即 S6 与 Alpha158 benchmark 在每个 (model, benchmark)
组合上都统计不可区分。Path-A 修正（S8_pf）修掉了一个小而真实的 MLP leakage
（ΔIC ≈ +0.010，BH p = 0.037）；修正后 S6 vs S8_pf 的 t 值仍远低于 1.96
（MLP t = +0.76，SAGE t = −0.08）。

**结论**。在以 S8 为 benchmark 的 Hansen SPA 下，**无任何候选子集（含 S6）在 α=0.05
下显著胜过 S8** (non-superiority)。这仅是一侧的结果——**我们未跑 SPA 以 S6 为 benchmark
的反方向测试**，故不能断言 "S8 未胜过 S6"。Panel (b) 的 per-pair |t|<1.96 是 failure-to-reject
point null of zero mean-IC difference，同样是 failure to reject 而非 positive evidence。
因此当前数据仅支持最弱叙事：**"S6 not shown to outperform S8 at α=0.05"**；任何形如
"S6 = S8" / "S6 ≈ S8" / "S6 与 S8 并列" / "S6 不差于 S8" 的 equivalence / non-inferiority
主张都**尚未被检验**，需在 paper submission 前补 TOST + 预设边际 δ（以及反向 SPA 以
完成对称 non-superiority 证据）。

---

## Finding 2 — Price-Only 模型在 21-Day Horizon 处 IC 最高

**是什么**。在 horizon ∈ {1, 5, 10, 21, 42, 63} 天上，SAGE-Mean 与 MLP 两个
price-only 模型的 mean IC 在 21 天处达到峰值（Folds 0–3，3 seeds）。MLP IC
由 0.012（1-day）上升到 0.026（21-day），再回落到 0.012（63-day）；SAGE-Mean
IC 由 0.009 → 0.024（21-day）→ 0.002（63-day）。

**怎么实现**。360-run 扫描由 `run_phase5_step3_feature_expansion.py` 驱动，
产出 `experiments/horizon_ablation_results.csv`（4 models × 6 horizons × 5
folds × 3 seeds = 360 行）。图中**排除了 Fold 4**，因为其 Q2-2025 tariff shock
regime 的方差异常（见 Finding 8）会掩盖跨 horizon 趋势；若纳入 Fold 4，会把
63-day mean IC 人为拉高。All-feature（news-augmented）变体未绘制——Finding 11
表明它跨折降低 price-only IC。

**图**。
![Finding 2](../plots/advisor/fig_02_horizon_ablation.png)

**图说**。两条曲线呈倒 U 形，最大值位于 21 天附近（金色竖带标出）。误差条
为 ±1 SE（3 seeds × 4 folds = 12 runs / point）。形状与 "cross-sectional rank
signal 在 2-4 周最丰富" 假说一致——太短被噪声主导，太长被个股 drift 主导。

**结论**。这是**预先设定的 horizon ablation**，不是 post-hoc 调参。21 天峰值
的绝对幅度不大（IC ≈ 0.025），且仅在 price-only 变体 + 本 universe 成立，
**不主张**存在普适的 21 天最优。

---

## Finding 3 — 5-Fold Walk-Forward：均值稳定，Fold 4 方差爆炸

**是什么**。严格的 5-fold walk-forward（90 runs）在 2024-H2 至 2025-H2 周期上
SAGE-Mean 与 MLP 的 mean IC 均为正，但 Fold 4（Q2-2025）的方差是其他 folds 的
3-5×（`wf5_results.csv` 上直接计算）：其他 folds 的 IC std ≈ 0.02，Fold 4
升至 ≈ 0.09；IC 范围由约 ±0.05 扩到 ±0.15。

**怎么实现**。由 `run_walkforward_5fold.py` 驱动，输出 `experiments/wf5_results.csv`
（90 行，6 个 model × feature 变体 × 5 folds × 3 seeds）。Per-fold 训练使用
train-only p1/p99 winsorization + cross-sectional z-score；validation 与 test
不拟合 normalization。Colab RTX Pro 6000 上跑约 13 分钟。

**图**。
![Finding 3](../plots/advisor/fig_03_wf5_stability.png)

**图说**。箱线图展示每个 fold 在 3 seeds 上的 IC 分布（叠加 strip 点）。黄色
竖带标出 Fold 4。注意 MLP price-only（浅红色，最右）在 Fold 4 达到最高中位数
IC，而 MLP all-features（深红色）崩塌到 −0.04——news-augmented 变体在 Fold 4
的范围最大，与 Finding 11 + Fold 4 作为 feature-distribution-shift 压力测试
（Finding 8）相吻合。

**结论**。Fold 级均值总体在 IC = 0.03 目标之上，但 Fold 4 的方差**不能**被
3 seeds 的平均所压制，应诚实报告而非聚合掩盖。

---

## Finding 4 — Permutation Importance：mom12m 贡献约为其他特征组的 5×

**是什么**。在覆盖 7 个特征组（14-dim 扩展集之上）的 30-run cross-sectional
permutation ranking 中，12-month momentum 组（`mom12m`）的 ΔIC = +0.0182
（baseline − shuffled），约为次名 `ret_mean_21d`（+0.0036）和
`ret_mean_10d`（+0.0025）的 5-8×。三个组的 mean ΔIC 为轻微负值
（`CORR5` −0.0002；`dolvol` −0.0005；`maxret` −0.0029），即 shuffle 后 IC
未下降或略有上升。

**怎么实现**。`run_step3_plan_z_part_a.py` 用 SHA-256 seeded RNG（Codex Round 5
修复）做分组 cross-sectional permutation。30 runs × 7 groups × 313 test days =
617,859 行 permuted 日度 IC，paired with 13,146 行 baseline IC。聚合在
`analyze_step3_plan_z.py`，汇总 JSON 为
`artifacts/step3_plan_z/part_a_ranking.json`。

**图**。
![Finding 4](../plots/advisor/fig_04_permutation_ranking.png)

**图说**。横向条形图显示每个特征组的 mean ΔIC，误差条为 ±1 SE（30 runs）。
`mom12m` 是**唯一**一个 shuffle 后模型 IC 显著下降的组。`ret_std_10d` 与
`maxret` 的误差条跨越 0，它们的点估计与 "无效应" 不可区分。

**结论**。信号集中在 long-horizon momentum；我们测试的大多数其他 engineered
组（volume、cross-stock correlation、tail return）贡献有限。这也为 Finding 1
的 S6 "PC probe" 选择提供了依据。

---

## Finding 5 — Normalization × Regime：图模型与非图模型同号效应

**是什么**。用标准 train-only cross-sectional z-score normalization 替换原始
scaling，IC 的改善或恶化**依赖 fold**，而且在 SAGE-Mean（graph）、NoGraph
（无图）、MLP（无图无池化）上的符号几乎一致：15 个 (model × fold) cells 中
有 14 个同号，因此**交互发生在 regime，而非 graph 结构**。

**怎么实现**。`run_diag1_normalization.py` 跑 SAGE-Mean × 30 runs（raw vs.
norm）；复现实验 `run_diag1b_replication.py` 跑 MLP × NoGraph × 60 runs，用于
检验是否 GNN 专属。输出：`experiments/diag1_normalization_results.csv`（30 行）
与 `diag1b_replication_results.csv`（60 行）。

**图**。
![Finding 5](../plots/advisor/fig_05_norm_regime.png)

**图说**。热图 cells 为 3 seeds 平均后的 (ΔIC = norm − raw)。蓝 cells：
normalization 损害（Folds 0、1、3）；红 cells：normalization 帮助（主要是
Fold 4）。Fold 3 下所有 3 个模型均被 normalization 拉低约 −0.10；Fold 4 下
均被 rescue +0.17 至 +0.28。跨模型符号一致（14/15 同号）排除了 GNN-specific
机制，剩余解释是第一线性层在 OOD 特征尺度下的 **input-scale saturation**，
regime 变化会让 per-fold scaler 的统计量产生这种效果。

**结论**。Normalization **不是**对弱平稳金融数据普适安全的预处理步骤，
是否采用与 regime 有关。总体 mean effect 统计上与 0 无差异
（Wilcoxon p = 0.60），因此也不能反过来主张 "normalize = 更差"。

---

## Finding 6 — SelectiveNet 在低覆盖率下输给简单 Threshold Baseline

**是什么**。在 10 个 coverage targets（10–100%）上，ICML-2019 SelectiveNet
3-head 架构（数据里标为 "E2E"）的 IC 曲线**全程低于**按分数幅值 drop 低
置信预测的简单 Threshold baseline。10% coverage 时：Threshold IC = 0.084 vs.
E2E IC = 0.048。

**怎么实现**。SelectiveNet 与两个 baseline（Threshold, Vol-Calibrated）
在 v3 pipeline N5 实验中训练。输出 `experiments/selectivenet_results.csv`
（70 行）。E2E（SelectiveNet）与 Vol-Calibrated 各有 3 个 calibration-target
变体（target ∈ {0.2, 0.4, 0.6}），图中每条曲线展示的是对应策略在每个
coverage 上**跨 calibration targets 的 envelope（IC 最大值）**——这是对
非 Threshold 策略最宽容的解读。

**图**。
![Finding 6](../plots/advisor/fig_06_selectivenet.png)

**图说**。Threshold（绿）在低覆盖（10–50%）区间领先最大——这正是 selective
prediction 系统本应增值的区间。60% 覆盖以上三条曲线相交，Vol-Calibrated 略
超 Threshold；100% 覆盖处三者 IC 收敛在 0.02 以内，与 selection head 在饱和
regime 坍缩至"全接受"一致。SelectiveNet（红）在低覆盖端是最弱的策略，而
低覆盖恰是 selective 的使用场景。

**结论**。在这个弱信号 + 高噪声场景下，learned selection head 并不能比
"挑最极端预测分数" 恢复出更有信心的子集。该结论仅针对本任务，**不是**对
SelectiveNet 的普适否定。

---

## Finding 7 — SEC Lazy-Prices 相似度特征降低 NN 排名 IC

**是什么**。在 price features 上加**同时包含两个** SEC 10-K/10-Q Lazy-Prices
特征（`lazy_sim` + `log1p(days_since_filing)`）的组合，IC 在 SAGE-Mean 上从
0.034 → 0.013（−61%）、在 MLP 上从 0.034 → 0.023（−34%），但在 LightGBM
基线上近似不变（0.016 → 0.019）。SAGE-Mean 的单特征消融表明破坏主要来自
`days_since_filing`：单独加 `lazy_sim` 仅下降 ~11%（0.034 → 0.031），
单独加 `days_since` 则把 IC 拉到近 0（−0.004）。

**怎么实现**。`run_gate1_experiment.py`（790 行；21 runs 限定在 Fold 0
Q2-2024）是唯一驱动。输出 `experiments/gate1_results.csv`（22 行）。消融在
SAGE-Mean 上测试 4 个变体（price / +lazy_sim / +days_since / +both），在 MLP
与 LGB 上测 2 个（price / +both）。

**图**。
![Finding 7](../plots/advisor/fig_07_sec_gate1.png)

**图说**。柱高为 Fold 0 上 3 seeds 的 mean IC。SAGE-Mean 加 `lazy_sim` 单独
仅损 ≈ 0.003 IC（可接受）；加 `days_since` 单独则把 IC 拉到近 0（橙色柱）
——该特征 0–7 的 log scale 主导了第一线性层的梯度。Tree baseline（LGB）
不受影响，这符合 "scaling/saturation 机制而非信号质量问题" 的解释。

**结论**。实验作用域限定在 Fold 0，因此只能报告**决定**（停止 SEC Layer 2/3）
而非跨 regime 的泛化结论。NN-vs-tree 的非对称性是可解释部分。

---

## Finding 8 — Fold 4 异常是 Regime Stress，不是 Label Leakage

**是什么**。Fold 4（Q2-2025）有升高的日度 feature-distribution drift
（z-drift）**和**升高的日度 IC。两者强烈共动：62 个测试日上 Pearson ρ 为
+0.420（MLP，p = 7×10⁻⁴）与 +0.476（SAGE-Mean，p = 9×10⁻⁵）。4 项泄露检测
（tail displacement、z-shift、rank preservation、tail concentration）
**全部通过**。因此升高的 IC 与 regime stress 一致（市场波动率在 2025 tariff
shock 期间升约 2×），不与 label leakage 一致。

**怎么实现**。`analyze_fold4_leakage.py` 执行 4-test 框架。输出
`experiments/step3_plan_z/fold4_zdrift_summary.csv`（158-feature drift 分数）、
`fold4_zdrift_per_day.csv`（62 日 × drift + IC 时间序列）、
`fold4_tail_concentration.csv`（158-feature tail 统计）。本图所用日度 rolling
指标取自 `fold4_zdrift_per_day.csv`。

**图**。
![Finding 8](../plots/advisor/fig_08_fold4_regime.png)

**图说**。Panel (a) 是 per-day scatter：每点为一个测试日。拟合虚线斜率为正，
对应文本标出的 ρ 值。若为 leakage 解释，IC 要么均匀偏高，要么与 drift 不
相关；实际上 "高 IC 日恰是高 drift 日"。Panel (b) 是同一组信号的时间序列
——day index 1055 附近的 z-drift spike 几天后对应 MLP rolling IC 峰值。

**结论**。Fold 4 的行为**不是**训练 artefact；而是模型在分布漂移下恰好
给出正确排序。这支持 "诚实报告 Fold 4 作为 stress-test" 的做法，而非丢弃。

---

## Finding 9 — 9-Dim Price Feature 的 Effective Rank ≈ 3

**是什么**。9-dim price features 的 9×9 correlation matrix 有 3 个特征值承载
约 89.7% 方差（PC1 49.5%、PC2 28.4%、PC3 11.8%）。三对 `ret_mean_k` 与
`momentum_k`（k ∈ {5, 10, 21}）数值上完全等同（Pearson ρ = 1.00）。

**怎么实现**。在 1,212 个有效交易日的 cross-sectional correlation matrix 上做
标准 PCA（`diag_phase5_effective_rank.csv`，9 行），并给出完整 9×9 相关矩阵
（`diag_phase5_collinearity.csv`，9 行）。流程属于 Phase-5 diagnostic，由
`diagnostic_phase5_step0.py` 触发。

**图**。
![Finding 9](../plots/advisor/fig_09_effective_rank.png)

**图说**。Panel (a) scree plot：蓝柱是每个 PC 的方差占比，红线是累积，金色
竖带标出累积跨过 90% 的 PC3 位置。Panel (b) 完整 correlation matrix。3 个
接近 1.0 的 off-diagonal cells 就是 `ret_mean_k ≡ momentum_k` 的数值等同
——在我们的特征定义下它们本质相同，**文中明确披露**。

**结论**。9-dim 特征集**不是**9 个独立信号。这为 Finding 1 的 3-feature PC
probe 选择提供动机；**不是**对金融因子普适性的新主张。

---

## Finding 10 — 架构比较：5 种架构下 Price-Only 的 IC 点估计都 **数值上** 高于 All-Features（算术比较；**未**跑跨架构的联合统计检验）

**是什么**。架构比较实验（5 models × 2 feature sets × 5 folds × 3 seeds = 150
行）显示，**每一种 price-only 变体的 mean IC 都高于其 all-features 对应**：
SAGE-Sum（0.039 vs. 0.010）、MLP（0.037 vs. −0.008）、Transformer（0.027 vs.
−0.009）、SAGE-Mean（0.026 vs. 0.011）、GAT（0.022 vs. −0.002）。在 price-only
五个模型间，mean IC 差异较小（0.022 → 0.039），variability bars（±1 σ）
大量重叠。

**怎么实现**。数据文件 `arch_comparison_results.csv`（150 行）。产出脚本
不在 repo 根目录——看起来是在 stability experiments 期间由 notebook 运行产出。
Canonical CSV 见 Appendix B。

**图**。
![Finding 10](../plots/advisor/fig_10_arch_stability.png)

**图说**。每个点是一个 model × feature-set 组合；横向线段为 ±1 σ（15 runs
= 5 folds × 3 seeds）。蓝点 price-only，红点 all-features。红点整体偏左，
是本图最清晰的定性信号（与 Finding 11 的 news 损害效应一致）。早期内部文档
报告的 "SAGE-Sum CV ≈ 5%" 在此 CSV 上**无法复现**——当 mean IC 近 0 时
|std/mean| 会发散，因此本图**改用**原始 IC 单位的 std，而非 CV%。

**结论**。Price-only 5 种架构内**没有**统计显著的两两差异（各 15 runs 不够）；
但 price-only vs. all-features 的差异在每种架构上都一致。

---

## Finding 11 — FinBERT News Embeddings 跨 5 Folds 降低日度 IC

**是什么**。把 384-dim FinBERT title embedding 拼接到 price features 上后，
seed-averaged mean IC 在 MLP 上**5/5 folds** 全部下降、在 SAGE-Mean 上
**4/5 folds** 下降（Fold 3 是唯一例外：news-augmented SAGE 略高于 price-only
小幅）。整体 5-fold walk-forward：MLP price-only mean IC = +0.037，
MLP all features = −0.008。

**怎么实现**。与 Finding 3 同驱动 `run_walkforward_5fold.py`，输出
`experiments/wf5_results.csv`。本图过滤到 4 个相关模型变体
（MLP_price、MLP_all、SAGE-Mean_price、SAGE-Mean_all）并按 fold 透视。

**图**。
![Finding 11](../plots/advisor/fig_11_finbert_harm.png)

**图说**。实心柱为 price-only（蓝），斜线纹理区分 SAGE-Mean 与 MLP。红柱
为 +FinBERT。MLP 上**每个 fold 的红柱都低于对应蓝柱**（5/5）；SAGE-Mean 在
Folds 0、1、2、4 成立（4/5），Fold 3 是唯一 news-augmented SAGE 略高于
price-only 的 fold（差距小）。

**结论**。在本 universe（SP500 大盘）+ 本新闻格式（title 级，平均 ~15 词
/ 事件）下，FinBERT embedding **不**增加排名信号，典型情况下减少。结果与
"大盘新闻在日收盘前已被定价" 的有效市场 prior 一致。

---

## Finding 12 — Binary Direction Prediction 在本 Universe 上不可学习

**是什么**。在 Phase 1d **news-event 触发、次日方向二分类**任务（SP500
universe，事件→stock-day 压缩后 ≈437K events，市场调整后标签）上，B1-B5
baseline matrix 每模型的 test-set AUC 点估计（来源：权威 run log
`progress.md` §2026-03-03-g）为：
B1 LR + FinBERT = 0.4993；B2 LR + Sentiment = 0.5031；
B3 LR + Sent + Momentum = 0.4965；B4 LR + Momentum = 0.4987；
B5 XGBoost + all = 0.5046。
即真实区间为 **[0.4965, 0.5046]**，最小值在 B3（动量 LR 过拟合，**不是** B1），
最大值在 B5。**没有任何一个**模型超过 pre-registered 0.52 "Go" 阈值。用
Qwen / GPT-4o LLM embedding 替代 FinBERT 仅带来 ΔAUC = +0.0009（与噪声
不可区分）。

**怎么实现**。Binary direction 实验位于 archived notebooks，见
`archived/docs/2026-03-27/notebook_phase1_2_B.md`；上面 5 个 per-model 数字
来自 **`progress.md` §2026-03-03-g** 权威 run log。**原始 per-seed CSV 并未
归档于 `experiments/`**，因此本图**只绘制 5 个真实的 per-model AUC 点
估计**，不捏造 per-seed 点。

**图**。
![Finding 12](../plots/advisor/fig_12_binary_failure.png)

**图说**。5 个圆点为 progress.md §2026-03-03-g 的 per-model test AUC：绿色
（最大）是 B5 XGBoost = 0.5046，红色（最小）是 B3 LR + Sent + Momentum =
0.4965（run log 标注其 Val → Test 跌落为"overfitting"）。实心黑线为 random
（0.50），虚红线为 pre-registered Go 阈值（0.52）。每个 baseline 都落在
random 的 ±0.005 以内（最大偏离：B5 为 +0.0046）；最佳 baseline（B5）距
Go 线仍差 0.0154。**有意**不
绘制 per-seed 点——progress.md 里也只保留了这 5 个 per-model 聚合数字。

**结论**。本结果是**news-event 触发 / 次日（短 horizon）二分类方向任务** 在
SP500 universe（事件→stock-day 压缩后 ≈437K events，市场调整后标签）+ 所测
特征上的负面结果——**既不是**对股票方向预测的一般性否定，**也不**适用于
21-day horizon（21-day 是 Finding 1-11 所用 v3 ranking 任务的 horizon，与
本 finding 的任务不同）。此结果是 v3 从 event-driven binary classification
转为 daily cross-sectional ranking 作为主任务的依据。

---

## Appendix A — 术语释义

- **IC (Information Coefficient)**：模型预测分数与当日实际次日收益的
  cross-sectional Spearman rank correlation，逐日计算后平均。
- **ICIR**：日均 IC 除以 IC 的日序列标准差。
- **Sharpe（gross / net）**：按预测分数构建 long-short portfolio 的年化
  收益/波动率。net 扣 15 bps round-trip 成本。
- **Newey-West (NW) t-stat**：对日度序列均值的 heteroskedasticity- and
  autocorrelation-robust t 值；本项目使用 lag = 5 天。
- **Hansen SPA**：Superior Predictive Ability 检验。原假设 "基准不劣于
  备选集合中最好的一个"。使用 studentized test statistic +
  stationary-bootstrap 抽样。p_consistent 是推荐的中间估计（介于 p_lower
  与 p_upper 之间）。
- **BH-FDR**：Benjamini-Hochberg false discovery rate correction for
  multiple testing。
- **Wilcoxon signed-rank test**：非参数 paired 检验，检验中位差是否为 0。
- **Bootstrap CI**：stationary bootstrap，block length ≈ n^{1/3}。
- **Walk-forward CV（purged / embargoed）**：train 在日期前缀，validate 在
  下一时段，test 在其后；滚动前移不 reshuffle。Purged = 丢弃 overlap 日，
  Embargoed = 两切分之间插入 buffer 日。
- **GAT / GraphSAGE (Mean, Sum) / HGT / Transformer / MLP**：节点级排名
  模型。GAT = Graph Attention Network；SAGE-Mean/Sum = GraphSAGE 的 mean/sum
  aggregator；HGT = Heterogeneous Graph Transformer；Transformer = 邻居集
  上的 permutation-invariant transformer。
- **NoGraph**：同 backbone 但不做 message passing 的消融，用于分离图特性。
- **Alpha158**：qlib 的 158 个 engineered price/volume 特征库
  （K-bar + rolling statistics）。
- **mom12m / momentum_k / ret_mean_k / ret_std_k**：特征组。mom12m = 12-month
  momentum（252-day lookback）；momentum_k = k 日动量；ret_mean_k = k 日
  rolling 收益均值；ret_std_k = k 日 rolling 波动率。
- **CORR5 / dolvol / maxret / RSV5**：扩展特征。CORR5 = 与大盘的 5 日
  rolling 相关；dolvol = dollar volume 代理；maxret = 窗口内最大日收益；
  RSV5 = 5 日 realized semi-variance。
- **Winsorization（p1/p99）**：将极值截断到第 1 / 99 百分位。
- **Cross-sectional z-score**：逐日（跨股票）z-score normalization，仅用
  train 统计量拟合。
- **Permutation importance (ΔIC)**：baseline IC 减去对特征（或组）做 cross-
  sectional shuffle 后的 IC。越高越重要。
- **Effective rank**：捕捉约 90% 方差所需的 PC 个数（scree method）或
  participation ratio 加权值。
- **PCA**：特征相关矩阵的 principal component analysis。
- **SelectiveNet**：ICML-2019 三头架构（prediction、selection、auxiliary），
  学习接受多大比例的输入。
- **Coverage**：selective 模型选择预测的样本比例。
- **Threshold baseline**：按 |score| 排序保留 top-k 比例的简单对照。
- **FinBERT**：在金融文本上 fine-tune 的 BERT 变体，本项目使用 title-level
  embedding。
- **TF-IDF**：term-frequency × inverse-document-frequency，经典文本向量化。
- **Lazy Prices**：Cohen-Malloy-Nguyen 2020 构造，度量 10-K/10-Q 季度间的
  文本变化。
- **SEC 10-K / 10-Q**：向美国 SEC 提交的年报 / 季报。
- **Tariff shock（Q2-2025）**：2025 年 4-6 月关税公告触发的大盘波动 regime；
  SP500 日均波动 ≈ 2×。
- **Regime stress / regime shift**：train 与 test 窗口的特征或收益联合分布
  变化。
- **Data leakage**：test 窗口信息影响训练。
- **Long-short portfolio**：做多预测 top-quantile、做空 bottom-quantile，
  日度 rebalance。
- **Turnover**：每次 rebalance 换仓比例。
- **HHI**：Herfindahl-Hirschman Index，组合集中度。
- **OOD**：out-of-distribution。
- **CV%**：coefficient of variation，|std/mean| × 100。均值近 0 时失真
  （Finding 10 改用原始 std IC）。

---

## Appendix B — 各 Finding 的实现代码清单

每个 finding 列出 (1) 产生数据的训练/分析脚本、(2) 图所消费的 canonical
CSV/JSON 文件。所有路径相对 repo 根 `/Users/heruixi/Desktop/GNN-Testing/`。

| # | Finding | 训练/分析脚本 | 数据文件 |
|---|---|---|---|
| 1 | S6 non-superiority vs S8 (parsimony) | [build_alpha158_features.py](../build_alpha158_features.py), [run_step3_plan_z_part_b.py](../run_step3_plan_z_part_b.py), [run_step3_plan_z_part_c.py](../run_step3_plan_z_part_c.py), [run_step3_plan_z_part_c_perfold.py](../run_step3_plan_z_part_c_perfold.py), [analyze_step3_plan_z.py](../analyze_step3_plan_z.py) | `experiments/step3_plan_z/part_b_summary.csv`, `part_c_s8_daily_ic.csv`, `part_c_s8_perfold_daily_ic.csv`, `hansen_spa_results.csv` |
| 2 | Horizon ablation | [run_phase5_step3_feature_expansion.py](../run_phase5_step3_feature_expansion.py), [diagnostic_phase5_step0.py](../diagnostic_phase5_step0.py) | `experiments/horizon_ablation_results.csv`（360 行） |
| 3 | Walk-forward 5-fold | [run_walkforward_5fold.py](../run_walkforward_5fold.py) | `experiments/wf5_results.csv`（90 行） |
| 4 | Permutation ranking | [run_step3_plan_z_part_a.py](../run_step3_plan_z_part_a.py), [analyze_step3_plan_z.py](../analyze_step3_plan_z.py) | `experiments/step3_plan_z/part_a_daily_ic.csv`, `part_a_permuted_ic.csv`, `artifacts/step3_plan_z/part_a_ranking.json` |
| 5 | Normalization × regime | [run_diag1_normalization.py](../run_diag1_normalization.py), [run_diag1b_replication.py](../run_diag1b_replication.py) | `experiments/diag1_normalization_results.csv`, `diag1b_replication_results.csv` |
| 6 | SelectiveNet coverage | v3 N5 notebook（见 `archived/notebooks/v3_ranking_pipeline.ipynb`） | `experiments/selectivenet_results.csv`（70 行） |
| 7 | SEC Gate 1 text | [run_gate1_experiment.py](../run_gate1_experiment.py) | `experiments/gate1_results.csv`（22 行） |
| 8 | Fold-4 regime stress | [analyze_fold4_leakage.py](../analyze_fold4_leakage.py) | `experiments/step3_plan_z/fold4_zdrift_summary.csv`, `fold4_zdrift_per_day.csv`, `fold4_tail_concentration.csv` |
| 9 | 9-dim effective rank | [diagnostic_phase5_step0.py](../diagnostic_phase5_step0.py), [diagnostic_phase5_fix.py](../diagnostic_phase5_fix.py) | `experiments/diag_phase5_effective_rank.csv`, `diag_phase5_collinearity.csv` |
| 10 | Architecture 比较 | 由 archived v3 stability notebook 产出；repo 根目录无对应 `.py`；见 `archived/notebooks/` | `experiments/arch_comparison_results.csv`（150 行） |
| 11 | FinBERT news 有害 | [run_walkforward_5fold.py](../run_walkforward_5fold.py) | `experiments/wf5_results.csv`（过滤至 `_price` vs. `_all`） |
| 12 | Binary direction 失败 | archived Phase-1d notebooks（见 `archived/docs/2026-03-27/notebook_phase1_2_B.md`） | `progress.md` §2026-03-03-g（权威 run log；5 个 per-model 点估计） |

### 图表生成驱动

全部 12 张图由 **[make_advisor_figures.py](../make_advisor_figures.py)**
（单文件约 450 行）生成，读取上表所列 CSV 并输出到
`plots/advisor/fig_NN_*.png`。调用方式：
`/opt/homebrew/Caskroom/miniforge/base/envs/gnn/bin/python make_advisor_figures.py`

### 特征 / 数据构建

- `build_alpha158_features.py` — 复现 qlib Alpha158DL 特征库。
- `build_phase5_features.py` — 构建 14-dim 扩展特征集。
- `cleanup_and_rebuild_features.py` — 统一特征重建工具。
- `download_ohlcv_yf.py` — yfinance OHLCV 抓取。
- `refetch_zts.py` — ZTS ticker 数据补抓。

### Paper / report aggregator（另立，现存）

- `run_figures_tables.py` — 原 paper pipeline 的图表生成器，保留以复现已有
  `plots/paper_*.png`；本次 advisor 包未修改此脚本。
