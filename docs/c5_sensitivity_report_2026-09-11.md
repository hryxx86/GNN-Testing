# C5 特征子集敏感性分析 — 完整报告

**日期**：2026-09-11　**任务来源**：H博士 简报 `docs/c5_rerun_brief_2026-09-10.md`　**状态**：全部完成，Colab 可断开
**Git**：本 session 5 个 commit（`46b3b8c` → `9008dbe` → `14c284a` → `a903c5e` → `9d9ed8e`），未 push
**评审链**：Codex TP1 A/B、Codex TP2 A、finance-gnn-reviewer TP2 B + TP3（Codex 用量上限 fallback）、4 代理 closeout 审计 PASS

---

## 0. 一页摘要

**任务**：论文 Limitation L1 承诺 "A re-run of L0 and L1 on the five surviving factor groups is the definitive check, and we have not run it"。本任务把它跑出来：在宇宙 C 中只保留 "T−1 re-rank 存活" 的 5 个因子组（20 列，记 C5），按 confirmatory 冻结协议重新调参并评估 L0（LightGBM）与 L1（MLP），报告 C5 上的 L1−L0，与 C（+0.0148）和 B（+0.0143）并列。

**最重要的发现（先于数字）**：**C5 不是 leak-free re-selection。** 选出这 5 组的两个排名都用了测试期标签打分，且 proxy 排名有无 T−1 shift 结果完全相同——"5/15 幸存 T−1" 是两种重要性度量的交集，不是去泄漏的结果。论文 L1 与附录的相关两句是误表述，需改口径；本次运行的正确定性是 **post-hoc、test-informed feature-subset sensitivity**，不能兑现 "definitive check"。

**数字（T4 主运行；source: `artifacts/storya_v21_family1_c5/c5_comparison.csv`、`c5_ex_fold.csv`、`c5_paired_contrast.csv`）**：

| 宇宙 | ΔIC L1−L0 | 21d block-boot 95% CI | HLN p（NW auto lag） | HLN p（lag 21） | MDE（≈2.8×SE） | 效应 vs MDE | per-seed 同号 |
|---|---|---|---|---|---|---|---|
| **C5**（post-hoc，test-informed） | **+0.0134** | [+0.0008, +0.0283]（边界排零） | 0.008 | **0.054** | 0.0197 | 低于 | 10/10 |
| C（confirmatory） | +0.0148 | [−0.0004, +0.0304] | 0.011 | 0.063 | 0.0220 | 低于 | 10/10 |
| B（confirmatory） | +0.0143 | [−0.0051, +0.0341] | 0.052 | 0.181 | 0.0275 | 低于 | 10/10 |

- 配对差 (L1−L0)_C − (L1−L0)_C5 = **+0.0013**，95% CI [−0.016, +0.019]；配对检验 MDE ≈ 0.025 > C 效应 0.0148 → 点估计基本未变，但**既不排除减半也不排除加倍，等价不成立**。
- **fold 9（2025Q2）贡献 pooled ΔIC 的 53%（C 44%，B 31%）**；剔除后 C5 ΔIC = +0.0069 [−0.0032, +0.0179]，p = 0.13（lag 21：0.23）。
- **lag-21 下 C5、C、B 都未达 nominal 0.05**；三者效应都低于设计 MDE，属边际、功效不足的检出。
- C5 调参两臂全部 5 个决赛配置在 2022H2 的 val-IC 为负（C 为 +0.07 / +0.06）；选出的 MLP 仅 2,337 参数（C 的 31,745）。
- T4 主运行与 Mac 复现一致（pooled ΔIC 0.0134 vs 0.0132；L0 逐位相同）。

**一句话结论（评审许可口径）**：在对重要性度量稳健的 20 列子集上，MLP 对 LightGBM 的点估计与 C 基本相同，但证据强度与 C/B 一样是边际的、集中于 2025Q2；因子子集本身是 test-informed 选出的，**本结果不解决、不界定、也不估计 C 的特征选择泄漏**；B 仍是 leak-free 的特征基锚点。

**建议**：① 论文两句改口径（9/25 前，插 C5 段之前）；② C-pre（pre-test 选择器）只在想对泄漏做量化陈述时才跑；③ C5h、L2 层不必跑；④ 确认后 push。

---

## 1. 任务与背景

审稿意见 2 指出：宇宙 C 的正结果（MLP − LightGBM = +0.0148，HLN p = 0.011，BH 拒绝）因特征基选择泄漏只算 "suggestive"，但论文没有给出泄漏把对比吹大了多少。论文现用无泄漏的宇宙 B 做锚点（+0.0143，p = 0.052），并在 L1 里承诺对 "five surviving factor groups" 做 L0/L1 的 re-run 作为 definitive check。

简报要求：C5 = `artifacts/plan_aaa_t1_diagnostic/group_ranking_comparison.csv` 中 `proxy_rank_t1 <= 15` 的 5 组（ROC30、KMID、KUP、CNTP20、CORR60）→ 按 `artifacts/plan_aaa/ranking.csv` 的 `group_members` 映射为 20 列；协议与 confirmatory 完全一致；两臂各自重调 30 trials；240 cell；报告 seed-averaged daily ΔIC、HLN p、MDE、per-seed k/10、LOSO m/10；不做 BH；不动任何 confirmatory 表。

## 2. 定性问题：为什么 C5 不是 leak-free（Codex TP1 CRITICAL/MAJOR，Claude 亲自核实）

**2.1 选择器在测试期内打分**

| 排名 | 打分窗口 | 位于 12 折测试期（2023Q1–2025Q4）内？ | 证据 |
|---|---|---|---|
| T−1 proxy（单特征 \|IC\|） | 面板最后 313 个有效标签日 = **2024-09-27 → 2025-12-26** | 是 | `analyze_plan_aaa_t1_diagnostic.py:86-99`（"use the LAST 313 days of valid labels"；脚本注释写的 "Q2-2024→Q2-2025" 与实际不符，`summary.md` caveat 3 已承认日历漂移）；本机用价格面板重建 |
| Plan AAA 原排名（permutation ΔIC，SAGE-Mean/MLP） | 5 折测试季 = **2024-04-01 → 2025-06-30**（313 天 = 12 折的 fold 5–9） | 是 | `data/reference/fold_manifest_expanding.json`；`artifacts/plan_aaa/baseline_ic_per_cell.csv` 列 `arch` ∈ {SAGE-Mean, MLP} |

结论：C 与 C5 的列选择都使用了 confirmatory 测试期的标签（Cawley–Talbot 类的 selection-on-test）。T−1 shift 后再评估无法移除已经进入"选哪些列"的信息。此外 Plan AAA 用的是 **NN 模型的 permutation 重要性**，所以 C/C5 内涉及 NN 臂的对比（含 L1−L0 的方向）并非 selection-neutral（TP3 R-A-05）。

**2.2 "5/15 幸存 T−1" 是度量方法之差**

`group_ranking_comparison.csv` 中，`proxy_rank_raw <= 15` 与 `proxy_rank_t1 <= 15` 的 15 组**集合完全相同**；orig ∩ raw = orig ∩ t1 = 同一 5 组。即 T−1 shift 一组也没移除；5/15 = Plan AAA permutation top-15 ∩ 单特征 |IC| proxy top-15。`summary.md` 本身也写着 "proxy-raw ∩ proxy-T1 = 15/15"。

**2.3 对论文的影响**

`paper/iclr2027/main.tex:290`（L1）、`:998`、`:1012`（图 caption）的 "only 5 of the 15 groups survive strict T−1 re-ranking" 与 L1 的 "A re-run … is the definitive check" 两句需要改口径（§7 给出改法）。本次运行按 Codex option 1 照跑并重新定性；option 2（pre-test 选择器 "C-pre"）作为提案见 §8。

## 3. 设计与执行

**3.1 C5 定义（20 列，与 `ranking.csv` 逐组核对一致）**

| Plan AAA 组 | Plan AAA rank | proxy_rank_t1 | 成员列 |
|---|---|---|---|
| ROC30+5 | 2 | 8 | ROC30, MA60, MAX60, MIN60, QTLU60, QTLD60 |
| KMID+6 | 4 | 13 | KMID, KMID2, KSFT, KSFT2, OPEN0, HIGH0, VWAP0 |
| KUP+1 | 10 | 6 | KUP, KUP2 |
| CNTP20+3 | 12 | 15 | CNTP20, CNTD20, CNTP30, CNTD30 |
| CORR60 | 15 | 9 | CORR60 |

实现：`build_universe_C5` 直接调用 `build_universe_C` 后按名选列（逐列 `array_equal` 断言 == C；T−1 shift 与 row-0 零由 C 的构造保证），不含 3 个 hc 列；下游按折 train-only winsorize/standardize 与 B/C 完全相同。

**3.2 协议（与 confirmatory 一致，一项未改）**：12 折 expanding walk-forward（2023Q1–2025Q4，T = 749 天）、21 天 purge、21 日前向收益横截面去均值 + z-score 标签、10 canonical seeds、相关图冻结快照 ≤ train_end。调参：`run_storya_v21_tune.py --n-trials 30 --top-k 5`，tune seeds [11, 22, 33]，窗口 train ≤ 2022-06-30 / val 2022H2（purge 后标签终点 ≤ 2022-12-30，严格早于测试期）。

**3.3 调参结果（选模指标，不是结果；source: `artifacts/storya_v21_family1_c5/c5_tuned_hparams.csv`、`experiments/storya_v21_tune/C5_{L0,L1}.json`）**

| 臂 | 冠军超参 | val-IC（3-seed 均值） | 决赛 5 配置 | 对比 C 冠军 |
|---|---|---|---|---|
| L0 LightGBM | num_leaves 63, lr 0.0129, min_data_in_leaf 100, λ1 4.7e-4, λ2 0.016 | **−0.012** | 全部负（−0.0121…−0.0122） | +0.0735 |
| L1 MLP | lr 0.0092, wd 6.6e-5, dropout 0.3, hidden 32, 1 层（2,337 参数） | **−0.045**（per-tune-seed −0.006 / −0.045 / −0.084） | 全部负（−0.0448…−0.0453），5 个都是 hidden 32 / 1 层 / dropout 0.3 | +0.0597（31,745 参数） |

含义：C5 上的等预算调参在**程序上合规、实质上无信息**——冠军是在近零/负信号配置里机械选出的。这本身与 "子集信号集中于其（测试期）选择窗口" 的解释相容。不得把 C5 的对比（或其与 C 的相似）归因于特征限制或容量。frozen 文件 `frozen_hparams_c5.json` md5 `cdb4d92314b0d43d3287ea6d403d840d`。

**3.4 运行**：主运行 = Colab T4（H博士 指示后于结果出来前预先声明；L1 29.4 s/cell，全程 1.04 h），复现 = Mac M4 MPS/CPU（L1 51.2 s/cell）。两者同 frozen、同代码；T4 代码身份事后在 Colab VM 上 md5sum 全部 7 个导入模块 == commit `9008dbe`（`_code_identity_t4.json`）。

**3.5 完整性（source: `artifacts/storya_v21_family1_c5/c5_run_integrity.json`）**：240/240 cell、cell_id 2400–2639（与 confirmatory 的 0–2399 不交）、0 failed、全部 converged、240 个 per-day IC 数组每折长度 == 冻结日历 [62,62,63,63,61,63,64,64,60,62,64,61]（和 749）、20 列名 == `UNIVERSE_C5_NAMES`、provenance gate（mode TUNED per-arm、md5 匹配、applied == frozen 冠军）通过、统计目录与 cell 目录 results.csv md5 一致。

## 4. 结果（T4 主运行）

**4.1 主表（source: `c5_comparison.csv`；C/B 行来自 `artifacts/storya_v21_family1/family1_{dm_hln,ic_ci,mde}.csv`）**

| 宇宙 | ΔIC（seed-averaged daily） | 95% CI（对 10-seed 平均） | p auto lag | p lag 21 | SE_block | MDE | IC L0 [CI] | IC L1 [CI] | k/10 | LOSO |
|---|---|---|---|---|---|---|---|---|---|---|
| C5 | +0.01343 | [+0.00075, +0.02833] | 0.0080 | 0.0537 | 0.00704 | 0.0197 | 0.0203 [−0.0037, 0.0472] | 0.0337 [0.0021, 0.0704] | 10/10 | 0/10 |
| C | +0.01477 | [−0.00036, +0.03041] | 0.0109 | 0.0632 | 0.00786 | 0.0220 | 0.0195 [−0.0070, 0.0481] | 0.0343 [0.0044, 0.0669] | 10/10 | 0/10 |
| B | +0.01428 | [−0.00507, +0.03414] | 0.0524 | 0.1812 | 0.00982 | 0.0275 | 0.0228 [0.0024, 0.0429] | 0.0371 [0.0118, 0.0630] | 10/10 | 0/10 |

读法（统计审计 EXPL-STAT-01/02/03/05/06/10）：
- NW auto lag（T=749 时 ≈6）是实现默认值，**不是协议 §6 冻结项**；21 日重叠标签下 ΔIC 在 lag 6 的自相关仍 ≈0.32，故必须并报 lag-21：**lag-21 下三者都未达 0.05**。
- lag-21 HAC SE（0.0068）与 block-bootstrap SE（0.0070）一致，但 C5 的 5% 判定边际相反：百分位 CI 排零而 lag-21 p = 0.054；1.96×SE = 0.0138 > 0.0134，CI 排零是边界情形。
- 分开说：C5 只在 auto-lag / 百分位 CI 读法下拒绝；C 在 auto-lag 下 BH 显著但 CI 含 0；B 是 non-detection。三者点估计都低于 ≈2.8×SE 的 80% 功效阈值。
- C5 的 p 最小但点估计最小——由方差驱动（SE 0.0070 vs 0.0079 / 0.0098），不是更大的效应。
- k/10 是同一数据上的 seed/初始化稳定性（非独立复制）；k = 10 时 LOSO m = 0 是必然。
- per-seed ΔIC 范围 +0.0029…+0.0250；0 塌缩 cell。C5 的 per-arm IC 水平条件于 test-informed 选择，不作为样本外表现引用。
- 本运行共发布 16 个 nominal p（`c5_tests_reported.json`），未做多重校正。

**4.2 季度集中（source: `c5_ex_fold.csv`）**

| 宇宙 | fold 9（2025Q2）ΔIC | 占 pooled ΔIC 份额 | 剔除 fold 9 后 ΔIC | 95% CI | p auto | p lag 21 |
|---|---|---|---|---|---|---|
| C5 | +0.0862 | 53% | +0.0069 | [−0.0032, +0.0179] | 0.132 | 0.231 |
| C | +0.0785 | 44% | +0.0090 | [−0.0052, +0.0228] | 0.125 | 0.248 |
| B | +0.0529 | 31%（B 最大折是 fold 7，+0.0574） | +0.0108 | [−0.0089, +0.0300] | 0.143 | 0.312 |

C5 继承了 C/B 的季度集中，不是均匀持续。12 折 LOFO 无符号翻转。

**4.3 配对日度对比（source: `c5_paired_contrast.csv`；正 = confirmatory 宇宙的 L1−L0 大于 C5）**

| 对比 | 均值 | 95% CI | p auto | p lag 21 | SE | 配对 MDE |
|---|---|---|---|---|---|---|
| (L1−L0)_C − (L1−L0)_C5 | +0.0013 | [−0.0159, +0.0189] | 0.838 | 0.882 | 0.0088 | 0.0245 |
| (L1−L0)_B − (L1−L0)_C5 | +0.0008 | [−0.0186, +0.0200] | 0.915 | 0.938 | 0.0098 | 0.0276 |

两条配对 MDE 都大于 C 的效应本身（0.0148）：点估计基本未变，但区间不能区分"未变"与"减半/加倍"，**等价不成立**（underpowered non-rejection）。它是 51→20 列限制 + 重调后的变化量，不是泄漏膨胀的识别量。C（Mac）vs C5（T4）的设备混杂以 4.4 的复现为界。

**4.4 设备复现（source: `c5_device_replication.md`；Mac 统计 `artifacts/storya_v21_family1_c5_mac/`）**

| 臂 | cell 数 | cell-IC 相关 | 平均 \|Δ\| | 最大 \|Δ\| | 逐位相同 |
|---|---|---|---|---|---|
| L0 | 120 | 1.000 | 0 | 0 | 120/120 |
| L1 | 120 | 0.951 | 0.018 | 0.103 | 0 |

pooled ΔIC：T4 +0.0134 vs Mac +0.0132；Mac 复现 CI [+0.0002, +0.0279]，p 0.013 / lag-21 0.067，配对 C−C5 +0.0016，ex-fold-9 +0.0064（p 0.19）。L1 的差异来自 early-stop 在 val loss ≈ 0.998 的平坦曲线上的后端非确定性，对 pooled 推断无影响。

## 5. 解读边界

**采用的表述**（TP3 许可清单）：
- "On the 20-column subset selected by the intersection of two test-informed importance rankings, the MLP−LightGBM point estimate is +0.0134 (95% 21d-block CI [+0.0008, +0.0283], a boundary exclusion; nominal HLN p 0.008 at the implementation-default NW auto lag, 0.054 at HAC lag = horizon, where C is 0.063 and B 0.181; 10/10 seeds same sign), versus +0.0148 in C and +0.0143 in B."
- "The paired change C − C5 is +0.0013 [−0.016, +0.019]: the point estimate is essentially unchanged, but the interval does not distinguish an unchanged contrast from a halved or doubled one; equivalence is not established."
- "Observed ΔIC is below the design's approximate MDE (≈0.020) in C5, C and B; these are marginal, underpowered detections."
- "About half of the pooled contrast in C5 (53%; C 44%, B 31%) comes from 2025Q2; excluding it, C5 ΔIC = +0.007 [−0.003, +0.018]."
- "Both arms' tuning-window validation IC on C5 was negative for every finalist (L0 −0.012, L1 −0.045); the frozen HPs are protocol-consistent but not a validated optimum, and the C5 MLP has 2,337 parameters vs 31,745 in C."
- "C5's selection used evaluation-period labels and NN-based permutation importance; this sensitivity does not resolve, bound, or estimate feature-selection leakage in C. B remains the leak-free feature-basis anchor."

**不采用的表述**："leak-free re-selection"、"survives T−1 re-ranking"、"definitive check"、"confirms / validates C"、"the advantage persists"（不带 fold-9 与 MDE 限定）、"robust to leakage"、"did not materially change / unchanged / equivalent"、单独引用 p = 0.008、任何容量归因、把 C5 per-arm IC 当样本外表现、"the L1−L0 contrast is unaffected by selection"。

## 6. 对论文的具体修改建议

**6.1 必改的两句（main.tex:290 L1；:998、:1012 附录）**

- 原："Universe C's 51 columns come from Plan-AAA top-15 Alpha158 groups ranked under same-day OHLC; only 5 of the 15 groups survive strict T−1 re-ranking"
  建议："Universe C's 51 columns come from Plan-AAA top-15 Alpha158 groups ranked by NN permutation importance on quarters that lie inside the evaluation window and under same-day OHLC; only 5 of the 15 groups are also top-15 under a single-feature-IC proxy, and that proxy ranking is identical with and without the T−1 shift, so the disagreement reflects the importance measure rather than the lag."
- 原："A re-run of L0 and L1 on the five surviving factor groups is the definitive check, and we have not run it."
  建议撤回，改为："A re-run of L0 and L1 on that 20-column subset (Appendix X) leaves the point estimate essentially unchanged but is itself conditional on the test-informed selection; a pre-evaluation re-selection would be required to bound the leakage, and we have not run one."

**6.2 附录段落草稿（英文，可直接改用）**

> *Feature-subset sensitivity (post-hoc, test-informed).* We re-tuned and re-evaluated L0 (LightGBM) and L1 (MLP) on the 20 Universe-C columns belonging to the five Plan-AAA groups that are also top-15 under a single-feature-IC proxy (ROC30, KMID, KUP, CNTP20 and CORR60 groups), under the frozen protocol (12-fold expanding walk-forward, 21-day purge, 10 seeds, 30-trial equal-budget tuning per arm). This is a sensitivity analysis, not a confirmatory test: both rankings that define the subset were scored on quarters inside the evaluation window, so the subset cannot resolve or bound the selection leakage discussed in Limitation L1, and no multiple-testing family is opened (nominal, unadjusted p-values). On the subset the MLP−LightGBM contrast is +0.0134 (95% 21-day block-bootstrap CI [+0.0008, +0.0283]; nominal HLN p = 0.008 at the NW automatic lag and 0.054 at lag = horizon; 10/10 seeds same sign), versus +0.0148 in Universe C and +0.0143 in Universe B. The paired daily change C − subset is +0.0013 [−0.016, +0.019], whose interval does not distinguish an unchanged contrast from a halved or doubled one. As in C and B, the contrast is below the design's approximate MDE (≈0.020) and concentrated in 2025Q2 (53% of the pooled value; excluding that quarter, +0.007 [−0.003, +0.018]). Both arms' tuning-window validation IC on the subset was negative for every finalist (L0 −0.012, L1 −0.045), so the tuned configurations are protocol-consistent but not validated optima, and the subset MLP is ≈14× smaller than the Universe-C MLP. A replicate on a different backend (Mac MPS vs CUDA) gives +0.0132 [+0.0002, +0.0279]. Universe B remains the leak-free feature-basis anchor.

**6.3 附录表建议**：直接引用 §4.1 主表 + §4.2 ex-fold 表 + §4.3 配对表（数字 source 见 §10）。

## 7. 后续选项与建议

| 选项 | 内容 | 成本 | 建议 |
|---|---|---|---|
| 论文改口径 | §6.1 两句 + §6.2 段落 | 文字 | **必做，9/25 前，插 C5 段之前** |
| C-pre（pre-test 选择器） | 对 168 候选（158 Alpha158 T−1 + 10 hc）在 2021-07-01→2022-06-30（purge，标签终点 ≤ 2022-06-30）算单特征 rank-IC → 按 Plan AAA 61 组（组定义校准窗 2021-01-29→2022-01-27，Codex 核实在 pre-test）取 top-15 → 重调 L0/L1 → 240 cell + 配对对比。启动前需冻结：`hc_mom12m` 252 日 warm-up 的覆盖规则（231 天窗口仅 85 天非常数，Codex B-01）、组分数定义（成员时间均值 \|IC\| 的均值）、组定义复用 vs 重聚类；单独 TP1 | ≈3 h 算力 + 评审 | **只在论文想对泄漏做量化陈述时才跑**；若保持 "suggestive；B 为锚；泄漏未量化"，不需要 |
| C5h（+3 hc 列） | 23 列变体 | ≈3 h | **不必要**（TP3：对论断无影响） |
| L2 层（C5 上 correlation-GAT） | 重调 ~5 h + 120 cell ~5 h（T4） | ≈10 h | **不建议**（与论文当前主张关系弱） |
| push | 5 个本地 commit | — | 待 H博士 确认 |

## 8. 评审链

| 触点 | 评审者 | 结果 | 处置 |
|---|---|---|---|
| TP1 Round A（plan） | Codex gpt-6-astra | BLOCK-EXECUTION：1 C（选择器在测试期）+ 3 M + 2 Cn | 两项核心指控亲自核实成立 → 按 option 1 重新定性；配对对比、双 lag、参数量、provenance 全部落地；C-pre 写成提案 |
| TP1 Round B | Codex | PROCEED-WITH-FIXES（1 Cn：C-pre 覆盖规则） | 残留 3 项（措辞/配对/provenance）在 TP2 修复中关闭 |
| TP2 Round A（code） | Codex | PROCEED-WITH-FIXES：4 M + 1 Cn | 5/5 修复 + fixture 验证（默认 merge 字节一致；strict integrity 门；provenance 门；措辞；ledger） |
| TP2 Round B | finance-gnn-reviewer（Codex 用量上限 fallback） | PROCEED-WITH-FIXES：2 M + 2 Cn；A 轮 5 项全 FIXED | T4 provenance 空真修复 + 代码身份事后核实；inputs 块；配对 SE/MDE；负 val-IC 披露 |
| TP3 Round A（results） | finance-gnn-reviewer（fallback） | PROCEED-WITH-FIXES：4 M + 3 Cn；计算可信度 PASS | fold-9 集中、双 lag + MDE、配对非等价、调参披露全部落地；NN 选择器、设备差异、论文措辞 ACCEPTED |
| Closeout（4 Explore 代理） | leakage / statistics / correctness / doc-drift | 0 / 2 / 4 / 2(C)+2(M) → 全部当场修复 | PASS |

评审文件：`artifacts/reviews/2026-09-10_codex_{plan_A,plan_B,code_A}.md`、`2026-09-11_finance-gnn-reviewer_{code_B,results_A}.md`、`2026-09-11_explore-{leakage,statistics,correctness,doc-drift}_closeout.md`。

## 9. 与简报的偏离（均已记录、评审通过）

1. 不把 C5 加进 `ALL_UNIVERSES`/`both`（显式 `--universe C5`），confirmatory 默认调用零改动。
2. `compute_family1_ladder.py` 用 CLI 覆盖（`--universes/--arms/--sensitivity`）而非改冻结常量；默认路径复现 confirmatory C 行（p = 0.010852，`DataFrame.equals` True）。
3. 主运行设备：先按设备一致性在 Mac 起跑，H博士 指示后改 T4 为主（结果出来前预先声明），Mac 降为复现。
4. 简报中 "5 组 = proxy_rank_t1 <= 15" 的描述不准确（该条件返回 15 组）；实际定义为 Plan AAA top-15 ∩ proxy top-15，已按此描述。
5. 未跑 L2 层、C5h。

## 10. 产物与复现

**结果目录**
- 主：`experiments/storya_v21_main12_c5_t4/`（results.csv、manifest.csv、per_day_ic/ 240 个 .npy、`_universe_c5.json`、`_frozen_hp_provenance.json`、`_run_provenance.json`（含 correction 条目）、`_code_identity_t4.json`）
- 复现：`experiments/storya_v21_main12_c5/`（同一套）
- 统计：`artifacts/storya_v21_family1_c5/`（family1_{dm_hln,ic_ci,mde,lofo,stability}.csv、family1_ledger.json、family1_summary.md、c5_comparison.{csv,md}、c5_paired_contrast.csv、c5_ex_fold.csv、c5_seed_robustness.csv、c5_run_integrity.json、c5_tuned_hparams.csv、c5_device_replication.{csv,md}、c5_tests_reported.json）；复现统计 `artifacts/storya_v21_family1_c5_mac/`
- 调参：`artifacts/storya_v21_tune/{C5_L0,C5_L1,frozen_hparams_c5}.json`、`c5_tune_archive_md5.json`、`studies_c5/*.db`

**复现命令（本机 gnn 环境）**
```bash
python run_storya_v21_tune.py --arm L0 --universe C5 --n-trials 30 --top-k 5
python run_storya_v21_tune.py --arm L1 --universe C5 --n-trials 30 --top-k 5
python run_v21_tune_launcher.py --merge --merge-universes C5 --merge-arms L0,L1 --merge-out frozen_hparams_c5.json
python run_storya_v21_main12.py --universe C5 --arms L0,L1 --frozen-hparams experiments/storya_v21_tune/frozen_hparams_c5.json --out-dir experiments/storya_v21_main12_c5_t4
python compute_family1_ladder.py --main-dir experiments/storya_v21_main12_c5_t4 --output-dir artifacts/storya_v21_family1_c5 --universes C5 --arms L0,L1 --sensitivity
python analyze_c5_sensitivity.py --c5-main-dir experiments/storya_v21_main12_c5_t4 --c5-family-dir artifacts/storya_v21_family1_c5 --replicate-main-dir experiments/storya_v21_main12_c5 --ex-fold 9
```

**代码改动**：`run_storya_e1_anchor.py`（`SENSITIVITY_UNIVERSES`、`UNIVERSE_C5_GROUPS`、`build_universe_C5`）；`run_storya_v21_main12.py`（显式 `--universe C5`、cell_id 2400–3599、`_universe_c5.json` + `_run_provenance.json`）；`run_storya_v21_tune.py`（C5 分发、执行元数据）；`run_v21_tune_launcher.py`（子集 merge，默认输出字节一致）；`compute_family1_ladder.py`（`--universes/--arms/--sensitivity`，sensitivity ledger）；新 `analyze_c5_sensitivity.py`。另补交 6 月起从未提交、confirmatory 分析已依赖的 `compute_e6_dm_spa.py` / e3 / e4 改动（`46b3b8c`）。

**文档**：`docs/analysis.md` 2026-09-11-a；`progress.md` 2026-09-10-a … 2026-09-11-e；`plan.md` 2026-09-11-a + Decision Log；`docs/c5_rerun_brief_2026-09-10.md` §9（偏离账本 + C-pre 提案）；`docs/session_handoff_2026-09-11.md`。

## 11. 未在本任务范围内的遗留

工作树中 `analyze_e1_lofo.py`、`paper_figs/fig_family2.py`、`figures/family2_edge_causal.{pdf,png}` 及 archived/ 的移动为早前 session 遗留的未提交改动，本次未动。
