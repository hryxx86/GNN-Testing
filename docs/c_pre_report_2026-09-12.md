# C-pre 报告 — 用评估前信息重选特征基底后的 MLP vs LightGBM

**日期**：2026-09-12 **任务**：H博士 2026-09-12 "go"（方案 `docs/c_pre_plan_2026-09-11.md`）
**状态**：全部完成。Rule 9 三触发点 + 4-agent closeout 全通过。本机 commit 未 push（待您确认）。
**一句话**：在只用 2022-06-30 之前信息选出的 48 列基底上，MLP−LightGBM 的点估计是 **−0.0024**，95% 区间 **[−0.0256, +0.0178]**（含 0）。C / C5 / B 上的正向点估计没有重现；但三条配对区间也都含 0，所以**没有**证明这些基底下的真实对比不同。

---

## 0. 这回答了教授的什么问题

论文 Limitation L1 说 C 池的正向结果 "suggestive pending leak-free re-selection"。上一轮的 C5 回答不了——它的 20 列是用评估期信息选的。C-pre 是能正面回答的最小实验：**把 C 池的特征基底用只到 2022-06-30 的信息重选一次**，再按冻结协议重调 L0/L1 并跑完整 12 折。

Codex 认可的论文级描述（英文原句）：

> We conducted a retrospective sensitivity analysis using feature re-selection whose scoring and grouping inputs were restricted to information through June 2022, conditional on the study's fixed stock panel, and re-tuned LightGBM and MLP under the existing evaluation protocol.

**它不是什么**：不是 confirmatory（事后设计、无 BH 族、nominal p）；不估计 C 被泄漏抬高了多少；不重评图/边对比；不解除面板非 point-in-time 的限制（论文 L8）；不把 C 的原结果变成干净证据。B 仍是唯一的无泄漏锚点。

---

## 1. 选择器（只选列，不训练）

在调参 train 段 **231 个特征日（2021-07-01→2022-05-31，标签终点 ≤ 2022-06-30）** 上，对 168 个候选（10 hc + 158 Alpha158，均 T−1）算单特征日度 Spearman IC；τ=0.50 最小覆盖；按 Plan-AAA 的 61 组取"成员 |时间均值 IC| 的均值"排名；**top-15 组的成员并集 = 48 列（5 hc + 43 Alpha158）**。与 C 的 51 列重叠 22 列，与 C5 的 20 列重叠 4 列。

| 项 | 值 |
|---|---|
| 选择窗 | 231 日 2021-07-01→2022-05-31（断言；标签终点 = 2022-06-30，正好落在边界上） |
| 未打分特征 | `hc_mom12m`（85/231 = 0.368；2022-01-28 前横截面全零） |
| 其余 167 个候选覆盖率 | 231/231 |
| τ 稳健性 | τ=0.75 排名相同；τ=0 让 `hc_mom12m` 排第一并挤掉 RESI60（14/15） |
| 选中的 15 组 | hc_ret_std_5d+1 / hc_dolvol / hc_ret_std_21d+1 / STD5+1 / KLEN / MAX20+3 / CNTN20+1 / CNTP20+3 / BETA20+8 / MAX5+5 / STD20+1 / WVMA60 / CNTP5+5 / ROC20+4 / RESI60 |

（source: `artifacts/storya_cpre_select/selection.json`, `group_scores.csv`, `selector_robustness.csv`）

**两条必须随结果一起说的限定**：
- **"pre-evaluation" 只约束输入**。规则本身（边际 |IC| 的分数形式、61 组划分、top-15 基数、合格阈值）沿用自 2026-05-27 的诊断脚本（它的打分窗是面板最后 313 个标签日，落在评估期内）与 test-informed 的 Plan-AAA → C 构造。保留是为了口径与宽度可比，未重新优化。
- **top-15 的切分几乎是并列**。第 15 名 RESI60 0.03198 与第 16 名 IMIN10 0.03150 只差 0.00048；单特征时间均值 IC 的标准误约 0.0099，按 21 日标签重叠放大后约 0.045——**60 个可排名组全部落在第 15 名 ± 一个放大 SE 之内**。这 48 列是许多近似并列集合中的一个抽样。

---

## 2. 运行与调参

Mac M4（mps，单次调用，git `46ca6e3`，source_clean True），**240/240 cell 完成且收敛**，cell_id [3600, 3839]（与 confirmatory [0,2399]、C5 [2400,3599] 不交），每个 cell 都是冻结日历全长（749 天），frozen-HP 门通过（md5 `a8fdfb8f`）。integrity **PASS**。

调参（30 trials，top-5 × 3 seeds，train ≤ 2022-06-30 / val 2022H2）：L0 冠军 val-IC **+0.0311**，L1 **+0.0135**——两臂决赛全为正（C5 当时全为负）。L1 冠军 1 层 × 128、31,361 参数，与 C 的 31,745 基本相同。**限定**：LightGBM 在冻结参数下是确定性的，L0 三个调参 seed 的 val-IC 完全相同，"三 seed 平均"对 L0 不含初始化信息；这些是**选择指标**，既不证明调参充分，也不是独立验证。

---

## 3. 结果

**主表**（seed 平均日度 ΔIC = L1 − L0；21d 平稳块自助 5000 次；HLN 自动 lag / lag 21；source: `artifacts/storya_v21_family1_cpre/cpre_comparison.csv`）

| 基底 | ΔIC L1−L0 | 95% CI | HLN p auto / lag21 | IC L0 | IC L1 | 自身 MDE | k/10 | LOSO |
|---|---|---|---|---|---|---|---|---|
| **C-pre**（48 列，pre-evaluation） | **−0.0024** | **[−0.0256, +0.0178]** | 0.786 / 0.847 | 0.0057 [−0.0230, +0.0376] | 0.0033 [−0.0181, +0.0245] | 0.0313 | 5/10 | 1/10 |
| C（51 列，test-informed） | +0.0148 | [−0.0004, +0.0304] | 0.011 / 0.063 | 0.0195 | 0.0343 | 0.0220 | 10/10 | 0/10 |
| B（价量，无泄漏） | +0.0143 | [−0.0051, +0.0341] | 0.052 / 0.181 | 0.0228 | 0.0371 | 0.0275 | 10/10 | 0/10 |
| C5（20 列，test-informed） | +0.0134 | [+0.0008, +0.0283] | 0.008 / 0.054 | 0.0203 | 0.0337 | 0.0197 | 10/10 | 0/10 |

**配对日度对比**（比较基底 − C-pre，同 749 天；只报绝对变化，不做比例推断；source: `cpre_paired_contrast.csv`）

| 对比 | 均值 | 95% CI | HLN p auto / lag21 | 配对 MDE |
|---|---|---|---|---|
| C − C-pre | +0.0172 | [−0.0111, +0.0494] | 0.102 / 0.271 | 0.0433 |
| B − C-pre | +0.0167 | [−0.0133, +0.0509] | 0.131 / 0.309 | 0.0447 |
| C5 − C-pre | +0.0158 | [−0.0095, +0.0481] | 0.131 / 0.292 | 0.0414 |

**折结构**：C-pre 的对比对 2025Q2（fold 9）敏感，且方向与别的基底相反——该季 LightGBM IC 0.204、MLP 0.066，ΔIC −0.139（12 折中贡献最负；C、C5 在同一季是最大正贡献）。剔除该季后 C-pre = +0.0099 [−0.0069, +0.0265]（p 0.198 / 0.348）。**全期结果为主，剔除只是诊断**（拼接跨过被删季度，有一个人为接缝）。

---

## 4. 怎么读（Codex TP3 许可的读法）

- **主句**：On C-pre, the seed-averaged daily MLP−LightGBM contrast was −0.0024 (95% CI [−0.0256, +0.0178]; nominal HLN p = 0.786 / 0.847). **The interval contains zero; this does not establish absence of a contrast.** |Δ| 低于 C-pre 自身的 MDE 0.0313。
- **未重现**：The positive pooled **point estimate** observed in C, C5 and B was not reproduced on C-pre. **However, all three paired difference intervals contain zero**, so the analysis does not establish that the underlying contrasts differ across these bases.（两句必须连着说。）
- **两臂水平**：Both models had low pooled IC point estimates on C-pre (LightGBM 0.0057, MLP 0.0033, intervals include 0), lower than on C and B; **this descriptive pattern does not establish negligible predictive content or identify the cause.**
- **不解决什么**：C-pre does not estimate how much leakage inflated C. "Clean basis" = 选择与分组输入 ≤ 2022-06-30，以固定面板与事后协议为条件。B remains the leak-free anchor.

**禁用**（TP3 清单，节选）："not reproduced" 不带点估计限定与配对区间；"the selected features contain little or no predictive signal, whereas B contains signal"；"removing leakage eliminated / halved / reduced the MLP advantage"；"the models are equivalent on C-pre"；"excluding the anomalous quarter reveals the true MLP advantage"；"C-pre resolves L1 / provides leak-free confirmation"。完整清单在 `docs/analysis.md` 2026-09-12-a。

---

## 5. 对论文的修改（需要您定）

1. **main.tex:290（L1）、:998、:1012**：改正 "only 5 of the 15 groups survive strict T−1 re-ranking"——那是两种排名方法的交集，与 T−1 修正无关（图已按新口径重画）。
2. **L1 增补一句**：承认已完成的 L0/L1 pre-evaluation 重选敏感性，用 §4 的许可句；**保留**对 C 原结果的限定；**明确说**图/边对比的重选没有做。
3. **附录**：插入 §3 的主表 + 配对表，配 §4 的读法。
4. **C5 段落**：删掉或改写 "does not distinguish an unchanged contrast from a halved or doubled one"（那不是被检验的命题）。

---

## 6. 评审链与产物

| 触发点 | 评审 | 结论 |
|---|---|---|
| TP1 Plan A/B | Codex | PROCEED-WITH-FIXES（4 条全落实）→ Round B 全 FIXED |
| TP2 Code A | Codex | PASS-WITH-CONCERNS（1 条已修）："production run may start" |
| TP3 Results A | Codex | PROCEED-WITH-FIXES（1 MAJOR 措辞 + 3 CONCERN，全部落实/接受） |
| Closeout | 4 个 Explore agent | PASS（0 CRITICAL；1 MAJOR + 20 CONCERN 全部当场修复） |

Closeout 的独立复核值得一提：泄漏 agent **从原始产物完整重算了整个选择**（与归档的 IC 最大差 5.0e-7，48 列完全相同），并把价格面板截断到 2022-06-30 重算 hc 与标签，231 个选择日上逐位相同；统计 agent 逐一核对了 114 个数字，全部无误。

**产物**：`artifacts/storya_cpre_select/`（选择器归档）、`experiments/storya_v21_main12_cpre/`（240 cell）、`artifacts/storya_v21_family1_cpre/`（统计，`cpre_comparison.md` 是一页版）、`artifacts/storya_v21_tune/{CPRE_L0,CPRE_L1,frozen_hparams_cpre,cpre_tune_archive_md5}.json`、`artifacts/reviews/2026-09-1{1,2}_*`。

**复现**：
```bash
python run_storya_cpre_select.py                       # 选择器（~7 s）
python run_storya_v21_tune.py --arm L0 --universe CPRE --n-trials 30 --top-k 5
python run_storya_v21_tune.py --arm L1 --universe CPRE --n-trials 30 --top-k 5
python run_v21_tune_launcher.py --merge --merge-universes CPRE --merge-arms L0,L1 --merge-out frozen_hparams_cpre.json
python run_storya_v21_main12.py --universe CPRE --arms L0,L1 \
  --frozen-hparams experiments/storya_v21_tune/frozen_hparams_cpre.json \
  --out-dir experiments/storya_v21_main12_cpre                      # 240 cell，Mac ≈ 1.5 h
python compute_family1_ladder.py --main-dir experiments/storya_v21_main12_cpre \
  --output-dir artifacts/storya_v21_family1_cpre --universes CPRE --arms L0,L1 --sensitivity
python analyze_c5_sensitivity.py --universe CPRE \
  --frozen experiments/storya_v21_tune/frozen_hparams_cpre.json --ex-fold 9
```

## 7. 遗留

- **push**：本 session 的 C-pre commit（`b969a62` … `0bbe9c8`）仅在本机，待您确认。
- **不建议再跑**：C-pre 上的 L2（correlation-GAT）层、C5h、逐臂水平的配对检验（只有在要断言"某一臂的水平下降"时才需要）。
- 工作树里仍有与本任务无关的旧改动（`analyze_e1_lofo.py`、`paper_figs/fig_family2.py`、`figures/family2_edge_causal.*`），未触碰。
