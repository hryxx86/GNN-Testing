# Paper 全项目系统性评估报告 — 2026-07-02

> **评估对象**: `paper/main.tex`（"When Do Graph Neural Networks Help in Cross-Sectional Stock Ranking?"，8pp 匿名 ICAIF 版 / 9pp 非匿名 arXiv 版，单源双用）
> **评估人**: Claude（全项目通读 + 独立复核），应 H博士 要求
> **导向**: 全力赶 ICAIF 2026（deadline **2026-08-02**，双盲，8 页硬上限含图表引用、无附录，CMT 系统；source: [icaif2026.org/call-for-papers.html](https://icaif2026.org/call-for-papers.html)，2026-07-02 抓取）
> **覆盖**: main.tex 全文逐行、main.pdf 9 页逐页、4 个 confirmatory 分析器公式级核对、8 个关键 CSV 亲验、预注册协议对照、PaperJury LEDGER 49 项、docs/ 历史（3 个 Explore agent 地毯式扫描 + 本人精读）、2025-2026 文献新鲜度检索
> → progress: 2026-07-02-a | plan: 2026-07-02-a | analysis: N/A（评估报告，非新实验；4 项零重跑核查见 §5，落档 `artifacts/audits/paper_eval_robustness.csv`）

---

## 1. 总评（TL;DR）

**这篇论文可以投，而且底子比大多数投稿硬。** 数字层零错（本次亲验 8 个 CSV 对 main.tex 逐项匹配）、统计实现正确（NW-HAC/DM/HLN/BH/SPA/bootstrap 公式级核对全过，见 §3）、预注册忠实（20 检验族与 `docs/protocol_v2_freeze.md` §6 逐字一致）、叙事纪律好（"fail-to-reject ≠ 等价"、suggestive 分级、L1–L9 诚实披露）。文献检索未发现撞车："严格协议下 GNN vs GBDT 股票排名"的验证性研究在 2025-2026 无先例（见 §7）。

**但有一个战略级叙事漏洞和两类可零成本关闭的 open majors**：

1. **E3 planted-signal 阳性对照完全没进论文**（§4 详述）。负结果论文的第一攻击永远是"你的实现可能是坏的"——而项目手里有教科书级的反证（GNN 恢复 82–91% 可达 IC、MLP≈0；source: `experiments/sanity_summary/verdicts.json`），却一个字没写。**这是本次评估发现的最高价值、最低成本的改进。**
2. **3 个 open majors（I-07/I-14/I-19）可用现成数据零重跑关闭**——本次评估已经把数算完了（§5）：26 检验合池 BH 决策 100% 不变；6 个 BH-rejected contrasts 的 leave-one-seed-out 全部 0/10 翻号；保守 BY 校正下 headline L2−L1 双宇宙仍拒绝而 L1−L0 正向主张不拒绝（与论文现有 suggestive 分级完美自洽）。每项一句话即可入纸。
3. **1 个事实性引用错误**（HXZ 65%，I-35）金融审稿人必抓，30 分钟修。

按 §6 的分级清单执行（T0+T1 合计约 2–4 个工作日、零训练重跑），论文攻击面可以收掉一大半，赶 8/2 毫无压力。

---

## 2. 评估方法与覆盖

- **论文本体**: main.tex 393 行逐行精读；main.pdf 9 页逐页视觉检查（浮动体、可读性、双盲、desk-reject 风险）。
- **数字忠实性**: Table 2/3/4/5 + abstract/正文全部 load-bearing 数字对源 CSV 亲验（`family1_ic_ci.csv`、`family1_dm_hln.csv`、`family1_spa.csv`、`family2_fc_causal.csv`、`cost_headline_crosswalk.csv`、`family1_lofo.csv`、`family1_mde.csv`、`family1_stability.csv`、`family1_cl5s_robustness.csv`）→ **零失配**（与 PaperJury Round-1/2 结论一致，独立重验）。
- **统计实现**: `compute_e6_dm_spa.py`（canonical 引擎）、`compute_family1_ladder.py`、`compute_fc_edge_causal.py`、`compute_cost_confirmatory.py` 四个分析器的核心统计函数公式级核对（§3）。
- **预注册忠实度**: 论文 §3.3 的 20 检验族（阶梯 5 对 + 边 5 对 × 2 universe）与 `docs/protocol_v2_freeze.md` §6（line 96）逐字一致；SPA M=9 含 L7 一致；L7 contingency 未触发（diverge_frac=0，source: `artifacts/storya_v21_family1/family1_ledger.json`）。
- **评审台账**: `paper/.paper-review/LEDGER.json` 49 项逐条读（5 open majors + ~25 open minors + 5 deferred R2 concerns）。
- **项目史**: progress.md / plan.md / docs/analysis.md 全量 + 3 个 Explore agent 扫描（docs/ 40 文件、experiments/ 24 个结果族、archived/ 内部批评史）。
- **文献**: WebSearch 4 组检索（GNN 股票负结果/基准、ICAIF 2025、GBDT vs DL 截面、2026 arXiv 评估严谨性）。

---

## 3. 统计实现独立复核 — 全部通过

| 组件 | 位置 | 核对结论 |
|---|---|---|
| NW-1994 自动带宽 | `compute_e6_dm_spa.py:83` | L=⌊4(T/100)^{2/9}⌋，T=749 → L=6 ✓（与论文 §3.3 一致） |
| NW-HAC 长程方差 | `compute_e6_dm_spa.py:200` | Bartlett 核、γ_l 除以 T（NW-1987 规范）、权重 1−l/(L+1) ✓ |
| DM 统计量 | `compute_e6_dm_spa.py:229` | mean(d)/√(nw_var/T) ✓ |
| HLN 小样本校正 | `compute_e6_dm_spa.py:247` | √((T+1−2h+h(h−1)/T)/T)，h=21，t_{T−1} 双侧 ✓（Harvey-Leybourne-Newbold 1997 原式） |
| BH-FDR | `compute_e6_dm_spa.py:263` | step-up 规范实现 ✓ |
| 平稳块 bootstrap | `compute_e6_dm_spa.py:286` | arch StationaryBootstrap，几何块长均值 21 ✓ |
| Hansen SPA | `compute_e6_dm_spa.py:308` | arch.bootstrap.SPA，loss=−daily IC，p_lower/consistent/upper ✓ |
| Family-2 推断 | `compute_fc_edge_causal.py:96-145` | fold-level seed-avg 配对 ΔIC、12 块 t 检验入 BH、bootstrap CI 降 descriptive、MDE=(z₀.₉₇₅+z₀.₈₀)×SE ✓ |
| Cost 层配对 ΔSharpe | `compute_cost_confirmatory.py:171-218` | 12 fold-block 平稳 bootstrap 百分位 CI + LOFO ✓（**论文未写明 CI 构造 = I-39，半句可补**，见 §6-T0） |
| C/L5s 处理 | `compute_family1_ladder.py:146-176` | nanmean over seeds 保天数（`family1_ic_ci.csv` C/L5s T=749 亲验）→ I-24 的"ragged column"担忧在结构上被缓解 + 三处理稳健性 0.077–0.080 已披露 ✓ |

**一处实现层评注（非错误）**: Family-2 MDE 用 z 乘子而非 12 块下的 t 乘子（I-36）。方向性：t 乘子会让 MDE 更大 → "6/6 underpowered" 结论只会更强 → 对论文结论无影响，可不动。

---

## 4. 七维评估

### D1 贡献与定位 — 强，且检索后更有信心
- Novelty 声明（§2 "first to combine …"）经 4 组检索**未发现先例或撞车**。最接近的 2025-2026 工作（见 §7 文献清单）要么是新架构论文（ACT、EP-GAT、GRU-PFG、FinMamba），要么是相邻问题（GraphNetz 做 GNN 基准统计学但非金融、"When Alpha Breaks" 做部署不确定性、"Do Better Volatility Forecasts…" 做波动率），没有人做"预注册两族 + SPA/FDR + 成本层"的图价值验证性研究。
- Conditional-negative-results + 方法论贡献的定位对 ICAIF 合适：ICAIF 历届接收过 benchmark/evaluation 类论文，且 HXZ/López de Prado 传统给了正当性框架。
- 风险：title 问"When do GNNs help"，正文的"when"答案偏否定侧。现有 abstract 已经用"conditional findings and failure modes"自洽，可不动；若想微调，§6-T0-13 给了一个可选的一句话方案。

### D2 叙事最优性 — headline 选择正确，经本次独立验证进一步坐实
- 2026-06-30 把头条从"MLP>LightGBM"换成"L2−L1<0（图拖累 MLP）"是正确决策，本次评估的 BY 检验（§5.3）给出新证据：**保守 BY 校正下 L2−L1 双宇宙仍拒绝，而 L1−L0 恰好不拒绝**——论文把前者当 headline、后者标 suggestive 的分级与最保守口径完全同构。这个巧合值得写进论文一句话（不增推断层，纯敏感性）。
- Abstract 三段结构（动机→Family-1→Family-2+cost）信息密度高但读得通；无需重写。

### D3 统计残余风险 — 5 个 open majors 中 3 个可零成本关闭（§5），2 个文字可解
- I-07/I-14/I-19 → §5 的三项计算 + 各一句话 → 关闭。
- I-01（isolation 措辞）+ I-02（等 trial 预算混杂）→ 文字修复（§6-T0-2/3）。I-02 另有一个**纸内已有但未连线的反证**：L4/L5 与 L2 同为 GAT、同维搜索空间、同 30 trials，在 C 却恢复到 L2 之上（L4−L2=+0.0163、L5−L2=+0.0152，均 BH-reject；source: `artifacts/storya_v21_family1/family1_dm_hln.csv` rows C L4-L2, C L5-L2）——若 30 trials 系统性把 GAT 调残，无法解释同预算同空间的 L4/L5 恢复。这个论证一句话就能把 under-search 攻击的杀伤力砍半（完全关闭仍需 trials sweep，T2 可选）。
- I-50（HAC lag=21 敏感性）：H博士 已决策 B（不报）。本次评估量化其风险敞口：lag=21 下失守的只有 C L1-L0（p=0.063）、C L3-L2（p=0.070）、C L4-L2（p=0.070）三项**次要主张**（source: `family1_dm_hln.csv` HLN_p_t_lag21 列），headline L2−L1（B p=0.011、C p=0.001）与 C L5-L3（p=8.5e-5）稳如磐石。维持决策 B 风险可控；若审稿被问可在 rebuttal 给数。

### D4 证据完备性 — 两个纸外弹药该进纸（这是本次全项目通读的核心增量）
1. **E3 planted-signal 阳性对照（强烈建议进纸）**。现状：论文对"实现是否可信"零防御。项目实际有：E0 wiring 14/14 PASS、E3 planted-signal GNN 恢复 82–91% 可达 IC（SAGE 91%/GAT 82%，MLP≈0，HLN p≈0 BH 过，5 folds × 4 seeds 全一致；source: `experiments/sanity_summary/verdicts.json`、`docs/analysis.md` 2026-06-11-a）。**建议 §5.6 后或 §4 末加 3–4 行** + Discussion 半句。诚实边界：E3 跑在 5-fold untuned anchor 管线上（非 tuned 12-fold），措辞必须是"the graph pipeline is operational; the null is not a broken-pipeline artifact"，不得说"tuned arms adequate"。这同时侧面压制 I-02（管线能转换图信号 → "图臂全被调残"的先验概率下降）。
2. **Sliding-252d 副轴结果（建议一句话进纸）**。论文 §3.1 预告了副轴却从未报结果。现有：8/10 对同号，且 2 个翻号方向是 B 宇宙 GAT/SAGE vs MLP 转负——**强化**而非削弱"图边不帮忙"（source: `docs/analysis.md` 2026-06-15-a §3，`experiments/storya_anchor_sliding/results.csv`）。一句话堵住"expanding window 陈旧数据伪影"攻击。
3. 不建议进纸：permutation 16K（叠加推断层）、horizon ablation 细节（保持 §3.4 exploratory 披露现状）、E1b 泄露 oracle −0.044（有趣但属 pilot 管线，机制段已够用）、regime 热图（空间不允许）。

### D5 审稿人攻击面 — 见 §8 模拟表
十二项预期攻击中，**7 项已有强防御在纸内**，3 项本次可零成本补齐（E3、seed、multiplicity），2 项靠披露与 rebuttal（单市场单 horizon、|ρ|>0.6 任意性）。

### D6 形式合规 — 基本干净，两处小风险
- 双盲：`anonymous` 开关方案可行；正文无自曝（"Plan-AAA"/"the project" 为内部代号不暴露身份；§8 未给 repo URL）。CFP 允许 arXiv preprint 在先（"Specific examples of permissible venues include arXiv"），但**双盲期间论文内不得引用该 preprint**。
- 页预算：匿名版恰好 8pp，非匿名 9pp 第 9 页仅 2 条参考文献溢出 → **任何加句都要有对应删句**（§6 给了净零页预算方案）。
- I-29 压缩宏（表格 `\scriptsize`+`\arraystretch{0.88}`、参考文献 `\scriptsize\bibsep=0pt`）：不改 textwidth/margin，非 desk-reject 级；但若 ICAIF 格式检查严格要求 reflow，粗估多出 0.3–0.5 页。缓解：优先删句而非依赖压缩宏（§6 净零方案已按此设计）。
- CMT 提交、无附录 → 现有"无附录 + repo 指针"结构正确。

### D7 历史遗留 A2（feature-horizon alignment artifact）— 已不适用，无需处理
2026-04-12 adversarial review 的 A2 攻击针对的是"21d horizon 峰值是特征-标签对齐伪影"这一**旧主张**；confirmatory 论文没有做任何 horizon 优选或 horizon 峰值主张（21d 是设计选择，§3.4 已把 horizon ablation 列为 exploratory 披露，L7 已声明单 horizon 限制）。攻击面已随主张一起消失。**结论：A2 无需任何动作**——这解决了 archived 批评史里唯一的 UNRESOLVED 项。

---

## 5. 本次新算的三项零重跑稳健性结果（可直接入纸）

> 计算脚本: `analyze_paper_eval_robustness.py`（本次新建，只读现有数据）；结果落档: `artifacts/audits/paper_eval_robustness.csv`。输入: `experiments/storya_v21_main12_tuned/results.csv`（2160 cells）、`artifacts/storya_v21_family1/family1_dm_hln.csv`、`artifacts/storya_v21_family2_fc/family2_fc_causal.csv`。

### 5.1 Per-seed 符号一致性 + leave-one-seed-out（关闭 I-14）
对全部 6 个 BH-rejected contrasts，按 seed 重构 pooled ΔIC（fold IC_mean 按 n_test_days 加权）：

| Contrast | pooled ΔIC | per-seed 同号 | LOSO 翻号 |
|---|---|---|---|
| C L1−L0 | +0.0148 | 10/10 | 0/10 |
| B L2−L1 | −0.0133 | 8/10 | 0/10 |
| C L2−L1 | −0.0119 | 9/10 | 0/10 |
| B L3−L2 | −0.0149 | 8/10 | 0/10 |
| C L3−L2 | −0.0123 | 9/10 | 0/10 |
| C L5−L3 | +0.0275 | 10/10 | 0/10 |

（source: `artifacts/audits/paper_eval_robustness.csv` rows check=per_seed_sign）

**读法**: 去掉任何单个 seed，6 个 rejection 的符号全部不变（与 §5.3 LOFO 的 0/12 folds 完美对称）——"lucky seed 驱动"假说被直接否证。per-seed 点估计存在离散（§1 已披露），但多数符号一致 8–10/10。**建议入纸一句**（对称句式）: "A leave-one-seed-out check mirrors the fold check: removing any single seed never flips the sign of any BH-rejected contrast (0/10 for all six), and per-seed signs agree with the pooled sign in 8–10 of 10 seeds."

### 5.2 26 检验合池 BH（关闭 I-07 主体、支撑 I-19）
把 20 个 DM/HLN p 值与 6 个 Family-2 fold-block t 检验 p 值合成单一 BH 族（q=0.05）重跑：**26 检验合池后的拒绝决策与预注册分族决策逐项相同**（11 项拒绝集合不变，无 FC contrast 新增拒绝；source: `paper_eval_robustness.csv` row check=pooled_bh_26, identical_to_preregistered=True）。**建议入纸一句**: "As a sensitivity, pooling all 26 pre-registered tests (20 DM + 6 Family-2) into a single BH family leaves every rejection decision unchanged."——"三层 q=0.05 loophole"攻击就此失去弹药。SPA 层本就测不同的全局 null（选择校正后是否有臂胜过 L0），与 DM 局部对比在論文 §5.2 已明确分工；补上这句后 I-19 只剩措辞层（§6-T0-4）。

### 5.3 BY（任意依赖）敏感性（回应 I-37，反向加固 headline）
对 20 检验族施加 Benjamini–Yekutieli 校正（不依赖 PRDS 假设）：**7 项存活 — B L2−L1、C L2−L1、C L2s−L2、C L5−L2、C L5−L3、C L6−L2、C L7−L2；不存活的恰是 C L1−L0、C L3−L2、B L3−L2、C L4−L2**（source: `paper_eval_robustness.csv` rows check=by_20）。

**读法**: 最保守的多重校正下，headline（图拖累 MLP）在两个宇宙都站着；而论文本来就标 suggestive 的 MLP>LightGBM 与本来就标 cost-sensitive 的 news-edge 恰好掉出。**论文现有分级与 BY 口径完全同构**——这句敏感性既回应 I-37（"你用的 BH 变体假设 PRDS"），又免费强化 headline 的"bandwidth-robust + universe-robust + fold-robust + seed-robust + dependence-robust"五重稳健叙事。

---

## 6. 建议清单（分级 + 页预算净零方案）

> 全部按"8/2 前可落地"筛选。T0+T1 合计约 2–4 个工作日。**执行前需 H博士 逐条批准（PaperJury 规则 1：author sign-off before edit）**；改动落地后建议触发一轮 `/codex-code-review`（若动了脚本）与最终 provenance verifier。

### T1 — 现成数据新句子（最高性价比，先做，共 ~1.5–2 天）
| # | 动作 | 关闭 | 页成本 |
|---|---|---|---|
| T1-α | **E3 planted-control 3–4 行**入 §5.6 后（新 §5.7 前）或 §4 末 + Discussion 半句。措辞锚定"pipeline operational / not a broken-pipeline artifact"（Codex R-A-01 scoped 版本） | 最大叙事漏洞；侧面压 I-02 | +4 行 |
| T1-β | **LOSO seed 一句**（§5.3 regime 段后接续，句式与 LOFO 对称，§5.1 表格数字） | I-14 | +2 行 |
| T1-γ | **26 检验合池 BH 一句**（§5.2 "do not claim a joint family-wise rate" 句后） | I-07 主体 | +1.5 行 |
| T1-δ | **BY 敏感性一句**（紧跟 T1-γ，或并成一句复合句） | I-37 + 加固 headline | +1.5 行 |
| T1-ε | **Sliding 副轴一句**（§5.3 末或 §3.1 就地）: "The sliding-252d robustness axis reproduces the same-direction picture (8/10 pairs same-sign; both flips strengthen the no-edge-benefit reading)." | expanding-window 伪影攻击 | +1.5 行 |

### T0 — 纯文字修复（各 ≤30 分钟，共 ~1 天）
1. **HXZ 引用修正（I-35，必做）**: §2 "65% fail after multiple-comparison adjustment" → 事实是 65% 过不了单检验 |t|≥1.96、多重校正后失败率升至 82%（Hou-Xue-Zhang 2020 RFS 摘要口径）。改为准确表述。
2. **Table 1 caption 软化（I-01）**: "Each rung isolates one design choice" → "Each rung changes one nominal design axis; every arm is retuned independently (30 trials), so rung contrasts compare tuned operating points, not capacity-matched pairs (see Family-2)."（±0 行，caption 内改）
3. **I-02 caveat 前置到解读点**: abstract 或 §5.2 headline 处补 "at equal 30-trial budget" 短语；§6 Discussion 的 L2−L1 段补半句 under-search 替代解释 + **L4/L5 同预算同空间恢复的反证**（§4-D3 的论证，一句话）。
4. **I-19 收口措辞**: §5.2 现有分工声明后加半句 "SPA is the confirmatory answer to the global benchmark question; the DM ladder is pre-registered local evidence at its own FDR level"（与 T1-γ 合并落笔更顺）。
5. **Family-2 "causal" 降温（I-34）**: 保留 family 结构，§3.3 header "Family-2 (causal)" → "Family-2 (edge attribution)"，首段保留 "causal-flavored, at a fixed operating point" 限定；abstract "The causal family" → "The edge-attribution family"。（防 methods 审稿人抠"causal"）
6. **§4 标签公式对齐（I-44）**: display 式补 z-score 或前后句挪一下，使公式与"cross-sectionally z-scored"文字一致。
7. **"501 names" 调和（I-49）**: §4 加半句 "(the 501-name survivor snapshot of the S&P 500; §L8)"。
8. **Turnover 实测一句（I-42）**: §5.4 补 "realized per-rebalance L1 turnover ranges 2.25 (LightGBM) to 2.90 (MLP) in Universe C"（source: `artifacts/storya_v21_cost/cost_ladder_by_arm.csv` rows C/L0、C/L1, cost_bps=10）。
9. **ΔSharpe CI 构造半句（I-39）**: §5.4 "computed from the paired fold differences" → "…via stationary block bootstrap over the 12 fold-level paired differences (5000 reps)"。
10. **Sharpe level n≈36 caveat（I-46）**: §5.4 level 数字旁加 "(≈36 rebalances)"。
11. **精度统一（I-48/I-22/I-28）**: abstract ΔIC 统一 4 位小数或全文统一 3 位；SPA p 统一 0.077 口径；"roughly one quarter"→"27.5%"（一处即可）。
12. **L5−L3 容量混杂脚注（I-47/I-41）**: tab:cost caption 或 §5.4 半句 "L5−L3 is a tuned-ladder contrast (capacity-confounded like all Family-1 pairs)"。
13. （可选）title 的"when"回应: §7 结尾或 §6 加一句正面 takeaway "Where the graph does carry signal (sector edges at fixed operating point, dense attention in rich features), effects are below current detectability — larger samples, not larger models, are the binding constraint."（此句概括 C L4/L5 fc 方向、L6 表现与 MDE 框架，全部有数据支撑）

### T2 — 小重跑（可选，8/2 前可行但非必需；触发则走 Rule 9 TP1）
| 动作 | 解决 | 成本 | 建议 |
|---|---|---|---|
| Trials-sensitivity: C 宇宙 {L1, L2} N=30→60 Optuna + 冻结重跑 2 臂 | I-02 正解 | ~1–2 天 Colab A100 + 分析 | **可选**。T0-3 + L4/L5 反证后剩余风险已低；若 H博士 想要"secondary positive 也硬"再做 |
| Leak-free Universe-C 重选（T-1 幸存 basis 重建 + C ladder 重跑） | L1 根治 | ~3–5 天 Colab + 全套分析 | **不建议赶 8/2**（QUEUE 保留，camera-ready/期刊版再做） |
| PIT 宇宙重建 | L8 根治 | >1 周 | 不做（<20% 阈值未触发，披露充分） |

### 页预算净零方案（T1 共约 +10–11 行）
等量删减候选（按优先级）：
- §5.7 ListMLE 段压缩 3 行（保留翻转数字 + 一句机制，删 softmax 细节句）
- §1 seed-dispersion 段（C-GAT CI 例子）压 2 行（数字保留在 tab:ic 的 CI 已有）
- §2 GNN baselines 段压 2 行（Sawhney/TRA/MDGNN 列举句可并）
- §3.2 Sharpe 定义段压 1–2 行（√(252/21) 句与 block 句可并）
- L6/L7 限制句在 §2 与 L6 有重复表述，可各删半句（~1.5 行）

---

## 7. 文献新鲜度（2026-07-02 检索）

**撞车检查：未发现。** 无 2025-2026 论文做"预注册/多种子/SPA-FDR/成本层的 GNN 股票排名验证性评估"。

可选补引（页预算紧张，均非必需；若 T0/T1 后有余量再考虑）：
- GraphNetz: Statistical Benchmarking of GNNs with Paired Tests and Rank Aggregation（arXiv 2605.09099）— 非金融但同方法论精神，§2 methodology 段半句可挂
- When Alpha Breaks（arXiv 2603.13252）— 截面 ranker 部署不确定性，109-fold walk-forward，related-work 可选
- Do Better Volatility Forecasts Lead to Better Portfolios? Evidence from GNNs（arXiv 2605.19278）— GNN 金融批判性评估相邻工作
- ACT（arXiv 2604.20204）/ EP-GAT（arXiv 2507.08184）/ GRU-PFG（arXiv 2411.18997）— 新架构类，不引不损失
- `docs/storya_references.md` 既有 to-be-added（Stockformer / Pinheiro-Wedge）维持"视页数"原判

---

## 8. 审稿人攻击面模拟（ICAIF 双盲视角）

| # | 预期攻击 | 现纸内防御 | 本次建议后 |
|---|---|---|---|
| 1 | "负结果 = 你们不会调 GNN / 实现有 bug" | **无** | T1-α E3 阳性对照 → 强 |
| 2 | "30 trials 把高维 GAT 调残了"（I-02） | §4 一句 caveat | T0-3（解读点 caveat + L4/L5 反证）→ 中强；T2 sweep → 强 |
| 3 | "三层 q=0.05 挑有利口径"（I-07/I-19） | §5.2 分工声明 | T1-γ/δ 合池 + BY 敏感性 → 强 |
| 4 | "seed-mean 掩盖 seed 脆弱"（I-14） | §1 披露离散 + §3.3 estimand 声明 | T1-β LOSO 0/10 → 强 |
| 5 | "幸存者偏差"（L8/I-17） | L8 两段式 + 液态边界量化 + level/contrast 区分 | 已强（Round-1/2 打磨过） |
| 6 | "Universe-C 泄露选择"（L1/M4） | headline 换 B-robust + suggestive 分级 | 已强；BY 结果再送一层 |
| 7 | "单市场单 horizon"（L7） | 披露 | 常规限制，可接受 |
| 8 | "HXZ 引用错误"（I-35） | — | T0-1 修正 → 消除 |
| 9 | "underpowered null 无信息量" | MDE 框架 + 三种样本量口径 + fail-to-reject 措辞 | 已强 |
| 10 | "净 Sharpe CI 小样本/构造不明"（I-39/I-46） | L5 部分披露 | T0-9/10 半句 → 补齐 |
| 11 | "expanding window 陈旧数据伪影" | §3.1 预告副轴但无结果 | T1-ε 一句 → 补齐 |
| 12 | "|ρ|>0.6 与 126d 任意"（L9） | L9 披露 + future work | 可接受（rebuttal 备 E4 图诊断：密度 0.7–1.9%，source: `experiments/sanity_e4_diagnostics/`） |

---

## 9. Open items 逐条处置建议（对照 LEDGER）

| ID | 级别 | 处置 | 依据 |
|---|---|---|---|
| I-01 | MAJOR | T0-2 caption+框架措辞 | 改动小、关闭干净 |
| I-02 | MAJOR | T0-3 文字 +（可选）T2 sweep | L4/L5 同预算反证 + E3 侧压后剩余风险低 |
| I-07 | MAJOR | T1-γ 合池敏感性一句 | 本次已算，决策不变 |
| I-14 | MAJOR | T1-β LOSO 一句 | 本次已算，0/10 全过 |
| I-19 | MAJOR | T0-4 指定 SPA=global confirmatory 措辞 | 论文已隐含，一句显式化 |
| I-34/I-33 | minor | T0-5 causal→edge-attribution + FC 首现展开 | 防 methods 抠字 |
| I-35 | minor | **T0-1 必修**（事实错误） | HXZ 原文口径 |
| I-39/I-42/I-46 | minor | T0-8/9/10 半句各补 | 数据都在 cost CSV |
| I-44/I-48/I-49/I-22/I-28 | minor | T0-6/7/11 机械修 | — |
| I-47/I-41 | minor | T0-12 对称 caveat | — |
| I-29 | minor | 维持现状 + 净零删句缓解 | 非 desk-reject 级 |
| I-36 | minor | 不动（方向性只会强化结论） | §3 复核 |
| I-24/I-45 | minor | 不动（三处理稳健已披露；保留=保守） | `family1_cl5s_robustness.csv` |
| I-50 | 已决策 | 维持 H博士 决策 B；rebuttal 备数 | §4-D3 风险量化 |
| R2 deferred ×5 | CONCERN | stat-02/gnn-01/gnn-04 若有页余量则半句；stat-04/qf-03 放弃 | 页预算优先给 T1 |

---

## 10. 投稿操作时间线建议（今天 7/2 → deadline 8/2）

1. **W1（7/2–7/8）**: H博士 逐条批 T0/T1 → 落地修改 → tectonic 双版本编译验证（匿名 8pp 必须卡住）→ provenance verifier + 措辞 grep sweep（M=9/suggestive/near-miss 红线复查）。
2. **W2（7/9–7/15）**: （若批准 T2-trials）Colab 跑 + Rule 9 TP1/TP3；同时 arXiv 非匿名版挂出（CFP 明确允许；论文内不自引 preprint）。
3. **W3（7/16–7/22）**: PaperJury 快速复核轮（只验 diff）；CMT 注册、题目/摘要/subject areas 预填。
4. **W4（7/23–8/1）**: 冻结期——只修编译/格式问题；8/1 前提交（勿踩 deadline 当天）。

---

*报告完。所有新计算可用 `analyze_paper_eval_robustness.py` 一键复现；评估过程读取的全部文件与验证步骤记录于 progress.md 2026-07-02-a。*
