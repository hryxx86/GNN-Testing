# 实验协议 v2.2 —— 冻结版

> 状态：**已冻结（Touchpoint 1 九条 disposition 已并入；v2.2 §4 搜索空间显式冻结）**。
> 决议史：S1=2022H2 降 case study；S2=全局 cold-start；S3=N=30；S4=L7 HATS 加回；S5=累积 α 语义；
> T1 处置（2026-06-12）：C1/M1/M2/Cn1–Cn5 接受并落地，**M3 后半（复用旧 anchor 结果）驳回**——主表 12 fold 全部在冻结新超参下重跑，旧 5-fold 仅作 pilot/smoke 对照（避免主表混超参）。
> M1 一致性判据 = **同号**（仅符号，不加 CI 重叠条件）；Cn5 阈值 = **20%**。

---

## 1. 数据与时间轴

| 段 | 区间 | 交易日 | 用途 |
|---|---|---|---|
| burn-in | 2021-01-29 → 2021-06-30 | ~106 | 特征预热（α1 满 126d 窗实跑核验 PASS；Alpha158 最长回看待 Cn4 grep 关尾） |
| 调参 train | 2021-07 → 2022-06 | ~252 | 仅 Phase 2 |
| 调参 val | 2022-07 → 2022-12 | ~126 | 仅 Phase 2 选模/early-stop → **2022H2 不得作任何正式测试**；与主轴 fold-1 训练段重叠属标准做法（GKX 先例），论文一句注明（Cn3） |
| 正式测试 | 2023Q1 → 2025Q4 | ~750 | 主轴 confirmatory（唯一推断家族） |
| 压力段 case study | 2022Q3–Q4 | ~126 | **"2022H2 压力段（回撤+Q4 反弹混合 regime）"**，带超参泄漏标注，描述性专节（Cn3 修正 regime 标签） |
| 弃用 | 2026-01（~19d） | — | 不足一季 |

Universe B（10 维手工）/ C（51 维 Alpha158 子集，**runtime T-1 shift**）；图 = per-fold frozen snapshot（train_end 前最后一张满 126d 窗，|ρ|>0.6）；新闻边 PIT = nyse_session_close_utc DST-aware cutoff（schema v2）。survivorship、sector 单快照（fetch 2026-02-09）、图新鲜度未作处理变量 → Limitations。

## 2. 双轴评估

### 2a. 主轴：expanding 12 fold（唯一 confirmatory，全局 cold-start）

| fold | train（起点 2021-07 固定） | fold 内 val（仅 early-stop） | test |
|---|---|---|---|
| 1 | → 2022-12 | 2022Q4 | 2023Q1 |
| … | 逐季扩 | 恒为训练段最后一季 | 逐季后移 |
| 12 | → 2025-09 | 2025Q3 | 2025Q4 |

### 2b. 副轴：sliding-252d（robustness/复现轴，无独立推断 —— M1 处置）

- 角色：堵"null 是训练过期伪影"的替代解释。**成功标准 = 方向一致性，非独立显著性。**
- 训练窗固定 252 交易日逐季滑动；fold 内 val = 窗内最后一季（有效训练 3 季 → Limitations，Cn2）；测试季 2023Q1→2025Q4（12 季，与主轴同季配对）+ 2022Q3/Q4 case study（泄漏标注）。
- 臂：L0/L1/L2/L5/L6；超参复用主轴冻结表（声明理由照旧）。
- **报告规范（预注册）**：每对只报 ΔIC 符号 + block-bootstrap CI；一致性判据 = **与主轴点估计同号**；汇报为 X/8 计数（4 对 × 2 universe）。副轴表格**无 p 值、无星号、无 BH-FDR、无 SPA**。
- **逃生口封条**：副轴任何"有意思"的模式（如 sliding 下某臂转正）只能作 exploratory/假设生成报告，**不得进摘要、不得进任何 claim**。

## 3. 消融阶梯

纪律：对比集预注册（§6）；只许加行不许改行；α 沿累积语义。**"现成"拆为两列**（M3 前半）：

| 行 | 配置 | 回答什么 | 代码状态 | 结果状态 |
|---|---|---|---|---|
| L0 | LightGBM | 非神经基线（C=墙，B=地板 IC≈0.006，叙事分 universe） | 现有 | **全新跑**（旧 anchor 仅 pilot；M3 驳回复用） |
| L1 | MLP 无图 | 神经化价值 | 现有 | 全新跑 |
| L2 | GAT + α1（corr） | 加图价值 | 现有 | 全新跑 |
| L3 | GAT + α1∪news | news 边边际价值 | 现有（**配置先例仅 SAGE/Univ-B/5-fold**） | 全新跑 |
| L4 | GAT + α2（corr+sector） | sector 边边际价值 | 现有（同上） | 全新跑 |
| L5 | GAT + α4（全） | 边叠加互补性 | 现有 | 全新跑 |
| L6 | full-attention-no-graph | attention-vs-structure 裁决器；**learned-attention 代表仅限 dense 家族（MASTER/AD-GAT 式 unmasked），不覆盖 learned-sparse（FinMamba 剪枝、ADB-TRM 自适应图 → Limitations/future work）**（Cn1 收窄） | **唯一新代码** | 全新跑 |
| L7 | HATS-3R-adapt | 领域修法代表（关系注意力）；Template-1 承重 | 现有（T2 PASS） | 全新跑；**受 §6 contingency 规则约束** |
| L2s/L5s | SAGE-Mean（α1/α4） | 聚合算子对照 | 现有 | 全新跑 |

## 4. 等预算调参

逐臂独立：10 配置 × 2 universe = 20 作业 × Optuna N=30（LightGBM 同 N，CPU）；窗口 2021-07→2022-06 / val 2022H2；选模指标 val 日均 Rank IC；top-5 各 3 调参 seed 复跑定冠军；评估 seed = canonical 10-seed，调参 seed 不相交（预检 #6）；**搜索空间见下表（v2.2 显式冻结，开调后不可改）**；冻结超参表进附录；regime 错配 + GKX 措辞照旧；2023 年底重调一次留作附录敏感性弹药不跑。

**搜索空间（v2.2 显式冻结 2026-06-15；中心 = pilot 默认值 → pilot 即 N=1 中心点样本，确保 pilot-vs-调参可对比）**：

NN / GAT / L6 / L7（每维中心=`run_storya_e1_anchor.NN_HPARAMS`）：
| 维度 | 中心(pilot) | 搜索范围 |
|---|---|---|
| lr | 1e-3 | log-uniform [1e-4, 1e-2] |
| weight_decay | 1e-4 | log-uniform [1e-5, 1e-3] |
| dropout | 0.3 | {0.1, 0.2, 0.3, 0.5} |
| hidden_channels | 64 | {32, 64, 128} |
| num_layers | 2 | {1, 2, 3} |
| heads（GAT gat_heads / L6 self-attn / L7 per-relation 共享）| 4 | {2, 4, 8} |

LGB（中心=`LGB_HPARAMS`；6 维以匹配 NN 维数、保等预算）：
| 维度 | 中心(pilot) | 搜索范围 |
|---|---|---|
| num_leaves | 31 | {15, 31, 63, 127} |
| learning_rate | 0.05 | log-uniform [0.01, 0.1] |
| min_data_in_leaf | 20 | {10, 20, 50, 100} |
| n_estimators | 100 | early-stop（不固定网格）|
| lambda_l1 | 1e-8 | log-uniform [1e-8, 1.0] |
| lambda_l2 | 1e-8 | log-uniform [1e-8, 1.0] |

- **固定不调**（NN/GAT/HATS）：epochs=100、patience=15、grad_accum=32（early-stop 控制）；HATS 结构 num_relations=3、rel_attn_arch=linear_shared（定义 L7 本身，非超参）。
- **L6 self-attn heads 在表中单列**，不得与 GAT 的 gat_heads 混（守 Cn1「L6≈GAT，差异仅在 mask」边界）。
- 偏离 v2-frozen 三处（见 §11 v2.2 行）：dropout 连续[0,0.5]→离散；LGB 8维(原含 feature_fraction/bagging_fraction)→6维(去二者、补 lambda_l1/l2) 以匹配 NN；hidden {64,128,256}→{32,64,128}（对称、中心=64=pilot 默认，防 hidden 卡底档 confound 模型对比）。

## 5. 训练协议（C1 落地）

- 全局 cold-start；early-stopping 沿 anchor 现行实现，全臂同值。
- **import-only 铁律**：12-fold runner、各臂、L6 一律 `import build_universe_C / nyse_session_close_utc / 冻结快照 helper`；**禁止重新加载原始 npy、禁止自写 cutoff、数据/边/快照构造逻辑零重写**。
- **运行时 assert（每 fold 构建时触发）**：(a) Univ-C `a158_slice[1]==raw[0] 且 row0==0`（T-1 shift 在位）；(b) 新闻边 `max(pub_ts) <= session_close(t-1)`。
- 每 cell 落盘三件套（日 IC .npy / 构造×成本档日收益 / fold 汇总）；results.csv 严格保持 E1 RESULTS_COLUMNS schema + HATS 7 诊断列；统计层只读不回训。

## 6. 统计层

- **唯一 confirmatory 家族 = 主轴**：DM-HLN 对子表（预注册，不许加对）= 阶梯五对（L1−L0, L2−L1, L6−L2, L7−L2, L2s−L2）+ 边 DAG 五对（L3−L2, L4−L2, L5−L2, L5−L4, L5−L3）× 2 universe = 20 检验，BH-FDR q=0.05。
- Hansen SPA：仅主轴，每 universe 一次，候选 M=9（含 HATS；本家族整体预注册，旧 E1 家族在 ledger 单列）。
- **L7 contingency（Cn5，机械规则、只触发于健康诊断、开跑前锁定）**：240 cells 中 (a) 任一 cell_id 注入 assert 失败，或 (b) >**20%** cells 发散（max-epoch 无 val 改善 / IC=NaN），或 (c) >**20%** cells α 塌缩（max_frac_collapsed>0.9）→ HATS 整臂降 exploratory，移出对子表与 SPA（M=9→8），ledger 留痕含触发器。A-11 uniform-α 扩展规则照旧。
- block-bootstrap CI（block=21, n_boot=5000）；LOFO 全 12 fold 扫描 + 单 fold 贡献重算。
- **MDE（M2 替换）**：MDE ≈ 2.8 × SE_block-bootstrap(mean ΔIC)（复用 stationary_bootstrap_ci），同时报告 n_eff；√750 公式废除。正文写明定量含义：**本设计可探测模型级差距（~0.025–0.03 量级在边缘），不可探测观测到的边级增量（+0.006–0.009）**——"fail-to-reject ≠ no effect" 的定量版，主动披露。
- 副轴：见 §2b 报告规范（无推断）。
- multi-testing ledger：新增 12-fold confirmatory 块；旧 E1/E3/E4 与 sanity 归 historical/exploratory。
- 附录三明治：E2 ≤ 真实边 ≤ E3 上限；oracle/泄漏数字禁入主表。

## 7. 组合层

LS-decile（主）、top-50、rank-weighted；min-variance 不做；成本阶梯沿 compute_e6_dm_spa 现行档位（预检 #8 核口径）。

## 8. 算力账（实测单价；M3 驳回复用 → 全新跑，预算不变）

| 块 | cells | 单价 | 小计 |
|---|---|---|---|
| 主轴标准神经（8×12×10×2） | 1920 | 60–90s | 1.5–2 天 |
| 主轴 L7 HATS（12×10×2） | 240 | 4–5min | ~0.8 天 |
| 副轴神经（4×14×10×2，含 case study 季） | 1120 | 30–50s | ~0.6 天 |
| 调参（20×30 + top-5 复核） | — | 1–3min/trial | 0.7–1 天 |
| LGB 双轴 | ~520 | CPU | 忽略 |
| **合计** | | | **≈ 3.5–4.5 A100·天**（本地 MPS 后备） |

## 9. 并行批次与写作

- 挂机：E2 补 canonical 10 seed（TOST）；E1b 补 10 seed + fold 分解。
- 写作：HATS preprint 措辞；related work 已发表锚点（ADB-TRM/GKX/RSR/STHAN-SR/AD-GAT/MAN-SF/DGATS/LSR-IGRU/MCI-GRU/THGNN/MASTER；Wade/FinMamba=concurrent preprint 引而不锚）；机制叙事（共线 descriptive/hub 候选/密度反例/"平滑伤排序"=H-pending）；Limitations 八条（正式家族无 2022 熊市→case study 部分修复；早 fold 幸存偏差；fold1 薄+fold 序混淆；超参 regime 错配；副轴 3 季训练；sector 单快照 PIT；图新鲜度未作变量；**MDE 不可探测边级增量**）。
- caption/数字 sweep grep 清单照 v2-frozen（重跑后执行）。

## 10. 预检清单

1. ☑ Touchpoint 1 完成（九条 disposition 并入本版）
2. ☐ Cn4：`grep -E 'ROC|STD|MA|RSV|rolling' build_alpha158_features.py` 确认最长回看 ≤106d，关 #2 尾
3. ☐ smoke：每臂 1 fold × 1 seed（主轴 fold12 最大训练集 + 副轴各一），单价回填 §8
4. ☐ L6 评审：参数量对齐报告 + 与 GAT 臂 diff 仅 mask 一处
5. ☐ L7：cell_id 域重映射 + 注入 assert + contingency 触发器实现（§6 规则的代码化）
6. ☐ seed 池互斥核对（canonical 10 / 调参 3 / sanity 批次）
7. ☐ **C1 验收**：新 runner diff 零数据构造代码 + 两条运行时 assert 在位 + **E0-canary-on-new-runner**（置换/off-by-1 负测试必 FAIL）
8. ☐ 成本口径核对（§7 vs 实现）
9. ☐ 旧 5-fold 归档"旧超参 pilot，不入主表"；ledger 新增 confirmatory 块
10. ☐ sliding case-study 输出带泄漏标注字段

## 11. 修订日志

| 日期 | 版本 | 修改 | 理由 |
|---|---|---|---|
| 2026-06-11 | v2-draft → v2-freeze-pending | 双轴设计，S1–S3 待签 | 讨论 |
| 2026-06-12 | v2-frozen | S1–S5 决议 + progress.md 校正（实测成本/canonical seed/q=0.05/schema 约束） | H博士决议 |
| 2026-06-12 | **v2.1-frozen** | T1 九条处置：C1 import-only+asserts+canary；M1 副轴降 robustness（同号判据、无推断、逃生口封条）；M2 MDE 改 block-bootstrap SE+n_eff+定量披露句；M3 标签双列化、**驳回复用旧 anchor**；Cn1 L6 claim 收窄 dense-only；Cn2 并入 M1；Cn3 重叠注明+regime 标签改"压力段"；Cn4 grep 预检项；Cn5 HATS contingency 20% 机械规则 | Codex T1 + H博士确认 |
| 2026-06-15 | **v2.2-frozen** | §4 搜索空间**显式冻结**（此前仅"同 v2-frozen 版"空指针、实体从未落档 → 本次踩坑根因）：6 维 NN/GAT/L6/L7 + 6 维 LGB，每维**中心=pilot 默认值**（`NN_HPARAMS`/`LGB_HPARAMS`）使 pilot 成 N=1 中心点、保 pilot-vs-调参可对比。三处偏离 v2-frozen：(1) dropout 连续[0,0.5]→离散{0.1,0.2,0.3,0.5}（N=30 预算不配连续）；(2) LGB 8维→6维（去 feature_fraction/bagging_fraction、补 lambda_l1/l2）匹配 NN 维数+保等预算（thesis 防护：基线不可因少调维显弱）；(3) hidden {64,128,256}→{32,64,128}（对称、中心 64=pilot 默认，防 hidden 卡底档 confound「图 vs 无图」对比）| Codex 顾问 + H博士确认 |
