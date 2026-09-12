---
reviewer: codex
touchpoint: plan
round: T1
target_files:
  - protocol_v2_freeze.md
target_plan: docs/protocol_v2_freeze.md
findings:
  - id: C1
    severity: CRITICAL
    category: data-leakage
    claim: "大重构是 same-day-leak 回归入口；schema diff 抓不到 T-1 shift / PIT cutoff 被绕过"
    evidence: "12-fold runner 若重写数据/边构造，可能引入 same-day 泄漏，schema diff 无法捕获"
    suggested_fix: "§5 import-only 铁律 + 两条 per-fold 运行时 assert + E0-canary-on-new-runner"
    status: FIXED
    resolution_notes: "§5 落地：禁重载 npy/自写 cutoff/重写构造；assert (a) a158_slice[1]==raw[0]&row0==0 (b) max(pub_ts)<=session_close(t-1)；预检 #7 升级 schema diff + 置换/off-by-1 负测试"
  - id: M1
    severity: MAJOR
    category: statistics
    claim: "双 confirmatory family 并列 BH-FDR = 双重捞鱼，整体 FDR>0.05"
    evidence: "主轴 + 副轴 两套同时做 BH-FDR 会膨胀整体 FDR"
    suggested_fix: "主轴=唯一 confirmatory；副轴降 robustness（仅 ΔIC 符号 + bootstrap CI）"
    status: FIXED
    resolution_notes: "选项 a 采纳：副轴无 p/星号/FDR/SPA，判据=与主轴点估计同号，报 X/8 计数，逃生口封条（副轴模式只能 exploratory，不进摘要/claim）。选项 b/c 弃"
  - id: M2
    severity: MAJOR
    category: statistics
    claim: "MDE 的 √750 忽略 21d 标签重叠自相关，功效虚高，与 block=21 bootstrap 自相矛盾"
    evidence: "21d-forward 标签重叠 → 有效样本 < 750；√750 SE 低估"
    suggested_fix: "MDE := 2.8×SE_block-bootstrap(mean ΔIC) + 报 n_eff；√750 废除"
    status: FIXED
    resolution_notes: "复用 stationary_bootstrap_ci。正文定量披露：可探测模型级差距(~0.025–0.03 边缘)，不可探测边级增量(+0.006–0.009)。DM-HLN 已 HAC，推断不受影响"
  - id: M3
    severity: MAJOR
    category: correctness
    claim: "(i) 现成 混淆代码/结果就绪度；(ii) 建议复用 anchor 旧 5-fold 的 L0/L1/L2 结果省 ~40%"
    evidence: "主表若混旧超参 5-fold + 新超参 12-fold = 秒杀级硬伤"
    suggested_fix: "(i) §3 拆代码状态/结果状态双列；(ii) 复用 → 全新跑"
    status: FIXED
    resolution_notes: "PARTIAL-REJECT。(i) FIXED：双列化，L3/L4 注明配置先例仅 SAGE/Univ-B/5-fold。(ii) REJECTED：与 D-RERUN-12F 冲突（H博士不用管钱必须调选b）；省额实际 <0.3 A100·天；旧 5-fold 维持 pilot/smoke 对照"
  - id: Cn1
    severity: CONCERN
    category: prior-art
    claim: "L6 完全图注意力 ≠ learned-sparse-graph，勿充当 测了 learned graph"
    evidence: "L6 = full-attention-no-graph；learned-sparse 是另一族"
    suggested_fix: "claim 收窄"
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "ACCEPTED-NARROWED：AD-GAT 本身 unmasked 全连接，L6 引 AD-GAT 成立；claim_scope 改 dense learned attention 家族(MASTER/AD-GAT)；learned-sparse(FinMamba/ADB-TRM) 入 Limitations/future work"
  - id: Cn2
    severity: CONCERN
    category: statistics
    claim: "副轴 252d 扣 val 仅 3 季有效训练，偏噪"
    evidence: "sliding-252d 窗内扣 val 后有效训练仅 3 季"
    suggested_fix: "Limitations 注明"
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "并入 M1：副轴已降方向性 robustness，与薄训练角色匹配；Limitations 条目保留"
  - id: Cn3
    severity: CONCERN
    category: other
    claim: "调参 val 与 fold-1 训练重叠应注明；牛市 标签不准"
    evidence: "2022H2 为混合段，任何单一 regime 标签均不准；reviewer 误引 牛市，v2-frozen §1 原文 熊市"
    suggested_fix: "重叠一句注明（GKX 先例）；regime 标签中性化"
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "重叠=GKX 标准做法注明（非测试泄漏）；regime 标签改 2022H2 压力段（回撤+Q4 反弹混合 regime）"
  - id: Cn4
    severity: CONCERN
    category: correctness
    claim: "Alpha158 子集最长回看 ≤106d 未逐条确认"
    evidence: "106d=burn-in 预算阈值；60d=预期实测 max（Qlib 滚动窗族 {5,10,20,30,60}）"
    suggested_fix: "grep build_alpha158_features.py 确认"
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "预检 #2 新增 grep；判定=实测 max ≤106；预期 PASS（裕度 ~46 交易日），以 grep 为准"
  - id: Cn5
    severity: CONCERN
    category: statistics
    claim: "HATS ~30-40% 风险流入 SPA(M=9)，contingency 须开跑前定"
    evidence: "HATS 训练不稳风险若流入 SPA 候选会污染"
    suggested_fix: "开跑前锁定机械 contingency 规则"
    status: ACCEPTED-AS-CONCERN
    resolution_notes: "§6 机械规则（只触发于健康诊断）：任一 cell_id assert 失败 / >20% cells 发散 / >20% α 塌缩(max_frac>0.9) → 整臂降 exploratory，移出对子表+SPA(M=9→8)，ledger 留痕"
summary:
  critical: 1
  major: 3
  concern: 5
  fixed_before_reply: 8
overall_verdict: PASS-WITH-CONCERNS
---

# Disposition Record — Touchpoint 1 (Plan) on 实验协议 v2 → v2.1

> 落档说明（2026-06-12）：本文件为 Rule 9 Touchpoint 1 disposition 的 canonical 副本，对应冻结协议 `docs/protocol_v2_freeze.md`。源原件在 H博士 桌面，逐字落盘。frontmatter 为机械可审 schema（`.claude/rules/docs.md §6`），正文为 H博士确认的原始 disposition。

- Target: protocol_v2_freeze.md (v2-frozen → v2.1-frozen)
- Review date: 2026-06-12 | Disposition date: 2026-06-12（H博士确认：M3 驳回复用、Cn5 阈值 20%、M1 判据=同号）
- Summary: 1 CRITICAL + 3 MAJOR + 5 CONCERN → 8 ACCEPTED(含 1 收窄) + 1 PARTIAL-REJECT(M3 后半)
- **引用规范（本次起生效）**：禁用裸字母选项引用（项目史中已存在 ≥2 套 (a)/(b)/(c) 选项表）；跨文档引用决议一律用决议 ID。已注册：**D-RERUN-12F** = 调参后 12-fold 全量于冻结新超参下重跑、旧 5-fold 降 pilot（源：H博士"不用管钱，必须调参，选b"决议）。
- Pre-credit（reviewer 已核干净项）：新闻边 PIT（nyse_session_close_utc, D-03）；Univ-C runtime T-1 shift（CR-A-01）；burn-in 126 满窗实跑 PASS；cold-start 对齐 anchor；FDR q=0.05；canonical seed；MDE 返正文。

| ID | Sev | Finding | Disposition | Resolution |
|---|---|---|---|---|
| C1 | CRITICAL | 大重构是 same-day-leak 回归入口；schema diff 抓不到 T-1 shift / PIT cutoff 被绕过 | **FIXED** | §5 import-only 铁律（禁重载 npy/自写 cutoff/重写数据构造）+ 两条 per-fold 运行时 assert（a158_slice[1]==raw[0]&row0==0；max(pub_ts)<=session_close(t-1)）+ 预检 #7 升级为 schema diff + E0-canary-on-new-runner（置换/off-by-1 负测试） |
| M1 | MAJOR | 双 confirmatory family 并列 BH-FDR = 双重捞鱼，整体 FDR>0.05 | **FIXED（选项 a）** | 主轴=唯一 confirmatory；副轴降 robustness：仅 ΔIC 符号+bootstrap CI，判据=与主轴点估计**同号**，报 X/8 计数；副轴无 p/星号/FDR/**SPA 一并删除**；逃生口封条（副轴模式只能 exploratory，不进摘要/claim）。选项 b(gatekeeping) 弃：复杂度不承重；选项 c(合并 28 检验) 弃：高相关检验同池稀释功效且家族定义不清 |
| M2 | MAJOR | MDE 的 √750 忽略 21d 标签重叠自相关，功效虚高，与 block=21 bootstrap 自相矛盾 | **FIXED** | MDE := 2.8×SE_block-bootstrap(mean ΔIC)（复用 stationary_bootstrap_ci）+ 报 n_eff；√750 废除。正文加定量披露：可探测模型级差距(~0.025–0.03 边缘)，不可探测边级增量(+0.006–0.009)。推断本身不受影响（DM-HLN 已 HAC） |
| M3 | MAJOR | (i)"现成"混淆代码/结果就绪度；(ii) 建议复用 anchor 旧 5-fold 的 L0/L1/L2 结果省 ~40% | **(i) FIXED / (ii) REJECTED** | (i) §3 拆"代码状态/结果状态"双列，L3/L4 注明配置先例仅 SAGE/Univ-B/5-fold。(ii) 驳回：与已锁决议 **D-RERUN-12F**（H博士"不用管钱，必须调参，选b"——调参后 12-fold 全量于冻结新超参下重跑；该 (b) 出自复用-vs-重跑选项表，与本表 M1 的 family 选项 a/b/c 无关）冲突——主表混"旧超参 5 fold+新超参 7 fold"=秒杀级硬伤；且省额实际 <0.3 A100·天（L0 CPU 秒级、L2 共 240 cells≈4.7h）。旧结果维持 pilot/smoke 对照 |
| Cn1 | CONCERN | L6 完全图注意力 ≠ learned-sparse-graph，勿充当"测了 learned graph" | **ACCEPTED-NARROWED** | 收窄而非全收：AD-GAT 本身即 unmasked 全连接学习注意力（原文核过），L6 引 AD-GAT 成立；claim_scope 改为"dense learned attention 家族（MASTER/AD-GAT）"，learned-sparse（FinMamba 剪枝/ADB-TRM 自适应）入 Limitations/future work |
| Cn2 | CONCERN | 副轴 252d 扣 val 仅 3 季有效训练，偏噪 | **ACCEPTED（并入 M1）** | 副轴已降方向性 robustness，与薄训练角色匹配；Limitations 条目保留 |
| Cn3 | CONCERN | 调参 val 与 fold-1 训练重叠应注明；"牛市"标签不准 | **ACCEPTED** | 重叠=GKX 标准做法一句注明（非测试泄漏）；regime 标签改"2022H2 压力段（回撤+Q4 反弹混合 regime）"。注记（事实链）：reviewer 引述协议标签时误写为"牛市"——v2-frozen §1 原文为"熊市 case study"；该误引不影响其实质论点（2022H2 为混合段，任何单一 regime 标签均不准），故 relabel 为中性"压力段" |
| Cn4 | CONCERN | Alpha158 子集最长回看 ≤106d 未逐条确认 | **ACCEPTED** | 预检 #2 新增 grep build_alpha158_features.py 关尾。两数字角色不同：**106d=burn-in 预算（阈值）**；**60d=预期实测 max**（Qlib 标准 Alpha158 滚动窗族 {5,10,20,30,60}，51 维子集若不含 60d 窗因子则更小；T-1 shift +1d 在裕度内）。判定 = 实测 max ≤106；预期 PASS（裕度 ~46 交易日），以 grep 为准 |
| Cn5 | CONCERN | HATS ~30-40% 风险流入 SPA(M=9)，contingency 须开跑前定 | **ACCEPTED** | §6 机械规则锁定（只触发于健康诊断非性能）：任一 cell_id assert 失败 / >20% cells 发散 / >20% α 塌缩(max_frac>0.9) → 整臂降 exploratory，移出对子表+SPA(M→8)，ledger 留痕含触发器；A-11 uniform-α 规则照旧 |

## 协议落点对照

- §1 Cn3 标签+重叠注明 | §2b M1 全套 | §3 M3(i)+Cn1+Cn5 引用 | §5 C1 | §6 M1/M2/Cn5 | §8 M3(ii) 驳回注记 | §9 Limitations 第八条(MDE) | §10 预检 #2/#5/#7/#9 | §11 修订日志 v2.1
