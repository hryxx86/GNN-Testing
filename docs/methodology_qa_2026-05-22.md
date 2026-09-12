# Methodology Q&A + Session Walkthrough (2026-05-21 → 2026-05-22)

> H博士 在 2026-05-21 整理代码 + 2026-05-22 深度方法论 Q&A 的完整记录。按项目逻辑顺序组织（代码组织 → 整理过程 → 项目状态 → 方法论 Q&A → 发现的 limitation → 下一步行动）。
>
> **不要遗漏的细节**：Codex stop-hook 5 轮修正全程、所有 Q&A 答案、关键数据证据、Web research 引用、Plan Z++ universe limitation 的诚实诊断。

---

## Part 0：Session 启动 — Handoff 状态恢复

### 上次 (2026-05-20) handoff 核心状态

来源：`docs/session_handoff_2026-05-20.md`

- **2026-05-20-a 已完成**：10-seed expansion + finalize.v3 → Tier 1.D verdict **FLIP 为 NULL**
- **数据规模**：Phase A+B 2,604 cells + Stage 1 600 cells = **3,204 cells**；~43h M4 wall clock
- **累积统计 (Story C+ v3)**：**0/36 BH-FDR rejections** + 7 nulls + 3 mechanism findings + 1 regime-conditional finding
- **Tier 1.D 翻转**：5-seed h2 (AdamW lr=5e-4 wd=1e-3 patience=5) 看似 marginal positive (NW p=0.059)，10-seed 后 **全部 4 hparam configs Score-negative + p>0.5**
- **待办**：paper v3 重写（基于 paper_draft_2026-05-18_v2.md，2-4h）；Codex Touchpoint 2/3 仍 pending（quota 不可用）
- **Open questions**：paper 标题选择（3 备选）、Stage 1 整合策略、Sharpe 是否用 raw fwd_ret 重算、anchored RankNet σ-guard 的定位

---

## Part 1：代码整理 (2026-05-21-a) — 32 → 14 文件

### 触发

H博士 在 session 启动后立即要求"全面检查代码并整理规整，告诉我每一个都做了什么，有什么用"。根目录已积累 32 个 `.py` 文件，跨越 Phase 5 → Plan Z++ → Tier 1 → Phase B → 10-seed → Paper v2/v3 多阶段。

### Inventory 执行

并行 3 个 Explore agents 完成全 32 文件分类登记：

| 类别 | 数量 |
|------|------|
| Build/Data-Prep | 7 |
| Experiment Runner | 11 |
| Analysis Script | 8 |
| Diagnostic | 2 |
| Orchestration/Utility | 4 |

### Dependency Scan 关键发现

`grep -nE '^(from\|import) run_step3_plan_z_part_a' *.py` 显示该文件被 9 个脚本 import：
- **5 个 active root**：`run_tier1_phase_a.py`, `run_tier1a/b_h2/c_phase_b.py`, `run_loss_horserace.py`
- **4 个 plan_z 兄弟**（即将归档）：`run_step3_plan_z_part_b/c/c_perfold.py`, `smoke_test_part_a.py`

→ **`run_step3_plan_z_part_a.py` 是 shared library，不能归档**。它提供：
1. `load_data_and_features()` — 统一数据入口
2. `RankingGNN` / `RankingMLP` 模型类
3. `train_one()` 标准训练循环
4. `assert_graph_train_only()` 泄露护栏
5. `set_seed`, `fit_feature_scaler`, `apply_scaler`, `build_correlation_snapshots`, `build_sector_edges`

### H博士 决策（4 项）

通过 `AskUserQuestion` 收集，全选 Recommended：
1. ✅ `run_loss_horserace.py` + `analyze_loss_horserace.py` 保留在根目录（pending Codex Touchpoint 2 + paper v3 可能复跑）
2. ✅ `analyze_seed_diagnostic.py` 归档（数据已在 `experiments/loss_horserace/seed_diagnostic/`）
3. ✅ 根目录扁平结构，不分 `run/`, `analyze/`, `build/` 子目录
4. ✅ 立刻执行（paper v3 工作前完成）

### 归档执行

创建 `archived/scripts/2026-05-21/`（与现有 `archived/scripts/` 平铺约定一致，内部 README 分节）：

**Phase 5 legacy (7)**：
- `run_walkforward_5fold.py` — Phase 5 baseline 5-fold WF（被 Tier 1 取代）
- `run_diag1_normalization.py` — z-score 归一化消融
- `run_diag1b_replication.py` — 扩展 Diag 1 到 MLP + NoGraph
- `run_gate1_experiment.py` — SEC 10-K/10-Q Layer 1 test（Gate STOP）
- `run_phase5_step3_feature_expansion.py` — 9/6/13-dim 特征集消融
- `diagnostic_phase5_step0.py` — Fold-4 异常 + 特征重要性 + collinearity
- `diagnostic_phase5_fix.py` — 修复 \|corr\| bias + effective-rank speculation

**Plan Z++ completed (6)**：
- `run_step3_plan_z_part_b.py` — 7 subsets × 2 models × 3 seeds × 5 folds = 210 runs
- `run_step3_plan_z_part_c.py` — S8 Alpha158 baseline 30 runs
- `run_step3_plan_z_part_c_perfold.py` — Part C with per-fold winsor 复跑
- `smoke_test_part_a.py` — Part A 烟雾测试
- `analyze_step3_plan_z.py` — Hansen SPA + BH-FDR pairwise
- `analyze_fold4_leakage.py` — Fold-4 tail concentration + z-drift

**One-off fixes (5)**：
- `refetch_zts.py` — ZTS ticker 补采（2026-04-17 batch gap）
- `cleanup_and_rebuild_features.py` — Parquet schema 修复
- `run_figures_tables.py` — Phase 5 era figure 生成
- `make_advisor_figures.py` — 2026-04-21 advisor 12 PNG
- `analyze_seed_diagnostic.py` — Stage 1 descriptive seed 分析

**总计**：18 files 归档；根目录 32 → 14 (56% 减少)。

### Active 14 个文件最终清单

```
Active runners (5):    run_tier1_phase_a, run_tier1a_phase_b, run_tier1b_h2_phase_b,
                        run_tier1c_phase_b, run_loss_horserace
Active analyzers (5):  analyze_tier1_phase_a, analyze_phase_b_finalize,
                        analyze_tier1e_regime_forensic, analyze_tier2c_sector_ic,
                        analyze_loss_horserace
Active data prep (3):  build_alpha158_features, build_phase5_features, download_ohlcv_yf
Shared library (1):    run_step3_plan_z_part_a
```

---

## Part 2：Codex Stop-Hook 5 轮修正（关键诚信记录）

H博士 在归档完成后，Codex stop-hook 触发 **5 轮 blocking errors**，每一轮抓住一个真实缺陷。这部分是这次会话最有教育价值的一段——展示了"声称完成"和"实际正确"的差距。

### Round 1：active docs 中存在 stale 路径引用

**Codex 反馈**："archived script moves left active docs pointing at missing root paths"

修复 4 个 active README（path 修复）：
- `README.md` — 重写"根目录脚本/Notebook"节：27 个 → 14 active + 1 shared library + 18 归档
- `utils/README.md:30-31` — 拆分 "active import 者" vs "历史 import 者（已归档）"
- `experiments/step3_plan_z/README.md:45-48` — part_b/c/c_perfold + analyze_step3 + analyze_fold4 标注 "已归档 2026-05-21"
- `artifacts/README.md:32` — 三个引用脚本改为完整归档路径
- `plan.md` 顶部加 archive pointer note（历史 plan 条目引用的脚本说明已归档）

### Round 2：事实陈述错误

**Codex 反馈**："edited docs still contain stale/incorrect archive references"

精确 grep 暴露两个错误：

#### 错误 1：part_a importer 数量

| 之前声称 | 实际事实 |
|---------|---------|
| "9 个 active script 依赖 part_a" | **5 个 active**：4 个 Tier 1 runner + `run_loss_horserace.py`。另外 4 个声称依赖的脚本（part_b/c/c_perfold + smoke_test_part_a）自己就在 archived 目录，不算 active dependency |

#### 错误 2：utils/ 的 active importers

| 之前声称 | 实际事实 |
|---------|---------|
| `utils/README.md` 列了 `run_step3_plan_z_part_a` + 4 Tier 1 + loss_horserace + 3 analyzers 为 "Active import 者" | **全部错误**：精确 grep `^(from utils\|import utils)` 显示**零** active 用户。`analyze_tier1_phase_a.py` 自己 inline 实现了 NW-HAC + bootstrap + BH-FDR；`analyze_phase_b_finalize.py` 从 `analyze_tier1_phase_a` import 而非 utils |

**这是个意外发现**：`utils/` 是隐性 dead module。当前只在 archived 脚本里被 import。

修复：4 处文档统一为 "5 active"；utils/README.md 诚实写 "Active import 者 → 当前无"。

### Round 3：progress.md 验证命令不可重跑

**Codex 反馈**："progress.md documents non-reproducible verification commands"

具体问题：progress.md 写 "grep 列出 9 处依赖 → 全部仍在根目录 ✓"，但归档后重跑会得到只 5 处（4 处已不在根目录），命令 + 结果不再一致。

修复：将 verification 节重写为 6 个带 ` ```bash ` 代码块的精确命令 + expected stdout 注释，每个命令在当前 repo 状态下可重跑。

### Round 4：stdout 注释非字面一致

**Codex 反馈**："verification comments are not exact rerunnable stdout"

之前 verification 注释是 paraphrase 风格（"5 hits"、`(4 Tier 1 + loss_horserace)`），不是命令的字面 stdout。

修复：实跑 6 个命令，把每个命令的**逐字符 stdout**贴进 progress.md。包括：
- 完整 `path:line_num:source_line` 格式
- `wc -l` 默认输出的 leading whitespace（`      19` / `      18`）
- Python script 的精确 stdout (`14/14 OK`)
- 空 stdout 的精确注释（`# (no stdout; grep exit code 1)`）

### Round 5：active runner 仍 import 归档模块（真实破坏）

**Codex 反馈**："Active runner imports an archived module"

**Codex 抓到一个真实 bug**：[run_loss_horserace.py:708](../run_loss_horserace.py#L708)（**indented** import 在 `load_s6_features()` 函数体内）：

```python
def load_s6_features(data: dict) -> tuple[np.ndarray, list[str]]:
    from run_step3_plan_z_part_b import build_full_feature_universe, select_subset_tensor
    ...
```

`run_step3_plan_z_part_b.py` 在 2026-05-21-a 已归档 → `load_s6_features()` 会在 Stage 1 跑 S6 时 `ModuleNotFoundError` 崩溃。

**为什么前 4 轮没抓到**：之前 dependency scan 用 `^(from|import)` **anchored** grep，只检测顶层 import，**漏掉了 indented imports**（lazy import 在函数体内）。

**修复 + Lesson**：
1. 删除第 708 行 dead import（静态分析证明两个 symbol 在函数体内零调用；comment 本身已注明 "instead we replicate its logic using `data` already loaded"）
2. 更新 comment 注明 part_b 已归档
3. 修正 progress.md Cmd 3 改为 unanchored 形式
4. 新增 [progress.md 2026-05-21-b](../progress.md) 条目记录此缺陷
5. **Lesson**: archive 操作的 dependency scan 必须用 **unanchored** regex 才能抓 indented/conditional imports

### Codex 5 轮总结

| Round | 抓到的问题 | 严重性 |
|-------|----------|--------|
| 1 | active docs 路径未更新 | Medium |
| 2 | 事实数字错误（9 vs 5） | High（误导后人） |
| 3 | verification 命令不可重跑 | Medium |
| 4 | stdout 注释非字面一致 | Low（但 reproducibility 标准） |
| 5 | **active runner 真实 broken import** | **Critical**（Stage 1 会崩溃） |

**核心教训**：anchored regex 的盲区在 indented imports。这次 Codex 帮项目避免了一个 Stage 1 复跑时的运行时错误。

---

## Part 3：项目当前状态（2026-05-22）

### 实验数据层（FROZEN）

| 实验 | Cells | 主要结果 |
|------|-------|---------|
| Stage 1 (loss horse race) | 600 | 0/8 BH-FDR rejection (4 losses × 2 models / co-primary) |
| Tier 1.B Adam | 400 | 0/12 BH-FDR rejection (robust losses null) |
| Tier 1.B h2 | 800 | 0/12 BH-FDR rejection（h2 baseline cross-validation） |
| Tier 1.A | 200 | rolling 2y ListMLE fold-4 attenuation NW p<0.001 (regime-conditional) |
| Tier 1.C | 400 | 0/4 anchored RankNet σ-guard 失败（mechanism finding） |
| Tier 1.D | 280 | 10-seed FULL NULL（5-seed marginal positive 被推翻） |

**累积**：3,204 cells = 600 + 2,604；**0/36 BH-FDR rejections** + 7 nulls + 3 mechanism findings + 1 regime-conditional finding。

### 代码层（刚整理）

- 根目录 `.py`: 32 → 14
- `archived/scripts/2026-05-21/` 收纳 18 legacy
- `utils/` dormant（如 paper v3 需复用 stats_tests，需重新 wire）
- 14/14 active scripts AST parse 通过
- 零 active 脚本依赖 archived files（含 indented imports，2026-05-21-b 修复后）

### 待办（next session）

1. **Paper v3 重写**（最高优先级，2-4h）：摘要 + §1.2 + §4.5/§4.7/§6.3 + §7 Limitations + 所有数值表 → 10-seed
2. **Codex Touchpoint 2 + 3**（quota 不可用，fallback claude-self-review）
3. **本会话发现的 limitation 修复**（Plan Z++ feature universe，见 Part 9）

---

## Part 4：Q&A — Permutation Tests 的作用与方法

### 作用

回答的问题：**"我的模型 IC=0.05，这个数字是真的有信号，还是随机也能跑出来？"**

把真实 labels **随机打乱** N 次重新算 IC，得到 "null distribution"。如果实际 IC 落在 null 的 99% 分位以上，就有信心说模型确实学到了 cross-sectional 信号。

### 怎么做（per-day cross-sectional shuffle）

```python
real_ic = compute_daily_ic(predictions, true_labels)   # → 比如 0.0436

null_ics = []
for shuffle_iter in range(16000):                       # 16K iterations
    shuffled_labels = []
    for day in trading_days:
        labels_today = true_labels[day]                 # 这一天 ~500 只股票的 label
        perm = rng.permutation(len(labels_today))       # 只在"截面"打乱
        shuffled_labels.append(labels_today[perm])
    null_ic = compute_daily_ic(predictions, shuffled_labels)
    null_ics.append(null_ic)

p_value = (sum(null_ics >= real_ic) + 1) / 16001       # one-sided
```

### 关键设计选择

| 选择 | 为什么 |
|------|-------|
| **只在 cross-section 打乱**，不跨天 | 保留时间序列自相关；只破坏 "predictions 与 labels 截面对应关系" |
| **16K iterations** (v2，v1 是 8K) | p-value 分辨率到 1/16001 ≈ 6.2e-5，支撑 p<0.001 声明 |
| **predictions 不动，只动 labels** | 测的是 "model output 是否包含真信号" |
| **复用 cached preds** | 16K × 5 folds × 6 models = 480K IC 计算，不重训 |

### 实际结果（experiments/permutation_v2_results.csv）

| Model | Feature | real_IC | p_value | 显著 (α=0.001) |
|-------|---------|---------|---------|---------------|
| MLP | all (158-dim) | **-0.010** | 1.000 | ❌ 反向 |
| MLP | price (9-dim) | **+0.034** | **0.000** | ✅ |
| SAGE-Mean | all (158-dim) | **+0.008** | 0.002 | ✅ (边缘) |
| SAGE-Mean | price (9-dim) | **+0.033** | **0.000** | ✅ |

### 核心洞察

1. **price 子集 (9-dim) 显著优于 random** — IC 是 null std 的 ~13 倍
2. **Alpha158 全集反而失效** — MLP_all 反向，SAGE_all 边缘显著但 IC 仅 +0.008
3. **"越多越好" 不成立** → 直接催生 Plan Z++ "pre-registered subset selection"

---

## Part 5：Q&A — Runner 概念 + 两个模型设计 + 参数调优

### Runner 定义

**Runner = 跑训练循环、产 `preds/*.npy` + `results.csv` 的脚本**，不做统计。命名约定 `run_*.py`。

### 标准结构

```python
# 1. 配置 (固定 hparams, seed list, fold list, model list, loss list)
SEEDS = [86, 123, 456, 789, 1024, 2024, 7, 34, 99, 2026]
FOLDS = list(range(5))

# 2. 载入数据 (import shared library)
import run_step3_plan_z_part_a as pa
data = pa.load_data_and_features()

# 3. 主循环 —— 每个 (model, loss, feat, fold, seed) 组合 = 1 cell
for fold, model, loss, seed in itertools.product(...):
    result = pa.train_one(...)
    np.save(f'preds/...', result['test_preds'])

# 4. 保存 results.csv + log
```

### Runner ↔ Analyzer 分工

```
Runner (run_tier1_phase_a.py)         Analyzer (analyze_tier1_phase_a.py)
├─ 输入：data + manifest                ├─ 输入：preds/*.npy + manifest
├─ 跑：680 cells（GPU 数小时）          ├─ 算：NW-HAC + bootstrap + BH-FDR（CPU 数分钟）
└─ 输出：preds/*.npy + results.csv      └─ 输出：stat_per_cell.csv + stat_report.md
```

**为什么分开**：改了统计方法不必重训模型。

### 两个模型详解（run_step3_plan_z_part_a.py:270-313）

#### RankingGNN（图神经网络）

```python
class RankingGNN(nn.Module):
    def __init__(self, in_ch, hidden=64, num_layers=2, dropout=0.3):
        self.lin = nn.Linear(in_ch, hidden)              # 输入投影
        self.convs = [SAGEConv(hidden, hidden, aggr='mean') × 2]
        self.norms = [LayerNorm(hidden) × 2]
        self.head = Linear(hidden, hidden//2) → ReLU → Dropout → Linear(hidden//2, 1)

    def forward(self, x, edge_index):
        h = ReLU(self.lin(x))
        for conv, norm in zip(convs, norms):
            h = norm(Dropout(conv(h, edge_index)) + h)   # ← residual connection
        return self.head(h).squeeze(-1)
```

#### RankingMLP（无图对照）

```python
class RankingMLP(nn.Module):
    # 完全一样的结构，唯一差异：把 SAGEConv 换成普通 Linear，不接 edge_index
    def __init__(self, in_ch, hidden=64, num_layers=2, dropout=0.3):
        self.lin = Linear(in_ch, hidden)
        self.layers = [Linear(hidden, hidden) × 2]       # ← 不是 SAGEConv
        ...
```

### 为什么是这两个

| 设计原则 | 理由 |
|---------|------|
| **同参数量 + 同深度 + 同 dropout** | 控制变量：MLP vs GNN 唯一差异是 "有没有用图"，干净测出图的边际价值 |
| **SAGEConv with `aggr='mean'`** | Week 3 ablation 比过 GAT / SAGE-Sum / TransformerConv，mean 聚合最稳；GAT 在 fold-4 collapse 严重 |
| **2 层** | 浅层够用（cross-sectional 信号不需远程信息）；深层 over-smoothing |
| **Residual + LayerNorm** | Codex Round 4 加的；之前训练后期梯度不稳，IC 在 fold-4 抖动大 |
| **Head 是 2-layer MLP** | 给 representation 一些 capacity 转成 ranking score；单 linear 表达力不足 |

### 参数调过吗 — 怎么调

**调过两次**：

#### Stage 0 (2026-04-28): grid search 选 default

`lr ∈ {1e-3, 5e-4, 1e-4}` × `wd ∈ {1e-4, 1e-3}` × `dropout ∈ {0.2, 0.3, 0.5}`，按 val IC 选 winner：

```python
HPARAMS_DEFAULT = dict(
    hidden=64, num_layers=2, dropout=0.3,
    lr=1e-3, weight_decay=1e-4,
    epochs=50, patience=10, grad_accum=4,
    optimizer='adam',
)
```

这套 **锁定**（Plan Z++ §1.B），所有 Stage 1 + Tier 1.B + Tier 1.A + Tier 1.C 都用它。

#### Tier 1.D (2026-05-18): hparam sweep 测稳健性

```python
TIER1D_HPARAM_GRID = [
    dict(weight_decay=3e-4, lr=5e-4, patience=5, optimizer='adamw'),
    dict(weight_decay=3e-4, lr=2e-4, patience=5, optimizer='adamw'),
    dict(weight_decay=1e-3, lr=5e-4, patience=5, optimizer='adamw'),  # ← "h2"
    dict(weight_decay=1e-3, lr=2e-4, patience=5, optimizer='adamw'),
]
```

**关键发现**：5-seed h2 看似 marginal positive，10-seed 全 NULL → 推翻"正则化能救场"假说。

### 未调的硬约束

`hidden=64` / `num_layers=2` / SAGEConv aggr `'mean'` 没改。进 paper v3 "Limitations" 节。

---

## Part 6：Q&A — Winsorization + 全 Panel 拟合的 Leakage 风险

### Winsor (winsorization) 是什么

把极端值"按比例剪到合理范围"。给定一组数：
```
原始: [-100, -5, -3, -2, 0, 1, 2, 3, 4, 8, 200, 500]
```
计算 1%/99% 分位数（-50 和 100），**剪裁**：
```
winsorized: [-50, -5, -3, -2, 0, 1, 2, 3, 4, 8, 100, 100]
```
和"删除"不同，winsor 是**把极端值替换为分位数边界**，保持样本量不变。对金融数据特别重要——一个 200% 单日涨幅会撑爆 z-score。

### "Bounds 在全 Panel 拟合" 是什么意思

**Panel** = 5y × 501 stocks × 158 features（30 亿数据点）。

**全 panel 拟合（错误，Phase 5 之前在用）**：
```python
for f in range(158):
    p1, p99 = np.percentile(features_np[:, :, f], [1, 99])   # ← 用了所有日子
    features_winsorized[:, :, f] = np.clip(features_np[:, :, f], p1, p99)
```
**问题**：计算 p1/p99 时包含了 val 和 test 的日子。训练时模型看到的特征已经被"未来信息"影响过。

**Per-fold train-only winsor（正确，Plan Z++ 之后强制）** — [run_tier1_phase_a.py:132-151](../run_tier1_phase_a.py#L132-L151)：
```python
def per_fold_winsorize(raw, train_days, q_lo=0.01, q_hi=0.99):
    bounds = np.zeros((F, 2))
    for f in range(F):
        train_slice = raw[train_days, :, f]                  # ← 只看 train
        lo, hi = np.percentile(train_slice, [1, 99])         # bounds 仅基于 train
        bounds[f] = (lo, hi)
        out[:, :, f] = np.clip(raw[:, :, f], lo, hi)         # 用 train bounds 剪 full panel
    return out, bounds
```

关键差异：
1. 分位数计算**只用 `train_days` 切片**
2. 每 fold 独立做
3. 应用时把 train bounds 用在 full panel 上（test 的极端值会被剪掉，但 train 的边界没看 test）

### 影响有多大

Paper v2 §1.2 提到："we catch and fix a global p1/p99 winsorization in the Alpha158 feature builder that had quietly contaminated prior baselines"。Pre-fix S8 跑出 fold-4 IC = +0.20，post-fix 降到 +0.05 ~ +0.08，说明 +0.15 的"信号"其实来自 leakage。

---

## Part 7：Q&A — Sentinel Leakage Test 详解 + Novelty Research

### 重新解释（一图流）

```
真实数据 → [Pipeline] → 训练产物 A
   ↓ 故意扰动 val 边界之后的所有数据 (σ=1e-3 高斯噪声)
扰动数据 → [Pipeline] → 训练产物 B

断言：A == B (字节级别完全相等)
  ├─ 通过 → pipeline 在训练时确实没碰 val/test 数据 ✓
  └─ 失败 → 有 leakage
```

### 为什么这么设计

**核心思想**：把 "code review + 程序员声称没用 future data" 这个**主观判断**变成**客观可验证的机器断言**。

1. **"假因果反演"**：如果 train 之后的事件能影响 train 输出，那就有 leakage。Sentinel 故意制造一个 "假事件"（val 时点之后的噪声），看 train 输出是否真的不变
2. **σ=1e-3 故意微小**：用真实噪声会被淹没；微扰动 + winsor 的 max/min 操作会把任何 leakage 放大成可检测的数值差异
3. **8 个 artifact 全部比对**：覆盖 winsor bounds / scaler / labels / graph snapshots → 一个都不能漏
4. **必须配 control pipeline 一起跑**：如果只测 Plan Z++ pipeline 10/10 PASS，无法证明 sentinel 本身有诊断力。我们保留旧 global-winsor pipeline 作为对照 — 它 **10/10 FAIL**，证明 sentinel 真能抓 leakage

### 实际结果（artifacts/audits/sentinel_leakage_test.md）

```yaml
test_method: perturb prices/features at indices >= min(val_days),
             recompute train artifacts, assert bitwise equality
perturb_scale_sigma: 0.001
pipelines:
  per_fold_winsor:                # Plan Z++ Tier 1 proposed pipeline
    folds_tested: 10
    pass: 10
    fail: 0                       # ✅ 完美通过
  global_winsor_legacy_control:   # 故意保留旧 pipeline 做对照
    folds_tested: 10
    pass: 0
    fail: 10                      # ✅ 全失败，证明 sentinel 真能抓 leakage
overall_verdict: PASS
```

- **10 cells = 5 folds × {expanding, roll2y manifest}**，两种 walk-forward 方案都测
- **Control pipeline 全失败** = sentinel 有诊断力（不是"测试不灵敏"的假阴性）
- **σ=1e-3 扰动量** 故意选——比真实噪声小 100 倍，纯数学边界探针

### 我们是不是第一个（Web Research 结论）

| 比较点 | 结论 |
|-------|------|
| 以 "Sentinel leakage testing" 为名发表的论文 | **没有** |
| Medium blog "Leakage Sentinel" ([gsparsh2/...](https://medium.com/@gsparsh2/detecting-data-leakage-in-machine-learning-with-leakage-sentinel-ac8beead3e3a)) | 特征级 permutation importance，**与我们方法不同** |
| 最近 leakage 论文 ([arXiv:2512.06932 2024](https://arxiv.org/html/2512.06932v1)) | 关注 detection（统计 + 时间切分），不是 differential bitwise 比对 |
| DeepMind reproducibility checklist | 没 codify 这个 pattern |
| Software engineering 里的 differential testing（Csmith / CompCert） | 概念存在已久，**没人搬到 ML leakage 检测** |

**Verdict**：可能是 quant finance ML 领域第一个完整实现。Paper v3 可作为 methodological contribution 写：

> "We adapt differential testing — a technique from software engineering — to validate the leakage-free property of walk-forward training pipelines. To our knowledge, this is the first systematic application in cross-sectional equity prediction."

**安全说法**："first publicly documented" 或 "to our knowledge first in this domain"。

---

## Part 8：Q&A — Paper v2/v3 的 Baseline Reference

Paper 里 "baseline" 出现在 3 个层面：

### 1. Loss baseline = MSE

Paper v3 摘要：
> "After Benjamini-Hochberg correction at α=0.05, **0 of 28 (loss × architecture × feature) contrasts beat mean-squared error**"

所有备选 loss (Huber, Tukey biweight, truncated MSE, ListMLE, RankNet, anchored RankNet) 都和 MSE 比 ΔIC。MSE 是 "最简单 + 最常见" 的 ranking proxy，零假设基线。`delta_IC = IC(other_loss) - IC(MSE)`。

### 2. Hparam baseline = Adam vs h2 (AdamW + 10×wd)

跨 baseline robustness check：

| Adam baseline | h2 baseline |
|--------------|-------------|
| lr=1e-3, wd=1e-4, patience=10 | lr=5e-4, wd=1e-3, patience=5 |
| Stage 0 grid winner | Tier 1.D 5-seed 看似最优 |
| Tier 1.B 主结果用它 | Tier 1.B-h2 复跑用它 |

两套都 0/12 BH-FDR 拒绝 → null hypothesis 对 hparam 选择稳健。

### 3. Feature baseline = S8 (Alpha158 158-dim)

Hansen SPA：
```
SAGE-Mean × S8 (benchmark) vs S6 (3-dim PC probe):
  T_spa = 1.231, p_consistent = 0.5509   ← 不显著
```
S6 没显著优于 S8，但也没显著劣于 S8 → **parsimony argument**："用 3 个特征跟用 158 个特征 IC 没差"。

S8 是 "标准 quant 工业界 baseline"（qlib Alpha158），写进 paper 作为对照说"我们不是因为特征工程做得差才没拿到 alpha"。

### 4. Cell baseline 概念

每个 "contrast" = `(other_loss vs MSE)` at 固定 `(model, feature, fold)` cell。比如：
- cell = (Huber, MLP, S6) at fold 4
- baseline = (MSE, MLP, S6) at fold 4
- delta_IC = IC(Huber/MLP/S6/f4) - IC(MSE/MLP/S6/f4)

每个 cell 配对自己的 baseline 算 ΔIC → 全部 28 个 ΔIC 序列做 NW-HAC + BH-FDR。

---

## Part 9：Q&A — Plan Z++ S1-S7 设计 + Critical Limitation 发现

### S1-S7 实际定义（artifacts/step3_plan_z/subsets_frozen.json）

| Subset | 维度 | 特征 | 设计理由 |
|--------|------|------|---------|
| **S1** | 10 | 全特征：ret_mean_{5,10,21}d, ret_std_{5,10,21}d, mom12m, maxret, dolvol, CORR5 | 全集 baseline |
| **S2** | 7 | top-4 groups by Δ-IC（去掉 maxret, dolvol, CORR5） | 嵌套递减 1 |
| **S3** | 4 | top-3 groups（去掉 std + 短期变量） | 嵌套递减 2 |
| **S4** | 2 | top-2 groups: ret_mean_21d + mom12m | 嵌套递减 3 |
| **S5** | 1 | top-1 group: **只有 mom12m** | 嵌套极简 |
| **S6** | 3 | **PC-representative 3D probe**: ret_mean_10d + ret_std_10d + mom12m | **非嵌套对照** |
| **S7** | 9 | wf5 baseline: 含 momentum_Nd 与 ret_mean_Nd 的语义冗余 | apples-to-apples 对比旧 Phase 5 |

### 怎么设计（4 步）

1. **Part A 跑完**：10-feature 全集 30 runs，每个 group 做 permutation Δ-IC 排序：
   ```
   mom12m:        Δ-IC = +0.0182  ← 最重要
   ret_mean_21d:  Δ-IC = +0.0036
   ret_mean_10d:  Δ-IC = +0.0025
   ret_std_10d:   Δ-IC = +0.0012
   CORR5:         Δ-IC = -0.0002
   dolvol:        Δ-IC = -0.0005
   maxret:        Δ-IC = -0.0029  ← 实际有害
   ```
2. **按排序构造嵌套 S2-S5**
3. **S6 非嵌套 control**：从 Diag 3 PCA 选 3 个 PC 代表
4. **S7 是历史桥**：保留旧 wf5 9-dim 定义

### Hansen SPA 实测结果（experiments/step3_plan_z/hansen_spa_results.csv）

| Benchmark | T_SPA | p_consistent | 结论 |
|-----------|-------|--------------|------|
| **S1 (vs 10 维全集)** | 2.69 | 0.125 | 不显著 |
| **S7 (vs 9-dim 旧 baseline)** | **4.82** | **0.007** | ✅ 显著 — 新 subset 比旧 baseline 强 |
| **S8 (vs Alpha158 158-dim)** | 1.23 | 0.551 | 不显著（用更少特征不亏） |
| **S8_pf (per-fold winsor 后)** | 0.00 | 0.700 | 完全 null |

### 后续 Stage 1 + Tier 1 实际用了哪些

- **Stage 1**：S6 + S8 + S_price（不是 S7）
- **Tier 1.A/B/C/D**：S6 + S8
- **S1, S2, S3, S4, S5, S7 全部"做完即退役"**

为什么不全带：Plan Z++ Part B 验证完 "S6 ≈ S8" 后，不需保留所有 7 个 subset 跑 600+ runs。计算预算用在更多 loss × seed 上更有价值。

### CRITICAL: H博士 发现的 Universe Limitation

H博士 问："S1 全集才 10 个，158 为什么没进去一起排序？我们能找到正确的 Δ-IC 排序吗，S 子集会不会缺？"

#### 10 个特征怎么定的（来源 plan.md:1134）

13 个起点：Phase 5 era 9 个 baseline + 4 个经典文献 alpha factor (mom12m, maxret, dolvol, CORR5)。先语义折叠 3 对 duplicate → 13 → 10。

**这 10 个不是从 158 里挑的，是人工先验 universe**。

#### 为什么 Part A ranking 没在 158 上做

| 理由 | 详细 |
|------|------|
| **Compute 不现实** | 158 dim × grouped permutation × 16K iter × 30 runs = 50-100h M4；10 dim 只要 ~3h |
| **方法论哲学** | Plan Z++ 追求 "small interpretable universe + 严谨多重比较"；158 暴搜会让 multiple-testing burden 爆炸（$\binom{158}{7} ≈ 10^{10}$ subset 组合） |

#### Plan Z++ 怎么对冲这个风险（实际做了什么）

Part C 单独跑 S8 (Alpha158 158-dim) 作为"工业级 baseline"，Hansen SPA 比 S1 vs S8：

| Model | S1 vs S8 (T_SPA, p_consistent) | 解读 |
|-------|-------------------------------|------|
| SAGE-Mean | T = 1.04, p = 0.55 | S8 不显著优于 S1 |
| MLP | T = 0.81, p = 0.65 | S8 不显著优于 S1 |

→ 间接说"148 个额外特征**整体**没贡献"。

#### 但对冲不完美（真实 limitation）

| 能回答 | 不能回答 |
|-------|---------|
| "158 整体 vs 10 整体" 哪个 IC 更高 ✅ | 148 个里"是否藏有 1-2 个 individual feature"能进 top-7 ❌ |
| Subset 嵌套结构合理性 ✅ (在 10 dim 内) | Individual feature ranking 是否在 158 dim 内成立 ❌ |

**严格说**：找到的是 **conditional ranking**——"**在 10 个先验特征 universe 内**的 Δ-IC 排序"。不是 158 维 universe 上的真排序。

#### Paper v3 现状

Paper v3 [§1.2 Contribution #1](paper_draft_2026-05-18_v2.md) 写："Plan Z++ fixes contrast sets, statistical primaries, and pass thresholds *before* any experiment"，但**没明确说 contrast set 怎么定的、为什么只在 10 维 universe 内**。§7 Limitations **也没承认这个 specific limitation**。

#### 建议修复（按工作量从小到大）

##### Option A（0 compute, 30min 文本）

§7 Limitations 加：

> "**Feature universe scope**. Our preregistered S1-S5 subsets are constructed by grouped permutation importance ranking *within* a hand-curated 10-feature universe (`ret_mean_{5,10,21}d`, `ret_std_{5,10,21}d`, `mom12m`, `maxret`, `dolvol`, `CORR5`), not within the full 158-dimensional Alpha158 panel. The S8 (Alpha158) baseline is benchmarked at aggregate level via Hansen SPA, which shows no significant aggregate improvement over S1, but does not rule out the existence of individual high-signal features within the remaining 148 dimensions. We chose this scope to keep the multiple-testing burden tractable; an extension to 158-feature ranking is left to future work."

##### Option B（~15min compute）— ✅ 已执行（见 §Part 10.5）

LightGBM gain-based feature importance 在 158 维上，看 10 个 hand-curated 是否真的在 top-N。

**实际结果（2026-05-22 执行，基于 raw Alpha158 + per-fold train-only winsor）**：4/10 进 top-30（mom12m rank 6, ret_std_21d rank 8, ret_std_10d rank 16, maxret rank 28）；6 个不进，其中 3 个排到 100 名以后（CORR5 rank 114, dolvol rank 118, ret_mean_5d rank 136）。Verdict: universe choice partially weakened — 见 §Part 10.5 完整数据 + §Part 11 paper v3 立场调整。

##### Option C（50-100h compute）

158 维完整 grouped permutation Δ-IC × 16K × 30 runs，看 individual feature 是否进 top-10。

##### Option D（超出 paper v3 scope）

Plan Z++ Phase A 重设计：先在 158 维做 group-level Spearman 聚类得到 K=20-30 groups，再做 grouped permutation Δ-IC。Plan AAA / paper v4 级别。

#### H博士 选择

H博士 选 **Option B**（**已完成**，见 §Part 10.5；verdict: 4/10 in top-30，universe 部分弱化）。

---

## Part 10：Q&A — 其他方法论问题

### lr / wd 参数含义

| 参数 | 全名 | 作用 | 我们的值 | 直觉 |
|------|------|------|---------|------|
| **lr** | learning rate | 每次反向传播后 weights 更新步长 | `1e-3` (Adam) / `5e-4` (h2) | 太大跑飞，太小没动 |
| **wd** | weight decay | L2 正则化系数；每步 weights × `(1 - lr·wd)` | `1e-4` (Adam) / `1e-3` (h2) | 防过拟合 |
| **patience** | early stop patience | val IC 连续多少 epoch 不涨就停 | `10` (Adam) / `5` (h2) | 太大浪费 + 过拟合，太小可能在抖动期停 |
| **dropout** | dropout rate | 训练时随机 mask neurons | `0.3` | 隐式正则 |
| **grad_accum** | gradient accumulation | 多少 batch 累积后一次 `optimizer.step()` | `4` | 等效 batch size × 4 |
| **hidden** | hidden dim | 每层神经元数 | `64` | Capacity 控制 |

### AdamW 是不是过时（Web Research）

**结论**：没过时，2026 仍 production default。

来自 [Prodia blog 2025](https://blog.prodia.com/post/adam-w-vs-adam-key-differences-and-best-use-cases-for-developers)：

| 优化器 | 状态 (2026) |
|--------|-----------|
| **AdamW** | 仍是 production / 论文默认 |
| **Lion (Google 2023)** | 研究中 0-20% 增益，但没在生产替代 AdamW |
| **Ranger21** | 组合 8 个 trick，paper 用得多，生产少见 |
| **Shampoo / Sophia** | 二阶方法，大模型 LLM 用，小网络不划算 |
| **Plain Adam** | 反而少用了（AdamW 完全替代它） |

Tabular / financial time-series 文献（2024-2025）**没有任何 finance-specific optimizer 成为 conventional wisdom**。我们用 Adam (Stage 1 default) + AdamW (Tier 1.D h2) 是合理的。

### 备选 Loss 都没比 MSE 好，合理吗（Web Research）

**结论**：惊人地合理。

[arXiv:2510.14156 (2025)](https://arxiv.org/pdf/2510.14156) 用 Transformer 在 S&P 500 系统跑了 MSE vs RankNet vs ListMLE vs ListNet：

> "IC Spearman remained remarkably consistent (0.073-0.077) across losses... ranking losses significantly better for portfolio performance but did NOT show markedly superior IC scores."

**与我们结果完全一致！** IC 维度上 loss 选择基本不重要，差异在 portfolio metrics。

### MSE 是最合理的 loss 吗 — 取决于优化目标

| 评估维度 | 最合理 loss | 我们怎么样 |
|---------|-----------|----------|
| **IC (Spearman)** | MSE / Huber / 任意 pointwise，差不多 | ✅ 用 MSE 合理 |
| **Top-k portfolio return** | Ranking losses (ListMLE, ListNet, LambdaRank) | ❌ paper v3 没测 |
| **Long-short Sharpe** | Ranking losses + 显式 long-short objective | ❌ 我们 Sharpe 是 z-score 近似 |
| **Tail risk / drawdown** | Huber / Tukey (downside-aware) | ⚠️ 我们 fold-4 测了，反而 MSE 更稳 |

[arXiv:2104.12484 (2021)](https://arxiv.org/pdf/2104.12484) ListMLE for long-short portfolio 是主流，**但前提是评估 metric 是 portfolio return**。

### 我们论文的处境

**Paper v3 的 null finding 实际上独立 replicate 了 arXiv:2510.14156 (2025) 的发现**：
- 他们：transformer + S&P 500，IC 维度看 loss 不重要
- 我们：GNN + S&P 500，IC 维度看 loss 不重要（+ 严谨多重比较校正 + 4 种 robust loss + 跨 hparam baseline + Bonferroni 双主要）

**升级路径**：
1. 加 raw fwd_ret 重算 Sharpe（5min compute）→ 看 ranking losses 在 portfolio metric 上是否真比 MSE 好
2. Discussion 节 cite arXiv:2510.14156
3. **不要写 "ranking losses are useless"** → 改写 "for IC-based selection, MSE is competitive; for portfolio-grade selection, ranking losses retain a known advantage we did not formally test"

### BH-FDR 拒绝是什么意思

#### 一句话

**BH-FDR** = Benjamini-Hochberg False Discovery Rate procedure。**"拒绝"** = 拒绝 null hypothesis（接受 alternative hypothesis）。

#### 我们场景里具体含义

对每个 contrast `(loss vs MSE, model, feature)`：
- **Null hypothesis (H₀)**：这个 loss 不优于 MSE（ΔIC ≤ 0）
- **Alternative (H₁)**：这个 loss 严格优于 MSE（ΔIC > 0）

**单 contrast** 算 NW-HAC p-value。p<0.05 → 单测试下"拒绝 H₀"。

**问题**：跑了 28 个 contrast，每个 5% 假阳性率 → 期望 ~1.4 个假阳性。

#### BH-FDR 修正

控制 **expected proportion of false discoveries ≤ 5%**（不是控制单个 test 的 type-I error）。算法：
```
1. 28 个 p-value 升序排列：p_(1) ≤ p_(2) ≤ ... ≤ p_(28)
2. 找最大的 k 使 p_(k) ≤ k/28 × α  (α=0.05)
3. 拒绝排名前 k 个 null hypothesis
```

直觉：最小的 p-value 必须 < 0.05/28 = 0.0018；第 2 小的 < 0.0036；以此类推。

#### "0/28 BH-FDR 拒绝" 字面意思

28 个 contrast 的 NW-HAC p-value 排升序后，**没有任何一个能通过 BH-FDR 修正阈值**。最小 p-value > 0.0018。

**这是强 null**：不只是"没显著"，是**严格控制 5% false discovery 后仍没显著**。

#### 为什么用 BH 而非 Bonferroni

Bonferroni (`α/N` per test) 更保守，会拒绝过少；BH 控制 FDR 比 FWER 更宽松，是 finance / genomics 标准。Paper v2/v3 主用 BH，同时 report Bonferroni 作 sensitivity check。

---

## Part 10.5：Option B 执行结果（2026-05-22 本会话执行）

### 执行细节

- Script: [option_b_lgbm_importance.py](../option_b_lgbm_importance.py)（一次性 audit，跑完后建议归档）
- Output: [artifacts/option_b_lgbm_importance/summary.md](../artifacts/option_b_lgbm_importance/summary.md)
- 复用 `part_a.load_data_and_features()` 拿到 ticker-aligned labels + label_valid
- 跑 LightGBM on Alpha158 158-dim, fold-0 train slice (352,660 samples)
- Pipeline: per-fold train-only winsor (p1/p99) → z-score → LGBMRegressor (gain importance)

### Methodology Fix（Codex stop-hook Round 6 抓到）

**第一次执行（错误）**：用了 `sp500_5y_alpha158_features.npy`，但这个文件在 `build_alpha158_features.py:389` build 时已经做过 **全 panel p1/p99 winsorization**——正是 paper v2 §1.2 提到的 leakage bug。我又在上面套了一层 "per-fold train-only winsor"，但底层数据已经被污染，winsor claim 与 data 不一致。

**修复（第二次执行，2026-05-22）**：换用 `sp500_5y_alpha158_features_raw.npy`（pre-winsor 原始版本），然后在 fold-0 train_days 上做 per-fold train-only winsor。Verification：raw ROC5 范围 [0, 2.42] vs winsorized ROC5 范围 [0.89, 1.14]，确认 raw 是真未污染版。

**最终结果（基于正确 methodology）**：verdict 不变 (4/10)，具体 ranks 微调。

### Top 30 by Gain（基于 raw + per-fold train-only winsor）

```
Rank 1-10:  WVMA60, CORR60, CORD60, STD60, IMXD60, ROC60, BETA60, STD20, STD30, WVMA30
Rank 11-20: CORD30, RESI60, CORR30, RSQR60, MAX60, STD10, IMAX60, WVMA20, IMXD30, MIN60
Rank 21-30: MIN30, KLEN, CNTD60, RSQR30, CORD20, CORR20, BETA30, MAX10, VSTD60, IMIN60
```

**Pattern**: 60-day window 占绝对主导（11/30 top features 是 _60）；30-day 次之；5-day 几乎没有。

### Hand-Curated 10 → Alpha158 排名（基于正确 methodology）

| Hand-Curated | Alpha158 Equiv | Best Rank | In Top 30? |
|--------------|----------------|-----------|------------|
| **mom12m** | ROC60 | **6** | ✓ |
| **ret_std_21d** | STD20 | **8** | ✓ |
| **ret_std_10d** | STD10 | **16** | ✓ |
| **maxret** | MAX10 | **28** | ✓ (边缘) |
| ret_std_5d | STD5 | 43 | ✗ (差一点) |
| ret_mean_21d | ROC20 | 58 | ✗ |
| ret_mean_10d | ROC10 | 82 | ✗ |
| CORR5 | CORR5 | 114 | ✗ |
| dolvol | VMA20 | 118 | ✗ |
| ret_mean_5d | ROC5 | 136 | ✗ |

### 实测 Verdict：4/10 进 top-30 — **Universe Choice WEAKENED**

只有 4 个 hand-curated concepts 在 Alpha158 top-30。**6 个不在**，其中 3 个排到 100 名以后（CORR5, dolvol, ret_mean_5d）。

### 关键失败点

1. **CORR5 rank 114**：我们 universe 里它是 sector correlation proxy，但 LightGBM 完全不喜欢 5d 窗口。Alpha158 top-2/top-3 是 CORR60 + CORD60（60d 版本），说明 LightGBM 认为相关性应该用更长窗口
2. **ret_mean_Nd 全部失败**（rank 58/82/136 for 21d/10d/5d）：LightGBM 认为 ROC 类的 mean return 在 cross-sectional 上不是 strong signal
3. **dolvol rank 118**：volume-based features 对 IC 帮助不大
4. **ret_std_5d 差一点（rank 43）**：短窗口 std 不如长窗口

### 但有局部成功

1. **mom12m → ROC60 排第 6**：最强 hand-curated 特征（12-month momentum）实测确实在 Alpha158 top-10
2. **ret_std_{10,21}d 排 16 / 8**：volatility 类是真信号
3. **maxret → MAX10 排 28**：边缘进入 top-30

### Paper v3 立场（必须诚实承认）

§7 Limitations 应加：

> "**Feature universe scope (verified post-hoc)**. The Plan Z++ S1-S5 subsets are constructed by grouped permutation importance ranking *within* a hand-curated 10-feature universe, not within the full 158-dimensional Alpha158 panel. We post-hoc benchmark this choice via a LightGBM gain-based importance ranking on the full Alpha158 panel (fold-0 train slice, 352K samples, per-fold train-only p1/p99 winsorization on the *raw* pre-winsor Alpha158 build). **Only 4 of 10 hand-curated concepts** (mom12m/ROC60 rank 6, ret_std_21d/STD20 rank 8, ret_std_10d/STD10 rank 16, maxret/MAX10 rank 28) **have semantic equivalents in the Alpha158 top-30 by gain**; the remaining 6 concepts rank outside top-30, including 3 that rank beyond 100 (CORR5 rank 114, dolvol/VMA20 rank 118, ret_mean_5d/ROC5 rank 136). Alpha158 top-30 is dominated by 60-day window features (variance, correlation, momentum) that intersect our universe only through mom12m. We acknowledge that S1-S5 are conditional rankings within a curated universe that diverges from data-driven Alpha158 importance, and that high-signal features in the 148 dimensions excluded from our universe may exist. The S8 (Alpha158) Hansen SPA benchmark in §4.1 shows no significant aggregate improvement over S1, providing aggregate-level reassurance but not feature-level coverage. A 158-feature replication of Plan Z++ Part A is reserved for future work (Plan AAA)."

附带 contributions 节调整：

> §1.2 contribution claim 不应说 "subset selection is optimal" — 改为 "within a curated 10-feature universe motivated by classical alpha factor literature (Jegadeesh & Titman 1993; Bali et al. 2011; Amihud 2002), subset rankings are stable and parsimonious (S6 ≈ S1 ≈ S8 via Hansen SPA)."

### 战略意义

H博士 的 critique 不是杞人忧天 — 真的暴露了 Plan Z++ 的 universe choice limitation。但**这不是 fatal**：
- 我们的 hand-curated universe **不是完全错的**：mom12m (rank 6) + ret_std_21d (rank 8) 进 top-10；ret_std_10d (rank 16) 进 top-20，是数据驱动验证的
- 但 ret_mean_Nd 与 dolvol 在 Alpha158 ranking 下确实不强 — paper v3 应主动 disclose 而非被 reviewer 抓
- Hansen SPA aggregate-level 对冲机制有效（S1 vs S8 not significant）
- Paper v3 加 Option B 结果作为 robustness check，**反而强化论文方法学严谨度**

### Paper v3 应该 emphasize 的 narrative shift

| 之前 (paper v2/前 v3) | Option B 后 (paper v3) |
|----------------------|----------------------|
| "S6 是 parsimony 的最佳证据" | "S6 ≈ S1 ≈ S8 in Hansen SPA, AND 4/10 hand-curated concepts validated by Alpha158 LightGBM gain top-30" |
| 隐含 "Plan Z++ universe 选择是优化的" | 诚实写 "literature-motivated + partial data-driven validation; 6/10 concepts outside top-30, including 3 outside top-100 (CORR5 rank 114, dolvol rank 118, ret_mean_5d rank 136)" |
| 没提 Alpha158 importance | §7 Limitations 完整 disclose + cite Option B 数据 |

---

## Part 11：下一步行动

### 已完成（本会话）

1. ✅ 代码整理（root .py 32 → 14）
2. ✅ 5 轮 Codex stop-hook 修正
3. ✅ Methodology Q&A 整理（本 doc）
4. ✅ **Option B 执行 + 诚实 verdict（4/10 进 top-30，universe 部分弱化）**

### Paper v3 重写时（next major task）

1. §1.2 Contribution #1 明确写 contrast set 是 within 10-feature curated universe
2. §7 Limitations 加 Plan Z++ universe scope limitation（per Option A）
3. Discussion 节 cite arXiv:2510.14156 + arXiv:2104.12484，定位为 IC-vs-portfolio divergence confirmation
4. 不要写 "ranking losses are useless"，改 "for IC-based selection, MSE is competitive; ranking losses retain a known portfolio-metric advantage we did not formally test"
5. Methodology 节加 sentinel leakage test 作为 contribution（safe wording: "to our knowledge first in this domain"）

### 外延（paper v4 / 未来）

1. **Plan AAA**：158 维上做 group-level Spearman 聚类 + grouped permutation Δ-IC（Option D）
2. Sharpe 用 raw fwd_ret 重算（5min compute）→ 测 ranking losses 在 portfolio metric 上的真实表现
3. Sentinel leakage test 单独写一篇 methodology paper（如果 paper v3 reviewer 反响好）

---

## Appendix A：关键数据证据 quick lookup

### Subsets frozen (subsets_frozen.json)

```json
{
  "S1": ["ret_mean_5d", "ret_mean_10d", "ret_mean_21d", "ret_std_5d", "ret_std_10d", "ret_std_21d", "mom12m", "maxret", "dolvol", "CORR5"],
  "S2": ["ret_mean_5d", "ret_mean_10d", "ret_mean_21d", "ret_std_5d", "ret_std_10d", "ret_std_21d", "mom12m"],
  "S3": ["ret_mean_5d", "ret_mean_10d", "ret_mean_21d", "mom12m"],
  "S4": ["ret_mean_21d", "mom12m"],
  "S5": ["mom12m"],
  "S6": ["ret_mean_10d", "ret_std_10d", "mom12m"],
  "S7": ["ret_mean_5d", "ret_mean_10d", "ret_mean_21d", "ret_std_5d", "ret_std_10d", "ret_std_21d", "momentum_5d", "momentum_10d", "momentum_21d"]
}
```

### Group ranking (Part A permutation Δ-IC)

```
mom12m:        +0.0182  ← top
ret_mean_21d:  +0.0036
ret_mean_10d:  +0.0025
ret_std_10d:   +0.0012
CORR5:         -0.0002
dolvol:        -0.0005
maxret:        -0.0029  ← harmful
```

### Hparam defaults

```python
HPARAMS_DEFAULT = dict(
    hidden=64, num_layers=2, dropout=0.3,
    lr=1e-3, weight_decay=1e-4,
    epochs=50, patience=10, grad_accum=4,
    optimizer='adam',
)
```

### Tier 1.D hparam grid

```python
TIER1D_HPARAM_GRID = [
    dict(weight_decay=3e-4, lr=5e-4, patience=5, optimizer='adamw'),
    dict(weight_decay=3e-4, lr=2e-4, patience=5, optimizer='adamw'),
    dict(weight_decay=1e-3, lr=5e-4, patience=5, optimizer='adamw'),  # "h2"
    dict(weight_decay=1e-3, lr=2e-4, patience=5, optimizer='adamw'),
]
```

### Permutation v2 主结果

| Model | Feature | real_IC | p_value |
|-------|---------|---------|---------|
| MLP_all | 158 | -0.010 | 1.000 |
| MLP_price | 9 | +0.034 | 0.000 |
| SAGE_all | 158 | +0.008 | 0.002 |
| SAGE_price | 9 | +0.033 | 0.000 |

### Sentinel test 结果

```
per_fold_winsor pipeline:           10/10 PASS
global_winsor_legacy_control:        10/10 FAIL
overall_verdict:                     PASS
```

### Cumulative Story C+ v3

```
0/36 BH-FDR rejections
  = Stage 1 ranking 0/8 + Tier 1.B Adam 0/12 + Tier 1.B-h2 0/12 + Tier 1.C 0/4
+ 7 nulls
+ 3 mechanism findings
+ 1 regime-conditional finding
```

---

## Appendix B：Web Research 引用

1. [arXiv:2510.14156 (2025) — On Evaluating Loss Functions for Stock Ranking](https://arxiv.org/pdf/2510.14156)
   - Transformer + S&P 500 systematic loss comparison
   - IC consistent (0.073-0.077) across losses; portfolio metrics diverge

2. [arXiv:2104.12484 (2021) — Constructing Long-Short Stock Portfolio with Listwise Learn-to-Rank](https://arxiv.org/pdf/2104.12484)
   - ListMLE for long-short portfolio
   - 主流参考

3. [arXiv:2512.06932 (2024) — Hidden Leaks in Time Series Forecasting](https://arxiv.org/html/2512.06932v1)
   - LSTM data leakage 评估
   - 不是 differential testing

4. [Refined Lion Optimizer (Nature 2025)](https://www.nature.com/articles/s41598-025-07112-4)
   - Lion 优化器 0-20% gain over AdamW

5. [Ranger21 (arXiv:2106.13731)](https://arxiv.org/pdf/2106.13731)
   - 组合 8 components 的优化器

6. [Prodia Blog 2025 — AdamW vs Adam](https://blog.prodia.com/post/adam-w-vs-adam-key-differences-and-best-use-cases-for-developers)
   - AdamW 仍 production default

7. [Medium Leakage Sentinel](https://medium.com/@gsparsh2/detecting-data-leakage-in-machine-learning-with-leakage-sentinel-ac8beead3e3a)
   - 同名但不同方法（feature-level permutation importance）

---

*Created: 2026-05-22 (本会话整理 by Claude per H博士 request)*
*Updated: 2026-05-22 (Option B 完成 + Codex Round 6/7 methodology fix + rank propagation)*
*Cross-ref: progress.md 2026-05-21-a, 2026-05-21-b, 2026-05-22-a | plan.md (Plan Z++ Decision Log) | artifacts/option_b_lgbm_importance/summary.md*
*Option B status: ✅ COMPLETED 本会话 (raw Alpha158 + per-fold train-only winsor, 4/10 in top-30)*
*Next session: Paper v3 rewrite with §7 Limitations including Option B data (per §Part 10.5 draft); Codex Touchpoint 2/3 if quota available*
