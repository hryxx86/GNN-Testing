# Session Handoff — 2026-04-15/16

> 新窗口开始前必读。当前 session 的完整状态。

---

## 当前状态：无后台进程

Gate 1 实验已手动停止（21/120 runs完成后提前终止）。无需检查PID。

---

## 本轮 Session 做了什么

### 核心事件：SEC Layer 1 Gate 1 实验 → STOP

SEC 10-K/10-Q 的 Lazy Prices TF-IDF similarity 特征对 S&P500 ranking **无效且有害**。

#### 实验设计
- 基于 v4 pipeline 创建 `run_gate1_experiment.py` (790 lines)
- Codex 6-point 审查通过（per-fold NaN fill, 4-tier gate criteria, no normalization）
- 8 model configs: SAGE/MLP/LGB × {price, priceL1} + SAGE × {priceLazy, priceDays}
- 只跑了 Fold 0 (21 runs)，信号足够强，提前终止

#### Gate 1 结果 (Fold 0, 3-seed mean)

| Model | price IC | priceL1 IC | Delta |
|-------|----------|-----------|-------|
| **SAGE-Mean** | 0.034 | 0.013 | **-0.021 (-61%)** |
| **MLP** | 0.034 | 0.023 | **-0.012 (-34%)** |
| **LGB** | 0.016 | 0.019 | **+0.003 (+17%)** |

#### 单特征 Ablation (SAGE, Fold 0)

| Feature | IC | Delta |
|---------|-----|-------|
| price only | 0.034 | — |
| + lazy_sim | 0.031 | -0.004 (-11%) |
| + days_since_filing | -0.001 | **-0.036 (catastrophic)** |
| + both | 0.013 | -0.021 (-61%) |

#### Root Cause
- `log1p_days_since_filing` (scale 0-7) 主导第一层梯度，破坏 ranking signal
- `lazy_sim` (~0.88 near-constant) 低跨截面方差，轻微有害
- LGB immune (tree scale-invariant)，但增量太小不值得

#### Decision
- **Gate 1: STOP**
- **Layer 2 (FinBERT sentiment): CANCELLED**
- **Layer 3 (Qwen structured): CANCELLED**
- Codex 同意：SEC filing 季度更新的 carry-forward 结构在日频 ranking 模型中根本无效

### LGB Crash Fix
- LightGBM `n_jobs=-1` + nohup 后台运行导致 silent crash (两次)
- 修复：改 `n_jobs=1` + `warnings.catch_warnings()` 抑制 sklearn warning

---

## 关键文件

| 文件 | 内容 |
|------|------|
| `run_gate1_experiment.py` | Gate 1 实验脚本 (790 lines) |
| `experiments/gate1_results.csv` | 21 runs 结果 (Fold 0 partial) |
| `experiments/gate1_log.txt` | 实验日志 |
| `data/sec_features/layer1_features.npy` | SEC L1 特征 (1255×503×2) |
| `data/sec_features/layer1_metadata.json` | 特征元数据 |
| `archived/plans/plan_sec_text_features.md` | 原始3层计划（已废弃） |
| `.claude/plans/dynamic-swinging-moonbeam.md` | Gate 1 实验计划 |

---

## 遗留问题

### 论文相关（需 H博士 决策）

1. **SEC negative finding 是否写进论文？** — "SEC filing text features do not improve stock ranking" 是有价值的 negative result
2. **下一步做什么？** — adversarial review P0 实验、图表完善、论文初稿？
3. **Qwen 新闻特征还做吗？** — Week 4 原计划的 Qwen 新闻（非SEC）结构化特征

### 技术遗留

1. **Gate 1 只跑了 Fold 0** — 如需论文数据，可重启跑完120 runs（~7h），但结论不会变
2. **其他脚本的 MLP 仍是空边 GNN** — 同 v4 handoff 记录的遗留问题

---

## v4 + Gate 1 综合状态

| 阶段 | 状态 |
|------|------|
| v4 5-fold walk-forward (90 runs) | ✅ DONE |
| True MLP baseline | ✅ DONE |
| Data leakage C1-C3 fix | ✅ DONE |
| **SEC Layer 1 Gate 1** | ✅ **STOP** |
| SEC Layer 2/3 | ❌ CANCELLED |
| Adversarial review P0 experiments | ⏳ PENDING |
| 论文图表 + 初稿 | ⏳ PENDING |

---

*Written: 2026-04-16*
