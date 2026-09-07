# paper/iclr2027/ — ICLR 2027 submission source

> **作用**: Story A 论文「When Do GNNs Help in Cross-Sectional Stock Ranking?」的 **ICLR 2027** 单栏投稿版。由 acmart 版（`../main.tex`）转换而来：正文重构压入 **9 页硬限**，被压内容全部移入附录（A–D，无信息删除）；每个数字与 acmart 源及底层 CSV/docs 逐字一致（82 项数字忠实性抽查通过，2026-09-06）。截稿 **2026-09-18**（H博士 2026-09-01 提供）。

## 当前内容

| 文件 | 说明 |
|---|---|
| `main.tex` | 全文单文件（正文 9pp + AI/Ethics/Repro statements + references + 附录 A–D）。附录含 `% source:` 逐数字溯源注释。 |
| `main.pdf` | tectonic 编译输出（23 页；双盲提交版，"Under review" 头 + 匿名）。 |
| `references.bib` | 30 条（acmart 版 26 条 + 新增 thgnn2022 / kmz2024jf / jkp2023jf / patel2024survey，DOI 已验证）。 |
| `iclr2027_conference.{sty,bst}`, `natbib.sty`, `fancyhdr.sty` | ICLR 2027 官方 kit（github.com/ICLR/Master-Template）。 |

图片：`\graphicspath{{../../figures/}}`，正文 4 图 + 附录 3 图（regime_perfold_ic / loss_listmle_inversion / plan_aaa_t1_stability）。

## 附录结构

- **A Experimental setup details** — fold 日历、HP 网格、冻结超参表（含 n_trials=25/29 偏差披露 + M14 90-trial 行）、cell 预算、DM/HLN 估计量公式（自正文移入）
- **B Extended related work & evaluation-protocol audit** — 8 系统协议法证矩阵（7/8 单切分、0/8 多重校正、0/8 成本）+ THGNN/RSR/STHAN-SR/AD-GAT/KMZ/JKP 定位段
- **C Additional robustness** — M14 trials-sensitivity 全表、per-seed/LOSO 表、pooled-BH/BY 决策网格、planted-signal 阳性对照全表、MDE 构造
- **D Survivorship audit, stability failure, exploratory diagnostics** — L8 全审计表（m10）、C/L5s 崩溃、per-fold regime 表+图、ListMLE/Plan-AAA 探索图

## 编译 / 双盲切换

```bash
cd paper/iclr2027 && tectonic -X compile main.tex   # 23pp, exit 0
```
- 默认（`\iclrfinalcopy` 注释态）= **双盲提交版**：作者块与 Acknowledgments 自动隐藏，头显 "Under review..."
- 取消注释 `\iclrfinalcopy`（main.tex 内有说明）= 署名版：Ruixi (Tracy) He + Lv 教授致谢显示，头变 "Published as..."

## 校验状态（2026-09-06）

- 9 页正文限：AI use statement 起于 p10 顶（正文恰好 9pp 内）；abstract 单段（模板要求）
- 数字忠实性：82 项抽查 PASS（冻结超参 JSON / robustness CSV / m10 审计 / analysis.md M14+planted+perfold / lit_benchmark）；正文相对 acmart 版无新增数字
- 0 未解析引用/标签；无 acmart 残留命令；红线 grep 通过（无单 seed 0.044 类表述）
- ICLR 必填 AI use statement 已含（如实披露 AI 辅助 + 作者验证责任）

## 变更日志

- 2026-09-06: 初版完成——workflow 转换（主转换 agent 中途撞月度限额，附录 4 片段自 transcript 恢复）+ 7 轮压页 + 数字审计 + 双盲/署名双模式编译验证（→ progress: 2026-09-06-a）
