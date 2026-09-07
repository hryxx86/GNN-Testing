# paper/ — LaTeX submission sources (acmart + ICLR 2027)

> **2026-09-06 起主线投稿源 = `iclr2027/`**（ICLR 2027 单栏版，截稿 9/18，见 `iclr2027/README.md`）。本目录根部的 acmart 版保留为 arXiv/期刊备用源。

> **作用**: Story A 论文「When Do GNNs Help in Cross-Sectional Stock Ranking?」的 **ACM SIGCONF (acmart)** 提交版 LaTeX 源。由 `docs/storya_paper_draft_v2.md`（confirmatory 工作草稿，已过 Codex T3 四轮）转换而来——**提交版去掉了工作草稿的 provenance 括注 / editor 清单 / 红线框 / verbose plain-English 框**；每个数字仍可溯源到草稿与其源 CSV。

## 当前内容

| 文件 | 说明 |
|---|---|
| `main.tex` | 全文（Abstract→§Reproducibility，无附录）。`\documentclass[sigconf,nonacm]{acmart}`。**8 页（ICAIF 硬上限）**，4 图 + 5 表 + 23 引用。**唯一正式投稿源**（2026-06-30 合并：原 9pp `main.tex` + 中间 codex 变体 + `main_jf_codex_compact.tex` 已合并为这一份 8pp 稿并删除多余文件，旧 9pp 可从 git 历史找回；备份在 session scratchpad）。 |
| `main.pdf` | 本地 tectonic 编译输出（8 页，干净）。 |
| `references.bib` | 23 条 BibTeX，`\bibliographystyle{ACM-Reference-Format}`。注释/验证轨迹见 `docs/storya_references.md`。 |
| `.paper-review/` | PaperJury 评审 ledger（`LEDGER.json`/`LEDGER.md` + 历史 round 记录）。 |

图片：`main.tex` 用 `\graphicspath{{../figures/}{figures/}}`，引用 `figures/*.pdf` 矢量母版（**4 张实际使用**：headline_ic_ladder / F9_spa_dm_confirmatory / cost_gross_net / family2_edge_causal；其余探索图在 9→8pp 裁剪中移出正文）。

## 编译

**本地 tectonic 可编译**（acmart + bib 内建，已验证 8 页、0 error）：
```bash
cd paper && tectonic -X compile main.tex     # 产出 main.pdf（8pp）
```
或用 **Overleaf**（内置 acmart）：上传 `main.tex` + `references.bib` + 4 张 `figures/*.pdf`，编译器 pdfLaTeX，流程 `pdflatex → bibtex → pdflatex ×2`（自动）。

## 校验状态（2026-06-30）

- **数字忠实性**：IC 表 / DM 20 对表 / Family-2 / SPA 值对源 CSV（`artifacts/storya_v21_family1`、`family2_fc`）→ 0 失配；compact↔旧 main.tex 数据数字零漂移（传递性继承）。
- **PaperJury ultracode 评审通过**（2026-06-30，`artifacts/reviews/2026-06-30_paperjury_compact-review_round1.md` + `.paper-review/LEDGER.json`）：压缩 SAFE，0 CRITICAL；scope-2 修复 + Tier-1 结论加强（头条改为 leak-robust L2−L1<0）已落地。剩 8 个 reviewer-anticipation 文字项待选做。
- **本地编译**：tectonic exit 0、**8 页**、`\ref`↔`\label` / `\cite`↔bib 全解析、4 图 `\includegraphics` 全命中。

## 待办（提交前）

- **作者块已填**（**独立作者 Ruixi (Tracy) He** / USC Viterbi / tracyhe@usc.edu；Lv 教授移至 Acknowledgments 致谢，2026-08-29）= **非匿名 preprint 版**（当前 `main.pdf`，9pp）。投**双盲**时只需在 `\documentclass` 加 `anonymous`（第 9 行 → `[sigconf,nonacm,anonymous]`），acmart 自动隐藏作者块与 acks 致谢，无需删真名（文件内有注释说明）。
- arXiv 上传：`main.tex` + `references.bib` + 4 张 `figures/*.pdf`（放 `figures/` 子目录匹配 `\graphicspath`）；可选把 `\acmConference[ICAIF '26]...` 改为中性 "Preprint. Under review." 以免未录用先挂会议页脚。
- 可选清剩余 8 个 reviewer-anticipation major（见 plan 2026-06-30-a / `.paper-review/LEDGER.json`，纯文字加 hedge，不重跑）。
- 可选补引：Stockformer / Pinheiro-Wedge（见 `docs/storya_references.md` 末「To-be-added」）。

## 变更日志

- 2026-09-06: **新增 `iclr2027/` 子目录**——ICLR 2027 单栏投稿版（9pp 正文 + 附录 A–D 回填 R3/R4/R5/M14/ST1/L8 全审计等），官方 kit + tectonic 23pp 编译验证 + 82 项数字忠实性审计；主线投稿源移交至该目录（→ progress: 2026-09-06-a）
- 2026-08-29: **作者署名变更（H博士 指示）**——独立作者 Ruixi (Tracy) He；Jinchi Lv 从作者块移除、移入 `acks` Acknowledgments 致谢（"supervision and guidance…All errors are my own"）；`\shortauthors{He}`；标题脚注 "Preprint—August 2026."（nonacm 下页脚不渲染，改用 `\titlenote`）+ `\acmConference` 日期字段同步。tectonic 重编译 9pp / 0 error 验证：标题页仅 1 作者，全文 "Lv" 仅剩致谢 1 处（→ progress: 2026-08-29-a）
- 2026-07-07: **R1/R2 补引落地**——§2 方法论段 +GKX（RFS'20，"no gain beyond shallow networks"）+ACM（MS'23，成本层外部验证）两句 + bib 2 条（DOI 级审计验证）。**排版（无内容变化）**：四图微缩至 .73/.73/.76/.81 + Repro 段冗余括号/§5.2 重复引用/Qlib bib 字段等 4 处零内容 trim → 匿名 8pp 恢复 / 非匿名 9pp。4-agent closeout 审计全 PASS（→ progress: 2026-07-07-a/-b）
- 2026-07-03: **M14 trials-sensitivity 两句入纸**——§4 "left for future work"→3×-budget sweep 结果句（B 存活 p=0.002 BH-reject / C 掉出 p=0.059 方向不变）、§6 under-search 备选解读→"narrows but does not close"限定句；摘要/§1 保留预注册口径；四图再微缩（.75/.75/.78/.83）保住**匿名 8pp**（非匿名 9pp）；措辞 Codex TP3 批准 + H博士 签核（→ progress: 2026-07-03-c）
- 2026-07-02: **投稿前评估建议落地（T0×21 + T1×5）**——新增 §4 Pipeline positive control（E3 planted-signal）、§5.2 合池 BH + BY 敏感性句（新引 benjamini2001by，现 24 条）、§5.3 LOSO seed 句、§5.2 under-search 边界论证；HXZ 引用修正（65%/82%）；Family-2 标签 causal→edge attribution；Table 1 caption 去 isolation 措辞；turnover/CI 构造/≈36 等半句补披露；页预算三轮收紧 + 四图微缩（.79/.79/.82/.85）→ **匿名 8pp / 非匿名 9pp 双版本验证**；Codex TP3 `artifacts/reviews/2026-07-02_codex_results_A.md`（→ progress: 2026-07-02-b）
- 2026-06-30: **合并为单一 8pp 投稿源**——把 codex compact（含 12 处评审修复 + Tier-1 结论加强）覆盖为 `main.tex`，删除中间 codex 变体 + 长名字 compact 文件 + 全部编译垃圾；本地 tectonic 验证 8 页；新增 `.paper-review/` 评审 ledger（→ progress: 2026-06-30-a）。
- 2026-06-25: 初版 LaTeX 源（`main.tex` + `references.bib`），由 confirmatory 草稿 v2 转换；数字交叉核验 + 静态审查通过（→ progress: 2026-06-25-a）。
