# paper/ — ICAIF 2026 LaTeX submission source

> **作用**: Story A 论文「When Do GNNs Help in Cross-Sectional Stock Ranking?」的 **ACM SIGCONF (acmart)** 提交版 LaTeX 源。由 `docs/storya_paper_draft_v2.md`（confirmatory 工作草稿，已过 Codex T3 四轮）转换而来——**提交版去掉了工作草稿的 provenance 括注 / editor 清单 / 红线框 / verbose plain-English 框**；每个数字仍可溯源到草稿与其源 CSV。

## 当前内容

| 文件 | 说明 |
|---|---|
| `main.tex` | 全文（Abstract→§Reproducibility + Appendix ST1）。`\documentclass[sigconf,nonacm]{acmart}`。8 图 + 8 表 + 21 引用。 |
| `references.bib` | 21 条 BibTeX（[1]–[21]），`\bibliographystyle{ACM-Reference-Format}`。注释/验证轨迹见 `docs/storya_references.md`。 |

图片：`main.tex` 用 `\graphicspath{{../figures/}{figures/}}`，引用 `figures/*.pdf` 矢量母版（8 张：pipeline / headline_ic_ladder / F9_spa_dm_confirmatory / regime_perfold_ic / cost_gross_net / family2_edge_causal / loss_listmle_inversion / plan_aaa_t1_stability）。

## 编译

**本机无 TeX**（无 pdflatex/acmart）→ 用 **Overleaf**（内置 acmart）：
1. 新建 Overleaf 项目，上传 `main.tex` + `references.bib` + 8 张 `figures/*.pdf`（放 `figures/` 子目录或同级，`\graphicspath` 两路径都试）。
2. 编译器选 pdfLaTeX；流程 `pdflatex → bibtex → pdflatex ×2`（Overleaf 自动）。

本地若装了 TeX：
```bash
cd paper && latexmk -pdf main.tex     # 或 pdflatex main; bibtex main; pdflatex main; pdflatex main
```

## 校验状态（2026-06-25）

- **数字忠实性**：脚本交叉核验 IC 表 / DM 20 对表 / Family-2 / SPA 值对源 CSV（`artifacts/storya_v21_family1`、`family2_fc`）→ **0 失配**。
- **静态 LaTeX 审查**：裸 `%`/`&`/`_` 仅出现在注释与 CCSXML（acmart 特殊处理）；`\ref`↔`\label` 全配对；8 图 `\includegraphics` 目标全部命中 `../figures/*.pdf`；表格列数逐表核对一致。**未本地编译**（无 TeX）——首次 Overleaf 编译后需肉眼核对溢出/浮动位置。

## 待办（提交前）

- 填 `\author` / `\affiliation` / `\email`（当前 Anonymous 占位）+ 确认 ICAIF '26 会议元数据（`\acmConference`）。
- 首次 Overleaf 编译后：核对页数（目标 8–10pp）、图表浮动、ACM-Reference-Format 渲染；按页数预算决定是否精简 §5.7 exploratory 或把 ST1 移补充材料。
- 可选补引：Stockformer / Pinheiro-Wedge（见 `docs/storya_references.md` 末「To-be-added」）。

## 变更日志

- 2026-06-25: 初版 LaTeX 源（`main.tex` + `references.bib`），由 confirmatory 草稿 v2 转换；数字交叉核验 + 静态审查通过（→ progress: 2026-06-25-a）。
