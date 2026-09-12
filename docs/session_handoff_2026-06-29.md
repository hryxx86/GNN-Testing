---
handoff_date: 2026-06-29
last_completed: "2026-06-29-a: ICAIF 8pp 页预算裁剪 — 删 pipeline图/§5.7两探索图/ST1附录/regime热图/related-work矩阵, 11pp→9pp (本地 tectonic 编译 0 错)"
in_flight:
  - id: page-budget-trim
    file: paper/main.tex
    status: "ICAIF'26 = 8pp 硬上限(图+表+引用全包含, 不收附录/补充材料, 超页 desk-reject — 官方 https://icaif2026.org/call-for-papers.html). 已删 6 浮动体+附录: 11pp→9pp, 现 4图+5表. 还差 1 页到 8."
    blockers:
      - "H博士 选 2 对 figure/table 各留哪个 (每删1全宽图≈0.4-0.5页, 删1张大概到8): A=删 fig:cost 保 fig:headline + 两表都留(推荐); B=两图都删两表都留(留余量); C=删 fig:headline 保 fig:cost"
  - id: author-metadata
    file: paper/main.tex
    status: "\\author/\\affiliation/\\email 仍是 Anonymous 占位; \\acmConference 未填 ICAIF'26. 注意 ICAIF 双盲评审 — 投稿版要匿名."
    blockers: ["H博士 提供作者信息 / 确认双盲匿名版"]
open_questions:
  - "页预算路线: 路A(死磕ICAIF 8pp, 删图)还是路B(保内容换更宽松场子, 9-12pp期刊/workshop)? 目前在执行路A, 已问到最后一对图/表取舍."
  - "5 条 Round-2 deferred CONCERN 是否要补(会增页数): stat-02 SPA-MDE-是-proxy / stat-04 abstract 补p-bracket / gnn-01 L6/L7-corroborating / gnn-04 Family-2-去混淆-role / qf-03 副标题 survivor-snapshot. 默认不补(8pp 压力)."
file_state:
  modified_uncommitted:
    - "paper/main.tex (大量: 24+10 审稿修复 + 6浮动体裁剪 + 5潜伏bug修复)"
    - "paper/references.bib (StockMixer/MDGNN 真BibTeX + sheppard_arch 补 year=2024)"
    - "progress.md (2026-06-26-a, 2026-06-27-a, 2026-06-29-a), plan.md (3 Decision Log), docs/analysis.md (M10 finding+成色), README.md (变更日志)"
    - "paper/.paper-review/REVIEW-ROUND-1.md (disposition表 + ROUND-1 CLOSED + Round-2 confirmation)"
  new_files:
    - "analyze_m10_universe_gap.py (M10 survivorship 审计脚本)"
    - "artifacts/audits/m10_universe_gap.{csv,md} + wikipedia_sp500_changes_cache.html"
    - "artifacts/reviews/2026-06-26_codex_code_A.md, 2026-06-26_codex_results_A.md, 2026-06-27_paperjury_round2.md"
    - "paper/main.pdf (本地 tectonic 编译产物, 当前 9pp)"
    - "docs/session_handoff_2026-06-29.md (本文件)"
  note: "全部未 commit, 等 H博士 过目. 另有历史 backlog(老文档删除/archived重组/.claude基建/~150路径)仍未提交, 与本次分开处理."
  new_local_infra: "tectonic 0.16.9 已 brew 安装 → 本地可端到端编译 acmart(之前无TeX). 命令: tectonic paper/main.tex --outdir paper"
rule9_status:
  touchpoint_2_code: PASSED        # M10 脚本: artifacts/reviews/2026-06-26_codex_code_A.md (2 FIXED: PIT分母+半开区间)
  touchpoint_3_results: PASSED     # M10 缺口: 2026-06-26_codex_results_A.md (PROCEED, 2 FIXED+3披露规格)
  paperjury_round1: CLOSED         # 24 findings 全 applied/disclosed/QUEUE, REVIEW-ROUND-1.md disposition表
  paperjury_round2: CLOSED         # 3 panels(stat/gnn/qf) 全 PASS-WITH-CONCERNS, 0 CRITICAL, 5 fixed+5 deferred
next_actions:
  - "H博士 选 figure/table 取舍 (A/B/C) → 我删图+重编译确认 8pp"
  - "填 author/affiliation/email + \\acmConference ICAIF'26 (双盲匿名版)"
  - "确认 8pp 达标后, 决定是否补 5 条 deferred CONCERN (权衡页数)"
  - "全部 OK → commit (本次 paper 里程碑改动; 历史 backlog 单独处理)"
  - "(QUEUE future work, 非投稿阻塞): M4 leak-free 重跑 / M13 L2 over-smoothing 证伪 / M14 trials sweep / m19 阈值消融 / M10 PIT universe 重建"
---

# Session Handoff — 2026-06-29

## TL;DR
论文 **`paper/main.tex` 的"内容关"已彻底过了**：PaperJury **两轮**审稿(Round-1 = 15 MAJOR+9 minor; Round-2 = 3 独立 panel 复审)**全部 CLOSED**, 四条红线全守, 所有 load-bearing 数字经多方独立核对 vs 源 CSV **零错误**, 且**本地 tectonic 能端到端编译出 PDF**(过程中救回 5 个会导致 desk-reject 的潜伏 LaTeX bug)。**唯一还开着的是页预算**: ICAIF'26 硬性 **8 页**(全包含、不收附录、超页拒稿), 已把 11pp 裁到 **9pp**, 还差 1 页 —— 卡在 H博士 选 headline-IC 和 cost 两对 figure/table 各留哪张。**所有改动未 commit**。

## 这次 session 干了什么(顺序)

### 1. PaperJury Round-1 处置 (progress 2026-06-26-a)
H博士 给了"按严重等级排序"的修复总纲, 逐条修 24 findings:
- **3 处现实修正**(代码核实推翻原假设): M2(√12 年化口径本就对, 21天不重叠)/M12(turnover_L1 单边定义清晰)/confirmatory-net(已用 tuned 成本层算好, 表数逐条对上) → 从"必须实修"降为"只澄清文本"。
- **阶段0**: M10 survivorship(写脚本实测)+ M4(Universe-C 正向降级 suggestive, null 不降级)。
- **阶段1/2/3**: M3/M7/M8/M9/M11 口径披露 + M1/M5/M6/M13/M14/M15 caveat + m16-m24 minor(含 StockMixer/MDGNN 真引用、EODHD news源)。

### 2. M10 survivorship 实测 + Rule9 (analysis 2026-06-26-a)
- `analyze_m10_universe_gap.py`: WebFetch Wikipedia S&P500 变更表, 实测固定 501 universe vs 真 PIT 成员。结果(`artifacts/audits/m10_universe_gap.md`): **名义缺口 14.8% / survivorship stock-days 8.2% / look-ahead 8.1% / 两侧成分错配 16.3%**。<20% 硬升级线 → **走披露不重建**。
- 成色: 剔除名 54 cap-change(指数小边缘)+28 M&A+0 微盘; look-ahead 76 名最小 $6.6B、无 <$5B → **边界大盘, 非小盘污染**。
- 写进 main.tex: Methods §3.1 条件估计量 + Limitation **L8 两段**(survivorship gap + composition mismatch, 禁"cancels", 点名 SVB/Signature/First Republic 距离名)。
- Rule9: TP2(`2026-06-26_codex_code_A.md` 2 FIXED) + TP3(`2026-06-26_codex_results_A.md` PROCEED)。

### 3. 5 个潜伏 LaTeX bug (装 tectonic 后端到端编译才暴露)
- **4 个 figure caption 漏闭合 `}`**(fig:headline/spa/regime/cost, commit eac6063 同样 Δ4 = 历史 bug 非本次引入) → 会"runaway \\caption"。
- **sheppard_arch 缺 year** → ACM-Reference-Format `[n.d.]` 路径破坏数学模式(`main.bbl:288 Missing $`), 编译前即 halt。补 `year=2024`。
- 全修后: 真 11pp PDF, 0 错。**brace 平衡过 ≠ 编译过 的活例**。

### 4. PaperJury Round-2 (progress 2026-06-27-a)
重跑 3-panel(stat/gnn/qf finance-gnn-reviewer, manuscript-only)于编辑后 main.tex:
- **全 PASS-WITH-CONCERNS, 0 CRITICAL, 1 MAJOR**; Round-1 全 24 条确认闭合(M7 PARTIAL→CLOSED); 数字零错引入。
- 本轮修 5: qf-01(L8 benignity 限定到流动性, 距离名缺口不被盖)/gnn-02(capacity-matched→operating-point)/qf-02(Wikipedia源caveat)/stat-03(BH用HLN_p_t)/stat-01(§1前向引用)。deferred 5(增页数、无correctness)。
- 全文: `artifacts/reviews/2026-06-27_paperjury_round2.md`。

### 5. 页预算裁剪 (progress 2026-06-29-a, 本次最后)
- 确认 ICAIF'26 = **8pp 硬上限**(官方 CFP, 全包含、不收附录、desk-reject)。README 老写的"8-10pp"乐观了。
- H博士 认可 3 个"减浮动体不减科学"切口 + 加删 pipeline 图: 删 §5.7两探索图/ST1附录(内化进§Reproducibility repo指针)/regime热图/related-work矩阵/pipeline图。
- 浮动体 **16→10**(中途), 再删 pipeline → **现 4图+5表, 9pp**。每删的图对应发现都留在正文。
- **还差 1 页**, 卡在两对 figure/table 取舍(见 in_flight blockers)。

## 现在论文的"硬事实"(给新 instance)
- **能编译**: `tectonic paper/main.tex --outdir paper` → `paper/main.pdf`(当前 9pp, 0错)。tectonic 0.16.9 本地已装。
- **红线(改任何东西都要守)**: SPA 未拒**禁 near-miss**(p_lower=0.055≥5%); Family-2 0/6 = **力不足非边无害**; Universe-C 正向 = **suggestive**(选择泄露基); IC 是唯一 confirmatory、net Sharpe 描述性; survivorship **禁"cancels"**(条件估计量)。
- **当前浮动体**: 图 fig:headline(IC森林)/fig:spa/fig:cost(成本双panel)/fig:family2; 表 tab:ladder/tab:ic/tab:dm/tab:cost/tab:family2。
- **待删候选**(到8pp): fig:headline↔tab:ic 同数据两视图; fig:cost↔tab:cost 同数据两视图。推荐删 fig:cost 保 fig:headline。

## 没做的(下次)
1. **页预算最后 1 页**: H博士 选 A/B/C → 删图 + 重编译验 8pp。
2. **author metadata + 双盲匿名版**(当前 Anonymous 占位)。
3. **commit**(本次 paper 里程碑; 历史 backlog 分开)。
4. (可选)5 条 deferred CONCERN; (QUEUE)M4 重跑/M13/M14/m19/M10 PIT 重建。

## Key paths
- 论文: `paper/main.tex` + `paper/references.bib` + `paper/main.pdf`(9pp) + `figures/*.pdf`(母版)。
- 审稿轨迹: `paper/.paper-review/REVIEW-ROUND-1.md`(disposition + Round-2 confirm) + `artifacts/reviews/2026-06-2{6,7}_*.md`。
- M10 证据: `analyze_m10_universe_gap.py` + `artifacts/audits/m10_universe_gap.{csv,md}`。
- 三件套: progress.md(2026-06-26-a/06-27-a/06-29-a) + plan.md(Decision Log) + docs/analysis.md(2026-06-26-a)。
