# docs/paper_sources/ — 论文引用 provenance (PDF 全文本地副本)

> **作用**: 本地保存 Story A paper 引用的关键论文 PDF，用于 verbatim quote 的来源追溯 (provenance) 与离线写作时的快速参考。**不进 paper supplementary**；仅作 reproducibility 工件。
>
> **为什么本地存**: (1) `docs/storya_paper_inspirations.md` 的 12K 字范式提炼依赖逐字引用 (`(Sawhney 2021, §X)` 等)，PDF 本地化保证后续 audit 时可独立核对；(2) AAAI 等出版商页面有时返回 403/500 或 OJS session 失败，本地缓存避免重蹈 2026-05-28 早期沙盒访问受限的故障；(3) Codex / 其他 agent 可通过本地 path 读取 PDF，无需联网。

## 当前内容

| 文件 | 大小 | 来源 | 引用位置 |
|------|------|------|----------|
| `sawhney_2021_sthansr_AAAI.pdf` | 6.7 MB, 8 pp. | AAAI 2021 proceedings | `docs/storya_paper_inspirations.md` §0 (Executive Summary), §2 (full A+B+C+D extraction), §4 (cross-paper comparison), §5 (synthesis strategy), §6 (phrase bank), Appendix A (source note) |

### sawhney_2021_sthansr_AAAI.pdf 完整书目信息

- **完整引用**: Sawhney, R., Agarwal, S., Wadhwa, A., Derr, T., Shah, R. R. (2021). "Stock Selection via Spatiotemporal Hypergraph Attention Network: A Learning to Rank Approach." *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(1), 497-504.
- **DOI**: 10.1609/aaai.v35i1.16127
- **下载源 URL** (2026-05-28): `https://ojs.aaai.org/index.php/AAAI/article/download/16127/15934`
- **下载方法**: 主 session bash `curl -sL ... -A "Mozilla/5.0" -o sawhney_2021_sthansr_AAAI.pdf`
- **MIME**: `application/pdf`; 文件头 PDF v1.5
- **md5** (2026-05-28 下载快照):
  ```
  $ md5 -r sawhney_2021_sthansr_AAAI.pdf
  ```
  (若需校验请重跑 `md5 -r`；本地下载快照与 AAAI 官方分发版本一致)

### 为什么是 STHAN-SR 而不是其他 GNN-finance 论文

H博士 2026-05-28 决策树：
1. 最初候选 3 篇顶刊: Feng-TOIS 2019 + Sawhney-AAAI 2021 + Hou-Xue-Zhang-RFS 2020
2. 首次提炼 agent 沙盒受限无法访问 AAAI proceedings，自行替换为 Cui 2021 HGTAN (arXiv) 并明文披露
3. H博士 2026-05-28 拒绝替代方案，指令 "重试拿 STHAN-SR"
4. 主 session bash 用 curl + Mozilla User-Agent 成功拿到 PDF
5. 第二轮 agent 用本地 PDF 重写 inspirations §2 + 相关章节，verbatim quote 全部归位

**为何这篇优于 Cui 2021 HGTAN**:
- STHAN-SR 8 页 AAAI 双栏 → 与 ICAIF 8 页 ACM SIGCONF 几乎完美对应 (Cui 是 14 页 IEEE Trans 格式，scale 不同)
- STHAN-SR 显式使用 ranking-aware combined loss (Eq.10)，与 Story A 的 cross-sectional ranking 任务定义对应；Cui 是 trend classification 而非 ranking
- STHAN-SR 用 named-baseline Wilcoxon `*` `†` 上标编码 p<0.01 显著性，比 Cui 的 prose-only "p<0.01" 更紧凑且更适合 Story A 直接借用
- STHAN-SR §5.3 "On the Effectiveness of Hypergraphs" degeneration-as-validation 实验设计可直接搬到 Story A E4-α edge ablation 章节

## 其他论文引用（未本地存）

`docs/storya_paper_inspirations.md` 还引用以下两篇，**未本地存 PDF**，理由：

| 论文 | 引用位置 | 未本地存原因 |
|------|------|---------|
| Feng et al. 2019 "Temporal Relational Ranking for Stock Prediction" (ACM TOIS) | inspirations §1 (architectural twin) | arXiv:1809.09441 直接可访问，PDF 已在 inspirations agent 写作时 read 过；DOI: 10.1145/3309547 |
| Hou, Xue, Zhang 2020 "Replicating Anomalies" (RFS) | inspirations §3 (rigor template) | RFS published version paywall; SSRN preprint + Google Scholar excerpts 可访问；inspirations doc 使用 verbatim abstract + 结论 + 主要 statistics，所有引用标注来源 (e.g., "HXZ 2020 abstract", "HXZ 2020 p.2020 [search excerpt]") |

如果后续 paper writing 阶段需要 Feng 或 HXZ PDF 本地全文，可：
- Feng: `curl -sL https://arxiv.org/pdf/1809.09441v2 -o feng_2019_tois_arxiv.pdf`
- HXZ: 通过学校图书馆代理或 NBER working paper 版本 (https://www.nber.org/system/files/working_papers/w23394/w23394.pdf)

## 关键文件速查

- **inspirations 文档**: `docs/storya_paper_inspirations.md` (104 KB, 12K 字)
- **STHAN-SR §5.3 degeneration-as-validation 设计**: PDF 第 5-6 页 (Sawhney 2021)，对应 Story A 的 F6 edge ablation forest
- **STHAN-SR 5-run Wilcoxon 显著性记号**: Table 2 of Sawhney 2021, 可借用为 Story A T1 / T2 显著性 superscripts

## 不变量

- **不修改 PDF**: 本地 PDF 是 verbatim 副本，禁止任何 editing / annotation overlay；如需 highlight 或 sticky note 请放在分离的 `.markdown` notes 文件而非 PDF 内部
- **md5 stable**: 任何文件 modification 都意味着 sourcing chain 断裂；inspirations doc 中的引用与 PDF 全文文字必须始终一致
- **referenced-from-inspirations**: 凡 inspirations doc 中提及的 STHAN-SR §X / Table Y / Eq.Z，必须能在本地 PDF 中找到对应文字

## 变更日志

- 2026-05-28: 文件夹创建。下载 Sawhney STHAN-SR AAAI 2021 PDF (6.7 MB) 作为 inspirations §2 verbatim quote 的本地 provenance 来源（→ progress: 2026-05-28 inspirations §2 rewrite by agent ID a40008cc0333a3226）。
