#!/usr/bin/env python
"""build_gallery.py — generate a SELF-CONTAINED bilingual figure gallery (figures/figure_gallery.html).

Images are base64-embedded, so the single .html opens anywhere (no relative-path / missing-PNG
issues). To ADD or UPDATE a figure: edit the FIGS list below and re-run:
    python paper_figs/build_gallery.py
Output: figures/figure_gallery.html
"""
from __future__ import annotations

import base64
import os

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

# ── figure entries (paper-narrative order). Add a dict to extend the gallery. ──
FIGS = [
    dict(
        id="pipeline", tag="METHODS", cls="methods", png="pipeline_confirmatory.png",
        title="Pipeline &amp; GNN message passing 流程与消息传递",
        script="paper_figs/fig_pipeline.py （scientific-schematics → matplotlib）",
        data="概念图，无数据 / conceptual, none",
        key="L2 = 基础相关图；L3–L5 叠加额外边类型 / L2 is the base graph; L3–L5 add edge types",
        zh=dict(
            what="整个研究的概念流程图。上排五个阶段：<b>数据</b>（标普 500，3 年日线 OHLCV + 价格特征 + 新闻 + 21 日收盘到收盘标签）→ <b>股票池划分</b>（B 池 10 手工特征 / C 池 51 Alpha158 特征）→ <b>模型梯子</b>（L0…L7，逐臂调参）→ <b>两个确认性家族</b> → <b>评估</b>（10 seed × 12 季度滚动折）。",
            read="下方两个放大图。左：<b>调参梯子</b>——L0 LightGBM（基线）→ L1 MLP（无图）→ L2 相关图 GAT → L3–L5 逐步加边 → L6 全注意力 → L7 HATS；底部括号标“无图 / 相关图 / 加额外边类型”。右：<b>GNN 消息传递</b>——一只股票从相关 / 行业 / 新闻邻居聚合信号。",
            take="L2 本身就是图（相关边），L3–L5 在其上叠加额外边类型。概念图，不含数据。"),
        en=dict(
            what="The conceptual study pipeline. Top row: <b>Data</b> (S&amp;P 500, 3 yr daily OHLCV + price feats + news + 21-day c-to-c label) → <b>Universe split</b> (B: 10 hand-crafted / C: 51 Alpha158) → <b>Model ladder</b> (L0…L7, tuned per arm) → <b>Two confirmatory families</b> → <b>Evaluation</b> (10 seeds × 12 quarterly folds).",
            read="Two zoom-ins below. Left, the <b>tuned ladder</b>: L0 LightGBM (baseline) → L1 MLP (non-graph) → L2 corr-graph GAT → L3–L5 add edges → L6 full attention → L7 HATS; brackets mark “no graph / corr graph / + extra edge types”. Right, <b>GNN message passing</b>: a stock aggregates from correlation / sector / news neighbours.",
            take="L2 is already a graph (correlation edges); L3–L5 stack extra edge types. Conceptual, no data."),
    ),
    dict(
        id="headline", tag="§5.1", cls="s51", png="headline_ic_ladder.png",
        title="Headline IC by arm 各臂 IC 水平",
        script="paper_figs/fig_headline_ic.py",
        data="artifacts/storya_v21_family1/family1_ic_ci.csv",
        key="C 池 MLP IC=0.034、LightGBM=0.020；多数臂 CI 含 0 / many arms’ CI include 0",
        zh=dict(
            what="Family-1 的“门面”图。每个模型臂的 <b>IC</b>（信息系数 = 模型打分排名与次日真实涨跌排名的相关度）及 95% 自助置信区间，C 池（上）/ B 池（下）。",
            read="黑色菱形 + 虚线 = LightGBM 基线（L0），是<b>对照参照</b>（中性色，不代表显著）。<b>蓝色实心</b> = IC 置信区间不含 0（可靠为正）；<b>灰色空心</b> = 含 0（测不准）。",
            take="所有臂（含 LightGBM）IC 都很小（约 0.01–0.04）且置信区间大幅重叠——没有哪个臂明显胜出。这张只看“水平”；正式“无人打赢 L0”检验在 §5.2。"),
        en=dict(
            what="The Family-1 headline. Each arm’s <b>IC</b> (rank correlation between the model’s score and next-day returns) with a 95% block-bootstrap CI, Univ C (top) / B (bottom).",
            read="The black diamond + dashed line is the LightGBM benchmark (L0), a <b>neutral reference</b> (colour does not imply significance). <b>Filled blue</b> = IC CI excludes 0 (reliably positive); <b>open grey</b> = CI includes 0.",
            take="Every arm (LightGBM included) has small IC (≈0.01–0.04) with widely overlapping CIs — no arm clearly wins. Levels only; the formal “no arm beats L0” test is §5.2."),
    ),
    dict(
        id="spa", tag="§5.2", cls="s52", png="F9_spa_dm_confirmatory.png",
        title="Hansen SPA + DM/HLN family 统计防御",
        script="paper_figs/fig_f9_confirmatory.py",
        data="family1_spa.csv · family1_dm_hln.csv · family1_mde.csv",
        key="SPA B=0.277, C=0.077（皆不拒绝）；DM 11/20 对 BH 显著（局部）",
        zh=dict(
            what="Family-1 的统计防御（防 cherry-pick）。左：Hansen SPA 的 p 值，B=0.277、C=0.077，红区 = 拒绝区（p&lt;0.05）。右：20 对 DM/HLN 配对检验森林（C 上 / B 下）。",
            read="左图两根柱都<b>在红区之上</b> → 两池都<b>不拒绝</b>原假设 → 没有任何调参臂被确认打赢调参 LightGBM。右图横轴 = ΔIC（arm_A − arm_B），<b>红实心 = BH-FDR 显著</b>、灰空心 = 不显著。",
            take="<b>措辞红线</b>：C 池 0.077 只能说“不显著”，<span class='flag'>禁止</span>“接近显著”。右图显著只是<b>梯子内局部</b>差异，不是“全局打赢 LightGBM”。"),
        en=dict(
            what="The Family-1 statistical defence. Left: Hansen SPA p-values, B=0.277, C=0.077; red band = rejection region (p&lt;0.05). Right: the 20-pair DM/HLN forest (C top / B bottom).",
            read="Both SPA bars sit <b>above</b> the red band → <b>neither universe rejects</b> H₀ → no tuned arm is confirmed to beat tuned LightGBM. In the forest, x = ΔIC (arm_A − arm_B); <b>filled red = BH-FDR significant</b>, open grey = not.",
            take="<b>Wording red line</b>: C’s 0.077 is “not significant”, <span class='flag'>never</span> “near-significant”. The pairwise rejections are <b>local ladder rungs</b>, not global superiority over LightGBM."),
    ),
    dict(
        id="regime", tag="§5.3", cls="s53", png="regime_perfold_ic.png",
        title="Regime dependence 体制依赖性",
        script="paper_figs/fig_regime.py",
        data="experiments/storya_v21_main12_tuned/results.csv (+L7)；退化 C/L5s 已排除",
        key="C 池 2025Q2 均值 IC≈0.145；约半数季度 ≤0",
        zh=dict(
            what="信号的“体制集中度”。(a) 热图：行 = 模型臂、列 = 12 季度、颜色 = 该季 IC（红=高、蓝=负），Univ C。(b) 每季平均 IC（B / C 两池柱）。",
            read="只有少数列<b>深红</b>（2024Q4、2025Q2 尤强），约一半季度淡或偏蓝（≈0 或负）。底部柱图同样显示强弱季交替。",
            take="预测信号<b>高度集中在特定市场体制</b>，不稳定。这解释了为什么 headline IC 在<b>留一折（LOFO）</b>下很脆——去掉一个强季度均值就大变。"),
        en=dict(
            what="How concentrated the signal is. (a) Heatmap: rows = arms, columns = 12 quarters, colour = that quarter’s IC (red=high, blue=negative), Univ C. (b) Per-quarter mean IC (bars, B / C).",
            read="Only a few columns are <b>deep red</b> (2024Q4 and 2025Q2 especially); about half the quarters are pale/bluish (≈0 or negative). The bars show the same strong/weak alternation.",
            take="The signal is <b>concentrated in specific regimes</b>, not steady — why headline IC is fragile under <b>leave-one-fold-out</b>: dropping one strong quarter moves the mean."),
    ),
    dict(
        id="cost", tag="§5.4", cls="s54", png="cost_gross_net.png",
        title="Cost gross/net 成本（毛/净）口径",
        script="paper_figs/fig_cost.py",
        data="artifacts/storya_v21_cost/cost_ladder_by_arm.csv · cost_headline_crosswalk.csv",
        key="C 池 MLP net@10bps=+0.95 vs LightGBM=−0.22；C news 净 ΔSharpe=+0.08（CI 跨 0）",
        zh=dict(
            what="把每句结论拿“扣交易成本后”的经济口径重检。(a) 净 Sharpe vs 成本（0–30bps），C 池关键臂。(b) 散点：横轴 = 毛口径（ΔIC）、纵轴 = 净口径（ΔSharpe@10bps）。",
            read="(a) 中 <b>LightGBM 扣几个 bps 就转负</b>，MLP / +all / 全注意力仍为正。(b) 中<b>粉色阴影象限</b> = 两口径符号不一致（成本敏感）；红点 = BH 显著且翻号、蓝 = BH 显著且一致、灰空 = 非 claim。",
            take="扣成本后两条主结论（神经网络赢树、图不帮忙）都站得住；BH 显著 claim 里<b>唯一翻号</b>的是 C 池“news 伤排序”——IC 说伤、净 Sharpe ≈ 0（阴影区，fold 脆），<b>不是</b> news 经济有用的证据。净 Sharpe 描述性，IC 确认性。"),
        en=dict(
            what="Re-checking each claim under a net-of-cost lens. (a) Net Sharpe vs cost (0–30 bps), Univ-C key arms. (b) Scatter: x = gross effect (ΔIC), y = net effect (ΔSharpe @10 bps).",
            read="In (a), <b>LightGBM turns negative after a few bps</b> while MLP / +all / full-attention stay positive. In (b), <b>pink shaded quadrants</b> = the two lenses disagree in sign (cost-sensitive); red = BH-sig &amp; reverses, blue = BH-sig &amp; agrees, open grey = not a claim.",
            take="After costs the two headline claims hold; among BH-significant claims the <b>only reversal</b> is Univ-C “news hurts ranking” — gross IC says harm, net Sharpe ≈ 0 (shaded, fold-fragile), <b>not</b> evidence news helps. Net Sharpe descriptive; IC confirmatory."),
    ),
    dict(
        id="family2", tag="§5.5", cls="s55", png="family2_edge_causal.png",
        title="Family-2 causal edge 因果边归因",
        script="paper_figs/fig_family2.py",
        data="artifacts/storya_v21_family2_fc/family2_fc_causal.csv",
        key="0/6 BH-FDR、6/6 欠功效；matched-ΔIC C-L4=+0.014 / C-L5=+0.014（CI 边缘）",
        zh=dict(
            what="Family-2 因果边归因：把容量<b>冻结在 L2 超参</b>上，只改边集，干净隔离“边本身”的效应。每行一个边对比（+news / +sector / +sector+news），C 上 / B 下。",
            read="<b>蓝实心 + CI</b> = matched-ΔIC（因果主指标）；<b>橙空心方块</b> = tuned-ΔIC（描述性）；<b>灰带</b> = ±MDE@80%“测不出区”（蓝点落带内即欠功效）；<b>红连线</b> = matched 与 tuned 符号相反（容量混淆）。",
            take="6 对比 <b>0 个</b>过 BH-FDR、<b>6 个全欠功效</b> → 边效应方向为正但<b>不可靠</b>。B 池“matched 正 vs tuned 负”翻号揭示：不固定容量时边看着有害，固定后真相是<b>微弱正</b>——“评估不干净会误判边”的活体证据。"),
        en=dict(
            what="Family-2 causal edge attribution: <b>freeze capacity at the L2 hyperparameters</b> and vary only the edge set. One row per edge contrast (+news / +sector / +sector+news), C top / B bottom.",
            read="<b>Filled blue + CI</b> = matched-ΔIC (causal primary); <b>open orange square</b> = tuned-ΔIC (descriptive); the <b>grey band</b> = ±MDE@80% “undetectable zone” (a dot inside = underpowered); a <b>red connector</b> = matched and tuned disagree in sign (capacity confound).",
            take="<b>0 of 6</b> survive BH-FDR and <b>all 6 are underpowered</b> → edge effects directionally positive but <b>not reliable</b>. The B matched(+)-vs-tuned(−) reversal shows edges look harmful without fixing capacity, but are a <b>tiny positive</b> once fixed — a live demo that unclean evaluation misjudges edges."),
    ),
    dict(
        id="loss_inv", tag="§5.7", cls="expl", png="loss_listmle_inversion.png",
        title="Listwise-loss inversion 列表损失翻转",
        script="paper_figs/fig_loss_inversion.py",
        data="experiments/loss_horserace/results.csv",
        key="listmle mean IC=−0.0458 vs mse=+0.0113（per-cell）",
        zh=dict(
            what="<b>探索性图（非确认性）</b>。损失函数赛马的副产品：MSE / Pairwise / ListMLE 三种排序损失在共享脚手架上的平均 IC（每个 seed×fold cell 取均值后再汇总，±1 标准误）。确认性梯子<b>锁定用 MSE</b>，这里只作失败模式展示。",
            read="三根柱配 0 参考线。MSE（青）正、Pairwise（紫）≈0、<b>ListMLE（红）显著为负</b>。柱上标注精确均值。",
            take="<b>ListMLE 在体制漂移下系统性翻号</b>：其 softmax-似然目标被训练期排序主导，测试期排名一变，损失面反转、模型给出反排名 → 平均 IC=−0.0458（vs MSE +0.0113）。对排序损失从业者的警示。<span class='flag'>探索性，不进确认性结论</span>。"),
        en=dict(
            what="<b>Exploratory (not confirmatory)</b>. A by-product of the loss horse race: mean IC for MSE / Pairwise / ListMLE on a shared scaffold (per-cell mean over seed×fold, then ±1 SE). The confirmatory ladder is <b>locked to MSE</b>; this is shown only as a failure mode.",
            read="Three bars against a 0 reference line. MSE (teal) positive, Pairwise (purple) ≈0, <b>ListMLE (red) clearly negative</b>; exact means annotated on each bar.",
            take="<b>ListMLE inverts under regime shift</b>: its softmax-likelihood objective is dominated by the in-distribution rank order, so when test ranks shift the loss landscape inverts and the model produces anti-rankings → mean IC = −0.0458 (vs MSE +0.0113). A caution for ranking-loss practitioners. <span class='flag'>Exploratory, not a confirmatory claim</span>."),
    ),
    dict(
        id="plan_aaa_t1", tag="§5.7", cls="expl", png="plan_aaa_t1_stability.png",
        title="Plan-AAA T−1 basis 基础脆弱性",
        script="paper_figs/fig_plan_aaa_t1.py",
        data="artifacts/plan_aaa_t1_diagnostic/group_ranking_comparison.csv",
        key="原 top-15 中 5 个存活 T−1（ROC30+5 / KMID+6 / KUP+1 / CNTP20+3 / CORR60）",
        zh=dict(
            what="<b>探索性 caveat 图（Limitation L1）</b>。C 池的特征来自早期 Plan-AAA 的 top-15 Alpha158 组，原排名在<b>当日 OHLC</b> 口径下算的。本图把“原排名”对“严格 T−1 修正后排名”作散点。",
            read="横轴=原 Plan-AAA 排名，纵轴=T−1 修正后排名；阴影框=top-15 区，虚线=完美稳定。<b>绿星=T−1 后仍留 top-15（5 个）</b>，灰圆=掉出（10 个）。",
            take="<b>原 top-15 只有 5 个 T−1 后仍 top-15</b> → C 池特征的<b>选取基础脆弱</b>（L1）。<span class='flag'>注意</span>：运行时特征<b>确实在 T−1</b>取值（无泄露），脆弱的只是“当初凭什么选这些”的基础。探索性。"),
        en=dict(
            what="<b>Exploratory caveat figure (Limitation L1)</b>. Universe C is built from the earlier Plan-AAA top-15 Alpha158 groups, originally ranked under a <b>same-day-OHLC</b> procedure. This scatters the original rank against the rank after strict T−1 leak correction.",
            read="x = original Plan-AAA rank, y = rank after T−1 correction; shaded box = top-15 region, dashed line = perfect stability. <b>Green stars = stay in the top 15 after T−1 (5)</b>, grey circles = drop out (10).",
            take="<b>Only 5 of the original top 15 stay in the top 15</b> after T−1 correction → the Universe-C selection <b>basis is fragile</b> (L1). <span class='flag'>Note</span>: runtime features <b>are</b> evaluated at T−1 (not leaked); only the basis for choosing them is fragile. Exploratory."),
    ),
]

CSS = """
  :root { --ink:#222; --muted:#666; --line:#e3e6ea; --accent:#2c7fb8; --bg:#fafbfc; }
  * { box-sizing: border-box; }
  body { font-family: Arial, Helvetica, sans-serif; color: var(--ink); line-height: 1.65;
         max-width: 1180px; margin: 0 auto; padding: 28px 22px 80px; background: #fff; }
  h1 { font-size: 26px; margin: 0 0 4px; }
  .sub { color: var(--muted); margin: 0 0 8px; font-size: 14px; }
  .legend-note { background: var(--bg); border: 1px solid var(--line); border-radius: 8px;
                 padding: 10px 14px; font-size: 13px; color: #444; margin: 14px 0 28px; }
  .toc { font-size: 14px; margin: 0 0 30px; }
  .toc a { color: var(--accent); text-decoration: none; margin-right: 14px; white-space: nowrap; }
  .fig-card { border: 1px solid var(--line); border-radius: 12px; padding: 20px 22px 8px;
              margin: 0 0 34px; box-shadow: 0 1px 3px rgba(0,0,0,.04); }
  .fig-head { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; margin-bottom: 2px; }
  .fig-head h2 { font-size: 19px; margin: 0; }
  .tag { font-size: 11px; font-weight: 700; color: #fff; background: var(--accent);
         border-radius: 5px; padding: 2px 8px; letter-spacing: .3px; }
  .tag.methods { background: #7a5cc0; } .tag.s51 { background: #1b9e77; }
  .tag.s52 { background: #d95f02; } .tag.s53 { background: #b8860b; }
  .tag.s54 { background: #c0392b; } .tag.s55 { background: #2c7fb8; }
  .tag.expl { background: #7a7f87; }
  .fig-img { text-align: center; margin: 14px 0 4px; }
  .fig-img img { max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 6px; }
  .bi { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin: 16px 0 6px; }
  @media (max-width: 760px) { .bi { grid-template-columns: 1fr; } }
  .bi h3 { font-size: 13px; text-transform: uppercase; letter-spacing: .5px; color: var(--muted);
           border-bottom: 1px solid var(--line); padding-bottom: 4px; margin: 0 0 8px; }
  .bi p { margin: 0 0 9px; font-size: 14px; }
  .bi b { color: #000; }
  .meta { font-size: 12.5px; color: var(--muted); border-top: 1px dashed var(--line);
          margin-top: 10px; padding: 9px 0 4px; }
  .meta code { background: var(--bg); padding: 1px 5px; border-radius: 4px; font-size: 12px; color:#333; }
  .meta .row { margin: 2px 0; }
  .flag { color: #c0392b; }
"""


def b64(png_path: str) -> str:
    with open(png_path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


def card(fig: dict) -> str:
    img = b64(os.path.join(FIG_DIR, fig["png"]))
    z, e = fig["zh"], fig["en"]
    return f"""
<section class="fig-card" id="{fig['id']}">
  <div class="fig-head"><span class="tag {fig['cls']}">{fig['tag']}</span><h2>{fig['title']}</h2></div>
  <div class="fig-img"><img src="{img}" alt="{fig['id']}"></div>
  <div class="bi">
    <div>
      <h3>中文</h3>
      <p><b>是什么。</b>{z['what']}</p>
      <p><b>怎么看。</b>{z['read']}</p>
      <p><b>结论。</b>{z['take']}</p>
    </div>
    <div>
      <h3>English</h3>
      <p><b>What it is.</b> {e['what']}</p>
      <p><b>How to read.</b> {e['read']}</p>
      <p><b>Takeaway.</b> {e['take']}</p>
    </div>
  </div>
  <div class="meta">
    <div class="row">脚本 / Script: <code>{fig['script']}</code></div>
    <div class="row">数据 / Data: <code>{fig['data']}</code></div>
    <div class="row">关键数 / Key: {fig['key']}</div>
  </div>
</section>"""


def main():
    toc = " ".join(f'<a href="#{f["id"]}">{f["tag"]} {f["title"].split(" ")[0]}</a>' for f in FIGS)
    cards = "\n".join(card(f) for f in FIGS)
    html = f"""<!DOCTYPE html>
<!-- Self-contained bilingual figure gallery. Regenerate: python paper_figs/build_gallery.py -->
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Story A — Figure Gallery 图库</title>
<style>{CSS}</style></head><body>
<h1>Story A — Figure Gallery / 图库</h1>
<p class="sub">“When Do GNNs Help in Cross-Sectional Stock Ranking” · ICAIF 2026 · 6 confirmatory figures (D-RERUN-12F) + 2 §5.7 exploratory (grey tag, non-confirmatory)</p>
<div class="legend-note"><b>全局约定 / Conventions.</b>
  sans-Arial；矢量 PDF + PNG；显著性<b>颜色 + 形状双编码</b>，色盲安全。
  净 Sharpe 为<b>描述性</b>，<b>IC 仍是唯一确认性指标</b>。
  &nbsp;|&nbsp; Sans-Arial; vector PDF + PNG; significance double-encoded (colour + shape), CB-safe.
  Net Sharpe is <b>descriptive</b>; <b>IC remains the sole confirmatory metric</b>. 图片已内嵌（base64），单文件可离线打开。</div>
<div class="toc"><b>目录 / Contents:</b> {toc}</div>
{cards}
<p class="sub" style="margin-top:40px;border-top:1px solid var(--line);padding-top:14px;">
  生成 / Generated by <code>paper_figs/build_gallery.py</code> · 三方 QA 通过（Codex 正确性 + nature-figure QA 清单 + 人工视检）。
  新图：在脚本 FIGS 列表加一项后重跑。 / New figures: add an entry to FIGS and re-run.</p>
</body></html>"""
    out = os.path.join(FIG_DIR, "figure_gallery.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[gallery] wrote {out}  ({len(html)//1024} KB, {len(FIGS)} figures, images embedded)")


if __name__ == "__main__":
    main()
