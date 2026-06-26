#!/usr/bin/env python
"""fig_pipeline.py — Story A confirmatory pipeline schematic (replaces the old pilot F1).

Top: the 5-stage flow (Data → Universe split → Models/ladder → Two confirmatory families →
Evaluation). Bottom: two zoom-ins — (left) the tuned L0–L7 ladder (the experimental axis;
edges added at L3–L5), (right) the GNN message-passing mechanism (a stock aggregates from
correlation / sector / news neighbours; this is what L2–L5 add over the non-graph MLP).

Conceptual map (no data). Sans-Arial global style, CB-safe, vector PDF.
Run: python paper_figs/fig_pipeline.py [--font serif]
Out: figures/pipeline_confirmatory.{pdf,png}
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from paper_figs.rcparams_storya import _apply_rc, save  # noqa: E402

INK = "#222222"
STAGE_FC = ["#E8F4F8", "#F3E5F5", "#E8F5E9", "#FFF3E0", "#FFFDE7"]


def rbox(ax, x, y, w, h, title, lines, fc, title_pt=8.2, line_pt=6.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2",
                                facecolor=fc, edgecolor=INK, linewidth=0.8, zorder=2))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=title_pt,
            fontweight="bold", color=INK, zorder=3)
    ax.text(x + w / 2, y + h - 6.0, "\n".join(lines), ha="center", va="top",
            fontsize=line_pt, color="#333333", zorder=3, linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, color=INK, style="-", lw=1.3, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
                                 color=color, lw=lw, linestyle=ls, zorder=4,
                                 shrinkA=0, shrinkB=0))


def chip(ax, cx, cy, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.2,rounding_size=1.2",
                                facecolor=fc, edgecolor=INK, linewidth=0.6, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=5.6, color=INK, zorder=4)


def draw(ax):
    # ── top: 5-stage flow ──
    n = 5
    w, gap = 16.5, 4.0
    x0 = (100 - (n * w + (n - 1) * gap)) / 2
    yb, h = 70, 24
    centers = []
    stages = [
        ("Data", ["S&P 500, 3 yr", "OHLCV + price feats", "news articles", "21-d c-to-c label"]),
        ("Universe split", ["B: 10 hand-crafted", "C: 51 Alpha158-top"]),
        ("Models — ladder", ["L0 … L7", "(tuned per arm)", "see zoom ↓"]),
        ("Confirmatory families", ["F1 predictive: SPA +", "  DM/HLN + BH/20", "F2 causal edge:", "  matched-ΔIC + BH/6"]),
        ("Evaluation", ["10 seeds × 12 folds", "IC (confirmatory)", "net Sharpe ladder", "  (descriptive)"]),
    ]
    for i, (t, ls) in enumerate(stages):
        x = x0 + i * (w + gap)
        rbox(ax, x, yb, w, h, t, ls, STAGE_FC[i])
        centers.append(x + w / 2)
        if i < n - 1:
            arrow(ax, x + w + 0.3, yb + h / 2, x + w + gap - 0.3, yb + h / 2)

    # ── bottom-left zoom: the tuned ladder ──
    lx, lw_, ly, lh = 3, 52, 6, 50
    ax.add_patch(FancyBboxPatch((lx, ly), lw_, lh, boxstyle="round,pad=0.6,rounding_size=2",
                                facecolor="#fafbfc", edgecolor="#999999", linewidth=0.7, zorder=1))
    ax.text(lx + lw_ / 2, ly + lh - 2.5, "The tuned ladder  (experimental axis)",
            ha="center", va="top", fontsize=7.4, fontweight="bold", color=INK)
    ladder = [("L0\nLGB", "#cfd4da"), ("L1\nMLP", "#aacbe6"), ("L2\nGAT", "#bfe0d0"),
              ("L3\n+news", "#a6d6c2"), ("L4\n+sect", "#a6d6c2"), ("L5\n+all", "#a6d6c2"),
              ("L6\nFullAtt", "#c9bde6"), ("L7\nHATS", "#cfe6b0")]
    cw, cy = (lw_ - 6) / len(ladder), ly + lh / 2 - 1
    for j, (txt, fc) in enumerate(ladder):
        cx = lx + 3 + cw * (j + 0.5)
        chip(ax, cx, cy, cw - 1.4, 9, txt, fc)
        if j < len(ladder) - 1:
            arrow(ax, cx + (cw - 1.4) / 2, cy, cx + cw - (cw - 1.4) / 2, cy, lw=0.8)
    # brackets: L0/L1 no graph | L2 base correlation graph | L3–L5 add extra edge types
    base = lx + 3

    def bracket(c0, c1, label, color):
        x0, x1 = base + cw * c0 + 0.6, base + cw * c1 - 0.6
        ax.plot([x0, x1], [cy - 6.3, cy - 6.3], color=color, lw=0.9)
        ax.text((x0 + x1) / 2, cy - 7.2, label, ha="center", va="top", fontsize=5.5, color=color)

    bracket(0, 2, "no graph", "#555555")
    bracket(2, 3, "corr graph", "#7a5cc0")
    bracket(3, 6, "+ extra edge types", "#1b7d5a")

    # ── bottom-right zoom: GNN message passing ──
    gx, gw, gy, gh = 58, 39, 6, 50
    ax.add_patch(FancyBboxPatch((gx, gy), gw, gh, boxstyle="round,pad=0.6,rounding_size=2",
                                facecolor="#fafbfc", edgecolor="#999999", linewidth=0.7, zorder=1))
    ax.text(gx + gw / 2, gy + gh - 2.5, "GNN message passing  (L2–L5)",
            ha="center", va="top", fontsize=7.4, fontweight="bold", color=INK)
    ccx, ccy = gx + gw / 2, gy + gh / 2 - 3
    # neighbours + typed edges
    nb = [(-13, 9, "#2c7fb8", "corr"), (13, 9, "#d95f02", "sector"),
          (-13, -9, "#1b9e77", "news"), (13, -9, "#2c7fb8", "corr")]
    for dx, dy, col, lab in nb:
        nx, ny = ccx + dx, ccy + dy
        ax.add_patch(Circle((nx, ny), 3.2, facecolor="#eef0f2", edgecolor=INK, linewidth=0.6, zorder=3))
        ax.text(nx, ny, "stock", ha="center", va="center", fontsize=5.2, color="#444444", zorder=4)
        arrow(ax, nx + (-3.0 if dx < 0 else 3.0) * 0.7, ny - 0.7 * (3.0 if dy > 0 else -3.0),
              ccx - (3.6 if dx < 0 else -3.6), ccy + (2.4 if dy > 0 else -2.4), color=col, lw=1.0)
        ax.text((nx + ccx) / 2 + (-2 if dx < 0 else 2), (ny + ccy) / 2, lab, ha="center",
                va="center", fontsize=5.3, color=col, zorder=5)
    ax.add_patch(Circle((ccx, ccy), 4.6, facecolor="#bfe0d0", edgecolor=INK, linewidth=0.9, zorder=5))
    ax.text(ccx, ccy, "stock i\naggregate", ha="center", va="center", fontsize=5.4,
            fontweight="bold", color=INK, zorder=6)
    ax.text(gx + gw / 2, gy + 2.5, "neighbours' signals pooled → node update", ha="center",
            va="bottom", fontsize=5.6, style="italic", color="#555555")

    # callouts: Models box → ladder zoom ; ladder graph arms → GNN zoom
    arrow(ax, centers[2], yb - 0.5, lx + lw_ / 2, ly + lh + 0.5, color="#999999", lw=0.8, ls=":")
    arrow(ax, lx + lw_ - 1, cy + 4, gx + 1, cy + 4, color="#999999", lw=0.8, ls=":")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 98)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--font", choices=["sans", "serif"], default="sans")
    args = ap.parse_args()
    _apply_rc()
    suffix = ""
    if args.font == "serif":
        mpl.rcParams.update({"font.family": "serif",
                             "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                             "mathtext.fontset": "stix"})
        suffix = "_serif"

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    draw(ax)
    paths = save(fig, "pipeline_confirmatory" + suffix)
    plt.close(fig)
    print(f"[pipeline] wrote {paths['pdf']}")


if __name__ == "__main__":
    main()
