#!/usr/bin/env python
"""fig_cost.py — §5.4 cost-口径 (gross/net): economic sensitivity of the headline claims.

(a) Net Sharpe vs transaction cost (0–30 bps) for the Univ-C key arms — tuned LightGBM (L0)
    turns negative after a few bps while MLP / +all / FullAttn stay positive.
(b) Gross IC effect vs net Sharpe@10bps effect for all 20 pre-registered contrasts — do the
    two口径 agree? Off-diagonal (shaded) = sign disagreement = cost-sensitive. Only the
    BH-significant "news hurts ranking" (C: +news−GAT) flips口径 (gross IC harm → net ≈ 0).

Net Sharpe is DESCRIPTIVE (IC stays confirmatory). Sources:
  artifacts/storya_v21_cost/cost_ladder_by_arm.csv
  artifacts/storya_v21_cost/cost_headline_crosswalk.csv
Run: python paper_figs/fig_cost.py [--font serif]
Out: figures/cost_gross_net.{pdf,png}
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from paper_figs.rcparams_storya import _apply_rc, save  # noqa: E402

COST = os.path.join(PROJECT_ROOT, "artifacts", "storya_v21_cost")
INK = "#222222"
C_SIG = "#d62728"        # cost-sensitive flip (BH-significant)
C_BLUE = "#2c7fb8"       # BH-significant,口径 agrees
C_NS = "#9aa0a6"         # not a BH claim
DISAGREE = "#fbe3e3"     # shaded sign-disagreement quadrants

LADDER_ARMS = [("L0", "LightGBM (L0)", "#d62728", "o"),
               ("L1", "MLP (L1)", "#2c7fb8", "s"),
               ("L2", "corr-GAT (L2)", "#7570b3", "^"),
               ("L5", "+all (L5)", "#1b9e77", "D"),
               ("L6", "FullAttn (L6)", "#e6ab02", "v")]


def panel_ladder(ax, lad):
    sub = lad[lad.universe == "C"]
    bps = [0, 5, 10, 15, 20, 30]
    ax.axhline(0, color=INK, lw=0.7, ls=(0, (4, 3)), zorder=1)
    for arm, label, col, mk in LADDER_ARMS:
        r = sub[sub.arm == arm].set_index("cost_bps")["Sharpe_net_mean"]
        ax.plot(bps, [r[b] for b in bps], color=col, marker=mk, ms=4, lw=1.4,
                label=label, zorder=3, mec=INK, mew=0.4)
    ax.axvline(10, color="#888888", lw=0.7, ls=":", zorder=1)
    ax.text(10, ax.get_ylim()[1], " headline\n 10 bps", fontsize=6.2, color="#666666",
            ha="left", va="top")
    ax.set_xlabel("transaction cost (bps, L1 one-way)")
    ax.set_ylabel("net Sharpe (Univ C, decile L/S)")
    ax.set_title("(a)  Net Sharpe vs cost — Univ C", loc="left", fontweight="bold")
    ax.set_xlim(-1, 31)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#dddddd",
              framealpha=0.96, fontsize=6.5, handletextpad=0.5, labelspacing=0.3).set_zorder(10)


def panel_crosswalk(ax, cw):
    xmin, xmax = -0.022, 0.034
    ymin, ymax = -1.0, 1.45
    # shade the two sign-DISAGREEMENT quadrants (cost-sensitive zones)
    ax.add_patch(plt.Rectangle((xmin, 0), -xmin, ymax, color=DISAGREE, zorder=0))      # x<0,y>0
    ax.add_patch(plt.Rectangle((0, ymin), xmax, -ymin, color=DISAGREE, zorder=0))      # x>0,y<0
    ax.axvline(0, color=INK, lw=0.7, zorder=1)
    ax.axhline(0, color=INK, lw=0.7, zorder=1)
    for _, r in cw.iterrows():
        x, y = float(r.gross_delta_IC), float(r.net_dSharpe_10bps)
        if bool(r.cost_sensitive) and bool(r.gross_BH_FDR_reject):
            col, fc, z, s = C_SIG, C_SIG, 6, 44
        elif bool(r.gross_BH_FDR_reject):
            col, fc, z, s = C_BLUE, C_BLUE, 4, 30
        else:
            col, fc, z, s = C_NS, "white", 3, 24
        ax.scatter([x], [y], marker="o", s=s, facecolor=fc,
                   edgecolor=col if fc == "white" else INK, linewidth=0.6, zorder=z)
    # faint quadrant watermark (top of the upper-left disagreement zone)
    ax.text(xmin * 0.5, ymax * 0.93, "gross & net DISAGREE", ha="center", va="top",
            fontsize=6.0, color=C_SIG, style="italic", alpha=0.8)
    # annotate the two key contrasts
    hl = cw[(cw.universe == "C") & (cw.pair == "L3-L2")].iloc[0]
    ax.annotate("C: +news − GAT\n(gross IC harm → net ≈ 0)",
                (hl.gross_delta_IC, hl.net_dSharpe_10bps), xytext=(-0.0205, 0.55),
                fontsize=6.3, color=C_SIG, va="center",
                arrowprops=dict(arrowstyle="->", color=C_SIG, lw=0.7))
    hd = cw[(cw.universe == "C") & (cw.pair == "L1-L0")].iloc[0]
    ax.annotate("C: MLP − LGB\n(holds at net)", (hd.gross_delta_IC, hd.net_dSharpe_10bps),
                xytext=(0.004, 1.22), fontsize=6.3, color=C_BLUE, va="center",
                arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=0.7))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("gross effect:  ΔIC (arm$_A$ − arm$_B$)")
    ax.set_ylabel("net effect:  ΔSharpe @10bps")
    ax.set_title("(b)  Do gross IC & net Sharpe agree?", loc="left", fontweight="bold")
    handles = [
        Line2D([0], [0], marker="o", color="none", mfc=C_SIG, mec=INK, ms=6.5,
               label="BH-sig & cost-sensitive"),
        Line2D([0], [0], marker="o", color="none", mfc=C_BLUE, mec=INK, ms=6,
               label="BH-sig, agrees"),
        Line2D([0], [0], marker="o", color="none", mfc="white", mec=C_NS, ms=5.5,
               label="not a BH claim"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True, facecolor="white",
              edgecolor="#dddddd", framealpha=0.96, fontsize=6.2,
              handletextpad=0.4, labelspacing=0.3).set_zorder(10)


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

    lad = pd.read_csv(os.path.join(COST, "cost_ladder_by_arm.csv"))
    cw = pd.read_csv(os.path.join(COST, "cost_headline_crosswalk.csv"))
    fig = plt.figure(figsize=(7.1, 4.0))
    gs = GridSpec(1, 2, width_ratios=[1.0, 1.12], left=0.075, right=0.985,
                  top=0.85, bottom=0.135, wspace=0.30, figure=fig)
    panel_ladder(fig.add_subplot(gs[0, 0]), lad)
    panel_crosswalk(fig.add_subplot(gs[0, 1]), cw)
    fig.text(0.5, 0.965,
             "After costs (net @10bps): headline claims hold; among BH-significant claims only \"news hurts ranking\" (C) reverses (gross IC harm → net ≈ 0)",
             ha="center", va="top", fontsize=8.2, fontweight="bold", color=INK)
    paths = save(fig, "cost_gross_net" + suffix)
    plt.close(fig)
    print(f"[cost] wrote {paths['pdf']}")


if __name__ == "__main__":
    main()
