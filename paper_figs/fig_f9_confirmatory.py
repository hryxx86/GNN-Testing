#!/usr/bin/env python
"""fig_f9_confirmatory.py — §5.2 Family-1 statistical-defense figure on CONFIRMATORY data.

Rebuild of the old pilot F9 (fig_e6_statistical). Layout:
  (a) Hansen SPA p_consistent per universe vs alpha=0.05 → neither rejects (left, full height).
  (b) DM/HLN 20-test pairwise family forest, split into two STACKED facets sharing the x-axis
      (Universe C on top, Universe B below) so the universe label never collides with the
      contrast tick labels. mean ΔIC ± 95% block-bootstrap CI, colour+marker double-encoded
      by BH-FDR significance.

Conclusion the figure defends (foregrounded as a top banner): no tuned arm is confirmed to
beat tuned LightGBM (SPA); the pairwise rejections are LOCAL ladder rungs, not global superiority.

Sources:
  artifacts/storya_v21_family1/family1_spa.csv      (SPA p_consistent)
  artifacts/storya_v21_family1/family1_dm_hln.csv   (mean_delta_IC, BH_FDR_reject_family)
  artifacts/storya_v21_family1/family1_mde.csv       (delta_ci_lo/hi block-bootstrap CI)

Run:  python paper_figs/fig_f9_confirmatory.py
Out:  figures/F9_spa_dm_confirmatory.{pdf,png}
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
F1 = os.path.join(PROJECT_ROOT, "artifacts", "storya_v21_family1")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")

mpl.rcParams.update({
    # global standard = sans-serif Arial (H博士 2026-06-23); --font serif for the alt render
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 0,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

C_SIG = "#d62728"          # BH-reject (signal red)
C_NS = "#9aa0a6"           # not rejected (neutral grey)
C_BAR = "#5a6b8c"          # SPA bars (neutral slate, NOT a significance colour)
C_REJECT_BAND = "#f4d7d7"  # SPA reject region shade
INK = "#222222"
BAND = "#f5f6f8"           # alternating row band

ARM = {"L0": "LGB", "L1": "MLP", "L2": "GAT", "L3": "+news", "L4": "+sector",
       "L5": "+all", "L6": "FullAttn", "L7": "HATS", "L2s": "SAGE", "L5s": "SAGE+all"}
PAIR_ORDER = ["L1-L0", "L2-L1", "L6-L2", "L7-L2", "L2s-L2",
              "L3-L2", "L4-L2", "L5-L2", "L5-L4", "L5-L3"]


def pair_label(pair: str) -> str:
    a, b = pair.split("-")
    return f"{ARM.get(a, a)} − {ARM.get(b, b)}"


def load():
    spa = pd.read_csv(os.path.join(F1, "family1_spa.csv"))
    dm = pd.read_csv(os.path.join(F1, "family1_dm_hln.csv"))
    mde = pd.read_csv(os.path.join(F1, "family1_mde.csv"))
    dm["pair"] = dm["arm_A"] + "-" + dm["arm_B"]
    m = dm.merge(mde[["universe", "pair", "delta_ci_lo", "delta_ci_hi"]],
                 on=["universe", "pair"], how="left", validate="one_to_one")
    return spa, m


def panel_spa(ax, spa):
    order = ["B", "C"]
    p = [float(spa.loc[spa.universe == u, "p_consistent"].iloc[0]) for u in order]
    x = np.arange(len(order))
    ax.axhspan(0, 0.05, color=C_REJECT_BAND, zorder=0)
    ax.axhline(0.05, color=C_SIG, lw=1.0, ls="--", zorder=2)
    ax.bar(x, p, width=0.62, color=C_BAR, edgecolor=INK, linewidth=0.5, zorder=3)
    for xi, pi in zip(x, p):
        ax.text(xi, pi + 0.008, f"p = {pi:.3f}", ha="center", va="bottom",
                fontsize=7.5, color=INK)
    ax.text(0.03, 0.046, "reject $H_0$ ($p<0.05$)", transform=ax.get_yaxis_transform(),
            ha="left", va="top", fontsize=6.5, color=C_SIG, style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Univ {u}" for u in order])
    ax.set_ylim(0, 0.34)
    ax.set_ylabel("Hansen SPA  $p_{\\mathrm{consistent}}$\n(benchmark = tuned LightGBM, $M=9$)")
    ax.set_title("(a)  Superior Predictive Ability", loc="left", fontweight="bold")
    ax.margins(x=0.18)


def forest_facet(ax, sub, universe_label, show_xlabel, with_legend):
    """One universe's 10-pair forest on its own axis (shared x with the sibling facet)."""
    sub = sub.set_index("pair")
    ys = np.arange(len(PAIR_ORDER))
    for i, pair in enumerate(PAIR_ORDER):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=BAND, zorder=0)
    ax.axvline(0, color=INK, lw=0.7, ls=(0, (4, 3)), zorder=1)
    for i, pair in enumerate(PAIR_ORDER):
        r = sub.loc[pair]
        d, lo, hi = float(r.mean_delta_IC), float(r.delta_ci_lo), float(r.delta_ci_hi)
        sig = bool(r.BH_FDR_reject_family)
        col = C_SIG if sig else C_NS
        ax.plot([lo, hi], [i, i], color=col, lw=1.3, zorder=3, solid_capstyle="round")
        if sig:
            ax.scatter([d], [i], s=26, facecolor=col, edgecolor=INK, linewidth=0.5, zorder=4)
        else:
            ax.scatter([d], [i], s=22, facecolor="white", edgecolor=col, linewidth=1.0, zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([pair_label(p) for p in PAIR_ORDER])
    ax.set_ylim(len(PAIR_ORDER) - 0.5, -0.5)   # first pair on top
    ax.set_ylabel(universe_label, fontweight="bold", fontsize=8)
    ax.set_xlim(-0.045, 0.072)
    if show_xlabel:
        ax.set_xlabel("Mean $\\Delta$IC  (arm$_A$ − arm$_B$),  95% block-bootstrap CI")
    else:
        ax.tick_params(labelbottom=False)
    if with_legend:
        handles = [
            Line2D([0], [0], marker="o", color=C_SIG, mfc=C_SIG, mec=INK, mew=0.5,
                   lw=1.3, ms=5.5, label="BH-FDR reject ($q=0.05$)"),
            Line2D([0], [0], marker="o", color=C_NS, mfc="white", mec=C_NS, mew=1.0,
                   lw=1.3, ms=5.0, label="not rejected"),
        ]
        ax.legend(handles=handles, loc="lower right", frameon=True, facecolor="white",
                  edgecolor="#dddddd", framealpha=0.96, handletextpad=0.5,
                  borderaxespad=0.4).set_zorder(10)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--font", choices=["sans", "serif"], default="sans",
                    help="sans (Arial, global standard) or serif (Times, ACM-body alt)")
    args = ap.parse_args()
    suffix = ""
    if args.font == "serif":
        mpl.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
        })
        suffix = "_serif"

    spa, m = load()
    fig = plt.figure(figsize=(7.1, 4.9))
    gs = GridSpec(2, 2, width_ratios=[1.0, 2.5], height_ratios=[1.0, 1.0],
                  left=0.115, right=0.985, top=0.815, bottom=0.115,
                  wspace=0.42, hspace=0.30, figure=fig)
    ax_spa = fig.add_subplot(gs[:, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, 1], sharex=ax_c)

    panel_spa(ax_spa, spa)
    ax_c.set_title("(b)  DM/HLN pairwise  (BH-FDR over 20 tests)", loc="left", fontweight="bold")
    forest_facet(ax_c, m[m.universe == "C"], "Universe C\n(51-feat)",
                 show_xlabel=False, with_legend=False)
    forest_facet(ax_b, m[m.universe == "B"], "Universe B\n(10-feat)",
                 show_xlabel=True, with_legend=True)

    fig.text(0.5, 0.965, "Family-1: no tuned arm is confirmed to beat tuned LightGBM",
             ha="center", va="top", fontsize=9.5, fontweight="bold", color=INK)
    fig.text(0.985, 0.012,
             "DM/HLN rejections are LOCAL ladder rungs — not global superiority over LightGBM.",
             ha="right", va="bottom", fontsize=6.8, style="italic", color="#555555")

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "F9_spa_dm_confirmatory" + suffix)
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=200)
    plt.close(fig)
    n_sig = int(m.BH_FDR_reject_family.sum())
    print(f"[F9] wrote {out}.pdf/.png  |  SPA B={spa.loc[spa.universe=='B','p_consistent'].iloc[0]:.4f} "
          f"C={spa.loc[spa.universe=='C','p_consistent'].iloc[0]:.4f}  |  BH-reject {n_sig}/{len(m)}")


if __name__ == "__main__":
    main()
