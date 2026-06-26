#!/usr/bin/env python
"""fig_family2.py — §5.5 Family-2 causal edge attribution (fixed-capacity, frozen L2).

Two facets (Univ C top, B bottom), one row per edge contrast (+news / +sector / +sector+news).
Per row:
  • matched-ΔIC (CAUSAL PRIMARY): filled circle + 95% block-bootstrap CI over 12 fold blocks.
  • tuned-ΔIC (DESCRIPTIVE): open square; a connector to the matched point turns RED when the two
    disagree in sign (capacity confound — the edge looks harmful when capacity is NOT held fixed).
  • ±MDE@80% "undetectable zone" (light grey span): every matched effect sits inside it → underpowered.

Result: 0/6 survive BH-FDR, 6/6 underpowered → edge effects are directionally positive but not
reliable. Source: artifacts/storya_v21_family2_fc/family2_fc_causal.csv
Run: python paper_figs/fig_family2.py [--font serif]
Out: figures/family2_edge_causal.{pdf,png}
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
from matplotlib.patches import Rectangle

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from paper_figs.rcparams_storya import _apply_rc, save  # noqa: E402

F2 = os.path.join(PROJECT_ROOT, "artifacts", "storya_v21_family2_fc", "family2_fc_causal.csv")
INK = "#222222"
C_MATCH = "#2c7fb8"     # matched-ΔIC (causal primary)
C_TUNED = "#d95f02"     # tuned-ΔIC (descriptive)
C_REV = "#d62728"       # sign-reversal connector (capacity confound)
C_MDE = "#e6e6e6"       # ±MDE undetectable zone
ROW_ORDER = ["news", "sector", "sector+news"]
ROW_LABEL = {"news": "+news", "sector": "+sector", "sector+news": "+sector+news"}


def facet(ax, sub, ulabel, show_xlabel, with_legend):
    sub = sub.set_index("edge_added")
    for i, e in enumerate(ROW_ORDER):
        r = sub.loc[e]
        mde = float(r.MDE_80pct)
        ax.add_patch(Rectangle((-mde, i - 0.3), 2 * mde, 0.6, color=C_MDE, zorder=0))
    ax.axvline(0, color=INK, lw=0.7, ls=(0, (4, 3)), zorder=1)
    for i, e in enumerate(ROW_ORDER):
        r = sub.loc[e]
        md, lo, hi = float(r.matched_delta_IC), float(r.ci_lo), float(r.ci_hi)
        td = float(r.tuned_delta_IC)
        reversal = not bool(r.same_sign_matched_vs_tuned)
        ax.plot([td, md], [i, i], color=C_REV if reversal else "#bbbbbb",
                lw=1.0, zorder=2, solid_capstyle="round")
        ax.plot([lo, hi], [i, i], color=C_MATCH, lw=1.4, zorder=3, solid_capstyle="round")
        ax.scatter([md], [i], s=30, facecolor=C_MATCH, edgecolor=INK, linewidth=0.5, zorder=5)
        ax.scatter([td], [i], s=26, marker="s", facecolor="white", edgecolor=C_TUNED,
                   linewidth=1.1, zorder=4)
    ax.set_yticks(range(len(ROW_ORDER)))
    ax.set_yticklabels([ROW_LABEL[e] for e in ROW_ORDER])
    ax.set_ylim(len(ROW_ORDER) - 0.5, -0.5)
    ax.set_ylabel(ulabel, fontweight="bold", fontsize=8)
    ax.set_xlim(-0.024, 0.034)
    if show_xlabel:
        ax.set_xlabel("edge effect:  matched / tuned ΔIC  vs frozen L2")
    else:
        ax.tick_params(labelbottom=False)


def legend_handles():
    return [
        Line2D([0], [0], marker="o", color=C_MATCH, mfc=C_MATCH, mec=INK, mew=0.5,
               lw=1.4, ms=5.5, label="matched ΔIC (causal) + 95% CI"),
        Line2D([0], [0], marker="s", color="none", mfc="white", mec=C_TUNED, mew=1.1,
               ms=5.5, label="tuned ΔIC (descriptive)"),
        Line2D([0], [0], color=C_REV, lw=1.2, label="sign reversal (capacity confound)"),
        Line2D([0], [0], marker="s", color="none", mfc=C_MDE, mec="none", ms=9,
               label="±MDE@80% (undetectable)"),
    ]


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

    df = pd.read_csv(F2)
    fig = plt.figure(figsize=(6.8, 4.2))
    gs = GridSpec(2, 1, height_ratios=[1, 1], left=0.16, right=0.975,
                  top=0.80, bottom=0.235, hspace=0.30, figure=fig)
    ax_c = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax_c)
    facet(ax_c, df[df.universe == "C"], "Universe C\n(51-feat)", show_xlabel=False, with_legend=False)
    facet(ax_b, df[df.universe == "B"], "Universe B\n(10-feat)", show_xlabel=True, with_legend=False)

    fig.text(0.5, 0.955,
             "Family-2 (fixed-capacity edge): 0/6 survive BH-FDR, 6/6 underpowered",
             ha="center", va="top", fontsize=9.2, fontweight="bold", color=INK)
    fig.text(0.5, 0.905,
             "edge effects directionally positive but not reliable; matched(+) vs tuned(−) reversal = capacity confound",
             ha="center", va="top", fontsize=7.2, color="#444444")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=2, frameon=False,
               fontsize=6.6, handletextpad=0.5, columnspacing=1.6, bbox_to_anchor=(0.5, 0.005))
    paths = save(fig, "family2_edge_causal" + suffix)
    plt.close(fig)
    print(f"[family2] wrote {paths['pdf']}  | BH-reject {int(df.BH_FDR_reject.sum())}/{len(df)} "
          f"underpowered {int(df.underpowered_vs_effect.sum())}/{len(df)}")


if __name__ == "__main__":
    main()
