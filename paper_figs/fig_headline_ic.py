#!/usr/bin/env python
"""fig_headline_ic.py — §5.1 Family-1 headline: IC level per tuned arm (L0–L7 ladder).

Two stacked facets (Univ C top, Univ B bottom) sharing the IC x-axis. Each arm = seed-averaged
block-bootstrap IC mean ± 95% CI. The tuned LightGBM benchmark (L0) is drawn as a vertical
reference line per facet. Points filled if the IC 95% CI excludes 0 (reliably positive IC),
open if it includes 0. Conclusion (banner): tuned IC is small and overlapping across all arms,
including LightGBM — the formal "no arm beats L0" test is the SPA figure (F9).

Source: artifacts/storya_v21_family1/family1_ic_ci.csv
Run:    python paper_figs/fig_headline_ic.py   [--font serif]
Out:    figures/headline_ic_ladder.{pdf,png}
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

F1 = os.path.join(PROJECT_ROOT, "artifacts", "storya_v21_family1")

C_POS = "#2c7fb8"     # IC CI excludes 0 (reliably positive) — accent blue
C_NS = "#9aa0a6"      # CI includes 0 — neutral grey
C_BENCH = "#333333"   # LightGBM benchmark — NEUTRAL dark (not a significance colour; L0 is the
#                       reference regardless of its own CI, so it must not read as "significant")
INK = "#222222"
BAND = "#f5f6f8"

ARM = {"L0": "LGB (bench)", "L1": "MLP", "L2": "GAT", "L2s": "SAGE", "L3": "+news",
       "L4": "+sector", "L5": "+all", "L5s": "SAGE+all", "L6": "FullAttn", "L7": "HATS"}
ARM_ORDER = ["L0", "L1", "L2", "L2s", "L3", "L4", "L5", "L5s", "L6", "L7"]


def facet(ax, sub, ulabel, show_xlabel, with_legend):
    sub = sub.set_index("arm")
    l0 = float(sub.loc["L0", "IC_mean"])
    for i, arm in enumerate(ARM_ORDER):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=BAND, zorder=0)
    ax.axvline(0, color=INK, lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.axvline(l0, color=C_BENCH, lw=1.0, ls="--", zorder=2)
    ax.text(l0, -0.85, "LightGBM (L0)", color=C_BENCH, fontsize=6.4, ha="center",
            va="bottom", style="italic")
    for i, arm in enumerate(ARM_ORDER):
        r = sub.loc[arm]
        ic, lo, hi = float(r.IC_mean), float(r.IC_ci_lo), float(r.IC_ci_hi)
        excl = bool(r.ci_excludes_0)
        if arm == "L0":
            ax.scatter([ic], [i], s=42, marker="D", facecolor=C_BENCH, edgecolor=INK,
                       linewidth=0.5, zorder=5)
            ax.plot([lo, hi], [i, i], color=C_BENCH, lw=1.3, zorder=4, solid_capstyle="round")
            continue
        col = C_POS if excl else C_NS
        ax.plot([lo, hi], [i, i], color=col, lw=1.3, zorder=3, solid_capstyle="round")
        if excl:
            ax.scatter([ic], [i], s=26, facecolor=col, edgecolor=INK, linewidth=0.5, zorder=4)
        else:
            ax.scatter([ic], [i], s=22, facecolor="white", edgecolor=col, linewidth=1.0, zorder=4)
    ax.set_yticks(range(len(ARM_ORDER)))
    ax.set_yticklabels([ARM[a] for a in ARM_ORDER])
    ax.set_ylim(len(ARM_ORDER) - 0.5, -0.5)
    ax.set_ylabel(ulabel, fontweight="bold", fontsize=8)
    ax.set_xlim(-0.035, 0.092)
    if show_xlabel:
        ax.set_xlabel("Seed-averaged IC  (95% block-bootstrap CI)")
    else:
        ax.tick_params(labelbottom=False)
    # legend is drawn once at figure level (bottom) — see main() — so it never covers CI lines


def legend_handles():
    return [
        Line2D([0], [0], marker="D", color="none", mfc=C_BENCH, mec=INK, mew=0.5,
               ms=6, label="LightGBM benchmark (L0)"),
        Line2D([0], [0], marker="o", color="none", mfc=C_POS, mec=INK, mew=0.5,
               ms=5.5, label="IC CI excludes 0"),
        Line2D([0], [0], marker="o", color="none", mfc="white", mec=C_NS, mew=1.0,
               ms=5, label="IC CI includes 0"),
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

    df = pd.read_csv(os.path.join(F1, "family1_ic_ci.csv"))
    fig = plt.figure(figsize=(6.6, 4.8))
    gs = GridSpec(2, 1, height_ratios=[1, 1], left=0.17, right=0.975,
                  top=0.85, bottom=0.165, hspace=0.28, figure=fig)
    ax_c = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax_c)
    facet(ax_c, df[df.universe == "C"], "Universe C\n(51-feat)", show_xlabel=False, with_legend=False)
    facet(ax_b, df[df.universe == "B"], "Universe B\n(10-feat)", show_xlabel=True, with_legend=False)

    fig.text(0.5, 0.965, "Family-1 headline: tuned IC is small and overlaps across all arms (incl. LightGBM)",
             ha="center", va="top", fontsize=9.2, fontweight="bold", color=INK)
    # compact legend at figure bottom (outside the axes → covers no CI line)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3, frameon=False,
               fontsize=6.3, handletextpad=0.45, columnspacing=1.6, bbox_to_anchor=(0.5, 0.052))
    fig.text(0.975, 0.01,
             "Levels only; the formal cherry-pick-robust test (no arm beats L0) is the SPA/DM figure.",
             ha="right", va="bottom", fontsize=6.4, style="italic", color="#555555")

    paths = save(fig, "headline_ic_ladder" + suffix)
    plt.close(fig)
    print(f"[headline_ic] wrote {paths['pdf']}")


if __name__ == "__main__":
    main()
