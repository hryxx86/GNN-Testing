#!/usr/bin/env python
"""fig_plan_aaa_t1.py — §5.7 / Limitation L1: Universe-C basis fragility (Plan-AAA, T-1).

EXPLORATORY / caveat figure. Universe C is composed from the project's earlier Plan-AAA
top-15 Alpha158 groups, originally ranked under a same-day-OHLC procedure. Under strict
T-1 leak correction only 5 of the top 15 survive in the top 15. Runtime features ARE T-1
(not leaked); the *basis* for selecting them is fragile.

Conclusion-first scatter: original rank (x) vs T-1-shifted proxy rank (y) for the top-15
groups; the diagonal is perfect stability; green stars = survive top-15 after T-1, grey
circles = drop out. Banner states "5/15 survive".

Source: artifacts/plan_aaa_t1_diagnostic/group_ranking_comparison.csv
Run: python paper_figs/fig_plan_aaa_t1.py [--font serif]
Out: figures/plan_aaa_t1_stability.{pdf,png}
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from paper_figs.rcparams_storya import _apply_rc, save  # noqa: E402

SRC = os.path.join(PROJECT_ROOT, "artifacts", "plan_aaa_t1_diagnostic",
                   "group_ranking_comparison.csv")
INK = "#222222"
C_SURV = "#1b9e77"   # teal-green — survives T-1
C_DROP = "#999999"   # grey — drops out
TOPK = 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", choices=["sans", "serif"], default="sans")
    args = ap.parse_args()
    _apply_rc()
    if args.font == "serif":
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]

    df = pd.read_csv(SRC)
    df["orig"] = pd.to_numeric(df["plan_aaa_orig_rank"], errors="coerce")
    df["t1"] = pd.to_numeric(df["proxy_rank_t1"], errors="coerce")
    top = df[df["orig"] <= TOPK].dropna(subset=["orig", "t1"]).copy()
    top["survives"] = top["t1"] <= TOPK
    n_surv = int(top["survives"].sum())

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    # top-15 survival box
    ax.axhspan(0.5, TOPK + 0.5, color=C_SURV, alpha=0.06, zorder=0)
    ax.axvspan(0.5, TOPK + 0.5, color=C_SURV, alpha=0.06, zorder=0)
    lim = max(top["orig"].max(), top["t1"].max()) + 1
    ax.plot([0, lim], [0, lim], ls="--", lw=0.9, color=INK, alpha=0.6, zorder=1,
            label="perfect stability")
    ax.axhline(TOPK + 0.5, ls=":", lw=0.8, color=INK, alpha=0.5, zorder=1)

    for _, r in top.iterrows():
        surv = r["survives"]
        ax.scatter(r["orig"], r["t1"],
                   marker="*" if surv else "o",
                   s=150 if surv else 42,
                   color=C_SURV if surv else C_DROP,
                   edgecolor=INK, linewidth=0.5, zorder=3)
        if surv:
            ax.annotate(str(r["group_label"]), (r["orig"], r["t1"]),
                        xytext=(4, 3), textcoords="offset points",
                        fontsize=6, color=INK)

    ax.set_xlabel("original Plan-AAA rank (same-day OHLC)")
    ax.set_ylabel("rank after strict T−1 correction")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.invert_yaxis()
    ax.invert_xaxis()  # rank 1 (best) at top-right corner of the survival box
    ax.set_title(f"Universe-C basis is fragile: only {n_surv}/15 top groups\n"
                 "stay in the top 15 after T−1 leak correction",
                 fontsize=8.2, loc="center")
    # legend proxies
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor=C_SURV,
               markeredgecolor=INK, markersize=11, label=f"survives ({n_surv})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_DROP,
               markeredgecolor=INK, markersize=7, label=f"drops out ({len(top)-n_surv})"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, framealpha=0.9,
              facecolor="white", edgecolor="#cccccc")
    ax.text(0.99, 0.02, "exploratory caveat (Limitation L1); features run at T−1",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6, color="#666666", style="italic")
    fig.tight_layout()
    paths = save(fig, "plan_aaa_t1_stability")
    print("wrote", paths)
    print(f"survivors ({n_surv}):", list(top[top['survives']]['group_label']))


if __name__ == "__main__":
    main()
