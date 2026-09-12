#!/usr/bin/env python
"""fig_plan_aaa_t1.py — Appendix (exploratory) / Limitation L1: Universe-C basis — two importance
measures disagree.

EXPLORATORY / caveat figure. REWRITTEN 2026-09-11 (C5 sensitivity work, Codex TP1 Round A finding
A-02, verified in artifacts/plan_aaa_t1_diagnostic/group_ranking_comparison.csv):

  * Universe C is composed from the earlier Plan-AAA top-15 Alpha158 groups, ranked by NN
    permutation ΔIC on same-day-OHLC features over the 5-fold test quarters (2024-04 .. 2025-06).
  * The 2026-05-27 diagnostic re-scored every group with a single-feature |IC| proxy on T−1-shifted
    features (last 313 label days, 2024-09 .. 2025-12).
  * Only 5 of the Plan-AAA top-15 groups are also top-15 under the proxy — BUT the proxy top-15 SET IS
    IDENTICAL with and without the T−1 shift (asserted below). The disagreement therefore reflects the
    importance MEASURE (permutation ΔIC vs single-feature IC), not the one-day lag.
  * The two pure-hc groups (hc_mom12m, hc_ret_std_5d+1) are NOT scored by the proxy (no Alpha158
    members; pinned to the bottom rank by na_option="bottom") and are drawn as a separate category.
  * Both rankings are scored on quarters inside the confirmatory evaluation window (test-informed).

The previous title "only 5/15 survive T−1 leak correction" was a misstatement and is retired
(docs/analysis.md 2026-09-11-a; docs/c5_sensitivity_report_2026-09-11.md §2, §6).

Scatter: x = original Plan-AAA rank; y = proxy rank on T−1 features; a thin tick per group marks the
proxy rank on UNSHIFTED features (same top-15 set). Green stars = also top-15 under the proxy; grey
circles = not; open squares at the bottom = unscored hc groups.

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
C_IN = "#1b9e77"     # teal-green — also top-15 under the proxy
C_OUT = "#999999"    # grey — outside the proxy top-15
C_UNS = "#d95f02"    # orange — not scored by the proxy (pure-hc groups)
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
    df["raw"] = pd.to_numeric(df["proxy_rank_raw"], errors="coerce")
    df["unscored"] = df["n_alpha158_members"].astype(int) == 0   # pure-hc groups: proxy NaN → bottom

    # Invariant behind the figure's message: the proxy top-15 SET is identical with and without the
    # T−1 shift (Codex TP1-A A-02, 2026-09-10; verified on the CSV). Fail loudly if the data changes.
    set_raw = set(df.loc[(df["raw"] <= TOPK) & ~df["unscored"], "group_label"])
    set_t1 = set(df.loc[(df["t1"] <= TOPK) & ~df["unscored"], "group_label"])
    assert set_raw == set_t1 and len(set_t1) == TOPK, \
        f"proxy top-{TOPK} differs with/without the T−1 shift: raw={sorted(set_raw)} t1={sorted(set_t1)}"

    top = df[df["orig"] <= TOPK].dropna(subset=["orig", "t1"]).copy()
    top["in_proxy15"] = (top["t1"] <= TOPK) & ~top["unscored"]
    n_in = int(top["in_proxy15"].sum())
    n_uns = int(top["unscored"].sum())
    n_out = len(top) - n_in - n_uns

    fig, ax = plt.subplots(figsize=(3.6, 3.35))
    ax.axhspan(0.5, TOPK + 0.5, color=C_IN, alpha=0.06, zorder=0)
    ax.axvspan(0.5, TOPK + 0.5, color=C_IN, alpha=0.06, zorder=0)
    lim = max(top["orig"].max(), top["t1"].max()) + 1
    ax.plot([0, lim], [0, lim], ls="--", lw=0.9, color=INK, alpha=0.6, zorder=1,
            label="perfect agreement")
    ax.axhline(TOPK + 0.5, ls=":", lw=0.8, color=INK, alpha=0.5, zorder=1)

    for _, r in top.iterrows():
        if r["unscored"]:
            ax.scatter(r["orig"], r["t1"], marker="s", s=46, facecolor="white",
                       edgecolor=C_UNS, linewidth=1.0, zorder=3)
            ax.annotate(str(r["group_label"]), (r["orig"], r["t1"]), ha="right",
                        xytext=(-3, 5), textcoords="offset points", fontsize=5.4, color=C_UNS)
            continue
        # unshifted proxy rank → T−1 proxy rank: thin connector + tick (the shift moves ranks by a
        # few places but never changes the top-15 set)
        ax.plot([r["orig"], r["orig"]], [r["raw"], r["t1"]], lw=0.7, color="#bbbbbb", zorder=2)
        ax.scatter(r["orig"], r["raw"], marker="_", s=40, color="#888888", linewidth=0.9, zorder=2)
        inside = bool(r["in_proxy15"])
        ax.scatter(r["orig"], r["t1"],
                   marker="*" if inside else "o",
                   s=150 if inside else 42,
                   color=C_IN if inside else C_OUT,
                   edgecolor=INK, linewidth=0.5, zorder=3)
        if inside:
            # per-label nudges so the five labels do not sit on neighbouring stars
            off, ha = {"CORR60": ((-6, 2), "right"), "CNTP20+3": ((-6, -7), "right")}.get(
                str(r["group_label"]), ((4, 3), "left"))
            ax.annotate(str(r["group_label"]), (r["orig"], r["t1"]), ha=ha,
                        xytext=off, textcoords="offset points", fontsize=6, color=INK)

    ax.set_xlabel("Plan-AAA rank (NN permutation ΔIC, same-day OHLC)")
    ax.set_ylabel("proxy rank (single-feature |IC|, T−1 features)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.invert_yaxis()
    ax.invert_xaxis()  # rank 1 (best) at the top-right corner of the top-15 box
    ax.set_title(f"Two importance measures disagree: {n_in}/{TOPK} Plan-AAA top groups\n"
                 f"are also top-{TOPK} under the single-feature-IC proxy",
                 fontsize=8.0, loc="center")
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor=C_IN,
               markeredgecolor=INK, markersize=11, label=f"also proxy top-{TOPK} ({n_in})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_OUT,
               markeredgecolor=INK, markersize=7, label=f"outside proxy top-{TOPK} ({n_out})"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="white",
               markeredgecolor=C_UNS, markersize=7, label=f"not scored by proxy: hc ({n_uns})"),
        Line2D([0], [0], marker="_", color="#888888", lw=0.7, markersize=8,
               label="proxy rank w/o T−1 shift"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.9,
              facecolor="white", edgecolor="#cccccc", fontsize=5.6)
    ax.text(0.01, 0.02,
            "proxy top-15 identical with/without the T−1 shift;\n"
            "both rankings scored inside the evaluation window;\n"
            "exploratory (L1); runtime features at T−1",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=5.2, color="#666666", style="italic")
    fig.tight_layout()
    paths = save(fig, "plan_aaa_t1_stability")
    print("wrote", paths)
    print(f"also proxy top-{TOPK} ({n_in}):", list(top[top["in_proxy15"]]["group_label"]))
    print(f"outside proxy top-{TOPK} ({n_out}):",
          list(top[~top["in_proxy15"] & ~top["unscored"]]["group_label"]))
    print(f"unscored hc ({n_uns}):", list(top[top["unscored"]]["group_label"]))
    print("proxy top-15 set identical with/without T−1 shift: True (asserted)")


if __name__ == "__main__":
    main()
