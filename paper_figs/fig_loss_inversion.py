#!/usr/bin/env python
"""fig_loss_inversion.py — §5.7 exploratory failure mode: listwise-loss inversion.

EXPLORATORY (not confirmatory): the loss-function horse race ran on a shared scaffold
(MSE was locked for the confirmatory ladder). ListMLE's softmax-likelihood objective
inverts under cross-sectional regime shift → systematically negative mean IC, while MSE
stays positive. Conclusion-first: one panel, mean IC per loss with ±1 SE across cells,
ListMLE highlighted; a 0 reference line; banner states the takeaway.

Source: experiments/loss_horserace/results.csv (per-day IC; aggregated to per-cell mean
over (model, feature_set, fold, seed) then to a grand mean per loss).
Run: python paper_figs/fig_loss_inversion.py [--font serif]
Out: figures/loss_listmle_inversion.{pdf,png}
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

SRC = os.path.join(PROJECT_ROOT, "experiments", "loss_horserace", "results.csv")
INK = "#222222"
C_MSE = "#1b9e77"       # teal — the locked confirmatory loss
C_PAIR = "#7570b3"      # purple — pairwise
C_LISTMLE = "#e41a1c"   # red — the inverting failure mode
ORDER = ["mse", "pairwise", "listmle"]
LABEL = {"mse": "MSE\n(locked baseline)", "pairwise": "Pairwise\nlog-loss", "listmle": "ListMLE\n(listwise)"}
COLOR = {"mse": C_MSE, "pairwise": C_PAIR, "listmle": C_LISTMLE}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", choices=["sans", "serif"], default="sans")
    args = ap.parse_args()
    _apply_rc()
    if args.font == "serif":
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]

    df = pd.read_csv(SRC)
    # per-cell mean IC, then grand mean + SE across cells per loss
    cell = (df.groupby(["loss", "model", "feature_set", "fold", "seed"])["ic"]
              .mean().reset_index())
    stat = (cell.groupby("loss")["ic"]
              .agg(["mean", "std", "count"]).reindex(ORDER))
    stat["se"] = stat["std"] / np.sqrt(stat["count"])

    fig, ax = plt.subplots(figsize=(3.33, 2.5))
    xs = np.arange(len(ORDER))
    means = stat["mean"].values
    ses = stat["se"].values
    bars = ax.bar(xs, means, width=0.62,
                  color=[COLOR[k] for k in ORDER],
                  yerr=ses, capsize=3, error_kw=dict(ecolor=INK, lw=0.9))
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[k] for k in ORDER])
    ax.set_ylabel("mean IC (per-cell, seed×fold)")
    for x, m in zip(xs, means):
        ax.annotate(f"{m:+.4f}", (x, m), ha="center",
                    va="bottom" if m >= 0 else "top",
                    xytext=(0, 4 if m >= 0 else -4), textcoords="offset points",
                    fontsize=7, color=INK)
    ax.set_title("Listwise loss inverts under regime shift:\n"
                 "ListMLE mean IC < 0 while MSE stays positive",
                 fontsize=8.2, loc="center")
    ax.margins(y=0.18)
    ax.text(0.99, 0.02,
            "exploratory (not confirmatory); MSE locked for the ladder",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6, color="#666666", style="italic")
    fig.tight_layout()
    paths = save(fig, "loss_listmle_inversion")
    print("wrote", paths)
    print(stat[["mean", "se", "count"]].round(4).to_string())


if __name__ == "__main__":
    main()
