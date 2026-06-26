#!/usr/bin/env python
"""fig_regime.py — §5.3 regime dependence / LOFO: signal is concentrated in a few quarters.

(a) Per-fold IC heatmap: tuned ladder arms (rows) × 12 walk-forward quarters (cols), Univ C,
    seed-averaged IC, diverging colour centred at 0.
(b) Per-quarter mean IC (averaged over arms) for Univ B & C — about half the 12 quarters are
    ≈0 or negative; a few (esp. 2024Q4, 2025Q2) carry most of the signal.

This is why headline IC is fragile under leave-one-fold-out (family1_lofo.csv): dropping a single
strong quarter materially moves the mean. Degenerate C/L5s cells EXCLUDED (matches Family-1).

Sources: experiments/storya_v21_main12_tuned/results.csv (+ L7 from the l7 staging dir)
Run: python paper_figs/fig_regime.py [--font serif]
Out: figures/regime_perfold_ic.{pdf,png}
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from paper_figs.rcparams_storya import _apply_rc, save  # noqa: E402

MAIN = os.path.join(PROJECT_ROOT, "experiments", "storya_v21_main12_tuned", "results.csv")
L7 = os.path.join(PROJECT_ROOT, "experiments", "_rerun_colab_staging",
                  "storya_v21_l7_hats_tuned", "results.csv")
INK = "#222222"
ARM = {"L0": "LGB", "L1": "MLP", "L2": "GAT", "L3": "+news", "L4": "+sector",
       "L5": "+all", "L6": "FullAttn", "L7": "HATS"}
ARM_ORDER = ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7"]
QUARTERS = ["23Q1", "23Q2", "23Q3", "23Q4", "24Q1", "24Q2",
            "24Q3", "24Q4", "25Q1", "25Q2", "25Q3", "25Q4"]


def load():
    m = pd.read_csv(MAIN)
    if os.path.exists(L7):
        m = pd.concat([m, pd.read_csv(L7)], ignore_index=True)
    ref = m.groupby(["universe", "fold"])["n_test_days"].transform("max")
    m = m[m.n_test_days >= ref]                       # EXCLUDE degenerate cells
    perfold = m.groupby(["universe", "arm", "fold"])["IC_mean"].mean().reset_index()
    perq = m.groupby(["universe", "fold"])["IC_mean"].mean().reset_index()
    return perfold, perq


def panel_heat(ax, cax, perfold):
    sub = perfold[(perfold.universe == "C")]
    M = np.full((len(ARM_ORDER), 12), np.nan)
    for i, arm in enumerate(ARM_ORDER):
        r = sub[sub.arm == arm].set_index("fold")["IC_mean"]
        for f in range(12):
            if f in r.index:
                M[i, f] = r[f]
    vmax = 0.12
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for i in range(len(ARM_ORDER)):
        for f in range(12):
            v = M[i, f]
            if not np.isnan(v):
                ax.text(f, i, f"{v:.02f}".replace("0.", ".").replace("-.", "−."),
                        ha="center", va="center", fontsize=5.2,
                        color="white" if abs(v) > 0.07 else "#333333")
    ax.set_xticks(range(12))
    ax.set_xticklabels([])
    ax.set_yticks(range(len(ARM_ORDER)))
    ax.set_yticklabels([ARM[a] for a in ARM_ORDER])
    ax.set_title("(a)  Per-quarter IC by arm — Universe C", loc="left", fontweight="bold")
    ax.tick_params(length=0)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label("seed-avg IC", fontsize=6.6)
    cb.ax.tick_params(labelsize=6)


def panel_bars(ax, perq):
    x = np.arange(12)
    w = 0.4
    for u, off, col, lab in [("B", -w / 2, "#5a6b8c", "Univ B"), ("C", w / 2, "#1b9e77", "Univ C")]:
        r = perq[perq.universe == u].set_index("fold")["IC_mean"]
        ax.bar(x + off, [r.get(f, np.nan) for f in range(12)], width=w,
               color=col, edgecolor=INK, linewidth=0.3, label=lab)
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(QUARTERS, rotation=45, ha="right")
    ax.set_xlim(-0.6, 11.6)
    ax.set_ylabel("mean IC\n(over arms)")
    ax.set_title("(b)  Per-quarter mean IC (averaged over arms)", loc="left", fontweight="bold")
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#dddddd",
              framealpha=0.96, fontsize=6.6, ncol=2, handletextpad=0.4,
              columnspacing=1.0).set_zorder(10)


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

    perfold, perq = load()
    fig = plt.figure(figsize=(7.1, 4.7))
    gs = GridSpec(2, 2, width_ratios=[1.0, 0.022], height_ratios=[1.5, 1.0],
                  left=0.085, right=0.95, top=0.85, bottom=0.135, hspace=0.20, wspace=0.02, figure=fig)
    panel_heat(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), perfold)
    panel_bars(fig.add_subplot(gs[1, 0]), perq)

    fig.text(0.5, 0.965,
             "Signal is regime-concentrated: a few quarters (esp. 2024Q4, 2025Q2) carry most of the IC; ~half are ≈0 or negative",
             ha="center", va="top", fontsize=8.6, fontweight="bold", color=INK)
    fig.text(0.5, 0.012,
             "→ headline IC is fragile under leave-one-fold-out (dropping one strong quarter moves the mean).",
             ha="center", va="bottom", fontsize=6.6, style="italic", color="#555555")
    paths = save(fig, "regime_perfold_ic" + suffix)
    plt.close(fig)
    print(f"[regime] wrote {paths['pdf']}")


if __name__ == "__main__":
    main()
