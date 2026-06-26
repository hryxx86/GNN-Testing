"""Story A — S12 (SelectiveNet coverage × IC curves).

# SOURCE_CONTRACT:
#   inputs:
#     - path: experiments/selectivenet_results.csv
#       columns: [target_coverage, actual_coverage, IC, IC_std, n_days,
#                 strategy, target]
#       md5: 26b044c210c9389acc5a9d44f7541592
#       n_rows: 70
#   outputs:
#     - path: figures/S12_selectivenet_coverage_ic.pdf
#       headline_values:
#         - threshold_peak_ic: max IC of Threshold strategy
#         - threshold_peak_coverage: actual_coverage at threshold peak
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paper_figs.rcparams_storya import setup, save, PALETTE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "experiments" / "selectivenet_results.csv"


def fig_S12() -> tuple[dict, float, float]:
    df = pd.read_csv(CSV_PATH).dropna(subset=["actual_coverage", "IC", "strategy"])

    strategies = sorted(df["strategy"].unique())
    color_map = {"Threshold": PALETTE["GAT"], "E2E": PALETTE["MLP"]}

    fig, ax = setup("single_col", height=2.6)
    peak_ic = np.nan
    peak_cov = np.nan
    for strat in strategies:
        sub = df[df["strategy"] == strat].sort_values("actual_coverage")
        if len(sub) == 0:
            continue
        color = color_map.get(strat, PALETTE["Baseline"])
        ax.plot(sub["actual_coverage"], sub["IC"],
                marker="o", color=color, linewidth=1.1,
                markersize=3.5, label=strat)
        if strat == "Threshold":
            idx = int(sub["IC"].idxmax())
            peak_ic = float(sub.loc[idx, "IC"])
            peak_cov = float(sub.loc[idx, "actual_coverage"])
            ax.scatter([peak_cov], [peak_ic], marker="*", s=120,
                       color=PALETTE["Warning"], edgecolor="black",
                       linewidth=0.4, zorder=6)
            ax.annotate(
                f"peak IC={peak_ic:.4f}\ncov={peak_cov:.2f}",
                xy=(peak_cov, peak_ic),
                xytext=(8, 8), textcoords="offset points",
                fontsize=6, color="black",
                arrowprops=dict(arrowstyle="-", lw=0.4, color="black"),
            )

    ax.set_xlabel("Actual coverage")
    ax.set_ylabel("IC")
    ax.set_title("S12 — SelectiveNet: coverage × IC", fontsize=8)
    ax.legend(loc="best", fontsize=6)
    fig.tight_layout()
    paths = save(fig, "S12_selectivenet_coverage_ic")
    plt.close(fig)
    return paths, peak_ic, peak_cov


def write_caption(peak_ic: float, peak_cov: float) -> None:
    out = PROJECT_ROOT / "tables" / "fig_selectivenet_caption.txt"
    text = (
        f"S12 — SelectiveNet coverage × IC. One line per strategy "
        f"(Threshold post-hoc vs E2E learnable gate). Threshold peak IC = "
        f"{peak_ic:.4f} at actual coverage {peak_cov:.2f} (star marker).\n"
    )
    out.write_text(text)


def main() -> None:
    s12, peak_ic, peak_cov = fig_S12()
    write_caption(peak_ic, peak_cov)
    print(f"[fig_selectivenet] S12 -> {s12['pdf']}")
    print(f"[fig_selectivenet] Threshold peak IC = {peak_ic:.4f} at coverage {peak_cov:.2f}")


if __name__ == "__main__":
    main()
