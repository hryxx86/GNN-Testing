"""Story A — E6 cost-ladder figures: F5, T4.

# SOURCE_CONTRACT:
#   inputs:
#     - path: artifacts/storya_e6_dm_spa/cost_ladder.csv
#       md5: a6a4751dde81f795412249cbcddc7c4c
#       n_rows: 48
#       columns: [universe, model, cost_bps, n_cells, Sharpe_net_mean,
#                 Sharpe_net_std, Sharpe_net_ci_lo, Sharpe_net_ci_hi]
#   outputs:
#     - path: figures/F5_cost_ladder.pdf
#       headline_values:
#         - net_sharpe_at_10bps_per_cell
#     - path: tables/T4_cost_ladder.tex
#     - path: tables/fig_e6_cost_ladder_caption.txt

# CAPTION REQUIREMENT (L1, verbatim):
#   "Universe C composition derives from Plan AAA which had same-day Alpha158
#    leak; T-1 diagnostic confirms LOW STABILITY (5/15)"
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paper_figs.rcparams_storya import (
    setup,
    model_color,
    save,
    write_tex_table,
    PALETTE,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "artifacts" / "storya_e6_dm_spa" / "cost_ladder.csv"

UNIVERSES = ["B", "C"]
MODELS = ["GAT", "SAGE-Mean", "MLP", "LightGBM"]
BPS_LEVELS = [0, 5, 10, 15, 20, 30]

L1_CAVEAT = ("Universe C composition derives from Plan AAA which had same-day "
             "Alpha158 leak; T-1 diagnostic confirms LOW STABILITY (5/15)")


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["Sharpe_net_mean", "cost_bps"])
    return df


# --------------------------------------------------------------------------- #
# F5 — Net Sharpe vs cost (8 lines)
# --------------------------------------------------------------------------- #
def fig_F5(df: pd.DataFrame) -> dict:
    fig, ax = setup("full_width", height=3.6)
    for u in UNIVERSES:
        for m in MODELS:
            sub = df[(df["universe"] == u) & (df["model"] == m)].sort_values(
                "cost_bps")
            if len(sub) == 0:
                continue
            xs = sub["cost_bps"].to_numpy()
            ys = sub["Sharpe_net_mean"].to_numpy()
            los = sub["Sharpe_net_ci_lo"].to_numpy()
            his = sub["Sharpe_net_ci_hi"].to_numpy()
            ls = "-" if u == "B" else "--"
            ax.plot(xs, ys, color=model_color(m), linestyle=ls, linewidth=1.1,
                    marker="o" if u == "B" else "s", markersize=4,
                    label=f"{m} / Univ {u}")
            ax.fill_between(xs, los, his, color=model_color(m), alpha=0.08,
                            linewidth=0)
    ax.axhline(0, color="black", linewidth=0.4, linestyle="--")
    ax.axhline(1.0, color=PALETTE["Highlight"], linewidth=0.7, linestyle=":",
               label="Sharpe = 1.0 (ICAIF benchmark)")
    ax.set_xticks(BPS_LEVELS)
    ax.set_xlabel("Transaction cost (bps, one-way L1)")
    ax.set_ylabel("Mean net Sharpe (across cells; 95% CI band)")
    ax.set_title("F5 — Cost-ladder: net Sharpe vs cost")
    ax.legend(ncol=3, fontsize=5.5, loc="upper right")
    fig.tight_layout()
    paths = save(fig, "F5_cost_ladder")
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- #
# T4 — Pivot table (8 (universe, model) rows × 6 bps cols)
# --------------------------------------------------------------------------- #
def table_T4(df: pd.DataFrame) -> str:
    lines = [
        r"\footnotesize",
        r"\begin{tabular}{lll" + "r" * len(BPS_LEVELS) + "}",
        r"\toprule",
        r"Univ & Model & $n_{cells}$ & " +
        " & ".join([f"{b} bps" for b in BPS_LEVELS]) + r" \\",
        r"\midrule",
    ]
    for u in UNIVERSES:
        for m in MODELS:
            sub = df[(df["universe"] == u) & (df["model"] == m)]
            if len(sub) == 0:
                continue
            n_cells = int(sub["n_cells"].iloc[0])
            cells = []
            for b in BPS_LEVELS:
                r = sub[sub["cost_bps"] == b]
                if len(r):
                    mean = float(r["Sharpe_net_mean"].iloc[0])
                    lo = float(r["Sharpe_net_ci_lo"].iloc[0])
                    hi = float(r["Sharpe_net_ci_hi"].iloc[0])
                    cells.append(f"{mean:.2f}\\;[{lo:.2f},{hi:.2f}]")
                else:
                    cells.append("--")
            lines.append(f"{u} & {m} & {n_cells} & " + " & ".join(cells) +
                         r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    latex = "\n".join(lines)
    write_tex_table("T4_cost_ladder", latex)
    return latex


def write_caption() -> None:
    out = PROJECT_ROOT / "tables" / "fig_e6_cost_ladder_caption.txt"
    text = (
        f"F5 — Cost-ladder: mean net Sharpe vs one-way L1 transaction cost "
        f"(bps), 8 lines (4 models × 2 universes). Solid lines = Universe B, "
        f"dashed = Universe C; shaded bands = 95% bootstrap CI. Dotted "
        f"reference at Sharpe = 1.0 (common ICAIF benchmark). LIMITATION "
        f"L1: {L1_CAVEAT}.\n\n"
        f"T4 — Cost-ladder pivot table. Mean net Sharpe with 95% CI for each "
        f"(universe, model) at 6 bps levels (0/5/10/15/20/30). LIMITATION "
        f"L1: {L1_CAVEAT}.\n"
    )
    out.write_text(text)


def main() -> None:
    df = load()
    fig_F5(df)
    table_T4(df)
    write_caption()
    print("[fig_e6_cost_ladder] OK — F5 + T4 + caption")


if __name__ == "__main__":
    main()
