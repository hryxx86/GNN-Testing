"""Story A — S6 (Hansen SPA forest), ST5 (Plan Z subset summary table).

# SOURCE_CONTRACT:
#   inputs:
#     - path: experiments/step3_plan_z/hansen_spa_results.csv
#       columns: [model, benchmark, T_spa, p_lower, p_consistent, p_upper,
#                 n_paired_days, t_stats]
#       md5: 608bdcf685f73fb49393bce9da48a63d
#       n_rows: 8
#     - path: experiments/step3_plan_z/part_b_summary.csv
#       columns: [subset, model, n_days, mean_IC, NW_SE, NW_t, NW_p,
#                 sharpe_point, sharpe_ci_lo, sharpe_ci_hi]
#       md5: fdb60f71f285e716ac188fbf2953f4fd
#       n_rows: 18
#   outputs:
#     - path: figures/S6_phase5_step3_subset_spa.pdf
#       headline_values:
#         - n_significant_at_05: count of rows with p_consistent < 0.05
#     - path: tables/ST5_phase5_step3.tex
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paper_figs.rcparams_storya import setup, save, write_tex_table, PALETTE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPA_CSV = PROJECT_ROOT / "experiments" / "step3_plan_z" / "hansen_spa_results.csv"
PARTB_CSV = PROJECT_ROOT / "experiments" / "step3_plan_z" / "part_b_summary.csv"


def _p_color(p: float) -> str:
    if not np.isfinite(p):
        return PALETTE["Baseline"]
    if p < 0.05:
        return PALETTE["Danger"]
    if p < 0.20:
        return PALETTE["Warning"]
    return PALETTE["Baseline"]


def fig_S6() -> tuple[dict, int]:
    df = pd.read_csv(SPA_CSV)
    df = df.dropna(subset=["T_spa", "p_consistent"])
    df = df.sort_values(["model", "benchmark"]).reset_index(drop=True)
    df["label"] = df["model"].astype(str) + " vs " + df["benchmark"].astype(str)

    fig, ax = setup("full_width", height=3.5)
    ys = np.arange(len(df))
    colors = [_p_color(p) for p in df["p_consistent"]]
    ax.scatter(df["T_spa"], ys, c=colors, s=60,
               edgecolor="black", linewidth=0.4, zorder=5)
    ax.axvline(0, color="black", linewidth=0.6, linestyle="--")

    # Annotate p_consistent next to each row (right of point)
    x_max = float(df["T_spa"].max())
    x_min = float(df["T_spa"].min())
    pad = 0.05 * (x_max - x_min + 1e-6)
    for i, (_, row) in enumerate(df.iterrows()):
        p_val = float(row["p_consistent"])
        p_str = "p<0.001" if p_val < 0.001 else f"p={p_val:.3f}"
        ax.text(row["T_spa"] + pad, i, p_str, va="center", fontsize=6)

    ax.set_yticks(ys)
    ax.set_yticklabels(df["label"], fontsize=7)
    ax.set_xlabel(r"$T_{SPA}$ (Hansen 2005)")
    ax.set_title("S6 — Hansen SPA test: subset benchmarks vs Story A models", fontsize=8)
    ax.invert_yaxis()

    # Legend (color bins)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=PALETTE["Danger"], markersize=7,
               label=r"$p_{consistent} < 0.05$"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=PALETTE["Warning"], markersize=7,
               label=r"$0.05 \leq p < 0.20$"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=PALETTE["Baseline"], markersize=7,
               label=r"$p \geq 0.20$"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=6)
    fig.tight_layout()
    paths = save(fig, "S6_phase5_step3_subset_spa")
    plt.close(fig)
    n_sig = int((df["p_consistent"] < 0.05).sum())
    return paths, n_sig


def table_ST5() -> str:
    df = pd.read_csv(PARTB_CSV)
    df = df.sort_values(["subset", "model"]).reset_index(drop=True)

    lines = [
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Subset & Model & $n_{days}$ & $\overline{IC}$ & NW-SE & NW-$t$ & NW-$p$ & $S$ & 95\% CI \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        subset = str(r["subset"]).replace("_", r"\_")
        model = str(r["model"]).replace("_", r"\_")
        ci = f"[{r['sharpe_ci_lo']:.2f}, {r['sharpe_ci_hi']:.2f}]"
        lines.append(
            f"{subset} & {model} & {int(r['n_days'])} & "
            f"{r['mean_IC']:.4f} & {r['NW_SE']:.4f} & "
            f"{r['NW_t']:.3f} & {r['NW_p']:.4f} & "
            f"{r['sharpe_point']:.3f} & {ci} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    latex = "\n".join(lines)
    write_tex_table("ST5_phase5_step3", latex)
    return latex


def write_caption(n_sig: int) -> None:
    out = PROJECT_ROOT / "tables" / "fig_phase5_step3_caption.txt"
    text = (
        f"S6 — Hansen SPA test (2005, consistent variant) for each (model, "
        f"benchmark-subset) pair. {n_sig} of 8 pairs reject SPA null at α=0.05. "
        f"Colors: red = p<0.05, amber = 0.05≤p<0.20, grey = p≥0.20.\n\n"
        "ST5 — Phase 5 Step 3 Plan Z Part B per-subset summary: 9 conditional subsets "
        "× 2 models. Reports mean IC, Newey-West SE, NW-t, NW-p, point Sharpe and "
        "95% block-bootstrap CI on Sharpe.\n"
    )
    out.write_text(text)


def main() -> None:
    s6, n_sig = fig_S6()
    table_ST5()
    write_caption(n_sig)
    print(f"[fig_phase5_step3] S6 -> {s6['pdf']}")
    print(f"[fig_phase5_step3] ST5 -> tables/ST5_phase5_step3.tex")
    print(f"[fig_phase5_step3] {n_sig}/8 pairs significant at p_consistent<0.05")


if __name__ == "__main__":
    main()
