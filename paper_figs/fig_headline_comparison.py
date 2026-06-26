"""Story A — F11 headline IC + Sharpe horizontal-bar comparison (2 panels).

Direct side-by-side comparison of all 8 (universe, model) cells on IC and
Sharpe simultaneously. Single figure that answers "do these models actually
have IC and Sharpe > 0?" — the most basic N1 narrative question.

Per H博士 directive 2026-05-28: candidate B (双 panel 横向 bar with CI).

# SOURCE_CONTRACT:
#   inputs:
#     - path: artifacts/storya_e6_dm_spa/bootstrap_ci.csv
#       columns: [universe, model, n_per_day_obs, n_cells, IC_mean, IC_mean_ci_lo,
#                 IC_mean_ci_hi, Sharpe_gross_mean, Sharpe_gross_std,
#                 Sharpe_gross_ci_lo, Sharpe_gross_ci_hi]
#       md5: b114bb06cdf7a69104251be5966bfd99
#       n_rows: 8
#   outputs:
#     - path: figures/F11_headline_ic_sharpe_bars.pdf
#       headline_values:
#         - 8 (universe, model) cells with paired IC mean + 95% bootstrap CI
#         - 8 cells with Sharpe_gross mean + 95% bootstrap CI
#         - Univ B GAT IC = 0.0355 [0.0181, 0.0526]; Sharpe_gross = 1.50 [0.82, 2.20]
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from paper_figs.rcparams_storya import setup, model_color, save, PALETTE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "artifacts" / "storya_e6_dm_spa" / "bootstrap_ci.csv"

UNIVERSES = ["B", "C"]
MODELS = ["GAT", "SAGE-Mean", "MLP", "LightGBM"]


def _ordered_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return df reindexed as Univ B (4 models) then Univ C (4 models), top→bottom."""
    rows = []
    for u in UNIVERSES:
        for m in MODELS:
            sub = df[(df["universe"] == u) & (df["model"] == m)]
            if len(sub) == 0:
                continue
            r = sub.iloc[0].to_dict()
            rows.append(r)
    return pd.DataFrame(rows)


def fig_F11() -> dict:
    df = pd.read_csv(CSV_PATH)
    df = _ordered_rows(df)
    n = len(df)
    assert n == 8, f"Expected 8 cells, got {n}"

    # y positions: Univ B at top (y=0..3), small gap, Univ C below (y=5..8)
    y_positions = []
    for i, row in df.iterrows():
        univ = row["universe"]
        within_idx = MODELS.index(row["model"])
        if univ == "B":
            y_positions.append(within_idx)
        else:
            y_positions.append(5 + within_idx)
    y_positions = np.array(y_positions)

    fig, axes = setup("two_panel", height=3.6)
    ax_ic, ax_sh = axes

    # ---- LEFT PANEL: IC mean ± 95% CI ----
    for i, row in df.iterrows():
        y = y_positions[i]
        m = row["model"]
        u = row["universe"]
        mean = float(row["IC_mean"])
        lo = float(row["IC_mean_ci_lo"])
        hi = float(row["IC_mean_ci_hi"])
        color = model_color(m)
        hatch = "" if u == "B" else "//"
        # Horizontal bar from 0 to mean
        ax_ic.barh(y, mean, height=0.7, color=color, edgecolor="black",
                   linewidth=0.4, hatch=hatch, alpha=0.85)
        # CI error bar overlay
        ax_ic.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]],
                       fmt="none", ecolor="black", capsize=2.5, linewidth=0.7)
        # Annotate point estimate
        text_x = max(hi, mean) + 0.003
        ax_ic.text(text_x, y, f"{mean:.4f}", va="center", ha="left",
                   fontsize=6.5, color="#1a202c")

    ax_ic.axvline(0, color="black", linewidth=0.5, linestyle="--")
    ax_ic.set_yticks(y_positions)
    ax_ic.set_yticklabels([f"{r['universe']}/{r['model']}" for _, r in df.iterrows()],
                          fontsize=7)
    ax_ic.invert_yaxis()
    ax_ic.set_xlabel("IC mean (95% bootstrap CI)")
    ax_ic.set_title("IC mean", fontsize=8)

    # ---- RIGHT PANEL: Sharpe_gross ± 95% CI ----
    for i, row in df.iterrows():
        y = y_positions[i]
        m = row["model"]
        u = row["universe"]
        mean = float(row["Sharpe_gross_mean"])
        lo = float(row["Sharpe_gross_ci_lo"])
        hi = float(row["Sharpe_gross_ci_hi"])
        color = model_color(m)
        hatch = "" if u == "B" else "//"
        ax_sh.barh(y, mean, height=0.7, color=color, edgecolor="black",
                   linewidth=0.4, hatch=hatch, alpha=0.85)
        ax_sh.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]],
                       fmt="none", ecolor="black", capsize=2.5, linewidth=0.7)
        text_x = max(hi, mean) + 0.05
        ax_sh.text(text_x, y, f"{mean:.2f}", va="center", ha="left",
                   fontsize=6.5, color="#1a202c")

    ax_sh.axvline(0, color="black", linewidth=0.5, linestyle="--")
    ax_sh.axvline(1.0, color=PALETTE["Highlight"], linewidth=0.6, linestyle=":")
    ax_sh.set_yticks(y_positions)
    ax_sh.set_yticklabels([])  # share with left panel visually
    ax_sh.invert_yaxis()
    ax_sh.set_xlabel("Sharpe_gross (95% bootstrap CI)")
    ax_sh.set_title("Sharpe_gross", fontsize=8)

    # Shared legend at bottom — model colors + universe hatches
    model_handles = [
        Patch(facecolor=model_color(m), edgecolor="black", linewidth=0.3, label=m)
        for m in MODELS
    ]
    univ_handles = [
        Patch(facecolor="white", edgecolor="black", hatch="", label="Universe B"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="Universe C"),
    ]
    fig.legend(handles=model_handles + univ_handles, ncol=6, fontsize=6,
               loc="lower center", bbox_to_anchor=(0.5, -0.05), frameon=False)

    # Light dividing line between Univ B and Univ C row groups (between y=3 and y=5)
    for ax in axes:
        ax.axhline(4, color=PALETTE["Baseline"], linewidth=0.3, alpha=0.5, linestyle=":")

    fig.suptitle(
        "F11 — Headline IC and Sharpe by (universe, model), 95% bootstrap CI",
        y=1.01, fontsize=9,
    )
    fig.tight_layout()
    paths = save(fig, "F11_headline_ic_sharpe_bars")
    plt.close(fig)
    return paths


def write_caption() -> None:
    out = PROJECT_ROOT / "tables" / "fig_headline_comparison_caption.txt"
    text = (
        "F11 — Headline IC mean (left panel) and Sharpe_gross mean (right panel) "
        "for 8 (universe, model) cells with 95% bootstrap CIs. Universe B group "
        "(rows 1–4) shown above Universe C group (rows 5–8); dashed grey line "
        "separates the two universe groups. Solid bars = Universe B; hatched "
        "(//) bars = Universe C. Bar colours: model identity (GAT teal, "
        "SAGE-Mean orange, MLP purple, LightGBM magenta). Black error bars = "
        "95% bootstrap CI on the mean. Vertical dashed black line marks 0; "
        "dotted blue line on right panel marks Sharpe = 1.0 (common ICAIF "
        "benchmark). Headline numbers: Univ B GAT IC=0.0355 [0.0181, 0.0526], "
        "Sharpe_gross=1.50 [0.82, 2.20]; Univ B SAGE-Mean IC=0.0320 "
        "[0.0144, 0.0498]. 7/8 cells have IC CI excluding 0 (Univ B LightGBM "
        "is the sole exception). N1 headline: small but consistently positive "
        "IC and gross Sharpe across most (universe, model) cells; see T2 / S17 "
        "for the same numbers split by LOFO-4 / Fold-4-only regimes.\n"
    )
    out.write_text(text)


def main() -> None:
    paths = fig_F11()
    write_caption()
    print(f"[fig_headline_comparison] F11 -> {paths['pdf']}")


if __name__ == "__main__":
    main()
