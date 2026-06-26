"""Story A — S10 (LGB permutation importance), S11 (sector attribution area).

# SOURCE_CONTRACT:
#   inputs:
#     - path: experiments/diag_phase5_permutation_importance_lgb.csv
#       columns: [feature, shuffled_IC, delta_IC]
#       md5: 4d7b834bab1f9cc3f412332cda6e6c3a
#       n_rows: 9
#     - path: experiments/diag_sector_attribution_sage_mean.csv
#       columns: [date, sector, long_contrib, short_contrib, ls_contrib,
#                 n_long, n_short]
#       md5: 44e48e090a7d658e45717bb3a949f52f
#       n_rows: 1408
#   outputs:
#     - path: figures/S10_lgb_perm_importance.pdf
#     - path: figures/S11_sector_attribution_area.pdf
#
# DEVIATION FROM ORIGINAL PROMPT — the permutation-importance CSV only has 9 rows
# (single-snapshot, no fold/seed breakdown). Adapted from "stacked bar per fold"
# to a single-snapshot bar chart sorted by |delta_IC|.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from paper_figs.rcparams_storya import setup, save, PALETTE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERM_CSV = PROJECT_ROOT / "experiments" / "diag_phase5_permutation_importance_lgb.csv"
SECTOR_CSV = PROJECT_ROOT / "experiments" / "diag_sector_attribution_sage_mean.csv"


def fig_S10() -> dict:
    df = pd.read_csv(PERM_CSV).dropna(subset=["delta_IC", "feature"])
    df["abs_delta"] = df["delta_IC"].abs()
    df = df.sort_values("abs_delta", ascending=False).reset_index(drop=True)

    fig, ax = setup("single_col", height=2.6)
    colors = [PALETTE["GAT"] if d > 0 else PALETTE["Danger"]
              for d in df["delta_IC"]]
    xs = np.arange(len(df))
    ax.bar(xs, df["delta_IC"], color=colors,
           edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(df["feature"], rotation=40, ha="right", fontsize=6)
    ax.set_ylabel("ΔIC after shuffle")
    ax.set_title("S10 — LightGBM permutation importance (single snapshot)", fontsize=8)
    fig.tight_layout()
    paths = save(fig, "S10_lgb_perm_importance")
    plt.close(fig)
    return paths


def fig_S11() -> dict:
    """Split into two panels: long_contrib (positive-only stack) and
    short_contrib (positive-only stack). Avoids the negative-stackplot visual
    confusion of plotting signed ls_contrib in a single stack.
    """
    df = pd.read_csv(SECTOR_CSV).dropna(
        subset=["date", "sector", "long_contrib", "short_contrib"]
    )
    df["date"] = pd.to_datetime(df["date"])

    pivot_long = df.pivot_table(index="date", columns="sector",
                                values="long_contrib", aggfunc="sum").fillna(0.0)
    pivot_short = df.pivot_table(index="date", columns="sector",
                                 values="short_contrib", aggfunc="sum").fillna(0.0)
    sectors = sorted(set(pivot_long.columns) | set(pivot_short.columns))
    # Align columns
    pivot_long = pivot_long.reindex(columns=sectors, fill_value=0.0)
    pivot_short = pivot_short.reindex(columns=sectors, fill_value=0.0)

    base = cm.tab20.colors[:max(len(sectors), 11)]
    palette = list(base[:len(sectors)])

    fig, axes = setup("two_panel", height=3.4)
    axes[0].stackplot(pivot_long.index, pivot_long.T.values,
                      labels=sectors, colors=palette,
                      edgecolor="none", alpha=0.85)
    axes[0].set_title("Long contribution by sector", fontsize=8)
    axes[0].set_ylabel("Contribution")
    axes[1].stackplot(pivot_short.index, pivot_short.T.values,
                      labels=sectors, colors=palette,
                      edgecolor="none", alpha=0.85)
    axes[1].set_title("Short contribution by sector", fontsize=8)

    for ax in axes:
        ax.set_xlabel("Date")
        ax.axhline(0, color="black", linewidth=0.4)

    axes[1].legend(loc="upper left", fontsize=5, ncol=3, frameon=False,
                   bbox_to_anchor=(0.0, -0.22))
    fig.suptitle("S11 — SAGE-Mean sector attribution (long / short separated)",
                 fontsize=8, y=1.02)
    fig.tight_layout()
    paths = save(fig, "S11_sector_attribution_area")
    plt.close(fig)
    return paths


def write_caption() -> None:
    out = PROJECT_ROOT / "tables" / "fig_phase5_diagnostics_caption.txt"
    text = (
        "S10 — LightGBM permutation importance, single-snapshot per feature "
        "(no fold or seed breakdown — adapted from the original plan's stacked "
        "bar specification to a single bar chart). Bars sorted descending by "
        "|ΔIC|; green = positive ΔIC after shuffle (feature was harmful), "
        "red = negative (feature was useful).\n\n"
        "S11 — SAGE-Mean per-day long/short contribution by GICS sector, split into "
        "two panels (long-side and short-side, both positive-only stacks). "
        "Splitting avoids the mirrored-around-zero visual that a signed ls_contrib "
        "stackplot would produce. 11-color qualitative palette shared across panels.\n"
    )
    out.write_text(text)


def main() -> None:
    s10 = fig_S10()
    s11 = fig_S11()
    write_caption()
    print(f"[fig_phase5_diagnostics] S10 -> {s10['pdf']}")
    print(f"[fig_phase5_diagnostics] S11 -> {s11['pdf']}")


if __name__ == "__main__":
    main()
