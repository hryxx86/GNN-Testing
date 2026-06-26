"""Story A — E1 anchor figures: F2, F3, F4, S1, S2, S3, S17 + T1, T2.

ADAPTATION NOTE (F2): Original plan called for "Cumulative L/S PnL curves", but
per_day_ic .npy files contain daily Spearman IC arrays (NOT daily L/S return
series). F2 is therefore implemented as "Cumulative daily IC trajectory" — a
common ranking-paper substitute that preserves the temporal-stability narrative
without inventing synthetic return data.

# SOURCE_CONTRACT:
#   inputs:
#     - path: experiments/storya_e1_anchor/results.csv
#       columns: [cell_id, universe, model, seed, fold, test_period, IC_mean,
#                 IC_std, n_test_days, Sharpe_gross, Sharpe_net_0bps, ...,
#                 Sharpe_net_30bps, mean_turnover_L1, n_periods, best_val_loss,
#                 epochs_run, wall_time_sec, converged_flag, cost_convention]
#       md5: c29851c0b4ae0457a8b3b24b6a7d6999
#       n_rows: 400
#     - path: experiments/storya_e1_anchor/per_day_ic/<universe>_<model>_s<seed>_f<fold>.npy
#       n_files: 402
#       dtype: float32, shape=(n_days_in_fold,), values=daily Spearman IC
#     - path: artifacts/storya_e6_dm_spa/bootstrap_ci.csv
#       md5: b114bb06cdf7a69104251be5966bfd99
#       n_rows: 8
#     - path: artifacts/storya_e6_dm_spa/lofo_diagnostic.csv
#       md5: 2a8bb2c59d087abbbabe2e084509abff
#       n_rows: 48
#     - path: artifacts/storya_e6_dm_spa/per_fold_table.csv
#       md5: 8cd4c55812aea797eae4b24a86dae4b5
#       n_rows: 40
#     - path: artifacts/storya_e6_dm_spa/per_cell_distribution.csv
#       md5: 513abd4359371b3581b39813cd7c6212
#       n_rows: 48
#     - path: artifacts/storya_e6_dm_spa/e1_three_column_summary.csv
#       md5: eaeb1847360e5bbf97974cb6bb201e3f
#       n_rows: 8
#   outputs:
#     - path: figures/F2_cumulative_ic_trajectory.pdf
#     - path: figures/F3_lofo_heatmap.pdf
#     - path: figures/F4_per_fold_ic_bars.pdf
#     - path: figures/S1_per_cell_ic_sharpe_scatter.pdf
#     - path: figures/S2_top_bottom_3_outliers.pdf
#     - path: figures/S3_per_day_ic_8_lines.pdf
#     - path: figures/S17_bootstrap_ci_3col.pdf
#     - path: tables/T1_headline.tex
#     - path: tables/T2_three_column_robustness.tex
#     - path: tables/fig_e1_anchor_caption.txt
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

from paper_figs.rcparams_storya import (
    setup,
    model_color,
    save,
    PALETTE,
    UNIVERSE_MARKER,
)
from paper_figs._fig_e1_anchor_tables import (
    table_T1,
    table_T2,
    write_caption,
)
from paper_figs._fig_e1_anchor_perday import (
    fig_F2,
    fig_S3,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
E1_DIR = PROJECT_ROOT / "experiments" / "storya_e1_anchor"
RESULTS_CSV = E1_DIR / "results.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "storya_e6_dm_spa"

UNIVERSES = ["B", "C"]
MODELS = ["GAT", "SAGE-Mean", "MLP", "LightGBM"]
FOLDS = [0, 1, 2, 3, 4]
LOFO_COLS = ["none", "0", "1", "2", "3", "4"]

L6_CAVEAT = ("Fold 4 (Q2-2025) is a known regime outlier; LOFO-4 column drops "
             "IC by 38-72%")


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV)
    df = df.dropna(subset=["IC_mean", "Sharpe_net_10bps", "universe", "model"])
    return df


# F2 (cumulative IC) and S3 (per-day series) live in _fig_e1_anchor_perday.py.


# --------------------------------------------------------------------------- #
# F3 — LOFO heatmap (8 rows × 6 cols)
# --------------------------------------------------------------------------- #
def fig_F3() -> dict:
    lofo = pd.read_csv(ARTIFACT_DIR / "lofo_diagnostic.csv")
    lofo["left_out_fold"] = lofo["left_out_fold"].astype(str)
    rows: list[tuple[str, str]] = [(u, m) for u in UNIVERSES for m in MODELS]
    mat = np.full((len(rows), len(LOFO_COLS)), np.nan)
    for i, (u, m) in enumerate(rows):
        for j, lof in enumerate(LOFO_COLS):
            sub = lofo[(lofo["universe"] == u) & (lofo["model"] == m) &
                       (lofo["left_out_fold"] == lof)]
            if len(sub):
                mat[i, j] = float(sub["IC_mean"].iloc[0])
    fig, ax = setup("full_width", height=3.2)
    vmax = float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 0.05
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(mat, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(LOFO_COLS)))
    ax.set_xticklabels([f"none\n(full)" if c == "none" else f"drop f{c}"
                        for c in LOFO_COLS])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{u}/{m}" for u, m in rows])
    ax.set_title("F3 — LOFO sensitivity: IC mean by left-out fold")
    for i in range(len(rows)):
        for j in range(len(LOFO_COLS)):
            v = mat[i, j]
            if np.isfinite(v):
                color = "white" if abs(v) > 0.55 * vmax else "black"
                ax.text(j, i, f"{v:.4f}", ha="center", va="center",
                        fontsize=6.5, color=color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.015)
    cbar.set_label("IC mean", fontsize=7)
    fig.tight_layout()
    paths = save(fig, "F3_lofo_heatmap")
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- #
# F4 — Per-fold IC bars (5 folds × 8 groups)
# --------------------------------------------------------------------------- #
def fig_F4() -> dict:
    pf = pd.read_csv(ARTIFACT_DIR / "per_fold_table.csv")
    fig, ax = setup("full_width", height=3.4)
    groups = [(u, m) for u in UNIVERSES for m in MODELS]
    n_groups = len(groups)  # 8
    bar_w = 0.10
    x = np.arange(len(FOLDS))
    for gi, (u, m) in enumerate(groups):
        sub = pf[(pf["universe"] == u) & (pf["model"] == m)].sort_values("fold")
        means = sub["IC_mean_mean"].to_numpy()
        stds = sub["IC_mean_std"].to_numpy()
        offset = (gi - (n_groups - 1) / 2.0) * bar_w
        hatch = "" if u == "B" else "//"
        ax.bar(x + offset, means, width=bar_w, color=model_color(m),
               edgecolor="black", linewidth=0.3, hatch=hatch,
               yerr=stds, capsize=1.2, error_kw=dict(elinewidth=0.4))
    ax.set_xticks(x)
    fold_labels = [f"f{f}" for f in FOLDS]
    fold_labels[-1] = "f4 (Q2-2025 regime)"
    ax.set_xticklabels(fold_labels)
    for label in ax.get_xticklabels():
        if "f4" in label.get_text():
            label.set_color(PALETTE["Danger"])
    ax.axhline(0, color="black", linewidth=0.4, linestyle="--")
    ax.set_ylabel("IC mean (across seeds)")
    ax.set_title("F4 — Per-fold IC (mean ± seed std)")
    # legend
    model_handles = [Patch(facecolor=model_color(m), edgecolor="black",
                           linewidth=0.3, label=m) for m in MODELS]
    univ_handles = [Patch(facecolor="white", edgecolor="black", hatch="",
                          label="Universe B"),
                    Patch(facecolor="white", edgecolor="black", hatch="//",
                          label="Universe C")]
    leg = ax.legend(handles=model_handles + univ_handles, ncol=3, fontsize=6,
                    loc="lower left")
    leg.get_frame().set_alpha(0.0)
    # L6 caveat now lives only in the caption.txt; on-figure annotation removed
    # per H博士 directive 2026-05-28.
    fig.tight_layout()
    paths = save(fig, "F4_per_fold_ic_bars")
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- #
# S1 — Per-cell IC vs Sharpe scatter (400 cells)
# --------------------------------------------------------------------------- #
def fig_S1(results: pd.DataFrame) -> dict:
    fig, ax = setup("full_width", height=3.0)
    for u in UNIVERSES:
        for m in MODELS:
            sub = results[(results["universe"] == u) & (results["model"] == m)]
            ax.scatter(sub["IC_mean"], sub["Sharpe_net_10bps"],
                       color=model_color(m), marker=UNIVERSE_MARKER[u],
                       s=14, alpha=0.55, edgecolors="black", linewidths=0.2,
                       label=f"{m} / Univ {u}")
    ax.axhline(0, color="black", linewidth=0.4, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.4, linestyle="--")
    ax.set_xlabel("IC mean (per cell)")
    ax.set_ylabel("Sharpe net (10 bps)")
    ax.set_title("S1 — Per-cell IC vs Sharpe scatter (400 cells)")
    ax.legend(ncol=4, fontsize=5.5, loc="upper left")
    fig.tight_layout()
    paths = save(fig, "S1_per_cell_ic_sharpe_scatter")
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- #
# S2 — Top-3 / bottom-3 outlier scatter
# --------------------------------------------------------------------------- #
def fig_S2() -> dict:
    pcd = pd.read_csv(ARTIFACT_DIR / "per_cell_distribution.csv")
    pcd = pcd.dropna(subset=["IC_mean", "Sharpe_net_10bps"])
    fig, ax = setup("full_width", height=3.4)
    for rank_class, color in [("TOP3_Sharpe", "#2ca02c"),
                              ("BOT3_Sharpe", PALETTE["Danger"])]:
        sub = pcd[pcd["rank_class"] == rank_class]
        ax.scatter(sub["IC_mean"], sub["Sharpe_net_10bps"], color=color,
                   s=30, alpha=0.85, edgecolors="black", linewidths=0.3,
                   label=rank_class)
        for _, row in sub.iterrows():
            ax.annotate(f"{row['universe']}/{row['model']}\n"
                        f"f{int(row['fold'])} s{int(row['seed'])}",
                        xy=(row["IC_mean"], row["Sharpe_net_10bps"]),
                        xytext=(3, 3), textcoords="offset points",
                        fontsize=5)
    ax.axhline(0, color="black", linewidth=0.4, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.4, linestyle="--")
    ax.set_xlabel("IC mean")
    ax.set_ylabel("Sharpe net (10 bps)")
    ax.set_title("S2 — Top-3 / Bottom-3 Sharpe outliers per (universe, model)")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    paths = save(fig, "S2_top_bottom_3_outliers")
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- #
# S17 — Bootstrap CI overlay: full / lofo4 / fold4only
# --------------------------------------------------------------------------- #
def fig_S17() -> dict:
    summ = pd.read_csv(ARTIFACT_DIR / "e1_three_column_summary.csv")
    fig, ax = setup("full_width", height=4.0)
    rows = [(u, m) for u in UNIVERSES for m in MODELS]
    regimes = [("full", PALETTE["GAT"]),
               ("lofo4", PALETTE["Warning"]),
               ("fold4only", PALETTE["Danger"])]
    y_positions = np.arange(len(rows))
    dodge = 0.22
    for ri, (regime_key, color) in enumerate(regimes):
        offsets = (ri - (len(regimes) - 1) / 2.0) * dodge
        means: list[float] = []
        los: list[float] = []
        his: list[float] = []
        for u, m in rows:
            sub = summ[(summ["universe"] == u) & (summ["model"] == m)]
            if len(sub) == 0:
                means.append(np.nan); los.append(np.nan); his.append(np.nan)
                continue
            means.append(float(sub[f"IC_{regime_key}_mean"].iloc[0]))
            los.append(float(sub[f"IC_{regime_key}_ci_lo"].iloc[0]))
            his.append(float(sub[f"IC_{regime_key}_ci_hi"].iloc[0]))
        means_a = np.array(means)
        los_a = np.array(los)
        his_a = np.array(his)
        xerr = np.vstack([means_a - los_a, his_a - means_a])
        ax.errorbar(means_a, y_positions + offsets, xerr=xerr, fmt="o",
                    color=color, ecolor=color, capsize=2, markersize=4,
                    linewidth=0.8, label=regime_key)
    ax.axvline(0, color="black", linewidth=0.4, linestyle="--")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{u}/{m}" for u, m in rows])
    ax.invert_yaxis()
    ax.set_xlabel("IC mean with 95% bootstrap CI")
    ax.set_title("S17 — Bootstrap CI: full vs LOFO-4 vs Fold-4-only")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    paths = save(fig, "S17_bootstrap_ci_3col")
    plt.close(fig)
    return paths


def main() -> None:
    results = load_results()
    fig_F2()
    fig_F3()
    fig_F4()
    fig_S1(results)
    fig_S2()
    fig_S3()
    fig_S17()
    table_T1()
    table_T2()
    write_caption()
    print("[fig_e1_anchor] OK — 7 figures + 2 tables + caption written")


if __name__ == "__main__":
    main()
