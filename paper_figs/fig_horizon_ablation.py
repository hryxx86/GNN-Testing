"""Story A — F7 (horizon×arch heatmap), F8 (news dilution forest), ST3 (full table).

Reads experiments/horizon_ablation_results.csv (360 rows = 4 models × 6 horizons × 3 seeds × 5 folds).

# SOURCE_CONTRACT:
#   inputs:
#     - path: experiments/horizon_ablation_results.csv
#       columns: [model, seed, fold, horizon, test_period, IC, IC_std, n_days,
#                 Sharpe_gross, Sharpe_net, n_periods, mean_turnover]
#       md5: dae8089fb12df086cef20a412fc057c1
#       n_rows: 360
#   outputs:
#     - path: figures/F7_horizon_arch_heatmap.pdf
#       headline_values:
#         - cell_annotation: IC mean over 15 cells (3 seeds x 5 folds)
#     - path: figures/F8_news_dilution_forest.pdf
#       headline_values:
#         - mlp_21d_delta_ic: ΔIC(MLP_all - MLP_price) at h=21d
#         - sage_21d_delta_ic: ΔIC(SAGE-Mean_all - SAGE-Mean_price) at h=21d
#     - path: tables/ST3_horizon_full.tex
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

from paper_figs.rcparams_storya import (
    setup,
    model_color,
    save,
    write_tex_table,
    PALETTE,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "experiments" / "horizon_ablation_results.csv"

MODELS = ["MLP_price", "MLP_all", "SAGE-Mean_price", "SAGE-Mean_all"]
HORIZONS = [1, 5, 10, 21, 42, 63]


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    # Defensive: drop rows with NaN in IC (none expected, but document)
    df = df.dropna(subset=["IC", "horizon", "model"])
    return df


def _cells_per(df: pd.DataFrame, model: str, horizon: int) -> pd.DataFrame:
    """Return per-(seed, fold) IC rows for given (model, horizon), sorted for paired joins."""
    sub = df[(df["model"] == model) & (df["horizon"] == horizon)]
    return sub[["seed", "fold", "IC"]].sort_values(["seed", "fold"]).reset_index(drop=True)


def _paired_delta(df: pd.DataFrame, m_all: str, m_price: str, horizon: int) -> np.ndarray:
    """Inner-join on (seed, fold); return paired delta-IC array (all − price)."""
    a = _cells_per(df, m_all, horizon)
    p = _cells_per(df, m_price, horizon)
    merged = a.merge(p, on=["seed", "fold"], suffixes=("_all", "_price"))
    return (merged["IC_all"] - merged["IC_price"]).to_numpy()


def fig_F7(df: pd.DataFrame) -> dict:
    mat = np.full((len(MODELS), len(HORIZONS)), np.nan)
    for i, m in enumerate(MODELS):
        for j, h in enumerate(HORIZONS):
            cells = _cells_per(df, m, h)["IC"].to_numpy()
            if len(cells):
                mat[i, j] = float(np.mean(cells))

    fig, ax = setup("full_width", height=2.8)
    vmax = float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 0.05
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(mat, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels([f"{h}d" for h in HORIZONS])
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS)
    ax.set_xlabel("Horizon (trading days)")
    ax.set_title("F7 — Horizon × Architecture IC (mean over 3 seeds × 5 folds)")
    for i in range(len(MODELS)):
        for j in range(len(HORIZONS)):
            v = mat[i, j]
            if np.isfinite(v):
                color = "white" if abs(v) > 0.6 * vmax else "black"
                ax.text(j, i, f"{v:.4f}", ha="center", va="center",
                        fontsize=7, color=color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.015)
    cbar.set_label("IC", fontsize=7)
    fig.tight_layout()
    paths = save(fig, "F7_horizon_arch_heatmap")
    plt.close(fig)
    return paths


def _bootstrap_paired_delta_ci(deltas: np.ndarray, B: int = 1000,
                               seed: int = 42) -> tuple[float, float, float]:
    """Paired bootstrap on per-(seed,fold) delta-IC array.

    Resamples the matched delta values themselves (one delta per (seed,fold) pair).
    This preserves the same-(seed,fold) correlation that the design provides; iid
    resampling of cells_all and cells_price independently overstates CI width.
    """
    rng = np.random.default_rng(seed)
    n = len(deltas)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(deltas.mean())
    boots = np.empty(B)
    for b in range(B):
        boots[b] = deltas[rng.integers(0, n, size=n)].mean()
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, float(lo), float(hi)


def fig_F8(df: pd.DataFrame) -> tuple[dict, float, float]:
    fig, axes = setup("two_panel", height=3.0)
    arches = [("MLP", "MLP_all", "MLP_price"),
              ("SAGE-Mean", "SAGE-Mean_all", "SAGE-Mean_price")]
    mlp_21d = sage_21d = np.nan
    for ax, (arch, m_all, m_price) in zip(axes, arches):
        ys = np.arange(len(HORIZONS))
        points = np.empty(len(HORIZONS))
        los = np.empty(len(HORIZONS))
        his = np.empty(len(HORIZONS))
        for i, h in enumerate(HORIZONS):
            deltas = _paired_delta(df, m_all, m_price, h)
            p, lo, hi = _bootstrap_paired_delta_ci(deltas, B=1000, seed=42)
            points[i] = p
            los[i] = lo
            his[i] = hi
            if h == 21:
                if arch == "MLP":
                    mlp_21d = p
                else:
                    sage_21d = p
        ax.errorbar(points, ys, xerr=[points - los, his - points],
                    fmt="o", color=model_color(arch),
                    ecolor=PALETTE["Baseline"], capsize=2.5,
                    markersize=5, linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{h}d" for h in HORIZONS])
        ax.set_xlabel("ΔIC (all − price)")
        ax.set_title(f"{arch}: news-feature dilution")
        ax.invert_yaxis()
    axes[0].set_ylabel("Horizon")
    fig.suptitle("F8 — News-feature dilution (paired 95% bootstrap CI, B=1000)", y=1.02)
    fig.tight_layout()
    paths = save(fig, "F8_news_dilution_forest")
    plt.close(fig)
    return paths, mlp_21d, sage_21d


def table_ST3(df: pd.DataFrame) -> str:
    rows = []
    for m in MODELS:
        for h in HORIZONS:
            sub = df[(df["model"] == m) & (df["horizon"] == h)]
            ic_mean = sub["IC"].mean()
            ic_std = sub["IC"].std(ddof=1)
            sh_mean = sub["Sharpe_net"].mean()
            sh_std = sub["Sharpe_net"].std(ddof=1)
            rows.append((m, h, ic_mean, ic_std, sh_mean, sh_std, len(sub)))

    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Model & Horizon & IC mean & IC std & $S_{net}$ mean & $S_{net}$ std & $n$ \\",
        r"\midrule",
    ]
    for m, h, im, isd, sm, ssd, n in rows:
        m_esc = m.replace("_", r"\_")
        lines.append(
            f"{m_esc} & {h}d & {im:.4f} & {isd:.4f} & "
            f"{sm:.3f} & {ssd:.3f} & {n} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    latex = "\n".join(lines)
    write_tex_table("ST3_horizon_full", latex)
    return latex


def write_caption(mlp_21d: float, sage_21d: float) -> None:
    out = PROJECT_ROOT / "tables" / "fig_horizon_ablation_caption.txt"
    text = (
        "F7 — Horizon × architecture IC heatmap. Each cell shows mean IC over 15 "
        "runs (3 seeds × 5 walk-forward folds). Diverging RdBu_r colormap centered "
        "at 0. News-augmented variants (*_all) underperform price-only variants "
        "(*_price) at long horizons.\n\n"
        f"F8 — News-feature dilution forest. ΔIC = IC(*_all) − IC(*_price), with "
        f"95% paired bootstrap CI (B=1000) over the 15 (seed, fold) pairs per horizon. "
        f"MLP news-feature dilution at 21d: ΔIC = {mlp_21d:.4f}. "
        f"SAGE-Mean news-feature dilution at 21d: ΔIC = {sage_21d:.4f}. "
        "Negative ΔIC at 21d–63d indicates news features hurt long-horizon ranking.\n\n"
        "ST3 — Full horizon × architecture table (4 models × 6 horizons), n=15 cells per row.\n"
    )
    out.write_text(text)


def main() -> None:
    df = load()
    f7_paths = fig_F7(df)
    f8_paths, mlp_21d, sage_21d = fig_F8(df)
    table_ST3(df)
    write_caption(mlp_21d, sage_21d)
    print(f"[fig_horizon_ablation] F7 -> {f7_paths['pdf']}")
    print(f"[fig_horizon_ablation] F8 -> {f8_paths['pdf']}")
    print(f"[fig_horizon_ablation] MLP 21d ΔIC = {mlp_21d:.4f}")
    print(f"[fig_horizon_ablation] SAGE 21d ΔIC = {sage_21d:.4f}")
    print("[fig_horizon_ablation] ST3 -> tables/ST3_horizon_full.tex")


if __name__ == "__main__":
    main()
