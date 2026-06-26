"""Story A — S9 (graph ablation grouped bar).

# SOURCE_CONTRACT:
#   inputs:
#     - path: experiments/graph_ablation_results.csv
#       columns: [config, desc, seed, edges, IC, IC_std, Sharpe_NO, n_periods]
#       md5: ba72ab2c9442bf6e46e5bd2bebc586e3
#       n_rows: 27
#   outputs:
#     - path: figures/S9_graph_ablation.pdf
#       headline_values:
#         - n_configs: 9
#         - true_mlp_baseline: IC of config 0_true_mlp (mean over 3 seeds)
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
CSV_PATH = PROJECT_ROOT / "experiments" / "graph_ablation_results.csv"


def fig_S9() -> tuple[dict, float]:
    df = pd.read_csv(CSV_PATH).dropna(subset=["IC", "config"])
    # Per-config mean IC across 3 seeds + std
    agg = df.groupby("config", as_index=False).agg(
        ic_mean=("IC", "mean"),
        ic_std=("IC", "std"),
    )
    # Preserve original config order from CSV
    order = df.drop_duplicates(subset=["config"])["config"].tolist()
    agg = agg.set_index("config").loc[order].reset_index()

    baseline_ic = float(
        agg.loc[agg["config"] == "0_true_mlp", "ic_mean"].iloc[0]
        if (agg["config"] == "0_true_mlp").any() else np.nan
    )

    # Color logic
    def _color(cfg, ic_mean):
        if cfg == "0_true_mlp":
            return PALETTE["GAT"]   # green baseline highlight
        if np.isfinite(baseline_ic) and ic_mean <= baseline_ic:
            return PALETTE["Danger"]
        return PALETTE["Baseline"]

    colors = [_color(c, m) for c, m in zip(agg["config"], agg["ic_mean"])]

    fig, ax = setup("full_width", height=3.0)
    xs = np.arange(len(agg))
    ax.bar(xs, agg["ic_mean"], yerr=agg["ic_std"],
           color=colors, edgecolor="black", linewidth=0.4,
           capsize=2.5, error_kw=dict(elinewidth=0.6))

    # Per-seed jitter scatter
    rng = np.random.default_rng(123)
    for i, cfg in enumerate(agg["config"]):
        sub = df[df["config"] == cfg]
        jitter = rng.uniform(-0.18, 0.18, size=len(sub))
        ax.scatter(np.full(len(sub), i) + jitter, sub["IC"],
                   color="black", s=10, alpha=0.65, zorder=5,
                   edgecolor="white", linewidth=0.3)

    if np.isfinite(baseline_ic):
        ax.axhline(baseline_ic, color=PALETTE["GAT"], linewidth=0.8,
                   linestyle="--", alpha=0.7, label=f"true_mlp = {baseline_ic:.4f}")
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xticks(xs)
    ax.set_xticklabels(agg["config"], rotation=30, ha="right", fontsize=6)
    ax.set_ylabel("IC (mean over 3 seeds)")
    ax.set_title("S9 — Graph ablation: configs at or below nn.Linear baseline", fontsize=8)
    ax.legend(loc="best", fontsize=6)
    fig.tight_layout()
    paths = save(fig, "S9_graph_ablation")
    plt.close(fig)
    return paths, baseline_ic


def write_caption(baseline_ic: float) -> None:
    out = PROJECT_ROOT / "tables" / "fig_graph_ablation_caption.txt"
    text = (
        f"S9 — Graph ablation grouped bar chart. 9 configs × 3 seeds = 27 runs. "
        f"Bars show mean IC ± std across seeds; black dots show individual seed runs "
        f"with x-jitter. Green bar = nn.Linear baseline (config '0_true_mlp', "
        f"IC = {baseline_ic:.4f}); red bars = configs whose mean IC is at or below "
        f"the nn.Linear baseline. Dashed green line marks the baseline.\n"
    )
    out.write_text(text)


def main() -> None:
    s9, baseline_ic = fig_S9()
    write_caption(baseline_ic)
    print(f"[fig_graph_ablation] S9 -> {s9['pdf']}")
    print(f"[fig_graph_ablation] true_mlp baseline IC = {baseline_ic:.4f}")


if __name__ == "__main__":
    main()
