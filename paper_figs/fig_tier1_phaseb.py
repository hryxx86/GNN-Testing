"""Story A — S13 (Tier1 Phase B 3-panel per-fold boxplots).

# SOURCE_CONTRACT:
#   inputs:
#     - path: artifacts/tier1a_phase_b/results.csv
#       columns: [cell_key, split, loss, model, feature_set, fold, seed,
#                 n_test_days, mean_test_ic, median_test_ic, n_test_ic_valid,
#                 best_val_ic, epochs_run, graph_snap_end, pred_cs_std_median,
#                 elapsed_s, cached]
#       md5: 87ccb8d78adb33a97b8116f66afd0026
#       n_rows: 200
#     - path: artifacts/tier1b_h2_phase_b/results.csv
#       columns: [cell_key, loss, model, feature_set, fold, seed,
#                 n_test_days, mean_test_ic, median_test_ic, n_test_ic_valid,
#                 best_val_ic, epochs_run, graph_snap_end, pred_cs_std_median,
#                 elapsed_s, cached]
#       md5: 5f32b3865543cbf966c2abfd27db2de2
#       n_rows: 800
#     - path: artifacts/tier1c_phase_b/results.csv
#       columns: same as tier1b_h2
#       md5: d7fd787e7ef654c0b0d21f9d1c283598
#       n_rows: 400
#   outputs:
#     - path: figures/S13_tier1_phaseb_boxes.pdf
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
TIER1A = PROJECT_ROOT / "artifacts" / "tier1a_phase_b" / "results.csv"
TIER1B = PROJECT_ROOT / "artifacts" / "tier1b_h2_phase_b" / "results.csv"
TIER1C = PROJECT_ROOT / "artifacts" / "tier1c_phase_b" / "results.csv"


def _box_per_fold(ax, df: pd.DataFrame, title: str) -> None:
    df = df.dropna(subset=["mean_test_ic", "fold"])
    folds = sorted(df["fold"].unique())
    data = [df[df["fold"] == f]["mean_test_ic"].to_numpy() for f in folds]
    box_colors = [PALETTE["Danger"] if int(f) == 4 else PALETTE["Baseline"]
                  for f in folds]

    bp = ax.boxplot(data, positions=range(len(folds)), widths=0.55,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=0.9),
                    whiskerprops=dict(linewidth=0.6),
                    capprops=dict(linewidth=0.6),
                    boxprops=dict(linewidth=0.6))
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    # Jittered scatter overlay
    rng = np.random.default_rng(7)
    for i, arr in enumerate(data):
        if len(arr) == 0:
            continue
        x = i + rng.uniform(-0.18, 0.18, size=len(arr))
        ax.scatter(x, arr, s=4, color="black", alpha=0.4, zorder=3)

    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xticks(range(len(folds)))
    ax.set_xticklabels([f"F{int(f)}" for f in folds])
    ax.set_xlabel("Walk-forward fold")
    ax.set_title(f"{title}  (n={len(df)})", fontsize=8)


def fig_S13() -> dict:
    fig, axes = setup("three_panel", height=3.0, sharey=True)

    tier1a = pd.read_csv(TIER1A)
    tier1b = pd.read_csv(TIER1B)
    tier1c = pd.read_csv(TIER1C)

    _box_per_fold(axes[0], tier1a, "Tier 1a")
    _box_per_fold(axes[1], tier1b, "Tier 1b (H2)")
    _box_per_fold(axes[2], tier1c, "Tier 1c")
    axes[0].set_ylabel("Mean test IC (per cell)")

    fig.suptitle("S13 — Tier 1 Phase B per-fold IC distributions (Fold 4 highlighted)",
                 y=1.02, fontsize=8)
    fig.tight_layout()
    paths = save(fig, "S13_tier1_phaseb_boxes")
    plt.close(fig)
    return paths


def write_caption() -> None:
    out = PROJECT_ROOT / "tables" / "fig_tier1_phaseb_caption.txt"
    text = (
        "S13 — Tier 1 Phase B per-fold mean-test-IC distributions across three "
        "tiers (Tier1a n=200, Tier1b H2 n=800, Tier1c n=400). Each box shows the "
        "distribution of cell-level mean_test_ic across all (model, loss, "
        "feature_set, seed) cells for that fold; black dots show individual cells "
        "with x-jitter. Fold 4 highlighted in red — the known regime-collapse fold "
        "documented in Phase 5 Step 3 Plan Z.\n"
    )
    out.write_text(text)


def main() -> None:
    s13 = fig_S13()
    write_caption()
    print(f"[fig_tier1_phaseb] S13 -> {s13['pdf']}")


if __name__ == "__main__":
    main()
