"""Helper for fig_e1_anchor.py — per-day-IC loader + F2 (cumulative IC) + S3
(per-day IC 8-line series).

Split out of fig_e1_anchor.py to keep that script under the 350-line cap.
No standalone entry-point; imported by fig_e1_anchor.main().
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from paper_figs.rcparams_storya import (
    setup,
    model_color,
    save,
    PALETTE,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NPY_DIR = PROJECT_ROOT / "experiments" / "storya_e1_anchor" / "per_day_ic"

UNIVERSES = ["B", "C"]
MODELS = ["GAT", "SAGE-Mean", "MLP", "LightGBM"]
FOLDS = [0, 1, 2, 3, 4]

NPY_PATTERN = re.compile(
    r"^(?P<u>[BC])_(?P<m>[A-Za-z\-]+)_s(?P<s>\d+)_f(?P<f>\d+)\.npy$"
)


def load_per_day_ic(universe: str, model: str) -> dict:
    """Return {seed: {fold: daily_ic_array}} for matching .npy files."""
    out: dict = {}
    prefix = f"{universe}_{model}_s"
    for path in sorted(NPY_DIR.glob(f"{prefix}*_f*.npy")):
        m = NPY_PATTERN.match(path.name)
        if not m or m["u"] != universe or m["m"] != model:
            continue
        seed = int(m["s"])
        fold = int(m["f"])
        arr = np.load(path).astype(np.float32, copy=False)
        out.setdefault(seed, {})[fold] = arr
    return out


# --------------------------------------------------------------------------- #
# F2 — Cumulative IC trajectory (8 panels: 4 rows x 2 cols by universe)
# --------------------------------------------------------------------------- #
def fig_F2() -> dict:
    fig, axes = setup("eight_panel", height=9.0)
    for col, universe in enumerate(UNIVERSES):
        for row, model in enumerate(MODELS):
            ax = axes[row, col]
            per_seed = load_per_day_ic(universe, model)
            seed_cum: dict[int, np.ndarray] = {}
            fold_boundaries: list[int] = []
            for seed, fold_map in per_seed.items():
                parts = [fold_map[f] for f in FOLDS if f in fold_map]
                if not parts:
                    continue
                seed_cum[seed] = np.cumsum(np.concatenate(parts))
                if not fold_boundaries:
                    cursor = 0
                    for f in FOLDS:
                        if f in fold_map:
                            cursor += len(fold_map[f])
                            fold_boundaries.append(cursor)
            if not seed_cum:
                ax.set_title(f"{model} / Univ {universe} (no data)")
                continue
            min_len = min(len(v) for v in seed_cum.values())
            stacked = np.stack([v[:min_len] for v in seed_cum.values()], axis=0)
            mean = stacked.mean(axis=0)
            for cum in seed_cum.values():
                ax.plot(np.arange(len(cum)), cum, color=PALETTE["Baseline"],
                        linewidth=0.5, alpha=0.30)
            ax.plot(np.arange(min_len), mean, color=model_color(model),
                    linewidth=1.2,
                    label=f"mean (n={stacked.shape[0]} seeds)")
            if len(fold_boundaries) >= 5:
                ax.axvspan(fold_boundaries[-2], fold_boundaries[-1],
                           color=PALETTE["Warning"], alpha=0.18, zorder=0)
            ax.axhline(0, color="black", linewidth=0.4, linestyle="--")
            ax.set_title(f"{model} / Univ {universe}", fontsize=8)
            if row == 3:
                ax.set_xlabel("Cumulative trading day")
            if col == 0:
                ax.set_ylabel("Cumulative IC")
            ax.legend(loc="upper left", fontsize=6)
    fig.suptitle("F2 — Cumulative daily IC trajectory (10 seeds per panel; "
                 "Fold 4 shaded amber)", y=0.995, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    paths = save(fig, "F2_cumulative_ic_trajectory")
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- #
# S3 — Per-day IC time series, 8 (universe, model) seed-averaged curves
# --------------------------------------------------------------------------- #
def fig_S3() -> dict:
    fig, ax = setup("full_width", height=3.5)
    fold_boundaries_global: list[int] = []
    for u in UNIVERSES:
        for m in MODELS:
            per_seed = load_per_day_ic(u, m)
            if not per_seed:
                continue
            per_fold_concat: list[np.ndarray] = []
            cursor = 0
            local_boundaries: list[int] = []
            for f in FOLDS:
                arrs = [v[f] for v in per_seed.values() if f in v]
                if not arrs:
                    continue
                min_len = min(len(a) for a in arrs)
                stack = np.stack([a[:min_len] for a in arrs], axis=0)
                mean_fold = stack.mean(axis=0)
                per_fold_concat.append(mean_fold)
                cursor += len(mean_fold)
                local_boundaries.append(cursor)
            if not per_fold_concat:
                continue
            series = np.concatenate(per_fold_concat)
            ax.plot(np.arange(len(series)), series, color=model_color(m),
                    linestyle="-" if u == "B" else "--", linewidth=0.7,
                    alpha=0.85, label=f"{m} / Univ {u}")
            if not fold_boundaries_global and len(local_boundaries) >= 5:
                fold_boundaries_global = local_boundaries
    if len(fold_boundaries_global) >= 5:
        ax.axvspan(fold_boundaries_global[-2], fold_boundaries_global[-1],
                   color=PALETTE["Danger"], alpha=0.12, zorder=0,
                   label="Fold 4 (Q2-2025 regime)")
    ax.axhline(0, color="black", linewidth=0.4, linestyle="--")
    ax.set_xlabel("Cumulative trading day (folds 0..4)")
    ax.set_ylabel("Daily Spearman IC (seed-averaged)")
    ax.set_title("S3 — Per-day IC, 8 (universe, model) curves")
    ax.legend(ncol=3, fontsize=5.5, loc="upper left")
    fig.tight_layout()
    paths = save(fig, "S3_per_day_ic_8_lines")
    plt.close(fig)
    return paths
