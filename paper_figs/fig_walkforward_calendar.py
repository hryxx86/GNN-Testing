"""Story A — S15 walk-forward calendar Gantt visualization.

# SOURCE_CONTRACT:
#   inputs:
#     - path: experiments/storya_e1_anchor/results.csv
#       md5: c29851c0b4ae0457a8b3b24b6a7d6999
#       n_rows: 400
#       columns: [..., fold, test_period, ...]
#       used_columns: [fold, test_period]
#   outputs:
#     - path: figures/S15_walkforward_calendar.pdf
#       headline_values:
#         - n_folds: 5
#         - purge_days: 21
#     - path: tables/fig_walkforward_calendar_caption.txt

# Layout: Gantt-style horizontal bars with 5 folds (y axis).
# For each fold, x axis shows: [train] -> [21d purge gap] -> [val] -> [test].
# Test period anchors derived from `test_period` column (e.g., Q2-2024).
# Train/val window lengths are illustrative (paper-config locked):
#   - Train: rolling ~750 trading days (~3 years) up to start_of_test - 21d - val
#   - Val:   63 trading days (~1 quarter)
#   - Purge: 21 trading days (= horizon)
#   - Test:  one quarter (~63 trading days)
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

from paper_figs.rcparams_storya import (
    setup,
    save,
    PALETTE,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "experiments" / "storya_e1_anchor" / "results.csv"

# Q-label to (start, end) calendar date mapping (illustrative; matches plan)
QUARTER_BOUNDS = {
    "Q1": ("01-01", "03-31"),
    "Q2": ("04-01", "06-30"),
    "Q3": ("07-01", "09-30"),
    "Q4": ("10-01", "12-31"),
}

TRAIN_YEARS = 3.0  # ~3 years rolling train
VAL_DAYS_CAL = 91   # ~1 quarter calendar days for val
PURGE_DAYS_CAL = 30  # 21 trading days ≈ ~30 calendar days

L6_CAVEAT = ("Fold 4 (Q2-2025) is a known regime outlier; LOFO-4 column drops "
             "IC by 38-72%")


def _parse_test_period(label: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Convert 'Q2-2024' -> (Apr 1 2024, Jun 30 2024)."""
    q, year = label.split("-")
    start_md, end_md = QUARTER_BOUNDS[q]
    start = pd.Timestamp(f"{year}-{start_md}")
    end = pd.Timestamp(f"{year}-{end_md}")
    return start, end


def _fold_windows(test_start: pd.Timestamp, test_end: pd.Timestamp) -> dict:
    """Compute illustrative [train, purge, val, test] window bounds."""
    val_end = test_start
    val_start = val_end - pd.Timedelta(days=VAL_DAYS_CAL)
    purge_end = val_start
    purge_start = purge_end - pd.Timedelta(days=PURGE_DAYS_CAL)
    train_end = purge_start
    train_start = train_end - pd.Timedelta(days=int(TRAIN_YEARS * 365))
    return {
        "train": (train_start, train_end),
        "purge": (purge_start, purge_end),
        "val": (val_start, val_end),
        "test": (test_start, test_end),
    }


def load_fold_periods() -> dict[int, str]:
    df = pd.read_csv(CSV_PATH, usecols=["fold", "test_period"])
    df = df.dropna(subset=["fold", "test_period"]).drop_duplicates()
    out: dict[int, str] = {}
    for _, r in df.iterrows():
        out.setdefault(int(r["fold"]), str(r["test_period"]))
    return dict(sorted(out.items()))


def fig_S15() -> dict:
    fold_periods = load_fold_periods()
    if not fold_periods:
        raise RuntimeError("no (fold, test_period) rows in results.csv")
    fig, ax = setup("full_width", height=3.0)

    segment_colors = {
        "train": "#bdbdbd",
        "purge": PALETTE["Danger"],
        "val": PALETTE["Warning"],
        "test": "#2ca02c",
    }
    bar_h = 0.55
    for fold, label in fold_periods.items():
        try:
            test_start, test_end = _parse_test_period(label)
        except (ValueError, KeyError):
            continue
        wins = _fold_windows(test_start, test_end)
        for seg in ("train", "purge", "val", "test"):
            start, end = wins[seg]
            width = (end - start).days
            ax.barh(fold, width, left=start, height=bar_h,
                    color=segment_colors[seg], edgecolor="black",
                    linewidth=0.25)
        # Label the test period inside the test bar
        ax.text(test_start + (test_end - test_start) / 2.0, fold, label,
                ha="center", va="center", fontsize=6.5, color="black")
        if fold == 4:
            ax.text(test_end + pd.Timedelta(days=10), fold,
                    "regime outlier", fontsize=5.5,
                    color=PALETTE["Danger"], va="center")

    ax.set_yticks(sorted(fold_periods.keys()))
    ax.set_yticklabels([f"Fold {f}" for f in sorted(fold_periods.keys())])
    ax.invert_yaxis()
    ax.set_xlabel("Calendar date")
    ax.set_title("S15 — Walk-forward calendar: 5 folds with 21-day purge")

    handles = [Patch(color=segment_colors["train"], label="Train (~3 yr)"),
               Patch(color=segment_colors["purge"],
                     label="Purge (21 trading d ≈ 30 cal d)"),
               Patch(color=segment_colors["val"], label="Val (~1 quarter)"),
               Patch(color=segment_colors["test"], label="Test (1 quarter)")]
    ax.legend(handles=handles, ncol=4, fontsize=6, loc="lower left",
              bbox_to_anchor=(0.0, -0.28))
    # Format x-axis as YYYY
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    paths = save(fig, "S15_walkforward_calendar")
    plt.close(fig)
    return paths


def write_caption() -> None:
    out = PROJECT_ROOT / "tables" / "fig_walkforward_calendar_caption.txt"
    text = (
        f"S15 — Walk-forward calendar: 5 folds × [train, 21-day purge gap, "
        f"validation, test]. Train window ≈ 3 years rolling; validation ≈ 1 "
        f"quarter; purge = 21 trading days (= label horizon); test = 1 "
        f"quarter (Q2-2024, Q3-2024, Q4-2024, Q1-2025, Q2-2025). The 21-day "
        f"purge eliminates train/test label overlap given the 21-day "
        f"close-to-close return horizon. Fold 4 (Q2-2025) is annotated as a "
        f"regime outlier. {L6_CAVEAT}.\n"
    )
    out.write_text(text)


def main() -> None:
    fig_S15()
    write_caption()
    print("[fig_walkforward_calendar] OK — S15 + caption")


if __name__ == "__main__":
    main()
