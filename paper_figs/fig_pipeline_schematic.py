"""Story A — F1 pipeline architecture schematic.

# SOURCE_CONTRACT:
#   inputs:
#     - path: (project constants — no CSV)
#       columns: derived from Story A v3 plan locked decisions
#       md5: N/A
#       n_rows: N/A
#     - reference: /Users/heruixi/.claude/plans/handoff-session-ranking-swirling-lemur.md
#         §1.1 — multi-seed model comparison locked at 4 models × 10 seeds × 5 folds × 2 univ
#         §1.2 — 4 edge configs (alpha1..alpha4)
#         §1.4 — statistical framework (SPA + DM/HLN + BH-FDR + bootstrap + LOFO + cost)
#   outputs:
#     - path: figures/F1_pipeline.pdf
#     - path: figures/F1_pipeline.svg
#       headline_values:
#         - cells_total: 400 (E1 anchor) + 50 (E3) + 100 (E4-α) + 50 (HATS) = 600
#         - layout: 5-stage left-to-right pipeline
#         - dimensions: 7.0in x 3.5in (ACM SIGCONF full-width)

Caveat highlights:
- Universe C box sub-row carries the L1 LOW STABILITY annotation
- Caption mentions 400 anchor + 50 E3 + 100 E4-α + 50 HATS = 600 cells total
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from paper_figs.rcparams_storya import _apply_rc, save, PALETTE, FIGURES_DIR


# ACM SIGCONF full-width landscape; 7" x 3.5"
FIG_W = 7.0
FIG_H = 3.5

# Box geometry (inches in axes coords)
BOX_W = 1.18
BOX_H = 2.20
BOX_Y_CENTER = 1.80
BOX_TOP = BOX_Y_CENTER + BOX_H / 2
BOX_BOTTOM = BOX_Y_CENTER - BOX_H / 2

# 5 box centers (x positions in inches)
N_BOXES = 5
LEFT_PAD = 0.10
RIGHT_PAD = 0.10
ARROW_GAP = 0.16
total_box_width = N_BOXES * BOX_W
arrow_width_total = FIG_W - LEFT_PAD - RIGHT_PAD - total_box_width
ARROW_GAP = arrow_width_total / (N_BOXES - 1)
BOX_X_CENTERS = [
    LEFT_PAD + BOX_W / 2 + i * (BOX_W + ARROW_GAP) for i in range(N_BOXES)
]


def _draw_box(ax, x_center, title, sub_lines, fillcolor=None,
              highlight_idx=None, highlight_color=None):
    """Draw a box at (x_center, BOX_Y_CENTER) with title + sub_lines.

    Args:
        x_center: float, x position
        title: str, top section heading
        sub_lines: list[str], stacked below the title
        fillcolor: hex string for box fill
        highlight_idx: optional index into sub_lines to color differently
        highlight_color: hex string for highlight
    """
    fillcolor = fillcolor or "#f5f5f5"
    box = FancyBboxPatch(
        (x_center - BOX_W / 2, BOX_BOTTOM),
        BOX_W, BOX_H,
        boxstyle="round,pad=0.04,rounding_size=0.06",
        linewidth=0.6, edgecolor="black", facecolor=fillcolor,
    )
    ax.add_patch(box)

    # Title at top of box
    ax.text(
        x_center, BOX_TOP - 0.18, title,
        ha="center", va="top", fontsize=8.5, fontweight="bold",
    )

    # Divider line under title
    ax.plot(
        [x_center - BOX_W / 2 + 0.06, x_center + BOX_W / 2 - 0.06],
        [BOX_TOP - 0.32, BOX_TOP - 0.32],
        color="black", linewidth=0.4,
    )

    # Sub-lines stacked
    n = len(sub_lines)
    line_y_start = BOX_TOP - 0.46
    line_spacing = 0.24
    for i, line in enumerate(sub_lines):
        y = line_y_start - i * line_spacing
        # Optional highlight background for one sub-row
        if highlight_idx is not None and i == highlight_idx:
            ax.add_patch(
                FancyBboxPatch(
                    (x_center - BOX_W / 2 + 0.05, y - 0.10),
                    BOX_W - 0.10, 0.20,
                    boxstyle="round,pad=0.01,rounding_size=0.02",
                    linewidth=0, facecolor=highlight_color or PALETTE["Warning"],
                    alpha=0.30, zorder=2,
                )
            )
        ax.text(
            x_center, y, line,
            ha="center", va="center", fontsize=6.8, zorder=3,
        )


def _draw_arrow(ax, x_from, x_to, y=BOX_Y_CENTER):
    """Right-pointing arrow between consecutive boxes."""
    arrow = FancyArrowPatch(
        (x_from, y), (x_to, y),
        arrowstyle="-|>", mutation_scale=10,
        linewidth=1.0, color="black", zorder=1,
    )
    ax.add_patch(arrow)


def main() -> None:
    _apply_rc()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    # === BOX 1 — Data ===
    _draw_box(
        ax, BOX_X_CENTERS[0],
        title="Data",
        sub_lines=[
            "S&P 500 (5y)",
            "Daily OHLCV",
            "Price features",
            "News articles",
            "21d c-to-c label",
        ],
        fillcolor="#eef4f8",
    )

    # === BOX 2 — Universe split ===
    _draw_box(
        ax, BOX_X_CENTERS[1],
        title="Universe split",
        sub_lines=[
            "B: 10-dim hc",
            "C: 51-dim",
            "Alpha158 top-15",
            "(LOW STABILITY",
            "caveat — L1)",
        ],
        fillcolor="#f1eef8",
        highlight_idx=3,
        highlight_color=PALETTE["Warning"],
    )

    # === BOX 3 — Edge config ===
    _draw_box(
        ax, BOX_X_CENTERS[2],
        title="Edge config",
        sub_lines=[
            "α1: corr only",
            "α2: + sector",
            "α3: + news",
            "α4: full",
            "(0/5 BH-FDR)",
        ],
        fillcolor="#f8f1ee",
        highlight_idx=4,
        highlight_color=PALETTE["Danger"],
    )

    # === BOX 4 — Models ===
    _draw_box(
        ax, BOX_X_CENTERS[3],
        title="Models",
        sub_lines=[
            "GAT",
            "SAGE-Mean",
            "MLP",
            "LightGBM",
            "(+ HATS-3R)",
        ],
        fillcolor="#eef8ee",
    )

    # === BOX 5 — Evaluation ===
    _draw_box(
        ax, BOX_X_CENTERS[4],
        title="Evaluation",
        sub_lines=[
            "10 seeds × 5 folds",
            "IC + Sharpe",
            "SPA + DM/HLN",
            "BH-FDR + boot CI",
            "LOFO-4 + cost",
        ],
        fillcolor="#f8f8ee",
    )

    # === Arrows between boxes ===
    for i in range(N_BOXES - 1):
        x_from = BOX_X_CENTERS[i] + BOX_W / 2 + 0.01
        x_to = BOX_X_CENTERS[i + 1] - BOX_W / 2 - 0.01
        _draw_arrow(ax, x_from, x_to)

    # === Title at top ===
    ax.text(
        FIG_W / 2, BOX_TOP + 0.30,
        "Story A pipeline",
        ha="center", va="bottom", fontsize=10, fontweight="bold",
    )

    # === Caption at bottom ===
    caption = (
        "S&P 500, 21d horizon. 2 universes × 4 models × 10 seeds × 5 walk-forward folds "
        "= 400 anchor cells; plus E3 50 + E4-α 100 + HATS 50 ablation cells "
        "(total 600). Universe C composition basis is leak-driven (L1)."
    )
    ax.text(
        FIG_W / 2, BOX_BOTTOM - 0.20,
        caption,
        ha="center", va="top", fontsize=6.8, style="italic",
        wrap=True,
    )

    # Axes setup
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect("equal")
    ax.axis("off")

    # Save
    fig.tight_layout(pad=0.1)
    pdf_path = FIGURES_DIR / "F1_pipeline.pdf"
    svg_path = FIGURES_DIR / "F1_pipeline.svg"
    png_path = FIGURES_DIR / "F1_pipeline.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.02,
                metadata={"Creator": "Story A paper_figs pipeline",
                          "Subject": "F1 schematic"})
    fig.savefig(svg_path, format="svg", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, format="png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)
    print(f"[fig_pipeline_schematic] F1 -> {pdf_path}")
    print(f"[fig_pipeline_schematic] F1 -> {svg_path}")
    print(f"[fig_pipeline_schematic] F1 -> {png_path}")


if __name__ == "__main__":
    main()
