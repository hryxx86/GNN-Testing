"""rcparams + helpers for Story A paper figures (ICAIF 2026 ACM SIGCONF).

Usage:
    from paper_figs.rcparams_storya import setup, model_color, save
    fig, ax = setup('single_col')
    ax.bar([0, 1], [0.04, 0.03], color=[model_color('GAT'), model_color('SAGE-Mean')])
    save(fig, 'F2_demo')

Layout reference: ACM SIGCONF (US Letter, 2-column body).
    single_col   = 3.33 in wide  (one column)
    full_width   = 7.00 in wide  (spans both columns)
    two_panel    = 7.00 in wide x 3.0 in tall, 2 axes

Colorblind palette: ColorBrewer Set2 (verified for protanopia/deuteranopia/tritanopia).
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import datetime as _dt
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR = PROJECT_ROOT / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# ACM SIGCONF column widths (inches)
SINGLE_COL_WIDTH = 3.33
FULL_WIDTH = 7.00

# Golden-ratio-ish default heights
DEFAULT_HEIGHT_SINGLE = 2.4
DEFAULT_HEIGHT_FULL = 3.0

# Font sizes (pt) — slightly larger than ICML default to read well at 8pt caption
BODY_PT = 9
LABEL_PT = 8
TICK_PT = 7
CAPTION_PT = 8
LEGEND_PT = 7

# ColorBrewer Set2 (CB-safe). 4 model slots + neutrals.
PALETTE = {
    "GAT":        "#1b9e77",   # teal — primary GNN
    "SAGE-Mean":  "#d95f02",   # orange — alt GNN
    "MLP":        "#7570b3",   # purple — neural non-graph
    "LightGBM":   "#e7298a",   # magenta — tabular baseline
    "HATS":       "#66a61e",   # green — 3-relation adapter
    "Baseline":   "#666666",   # grey
    "Highlight":  "#a6cee3",   # light blue (CI bands)
    "Warning":    "#e6ab02",   # amber — for caveats / Fold-4
    "Danger":     "#e41a1c",   # red — for failure-mode highlights
}

# Universe markers
UNIVERSE_MARKER = {"B": "o", "C": "s"}


def _resolve_serif():
    """Prefer Times; fall back to DejaVu Serif if unavailable."""
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    for candidate in ("Times", "Times New Roman", "DejaVu Serif"):
        if candidate in available:
            return candidate
    return "serif"


_BASE_RC = {
    # Global figure font standard (H博士 2026-06-23): sans-serif Arial — crisper at small
    # sizes + ML-venue convention (sans figures with serif ACM body is standard).
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": BODY_PT,
    "axes.titlesize": BODY_PT,
    "axes.labelsize": LABEL_PT,
    "xtick.labelsize": TICK_PT,
    "ytick.labelsize": TICK_PT,
    "legend.fontsize": LEGEND_PT,
    "figure.titlesize": BODY_PT,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 1.0,
    "lines.markersize": 3.5,
    "axes.grid": True,
    "grid.linewidth": 0.3,
    "grid.alpha": 0.4,
    "grid.color": "#cccccc",
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,  # TrueType embed — editable in Illustrator
    "ps.fonttype": 42,
    "figure.dpi": 110,   # screen preview
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "legend.borderaxespad": 0.3,
}


def _apply_rc():
    mpl.rcParams.update(_BASE_RC)


def setup(format: str = "single_col", height: float | None = None, **subplots_kwargs):
    """Apply Story A rcparams and create a figure at the requested layout.

    format:
        'single_col'  -> 3.33 x DEFAULT_HEIGHT_SINGLE single axis
        'full_width'  -> 7.00 x DEFAULT_HEIGHT_FULL single axis
        'two_panel'   -> 7.00 x 3.0 with (1 row, 2 cols) axes
        'four_panel'  -> 7.00 x 5.6 with (2 rows, 2 cols) axes
        'eight_panel' -> 7.00 x 9.0 with (4 rows, 2 cols) axes (heatmap-sized)
    """
    _apply_rc()
    if format == "single_col":
        w = SINGLE_COL_WIDTH
        h = height if height is not None else DEFAULT_HEIGHT_SINGLE
        fig, ax = plt.subplots(1, 1, figsize=(w, h), **subplots_kwargs)
        return fig, ax
    if format == "full_width":
        w = FULL_WIDTH
        h = height if height is not None else DEFAULT_HEIGHT_FULL
        fig, ax = plt.subplots(1, 1, figsize=(w, h), **subplots_kwargs)
        return fig, ax
    if format == "two_panel":
        w = FULL_WIDTH
        h = height if height is not None else 3.0
        fig, axes = plt.subplots(1, 2, figsize=(w, h), **subplots_kwargs)
        return fig, axes
    if format == "four_panel":
        w = FULL_WIDTH
        h = height if height is not None else 5.6
        fig, axes = plt.subplots(2, 2, figsize=(w, h), **subplots_kwargs)
        return fig, axes
    if format == "eight_panel":
        w = FULL_WIDTH
        h = height if height is not None else 9.0
        fig, axes = plt.subplots(4, 2, figsize=(w, h), **subplots_kwargs)
        return fig, axes
    if format == "three_panel":
        w = FULL_WIDTH
        h = height if height is not None else 3.0
        fig, axes = plt.subplots(1, 3, figsize=(w, h), **subplots_kwargs)
        return fig, axes
    raise ValueError(f"unknown format: {format!r}")


def model_color(name: str) -> str:
    """Return a hex color for a model name. Unknown names default to grey."""
    return PALETTE.get(name, PALETTE["Baseline"])


def _git_rev() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=3,
        )
        return out.stdout.strip() or "no-git"
    except Exception:
        return "no-git"


def save(fig, name: str, formats=("pdf", "png")) -> dict:
    """Save fig to figures/<name>.<fmt> with metadata. Returns dict of paths."""
    paths = {}
    git_rev = _git_rev()
    ts = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    metadata = {
        "Creator": "Story A paper_figs pipeline",
        "Subject": f"git={git_rev} ts={ts}",
        "Producer": "matplotlib",
    }
    for fmt in formats:
        path = FIGURES_DIR / f"{name}.{fmt}"
        if fmt == "pdf":
            fig.savefig(path, format="pdf", metadata=metadata)
        elif fmt == "png":
            fig.savefig(path, format="png", dpi=200)
        else:
            fig.savefig(path, format=fmt)
        paths[fmt] = str(path)
    return paths


def md5_of_file(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_tex_table(name: str, latex_str: str) -> str:
    path = TABLES_DIR / f"{name}.tex"
    path.write_text(latex_str)
    return str(path)


if __name__ == "__main__":
    fig, ax = setup("single_col")
    ax.bar(
        ["GAT", "SAGE-Mean", "MLP", "LightGBM"],
        [0.032, 0.027, 0.037, 0.020],
        color=[model_color(m) for m in ["GAT", "SAGE-Mean", "MLP", "LightGBM"]],
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_ylabel("IC (21d)")
    ax.set_title("rcparams smoke test")
    paths = save(fig, "test_rcparams")
    print("OK", paths)
