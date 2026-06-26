"""Story A — S7 (loss×arch ΔIC heatmap), S8 (ListMLE fold-4 collapse),
S14 (diagnostic_price replication failure), ST6 (loss pairwise table).

# SOURCE_CONTRACT:
#   inputs:
#     - path: experiments/loss_horserace/results.csv
#       columns: [model, loss, feature_set, fold, seed, day_idx, ic, pred_cs_std_day]
#       md5: 398c7a317c03c01064dce97bef536d85
#       n_rows: 37560
#     - path: experiments/loss_horserace/paired_delta_ic.csv
#       columns: [model, feature_set, fold, seed, day_idx, fold_day_id,
#                 delta_ic, loss_contrast]
#       md5: 89b8b99f03c7409831d53e6b8b83c90b
#       n_rows: 24721
#     - path: experiments/loss_horserace/results_diagnostic_price.csv
#       columns: [model, loss, feature_set, fold, seed, day_idx, ic, pred_cs_std_day]
#       md5: b575b1b7ec995783f6acd164304b5188
#       n_rows: 12520
#   outputs:
#     - path: figures/S7_loss_arch_delta_heatmap.pdf
#     - path: figures/S8_listmle_fold4_collapse.pdf
#     - path: figures/S14_diagnostic_price_replication.pdf
#       headline_values:
#         - caveat: "Part B v4 wf5 21d MLP_price IC=+0.037 / SAGE_price IC=+0.027
#                    did NOT replicate (Diagnostic_price IC ≈ -0.004 / -0.057)"
#     - path: tables/ST6_loss_pairwise.tex
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
    setup, model_color, save, write_tex_table, PALETTE
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = PROJECT_ROOT / "experiments" / "loss_horserace" / "results.csv"
PAIRED_CSV = PROJECT_ROOT / "experiments" / "loss_horserace" / "paired_delta_ic.csv"
DIAG_PRICE_CSV = PROJECT_ROOT / "experiments" / "loss_horserace" / "results_diagnostic_price.csv"


def _aggregate_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily-IC table to per-cell IC by mean over day_idx."""
    return (
        df.groupby(["model", "loss", "feature_set", "fold", "seed"], as_index=False)
        ["ic"].mean()
    )


def fig_S7() -> dict:
    """Two-panel heatmap of mean(ΔIC vs MSE) across (model × loss), stratified
    by feature_set. Averaging across feature_sets would mask the conditional
    finding documented in S14 (where price-only vs full-feature deltas diverge),
    so we report one heatmap per feature_set.
    """
    paired = pd.read_csv(PAIRED_CSV)
    paired = paired.dropna(subset=["delta_ic", "loss_contrast", "feature_set"])

    # loss_contrast like "huber_vs_mse" -> loss = "huber"
    paired["loss"] = paired["loss_contrast"].str.replace("_vs_mse", "", regex=False)

    # Cell-level aggregate first (per (model, feature_set, loss, fold, seed))
    cell = (
        paired.groupby(["model", "feature_set", "loss", "fold", "seed"], as_index=False)
        ["delta_ic"].mean()
    )
    # Then mean across cells for the heatmap cell value
    agg = (
        cell.groupby(["model", "feature_set", "loss"], as_index=False)
        ["delta_ic"].mean()
    )

    feature_sets = sorted(agg["feature_set"].unique())
    archs = sorted(agg["model"].unique())
    losses_all = sorted(agg["loss"].unique())
    losses = ["mse"] + [l for l in losses_all if l != "mse"]

    # Global vmax across feature_sets so panels are visually comparable
    panel_mats: dict[str, np.ndarray] = {}
    for fs in feature_sets:
        mat = np.zeros((len(archs), len(losses)))
        for i, a in enumerate(archs):
            for j, l in enumerate(losses):
                if l == "mse":
                    mat[i, j] = 0.0
                else:
                    sub = agg[(agg["model"] == a) & (agg["feature_set"] == fs)
                              & (agg["loss"] == l)]
                    mat[i, j] = float(sub["delta_ic"].mean()) if len(sub) else np.nan
        panel_mats[fs] = mat
    finite_vals = np.concatenate([m[np.isfinite(m)].ravel() for m in panel_mats.values()])
    vmax = float(np.nanmax(np.abs(finite_vals))) if finite_vals.size else 0.01
    vmax = max(vmax, 1e-4)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    n_panels = len(feature_sets)
    fig, axes = setup("full_width", height=2.6 + 0.8 * n_panels)
    if n_panels == 1:
        plt.close(fig)
        fig, ax = setup("full_width", height=3.2)
        axes = [ax]
    else:
        plt.close(fig)
        from paper_figs.rcparams_storya import FULL_WIDTH, _apply_rc
        _apply_rc()
        fig, axes = plt.subplots(1, n_panels, figsize=(FULL_WIDTH, 3.4),
                                 sharey=True)
        axes = list(axes)

    im = None
    for ax, fs in zip(axes, feature_sets):
        mat = panel_mats[fs]
        im = ax.imshow(mat, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_xticks(range(len(losses)))
        ax.set_xticklabels(losses, fontsize=7)
        ax.set_yticks(range(len(archs)))
        ax.set_yticklabels(archs, fontsize=7)
        ax.set_xlabel("Loss")
        ax.set_title(f"feature_set = {fs}", fontsize=8)
        for i in range(len(archs)):
            for j in range(len(losses)):
                v = mat[i, j]
                if np.isfinite(v):
                    color = "white" if abs(v) > 0.6 * vmax else "black"
                    ax.text(j, i, f"{v:+.4f}", ha="center", va="center",
                            fontsize=6, color=color)
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.85, pad=0.015).set_label("ΔIC vs MSE", fontsize=7)
    fig.suptitle("S7 — Loss × architecture ΔIC heatmap, stratified by feature_set "
                 "(cell-level mean over (fold × seed))", fontsize=8, y=1.02)
    paths = save(fig, "S7_loss_arch_delta_heatmap")
    plt.close(fig)
    return paths


def fig_S8() -> dict:
    """Per-fold IC trajectory for ListMLE runs; shade fold 4."""
    df = pd.read_csv(RESULTS_CSV)
    df = df[df["loss"] == "listmle"].dropna(subset=["ic"])
    cells = _aggregate_cells(df)
    # mean over seeds → per (model, feature_set, fold)
    per = (
        cells.groupby(["model", "feature_set", "fold"], as_index=False)
        ["ic"].mean()
    )

    fig, ax = setup("full_width", height=3.0)
    folds = sorted(per["fold"].unique())
    ax.axvspan(3.5, 4.5, color=PALETTE["Danger"], alpha=0.12, zorder=0)

    groups = per.groupby(["model", "feature_set"])
    for (m, fs), g in groups:
        g = g.sort_values("fold")
        c = model_color(m if m in ("MLP", "SAGE-Mean", "GAT") else "Baseline")
        ls = "-" if fs == "S_price" or "price" in str(fs).lower() else "--"
        ax.plot(g["fold"], g["ic"], marker="o", linewidth=1.0,
                color=c, linestyle=ls, label=f"{m} / {fs}", markersize=4)

    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xticks(folds)
    ax.set_xlabel("Walk-forward fold")
    ax.set_ylabel("Mean IC (over seed × day)")
    ax.set_title("S8 — ListMLE per-fold IC: Fold-4 regime collapse", fontsize=8)
    ax.legend(loc="best", fontsize=6, ncol=2)
    fig.tight_layout()
    paths = save(fig, "S8_listmle_fold4_collapse")
    plt.close(fig)
    return paths


def fig_S14() -> dict:
    """Histogram of per-cell IC from diagnostic_price; 2 panels MLP / SAGE-Mean.
    Both panels share a single bin-edge array so visual comparison is honest.
    """
    df = pd.read_csv(DIAG_PRICE_CSV)
    df = df.dropna(subset=["ic"])
    cells = _aggregate_cells(df)  # per-cell IC

    fig, axes = setup("two_panel", height=3.0)
    arches = ["MLP", "SAGE-Mean"]
    claimed = {"MLP": 0.037, "SAGE-Mean": 0.027}
    # Shared bins across both panels (combined data + claimed arrow positions)
    all_ic = cells["ic"].to_numpy()
    lo = float(min(all_ic.min(), min(claimed.values())))
    hi = float(max(all_ic.max(), max(claimed.values())))
    pad = 0.02 * (hi - lo + 1e-6)
    shared_bins = np.linspace(lo - pad, hi + pad, 21)

    for ax, arch in zip(axes, arches):
        sub = cells[cells["model"] == arch]
        for loss, color in (("mse", PALETTE["Baseline"]),
                            ("listmle", PALETTE["Danger"])):
            x = sub[sub["loss"] == loss]["ic"].to_numpy()
            if len(x) == 0:
                continue
            ax.hist(x, bins=shared_bins, alpha=0.55, color=color,
                    edgecolor="black", linewidth=0.3,
                    label=f"{loss} (n={len(x)}, mean={x.mean():+.4f})")
            ax.axvline(x.mean(), color=color, linewidth=0.8, linestyle="--")

        # Arrow to claimed Part B v4 wf5 21d value
        claim = claimed[arch]
        ax.annotate(
            f"Part B v4 claim: {claim:+.3f}",
            xy=(claim, 0), xytext=(claim, 4),
            ha="center", fontsize=6, color=PALETTE["Warning"],
            arrowprops=dict(arrowstyle="->", color=PALETTE["Warning"], lw=0.8),
        )
        ax.axvline(0, color="black", linewidth=0.5, linestyle=":")
        ax.set_xlabel("Per-cell IC")
        ax.set_ylabel("count")
        ax.set_title(f"{arch} (S_price, diagnostic)", fontsize=8)
        ax.legend(loc="upper left", fontsize=6)

    fig.suptitle("S14 — Diagnostic_price replication failure (Part B v4 wf5 NOT replicated)",
                 y=1.02, fontsize=8)
    fig.tight_layout()
    paths = save(fig, "S14_diagnostic_price_replication")
    plt.close(fig)
    return paths


def table_ST6() -> str:
    """Paired ΔIC summary per (model, feature_set, loss_contrast) using
    cluster-bootstrap on (fold, seed) cells (B=1000, seed=42). iid resampling
    of daily ΔIC observations would ignore within-cell temporal autocorrelation
    and produce too-narrow CIs.
    """
    paired = pd.read_csv(PAIRED_CSV).dropna(subset=["delta_ic", "loss_contrast"])

    rng = np.random.default_rng(42)
    rows = []
    for (m, fs, lc), g in paired.groupby(["model", "feature_set", "loss_contrast"]):
        # Cluster bootstrap unit = (fold, seed) cell.
        cell = g.groupby(["fold", "seed"], as_index=False)["delta_ic"].mean()
        cell_means = cell["delta_ic"].to_numpy()
        n_cells = len(cell_means)
        if n_cells == 0:
            continue
        mean = float(cell_means.mean())  # equal weight per cell
        B = 1000
        boots = np.empty(B)
        for b in range(B):
            boots[b] = cell_means[rng.integers(0, n_cells, size=n_cells)].mean()
        lo, hi = np.quantile(boots, [0.025, 0.975])
        rows.append((m, fs, lc, n_cells, mean, float(lo), float(hi)))

    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    lines = [
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Model & Features & Contrast & $n_{cells}$ & $\overline{\Delta IC}$ & CI lo & CI hi \\",
        r"\midrule",
    ]
    for m, fs, lc, n, mean, lo, hi in rows:
        m_esc = str(m).replace("_", r"\_")
        fs_esc = str(fs).replace("_", r"\_")
        lc_esc = str(lc).replace("_", r"\_")
        lines.append(
            f"{m_esc} & {fs_esc} & {lc_esc} & {n} & "
            f"{mean:+.4f} & {lo:+.4f} & {hi:+.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    latex = "\n".join(lines)
    write_tex_table("ST6_loss_pairwise", latex)
    return latex


def write_caption() -> None:
    out = PROJECT_ROOT / "tables" / "fig_loss_horserace_caption.txt"
    text = (
        "S7 — Loss × architecture ΔIC heatmap, stratified by feature_set "
        "(two panels: S6 full features vs S_price 9-dim). Each cell = mean ΔIC vs MSE "
        "baseline aggregated across (fold × seed) cells. MSE column anchored at 0 by "
        "definition. Stratification is needed because S14 documents opposite-sign "
        "ΔIC between feature_sets at long horizons.\n\n"
        "S8 — ListMLE per-fold IC trajectory across the 5 walk-forward folds. "
        "Fold 4 (red band) shows systematic collapse for ListMLE — see Phase 5 "
        "Step 3 Plan Z Fold-4 regime forensics.\n\n"
        "S14 — Diagnostic_price replication histogram. Caveat: Part B v4 wf5 21d "
        "MLP_price IC=+0.037 / SAGE_price IC=+0.027 did NOT replicate in Stage 1 "
        "framework (Diagnostic_price IC ≈ -0.004 / -0.057). Vertical dashed = "
        "sample mean per loss; amber arrow = original Part B v4 claim.\n\n"
        "ST6 — Paired ΔIC per (model, feature_set, loss_contrast); 95% cluster-bootstrap "
        "CI on the mean clustered by (fold, seed) cell (B=1000, seed=42). Cluster unit "
        "addresses within-cell temporal autocorrelation in daily ΔIC. n_cells column is "
        "the bootstrap unit count, not raw daily-observation count. No DM/HLN columns — "
        "not in source CSV.\n"
    )
    out.write_text(text)


def main() -> None:
    s7 = fig_S7()
    s8 = fig_S8()
    s14 = fig_S14()
    table_ST6()
    write_caption()
    print(f"[fig_loss_horserace] S7  -> {s7['pdf']}")
    print(f"[fig_loss_horserace] S8  -> {s8['pdf']}")
    print(f"[fig_loss_horserace] S14 -> {s14['pdf']}")
    print(f"[fig_loss_horserace] ST6 -> tables/ST6_loss_pairwise.tex")


if __name__ == "__main__":
    main()
