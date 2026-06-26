"""Story A — Edge-ablation figures: F6, S18 + T5.

# SOURCE_CONTRACT:
#   inputs:
#     - path: artifacts/storya_e6_edge_ablation/edge_bootstrap_ci.csv
#       md5: 8ed2f406d6037d8926511577ebe6280f
#       n_rows: 15
#       columns: [pair_id, description, regime_condition, delta_ic_mean,
#                 delta_ic_ci_lo, delta_ic_ci_hi, delta_sharpe_net10bps_mean,
#                 delta_sharpe_net10bps_ci_lo, delta_sharpe_net10bps_ci_hi]
#     - path: artifacts/storya_e6_edge_ablation/edge_pairs_dm.csv
#       md5: 9853884a41972b631fd3c84155365425
#       n_rows: 15
#       columns: [pair_id, description, regime_condition, n_days_paired,
#                 n_cells_paired, mean_delta_ic, mean_delta_sharpe_net10bps,
#                 DM_stat, HLN_stat, HLN_p_two_sided,
#                 BH_FDR_rejected_q05_full_family5]
#     - path: experiments/storya_e3_news_edge/news_snapshots_cache.npz
#       n_bytes: ~12 MB; schema not pre-verified — see S18 fallback note
#     - path: experiments/storya_e3_news_edge/results.csv  (S18 fallback)
#       md5: f00b2b3d787504ca6f60936078837a03
#       columns include: n_news_edges_avg, n_news_articles_avg
#   outputs:
#     - path: figures/F6_edge_ablation_forest.pdf
#       headline_values:
#         - n_pairs_surviving_bh_fdr_full: 0 (expected per N3)
#     - path: figures/S18_news_edge_density.pdf
#     - path: tables/T5_edge_ablation.tex
#     - path: tables/fig_edge_ablation_caption.txt

# CAPTION REQUIREMENT (N3 narrative, verbatim):
#   "0/5 pairs survive BH-FDR q=0.05 in full condition"
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paper_figs.rcparams_storya import (
    setup,
    model_color,
    save,
    write_tex_table,
    PALETTE,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ABL_DIR = PROJECT_ROOT / "artifacts" / "storya_e6_edge_ablation"
E3_DIR = PROJECT_ROOT / "experiments" / "storya_e3_news_edge"
BOOT_CSV = ABL_DIR / "edge_bootstrap_ci.csv"
DM_CSV = ABL_DIR / "edge_pairs_dm.csv"
NPZ_PATH = E3_DIR / "news_snapshots_cache.npz"
E3_RESULTS_CSV = E3_DIR / "results.csv"

REGIME_ORDER = ["full", "lofo4", "fold4only"]
REGIME_COLOR = {
    "full": model_color("SAGE-Mean"),
    "lofo4": PALETTE["Warning"],
    "fold4only": PALETTE["Danger"],
}

N3_CAVEAT = "0/5 pairs survive BH-FDR q=0.05 in full condition"


# --------------------------------------------------------------------------- #
# F6 — Edge-ablation forest (15 rows = 5 pairs × 3 regimes)
# --------------------------------------------------------------------------- #
def fig_F6() -> dict:
    df = pd.read_csv(BOOT_CSV)
    df = df.dropna(subset=["delta_ic_mean", "pair_id", "regime_condition"])
    pair_ids = list(dict.fromkeys(df["pair_id"].tolist()))
    n_pairs = len(pair_ids)
    fig, ax = setup("full_width", height=4.5)

    y_positions: list[float] = []
    y_labels: list[str] = []
    for pi, pair_id in enumerate(pair_ids):
        base_y = pi * (len(REGIME_ORDER) + 1)
        for ri, regime in enumerate(REGIME_ORDER):
            sub = df[(df["pair_id"] == pair_id) &
                     (df["regime_condition"] == regime)]
            if len(sub) == 0:
                continue
            r = sub.iloc[0]
            y = base_y + ri
            mean = float(r["delta_ic_mean"])
            lo = float(r["delta_ic_ci_lo"])
            hi = float(r["delta_ic_ci_hi"])
            color = REGIME_COLOR[regime]
            ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]],
                        fmt="o", color=color, ecolor=color, capsize=2.5,
                        markersize=4.5, linewidth=0.9)
            y_positions.append(y)
            y_labels.append(regime)
        # Pair separator label
        desc = df[df["pair_id"] == pair_id]["description"].iloc[0]
        ax.text(-0.02, base_y + 1, f"{pair_id}\n({desc})",
                transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=6)
    ax.axvline(0, color="black", linewidth=0.4, linestyle="--")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta$ IC (95% bootstrap CI)")
    ax.set_title("F6 — Edge-ablation forest: 5 pairs × 3 regimes")
    # legend
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=REGIME_COLOR[r], markersize=6,
                          label=r) for r in REGIME_ORDER]
    ax.legend(handles=handles, fontsize=6, loc="lower right")
    fig.tight_layout()
    paths = save(fig, "F6_edge_ablation_forest")
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- #
# S18 — News-edge density temporal profile
# --------------------------------------------------------------------------- #
def fig_S18() -> dict:
    """Attempt to read news_snapshots_cache.npz; on schema mismatch, fall back
    to per-cell aggregates from E3 results.csv."""
    fig, ax = setup("full_width", height=2.8)
    used_fallback = False
    try:
        with np.load(NPZ_PATH, allow_pickle=False) as npz:
            keys = list(npz.keys())
            # Heuristic: prefer arrays whose name contains 'edge', 'article', or 'count'
            candidates = [k for k in keys if any(tok in k.lower() for tok in
                          ("edge", "article", "count", "n_"))]
            plotted = False
            for k in candidates:
                arr = np.asarray(npz[k])
                if arr.ndim == 1 and arr.size > 1 and np.issubdtype(arr.dtype, np.number):
                    # Strip leading underscores; matplotlib ignores _-prefixed labels in legend.
                    label = k.strip("_") or k
                    ax.plot(np.arange(len(arr)), arr, linewidth=0.8, label=label)
                    plotted = True
            if not plotted:
                raise ValueError(f"no 1-D numeric per-day array in npz; "
                                 f"keys={keys}")
            ax.set_xlabel("Day index (within E3 fold pool)")
            ax.set_ylabel("Count")
            ax.set_title("S18 — News-edge density (from npz cache)")
            ax.legend(fontsize=6, loc="upper right")
    except Exception as exc:  # noqa: BLE001
        # Fallback: bar chart of per-cell n_news_edges_avg / n_news_articles_avg.
        used_fallback = True
        ax.clear()
        df = pd.read_csv(E3_RESULTS_CSV)
        df = df.dropna(subset=["n_news_edges_avg"])
        df = df.sort_values(["fold", "seed"]).reset_index(drop=True)
        x = np.arange(len(df))
        ax.bar(x - 0.2, df["n_news_edges_avg"], width=0.4,
               color=model_color("SAGE-Mean"), label="n_news_edges_avg",
               edgecolor="black", linewidth=0.2)
        if "n_news_articles_avg" in df.columns:
            ax.bar(x + 0.2, df["n_news_articles_avg"], width=0.4,
                   color=PALETTE["Highlight"], label="n_news_articles_avg",
                   edgecolor="black", linewidth=0.2)
        ax.set_xlabel(f"E3 cell index (n={len(df)}) — FALLBACK "
                      f"(npz unparseable: {type(exc).__name__})")
        ax.set_ylabel("Count")
        ax.set_title("S18 — News-edge density (per-cell aggregates, FALLBACK)")
        ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    paths = save(fig, "S18_news_edge_density")
    plt.close(fig)
    if used_fallback:
        print("[fig_edge_ablation] S18 used FALLBACK (per-cell aggregates)")
    return paths


# --------------------------------------------------------------------------- #
# T5 — Edge ablation LaTeX table (boot CI + DM)
# --------------------------------------------------------------------------- #
def table_T5() -> str:
    boot = pd.read_csv(BOOT_CSV)
    dm = pd.read_csv(DM_CSV)
    merged = boot.merge(dm, on=["pair_id", "description", "regime_condition"],
                        suffixes=("_b", "_d"))
    # Sort: by pair_id then regime in our defined order
    merged["regime_order"] = merged["regime_condition"].map(
        {r: i for i, r in enumerate(REGIME_ORDER)})
    merged = merged.sort_values(["pair_id", "regime_order"]).drop(
        columns=["regime_order"])
    lines = [
        r"\footnotesize",
        r"\begin{tabular}{lllrrrrl}",
        r"\toprule",
        r"Pair & Description & Regime & "
        r"$\overline{\Delta IC}$ & [CI lo, CI hi] & HLN stat & "
        r"HLN $p$ & BH-FDR \\",
        r"\midrule",
    ]
    for _, r in merged.iterrows():
        pair = str(r["pair_id"]).replace("_", r"\_")
        desc = str(r["description"]).replace("_", r"\_")
        regime = r["regime_condition"]
        mean_dic = float(r["delta_ic_mean"])
        lo = float(r["delta_ic_ci_lo"])
        hi = float(r["delta_ic_ci_hi"])
        hln = r.get("HLN_stat", np.nan)
        p = r.get("HLN_p_two_sided", np.nan)
        rej_raw = r.get("BH_FDR_rejected_q05_full_family5", "")
        # BH-FDR family-of-5 only applies to the `full` regime. lofo4 and
        # fold4_only rows have NaN here and must NOT be rendered as truthy
        # via bool(NaN)=True. Explicit NaN check.
        if pd.isna(rej_raw):
            rej_disp = "--"
        elif isinstance(rej_raw, str):
            rej_disp = rej_raw if rej_raw.strip() else "--"
        else:
            rej_disp = r"\checkmark" if bool(rej_raw) else "no"
        hln_s = f"{float(hln):.3f}" if pd.notna(hln) else "--"
        p_s = f"{float(p):.4f}" if pd.notna(p) else "--"
        lines.append(
            f"{pair} & {desc} & {regime} & "
            f"{mean_dic:.4f} & [{lo:.4f}, {hi:.4f}] & "
            f"{hln_s} & {p_s} & {rej_disp} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    latex = "\n".join(lines)
    write_tex_table("T5_edge_ablation", latex)
    return latex


def write_caption() -> None:
    out = PROJECT_ROOT / "tables" / "fig_edge_ablation_caption.txt"
    text = (
        f"F6 — Edge-ablation forest. 5 alpha-edge pairs × 3 regime conditions "
        f"(full / LOFO-4 / Fold-4-only). Each marker = mean ΔIC with 95% "
        f"bootstrap CI. KEY FINDING: {N3_CAVEAT}, indicating no edge "
        f"augmentation provides robust ranking-quality lift over the "
        f"corr-only baseline.\n\n"
        "S18 — News-edge density temporal profile. Primary view reads "
        "per-day arrays from news_snapshots_cache.npz; if the npz schema is "
        "unparseable, the script automatically falls back to per-cell "
        "aggregate counts (n_news_edges_avg, n_news_articles_avg) from E3 "
        "results.csv. The fallback condition is logged to stdout and noted "
        "in the figure title.\n\n"
        f"T5 — Edge-ablation full table. Joined bootstrap CIs (15 rows) and "
        f"DM/HLN paired-test statistics. BH-FDR column is populated only for "
        f"the `full` regime (family of 5 pairs); LOFO-4 and Fold-4-only "
        f"regimes report stats without family correction. {N3_CAVEAT}.\n"
    )
    out.write_text(text)


def main() -> None:
    fig_F6()
    fig_S18()
    table_T5()
    write_caption()
    print("[fig_edge_ablation] OK — F6 + S18 + T5 + caption")


if __name__ == "__main__":
    main()
