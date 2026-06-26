"""Story A — F10 (Plan AAA T-1 stability scatter), S4 (top-30 ranking dot plot), ST4.

# SOURCE_CONTRACT:
#   inputs:
#     - path: artifacts/plan_aaa/ranking.csv
#       columns: [rank, group_id, group_label, group_size, group_members,
#                 n_dates, mean_delta_IC, nw_lag, nw_se, nw_t,
#                 nw_p_two_sided, bootstrap_ci_lo, bootstrap_ci_hi,
#                 bh_fdr_rejected, bh_fdr_p_adj]
#       md5: fcc9b8390efbb21fa54cc858df693570
#       n_rows: 61
#     - path: artifacts/plan_aaa_t1_diagnostic/group_ranking_comparison.csv
#       columns: [group_id, group_label, group_size, n_alpha158_members,
#                 plan_aaa_orig_rank, plan_aaa_orig_mean_delta_ic,
#                 proxy_group_abs_ic_raw_leaky, proxy_group_abs_ic_t1_shifted,
#                 proxy_ic_drop_abs, leak_affected, proxy_rank_raw, proxy_rank_t1]
#       md5: aa7d8820c13c24a5f79a15d696fe8dcd
#       n_rows: 61
#   outputs:
#     - path: figures/F10_plan_aaa_t1_stability.pdf
#       headline_values:
#         - verdict: "Plan AAA orig ∩ proxy-T1 = 5/15 → LOW STABILITY"
#     - path: figures/S4_plan_aaa_ranking_top30.pdf
#     - path: tables/ST4_plan_aaa_top20.tex
#
# NOTE: S5 (per-group permutation distribution) is SKIPPED — no per-group
# permutation distribution CSV exists. Document the skip here.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paper_figs.rcparams_storya import setup, save, write_tex_table, PALETTE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RANKING_CSV = PROJECT_ROOT / "artifacts" / "plan_aaa" / "ranking.csv"
DIAG_CSV = PROJECT_ROOT / "artifacts" / "plan_aaa_t1_diagnostic" / "group_ranking_comparison.csv"

VERDICT = "Plan AAA orig ∩ proxy-T1 = 5/15 → LOW STABILITY"


def load_ranking() -> pd.DataFrame:
    return pd.read_csv(RANKING_CSV)


def load_diag() -> pd.DataFrame:
    return pd.read_csv(DIAG_CSV)


def fig_F10(diag: pd.DataFrame) -> dict:
    sub = diag[diag["plan_aaa_orig_rank"] <= 15].copy()
    # NaN proxy_rank_t1 means pure-hc (no alpha158 members) — annotate separately
    nan_mask = sub["proxy_rank_t1"].isna()
    nan_rows = sub[nan_mask]
    sub = sub[~nan_mask]

    stable = sub["proxy_rank_t1"] <= 15
    n_stable = int(stable.sum())

    fig, ax = setup("single_col", height=3.3)
    # Non-stable points (grey)
    ns = sub[~stable]
    ax.scatter(ns["plan_aaa_orig_rank"], ns["proxy_rank_t1"],
               color=PALETTE["Baseline"], s=28, alpha=0.7,
               edgecolor="black", linewidth=0.4, label="dropped from top-15")
    # Stable points (green star)
    st = sub[stable]
    ax.scatter(st["plan_aaa_orig_rank"], st["proxy_rank_t1"],
               color=PALETTE["GAT"], s=70, marker="*",
               edgecolor="black", linewidth=0.5, zorder=5,
               label=f"stable (n={n_stable}/15)")

    # y=x reference
    diag_max = max(15, sub["proxy_rank_t1"].max() if len(sub) else 15) + 2
    ax.plot([1, diag_max], [1, diag_max], color="black",
            linewidth=0.4, linestyle=":", alpha=0.5)

    # top-15 box
    ax.axhline(15, color=PALETTE["Danger"], linewidth=0.6, linestyle="--", alpha=0.7)
    ax.axvline(15, color=PALETTE["Danger"], linewidth=0.6, linestyle="--", alpha=0.7)

    ax.set_xlabel("Plan AAA original rank")
    ax.set_ylabel("Proxy (T-1 shifted) rank")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, diag_max)
    ax.invert_yaxis()
    ax.set_title(VERDICT, fontsize=8)
    ax.legend(loc="lower right", fontsize=6)

    # Annotate pure-hc rows (no alpha158 members)
    if len(nan_rows) > 0:
        ax.text(0.02, 0.02,
                f"+ {len(nan_rows)} pure-hc group(s) (no proxy): "
                + ", ".join(nan_rows["group_label"].astype(str).tolist()[:3]),
                transform=ax.transAxes, fontsize=5,
                color=PALETTE["Warning"], va="bottom")

    fig.tight_layout()
    paths = save(fig, "F10_plan_aaa_t1_stability")
    plt.close(fig)
    return paths


def _stable_set(diag: pd.DataFrame) -> set[str]:
    sub = diag[(diag["plan_aaa_orig_rank"] <= 15) & (diag["proxy_rank_t1"] <= 15)]
    return set(sub["group_id"].astype(str).tolist())


def fig_S4(ranking: pd.DataFrame, diag: pd.DataFrame) -> dict:
    stable_ids = _stable_set(diag)
    top = ranking.sort_values("rank").head(30).copy()
    top = top.iloc[::-1].reset_index(drop=True)  # plot top at top of axis

    fig, ax = setup("full_width", height=5.5)
    ys = np.arange(len(top))
    colors = [PALETTE["Danger"] if bool(r) else PALETTE["Baseline"]
              for r in top["bh_fdr_rejected"]]

    xerr_lo = top["mean_delta_IC"] - top["bootstrap_ci_lo"]
    xerr_hi = top["bootstrap_ci_hi"] - top["mean_delta_IC"]

    ax.errorbar(top["mean_delta_IC"], ys,
                xerr=[xerr_lo, xerr_hi],
                fmt="none", ecolor=PALETTE["Baseline"],
                elinewidth=0.6, capsize=2)
    ax.scatter(top["mean_delta_IC"], ys, c=colors,
               s=30, edgecolor="black", linewidth=0.3, zorder=4)

    # Star marker for the 5 stable groups
    for i, gid in enumerate(top["group_id"].astype(str)):
        if gid in stable_ids:
            ax.scatter(top.iloc[i]["mean_delta_IC"], i,
                       marker="*", s=120, color=PALETTE["GAT"],
                       edgecolor="black", linewidth=0.4, zorder=6)

    ax.axvline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_yticks(ys)
    ax.set_yticklabels(top["group_label"].astype(str), fontsize=6)
    ax.set_xlabel("mean ΔIC (with 95% bootstrap CI)")
    ax.set_title("S4 — Plan AAA: top-30 group ranking (star = T-1 stable)", fontsize=8)

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["Danger"],
               markersize=6, label="BH-FDR rejected"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["Baseline"],
               markersize=6, label="not rejected"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=PALETTE["GAT"],
               markersize=10, label="T-1 stable (top-15)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=6)

    fig.tight_layout()
    paths = save(fig, "S4_plan_aaa_ranking_top30")
    plt.close(fig)
    return paths


def table_ST4(ranking: pd.DataFrame, diag: pd.DataFrame) -> str:
    stable_ids = _stable_set(diag)
    top = ranking.sort_values("rank").head(20).copy()

    lines = [
        r"\begin{tabular}{rlrrrrcc}",
        r"\toprule",
        r"Rank & Group & $\overline{\Delta IC}$ & NW-$t$ & NW-$p$ & BH-$p_{adj}$ & FDR rej. & T-1 stable \\",
        r"\midrule",
    ]
    for _, row in top.iterrows():
        gid = str(row["group_id"])
        label = str(row["group_label"]).replace("_", r"\_")
        rej = "Y" if bool(row["bh_fdr_rejected"]) else "N"
        stable = "Y" if gid in stable_ids else "N"
        lines.append(
            f"{int(row['rank'])} & {label} & {row['mean_delta_IC']:.4f} & "
            f"{row['nw_t']:.3f} & {row['nw_p_two_sided']:.4f} & "
            f"{row['bh_fdr_p_adj']:.4f} & {rej} & {stable} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    latex = "\n".join(lines)
    write_tex_table("ST4_plan_aaa_top20", latex)
    return latex


def write_caption(diag: pd.DataFrame) -> None:
    out = PROJECT_ROOT / "tables" / "fig_plan_aaa_caption.txt"
    n_stable = int(((diag["plan_aaa_orig_rank"] <= 15) & (diag["proxy_rank_t1"] <= 15)).sum())
    text = (
        f"F10 — Plan AAA T-1 stability. Original top-15 groups (x-axis) versus their "
        f"T-1-shifted proxy ranks (y-axis). Green stars: groups remaining within top-15 "
        f"after T-1 shift. {VERDICT}.\n\n"
        f"S4 — Plan AAA top-30 ranked groups with 95% bootstrap CI on mean ΔIC. "
        f"Red dots: BH-FDR rejected at q=0.05 (none in the positive direction). "
        f"Stars mark the 5 of 15 groups surviving the T-1 stability filter ({VERDICT}).\n\n"
        "S5 — SKIPPED: no per-group permutation distribution CSV exists in this revision.\n\n"
        "ST4 — Top-20 Plan AAA groups; columns include BH-FDR adjusted p-values and T-1 stable flag.\n"
    )
    out.write_text(text)


def main() -> None:
    ranking = load_ranking()
    diag = load_diag()
    f10 = fig_F10(diag)
    s4 = fig_S4(ranking, diag)
    table_ST4(ranking, diag)
    write_caption(diag)
    print(f"[fig_plan_aaa] F10 -> {f10['pdf']}")
    print(f"[fig_plan_aaa] S4  -> {s4['pdf']}")
    print(f"[fig_plan_aaa] ST4 -> tables/ST4_plan_aaa_top20.tex")
    print(f"[fig_plan_aaa] verdict: {VERDICT}")


if __name__ == "__main__":
    main()
