"""Story A — E6 statistical figures: F9, S16 + T3, ST2.

# SOURCE_CONTRACT:
#   inputs:
#     - path: artifacts/storya_e6_dm_spa/spa_results.csv
#       md5: 2064760556dafcda9f12dbcf9fd7be61
#       n_rows: 3
#       columns: [universe, benchmark, candidates, M, T, p_lower, p_consistent,
#                 p_upper, reject_h0_at_5pct]
#     - path: artifacts/storya_e6_dm_spa/dm_hln_results.csv
#       md5: 26b39ba77763847a0a6e509c39188046
#       n_rows: 10
#       columns: [universe, model_A, model_B, mean_delta_IC, T, NW_lag, DM_stat,
#                 DM_p_normal, HLN_stat, HLN_p_t, BH_FDR_reject, bh_fdr_q]
#     - path: artifacts/storya_e6_dm_spa/multiple_testing_ledger.json
#       schema_version: 2026-05-27-a
#       keys: [primary_storya_trials, ablation_storya_trials,
#              historical_exploratory_trials, ...]
#   outputs:
#     - path: figures/F9_spa_dm_hln.pdf
#       headline_values:
#         - spa_p_consistent_per_universe
#         - dm_hln_n_bh_fdr_rejected
#     - path: figures/S16_multitest_ledger_pyramid.pdf
#     - path: tables/T3_spa_dm_hln.tex
#     - path: tables/ST2_multitest_ledger.tex
#     - path: tables/fig_e6_statistical_caption.txt
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paper_figs.rcparams_storya import (
    setup,
    save,
    write_tex_table,
    PALETTE,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "storya_e6_dm_spa"
SPA_CSV = ARTIFACT_DIR / "spa_results.csv"
DM_CSV = ARTIFACT_DIR / "dm_hln_results.csv"
LEDGER_JSON = ARTIFACT_DIR / "multiple_testing_ledger.json"


# --------------------------------------------------------------------------- #
# F9 — SPA bar (left) + DM/HLN forest (right)
# --------------------------------------------------------------------------- #
def fig_F9() -> dict:
    spa = pd.read_csv(SPA_CSV)
    dm = pd.read_csv(DM_CSV)
    fig, axes = setup("two_panel", height=3.4)
    ax_l, ax_r = axes

    # --- Left: SPA p_consistent bars ---
    labels = spa["universe"].astype(str).tolist()
    p_cons = spa["p_consistent"].astype(float).to_numpy()
    colors = [PALETTE["Danger"] if p < 0.05 else PALETTE["Baseline"]
              for p in p_cons]
    bars = ax_l.bar(labels, p_cons, color=colors, edgecolor="black",
                    linewidth=0.4)
    ax_l.axhline(0.05, color=PALETTE["Danger"], linestyle="--", linewidth=0.7,
                 label=r"$\alpha=0.05$")
    for bar, p in zip(bars, p_cons):
        ax_l.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.01,
                  f"{p:.4f}", ha="center", va="bottom", fontsize=6.5)
    ax_l.set_ylim(0, max(0.5, float(p_cons.max()) * 1.25))
    ax_l.set_ylabel(r"SPA $p_{consistent}$")
    ax_l.set_xlabel("Universe")
    ax_l.set_title("Hansen SPA (consistent variant)")
    ax_l.legend(fontsize=7, loc="upper left")

    # --- Right: DM/HLN forest ---
    dm = dm.reset_index(drop=True)
    n = len(dm)
    y = np.arange(n)
    deltas = dm["mean_delta_IC"].astype(float).to_numpy()
    rejected = dm["BH_FDR_reject"].astype(bool).to_numpy()
    for i in range(n):
        color = PALETTE["Danger"] if rejected[i] else PALETTE["Baseline"]
        ax_r.scatter(deltas[i], y[i], color=color, s=24, edgecolors="black",
                     linewidths=0.3, zorder=3)
        p_t = dm["HLN_p_t"].iloc[i]
        ax_r.text(deltas[i], y[i] + 0.15, f"p={p_t:.3f}",
                  fontsize=5.5, ha="left", va="bottom")
    ax_r.axvline(0, color="black", linewidth=0.4, linestyle="--")
    ax_r.set_yticks(y)
    ylabels = [f"{r['universe']}: {r['model_A']} vs {r['model_B']}"
               for _, r in dm.iterrows()]
    ax_r.set_yticklabels(ylabels, fontsize=6)
    ax_r.invert_yaxis()
    ax_r.set_xlabel(r"Mean $\Delta$IC (A $-$ B)")
    ax_r.set_title("DM/HLN paired tests")
    ax_r.legend(handles=[
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=PALETTE["Danger"], markersize=5,
                   label="BH-FDR reject (q=0.05)"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=PALETTE["Baseline"], markersize=5,
                   label="not rejected"),
    ], fontsize=6, loc="lower right")
    fig.suptitle("F9 — SPA + DM/HLN family", y=1.01, fontsize=9)
    fig.tight_layout()
    paths = save(fig, "F9_spa_dm_hln")
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- #
# S16 — Multi-testing ledger pyramid
# --------------------------------------------------------------------------- #
def _ledger_counts(ledger: dict) -> dict:
    primary = ledger.get("primary_storya_trials", {})
    ablation = ledger.get("ablation_storya_trials", {})
    hist = ledger.get("historical_exploratory_trials", {})

    primary_total = int(primary.get("total_E1_cells", 0))
    ablation_total = int(ablation.get("E3_cells_new", 0)) + \
        int(ablation.get("E4_cells_new", 0))
    hist_total = int(hist.get("total_exploratory_units", {}).get("sum", 0))
    return {
        "primary_storya_trials (post-E1 SPA family)": primary_total,
        "ablation_storya_trials (E3+E4 edge family)": ablation_total,
        "historical_exploratory_trials (NOT in SPA)": hist_total,
    }


def fig_S16() -> dict:
    with LEDGER_JSON.open() as f:
        ledger = json.load(f)
    counts = _ledger_counts(ledger)
    # Build pyramid: top (smallest = primary), bottom (largest = historical).
    # Sort descending by value so the widest sits at the bottom.
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    fig, ax = setup("full_width", height=3.0)
    max_v = max(v for _, v in items) or 1
    bar_h = 0.65
    y_positions = list(range(len(items)))
    colors = [PALETTE["Warning"], PALETTE["SAGE-Mean"], PALETTE["GAT"]]
    for i, ((label, value), color) in enumerate(zip(items, colors)):
        # Centered horizontal bar -> pyramid look
        half = value / 2.0
        ax.barh(y_positions[i], value, left=-half, height=bar_h, color=color,
                edgecolor="black", linewidth=0.4)
        ax.text(0, y_positions[i], f"{label}\n n = {value}", ha="center",
                va="center", fontsize=7,
                color="white" if value > 0.4 * max_v else "black")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(["bottom (widest)" if i == 0 else
                        ("middle" if i == 1 else "top (narrowest)")
                        for i in range(len(items))])
    ax.set_xlim(-max_v * 0.65, max_v * 0.65)
    ax.set_xlabel("Trial count (centered = pyramid width)")
    ax.set_title("S16 — Multi-testing ledger pyramid")
    ax.invert_yaxis()
    fig.tight_layout()
    paths = save(fig, "S16_multitest_ledger_pyramid")
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- #
# T3 — SPA + DM/HLN combined LaTeX table
# --------------------------------------------------------------------------- #
def table_T3() -> str:
    spa = pd.read_csv(SPA_CSV)
    dm = pd.read_csv(DM_CSV)
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"\multicolumn{7}{l}{\textbf{Panel A: Hansen SPA (consistent variant)}} \\",
        r"\midrule",
        r"Universe & Benchmark & Candidates & $M$ & $T$ & "
        r"$p_{consistent}$ & reject@5\% \\",
        r"\midrule",
    ]
    for _, r in spa.iterrows():
        rej = r"\checkmark" if bool(r["reject_h0_at_5pct"]) else "no"
        cands = str(r["candidates"]).replace("_", r"\_")
        lines.append(
            f"{r['universe']} & {r['benchmark']} & "
            f"{cands} & "
            f"{int(r['M'])} & {int(r['T'])} & "
            f"{float(r['p_consistent']):.4f} & {rej} \\\\"
        )
    lines += [
        r"\midrule",
        r"\multicolumn{7}{l}{\textbf{Panel B: DM/HLN paired tests "
        r"(NW-HAC SE, BH-FDR q=0.05)}} \\",
        r"\midrule",
        r"Universe & Pair (A vs B) & $\overline{\Delta IC}$ & DM stat & "
        r"HLN stat & HLN $p$ & BH-FDR \\",
        r"\midrule",
    ]
    for _, r in dm.iterrows():
        pair = f"{r['model_A']} vs {r['model_B']}"
        rej = r"\checkmark" if bool(r["BH_FDR_reject"]) else "no"
        lines.append(
            f"{r['universe']} & {pair} & "
            f"{float(r['mean_delta_IC']):.4f} & "
            f"{float(r['DM_stat']):.3f} & {float(r['HLN_stat']):.3f} & "
            f"{float(r['HLN_p_t']):.4f} & {rej} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    latex = "\n".join(lines)
    write_tex_table("T3_spa_dm_hln", latex)
    return latex


# --------------------------------------------------------------------------- #
# ST2 — Multi-testing ledger summary
# --------------------------------------------------------------------------- #
def table_ST2() -> str:
    with LEDGER_JSON.open() as f:
        ledger = json.load(f)
    primary = ledger.get("primary_storya_trials", {})
    ablation = ledger.get("ablation_storya_trials", {})
    hist = ledger.get("historical_exploratory_trials", {})
    dm_app = ledger.get("dm_hln_application", {})
    spa_app = ledger.get("spa_application", {})
    cost_bps = ledger.get("cost_ladder_bps", [])

    rows = [
        ("Primary E1 cells (SPA / DM-HLN family)",
         primary.get("total_E1_cells", 0),
         f"BH-FDR q={dm_app.get('bh_fdr_q', 0.05)}; "
         f"SPA M={spa_app.get('universe_B_M_candidates', 0)} per univ"),
        ("E3 news-encoding cells (ablation family)",
         ablation.get("E3_cells_new", 0),
         "edge-ablation BH-FDR q=0.05 family of 5 pairs"),
        ("E4 alpha-edge cells (ablation family)",
         ablation.get("E4_cells_new", 0),
         "edge-ablation BH-FDR q=0.05 family of 5 pairs"),
        ("Plan AAA group-rank tests (historical, disclosed)",
         hist.get("plan_aaa_group_ranking", {}).get("groups_tested", 0),
         "0/61 pass at q=0.05; NOT in post-E1 SPA family"),
        ("Horizon-ablation cells (historical)",
         hist.get("horizon_ablation", {}).get("total_cells", 0),
         "21d horizon selection; NOT in SPA family"),
        ("Loss-horserace cells (historical)",
         hist.get("loss_horserace_phase5_step3", {}).get("total_cells", 0),
         "MSE locked for E1; NOT in SPA family"),
        ("Cost-ladder bps levels",
         len(cost_bps),
         f"bps = {cost_bps}; reported as descriptive ladder, not multi-tested"),
    ]
    lines = [
        r"\begin{tabular}{lrl}",
        r"\toprule",
        r"Family & Count & Coverage note \\",
        r"\midrule",
    ]
    for label, count, note in rows:
        label_esc = label.replace("_", r"\_").replace("&", r"\&")
        note_esc = note.replace("_", r"\_").replace("&", r"\&")
        lines.append(f"{label_esc} & {count} & {note_esc} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    latex = "\n".join(lines)
    write_tex_table("ST2_multitest_ledger", latex)
    return latex


def write_caption() -> None:
    out = PROJECT_ROOT / "tables" / "fig_e6_statistical_caption.txt"
    text = (
        "F9 — Hansen SPA (consistent variant; left) + DM/HLN paired tests "
        "(right). Left panel: p_consistent per universe (B / C / joint) "
        "with dashed reference line at α=0.05; no universe rejects the SPA "
        "null. Right panel: 10 DM/HLN paired-test rows (per-universe model "
        "pairs); colour red = BH-FDR reject at q=0.05, grey = not rejected. "
        "Annotations show HLN small-sample t-test p-values.\n\n"
        "S16 — Multi-testing ledger pyramid. Centered horizontal bars sized "
        "by trial count: primary_storya_trials (post-E1 SPA family), "
        "ablation_storya_trials (E3+E4 edge family), and "
        "historical_exploratory_trials (NOT controlled by SPA; disclosed "
        "for transparency per Codex T3 A-bis-06).\n\n"
        "T3 — SPA + DM/HLN combined results. Panel A: SPA per universe "
        "(3 rows). Panel B: DM/HLN per pair (10 rows). BH-FDR q=0.05 "
        "applied within each family separately.\n\n"
        "ST2 — Multi-testing ledger summary. Lists all trial families with "
        "counts and coverage notes, clarifying which families are inside the "
        "post-E1 SPA control and which are disclosed historical exploration.\n"
    )
    out.write_text(text)


def main() -> None:
    fig_F9()
    fig_S16()
    table_T3()
    table_ST2()
    write_caption()
    print("[fig_e6_statistical] OK — F9, S16, T3, ST2 + caption")


if __name__ == "__main__":
    main()
