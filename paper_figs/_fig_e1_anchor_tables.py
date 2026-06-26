"""Helper for fig_e1_anchor.py — T1, T2 LaTeX tables + caption writer.

Split out of fig_e1_anchor.py to keep that script under the 350-line cap.
No standalone entry-point; imported by fig_e1_anchor.main().
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import pandas as pd

from paper_figs.rcparams_storya import write_tex_table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "storya_e6_dm_spa"

L1_CAVEAT = ("Universe C composition derives from Plan AAA which had same-day "
             "Alpha158 leak; T-1 diagnostic confirms LOW STABILITY (5/15)")
L6_CAVEAT = ("Fold 4 (Q2-2025) is a known regime outlier; LOFO-4 column drops "
             "IC by 38-72%")


def table_T1() -> str:
    df = pd.read_csv(ARTIFACT_DIR / "bootstrap_ci.csv")
    lines = [
        r"\begin{tabular}{llrcc}",
        r"\toprule",
        r"Universe & Model & $n_{cells}$ & IC mean [95\% CI] & "
        r"$S_{gross}$ mean [95\% CI] \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        ic = (f"{r['IC_mean']:.4f} "
              f"[{r['IC_mean_ci_lo']:.4f}, {r['IC_mean_ci_hi']:.4f}]")
        sh = (f"{r['Sharpe_gross_mean']:.2f} "
              f"[{r['Sharpe_gross_ci_lo']:.2f}, {r['Sharpe_gross_ci_hi']:.2f}]")
        lines.append(
            f"{r['universe']} & {r['model']} & {int(r['n_cells'])} & "
            f"{ic} & {sh} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    latex = "\n".join(lines)
    write_tex_table("T1_headline", latex)
    return latex


def table_T2() -> str:
    df = pd.read_csv(ARTIFACT_DIR / "e1_three_column_summary.csv")
    lines = [
        r"\footnotesize",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Univ & Model & "
        r"IC$_{full}$ [CI] & IC$_{lofo4}$ [CI] & IC$_{fold4}$ [CI] & "
        r"$S_{net,10}^{full}$ [CI] & $S_{net,10}^{lofo4}$ [CI] & "
        r"$S_{net,10}^{fold4}$ [CI] \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        def _ic(prefix: str) -> str:
            return (f"{r[f'IC_{prefix}_mean']:.4f} "
                    f"[{r[f'IC_{prefix}_ci_lo']:.4f},"
                    f"{r[f'IC_{prefix}_ci_hi']:.4f}]")

        def _sh(prefix: str) -> str:
            return (f"{r[f'Sharpe_net_10bps_{prefix}_mean']:.2f} "
                    f"[{r[f'Sharpe_net_10bps_{prefix}_ci_lo']:.2f},"
                    f"{r[f'Sharpe_net_10bps_{prefix}_ci_hi']:.2f}]")

        lines.append(
            f"{r['universe']} & {r['model']} & "
            f"{_ic('full')} & {_ic('lofo4')} & {_ic('fold4only')} & "
            f"{_sh('full')} & {_sh('lofo4')} & {_sh('fold4only')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    latex = "\n".join(lines)
    write_tex_table("T2_three_column_robustness", latex)
    return latex


def write_caption() -> None:
    out = PROJECT_ROOT / "tables" / "fig_e1_anchor_caption.txt"
    text = (
        "F2 — Cumulative daily IC trajectory (8 panels: 4 models × 2 universes). "
        "Grey lines = 10 individual seeds; coloured line = across-seed mean. "
        "Amber-shaded region = Fold 4 (Q2-2025 regime outlier). NOTE: Original "
        "plan called for cumulative L/S PnL; per_day_ic .npy files contain "
        "daily Spearman IC arrays (not L/S returns), so F2 reports cumulative "
        "IC as a ranking-quality proxy.\n\n"
        f"F3 — LOFO sensitivity heatmap (8 rows × 6 cols). Each cell shows "
        f"IC mean when the indicated fold is left out (`none` = full data). "
        f"Diverging RdBu_r colormap centered at 0. {L6_CAVEAT}.\n\n"
        f"F4 — Per-fold IC bars (mean ± seed std). 5 folds × 8 (universe × "
        f"model) groups. Solid bars = Universe B; hatched bars = Universe C. "
        f"Fold 4 tick is highlighted red. {L6_CAVEAT}.\n\n"
        "S1 — Per-cell IC vs net Sharpe scatter (400 cells = 4 models × 2 "
        "universes × 10 seeds × 5 folds). Color = model, marker = universe "
        "(o=B, s=C).\n\n"
        "S2 — Top-3 / Bottom-3 Sharpe outliers per (universe, model), annotated "
        "with (cell_id, fold, seed). Green = TOP3, red = BOT3.\n\n"
        f"S3 — Per-day IC time series, 8 (universe × model) seed-averaged "
        f"curves. Solid = Univ B, dashed = Univ C. Fold 4 shaded red. "
        f"{L6_CAVEAT}.\n\n"
        f"S17 — Bootstrap CI overlay: IC mean with 95% block-bootstrap CI "
        f"under three regimes (full / LOFO-4 / Fold-4-only). 8 rows × 3 "
        f"regimes. {L6_CAVEAT}.\n\n"
        "T1 — Headline IC + gross Sharpe with 95% bootstrap CI, 8 rows "
        "(2 universes × 4 models).\n\n"
        f"T2 — 3-column robustness table: IC and net-10bps Sharpe under full / "
        f"LOFO-4 / Fold-4-only conditions. {L6_CAVEAT}.\n"
    )
    out.write_text(text)
