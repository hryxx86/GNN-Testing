#!/usr/bin/env python
"""compute_cost_confirmatory.py — D-RERUN-12F cost-口径 (gross/net) crosswalk.

DESCRIPTIVE economic-sensitivity layer on the confirmatory tuned ladder. IC stays the SOLE
confirmatory metric (Family-1); this module does NOT open a third BH-FDR confirmatory family on
Sharpe (that would over-claim + add multiplicity; the project already treats Sharpe as economic
sensitivity, not robust alpha). It re-expresses each pre-registered Family-1 IC claim with a
net-of-cost Sharpe sibling and FLAGS claims whose gross-IC conclusion and net-Sharpe@10bps
conclusion disagree (`cost_sensitive`), with block-bootstrap CI + leave-one-fold-out so a "flip"
is robust, not an outlier artifact.

WHY this is needed: the confirmatory headline is stated entirely in IC (a gross rank metric); the
official Codex T3 review did not cover transaction cost. H博士 2026-06-21: cost口径 is a BLOCKING
item — every headline claim must carry a gross/net dual口径 because the paper criticizes prior work
for ignoring costs.

NO re-run: every confirmatory cell already stores Sharpe_gross + Sharpe_net_{0,5,10,15,20,30}bps +
mean_turnover_L1 (written at run-time by run_storya_e1_anchor.compute_cost_ladder_sharpe, the
L1-one-way cost convention: net_ret = gross_ret − turnover_L1 × bps/10000). This is a pure analyzer.

Conventions inherited VERBATIM from compute_family1_ladder.py / compute_fc_edge_causal.py:
  - arms L0–L7 + L2s + L5s; the 20-test pairwise family (LADDER_PAIRS + EDGE_PAIRS);
  - EXCLUDE degenerate C/L5s cells (constant predictor → arbitrary-tiebreak L/S Sharpe is
    meaningless, same 25 full + 8 partial = 33 cells Family-1 excludes for undefined IC);
  - fold-level seed-averaging (10 seeds, 12 folds); block bootstrap n_boot=5000.
Gross ΔIC + BH-FDR are COPIED VERBATIM from artifacts/storya_v21_family1/family1_dm_hln.csv (never
re-derived → no drift). Old E1/E6/pilot Sharpe numbers are NEVER read here (H博士: no mixing).

Headline net口径 = 10bps; 0 and 30bps reported as the sensitivity ladder.

Usage (from project root):
  python compute_cost_confirmatory.py
  python compute_cost_confirmatory.py --main-csv experiments/storya_v21_main12_tuned/results.csv \
      --fc-csv experiments/_rerun_colab_staging/storya_v21_main12_fc/results.csv \
      --gross-csv artifacts/storya_v21_family1/family1_dm_hln.csv \
      --output-dir artifacts/storya_v21_cost
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# Reuse the validated bootstrap engine (import-only; do NOT reimplement)
from compute_e6_dm_spa import stationary_bootstrap_ci

# ── CONFIG (LOCKED, mirrors compute_family1_ladder.py) ──
UNIVERSES = ['B', 'C']
CANONICAL_SEEDS = [86, 123, 456, 789, 1024, 2024, 7, 34, 99, 2026]
N_FOLDS = 12
HORIZON = 21
COST_LEVELS_BPS = (0, 5, 10, 15, 20, 30)
HEADLINE_BPS = 10                 # the paper's net口径 headline cost
LADDER_BPS = (0, 10, 30)          # crosswalk + pairwise reported at these (gross / headline / stress)
N_BOOT = 5000
FOLD_BLOCK_SIZE = 1               # 12 fold blocks are the resample unit (= Family-2 convention)
CELL_BLOCK_SIZE = 1               # per-arm ladder: cells treated independent (= run_cost_ladder)
Q2_2025_FOLD = 9                  # 2025Q2 = the high-dispersion regime fold (LOFO sensitivity probe)

# Pre-registered 20-test pairwise family (A-B → arm_A vs arm_B; +Δ means arm_A better)
LADDER_PAIRS = [('L1', 'L0'), ('L2', 'L1'), ('L6', 'L2'), ('L7', 'L2'), ('L2s', 'L2')]
EDGE_PAIRS = [('L3', 'L2'), ('L4', 'L2'), ('L5', 'L2'), ('L5', 'L4'), ('L5', 'L3')]
ALL_PAIRS = LADDER_PAIRS + EDGE_PAIRS

ARMS = ['L0', 'L1', 'L2', 'L2s', 'L3', 'L4', 'L5', 'L5s', 'L6', 'L7']
FC_ARMS = ['L3', 'L4', 'L5']
L2_BASE = 'L2'

CLAIM_LABEL = {
    ('L1', 'L0'): 'MLP (non-graph NN) vs tuned LightGBM',
    ('L2', 'L1'): 'corr-GAT (add graph) vs MLP',
    ('L6', 'L2'): 'complete-graph attention vs corr-GAT',
    ('L7', 'L2'): 'HATS relation-attention vs corr-GAT',
    ('L2s', 'L2'): 'SAGE-Mean vs corr-GAT (aggregation swap)',
    ('L3', 'L2'): '+news edge vs corr-GAT',
    ('L4', 'L2'): '+sector edge vs corr-GAT',
    ('L5', 'L2'): '+sector+news vs corr-GAT',
    ('L5', 'L4'): '+news on top of sector',
    ('L5', 'L3'): '+sector on top of news',
}

_MEAN = lambda x: float(np.mean(x))


# ══════════════════════════════════════════════════════════════
# EXCLUDE degenerate cells (consistency with Family-1)
# ══════════════════════════════════════════════════════════════

def add_exclude_mask(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Tag degenerate cells: n_test_days < the per-(universe,fold) reference (max across arms).

    A constant-prediction cell has no defined cross-sectional ranking → Family-1 drops its undefined
    IC days (its per-day .npy is short/empty). The same cells appear here with n_test_days below the
    fold's full count; their stored Sharpe is the arbitrary-tiebreak L/S portfolio (meaningless), so
    EXCLUDE them. Verified to match Family-1: this flags ONLY C/L5s (25 full + 8 partial = 33).
    """
    df = df.copy()
    ref = df.groupby(['universe', 'fold'])['n_test_days'].transform('max')
    df['_exclude'] = df['n_test_days'] < ref
    n_excl = int(df['_exclude'].sum())
    if n_excl:
        bad = df[df['_exclude']][['universe', 'arm']].drop_duplicates().values.tolist()
        print(f"  [{label}] EXCLUDE {n_excl} degenerate cells (arms: {sorted(set(a for _, a in bad))})")
    return df


# ══════════════════════════════════════════════════════════════
# Loaders — cell-level Sharpe_net from results.csv
# ══════════════════════════════════════════════════════════════

def fold_seedavg(df: pd.DataFrame, u: str, arm: str, col: str) -> dict:
    """{fold: seed-averaged value of `col`} over NON-excluded cells (mirrors Family-2 fold_seedavg)."""
    sub = df[(df.universe == u) & (df.arm == arm) & (~df['_exclude'])]
    out = {}
    for f in range(N_FOLDS):
        vals = sub[sub.fold == f][col].values.astype(np.float64)
        if len(vals):
            out[f] = float(np.mean(vals))
    return out


def _lofo_means(deltas: np.ndarray) -> list[float]:
    """Leave-one-out means of a delta vector (drop each element once)."""
    return [float(np.mean(np.delete(deltas, i))) for i in range(len(deltas))]


# ══════════════════════════════════════════════════════════════
# 1. Per-arm net cost-ladder (DESCRIPTIVE economic sensitivity)
# ══════════════════════════════════════════════════════════════

def run_arm_ladder(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """Per (universe, arm, cost_bps): mean net Sharpe + block-bootstrap CI over cells, PLUS the
    outlier guards learned from the prior E6 Sharpe=75 inflation (CODEX-RR-E1E6-A-bis-04): median,
    LOFO drop-Q2-2025, and the max |gross Sharpe| cell (heavy-tail flag)."""
    rows = []
    for u in UNIVERSES:
        for arm in ARMS:
            sub = df[(df.universe == u) & (df.arm == arm) & (~df['_exclude'])]
            if not len(sub):
                continue
            for c in COST_LEVELS_BPS:
                col = f'Sharpe_net_{c}bps'
                vals = sub[col].values.astype(np.float64)
                _, lo, hi = stationary_bootstrap_ci(vals, _MEAN, n_boot=N_BOOT, block_size=CELL_BLOCK_SIZE)
                noq2 = sub[sub.fold != Q2_2025_FOLD][col].values.astype(np.float64)
                rows.append({
                    'universe': u, 'arm': arm, 'model': sub['model'].iloc[0], 'cost_bps': c,
                    'n_cells': int(len(sub)),
                    'Sharpe_net_mean': round(float(np.mean(vals)), 4),
                    'Sharpe_net_median': round(float(np.median(vals)), 4),
                    'Sharpe_net_ci_lo': round(lo, 4), 'Sharpe_net_ci_hi': round(hi, 4),
                    'Sharpe_net_drop_q2_2025': round(float(np.mean(noq2)), 4) if len(noq2) else None,
                    'mean_turnover_L1': round(float(sub['mean_turnover_L1'].mean()), 3),
                    'max_abs_Sharpe_gross_cell': round(float(sub['Sharpe_gross'].abs().max()), 2),
                })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, 'cost_ladder_by_arm.csv'), index=False)
    return out


# ══════════════════════════════════════════════════════════════
# 2. Per-pair ΔSharpe_net (the 20-test family; DESCRIPTIVE + LOFO)
# ══════════════════════════════════════════════════════════════

def run_pairwise_dsharpe(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """For each (universe, pair, cost ∈ {0,10,30}): fold-level seed-averaged ΔSharpe (arm_A − arm_B),
    block bootstrap CI over the 12 fold blocks, + LOFO sign-stability (no single fold drives a flip).
    NO BH-FDR (descriptive layer; IC remains the sole confirmatory metric).

    Also emits cost_pairwise_folddeltas.csv (the 12 per-fold ΔSharpe values per pair × cost) so any
    per-fold statement (e.g. 'sign-split N pos / M neg') is source-citable (CODEX-A-07)."""
    period = dict(df[['fold', 'test_period']].drop_duplicates().values)
    rows, fold_rows = [], []
    for u in UNIVERSES:
        for a, b in ALL_PAIRS:
            for c in LADDER_BPS:
                col = f'Sharpe_net_{c}bps'
                fa, fb = fold_seedavg(df, u, a, col), fold_seedavg(df, u, b, col)
                folds = [f for f in range(N_FOLDS) if f in fa and f in fb]
                deltas = np.array([fa[f] - fb[f] for f in folds], dtype=np.float64)
                if len(deltas) < 2:
                    continue
                for f, dv in zip(folds, deltas):
                    fold_rows.append({'universe': u, 'pair': f'{a}-{b}', 'arm_A': a, 'arm_B': b,
                                      'cost_bps': c, 'fold': f, 'test_period': period.get(f, ''),
                                      'dSharpe': round(float(dv), 4)})
                _, lo, hi = stationary_bootstrap_ci(deltas, _MEAN, n_boot=N_BOOT, block_size=FOLD_BLOCK_SIZE)
                full = float(np.mean(deltas))
                lofo = _lofo_means(deltas)
                # STRICT robustness (deliberately conservative, NOT a majority vote): the sign is
                # "stable under LOFO" only if it survives EVERY single-fold removal. A near-zero noisy
                # mean (sign-split per-fold ΔSharpe) correctly comes out False. The bootstrap CI
                # (ci_excludes_0) is the PRIMARY inferential signal; this is a supplementary flag.
                sign_stable = bool(full != 0 and all(np.sign(v) == np.sign(full) for v in lofo))
                dq2 = None
                if Q2_2025_FOLD in folds:
                    idx = folds.index(Q2_2025_FOLD)
                    dq2 = round(float(np.mean(np.delete(deltas, idx))), 4)
                rows.append({
                    'universe': u, 'pair': f'{a}-{b}', 'arm_A': a, 'arm_B': b, 'cost_bps': c,
                    'n_fold_blocks': int(len(deltas)),
                    'dSharpe_mean': round(full, 4),
                    'dSharpe_ci_lo': round(lo, 4), 'dSharpe_ci_hi': round(hi, 4),
                    'ci_excludes_0': bool(lo > 0 or hi < 0),
                    'lofo_min_mean': round(min(lofo), 4), 'lofo_max_mean': round(max(lofo), 4),
                    'sign_stable_under_lofo': sign_stable,
                    'dSharpe_drop_q2_2025': dq2,
                })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, 'cost_pairwise_dsharpe.csv'), index=False)
    pd.DataFrame(fold_rows).to_csv(os.path.join(out_dir, 'cost_pairwise_folddeltas.csv'), index=False)
    return out


# ══════════════════════════════════════════════════════════════
# 3. FC arm net check (Family-2 contrasts; DESCRIPTIVE)
# ══════════════════════════════════════════════════════════════

def run_fc_dsharpe(df_main: pd.DataFrame, df_fc: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """6 FC contrasts (L3/L4/L5 fc − frozen L2) net ΔSharpe, fold-level (mirrors Family-2), at
    {0,10,30}bps. DESCRIPTIVE — Family-2's confirmatory call (matched-ΔIC, 0/6 BH, 6/6 underpowered)
    is unchanged; this only shows whether the tiny edge effects survive costs economically."""
    rows = []
    for u in UNIVERSES:
        for arm in FC_ARMS:
            for c in LADDER_BPS:
                col = f'Sharpe_net_{c}bps'
                l2 = fold_seedavg(df_main, u, L2_BASE, col)   # frozen L2 baseline from main table
                fc = fold_seedavg(df_fc, u, arm, col)         # fixed-capacity edge arm from FC table
                folds = [f for f in range(N_FOLDS) if f in l2 and f in fc]
                deltas = np.array([fc[f] - l2[f] for f in folds], dtype=np.float64)
                if len(deltas) < 2:
                    continue
                _, lo, hi = stationary_bootstrap_ci(deltas, _MEAN, n_boot=N_BOOT, block_size=FOLD_BLOCK_SIZE)
                rows.append({
                    'universe': u, 'fc_arm': arm, 'contrast': f'{arm}fc-L2',
                    'edge_added': {'L3': 'news', 'L4': 'sector', 'L5': 'sector+news'}[arm],
                    'cost_bps': c, 'n_fold_blocks': int(len(deltas)),
                    'dSharpe_net_mean': round(float(np.mean(deltas)), 4),
                    'ci_lo': round(lo, 4), 'ci_hi': round(hi, 4),
                    'ci_excludes_0': bool(lo > 0 or hi < 0),
                })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, 'cost_fc_dsharpe.csv'), index=False)
    return out


# ══════════════════════════════════════════════════════════════
# 4. Headline gross/net crosswalk (the deliverable)
# ══════════════════════════════════════════════════════════════

def build_crosswalk(pair_df: pd.DataFrame, gross_csv: str, out_dir: str) -> pd.DataFrame:
    """One row per pre-registered pair: gross ΔIC + BH (COPIED from family1_dm_hln.csv, not
    re-derived) next to net ΔSharpe@10bps + CI + LOFO stability + a cost_sensitive flag (gross-IC
    and net-Sharpe@10bps conclusions disagree in sign)."""
    g = pd.read_csv(gross_csv)
    gmap = {(r.universe, r.arm_A, r.arm_B): (float(r.mean_delta_IC), bool(r.BH_FDR_reject_family))
            for _, r in g.iterrows()}
    p10 = pair_df[pair_df.cost_bps == HEADLINE_BPS]
    rows, dropped_no_gross, missing_net = [], [], []
    for u in UNIVERSES:
        for a, b in ALL_PAIRS:
            key = (u, a, b)
            if key not in gmap:
                dropped_no_gross.append(key)   # legitimately absent from Family-1 (e.g. L7 demoted)
                continue
            gd, gbh = gmap[key]
            pr = p10[(p10.universe == u) & (p10.arm_A == a) & (p10.arm_B == b)]
            if not len(pr):
                missing_net.append(key)        # gross claim exists but net sibling missing → INTEGRITY FAIL
                continue
            pr = pr.iloc[0]
            net = float(pr.dSharpe_mean)
            gsign, nsign = int(np.sign(gd)), int(np.sign(net))
            cost_sensitive = bool(gsign != 0 and nsign != 0 and gsign != nsign)
            rows.append({
                'universe': u, 'pair': f'{a}-{b}', 'claim': CLAIM_LABEL[(a, b)],
                'gross_delta_IC': round(gd, 5), 'gross_BH_FDR_reject': gbh,
                'net_dSharpe_10bps': round(net, 4),
                'net_ci_lo': float(pr.dSharpe_ci_lo), 'net_ci_hi': float(pr.dSharpe_ci_hi),
                'net_ci_excludes_0': bool(pr.ci_excludes_0),
                'net_sign_stable_lofo': bool(pr.sign_stable_under_lofo),
                'gross_sign': gsign, 'net_sign': nsign,
                'cost_sensitive': cost_sensitive,
            })
    out = pd.DataFrame(rows)
    # INTEGRITY GATE (CODEX-A-02): a pair with a gross IC claim MUST get a net sibling — never
    # silently drop a claim from the gross/net crosswalk. Hard-fail, do not just warn.
    if missing_net:
        raise ValueError(f"crosswalk integrity: {len(missing_net)} gross pair(s) lack a net ΔSharpe "
                         f"row (silent-incomplete crosswalk): {missing_net}")
    assert len(out) == len(gmap), (f"crosswalk row count {len(out)} != gross family size {len(gmap)} "
                                   f"(every Family-1 pair must have exactly one net sibling)")
    if dropped_no_gross:
        print(f"  [crosswalk] {len(dropped_no_gross)} pair(s) absent from Family-1 gross (legitimate, "
              f"e.g. L7 demoted): {dropped_no_gross}")
    out.to_csv(os.path.join(out_dir, 'cost_headline_crosswalk.csv'), index=False)
    return out


# ══════════════════════════════════════════════════════════════
# Ledger + summary
# ══════════════════════════════════════════════════════════════

def write_ledger(out_dir: str, n_excluded: int, crosswalk: pd.DataFrame) -> None:
    ledger = {
        'module': 'cost-口径 (gross/net) crosswalk',
        'role': ('DESCRIPTIVE economic-sensitivity layer; IC is the SOLE confirmatory metric. '
                 'NO third BH-FDR confirmatory family on Sharpe (avoids over-claim + multiplicity).'),
        'why': ('confirmatory headline is IC-only (gross rank metric); Codex T3 did not cover cost; '
                'H博士 2026-06-21: cost口径 BLOCKING — every claim needs gross/net dual口径.'),
        'cost_convention': 'L1_one_way: net_ret = gross_ret − turnover_L1 × bps/10000 (run-time stored)',
        'headline_bps': HEADLINE_BPS, 'ladder_bps_reported': list(LADDER_BPS),
        'cost_levels_available': list(COST_LEVELS_BPS),
        'pairwise_family': {
            'n_pairs_per_universe': len(ALL_PAIRS), 'universes': len(UNIVERSES),
            'pairs': [f'{a}-{b}' for a, b in ALL_PAIRS],
            'inference_unit': 'fold-level seed-averaged ΔSharpe (n≈12 fold blocks)',
            'ci': f'stationary block bootstrap over fold blocks (block={FOLD_BLOCK_SIZE}), n_boot={N_BOOT}',
            'robustness': 'leave-one-fold-out sign stability + drop-Q2-2025 (fold 9)',
            'multiplicity': 'NONE — descriptive layer (no BH-FDR; IC family carries the confirmatory call)',
        },
        'gross_source': 'artifacts/storya_v21_family1/family1_dm_hln.csv (mean_delta_IC + BH_FDR_reject_family COPIED verbatim, never re-derived)',
        'degenerate_treatment': f'EXCLUDE {n_excluded} degenerate cells (C/L5s constant-predictor; matches Family-1 25 full + 8 partial)',
        'old_numbers_policy': 'old E1/E6/pilot Sharpe values are NEVER read or quoted here (H博士: no mixing into confirmatory narrative)',
        'cost_sensitive_flag': 'gross-IC sign != net-Sharpe@10bps sign → the claim does not hold across口径',
        'cost_sensitive_pairs': [f"{r.universe} {r.pair} ({r.claim})"
                                 for _, r in crosswalk[crosswalk.cost_sensitive].iterrows()],
    }
    with open(os.path.join(out_dir, 'cost_ledger.json'), 'w') as f:
        json.dump(ledger, f, indent=2, ensure_ascii=False)


def write_summary(out_dir: str, arm_df, pair_df, fc_df, crosswalk) -> None:
    L = [f"# Cost-口径 (gross/net) crosswalk — confirmatory tuned ladder  "
         f"(_generated {time.strftime('%Y-%m-%d %H:%M:%S')}_)\n",
         "**DESCRIPTIVE economic-sensitivity layer.** IC stays the sole confirmatory metric; net "
         "Sharpe is reported as cost sensitivity, NOT a second confirmatory family. Net口径 headline "
         f"= {HEADLINE_BPS}bps (L1-one-way). Gross ΔIC + BH copied verbatim from family1_dm_hln.csv.\n"]

    cs = crosswalk[crosswalk.cost_sensitive]
    cs_claims = cs[cs.gross_BH_FDR_reject]        # cost-sensitive AND a real (BH-significant) IC claim
    cs_noise = cs[~cs.gross_BH_FDR_reject]        # sign disagrees but the gross pair was never a claim
    L.append("## ⚠️ Cost-sensitive findings (gross-IC and net-Sharpe@10bps DISAGREE in sign)\n")
    L.append("**(a) BH-significant IC claims whose口径 does NOT carry to net Sharpe — the ones that "
             "MUST be flagged in the paper:**\n")
    if len(cs_claims):
        L.append(cs_claims[['universe', 'pair', 'claim', 'gross_delta_IC', 'net_dSharpe_10bps',
                            'net_ci_lo', 'net_ci_hi', 'net_ci_excludes_0',
                            'net_sign_stable_lofo']].to_markdown(index=False))
        L.append("\n_Read carefully (PRIMARY signal = the net bootstrap CI): a BH-significant IC harm "
                 "whose net ΔSharpe@10bps CI straddles 0 is NOT evidence the edge helps economically — "
                 "the IC-口径 conclusion simply does not reproduce in net economic口径, and the net "
                 "difference is itself indistinguishable from zero. `net_sign_stable_lofo` is a STRICT "
                 "supplementary check (sign survives EVERY single-fold drop); False here reflects "
                 "sign-split per-fold ΔSharpe around a near-zero mean, NOT a robust reversal._\n")
    else:
        L.append("_None — every BH-significant IC claim keeps its sign at net Sharpe@10bps._\n")
    L.append("\n**(b) Non-significant gross pairs that also disagree (noise-level — NOT headline "
             "claims either口径):**\n")
    if len(cs_noise):
        L.append(cs_noise[['universe', 'pair', 'claim', 'gross_delta_IC', 'net_dSharpe_10bps',
                           'net_sign_stable_lofo']].to_markdown(index=False))
    else:
        L.append("_None._\n")

    L.append("\n## Headline gross/net crosswalk (all 20 pre-registered pairs)\n")
    L.append(crosswalk[['universe', 'pair', 'claim', 'gross_delta_IC', 'gross_BH_FDR_reject',
                        'net_dSharpe_10bps', 'net_ci_lo', 'net_ci_hi', 'net_sign_stable_lofo',
                        'cost_sensitive']].to_markdown(index=False))

    L.append("\n## Per-arm net cost-ladder (descriptive; mean + median + LOFO-Q2-2025 + heavy-tail flag)\n")
    L.append("_`max_abs_Sharpe_gross_cell` flags heavy-tailed cells (the prior E6 had a single "
             "Sharpe=75 cell inflate a mean); compare `Sharpe_net_mean` vs `Sharpe_net_median` and "
             "`Sharpe_net_drop_q2_2025` for fragility._\n")
    L.append(arm_df[arm_df.cost_bps == HEADLINE_BPS][
        ['universe', 'arm', 'model', 'n_cells', 'Sharpe_net_mean', 'Sharpe_net_median',
         'Sharpe_net_ci_lo', 'Sharpe_net_ci_hi', 'Sharpe_net_drop_q2_2025', 'mean_turnover_L1',
         'max_abs_Sharpe_gross_cell']].to_markdown(index=False))
    L.append(f"\n_(net @ {HEADLINE_BPS}bps shown; full 0–30bps ladder in cost_ladder_by_arm.csv)_\n")

    L.append("\n## FC arm net ΔSharpe (Family-2 contrasts; descriptive)\n")
    if len(fc_df):
        L.append(fc_df[fc_df.cost_bps == HEADLINE_BPS][
            ['universe', 'fc_arm', 'edge_added', 'n_fold_blocks', 'dSharpe_net_mean',
             'ci_lo', 'ci_hi', 'ci_excludes_0']].to_markdown(index=False))

    with open(os.path.join(out_dir, 'cost_summary.md'), 'w') as f:
        f.write('\n'.join(L))


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--main-csv', default='experiments/storya_v21_main12_tuned/results.csv')
    p.add_argument('--l7-csv', default='experiments/_rerun_colab_staging/storya_v21_l7_hats_tuned/results.csv')
    p.add_argument('--fc-csv', default='experiments/_rerun_colab_staging/storya_v21_main12_fc/results.csv')
    p.add_argument('--gross-csv', default='artifacts/storya_v21_family1/family1_dm_hln.csv')
    p.add_argument('--output-dir', default='artifacts/storya_v21_cost')
    p.add_argument('--smoke', action='store_true', help='reduce n_boot for a fast wiring check')
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.smoke:
        global N_BOOT
        N_BOOT = 200

    print(f"[COST] loading {args.main_csv} + {args.fc_csv}")
    df_raw = pd.read_csv(args.main_csv)
    # L7 (HATS) lives in its own results.csv (not in main12_tuned) — merge so L7-L2 pairs are covered.
    # Verified: L7 shares L2's exact per-fold test-day counts, so the EXCLUDE reference is unaffected.
    if os.path.exists(args.l7_csv):
        l7 = pd.read_csv(args.l7_csv)
        df_raw = pd.concat([df_raw, l7], ignore_index=True)
        print(f"[COST] merged L7 ({len(l7)} cells) from {args.l7_csv}")
    else:
        print(f"WARN: L7 csv not found ({args.l7_csv}) — L7-L2 pairs will be absent from the crosswalk")
    df_main = add_exclude_mask(df_raw, 'main')
    df_fc = add_exclude_mask(pd.read_csv(args.fc_csv), 'fc')

    # ── VERIFICATION 1: EXCLUDE set matches Family-1 (only C/L5s; count == Family-1 ground truth) ──
    excl = df_main[df_main['_exclude']]
    bad_arms = set((u, a) for u, a in excl[['universe', 'arm']].values)
    assert bad_arms <= {('C', 'L5s')}, f"EXCLUDE leaked beyond C/L5s: {bad_arms}"
    # CODEX-A-03: cross-check the excluded COUNT against Family-1's stability ground truth (not a magic
    # 33) — catches degeneracy drift between this analysis and the IC family.
    stab_path = os.path.join(os.path.dirname(args.gross_csv), 'family1_stability.csv')
    if os.path.exists(stab_path):
        stab = pd.read_csv(stab_path)
        n_expected = int((stab['n_fully_degenerate'] + stab['n_partial_collapse']).sum())
        assert len(excl) == n_expected, (f"EXCLUDE count {len(excl)} != Family-1 stability total "
                                         f"{n_expected} (degeneracy drift vs the IC analysis)")
        print(f"[COST] EXCLUDE check: {len(excl)} cells == Family-1 stability total {n_expected}, all C/L5s")
    else:
        print(f"WARN: {stab_path} not found — exclude-count cross-check skipped ({len(excl)} excluded)")

    print("[COST] (1) per-arm net cost-ladder ...")
    arm_df = run_arm_ladder(df_main, args.output_dir)

    # ── VERIFICATION 2: net spot-check — analyzer means reproduce a direct pandas mean ──
    for u, arm, lab in [('C', 'L0', 'C/LGB'), ('C', 'L1', 'C/MLP')]:
        direct = df_main[(df_main.universe == u) & (df_main.arm == arm)]['Sharpe_net_10bps'].mean()
        viarow = arm_df[(arm_df.universe == u) & (arm_df.arm == arm) &
                        (arm_df.cost_bps == 10)]['Sharpe_net_mean'].iloc[0]
        assert abs(direct - viarow) < 1e-3, f"net spot-check {lab}: {direct} vs {viarow}"
        print(f"[COST] net spot-check {lab} Sharpe_net_10bps mean = {viarow:+.4f} (direct {direct:+.4f}) OK")

    print("[COST] (2) per-pair ΔSharpe (20-test family, fold-level + LOFO) ...")
    pair_df = run_pairwise_dsharpe(df_main, args.output_dir)

    print("[COST] (3) FC arm net ΔSharpe ...")
    fc_df = run_fc_dsharpe(df_main, df_fc, args.output_dir)

    print("[COST] (4) headline gross/net crosswalk ...")
    crosswalk = build_crosswalk(pair_df, args.gross_csv, args.output_dir)
    print(crosswalk[['universe', 'pair', 'gross_delta_IC', 'gross_BH_FDR_reject',
                     'net_dSharpe_10bps', 'cost_sensitive']].to_string(index=False))

    # ── VERIFICATION 3: any cost_sensitive flip must survive LOFO (not a single-fold artifact) ──
    cs = crosswalk[crosswalk.cost_sensitive]
    print(f"\n[COST] {len(cs)} cost-sensitive claim(s):")
    for _, r in cs.iterrows():
        robust = "LOFO-robust" if r.net_sign_stable_lofo else "FRAGILE (single-fold)"
        print(f"   {r.universe} {r.pair} ({r.claim}): gross ΔIC={r.gross_delta_IC:+.4f} "
              f"vs net ΔSharpe@10bps={r.net_dSharpe_10bps:+.4f} → {robust}")

    write_ledger(args.output_dir, int(len(excl)), crosswalk)
    write_summary(args.output_dir, arm_df, pair_df, fc_df, crosswalk)
    print(f"[COST] DONE → {args.output_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
