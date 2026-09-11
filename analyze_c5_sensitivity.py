#!/usr/bin/env python
"""analyze_c5_sensitivity.py — C5 test-informed feature-subset sensitivity (POST-HOC, not confirmatory).

Companion to `compute_family1_ladder.py --universes C5 --arms L0,L1 --sensitivity` (which produces the
HLN / block-bootstrap CI / MDE rows). This script adds the seed-level robustness numbers the paper's
appendix C.2 reports for the confirmatory contrasts, checks run integrity, and assembles the one-table
side-by-side C5 vs C vs B for L1-L0 (MLP - LightGBM). Spec: docs/c5_rerun_brief_2026-09-10.md §4.4/§6.

Estimands (identical to analyze_paper_eval_robustness.py checks 1+2, appendix C.2):
  per-seed pooled IC   = n_test_days-weighted mean of the 12 fold-level IC_mean of that seed
  per-seed ΔIC         = pooled_L1[s] - pooled_L0[s]
  k/10 same sign       = # seeds whose ΔIC sign == sign(mean over seeds)
  m/10 LOSO flips      = # seeds whose removal flips the sign of the mean ΔIC
Optional PAIRED daily contrast (C minus C5): d(t) = ΔIC_C(t) - ΔIC_C5(t) on the seed-averaged daily
ΔIC series (same 749 test days in the same chronological fold order, same 10 seeds) → HLN p + 21d
stationary block-bootstrap CI. Positive = the confirmatory C contrast exceeds the C5 one. This is a
CONDITIONAL SUBSET CONTRAST (feature restriction 51→20 + re-tuning); C5's columns were themselves
selected with evaluation-period outcomes (brief §9.9), so it does NOT identify leakage inflation and
does not establish selection-independent performance.

Inputs (read-only):
  --c5-main-dir      experiments/storya_v21_main12_c5     results.csv / manifest.csv / per_day_ic / _universe_c5.json
  --c5-family-dir    artifacts/storya_v21_family1_c5      family1_{dm_hln,ic_ci,mde}.csv (sensitivity run)
  --conf-main-dir    experiments/storya_v21_main12_tuned  confirmatory results.csv (+ per_day_ic for --paired)
  --conf-family-dir  artifacts/storya_v21_family1         confirmatory family1_{dm_hln,ic_ci,mde}.csv
  --frozen           experiments/storya_v21_tune/frozen_hparams_c5.json (md5 cross-checked vs provenance)
Outputs (→ --out-dir, default = --c5-family-dir):
  c5_run_integrity.json, c5_seed_robustness.csv, c5_paired_contrast.csv, c5_comparison.csv, c5_comparison.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

from analyze_paper_eval_robustness import seed_pooled          # appendix C.2 estimator (import-only)
from compute_family1_ladder import (                            # validated loaders (import-only)
    collect_arm_matrix, seed_avg_pooled, seed_avg_per_fold, CANONICAL_SEEDS, N_FOLDS, HORIZON, BLOCK_SIZE, N_BOOT,
    MDE_FACTOR,
)
from compute_e6_dm_spa import hln_test, stationary_bootstrap_ci

PAIR = ('L1', 'L0')


# ══════════════════════════════════════════════════════════════
# 1. run integrity (240 cells, no failures, no duplicate ids, per_day complete, 20 features, md5)
# ══════════════════════════════════════════════════════════════

def fold_calendar_days(conf_results_csv: str) -> dict:
    """{fold: full test-day count} from the confirmatory run (arm L0, identical across seeds/arms/universes:
    every cell of a fold scores the same calendar days unless it partially collapsed). This is the frozen
    12-fold calendar's per-fold day count (sum = 749)."""
    df = pd.read_csv(conf_results_csv)
    sub = df[(df.arm == 'L0')]
    cal = sub.groupby('fold')['n_test_days'].agg(['min', 'max'])
    assert (cal['min'] == cal['max']).all(), 'confirmatory L0 fold day counts differ across cells'
    return {int(f): int(v) for f, v in cal['max'].items()}


def run_integrity(c5_main: str, frozen_path: str | None, conf_results_csv: str, strict: bool = True) -> dict:
    """CODEX-TP2-A-02/A-03: every C5 cell must carry the FULL per-fold day count of the frozen calendar
    (a shortened .npy = undefined-IC days dropped → positional packing would silently misalign dates);
    outside smoke mode the frozen file, the provenance file (mode TUNED per-arm, md5 match, applied
    C5_L0/C5_L1 == frozen winners) and 240 completed cells are all REQUIRED for PASS."""
    res = pd.read_csv(os.path.join(c5_main, 'results.csv'))
    man = pd.read_csv(os.path.join(c5_main, 'manifest.csv'))
    uni = json.load(open(os.path.join(c5_main, '_universe_c5.json')))
    prov_p = os.path.join(c5_main, '_frozen_hp_provenance.json')
    prov = json.load(open(prov_p)) if os.path.exists(prov_p) else None
    frozen = json.load(open(frozen_path)) if (frozen_path and os.path.exists(frozen_path)) else None
    n_expected = 2 * len(CANONICAL_SEEDS) * N_FOLDS
    per_day = os.path.join(c5_main, 'per_day_ic')
    cal = fold_calendar_days(conf_results_csv)
    npy_ok, npy_len_mismatch, short_cells = 0, [], []
    for r in res.itertuples():
        p = os.path.join(per_day, f'{r.universe}_{r.arm}_s{r.seed}_f{r.fold}.npy')
        if os.path.exists(p):
            npy_ok += 1
            L = len(np.load(p))
            if L != int(r.n_test_days):
                npy_len_mismatch.append(os.path.basename(p))
            if L != cal[int(r.fold)]:                      # partial/full collapse vs frozen calendar
                short_cells.append({'file': os.path.basename(p), 'len': L, 'calendar': cal[int(r.fold)]})
    md5_frozen = hashlib.md5(open(frozen_path, 'rb').read()).hexdigest() if frozen else None
    # FINGNN-B-02 (TP2-B): identify the exact inputs these statistics were computed from
    run_prov_p = os.path.join(c5_main, '_run_provenance.json')
    run_prov = json.load(open(run_prov_p)) if os.path.exists(run_prov_p) else None
    if isinstance(run_prov, list):
        run_prov = run_prov[-1] if run_prov else None
    code_id_p = os.path.join(c5_main, '_code_identity_t4.json')
    code_id = json.load(open(code_id_p)) if os.path.exists(code_id_p) else None
    inputs = {
        'c5_main_dir': c5_main, 'conf_results_csv': conf_results_csv,
        'results_csv_md5': hashlib.md5(open(os.path.join(c5_main, 'results.csv'), 'rb').read()).hexdigest(),
        'manifest_csv_md5': hashlib.md5(open(os.path.join(c5_main, 'manifest.csv'), 'rb').read()).hexdigest(),
        'device': run_prov.get('device') if run_prov else None,
        'platform': run_prov.get('platform') if run_prov else None,
        'git_rev': run_prov.get('git_rev') if run_prov else None,
        'source_clean': run_prov.get('source_clean') if run_prov else None,
        'code_identity_post_hoc': ({'all_modules_match': code_id.get('all_modules_match_9008dbe'),
                                    'commit': '9008dbe', 'verified_at': code_id.get('verified_at')} if code_id else None),
    }
    prov_ok = None
    if prov is not None and frozen is not None:
        applied = prov.get('applied', {})
        prov_ok = bool(
            prov.get('mode') == 'TUNED per-arm' and prov.get('frozen_md5') == md5_frozen
            and set(applied) == {'C5_L0', 'C5_L1'}
            and all(applied[k]['src'] == k and applied[k]['params'] == frozen['studies'][k]['winner_params']
                    for k in ('C5_L0', 'C5_L1'))
            and frozen.get('complete') is True and frozen.get('n_studies') == frozen.get('expected') == 2
        )
    out = {
        'n_results_rows': int(len(res)), 'n_expected': n_expected,
        'universes': sorted(res.universe.unique().tolist()), 'arms': sorted(res.arm.unique().tolist()),
        'cell_id_unique': bool(res.cell_id.is_unique),
        'cell_id_range': [int(res.cell_id.min()), int(res.cell_id.max())] if len(res) else None,
        'cell_id_disjoint_from_confirmatory_0_2399': bool(len(res) and res.cell_id.min() >= 2400),
        'manifest_status_counts': man.status.value_counts().to_dict(),
        'n_failed': int((man.status != 'completed').sum()),
        'converged_all': bool((res.converged_flag == 1).all()) if len(res) else False,
        'n_test_days_total_per_arm': res.groupby(['arm', 'seed'])['n_test_days'].sum().unstack().iloc[:, 0].to_dict()
                                     if len(res) else {},
        'per_day_npy_present': npy_ok, 'per_day_npy_len_mismatch': npy_len_mismatch,
        'frozen_calendar_days_per_fold': cal, 'frozen_calendar_total': int(sum(cal.values())),
        'cells_not_full_calendar_length': short_cells,          # any entry = date alignment NOT guaranteed
        'n_features': uni['n_features'], 'feature_names': uni['feature_names'],
        'inputs': inputs,
        'frozen_hparams_path': frozen_path, 'frozen_present': frozen is not None, 'frozen_md5': md5_frozen,
        'provenance_present': prov is not None,
        'provenance_mode': prov.get('mode') if prov else None,
        'provenance_md5': prov.get('frozen_md5') if prov else None,
        'provenance_gate_ok': prov_ok,   # mode TUNED per-arm + md5 match + applied C5_L0/C5_L1 == frozen winners + 2/2
        'applied_hparams': prov.get('applied') if prov else None,
        'strict': strict,
    }
    out['PASS'] = bool(
        out['n_results_rows'] == n_expected and out['cell_id_unique'] and out['n_failed'] == 0
        and out['converged_all'] and npy_ok == n_expected and not npy_len_mismatch and not short_cells
        and out['n_features'] == 20 and out['cell_id_disjoint_from_confirmatory_0_2399']
        and (prov_ok is True if strict else prov_ok in (True, None))
    )
    return out


# ══════════════════════════════════════════════════════════════
# 2. per-seed sign k/10 + leave-one-seed-out m/10 (appendix C.2 estimator)
# ══════════════════════════════════════════════════════════════

def seed_robustness(results_csv: str, universe: str, a: str = PAIR[0], b: str = PAIR[1]) -> dict:
    df = pd.read_csv(results_csv)
    pa, pb = seed_pooled(df, universe, a), seed_pooled(df, universe, b)
    seeds = sorted(set(pa) & set(pb))
    d = np.array([pa[s] - pb[s] for s in seeds])
    full = float(d.mean())
    same = int((np.sign(d) == np.sign(full)).sum())
    loso = np.array([np.delete(d, i).mean() for i in range(len(d))])
    flips = int((np.sign(loso) != np.sign(full)).sum())
    return {'universe': universe, 'contrast': f'{a}-{b}', 'pooled_delta_IC': round(full, 5),
            'n_seeds': len(seeds), 'n_same_sign': same, 'loso_sign_flips': flips,
            'per_seed_min': round(float(d.min()), 5), 'per_seed_max': round(float(d.max()), 5),
            'per_seed_delta': {int(s): round(float(v), 5) for s, v in zip(seeds, d)},
            'pooled_IC_L1_mean_over_seeds': round(float(np.mean([pa[s] for s in seeds])), 5),
            'pooled_IC_L0_mean_over_seeds': round(float(np.mean([pb[s] for s in seeds])), 5)}


# ══════════════════════════════════════════════════════════════
# 3. paired daily contrast: (L1-L0)_C5 minus (L1-L0)_C on seed-averaged daily ΔIC
# ══════════════════════════════════════════════════════════════

def _delta_series(per_day_dir: str, universe: str) -> np.ndarray:
    ic_a = seed_avg_pooled(collect_arm_matrix(per_day_dir, universe, PAIR[0]))
    ic_b = seed_avg_pooled(collect_arm_matrix(per_day_dir, universe, PAIR[1]))
    n = min(len(ic_a), len(ic_b))
    return ic_a[:n] - ic_b[:n]


def paired_contrast(c5_main: str, conf_main: str, other: str, n_boot: int, strict: bool = True) -> dict:
    d5 = _delta_series(os.path.join(c5_main, 'per_day_ic'), 'C5')
    dc = _delta_series(os.path.join(conf_main, 'per_day_ic'), other)
    if strict:   # full run: both series must be the same 749 seed-averaged test days (fold order 0..11)
        assert len(d5) == len(dc), f'paired series length mismatch: C5 {len(d5)} vs {other} {len(dc)}'
    n = min(len(d5), len(dc))
    assert n >= 2 * BLOCK_SIZE, f'paired series too short: {n}'
    d = dc[:n] - d5[:n]                                   # positive = confirmatory contrast exceeds C5's
    hln_stat, hln_p, T = hln_test(d)
    _, hln_p21, _ = hln_test(d, lag=HORIZON)
    mean_d, lo, hi = stationary_bootstrap_ci(d, lambda x: float(np.mean(x)), n_boot=n_boot,
                                             block_size=BLOCK_SIZE)
    from arch.bootstrap import StationaryBootstrap          # same SE/MDE construction as family1 run_ci_and_mde
    sb = StationaryBootstrap(BLOCK_SIZE, d, seed=86)
    se_block = float(np.std(sb.apply(lambda x: float(np.mean(x)), n_boot).ravel(), ddof=1))
    return {'contrast': f'(L1-L0)_{other} - (L1-L0)_C5', 'T': int(T),
            'mean_delta_C5': round(float(d5[:n].mean()), 5), f'mean_delta_{other}': round(float(dc[:n].mean()), 5),
            'mean_paired_diff': round(float(mean_d), 5), 'ci_lo': round(float(lo), 5), 'ci_hi': round(float(hi), 5),
            'ci_excludes_0': bool(lo > 0 or hi < 0),
            'SE_block': round(se_block, 5), 'MDE_2p8xSE': round(MDE_FACTOR * se_block, 5),
            'HLN_stat': round(float(hln_stat), 4), 'HLN_p_t': round(float(hln_p), 5),
            'HLN_p_t_lag21': round(float(hln_p21), 5),
            'n_boot': n_boot, 'block_size': BLOCK_SIZE,
            'note': (f'post-hoc conditional subset contrast; same test days + same 10 seeds; positive = {other} '
                     f'contrast larger than C5; not an identified leakage-inflation effect (C5 selection is test-informed)')}


# ══════════════════════════════════════════════════════════════
# 3b. leave-one-fold-out pooled statistics for the dominant fold (FINGNN-R-A-01: 2025Q2 concentration)
# ══════════════════════════════════════════════════════════════

def ex_fold_stats(main_dir: str, universe: str, ex_fold: int, n_boot: int) -> dict:
    """Pooled seed-averaged daily ΔIC statistics with one fold EXCLUDED (HLN both lags + 21d block CI + SE/MDE)."""
    pd_dir = os.path.join(main_dir, 'per_day_ic')
    a = seed_avg_per_fold(collect_arm_matrix(pd_dir, universe, PAIR[0]))
    b = seed_avg_per_fold(collect_arm_matrix(pd_dir, universe, PAIR[1]))
    parts, fold_delta = [], None
    for f in range(N_FOLDS):
        if a.get(f) is None or b.get(f) is None:
            continue
        n = min(len(a[f]), len(b[f]))
        dd = a[f][:n] - b[f][:n]
        if f == ex_fold:
            fold_delta = float(dd.mean())
            continue
        parts.append(dd)
    d = np.concatenate(parts)
    _, p, T = hln_test(d)
    _, p21, _ = hln_test(d, lag=HORIZON)
    m, lo, hi = stationary_bootstrap_ci(d, lambda x: float(np.mean(x)), n_boot=n_boot, block_size=BLOCK_SIZE)
    from arch.bootstrap import StationaryBootstrap
    se = float(np.std(StationaryBootstrap(BLOCK_SIZE, d, seed=86).apply(lambda x: float(np.mean(x)), n_boot).ravel(), ddof=1))
    return {'universe': universe, 'excluded_fold': ex_fold, 'excluded_fold_delta_IC': round(fold_delta, 5) if fold_delta is not None else None,
            'T_ex': int(T), 'mean_delta_IC_ex': round(float(m), 5), 'ci_lo': round(float(lo), 5), 'ci_hi': round(float(hi), 5),
            'HLN_p_t': round(float(p), 5), 'HLN_p_t_lag21': round(float(p21), 5),
            'SE_block': round(se, 5), 'MDE_2p8xSE': round(MDE_FACTOR * se, 5)}


# ══════════════════════════════════════════════════════════════
# 3c. device replication: primary vs replicate result dirs, cell-level (FINGNN-B-02 generator in repo)
# ══════════════════════════════════════════════════════════════

def device_replication(primary_dir: str, replicate_dir: str) -> tuple:
    p = pd.read_csv(os.path.join(primary_dir, 'results.csv')); r = pd.read_csv(os.path.join(replicate_dir, 'results.csv'))
    j = p.merge(r, on=['arm', 'seed', 'fold'], suffixes=('_primary', '_replicate'))
    assert len(j) == len(p) == len(r), f'replicate join mismatch {len(j)} vs {len(p)}/{len(r)}'
    rows = []
    for arm in sorted(j.arm.unique()):
        s = j[j.arm == arm]; d = s.IC_mean_replicate - s.IC_mean_primary
        rows.append({'arm': arm, 'n_cells': int(len(s)),
                     'corr_cell_IC': round(float(np.corrcoef(s.IC_mean_primary, s.IC_mean_replicate)[0, 1]), 4),
                     'mean_IC_primary': round(float(s.IC_mean_primary.mean()), 5),
                     'mean_IC_replicate': round(float(s.IC_mean_replicate.mean()), 5),
                     'mean_abs_diff': round(float(d.abs().mean()), 5), 'max_abs_diff': round(float(d.abs().max()), 5),
                     'n_identical': int((d.abs() < 1e-9).sum()),
                     'wall_primary_s': round(float(s.wall_time_sec_primary.mean()), 1),
                     'wall_replicate_s': round(float(s.wall_time_sec_replicate.mean()), 1)})
    def pooled(df):
        return {arm: float(np.average(df[df.arm == arm].IC_mean, weights=df[df.arm == arm].n_test_days)) for arm in ('L0', 'L1')}
    pp, pr = pooled(p), pooled(r)
    extra = {'primary_dir': primary_dir, 'replicate_dir': replicate_dir,
             'pooled_delta_L1_L0_primary': round(pp['L1'] - pp['L0'], 5), 'pooled_delta_L1_L0_replicate': round(pr['L1'] - pr['L0'], 5)}
    return pd.DataFrame(rows), extra


# ══════════════════════════════════════════════════════════════
# 4. side-by-side table
# ══════════════════════════════════════════════════════════════

def _row(fam_dir: str, universe: str, seedrob: dict) -> dict:
    dm = pd.read_csv(os.path.join(fam_dir, 'family1_dm_hln.csv'))
    ci = pd.read_csv(os.path.join(fam_dir, 'family1_ic_ci.csv'))
    mde = pd.read_csv(os.path.join(fam_dir, 'family1_mde.csv'))
    r = dm[(dm.universe == universe) & (dm.arm_A == PAIR[0]) & (dm.arm_B == PAIR[1])].iloc[0]
    m = mde[(mde.universe == universe) & (mde.pair == f'{PAIR[0]}-{PAIR[1]}')].iloc[0]
    c0 = ci[(ci.universe == universe) & (ci.arm == PAIR[1])].iloc[0]
    c1 = ci[(ci.universe == universe) & (ci.arm == PAIR[0])].iloc[0]
    return {
        'universe': universe, 'T_days': int(r['T']),
        'mean_delta_IC': round(float(r['mean_delta_IC']), 5),
        'delta_ci_lo': float(m['delta_ci_lo']), 'delta_ci_hi': float(m['delta_ci_hi']),
        'HLN_p_t': round(float(r['HLN_p_t']), 5), 'HLN_p_t_lag21': round(float(r['HLN_p_t_lag21']), 5),
        'BH_reject_family': (None if pd.isna(r.get('BH_FDR_reject_family', np.nan))
                             else bool(r['BH_FDR_reject_family'])),
        'IC_L0': float(c0['IC_mean']), 'IC_L0_ci_lo': float(c0['IC_ci_lo']), 'IC_L0_ci_hi': float(c0['IC_ci_hi']),
        'IC_L1': float(c1['IC_mean']), 'IC_L1_ci_lo': float(c1['IC_ci_lo']), 'IC_L1_ci_hi': float(c1['IC_ci_hi']),
        'SE_block': float(m['SE_block']), 'MDE_2p8xSE': float(m['MDE_2p8xSE']),
        'per_seed_same_sign': f"{seedrob['n_same_sign']}/{seedrob['n_seeds']}",
        'loso_flips': f"{seedrob['loso_sign_flips']}/{seedrob['n_seeds']}",
        'role': 'post-hoc sensitivity (no BH)' if universe == 'C5' else 'confirmatory (BH over 20-test family)',
    }


def hparam_report(frozen_c5: str, frozen_conf: str = 'artifacts/storya_v21_tune/frozen_hparams.json') -> list:
    """Tuned winners for C5 (+ confirmatory C for reference) and the MLP parameter count at the actual
    input width (20 vs 51) — CODEX-A-05: report capacity changes alongside the contrast."""
    import torch
    import run_storya_e1_anchor as anchor
    rows = []
    for tag, path, n_in in [('C5', frozen_c5, 20), ('C', frozen_conf, 51)]:
        if not os.path.exists(path):
            continue
        st = json.load(open(path))['studies']
        for arm in ['L0', 'L1']:
            k = f'{tag}_{arm}'
            if k not in st:
                continue
            wp = st[k]['winner_params']
            n_params = None
            if arm == 'L1':
                orig = dict(anchor.NN_HPARAMS)
                anchor.NN_HPARAMS = {**orig, **wp}
                try:
                    m = anchor.make_nn_model('MLP', n_in, torch.device('cpu'))
                    n_params = int(sum(p.numel() for p in m.parameters()))
                finally:
                    anchor.NN_HPARAMS = orig
            rows.append({'universe': tag, 'arm': arm, 'model': st[k]['model'], 'n_inputs': n_in,
                         'winner_params': wp, 'winner_mean_val_ic_3seed': st[k].get('winner_mean_val_ic_3seed'),
                         'n_trials': st[k].get('n_trials'), 'mlp_n_params': n_params})
    return rows


def write_md(out_dir: str, comp: pd.DataFrame, paired: list, integ: dict, hp_rows: list | None = None,
             ex_rows: list | None = None, dev: tuple | None = None) -> None:
    inp = integ.get('inputs', {})
    L = ['# C5 feature-subset sensitivity (POST-HOC, TEST-INFORMED selection) — L1 (MLP) − L0 (LightGBM)\n',
         f"_C5 = {integ['n_features']} columns = Plan-AAA permutation top-15 ∩ single-feature-IC proxy top-15 "
         f"(proxy top-15 identical with/without the T-1 shift; both selectors scored inside the 12-fold test period, using "
         f"NN-based permutation importance — see docs/c5_rerun_brief_2026-09-10.md §9.9). {integ['n_results_rows']} cells; "
         f"frozen_hparams md5 {integ['frozen_md5']}; integrity PASS={integ['PASS']}. "
         f"Raw (unadjusted, nominal) HLN p; no BH family; not confirmatory._\n",
         f"_INPUT (primary): `{inp.get('c5_main_dir')}` — device {inp.get('device')}, platform {inp.get('platform')}, "
         f"results.csv md5 {inp.get('results_csv_md5')}, git_rev {inp.get('git_rev')}, source_clean {inp.get('source_clean')}, "
         f"post-hoc code identity {inp.get('code_identity_post_hoc')}._\n",
         '| universe | role | ΔIC (L1−L0) | 95% block-boot CI | HLN p | HLN p (lag 21) | IC L0 [CI] | IC L1 [CI] | MDE (≈2.8×SE, approx. nominal) | per-seed same sign | LOSO flips |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in comp.itertuples():
        L.append(f"| {r.universe} | {r.role} | {r.mean_delta_IC:+.4f} | [{r.delta_ci_lo:+.4f}, {r.delta_ci_hi:+.4f}] | "
                 f"{r.HLN_p_t:.3f} | {r.HLN_p_t_lag21:.3f} | {r.IC_L0:.4f} [{r.IC_L0_ci_lo:.4f}, {r.IC_L0_ci_hi:.4f}] | "
                 f"{r.IC_L1:.4f} [{r.IC_L1_ci_lo:.4f}, {r.IC_L1_ci_hi:.4f}] | {r.MDE_2p8xSE:.4f} | "
                 f"{r.per_seed_same_sign} | {r.loso_flips} |")
    L.append('\n(source: family1_{dm_hln,ic_ci,mde}.csv in artifacts/storya_v21_family1_c5 for C5 and '
             'artifacts/storya_v21_family1 for C/B; c5_seed_robustness.csv for k/10, m/10)\n')
    L.append('_Read with the CI and BOTH HAC lags (the frozen NW auto-lag ≈6 truncates while the 21-day-label autocorrelation '
             'is still ≈0.3; lag-21 and the 21d block bootstrap agree). In C5, C and B the observed |ΔIC| is BELOW the '
             "design's approximate MDE (≈2.8×SE): marginal, underpowered detections. Per-arm IC levels are conditional on the "
             'test-informed selection and are not out-of-sample performance figures._\n')
    if ex_rows:
        exf = ex_rows[0]['excluded_fold']
        L.append(f'## Fold concentration — pooled statistics EXCLUDING fold {exf} (the fold flagged by LOFO as dominant)\n')
        L.append('| universe | fold ΔIC (excluded fold) | ΔIC ex-fold | 95% block-boot CI | HLN p | HLN p (lag 21) | MDE (≈2.8×SE) | T |')
        L.append('|---|---|---|---|---|---|---|---|')
        for r in ex_rows:
            L.append(f"| {r['universe']} | {r['excluded_fold_delta_IC']:+.4f} | {r['mean_delta_IC_ex']:+.4f} | "
                     f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['HLN_p_t']:.3f} | {r['HLN_p_t_lag21']:.3f} | {r['MDE_2p8xSE']:.4f} | {r['T_ex']} |")
        L.append('\n(source: c5_ex_fold.csv; the excluded quarter carries a large share of the pooled contrast in C5 exactly as in C and B — '
                 'the C5 contrast inherits their quarter concentration; not evenly persistent)\n')
    if paired:
        L.append('## Paired daily contrast (seed-averaged daily ΔIC, same test days; conditional subset contrast)\n')
        L.append('| contrast | mean paired diff | 95% CI | HLN p | HLN p (lag 21) | SE_block | MDE (≈2.8×SE) | T |')
        L.append('|---|---|---|---|---|---|---|---|')
        for p in paired:
            L.append(f"| {p['contrast']} | {p['mean_paired_diff']:+.4f} | [{p['ci_lo']:+.4f}, {p['ci_hi']:+.4f}] | "
                     f"{p['HLN_p_t']:.3f} | {p['HLN_p_t_lag21']:.3f} | {p['SE_block']:.4f} | {p['MDE_2p8xSE']:.4f} | {p['T']} |")
        L.append('\n_The paired difference is not distinguishable from zero, but its interval is wider than the contrast itself '
                 '(approximate MDE of the paired test > the C contrast): it does NOT exclude a halving or a doubling of the '
                 'contrast under the subset. Equivalence is not established; this is an underpowered non-rejection._')
        L.append('\n(source: c5_paired_contrast.csv; positive = the confirmatory universe\'s L1−L0 exceeds the C5 one. '
                 'Conditional subset contrast — feature restriction + re-tuning; NOT an identified leakage-inflation effect: '
                 'C5\'s columns were selected with evaluation-period outcomes, brief §9.9)\n')
    if hp_rows:
        L.append('## Tuned winners (30 trials, top-5 × 3 tuning seeds) and MLP capacity at the actual input width\n')
        L.append('| universe | arm | model | n_inputs | winner params | val-IC (3-seed, SELECTION metric only) | MLP #params |')
        L.append('|---|---|---|---|---|---|---|')
        for r in hp_rows:
            L.append(f"| {r['universe']} | {r['arm']} | {r['model']} | {r['n_inputs']} | `{r['winner_params']}` | "
                     f"{r['winner_mean_val_ic_3seed']:.4f} | {r['mlp_n_params'] if r['mlp_n_params'] is not None else '—'} |")
        L.append('\n(source: experiments/storya_v21_tune/frozen_hparams_c5.json + artifacts/storya_v21_tune/frozen_hparams.json; '
                 'param count via run_storya_e1_anchor.make_nn_model at n_inputs)\n')
        L.append('_DISCLOSURE (TP2-B B-04 / TP3 R-A-04): on C5 every finalist of BOTH arms had NEGATIVE 2022H2 validation IC '
                 '(the frozen HPs are protocol-consistent but not a validated optimum; the selection carried no positive signal), '
                 'whereas the C winners had val-IC +0.07/+0.06; the C5 MLP is ≈14× smaller than the C MLP. Do not attribute the '
                 'contrast (or its similarity to C) to feature restriction or capacity alone._\n')
    if dev is not None:
        df_dev, extra = dev
        L.append('## Device replication — primary vs replicate result directories (same frozen HPs, same code)\n')
        L.append(df_dev.to_markdown(index=False))
        L.append(f"\n_{extra['primary_dir']} (primary) vs {extra['replicate_dir']} (replicate): pooled ΔIC L1−L0 "
                 f"{extra['pooled_delta_L1_L0_primary']:+.5f} vs {extra['pooled_delta_L1_L0_replicate']:+.5f}. L1 cell-level "
                 f"divergence is backend nondeterminism amplified by early stopping on a flat validation curve; the pooled inference "
                 f"is insensitive to it (source: c5_device_replication.csv)._\n")
    with open(os.path.join(out_dir, 'c5_comparison.md'), 'w') as f:
        f.write('\n'.join(L))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--c5-main-dir', default='experiments/storya_v21_main12_c5_t4',
                   help='PRIMARY result dir (pre-declared 2026-09-11-b: the Colab T4 run); Mac replicate = experiments/storya_v21_main12_c5')
    p.add_argument('--replicate-main-dir', default=None,
                   help='optional replicate result dir → c5_device_replication.{csv,md} (cell-level primary-vs-replicate)')
    p.add_argument('--ex-fold', type=int, default=None,
                   help='report pooled stats with this fold excluded for C5/C/B (fold flagged by LOFO as dominant, e.g. 9 = 2025Q2)')
    p.add_argument('--c5-family-dir', default='artifacts/storya_v21_family1_c5')
    p.add_argument('--conf-main-dir', default='experiments/storya_v21_main12_tuned')
    p.add_argument('--conf-family-dir', default='artifacts/storya_v21_family1')
    p.add_argument('--frozen', default='experiments/storya_v21_tune/frozen_hparams_c5.json')
    p.add_argument('--out-dir', default=None, help='default = --c5-family-dir')
    p.add_argument('--no-paired', action='store_true', help='skip the paired (C5 minus C / B) daily contrast')
    p.add_argument('--smoke', action='store_true', help='n_boot=200; tolerate partial C5 runs')
    p.add_argument('--conf-only', action='store_true',
                   help='cross-check mode: only the confirmatory C/B per-seed numbers (no C5 dirs needed)')
    args = p.parse_args()
    out_dir = args.out_dir or args.c5_family_dir
    os.makedirs(out_dir, exist_ok=True)
    n_boot = 200 if args.smoke else N_BOOT

    conf_results = os.path.join(args.conf_main_dir, 'results.csv')
    rows = [seed_robustness(conf_results, u) for u in ['C', 'B']]
    for r in rows:
        print(f"  [{r['universe']} L1-L0] pooled={r['pooled_delta_IC']:+.5f} same-sign {r['n_same_sign']}/{r['n_seeds']} "
              f"LOSO flips {r['loso_sign_flips']}/{r['n_seeds']}  (confirmatory cross-check)")
    if args.conf_only:
        pd.DataFrame(rows).drop(columns=['per_seed_delta']).to_csv(os.path.join(out_dir, 'c5_seed_robustness.csv'), index=False)
        print('conf-only cross-check done')
        return 0

    integ = run_integrity(args.c5_main_dir, args.frozen, conf_results, strict=not args.smoke)
    with open(os.path.join(out_dir, 'c5_run_integrity.json'), 'w') as f:
        json.dump(integ, f, indent=2)
    print(f"[integrity] rows={integ['n_results_rows']}/{integ['n_expected']} failed={integ['n_failed']} "
          f"unique_ids={integ['cell_id_unique']} npy={integ['per_day_npy_present']} n_features={integ['n_features']} "
          f"short_cells={len(integ['cells_not_full_calendar_length'])} provenance_gate={integ['provenance_gate_ok']} "
          f"→ PASS={integ['PASS']}")
    if not integ['PASS'] and not args.smoke:
        raise SystemExit('C5 run integrity FAILED — see c5_run_integrity.json; not building the comparison')

    r5 = seed_robustness(os.path.join(args.c5_main_dir, 'results.csv'), 'C5')
    print(f"  [C5 L1-L0] pooled={r5['pooled_delta_IC']:+.5f} same-sign {r5['n_same_sign']}/{r5['n_seeds']} "
          f"LOSO flips {r5['loso_sign_flips']}/{r5['n_seeds']}")
    rows = [r5] + rows
    pd.DataFrame(rows).drop(columns=['per_seed_delta']).to_csv(os.path.join(out_dir, 'c5_seed_robustness.csv'), index=False)
    with open(os.path.join(out_dir, 'c5_seed_robustness_per_seed.json'), 'w') as f:
        json.dump({r['universe']: r['per_seed_delta'] for r in rows}, f, indent=2)

    paired = []
    if not args.no_paired:
        for other in ['C', 'B']:
            try:
                pr = paired_contrast(args.c5_main_dir, args.conf_main_dir, other, n_boot, strict=not args.smoke)
                paired.append(pr)
                print(f"  [paired {pr['contrast']}] diff={pr['mean_paired_diff']:+.5f} CI=[{pr['ci_lo']:+.5f}, {pr['ci_hi']:+.5f}] "
                      f"HLN p={pr['HLN_p_t']:.4f} (lag21 {pr['HLN_p_t_lag21']:.4f}) T={pr['T']}")
            except AssertionError as e:
                if not args.smoke:
                    raise
                print(f"  [paired C5 vs {other}] skipped (smoke): {e}")
        if paired:
            pd.DataFrame(paired).to_csv(os.path.join(out_dir, 'c5_paired_contrast.csv'), index=False)

    seedrob = {r['universe']: r for r in rows}
    comp = pd.DataFrame([_row(args.c5_family_dir, 'C5', seedrob['C5']),
                         _row(args.conf_family_dir, 'C', seedrob['C']),
                         _row(args.conf_family_dir, 'B', seedrob['B'])])
    comp.to_csv(os.path.join(out_dir, 'c5_comparison.csv'), index=False)
    hp_rows = hparam_report(args.frozen)
    if hp_rows:
        pd.DataFrame(hp_rows).to_csv(os.path.join(out_dir, 'c5_tuned_hparams.csv'), index=False)
    ex_rows = []
    if args.ex_fold is not None:
        ex_rows = [ex_fold_stats(args.c5_main_dir, 'C5', args.ex_fold, n_boot)] + \
                  [ex_fold_stats(args.conf_main_dir, u, args.ex_fold, n_boot) for u in ('C', 'B')]
        pd.DataFrame(ex_rows).to_csv(os.path.join(out_dir, 'c5_ex_fold.csv'), index=False)
        for r in ex_rows:
            print(f"  [ex-fold {r['excluded_fold']} {r['universe']}] fold ΔIC={r['excluded_fold_delta_IC']:+.4f} → ex-fold ΔIC={r['mean_delta_IC_ex']:+.4f} "
                  f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] p={r['HLN_p_t']:.3f} (lag21 {r['HLN_p_t_lag21']:.3f})")
    dev = None
    if args.replicate_main_dir:
        dev = device_replication(args.c5_main_dir, args.replicate_main_dir)
        dev[0].to_csv(os.path.join(out_dir, 'c5_device_replication.csv'), index=False)
        with open(os.path.join(out_dir, 'c5_device_replication.md'), 'w') as f:
            f.write('# C5 device replication — primary vs replicate (same frozen HPs, same code)\n\n' + dev[0].to_markdown(index=False)
                    + '\n\n' + json.dumps(dev[1]) + '\n')
        print(dev[0].to_string(index=False)); print(dev[1])
    write_md(out_dir, comp, paired, integ, hp_rows, ex_rows, dev)
    print(comp[['universe', 'mean_delta_IC', 'delta_ci_lo', 'delta_ci_hi', 'HLN_p_t', 'IC_L0', 'IC_L1',
                'MDE_2p8xSE', 'per_seed_same_sign', 'loso_flips']].to_string(index=False))
    print(f'[C5] DONE → {out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
