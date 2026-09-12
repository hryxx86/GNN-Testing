#!/usr/bin/env python
"""analyze_c5_sensitivity.py — post-hoc feature-basis sensitivity analyses for the L1 (MLP) − L0 (LightGBM) contrast.

Universes handled (--universe):
  C5    the 20-column TEST-INFORMED subset of Universe C (docs/c5_rerun_brief_2026-09-10.md §9.9)      [default]
  CPRE  the PRE-EVALUATION re-selection (run_storya_cpre_select.py; docs/c_pre_plan_2026-09-11.md)     (2026-09-12)
Neither is confirmatory: nominal (unadjusted) HLN p-values, no BH family, no change to any confirmatory table.

Companion to `compute_family1_ladder.py --universes <U> --arms L0,L1 --sensitivity` (HLN / block-bootstrap CI / MDE
rows). This script adds the seed-level robustness numbers the paper's appendix C.2 reports for the confirmatory
contrasts, checks run integrity, and assembles the side-by-side table <U> vs C vs B (vs C5 for CPRE) for L1-L0.

Estimands (identical to analyze_paper_eval_robustness.py checks 1+2, appendix C.2):
  per-seed pooled IC   = n_test_days-weighted mean of the 12 fold-level IC_mean of that seed
  per-seed ΔIC         = pooled_L1[s] - pooled_L0[s]
  k/10 same sign       = # seeds whose ΔIC sign == sign(mean over seeds)
  m/10 LOSO flips      = # seeds whose removal flips the sign of the mean ΔIC (k = n ⇒ m = 0 by construction)
Paired daily contrast (comparator minus <U>): d(t) = ΔIC_other(t) − ΔIC_U(t) on the seed-averaged daily ΔIC series
(same 749 test days in the same chronological fold order, per-fold calendar asserted on both sides) → HLN p (auto lag
and lag 21) + 21d stationary block-bootstrap CI + SE/MDE. Reported as the ABSOLUTE paired change only — no
proportional ("halved / doubled") inference (Codex TP1-A A-02, 2026-09-11). It is a conditional contrast (feature
restriction/re-selection + re-tuning), NOT an identified leakage-inflation effect.

All result-dependent prose in the markdown is COMPUTED from the run's own numbers (Codex TP1-A: no literal
conclusions may propagate between universes).

Outputs (→ --out-dir, default = the universe's family dir), prefixed c5_ / cpre_:
  <p>_run_integrity.json, <p>_seed_robustness.csv (+_per_seed.json), <p>_paired_contrast.csv, <p>_comparison.{csv,md},
  <p>_tuned_hparams.csv, <p>_ex_fold.csv (--ex-fold), <p>_device_replication.{csv,md} (--replicate-main-dir),
  <p>_tests_reported.json
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

# per-universe specification (paths are defaults; every one is overridable on the CLI)
SPECS = {
    'C5': {
        'prefix': 'c5', 'names_attr': 'UNIVERSE_C5_NAMES', 'universe_json': '_universe_c5.json', 'block_min': 2400,
        'main_dir': 'experiments/storya_v21_main12_c5_t4',          # PRIMARY (pre-declared 2026-09-11-b: the Colab T4 run)
        'family_dir': 'artifacts/storya_v21_family1_c5',
        'frozen': 'experiments/storya_v21_tune/frozen_hparams_c5.json',
        'comparators': ['C', 'B'],
        'role': 'post-hoc sensitivity (no BH)',
        'title': 'C5 feature-subset sensitivity (POST-HOC, TEST-INFORMED selection) — L1 (MLP) − L0 (LightGBM)',
        'selection_desc': ('Plan-AAA permutation top-15 ∩ single-feature-IC proxy top-15 (proxy top-15 identical with/without the '
                           'T-1 shift; both selectors scored inside the 12-fold test period, using NN-based permutation importance '
                           '— see docs/c5_rerun_brief_2026-09-10.md §9.9)'),
        'levels_note': 'conditional on the test-informed selection and are not out-of-sample performance figures',
        'paired_note': 'not an identified leakage-inflation effect (C5 selection is test-informed)',
        'selection_kind': 'test-informed',
    },
    'CPRE': {
        'prefix': 'cpre', 'names_attr': 'UNIVERSE_CPRE_NAMES', 'universe_json': '_universe_cpre.json', 'block_min': 3600,
        'main_dir': 'experiments/storya_v21_main12_cpre',
        'family_dir': 'artifacts/storya_v21_family1_cpre',
        'frozen': 'experiments/storya_v21_tune/frozen_hparams_cpre.json',
        'comparators': ['C', 'B', 'C5'],
        'role': 'post-hoc sensitivity, pre-evaluation selection (no BH)',
        'title': 'C-pre feature re-selection sensitivity (POST-HOC, PRE-EVALUATION selection) — L1 (MLP) − L0 (LightGBM)',
        'selection_desc': ('union of all members of the top-15 Plan-AAA groups ranked by single-feature |IC| on the tuning-train window '
                           '2021-07-01..2022-05-31 (label end ≤ 2022-06-30; τ = 0.50 coverage; docs/c_pre_plan_2026-09-11.md §3) — '
                           'scoring and grouping inputs are bounded by 2022-06-30, the protocol was chosen retrospectively'),
        'levels_note': ('conditional on the pre-evaluation selection rule and on a retrospectively chosen protocol; nominal, '
                        'not confirmatory out-of-sample performance figures'),
        'paired_note': 'not an identified leakage-inflation effect (column sets differ and both arms were re-tuned)',
        'selection_kind': 'pre-evaluation',
    },
}


# ══════════════════════════════════════════════════════════════
# 1. run integrity (240 cells, no failures, no duplicate ids, per_day complete, feature list, frozen-HP gate)
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


def run_integrity(universe: str, main_dir: str, frozen_path: str | None, conf_results_csv: str, strict: bool = True) -> dict:
    """CODEX-TP2-A-02/A-03: every cell must carry the FULL per-fold day count of the frozen calendar (a shortened
    .npy = undefined-IC days dropped → positional packing would silently misalign dates); outside smoke mode the
    frozen file, the provenance file (mode TUNED per-arm, md5 match, applied <U>_L0/<U>_L1 == frozen winners) and
    240 completed cells are all REQUIRED for PASS."""
    spec = SPECS[universe]
    res = pd.read_csv(os.path.join(main_dir, 'results.csv'))
    man = pd.read_csv(os.path.join(main_dir, 'manifest.csv'))
    uni = json.load(open(os.path.join(main_dir, spec['universe_json'])))
    prov_p = os.path.join(main_dir, '_frozen_hp_provenance.json')
    prov = json.load(open(prov_p)) if os.path.exists(prov_p) else None
    frozen = json.load(open(frozen_path)) if (frozen_path and os.path.exists(frozen_path)) else None
    n_expected = 2 * len(CANONICAL_SEEDS) * N_FOLDS
    per_day = os.path.join(main_dir, 'per_day_ic')
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
    run_prov_p = os.path.join(main_dir, '_run_provenance.json')
    prov_entries = json.load(open(run_prov_p)) if os.path.exists(run_prov_p) else None
    if isinstance(prov_entries, dict):
        prov_entries = [prov_entries]
    prov_entries = prov_entries or []
    # EXPL-CODE-07: entries are appended per invocation (resume can span machines; a CORRECTION entry may be
    # appended later with a local-time stamp) → prefer an explicit correction entry, else the last one, and
    # surface the invocation count + the set of devices so a multi-device run cannot look single-device.
    corr = [e for e in prov_entries if 'CORRECTION' in str(e.get('entry_type', ''))]
    run_prov = (corr[-1] if corr else (prov_entries[-1] if prov_entries else None))
    devices = sorted({str(e.get('device')) for e in prov_entries if e.get('device')})
    code_id_p = os.path.join(main_dir, '_code_identity_t4.json')
    code_id = json.load(open(code_id_p)) if os.path.exists(code_id_p) else None
    cid_match = cid_commit = None
    if code_id is not None:   # EXPL-CODE-08: generic keys, with fallback to the commit-suffixed legacy keys
        cid_commit = code_id.get('commit') or next((k.split('_')[-1] for k in code_id if k.startswith('all_modules_match_')), None)
        cid_match = code_id.get('all_modules_match', code_id.get(f'all_modules_match_{cid_commit}'))
    inputs = {
        'universe': universe, 'main_dir': main_dir, 'conf_results_csv': conf_results_csv,
        'results_csv_md5': hashlib.md5(open(os.path.join(main_dir, 'results.csv'), 'rb').read()).hexdigest(),
        'manifest_csv_md5': hashlib.md5(open(os.path.join(main_dir, 'manifest.csv'), 'rb').read()).hexdigest(),
        'device': run_prov.get('device') if run_prov else None,
        'platform': run_prov.get('platform') if run_prov else None,
        'git_rev': run_prov.get('git_rev') if run_prov else None,
        'source_clean': run_prov.get('source_clean') if run_prov else None,
        'code_identity_post_hoc': ({'all_modules_match': cid_match, 'commit': cid_commit,
                                    'verified_at': code_id.get('verified_at')} if code_id else None),
        'n_invocations': len(prov_entries), 'devices_seen': devices,
        'multi_device_warning': (len(devices) > 1),
        'selector_inputs_md5': run_prov.get('selector_inputs_md5') if run_prov else None,
    }
    try:
        import run_storya_e1_anchor as _anchor          # EXPL-CODE-09: the names must be THE frozen definition
        expected_names = list(getattr(_anchor, spec['names_attr']) or [])
    except Exception:
        expected_names = None
    names_ok = (uni.get('feature_names') == expected_names) if expected_names else None
    n_feat_expected = len(expected_names) if expected_names else uni['n_features']
    prov_ok = None
    keys = {f'{universe}_L0', f'{universe}_L1'}
    if prov is not None and frozen is not None:
        applied = prov.get('applied', {})
        prov_ok = bool(
            prov.get('mode') == 'TUNED per-arm' and prov.get('frozen_md5') == md5_frozen
            and set(applied) == keys
            and all(applied[k]['src'] == k and applied[k]['params'] == frozen['studies'][k]['winner_params']
                    for k in sorted(keys))
            and frozen.get('complete') is True and frozen.get('n_studies') == frozen.get('expected') == 2
        )
    block_lo, block_hi = spec['block_min'], spec['block_min'] + 1199
    out = {
        'universe': universe,
        'n_results_rows': int(len(res)), 'n_expected': n_expected,
        'universes': sorted(res.universe.unique().tolist()), 'arms': sorted(res.arm.unique().tolist()),
        'cell_id_unique': bool(res.cell_id.is_unique),
        'cell_id_range': [int(res.cell_id.min()), int(res.cell_id.max())] if len(res) else None,
        'cell_id_block_expected': [block_lo, block_hi],
        'cell_id_inside_block': bool(len(res) and res.cell_id.min() >= block_lo and res.cell_id.max() <= block_hi),
        'cell_id_disjoint_from_confirmatory_0_2399': bool(len(res) and res.cell_id.min() >= 2400),
        'manifest_status_counts': man.status.value_counts().to_dict(),
        'n_failed': int((man.status != 'completed').sum()),
        'converged_all': bool((res.converged_flag == 1).all()) if len(res) else False,
        'n_test_days_per_arm_per_seed_min_max': ({arm: [int(g.min()), int(g.max())] for arm, g in
                                                  res.groupby(['arm', 'seed'])['n_test_days'].sum().groupby(level=0)}
                                                 if len(res) else {}),   # EXPL-CODE-13 (each should be [749, 749])
        'per_day_npy_present': npy_ok, 'per_day_npy_len_mismatch': npy_len_mismatch,
        'frozen_calendar_days_per_fold': cal, 'frozen_calendar_total': int(sum(cal.values())),
        'cells_not_full_calendar_length': short_cells,          # any entry = date alignment NOT guaranteed
        'n_features': uni['n_features'], 'n_features_expected': n_feat_expected, 'feature_names': uni['feature_names'],
        f"feature_names_match_{spec['names_attr']}": names_ok,
        'inputs': inputs,
        'frozen_hparams_path': frozen_path, 'frozen_present': frozen is not None, 'frozen_md5': md5_frozen,
        'provenance_present': prov is not None,
        'provenance_mode': prov.get('mode') if prov else None,
        'provenance_md5': prov.get('frozen_md5') if prov else None,
        'provenance_gate_ok': prov_ok,   # mode TUNED per-arm + md5 match + applied <U>_L0/<U>_L1 == frozen winners + 2/2
        'applied_hparams': prov.get('applied') if prov else None,
        'strict': strict,
    }
    out['PASS'] = bool(
        out['n_results_rows'] == n_expected and out['cell_id_unique'] and out['n_failed'] == 0
        and out['converged_all'] and npy_ok == n_expected and not npy_len_mismatch and not short_cells
        and out['n_features'] == n_feat_expected and out['cell_id_inside_block']
        and out['cell_id_disjoint_from_confirmatory_0_2399']
        and (names_ok is True if strict else names_ok in (True, None))
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
# 3. paired daily contrast: (L1-L0)_other minus (L1-L0)_U on seed-averaged daily ΔIC (positive = other larger)
# ══════════════════════════════════════════════════════════════

def _delta_per_fold(per_day_dir: str, universe: str, calendar: dict | None) -> dict:
    """{fold: seed-averaged daily ΔIC (L1−L0)}; with `calendar` given, EVERY fold of BOTH arms must carry the
    full frozen-calendar day count (EXPL-LEAK-01 closeout: per-fold alignment asserted on both sides of a
    pairing, never inferred from a pooled length)."""
    a = seed_avg_per_fold(collect_arm_matrix(per_day_dir, universe, PAIR[0]))
    b = seed_avg_per_fold(collect_arm_matrix(per_day_dir, universe, PAIR[1]))
    out = {}
    for f in range(N_FOLDS):
        if a.get(f) is None or b.get(f) is None:
            if calendar is not None:
                raise AssertionError(f'{universe}: fold {f} missing for L1 or L0')
            continue
        if calendar is not None:
            assert len(a[f]) == len(b[f]) == calendar[f], \
                f'{universe} fold {f}: L1 {len(a[f])} / L0 {len(b[f])} days != frozen calendar {calendar[f]}'
        n = min(len(a[f]), len(b[f]))
        out[f] = a[f][:n] - b[f][:n]
    return out


def _delta_series(per_day_dir: str, universe: str, calendar: dict | None = None) -> np.ndarray:
    per_fold = _delta_per_fold(per_day_dir, universe, calendar)
    return np.concatenate([per_fold[f] for f in sorted(per_fold)]) if per_fold else np.array([])


def paired_contrast(universe: str, main_dir: str, other: str, other_main: str, n_boot: int, strict: bool = True,
                    calendar: dict | None = None) -> dict:
    cal = calendar if strict else None    # strict: per-fold day counts asserted for BOTH universes
    du = _delta_series(os.path.join(main_dir, 'per_day_ic'), universe, cal)
    dc = _delta_series(os.path.join(other_main, 'per_day_ic'), other, cal)
    if strict:   # full run: both series must be the same 749 seed-averaged test days (fold order 0..11)
        assert len(du) == len(dc), f'paired series length mismatch: {universe} {len(du)} vs {other} {len(dc)}'
    n = min(len(du), len(dc))
    assert n >= 2 * BLOCK_SIZE, f'paired series too short: {n}'
    d = dc[:n] - du[:n]                                   # positive = comparator contrast exceeds the target's
    hln_stat, hln_p, T = hln_test(d)
    _, hln_p21, _ = hln_test(d, lag=HORIZON)
    mean_d, lo, hi = stationary_bootstrap_ci(d, lambda x: float(np.mean(x)), n_boot=n_boot,
                                             block_size=BLOCK_SIZE)
    from arch.bootstrap import StationaryBootstrap          # same SE/MDE construction as family1 run_ci_and_mde
    sb = StationaryBootstrap(BLOCK_SIZE, d, seed=86)
    se_block = float(np.std(sb.apply(lambda x: float(np.mean(x)), n_boot).ravel(), ddof=1))
    spec = SPECS[universe]
    row = {'contrast': f'(L1-L0)_{other} - (L1-L0)_{universe}', 'other_universe': other, 'T': int(T),
           f'mean_delta_{spec["prefix"]}': round(float(du[:n].mean()), 5), 'mean_delta_other': round(float(dc[:n].mean()), 5),
           'mean_paired_diff': round(float(mean_d), 5), 'ci_lo': round(float(lo), 5), 'ci_hi': round(float(hi), 5),
           'ci_excludes_0': bool(lo > 0 or hi < 0),
           'SE_block': round(se_block, 5), 'MDE_2p8xSE': round(MDE_FACTOR * se_block, 5),
           'HLN_stat_on_IC_diff': round(float(hln_stat), 4),   # IC-difference convention (family1 uses loss diff = −ΔIC)
           'HLN_p_t': round(float(hln_p), 5),
           'HLN_p_t_lag21': round(float(hln_p21), 5),
           'n_boot': n_boot, 'block_size': BLOCK_SIZE,
           'note': (f'post-hoc conditional contrast; same test days + same 10 seeds; positive = {other} contrast larger than '
                    f'{universe}; absolute paired change only (no proportional inference); {spec["paired_note"]}')}
    return row


# ══════════════════════════════════════════════════════════════
# 3b. leave-one-fold-out pooled statistics for the dominant fold (FINGNN-R-A-01: 2025Q2 concentration)
# ══════════════════════════════════════════════════════════════

def ex_fold_stats(main_dir: str, universe: str, ex_fold: int, n_boot: int, calendar: dict | None = None) -> dict:
    """Pooled seed-averaged daily ΔIC statistics with one fold EXCLUDED (HLN both lags + 21d block CI + SE/MDE),
    plus the excluded fold's share of the pooled contrast and its rank among the per-fold contributions."""
    if ex_fold not in range(N_FOLDS):
        raise ValueError(f'--ex-fold {ex_fold} outside 0..{N_FOLDS - 1}')
    pd_dir = os.path.join(main_dir, 'per_day_ic')
    a = seed_avg_per_fold(collect_arm_matrix(pd_dir, universe, PAIR[0]))
    b = seed_avg_per_fold(collect_arm_matrix(pd_dir, universe, PAIR[1]))
    if a.get(ex_fold) is None or b.get(ex_fold) is None:
        raise ValueError(f'{universe}: fold {ex_fold} has no per-day data for L1/L0 — cannot exclude it')
    parts, fold_delta, contrib, ex = [], None, {}, np.array([])
    for f in range(N_FOLDS):
        if a.get(f) is None or b.get(f) is None:
            continue
        if calendar is not None:      # EXPL-CODE-05: same per-fold calendar assert as _delta_per_fold, both arms
            assert len(a[f]) == len(b[f]) == calendar[f], \
                f'{universe} fold {f}: L1 {len(a[f])} / L0 {len(b[f])} days != frozen calendar {calendar[f]}'
        n = min(len(a[f]), len(b[f]))
        dd = a[f][:n] - b[f][:n]
        contrib[f] = float(dd.sum())                     # n_f × fold ΔIC = contribution to the pooled sum
        if f == ex_fold:
            fold_delta = float(dd.mean())
            ex = dd                                       # EXPL-CODE-05: reuse the truncated series, never re-subtract
            continue
        parts.append(dd)
    if not parts:
        raise ValueError(f'{universe}: no fold left after excluding fold {ex_fold} (only {sorted(contrib)} present)')
    d = np.concatenate(parts)
    stat, p, T = hln_test(d)
    _, p21, _ = hln_test(d, lag=HORIZON)
    m, lo, hi = stationary_bootstrap_ci(d, lambda x: float(np.mean(x)), n_boot=n_boot, block_size=BLOCK_SIZE)
    from arch.bootstrap import StationaryBootstrap
    se = float(np.std(StationaryBootstrap(BLOCK_SIZE, d, seed=86).apply(lambda x: float(np.mean(x)), n_boot).ravel(), ddof=1))
    n_ex = len(ex)
    pooled_all = float(np.mean(np.concatenate(parts + ([ex] if n_ex else []))))
    share = (n_ex * fold_delta) / ((T + n_ex) * pooled_all) if (fold_delta is not None and pooled_all) else None
    order = sorted(contrib, key=lambda f: contrib[f], reverse=True)     # largest positive contribution first
    # EXPL-CODE-01 (MAJOR, closeout 2026-09-12): the share RATIO is only meaningful when its denominator (the pooled
    # contrast) is itself separated from 0 — otherwise it explodes and can even flip sign. Decide it HERE, emit the
    # flag in the CSV, and let every consumer (stdout, markdown) read the flag instead of recomputing it.
    from arch.bootstrap import StationaryBootstrap as _SB
    se_pooled = float(np.std(_SB(BLOCK_SIZE, np.concatenate(parts + ([ex] if n_ex else [])), seed=86)
                             .apply(lambda x: float(np.mean(x)), n_boot).ravel(), ddof=1))
    share_ok = bool(share is not None and abs(pooled_all) >= se_pooled)
    return {'universe': universe, 'excluded_fold': ex_fold, 'excluded_fold_delta_IC': round(fold_delta, 5) if fold_delta is not None else None,
            'pooled_delta_IC_all_folds': round(pooled_all, 5), 'pooled_SE_block_all_folds': round(se_pooled, 5),
            'excluded_fold_share': (round(float(share), 4) if share_ok else None),
            'excluded_fold_share_raw': round(float(share), 4) if share is not None else None,
            'excluded_fold_share_is_meaningful': share_ok,
            'excluded_fold_contribution_rank': int(order.index(ex_fold) + 1), 'largest_contribution_fold': int(order[0]),
            'n_folds_present': int(len(contrib)),
            'T_ex': int(T), 'mean_delta_IC_ex': round(float(m), 5), 'ci_lo': round(float(lo), 5), 'ci_hi': round(float(hi), 5),
            'HLN_stat_on_IC_diff': round(float(stat), 4), 'HLN_p_t': round(float(p), 5), 'HLN_p_t_lag21': round(float(p21), 5),
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
        'role': SPECS[universe]['role'] if universe in SPECS else 'confirmatory (BH over 20-test family)',
    }


def hparam_report(universe: str, frozen_target: str, n_in_target: int,
                  frozen_conf: str = 'artifacts/storya_v21_tune/frozen_hparams.json',
                  extra: list | None = None, sources: list | None = None) -> list:
    """Tuned winners for the target universe (+ confirmatory C and any extra comparator for reference) and the MLP
    parameter count at the actual input width — CODEX-A-05: report capacity changes alongside the contrast."""
    import torch
    import run_storya_e1_anchor as anchor
    rows = []   # C width from the anchor's own name lists (EXPL-CODE-04: no hardcoded 51/20)
    specs = [(universe, frozen_target, n_in_target), ('C', frozen_conf, len(anchor.UNIVERSE_C_ALPHA158_NAMES) + len(anchor.UNIVERSE_C_EXTRA_NAMES))] + list(extra or [])
    for tag, path, n_in in specs:
        if not os.path.exists(path):
            continue
        if sources is not None and path not in sources:
            sources.append(path)      # EXPL-CODE-04: the md source line names every frozen file actually read
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


def reading_notes(universe: str, comp: pd.DataFrame, seedrob: dict) -> list:
    """Every sentence below is derived from the comparison table / seed table of THIS run (no literal conclusions)."""
    spec = SPECS[universe]
    t = comp[comp.universe == universe].iloc[0]
    notes = []
    sig_auto = [r.universe for r in comp.itertuples() if r.HLN_p_t < 0.05]
    sig_21 = [r.universe for r in comp.itertuples() if r.HLN_p_t_lag21 < 0.05]
    notes.append('(i) The headline HLN p uses the Newey-West AUTO lag — an implementation default, NOT a protocol-specified '
                 'choice; the label overlaps 21 days, so the horizon-matched lag-21 p is reported alongside. Nominal p < 0.05 '
                 f"at the auto lag: {sig_auto or 'none'}; at lag 21: {sig_21 or 'none'}.")
    parts_ii = []   # EXPL-STAT-04 (2026-09-12 closeout): the boundary-case check runs on EVERY row, target first
    for rr in [t] + [r for r in comp.itertuples() if r.universe != universe]:
        half = 1.96 * float(rr.SE_block); excl = bool(rr.delta_ci_lo > 0 or rr.delta_ci_hi < 0); a = abs(float(rr.mean_delta_IC))
        if excl and a < half:
            parts_ii.append(f'{rr.universe}: CI excludes 0 although 1.96×SE_block ({half:.4f}) exceeds |ΔIC| ({a:.4f}) — a boundary '
                            'case (percentile asymmetry), not a robust rejection')
        elif excl:
            parts_ii.append(f'{rr.universe}: CI excludes 0 and |ΔIC| ({a:.4f}) exceeds 1.96×SE_block ({half:.4f})')
        else:
            parts_ii.append(f'{rr.universe}: CI includes 0 (|ΔIC| {a:.4f} vs 1.96×SE_block {half:.4f})')
    notes.append('(ii) Percentile-CI boundary check — ' + '; '.join(parts_ii) + '.')
    verdicts = []
    for r in comp.itertuples():
        e = bool(r.delta_ci_lo > 0 or r.delta_ci_hi < 0)
        bh = '' if r.BH_reject_family is None or (isinstance(r.BH_reject_family, float) and np.isnan(r.BH_reject_family)) \
            else f", BH {'reject' if r.BH_reject_family else 'no-reject'}"
        verdicts.append(f"{r.universe}: CI {'excludes' if e else 'includes'} 0, auto-lag p {r.HLN_p_t:.3f}, lag-21 p {r.HLN_p_t_lag21:.3f}{bh}")
    below = [r.universe for r in comp.itertuples() if abs(r.mean_delta_IC) < r.MDE_2p8xSE]
    above = [r.universe for r in comp.itertuples() if abs(r.mean_delta_IC) >= r.MDE_2p8xSE]
    notes.append('(iii) 5% verdicts per universe — ' + '; '.join(verdicts) + f". |ΔIC| below its own ≈2.8×SE MDE: {below or 'none'}"
                 + (f"; at or above: {above}" if above else '') + '.')
    smallest = comp.loc[comp.HLN_p_t.idxmin()]; smallest_eff = comp.loc[comp.mean_delta_IC.abs().idxmin()]
    if smallest.universe == smallest_eff.universe and len(comp) > 1:
        notes.append(f"(iv) The smallest nominal p ({smallest.universe}, {smallest.HLN_p_t:.3f}) accompanies the smallest |ΔIC| — the "
                     f"ordering is variance-driven (SE_block {', '.join(f'{r.universe} {r.SE_block:.4f}' for r in comp.itertuples())}), "
                     'not a larger effect.')
    else:
        notes.append(f"(iv) Smallest nominal p: {smallest.universe} ({smallest.HLN_p_t:.3f}); smallest |ΔIC|: {smallest_eff.universe} "
                     f"(SE_block {', '.join(f'{r.universe} {r.SE_block:.4f}' for r in comp.itertuples())}).")
    sr = seedrob[universe]
    notes.append(f"(v) k/10 = {sr['n_same_sign']}/{sr['n_seeds']} is a seed/initialisation stability check on the SAME data (not "
                 f"independent replication)" + ('; m = 0 LOSO flips is implied by k = n.' if sr['n_same_sign'] == sr['n_seeds']
                                                else f"; LOSO flips = {sr['loso_sign_flips']}."))
    notes.append(f"(vi) The CI is for the seed-averaged ensemble (day-to-day variance only; per-seed ΔIC in {universe} spans "
                 f"{sr['per_seed_min']:+.4f}…{sr['per_seed_max']:+.4f}). Per-arm IC levels are {spec['levels_note']}.")
    return notes


def write_md(universe: str, out_dir: str, comp: pd.DataFrame, paired: list, integ: dict, seedrob: dict,
             hp_rows: list | None = None, ex_rows: list | None = None, dev: tuple | None = None,
             fam_dirs: dict | None = None, frozen_target: str | None = None, hp_sources: list | None = None) -> None:
    spec = SPECS[universe]; pfx = spec['prefix']
    inp = integ.get('inputs', {})
    L = [f"# {spec['title']}\n",
         f"_{universe} = {integ['n_features']} columns = {spec['selection_desc']}. {integ['n_results_rows']} cells; "
         f"frozen_hparams md5 {integ['frozen_md5']}; integrity PASS={integ['PASS']}. "
         f"Raw (unadjusted, nominal) HLN p; no BH family; not confirmatory._\n",
         f"_INPUT (primary): `{inp.get('main_dir')}` — device {inp.get('device')} (devices seen {inp.get('devices_seen')}, "
         f"{inp.get('n_invocations')} invocation(s)), platform {inp.get('platform')}, results.csv md5 {inp.get('results_csv_md5')}, "
         f"git_rev {inp.get('git_rev')}, source_clean {inp.get('source_clean')}, post-hoc code identity {inp.get('code_identity_post_hoc')}; "
         f"family-dir stats from the same results.csv: {integ.get('family_dir_matches_main_dir')}; "
         f"analysis mode {integ.get('analysis_mode')}._\n",
         '| universe | role | ΔIC (L1−L0, seed-averaged daily) | 95% block-boot CI (on the 10-seed average) | HLN p (NW auto lag) | HLN p (lag 21 = horizon) | IC L0 [CI] | IC L1 [CI] | MDE (≈2.8×SE, approx. nominal) | per-seed same sign | LOSO flips |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in comp.itertuples():
        L.append(f"| {r.universe} | {r.role} | {r.mean_delta_IC:+.4f} | [{r.delta_ci_lo:+.4f}, {r.delta_ci_hi:+.4f}] | "
                 f"{r.HLN_p_t:.3f} | {r.HLN_p_t_lag21:.3f} | {r.IC_L0:.4f} [{r.IC_L0_ci_lo:.4f}, {r.IC_L0_ci_hi:.4f}] | "
                 f"{r.IC_L1:.4f} [{r.IC_L1_ci_lo:.4f}, {r.IC_L1_ci_hi:.4f}] | {r.MDE_2p8xSE:.4f} | "
                 f"{r.per_seed_same_sign} | {r.loso_flips} |")
    src = ', '.join(f'{d} for {u}' for u, d in (fam_dirs or {}).items())
    L.append(f'\n(source: family1_{{dm_hln,ic_ci,mde}}.csv in {src}; {pfx}_seed_robustness.csv for k/10, m/10)\n')
    L.append('_Reading notes (computed from this run; closeout EXPL-STAT-01/02/03/05/06/10 checks). '
             + ' '.join(reading_notes(universe, comp, seedrob)) + '_\n')
    if ex_rows:
        exf = ex_rows[0]['excluded_fold']
        largest = [r['universe'] for r in ex_rows if r['excluded_fold_contribution_rank'] == 1]
        others = [f"{r['universe']} (rank {r['excluded_fold_contribution_rank']}; largest = fold {r['largest_contribution_fold']})"
                  for r in ex_rows if r['excluded_fold_contribution_rank'] != 1]
        L.append(f'## Fold concentration — pooled statistics EXCLUDING fold {exf} '
                 f"(largest single-fold contribution in {largest or 'none'}{'; ' + ', '.join(others) if others else ''})\n")
        L.append('| universe | fold ΔIC (excluded fold) | share of pooled ΔIC | ΔIC ex-fold | 95% block-boot CI | HLN p | HLN p (lag 21) | MDE (≈2.8×SE) | T |')
        L.append('|---|---|---|---|---|---|---|---|---|')
        # EXPL-CODE-01: the meaningfulness flag is computed in ex_fold_stats and published in the CSV; read it here
        def _share(r):
            rank = f"rank {r['excluded_fold_contribution_rank']}/{r.get('n_folds_present', N_FOLDS)}"
            if not r.get('excluded_fold_share_is_meaningful'):
                return f"n/a (pooled ΔIC {r.get('pooled_delta_IC_all_folds'):+.4f} within one SE of 0; contribution {rank})"
            return f"{r['excluded_fold_share']:.0%} ({rank})"
        for r in ex_rows:
            L.append(f"| {r['universe']} | {r['excluded_fold_delta_IC']:+.4f} | {_share(r)} | {r['mean_delta_IC_ex']:+.4f} | "
                     f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['HLN_p_t']:.3f} | {r['HLN_p_t_lag21']:.3f} | {r['MDE_2p8xSE']:.4f} | {r['T_ex']} |")
        shares = ', '.join(f"{r['universe']} {_share(r)}" for r in ex_rows)
        L.append(f'\n(source: {pfx}_ex_fold.csv; share = n_days(fold) × fold ΔIC / (T × pooled ΔIC): {shares}. '
                 'A share near or above one half means the pooled contrast is not evenly persistent across quarters; the ratio is '
                 'not meaningful when the pooled contrast is within one SE of zero, and its denominator carries the same uncertainty '
                 'as the headline. The ex-fold series joins the retained observations across the removed quarter, so the HAC window '
                 'and the 21-day blocks straddle one artificial seam — the ex-fold row is a diagnostic; the full-period row is primary)\n')
    if paired:
        L.append('## Paired daily contrast (seed-averaged daily ΔIC, same test days; conditional contrast — absolute change only)\n')
        L.append('| contrast | mean paired diff | 95% CI | HLN p | HLN p (lag 21) | SE_block | MDE (≈2.8×SE) | T |')
        L.append('|---|---|---|---|---|---|---|---|')
        for p in paired:
            L.append(f"| {p['contrast']} | {p['mean_paired_diff']:+.4f} | [{p['ci_lo']:+.4f}, {p['ci_hi']:+.4f}] | "
                     f"{p['HLN_p_t']:.3f} | {p['HLN_p_t_lag21']:.3f} | {p['SE_block']:.4f} | {p['MDE_2p8xSE']:.4f} | {p['T']} |")
        for p in paired:
            o = comp[comp.universe == p['other_universe']]
            o_eff = abs(float(o.iloc[0].mean_delta_IC)) if len(o) else float('nan')
            sig = 'excludes 0' if p['ci_excludes_0'] else 'includes 0'
            L.append(f"\n_{p['contrast']}: CI {sig}; paired SE {p['SE_block']:.4f} → paired MDE {p['MDE_2p8xSE']:.4f} "
                     f"{'>' if p['MDE_2p8xSE'] > o_eff else '<='} |{p['other_universe']} contrast| {o_eff:.4f}. "
                     + ('The comparator contrast exceeds the target contrast in the paired sense; consistent with, but not an '
                        'identification of, selection inflation.' if (p['ci_excludes_0'] and p['mean_paired_diff'] > 0) else
                        'The comparator contrast is smaller than the target contrast in the paired sense.' if (p['ci_excludes_0']) else
                        'Read the absolute point estimate with its interval; equivalence is not tested and a non-rejection is not '
                        'equivalence' + (' (paired MDE exceeds the comparator contrast: underpowered comparison).' if p['MDE_2p8xSE'] > o_eff else '.'))
                     + '_')
        L.append(f"\n(source: {pfx}_paired_contrast.csv; positive = the comparator's L1−L0 exceeds the {universe} one. "
                 f"Conditional contrast — {spec['paired_note']}; no proportional (halved/doubled) inference)\n")
    below = {r.universe: abs(r.mean_delta_IC) < r.MDE_2p8xSE for r in comp.itertuples()}
    L.append('_Data-derived checks (EXPL-CODE-04): |ΔIC| vs its own MDE — '
             + ', '.join(f"{u}: {'BELOW' if b else 'ABOVE'}" for u, b in below.items()) + '._\n')
    if hp_rows:
        L.append('## Tuned winners (30 trials, top-5 × 3 tuning seeds) and MLP capacity at the actual input width\n')
        L.append('| universe | arm | model | n_inputs | winner params | val-IC (3-seed, SELECTION metric only) | MLP #params |')
        L.append('|---|---|---|---|---|---|---|')
        for r in hp_rows:
            L.append(f"| {r['universe']} | {r['arm']} | {r['model']} | {r['n_inputs']} | `{r['winner_params']}` | "
                     f"{r['winner_mean_val_ic_3seed']:.4f} | {r['mlp_n_params'] if r['mlp_n_params'] is not None else '—'} |")
        L.append(f"\n(source: {', '.join(hp_sources or [frozen_target])}; "
                 'param count via run_storya_e1_anchor.make_nn_model at n_inputs)\n')
        # EXPL-CODE-04: derive the disclosure from the actual finalist tables / winners, not literal text
        fin, degenerate = {}, {}
        for arm in ('L0', 'L1'):
            fp_ = f'experiments/storya_v21_tune/{universe}_{arm}.json'
            if os.path.exists(fp_):
                tt = json.load(open(fp_)).get('top_table', [])
                vals = [x['mean_val_ic_3seed'] for x in tt]
                fin[arm] = (len(vals), sum(v < 0 for v in vals), min(vals) if vals else None, max(vals) if vals else None)
                # EXPL-STAT-05: a deterministic arm gives identical val-IC across the 3 tuning seeds → the "3-seed average"
                # carries no initialisation information for that arm
                seeds_ = [x.get('tune_seed_ics') for x in tt if x.get('tune_seed_ics')]
                degenerate[arm] = bool(seeds_) and all(len({round(float(v), 8) for v in sx}) == 1 for sx in seeds_)
        c_val = {r['arm']: r['winner_mean_val_ic_3seed'] for r in hp_rows if r['universe'] == 'C'}
        np_t = next((r['mlp_n_params'] for r in hp_rows if r['universe'] == universe and r['arm'] == 'L1'), None)
        npc = next((r['mlp_n_params'] for r in hp_rows if r['universe'] == 'C' and r['arm'] == 'L1'), None)
        parts_ = [f"{arm}: {neg}/{n} finalists with negative 2022H2 val-IC (range {lo:+.4f}…{hi:+.4f})"
                  + (' — the 3 tuning seeds give identical val-IC for this arm (deterministic), so the 3-seed average carries no '
                     'initialisation information here' if degenerate.get(arm) else '')
                  for arm, (n, neg, lo, hi) in fin.items()]
        cap = (f"; {universe} MLP {np_t:,} params vs C MLP {npc:,} (ratio {npc / np_t:.1f}×)" if (np_t and npc) else '')
        cref = ', '.join(f"C {a} winner val-IC {v:+.4f}" for a, v in c_val.items())
        any_neg = any(neg > 0 for (_, neg, _, _) in fin.values())
        L.append('_DISCLOSURE (TP2-B B-04 / TP3 R-A-04; values computed from the tune JSONs): ' + '; '.join(parts_)
                 + (f" — vs {cref}" if cref else '') + cap
                 + ('. The frozen HPs are protocol-consistent but not a validated optimum where the finalists are negative; '
                    if any_neg else '. The frozen HPs are protocol-consistent; ')
                 + 'the contrast (or its similarity to C) is not attributed to feature restriction/re-selection or capacity alone._\n')
    if dev is not None:
        df_dev, extra = dev
        L.append('## Device replication — primary vs replicate result directories (same frozen HPs, same code)\n')
        L.append(df_dev.to_markdown(index=False))
        # CODEX TP2-A A-01 (2026-09-12): descriptive only — no causal attribution of cell-level divergence and no claim
        # about inference (the replicate's own inferential statistics live in its own family1/analysis run, if any).
        same_sign = (np.sign(extra['pooled_delta_L1_L0_primary']) == np.sign(extra['pooled_delta_L1_L0_replicate']))
        per_arm = '; '.join(f"{r['arm']}: cell-IC corr {r['corr_cell_IC']:.3f}, mean |diff| {r['mean_abs_diff']:.4f}, "
                            f"max |diff| {r['max_abs_diff']:.4f}, {r['n_identical']}/{r['n_cells']} identical" for r in df_dev.to_dict('records'))
        L.append(f"\n_{extra['primary_dir']} (primary) vs {extra['replicate_dir']} (replicate): pooled ΔIC L1−L0 "
                 f"{extra['pooled_delta_L1_L0_primary']:+.5f} vs {extra['pooled_delta_L1_L0_replicate']:+.5f} "
                 f"({'same' if same_sign else 'DIFFERENT'} sign; absolute gap {abs(extra['pooled_delta_L1_L0_primary'] - extra['pooled_delta_L1_L0_replicate']):.5f}). "
                 f"Per arm — {per_arm}. These are descriptive replication statistics; whether the replicate's inference agrees is read "
                 f"from the replicate's own family1/analysis outputs, not from this table (source: {pfx}_device_replication.csv)._\n")
    with open(os.path.join(out_dir, f'{pfx}_comparison.md'), 'w') as f:
        f.write('\n'.join(L))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--universe', choices=sorted(SPECS), default='C5')
    p.add_argument('--main-dir', default=None, help='target universe result dir (default per --universe)')
    p.add_argument('--family-dir', default=None, help='target universe family1 --sensitivity dir (default per --universe)')
    p.add_argument('--frozen', default=None, help='target frozen_hparams json (default per --universe)')
    p.add_argument('--c5-main-dir', default=None,
                   help='C5: the target dir (legacy alias of --main-dir); CPRE: the C5 comparator dir')
    p.add_argument('--c5-family-dir', default=None, help='C5: legacy alias of --family-dir; CPRE: the C5 comparator family dir')
    p.add_argument('--c5-frozen', default=None, help='CPRE only: the C5 comparator frozen_hparams json (default per SPECS)')
    p.add_argument('--replicate-main-dir', default=None,
                   help='optional replicate result dir → <p>_device_replication.{csv,md} (cell-level primary-vs-replicate)')
    p.add_argument('--ex-fold', type=int, default=None,
                   help='report pooled stats with this fold excluded (fold flagged by LOFO as dominant, e.g. 9 = 2025Q2)')
    p.add_argument('--conf-main-dir', default='experiments/storya_v21_main12_tuned')
    p.add_argument('--conf-family-dir', default='artifacts/storya_v21_family1')
    p.add_argument('--out-dir', default=None, help='default = the target family dir')
    p.add_argument('--no-paired', action='store_true', help='skip the paired daily contrasts')
    p.add_argument('--smoke', action='store_true', help='n_boot=200; tolerate partial runs')
    p.add_argument('--conf-only', action='store_true',
                   help='cross-check mode: only the confirmatory C/B per-seed numbers (no target dirs needed)')
    args = p.parse_args()
    U = args.universe; spec = SPECS[U]; pfx = spec['prefix']
    if U == 'C5':
        main_dir = args.main_dir or args.c5_main_dir or spec['main_dir']
        family_dir = args.family_dir or args.c5_family_dir or spec['family_dir']
    else:
        main_dir = args.main_dir or spec['main_dir']
        family_dir = args.family_dir or spec['family_dir']
    frozen = args.frozen or spec['frozen']
    comp_dirs = {'C': (args.conf_main_dir, args.conf_family_dir), 'B': (args.conf_main_dir, args.conf_family_dir)}
    if 'C5' in spec['comparators']:
        comp_dirs['C5'] = (args.c5_main_dir or SPECS['C5']['main_dir'], args.c5_family_dir or SPECS['C5']['family_dir'])
    degraded = args.smoke or args.conf_only
    out_dir = args.out_dir or (family_dir + ('_smoke' if args.smoke else '_confonly') if degraded else family_dir)
    if degraded and os.path.abspath(out_dir) == os.path.abspath(family_dir):
        raise SystemExit('--smoke / --conf-only must not write into the published family dir; pass a different --out-dir')
    os.makedirs(out_dir, exist_ok=True)
    n_boot = 200 if args.smoke else N_BOOT

    conf_results = os.path.join(args.conf_main_dir, 'results.csv')
    rows = [seed_robustness(conf_results, u) for u in ['C', 'B']]
    for r in rows:
        print(f"  [{r['universe']} L1-L0] pooled={r['pooled_delta_IC']:+.5f} same-sign {r['n_same_sign']}/{r['n_seeds']} "
              f"LOSO flips {r['loso_sign_flips']}/{r['n_seeds']}  (confirmatory cross-check)")
    if args.conf_only:
        pd.DataFrame(rows).drop(columns=['per_seed_delta']).to_csv(os.path.join(out_dir, f'{pfx}_seed_robustness.csv'), index=False)
        print('conf-only cross-check done')
        return 0

    integ = run_integrity(U, main_dir, frozen, conf_results, strict=not args.smoke)
    integ['analysis_mode'] = {'n_boot': n_boot, 'strict': not args.smoke, 'smoke': bool(args.smoke), 'out_dir': out_dir}
    # EXPL-CODE-02: the family-dir statistics must have been computed from THIS main dir (results.csv md5)
    led_p = os.path.join(family_dir, 'family1_ledger.json')
    led = json.load(open(led_p)) if os.path.exists(led_p) else {}
    fam_md5 = (led.get('inputs') or {}).get('results_csv_md5')
    integ['family_dir_results_md5'] = fam_md5
    integ['family_dir_matches_main_dir'] = (fam_md5 == integ['inputs']['results_csv_md5']) if fam_md5 else None
    if not args.smoke and integ['family_dir_matches_main_dir'] is not True:
        raise SystemExit(f"--family-dir {family_dir} ledger inputs.results_csv_md5={fam_md5} does not match "
                         f"--main-dir {main_dir} results.csv md5={integ['inputs']['results_csv_md5']}; "
                         f"re-run compute_family1_ladder.py --sensitivity on this main dir first")
    with open(os.path.join(out_dir, f'{pfx}_run_integrity.json'), 'w') as f:
        json.dump(integ, f, indent=2)
    print(f"[integrity {U}] rows={integ['n_results_rows']}/{integ['n_expected']} failed={integ['n_failed']} "
          f"unique_ids={integ['cell_id_unique']} npy={integ['per_day_npy_present']} n_features={integ['n_features']} "
          f"short_cells={len(integ['cells_not_full_calendar_length'])} provenance_gate={integ['provenance_gate_ok']} "
          f"→ PASS={integ['PASS']}")
    if not integ['PASS'] and not args.smoke:
        raise SystemExit(f'{U} run integrity FAILED — see {pfx}_run_integrity.json; not building the comparison')

    rt = seed_robustness(os.path.join(main_dir, 'results.csv'), U)
    print(f"  [{U} L1-L0] pooled={rt['pooled_delta_IC']:+.5f} same-sign {rt['n_same_sign']}/{rt['n_seeds']} "
          f"LOSO flips {rt['loso_sign_flips']}/{rt['n_seeds']}")
    rows = [rt] + rows
    if 'C5' in spec['comparators']:
        r5 = seed_robustness(os.path.join(comp_dirs['C5'][0], 'results.csv'), 'C5')
        print(f"  [C5 L1-L0] pooled={r5['pooled_delta_IC']:+.5f} same-sign {r5['n_same_sign']}/{r5['n_seeds']} (comparator)")
        rows.append(r5)
    pd.DataFrame(rows).drop(columns=['per_seed_delta']).to_csv(os.path.join(out_dir, f'{pfx}_seed_robustness.csv'), index=False)
    with open(os.path.join(out_dir, f'{pfx}_seed_robustness_per_seed.json'), 'w') as f:
        json.dump({r['universe']: r['per_seed_delta'] for r in rows}, f, indent=2)

    paired = []
    if not args.no_paired:
        for other in spec['comparators']:
            try:
                pr = paired_contrast(U, main_dir, other, comp_dirs[other][0], n_boot, strict=not args.smoke,
                                     calendar=integ['frozen_calendar_days_per_fold'])
                paired.append(pr)
                print(f"  [paired {pr['contrast']}] diff={pr['mean_paired_diff']:+.5f} CI=[{pr['ci_lo']:+.5f}, {pr['ci_hi']:+.5f}] "
                      f"HLN p={pr['HLN_p_t']:.4f} (lag21 {pr['HLN_p_t_lag21']:.4f}) T={pr['T']}")
            except AssertionError as e:
                if not args.smoke:
                    raise
                print(f"  [paired {U} vs {other}] skipped (smoke): {e}")
        if paired:
            pd.DataFrame(paired).to_csv(os.path.join(out_dir, f'{pfx}_paired_contrast.csv'), index=False)

    seedrob = {r['universe']: r for r in rows}
    fam_dirs = {U: family_dir, **{o: comp_dirs[o][1] for o in spec['comparators']}}
    comp = pd.DataFrame([_row(family_dir, U, seedrob[U])] + [_row(comp_dirs[o][1], o, seedrob[o]) for o in spec['comparators']])
    comp.to_csv(os.path.join(out_dir, f'{pfx}_comparison.csv'), index=False)
    import run_storya_e1_anchor as _anchor
    extra_hp = ([('C5', args.c5_frozen or SPECS['C5']['frozen'], len(_anchor.UNIVERSE_C5_NAMES))]
                if 'C5' in spec['comparators'] else None)
    hp_sources = []
    hp_rows = hparam_report(U, frozen, int(integ['n_features']), extra=extra_hp, sources=hp_sources)
    if hp_rows:
        pd.DataFrame(hp_rows).to_csv(os.path.join(out_dir, f'{pfx}_tuned_hparams.csv'), index=False)
    ex_rows = []
    if args.ex_fold is not None:
        _cal = integ['frozen_calendar_days_per_fold'] if not args.smoke else None
        ex_rows = [ex_fold_stats(main_dir, U, args.ex_fold, n_boot, _cal)] + \
                  [ex_fold_stats(comp_dirs[o][0], o, args.ex_fold, n_boot, _cal) for o in spec['comparators']]
        pd.DataFrame(ex_rows).to_csv(os.path.join(out_dir, f'{pfx}_ex_fold.csv'), index=False)
        for r in ex_rows:
            _sh = (f"share {r['excluded_fold_share']:.0%}" if r['excluded_fold_share_is_meaningful']
                   else f"share n/a (pooled {r['pooled_delta_IC_all_folds']:+.4f} within 1 SE of 0)")
            print(f"  [ex-fold {r['excluded_fold']} {r['universe']}] fold ΔIC={r['excluded_fold_delta_IC']:+.4f} ({_sh}, "
                  f"contribution rank {r['excluded_fold_contribution_rank']}/{r['n_folds_present']}) → ex-fold ΔIC={r['mean_delta_IC_ex']:+.4f} "
                  f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] p={r['HLN_p_t']:.3f} (lag21 {r['HLN_p_t_lag21']:.3f})")
    dev = None
    if args.replicate_main_dir:
        dev = device_replication(main_dir, args.replicate_main_dir)
        dev[0].to_csv(os.path.join(out_dir, f'{pfx}_device_replication.csv'), index=False)
        with open(os.path.join(out_dir, f'{pfx}_device_replication.md'), 'w') as f:
            f.write(f'# {U} device replication — primary vs replicate (same frozen HPs, same code)\n\n' + dev[0].to_markdown(index=False)
                    + '\n\n' + json.dumps(dev[1]) + '\n')
        print(dev[0].to_string(index=False)); print(dev[1])
    # EXPL-STAT-04 (closeout): inventory of every nominal p-value this run publishes (no BH family opened)
    # EXPL-STAT-06 (closeout 2026-09-12): each entry carries a ROLE — the comparator L1-L0 rows are RE-PUBLISHED
    # (C/B: members of the confirmatory 20-test BH family at the auto lag, with their lag-21 column; C5: the earlier
    # sensitivity run), only the target / ex-fold / paired rows are NEW nominal tests of this run.
    def _role(u):
        return ('republished: confirmatory family1 (BH-family member at auto lag)' if u in ('B', 'C')
                else 'republished: earlier sensitivity run' if u != U else 'this run: nominal')
    tests = ([(f'{r.universe} L1-L0 HLN p (auto lag)', float(r.HLN_p_t), _role(r.universe)) for r in comp.itertuples()]
             + [(f'{r.universe} L1-L0 HLN p (lag 21)', float(r.HLN_p_t_lag21), _role(r.universe)) for r in comp.itertuples()]
             + [(f"{r['universe']} L1-L0 ex-fold-{r['excluded_fold']} HLN p (auto lag)", float(r['HLN_p_t']), 'this run: nominal') for r in ex_rows]
             + [(f"{r['universe']} L1-L0 ex-fold-{r['excluded_fold']} HLN p (lag 21)", float(r['HLN_p_t_lag21']), 'this run: nominal') for r in ex_rows]
             + [(f"paired {q['contrast']} HLN p (auto lag)", float(q['HLN_p_t']), 'this run: nominal') for q in paired]
             + [(f"paired {q['contrast']} HLN p (lag 21)", float(q['HLN_p_t_lag21']), 'this run: nominal') for q in paired])
    headline = f'{U} L1-L0 HLN p (auto lag)'
    head_p = {t[0]: t[1] for t in tests}[headline]; min_name, min_p, _ = min(tests, key=lambda t: t[1])
    n_new = sum(t[2].startswith('this run') for t in tests)
    inv = {'note': ('p-values published by this sensitivity run: the rows marked "this run" are nominal, unadjusted HLN p-values '
                    '(no multiplicity correction; no BH family opened); the rows marked "republished" are quoted from the '
                    'confirmatory family1 tables (BH-family members at the auto lag) or from the earlier sensitivity run'),
           'tests': [t[0] for t in tests], 'p_values': {t[0]: t[1] for t in tests}, 'roles': {t[0]: t[2] for t in tests},
           'n_tests_reported': len(tests), 'n_new_nominal_this_run': n_new, 'n_republished': len(tests) - n_new,
           'headline': headline, 'headline_p': head_p, 'smallest_p_test': min_name, 'smallest_p': min_p,
           'smallest_p_is_headline': bool(min_name == headline)}
    with open(os.path.join(out_dir, f'{pfx}_tests_reported.json'), 'w') as f:
        json.dump(inv, f, indent=2)
    print(f"[tests reported] {inv['n_tests_reported']} p-values ({n_new} new nominal this run + {len(tests) - n_new} republished; no BH); "
          f"smallest = {min_name} ({min_p:.4f})")
    write_md(U, out_dir, comp, paired, integ, seedrob, hp_rows, ex_rows, dev, fam_dirs, frozen, hp_sources)
    print(comp[['universe', 'mean_delta_IC', 'delta_ci_lo', 'delta_ci_hi', 'HLN_p_t', 'IC_L0', 'IC_L1',
                'MDE_2p8xSE', 'per_seed_same_sign', 'loso_flips']].to_string(index=False))
    print(f'[{U}] DONE → {out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
