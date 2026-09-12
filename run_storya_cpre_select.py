#!/usr/bin/env python
"""run_storya_cpre_select.py — C-pre: PRE-EVALUATION feature re-selection for Universe C (post-hoc sensitivity).

Frozen protocol: docs/c_pre_plan_2026-09-11.md §2–§3 (Codex Touchpoint-1 Round A 2026-09-11, all decisions agreed).
This script only SELECTS columns; it trains nothing. Everything it reads is dated on or before 2022-06-30
(label end) — strictly before the tuning-validation window (2022H2) and the 12-fold evaluation window
(2023Q1–2025Q4). The decision to run it was taken after the paper's test results were known (post-hoc).
SCOPE OF THE "PRE-EVALUATION" CLAIM (closeout EXPL-LEAK-03): only the selector's INPUTS are bounded by 2022-06-30.
The RULE is not: the marginal-|IC| group score, the reused 61-group partition, the top-15 cut and the eligibility
thresholds were carried over from analyze_plan_aaa_t1_diagnostic.py (whose own scoring window, the last 313 valid
label days, lies inside the evaluation period) and from the test-informed Plan-AAA -> Universe C construction.

Candidates (168 = artifacts/plan_aaa/groups_168.json feature_order):
  * 10 hc  — run_step3_plan_z_part_a.load_data_and_features()['features_np'] exactly as Plan AAA consumed them
             (rolling(w).{mean,std}().shift(1); phase5 columns carry shift(1)/shift(22) from build time) → T−1.
  * 158 Alpha158 — data/reference/sp500_5y_alpha158_features_raw.npy (SAME-DAY OHLC at build time), T−1-shifted
             here exactly as run_storya_e1_anchor.build_universe_C does at runtime: np.roll(axis=0, 1), row 0 := 0.
Label: run_storya_e1_anchor.build_labels(prices, 21) (the ladder's label; locked).

Selector (§3):
  D_sel      = train days of run_storya_v21_tune.TUNE_FOLD (train_end 2022-06-30) via create_fold_masks — feature
               dates 2021-07-01 … 2022-05-31 whose 21d label ends ≤ 2022-06-30 (231 dates; asserted).
  eligible   = for feature f and date t: ≥ 30 stocks with a valid label and a finite feature value, feature
               cross-section non-constant (std > 1e-9) [same conditions as analyze_plan_aaa_t1_diagnostic].
               NOTE (closeout EXPL-LEAK-01): BOTH candidate sources are NaN->0-filled at build time
               (build_alpha158_features.py:362, before the _raw.npy save; run_step3_plan_z_part_a.py:113), so the
               "finite" test is vacuous and missing observations enter as imputed zeros (build-time NaN rate for the
               selected Alpha158 columns: median 1.7%, max 3.3%). In practice the coverage rule binds only through the
               NON-CONSTANT test - which is exactly what excludes hc_mom12m (all-zero cross-sections before 2022-01-28).
  coverage   = n_eligible(f) / |D_sel|; scored iff coverage ≥ TAU (0.50); UNSCORED contributes nothing (undefined
               IC is never treated as 0). Known consequence: hc_mom12m (85/231 = 0.37) is UNSCORED.
  IC̄_f       = time-mean over f's eligible dates of the daily cross-sectional Spearman IC(feature_f(t,·), label(t,·)).
  S(g)       = mean over SCORED members m of g of |IC̄_m|  (groups = the 61 Plan-AAA groups, reused, not re-clustered);
               0 scored members → UNRANKED. Rank by S desc, then n_scored desc, then group_id asc.
  C-pre      = union of ALL members of the top-15 groups (group = unit of selection), ordered by group rank then by
               member order in groups_168.json. Width is data-determined; NOT intersected with Universe C.
Robustness (rankings only, no model runs): the ranking is re-tabulated at TAU = 0 (admit the short-history feature)
and TAU = 0.75; top-15 overlap vs the frozen TAU = 0.50 ranking is reported. The frozen rule is TAU = 0.50 regardless.

Outputs → artifacts/storya_cpre_select/ (git-whitelisted):
  feature_scores.csv, eligible_dates.json, group_scores.csv, selection.json (rules, window, ordered columns + md5,
  input md5s incl. the resolved ticker/date axes, git rev of THIS committed source), selector_robustness.csv, summary.md
Run: python run_storya_cpre_select.py [--smoke]   (--smoke: first 5 candidates only → artifacts/storya_cpre_select_smoke/)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
import run_storya_e1_anchor as anchor                       # noqa: E402  (load_core_data / build_labels / create_fold_masks)
import run_step3_plan_z_part_a as pa                        # noqa: E402  (the 10 hc features, Plan-AAA source)
from run_storya_v21_tune import TUNE_FOLD                  # noqa: E402  (train_end 2022-06-30 / val 2022H2)

GROUPS_JSON = 'artifacts/plan_aaa/groups_168.json'
OUT_DIR = 'artifacts/storya_cpre_select'
TAU = 0.50                      # frozen minimum coverage (plan §3.3, decision D1)
TAU_ROBUSTNESS = (0.0, 0.75)    # rankings only
TOP_K = 15                      # cardinality carried over from the Plan-AAA -> Universe C construction (whose own top-15 was a
                                # test-informed ranking); kept for width continuity, NOT re-optimised — the only residual
                                # test-informed input of this selector (closeout EXPL-STAT-07, 2026-09-12); the ranking itself is pre-evaluation
MIN_STOCKS = 30
CONST_EPS = 1e-9
EXPECTED_N_DATES = 231          # Codex TP1-B / Claude 2026-09-11 (asserted, not assumed)
EXPECTED_FIRST, EXPECTED_LAST = '2021-07-01', '2022-05-31'
SELECTION_LABEL_END = TUNE_FOLD['train_end']   # '2022-06-30'


def _md5_file(p: str) -> str:
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def _md5_text(s: str) -> str:
    return hashlib.md5(s.encode('utf-8')).hexdigest()


def git_identity(paths: list) -> dict:
    """git rev + per-file status/blob for the selector source and the modules it imports (plan §3.6 / Codex A-04)."""
    out = {'git_rev': None, 'git_error': None, 'files': {}}
    for p in paths:
        out['files'][os.path.relpath(p, PROJECT_ROOT)] = {'md5': _md5_file(p)}
    try:
        out['git_rev'] = subprocess.check_output(['git', '-C', PROJECT_ROOT, 'rev-parse', 'HEAD'], text=True).strip()
        top = subprocess.check_output(['git', '-C', PROJECT_ROOT, 'rev-parse', '--show-toplevel'], text=True).strip()
        rel = [os.path.relpath(p, top) for p in paths]
        # EXPL-CODE-02 (closeout 2026-09-12): `git status --porcelain -- <p>` is EMPTY for an IGNORED file, which
        # would read as 'clean'. Probe tracked-ness explicitly so an untracked/ignored module fails source_clean.
        tracked = set()
        try:
            tracked = set(subprocess.check_output(['git', '-C', top, 'ls-files', '--'] + rel, text=True).split())
        except Exception:
            pass
        st = subprocess.check_output(['git', '-C', top, 'status', '--porcelain', '--'] + rel, text=True)
        status = {ln[3:]: ln[:2].strip() for ln in st.splitlines()}
        for _r in rel:
            if _r not in tracked:
                status[_r] = '!!untracked-or-ignored'
        shas = subprocess.check_output(['git', '-C', top, 'hash-object'] + rel, text=True).split()
        for p, r, sha in zip(paths, rel, shas):
            out['files'][os.path.relpath(p, PROJECT_ROOT)].update({'blob_sha': sha, 'git_status': status.get(r, '')})
        out['source_clean'] = all(v.get('git_status') == '' for v in out['files'].values())
    except Exception as e:  # git unavailable → md5s stand alone, source_clean unknown
        out['git_error'] = str(e)
        out['source_clean'] = None
    return out


def load_candidates() -> dict:
    """168 candidates in groups_168.json feature_order: hc (part_a tensor) + Alpha158 (T−1 rolled)."""
    groups = json.load(open(GROUPS_JSON))
    feature_order = list(groups['feature_order'])
    assert len(feature_order) == groups['num_features'] == 168, feature_order[:3]

    anchor.setup_workdir()
    core = anchor.load_core_data()
    prices, returns, all_dates = core['prices'], core['returns'], core['all_dates']
    labels_np, label_valid_np = anchor.build_labels(prices, anchor.HORIZON)

    base = pa.load_data_and_features()
    hc = base['features_np'].astype(np.float32)                              # (T, N, 10), T−1 by construction
    hc_names = [f'hc_{n}' for n in base['feature_names']]
    # the two loaders must resolve the SAME ordered ticker and date axes (Codex TP1-A checked: identical 501 lists)
    assert list(base['valid_tickers']) == list(core['valid_tickers']), 'ticker axis differs between anchor and part_a'
    assert list(base['all_dates']) == list(all_dates), 'date axis differs between anchor and part_a'
    assert hc.shape[:2] == labels_np.shape, (hc.shape, labels_np.shape)

    meta = json.load(open(anchor.PATHS['alpha158_meta']))
    a_names = list(meta['feature_order'])
    a_raw = np.load(anchor.PATHS['alpha158_npy']).astype(np.float32)        # (T, N, 158) SAME-DAY at build time
    assert a_raw.shape == (labels_np.shape[0], labels_np.shape[1], 158), a_raw.shape
    a_t1 = np.roll(a_raw, shift=1, axis=0)                                   # == build_universe_C's runtime shift
    a_t1[0] = 0.0
    assert np.array_equal(a_t1[1], a_raw[0]) and np.all(a_t1[0] == 0.0)

    assert feature_order[:10] == hc_names and feature_order[10:] == a_names, 'groups_168 feature_order mismatch'
    cand = np.concatenate([hc, a_t1], axis=-1)                               # (T, N, 168) in feature_order
    source = ['hc(part_a, T-1 by construction)'] * 10 + ['alpha158_raw shifted np.roll(1), row0=0'] * 158
    return {'groups': groups, 'feature_order': feature_order, 'cand': cand, 'source': source,
            'labels_np': labels_np, 'label_valid_np': label_valid_np, 'all_dates': all_dates,
            'tickers': list(core['valid_tickers']), 'prices': prices,
            'news_tickers_md5': _md5_text(','.join(sorted(pd.read_parquet(
                'data/fullscale/sp500_news_events.parquet', columns=['ticker'])['ticker'].unique().tolist())))}


def selection_window(all_dates) -> np.ndarray:
    """D_sel = TUNE_FOLD train days (feature dates whose 21d label ends ≤ train_end), ≥ TRAIN_START."""
    train_days, val_days, _ = anchor.create_fold_masks(TUNE_FOLD, all_dates, anchor.HORIZON)
    train_days = np.asarray(train_days, dtype=int)
    first, last = all_dates[train_days[0]].strftime('%Y-%m-%d'), all_dates[train_days[-1]].strftime('%Y-%m-%d')
    assert first >= anchor.TRAIN_START, (first, anchor.TRAIN_START)
    assert len(train_days) == EXPECTED_N_DATES and (first, last) == (EXPECTED_FIRST, EXPECTED_LAST), \
        f'selection window changed: n={len(train_days)} {first}..{last} (expected {EXPECTED_N_DATES} {EXPECTED_FIRST}..{EXPECTED_LAST})'
    # label of the last selection date ends ≤ train_end; first val date is after train_end
    last_label_end = all_dates[train_days[-1] + anchor.HORIZON].strftime('%Y-%m-%d')
    assert last_label_end <= SELECTION_LABEL_END, (last_label_end, SELECTION_LABEL_END)
    val_first = all_dates[int(np.asarray(val_days)[0])].strftime('%Y-%m-%d')
    assert val_first > SELECTION_LABEL_END, (val_first, SELECTION_LABEL_END)
    return train_days


def score_features(cand: np.ndarray, labels_np: np.ndarray, label_valid_np: np.ndarray, days: np.ndarray,
                   names: list, source: list, n_limit: int | None = None) -> tuple:
    """Per-feature eligible dates + daily Spearman IC; returns (feature_scores rows, eligible_dates dict)."""
    rows, elig = [], {}
    n_feat = cand.shape[-1] if n_limit is None else min(n_limit, cand.shape[-1])
    t0 = time.time()
    for j in range(n_feat):
        ics, dates_ok = [], []
        for d in days:
            x = cand[d, :, j]
            m = label_valid_np[d] & np.isfinite(x)
            if int(m.sum()) < MIN_STOCKS:
                continue
            xd, yd = x[m], labels_np[d, m]
            if xd.std() < CONST_EPS or yd.std() < CONST_EPS:
                continue
            rho, _ = spearmanr(xd, yd)
            if np.isnan(rho):
                continue
            ics.append(float(rho)); dates_ok.append(int(d))
        n_el = len(ics)
        cov = n_el / len(days)
        ic_bar = float(np.mean(ics)) if n_el else float('nan')
        rows.append({'feature': names[j], 'idx': j, 'source': source[j], 'n_eligible': n_el, 'n_window': int(len(days)),
                     'coverage': round(cov, 4), 'ic_mean': round(ic_bar, 6) if n_el else np.nan,
                     'abs_ic_mean': round(abs(ic_bar), 6) if n_el else np.nan,
                     'ic_daily_sd': round(float(np.std(ics, ddof=1)), 6) if n_el > 1 else np.nan})
        elig[names[j]] = dates_ok
        if j % 40 == 0:
            print(f'  [score] {j}/{n_feat} {names[j]}: n_eligible={n_el} coverage={cov:.3f} ({time.time() - t0:.0f}s)')
    return rows, elig


def rank_groups(feat: pd.DataFrame, groups: dict, tau: float) -> pd.DataFrame:
    """S(g) = mean over scored members of |IC̄|; UNRANKED if no scored member; deterministic tie-break."""
    scored = {r.feature: (r.coverage >= tau) and np.isfinite(r.abs_ic_mean) for r in feat.itertuples()}
    absic = {r.feature: r.abs_ic_mean for r in feat.itertuples()}
    rows = []
    for g in groups['groups']:
        members = list(g['members'])
        sm = [m for m in members if scored.get(m, False)]
        S = float(np.mean([absic[m] for m in sm])) if sm else float('nan')
        rows.append({'group_id': int(g['group_id']), 'group_label': g['label'], 'n_members': len(members),
                     'n_scored': len(sm), 'n_unscored': len(members) - len(sm),
                     'n_missing_from_scoring': sum(m not in scored for m in members),
                     'score_mean_abs_ic': round(S, 6) if sm else np.nan,
                     'members': ','.join(members),
                     'member_abs_ic': ';'.join(f'{m}={absic[m]:.6f}' if scored.get(m, False) else f'{m}=UNSCORED' for m in members)})
    df = pd.DataFrame(rows)
    ranked = df[df.n_scored > 0].sort_values(['score_mean_abs_ic', 'n_scored', 'group_id'],
                                             ascending=[False, False, True]).copy()
    ranked['rank'] = np.arange(1, len(ranked) + 1)
    unranked = df[df.n_scored == 0].copy(); unranked['rank'] = np.nan
    out = pd.concat([ranked, unranked]).reset_index(drop=True)
    out['selected_top15'] = out['rank'] <= TOP_K
    return out


def columns_from_ranking(gr: pd.DataFrame, groups: dict) -> list:
    by_id = {int(g['group_id']): list(g['members']) for g in groups['groups']}
    cols = []
    for r in gr[gr.selected_top15].sort_values('rank').itertuples():
        for m in by_id[r.group_id]:
            if m not in cols:
                cols.append(m)
    return cols


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--smoke', action='store_true', help='score the first 5 candidates only; write to *_smoke/')
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or (OUT_DIR + '_smoke' if args.smoke else OUT_DIR)
    if args.smoke and os.path.abspath(out_dir) == os.path.abspath(OUT_DIR):
        raise SystemExit('--smoke must not write into the published selection dir')
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    src_files = [os.path.abspath(__file__), os.path.abspath(anchor.__file__), os.path.abspath(pa.__file__),
                 os.path.abspath(sys.modules['run_storya_v21_tune'].__file__)]
    ident = git_identity(src_files)
    print(f'[cpre] git_rev={ident["git_rev"]} source_clean={ident.get("source_clean")}')

    C = load_candidates()
    days = selection_window(C['all_dates'])
    all_dates = C['all_dates']
    print(f'[cpre] D_sel: {len(days)} feature dates {all_dates[days[0]].date()}..{all_dates[days[-1]].date()}; '
          f'label end ≤ {SELECTION_LABEL_END}; {C["cand"].shape[-1]} candidates; {C["groups"]["num_groups"]} groups')

    rows, elig = score_features(C['cand'], C['labels_np'], C['label_valid_np'], days, C['feature_order'], C['source'],
                                n_limit=5 if args.smoke else None)
    feat = pd.DataFrame(rows)
    feat['scored_tau'] = (feat.coverage >= TAU) & feat.abs_ic_mean.notna()
    feat['exclusion_reason'] = np.where(feat.scored_tau, '', np.where(feat.n_eligible == 0, 'no eligible date',
                                                                      f'coverage < {TAU}'))
    feat.to_csv(os.path.join(out_dir, 'feature_scores.csv'), index=False)
    with open(os.path.join(out_dir, 'eligible_dates.json'), 'w') as f:
        json.dump({'note': 'per-feature eligible date INDICES into the panel date axis (see selection.json date_axis)',
                   'window_indices': [int(d) for d in days], 'eligible': elig}, f)

    gr = rank_groups(feat, C['groups'], TAU)
    gr.to_csv(os.path.join(out_dir, 'group_scores.csv'), index=False)
    cols = columns_from_ranking(gr, C['groups'])
    sel_groups = gr[gr.selected_top15].sort_values('rank')
    print(f'[cpre] top-{TOP_K} groups → {len(cols)} columns ({sum(c.startswith("hc_") for c in cols)} hc + '
          f'{sum(not c.startswith("hc_") for c in cols)} Alpha158)')
    for r in sel_groups.itertuples():
        print(f'   #{int(r.rank):2d} {r.group_label:<18s} S={r.score_mean_abs_ic:.5f} scored {r.n_scored}/{r.n_members}')

    # robustness (rankings only) — τ variants vs the frozen τ
    rob = []
    frozen_set = set(sel_groups.group_id)
    for tau in sorted(set(TAU_ROBUSTNESS) | {TAU}):
        g2 = rank_groups(feat, C['groups'], tau)
        s2 = set(g2[g2.selected_top15].group_id)
        rob.append({'tau': tau, 'frozen': tau == TAU, 'n_scored_features': int(((feat.coverage >= tau) & feat.abs_ic_mean.notna()).sum()),
                    'top15_overlap_with_frozen': len(s2 & frozen_set), 'identical_to_frozen': s2 == frozen_set,
                    'top15_groups': ','.join(g2[g2.selected_top15].sort_values('rank').group_label)})
    pd.DataFrame(rob).to_csv(os.path.join(out_dir, 'selector_robustness.csv'), index=False)

    date_axis = [d.strftime('%Y-%m-%d') for d in all_dates]
    unscored = feat[~feat.scored_tau][['feature', 'coverage', 'n_eligible', 'exclusion_reason']].to_dict('records')
    selection = {
        'universe': 'CPRE', 'protocol': 'docs/c_pre_plan_2026-09-11.md §2-§3 (frozen; Codex TP1-A 2026-09-11)',
        'smoke': bool(args.smoke), 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'rules': {'tau_min_coverage': TAU, 'top_k_groups': TOP_K, 'min_stocks_per_date': MIN_STOCKS, 'const_eps': CONST_EPS,
                  'group_score': 'mean over SCORED members of |time-mean daily Spearman IC|',
                  'tie_break': 'score desc, n_scored desc, group_id asc', 'column_set': 'union of ALL members of the top-15 groups, '
                  'ordered by group rank then member order in groups_168.json', 'undefined_ic': 'never treated as 0 (UNSCORED)'},
        'selection_window': {'first_feature_date': date_axis[days[0]], 'last_feature_date': date_axis[days[-1]],
                             'n_feature_dates': int(len(days)), 'label_end_max': SELECTION_LABEL_END,
                             'tune_fold': TUNE_FOLD, 'train_start': anchor.TRAIN_START, 'horizon': anchor.HORIZON,
                             'window_indices_md5': _md5_text(','.join(map(str, days)))},
        'candidates': {'n': int(C['cand'].shape[-1]), 'n_hc': 10, 'n_alpha158': 158, 'groups_json': GROUPS_JSON,
                       'n_groups': int(C['groups']['num_groups']),
                       'group_calibration_window': C['groups'].get('calibration_window_dates')},
        'unscored_features': unscored,
        'selected_groups': [{'rank': int(r.rank), 'group_id': int(r.group_id), 'label': r.group_label,
                             'score': float(r.score_mean_abs_ic), 'n_scored': int(r.n_scored), 'n_members': int(r.n_members),
                             'members': r.members.split(',')} for r in sel_groups.itertuples()],
        'columns': cols, 'n_columns': len(cols), 'columns_md5': _md5_text(','.join(cols)),
        'overlap_with_universe_C': sorted(set(cols) & set(anchor.UNIVERSE_C_ALPHA158_NAMES + anchor.UNIVERSE_C_EXTRA_NAMES)),
        'overlap_with_C5': sorted(set(cols) & set(anchor.UNIVERSE_C5_NAMES)),
        'ticker_axis': {'n': len(C['tickers']), 'first': C['tickers'][0], 'last': C['tickers'][-1],
                        'md5': _md5_text(','.join(C['tickers']))},
        'date_axis': {'n': len(date_axis), 'first': date_axis[0], 'last': date_axis[-1], 'md5': _md5_text(','.join(date_axis))},
        'input_md5': {'alpha158_npy': _md5_file(anchor.PATHS['alpha158_npy']), 'alpha158_meta': _md5_file(anchor.PATHS['alpha158_meta']),
                      'phase5_npy': _md5_file(anchor.PATHS['phase5_npy']), 'prices': _md5_file(anchor.PATHS['prices']),
                      'sectors': _md5_file(anchor.PATHS['sectors']), 'news_events_ticker_membership': C['news_tickers_md5'],
                      'groups_168_json': _md5_file(GROUPS_JSON)},
        'source_identity': ident,
        'robustness': rob,
    }
    with open(os.path.join(out_dir, 'selection.json'), 'w') as f:
        json.dump(selection, f, indent=2, ensure_ascii=False)

    L = ['# C-pre selection — pre-evaluation feature re-selection (post-hoc sensitivity)\n',
         f'Run: {selection["timestamp"]}{" (SMOKE)" if args.smoke else ""} | git {ident["git_rev"]} (source_clean={ident.get("source_clean")}) | '
         f'wall {time.time() - t0:.0f}s\n',
         f'Window: {len(days)} feature dates {date_axis[days[0]]}..{date_axis[days[-1]]}, label end ≤ {SELECTION_LABEL_END} '
         f'(TUNE_FOLD train days; val 2022H2 and the 12-fold test window are NOT used). τ = {TAU}; top-{TOP_K} groups.\n',
         f'Candidates: {C["cand"].shape[-1]} (10 hc + 158 Alpha158, T−1). Unscored: '
         + (', '.join(f"{u['feature']} (coverage {u['coverage']})" for u in unscored) or 'none') + '\n',
         f'**Selected: {len(cols)} columns** from {len(sel_groups)} groups; overlap with Universe C = '
         f'{len(selection["overlap_with_universe_C"])} columns, with C5 = {len(selection["overlap_with_C5"])}.\n',
         '| rank | group | score = mean member abs-IC | scored/members | members |', '|---|---|---|---|---|']
    for r in sel_groups.itertuples():
        L.append(f'| {int(r.rank)} | {r.group_label} | {r.score_mean_abs_ic:.5f} | {r.n_scored}/{r.n_members} | {r.members} |')
    L.append('\nRobustness (rankings only; the frozen rule is τ = 0.50 regardless):\n')
    L.append(pd.DataFrame(rob).drop(columns=['top15_groups']).to_markdown(index=False))
    L.append('\n(source: feature_scores.csv / group_scores.csv / selection.json / selector_robustness.csv in this directory)\n')
    with open(os.path.join(out_dir, 'summary.md'), 'w') as f:
        f.write('\n'.join(L))
    print(f'[cpre] DONE → {out_dir} ({time.time() - t0:.0f}s); columns_md5={selection["columns_md5"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
