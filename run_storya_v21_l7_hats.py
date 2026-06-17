#!/usr/bin/env python
"""
Story A v2.1 — L7 HATS-3R-adapt on the 12-fold main axis (SEPARATE runner per protocol §6/§8).

Per docs/protocol_v2_freeze.md (L7 = 领域修法代表 / domain-method baseline, relation attention)
+ Touchpoint 1 disposition artifacts/reviews/2026-06-12_codex_plan_T1.md (Cn5 contingency).

WHY SEPARATE (protocol §8): L7 HATS is budgeted + run apart from the 8 standard-neural arms
(its 3-relation attention + per-cell α-diagnostics differ from the ladder). It still lives in the
v2.1 cell_id space (arm 'L7' = index 7) so it joins the same DM-HLN/SPA family as a candidate.

IMPORT-ONLY (§5 铁律): the HATS model + trainer + 3-relation edge builder are imported from the
E0/T2-validated run_storya_e1_6_hats.py; the 12-fold definition + cell_id from run_storya_v21_main12.py;
all data construction from run_storya_e1_anchor.py. This file writes ONLY: the 12-fold HATS loop,
the cell_id remap into the v2.1 space, an injection canary (relation-assignment provenance), and the
§6 contingency trigger.

§6 CONTINGENCY (Cn5, mechanical, locked BEFORE the run; health diagnostics only, NOT performance):
  Over the 240 cells (12 fold × 10 seed × 2 universe), DEMOTE HATS to exploratory (remove from the
  confirmatory pair table + SPA M=9→8) if ANY of:
    (a) any cell_id injection assert fails, OR
    (b) > 20% cells diverged (IC=NaN / no valid eval days, or hit max epochs with no val improvement), OR
    (c) > 20% cells α-collapsed (alpha_max_fraction_collapsed_test > 0.9).
  Otherwise KEEP HATS confirmatory. The trigger + counts are logged to the ledger regardless.

CLI:
  --canary        relation-assignment injection canary (no training) — verifies the 3 relations are
                  wired corr/sector/news in order, + off-by-one negative test, + cell_id injectivity.
  --smoke         1 fold × 1 seed (Univ B), --smoke-fold default 11.
  --contingency   read results.csv and print the §6 verdict (no training).
  --universe/--folds/--seeds/--resume   as in the main runner.
"""

import os
import sys
import time
import argparse
import json
import hashlib
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: F401  (HATS module uses F internally; kept for parity)

# ── import-only reuse (§5): 12-fold + cell_id from main runner; data from anchor; HATS from e1_6 ──
import run_storya_e1_anchor as anchor
from run_storya_v21_main12 import WALK_FORWARD_FOLDS_12, cell_id, N_FOLDS, ARM_ORDER, load_frozen_hparams
from run_storya_e1_anchor import (
    CANONICAL_SEEDS, HORIZON, COST_LEVELS_BPS, COST_CONVENTION,
    load_core_data, build_universe_B, build_universe_C, build_labels,
    build_correlation_snapshots, get_frozen_snapshot_idx, create_fold_masks,
    winsorize_train_only, standardize_train_only, compute_daily_ic,
    compute_cost_ladder_sharpe, get_device,
)
import run_storya_e1_6_hats as hats
from run_storya_e1_6_hats import (
    HATS3RAdapt, train_hats, build_three_relation_edges_per_fold, HATS_HPARAMS,
)
from run_storya_e3_news_edge import load_news_edge_source, build_per_day_news_edges
from run_storya_e4_alpha import build_sector_edges

ARM = 'L7'
ALL_UNIVERSES = ['B', 'C']
ALPHA_COLLAPSE_FRAC_THRESHOLD = 0.9     # per-cell α collapse indicator (Cn5)
CONTINGENCY_CELL_FRACTION = 0.20        # >20% rule (Cn5, locked = 20%)

OUT_DIR = 'experiments/storya_v21_l7_hats'
RESULTS_CSV = f'{OUT_DIR}/results.csv'
MANIFEST_CSV = f'{OUT_DIR}/manifest.csv'
PER_DAY_IC_DIR = f'{OUT_DIR}/per_day_ic'
ALPHA_DIAG_DIR = f'{OUT_DIR}/alpha_diag'

# §4 frozen-HP injection for L7 (D-RERUN-12F): preserved HATS pilot defaults = the no-flag baseline
# (byte-identical to the untuned L7). Injection patches hats.HATS_HPARAMS per universe (B_L7 / C_L7).
_ORIG_HATS = dict(hats.HATS_HPARAMS)

# Schema = base (parity with main runner) + 4 HATS α-diagnostic cols + diverged flag.
L7_RESULTS_COLUMNS = (
    ['cell_id', 'universe', 'arm', 'model', 'seed', 'fold', 'test_period',
     'IC_mean', 'IC_std', 'n_test_days', 'Sharpe_gross']
    + [f'Sharpe_net_{c}bps' for c in COST_LEVELS_BPS]
    + ['mean_turnover_L1', 'n_periods', 'best_val_loss', 'epochs_run',
       'alpha_mean_corr_test', 'alpha_mean_sector_test', 'alpha_mean_news_test',
       'alpha_max_fraction_collapsed_test', 'diverged_flag',
       'wall_time_sec', 'converged_flag', 'cost_convention']
)
MANIFEST_COLUMNS = ['cell_id', 'universe', 'arm', 'seed', 'fold',
                    'status', 'start_ts', 'end_ts', 'wall_time_sec', 'err']


# ══════════════════════════════════════════════════════════════
# CELL_ID — L7 lives in the v2.1 space (arm 'L7' = index 7)
# ══════════════════════════════════════════════════════════════

def cell_id_l7(universe_idx: int, fold_idx: int, seed_idx: int) -> int:
    return cell_id(universe_idx, ARM, fold_idx, seed_idx)


def assert_cell_id_l7_injective() -> None:
    """L7's 240 ids (2 universe × 12 fold × 10 seed) must be injective and disjoint from the other
    arms — guaranteed by the shared v2.1 cell_id radix (arm block 'L7'=index 7 → [840+u*1200, ...])."""
    seen = set()
    for u in range(2):
        for f in range(N_FOLDS):
            for s in range(10):
                cid = cell_id_l7(u, f, s)
                assert cid not in seen, f"L7 cell_id collision u={u} f={f} s={s}"
                seen.add(cid)
    assert len(seen) == 240, f"L7 expected 240 cells, got {len(seen)}"
    # disjoint-from-other-arms is structural (arm index 7 occupies its own *120 block per universe)
    lo, hi = min(seen), max(seen)
    print(f"✓ L7 cell_id injective, n=240, range [{lo}, {hi}] (arm 'L7' block in v2.1 space)")


# ══════════════════════════════════════════════════════════════
# INJECTION CANARY (§10 #5) — relation-assignment provenance, no training
# ══════════════════════════════════════════════════════════════

def _sig(ei_np) -> tuple:
    """Order-invariant edge signature: (n, n_unique_pairs, hash) — sensitive to relation mix-up."""
    if ei_np.shape[1] == 0:
        return (0, 0, 0)
    pairs = sorted(set(zip(ei_np[0].tolist(), ei_np[1].tolist())))
    return (ei_np.shape[1], len(pairs), hash(tuple(pairs)))


def run_injection_canary(returns, all_dates, num_days, num_stocks, ticker_to_idx) -> bool:
    """Verify build_three_relation_edges_per_fold assigns the 3 relations in the documented order
    [corr_frozen, sector_static, news_per_day] — a mis-assignment (e.g. sector where news belongs)
    would silently confound L7. Checks, all must PASS:
      (a) relation 0 == the frozen corr snapshot the runner feeds; relation 1 == build_sector_edges;
          relation 2 == build_per_day_news_edges for the sampled day (provenance, exact signature).
      (b) off-by-one: relation 0 != an adjacent corr snapshot (catches snapshot off-by-one).
      (c) HATS3RAdapt forward contract: exactly 3 edge lists expected (model asserts len==3)."""
    import run_storya_v21_main12 as _m  # only for parity of constants; no side effects
    print('\n=== L7 INJECTION CANARY (relation-assignment provenance, no training) ===')
    ok = True
    fold = WALK_FORWARD_FOLDS_12[5]
    snaps, _, sps = build_correlation_snapshots(returns, num_days)
    tr, va, te = create_fold_masks(fold, all_dates, HORIZON)
    fsi = get_frozen_snapshot_idx(tr[-1], sps)
    corr_np = snaps[fsi].cpu().numpy()
    sector_np = build_sector_edges(anchor.PATHS['sectors'], ticker_to_idx)
    news_df = load_news_edge_source()
    news_snaps, _ = build_per_day_news_edges(news_df, all_dates, ticker_to_idx)
    three = build_three_relation_edges_per_fold(corr_np, sector_np, news_snaps, tr, va, te)

    sample_day = int(te[len(te) // 2])
    rel = three[sample_day]  # [ei_corr, ei_sector, ei_news]
    a = (len(rel) == 3
         and _sig(rel[0]) == _sig(corr_np)
         and _sig(rel[1]) == _sig(sector_np)
         and _sig(rel[2]) == _sig(news_snaps.get(sample_day, np.zeros((2, 0), dtype=np.int64))))
    ok &= a
    print(f'[L7-a] relation assignment @day{sample_day}: corr={_sig(rel[0])==_sig(corr_np)}, '
          f'sector={_sig(rel[1])==_sig(sector_np)}, news={_sig(rel[2])==_sig(news_snaps.get(sample_day, np.zeros((2,0),dtype=np.int64)))} '
          f'-> {"PASS" if a else "FAIL"}')

    offby1 = True
    for adj in (fsi - 1, fsi + 1):
        if 0 <= adj < len(sps) and _sig(snaps[adj].cpu().numpy()) == _sig(corr_np):
            offby1 = False
    ok &= offby1
    print(f'[L7-b] corr off-by-one caught={offby1} -> {"PASS" if offby1 else "FAIL"}')

    try:
        m = HATS3RAdapt(in_channels=10, hidden=HATS_HPARAMS['hidden_channels'],
                        num_relations=HATS_HPARAMS['num_relations'])
        x = torch.zeros(num_stocks, 10)
        eis = [torch.from_numpy(e).long() for e in rel]
        with torch.no_grad():
            out = m(x, eis)
        c = (out.shape[0] == num_stocks) and (m.last_alpha is not None) and (m.last_alpha.shape[1] == 3)
    except Exception as e:
        c = False
        print(f'    forward error: {e}')
    ok &= c
    print(f'[L7-c] HATS forward(3 relations) -> alpha (N,3): {"PASS" if c else "FAIL"}')
    print(f'=== L7 INJECTION CANARY {"ALL PASS ✓" if ok else "FAIL ✗"} ===\n')
    return ok


# ══════════════════════════════════════════════════════════════
# §6 CONTINGENCY (Cn5) — mechanical, on the 240-cell results
# ══════════════════════════════════════════════════════════════

def evaluate_contingency(results_df: pd.DataFrame, manifest_df: pd.DataFrame = None,
                         expected_n: int = 240) -> dict:
    """Apply the locked §6 (Cn5) rule over the FULL expected grid (240 cells), NOT the reduced
    completed-row count (CODEX-C-02 fix).
      diverged := diverged_flag==1 (runtime: no valid IC / non-finite val loss); a cell that was
        ATTEMPTED but FAILED (exception → manifest 'failed', no results row) is itself a health
        failure → counted as diverged.
      collapsed := alpha_max_fraction_collapsed_test > 0.9 (only defined on cells that produced α).
    Denominator = expected_n always. Verdict is INCOMPLETE/PENDING until all expected cells are
    attempted (unless a trigger already fires definitively against the full grid)."""
    n_completed = len(results_df)
    n_failed = 0
    if manifest_df is not None and 'status' in manifest_df.columns and len(manifest_df):
        n_failed = int((manifest_df['status'] == 'failed').sum())
    if n_completed == 0 and n_failed == 0:
        return {'verdict': 'PENDING', 'n_cells': 0, 'n_failed': 0, 'n_expected': expected_n,
                'triggers': ['no cells yet']}
    diverged_done = int((results_df['diverged_flag'].astype(int) == 1).sum()) if n_completed else 0
    collapsed_done = int((results_df['alpha_max_fraction_collapsed_test']
                          > ALPHA_COLLAPSE_FRAC_THRESHOLD).sum()) if n_completed else 0
    n_attempted = n_completed + n_failed
    incomplete = n_attempted < expected_n
    # failed/attempted cells with no result are health failures → counted as diverged. Denominator
    # is the full expected grid (never the shrunk completed-row count).
    frac_d = (diverged_done + n_failed) / expected_n
    frac_c = collapsed_done / expected_n
    triggers = []
    if frac_d > CONTINGENCY_CELL_FRACTION:
        triggers.append(f'>{CONTINGENCY_CELL_FRACTION:.0%} diverged ({frac_d:.1%} of {expected_n}; {n_failed} failed)')
    if frac_c > CONTINGENCY_CELL_FRACTION:
        triggers.append(f'>{CONTINGENCY_CELL_FRACTION:.0%} α-collapsed ({frac_c:.1%} of {expected_n})')
    if triggers:
        verdict = 'DEMOTE-EXPLORATORY (move out of confirmatory pairs + SPA M=9→8)'
    elif incomplete:
        verdict = f'KEEP-PENDING (incomplete: {n_attempted}/{expected_n} attempted)'
    else:
        verdict = 'KEEP-CONFIRMATORY'
    return {'verdict': verdict, 'n_cells': n_completed, 'n_failed': n_failed, 'n_expected': expected_n,
            'frac_diverged': frac_d, 'frac_collapsed': frac_c, 'incomplete': incomplete,
            'triggers': triggers}


# ══════════════════════════════════════════════════════════════
# IO
# ══════════════════════════════════════════════════════════════

def init_csv_files():
    if not os.path.exists(RESULTS_CSV):
        pd.DataFrame(columns=L7_RESULTS_COLUMNS).to_csv(RESULTS_CSV, index=False)
    if not os.path.exists(MANIFEST_CSV):
        pd.DataFrame(columns=MANIFEST_COLUMNS).to_csv(MANIFEST_CSV, index=False)


def load_done():
    if not os.path.exists(MANIFEST_CSV):
        return set()
    df = pd.read_csv(MANIFEST_CSV)
    if 'status' not in df.columns or len(df) == 0:
        return set()
    d = df[df['status'] == 'completed']
    return set(zip(d['universe'].astype(str), d['seed'].astype(int), d['fold'].astype(int)))


def append_row(path, row, cols):
    pd.DataFrame([row], columns=cols).to_csv(path, mode='a', header=False, index=False)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--universe', choices=['B', 'C', 'both'], default='both')
    p.add_argument('--seeds', type=str, default=','.join(str(s) for s in CANONICAL_SEEDS))
    p.add_argument('--folds', type=str, default=','.join(str(i) for i in range(N_FOLDS)))
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--smoke-fold', type=int, default=11)
    p.add_argument('--canary', action='store_true')
    p.add_argument('--contingency', action='store_true', help='Evaluate §6 rule on results.csv, exit')
    p.add_argument('--resume', action='store_true', default=True)
    p.add_argument('--no-resume', dest='resume', action='store_false')
    p.add_argument('--frozen-hparams', type=str, default=None,
                   help='Path to frozen_hparams.json → inject per-universe L7 HATS tuned HPs '
                        '(D-RERUN-12F). Absent = HATS pilot defaults (byte-identical untuned L7).')
    p.add_argument('--out-dir', type=str, default=None,
                   help='Override output dir (tuned → experiments/storya_v21_l7_hats_tuned/).')
    args = p.parse_args()

    global OUT_DIR, RESULTS_CSV, MANIFEST_CSV, PER_DAY_IC_DIR, ALPHA_DIAG_DIR
    _pilot_out = OUT_DIR
    if args.out_dir:
        OUT_DIR = args.out_dir.rstrip('/')
        RESULTS_CSV = f'{OUT_DIR}/results.csv'
        MANIFEST_CSV = f'{OUT_DIR}/manifest.csv'
        PER_DAY_IC_DIR = f'{OUT_DIR}/per_day_ic'
        ALPHA_DIAG_DIR = f'{OUT_DIR}/alpha_diag'
    # CODEX-A-01 (fail closed): frozen L7 must NOT write into the untuned pilot dir (resume cell-id clash).
    if args.frozen_hparams and OUT_DIR == _pilot_out:
        raise SystemExit(f'--frozen-hparams requires a NEW --out-dir (not the pilot dir {_pilot_out}); '
                         f'e.g. --out-dir experiments/storya_v21_l7_hats_tuned')

    anchor.setup_workdir()
    for d in [OUT_DIR, PER_DAY_IC_DIR, ALPHA_DIAG_DIR]:
        os.makedirs(d, exist_ok=True)

    if args.contingency:
        if not os.path.exists(RESULTS_CSV):
            print('no results.csv yet'); sys.exit(0)
        man = pd.read_csv(MANIFEST_CSV) if os.path.exists(MANIFEST_CSV) else None
        res = evaluate_contingency(pd.read_csv(RESULTS_CSV), man)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        sys.exit(0)

    device = get_device()
    print(f'Device: {device}')
    assert_cell_id_l7_injective()

    core = load_core_data()
    prices, returns, all_dates = core['prices'], core['returns'], core['all_dates']
    num_days, num_stocks = core['num_days'], core['num_stocks']
    ticker_to_idx = core['ticker_to_id']

    if args.canary:
        ok = run_injection_canary(returns, all_dates, num_days, num_stocks, ticker_to_idx)
        sys.exit(0 if ok else 1)

    labels_np, label_valid_np = build_labels(prices, HORIZON)
    labels_t = torch.tensor(labels_np, dtype=torch.float32)
    label_valid_t = torch.tensor(label_valid_np, dtype=torch.bool)
    corr_snaps, _, sps = build_correlation_snapshots(returns, num_days)
    sector_np = build_sector_edges(anchor.PATHS['sectors'], ticker_to_idx)
    news_df = load_news_edge_source()
    news_snaps, _ = build_per_day_news_edges(news_df, all_dates, ticker_to_idx)  # C1 assert b
    print(f'3 relations ready: corr snapshots={len(sps)}, sector |E|={sector_np.shape[1]}, '
          f'news per-day snapshots={len(news_snaps)} (C1 assert b PIT-checked)')

    if args.smoke:
        universes_run, seeds_run, folds_run = ['B'], [CANONICAL_SEEDS[0]], [args.smoke_fold]
        print(f'\n=== L7 SMOKE: fold {args.smoke_fold} × seed {seeds_run[0]} (Univ B) ===\n')
    else:
        universes_run = ALL_UNIVERSES if args.universe == 'both' else [args.universe]
        seeds_run = [int(s) for s in args.seeds.split(',') if s]
        folds_run = [int(f) for f in args.folds.split(',') if f != '']
        for s in seeds_run:
            assert s in CANONICAL_SEEDS, f'seed {s} not canonical'

    init_csv_files()
    seed_idx_map = {s: i for i, s in enumerate(CANONICAL_SEEDS)}
    u_idx_map = {'B': 0, 'C': 1}
    done = load_done() if args.resume else set()

    features_raw = {}
    if 'B' in universes_run:
        features_raw['B'], _ = build_universe_B(prices, returns)
    if 'C' in universes_run:
        fC, _ = build_universe_C(prices, returns)
        assert np.all(fC[0] == 0.0), "Univ-C T-1 contract broken (row0)"
        features_raw['C'] = fC

    frozen = load_frozen_hparams(args.frozen_hparams) if args.frozen_hparams else None
    if frozen:
        # CODEX-A-02: L7 now persists provenance too (merge + md5 validate, like main12).
        fz_md5 = hashlib.md5(open(args.frozen_hparams, 'rb').read()).hexdigest()
        prov_path = f'{OUT_DIR}/_frozen_hp_provenance.json'
        if os.path.exists(prov_path):
            prov = json.load(open(prov_path))
            if prov.get('frozen_md5') != fz_md5:
                raise SystemExit(f'{prov_path} md5={prov.get("frozen_md5","?")[:8]} != this run {fz_md5[:8]}; '
                                 f'refuse to mix frozen-HP files in one dir')
        else:
            prov = {'mode': 'L7 TUNED per-universe', 'frozen_hparams': args.frozen_hparams,
                    'frozen_md5': fz_md5, 'applied': {}}
        print(f'Frozen-HP injection [L7 TUNED per-universe] from {args.frozen_hparams} (md5 {fz_md5[:8]}):')
        for u in universes_run:
            wp = frozen[f'{u}_L7']['winner_params']
            prov['applied'][f'{u}_L7'] = {'src': f'{u}_L7', 'params': wp}
            print(f'  U{u} L7 ← {u}_L7: {wp}')
        with open(prov_path, 'w') as f:
            json.dump(prov, f, indent=2)
    else:
        print('Frozen-HP injection: OFF (HATS pilot defaults — untuned L7 baseline)')

    t0 = time.time()
    for universe in universes_run:
        if frozen:   # inject this universe's L7 HATS tuned HPs (patch hats.HATS_HPARAMS module global)
            hats.HATS_HPARAMS = {**_ORIG_HATS, **frozen[f'{universe}_L7']['winner_params']}
        feats_raw = features_raw[universe]
        u_idx = u_idx_map[universe]
        for fold in WALK_FORWARD_FOLDS_12:
            if fold['id'] not in folds_run:
                continue
            tr, va, te = create_fold_masks(fold, all_dates, HORIZON)
            fsi = get_frozen_snapshot_idx(tr[-1], sps)
            corr_np = corr_snaps[fsi].cpu().numpy()
            three = build_three_relation_edges_per_fold(corr_np, sector_np, news_snaps, tr, va, te)
            feats_std = standardize_train_only(winsorize_train_only(feats_raw, tr), tr)
            feats_std_t = torch.tensor(feats_std, dtype=torch.float32)
            print(f'\n[U{universe}] fold {fold["id"]} ({fold["desc"]}): train={len(tr)}d val={len(va)}d test={len(te)}d')

            for seed in seeds_run:
                s_idx = seed_idx_map[seed]
                cid = cell_id_l7(u_idx, fold['id'], s_idx)
                key = (universe, int(seed), int(fold['id']))
                if key in done:
                    continue
                print(f'  [cid={cid:04d}] U{universe} L7/HATS seed={seed} fold={fold["id"]} ...', end=' ', flush=True)
                start_ts = time.time()
                status, err = 'running', ''
                try:
                    alpha_log = f'{ALPHA_DIAG_DIR}/{universe}_s{seed}_f{fold["id"]}.csv'
                    preds, info = train_hats(feats_std_t, labels_t, label_valid_t, tr, va, te,
                                             three, num_days, num_stocks, seed, device, alpha_log)
                    ic_arr = compute_daily_ic(preds, te, labels_np, label_valid_np)
                    ic_mean = float(ic_arr.mean()) if len(ic_arr) else float('nan')
                    ic_std = float(ic_arr.std()) if len(ic_arr) else float('nan')
                    sh = compute_cost_ladder_sharpe(preds, te, prices, label_valid_np, num_stocks, num_days, horizon=HORIZON)
                    np.save(f'{PER_DAY_IC_DIR}/{universe}_L7_s{seed}_f{fold["id"]}.npy', ic_arr)
                    # diverged := genuine HEALTH failure only (Cn5 is a health gate, NOT performance).
                    # CODEX-C-01 fix: dropped the `epochs_run >= max` proxy — it is BACKWARDS under
                    # patience-based early-stopping. A run that reaches max epochs did so because val
                    # kept improving within the last `patience` epochs (else early-stop fires), i.e.
                    # epochs_run>=max marks a LATE-IMPROVER, not divergence; a truly non-learning run
                    # early-stops near `patience`. So divergence = no usable output (no valid eval
                    # days / IC NaN) OR a non-finite best val loss (numerical blow-up). epochs_run is
                    # still recorded as a column for inspection.
                    diverged = int((len(ic_arr) == 0) or np.isnan(ic_mean)
                                   or (not np.isfinite(info['best_val_loss'])))
                    wall = time.time() - start_ts
                    row = {
                        'cell_id': cid, 'universe': universe, 'arm': ARM, 'model': 'HATS-3R-adapt',
                        'seed': seed, 'fold': fold['id'], 'test_period': fold['desc'],
                        'IC_mean': round(ic_mean, 6) if not np.isnan(ic_mean) else float('nan'),
                        'IC_std': round(ic_std, 6) if not np.isnan(ic_std) else float('nan'),
                        'n_test_days': len(ic_arr), 'Sharpe_gross': round(sh['Sharpe_gross'], 4),
                        **{k: round(v, 4) for k, v in sh.items() if k.startswith('Sharpe_net_')},
                        'mean_turnover_L1': round(sh['mean_turnover_L1'], 4), 'n_periods': sh['n_periods'],
                        'best_val_loss': round(info['best_val_loss'], 6), 'epochs_run': info['epochs_run'],
                        'alpha_mean_corr_test': round(info['alpha_mean_corr_test'], 5),
                        'alpha_mean_sector_test': round(info['alpha_mean_sector_test'], 5),
                        'alpha_mean_news_test': round(info['alpha_mean_news_test'], 5),
                        'alpha_max_fraction_collapsed_test': round(info['alpha_max_fraction_collapsed_test'], 5),
                        'diverged_flag': diverged,
                        'wall_time_sec': round(wall, 1),
                        'converged_flag': int(not diverged), 'cost_convention': COST_CONVENTION,
                    }
                    append_row(RESULTS_CSV, row, L7_RESULTS_COLUMNS)
                    status = 'completed'
                    print(f'IC={ic_mean:+.5f} α_collapse={info["alpha_max_fraction_collapsed_test"]:.3f} '
                          f'diverged={diverged} ({wall:.0f}s)')
                except Exception as e:
                    status, err = 'failed', str(e)[:200]
                    import traceback; traceback.print_exc()
                    print(f'  FAILED: {err}')
                end_ts = time.time()
                append_row(MANIFEST_CSV, {
                    'cell_id': cid, 'universe': universe, 'arm': ARM, 'seed': seed, 'fold': fold['id'],
                    'status': status, 'start_ts': round(start_ts, 1), 'end_ts': round(end_ts, 1),
                    'wall_time_sec': round(end_ts - start_ts, 1), 'err': err,
                }, MANIFEST_COLUMNS)
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            del feats_std, feats_std_t
            gc.collect()

    # §6 contingency snapshot on whatever has run so far (informational; full 240-grid decides)
    if os.path.exists(RESULTS_CSV):
        man = pd.read_csv(MANIFEST_CSV) if os.path.exists(MANIFEST_CSV) else None
        res = evaluate_contingency(pd.read_csv(RESULTS_CSV), man)
        print(f'\n[§6 contingency: {res["n_cells"]} done / {res.get("n_failed",0)} failed / {res["n_expected"]} expected] '
              f'{res["verdict"]} | diverged={res.get("frac_diverged", 0):.1%} '
              f'collapsed={res.get("frac_collapsed", 0):.1%} | triggers={res["triggers"] or "none"}')
    print(f'\nTotal wall: {(time.time()-t0)/3600:.2f}h')
    print('=== run_storya_v21_l7_hats.py DONE ===')


if __name__ == '__main__':
    main()
