"""Paper-evaluation robustness checks (2026-07-02) — zero-rerun, existing data only.

Four checks supporting the paper evaluation (docs/paper_evaluation_2026-07-02.md),
each targeting an open PaperJury major on paper/main.tex:

  1. PER-SEED SIGN       (I-14): per-seed pooled ΔIC sign agreement for the six
     BH-rejected contrasts. Per-seed pooled IC = n_test_days-weighted mean of
     fold-level IC_mean, which reconstructs the pooled daily mean per seed.
  2. LEAVE-ONE-SEED-OUT  (I-14): drop each of the 10 seeds, recompute pooled ΔIC,
     count sign flips (mirrors the LOFO fold check already in the paper).
  3. POOLED 26-TEST BH   (I-07/I-19): single BH family over 20 DM/HLN + 6 Family-2
     tests; compare rejection decisions vs the pre-registered separate families.
  4. BY SENSITIVITY      (I-37): Benjamini-Yekutieli (arbitrary dependence) over the
     20-test DM family; which BH rejections survive the conservative correction.

Inputs (read-only):
  experiments/storya_v21_main12_tuned/results.csv
  artifacts/storya_v21_family1/family1_dm_hln.csv
  artifacts/storya_v21_family2_fc/family2_fc_causal.csv

Output:
  artifacts/audits/paper_eval_robustness.csv  (one row per check result)
  console summary
"""

import os

import numpy as np
import pandas as pd

RESULTS = 'experiments/storya_v21_main12_tuned/results.csv'
DM_CSV = 'artifacts/storya_v21_family1/family1_dm_hln.csv'
FC_CSV = 'artifacts/storya_v21_family2_fc/family2_fc_causal.csv'
OUT = 'artifacts/audits/paper_eval_robustness.csv'

# The six BH-rejected (universe, arm_A, arm_B) ladder/edge contrasts (family1_dm_hln.csv
# BH_FDR_reject_family=True rows, restricted to the four pairs named in paper §5.3 LOFO).
BH_REJECTED = [('C', 'L1', 'L0'), ('B', 'L2', 'L1'), ('C', 'L2', 'L1'),
               ('B', 'L3', 'L2'), ('C', 'L3', 'L2'), ('C', 'L5', 'L3')]
Q = 0.05


def seed_pooled(df: pd.DataFrame, u: str, arm: str) -> dict:
    """{seed: pooled IC} — n_test_days-weighted mean of fold IC_mean per seed."""
    sub = df[(df.universe == u) & (df.arm == arm)]
    return {s: float(np.average(g['IC_mean'].values, weights=g['n_test_days'].values))
            for s, g in sub.groupby('seed')}


def bh_reject(pvals: np.ndarray, q: float) -> np.ndarray:
    m = len(pvals)
    order = np.argsort(pvals)
    ok = pvals[order] <= (np.arange(1, m + 1) / m * q)
    if not ok.any():
        return np.zeros(m, bool)
    return pvals <= pvals[order][np.where(ok)[0].max()]


def main() -> int:
    df = pd.read_csv(RESULTS)
    dm = pd.read_csv(DM_CSV)
    fc = pd.read_csv(FC_CSV)
    rows = []

    # ── Check 1 + 2: per-seed sign agreement and leave-one-seed-out ──
    print('=== Check 1+2: per-seed sign / leave-one-seed-out (I-14) ===')
    for u, a, b in BH_REJECTED:
        pa, pb = seed_pooled(df, u, a), seed_pooled(df, u, b)
        seeds = sorted(set(pa) & set(pb))
        d = np.array([pa[s] - pb[s] for s in seeds])
        full = d.mean()
        same = int((np.sign(d) == np.sign(full)).sum())
        loso = np.array([np.delete(d, i).mean() for i in range(len(d))])
        flips = int((np.sign(loso) != np.sign(full)).sum())
        print(f'  {u} {a}-{b}: pooled={full:+.5f}  per-seed same-sign {same}/{len(seeds)}  '
              f'LOSO flips {flips}/{len(seeds)}')
        rows.append({'check': 'per_seed_sign', 'universe': u, 'contrast': f'{a}-{b}',
                     'pooled_delta_IC': round(full, 5), 'n_seeds': len(seeds),
                     'n_same_sign': same, 'loso_sign_flips': flips,
                     'per_seed_min': round(float(d.min()), 5),
                     'per_seed_max': round(float(d.max()), 5)})

    # ── Check 3: pooled 26-test BH (I-07 / I-19) ──
    p_dm = dm['HLN_p_t'].to_numpy()
    lab_dm = [f'{r.universe} {r.arm_A}-{r.arm_B}' for r in dm.itertuples()]
    p_fc = fc['t_p_two_sided'].to_numpy()
    lab_fc = [f'{r.universe} {r.contrast}' for r in fc.itertuples()]
    pooled = np.concatenate([p_dm, p_fc])
    labels = lab_dm + lab_fc
    rej = bh_reject(pooled, Q)
    pooled_set = {l for l, r in zip(labels, rej) if r}
    orig_set = {f'{r.universe} {r.arm_A}-{r.arm_B}' for r in dm[dm.BH_FDR_reject_family].itertuples()}
    identical = (pooled_set & set(lab_dm) == orig_set) and not (pooled_set & set(lab_fc))
    print(f'\n=== Check 3: pooled BH over {len(pooled)} tests (I-07/I-19) ===')
    print(f'  decisions identical to separate pre-registered families: {identical}')
    rows.append({'check': 'pooled_bh_26', 'universe': 'both', 'contrast': 'all',
                 'n_tests': len(pooled), 'n_reject': int(rej.sum()),
                 'identical_to_preregistered': bool(identical)})

    # ── Check 4: BY over the 20 DM tests (I-37) ──
    m = len(p_dm)
    c_m = float(np.sum(1.0 / np.arange(1, m + 1)))
    rej_by = bh_reject(p_dm, Q / c_m)
    by_set = {l for l, r in zip(lab_dm, rej_by) if r}
    print(f'\n=== Check 4: BY (arbitrary dependence) over 20 DM tests (I-37) ===')
    print(f'  BY survivors ({len(by_set)}): {sorted(by_set)}')
    print(f'  BH-only (dropped under BY): {sorted(orig_set - by_set)}')
    for l in lab_dm:
        rows.append({'check': 'by_20', 'universe': l.split()[0], 'contrast': l.split()[1],
                     'bh_reject': l in orig_set, 'by_reject': l in by_set})

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f'\nwrote {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
