#!/usr/bin/env python
"""
Launcher for the v2.2 §4 equal-budget tuning — STUDY-LEVEL parallelism (cross-machine + intra-GPU).

Each study (one arm × one universe) is an independent, internally-sequential, deterministic Optuna
run (run_storya_v21_tune.py). This launcher keeps K such study-processes running concurrently,
pulling the next study from a queue as slots free (handles uneven per-study cost). Concurrency
across studies fills the GPU WITHOUT perturbing per-study TPE determinism (we never parallelize
trials inside one study).

WHY study-level (not Optuna n_jobs): concurrent trials inside one study make TPE suggest before
in-flight trials report → non-reproducible search. 20 studies give plenty to parallelize across.

CROSS-MACHINE SPLIT (disjoint; union = all 20 = 10 arms × {B,C}):
  --machine mac : LGB (L0) on CPU + the lighter MLP/SAGE arms on MPS         (default concurrency 2)
  --machine t4  : the GPU-heavy GAT/L6/L7 arms                                (default concurrency 4)
  --machine all : every study (single-box run)
  --studies "C:L2,B:L6,..." : explicit override

On the T4 box, START THIS LAUNCHER with the CUDA libs visible so child workers inherit them:
  LD_LIBRARY_PATH=/usr/lib64-nvidia python run_v21_tune_launcher.py --machine t4 --concurrency 4

After both machines finish, merge:  python run_v21_tune_launcher.py --merge
"""

import os
import sys
import time
import json
import glob
import argparse
import subprocess

OUT_DIR = 'experiments/storya_v21_tune'
TUNE_SCRIPT = 'run_storya_v21_tune.py'

# study split (disjoint; union = all 20). LGB is CPU; MLP/SAGE lighter; GAT/L6/L7 GPU-heavy.
MAC_ARMS = ['L0', 'L1', 'L2s', 'L5s']        # 4 arms × 2 univ = 8 studies (LGB CPU + MLP/SAGE MPS)
T4_ARMS = ['L2', 'L3', 'L4', 'L5', 'L6', 'L7']  # 6 arms × 2 univ = 12 studies (GAT/L6/L7 GPU)


def studies_for(machine: str):
    if machine == 'mac':
        arms = MAC_ARMS
    elif machine == 't4':
        arms = T4_ARMS
    elif machine == 'all':
        arms = MAC_ARMS + T4_ARMS
    else:
        raise ValueError(machine)
    return [(u, a) for u in ['B', 'C'] for a in arms]


def parse_studies(s: str):
    out = []
    for tok in s.split(','):
        tok = tok.strip()
        if not tok:
            continue
        u, a = tok.split(':')
        out.append((u, a))
    return out


def run_pool(studies, concurrency: int, python_exe: str, n_trials, top_k):
    """Keep `concurrency` study-processes running; pull from queue as slots free."""
    queue = list(studies)
    running = {}   # popen -> (univ, arm, start_ts)
    results = {}   # (univ,arm) -> returncode
    total = len(queue)
    print(f'[launcher] {total} studies, concurrency={concurrency}, python={python_exe}')

    def launch(u, a):
        log = f'{OUT_DIR}/log_{u}_{a}.txt'
        cmd = [python_exe, TUNE_SCRIPT, '--arm', a, '--universe', u,
               '--n-trials', str(n_trials), '--top-k', str(top_k)]
        f = open(log, 'w')
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        running[p] = (u, a, time.time(), f)
        print(f'[launcher] ▶ start {u}:{a}  (running {len(running)}/{concurrency})  → {log}')

    while queue and len(running) < concurrency:
        u, a = queue.pop(0)
        launch(u, a)

    while running:
        time.sleep(3)
        for p in list(running):
            rc = p.poll()
            if rc is None:
                continue
            u, a, ts, f = running.pop(p)
            f.close()
            results[(u, a)] = rc
            dur = time.time() - ts
            status = 'OK' if rc == 0 else f'FAIL(rc={rc})'
            done = len(results)
            print(f'[launcher] ■ {status} {u}:{a}  ({dur:.0f}s)  [{done}/{total}]')
            if queue:
                nu, na = queue.pop(0)
                launch(nu, na)

    ok = sum(1 for rc in results.values() if rc == 0)
    print(f'[launcher] DONE {ok}/{total} OK')
    fails = [k for k, rc in results.items() if rc != 0]
    if fails:
        print(f'[launcher] FAILED: {fails}  (see {OUT_DIR}/log_*.txt; re-run those, --resume via study db)')
    return results


def merge(allow_partial: bool = False):
    """Collect per-study {univ}_{arm}.json into frozen_hparams.json (winner per arm×universe).

    CODEX-A-01 (fail closed): unless all 20 studies are present (or --allow-partial), do NOT write
    frozen_hparams.json and exit nonzero — downstream must never silently consume a partial main-table
    HP set (e.g. an L7 crash → 18/20). --allow-partial emits frozen_hparams.PARTIAL.json (complete:false).
    CODEX-A-02: smoke artifacts (_smoke_*.json or smoke==True) are excluded."""
    rows = {}
    skipped_smoke = []
    for fp in sorted(glob.glob(f'{OUT_DIR}/*_*.json')):
        base = os.path.basename(fp)
        if base in ('frozen_hparams.json', 'frozen_hparams.PARTIAL.json'):
            continue
        if base.startswith('_smoke'):
            skipped_smoke.append(base); continue
        r = json.load(open(fp))
        if r.get('smoke'):                       # belt-and-suspenders vs a mislabeled smoke file
            skipped_smoke.append(base); continue
        key = f"{r['universe']}_{r['arm']}"
        rows[key] = {
            'universe': r['universe'], 'arm': r['arm'], 'model': r['model'],
            'winner_params': r['winner_params'],
            'winner_mean_val_ic_3seed': r['winner_mean_val_ic_3seed'],
            'n_trials': r['n_trials'],
        }
    if skipped_smoke:
        print(f'[merge] skipped {len(skipped_smoke)} smoke JSON(s): {skipped_smoke}')
    n = len(rows)
    missing = [f'{u}_{a}' for u in ['B', 'C']
               for a in (MAC_ARMS + T4_ARMS) if f'{u}_{a}' not in rows]
    if n != 20 and not allow_partial:
        print(f'[merge] ERROR incomplete: {n}/20 studies present; missing {missing}. '
              f'NOT writing frozen_hparams.json — re-run the missing studies, or pass --allow-partial '
              f'to emit a clearly-marked partial artifact.')
        sys.exit(1)
    frozen = {
        'source': 'run_storya_v21_tune.py per-study outputs',
        'search_space_ref': 'docs/protocol_v2_freeze.md v2.2 §4',
        'n_studies': n, 'expected': 20, 'complete': n == 20, 'missing': missing,
        'studies': rows,
    }
    out = f'{OUT_DIR}/frozen_hparams.json' if n == 20 else f'{OUT_DIR}/frozen_hparams.PARTIAL.json'
    with open(out, 'w') as f:
        json.dump(frozen, f, indent=2)
    print(f'[merge] {n}/20 studies → {out}' + ('' if n == 20 else f'  (PARTIAL; missing {missing})'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--machine', choices=['mac', 't4', 'all'])
    ap.add_argument('--studies', type=str, help='explicit "U:arm,..." override')
    ap.add_argument('--concurrency', type=int, default=None)
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--n-trials', type=int, default=30)
    ap.add_argument('--top-k', type=int, default=5)
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--allow-partial', action='store_true',
                    help='merge: emit frozen_hparams.PARTIAL.json instead of failing closed on <20 studies')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.merge:
        merge(allow_partial=args.allow_partial)
        return
    if args.studies:
        studies = parse_studies(args.studies)
    elif args.machine:
        studies = studies_for(args.machine)
    else:
        ap.error('need --machine {mac|t4|all} or --studies or --merge')
    concurrency = args.concurrency or ({'mac': 2, 't4': 4, 'all': 4}.get(args.machine, 2))
    results = run_pool(studies, concurrency, args.python, args.n_trials, args.top_k)
    n_fail = sum(1 for rc in results.values() if rc != 0)
    if n_fail:                                   # CODEX-A-01: surface failures as a nonzero exit
        print(f'[launcher] {n_fail} study(ies) failed — exit 1 (re-run them before --merge)')
        sys.exit(1)


if __name__ == '__main__':
    main()
