#!/usr/bin/env python
"""Recycle-resilience for the v2.2 §4 TUNING on Colab T4.

WHY: the Optuna sqlite study DBs must live on LOCAL /content disk — a Google Drive FUSE mount
cannot host sqlite ("unable to open database file"; no POSIX locking). But /content is WIPED on a
Colab runtime recycle, and the optuna resume (load_if_exists) needs the DB. Without a backup, a
recycle loses every in-progress study back to trial 0 (this bit us 2026-06-16). This script makes
the run recycle-resilient:

  --backup   loop: every INTERVAL s, snapshot each LOCAL db → DRIVE backup dir. The snapshot is a
             CONSISTENT online backup (sqlite .backup API) to a LOCAL temp, then a plain byte-copy
             to Drive — we NEVER open sqlite directly on the FUSE mount. Safe to run while the
             launcher is writing the dbs. Run it in its own tmux session alongside the launcher.
  --restore  one-shot: copy DRIVE backups → LOCAL. Run this BEFORE the launcher on a fresh runtime;
             the launcher's load_if_exists then resumes each study from its last-backed-up trial.

Completed studies already persist their winner JSON to OUT_DIR (Drive); this only protects the
IN-PROGRESS study DBs. Worst-case loss on a recycle = trials since the last snapshot (≤ INTERVAL).

Paths (env, with defaults matching run_storya_v21_tune.py / the T4 launch):
  V21_TUNE_STUDY_DIR     local sqlite dir   (default /content/v21_tune_studies)
  V21_TUNE_DRIVE_BACKUP  Drive backup dir   (default experiments/storya_v21_tune/studies_backup)
"""
import os
import glob
import time
import shutil
import sqlite3
import argparse

LOCAL = os.environ.get('V21_TUNE_STUDY_DIR', '/content/v21_tune_studies')
DRIVE = os.environ.get('V21_TUNE_DRIVE_BACKUP', 'experiments/storya_v21_tune/studies_backup')


def snapshot_one(db: str, dst_dir: str) -> None:
    """Consistent online backup of one sqlite db → a LOCAL temp, then byte-copy to Drive.
    Avoids opening sqlite on the FUSE mount and avoids copying a db mid-transaction."""
    tmp = db + '.snap'
    src = sqlite3.connect(db)
    dst = sqlite3.connect(tmp)
    try:
        src.backup(dst)          # online backup API — consistent even while the writer is active
    finally:
        dst.close()
        src.close()
    shutil.copy(tmp, os.path.join(dst_dir, os.path.basename(db)))
    os.remove(tmp)


def backup_loop(interval: int) -> None:
    os.makedirs(DRIVE, exist_ok=True)
    print(f'[dbsync] backup loop: {LOCAL}/*.db → {DRIVE} every {interval}s', flush=True)
    while True:
        n, errs = 0, 0
        for db in glob.glob(f'{LOCAL}/*.db'):
            try:
                snapshot_one(db, DRIVE)
                n += 1
            except Exception as e:
                errs += 1
                print(f'[dbsync] skip {os.path.basename(db)}: {type(e).__name__}: {e}', flush=True)
        print(f'[dbsync] backed up {n} dbs ({errs} skipped)', flush=True)
        time.sleep(interval)


def restore() -> None:
    if not os.path.isdir(DRIVE):
        print(f'[dbsync] no Drive backup dir {DRIVE}; nothing to restore', flush=True)
        return
    os.makedirs(LOCAL, exist_ok=True)
    n = 0
    for db in glob.glob(f'{DRIVE}/*.db'):
        shutil.copy(db, os.path.join(LOCAL, os.path.basename(db)))
        n += 1
    print(f'[dbsync] restored {n} dbs → {LOCAL}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--backup', action='store_true', help='periodic local→Drive snapshot loop')
    ap.add_argument('--restore', action='store_true', help='one-shot Drive→local copy (before launch)')
    ap.add_argument('--interval', type=int, default=180)
    a = ap.parse_args()
    if a.restore:
        restore()
    if a.backup:
        backup_loop(a.interval)
    if not (a.restore or a.backup):
        ap.error('need --backup or --restore')
