#!/bin/bash
# Pull Story A v3 experiment outputs from Google Drive Desktop mount → local.
# Mirrors the reverse direction of scripts/sync_to_drive.sh.
#
# WHY: experiments/** is .gitignored (large outputs live on Drive, not GitHub).
# Local has only smoke cells after Colab runs finish. Paper-figure scripts need
# the full per_day_ic .npy arrays + full results.csv, so we pull once + iterate
# fig scripts locally without burning Colab Pro hours.
#
# WHAT IS PULLED (Story A v3 confirmatory experiments only):
#   experiments/storya_e1_anchor/{per_day_ic/, results.csv, manifest.csv, _meta.json}
#   experiments/storya_e3_news_edge/{per_day_ic/, results.csv, manifest.csv, _meta.json}
#   experiments/storya_e4_alpha/{per_day_ic/, results.csv, manifest.csv, _meta.json}
#
# Total ~50MB across 550 .npy files + 3 results.csv + manifests.
#
# Usage: bash scripts/sync_storya_from_drive.sh [--dry-run]

set -euo pipefail

DRIVE_BASE="/Users/heruixi/Library/CloudStorage/GoogleDrive-hryxx86@gmail.com/我的云端硬盘/GNN测试"
LOCAL_BASE="/Users/heruixi/Desktop/GNN-Testing"

if [[ ! -d "$DRIVE_BASE" ]]; then
  echo "[error] Drive mount not found: $DRIVE_BASE" >&2
  echo "[hint] Open Google Drive Desktop app and verify 'GNN测试' folder is synced locally." >&2
  exit 1
fi

DRY=""
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY="--dry-run"
  echo "[sync] DRY-RUN — no files written"
fi

# rsync flags:
#   -a  archive (preserve mtime/perms)
#   -v  verbose (lists transferred files)
#   -h  human-readable sizes
#   -c  checksum-based diff (Drive rewrites mtime, so mtime+size diff would skip stale files)
# Direction: Drive → Local (one-way; Colab is source of truth for results)
RSYNC="rsync -avhc $DRY"

EXPERIMENTS=("storya_e1_anchor" "storya_e3_news_edge" "storya_e4_alpha")

for EXP in "${EXPERIMENTS[@]}"; do
  SRC="$DRIVE_BASE/experiments/$EXP"
  DST="$LOCAL_BASE/experiments/$EXP"

  if [[ ! -d "$SRC" ]]; then
    echo "[skip] $EXP not on Drive: $SRC"
    continue
  fi

  echo
  echo "════════════════════════════════════════════════════════════════════"
  echo "Pulling: $EXP"
  echo "  from: $SRC"
  echo "  to:   $DST"
  echo "════════════════════════════════════════════════════════════════════"

  mkdir -p "$DST/per_day_ic"

  # Per-day IC .npy directory (the bulk; ~500B-1KB per file)
  if [[ -d "$SRC/per_day_ic" ]]; then
    $RSYNC --include='*.npy' --exclude='*' "$SRC/per_day_ic/" "$DST/per_day_ic/"
  fi

  # Top-level result files
  for f in results.csv manifest.csv _meta.json hp_grid.json smoke_benchmark.csv news_edge_source_schema.md; do
    if [[ -f "$SRC/$f" ]]; then
      $RSYNC "$SRC/$f" "$DST/$f"
    fi
  done
done

if [[ -n "$DRY" ]]; then
  echo
  echo "[sync] dry-run done; skipping post-sync verification."
  exit 0
fi

# ── Post-sync verification ──────────────────────────────────────────────────
echo
echo "════════════════════════════════════════════════════════════════════"
echo "Post-sync verification"
echo "════════════════════════════════════════════════════════════════════"

VERIFY_FAIL=0
EXPECTED_COUNTS=(400 50 100)   # E1 / E3 / E4 .npy counts
for i in "${!EXPERIMENTS[@]}"; do
  EXP="${EXPERIMENTS[$i]}"
  EXPECTED="${EXPECTED_COUNTS[$i]}"
  LOCAL_DIR="$LOCAL_BASE/experiments/$EXP/per_day_ic"

  if [[ ! -d "$LOCAL_DIR" ]]; then
    echo "[VERIFY-FAIL] $EXP: per_day_ic dir missing locally: $LOCAL_DIR"
    VERIFY_FAIL=1
    continue
  fi

  ACTUAL=$(ls "$LOCAL_DIR"/*.npy 2>/dev/null | wc -l | tr -d ' ')
  if [[ "$ACTUAL" -ne "$EXPECTED" ]]; then
    echo "[VERIFY-FAIL] $EXP: expected $EXPECTED .npy, got $ACTUAL"
    VERIFY_FAIL=1
  else
    echo "[ok] $EXP: $ACTUAL .npy files synced (matches expected $EXPECTED)"
  fi

  # results.csv row count check (header + N data rows)
  CSV="$LOCAL_BASE/experiments/$EXP/results.csv"
  if [[ -f "$CSV" ]]; then
    ROWS=$(wc -l < "$CSV" | tr -d ' ')
    echo "[ok] $EXP: results.csv has $ROWS lines (1 header + $((ROWS - 1)) data rows)"
  else
    echo "[VERIFY-FAIL] $EXP: results.csv missing locally"
    VERIFY_FAIL=1
  fi
done

if [[ "$VERIFY_FAIL" -ne 0 ]]; then
  echo
  echo "[sync] FAILED — some files missing or count mismatch." >&2
  exit 1
fi

# Total disk usage report
TOTAL_KB=0
for EXP in "${EXPERIMENTS[@]}"; do
  KB=$(du -sk "$LOCAL_BASE/experiments/$EXP" 2>/dev/null | awk '{print $1}')
  TOTAL_KB=$((TOTAL_KB + KB))
done
echo
echo "[sync] done. Total disk usage: ~$((TOTAL_KB / 1024)) MB across 3 experiments."
echo "[next] Paper-figure scripts (paper_figs/fig_*.py) can now read per_day_ic .npy locally."
