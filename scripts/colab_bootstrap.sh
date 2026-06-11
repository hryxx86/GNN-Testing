#!/usr/bin/env bash
# Colab per-runtime bootstrap — realizes the project architecture "Code = GitHub, Data = Drive".
#
# WHY this exists (root cause 2026-06-10):
#   The Drive mount `/content/drive/MyDrive/GNN测试` is NOT a git repo (no `.git`), so the old
#   `cd <drive> && git pull` step always failed silently. Code reached Colab only via flaky
#   Google-Drive sync, and `scripts/` wasn't even covered by `sync_to_drive.sh`.
#
# WHAT this does instead:
#   1. CODE  → `git clone`/`git pull` into Colab LOCAL disk (`/content/GNN-Testing`), a real git
#              working tree where `git pull` actually works and code is always GitHub-current.
#   2. DATA  → symlink the fully-.gitignored input/output dirs from Drive into the repo, so input
#              data is found and outputs survive runtime death (Colab local disk is ephemeral).
#   For `experiments/` (mixed: git-tracked config + gitignored outputs) the git config is rsynced
#   INTO Drive first (preserving Drive's outputs), then symlinked — git stays config source-of-truth.
#   `artifacts/` is intentionally NOT symlinked: it is git-managed (reviews + allowlisted outputs).
#
# USAGE (Colab cell, AFTER drive.mount):
#   !curl -sSL https://raw.githubusercontent.com/hryxx86/GNN-Testing/main/scripts/colab_bootstrap.sh | bash
#   %cd /content/GNN-Testing
#
# Override paths via env: REPO=... DRIVE=... GIT_URL=...

set -e
REPO="${REPO:-/content/GNN-Testing}"
DRIVE="${DRIVE:-/content/drive/MyDrive/GNN测试}"
GIT_URL="${GIT_URL:-https://github.com/hryxx86/GNN-Testing.git}"

if [[ ! -d "$DRIVE" ]]; then
    echo "ERROR: Drive folder not found: $DRIVE"
    echo "       Run drive.mount('/content/drive') first (Cell 1)."
    exit 1
fi

# ─── 1. Code: clone fresh, or fast-forward pull if already cloned ───
if [[ -d "$REPO/.git" ]]; then
    echo "[bootstrap] repo exists → git pull --ff-only"
    git -C "$REPO" pull --ff-only
else
    echo "[bootstrap] cloning $GIT_URL → $REPO"
    git clone "$GIT_URL" "$REPO"
fi

# ─── 2. experiments/ : push git config to Drive (keep Drive outputs), then symlink ───
# rsync from the clone copies ONLY git-tracked config (prereg.json / hp_grid.json / READMEs —
# the clone has no large outputs, they're .gitignored). No --delete, so Drive outputs are kept.
mkdir -p "$DRIVE/experiments"
rsync -a "$REPO/experiments/" "$DRIVE/experiments/" 2>/dev/null || true
rm -rf "$REPO/experiments"
ln -sfn "$DRIVE/experiments" "$REPO/experiments"
echo "[bootstrap] experiments/ → Drive (git config seeded, outputs preserved)"

# ─── 3. Pure data/output dirs (clone has none of these — all .gitignored): plain symlink ───
for d in data plots wandb; do
    mkdir -p "$DRIVE/$d"
    rm -rf "$REPO/$d"
    ln -sfn "$DRIVE/$d" "$REPO/$d"
    echo "[bootstrap] $d/ → Drive"
done

echo ""
echo "===================================================="
echo "✓ bootstrap done: code @ $REPO (git), data @ Drive (symlinked)"
echo "  Next cell:  %cd $REPO"
echo "  Then SSH:   !bash scripts/colab_ssh_tunnel.sh"
echo "===================================================="
