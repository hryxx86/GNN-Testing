#!/usr/bin/env bash
# One-line launcher for Colab GPU training jobs via SSH.
#
# IMPORTANT (2026-05-26 update): Code lives on GitHub; DATA lives in Google Drive.
# The project root on Colab is the Drive mount, NOT a fresh GitHub clone:
#
#   /content/drive/MyDrive/GNN测试   ← project root (Drive mount, also a git working tree)
#
# Usage (run from project root on Colab side):
#   bash scripts/colab_launch.sh <run_script.py> [run_args...]
#
# Examples:
#   bash scripts/colab_launch.sh run_storya_e1_anchor.py
#   bash scripts/colab_launch.sh run_tier1_phase_a_wandb.py --mode smoke
#
# Or remote-trigger from local Mac (cloudflared SSH):
#   sshpass -p "GNNTEST" ssh <host>.trycloudflare.com \
#     "cd '/content/drive/MyDrive/GNN测试' && git pull && \
#      bash scripts/colab_launch.sh run_storya_e1_anchor.py"
#
# What it does:
#   0. Sanity-checks cwd is the project root (must contain data/reference/sp500_5y_prices.csv)
#   1. Verifies tmux + python + wandb are available (installs if missing)
#   2. Optionally logs into wandb if WANDB_API_KEY is set in environment
#   3. Creates artifacts/colab_runs/ log directory
#   4. Starts a tmux session named 'train' running the python script
#   5. Detaches; training continues even after SSH disconnect
#
# To watch / attach back:
#   tmux attach -t train          (Ctrl+B then D to detach again)
#   tail -f artifacts/colab_runs/<latest>.log

set -e

# ─── Step 0: cwd sanity check ───
# Fail fast if invoked from the wrong directory (e.g. a bare GitHub clone in ~ that has no data/).
SENTINEL="data/reference/sp500_5y_prices.csv"
if [[ ! -f "$SENTINEL" ]]; then
    echo "ERROR: cwd=$PWD does not look like the GNN-Testing project root."
    echo "       Missing sentinel: $SENTINEL"
    echo ""
    echo "       The project root (with data/) lives in Google Drive on Colab:"
    echo "           cd '/content/drive/MyDrive/GNN测试'"
    echo ""
    echo "       Then re-run: bash scripts/colab_launch.sh <run_script.py> [args...]"
    exit 2
fi

RUN_SCRIPT="${1:-}"
shift || true
RUN_ARGS="$@"

if [[ -z "$RUN_SCRIPT" ]]; then
    echo "Usage: bash $0 <run_script.py> [run_args...]"
    echo "Example: bash $0 run_storya_e1_anchor.py"
    exit 1
fi

if [[ ! -f "$RUN_SCRIPT" ]]; then
    echo "ERROR: run script not found: $RUN_SCRIPT (cwd=$PWD)"
    exit 1
fi

# ─── Step 1: dependencies ───
if ! command -v tmux &> /dev/null; then
    echo "[setup] tmux missing, installing ..."
    apt-get install -y tmux > /dev/null 2>&1 || sudo apt-get install -y tmux > /dev/null
fi

PY="${PYTHON_BIN:-python}"
if ! $PY -c "import wandb" &> /dev/null; then
    echo "[setup] wandb missing, installing ..."
    $PY -m pip install -q wandb
fi

# ─── Step 2: wandb auto-login (optional) ───
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "[setup] WANDB_API_KEY found in env, logging in ..."
    wandb login --relogin "$WANDB_API_KEY" > /dev/null 2>&1 || true
fi

# ─── Step 3: log dir + tmux session ───
mkdir -p artifacts/colab_runs
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="artifacts/colab_runs/${TS}_$(basename $RUN_SCRIPT .py).log"
SESSION="train"

# Kill any prior 'train' session so we always launch fresh
tmux kill-session -t $SESSION 2>/dev/null || true

# Launch inside tmux. Use `tee` so we get both terminal output AND a log file.
tmux new-session -d -s $SESSION \
    "$PY -u $RUN_SCRIPT $RUN_ARGS 2>&1 | tee $LOG_FILE"

# ─── Step 4: status report ───
echo ""
echo "===================================================="
echo "✓ Training started in tmux session '$SESSION'"
echo "===================================================="
echo "Command:      $PY -u $RUN_SCRIPT $RUN_ARGS"
echo "Log file:     $LOG_FILE"
echo ""
echo "Attach live:  tmux attach -t $SESSION"
echo "Detach:       Ctrl+B then D  (training keeps running)"
echo "Tail log:     tail -f $LOG_FILE"
echo "Kill:         tmux kill-session -t $SESSION"
echo "===================================================="
