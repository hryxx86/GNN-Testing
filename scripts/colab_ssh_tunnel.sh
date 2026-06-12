#!/usr/bin/env bash
# Manual cloudflared SSH tunnel for Colab — replaces the unmaintained `colab_ssh`
# package (PyPI 0.3.27, last updated 2021-10).
#
# WHY this exists (diagnosis 2026-06-10):
#   The colab_ssh path broke with a local "websocket: bad handshake" /
#   "Connection closed by UNKNOWN port 65535". Live probe of the tunnel hostname
#   returned HTTP 502 from the Cloudflare edge — i.e. the cloudflared tunnel was
#   UP but the Colab-side sshd was NOT listening on localhost:22 (origin down).
#   colab_ssh started the tunnel but its sshd was dead / on the wrong port.
#
# WHAT this does instead (explicit, debuggable, current):
#   1. Installs + (re)starts a real sshd bound to :22 with root password auth.
#   2. Downloads cloudflared from the CURRENT GitHub release URL (the Equinox.io
#      URL baked into colab_ssh 0.3.27 has been 404 since 2021).
#   3. Opens a quick tunnel pinned to --protocol http2 (TCP). cloudflared defaults
#      to QUIC (UDP); UDP is frequently blocked/idle-timed-out on the path, which
#      is a separate well-documented cause of hanging SSH connects. http2 sidesteps it.
#   4. Parses the *.trycloudflare.com hostname itself and prints a ready-to-paste
#      local command. No dependency on colab_ssh's brittle log parser.
#
# Local side is UNCHANGED. ~/.ssh/config already proxies *.trycloudflare.com via:
#     ProxyCommand /opt/homebrew/bin/cloudflared access ssh --hostname %h
#
# USAGE — run ONCE per Colab session, from a notebook cell:
#     !bash scripts/colab_ssh_tunnel.sh
#
# Override the password (must match local sshpass) via env: COLAB_SSH_PASS=...

set -e

PASS="${COLAB_SSH_PASS:-GNNTEST}"
LOG="/content/cloudflared_tunnel.log"
SESSION="cf-tunnel"

# ─── Step 1: sshd — install, host keys, root password, FORCE port 22, restart ───
echo "[1/4] Configuring sshd (forcing port 22) ..."
if ! command -v sshd >/dev/null 2>&1 && [[ ! -x /usr/sbin/sshd ]]; then
    apt-get -qq update >/dev/null 2>&1 || true
    apt-get -qq install -y openssh-server >/dev/null 2>&1
fi
mkdir -p /run/sshd
ssh-keygen -A >/dev/null 2>&1 || true          # generate any missing host keys
echo "root:${PASS}" | chpasswd

# WHY force port 22 (root cause observed live 2026-06-10): a stale colab_ssh setup
# left sshd on :2222 while its tunnel pointed at :22 → permanent HTTP 502 / "websocket:
# bad handshake". Neutralize every pre-existing Port directive, then pin Port 22 via a
# high-priority drop-in (Ubuntu/Colab parses sshd_config.d/*.conf first), so sshd binds
# exactly :22 and the tunnel below (→ ssh://localhost:22) can never port-mismatch.
# The drop-in also wins on PermitRootLogin/PasswordAuthentication over any cloud-image conf.
sed -i -E 's/^[[:space:]]*(Port|PasswordAuthentication|PermitRootLogin)[[:space:]].*/# &  (disabled by colab_ssh_tunnel.sh)/' /etc/ssh/sshd_config 2>/dev/null || true
mkdir -p /etc/ssh/sshd_config.d
for f in /etc/ssh/sshd_config.d/*.conf; do
    [ -f "$f" ] && sed -i -E 's/^[[:space:]]*(Port|PasswordAuthentication|PermitRootLogin)[[:space:]].*/# &  (disabled by colab_ssh_tunnel.sh)/' "$f" 2>/dev/null || true
done
cat > /etc/ssh/sshd_config.d/00-colab-ssh.conf <<'EOF'
Port 22
PermitRootLogin yes
PasswordAuthentication yes
EOF

# Kill any pre-existing sshd (e.g. a stale colab_ssh one on :2222) so the fresh start
# binds cleanly on :22. Safe here: this runs from a Colab notebook cell, not over SSH.
pkill -x sshd 2>/dev/null || true
pkill -f '/usr/sbin/sshd' 2>/dev/null || true
sleep 1
service ssh restart >/dev/null 2>&1 || /usr/sbin/sshd

# Confirm sshd is actually listening on :22 before bothering with the tunnel.
if ! (ss -tlnp 2>/dev/null | grep -q ':22 ' || netstat -tlnp 2>/dev/null | grep -q ':22 '); then
    echo "ERROR: sshd is not listening on :22 after restart. Aborting."
    echo "       Debug: /usr/sbin/sshd -t  (config test)  ;  ps aux | grep sshd"
    exit 1
fi
echo "      sshd listening on :22 ✓"

# ─── Step 2: cloudflared binary (current URL) ───
echo "[2/4] Ensuring cloudflared ..."
if [[ ! -x /usr/local/bin/cloudflared ]]; then
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
        -O /usr/local/bin/cloudflared
    chmod +x /usr/local/bin/cloudflared
fi
/usr/local/bin/cloudflared --version || true

# ─── Step 3: quick tunnel (http2, not QUIC) inside tmux so it survives the cell ───
echo "[3/4] Starting cloudflared quick tunnel (http2 → ssh://localhost:22) ..."
if ! command -v tmux >/dev/null 2>&1; then
    apt-get -qq install -y tmux >/dev/null 2>&1
fi
: > "$LOG"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" \
    "/usr/local/bin/cloudflared tunnel --no-autoupdate --protocol http2 --url ssh://localhost:22 > '$LOG' 2>&1"

# ─── Step 4: parse the trycloudflare hostname from the log ───
echo "[4/4] Waiting for tunnel hostname (up to ~60s) ..."
HOST=""
for _ in $(seq 1 30); do
    sleep 2
    HOST=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" 2>/dev/null | head -1 | sed 's#https://##')
    [[ -n "$HOST" ]] && break
done

if [[ -z "$HOST" ]]; then
    echo "ERROR: tunnel did not report a hostname within ~60s. Last 20 log lines:"
    tail -20 "$LOG"
    exit 1
fi

echo ""
echo "===================================================="
echo "✓ Colab SSH tunnel UP (http2, sshd on :22)"
echo "  hostname: ${HOST}"
echo ""
echo "On the local Mac, paste (path B: code = GitHub clone, data symlinked from Drive):"
echo "  sshpass -p ${PASS} ssh ${HOST} \\"
echo "    \"cd /content/GNN-Testing && git pull && bash scripts/colab_launch.sh <run_script.py>\""
echo ""
echo "Tunnel logs:  tail -f ${LOG}    |    Tunnel tmux: tmux attach -t ${SESSION}"
echo "===================================================="
