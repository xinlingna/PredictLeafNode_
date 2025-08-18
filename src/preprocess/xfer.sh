#!/usr/bin/env bash
set -euo pipefail

# xfer.sh — One-click cross-server file/dir transfer via SSH
# Default: rsync -e ssh (resumable, preserves attrs). Fallback: scp -3.
# Usage:
#   ./xfer.sh --src-user xln --src-host 192.169.1.13 --src-path /home/xln/elpis/index/gist_query_10k/txt/query_knn_distributions.txt \
#             --dst-user xln --dst-host 192.168.1.12 --dst-path /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize10k/ \
#             [--port 22] [--identity ~/.ssh/id_rsa] [--mode auto|rsync|scp] [--dry-run]

SRC_USER="" SRC_HOST="" SRC_PATH=""
DST_USER="" DST_HOST="" DST_PATH=""
SSH_PORT="22" ID_FILE="" MODE="auto" DRY_RUN=0

die(){ echo "ERROR: $*" >&2; exit 1; }

usage(){
  sed -n '2,30p' "$0"
  exit 1
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --src-user) SRC_USER="$2"; shift 2;;
    --src-host) SRC_HOST="$2"; shift 2;;
    --src-path) SRC_PATH="$2"; shift 2;;
    --dst-user) DST_USER="$2"; shift 2;;
    --dst-host) DST_HOST="$2"; shift 2;;
    --dst-path) DST_PATH="$2"; shift 2;;
    --port)     SSH_PORT="$2"; shift 2;;
    --identity) ID_FILE="$2"; shift 2;;
    --mode)     MODE="$2"; shift 2;; # auto|rsync|scp
    --dry-run)  DRY_RUN=1; shift;;
    -h|--help)  usage;;
    *) echo "Unknown arg: $1"; usage;;
  esac
done

[[ -n "$SRC_USER" && -n "$SRC_HOST" && -n "$SRC_PATH" && -n "$DST_USER" && -n "$DST_HOST" && -n "$DST_PATH" ]] \
  || die "Missing required arguments. Run with --help."

SSH_BASE=(ssh -p "$SSH_PORT" -o BatchMode=yes -o StrictHostKeyChecking=accept-new)
[[ -n "$ID_FILE" ]] && SSH_BASE+=(-i "$ID_FILE")

# Helper to run ssh to specific host/user
ssh_as(){ local USER="$1" HOST="$2"; shift 2; "${SSH_BASE[@]}" "$USER@$HOST" "$@"; }

echo "==> Checking SSH connectivity..."
ssh_as "$SRC_USER" "$SRC_HOST" "echo ok" >/dev/null || die "Cannot SSH to source $SRC_USER@$SRC_HOST"
ssh_as "$DST_USER" "$DST_HOST" "echo ok" >/dev/null || die "Cannot SSH to destination $DST_USER@$DST_HOST"

echo "==> Checking source path type (file/dir)..."
SRC_TYPE=$(ssh_as "$SRC_USER" "$SRC_HOST" "if [ -f '$SRC_PATH' ]; then echo file; elif [ -d '$SRC_PATH' ]; then echo dir; else echo missing; fi")
[[ "$SRC_TYPE" != "missing" ]] || die "Source path not found on $SRC_HOST: $SRC_PATH"

echo "==> Ensuring destination directory exists..."
# If SRC is a file: ensure parent dir of DST_PATH (if dst path ends with / treat as dir)
if [[ "$SRC_TYPE" == "file" ]]; then
  if [[ "$DST_PATH" == */ ]]; then
    ssh_as "$DST_USER" "$DST_HOST" "mkdir -p '$DST_PATH'"
  else
    ssh_as "$DST_USER" "$DST_HOST" "mkdir -p \"\$(dirname '$DST_PATH')\""
  fi
else
  ssh_as "$DST_USER" "$DST_HOST" "mkdir -p '$DST_PATH'"
fi

# Decide mode
choose_mode(){
  case "$MODE" in
    rsync) echo rsync;;
    scp)   echo scp;;
    auto)
      if command -v rsync >/dev/null 2>&1; then echo rsync; else echo scp; fi
      ;;
    *) die "Invalid --mode $MODE";;
  esac
}

MODE_CHOSEN=$(choose_mode)
echo "==> Transfer mode: $MODE_CHOSEN"

# Build SSH option string for rsync/scp
SSH_OPT_STR="-p $SSH_PORT -o StrictHostKeyChecking=accept-new"
[[ -n "$ID_FILE" ]] && SSH_OPT_STR="$SSH_OPT_STR -i $ID_FILE"

set -x
if [[ "$MODE_CHOSEN" == "rsync" ]]; then
  # rsync remote-src to remote-dst via local relay: rsync supports SRC as ssh, DEST as ssh
  # If SRC is file and DST_PATH ends with /, rsync will drop into that dir; else it uses exact destination path.
  ROPT=(-e "ssh $SSH_OPT_STR" -a --progress)
  [[ $DRY_RUN -eq 1 ]] && ROPT+=(--dry-run)
  if [[ "$SRC_TYPE" == "file" ]]; then
    rsync "${ROPT[@]}" "$SRC_USER@$SRC_HOST:$SRC_PATH" "$DST_USER@$DST_HOST:$DST_PATH"
  else
    # For directories, append trailing slash to copy contents vs dir itself; default here: copy the dir itself into DST_PATH
    rsync "${ROPT[@]}" -r "$SRC_USER@$SRC_HOST:$SRC_PATH" "$DST_USER@$DST_HOST:$DST_PATH"
  fi
else
  # Fallback: scp -3 (force through local)
  # -r for directories, -p to preserve times/modes
  SCP_OPTS=(-P "$SSH_PORT" -o StrictHostKeyChecking=accept-new -3 -p)
  [[ -n "$ID_FILE" ]] && SCP_OPTS+=(-i "$ID_FILE")
  [[ "$SRC_TYPE" == "dir" ]] && SCP_OPTS+=(-r)
  if [[ $DRY_RUN -eq 1 ]]; then
    set +x
    echo "[DRY-RUN] scp ${SCP_OPTS[*]} $SRC_USER@$SRC_HOST:$SRC_PATH $DST_USER@$DST_HOST:$DST_PATH"
    set -x
  else
    scp "${SCP_OPTS[@]}" "$SRC_USER@$SRC_HOST:$SRC_PATH" "$DST_USER@$DST_HOST:$DST_PATH"
  fi
fi
set +x

echo "==> Done."


# ./xfer.sh \
#   --src-user xln --src-host 192.169.1.13 \
#   --src-path /home/xln/elpis/index/gist_query_10k/txt/query_knn_distributions.txt \
#   --dst-user xln --dst-host 192.168.1.12 \
#   --dst-path /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize10k/ \
#   --mode auto
