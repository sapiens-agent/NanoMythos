#!/usr/bin/env bash
# FineWeb-Edu 10K: baseline vs T=1, 5k steps, multi-seed (521 protocol)
set -euo pipefail
export NANOWHALE_DEVICE_SELECTOR="${NANOWHALE_DEVICE_SELECTOR:-0}"
REPO_ROOT="${NANOWHALE_ROOT:-/home/zetyun/nanowhale}"
PY="${NANOWHALE_PYTHON:-python}"
cd "$REPO_ROOT"
mkdir -p /data/zetyun/eval/521/logs
TS=$(date +%Y%m%d_%H%M%S)
LOG=/data/zetyun/eval/521/logs/t1_cross_seed_${TS}.log

echo "Using REPO_ROOT=$REPO_ROOT NANOWHALE_DEVICE_SELECTOR=$NANOWHALE_DEVICE_SELECTOR" | tee "$LOG"
"$PY" "${BASH_SOURCE%/*}/521_fw_cross_seed_5k.py" --mode "${1:-all}" --seeds "${2:-2025,2027,2048}" \
  2>&1 | tee -a "$LOG"
"$PY" "$REPO_ROOT/scripts/521_summarize_cross_seed.py" 2>&1 | tee -a "$LOG"
echo "Done. See /data/zetyun/eval/521/ and results/521_fineweb10k_cross_seed.csv" | tee -a "$LOG"
