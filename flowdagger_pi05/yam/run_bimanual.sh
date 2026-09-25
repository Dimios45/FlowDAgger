#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="/home/sra/molmoact2/third_party/lerobot-pi05-server/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
mkdir -p outputs/yam-flowdagger-v2/training
exec 9>outputs/yam-flowdagger-v2/training.lock
flock -n 9 || { echo "A bimanual training pipeline is already running" >&2; exit 1; }
RUN_DIR=$(mktemp -d "outputs/yam-flowdagger-v2/training/bimanual-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
printf '%s\n' "$RUN_DIR" | tee outputs/yam-flowdagger-v2/latest-run.txt
/home/sra/molmoact2/.venv-pi05-server/bin/python -u \
  -m flowdagger_pi05.yam.run_bimanual --output "$RUN_DIR" "$@" \
  2>&1 | tee "$RUN_DIR/run.log"
