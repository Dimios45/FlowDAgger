#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="/home/sra/molmoact2/third_party/lerobot-pi05-server/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
mkdir -p outputs/yam-flowdagger-v1/training
RUN_DIR=$(mktemp -d "outputs/yam-flowdagger-v1/training/reconstructed-two-buffer-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
printf 'Run directory: %s\n' "$RUN_DIR"
/home/sra/molmoact2/.venv-pi05-server/bin/python -u \
  -m flowdagger_pi05.yam.run_two_buffer --output "$RUN_DIR" "$@" \
  2>&1 | tee "$RUN_DIR/run.log"
