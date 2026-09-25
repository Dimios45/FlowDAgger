#!/usr/bin/env bash
# Offline YAM baseline: prepare if needed, invert, verify, train, validate.
# --check checks the runtime/configuration without loading weights or training.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--check" ) ]]; then
  echo "Usage: bash flowdagger_pi05/yam/run_training.sh [--check]" >&2
  exit 2
fi

PY="${YAM_PYTHON:-/home/sra/molmoact2/.venv-pi05-server/bin/python}"
LEROBOT_SRC="${YAM_LEROBOT_SRC:-/home/sra/molmoact2/third_party/lerobot-pi05-server/src}"
CKPT="${YAM_CHECKPOINT:-/home/sra/molmoact2/outputs/models/molmoact2-yam-pi05}"
TOKENIZER="${YAM_TOKENIZER:-/home/sra/molmoact2/outputs/models/paligemma-tokenizer}"
RUN="${YAM_RUN_DIR:-$ROOT/outputs/yam-flowdagger-v1}"
DATASET="$ROOT/outputs/yam-flowdagger/dataset"
CONVENTIONS="$ROOT/outputs/yam-gripper-audit/conventions.json"
export PYTHONPATH="$ROOT:$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"

trap 'echo "Stopped at line $LINENO. See the current stage log; later stages were not run." >&2' ERR

[[ -x "$PY" ]] || { echo "Python runtime missing: $PY" >&2; exit 1; }
for path in "$CKPT/model.safetensors" "$TOKENIZER/tokenizer.json" \
            "$DATASET/meta/info.json" "$CONVENTIONS"; do
  [[ -r "$path" ]] || { echo "Required file missing: $path" >&2; exit 1; }
done

"$PY" - "$CONVENTIONS" <<'PY'
import json
import sys
import av
import pandas
import pyarrow
import torch
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from flowdagger_pi05.yam.core import verify_conventions

with open(sys.argv[1]) as f:
    verify_conventions(json.load(f))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this process. Run on the GPU host with CUDA access.")
print("Runtime/configuration check passed. GPU:", torch.cuda.get_device_name(0))
PY

if [[ "${1:-}" == "--check" ]]; then
  exit 0
fi

mkdir -p "$RUN/training"
# Each invocation gets a fresh actor path; the inversion cache is reusable.
TRAIN_DIR="$(mktemp -d "$RUN/training/baseline-seed0-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
COMMON=(--checkpoint "$CKPT" --tokenizer "$TOKENIZER" --device cuda)
echo "Logs and actor: $TRAIN_DIR"
echo "AdamW lr=1e-4 wd=1e-4; batch=64; epochs<=250; patience=20; seed=0"

stage() {
  local name="$1"
  shift
  echo "Starting $name"
  "$PY" -u -m flowdagger_pi05.yam "$@" 2>&1 | tee "$TRAIN_DIR/$name.log"
}

if [[ ! -f "$RUN/manifest.json" ]]; then
  stage prepare prepare --dataset "$DATASET" --conventions "$CONVENTIONS" \
    --output "$RUN/manifest.json" --stride 10 --seed 0
fi

stage invert invert "${COMMON[@]}" \
  --manifest "$RUN/manifest.json" --output "$RUN/cache"

stage verify verify "${COMMON[@]}" --cache "$RUN/cache"

stage train train "${COMMON[@]}" --cache "$RUN/cache" \
  --output "$TRAIN_DIR/actor.pt" \
  --batch-size 64 --epochs 250 --patience 20 --seed 0

stage validation evaluate "${COMMON[@]}" --cache "$RUN/cache" \
  --actor "$TRAIN_DIR/actor.pt" --split validation \
  --output "$TRAIN_DIR/validation.json"

echo "Finished. Best actor: $TRAIN_DIR/actor.pt"
echo "Validation comparison: $TRAIN_DIR/validation.json"
echo "Test episodes were not evaluated. No robot commands were issued."
