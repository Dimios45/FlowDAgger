#!/usr/bin/env bash
# Three fresh actors, same seed/split/cache; no inversion or test-set evaluation.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY=/home/sra/molmoact2/.venv-pi05-server/bin/python
export PYTHONPATH="$ROOT:/home/sra/molmoact2/third_party/lerobot-pi05-server/src${PYTHONPATH:+:$PYTHONPATH}"
RUN="$ROOT/outputs/yam-flowdagger-v1"
[[ -f "$RUN/cache/metadata.json" ]] || { echo "Complete the baseline inversion first." >&2; exit 1; }
mkdir -p "$RUN/training"
OUT="$(mktemp -d "$RUN/training/tuning-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
COMMON=(
  --checkpoint /home/sra/molmoact2/outputs/models/molmoact2-yam-pi05
  --tokenizer /home/sra/molmoact2/outputs/models/paligemma-tokenizer
  --device cuda
  --cache "$RUN/cache"
)
trap 'echo "Stopped at line $LINENO; inspect logs in $OUT" >&2' ERR
echo "Tuning outputs: $OUT"

for config in "3e-5 1e-4 lower_lr" "1e-4 1e-3 stronger_decay" "3e-5 1e-3 combined"; do
  read -r lr wd name <<< "$config"
  mkdir "$OUT/$name"
  echo "Training $name: learning rate=$lr, weight decay=$wd"
  "$PY" -u -m flowdagger_pi05.yam train "${COMMON[@]}" \
    --output "$OUT/$name/actor.pt" \
    --learning-rate "$lr" --weight-decay "$wd" \
    --batch-size 64 --epochs 250 --patience 20 --seed 0 \
    2>&1 | tee "$OUT/$name/train.log"

  "$PY" -u -m flowdagger_pi05.yam evaluate "${COMMON[@]}" \
    --actor "$OUT/$name/actor.pt" --split validation \
    --output "$OUT/$name/validation.json" \
    2>&1 | tee "$OUT/$name/validation.log"
done

"$PY" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
summary = {}
for path in sorted(root.glob("*/validation.json")):
    report = json.loads(path.read_text())
    episodes = sorted({e["window"]["episode"] for e in report["examples"]})
    per_episode = {}
    for episode in episodes:
        rows = [e for e in report["examples"] if e["window"]["episode"] == episode]
        per_episode[episode] = {
            method: sum(e["methods"][method]["normalized"]["rmse"] for e in rows) / len(rows)
            for method in ("steered", "zero", "random")
        }
    summary[path.parent.name] = {
        "mean_normalized_rmse": report["mean_normalized_rmse"],
        "per_episode": per_episode,
    }
(root / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
print(f"Finished. Comparison: {root / 'comparison.json'}")
PY
