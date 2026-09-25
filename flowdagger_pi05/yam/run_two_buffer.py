"""Reproducible reviewed two-buffer experiment using one frozen model instance."""
import argparse
import json
from pathlib import Path

from .backend import NativeBackend
from .core import load_actor
from .data import write_json
from .pipeline import invert_dataset, train, read_cache, evaluate
from .preservation import prepare_preservation, load_preservation_cache
from .verify import verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="outputs/yam-flowdagger-v1")
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", default="/home/sra/molmoact2/outputs/models/molmoact2-yam-pi05")
    p.add_argument("--tokenizer", default="/home/sra/molmoact2/outputs/models/paligemma-tokenizer")
    p.add_argument("--device", default="cuda")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    root, out = Path(args.root), Path(args.output)
    if (out / "actor.pt").exists():
        raise ValueError("Use a fresh run directory; existing actors are never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    review = root / "preservation-review"
    summary = prepare_preservation(root / "manifest.json", review / "review.json", review / "manifest.json")
    print(json.dumps(summary), flush=True)
    backend = NativeBackend(args.checkpoint, args.tokenizer, args.device)
    preservation = root / "preservation-cache"
    invert_dataset(backend, review / "manifest.json", preservation)
    meta, human_examples = read_cache(root / "cache", accepted_only=False)
    auto_meta, auto_examples, auto_train = load_preservation_cache(preservation, meta)
    verification = verify(backend, preservation)
    verification.update(preservation=summary, accepted_preservation_train=len(auto_train))
    write_json(out / "verification.json", verification)
    if args.verify_only:
        print(json.dumps(verification, indent=2), flush=True)
        return
    train(backend, root / "cache", out / "actor.pt", epochs=args.epochs, patience=args.patience,
          batch_size=args.batch_size, seed=args.seed, learning_rate=args.learning_rate,
          weight_decay=args.weight_decay, preservation_cache=preservation)
    actor, _ = load_actor(out / "actor.pt", meta["identity"], args.device)
    human_report = evaluate(backend, actor, meta, human_examples, "validation")
    auto_report = evaluate(backend, actor, auto_meta, auto_examples, "validation")
    write_json(out / "validation-human.json", human_report)
    write_json(out / "validation-preservation.json", auto_report)
    comparison = dict(human=human_report["mean_normalized_rmse"],
                      reconstructed_preservation=auto_report["mean_normalized_rmse"])
    comparison["equal_buffer_mean"] = {name: .5 * (comparison["human"][name] + comparison["reconstructed_preservation"][name])
                                     for name in comparison["human"]}
    write_json(out / "comparison.json", comparison)
    print(json.dumps(comparison, indent=2), flush=True)
    print(f"Finished. Actor: {out / 'actor.pt'}. No test evaluation or robot commands.", flush=True)


if __name__ == "__main__":
    main()
