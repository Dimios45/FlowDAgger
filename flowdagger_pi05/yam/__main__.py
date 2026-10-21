import argparse
import json
from pathlib import Path


def main(default_backend="torch"):
    parser = argparse.ArgumentParser(description=f"Frozen pi0.5 YAM steering (default backend: {default_backend})")
    commands = parser.add_subparsers(dest="command", required=True)
    merge = commands.add_parser("merge-caches")
    merge.add_argument("--sources", required=True, help="JSON array of {cache, traces} round sources")
    merge.add_argument("--output", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--conventions", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--splits", help="JSON train/validation/test episode lists; preserve splits across rounds")
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--stride", type=int, default=10)
    preservation = commands.add_parser("prepare-preservation")
    preservation.add_argument("--source-manifest", required=True)
    preservation.add_argument("--review", required=True)
    preservation.add_argument("--output", required=True)
    preservation.add_argument("--stride", type=int, default=30)
    audit = commands.add_parser("audit")
    audit.add_argument("--dataset", required=True)
    for name in ("invert", "train", "evaluate", "serve", "verify", "diagnose"):
        p = commands.add_parser(name)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--tokenizer", required=True)
        p.add_argument("--device", default="cuda")
        p.add_argument("--backend", choices=("torch", "jax"), default=default_backend)
        p.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
        if name == "invert":
            p.add_argument("--manifest", required=True)
            p.add_argument("--output", required=True)
            p.add_argument("--limit", type=int)
        elif name in ("train", "evaluate", "verify"):
            p.add_argument("--cache", required=True)
            if name == "train":
                p.add_argument("--output", required=True)
                p.add_argument("--epochs", type=int, default=50)
                p.add_argument("--patience", type=int, default=5)
                p.add_argument("--batch-size", type=int, default=64)
                p.add_argument("--seed", type=int, default=0)
                p.add_argument("--learning-rate", type=float, default=3e-5)
                p.add_argument("--weight-decay", type=float, default=1e-3)
                buffers = p.add_mutually_exclusive_group()
                buffers.add_argument("--autonomous")
                buffers.add_argument("--preservation-cache")
                p.add_argument("--allow-human-only", action="store_true", help="Diagnostic ablation; not two-buffer FlowDAgger")
            elif name == "evaluate":
                p.add_argument("--actor", required=True)
                p.add_argument("--split", choices=("validation", "test"), default="test")
                p.add_argument("--output", required=True)
        elif name == "diagnose":
            p.add_argument("--manifest", help="Prepared human manifest; omit for synthetic sampler-only smoke test")
            p.add_argument("--window", type=int, default=0)
            p.add_argument("--output", required=True)
            p.add_argument("--iteration-sweep", action="store_true")
        else:
            p.add_argument("--actor", help="Omit for original random-noise base collection")
            p.add_argument("--conventions", required=True)
            p.add_argument("--traces", required=True)
            p.add_argument("--normalization-tag", required=True)
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8204)
    args = parser.parse_args()
    if args.command == "train" and not (args.autonomous or args.preservation_cache or args.allow_human_only):
        parser.error("Training requires --autonomous or --preservation-cache; --allow-human-only is an explicit ablation")
    from .data import Dataset, prepare as prepare_dataset, windows, write_json
    if args.command == 'merge-caches':
        from .pipeline import merge_caches
        print(json.dumps(merge_caches(json.loads(Path(args.sources).read_text()),args.output),indent=2))
        return
    if args.command == "prepare-preservation":
        from .preservation import prepare_preservation
        print(json.dumps(prepare_preservation(args.source_manifest, args.review, args.output, args.stride), indent=2))
        return
    if args.command == "prepare":
        m = prepare_dataset(args.dataset, json.loads(Path(args.conventions).read_text()), args.output, args.seed, args.stride,
                            json.loads(Path(args.splits).read_text()) if args.splits else None)
        print(json.dumps(dict(windows=len(m["windows"]), excluded_short_segments=m["excluded_short_segments"], splits=m["splits"]), indent=2))
        return
    if args.command == "audit":
        d = Dataset(args.dataset)
        w, excluded = windows(d.frame)
        print(json.dumps(dict(episodes=len(d.episodes), frames=len(d.frame), complete_human_windows=len(w), excluded_short_segments=excluded), indent=2))
        return
    if args.backend == "jax":
        from flowdagger_pi05.yam_jax.backend import JaxBackend
        backend = JaxBackend(args.checkpoint, args.tokenizer, args.dtype)
    else:
        from .backend import NativeBackend
        backend = NativeBackend(args.checkpoint, args.tokenizer, args.device)
    from .pipeline import invert_dataset, train, read_cache, evaluate
    if args.command == "diagnose":
        import numpy as np
        from .diagnostics import diagnose
        from .data import model_example
        if args.manifest:
            m = json.loads(Path(args.manifest).read_text())
            d = Dataset(m['dataset'])
            obs, actions, _ = model_example(d, m['windows'][args.window], m['conventions'])
        else:
            obs = dict(state=np.zeros(14, np.float32), instruction="fold the towel",
                **{k+'_cam':np.zeros((224,224,3),np.uint8) for k in ('top','left','right')})
            actions = None
        write_json(args.output, diagnose(backend, obs, actions, iteration_sweep=args.iteration_sweep))
    elif args.command == "invert":
        invert_dataset(backend, args.manifest, args.output, args.limit)
    elif args.command == "train":
        train(backend, args.cache, args.output, args.epochs, args.patience, args.batch_size, args.seed,
              args.autonomous, learning_rate=args.learning_rate, weight_decay=args.weight_decay,
              preservation_cache=args.preservation_cache)
    elif args.command == "evaluate":
        from .core import load_actor
        meta, examples = read_cache(args.cache, accepted_only=False)
        from .pipeline import verify_cache_datasets
        verify_cache_datasets(meta)
        identity = dict(backend.identity, conventions=meta["identity"]["conventions"])
        if identity != meta["identity"]:
            raise ValueError("Evaluation backend/cache mismatch")
        actor, _ = load_actor(args.actor, identity, backend.device)
        write_json(args.output, evaluate(backend, actor, meta, examples, args.split))
    elif args.command == "verify":
        from .verify import verify
        print(json.dumps(verify(backend, args.cache), indent=2))
    else:
        from .serve import Policy, build_app
        import uvicorn
        policy = Policy(backend, args.actor, json.loads(Path(args.conventions).read_text()), args.traces, args.normalization_tag)
        if args.backend == 'jax':
            import numpy as np
            import torch
            print('Warming JAX sampler before accepting robot requests...', flush=True)
            p = backend.prepare(dict(state=np.zeros(14,np.float32),instruction='fold the towel',
                **{c+'_cam':np.zeros((224,224,3),np.uint8) for c in ('top','left','right')}))
            warm = backend.sample(p,torch.zeros(1,30,32))
            if not torch.isfinite(warm).all():
                raise RuntimeError('Nonfinite warmup output')
            print('JAX warmup complete.',flush=True)
        uvicorn.run(build_app(policy), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
