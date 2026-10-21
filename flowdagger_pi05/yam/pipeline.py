"""Persistent inversion examples and episode-held-out decoded-action training."""
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .core import CONTRACT, SteeringActor, digest, errors, load_actor
from .data import Dataset, model_example, write_json


def save_tensor(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(data, tmp)
    tmp.replace(path)


def invert_dataset(backend, manifest_path, output, limit=None):
    manifest = json.loads(Path(manifest_path).read_text())
    from .core import verify_conventions
    verify_conventions(manifest["conventions"])
    if manifest["contract"] != CONTRACT:
        raise ValueError("Manifest sampler contract mismatch")
    dataset = Dataset(manifest["dataset"], **({"episodes": manifest["dataset_episodes"]} if "dataset_episodes" in manifest else {}))
    if dataset.identity() != manifest["dataset_identity"]:
        raise ValueError("Dataset changed after preparation")
    identity = dict(backend.identity, conventions=manifest["conventions"])
    output = Path(output)
    metadata = dict(identity=identity, manifest=manifest, cache_id=digest([identity, manifest["id"]]))
    if (output / "metadata.json").exists() and json.loads((output / "metadata.json").read_text()) != metadata:
        raise ValueError("Refusing to mix incompatible inversion caches")
    write_json(output / "metadata.json", metadata)
    for number, window in enumerate(manifest["windows"][:limit]):
        path = output / "examples" / f"{number:06d}.pt"
        if path.exists():
            continue
        obs, actions, alignment = model_example(dataset, window, manifest["conventions"])
        p = backend.prepare(obs, actions)
        zero = torch.zeros(1, 30, 32, device=backend.device)
        completion = backend.sample(p, zero)
        target = completion.clone()
        target[..., :14] = p["batch"]["action"]
        noise = backend.invert(p, target)
        if torch.isfinite(noise).all():
            decoded = backend.sample(p, noise)
            normalized = errors(decoded, target)
            physical = errors(backend.decode_actions(decoded), torch.from_numpy(actions)[None])
            normalized["accepted"] &= bool(torch.isfinite(decoded).all() and physical["finite"])
        else:
            normalized = dict(finite=False, accepted=False, reason='nonfinite inverted noise')
            physical = dict(finite=False, accepted=False)
        from .diagnostics import noise_stats
        save_tensor(path, dict(cache_id=metadata["cache_id"], window=window, alignment=alignment,
            patches=p["patches"].cpu()[0], state=p["state"].cpu()[0], noise=noise.cpu()[0],
            target=target.cpu()[0], normalized=normalized, physical=physical,
            inversion_diagnostics=dict(noise=noise_stats(noise), **getattr(backend, "last_inversion_diagnostics", {}))))
        print(json.dumps(dict(example=number, **normalized)), flush=True)
    from .diagnostics import summarize
    _, rows = read_cache(output, accepted_only=False, require_complete=False)
    if rows:
        write_json(output / "inversion-summary.json", summarize(rows))


def read_cache(root, accepted_only=True, require_complete=True):
    root = Path(root)
    meta = json.loads((root / "metadata.json").read_text())
    examples = [torch.load(p, weights_only=True) for p in sorted((root / "examples").glob("*.pt"))]
    if require_complete and len(examples) != len(meta["manifest"]["windows"]):
        raise ValueError("Incomplete inversion cache; finish inversion before training/evaluation")
    for path, example in zip(sorted((root / "examples").glob("*.pt")), examples):
        if example["window"] != meta["manifest"]["windows"][int(path.stem)]:
            raise ValueError("Cache window provenance mismatch")
    if any(e["cache_id"] != meta["cache_id"] for e in examples):
        raise ValueError("Mixed cache identities")
    return meta, [e for e in examples if e["normalized"]["accepted"] or not accepted_only]


def verify_cache_datasets(meta):
    sources = meta.get('sources', {'single':meta})
    for source in sources.values():
        m = source['manifest']
        kwargs = {'episodes':m['dataset_episodes']} if 'dataset_episodes' in m else {}
        if Dataset(m['dataset'], **kwargs).identity() != m['dataset_identity']:
            raise ValueError('Dataset changed since inversion')


def example_input(meta, example):
    source = meta['sources'][example['source_id']] if 'sources' in meta else meta
    m = source['manifest']
    kwargs = {'episodes':m['dataset_episodes']} if 'dataset_episodes' in m else {}
    return model_example(Dataset(m['dataset'], **kwargs),
        example.get('source_window',example['window']), source['identity']['conventions'])


def merge_caches(specification, output):
    """Aggregate disjoint collection rounds without relabeling their holdouts.

    specification: [{cache: path, traces: optional original-noise trace root}].
    Copies tensors into a new immutable cache; raw datasets remain referenced.
    """
    output = Path(output)
    if output.exists():
        raise ValueError('Use a new aggregate cache directory')
    if not specification:
        raise ValueError('No source caches')
    sources, rows, mapping, trace_roots = {}, [], {}, {}
    identity = None
    dataset_ids = set()
    for entry in specification:
        source, examples = read_cache(entry['cache'], accepted_only=False)
        if 'sources' in source:
            raise ValueError('List original round caches, not previously merged caches')
        verify_cache_datasets(source)
        sid = source['cache_id']
        did = digest(source['manifest']['dataset_identity'])
        if sid in sources or did in dataset_ids:
            raise ValueError('Duplicate dataset/round in aggregate')
        dataset_ids.add(did)
        if identity is not None and source['identity'] != identity:
            raise ValueError('Round model/processor/convention mismatch')
        identity = source['identity']
        sources[sid] = source
        mapping[sid] = {}
        for split, episodes in source['manifest']['splits'].items():
            for ep in episodes:
                mapping[sid][str(ep)] = sum(len(v) for v in mapping.values())
        if entry.get('traces'):
            trace_roots[sid] = str(Path(entry['traces']).resolve())
        for e in examples:
            window = dict(e['window'], episode=mapping[sid][str(e['window']['episode'])])
            rows.append(dict(e, window=window, source_window=e['window'], source_id=sid))
    splits = {split:[mapping[sid][str(ep)] for sid,s in sources.items()
                    for ep in s['manifest']['splits'][split]] for split in ('train','validation','test')}
    manifest = dict(windows=[e['window'] for e in rows], splits=splits, aggregate=True)
    cache_id = digest([list(sources), mapping, trace_roots])
    meta = dict(identity=identity, manifest=manifest, cache_id=cache_id, sources=sources,
                episode_mapping=mapping, autonomous_roots=trace_roots)
    write_json(output/'metadata.json',meta)
    for i,e in enumerate(rows):
        save_tensor(output/'examples'/f'{i:06d}.pt', dict(e, cache_id=cache_id))
    return dict(examples=len(rows), rounds=len(sources), splits=splits)


@torch.no_grad()
def evaluate(backend, actor, meta, examples, split, baselines=True):
    results = []
    generator = torch.Generator(device=backend.device).manual_seed(0)
    for e in examples:
        if e["window"]["split"] != split:
            continue
        obs, actions, _ = example_input(meta, e)
        p = backend.prepare(obs, actions)
        noise = actor(p["patches"], p["state"])
        noises = {"steered": noise}
        if baselines:
            noises.update(zero=torch.zeros_like(noise), random=torch.randn(noise.shape, generator=generator, device=backend.device))
        row = dict(window=e["window"], methods={})
        target = p["batch"]["action"]
        grip = actions[:, [6, 13]]
        # Report all changing-gripper windows, including closing/grasp transitions.
        row["gripper_transition"] = bool(abs(grip[1:] - grip[:-1]).max() > .05)
        for name, w in noises.items():
            decoded = backend.sample(p, w)
            row["methods"][name] = dict(normalized=errors(decoded, target),
                physical=errors(backend.decode_actions(decoded), torch.from_numpy(actions)[None]))
        results.append(row)
    if not results:
        raise ValueError(f"No accepted {split} examples")
    scores = {}
    for name in results[0]["methods"]:
        metrics = [r["methods"][name]["normalized"] for r in results]
        scores[name] = sum(m.get("rmse", float("inf")) for m in metrics) / len(metrics)
    return dict(split=split, mean_normalized_rmse=scores, examples=results)


@torch.no_grad()
def evaluate_traces(backend, actor, records):
    scores = []
    for r in records:
        p = backend.prepare(r['observation'])
        noise = actor(p['patches'], p['state'])
        target = r.get('terminal')
        if target is None:
            target = backend.sample(p, r['noise'][None].to(backend.device))[0]
        scores.append(errors(backend.sample(p, noise), target[None].to(backend.device))['rmse'])
    return sum(scores)/len(scores) if scores else None


def train(backend, cache, output, epochs=50, patience=5, batch_size=64, seed=0, autonomous=None,
          learning_rate=3e-5, weight_decay=1e-3, preservation_cache=None, checkpoint_selection='best',
          all_training=False):
    if all_training and checkpoint_selection != 'last':
        raise ValueError('All-training mode requires last-epoch checkpoint selection')
    if checkpoint_selection not in ('best', 'last'):
        raise ValueError('checkpoint_selection must be best or last')
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if Path(output).exists():
        raise ValueError("Actor output already exists; use a new versioned path for each training round")
    torch.manual_seed(seed)
    meta, examples = read_cache(cache, accepted_only=False)
    if dict(backend.identity, conventions=meta["identity"]["conventions"]) != meta["identity"]:
        raise ValueError("Backend does not match cache")
    human = [e for e in examples if (all_training or e["window"]["split"] == "train") and e["normalized"]["accepted"]]
    if not human:
        raise ValueError("No accepted human training examples")
    if not all_training:
        train_eps = {e['window']['episode'] for e in examples if e['window']['split'] == 'train'}
        val_eps = {e['window']['episode'] for e in examples if e['window']['split'] == 'validation'}
        test_eps = {e['window']['episode'] for e in examples if e['window']['split'] == 'test'}
        if train_eps & val_eps or train_eps & test_eps or val_eps & test_eps:
            raise ValueError("Episode split leakage")
    verify_cache_datasets(meta)
    actor = SteeringActor(human[0]["patches"].shape[-1]).to(backend.device)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=learning_rate, weight_decay=weight_decay)
    auto = []
    trace_validation = []
    preservation_meta, preservation_validation = None, []
    excluded = []
    if autonomous and preservation_cache:
        raise ValueError("Choose original traces or reconstructed preservation, not both")
    if preservation_cache:
        from .preservation import load_preservation_cache, overlaps
        preservation_meta, preservation_validation, auto = load_preservation_cache(preservation_cache, meta)
        if bool(preservation_meta['manifest'].get('all_training')) != all_training:
            raise ValueError('Training and preservation holdout modes differ')
        excluded = preservation_meta["manifest"]["supervision"]["review"].get("human_exclusions", [])
        human = [e for e in human if not overlaps(e["window"], excluded)]
        if not human:
            raise ValueError("Human exclusions removed all accepted training examples")
    if autonomous:
        from .traces import preservation_examples
        if autonomous == 'merged':
            if not meta.get('autonomous_roots'):
                raise ValueError('Aggregate has no trace sources')
            for sid,root in meta['autonomous_roots'].items():
                splits = meta['sources'][sid]['manifest']['splits']
                auto.extend(preservation_examples(root, meta['identity'], splits['train'], allow_empty=True))
                trace_validation.extend(preservation_examples(root, meta['identity'], splits['validation'], allow_empty=True, records=True))
            if not auto:
                raise ValueError('No eligible autonomous anchors in aggregate')
        else:
            auto = preservation_examples(autonomous, meta["identity"], meta["manifest"]["splits"]["train"])
            trace_validation = preservation_examples(autonomous, meta['identity'],
                meta['manifest']['splits']['validation'], allow_empty=True, records=True)
    if epochs < 1 or epochs > 250 or patience < 1 or batch_size < 1:
        raise ValueError("Use 1..250 epochs and positive patience/batch size")
    if auto and (batch_size < 2 or batch_size % 2):
        raise ValueError("Two-buffer training requires an even total batch size")
    human_batch_size = batch_size // 2 if auto else batch_size
    training_config = dict(epochs=epochs, patience=patience, batch_size=batch_size,
        all_training=all_training, validation_enabled=not all_training,
        human_training_episodes=sorted({e['window']['episode'] for e in human}),
        autonomous_training_episodes=sorted({e['window']['episode'] for e in auto}) if preservation_meta else [],
        checkpoint_selection=checkpoint_selection, early_stopping=checkpoint_selection == 'best',
        human_batch_size=human_batch_size, preservation_batch_size=human_batch_size if auto else 0,
        learning_rate=learning_rate, weight_decay=weight_decay, gradient_clip=1.,
        preservation=bool(auto), human_examples=len(human), preservation_examples=len(auto),
        preservation_kind=preservation_meta["manifest"]["supervision"]["kind"] if preservation_meta else
            ("recorded-successful-autonomous-noise" if auto else None),
        preservation_cache_id=preservation_meta["cache_id"] if preservation_meta else None,
        human_exclusions=excluded,
        selection_metric=None if all_training else ("equal-buffer-validation-decoded-rmse" if preservation_meta else "human-validation-decoded-rmse"))
    write_json(Path(output).with_suffix(".training.json"), training_config)
    best, stale, history = float("inf"), 0, []
    for epoch in range(epochs):
        actor.train()
        loss_sum, samples, human_sum, auto_sum = 0., 0, 0., 0.
        # Each intervention segment receives equal total sampling probability.
        from collections import Counter
        from torch.utils.data import WeightedRandomSampler
        segments = [(e['window']['episode'], e['window'].get('segment', 0)) for e in human]
        counts = Counter(segments)
        episode_segments = Counter(s[0] for s in counts)
        sampler = WeightedRandomSampler([1/(counts[s]*episode_segments[s[0]]) for s in segments], len(human), replacement=True)
        auto_episodes = [e.get('episode',e.get('window',{}).get('episode',0)) for e in auto]
        auto_counts = Counter(auto_episodes)
        auto_weights = torch.tensor([1/auto_counts[ep] for ep in auto_episodes])
        for batch in DataLoader(human, batch_size=human_batch_size, sampler=sampler, collate_fn=lambda rows: rows):
            def loss(rows):
                patches, state, noise = [torch.stack([r[k] for r in rows]).to(backend.device) for k in ("patches", "state", "noise")]
                return (actor(patches, state) - noise).square().mean()
            value = loss(batch)
            human_sum += value.detach().item()*len(batch)
            if auto:
                indices = torch.multinomial(auto_weights, len(batch), replacement=True).tolist()
                auto_loss = loss([auto[i] for i in indices])
                auto_sum += auto_loss.detach().item()*len(batch)
                value = .5 * value + .5 * auto_loss
            if not torch.isfinite(value):
                raise RuntimeError("Nonfinite training loss")
            optimizer.zero_grad()
            value.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.)
            optimizer.step()
            loss_sum += value.detach().item() * len(batch)
            samples += len(batch)
        actor.eval()
        if all_training:
            if not all(torch.isfinite(v).all() for v in actor.state_dict().values()):
                raise RuntimeError('Nonfinite actor weights; refusing to save checkpoint')
            history.append(dict(epoch=epoch, training_noise_mse=loss_sum/samples))
            write_json(Path(output).with_suffix('.history.json'), history)
            print(history[-1], flush=True)
            save_tensor(output, dict(state_dict={k:v.detach().cpu() for k,v in actor.state_dict().items()},
                identity=meta['identity'], feature_dim=actor.feature_dim, cache_id=meta['cache_id'],
                epoch=epoch, seed=seed, schema=1, training=training_config))
            continue
        report = evaluate(backend, actor, meta, examples, "validation", baselines=False)
        score = report["mean_normalized_rmse"]["steered"]
        human_score = score
        preservation_score = None
        if preservation_meta:
            auto_report = evaluate(backend, actor, preservation_meta, preservation_validation, "validation", baselines=False)
            preservation_score = auto_report["mean_normalized_rmse"]["steered"]
            score = .5 * (human_score + preservation_score)
        elif trace_validation:
            preservation_score = evaluate_traces(backend, actor, trace_validation)
            score = .5 * (human_score + preservation_score)
        history.append(dict(epoch=epoch, human_noise_mse=human_sum/samples,
                            autonomous_noise_mse=auto_sum/samples if auto else None,
                            validation_decoded_rmse=human_score,
                            preservation_validation_decoded_rmse=preservation_score, selection_score=score))
        write_json(Path(output).with_suffix(".history.json"), history)
        print(history[-1], flush=True)
        improved = score < best
        if checkpoint_selection == 'last' or improved:
            if not all(torch.isfinite(v).all() for v in actor.state_dict().values()):
                raise RuntimeError('Nonfinite actor weights; refusing to save checkpoint')
            save_tensor(output, dict(state_dict={k:v.detach().cpu() for k,v in actor.state_dict().items()},
                identity=meta["identity"], feature_dim=actor.feature_dim, cache_id=meta["cache_id"],
                epoch=epoch, validation_decoded_rmse=human_score, seed=seed, schema=1,
                preservation_validation_decoded_rmse=preservation_score, selection_score=score,
                training=training_config))
        if improved:
            best, stale = score, 0
        else:
            stale += 1
            if checkpoint_selection == 'best' and stale >= patience:
                break
    if not Path(output).exists():
        raise RuntimeError("No finite validation score; no actor produced")
