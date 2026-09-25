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
        decoded = backend.sample(p, noise)
        normalized = errors(decoded, target)
        physical = errors(backend.decode_actions(decoded), torch.from_numpy(actions)[None])
        normalized["accepted"] &= bool(torch.isfinite(noise).all() and torch.isfinite(decoded).all() and physical["finite"])
        save_tensor(path, dict(cache_id=metadata["cache_id"], window=window, alignment=alignment,
            patches=p["patches"].cpu()[0], state=p["state"].cpu()[0], noise=noise.cpu()[0],
            target=target.cpu()[0], normalized=normalized, physical=physical))
        print(json.dumps(dict(example=number, **normalized)), flush=True)


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


@torch.no_grad()
def evaluate(backend, actor, meta, examples, split, baselines=True):
    m = meta["manifest"]
    dataset = Dataset(m["dataset"], **({"episodes": m["dataset_episodes"]} if "dataset_episodes" in m else {}))
    results = []
    generator = torch.Generator(device=backend.device).manual_seed(0)
    for e in examples:
        if e["window"]["split"] != split:
            continue
        obs, actions, _ = model_example(dataset, e["window"], meta["identity"]["conventions"])
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


def train(backend, cache, output, epochs=250, patience=20, batch_size=64, seed=0, autonomous=None,
          learning_rate=1e-4, weight_decay=1e-4, preservation_cache=None, checkpoint_selection='best',
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
    if Dataset(meta["manifest"]["dataset"]).identity() != meta["manifest"]["dataset_identity"]:
        raise ValueError("Dataset changed since inversion")
    actor = SteeringActor(human[0]["patches"].shape[-1]).to(backend.device)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=learning_rate, weight_decay=weight_decay)
    auto = []
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
        auto = preservation_examples(autonomous, meta["identity"], meta["manifest"]["splits"]["train"])
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
        loss_sum, samples = 0., 0
        for batch in DataLoader(human, batch_size=human_batch_size, shuffle=True, collate_fn=lambda rows: rows):
            def loss(rows):
                patches, state, noise = [torch.stack([r[k] for r in rows]).to(backend.device) for k in ("patches", "state", "noise")]
                return (actor(patches, state) - noise).square().mean()
            value = loss(batch)
            if auto:
                indices = torch.randint(len(auto), (len(batch),)).tolist()
                value = .5 * value + .5 * loss([auto[i] for i in indices])
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
        history.append(dict(epoch=epoch, validation_decoded_rmse=human_score,
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
