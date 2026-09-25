"""Explicit real-checkpoint acceptance checks (requires the native runtime)."""
import torch

from .core import errors
from .data import Dataset, model_example
from .pipeline import read_cache


@torch.no_grad()
def verify(backend, cache):
    meta, examples = read_cache(cache, require_complete=False)
    if not examples:
        raise ValueError("No accepted examples to verify")
    if dict(backend.identity, conventions=meta["identity"]["conventions"]) != meta["identity"]:
        raise ValueError("Verification backend/cache mismatch")
    m = meta["manifest"]
    dataset = Dataset(m["dataset"], **({"episodes": m["dataset_episodes"]} if "dataset_episodes" in m else {}))
    obs, actions, _ = model_example(dataset, examples[0]["window"], meta["identity"]["conventions"])
    p = backend.prepare(obs, actions)
    torch.testing.assert_close(backend.decode_actions(p["batch"]["action"]), torch.tensor(actions)[None], atol=1e-5, rtol=1e-5)
    noise = torch.randn(1, 30, 32, device=backend.device, generator=torch.Generator(device=backend.device).manual_seed(0))
    generated = backend.sample(p, noise)
    torch.testing.assert_close(generated, backend.native_sample(p, noise), atol=1e-5, rtol=1e-5)
    reconstructed = backend.sample(p, backend.invert(p, generated))
    metrics = errors(reconstructed, generated)
    if not metrics["accepted"]:
        raise AssertionError(f"Generated-action inversion failed: {metrics}")
    assert all(not param.requires_grad for param in backend.policy.parameters())
    torch.testing.assert_close(p["patches"][0].cpu(), examples[0]["patches"], atol=1e-5, rtol=1e-5)
    return dict(processor_round_trip=True, native_sampler_parity=True, frozen_base=True,
                training_serving_features=True, generated_inversion=metrics,
                expert_inversion=examples[0]["normalized"])
