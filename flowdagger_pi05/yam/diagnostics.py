"""Measure reconstruction, conditioning and utility in the same action space."""
import numpy as np
import torch
import time
from .core import errors


def noise_stats(noise):
    x = noise.detach().float().cpu()
    if not torch.isfinite(x).all():
        return dict(finite=False)
    return dict(finite=True, rms=float(x.square().mean().sqrt()),
                p99_abs=float(torch.quantile(x.abs().flatten(), .99)),
                max_abs=float(x.abs().max()),
                physical_channel_rms=float(x[..., :14].square().mean().sqrt()),
                other_channel_rms=float(x[..., 14:].square().mean().sqrt()))


@torch.no_grad()
def diagnose(backend, observation, actions=None, seed=0, random_trials=3, perturbation=.001,
             iteration_sweep=False):
    """No robot commands. A real human window is needed to assess relevance.

    Finite-difference amplification is a local diagnostic, not a stability proof.
    Joint errors are radians, gripper errors are normalized gripper units.
    """
    if random_trials < 1 or perturbation <= 0:
        raise ValueError("Positive random_trials and perturbation required")
    p = backend.prepare(observation, actions)
    gen = torch.Generator(device=backend.device).manual_seed(seed)
    random = lambda: torch.randn((1,30,32), generator=gen, device=backend.device)
    w = random()
    started = time.perf_counter()
    generated = backend.sample(p, w)
    sample_seconds = time.perf_counter()-started
    native = backend.native_sample(p, w)
    started = time.perf_counter()
    recovered = backend.invert(p, generated)
    inversion_seconds = time.perf_counter()-started
    if not torch.isfinite(recovered).all():
        return dict(generated_reconstruction={'finite':False,'accepted':False},
                    generated_noise=noise_stats(recovered),human_relevance_measured=False)
    reconstructed = backend.sample(p, recovered)
    report = dict(seed=seed, native_max_abs=float((generated-native).abs().max()),
        timing=dict(sample_seconds=sample_seconds, generated_inverse_seconds=inversion_seconds,
                    includes_first_call_compilation=True),
        generated_reconstruction=errors(reconstructed, generated),
        recovered_noise_rmse=float((recovered-w).square().mean().sqrt()),
        generated_noise=noise_stats(recovered),
        fixed_point=getattr(backend, "last_inversion_diagnostics", {}))
    def sweep(target):
        if not getattr(backend, 'supports_iteration_sweep', False):
            raise ValueError('Iteration sweep is supported by the JAX backend')
        results = {}
        for n in (5,10,20):
            noise = backend.invert(p, target, iterations=n)
            results[str(n)] = dict(noise=noise_stats(noise),
                reconstruction=errors(backend.sample(p,noise),target) if torch.isfinite(noise).all() else {'finite':False,'accepted':False},
                **backend.last_inversion_diagnostics)
        return results
    if actions is None:
        report["human_relevance_measured"] = False
        if iteration_sweep:
            report['iteration_sweep'] = sweep(generated)
        return report
    zero = backend.sample(p, torch.zeros_like(w))
    target = zero.clone()
    target[..., :14] = p["batch"]["action"]
    started = time.perf_counter()
    inv = backend.invert(p, target)
    report['timing']['target_inverse_seconds'] = time.perf_counter()-started
    residual = getattr(backend, "last_inversion_diagnostics", {})
    if not torch.isfinite(inv).all():
        report['human_relevance_measured'] = True
        report['human'] = dict(reconstruction={'finite':False,'accepted':False}, noise=noise_stats(inv), fixed_point=residual)
        return report
    decoded = backend.sample(p, inv)
    perturbed = target.clone()
    direction = random()[..., :14]
    direction = direction / direction.square().mean().sqrt()
    perturbed[..., :14] += perturbation*direction
    inv_perturbed = backend.invert(p, perturbed)
    forward_perturbed = backend.sample(p, inv + perturbation*random())
    random_errors = [errors(backend.sample(p, random()), target)["rmse"] for _ in range(random_trials)]
    physical = backend.decode_actions(decoded)-torch.as_tensor(actions, device=backend.device)[None]
    channel = physical.square().mean((0,1)).sqrt().cpu().tolist()
    report["human_relevance_measured"] = True
    report["human"] = dict(reconstruction=errors(decoded, target), noise=noise_stats(inv),
        fixed_point=residual, physical_per_channel_rmse=channel,
        zero_rmse=errors(zero, target)["rmse"], random_rmse=random_errors,
        target_to_noise_amplification=float((inv_perturbed-inv).square().mean().sqrt()/perturbation) if torch.isfinite(inv_perturbed).all() else None,
        noise_to_action_amplification=float((forward_perturbed[...,:14]-decoded[...,:14]).square().mean().sqrt()/perturbation),
        perturbed_target_reconstruction=errors(backend.sample(p, inv_perturbed), perturbed) if torch.isfinite(inv_perturbed).all() else {'finite':False,'accepted':False})
    if iteration_sweep:
        report['iteration_sweep'] = sweep(target)
    return report


def summarize(examples):
    """Include rejected samples in denominators; never hide difficult corrections."""
    if not examples:
        raise ValueError("No inversion examples")
    accepted = [e for e in examples if e["normalized"].get("accepted", False)]
    finite = [e["normalized"]["rmse"] for e in examples if e["normalized"].get("finite")]
    return dict(total=len(examples), accepted=len(accepted), acceptance_rate=len(accepted)/len(examples),
                rmse_quantiles=dict(zip(("p50", "p90", "p99"), np.quantile(finite, [.5,.9,.99]).tolist())) if finite else {},
                per_episode={str(ep): dict(total=len(rows), accepted=sum(e["normalized"].get("accepted", False) for e in rows))
                    for ep in sorted({e["window"]["episode"] for e in examples})
                    for rows in [[e for e in examples if e["window"]["episode"] == ep]]})
