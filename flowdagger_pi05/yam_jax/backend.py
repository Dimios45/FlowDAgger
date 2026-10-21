"""Exact checkpoint processing, cached JAX flow, and ordered spatial features.

Torch is used for the small CPU actor and the existing portable cache format.
The frozen foundation model and inversion stay in JAX on its selected device.
"""
import importlib.metadata
import inspect
from pathlib import Path

import numpy as np
import torch

from flowdagger_pi05.yam.backend import image_tensor
from flowdagger_pi05.yam.core import CONTRACT, file_hash


def tensor(x):
    return torch.from_numpy(np.array(x, dtype=np.float32, copy=True))


class JaxBackend:
    device = "cpu"
    supports_iteration_sweep = True

    def __init__(self, checkpoint, tokenizer, dtype="float32"):
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from expo_ft.conversion import yam_loader, yam_pi05
        from openpi.models.pi0 import make_attn_mask
        from openpi.models import model as model_module

        self.processor = yam_loader.YamProcessor(checkpoint, tokenizer)
        model = yam_pi05.load_model(checkpoint, dtype)
        model.eval()
        self.graph, self.weights = nnx.split(model)
        self.as_observation = yam_loader.as_observation
        root = Path(checkpoint)
        # Hash the actual parameter files, not just a mutable path/manifest.
        files = sorted(p for p in root.rglob("*") if p.is_file() and
                       (p.relative_to(root).parts[0] == "params" or p.parent == root))
        self.identity = dict(
            checkpoint={str(p.relative_to(root)): file_hash(p) for p in files},
            tokenizer={p.name: file_hash(p) for p in sorted(Path(tokenizer).iterdir()) if p.is_file()},
            contract=CONTRACT, backend="jax-yam-v1", dtype=dtype,
            features="ordered-prefix-image-embeddings-4x4-v1",
            implementation={p.name: file_hash(p) for p in [Path(__file__),
                Path(inspect.getfile(yam_loader)), Path(inspect.getfile(yam_pi05)),
                Path(inspect.getfile(type(model))), Path(inspect.getfile(model_module))]},
            runtime={k: importlib.metadata.version(k) for k in ("jax", "jaxlib", "flax", "torch", "transformers")})

        def compile_fn(fn):
            def highest(*args):
                with jax.default_matmul_precision("highest"):
                    return fn(*args)
            return jax.jit(highest)

        def encode(weights, obs):
            m = nnx.merge(self.graph, weights)
            tokens, mask, ar = m.embed_prefix(obs)
            _, cache = m.PaliGemma.llm([tokens, None], mask=make_attn_mask(mask, ar),
                                     positions=jnp.cumsum(mask, axis=1)-1)
            # Three ordered 16x16 patch grids; pool each to 4x4, never mix cameras.
            grids = tokens[:, :3*256].reshape(tokens.shape[0], 3, 4, 4, 4, 4, -1)
            grids = grids.mean(axis=(3, 5)).reshape(tokens.shape[0], 3, 16, -1)
            return mask, cache, grids

        def velocity(weights, obs, mask, cache, x, t):
            m = nnx.merge(self.graph, weights)
            suffix, sm, sa, cond = m.embed_suffix(obs, x, jnp.broadcast_to(t, (x.shape[0],)))
            attn = jnp.concatenate((jnp.broadcast_to(mask[:, None, :],
                        (x.shape[0], suffix.shape[1], mask.shape[1])), make_attn_mask(sm, sa)), -1)
            (_, hidden), _ = m.PaliGemma.llm([None, suffix], mask=attn,
                positions=mask.sum(-1)[:, None]+jnp.cumsum(sm, axis=1)-1,
                kv_cache=cache, adarms_cond=[None, cond])
            return m.action_out_proj(hidden[:, -30:]).astype(jnp.float32)

        def sample(weights, obs, mask, cache, noise):
            def step(i, x):
                t = jnp.float32(1.) + i*jnp.float32(-.1)
                return x - jnp.float32(.1)*velocity(weights, obs, mask, cache, x, t)
            return jax.lax.fori_loop(0, 10, step, noise.astype(jnp.float32))

        def invert(weights, obs, mask, cache, target, iterations):
            def step(y, i):
                t = jnp.float32(1.) + (9-i)*jnp.float32(-.1)
                def fp(_, z):
                    return y + jnp.float32(.1)*velocity(weights, obs, mask, cache, z, t)
                z = jax.lax.fori_loop(0, iterations, fp, y)
                residual = z - jnp.float32(.1)*velocity(weights, obs, mask, cache, z, t)-y
                return z, jnp.sqrt(jnp.mean(residual**2))
            return jax.lax.scan(step, target.astype(jnp.float32), jnp.arange(10))

        self._encode = compile_fn(encode)
        self._sample = compile_fn(sample)
        self._invert = compile_fn(invert)
        self._native = compile_fn(lambda w, obs, n: nnx.merge(self.graph, w).sample_actions(
            jax.random.key(0), obs, noise=n[:, None], num_steps=10))

    def prepare(self, observation, actions=None):
        if observation.get("instruction", CONTRACT["task"]) != CONTRACT["task"]:
            raise ValueError("This actor only supports 'fold the towel'")
        images = {c: np.rint(image_tensor(observation[c+"_cam"]).permute(1, 2, 0).numpy()*255).astype(np.uint8)
                  for c in ("top", "left", "right")}
        data = self.processor.prepare_numpy(images, observation["state"], CONTRACT["task"])
        obs = self.as_observation(data)
        mask, cache, patches = self._encode(self.weights, obs)
        batch = {}
        if actions is not None:
            actions = np.asarray(actions, dtype=np.float32)
            if actions.shape != (30, 14) or not np.isfinite(actions).all():
                raise ValueError("Expected finite 30x14 absolute targets")
            batch["action"] = tensor(self.processor.normalize(actions, "action"))[None]
        return dict(obs=obs, mask=mask, cache=cache, patches=tensor(patches),
                    state=tensor(data["state"][:, :14]), batch=batch)

    def _args(self, p, x):
        import jax.numpy as jnp
        x = np.asarray(x.detach().cpu() if isinstance(x, torch.Tensor) else x, dtype=np.float32)
        if x.shape != (1, 30, 32) or not np.isfinite(x).all():
            raise ValueError("Expected finite (1,30,32) latent")
        return self.weights, p["obs"], p["mask"], p["cache"], jnp.asarray(x)

    def sample(self, p, noise):
        return tensor(self._sample(*self._args(p, noise)))

    def invert(self, p, target, iterations=5):
        if not 1 <= iterations <= 100:
            raise ValueError("Use 1..100 fixed-point iterations")
        noise, residual = self._invert(*self._args(p, target), iterations)
        self.last_inversion_diagnostics = dict(iterations=iterations, fixed_point_residuals=np.asarray(residual).tolist())
        return tensor(noise)

    def native_sample(self, p, noise):
        args = self._args(p, noise)
        return tensor(self._native(args[0], args[1], args[-1]))

    def decode_actions(self, latent):
        return tensor(self.processor.unnormalize_actions(np.asarray(latent)))
