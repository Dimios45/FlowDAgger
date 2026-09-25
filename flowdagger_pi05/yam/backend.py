"""Native LeRobot adapter. Imports are lazy so preparation needs no VLM runtime."""
import inspect
import importlib.metadata
import io
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .core import CAMERAS, CONTRACT, JOINTS, checkpoint_identity, file_hash, invert, sample


def image_tensor(value):
    a = np.asarray(value)
    if a.ndim == 1 and a.dtype == np.uint8:
        a = np.asarray(Image.open(io.BytesIO(a.tobytes())).convert("RGB"))
    if a.ndim != 3 or a.shape[-1] != 3 or a.dtype != np.uint8:
        raise ValueError("Expected RGB HWC uint8 or encoded JPEG uint8 array")
    return torch.from_numpy(a.copy()).permute(2, 0, 1).float() / 255


class NativeBackend:
    def __init__(self, checkpoint, tokenizer, device="cuda"):
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.factory import make_pre_post_processors
        from safetensors.torch import load_file
        root = Path(checkpoint)
        config = PI05Config.from_pretrained(str(root))
        if (config.chunk_size, config.max_action_dim, config.num_inference_steps) != (30, 32, 10):
            raise ValueError("Actor requires 30x32 latent and ten sampler steps")
        if config.use_relative_actions or config.rtc_config or tuple(config.action_feature_names) != JOINTS:
            raise ValueError("Expected absolute YAM joint order with RTC disabled")
        if list(config.image_features) != ["observation.images." + c for c in CAMERAS]:
            raise ValueError("Checkpoint camera order differs from top, left, right")
        if tuple(config.output_features["action"].shape) != (14,):
            raise ValueError("Expected 14 real action dimensions")
        for key in ("use_visual_memory", "use_proprioceptive_memory"):
            if getattr(config, key, False):
                raise ValueError("Temporal memory is not supported in this actor contract")
        config.device, config.compile_model, config.gradient_checkpointing = device, False, False
        self.pre, self.post = make_pre_post_processors(config, pretrained_path=str(root),
            preprocessor_overrides={"device_processor": {"device": device},
                                     "tokenizer_processor": {"tokenizer_name": str(tokenizer)}})
        self.policy = PI05Policy(config)
        weights = self.policy._fix_pytorch_state_dict_keys(load_file(str(root / "model.safetensors")), config)
        self.policy.load_state_dict({k if k.startswith("model.") else "model." + k: v for k, v in weights.items()}, strict=True)
        del weights
        self.policy.to(device).eval().requires_grad_(False)
        self.model, self.device = self.policy.model, device
        self.identity = dict(adapter={p.name: file_hash(p) for p in (Path(__file__), Path(__file__).with_name("core.py"))},
                             contract=CONTRACT, checkpoint=checkpoint_identity(root), dtype=config.dtype,
                             implementation={str(p.relative_to(Path(inspect.getfile(PI05Policy)).parents[2])): file_hash(p)
                                 for sub in ("policies/pi05", "policies/common", "processor")
                                 for p in sorted((Path(inspect.getfile(PI05Policy)).parents[2] / sub).rglob("*.py"))},
                             runtime={name: importlib.metadata.version(name) for name in ("torch", "transformers", "safetensors")},
                             tokenizer={p.name: file_hash(p) for p in sorted(Path(tokenizer).glob("*")) if p.is_file()})
        if not self.identity["tokenizer"]:
            raise ValueError("An explicit local saved tokenizer directory is required")

    @torch.no_grad()
    def prepare(self, observation, actions=None):
        state = np.asarray(observation["state"], dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("Expected 14 finite model-convention state values")
        if observation.get("instruction", CONTRACT["task"]) != CONTRACT["task"]:
            raise ValueError("This actor only supports 'fold the towel'")
        batch = {"observation.state": torch.from_numpy(state.copy()), "task": CONTRACT["task"]}
        for c in CAMERAS:
            batch["observation.images." + c] = image_tensor(observation[c + "_cam"])
        if actions is not None:
            a = torch.as_tensor(actions, dtype=torch.float32)
            if a.shape != (30, 14) or not torch.isfinite(a).all():
                raise ValueError("Expected 30x14 finite commanded targets")
            batch["action"] = a.unsqueeze(0)
        batch = self.pre(batch)
        images, image_masks = self.policy._preprocess_images(batch)
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks, prepare_attention_masks_4d
        states, state_masks = self.policy._prepare_memory_states(batch)
        args = (images, image_masks, batch["observation.language.tokens"], batch["observation.language.attention_mask"])
        emb, pad, att = self.model.embed_prefix(*args, states, state_masks)
        self.model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        _, cache = self.model.paligemma_with_expert.forward(
            attention_mask=prepare_attention_masks_4d(make_att_2d_masks(pad, att)),
            position_ids=torch.cumsum(pad, dim=1) - 1, past_key_values=None,
            inputs_embeds=[emb, None], use_cache=True)
        grids = []
        for img in images:
            features = self.model.paligemma_with_expert.embed_image(img).float()
            side = math.isqrt(features.shape[1])
            if side * side != features.shape[1]:
                raise ValueError("Image feature sequence is not a square patch grid")
            grid = features.transpose(1, 2).reshape(features.shape[0], -1, side, side)
            grids.append(F.adaptive_avg_pool2d(grid, (4, 4)).flatten(2).transpose(1, 2))
        return dict(batch=batch, pad=pad, cache=cache, patches=torch.stack(grids, 1),
                    state=batch["observation.state"][..., :14], native_args=args,
                    states=states, state_masks=state_masks)

    def velocity(self, prepared):
        return lambda x, t: self.model.denoise_step(prepared["pad"], prepared["cache"], x, t)

    def sample(self, prepared, noise):
        return sample(self.velocity(prepared), noise)

    def invert(self, prepared, target):
        return invert(self.velocity(prepared), target)

    @torch.no_grad()
    def decode_actions(self, latent):
        return self.post(latent[..., :14]).float()

    @torch.no_grad()
    def native_sample(self, prepared, noise):
        return self.model.sample_actions(*prepared["native_args"], states=prepared["states"],
                                        state_masks=prepared["state_masks"], noise=noise, num_steps=10)
