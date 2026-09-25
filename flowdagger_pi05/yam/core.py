"""Versioned contracts, exact Euler inversion, and the trainable actor."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

CAMERAS = ("top", "left", "right")
DATA_CAMERAS = ("top", "left_wrist", "right_wrist")
JOINTS = tuple([f"left_joint_{i}.pos" for i in range(6)] + ["left_gripper.pos"] +
               [f"right_joint_{i}.pos" for i in range(6)] + ["right_gripper.pos"])
CONTRACT = dict(schema=1, horizon=30, latent_dim=32, action_dim=14, steps=10,
                task="fold the towel", completion="zero-noise-native-terminal-unused-channels",
                fixed_point_iterations=5, grid="1.0 + step * (-1.0 / 10)",
                features="native-image-embeddings-adaptive-average-4x4", rtc=False)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_identity(root):
    root = Path(root)
    names = ["config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"]
    names += sorted(p.name for p in root.glob("policy_*processor_step_*.safetensors"))
    return {name: file_hash(root / name) for name in names}


def verify_conventions(c):
    for key in ("dataset_open", "model_open", "wire_open"):
        if c.get(key) not in (0, 1) or isinstance(c.get(key), bool):
            raise ValueError(f"Explicit {key}=0 or 1 is required")
    for key in ("video_evidence", "training_evidence", "karma_revision", "wire_test_evidence"):
        if not isinstance(c.get(key), str) or not c[key].strip():
            raise ValueError(f"Gripper audit requires {key}")
    if c.get("verified") is not True:
        raise ValueError("Gripper conventions are not verified; complete the labeled-frame and wire audit")
    return c


def polarity(x, source_open, destination_open):
    x = x.clone() if isinstance(x, torch.Tensor) else np.array(x, copy=True)
    if source_open != destination_open:
        x[..., [6, 13]] = 1 - x[..., [6, 13]]
    return x


@torch.no_grad()
def sample(velocity, noise):
    x = noise.float().clone()
    dt = -1.0 / 10
    for step in range(10):
        t = torch.tensor(1.0 + step * dt, dtype=torch.float32, device=x.device).expand(x.shape[0])
        x = x + dt * velocity(x, t).float()
    return x


@torch.no_grad()
def invert(velocity, target):
    y = target.float().clone()
    dt = -1.0 / 10
    for step in reversed(range(10)):
        t = torch.tensor(1.0 + step * dt, dtype=torch.float32, device=y.device).expand(y.shape[0])
        z = y.clone()
        for _ in range(5):
            z = y - dt * velocity(z, t).float()
        y = z
    return y


def errors(prediction, target):
    e = (prediction[..., :14].float() - target[..., :14].float()).cpu()
    if not torch.isfinite(e).all():
        return {"finite": False, "accepted": False}
    channel = e.square().reshape(-1, 14).mean(0).sqrt()
    rmse = e.square().mean().sqrt().item()
    return dict(finite=True, rmse=rmse, per_channel=channel.tolist(),
                left_arm=e[..., :6].square().mean().sqrt().item(),
                right_arm=e[..., 7:13].square().mean().sqrt().item(),
                grippers=e[..., [6, 13]].square().mean().sqrt().item(),
                accepted=rmse <= .03 and channel.max().item() <= .05)


class SteeringActor(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.projection = nn.Linear(feature_dim, 64)
        self.mlp = nn.Sequential(nn.Linear(3 * 16 * 64 + 14, 512), nn.ReLU(),
                                 nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 960))

    def forward(self, patches, state):
        if patches.shape[1:3] != (3, 16) or state.shape[-1] != 14:
            raise ValueError("Expected three ordered 4x4 camera grids and 14 state channels")
        return self.mlp(torch.cat((self.projection(patches.float()).flatten(1), state.float()), -1)).view(-1, 30, 32)


def load_actor(path, identity, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["identity"] != identity:
        raise ValueError("Actor/base/processor/convention identity mismatch")
    actor = SteeringActor(saved["feature_dim"]).to(device)
    actor.load_state_dict(saved["state_dict"], strict=True)
    return actor.eval(), saved
