import io
import json
import threading

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
torch.set_num_threads(2)

from flowdagger_pi05.yam.backend import image_tensor
from flowdagger_pi05.yam.core import SteeringActor, errors, invert, load_actor, polarity, sample, verify_conventions
from flowdagger_pi05.yam.data import windows
from flowdagger_pi05.yam.pipeline import save_tensor
from flowdagger_pi05.yam.serve import ExecutionGuard, Policy, build_app
from flowdagger_pi05.yam.traces import TraceStore, preservation_examples


def conventions():
    return dict(verified=True, dataset_open=0, model_open=1, wire_open=0,
                video_evidence="labeled left/right open/closed fixtures", training_evidence="training fixture",
                karma_revision="fixture-revision", wire_test_evidence="both grippers fixture")


def test_discrete_inverse_coupled_channels():
    generator = torch.Generator().manual_seed(12)
    noise = torch.randn(2, 30, 32, generator=generator)
    times = []
    def velocity(x, t):
        times.append(float(t[0]))
        return .2 * x + .1 * x.roll(1, -1) + t[:, None, None]
    target = sample(velocity, noise)
    forward_times = times.copy()
    times.clear()
    recovered = invert(velocity, target)
    assert times[::5] == list(reversed(forward_times))
    torch.testing.assert_close(recovered, noise, atol=1e-6, rtol=1e-6)
    assert errors(sample(velocity, recovered), target)["accepted"]


def test_right_arm_and_each_gripper_gate():
    for channel in (7, 12, 6, 13):
        target = torch.zeros(1, 30, 32)
        bad = target.clone()
        bad[..., channel] = .06
        assert not errors(bad, target)["accepted"]
    assert not errors(torch.full((1, 30, 32), float("nan")), target)["finite"]


def test_polarity_preserves_joint_order_and_is_involution():
    x = np.arange(14, dtype=np.float32) / 14
    y = polarity(x, 0, 1)
    assert y[6] == 1-x[6] and y[13] == 1-x[13]
    np.testing.assert_array_equal(y[[0,1,2,3,4,5,7,8,9,10,11,12]], x[[0,1,2,3,4,5,7,8,9,10,11,12]])
    np.testing.assert_allclose(polarity(y, 1, 0), x, atol=1e-7)
    for key in ("verified", "video_evidence", "training_evidence", "wire_test_evidence", "karma_revision"):
        invalid = conventions()
        invalid[key] = False if key == "verified" else ""
        with pytest.raises(ValueError):
            verify_conventions(invalid)


def test_boundary_and_gap_exclusion():
    n = 100
    frame = pd.DataFrame(dict(episode_index=[0]*60+[1]*40, frame_index=list(range(60))+list(range(40)),
        timestamp=np.r_[np.arange(60)/30, np.arange(40)/30], intervention=[1]*35+[0]*5+[1]*60, index=range(n)))
    candidates, short = windows(frame)
    assert [(w["episode"], w["frame"]) for w in candidates] == [(0,0),(1,0),(1,10)]
    assert short[0]["frames"] == 20
    frame.loc[80:, "timestamp"] += .1
    assert len(windows(frame)[0]) == 1


def test_actor_unbounded_save_load_and_optimizer(tmp_path):
    actor = SteeringActor(8)
    base = torch.nn.Linear(8, 8).requires_grad_(False)
    before = {k:v.clone() for k,v in base.state_dict().items()}
    patches, state = torch.randn(2,3,16,8), torch.randn(2,14)
    optimizer = torch.optim.AdamW(actor.parameters())
    loss = (actor(patches, state)-10).square().mean()
    loss.backward()
    optimizer.step()
    assert not set(map(id, base.parameters())) & {id(p) for group in optimizer.param_groups for p in group["params"]}
    for key, v in base.state_dict().items():
        torch.testing.assert_close(v, before[key], rtol=0, atol=0)
    with torch.no_grad():
        actor.mlp[-1].bias.fill_(20)
    assert actor(patches, state).max() > 1
    path = tmp_path / "actor.pt"
    save_tensor(path, dict(state_dict=actor.state_dict(), identity={"test":1}, feature_dim=8))
    loaded, _ = load_actor(path, {"test":1}, "cpu")
    torch.testing.assert_close(loaded(patches, state), actor(patches, state), rtol=0, atol=0)
    with pytest.raises(ValueError):
        load_actor(path, {"test":2}, "cpu")


def test_raw_and_jpeg():
    rgb = np.full((10, 20, 3), 128, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG")
    torch.testing.assert_close(image_tensor(rgb), image_tensor(np.frombuffer(buffer.getvalue(), dtype=np.uint8)), atol=.01, rtol=0)
    with pytest.raises(ValueError):
        image_tensor(rgb.astype(float))


class FakeBackend:
    device = "cpu"
    identity = {"checkpoint":{"fixture":"hash"}}
    def prepare(self, obs):
        # Preserve camera order in features so server tests observe all three.
        patches = torch.stack([image_tensor(obs[c+"_cam"]).mean().expand(16,8) for c in ("top","left","right")])[None]
        return dict(patches=patches, state=torch.as_tensor(obs["state"])[None])
    def sample(self, p, noise):
        return noise
    def decode_actions(self, x):
        return x[..., :14]


def test_http_contract_and_traces(tmp_path):
    import json_numpy
    from fastapi.testclient import TestClient
    backend = FakeBackend()
    actor = SteeringActor(8)
    actor_path = tmp_path / "actor.pt"
    identity = dict(backend.identity, conventions=conventions())
    save_tensor(actor_path, dict(state_dict=actor.state_dict(), identity=identity, feature_dim=8))
    policy = Policy(backend, actor_path, conventions(), tmp_path / "traces", "yam")
    client = TestClient(build_app(policy))
    health = client.get("/act").json()
    assert health["normalization_tag"] == "yam" and health["sampler_steps"] == 10
    assert health["norm_tag"] == "yam"
    frame = np.zeros((8,8,3), np.uint8)
    payload = dict(state=np.zeros(14), top_cam=frame, left_cam=frame+30, right_cam=frame+60,
                   instruction="fold the towel", normalization_tag="yam", episode_id="e", dataset_episode=0)
    response = client.post("/act", content=json_numpy.dumps(payload))
    assert response.status_code == 200
    r = json_numpy.loads(response.text)
    assert r["actions"].shape == (30,14) and np.isfinite(r["actions"]).all()
    record = torch.load(tmp_path / "traces" / (r["request_id"]+".pt"), weights_only=True)
    assert record["patches"][0].mean() < record["patches"][1].mean() < record["patches"][2].mean()
    for change in ({"num_steps":11}, {"normalization_tag":"bad"}, {"instruction":"other"}, {"guidance":1}):
        assert client.post("/act", content=json_numpy.dumps(dict(payload, **change))).status_code == 400
    event = dict(kind="executed", episode_id="e", request_id=r["request_id"], indices=list(range(30)), commands=r["actions"].tolist())
    assert client.post("/events", json=event).status_code == 200
    assert client.post("/events", json=dict(kind="outcome", episode_id="e", success=True)).status_code == 200
    assert len(preservation_examples(tmp_path / "traces", identity, [0])) == 1
    guard = ExecutionGuard(r["session"])
    assert guard.accepts(r)
    guard.takeover()
    assert not guard.accepts(r)
    payload["episode_id"] = "takeover"
    r = json_numpy.loads(client.post("/act", content=json_numpy.dumps(payload)).text)
    assert client.post("/events", json=dict(kind="takeover", episode_id="takeover")).status_code == 200
    event.update(episode_id="takeover", request_id=r["request_id"])
    assert client.post("/events", json=event).status_code == 400
    assert client.post("/act", content=json_numpy.dumps(dict(payload, generation=0))).status_code == 400


def test_offline_pipeline_invert_train_evaluate(tmp_path, monkeypatch):
    from flowdagger_pi05.yam import pipeline
    from flowdagger_pi05.yam.core import CONTRACT
    from flowdagger_pi05.yam.data import write_json
    class DatasetFixture:
        def __init__(self, root):
            pass
        def identity(self):
            return {"fixture": "hash"}
    class Backend(FakeBackend):
        def prepare(self, obs, actions=None):
            p = super().prepare(obs)
            p["batch"] = {"action":torch.as_tensor(actions)[None]}
            return p
        def invert(self, p, target):
            return target.clone()
    backend = Backend()
    obs = dict(state=np.zeros(14, np.float32), instruction="fold the towel",
               **{c+"_cam":np.full((8,8,3), i*20, np.uint8) for i,c in enumerate(("top","left","right"))})
    monkeypatch.setattr(pipeline, "Dataset", DatasetFixture)
    monkeypatch.setattr(pipeline, "model_example", lambda *args: (obs, np.zeros((30,14), np.float32), {}))
    manifest = dict(id="fixture", contract=CONTRACT, conventions=conventions(), dataset="fixture",
                    dataset_identity={"fixture":"hash"}, splits=dict(train=[0],validation=[1],test=[2]),
                    windows=[dict(episode=i, frame=0, segment=0, indices=list(range(30)), split=split)
                             for i,split in enumerate(("train","validation","test"))])
    path = tmp_path / "manifest.json"
    write_json(path, manifest)
    pipeline.invert_dataset(backend, path, tmp_path / "cache")
    meta, examples = pipeline.read_cache(tmp_path / "cache")
    assert len(examples) == 3 and all(e["normalized"]["accepted"] for e in examples)
    actor_path = tmp_path / "actor.pt"
    pipeline.train(backend, tmp_path / "cache", actor_path, epochs=1,
                   learning_rate=3e-5, weight_decay=1e-3)
    actor, saved = load_actor(actor_path, meta["identity"], "cpu")
    assert saved["training"]["learning_rate"] == 3e-5
    assert saved["training"]["weight_decay"] == 1e-3
    report = pipeline.evaluate(backend, actor, meta, examples, "test")
    assert set(report["mean_normalized_rmse"]) == {"steered","zero","random"}
    assert report["mean_normalized_rmse"]["zero"] == 0
    assert report["examples"][0]["window"]["episode"] == 2
