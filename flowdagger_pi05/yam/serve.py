"""Karma-compatible /act plus an explicit collection-event extension."""
import threading
import time

import numpy as np
import torch

from .core import CONTRACT, digest, file_hash, load_actor, polarity, verify_conventions
from .traces import TraceStore


class Policy:
    def __init__(self, backend, actor_path, conventions, trace_dir, normalization_tag):
        self.backend = backend
        self.conventions = verify_conventions(conventions)
        self.identity = dict(backend.identity, conventions=conventions)
        self.actor, _ = load_actor(actor_path, self.identity, backend.device)
        self.actor.requires_grad_(False)
        self.actor_version = file_hash(actor_path)
        self.traces = TraceStore(trace_dir)
        self.normalization_tag = normalization_tag
        self.lock = threading.Lock()

    def health(self):
        return dict(status="ok", policy_type="pi05-flowdagger-yam", checkpoint=self.identity["checkpoint"],
                    actor_version=self.actor_version, normalization_tag=self.normalization_tag,
                    norm_tag=self.normalization_tag,
                    unnorm_key=self.normalization_tag, state_dim=14, action_dim=14, num_cameras=3,
                    action_horizon=30, n_action_steps=30, num_steps=10, sampler_steps=10,
                    task=CONTRACT["task"], action_mode="absolute", device=self.backend.device,
                    generation=self.traces.generation, session=self.traces.session)

    @torch.no_grad()
    def predict(self, payload):
        if payload.get("num_steps", 10) != 10:
            raise ValueError("This actor requires exactly ten sampler steps")
        tag = payload.get("normalization_tag", payload.get("unnorm_key"))
        if tag != self.normalization_tag:
            raise ValueError("Missing or incompatible normalization tag")
        if any(k in payload for k in ("rtc", "guidance", "prev_chunk_left_over", "inference_delay")):
            raise ValueError("RTC and guidance are disabled")
        if payload.get("instruction") != CONTRACT["task"]:
            raise ValueError("This actor only supports 'fold the towel'")
        with self.lock:
            if "generation" in payload and payload["generation"] != self.traces.generation:
                raise ValueError("Stale request generation")
            if "session" in payload and payload["session"] != self.traces.session:
                raise ValueError("Stale server session")
            episode = payload.get("episode_id", "unassigned")
            if not isinstance(episode, str) or not episode or episode in self.traces.closed_episodes:
                raise ValueError("Invalid or closed episode")
            start = time.perf_counter()
            obs = {k: payload[k] for k in ("state", "top_cam", "left_cam", "right_cam", "instruction")}
            obs["state"] = polarity(np.asarray(obs["state"], dtype=np.float32),
                                     self.conventions["wire_open"], self.conventions["model_open"])
            p = self.backend.prepare(obs)
            noise = self.actor(p["patches"], p["state"])
            actions = self.backend.decode_actions(self.backend.sample(p, noise))[0].cpu()
            actions = polarity(actions, self.conventions["model_open"], self.conventions["wire_open"])
            if actions.shape != (30, 14) or not torch.isfinite(actions).all():
                raise RuntimeError("Sampler produced invalid actions")
            record = dict(identity=self.identity, actor_version=self.actor_version,
                episode_id=episode, dataset_episode=payload.get("dataset_episode"),
                observation={k: torch.as_tensor(np.asarray(v).copy()) if k != "instruction" else v for k, v in obs.items()},
                patches=p["patches"].cpu()[0], state=p["state"].cpu()[0], noise=noise.cpu()[0], actions=actions)
            request_id = self.traces.record(record)
            return dict(actions=actions.numpy(), dt_ms=(time.perf_counter()-start)*1000,
                        request_id=request_id, chunk_id=request_id, generation=self.traces.generation,
                        session=self.traces.session, actor_version=self.actor_version)


def build_app(policy):
    import json_numpy
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response
    from starlette.concurrency import run_in_threadpool
    app = FastAPI(title="Frozen pi0.5 YAM FlowDAgger")

    @app.get("/act")
    @app.get("/healthz")
    def health():
        return policy.health()

    # Request must be in globals for FastAPI annotation resolution.
    globals()["Request"] = Request

    @app.post("/act")
    async def act(request: Request):
        try:
            payload = json_numpy.loads((await request.body()).decode())
            result = await run_in_threadpool(policy.predict, payload)
            return Response(json_numpy.dumps(result), media_type="application/json")
        except (KeyError, ValueError, TypeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.post("/events")
    async def events(request: Request):
        try:
            event = await request.json()
            def record():
                with policy.lock:
                    return policy.traces.event(event)
            return await run_in_threadpool(record)
        except (KeyError, ValueError, TypeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
    return app


class ExecutionGuard:
    """Client-side gate: call takeover immediately, then discard prefetched chunks.

    An HTTP server cannot retract actions already received by a robot client.
    The client must use this gate before every execution, and report /events.
    """
    def __init__(self, session, generation=0):
        self.session, self.generation = session, generation

    def takeover(self):
        self.generation += 1

    def accepts(self, response):
        return response.get("session") == self.session and response.get("generation") == self.generation
