"""Append-only collection records; only proven executed autonomous chunks train."""
import json
import os
from pathlib import Path
import uuid

import torch

from .core import digest
from .pipeline import save_tensor


class TraceStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.generation = 0
        self.session = uuid.uuid4().hex
        self.requests = {}
        self.closed_episodes = set()
        self.discarded = set()
        self.executed = {}
        self.modified = set()
        self.event_ids = {}

    def record(self, record):
        request_id = uuid.uuid4().hex
        record = dict(record, request_id=request_id, generation=self.generation, session=self.session)
        save_tensor(self.root / (request_id + ".pt"), record)
        self.requests[request_id] = record
        return request_id

    def event(self, event):
        event_id = event.get("event_id")
        if event_id is not None and (not isinstance(event_id,str) or not event_id):
            raise ValueError('event_id must be a nonempty string')
        if event_id in self.event_ids:
            prior, result = self.event_ids[event_id]
            if prior != event:
                raise ValueError("Event ID reused with different content")
            return result
        if "session" in event and event["session"] != self.session:
            raise ValueError("Stale server session")
        kind = event.get("kind")
        if kind not in ("executed", "takeover", "handback", "discarded", "outcome"):
            raise ValueError("Unknown trace event")
        episode = event.get("episode_id")
        if not isinstance(episode, str) or not episode:
            raise ValueError("Trace events require episode_id")
        if episode in self.closed_episodes:
            raise ValueError("Episode is already closed")
        modified = False
        if kind in ("executed", "discarded"):
            record = self.requests.get(event.get("request_id"))
            if record is None or record["episode_id"] != episode:
                raise ValueError("Unknown request or episode mismatch")
            if kind == "executed":
                if event["request_id"] in self.discarded:
                    raise ValueError("Discarded request")
                if record["generation"] != self.generation:
                    raise ValueError("Stale request after takeover")
                indices = event.get("indices", [])
                if not indices or any(type(i) is not int or i < 0 or i >= 30 for i in indices) or len(set(indices)) != len(indices):
                    raise ValueError("Invalid executed chunk indices")
                previous = self.executed.get(event["request_id"], [])
                if indices != list(range(len(previous), len(previous) + len(indices))):
                    raise ValueError("Execution indices must be contiguous, ordered, and never repeated")
                commands = torch.as_tensor(event.get("commands"), dtype=torch.float32)
                expected = record["actions"][indices]
                if commands.shape != expected.shape or not torch.isfinite(commands).all():
                    raise ValueError("Invalid executed commands")
                # Record the truth, but never train an exact-noise anchor from
                # materially changed commands (including gripper clamps).
                if not torch.allclose(commands, expected, atol=1e-6, rtol=0):
                    modified = True
        if kind == 'outcome' and type(event.get('success')) is not bool:
            raise ValueError('Outcome needs boolean success')
        next_generation = self.generation + (kind == 'takeover')
        line = json.dumps(dict(event, session=self.session, generation=next_generation), allow_nan=False) + '\n'
        # Persist before advancing in-memory execution/generation state.
        with (self.root / 'events.jsonl').open('a') as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        if modified:
            self.modified.add(event['request_id'])
        if kind == "executed":
            self.executed.setdefault(event["request_id"], []).extend(event["indices"])
        if kind == "discarded":
            self.discarded.add(event["request_id"])
        if kind == "takeover":
            self.generation += 1
        if kind == "outcome":
            self.closed_episodes.add(episode)
        result = dict(generation=self.generation, session=self.session)
        if event_id:
            self.event_ids[event_id] = (dict(event), result)
        return result


def preservation_examples(root, identity, train_episodes, allow_empty=False, records=False):
    root = Path(root)
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    unique, ids = [], {}
    for event in events:
        eid = event.get('event_id')
        if eid and eid in ids:
            if event != ids[eid]:
                raise ValueError('Conflicting duplicate event ID in journal')
            continue
        if eid:
            ids[eid] = event
        unique.append(event)
    events = unique
    result = []
    for path in sorted(root.glob("*.pt")):
        r = torch.load(path, weights_only=True)
        if r["identity"] != identity:
            raise ValueError("Autonomous trace identity mismatch")
        if r.get("dataset_episode") not in train_episodes:
            continue  # Must assign new episodes to the recorded split before using them.
        relevant = [e for e in events if e["session"] == r["session"] and e["episode_id"] == r["episode_id"]]
        if any(e["kind"] == "takeover" for e in relevant):
            continue
        outcomes = [e for e in relevant if e["kind"] == "outcome"]
        if len(outcomes) != 1 or not outcomes[0]["success"]:
            continue
        request = [e for e in relevant if e.get("request_id") == r["request_id"]]
        if any(e["kind"] == "discarded" for e in request):
            continue
        # Event retries may appear in imported legacy logs; deduplicate IDs.
        seen = set()
        execution = []
        for e in request:
            if e['kind'] != 'executed':
                continue
            if e.get('event_id') and e['event_id'] in seen:
                continue
            seen.add(e.get('event_id'))
            execution.append(e)
        indices = [i for e in execution for i in e["indices"]]
        if indices != list(range(30)):
            continue
        if any(not torch.allclose(torch.tensor(e['commands'], dtype=torch.float32),
                    r['actions'][e['indices']], atol=1e-6, rtol=0) for e in execution):
            continue
        result.append(r if records else dict({k: r[k] for k in ("patches", "state", "noise")},
            episode=f"{r['session']}:{r['dataset_episode']}"))
    if not result and not allow_empty:
        raise ValueError("No complete successful autonomous training traces")
    return result
