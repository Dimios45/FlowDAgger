"""Opt-in FlowDAgger provenance for Karma; no ML/GPU dependencies.

KARMA_FLOWDAGGER_TRACE_DIR enables this extension. Events are journaled before
asynchronous delivery; robot control never waits for an event HTTP response.
"""
import json
import logging
import os
from pathlib import Path
import queue
import threading
import time
import uuid

import numpy as np


class TraceActions(np.ndarray):
    def __new__(cls, actions, metadata):
        obj = np.asarray(actions).view(cls)
        obj.metadata = dict(metadata)
        return obj

    def __array_finalize__(self, source):
        self.metadata = getattr(source, 'metadata', {})


class TraceBridge:
    def __init__(self, url, health, root):
        self.url = url.rsplit('/act', 1)[0] + '/events'
        self.session = health['session']
        self.generation = health['generation']
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / (self.session + '.jsonl')
        self.queue = queue.Queue()
        self.lock = threading.RLock()
        self.episode = None
        self.dataset_episode = None
        self.deadline = float('inf')
        self.pending = None
        self.error = None
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def _worker(self):
        import requests
        session = requests.Session()
        while True:
            event = self.queue.get()
            while True:
                try:
                    response = session.post(self.url, json=event, timeout=5)
                    response.raise_for_status()
                    self.error = None
                    break
                except Exception as exc:
                    if self.error != str(exc):
                        logging.getLogger(__name__).warning('FlowDAgger event delivery pending in %s: %s', self.path, exc)
                    self.error = str(exc)
                    time.sleep(.25)
            self.queue.task_done()

    def start(self, dataset_episode, seconds):
        with self.lock:
            self.episode = uuid.uuid4().hex
            self.dataset_episode = int(dataset_episode)
            self.seconds = float(seconds)
            self.deadline = float('inf')

    def activate(self):
        if self.deadline == float('inf'):
            self.deadline = time.monotonic() + self.seconds

    def context(self):
        # Called only by the inference worker, never by the command loop.
        self.queue.join()
        with self.lock:
            return dict(session=self.session, generation=self.generation,
                        episode_id=self.episode, dataset_episode=self.dataset_episode)

    def current(self, metadata):
        return (                metadata.get('session') == self.session and
                metadata.get('generation') == self.generation and
                metadata.get('episode_id') == self.episode)

    def accepts(self, metadata):
        return time.monotonic() < self.deadline and self.current(metadata)

    def event(self, kind, **values):
        with self.lock:
            event = dict(kind=kind, session=self.session, episode_id=self.episode,
                         event_id=uuid.uuid4().hex, monotonic=time.monotonic(), **values)
            with self.path.open('a') as stream:
                stream.write(json.dumps(event, allow_nan=False)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
            self.queue.put(event)

    def flush(self):
        with self.lock:
            if self.pending:
                self.event('executed', **self.pending)
                self.pending = None

    def executed(self, metadata, index, commands):
        with self.lock:
            if not self.current(metadata):
                raise RuntimeError('Expired/stale FlowDAgger command')
            request_id = metadata['request_id']
            if self.pending and self.pending['request_id'] != request_id:
                self.flush()
            if self.pending is None:
                self.pending = dict(request_id=request_id, indices=[], commands=[])
            self.pending['indices'].append(int(index))
            self.pending['commands'].append(np.asarray(commands).tolist())
            if index == 29:
                self.flush()

    def takeover(self):
        with self.lock:
            self.flush()
            self.event('takeover')
            self.generation += 1

    def handback(self):
        self.event('handback')

    def drain(self, timeout=5):
        deadline = time.monotonic()+timeout
        while self.queue.unfinished_tasks and time.monotonic()<deadline:
            time.sleep(.01)
        if self.queue.unfinished_tasks:
            logging.getLogger(__name__).warning('Replay pending events from %s before training', self.path)
        return not self.queue.unfinished_tasks

    def outcome(self, success):
        with self.lock:
            self.flush()
            self.event('outcome', success=bool(success))
            self.deadline = 0


def attach(client):
    root = os.environ.get('KARMA_FLOWDAGGER_TRACE_DIR')
    if root:
        health = client.health()
        if health.get('state_dim') != 14 or 'session' not in health:
            raise ValueError('FlowDAgger tracing requires the instrumented YAM server')
        client.flow_trace = TraceBridge(client.url, health, root)
    else:
        client.flow_trace = None


def driver(source):
    return getattr(source, '_policy', source)


def bridge(source):
    return getattr(getattr(driver(source), '_client', None), 'flow_trace', None)


def replay(path, server):
    """Retry a journal against the same running server/session, idempotently."""
    import requests
    url = server.rstrip('/').removesuffix('/act')+'/events'
    with requests.Session() as session:
        for line in Path(path).read_text().splitlines():
            response = session.post(url, json=json.loads(line), timeout=10)
            response.raise_for_status()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Replay durable FlowDAgger events to the original server session')
    parser.add_argument('--journal',required=True)
    parser.add_argument('--server',required=True)
    args = parser.parse_args()
    replay(args.journal,args.server)
