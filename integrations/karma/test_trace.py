"""Run against the patched pinned Karma checkout; no hardware is opened."""
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from openpi_control.flowdagger_trace import TraceActions, TraceBridge
from openpi_control.inference import ChunkPrefetcher
from flowdagger_pi05.yam.serve import Policy, build_app
from flowdagger_pi05.yam.tests.test_yam import FakeBackend
from flowdagger_pi05.yam.traces import preservation_examples


def drained(bridge):
    deadline=time.monotonic()+5
    while bridge.queue.unfinished_tasks and time.monotonic()<deadline:
        time.sleep(.01)
    assert not bridge.queue.unfinished_tasks,bridge.error


def test_metadata_survives_prefix_slice():
    a=TraceActions(np.zeros((30,14)),{'request_id':'abc','generation':2})
    assert a[:15].metadata==a.metadata


def test_prefetch_wait_does_not_block_control_thread():
    release=threading.Event()
    class Client:
        def infer(self,*args):
            release.wait(3)
            return TraceActions(np.zeros((30,14)),{'generation':0})
    prefetch=ChunkPrefetcher(Client())
    start=time.monotonic()
    assert prefetch.take_ready(None,'fold the towel') is None
    assert prefetch.take_ready(None,'fold the towel') is None
    assert time.monotonic()-start<.2
    prefetch.drop()
    release.set()
    assert not prefetch.busy


def test_journal_delivery_exact_noise_and_deadline(tmp_path,monkeypatch):
    import json_numpy
    import requests
    conventions=dict(mapping='identity',verified=True,evidence='fixture',dataset_open=1,model_open=1,wire_open=1)
    policy=Policy(FakeBackend(),None,conventions,tmp_path/'server','yam')
    http=TestClient(build_app(policy))
    class Transport:
        def post(self,url,json,timeout):
            return http.post('/events',json=json)
    monkeypatch.setattr(requests,'Session',Transport)
    bridge=TraceBridge('http://fixture/act',policy.health(),tmp_path/'client')
    bridge.start(0,60)
    context=bridge.context()
    frame=np.zeros((8,8,3),np.uint8)
    r=json_numpy.loads(http.post('/act',content=json_numpy.dumps(dict(context,
        state=np.zeros(14),top_cam=frame,left_cam=frame,right_cam=frame,
        instruction='fold the towel',normalization_tag='yam'))).text)
    metadata={**context,**{k:r[k] for k in ('session','generation','request_id')}}
    for i,command in enumerate(r['actions']):
        bridge.executed(metadata,i,command)
    bridge.outcome(True)
    drained(bridge)
    assert len(preservation_examples(tmp_path/'server',policy.identity,[0]))==1
    assert not bridge.accepts(metadata)
    assert len(bridge.path.read_text().splitlines())==2
    bridge.start(1,60)
    old=dict(metadata,episode_id=bridge.episode)
    bridge.takeover()
    assert not bridge.accepts(old)
    drained(bridge)
    assert policy.traces.generation==bridge.generation==1
    assert bridge.context()['generation']==1


def test_traced_recorder_preserves_both_grippers_and_precommand_images():
    from types import SimpleNamespace
    from test_record import FakeArm, FakeCamera
    from openpi_control.record import MemorySink, ScriptedSource, TeleopStep, EpisodeEvent, ArmTarget, record_session
    class Bridge:
        deadline=float('inf')
    source=ScriptedSource([
        TeleopStep(event=EpisodeEvent.START, targets={
            'left':ArmTarget((.3,)*6,.2),'right':ArmTarget((.4,)*6,.8)}),
        TeleopStep(event=EpisodeEvent.SAVE),
        TeleopStep(event=EpisodeEvent.STOP)])
    source._client=SimpleNamespace(flow_trace=Bridge())
    left,right=FakeArm(effector=1.),FakeArm(effector=0.)
    # First frame is used by warmup; the second is captured before poll/command.
    camera=FakeCamera([np.full((2,2,3),10,np.uint8), np.full((2,2,3),20,np.uint8),np.full((2,2,3),30,np.uint8)])
    sink=MemorySink()
    record_session(arms={'left':left,'right':right},source=source,sink=sink,
        cameras={'top':camera},task='fold the towel',fps=1000)
    row=sink.episodes[0][0]
    np.testing.assert_allclose(row['observation.state'][[6,13]],[1.,0.])
    np.testing.assert_allclose(row['action'][[6,13]],[.2,.8])
    assert row['action.command_monotonic'][0]>=row['observation.capture_monotonic'][0]
    assert row['observation.images.top'][0,0,0]==20
