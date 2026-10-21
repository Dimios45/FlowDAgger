import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from flowdagger_pi05.yam.core import polarity, verify_conventions, sample, invert
from flowdagger_pi05.yam.data import windows
from flowdagger_pi05.yam.diagnostics import diagnose, summarize
from flowdagger_pi05.yam.traces import TraceStore, preservation_examples


def test_new_collection_mapping_is_identity():
    c = json.loads((Path(__file__).parents[1]/'configs/yam_identity.json').read_text())
    verify_conventions(c)
    for shape in ((14,), (30,14)):
        x = np.random.default_rng(3).random(shape).astype(np.float32)
        np.testing.assert_array_equal(polarity(x, c['dataset_open'], c['model_open']), x)
        np.testing.assert_array_equal(polarity(x, c['model_open'], c['wire_open']), x)


def test_real_clock_gap_rejected_despite_nominal_timestamps():
    frame = pd.DataFrame(dict(episode_index=[0]*60, frame_index=range(60),
        timestamp=np.arange(60)/30, intervention=[1]*60, index=range(60),
        **{'observation.capture_monotonic':100+np.arange(60)/30,
           'action.command_monotonic':100+np.arange(60)/30+.001}))
    assert len(windows(frame)[0]) == 4
    frame.loc[15:, 'observation.capture_monotonic'] += .2
    frame.loc[15:, 'action.command_monotonic'] += .2
    assert all(w['frame'] >= 15 for w in windows(frame)[0])


def trace_fixture(tmp_path):
    store = TraceStore(tmp_path)
    row = dict(identity={'fixture':1}, dataset_episode=0, episode_id='episode',
        patches=torch.zeros(3,16,8), state=torch.zeros(14), noise=torch.zeros(30,32),
        actions=torch.zeros(30,14))
    return store, store.record(row)


def test_modified_commands_logged_but_not_original_noise_anchors(tmp_path):
    store, request_id = trace_fixture(tmp_path)
    event = dict(kind='executed', request_id=request_id, episode_id='episode',
        event_id='one', indices=list(range(30)), commands=torch.ones(30,14).tolist())
    result = store.event(event)
    assert store.event(event) == result  # retry does not duplicate execution
    with pytest.raises(ValueError, match='reused'):
        store.event(dict(event, commands=torch.zeros(30,14).tolist()))
    store.event(dict(kind='outcome', episode_id='episode', success=True))
    assert preservation_examples(tmp_path, {'fixture':1}, [0], allow_empty=True) == []
    assert len((tmp_path/'events.jsonl').read_text().splitlines()) == 2


def test_takeover_retry_cannot_advance_generation_twice(tmp_path):
    store, _ = trace_fixture(tmp_path)
    e = dict(kind='takeover', episode_id='episode', event_id='take')
    assert store.event(e)['generation'] == 1
    assert store.event(e)['generation'] == 1
    with pytest.raises(ValueError, match='session'):
        store.event(dict(kind='handback', episode_id='episode', session='wrong'))


def test_invalid_event_does_not_advance_generation(tmp_path):
    store, _ = trace_fixture(tmp_path)
    with pytest.raises(ValueError):
        store.event(dict(kind='takeover',episode_id='episode',event_id='take',monotonic=float('nan')))
    assert store.generation == 0
    assert store.event(dict(kind='takeover',episode_id='episode',event_id='take'))['generation'] == 1


def test_inversion_report_includes_baselines_sensitivity_and_rejections():
    class Backend:
        device = 'cpu'
        def prepare(self, obs, actions): return {'batch': {'action':torch.tensor(actions)[None]}}
        def sample(self, p, w): return sample(lambda x,t: .2*x, w)
        native_sample = sample
        def invert(self, p, x): return invert(lambda x,t: .2*x, x)
        def decode_actions(self, x): return x[...,:14]
    report = diagnose(Backend(), {}, np.ones((30,14),np.float32)*.2)
    assert report['native_max_abs'] == 0
    assert report['human']['reconstruction']['rmse'] < 1e-5
    assert report['human']['zero_rmse'] > .1
    assert report['human']['target_to_noise_amplification'] > 0
    rows = [dict(window={'episode':0}, normalized={'accepted':a, 'finite':True, 'rmse':v})
            for a,v in ((True,.01),(False,.2))]
    assert summarize(rows)['acceptance_rate'] == .5


def test_round_aggregation_keeps_holdouts_and_source_observations(tmp_path, monkeypatch):
    from flowdagger_pi05.yam import pipeline
    from flowdagger_pi05.yam.data import write_json
    class Dataset:
        def __init__(self, root): self.root=root
        def identity(self): return {'dataset':self.root}
    monkeypatch.setattr(pipeline,'Dataset',Dataset)
    monkeypatch.setattr(pipeline,'model_example',lambda d,w,c:(d.root,w['episode'],{}))
    entries=[]
    for source in ('round0','round1'):
        root=tmp_path/source
        ws=[dict(episode=i,split=s,indices=[i]) for i,s in enumerate(('train','validation','test'))]
        m=dict(identity={'base':'frozen','conventions':{}},cache_id=source,
            manifest=dict(dataset=source,dataset_identity={'dataset':source},windows=ws,
                          splits=dict(train=[0],validation=[1],test=[2])))
        write_json(root/'metadata.json',m)
        for i,w in enumerate(ws):
            pipeline.save_tensor(root/'examples'/f'{i:06d}.pt',dict(cache_id=source,window=w,
                normalized={'accepted':True},patches=torch.zeros(3,16,8),state=torch.zeros(14),noise=torch.zeros(30,32)))
        entries.append({'cache':str(root)})
    pipeline.merge_caches(entries,tmp_path/'merged')
    meta,rows=pipeline.read_cache(tmp_path/'merged')
    assert meta['manifest']['splits']==dict(train=[0,3],validation=[1,4],test=[2,5])
    assert pipeline.example_input(meta,rows[4])[:2]==('round1',1)
    with pytest.raises(ValueError,match='Duplicate'):
        pipeline.merge_caches([entries[0],entries[0]],tmp_path/'duplicate')


def test_divergent_inversion_is_saved_as_rejected_not_sampled(tmp_path, monkeypatch):
    from flowdagger_pi05.yam import pipeline
    from flowdagger_pi05.yam.core import CONTRACT
    from flowdagger_pi05.yam.data import write_json
    class Dataset:
        def __init__(self,root): pass
        def identity(self): return {'data':'fixture'}
    class Backend:
        device='cpu'
        identity={'base':'fixture'}
        def prepare(self,*args):
            return dict(batch={'action':torch.zeros(1,30,14)},patches=torch.zeros(1,3,16,8),state=torch.zeros(1,14))
        def sample(self,p,noise):
            assert torch.isfinite(noise).all()
            return noise
        def invert(self,p,target): return torch.full_like(target,float('nan'))
    c=json.loads((Path(__file__).parents[1]/'configs/yam_identity.json').read_text())
    monkeypatch.setattr(pipeline,'Dataset',Dataset)
    monkeypatch.setattr(pipeline,'model_example',lambda *args:({},np.zeros((30,14),np.float32),{}))
    m=dict(id='one',dataset='fixture',dataset_identity={'data':'fixture'},conventions=c,contract=CONTRACT,
        windows=[dict(episode=0,split='train',indices=list(range(30)))])
    write_json(tmp_path/'manifest.json',m)
    pipeline.invert_dataset(Backend(),tmp_path/'manifest.json',tmp_path/'cache')
    _,rows=pipeline.read_cache(tmp_path/'cache',accepted_only=False)
    assert rows[0]['normalized']['accepted'] is False
    assert json.loads((tmp_path/'cache/inversion-summary.json').read_text())['acceptance_rate']==0
