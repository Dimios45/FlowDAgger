import copy
import numpy as np
import pandas as pd
import pytest
import torch

from flowdagger_pi05.yam.preservation import KIND, reviewed_windows, overlaps, load_preservation_cache


def fixture():
    class Dataset:
        frame = pd.DataFrame(dict(episode_index=np.repeat([0, 1, 2], 120),
            frame_index=list(range(120))*3, timestamp=list(np.arange(120)/30)*3,
            intervention=np.zeros(360), index=range(360)))
    source = dict(id="human", splits=dict(train=[0], validation=[1], test=[2]))
    review = dict(kind=KIND, source_manifest_id="human", review_method="fixture", limitations=["reconstructed"],
                  ranges=[dict(episode=ep, start=0, end=90, evidence="fixture", behavior="reach") for ep in (0,1)])
    outcomes = dict(episodes=[dict(episode_index=ep, saved=True, success=True, aborted=False) for ep in range(3)])
    return Dataset(), source, review, outcomes


def test_review_excludes_leakage_handoffs_and_duplicates():
    d, source, review, outcomes = fixture()
    assert len(reviewed_windows(d, source, review, outcomes)) == 6
    bad = copy.deepcopy(review); bad["ranges"][0]["episode"] = 2
    with pytest.raises(ValueError, match="test"):
        reviewed_windows(d, source, bad, outcomes)
    bad = copy.deepcopy(review); bad["ranges"].append(bad["ranges"][0])
    with pytest.raises(ValueError, match="duplicate"):
        reviewed_windows(d, source, bad, outcomes)
    d.frame.loc[60, "intervention"] = 1
    with pytest.raises(ValueError, match="intervention"):
        reviewed_windows(d, source, review, outcomes)


def test_failed_outcome_gaps_and_reset_exclusion():
    d, source, review, outcomes = fixture()
    outcomes["episodes"][0]["success"] = False
    with pytest.raises(ValueError, match="successful"):
        reviewed_windows(d, source, review, outcomes)
    outcomes["episodes"][0]["success"] = True
    d.frame.loc[15:, "timestamp"] += .1
    selected = reviewed_windows(d, source, review, outcomes)
    assert not any(w["episode"] == 0 and w["frame"] < 15 for w in selected)
    assert overlaps(dict(episode=0, frame=30), [dict(episode=0, start=59, end=90)])
    assert not overlaps(dict(episode=0, frame=30), [dict(episode=0, start=60, end=90)])


def test_cache_rejects_wrong_identity_split_and_rejected_targets(monkeypatch):
    from flowdagger_pi05.yam import pipeline
    human = dict(identity={"base":"a"}, manifest=dict(id="human", dataset_identity={"d":"a"},
                splits=dict(train=[0], validation=[1], test=[2])))
    meta = copy.deepcopy(human)
    meta["manifest"]["supervision"] = dict(kind=KIND, original_noise_available=False, source_manifest_id="human")
    examples = [dict(window=dict(episode=i, split=s), normalized=dict(accepted=True))
                for i,s in [(0,"train"),(1,"validation")]]
    monkeypatch.setattr(pipeline, "read_cache", lambda *a, **kw: (meta, examples))
    assert len(load_preservation_cache("unused", human)[2]) == 1
    examples[0]["normalized"]["accepted"] = False
    with pytest.raises(ValueError, match="accepted"):
        load_preservation_cache("unused", human)
    examples[0]["normalized"]["accepted"] = True
    examples[0]["window"]["episode"] = 2
    with pytest.raises(ValueError, match="leakage"):
        load_preservation_cache("unused", human)
    examples[0]["window"]["episode"] = 0
    meta["identity"] = {"base":"different"}
    with pytest.raises(ValueError, match="mismatch"):
        load_preservation_cache("unused", human)


def test_two_buffer_training_balance_and_selection(tmp_path, monkeypatch):
    from flowdagger_pi05.yam import pipeline, preservation
    from flowdagger_pi05.yam.core import SteeringActor
    rows = [dict(window=dict(episode=0, split="train", frame=i*30), normalized=dict(accepted=True),
                 patches=torch.zeros(3,16,8), state=torch.zeros(14), noise=torch.zeros(30,32)) for i in range(4)]
    auto = [dict(r, noise=torch.ones(30,32)) for r in rows]
    meta = dict(identity={"x":1}, cache_id="human", manifest=dict(dataset="fixture", dataset_identity={"d":1}))
    pm = dict(cache_id="auto", manifest=dict(supervision=dict(kind=KIND, review={})))
    monkeypatch.setattr(pipeline, "read_cache", lambda *a, **kw: (meta, rows))
    class Dataset:
        def __init__(self,*a): pass
        def identity(self): return {"d":1}
    monkeypatch.setattr(pipeline,"Dataset",Dataset)
    monkeypatch.setattr(preservation,"load_preservation_cache",lambda *a: (pm, auto, auto))
    forwards=[]
    class Actor(SteeringActor):
        def forward(self, patches, state):
            forwards.append(len(state))
            return super().forward(patches,state)
    monkeypatch.setattr(pipeline,"SteeringActor",Actor)
    def evaluate(backend, actor, which, *a, **kw):
        return dict(mean_normalized_rmse=dict(steered=.2 if which is meta else .6))
    monkeypatch.setattr(pipeline,"evaluate",evaluate)
    class Backend:
        identity={"x":1}
        device="cpu"
    meta["identity"]["conventions"]={}
    out=tmp_path/'actor.pt'
    pipeline.train(Backend(), 'unused', out, epochs=1, batch_size=4, preservation_cache='unused')
    saved=torch.load(out,weights_only=True)
    assert forwards==[2,2,2,2]  # two updates, each with 2 human + 2 preservation
    assert saved['selection_score']==pytest.approx(.4)
    assert saved['validation_decoded_rmse']==.2
    assert saved['preservation_validation_decoded_rmse']==.6
    assert saved['training']['preservation_batch_size']==2


def test_last_checkpoint_runs_full_epochs_even_when_validation_worsens(tmp_path, monkeypatch):
    from flowdagger_pi05.yam import pipeline
    rows = [dict(window=dict(episode=0, split='train', frame=0), normalized=dict(accepted=True),
                 patches=torch.zeros(3,16,8), state=torch.zeros(14), noise=torch.zeros(30,32))]
    meta = dict(identity={'conventions':{}}, cache_id='human',
                manifest=dict(dataset='fixture', dataset_identity={'d':1}))
    monkeypatch.setattr(pipeline, 'read_cache', lambda *a, **kw: (meta, rows))
    class Dataset:
        def __init__(self,*a): pass
        def identity(self): return {'d':1}
    monkeypatch.setattr(pipeline, 'Dataset', Dataset)
    scores=iter([.1,.2,.3])
    monkeypatch.setattr(pipeline, 'evaluate', lambda *a,**kw: dict(mean_normalized_rmse=dict(steered=next(scores))))
    class Backend:
        identity={}
        device='cpu'
    out=tmp_path/'actor.pt'
    pipeline.train(Backend(), 'unused', out, epochs=3, patience=1, batch_size=1,
                   checkpoint_selection='last')
    saved=torch.load(out,weights_only=True)
    assert saved['epoch']==2
    assert saved['validation_decoded_rmse']==.3
    assert saved['training']['checkpoint_selection']=='last'
    assert saved['training']['early_stopping'] is False


def test_all_training_uses_previous_holdouts_without_evaluation(tmp_path, monkeypatch):
    from flowdagger_pi05.yam import pipeline
    rows=[dict(window=dict(episode=i,split=s,frame=0),normalized=dict(accepted=True),
          patches=torch.zeros(3,16,8),state=torch.zeros(14),noise=torch.zeros(30,32))
          for i,s in enumerate(['train','validation','test'])]
    meta=dict(identity={'conventions':{}},cache_id='human',manifest=dict(dataset='fixture',dataset_identity={}))
    monkeypatch.setattr(pipeline,'read_cache',lambda *a,**kw:(meta,rows))
    class Dataset:
        def __init__(self,*a):pass
        def identity(self):return {}
    monkeypatch.setattr(pipeline,'Dataset',Dataset)
    def no_eval(*a,**kw):raise AssertionError('Holdout evaluation must not run')
    monkeypatch.setattr(pipeline,'evaluate',no_eval)
    class Backend:
        identity={}
        device='cpu'
    out=tmp_path/'actor.pt'
    pipeline.train(Backend(),'unused',out,epochs=2,patience=1,all_training=True,checkpoint_selection='last')
    saved=torch.load(out,weights_only=True)
    assert saved['epoch']==1
    assert saved['training']['human_training_episodes']==[0,1,2]
    assert saved['training']['human_examples']==3
    assert saved['training']['validation_enabled'] is False
    assert 'validation_decoded_rmse' not in saved
