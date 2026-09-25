import copy
import numpy as np
import pandas as pd
import pytest
from flowdagger_pi05.yam.autonomous_dataset import episode_splits, playback_windows


def fixture():
    rng=np.random.default_rng(4)
    native=rng.normal(size=(12,30,14))
    frames=[]
    for chunk in native:
        stretch=np.empty((59,14));stretch[::2]=chunk
        stretch[1::2]=(chunk[:-1]+chunk[1:])/2
        frames.extend(stretch)
    a=np.array(frames)
    return pd.DataFrame({'episode_index':0,'frame_index':np.arange(len(a)),
        'timestamp':np.arange(len(a))/30,'index':np.arange(len(a))+1000,
        'action':list(a),'observation.state':list(a)}),native


def test_recovers_native_spacing_without_padding_and_drops_clipping():
    frame,native=fixture()
    windows,report=playback_windows(frame)
    assert len(windows)==12 and report['phase']==0
    for w,target in zip(windows,native):
        np.testing.assert_allclose(np.stack(frame.set_index('index').loc[w['indices']].action),target)
        assert len(w['indices'])==30 and np.diff(w['indices']).tolist()==[2]*29
    frame.at[1,'action']=np.zeros(14)
    windows,_=playback_windows(frame)
    assert len(windows)==11 and windows[0]['frame']==59
    frame.loc[70:,'timestamp']+=.2
    windows,_=playback_windows(frame)
    assert not any(w['frame']==59 for w in windows)


def test_shifted_recording_and_stationary_ambiguity():
    frame,_=fixture()
    windows,report=playback_windows(frame.iloc[1:].copy())
    assert report['phase']==58 and windows[0]['frame']==59
    frame['action']=[np.zeros(14) for _ in range(len(frame))]
    assert playback_windows(frame)[0]==[]


def test_splits_exclude_failure_wrong_task_and_are_disjoint():
    rows=[dict(episode_index=i,success=True,saved=True,aborted=False,
               prompt='fold the towel',source_dataset='a') for i in range(20)]
    rows[0]['success']=False; rows[1]['prompt']='fold the towel \\'
    splits,excluded=episode_splits(dict(episodes=rows))
    ids=[i for x in splits.values() for i in x]
    assert len(ids)==len(set(ids))==18 and 0 not in ids and 1 not in ids
    assert len(excluded)==2
    assert splits==episode_splits(dict(episodes=rows))[0]
    with pytest.raises(ValueError,match='Duplicate'):
        episode_splits(dict(episodes=rows+[copy.deepcopy(rows[0])]))


def test_separate_datasets_allow_same_episode_number_but_reject_test_leakage(monkeypatch):
    from flowdagger_pi05.yam import pipeline,preservation
    human=dict(identity={'base':'same'},manifest=dict(id='h',dataset_identity={'human':1}))
    meta=dict(identity=human['identity'],manifest=dict(dataset='auto',dataset_identity={'auto':1},
        dataset_episodes=[0,1,2],splits=dict(train=[0],validation=[1],test=[2]),
        supervision=dict(kind=preservation.INDEPENDENT_KIND,original_noise_available=False,
                         source_manifest_id='h',snapshot={'revision':'pinned'})))
    rows=[dict(window=dict(episode=ep,split=split),normalized=dict(accepted=True))
          for ep,split in [(0,'train'),(1,'validation')]]
    monkeypatch.setattr(pipeline,'read_cache',lambda *a,**kw:(meta,rows))
    class Dataset:
        def __init__(self,*a,**kw):pass
        def identity(self):return {'auto':1}
    monkeypatch.setattr(preservation,'Dataset',Dataset)
    assert len(preservation.load_preservation_cache('unused',human)[2])==1
    rows[0]['window']['episode']=2
    with pytest.raises(ValueError,match='leakage'):
        preservation.load_preservation_cache('unused',human)


def test_selected_episode_identity_ignores_unrelated_downloads(tmp_path):
    from flowdagger_pi05.yam.data import Dataset
    d=Dataset.__new__(Dataset)
    d.root=tmp_path;d.selected_episodes=[0]
    d.info={'video_path':'videos/{video_key}/{file_index}.mp4'}
    row={f'videos/observation.images.{c}/{field}':0
         for c in ('top','left_wrist','right_wrist') for field in ('chunk_index','file_index')}
    d.episodes=pd.DataFrame([row])
    (tmp_path/'meta').mkdir();(tmp_path/'meta/info.json').write_text('{}')
    for c in ('top','left_wrist','right_wrist'):
        path=tmp_path/f'videos/observation.images.{c}/0.mp4'
        path.parent.mkdir(parents=True);path.write_bytes(b'selected')
    before=d.identity()
    (tmp_path/'videos/extra.mp4').write_bytes(b'unrelated download')
    assert d.identity()==before
    (tmp_path/'videos/observation.images.top/0.mp4').write_bytes(b'changed')
    assert d.identity()!=before


def test_recorded_mode_preserves_every_command_and_excludes_gaps():
    from flowdagger_pi05.yam.autonomous_dataset import recorded_windows
    frame,_=fixture()
    frame=frame.iloc[:95].copy()
    selected,report=recorded_windows(frame)
    assert len(selected)==3 and report['resampled'] is False
    assert selected[0]['indices']==list(range(1000,1030))
    assert selected[1]['indices']==list(range(1030,1060))
    assert all(w['recorded_action_stride']==1 for w in selected)
    frame.loc[15:,'timestamp']+=.1
    selected,report=recorded_windows(frame)
    assert selected[0]['frame']==15
    assert all(not (w['frame']<15<w['frame']+30) for w in selected)
    assert report['excluded_short_segments'][0]['frames']==15
