"""Conservative reconstruction from successful, half-speed recorded rollouts.

Interpolation consistency is evidence about playback, NOT proof of the original
request boundary/observation, unclipped model action, or original sampler noise.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np

from .core import CONTRACT, digest, file_hash
from .data import Dataset, write_json, windows as contiguous_windows
from .preservation import INDEPENDENT_KIND


def episode_splits(outcomes, seed=0):
    groups = defaultdict(list)
    excluded = []
    seen = set()
    for row in outcomes['episodes']:
        ep = row['episode_index']
        if ep in seen:
            raise ValueError('Duplicate episode outcomes')
        seen.add(ep)
        if row.get('success') is not True or row.get('saved') is not True or row.get('aborted'):
            excluded.append(dict(episode=ep, reason='not saved successful non-aborted'))
        elif row.get('prompt') != CONTRACT['task']:
            excluded.append(dict(episode=ep, reason='different task instruction'))
        else:
            groups[row['source_dataset']].append(ep)
    rng = np.random.default_rng(seed)
    splits = dict(train=[], validation=[], test=[])
    for source, episodes in sorted(groups.items()):
        ids = rng.permutation(sorted(episodes)).tolist()
        n = max(1, round(len(ids)*.1))
        if len(ids) <= 2*n:
            raise ValueError(f'Insufficient successful episodes in {source}')
        for key, values in zip(splits, (ids[:-2*n], ids[-2*n:-n], ids[-n:])):
            splits[key].extend(values)
    return splits, excluded


def playback_windows(group, tolerance=1e-4):
    """Choose strongest episode-wide 59-frame phase; reject inconsistent chunks.

    Skipped/missing records can change phase mid-episode: those windows are
    deliberately lost, not interpolated or padded. Full recorded span is tested.
    """
    group = group.sort_values('frame_index')
    a = np.stack(group.action).astype(np.float64)
    state = np.stack(group['observation.state'])
    if a.shape[1:] != (14,) or not np.isfinite(a).all() or not np.isfinite(state).all():
        raise ValueError('Nonfinite or malformed recorded actions/state')
    ids = group.frame_index.to_numpy()
    ts = group.timestamp.to_numpy()
    candidates = []
    for phase in range(59):
        n = (len(a)-phase)//59
        if n < 1:
            continue
        v = a[phase:phase+n*59].reshape(n,59,14)
        residual = np.abs(v[:,1::2] - (v[:,:-2:2]+v[:,2::2])/2)
        maxima = residual.max((1,2))
        candidates.append((int((maxima <= tolerance).sum()), -float(residual.mean()), phase, maxima))
    if not candidates:
        return [], dict(frames=len(a), reason='short episode')
    candidates.sort(key=lambda x:(x[0],x[1]), reverse=True)
    count, _, phase, maxima = candidates[0]
    # Avoid assigning an arbitrary phase to an entirely stationary recording.
    informative = np.ptp(a, axis=0).max() > .01
    phase_margin = count - (candidates[1][0] if len(candidates)>1 else 0)
    report = dict(frames=len(a), phase=phase, interpolation_tolerance=tolerance,
                  phase_margin_windows=phase_margin, complete_spans=len(maxima),
                  interpolation_pass=int(count), retained=0,
                  request_boundaries_verified=False)
    if not informative or phase_margin < 3:
        report['reason'] = 'ambiguous interpolation phase'
        return [], report
    selected = []
    for j, error in enumerate(maxima):
        at = phase+j*59
        if error > tolerance:
            continue
        span = slice(at, at+59)
        if not (np.diff(ids[span]) == 1).all() or not (np.abs(np.diff(ts[span])-1/30)<.002).all():
            continue
        selected.append(dict(episode=int(group.episode_index.iloc[0]), segment=0,
            frame=int(ids[at]), indices=group.iloc[at:at+59:2]['index'].astype(int).tolist(),
            recorded_span_frames=59, recorded_action_stride=2,
            interpolation_max_error=float(error), boundary_source='inferred-episode-playback-phase'))
    report['retained'] = len(selected)
    return selected, report


def recorded_windows(group):
    """Keep 30 consecutive recorded commands, without playback resampling."""
    frame = group.copy()
    for key in ('action', 'observation.state'):
        values = np.stack(frame[key])
        if values.shape[1:] != (14,) or not np.isfinite(values).all():
            raise ValueError('Nonfinite or malformed recorded actions/state')
    # Reuse the continuity splitter; control provenance is checked by prepare.
    frame['intervention'] = 1
    selected, short = contiguous_windows(frame, stride=30)
    selected = [dict(w, recorded_span_frames=30, recorded_action_stride=1,
                     boundary_source='consecutive-recorded-frames') for w in selected]
    return selected, dict(frames=len(frame), retained=len(selected),
                         excluded_short_segments=short, resampled=False,
                         request_boundaries_verified=False)


def prepare(root, source_manifest, snapshot_path, output, review_path, seed=0,
            timing_mode='restore-native', all_training=False):
    if timing_mode not in ('restore-native', 'recorded'):
        raise ValueError('Unknown timing mode')
    root, output = Path(root), Path(output)
    source = json.loads(Path(source_manifest).read_text())
    outcomes_path = root/'openpi_control_rollouts.json'
    outcomes = json.loads(outcomes_path.read_text())
    if outcomes.get('format') != 'openpi-control.inference-rollout-v1':
        raise ValueError('Expected an explicitly autonomous rollout recording')
    if (outcomes.get('speed'), outcomes.get('chunk_size'), outcomes.get('fps')) != (.5,30,30):
        raise ValueError('This reconstruction is specific to 30-action chunks at playback speed .5 / 30 Hz')
    splits, excluded = episode_splits(outcomes, seed)
    if all_training:
        splits = dict(train=sorted(ep for ids in splits.values() for ep in ids), validation=[], test=[])
    split_record = dict(splits=splits, excluded=excluded, seed=seed,
                        outcomes_sha256=file_hash(outcomes_path))
    split_path = output.with_name('autonomous-splits.json')
    if split_path.exists() and json.loads(split_path.read_text()) != split_record:
        raise ValueError('Refusing to change an existing episode split')
    write_json(split_path, split_record)
    episodes = sorted(ep for ids in splits.values() for ep in ids)
    dataset = Dataset(root, episodes=episodes)
    if 'intervention' in dataset.frame:
        flags = dataset.frame.intervention.map(lambda x: float(np.asarray(x).reshape(-1)[0]))
        if not (flags == 0).all():
            raise ValueError('Autonomous source contains intervention or invalid control flags')
    selected, timing = [], {}
    for split in ('train','validation'):
        for ep in splits[split]:
            selector = recorded_windows if timing_mode == 'recorded' else playback_windows
            windows, report = selector(dataset.frame[dataset.frame.episode_index==ep])
            timing[str(ep)] = dict(report, split=split)
            selected.extend(dict(w,split=split) for w in windows)
    required_splits = ('train',) if all_training else ('train','validation')
    if any(not any(w['split']==s for w in selected) for s in required_splits):
        raise ValueError('No timing-consistent examples in one training/validation buffer')
    review = json.loads(Path(review_path).read_text())
    if review['source_manifest_id'] != source['id']:
        raise ValueError('Human exclusion review belongs to another dataset')
    manifest = dict(schema=1, dataset=str(root.resolve()), dataset_episodes=episodes,
        dataset_identity=dataset.identity(), conventions=source['conventions'], contract=CONTRACT,
        splits=splits, seed=seed, windows=sorted(selected,key=lambda w:(w['episode'],w['frame'])),
        stride=30 if timing_mode == 'recorded' else 59,
        excluded_short_segments=[], video_pts=dataset.validate_video_pts(),
        supervision=dict(kind=INDEPENDENT_KIND, original_noise_available=False,
            original_request_boundaries_verified=False, source_manifest_id=source['id'],
            snapshot=json.loads(Path(snapshot_path).read_text()),
            outcomes_sha256=file_hash(outcomes_path),
            review=dict(human_exclusions=review.get('human_exclusions',[])),
            timing=timing, excluded_episodes=excluded,
            gripper_evidence='audit/autonomous-grippers.png: episode 0 frame 0 open, frame 1500 closed; dataset_open=0',
            limitations=['Executed commands may be clipped; original model proposals and noise are unavailable.',
                        'Interpolation phase is inferred, not an original request boundary.',
                        'Recorded observation is not proven identical to the inference request.',
                        'Metadata speed .5 is assumed for all merged sources; inconsistent windows are excluded.',
                        'This is offline reconstruction, not an interleaved on-policy DAgger round.']))
    if timing_mode == 'recorded':
        manifest['supervision']['timing_mode'] = 'recorded-no-resampling'
        manifest['supervision']['limitations'] = [
            'Targets are consecutive recorded commands; playback interpolation is retained.',
            'Recorded windows are not verified original model chunks or request observations.',
            'Original noise and unclipped model proposals are unavailable.',
            'Client playback speed .5 will stretch these already recorded-cadence predictions again.',
            'This is offline reconstruction, not an interleaved on-policy DAgger round.']
    if all_training:
        manifest['all_training'] = True
    manifest['id'] = digest(manifest)
    if output.exists() and json.loads(output.read_text()) != manifest:
        raise ValueError('Refusing to overwrite a different preparation manifest')
    write_json(output,manifest)
    summary=dict(windows=dict(Counter(w['split'] for w in selected)),splits=splits,
                 timing=timing, excluded_episodes=excluded,manifest_id=manifest['id'])
    write_json(output.with_name('preparation-summary.json'),summary)
    return summary
