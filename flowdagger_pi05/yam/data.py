"""LeRobot v3 windows; no padding, outcome filtering, or inferred noise labels."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .core import CONTRACT, DATA_CAMERAS, CAMERAS, digest, file_hash, polarity, verify_conventions


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def windows(frame, stride=10):
    if stride < 1:
        raise ValueError("stride must be positive")
    result, excluded = [], []
    for episode, group in frame.groupby("episode_index", sort=True):
        group = group.sort_values("frame_index")
        ids = group.frame_index.to_numpy()
        ts = group.timestamp.to_numpy()
        human = group.intervention.map(lambda x: float(np.asarray(x).reshape(-1)[0])).to_numpy()
        if not np.isin(human, [0, 1]).all():
            raise ValueError("Intervention labels must be binary")
        contiguous = (np.diff(ids) == 1) & (np.abs(np.diff(ts) - 1 / 30) < .002)
        breaks = np.r_[True, (np.diff(human) != 0) | ~contiguous, True]
        bounds = np.flatnonzero(breaks)
        for segment, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
            if human[start] != 1:
                continue
            if end - start < 30:
                excluded.append(dict(episode=int(episode), segment=segment, frames=int(end-start)))
            for at in range(start, end - 29, stride):
                result.append(dict(episode=int(episode), segment=segment, frame=int(ids[at]),
                                   indices=group.iloc[at:at+30]["index"].astype(int).tolist()))
    return result, excluded


class Dataset:
    def __init__(self, root, episodes=None):
        self.root = Path(root)
        self.selected_episodes = episodes
        self.info = json.loads((self.root / "meta/info.json").read_text())
        if self.info["fps"] != 30:
            raise ValueError("Expected 30 Hz dataset")
        self.frame = pd.concat([pd.read_parquet(p) for p in sorted(self.root.glob("data/**/*.parquet"))], ignore_index=True)
        if self.frame["index"].duplicated().any():
            raise ValueError("Duplicate global frame indices")
        expected = [f"left_joint_{i}" for i in range(1, 7)] + ["left_gripper"] + [f"right_joint_{i}" for i in range(1, 7)] + ["right_gripper"]
        for key in ("observation.state", "action"):
            feature = self.info["features"][key]
            if feature["shape"] != [14] or feature["names"] != expected:
                raise ValueError(f"Unrecognized dataset joint order: {key}")
        if not np.isfinite(self.frame.timestamp.to_numpy()).all():
            raise ValueError("Nonfinite dataset timestamps")
        if self.frame.duplicated(["episode_index", "frame_index"]).any():
            raise ValueError("Duplicate episode/frame indices")
        self.frame = self.frame.set_index("index", drop=False)
        self.episodes = pd.concat([pd.read_parquet(p) for p in sorted(self.root.glob("meta/episodes/**/*.parquet"))]).set_index("episode_index")
        if episodes is not None:
            if not episodes or len(set(episodes)) != len(episodes):
                raise ValueError("Selected episodes must be nonempty and unique")
            self.episodes = self.episodes.loc[episodes]
            self.frame = self.frame[self.frame.episode_index.isin(episodes)]
        if any(list(tasks) != [CONTRACT["task"]] for tasks in self.episodes.tasks):
            raise ValueError("Dataset must contain only 'fold the towel'")

    def identity(self):
        videos = sorted(self.root.glob("videos/**/*.mp4"))
        if self.selected_episodes is not None:
            videos = sorted({self.root / self.info["video_path"].format(
                video_key="observation.images." + c,
                chunk_index=int(ep[f"videos/observation.images.{c}/chunk_index"]),
                file_index=int(ep[f"videos/observation.images.{c}/file_index"]))
                for c in DATA_CAMERAS for _, ep in self.episodes.iterrows()})
        paths = sorted(self.root.glob("**/*.parquet")) + videos + [self.root / "meta/info.json"]
        return {str(p.relative_to(self.root)): file_hash(p) for p in paths}

    def validate_video_pts(self):
        """Check encoded PTS coverage/gaps; this cannot prove wall-clock camera sync."""
        import av
        reports = {}
        for camera in DATA_CAMERAS:
            key = "observation.images." + camera
            for _, episode in self.episodes.iterrows():
                relative = self.info["video_path"].format(video_key=key,
                    chunk_index=int(episode[f"videos/{key}/chunk_index"]),
                    file_index=int(episode[f"videos/{key}/file_index"]))
                if relative not in reports:
                    with av.open(str(self.root / relative)) as container:
                        stream = container.streams.video[0]
                        pts = sorted(float(packet.pts * stream.time_base) for packet in container.demux(stream)
                                     if packet.pts is not None)
                    times = np.asarray(pts)
                    if not len(times) or (len(times) > 1 and not np.all(np.abs(np.diff(times) - 1/30) < .002)):
                        raise ValueError(f"Video PTS gaps or duplicates: {relative}")
                    reports[relative] = dict(first=float(times[0]), last=float(times[-1]), frames=len(times))
                report = reports[relative]
                start = float(episode[f"videos/{key}/from_timestamp"])
                end = float(episode[f"videos/{key}/to_timestamp"])
                if abs((end-start)*30 - int(episode.length)) > .1:
                    raise ValueError(f"Episode/video length mismatch: {relative}")
                if report["first"] > start + .002 or report["last"] < end - 1/30 - .002:
                    raise ValueError(f"Video does not cover episode: {relative}")
        return reports

    def observation(self, window):
        import av
        row = self.frame.loc[window["indices"][0]]
        episode = self.episodes.loc[window["episode"]]
        result = dict(state=np.asarray(row["observation.state"], dtype=np.float32), instruction=CONTRACT["task"])
        alignment = {}
        for source, dest in zip(DATA_CAMERAS, CAMERAS):
            key = "observation.images." + source
            path = self.root / self.info["video_path"].format(video_key=key,
                chunk_index=int(episode[f"videos/{key}/chunk_index"]), file_index=int(episode[f"videos/{key}/file_index"]))
            wanted = float(episode[f"videos/{key}/from_timestamp"]) + float(row.timestamp)
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                container.seek(max(0, int((wanted - .1) / stream.time_base)), stream=stream)
                best = None
                for frame in container.decode(stream):
                    if frame.pts is None:
                        raise ValueError(f"Missing PTS: {path}")
                    actual = float(frame.pts * stream.time_base)
                    if best is None or abs(actual-wanted) < abs(best[0]-wanted):
                        best = (actual, frame.to_ndarray(format="rgb24"))
                    if actual >= wanted:
                        break
            if best is None or abs(best[0] - wanted) > 1 / 60 + .002:
                raise ValueError(f"Video alignment failed: {path} at {wanted}")
            result[dest + "_cam"] = best[1]
            alignment[dest] = dict(path=str(path), requested=wanted, pts=best[0])
        return result, alignment


def prepare(root, conventions, output, seed=0, stride=10, splits=None):
    conventions = verify_conventions(conventions)
    dataset = Dataset(root)
    episodes = sorted(int(e) for e in dataset.episodes.index)
    if len(episodes) < 3:
        raise ValueError("At least three episodes are required")
    shuffled = np.random.default_rng(seed).permutation(episodes).tolist()
    n = max(1, round(len(episodes) * .1))
    if splits is None:
        splits = dict(train=shuffled[:-2*n], validation=shuffled[-2*n:-n], test=shuffled[-n:])
    if set(splits) != {"train", "validation", "test"} or any(not v for v in splits.values()):
        raise ValueError("Supply nonempty train/validation/test episode lists")
    assigned = [e for values in splits.values() for e in values]
    if sorted(assigned) != episodes or len(set(assigned)) != len(assigned):
        raise ValueError("Splits must assign every episode exactly once")
    candidates, excluded = windows(dataset.frame, stride)
    for w in candidates:
        w["split"] = next(k for k, v in splits.items() if w["episode"] in v)
    manifest = dict(schema=1, dataset=str(Path(root).resolve()), dataset_identity=dataset.identity(),
                    conventions=conventions, contract=CONTRACT, splits=splits, seed=seed,
                    video_pts=dataset.validate_video_pts(),
                    stride=stride, windows=candidates, excluded_short_segments=excluded,
                    synchronization="PTS alignment checked on decode; wall-clock synchronization unverified")
    manifest["id"] = digest(manifest)
    write_json(output, manifest)
    return manifest


def model_example(dataset, window, conventions):
    obs, alignment = dataset.observation(window)
    actions = np.stack(dataset.frame.loc[window["indices"], "action"].to_numpy()).astype(np.float32)
    obs["state"] = polarity(obs["state"], conventions["dataset_open"], conventions["model_open"])
    actions = polarity(actions, conventions["dataset_open"], conventions["model_open"])
    return obs, actions, alignment
