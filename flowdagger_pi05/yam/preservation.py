"""Explicitly reviewed, reconstructed preservation; never claims original noise."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .core import digest, file_hash
from .data import Dataset, windows, write_json

KIND = "reviewed-autonomous-segments-reconstructed-noise-v1"
INDEPENDENT_KIND = "successful-separate-dataset-reconstructed-noise-v1"


def overlaps(window, ranges):
    return any(window["episode"] == r["episode"] and
               window["frame"] < r["end"] and window["frame"] + 30 > r["start"] for r in ranges)


def reviewed_windows(dataset, source, review, outcomes, stride=30):
    if review.get("kind") != KIND or review.get("source_manifest_id") != source["id"]:
        raise ValueError("Review must identify this human manifest and reconstructed supervision")
    if not review.get("review_method") or not review.get("limitations"):
        raise ValueError("Document review method and limitations")
    if stride < 30:
        raise ValueError("Use nonoverlapping preservation windows (stride >= 30)")
    frame = dataset.frame.copy()
    flags = frame.intervention.map(lambda x: float(np.asarray(x).reshape(-1)[0]))
    if not flags.isin([0, 1]).all():
        raise ValueError("Invalid intervention flags")
    frame["intervention"] = 1 - flags
    candidates, _ = windows(frame, stride)
    selected = []
    seen = set()
    allowed = set(source["splits"]["train"] + source["splits"]["validation"])
    for r in review["ranges"]:
        ep, start, end = r["episode"], r["start"], r["end"]
        if any(type(x) is not int for x in (ep, start, end)) or start < 0 or end <= start:
            raise ValueError("Ranges use integer frames [start, end)")
        if ep not in allowed:
            raise ValueError("Preservation review must not use test or unknown episodes")
        if not r.get("evidence") or not r.get("behavior"):
            raise ValueError("Each range needs visual evidence and behavior description")
        matching = [e for e in outcomes["episodes"] if e.get("episode_index") == ep and e.get("saved")]
        if len(matching) != 1 or matching[0].get("success") is not True or matching[0].get("aborted"):
            raise ValueError("Reviewed portions require a saved, successful, non-aborted source episode")
        rows = frame[(frame.episode_index == ep) & frame.frame_index.between(start, end-1)]
        if len(rows) != end-start or not (rows.intervention == 1).all():
            raise ValueError("Reviewed range crosses intervention or missing frames")
        count = 0
        for w in candidates:
            if w["episode"] != ep or not start <= w["frame"] or w["frame"]+30 > end:
                continue
            key = (ep, w["frame"])
            if key in seen:
                raise ValueError("Overlapping review ranges duplicate examples")
            seen.add(key)
            selected.append(dict(w, split="train" if ep in source["splits"]["train"] else "validation"))
            count += 1
        if not count:
            raise ValueError("Reviewed range contains no complete autonomous window")
    for r in review.get("human_exclusions", []):
        if r["episode"] not in source["splits"]["train"] or r["start"] < 0 or r["end"] <= r["start"] or not r.get("evidence"):
            raise ValueError("Human exclusions need training-only ranges and evidence")
    if set(w["split"] for w in selected) != {"train", "validation"}:
        raise ValueError("Review needs independent train and validation episodes")
    return sorted(selected, key=lambda w: (w["episode"], w["frame"]))


def prepare_preservation(source_manifest, review_path, output, stride=30):
    source = json.loads(Path(source_manifest).read_text())
    review = json.loads(Path(review_path).read_text())
    dataset = Dataset(source["dataset"])
    if dataset.identity() != source["dataset_identity"]:
        raise ValueError("Dataset changed since the human manifest")
    outcomes_path = dataset.root / "openpi_control_hitl.json"
    outcomes = json.loads(outcomes_path.read_text())
    selected = reviewed_windows(dataset, source, review, outcomes, stride)
    manifest = dict(source)
    manifest.pop("id")
    manifest.update(windows=selected, stride=stride, excluded_short_segments=[],
        supervision=dict(kind=KIND, original_noise_available=False,
                         original_request_boundaries_verified=False, source_manifest_id=source["id"],
                         review=review, review_sha256=file_hash(review_path),
                         outcomes_sha256=file_hash(outcomes_path)),
        video_pts=dataset.validate_video_pts())
    manifest["id"] = digest(manifest)
    path = Path(output)
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Refusing to replace a different preservation manifest")
    write_json(path, manifest)
    return dict(windows=dict(Counter(w["split"] for w in selected)), kind=KIND,
                original_noise_available=False, manifest_id=manifest["id"])


def load_preservation_cache(root, human_meta):
    from .pipeline import read_cache
    meta, examples = read_cache(root, accepted_only=False)
    m, h = meta["manifest"], human_meta["manifest"]
    s = m.get("supervision", {})
    if s.get("kind") not in (KIND, INDEPENDENT_KIND) or s.get("original_noise_available") is not False:
        raise ValueError("Expected explicitly reconstructed preservation cache")
    if meta["identity"] != human_meta["identity"]:
        raise ValueError("Preservation base/processors/dataset mismatch")
    if s["source_manifest_id"] != h["id"]:
        raise ValueError("Preservation split/source manifest mismatch")
    if s["kind"] == KIND:
        if m["dataset_identity"] != h["dataset_identity"] or m["splits"] != h["splits"]:
            raise ValueError("Preservation dataset/split mismatch")
    else:
        if m["dataset_identity"] == h["dataset_identity"] or not s.get("snapshot"):
            raise ValueError("Independent preservation needs a separate versioned dataset")
        assigned = [ep for ids in m["splits"].values() for ep in ids]
        if len(set(assigned)) != len(assigned) or sorted(assigned) != sorted(m["dataset_episodes"]):
            raise ValueError("Independent preservation split leakage")
        if Dataset(m["dataset"], episodes=m["dataset_episodes"]).identity() != m["dataset_identity"]:
            raise ValueError("Independent preservation dataset changed")
    for e in examples:
        w = e["window"]
        if w["split"] not in ("train", "validation") or w["episode"] not in m["splits"][w["split"]]:
            raise ValueError("Preservation split leakage")
    train = [e for e in examples if e["window"]["split"] == "train" and e["normalized"]["accepted"]]
    if m.get('all_training'):
        if m['splits']['validation'] or m['splits']['test'] or any(e['window']['split'] != 'train' for e in examples):
            raise ValueError('All-training preservation must have no holdouts')
    if not train or (not m.get('all_training') and not any(e["window"]["split"] == "validation" for e in examples)):
        raise ValueError("Need accepted preservation training targets and validation examples")
    return meta, examples, train
