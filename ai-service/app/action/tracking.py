"""M5C.3: conservative clip-local association of previously sampled YOLO pose observations.

This is *not* fighter identity tracking and does not detect punches, hits, or strike type.
It uses only pose_observations.json; no video decode, GPU, or model download.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

INPUT_NAME = "pose_observations.json"
OUTPUT_NAME = "pose_tracks.json"
METHOD = "adjacent-frame-bbox-and-body-relative-wrist-v1"
MAX_CLIPS = 50
MAX_FRAMES_PER_CLIP = 12
MAX_PEOPLE_PER_FRAME = 4
MAX_GAP_SECONDS = 1.0
MAX_CENTER_SHIFT = 0.28
AMBIGUITY_MARGIN = 0.065
MIN_DELTA_SECONDS = 0.04
COCO_ARMS = ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist")


class TrackingError(RuntimeError):
    """Invalid pose observations or unsafe tracking output."""


@dataclass(frozen=True)
class TrackingResult:
    path: str
    clips: int
    tracks: int
    matches: int
    hand_motion_samples: int
    ambiguous_restarts: int


def _number(value: Any, name: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrackingError(f"Invalid {name}: must be numeric")
    value = float(value)
    if not math.isfinite(value) or (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise TrackingError(f"Invalid {name}: out of range")
    return value


def _bbox(person: dict) -> tuple[float, float, float, float]:
    coords = person.get("bboxXyxyNormalized")
    if not isinstance(coords, list) or len(coords) != 4:
        raise TrackingError("Invalid person bounding box")
    x1, y1, x2, y2 = (_number(v, "bbox", minimum=0, maximum=1) for v in coords)
    if not x2 > x1 or not y2 > y1:
        raise TrackingError("Degenerate person bounding box")
    return x1, y1, x2, y2


def _point(person: dict, name: str) -> tuple[float, float] | None:
    points = person.get("keypoints")
    if not isinstance(points, dict):
        raise TrackingError("Invalid person keypoints")
    item = points.get(name)
    if not isinstance(item, dict) or not isinstance(item.get("reported"), bool):
        raise TrackingError(f"Invalid {name} keypoint")
    _number(item.get("score"), f"{name} score", minimum=0, maximum=1)
    if not item["reported"]:
        return None
    return (
        _number(item.get("x"), f"{name} x", minimum=0, maximum=1),
        _number(item.get("y"), f"{name} y", minimum=0, maximum=1),
    )


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersect = x * y
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return intersect / (aa + bb - intersect)


def _pair_cost(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float | None:
    ax, ay = _center(a)
    bx, by = _center(b)
    dist = math.hypot(ax - bx, ay - by)
    if dist > MAX_CENTER_SHIFT:
        return None
    # Penalize rapid changes in scale; broadcasting cuts and occlusion may break a track.
    a_area = (a[2] - a[0]) * (a[3] - a[1])
    b_area = (b[2] - b[0]) * (b[3] - b[1])
    if max(a_area, b_area) / min(a_area, b_area) > 3.0:
        return None
    return dist + 0.16 * (1 - _iou(a, b))


def _hand_offset(person: dict, side: str) -> tuple[float, float] | None:
    shoulder, wrist = _point(person, f"{side}_shoulder"), _point(person, f"{side}_wrist")
    if shoulder is None or wrist is None:
        return None
    box = _bbox(person)
    scale = max(box[2] - box[0], 0.08)
    return ((wrist[0] - shoulder[0]) / scale, (wrist[1] - shoulder[1]) / scale)


def _movement(old: dict, new: dict, seconds: float) -> dict:
    """2D shoulder-relative wrist displacement rate, NOT real-world speed or punch evidence."""
    result = {}
    for side in ("left", "right"):
        a, b = _hand_offset(old, side), _hand_offset(new, side)
        if a is None or b is None or not MIN_DELTA_SECONDS <= seconds <= MAX_GAP_SECONDS:
            result[side] = {"measurable": False, "shoulderRelativeRate": None}
        else:
            rate = math.hypot(a[0] - b[0], a[1] - b[1]) / seconds
            result[side] = {"measurable": True, "shoulderRelativeRate": round(rate, 4)}
    return result


def _read_doc(job_dir: Path) -> dict:
    if not job_dir.is_dir() or job_dir.is_symlink():
        raise TrackingError("Job directory is missing or unsafe")
    source = job_dir / INPUT_NAME
    if not source.is_file() or source.is_symlink():
        raise TrackingError("M5C.2 pose observations missing or unsafe")
    try:
        doc = json.loads(source.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise TrackingError("Could not read pose observations JSON") from exc
    if (not isinstance(doc, dict) or doc.get("stage") != "UNVERIFIED_POSE_OBSERVATIONS"
            or doc.get("combatEventsDetected") is not False or not isinstance(doc.get("annotations"), list)):
        raise TrackingError("Unexpected M5C.2 pose observations schema")
    clips = doc["annotations"]
    if len(clips) > MAX_CLIPS or doc.get("clipCount") != len(clips):
        raise TrackingError("Pose clip count exceeds limit or is inconsistent")
    ids = set()
    for clip in clips:
        if not isinstance(clip, dict) or not isinstance(clip.get("frames"), list):
            raise TrackingError("Invalid pose clip")
        cid = clip.get("candidateIndex")
        if type(cid) is not int or cid < 1 or cid in ids:
            raise TrackingError("Invalid or duplicate candidate index")
        ids.add(cid)
        frames = clip["frames"]
        if len(frames) > MAX_FRAMES_PER_CLIP or clip.get("sampledFrameCount") != len(frames):
            raise TrackingError("Pose frame count exceeds limit or is inconsistent")
        last_time = -1.0
        last_index = -1
        for frame in frames:
            if not isinstance(frame, dict) or not isinstance(frame.get("people"), list):
                raise TrackingError("Invalid pose frame")
            people = frame["people"]
            if len(people) > MAX_PEOPLE_PER_FRAME or frame.get("personCount") != len(people):
                raise TrackingError("Pose people count exceeds limit or is inconsistent")
            timestamp = _number(frame.get("clipTimeSeconds"), "frame timestamp", minimum=0)
            index = frame.get("frameIndex")
            if type(index) is not int or index <= last_index or timestamp < last_time:
                raise TrackingError("Pose frames must have increasing indices and timestamps")
            last_index, last_time = index, timestamp
            for person in people:
                if not isinstance(person, dict):
                    raise TrackingError("Invalid person observation")
                _bbox(person)
                for name in COCO_ARMS:
                    _point(person, name)
    return doc


def _track_clip(clip: dict) -> tuple[dict, dict]:
    """Assign short-lived observation IDs only to pairs in adjacent sampled frames."""
    cid = clip["candidateIndex"]
    seq = 0
    prior: list[dict] = []
    outputs: list[dict] = []
    durations: dict[str, int] = {}
    counts = {"matches": 0, "handSamples": 0, "ambiguousRestarts": 0}
    prev_time: float | None = None
    for frame in clip["frames"]:
        t = float(frame["clipTimeSeconds"])
        people = frame["people"]
        boxes = [_bbox(p) for p in people]
        delta = None if prev_time is None else t - prev_time
        feasible = []
        if delta is not None and MIN_DELTA_SECONDS <= delta <= MAX_GAP_SECONDS:
            for i, tr in enumerate(prior):
                for j, box in enumerate(boxes):
                    cost = _pair_cost(tr["box"], box)
                    if cost is not None:
                        feasible.append((cost, i, j))
        # Abstain when two feasible alternatives are too close in matching cost.
        ambiguous_t = set()
        ambiguous_d = set()
        for i in range(len(prior)):
            costs = sorted(c for c, pi, _ in feasible if pi == i)
            if len(costs) >= 2 and costs[1] - costs[0] < AMBIGUITY_MARGIN:
                ambiguous_t.add(i)
        for j in range(len(people)):
            costs = sorted(c for c, _, dj in feasible if dj == j)
            if len(costs) >= 2 and costs[1] - costs[0] < AMBIGUITY_MARGIN:
                ambiguous_d.add(j)
        paired: dict[int, tuple[int, float]] = {}
        used_tracks: set[int] = set()
        for cost, i, j in sorted(feasible):
            if i in used_tracks or j in paired or i in ambiguous_t or j in ambiguous_d:
                continue
            paired[j] = (i, cost)
            used_tracks.add(i)
        next_prior: list[dict] = []
        associations = []
        for j, person in enumerate(people):
            if j in paired:
                i, cost = paired[j]
                tr = prior[i]
                tid = tr["id"]
                status = "MATCHED_ADJACENT"
                hands = _movement(tr["person"], person, float(delta))
                counts["matches"] += 1
                counts["handSamples"] += sum(x["measurable"] for x in hands.values())
                association_cost = round(cost, 4)
            else:
                seq += 1
                tid = f"c{cid:02d}_t{seq:02d}"
                ambiguous = j in ambiguous_d or any(i in ambiguous_t and dj == j for _, i, dj in feasible)
                status = "AMBIGUOUS_NEW_TRACK" if ambiguous else "NEW_TRACK"
                if ambiguous:
                    counts["ambiguousRestarts"] += 1
                hands = {side: {"measurable": False, "shoulderRelativeRate": None} for side in ("left", "right")}
                association_cost = None
            durations[tid] = durations.get(tid, 0) + 1
            associations.append({
                "detectionIndex": j,
                "trackId": tid,
                "associationStatus": status,
                "associationCost": association_cost,
                "handMotion": hands,
            })
            next_prior.append({"id": tid, "box": boxes[j], "person": person})
        prior = next_prior
        prev_time = t
        outputs.append({
            "frameIndex": frame["frameIndex"],
            "clipTimeSeconds": t,
            "associations": associations,
        })
    return ({
        "candidateIndex": cid,
        "trackIdScope": "THIS_CLIP_ONLY",
        "trackCount": len(durations),
        "trackLengthsInSampledFrames": durations,
        "frames": outputs,
    }, counts)


def run_tracking(job_directory: Path) -> TrackingResult:
    """Derive separate atomic report from M5C.2; never rewrite prior milestone artifacts."""
    source_dir = Path(job_directory)
    if not source_dir.is_dir() or source_dir.is_symlink():
        raise TrackingError("Job directory is missing or unsafe")
    directory = source_dir.resolve(strict=True)
    doc = _read_doc(directory)
    clips = []
    sums = {"matches": 0, "handSamples": 0, "ambiguousRestarts": 0}
    total_tracks = 0
    for clip in doc["annotations"]:
        result, counts = _track_clip(clip)
        clips.append(result)
        total_tracks += result["trackCount"]
        for key in sums:
            sums[key] += counts[key]
    report = {
        "schemaVersion": 1,
        "stage": "UNVERIFIED_CLIP_LOCAL_POSE_ASSOCIATIONS",
        "method": METHOD,
        "sourceArtifact": INPUT_NAME,
        "sourceModelId": doc.get("modelId"),
        "jobId": doc.get("jobId"),
        "videoId": doc.get("videoId"),
        "clipCount": len(clips),
        "totalClipLocalTracks": total_tracks,
        "matchedAdjacentDetections": sums["matches"],
        "measuredHandMotionSamples": sums["handSamples"],
        "ambiguousRestarts": sums["ambiguousRestarts"],
        "bodyRelativeMotionUnits": "2D shoulder-relative wrist displacement per bounding-box width per second",
        "stableFighterIdentities": False,
        "combatEventsDetected": False,
        "punchesDetected": False,
        "limitations": (
            "Heuristic short-lived box associations ONLY within each clip, using at most 12 sampled frames. "
            "No cross-clip re-identification, reliable fighter A/B assignment, punch or technique detection, "
            "contact estimation, camera-motion correction, calibration, or real-world hand velocity. "
            "Occlusion, overlap, referee, cuts, and ambiguous identity can fragment or swap tracks."
        ),
        "annotations": clips,
    }
    destination = directory / OUTPUT_NAME
    tmp = directory / (OUTPUT_NAME + ".tmp")
    if destination.is_symlink() or tmp.is_symlink():
        raise TrackingError("Unsafe tracking output path")
    try:
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, destination)
    finally:
        if tmp.exists():
            tmp.unlink()
    return TrackingResult(str(destination), len(clips), total_tracks, sums["matches"], sums["handSamples"], sums["ambiguousRestarts"])


def main() -> None:
    parser = argparse.ArgumentParser(description="M5C.3: clip-local pose association from M5C.2 JSON")
    parser.add_argument("--job-dir", required=True, type=Path)
    args = parser.parse_args()
    result = run_tracking(args.job_dir)
    print(f"M5C.3 completed: {result.clips} clips; {result.tracks} tentative clip-local tracks")
    print(f"Adjacent matches: {result.matches}; measured hand samples: {result.hand_motion_samples}; ambiguous restarts: {result.ambiguous_restarts}")
    print(f"Pose tracking JSON: {result.path}")
    print("WARNING: no verified fighter IDs, punch types, contact, or combat events.")


if __name__ == "__main__":
    main()
