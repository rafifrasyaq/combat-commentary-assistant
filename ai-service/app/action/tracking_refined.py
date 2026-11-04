"""M5C.4: experimental, gap-aware, clip-local association from saved pose observations.

No model inference; no verified fighter identities, punches, contacts, or techniques.
Writes pose_tracks_refined.json and tracking_comparison.json without overwriting M5C.3.
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import tracking as v1

METHOD = "clip-local-gap-aware-global-box-pose-v2-experimental"
OUTPUT_NAME = "pose_tracks_refined.json"
COMPARISON_NAME = "tracking_comparison.json"
COMPARISON_MARKDOWN = "tracking_comparison.md"
MAX_LINK_GAP_FRAMES = 1   # at most one missed sampled frame
MAX_LINK_SECONDS = 0.85
MAX_ADJACENT_CENTER_SHIFT = 0.33
MAX_GAP_CENTER_SHIFT = 0.22
MAX_ADJACENT_SCALE_RATIO = 4.5
MAX_GAP_SCALE_RATIO = 2.7
MAX_PAIR_COST_ADJACENT = 0.43
MAX_PAIR_COST_GAP = 0.29
AMBIGUITY_MARGIN = 0.050
UNMATCHED_COST = 0.46


@dataclass(frozen=True)
class RefinedResult:
    path: str
    comparison_path: str
    clips: int
    tracks: int
    one_frame_tracks: int
    matched_adjacent: int
    gap_reconnections: int
    ambiguous_restarts: int
    hand_samples: int


def _body_anchor(person: dict) -> tuple[float, float] | None:
    """Torso location is less arm-motion-dependent than wrists."""
    left = v1._point(person, "left_shoulder")
    right = v1._point(person, "right_shoulder")
    if left is not None and right is not None:
        return ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
    return None


def _candidate_cost(old_person: dict, new_person: dict, *, missing_frames: int, dt: float) -> float | None:
    if missing_frames not in (0, 1) or not v1.MIN_DELTA_SECONDS <= dt <= MAX_LINK_SECONDS:
        return None
    a, b = v1._bbox(old_person), v1._bbox(new_person)
    distance = math.dist(v1._center(a), v1._center(b))
    scale_a = (a[2] - a[0]) * (a[3] - a[1])
    scale_b = (b[2] - b[0]) * (b[3] - b[1])
    ratio = max(scale_a, scale_b) / min(scale_a, scale_b)
    if missing_frames:
        if distance > MAX_GAP_CENTER_SHIFT or ratio > MAX_GAP_SCALE_RATIO:
            return None
    else:
        if distance > MAX_ADJACENT_CENTER_SHIFT or ratio > MAX_ADJACENT_SCALE_RATIO:
            return None
        # Broad gate is only allowed for spatially close observations.
        if distance > 0.20 and ratio > 2.5:
            return None
        if ratio > 3.0 and distance > 0.11:
            return None
    old_anchor, new_anchor = _body_anchor(old_person), _body_anchor(new_person)
    anchor_penalty = 0.0
    if old_anchor and new_anchor:
        anchor_delta = math.dist(old_anchor, new_anchor)
        if anchor_delta > (0.24 if missing_frames else 0.34):
            return None
        anchor_penalty = 0.07 * anchor_delta
    cost = (distance + 0.15 * (1.0 - v1._iou(a, b))
            + 0.025 * math.log(ratio) + anchor_penalty + (0.07 if missing_frames else 0.0))
    if cost > (MAX_PAIR_COST_GAP if missing_frames else MAX_PAIR_COST_ADJACENT):
        return None
    return cost


def _global_assignment(eligible: list[dict], people: list[dict], frame_idx: int, t: float):
    """One-to-one minimal cost assignment with conservative pair ambiguity vetoes.

    Up to 4 detections per frame, at most 8 prior tracks with 1-frame gap.
    """
    edges: list[tuple[float, int, int, int]] = []  # cost, prior idx, person idx, missing frames
    for i, tr in enumerate(eligible):
        missing = frame_idx - tr["sampleIndex"] - 1
        for j, person in enumerate(people):
            cost = _candidate_cost(tr["person"], person, missing_frames=missing, dt=t - tr["time"])
            if cost is not None:
                edges.append((cost, i, j, missing))
    ambiguous_tracks: set[int] = set()
    ambiguous_detection: set[int] = set()
    for i in range(len(eligible)):
        options = sorted(x[0] for x in edges if x[1] == i)
        if len(options) >= 2 and options[1] - options[0] < AMBIGUITY_MARGIN:
            ambiguous_tracks.add(i)
    for j in range(len(people)):
        options = sorted(x[0] for x in edges if x[2] == j)
        if len(options) >= 2 and options[1] - options[0] < AMBIGUITY_MARGIN:
            ambiguous_detection.add(j)
    allowed = {(i,j): (cost, missing) for cost,i,j,missing in edges if i not in ambiguous_tracks and j not in ambiguous_detection}

    @functools.lru_cache(None)
    def dp(j: int, used: int) -> tuple[float, tuple]:
        if j == len(people):
            return (0.0, ())
        suffix_cost, suffix_mapping = dp(j + 1, used)
        best: tuple[float, tuple] = (UNMATCHED_COST + suffix_cost, (None,) + suffix_mapping)
        for i in range(len(eligible)):
            if used & (1 << i) or (i,j) not in allowed:
                continue
            cost, missing = allowed[i,j]
            extra, remaining = dp(j + 1, used | (1 << i))
            candidate = (cost + extra, (i,) + remaining)
            if candidate[0] < best[0] - 1e-8:
                best = candidate
        return best

    _, solution = dp(0, 0)
    paired = {}
    for j, i in enumerate(solution):
        if i is not None:
            cost, missing = allowed[i,j]
            paired[j] = (i, cost, missing)
    unpaired_ambiguity = set()
    for j in range(len(people)):
        if j in paired:
            continue
        if j in ambiguous_detection or any(i in ambiguous_tracks and dj == j for _, i, dj, _ in edges):
            unpaired_ambiguity.add(j)
    return paired, unpaired_ambiguity


def _track_clip(clip: dict) -> tuple[dict, dict]:
    cid = clip["candidateIndex"]
    next_id = 0
    tracks: dict[str, dict] = {}
    durations: dict[str, int] = {}
    outputs = []
    counts = {"matches": 0, "gapMatches": 0, "handSamples": 0, "ambiguousRestarts": 0}
    for sample_idx, frame in enumerate(clip["frames"]):
        t = float(frame["clipTimeSeconds"])
        people = frame["people"]
        eligible = [tr for tr in tracks.values()
                    if 0 < sample_idx - tr["sampleIndex"] <= MAX_LINK_GAP_FRAMES + 1
                    and v1.MIN_DELTA_SECONDS <= t-tr["time"] <= MAX_LINK_SECONDS]
        eligible.sort(key=lambda tr: tr["id"])
        paired, unpaired_ambiguity = _global_assignment(eligible, people, sample_idx, t)
        associations = []
        for j, person in enumerate(people):
            if j in paired:
                i, cost, missing = paired[j]
                old = eligible[i]
                tid = old["id"]
                status = "RECONNECTED_AFTER_GAP" if missing else "MATCHED_ADJACENT"
                # Never measure velocity/rate across an unobserved frame.
                hands = (v1._movement(old["person"], person, t-old["time"])
                         if not missing else _empty_motion())
                if missing:
                    counts["gapMatches"] += 1
                else:
                    counts["matches"] += 1
                counts["handSamples"] += sum(int(h["measurable"]) for h in hands.values())
                rounded_cost = round(cost, 4)
            else:
                next_id += 1
                tid = f"c{cid:02d}_r{next_id:02d}"
                ambiguous = j in unpaired_ambiguity
                status = "AMBIGUOUS_NEW_TRACK" if ambiguous else "NEW_TRACK"
                counts["ambiguousRestarts"] += int(ambiguous)
                rounded_cost = None
                hands = _empty_motion()
                missing = None
            durations[tid] = durations.get(tid, 0) + 1
            tracks[tid] = {"id": tid, "sampleIndex": sample_idx, "time": t, "person": person}
            associations.append({
                "detectionIndex": j,
                "trackId": tid,
                "associationStatus": status,
                "associationCost": rounded_cost,
                "missingSampledFrames": missing,
                "handMotion": hands,
            })
        outputs.append({"frameIndex": frame["frameIndex"], "clipTimeSeconds": t,
                        "associations": associations})
    return ({"candidateIndex": cid, "trackIdScope": "THIS_CLIP_ONLY_UNVERIFIED",
             "trackCount": len(durations), "trackLengthsInSampledFrames": durations,
             "frames": outputs}, counts)


def _empty_motion():
    return {side: {"measurable": False, "shoulderRelativeRate": None} for side in ("left", "right")}


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    if path.is_symlink() or tmp.is_symlink():
        raise v1.TrackingError(f"Refusing unsafe symlink output: {path.name}")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def run_refined_tracking(job_directory: Path) -> RefinedResult:
    job_dir = Path(job_directory)
    if not job_dir.is_dir() or job_dir.is_symlink():
        raise v1.TrackingError("Invalid or unsafe job directory")
    job_dir = job_dir.resolve(strict=True)
    doc = v1._read_doc(job_dir)
    clips = []
    sums = {"matches": 0, "gapMatches": 0, "handSamples": 0, "ambiguousRestarts": 0}
    for clip in doc["annotations"]:
        out, cnt = _track_clip(clip)
        clips.append(out)
        for key in sums:
            sums[key] += cnt[key]
    new_tracks = sum(c["trackCount"] for c in clips)
    one_frame = sum(sum(v == 1 for v in c["trackLengthsInSampledFrames"].values()) for c in clips)
    report = {
        "schemaVersion": 1,
        "stage": "UNVERIFIED_EXPERIMENTAL_CLIP_LOCAL_POSE_ASSOCIATIONS",
        "method": METHOD,
        "sourceArtifact": v1.INPUT_NAME,
        "jobId": doc.get("jobId"), "videoId": doc.get("videoId"),
        "clipCount": len(clips),
        "totalClipLocalTracks": new_tracks,
        "oneFrameTracks": one_frame,
        "matchedAdjacentDetections": sums["matches"],
        "reconnectedAfterMissedSampledFrame": sums["gapMatches"],
        "measuredHandMotionSamples": sums["handSamples"],
        "ambiguousRestarts": sums["ambiguousRestarts"],
        "stableFighterIdentities": False, "combatEventsDetected": False,
        "punchesDetected": False,
        "limitations": "Experimental heuristic links only detections within one sampled clip, with at most one missing sampled frame. No ground-truth identity, no pose re-identification, fighter IDs, strike type, impact, camera-cut detection or verified tracking accuracy. Fewer tracks does not prove fewer identity switches. Motion rate not estimated for gap reconnection.",
        "annotations": clips,
    }
    comparison = {
        "stage": "UNVERIFIED_TRACKING_COMPARISON",
        "inputDetections": sum(len(f["people"]) for c in doc["annotations"] for f in c["frames"]),
        "sampledFrames": sum(len(c["frames"]) for c in doc["annotations"]),
        "newMethod": METHOD,
        "new": {k: report[k] for k in ("totalClipLocalTracks", "oneFrameTracks", "matchedAdjacentDetections", "reconnectedAfterMissedSampledFrame", "measuredHandMotionSamples", "ambiguousRestarts")},
        "old": None,
        "interpretation": "Lower fragmentation indicators do not demonstrate correct fighter identity, and gap links may introduce ID switches; independent hand-labeled evaluation required.",
    }
    per_clip = [{
        "candidateIndex": c["candidateIndex"],
        "newTracks": c["trackCount"],
        "newSingleFrame": sum(v == 1 for v in c["trackLengthsInSampledFrames"].values()),
        "gapReconnects": sum(a["associationStatus"] == "RECONNECTED_AFTER_GAP" for f in c["frames"] for a in f["associations"]),
        "oldTracks": None,
        "oldSingleFrame": None,
    } for c in clips]
    comparison["byClip"] = per_clip
    old_file = job_dir / v1.OUTPUT_NAME
    if old_file.exists():
        if not old_file.is_file() or old_file.is_symlink():
            raise v1.TrackingError("Unsafe baseline track path")
        try:
            old = json.loads(old_file.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise v1.TrackingError("Invalid baseline tracks") from exc
        if (old.get("stage") != "UNVERIFIED_CLIP_LOCAL_POSE_ASSOCIATIONS"
            or old.get("jobId") != doc.get("jobId")
            or old.get("clipCount") != len(clips)):
            raise v1.TrackingError("Baseline artifacts are not comparable")
        if (not isinstance(old.get("annotations"), list)
            or len(old["annotations"]) != len(clips)
            or any(type(c.get("candidateIndex")) is not int or c.get("candidateIndex") != cl["candidateIndex"]
                   or len(c.get("frames", [])) != len(cl["frames"])
                   or any(len(f.get("associations", [])) != len(pf["people"])
                          for f, pf in zip(c["frames"], source["frames"]))
                   for c, cl, source in zip(old["annotations"], clips, doc["annotations"]))):
            raise v1.TrackingError("Baseline frame/detection counts are not comparable")
        baseline_one_frame = sum(sum(n == 1 for n in c["trackLengthsInSampledFrames"].values()) for c in old["annotations"])
        for item, baseline_clip in zip(per_clip, old["annotations"]):
            item["oldTracks"] = baseline_clip["trackCount"]
            item["oldSingleFrame"] = sum(v == 1 for v in baseline_clip["trackLengthsInSampledFrames"].values())
        comparison["old"] = {
            "totalClipLocalTracks": old["totalClipLocalTracks"],
            "oneFrameTracks": baseline_one_frame,
            "matchedAdjacentDetections": old["matchedAdjacentDetections"],
            "measuredHandMotionSamples": old["measuredHandMotionSamples"],
            "ambiguousRestarts": old["ambiguousRestarts"],
        }
    _atomic_json(job_dir / OUTPUT_NAME, report)
    _atomic_json(job_dir / COMPARISON_NAME, comparison)
    md_path = job_dir / COMPARISON_MARKDOWN
    md_tmp = md_path.with_suffix(md_path.suffix + ".tmp")
    if md_path.is_symlink() or md_tmp.is_symlink():
        raise v1.TrackingError("Unsafe comparison report output")
    old_counts = comparison["old"]
    lines = [
        "# M5C.4 — Tracking comparison (experimental)", "",
        "Comparison uses the same sampled YOLO Pose observations. No verified fighter identities or punch events.", "",
        "| Metric | M5C.3 baseline | M5C.4 refined |", "|---|---:|---:|",
        f"| Clip-local tracks | {old_counts['totalClipLocalTracks'] if old_counts else 'N/A'} | {new_tracks} |",
        f"| One-frame tracks | {old_counts['oneFrameTracks'] if old_counts else 'N/A'} | {one_frame} |",
        f"| Adjacent-frame links | {old_counts['matchedAdjacentDetections'] if old_counts else 'N/A'} | {sums['matches']} |",
        f"| Gap reconnections | N/A | {sums['gapMatches']} |",
        f"| Ambiguous restarts | {old_counts['ambiguousRestarts'] if old_counts else 'N/A'} | {sums['ambiguousRestarts']} |",
        f"| Hand-motion samples | {old_counts['measuredHandMotionSamples'] if old_counts else 'N/A'} | {sums['handSamples']} |", "",
        "## Per-clip fragmentation", "",
        "| Clip | Old tracks | Refined tracks | Old one-frame | Refined one-frame | Gap links |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in per_clip:
        lines.append(f"| {item['candidateIndex']} | {item['oldTracks'] if item['oldTracks'] is not None else 'N/A'} | {item['newTracks']} | {item['oldSingleFrame'] if item['oldSingleFrame'] is not None else 'N/A'} | {item['newSingleFrame']} | {item['gapReconnects']} |")
    lines += ["", "## Interpretation", "",
              "Lower track counts and fewer ambiguity restarts are *fragmentation proxies*, NOT confirmed improvements in fighter identity accuracy.",
              "A gap reconnection could connect different people; assess with manually labeled fighter IDs across frames before using these results as evidence of reliable identity tracking.",
              "Motion rates across gap reconnects are deliberately omitted rather than interpolated.",
              "The original pose_tracks.json and pose_observations.json remain unchanged.", ""]
    try:
        md_tmp.write_text("\n".join(lines), encoding="utf-8")
        os.replace(md_tmp, md_path)
    finally:
        if md_tmp.exists():
            md_tmp.unlink()
    return RefinedResult(str(job_dir / OUTPUT_NAME), str(job_dir / COMPARISON_NAME), len(clips), new_tracks, one_frame, sums["matches"], sums["gapMatches"], sums["ambiguousRestarts"], sums["handSamples"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental clip-local pose tracking refinement M5C.4")
    parser.add_argument("--job-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_refined_tracking(args.job_dir)
    print(f"M5C.4 REFINED TRACKING: {result.clips} clips, {result.tracks} tracks, {result.one_frame_tracks} single-frame tracks")
    print(f"Adjacent matches: {result.matched_adjacent}; gap reconnections: {result.gap_reconnections}; ambiguous restarts: {result.ambiguous_restarts}; hand samples: {result.hand_samples}")
    print(f"Refined JSON: {result.path}")
    print(f"Comparison JSON: {result.comparison_path}")
    print(f"Readable comparison: {Path(result.comparison_path).with_suffix('.md')}")
    print("WARNING: IDs remain clip-local heuristics; fewer tracks is not verified ID accuracy.")


if __name__ == "__main__":
    main()
