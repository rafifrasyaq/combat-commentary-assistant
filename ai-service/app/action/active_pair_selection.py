"""M5C.6: conservative, human-verifiable active pair proposals.

No fighter identification or combat event classification. Uses only stored M5C.2
pose and M5C.4 refined associations; does NOT infer punches or real impacts.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

STAGE = "UNVERIFIED_ACTIVE_FIGHTER_PAIR_PROPOSALS"
MAX_CLIPS = 10
MAX_SAMPLED_FRAMES = 100
MAX_PEOPLE_PER_FRAME = 12
MIN_AREA = 0.075
MIN_HEIGHT = 0.28
MIN_DETECTION_SCORE = 0.4
MIN_PAIR_SCORE = 0.64
MIN_PAIR_MARGIN = 0.12
REVIEW_COLUMNS = [
    "candidateIndex", "frameIndex", "sourceTimeSeconds", "autoStatus",
    "proposedDetectionIndexes", "proposedTrackIds", "provisionalScore",
    "runnerUpScore", "reviewStatus", "humanPairVerdict", "notes"
]


class PairSelectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class PairSelectionResult:
    frame_count: int
    proposed_count: int
    ambiguous_count: int
    unknown_count: int
    preview_count: int
    output_dir: Path


def _number(v: Any) -> bool:
    return type(v) in (int, float) and math.isfinite(v)


def _read_json(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise PairSelectionError(f"Missing or unsafe input: {path.name}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise PairSelectionError(f"Cannot read {path.name}") from exc
    if not isinstance(obj, dict):
        raise PairSelectionError(f"Expected JSON object: {path.name}")
    return obj


def _validate_input(pose: dict, tracking: dict) -> list[tuple[dict, dict]]:
    if pose.get("stage") != "UNVERIFIED_POSE_OBSERVATIONS" or tracking.get("stage") != "UNVERIFIED_EXPERIMENTAL_CLIP_LOCAL_POSE_ASSOCIATIONS":
        raise PairSelectionError("Unexpected source stage; refuse unrelated artifacts")
    if not pose.get("jobId") or not pose.get("videoId") or pose.get("jobId") != tracking.get("jobId") or pose.get("videoId") != tracking.get("videoId"):
        raise PairSelectionError("Job/video metadata mismatch")
    a = pose.get("annotations")
    b = tracking.get("annotations")
    if not isinstance(a, list) or not isinstance(b, list) or not 1 <= len(a) <= MAX_CLIPS or len(a) != len(b) or pose.get("clipCount") != len(a) or tracking.get("clipCount") != len(b):
        raise PairSelectionError("Clip count mismatch")
    for clipno, (pa, ta) in enumerate(zip(a, b), 1):
        if pa.get("candidateIndex") != clipno or ta.get("candidateIndex") != clipno:
            raise PairSelectionError("Candidate ordering mismatch")
        pf, tf = pa.get("frames"), ta.get("frames")
        if not isinstance(pf, list) or not isinstance(tf, list) or not 1 <= len(pf) <= MAX_SAMPLED_FRAMES or len(pf) != len(tf):
            raise PairSelectionError("Frame alignment mismatch")
        prev = -1
        for frame, tr in zip(pf, tf):
            idx = frame.get("frameIndex")
            if type(idx) is not int or idx <= prev or tr.get("frameIndex") != idx or abs(frame.get("clipTimeSeconds", -999) - tr.get("clipTimeSeconds", 999)) > .03:
                raise PairSelectionError("Frame index/time mismatch")
            prev = idx
            people = frame.get("people")
            assocs = tr.get("associations")
            if not isinstance(people, list) or not isinstance(assocs, list) or len(people) != len(assocs) or len(people) > MAX_PEOPLE_PER_FRAME:
                raise PairSelectionError("Detection and track counts mismatch")
            for detindex, (person, assoc) in enumerate(zip(people, assocs)):
                if assoc.get("detectionIndex") != detindex or not isinstance(assoc.get("trackId"), str):
                    raise PairSelectionError("Invalid track-to-detection mapping")
                box = person.get("bboxXyxyNormalized")
                if not isinstance(box, list) or len(box) != 4 or not all(_number(x) and 0 <= x <= 1 for x in box) or not (box[0] < box[2] and box[1] < box[3]):
                    raise PairSelectionError("Invalid bounding box")
                score = person.get("personDetectionScore")
                if not _number(score) or not 0 <= score <= 1:
                    raise PairSelectionError("Invalid detection score")
    return list(zip(a, b))


def _clamp(n: float) -> float:
    return max(0.0, min(1.0, n))


def _keypoint(person: dict, name: str):
    k = person.get("keypoints", {}).get(name)
    if not isinstance(k, dict) or k.get("reported") is not True:
        return None
    x, y = k.get("x"), k.get("y")
    if _number(x) and _number(y) and 0 <= x <= 1 and 0 <= y <= 1:
        return (float(x), float(y))
    return None


def _geom(person: dict) -> dict:
    x1, y1, x2, y2 = person["bboxXyxyNormalized"]
    return {
        "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
        "w": x2 - x1, "h": y2 - y1, "area": (x2 - x1) * (y2 - y1),
    }


def _eligible(person: dict) -> bool:
    g = _geom(person)
    torso = any(_keypoint(person, k) is not None for k in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"))
    return g["area"] >= MIN_AREA and g["h"] >= MIN_HEIGHT and person["personDetectionScore"] >= MIN_DETECTION_SCORE and torso


def _arm_relevance(source: dict, target: dict) -> float:
    """Upper-body proximity only, NOT evidence of punch contact."""
    wrists = [_keypoint(source, x) for x in ("left_wrist", "right_wrist")]
    wrists = [p for p in wrists if p is not None]
    upper = [_keypoint(target, x) for x in ("left_shoulder", "right_shoulder")]
    upper = [p for p in upper if p is not None]
    if not wrists or not upper:
        return 0.0
    torso_center = (sum(x for x, _ in upper) / len(upper), sum(y for _, y in upper) / len(upper))
    width = max(_geom(target)["w"], .13)
    closest = min(math.dist(w, torso_center) for w in wrists)
    return _clamp(1.0 - closest / (1.7 * width))


def _pair_score(a: dict, b: dict, a_track: dict, b_track: dict) -> dict:
    ga, gb = _geom(a), _geom(b)
    dist = math.hypot(ga["cx"] - gb["cx"], ga["cy"] - gb["cy"])
    mean_height = (ga["h"] + gb["h"]) / 2
    proximity = _clamp(1.0 - dist / max(1e-6, 1.25 * mean_height))
    vertical = _clamp(1.0 - abs(ga["cy"] - gb["cy"]) / max(1e-6, .75 * mean_height))
    relative_size = _clamp(1.0 - abs(math.log(ga["area"] / gb["area"])) / math.log(6.0))
    arms = max(_arm_relevance(a, b), _arm_relevance(b, a))
    detection = (a["personDetectionScore"] + b["personDetectionScore"]) / 2
    # Track continuity contributes only a tiny amount: IDs are unverified.
    continuity = sum(t.get("associationStatus") in ("MATCHED_ADJACENT", "RECONNECTED_AFTER_GAP") for t in (a_track, b_track)) / 2
    s = .34 * proximity + .12 * vertical + .16 * relative_size + .25 * arms + .10 * detection + .03 * continuity
    return {"score": round(s, 4), "evidence": {"proximity": round(proximity, 3), "verticalAlignment": round(vertical, 3), "relativeSize": round(relative_size, 3), "armProximity": round(arms, 3), "detectorConfidence": round(detection, 3), "unverifiedTrackContinuity": round(continuity, 3)}}


def propose_frame(frame: dict, tracked_frame: dict) -> dict:
    people = frame["people"]
    assocs = tracked_frame["associations"]
    eligible = [i for i, obj in enumerate(people) if _eligible(obj)]
    pairs = []
    for i, j in combinations(eligible, 2):
        scored = _pair_score(people[i], people[j], assocs[i], assocs[j])
        pairs.append({"detectionIndexes": [i, j], "trackIds": [assocs[i]["trackId"], assocs[j]["trackId"]], **scored})
    pairs.sort(key=lambda x: (-x["score"], x["detectionIndexes"]))
    status = "UNKNOWN_INSUFFICIENT_PEOPLE"
    chosen = None
    if pairs:
        first = pairs[0]
        next_score = pairs[1]["score"] if len(pairs) > 1 else None
        if first["score"] < MIN_PAIR_SCORE:
            status = "UNKNOWN_WEAK_PAIR_EVIDENCE"
        elif next_score is not None and first["score"] - next_score < MIN_PAIR_MARGIN:
            status = "AMBIGUOUS_COMPETING_PAIRS"
        else:
            status = "PAIR_CANDIDATE_UNVERIFIED"
            chosen = first
    assigned = set(chosen["detectionIndexes"]) if chosen else set()
    return {
        "frameIndex": frame["frameIndex"],
        "clipTimeSeconds": frame["clipTimeSeconds"],
        "sourceTimeSeconds": frame["sourceTimeSeconds"],
        "detectedPeople": len(people),
        "eligiblePeople": len(eligible),
        "status": status,
        "selectedDetectionIndexes": sorted(assigned),
        "selectedTrackIds": chosen["trackIds"] if chosen else [],
        "provisionalScore": chosen["score"] if chosen else None,
        "runnerUpScore": pairs[1]["score"] if len(pairs) > 1 else None,
        "personAssignments": [{"detectionIndex": k, "trackId": a["trackId"], "assignment": "PAIR_PROPOSAL_UNVERIFIED" if k in assigned else "UNASSIGNED"} for k, a in enumerate(assocs)],
        "topPairsForReview": pairs[:3],
    }


def generate(pose: dict, tracking: dict) -> dict:
    items = _validate_input(pose, tracking)
    annotations = []
    count_status: dict[str, int] = {}
    with_crowd = 0
    for pa, ta in items:
        out_frames = []
        for pf, tf in zip(pa["frames"], ta["frames"]):
            proposal = propose_frame(pf, tf)
            out_frames.append(proposal)
            count_status[proposal["status"]] = count_status.get(proposal["status"], 0) + 1
            if len(pf["people"]) > 2:
                with_crowd += 1
        annotations.append({
            "candidateIndex": pa["candidateIndex"],
            "sampledFrameCount": len(out_frames),
            "candidateFrames": sum(f["status"] == "PAIR_CANDIDATE_UNVERIFIED" for f in out_frames),
            "ambiguousFrames": sum(f["status"] == "AMBIGUOUS_COMPETING_PAIRS" for f in out_frames),
            "unknownFrames": sum(f["status"].startswith("UNKNOWN") for f in out_frames),
            "frames": out_frames,
        })
    return {
        "schemaVersion": 1,
        "stage": STAGE,
        "method": "two-person-geometry-pose-proximity-abstention-pilot-v1",
        "jobId": pose["jobId"],
        "videoId": pose["videoId"],
        "sourceArtifacts": ["pose_observations.json", "pose_tracks_refined.json"],
        "clipCount": len(annotations),
        "sampledFrameCount": sum(len(a["frames"]) for a in annotations),
        "framesWithMoreThanTwoDetectedPeople": with_crowd,
        "statusCounts": count_status,
        "provisionalScoreIsCalibratedProbability": False,
        "verifiedFighterIdentity": False,
        "verifiedPairSelectionAccuracy": False,
        "combatEventsDetected": False,
        "limitations": "Pose geometry and proximity are heuristics. A nearby bystander may still be selected, camera cuts and occlusions may confuse assignment. Unassigned does not prove bystander. Frame-level pair proposals are NOT stable fighter IDs, true punches or contact. Human review is mandatory.",
        "annotations": annotations,
    }


def _as_csv_rows(result: dict) -> list[dict]:
    rows = []
    for item in result["annotations"]:
        for frame in item["frames"]:
            rows.append({
                "candidateIndex": item["candidateIndex"],
                "frameIndex": frame["frameIndex"],
                "sourceTimeSeconds": frame["sourceTimeSeconds"],
                "autoStatus": frame["status"],
                "proposedDetectionIndexes": ";".join(map(str, frame["selectedDetectionIndexes"])),
                "proposedTrackIds": ";".join(frame["selectedTrackIds"]),
                "provisionalScore": frame["provisionalScore"] if frame["provisionalScore"] is not None else "",
                "runnerUpScore": frame["runnerUpScore"] if frame["runnerUpScore"] is not None else "",
                "reviewStatus": "UNREVIEWED",
                "humanPairVerdict": "UNREVIEWED",
                "notes": "",
            })
    return rows


def _render_markdown(result: dict) -> str:
    counts = result["statusCounts"]
    parts = ["# M5C.6 — Active Pair Selection Pilot", "", "**UNVERIFIED HEURISTIC — NOT fighter recognition.**", "", f"Analyzed **{result['sampledFrameCount']}** sampled frames from **{result['clipCount']}** clips. Frames containing 3+ detected people: **{result['framesWithMoreThanTwoDetectedPeople']}**.", "", "## Frame statuses", ""]
    for key in ("PAIR_CANDIDATE_UNVERIFIED", "AMBIGUOUS_COMPETING_PAIRS", "UNKNOWN_WEAK_PAIR_EVIDENCE", "UNKNOWN_INSUFFICIENT_PEOPLE"):
        parts.append(f"- {key}: **{counts.get(key,0)}**")
    parts.extend(["", "## Per-clip summary", "", "| Clip | Candidate | Ambiguous | Unknown |", "|---:|---:|---:|---:|"])
    for a in result["annotations"]:
        parts.append(f"| {a['candidateIndex']} | {a['candidateFrames']} | {a['ambiguousFrames']} | {a['unknownFrames']} |")
    parts.extend(["", "## Important limitations", "", "- These are frame-level pair proposals, **NOT** persistent fighter A/B IDs.", "- A proposed pair can contain a bystander; `UNASSIGNED` does not mean confirmed non-fighter.", "- Scores are ranking heuristics, **NOT** calibrated probabilities or accuracy.", "- Any ambiguous/unknown frame abstains from selecting a pair.", "- Review `active_pair_review.csv` and preview contact sheets before using these results for strike analysis.", "- No punches, strike types, or impacts have been classified.", ""])
    return "\n".join(parts)


def _write_output(job_dir: Path, result: dict) -> Path:
    outdir = job_dir / "active_pair_review"
    if outdir.exists() and (not outdir.is_dir() or outdir.is_symlink()):
        raise PairSelectionError("Unsafe output directory")
    outdir.mkdir(exist_ok=True)
    review_csv = outdir / "active_pair_review.csv"
    # NEVER silently overwrite human judgments or allow stale judgments to point to new proposal data.
    if review_csv.exists() or review_csv.is_symlink():
        raise PairSelectionError(f"Review CSV already exists; refusing to overwrite: {review_csv.name}. Save existing review before rerunning.")
    for target in (outdir / "active_pair_proposals.json", outdir / "active_pair_report.md"):
        if target.is_symlink():
            raise PairSelectionError("Unsafe existing output")
    (outdir / "active_pair_proposals.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (outdir / "active_pair_report.md").write_text(_render_markdown(result), encoding="utf-8")
    with review_csv.open("x", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        w.writeheader()
        w.writerows(_as_csv_rows(result))
    return outdir


def _annotate_frame(rgb, pf: dict, selection: dict, cv2):
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    selected = set(selection["selectedDetectionIndexes"])
    for i, person in enumerate(pf["people"]):
        x1, y1, x2, y2 = person["bboxXyxyNormalized"]
        left, top, right, bottom = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
        color = (70, 220, 255) if i in selected else (160, 160, 160)
        cv2.rectangle(img, (left, top), (right, bottom), color, 3 if i in selected else 1)
        text = f"P? {i}" if i in selected else f"U {i}"
        cv2.putText(img, text, (max(2, left), max(18, top-5)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2, cv2.LINE_AA)
    return img


def _contact_sheet(tiles, clip_id: int, cv2):
    import numpy as np
    cell_w, cell_h = 440, 275
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    canvas = np.full((68 + rows*cell_h, cell_w*cols, 3), 24, dtype=np.uint8)
    cv2.putText(canvas, f"M5C.6 CLIP {clip_id:02d} - UNVERIFIED PAIR PROPOSALS", (18, 26), cv2.FONT_HERSHEY_SIMPLEX, .73, (245,245,245), 2, cv2.LINE_AA)
    cv2.putText(canvas, "yellow = tentative pair   gray = unassigned   UNKNOWN / AMBIGUOUS = no assignment", (18, 50), cv2.FONT_HERSHEY_SIMPLEX, .52, (210,210,210), 1, cv2.LINE_AA)
    for k, (img, fr) in enumerate(tiles):
        yy, xx = divmod(k, cols)
        rh, rw = img.shape[:2]
        scale = min((cell_w-12)/rw, (cell_h-42)/rh)
        nw, nh = max(1, int(rw*scale)), max(1, int(rh*scale))
        small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        x = xx*cell_w+(cell_w-nw)//2
        y = 68+yy*cell_h+36+(cell_h-42-nh)//2
        canvas[y:y+nh, x:x+nw] = small
        label = f"frame {fr['frameIndex']}  {fr['status'].replace('_',' ')[:27]}"
        cv2.putText(canvas, label, (xx*cell_w+7, 68+yy*cell_h+23), cv2.FONT_HERSHEY_SIMPLEX, .46, (240,240,240), 1, cv2.LINE_AA)
    return canvas


def _render_previews(job_dir: Path, result: dict, *, frame_reader=None, clips=None) -> int:
    try:
        import cv2
        if frame_reader is None:
            from .pose import read_sampled_frames as frame_reader
        if clips is None:
            from .recognition import _validated_clips
            _, clips = _validated_clips(job_dir)
    except ImportError as exc:
        raise PairSelectionError("Preview requires OpenCV and prior M5A/M5C modules") from exc
    pose = _read_json(job_dir / "pose_observations.json")
    outdir = job_dir / "active_pair_review"
    outdir.mkdir(exist_ok=True)
    if len(clips) != len(pose["annotations"]):
        raise PairSelectionError("Video clip count mismatch")
    n = 0
    for pa, proposed, (_, path) in zip(pose["annotations"], result["annotations"], clips):
        samples = frame_reader(path)
        if len(samples) != len(pa["frames"]):
            raise PairSelectionError("Video sampled frames changed from original pose run")
        frames = []
        for sample, pf, sf in zip(samples, pa["frames"], proposed["frames"]):
            if sample.frame_index != pf["frameIndex"] or abs(sample.clip_time_seconds-pf["clipTimeSeconds"]) > .025:
                raise PairSelectionError("Video frame does not align with saved pose")
            frames.append((_annotate_frame(sample.rgb, pf, sf, cv2), sf))
        sheet = _contact_sheet(frames, pa["candidateIndex"], cv2)
        dest = outdir / f"candidate_{pa['candidateIndex']:02d}_pair_proposals.jpg"
        if dest.is_symlink():
            raise PairSelectionError("Unsafe output image")
        temp = dest.with_suffix(".tmp.jpg")
        if not cv2.imwrite(str(temp), sheet, [cv2.IMWRITE_JPEG_QUALITY, 87]):
            raise PairSelectionError("Could not write pair preview")
        os.replace(temp, dest)
        n += 1
    return n


def run(job_dir: Path, *, previews=True, frame_reader=None, clips=None) -> PairSelectionResult:
    job_dir = Path(job_dir)
    pose = _read_json(job_dir / "pose_observations.json")
    tracking = _read_json(job_dir / "pose_tracks_refined.json")
    result = generate(pose, tracking)
    # Validate no existing human review before touching any output. Generate
    # previews first so an unavailable clip cannot leave a new CSV behind.
    old_csv = job_dir / "active_pair_review" / "active_pair_review.csv"
    if old_csv.exists() or old_csv.is_symlink():
        raise PairSelectionError("Existing human review CSV; refusing to overwrite. Do not rerun over existing judgments.")
    n = _render_previews(job_dir, result, frame_reader=frame_reader, clips=clips) if previews else 0
    out = _write_output(job_dir, result)
    counts = result["statusCounts"]
    return PairSelectionResult(result["sampledFrameCount"], counts.get("PAIR_CANDIDATE_UNVERIFIED",0), counts.get("AMBIGUOUS_COMPETING_PAIRS",0), sum(v for k,v in counts.items() if k.startswith('UNKNOWN')), n, out)


def main():
    parser = argparse.ArgumentParser(description="M5C.6 experimental active pair proposal; no fighter ID verification")
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--skip-previews", action="store_true")
    args = parser.parse_args()
    try:
        res = run(args.job_dir, previews=not args.skip_previews)
    except PairSelectionError as exc:
        parser.exit(1, f"M5C.6 FAILED: {exc}\n")
    print(f"M5C.6 completed: {res.frame_count} sampled frames")
    print(f"PAIR_CANDIDATE_UNVERIFIED: {res.proposed_count}")
    print(f"AMBIGUOUS_COMPETING_PAIRS: {res.ambiguous_count}")
    print(f"UNKNOWN: {res.unknown_count}")
    print(f"Preview contact sheets: {res.preview_count}")
    print(f"Output: {res.output_dir}")
    print("WARNING: no pair is verified, no stable fighter ID, no punch or impact detected.")


if __name__ == "__main__":
    main()
