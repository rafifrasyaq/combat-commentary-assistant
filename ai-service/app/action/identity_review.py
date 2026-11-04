"""M5C.5: human identity validation aids for experimental clip-local tracks.

Consumes saved pose and refined track JSON plus existing M5A clips. No neural
inference, punch recognition, or automatic fighter identity claims.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

MAX_CLIPS = 10
MAX_FRAMES = 12
MAX_PEOPLE = 4
REVIEW_DIR = "identity_review"
TRACK_RE = re.compile(r"^c\d{2}_r\d{2,3}$")
REVIEW_FIELDS = ["candidateIndex", "reviewStatus", "idSwitchObserved", "nonFighterDetections", "occlusionOrCameraCut", "notes"]
PRIORITY_FIELDS = ["candidateIndex", "frameIndex", "clipTimeSeconds", "sourceTimeSeconds", "trackId", "associationStatus", "visualVerdict", "notes"]
PRIORITY_STATES = {"AMBIGUOUS_NEW_TRACK", "RECONNECTED_AFTER_GAP"}
VALID_STATES = PRIORITY_STATES | {"NEW_TRACK", "MATCHED_ADJACENT"}


class IdentityReviewError(RuntimeError):
    """Invalid or inconsistent source tracking artifacts."""


@dataclass(frozen=True)
class IdentityReviewResult:
    report_dir: str
    rendered_clips: int
    rendered_frames: int
    priority_cases: int


def _read_json(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise IdentityReviewError(f"Required tracking input missing or unsafe: {path.name}")
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise IdentityReviewError(f"Could not read {path.name}") from exc
    if not isinstance(parsed, dict):
        raise IdentityReviewError(f"Invalid JSON document: {path.name}")
    return parsed


def _load_and_validate(job_dir: Path, clips: list[tuple[dict, Path]]) -> list[tuple[dict, dict, dict, Path]]:
    pose = _read_json(job_dir / "pose_observations.json")
    tracking = _read_json(job_dir / "pose_tracks_refined.json")
    if pose.get("stage") != "UNVERIFIED_POSE_OBSERVATIONS" or \
       tracking.get("stage") != "UNVERIFIED_EXPERIMENTAL_CLIP_LOCAL_POSE_ASSOCIATIONS":
        raise IdentityReviewError("Unexpected or unverified tracking stage")
    if pose.get("jobId") != tracking.get("jobId") or pose.get("videoId") != tracking.get("videoId"):
        raise IdentityReviewError("Source artifacts refer to different jobs/videos")
    if not pose.get("jobId") or not pose.get("videoId"):
        raise IdentityReviewError("Missing source job metadata")
    pitems, titems = pose.get("annotations"), tracking.get("annotations")
    if not isinstance(pitems, list) or not isinstance(titems, list) or len(pitems) != len(titems) or \
       len(pitems) != len(clips) or len(pitems) > MAX_CLIPS or \
       pose.get("clipCount") != len(clips) or tracking.get("clipCount") != len(clips):
        raise IdentityReviewError("Clip count mismatch")
    validated = []
    for index, ((meta, clip_path), pa, ta) in enumerate(zip(clips, pitems, titems), 1):
        if any(item.get("candidateIndex") != index for item in (meta, pa, ta)):
            raise IdentityReviewError("Candidate index mismatch")
        pf, tf = pa.get("frames"), ta.get("frames")
        if not isinstance(pf, list) or not isinstance(tf, list) or not 1 <= len(pf) <= MAX_FRAMES or len(pf) != len(tf):
            raise IdentityReviewError("Sampled frame count mismatch")
        for frame, tr in zip(pf, tf):
            if frame.get("frameIndex") != tr.get("frameIndex") or frame.get("clipTimeSeconds") != tr.get("clipTimeSeconds"):
                raise IdentityReviewError("Pose and track frame alignment mismatch")
            people, associations = frame.get("people"), tr.get("associations")
            if not isinstance(people, list) or not isinstance(associations, list) or \
               len(people) != len(associations) or len(people) > MAX_PEOPLE:
                raise IdentityReviewError("Detection and association counts differ")
            if [a.get("detectionIndex") for a in associations] != list(range(len(people))):
                raise IdentityReviewError("Track association order/index mismatch")
            track_ids = set()
            for person, a in zip(people, associations):
                tid = a.get("trackId")
                if not isinstance(tid, str) or not TRACK_RE.fullmatch(tid) or not tid.startswith(f"c{index:02d}_"):
                    raise IdentityReviewError("Invalid track ID")
                if tid in track_ids or a.get("associationStatus") not in VALID_STATES:
                    raise IdentityReviewError("Duplicate track or invalid association state")
                track_ids.add(tid)
                box = person.get("bboxXyxyNormalized")
                if not isinstance(box, list) or len(box) != 4 or any(not isinstance(v, (int,float)) or not math.isfinite(v) or v < 0 or v > 1 for v in box) or not(box[0]<box[2] and box[1]<box[3]):
                    raise IdentityReviewError("Invalid normalized detection bbox")
        validated.append((pa, ta, meta, clip_path))
    return validated


def _palette(track_id: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(track_id.encode()).digest()
    return (int(70 + digest[0] % 160), int(70 + digest[1] % 160), int(70 + digest[2] % 160))


def _draw_frame(rgb, pose: dict, tr: dict, *, candidate: int, cv2):
    """Render one sampled frame with stable, clip-local ID colors and uncertainty states."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    for person, assoc in zip(pose["people"], tr["associations"]):
        x1, y1, x2, y2 = person["bboxXyxyNormalized"]
        a, b, c, d = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
        tid = assoc["trackId"]
        color = _palette(tid)
        cv2.rectangle(bgr, (a,b), (c,d), color, 2)
        kp = person.get("keypoints", {})
        if isinstance(kp, dict):
            for name in ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"):
                entry = kp.get(name)
                if isinstance(entry, dict) and entry.get("reported") and entry.get("x") is not None and entry.get("y") is not None:
                    cv2.circle(bgr, (int(entry["x"]*w), int(entry["y"]*h)), 3, color, -1)
        state = assoc["associationStatus"]
        flag = {"AMBIGUOUS_NEW_TRACK": "?", "RECONNECTED_AFTER_GAP": "G", "NEW_TRACK": "N", "MATCHED_ADJACENT": "M"}[state]
        label = f"{tid[-3:]} {flag}"
        baseline_y = max(15, b-5)
        font = cv2.FONT_HERSHEY_SIMPLEX
        size, _ = cv2.getTextSize(label, font, 0.48, 1)
        cv2.rectangle(bgr, (a, baseline_y-size[1]-4), (min(w-1,a+size[0]+5), baseline_y+3), (0,0,0), -1)
        cv2.putText(bgr, label, (a+2, baseline_y), font, 0.48, color, 1, cv2.LINE_AA)
    return bgr


def _contact_sheet(images: list, *, cv2, candidate: int):
    import numpy as np
    tile_w, tile_h = 420, 256
    cols = 3
    rows = (len(images)+cols-1)//cols
    header_h = 76
    canvas = np.full((header_h + rows*tile_h, cols*tile_w, 3), 25, dtype=np.uint8)
    cv2.putText(canvas, f"M5C.5  CLIP {candidate:02d}   UNVERIFIED TRACK IDS", (16, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.73, (240,240,240), 2, cv2.LINE_AA)
    cv2.putText(canvas, "N=new  M=matched  G=gap reconnect  ?=ambiguous     Different colors = different track IDs", (16, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190,190,190), 1, cv2.LINE_AA)
    for i, (bgr, frame) in enumerate(images):
        hh, ww = bgr.shape[:2]
        scale = min((tile_w-12)/ww, (tile_h-32)/hh)
        rw, rh = max(1,int(ww*scale)), max(1,int(hh*scale))
        resized = cv2.resize(bgr, (rw, rh), interpolation=cv2.INTER_AREA)
        row, col = divmod(i, cols)
        x = col*tile_w + (tile_w-rw)//2
        y = header_h + row*tile_h + 26 + (tile_h-32-rh)//2
        canvas[y:y+rh,x:x+rw] = resized
        cap = f"frame {frame['frameIndex']:>3}  t={frame['clipTimeSeconds']:.2f}s  people={len(frame['people'])}"
        cv2.putText(canvas, cap, (col*tile_w+10, header_h+row*tile_h+18), cv2.FONT_HERSHEY_SIMPLEX, 0.51, (240,240,240), 1, cv2.LINE_AA)
    return canvas


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_review(
    job_directory: Path,
    *,
    frame_reader: Callable | None = None,
    clip_validator: Callable | None = None,
) -> IdentityReviewResult:
    """Produce 7 labeled contact sheets, per-clip and priority manual review CSVs.

    Uses existing clips and artifacts; never alters source JSON or trained models.
    Test dependencies can be injected to avoid video decoding in unit tests.
    """
    try:
        import cv2
    except ImportError as exc:
        raise IdentityReviewError("OpenCV dependency missing; run requirements-pose.txt") from exc
    if frame_reader is None:
        from .pose import read_sampled_frames as frame_reader
    if clip_validator is None:
        from .recognition import _validated_clips as clip_validator
    job_dir = Path(job_directory)
    try:
        _meta, clips = clip_validator(job_dir)
    except Exception as exc:
        raise IdentityReviewError(f"M5A clips not available or invalid: {exc}") from exc
    entries = _load_and_validate(job_dir, clips)
    outdir = job_dir / REVIEW_DIR
    if outdir.exists() and (not outdir.is_dir() or outdir.is_symlink()):
        raise IdentityReviewError("Unsafe review output directory")
    if any((outdir / name).exists() for name in ("manual_clip_review.csv", "priority_cases.csv")):
        raise IdentityReviewError("Review CSVs already exist; refusing to overwrite human judgments")
    outdir.mkdir(exist_ok=True)
    priority, overall = [], []
    frame_count = 0
    for candidate, (pa, ta, meta, clip_path) in enumerate(entries, 1):
        decoded = frame_reader(clip_path)
        if len(decoded) != len(pa["frames"]):
            raise IdentityReviewError(f"Clip {candidate}: sampled frame count changed since YOLO inference")
        annotated = []
        for sample, frame, assoc_frame in zip(decoded, pa["frames"], ta["frames"]):
            if sample.frame_index != frame["frameIndex"] or abs(sample.clip_time_seconds-frame["clipTimeSeconds"]) > 0.025:
                raise IdentityReviewError(f"Clip {candidate}: video frames no longer align with saved pose observations")
            bgr = _draw_frame(sample.rgb, frame, assoc_frame, candidate=candidate, cv2=cv2)
            annotated.append((bgr, frame))
            frame_count += 1
            for assoc in assoc_frame["associations"]:
                if assoc["associationStatus"] in PRIORITY_STATES:
                    priority.append({
                        "candidateIndex": candidate,
                        "frameIndex": frame["frameIndex"],
                        "clipTimeSeconds": frame["clipTimeSeconds"],
                        "sourceTimeSeconds": frame["sourceTimeSeconds"],
                        "trackId": assoc["trackId"],
                        "associationStatus": assoc["associationStatus"],
                        "visualVerdict": "UNREVIEWED",
                        "notes": "",
                    })
        sheet = _contact_sheet(annotated, cv2=cv2, candidate=candidate)
        dest = outdir / f"candidate_{candidate:02d}_tracks.jpg"
        if dest.is_symlink():
            raise IdentityReviewError("Unsafe output image path")
        temp = dest.with_suffix(".tmp.jpg")
        if not cv2.imwrite(str(temp), sheet, [cv2.IMWRITE_JPEG_QUALITY, 87]):
            raise IdentityReviewError("Could not write tracking review image")
        os.replace(temp, dest)
        overall.append({
            "candidateIndex": candidate,
            "reviewStatus": "UNREVIEWED",
            "idSwitchObserved": "",
            "nonFighterDetections": "",
            "occlusionOrCameraCut": "",
            "notes": "",
        })
    _write_csv(outdir / "priority_cases.csv", PRIORITY_FIELDS, priority)
    _write_csv(outdir / "manual_clip_review.csv", REVIEW_FIELDS, overall)
    readme = (
        "# M5C.5 — Human validation needed\n\n"
        "These images show **unverified clip-local IDs**, not confirmed fighter identities.\n\n"
        "Inspect candidate_XX_tracks.jpg left-to-right and top-to-bottom; a given color/ID should follow the same visible person.\n"
        "Track labels: N=NEW, M=MATCHED, G=RECONNECTED AFTER GAP, ?=AMBIGUOUS NEW.\n\n"
        "Fill manual_clip_review.csv: reviewStatus=PASS/FAIL/UNCLEAR, idSwitchObserved=YES/NO/UNCLEAR, "
        "nonFighterDetections=YES/NO/UNCLEAR, occlusionOrCameraCut=YES/NO/UNCLEAR.\n"
        "For priority_cases.csv, set visualVerdict=CORRECT/INCORRECT/UNCLEAR where possible. "
        "A new ambiguous ID is intentionally cautious, not automatically an error.\n\n"
        "Do not infer strike type, punch impact, or accurate long-term fighter IDs from these images.\n"
    )
    (outdir / "README.md").write_text(readme, encoding="utf-8")
    manifest = {
        "schemaVersion": 1,
        "stage": "HUMAN_IDENTITY_REVIEW_PENDING",
        "jobId": _read_json(job_dir / "pose_observations.json").get("jobId"),
        "reviewedClips": 0,
        "renderedClips": len(entries),
        "renderedFrames": frame_count,
        "priorityCases": len(priority),
        "limitations": "Unverified, sampled clip-local track identity. Visual human review required before accuracy claims."
    }
    (outdir / "review_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return IdentityReviewResult(str(outdir.resolve()), len(entries), frame_count, len(priority))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render annotated contact sheets for human M5C.5 tracking validation")
    parser.add_argument("--job-dir", required=True, type=Path)
    args = parser.parse_args()
    result = run_review(args.job_dir.resolve())
    print(f"M5C.5 VISUAL REVIEW READY: {result.rendered_clips} clips; {result.rendered_frames} sampled frames")
    print(f"Priority identity review cases: {result.priority_cases}")
    print(f"Review images + CSV: {result.report_dir}")
    print("UNVERIFIED: no fighter identities or punches confirmed. Human visual review required.")


if __name__ == "__main__":
    main()
