"""M5C.2: bounded COCO17 pose observations for existing M5A action clips.

NO punch classification, stable fighter tracking, contact, or hit verification.
YOLO is an opt-in dependency and is lazily imported only for real inference.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from app.action.recognition import RecognitionError, _validated_clips

MODEL_ID = "yolo11n-pose.pt"
METHOD = "yolo11n-coco17-sampled-pose-v1"
MAX_DECODE_FRAMES = 120
MAX_SAMPLE_FRAMES = 12
MAX_FRAME_SIDE = 640
KEYPOINT_MIN_SCORE = 0.30  # visibility reporting only; not punch confidence
MAX_PEOPLE_PER_FRAME = 4  # includes referee or spectators if visible
COCO_KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


class PoseError(RuntimeError):
    """Pose extraction input/inference error with safe message."""


@dataclass(frozen=True)
class FrameSample:
    frame_index: int
    clip_time_seconds: float
    rgb: object


@dataclass(frozen=True)
class PoseRunResult:
    path: str
    clip_count: int
    analyzed_frames: int
    previews: int


class PoseEstimator(Protocol):
    def predict(self, rgb_frame: object, *, preview: bool = False) -> dict: ...


def _sample_indices(frame_count: int, target: int) -> list[int]:
    """Evenly spaced distinct samples for a bounded decoded clip."""
    if frame_count < 1 or not 1 <= target <= MAX_SAMPLE_FRAMES:
        raise PoseError("Invalid pose frame sampling request")
    count = min(frame_count, target)
    if count == 1:
        return [0]
    return [(i * (frame_count - 1)) // (count - 1) for i in range(count)]


def read_sampled_frames(clip_path: Path) -> list[FrameSample]:
    """Decode max 120 frames; downscale before keeping pixels in memory."""
    try:
        import av
    except ImportError as exc:
        raise PoseError("PyAV is missing; install requirements-ai.txt") from exc
    decoded: list[FrameSample] = []
    try:
        with av.open(str(clip_path)) as container:
            first_time: float | None = None
            for idx, frame in enumerate(container.decode(video=0)):
                if idx >= MAX_DECODE_FRAMES:
                    raise PoseError("Pose input exceeds decoded frame limit")
                if frame.width <= 0 or frame.height <= 0:
                    raise PoseError("Pose input has invalid frame dimensions")
                ratio = min(1.0, MAX_FRAME_SIDE / max(frame.width, frame.height))
                width = max(2, int(frame.width * ratio) // 2 * 2)
                height = max(2, int(frame.height * ratio) // 2 * 2)
                small = frame.reformat(width=width, height=height, format="rgb24")
                raw_time = float(frame.time) if frame.time is not None else idx / 12.0
                if not math.isfinite(raw_time):
                    raise PoseError("Pose input has an invalid frame timestamp")
                if first_time is None:
                    first_time = raw_time
                seconds = max(0.0, raw_time - first_time)
                decoded.append(FrameSample(idx, round(seconds, 4), small.to_ndarray()))
    except PoseError:
        raise
    except Exception as exc:
        raise PoseError("Could not decode action clip for pose estimation") from exc
    if not decoded:
        raise PoseError("Pose input contains no readable frames")
    sampled = [decoded[i] for i in _sample_indices(len(decoded), MAX_SAMPLE_FRAMES)]
    return sampled


def _format_keypoints(keypoint_xy: list, keypoint_scores: list) -> dict:
    if len(keypoint_xy) != len(COCO_KEYPOINT_NAMES) or len(keypoint_scores) != len(COCO_KEYPOINT_NAMES):
        raise PoseError("YOLO returned an unexpected keypoint layout")
    result = {}
    for name, coordinates, score in zip(COCO_KEYPOINT_NAMES, keypoint_xy, keypoint_scores):
        probability = float(score)
        if not math.isfinite(probability):
            raise PoseError("YOLO returned a non-finite keypoint score")
        x, y = float(coordinates[0]), float(coordinates[1])
        visible = probability >= KEYPOINT_MIN_SCORE and math.isfinite(x) and math.isfinite(y)
        result[name] = {
            "x": round(max(0.0, min(1.0, x)), 5) if visible else None,
            "y": round(max(0.0, min(1.0, y)), 5) if visible else None,
            "score": round(probability, 5),
            "reported": visible,
        }
    return result


class Yolo11PoseEstimator:
    """Single YOLO11n pose model reused across all clips in one CLI run."""

    def __init__(self):
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise PoseError("YOLO Pose dependencies missing; install requirements-pose.txt") from exc
        if not torch.cuda.is_available():
            raise PoseError("CUDA GPU unavailable; M5C.2 GPU pilot requires CUDA")
        try:
            self.model = YOLO(MODEL_ID)
        except Exception as exc:
            raise PoseError("Could not load official YOLO11n-Pose checkpoint") from exc

    def predict(self, rgb_frame: object, *, preview: bool = False) -> dict:
        try:
            import cv2
            # Model accepts RGB arrays; Ultralytics numpy input expects BGR.
            bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            output = self.model.predict(
                source=bgr, device=0, imgsz=640, conf=0.25,
                max_det=MAX_PEOPLE_PER_FRAME, classes=[0], verbose=False,
            )[0]
            people = []
            if output.boxes is not None and output.keypoints is not None:
                boxes_xy = output.boxes.xyxyn.cpu().tolist()
                box_scores = output.boxes.conf.cpu().tolist()
                points_xy = output.keypoints.xyn.cpu().tolist()
                point_scores_tensor = output.keypoints.conf
                point_scores = (point_scores_tensor.cpu().tolist()
                                if point_scores_tensor is not None
                                else [[1.0] * len(COCO_KEYPOINT_NAMES) for _ in points_xy])
                for box, box_score, pts, pts_scores in zip(
                    boxes_xy, box_scores, points_xy, point_scores
                ):
                    people.append({
                        "personDetectionScore": round(float(box_score), 5),
                        "bboxXyxyNormalized": [round(max(0.0, min(1.0, float(c))), 5) for c in box],
                        "keypoints": _format_keypoints(pts, pts_scores),
                    })
            preview_jpeg = None
            if preview:
                ok, jpeg = cv2.imencode(".jpg", output.plot(), [cv2.IMWRITE_JPEG_QUALITY, 82])
                if not ok:
                    raise PoseError("Could not encode YOLO preview")
                preview_jpeg = jpeg.tobytes()
            return {"people": people, "previewJpeg": preview_jpeg}
        except PoseError:
            raise
        except Exception as exc:
            raise PoseError("YOLO pose inference failed") from exc


def run_pose_observations(
    job_directory: Path,
    *,
    estimator: PoseEstimator | None = None,
    frame_reader: Callable[[Path], list[FrameSample]] = read_sampled_frames,
) -> PoseRunResult:
    """Analyze existing clips, write separate preview JPEGs + atomic JSON.

    Does not change M5A/M5B manifest, action_predictions.json, or backend API.
    Fake estimator/reader may be injected for unit tests (no GPU required).
    """
    job_directory = Path(job_directory)
    try:
        clip_doc, clips = _validated_clips(job_directory)
    except RecognitionError as exc:
        raise PoseError(str(exc)) from exc
    model = estimator if estimator is not None else (Yolo11PoseEstimator() if clips else None)
    preview_dir = job_directory / "pose_previews"
    if preview_dir.is_symlink():
        raise PoseError("Unsafe pose preview output location")
    preview_dir.mkdir(exist_ok=True)
    if not preview_dir.is_dir():
        raise PoseError("Pose preview output directory is invalid")
    entries: list[dict] = []
    total_frames = 0
    preview_count = 0
    for clip_info, clip_path in clips:
        sampled = frame_reader(clip_path)
        if not 1 <= len(sampled) <= MAX_SAMPLE_FRAMES:
            raise PoseError("Pose sampler returned an invalid frame count")
        observations = []
        preview_path = None
        for i, frame in enumerate(sampled):
            should_preview = i == len(sampled) // 2
            raw = model.predict(frame.rgb, preview=should_preview)
            if not isinstance(raw, dict) or not isinstance(raw.get("people"), list):
                raise PoseError("Pose estimator returned invalid results")
            for person in raw["people"]:
                if not isinstance(person, dict) or len(person.get("keypoints", {})) != 17:
                    raise PoseError("Pose estimator returned invalid keypoints")
            observed = {
                "frameIndex": frame.frame_index,
                "clipTimeSeconds": frame.clip_time_seconds,
                "sourceTimeSeconds": round(float(clip_info["startSeconds"]) + frame.clip_time_seconds, 4),
                "personCount": len(raw["people"]),
                "people": raw["people"],
            }
            observations.append(observed)
            if should_preview and raw.get("previewJpeg") is not None:
                jpeg = raw["previewJpeg"]
                if not isinstance(jpeg, bytes) or not jpeg.startswith(b"\xff\xd8"):
                    raise PoseError("Pose estimator returned invalid preview bytes")
                jpg = preview_dir / f"candidate_{clip_info['candidateIndex']:02d}.jpg"
                if jpg.is_symlink():
                    raise PoseError("Unsafe pose preview file location")
                temp_jpg = jpg.with_suffix(".jpg.tmp")
                temp_jpg.write_bytes(jpeg)
                os.replace(temp_jpg, jpg)
                preview_path = str(jpg.resolve())
                preview_count += 1
        total_frames += len(observations)
        entries.append({
            "candidateIndex": clip_info["candidateIndex"],
            "startSeconds": clip_info["startSeconds"],
            "endSeconds": clip_info["endSeconds"],
            "sampledFrameCount": len(observations),
            "framesWithPerson": sum(o["personCount"] >= 1 for o in observations),
            "framesWithTwoOrMorePeople": sum(o["personCount"] >= 2 for o in observations),
            "previewPath": preview_path,
            "frames": observations,
        })
    doc = {
        "schemaVersion": 1,
        "stage": "UNVERIFIED_POSE_OBSERVATIONS",
        "method": METHOD,
        "modelId": MODEL_ID,
        "jobId": clip_doc.get("jobId"),
        "videoId": clip_doc.get("videoId"),
        "clipCount": len(entries),
        "analyzedFrameCount": total_frames,
        "previewCount": preview_count,
        "coordinateSystem": "xy normalized to the input clip frame (range 0..1)",
        "keypointReportingThreshold": KEYPOINT_MIN_SCORE,
        "annotations": entries,
        "combatEventsDetected": False,
        "limitations": (
            "COCO17 body keypoints only. Person index is per-frame, not stable fighter identity. "
            "No jab/cross/hook classification, punch count, contact, impact, or verified event. "
            "Occlusion, broadcast cuts, referee, and spectator detections may occur."
        ),
    }
    output_path = job_directory / "pose_observations.json"
    if output_path.is_symlink():
        raise PoseError("Unsafe pose observation output location")
    temp_path = output_path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, output_path)
    return PoseRunResult(str(output_path.resolve()), len(entries), total_frames, preview_count)


def main() -> None:
    parser = argparse.ArgumentParser(description="M5C.2: YOLO11n COCO17 pose on existing clips")
    parser.add_argument("--job-dir", required=True, type=Path)
    args = parser.parse_args()
    result = run_pose_observations(args.job_dir.resolve())
    print(f"M5C.2 completed: {result.clip_count} clips, {result.analyzed_frames} sampled frames")
    print(f"Pose previews written: {result.previews}")
    print(f"Pose JSON: {result.path}")
    print("WARNING: keypoints are observations, not confirmed punches, contacts, or fighter IDs.")


if __name__ == "__main__":
    main()
