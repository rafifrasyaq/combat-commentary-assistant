"""Optional M5B baseline: unverified Kinetics-400 activity labels for M5A clips.

This is NOT punch-technique recognition, fighter tracking, or contact detection.
Heavy ML dependencies load lazily, so M5A works without torch/transformers.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"
METHOD = "videomae-kinetics400-activity-baseline-v1"
MAX_CLIPS = 8
MAX_CLIP_BYTES = 40_000_000
MAX_DECODE_FRAMES = 120
TOP_K = 5
MIN_REPORTED_SCORE = 0.40  # A UI abstention rule, NOT a calibrated confidence cutoff.


class RecognitionError(RuntimeError):
    """Explicit failure to obtain trustworthy baseline predictions."""


@dataclass(frozen=True)
class RecognitionResult:
    path: str
    clip_count: int
    unknown_count: int
    model_id: str


class ActivityClassifier(Protocol):
    def predict(self, clip_path: Path) -> dict: ...


def uniform_indices(frame_count: int, sample_count: int) -> list[int]:
    """Uniformly sample frames with repeat when a short clip lacks enough frames."""
    if frame_count < 1 or not 1 <= sample_count <= 64:
        raise RecognitionError("Invalid frame sampling request")
    if sample_count == 1:
        return [0]
    return [min(frame_count - 1, (i * (frame_count - 1)) // (sample_count - 1))
            for i in range(sample_count)]


def _decode_rgb_frames(clip_path: Path) -> list:
    """Decode a strictly bounded, already-prepared MP4. PyAV is lazy-imported."""
    try:
        import av
    except ImportError as exc:
        raise RecognitionError("PyAV is missing; install requirements-ai.txt") from exc
    frames = []
    try:
        with av.open(str(clip_path)) as container:
            for frame in container.decode(video=0):
                if len(frames) >= MAX_DECODE_FRAMES:
                    raise RecognitionError("Action clip has too many frames")
                frames.append(frame.to_ndarray(format="rgb24"))
    except RecognitionError:
        raise
    except Exception as exc:
        raise RecognitionError("Action clip could not be decoded") from exc
    if not frames:
        raise RecognitionError("Action clip contains no readable video frames")
    return frames


class VideoMaeActivityClassifier:
    """One model per Python process; inference serialized for bounded VRAM usage."""

    def __init__(self):
        try:
            import torch
            from transformers import VideoMAEForVideoClassification, VideoMAEVideoProcessor
        except ImportError as exc:
            raise RecognitionError("VideoMAE dependencies missing; install requirements-ai.txt") from exc
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device != "cuda":
            raise RecognitionError("CUDA GPU unavailable; M5B baseline currently requires CUDA")
        try:
            self.processor = VideoMAEVideoProcessor.from_pretrained(MODEL_ID)
            self.model = VideoMAEForVideoClassification.from_pretrained(MODEL_ID).to(self.device).eval()
        except Exception as exc:
            raise RecognitionError("Could not load pretrained VideoMAE checkpoint") from exc
        self.num_frames = int(self.model.config.num_frames)
        if not 1 <= self.num_frames <= 64:
            raise RecognitionError("Model requires an unsupported frame count")
        self.lock = threading.Lock()

    def predict(self, clip_path: Path) -> dict:
        frames = _decode_rgb_frames(clip_path)
        selected = [frames[i] for i in uniform_indices(len(frames), self.num_frames)]
        del frames
        try:
            batch = self.processor(selected, return_tensors="pt", do_sample_frames=False)
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with self.lock, self.torch.inference_mode():
                logits = self.model(**batch).logits[0]
                probabilities = logits.softmax(dim=-1)
                values, indices = probabilities.topk(min(TOP_K, probabilities.numel()))
            top = [{"label": str(self.model.config.id2label[int(idx)]),
                    "softmaxScore": round(float(score), 6)}
                   for score, idx in zip(values.tolist(), indices.tolist())]
        except Exception as exc:
            raise RecognitionError("VideoMAE inference failed") from exc
        leader = top[0]
        unknown = leader["softmaxScore"] < MIN_REPORTED_SCORE
        return {
            "activityLabel": "UNKNOWN" if unknown else leader["label"],
            "status": "UNKNOWN_LOW_SCORE" if unknown else "UNVERIFIED_ACTIVITY_LABEL",
            "topPredictions": top,
            "sampledFrameCount": len(selected),
            "inferenceDevice": self.device,
        }


_model_load_lock = threading.Lock()


@lru_cache(maxsize=1)
def _cached_classifier() -> ActivityClassifier:
    return VideoMaeActivityClassifier()


def _get_classifier() -> ActivityClassifier:
    # The lock also prevents duplicate model loads if jobs arrive concurrently.
    with _model_load_lock:
        return _cached_classifier()


def recognition_enabled() -> bool:
    return os.getenv("ACTION_RECOGNITION_ENABLED", "false").strip().lower() in {"1", "true", "yes"}


def _atomic_json(target: Path, data: dict) -> None:
    temp = target.with_name(target.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, target)


def _validated_clips(job_directory: Path) -> tuple[dict, list[tuple[dict, Path]]]:
    metadata_path = job_directory / "action_clips.json"
    if not metadata_path.is_file() or metadata_path.is_symlink():
        raise RecognitionError("M5A action_clips.json is missing")
    try:
        doc = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RecognitionError("Invalid action_clips.json") from exc
    clips = doc.get("clips")
    if not isinstance(clips, list) or len(clips) > MAX_CLIPS or doc.get("clipCount") != len(clips):
        raise RecognitionError("Invalid M5A clip count")
    expected_dir = job_directory / "action_clips"
    if not expected_dir.is_dir() or expected_dir.is_symlink():
        raise RecognitionError("M5A clip directory missing or unsafe")
    validated = []
    for index, item in enumerate(clips, 1):
        if not isinstance(item, dict) or item.get("candidateIndex") != index:
            raise RecognitionError("M5A clip metadata has invalid index")
        expected = expected_dir / f"candidate_{index:02d}.mp4"
        try:
            supplied = Path(item["path"])
            if supplied.resolve() != expected.resolve() or supplied.is_symlink() or expected.is_symlink():
                raise RecognitionError("M5A clip path is outside expected directory")
            if not expected.is_file() or not 0 < expected.stat().st_size <= MAX_CLIP_BYTES:
                raise RecognitionError("M5A clip missing or exceeds size limit")
            start, end = float(item["startSeconds"]), float(item["endSeconds"])
            if not math.isfinite(start) or not math.isfinite(end) or not (0 <= start < end <= start + 5.01):
                raise RecognitionError("M5A clip timestamps invalid")
        except (KeyError, ValueError, TypeError, OSError) as exc:
            raise RecognitionError("Invalid M5A clip metadata") from exc
        validated.append((item, expected))
    return doc, validated


def run_action_recognition(job_directory: Path, classifier: ActivityClassifier | None = None) -> RecognitionResult:
    """Classify prepared clips and publish a separate JSON artifact atomically.

    A classifier can be injected in unit tests without downloading a model.
    """
    job_directory = Path(job_directory)
    source, clips = _validated_clips(job_directory)
    model = classifier if classifier is not None else (_get_classifier() if clips else None)
    predictions = []
    for item, path in clips:
        try:
            raw = model.predict(path)
        except RecognitionError:
            raise
        except Exception as exc:
            raise RecognitionError("Activity classifier failed") from exc
        if not isinstance(raw, dict) or raw.get("status") not in {
            "UNKNOWN_LOW_SCORE", "UNVERIFIED_ACTIVITY_LABEL"
        }:
            raise RecognitionError("Classifier returned an invalid prediction")
        predictions.append({
            "candidateIndex": item["candidateIndex"],
            "startSeconds": item["startSeconds"],
            "endSeconds": item["endSeconds"],
            "peakSeconds": item["peakSeconds"],
            "activityLabel": raw["activityLabel"],
            "status": raw["status"],
            "topPredictions": raw["topPredictions"],
            "sampledFrameCount": raw["sampledFrameCount"],
            "inferenceDevice": raw["inferenceDevice"],
        })
    unknown_count = sum(p["status"] == "UNKNOWN_LOW_SCORE" for p in predictions)
    artifact = {
        "schemaVersion": 1,
        "stage": "UNVERIFIED_ACTIVITY_PREDICTIONS",
        "analysisType": "PRETRAINED_KINETICS400_ACTIVITY_ONLY",
        "model": {"id": MODEL_ID, "method": METHOD, "scoreType": "uncalibrated_softmax",
                  "abstentionThreshold": MIN_REPORTED_SCORE},
        "jobId": source.get("jobId"),
        "videoId": source.get("videoId"),
        "clipCount": len(predictions),
        "unknownCount": unknown_count,
        "predictions": predictions,
        "combatEventsDetected": False,
        "limitations": (
            "Kinetics-400 general activities only. Scores are not calibrated probabilities of actual strikes; "
            "a boxing label does not establish which fighter punched, technique, contact, or impact. "
            "Upstream motion candidates may miss activity or include camera cuts."
        ),
    }
    path = job_directory / "action_predictions.json"
    _atomic_json(path, artifact)
    return RecognitionResult(path=str(path.resolve()), clip_count=len(predictions),
                             unknown_count=unknown_count, model_id=MODEL_ID)


def publish_recognition_manifest(manifest_path: Path, recognition: RecognitionResult) -> None:
    """Extend M5A manifest without changing backend-facing response contracts."""
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RecognitionError("Could not read M5A manifest") from exc
    if "actionPreparation" not in manifest:
        raise RecognitionError("Action preparation is missing from manifest")
    manifest["schemaVersion"] = 4
    manifest["stage"] = "UNVERIFIED_ACTIVITY_PREDICTIONS"
    manifest["combatEventsDetected"] = False
    manifest["actionPreparation"]["modelInferenceStatus"] = "COMPLETED"
    manifest["actionRecognition"] = {
        "modelInferenceStatus": "COMPLETED",
        "modelId": recognition.model_id,
        "resultPath": recognition.path,
        "clipCount": recognition.clip_count,
        "unknownCount": recognition.unknown_count,
        "combatEventsVerified": False,
    }
    _atomic_json(Path(manifest_path), manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VideoMAE on existing M5A clips without re-upload")
    parser.add_argument("--job-dir", type=Path, required=True, help="Existing backend/storage/analysis/<jobId> directory")
    args = parser.parse_args()
    job_dir = args.job_dir.resolve()
    result = run_action_recognition(job_dir)
    publish_recognition_manifest(job_dir / "manifest.json", result)
    print(f"M5B inference completed: {result.clip_count} clips; {result.unknown_count} UNKNOWN")
    print(f"Predictions JSON: {result.path}")
    print("These are UNVERIFIED activity labels, NOT verified combat events.")


if __name__ == "__main__":
    main()
