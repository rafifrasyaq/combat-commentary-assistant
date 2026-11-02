"""Prepare short video clips around M4 motion candidates for later model inference.

This module extracts media only. It does NOT detect techniques or generate
confidence scores. In particular, M4 motion candidates can be editing cuts.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from app.media.preprocessing import MediaDecodeError, MediaTimeout, MediaToolMissing, _tools

CLIP_METHOD = "bounded-temporal-clip-extraction-v1"
CLIP_FPS = 12
MAX_WIDTH = 640
MAX_CLIPS = 8
MAX_CLIP_SECONDS = 5.0
CONTEXT_SECONDS = 0.75


@dataclass(frozen=True)
class ClipPreparationResult:
    path: str
    count: int
    clips: list[dict]


def _clip_window(candidate: dict, total_duration: float) -> tuple[float, float]:
    """Choose an in-bounds clip, including context around a validated peak."""
    try:
        start = float(candidate["startSeconds"])
        peak = float(candidate["peakSeconds"])
        end = float(candidate["endSeconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MediaDecodeError("Motion candidate lacks usable timestamps") from exc
    if not all(math.isfinite(n) for n in (start, peak, end, total_duration)):
        raise MediaDecodeError("Non-finite motion candidate timestamp")
    if total_duration <= 0 or not (0 <= start <= peak <= end <= total_duration + 0.01):
        raise MediaDecodeError("Motion candidate timestamps are outside the video")
    # For very long events retain the region around the peak; do not silently
    # claim to have extracted the whole event.
    clip_start = max(0.0, start - CONTEXT_SECONDS)
    clip_end = min(total_duration, end + CONTEXT_SECONDS)
    if clip_end - clip_start > MAX_CLIP_SECONDS:
        clip_start = max(0.0, peak - MAX_CLIP_SECONDS / 2)
        clip_end = min(total_duration, clip_start + MAX_CLIP_SECONDS)
        clip_start = max(0.0, clip_end - MAX_CLIP_SECONDS)
    if clip_end - clip_start < 0.25:
        # A candidate right on a boundary still needs a decodable interval.
        clip_start = max(0.0, peak - 0.5)
        clip_end = min(total_duration, max(clip_start + 0.25, peak + 0.5))
    return round(clip_start, 3), round(clip_end, 3)


def prepare_action_clips(*, video_path: Path, video_id: UUID, job_id: UUID,
                         job_directory: Path, motion_candidates: list[dict],
                         duration_seconds: float) -> ClipPreparationResult:
    """Extract up to eight 12 FPS MPEG-4 clips and atomically publish clip metadata.

    The prior /analyze boundary has validated the storage path. Clip output is
    only written under the UUID-owned job directory created by preprocessing.
    """
    if not isinstance(motion_candidates, list) or len(motion_candidates) > MAX_CLIPS:
        raise MediaDecodeError("Unexpected number of motion candidates")
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise MediaDecodeError("Video duration must be positive")
    if not job_directory.is_dir():
        raise MediaDecodeError("Job artifact directory is missing")
    _, ffmpeg = _tools()
    clip_directory = job_directory / "action_clips"
    clip_directory.mkdir(exist_ok=False)
    results: list[dict] = []
    for index, candidate in enumerate(motion_candidates, start=1):
        if not isinstance(candidate, dict):
            raise MediaDecodeError("Invalid motion candidate")
        start, end = _clip_window(candidate, duration_seconds)
        target = clip_directory / f"candidate_{index:02d}.mp4"
        command = [
            ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start:.3f}", "-i", str(video_path),
            "-t", f"{end-start:.3f}", "-map", "0:v:0", "-an", "-sn", "-dn",
            "-vf", f"fps={CLIP_FPS},scale={MAX_WIDTH}:-2:flags=fast_bilinear",
            "-c:v", "mpeg4", "-q:v", "5", "-movflags", "+faststart", str(target),
        ]
        try:
            subprocess.run(command, capture_output=True, check=True, timeout=45)
        except FileNotFoundError as exc:
            raise MediaToolMissing("FFmpeg executable is not installed") from exc
        except subprocess.TimeoutExpired as exc:
            raise MediaTimeout("Extracting an action clip exceeded the time limit") from exc
        except subprocess.CalledProcessError as exc:
            raise MediaDecodeError("Could not extract an action clip") from exc
        if not target.is_file() or target.stat().st_size == 0:
            raise MediaDecodeError("Action clip extraction produced no video")
        results.append({
            "candidateIndex": index,
            "startSeconds": start,
            "endSeconds": end,
            "peakSeconds": float(candidate["peakSeconds"]),
            "path": str(target.resolve()),
            "durationSeconds": round(end-start, 3),
            "sourceType": "MOTION_ACTIVITY_CANDIDATE",
            "modelPredictionStatus": "NOT_RUN",
        })

    metadata = {
        "schemaVersion": 1,
        "stage": "ACTION_CLIPS_PREPARED_ONLY",
        "method": CLIP_METHOD,
        "jobId": str(job_id),
        "videoId": str(video_id),
        "clipSettings": {"fps": CLIP_FPS, "maxWidth": MAX_WIDTH, "maxClipSeconds": MAX_CLIP_SECONDS},
        "clipCount": len(results),
        "clips": results,
        "modelInference": {"status": "NOT_CONFIGURED", "predictedEvents": []},
        "limitations": (
            "Action clips are inputs for future AI inference, NOT action predictions. "
            "Motion candidates may reflect camera motion or editing. "
            "A clip being present does not mean a strike occurred."
        ),
    }
    final = job_directory / "action_clips.json"
    temp = job_directory / "action_clips.json.tmp"
    temp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, final)
    return ClipPreparationResult(path=str(final.resolve()), count=len(results), clips=results)
