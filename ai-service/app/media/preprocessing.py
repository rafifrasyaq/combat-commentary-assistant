"""Bounded, deterministic video preprocessing via installed FFmpeg tools.

A manifest is stored on local disk for future computer-vision and ASR stages.
The service never claims to have recognized fighter identities or combat events.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from uuid import UUID


class MediaToolMissing(Exception):
    pass


class MediaDecodeError(Exception):
    pass


class MediaTimeout(Exception):
    pass


@dataclass(frozen=True)
class PreprocessingResult:
    manifest_path: str
    frame_count: int
    audio_extracted: bool
    duration_seconds: float


def _tools() -> tuple[str, str]:
    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    if ffprobe is None or ffmpeg is None:
        raise MediaToolMissing("ffprobe and ffmpeg must be installed and available on PATH")
    return ffprobe, ffmpeg


def _run(args: list[str], *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise MediaToolMissing("FFmpeg tools are missing") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaTimeout("Video preprocessing took too long") from exc
    except subprocess.CalledProcessError as exc:
        raise MediaDecodeError("Could not decode video or audio") from exc


def _frame_rate(video_stream: dict) -> float | None:
    value = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")
    try:
        rate = float(Fraction(str(value)))
        return round(rate, 3) if 0 < rate <= 1000 else None
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def _sample_times(duration: float, max_frames: int) -> list[float]:
    """Evenly spaced representative timestamps, excluding exact end-of-file."""
    if duration <= 0:
        return []
    usable = max(0.0, duration - 0.1)
    count = min(max_frames, max(1, int(duration) + 1))
    return [round(usable * (i + 0.5) / count, 3) for i in range(count)]


def preprocess(
    *,
    video_path: Path,
    video_id: UUID,
    job_id: UUID,
    artifact_root: Path,
    max_frames: int = 6,
    audio_seconds: int = 20,
) -> PreprocessingResult:
    """Decode authentic metadata, sample frames/audio and atomically write manifest.

    Only called *after* HTTP-boundary storage-path validation. Output is tied to
    a unique processing job so retries never overwrite other attempts.
    """
    if not (1 <= max_frames <= 12 and 0 <= audio_seconds <= 30):
        raise ValueError("Extraction limits out of range")
    ffprobe, ffmpeg = _tools()
    probe_output = _run([
        ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(video_path),
    ], timeout=25)
    try:
        data = json.loads(probe_output.stdout)
        streams = data.get("streams", [])
        video = next(s for s in streams if s.get("codec_type") == "video")
        audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
        duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0)
        width = int(video.get("width") or 0)
        height = int(video.get("height") or 0)
        if duration <= 0 or not (width > 0 and height > 0):
            raise ValueError("Video is missing a valid duration or dimensions")
    except (StopIteration, ValueError, TypeError, KeyError) as exc:
        raise MediaDecodeError("No decodable video stream with usable metadata") from exc

    artifact_root = artifact_root.resolve()
    job_dir = artifact_root / str(job_id)
    # Never accept output directory as an existing job; avoid stale results.
    try:
        job_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise MediaDecodeError("Output for this processing job already exists") from exc

    try:
        frames_dir = job_dir / "frames"
        frames_dir.mkdir()
        frames = []
        for index, seconds in enumerate(_sample_times(duration, max_frames), start=1):
            dest = frames_dir / f"frame_{index:02d}.jpg"
            _run([
                ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{seconds:.3f}", "-i", str(video_path),
                "-map", "0:v:0", "-frames:v", "1", "-vf", "scale=960:-2", str(dest),
            ], timeout=30)
            if not dest.is_file() or dest.stat().st_size == 0:
                raise MediaDecodeError("Frame extraction produced an empty output")
            frames.append({"timestampSeconds": seconds, "path": str(dest.resolve())})

        audio_path = None
        if audio is not None and audio_seconds > 0:
            dest = job_dir / "audio_preview.wav"
            _run([
                ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(video_path), "-map", "0:a:0", "-t", str(audio_seconds),
                "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dest),
            ], timeout=45)
            if dest.is_file() and dest.stat().st_size > 44:
                audio_path = str(dest.resolve())

        manifest = {
            "schemaVersion": 1,
            "jobId": str(job_id),
            "videoId": str(video_id),
            "stage": "MEDIA_PREPROCESSING_ONLY",
            "combatEventsDetected": False,
            "metadata": {
                "durationSeconds": round(duration, 3),
                "width": width,
                "height": height,
                "fps": _frame_rate(video),
                "videoCodec": video.get("codec_name"),
                "hasAudio": audio is not None,
                "fileSizeBytes": video_path.stat().st_size,
            },
            "frames": frames,
            "audioPreviewPath": audio_path,
        }
        temporary = job_dir / "manifest.json.tmp"
        final = job_dir / "manifest.json"
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, final)
        return PreprocessingResult(
            manifest_path=str(final.resolve()),
            frame_count=len(frames),
            audio_extracted=audio_path is not None,
            duration_seconds=duration,
        )
    except Exception:
        # Avoid falsely returning stale/partial manifests from a failed run.
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
