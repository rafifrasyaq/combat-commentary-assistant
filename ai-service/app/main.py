"""AI-service boundary: media preprocessing plus bounded motion *candidates*."""

import json
import os
import shutil
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.media.preprocessing import (
    MediaDecodeError,
    MediaTimeout,
    MediaToolMissing,
    preprocess,
)
from app.motion.detection import METHOD, detect_motion
from app.action.clips import CLIP_METHOD, prepare_action_clips

app = FastAPI(title="Combat Commentary AI Service", version="0.4.0")


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    jobId: UUID
    videoId: UUID
    storagePath: str = Field(min_length=1)


class AnalyzeResponse(BaseModel):
    engine: str
    fileSizeBytes: int
    summary: str
    manifestPath: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "UP", "service": "combat-commentary-ai-motion-candidates"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    default_storage = Path(__file__).resolve().parents[2] / "backend" / "storage" / "videos"
    root = Path(os.getenv("VIDEO_STORAGE_ROOT", str(default_storage))).resolve()
    file_path = Path(payload.storagePath).resolve()
    if not file_path.is_relative_to(root) or file_path.suffix.lower() not in {".mp4", ".mov", ".mkv"}:
        raise HTTPException(status_code=422, detail="Video path is outside approved storage")
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Video file not found")
    if file_path.stat().st_size <= 0:
        raise HTTPException(status_code=422, detail="Video file is empty")

    default_artifacts = Path(__file__).resolve().parents[2] / "backend" / "storage" / "analysis"
    artifacts = Path(os.getenv("VIDEO_ARTIFACT_ROOT", str(default_artifacts)))
    result = None
    try:
        result = preprocess(
            video_path=file_path, video_id=payload.videoId, job_id=payload.jobId,
            artifact_root=artifacts,
        )
        job_dir = Path(result.manifest_path).parent
        motion = detect_motion(
            video_path=file_path, video_id=payload.videoId, job_id=payload.jobId,
            job_directory=job_dir, duration_seconds=result.duration_seconds,
        )
        # M5A: prepare short, real video clips for a future video-action model.
        # This operation intentionally does NOT run a learned classifier.
        clips = prepare_action_clips(
            video_path=file_path, video_id=payload.videoId, job_id=payload.jobId,
            job_directory=job_dir, motion_candidates=motion.candidates,
            duration_seconds=result.duration_seconds,
        )
        # Upgrade manifest only after the motion and action-clip artifacts were fully written.
        # A temporary file and atomic replacement prevent partial JSON reads.
        manifest_path = Path(result.manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["schemaVersion"] = 3
        manifest["stage"] = "ACTION_CLIPS_PREPARED_ONLY"
        manifest["combatEventsDetected"] = False
        manifest["motionAnalysis"] = {
            "method": METHOD,
            "resultPath": motion.path,
            "candidateCount": len(motion.candidates),
            "analyzedDurationSeconds": motion.analyzed_seconds,
            "truncatedByDurationLimit": motion.truncated,
        }
        manifest["actionPreparation"] = {
            "method": CLIP_METHOD,
            "resultPath": clips.path,
            "clipCount": clips.count,
            "modelInferenceStatus": "NOT_CONFIGURED",
        }
        temp_path = job_dir / "manifest.json.tmp"
        temp_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp_path, manifest_path)
    except (MediaToolMissing, MediaTimeout, MediaDecodeError) as exc:
        if result is not None:
            shutil.rmtree(Path(result.manifest_path).parent, ignore_errors=True)
        code = 503 if isinstance(exc, MediaToolMissing) else 504 if isinstance(exc, MediaTimeout) else 422
        raise HTTPException(status_code=code, detail=str(exc)) from exc
    except Exception:
        if result is not None:
            shutil.rmtree(Path(result.manifest_path).parent, ignore_errors=True)
        raise

    return AnalyzeResponse(
        engine="ffmpeg-motion-clips-v1",
        fileSizeBytes=file_path.stat().st_size,
        summary=(f"Media preprocessed: {result.duration_seconds:.1f}s, "
                 f"{result.frame_count} still frames, "
                 f"audio preview {'yes' if result.audio_extracted else 'no'}; "
                 f"motion analysis {motion.analyzed_seconds:.1f}s, "
                 f"{len(motion.candidates)} motion candidates, "
                 f"{motion.excluded_scene_changes} likely scene cuts excluded; "
                 f"{clips.count} action-ready clips; model inference NOT RUN."),
        manifestPath=result.manifest_path,
    )
