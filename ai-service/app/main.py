"""AI service boundary — milestone 3 video/media preprocessing (not combat inference)."""

import os
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

app = FastAPI(title="Combat Commentary AI Service", version="0.2.0")


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
    return {"status": "UP", "service": "combat-commentary-ai-preprocessor"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    # Dev-only local filesystem contract; object storage will replace this in deployment.
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
    try:
        result = preprocess(video_path=file_path, video_id=payload.videoId, job_id=payload.jobId,
                            artifact_root=artifacts)
    except MediaToolMissing as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MediaTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except MediaDecodeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return AnalyzeResponse(
        engine="ffmpeg-preprocessing-v1",
        fileSizeBytes=file_path.stat().st_size,
        summary=(f"Media preprocessed: {result.duration_seconds:.1f}s, "
                 f"{result.frame_count} frames, "
                 f"audio preview {'yes' if result.audio_extracted else 'no'}; "
                 "combat analysis not yet implemented."),
        manifestPath=result.manifest_path,
    )
