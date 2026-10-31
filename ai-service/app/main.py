"""Development-only processing boundary; no video understanding or commentary yet."""

import os
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI(title="Combat Commentary AI Service", version="0.1.0")


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    jobId: UUID
    videoId: UUID
    storagePath: str = Field(min_length=1)


class AnalyzeResponse(BaseModel):
    engine: str
    fileSizeBytes: int
    summary: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "UP", "service": "combat-commentary-ai-stub"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    # The local dev worker has read-only access to an explicitly configured folder.
    root_setting = os.getenv("VIDEO_STORAGE_ROOT", "../backend/storage/videos")
    root = Path(root_setting).resolve()
    file_path = Path(payload.storagePath).resolve()
    if not file_path.is_relative_to(root) or file_path.suffix.lower() not in {".mp4", ".mov", ".mkv"}:
        raise HTTPException(status_code=422, detail="Video path is outside approved storage")
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Video file not found")
    size = file_path.stat().st_size
    if size <= 0:
        raise HTTPException(status_code=422, detail="Video file is empty")

    # No inference: returning only verified file metadata deliberately prevents fabricated moments.
    return AnalyzeResponse(
        engine="filesystem-stub-v1",
        fileSizeBytes=size,
        summary="Video readable by AI service; combat analysis has not been implemented.",
    )
