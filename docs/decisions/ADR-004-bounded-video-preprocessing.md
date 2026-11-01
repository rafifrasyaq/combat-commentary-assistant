# ADR-004 — Bounded FFmpeg-based video preprocessing

Status: Accepted for local development / Milestone 3

## Context

The Java processing boundary is operational but Python previously only checked file readability.
The next stages need genuine video metadata and timestamped media artifacts without inventing combat events.

## Decision

- Keep Spring Boot's Processing Job orchestration unchanged.
- Use Python + locally installed `ffprobe` for media stream metadata.
- Extract at most six representative JPEG frames and a 20-second, 16 kHz mono WAV preview through `ffmpeg`.
- Write a versioned JSON manifest per processing job under `backend/storage/analysis/<job-id>/` (ignored by Git).
- Store only `artifact_manifest_path` in `processing_jobs` via Flyway V3; user-facing job response remains a status + summary.
- Preserve `AiProcessingResponse` fields and add a required manifest path; extend HTTP read timeout for preprocessing.
- Validate input paths against developer-controlled `VIDEO_STORAGE_ROOT`, reject invalid media, bound extraction work.
- Do NOT infer fighter identities, fight events, or commentary in this milestone.

## Trade-offs / limitations

- Current architecture uses a shared local filesystem between Java and Python; deployment to separate machines will require object-storage references, storage credentials and controlled artifact access.
- The audio is a preview, not full ASR; frames are sparse, not fight event detection.
- `COMPLETED` means this preprocessing job succeeded, **not** that combat commentary has been generated.
- Backend persists the manifest path, not the full manifest; downstream integration should read/validate it via a dedicated internal service.
- No distributed queue/lease mechanism yet; this is a development-scale vertical slice.
