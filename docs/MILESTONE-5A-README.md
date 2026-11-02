# Milestone 5A — Clip preparation for future AI action recognition

## Why this milestone exists

M4 candidates are based on pixel differences. An actual video-action model
needs a short sequence of frames around candidate peaks. This milestone
prepares those clips **without claiming to classify combat events**.

## Boundaries and ownership

- Spring Boot: no changes. Owns upload, job state and `/moments` response.
- FastAPI: after M4 motion detection, prepares 0–8 short video clips.
- FFmpeg: re-decodes clips at 12 FPS, scales to 640 px max width, no audio.
- Local disk: saves `action_clips/candidate_NN.mp4` and `action_clips.json`.
- Manifest: `schemaVersion=3`, `stage=ACTION_CLIPS_PREPARED_ONLY`, `combatEventsDetected=false`,
  `actionPreparation.modelInferenceStatus=NOT_CONFIGURED`.
- Existing `/moments` continues to return motion-candidate JSON, not classifications.

## Acceptance criteria

1. Existing API contract with Java remains unchanged.
2. A normal video still reaches `COMPLETED`, with `motion_candidates.json` available.
3. Clip metadata is written even for zero candidates.
4. Clips do not exceed five seconds and contain the expected video stream.
5. Invalid timestamps cannot create arbitrary output paths.
6. No fake confidence/technique labels are created.
7. Existing and new Python tests pass.

## Local test

`cd ai-service; .\.venv\Scripts\python.exe -m pytest -q`

The Java backend is unchanged; re-running its tests is optional for this
small Python-only change. After Python tests pass, restart FastAPI.

## Verify on real putra.mp4

Upload a NEW video ID and POST `/processing` once. GET processing until
COMPLETED. Preserve the returned jobId; open:

`backend/storage/analysis/<jobId>/manifest.json`

Look at `actionPreparation.resultPath` and open the JSON it points to. Look
under `backend/storage/analysis/<jobId>/action_clips/` for MP4 clips.

## Next: Milestone 5B

Select a real video-recognition model with a compatible label ontology,
document the model version/license/inputs/costs, provide a CPU/GPU execution
plan, and evaluate against human-labeled combat footage. Only then add
`combat_events.json` and classification-specific API routes. No generic
model's output should be presented as verified strike contact without
appropriate evaluation.

## One-command end-to-end smoke test

With backend + Python services running and PostgreSQL started, from the
repository root:

```powershell
& .\scripts\smoke-milestone5a.ps1 -VideoFile 'C:\Users\massr\Downloads\putra.mp4' -ProjectPath 'D:\Milestone\combat-commentary-assistant'
```

The script performs one upload, one POST to create a job, uses GET for polling,
and checks the manifest, clip count, and output paths. It never commits/pushes.
