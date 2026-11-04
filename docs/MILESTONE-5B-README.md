# Milestone 5B — Pretrained VideoMAE inference baseline

## Requirements and scope

Milestone 5A generated up to 8 bounded action clips (not verified combat events).
Milestone 5B adds an **optional** inference stage using the pretrained checkpoint
`MCG-NJU/videomae-base-finetuned-kinetics` (Kinetics-400). This model predicts
**general activity categories**, not jab/cross/hook, contact, knockdowns,
which fighter acted, or match results.

The user-confirmed baseline on their RTX 4060 before integration:
`candidate_01.mp4` → `punching person (boxing)` softmax 55.09% (plus other classes).
This is a single model observation, **not** a validated accuracy measurement.

### Components and ownership

| Component | Responsibility |
| --- | --- |
| M5A clip extractor | Generate short MP4s from motion candidates |
| M5B recognition.py | Bounded PyAV decode, 16-frame sampling, one cached model per process, GPU inference |
| FastAPI `/analyze` | Optional orchestrator (`ACTION_RECOGNITION_ENABLED=true`) |
| `action_predictions.json` | Unverified Kinetics activity predictions, score type, timestamps |
| `manifest.json` | Add `actionRecognition`; preserve `combatEventsDetected=false` |
| Spring Boot | Unchanged upload/job state and `/moments` API |

The model is loaded lazily and only once per Python process, and inference
runs under a lock to limit VRAM concurrency. The model stage is **disabled by
default**, keeping all M5A flows working even without ML libraries.

### Dependencies

In PowerShell from `ai-service`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ai.txt
```

PyTorch and torchvision need the compatible official CUDA wheel installation
(the user already confirmed torch 2.14.1+cu126 and RTX 4060 CUDA availability):

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
```

Do not commit `.venv/`, model weights or generated videos. A Hugging Face
checkpoint is cached outside the repository. Check the Kinetics checkpoint
license (noncommercial) before using it in a commercial product.

### Try all seven EXISTING clips without re-uploading

Run this from the **repository root**; no Java, Docker, or FastAPI needed:

```powershell
& .\scripts\run-milestone5b.ps1 -JobId '746233c4-2e86-4c47-8d64-14d67f45adcc'
```

This reuses previously generated clips on disk, writes `action_predictions.json`
and updates that job's `manifest.json` (schemaVersion 4) to reflect model inference.
All source files and Git history are untouched. This command is not a fresh
end-to-end upload or backend API check.

### Verify coding changes (fake classifier, no GPU/network)

```powershell
& .\scripts\verify-milestone5b.ps1
```

### Enable automatic predictions for NEW processing jobs

In the FastAPI process's PowerShell terminal (before starting Uvicorn):

```powershell
cd D:\Milestone\combat-commentary-assistant\ai-service
$env:VIDEO_STORAGE_ROOT = (Resolve-Path ..\backend\storage\videos).Path
$env:ACTION_RECOGNITION_ENABLED = 'true'
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then **new jobs** will run VideoMAE automatically after M5A and publish:

```
backend/storage/analysis/<jobId>/
    motion_candidates.json
    action_clips.json
    action_clips/candidate_XX.mp4
    action_predictions.json
    manifest.json
```

When the flag is absent or false, existing M5A behavior is preserved. Inference
errors when enabled are explicit failures rather than false completed analysis.
The Java HTTP client has a 120-second read timeout; pretrained weights should
be cached before enabling new-job inference. Follow-up milestone should make
model initialization and long-running inference independent of synchronous
FastAPI request duration.

### Output contract

The file `action_predictions.json` contains `analysisType=PRETRAINED_KINETICS400_ACTIVITY_ONLY`,
`predictions[]` with `candidateIndex`, clip timestamps, `activityLabel`,
`status`, `topPredictions[]` containing `softmaxScore`, and device metadata.
The threshold of 0.4 for displaying `UNKNOWN` is a **heuristic**;
softmax scores are uncalibrated and not accuracy or probability of strike
contact. Even a high `punching person (boxing)` result is not a verified punch.

### Pending work

- Compare outputs with human-labeled clips, quantify error/precision/recall.
- Evaluate boxing-specific pretrained or fine-tuned models and dataset licensing.
- Robust fighter tracking, strike contact classification, alternative windows
  for motion detector false negatives, GPU worker queue and per-job cancellation.
- Separate backend endpoint for unverified predictions (do not repurpose `/moments`).

## Milestone acceptance

- Old M5A works unchanged when M5B flag disabled.
- All fast Python tests pass with fake model; zero clip and invalid path covered.
- Existing-job CLI uses one model load and classifies all available clips.
- Actual model prediction stays unverified and never sets `combatEventsDetected=true`.
- No secrets, caches, videos, or heavy model files in Git.
