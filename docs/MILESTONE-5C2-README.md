# Milestone 5C.1–5C.2 — Human review and YOLO pose pilot

## 5C.1 — Human review (completed)

The user manually viewed the seven action clips from M5B full E2E job
`c22eb136-112b-4529-b3a2-b72834624993` and reported **punch present** in
all seven clips. The seven VideoMAE top category labels are also
`punching person (boxing)`; see `docs/evaluation/m5c1-manual-review.csv`.
Scores in CSV were rounded to one decimal percent in the original terminal.

**Do not call this 100% accuracy, precision, recall, F1, hit detection, or punch
classification.** The dataset has zero human-reviewed negatives, no timestamp
annotations of individual punches, and no checked motion-detector misses.
What has been observed is 7/7 agreement on these hand-reviewed *positive*
clips with a *general activity* classifier. For real metrics, collect at least
clips without punches, hard negatives (feints, camera cuts, guard changes),
and timestamped punches from full video, not just algorithm-selected clips.

## 5C.2 — YOLO11n Pose baseline (opt-in, CLI only)

Adds `app/action/pose.py` which uses `yolo11n-pose.pt` pretrained on COCO17:
body keypoints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees,
ankles). It analyzes at most 8 existing M5A clips and at most 12 evenly
sampled 640-pixel frames per clip, with at most 120 decoded frames per clip.
It downloads YOLO11n-Pose weights the first time, reuses one model for all
clips, and runs on CUDA GPU 0.

**No changes to FastAPI or the M5B processing pipeline.** 5C.2 is a pilot
before automatically invoking pose estimation on new uploads.

### Installation (PowerShell, after copying the update files)

```powershell
cd D:\Milestone\combat-commentary-assistant
.\ai-service\.venv\Scripts\python.exe -m pip install -r .\ai-service\requirements-pose.txt
& .\scripts\verify-milestone5c2.ps1
& .\scripts\run-milestone5c2.ps1 -JobId 'c22eb136-112b-4529-b3a2-b72834624993'
```

### Results (not tracked in Git)

- `backend/storage/analysis/<jobId>/pose_observations.json` — per-frame
  person detections, 17 normalized keypoints and reported keypoint confidence.
- `backend/storage/analysis/<jobId>/pose_previews/candidate_01.jpg` … — one
  YOLO-annotated middle sampled frame per clip.
- `combatEventsDetected=false`; no global fighter IDs, hit verification,
  punch technique classifier or punch count.
- VideoMAE M5B `action_predictions.json` is **not changed**.

The M5C.2 model can detect referees or spectators as "person". Box indexes
are *per frame* only, and may switch when people overlap. No athlete tracking
or affiliation should be inferred from this pilot.

### Planned evaluation gates

1. Inspect the 7 JPEG overlays: are wrists/elbows/shoulders reliably aligned?
2. Compare visibility during guard, feints, attacks and occluded moments.
3. Collect negative clips and multi-person clutter; quantify detection quality.
4. Only then consider stable person tracking, hand trajectory features and
   boxing-specific action classifier. Always keep `UNKNOWN` when evidence is
   insufficient.

### Licensing

Ultralytics YOLO code and weights require evaluation under their AGPL-3.0
(or appropriate enterprise) license. The VideoMAE checkpoint previously used
has noncommercial restrictions. This is an experimental research/prototype
workflow; check licensing before distributing, operating as closed-source
service, or commercially deploying.
