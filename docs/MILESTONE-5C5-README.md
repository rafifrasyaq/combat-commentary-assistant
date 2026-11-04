# Milestone 5C.5 — Human identity validation (pilot)

**Scope:** Inspect whether M5C.4's experimental clip-local IDs are visually consistent. Not strike classification and not verified fighters.

## Inputs

- `backend/storage/analysis/<jobId>/action_clips/` + `action_clips.json` (M5A, unchanged)
- `pose_observations.json` (M5C.2, unchanged)
- `pose_tracks_refined.json` (M5C.4, unchanged)

Requires existing `opencv-python` (installed with requirements-pose.txt) and PyAV (previously installed with VideoMAE). **No GPU, model weights, backend, or additional dependencies.** Decoding uses the **same** sampling code as M5C.2 to align frames exactly. If clips or sampling changed, the tool refuses to silently mislabel them.

## Output

`identity_review/candidate_01_tracks.jpg` ... `candidate_07_tracks.jpg`, `manual_clip_review.csv`, `priority_cases.csv`, `review_manifest.json` and `README.md`.

Images contain sampled frames with clip-local IDs, normalized YOLO bounding boxes, arm keypoints, a per-ID deterministic color, and N/M/G/? for **new/matched/gap reconnect/ambiguous new**. Different colors do not prove different people, and same color does not prove same person. IDs reset for each clip.

`manual_clip_review.csv` asks for one judgment per clip. Set `reviewStatus` to `PASS`, `FAIL`, or `UNCLEAR`, and mark suspected identity switches, extra people, camera cuts and occlusion. `priority_cases.csv` includes each gap reconnection and ambiguous new ID; `visualVerdict` can be `CORRECT`, `INCORRECT` or `UNCLEAR`. Empty cells are deliberately unlabeled. **Do not treat the CSV as automated ground truth.**

## Run (VS Code PowerShell)

```powershell
& .\scripts\verify-milestone5c5-review.ps1
& .\scripts\run-milestone5c5-review.ps1 -JobId 'c22eb136-112b-4529-b3a2-b72834624993'
explorer '.\backend\storage\analysis\c22eb136-112b-4529-b3a2-b72834624993\identity_review'
```

Review all seven sheets and note any ID swaps, paying attention to clip 6's detection drops. No Git commit or PR until human review.

## Limitations and future improvements

12 sampled frames per clip are insufficient to prove stable full-fight identities, jab/cross/hook, contact, or impacts. Human review can identify obvious ID switches only; robust tracking will eventually require a stable detection/re-ID benchmark, higher sample rate, handling broadcast cuts/occlusion and comparing to manually labeled fighter trajectories on multiple videos. Respect Ultralytics AGPL-3.0/enterprise license and original model/data licenses.
