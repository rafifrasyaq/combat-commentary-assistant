# M5C.3 — Conservative clip-local pose association (pilot)

## Purpose and input

Consume M5C.2 `pose_observations.json` from the existing analysis job directory. **No new upload, backend, FastAPI, CUDA, YOLO download, or video decode is needed.** Outputs `pose_tracks.json` in the same job directory. Neither M5A/M5B nor M5C.2 artifacts are changed. No additional Python dependencies.

## Matching algorithm

For each clip separately, associate people detected in **adjacent sampled frames only** by normalized bounding-box center displacement and intersection-over-union (IoU). Reject implausible center movement (`>0.28` normalized frame units), large scale changes (`>3x` area), and frame gaps (`>1 second`). When multiple plausible assignments have similar costs, **abstain** and restart tracks. A disappeared person is not re-identified on reappearance. Track IDs (`c01_t01`, etc.) reset by clip. This is heuristic, not a robust broadcast re-identification system or stable fighter identity.

For confidently reported wrists and shoulders on consecutively associated people, compute the change in *wrist position relative to its own shoulder* and express it per observed person bounding-box width per elapsed second. This is a **2D image-based relative displacement rate**, not physical hand velocity, punching rate, contact, or a jab/cross/hook classifier. Inadequate keypoints or timing produce `measurable: false` rather than guessed numbers.

## Expected fields

`pose_tracks.json` includes:

- `stage: UNVERIFIED_CLIP_LOCAL_POSE_ASSOCIATIONS`
- `clipCount`, `totalClipLocalTracks`, `matchedAdjacentDetections`, `measuredHandMotionSamples`, `ambiguousRestarts`
- `annotations[].frames[].associations[]`: original `detectionIndex`, clip-local `trackId`, matching status, cost if matched, left/right relative motion observations
- `stableFighterIdentities: false`, `combatEventsDetected: false`, `punchesDetected: false`

The output is intentionally conservative. A high measured movement rate **is not proof of a punch**.

## Validation and limitations

The script checks the M5C.2 artifact schema, clip/frame/person limits and normalized keypoint locations before publishing atomic JSON. Automated tests use synthetic inputs and cover matching, overlapping-person ambiguity, occlusion, missing wrist keypoints, invalid inputs, no-person frames, and preservation of existing artifacts. They do **not** validate tracker identity accuracy on real videos.

A quick human review should check whether `cXX_tYY` stays on the same person in adjacent frames, especially when fighters cross, overlap, or the referee appears. For production-quality fighter tracking, consider video-level detector tracking, ID-switch measurement, referee separation, and more densely sampled frames. Public releases using Ultralytics must account for its licensing conditions.

## Run

From PowerShell in repo root:

```powershell
& .\scripts\verify-milestone5c3.ps1
& .\scripts\run-milestone5c3.ps1 -JobId 'c22eb136-112b-4529-b3a2-b72834624993'
```

No Pull Request yet; keep commits on `feature/combat-action-recognition` until the feature series is ready for review. `try_videomae.py` is a local experiment and is not part of this change.
