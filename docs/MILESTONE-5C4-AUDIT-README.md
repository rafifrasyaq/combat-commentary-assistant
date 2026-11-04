# M5C.4 — Tracking Quality Audit (diagnostic pilot)

This stage reads `pose_observations.json` and `pose_tracks.json` from an existing job; it does **not** run YOLO, VideoMAE, or the M5C.3 tracker. It generates `tracking_quality_report.json` and `tracking_quality_report.md` alongside the existing outputs.

## What it measures

- Fresh track decisions, matched adjacent detections, and single-frame tracks.
- Breakdown of *rule-based plausible causes* of fresh tracks: first frame, previous frame with no people, unsupported sampling gap, center movement gate, bounding-box scale gate, ambiguity abstention, and unpaired association despite feasible candidates.
- Per-clip comparisons. Time between sampled frames and descriptive 2D wrist rate summary.

These counts **cannot** prove ID switches or improved tracking accuracy without manually labeled correspondence. Lower track fragmentation is not inherently better; it may conceal incorrect fighter swaps.

## Run

```powershell
& .\scripts\verify-milestone5c4-audit.ps1
& .\scripts\run-milestone5c4-audit.ps1 -JobId 'c22eb136-112b-4529-b3a2-b72834624993'
```

Open the report in VS Code:

```powershell
code .\backend\storage\analysis\c22eb136-112b-4529-b3a2-b72834624993\tracking_quality_report.md
```

Share the brief terminal summary or `tracking_quality_report.json` to proceed with controlled improvements. No need to upload original 77MB video or rerun inference.

## Safety

- No changes to M5A/M5B/M5C.2/M5C.3 outputs or source.
- Audit artifacts use a separate filename. Checks prevent symlink source/output use.
- No identity claims (Fighter A/B), punch types, contact or confirmed combat events.
