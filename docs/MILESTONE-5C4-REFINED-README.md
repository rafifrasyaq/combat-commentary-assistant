# M5C.4 — Experimental tracking refinement

## Purpose

Compare a gap-aware, clip-local tracker against the M5C.3 baseline **on the same YOLO Pose samples**, without changing earlier code or processing any video again.

## Context and observed limitations

The user's M5C.3 pilot had 7 clips, 84 sampled frames, and 245 person detections (sometimes 3–4 people in one frame). M5C.3 recorded **94 tracks, 43 single-frame tracks, 151 adjacent associations, 24 ambiguous restarts and 201 measured hand-motion samples**. The separate M5C.4 audit found 23 scale-gate failures, 13 center-shift-gate failures, 24 ambiguity abstentions, 12 unpaired candidates and 22 frame-initialization tracks.

## Heuristic changes

1. Keeps a missing track eligible for **one missed sampled frame only**, subject to tighter center/scale gates; no cross-clip matching.
2. Uses shoulder location as minor supporting evidence alongside center displacement, bounding-box overlap, and scale change.
3. Performs an exact minimum-cost one-to-one assignment over the tiny set of recent tracks and detections (at most four people detected per frame).
4. Abstains when competing links have similar costs; an ambiguous detection starts a new unverified track rather than a fabricated fighter identity.
5. Marks reconnections `RECONNECTED_AFTER_GAP` and does **not** estimate hand-motion rate across an unobserved frame.

All matches remain speculative heuristic associations. A reduction in track count can hide identity swaps and **is not a tracking accuracy score**. Several detections may be referees, spectators, or duplicates.

## Expected comparison on the supplied seven-clip pilot

| Metric | Existing M5C.3 | Experimental M5C.4 |
|---|---:|---:|
| Clip-local tracks | 94 | 84 |
| One-frame tracks | 43 | 34 |
| Adjacent-frame links | 151 | 149 |
| One-missing-frame reconnections | Not supported | 12 |
| Ambiguous restarts | 24 | 17 |
| Measured hand-motion samples | 201 | 195 |

These numbers are **specific to the supplied seven clips and this code version** and may differ with other data. Use manual annotation of actual fighter identity per sampled frame to evaluate identity switches.

## Run

From the project root (PowerShell):

```powershell
& .\scripts\verify-milestone5c4-refined.ps1
& .\scripts\run-milestone5c4-refined.ps1 -JobId 'c22eb136-112b-4529-b3a2-b72834624993'
```

### Outputs (new files only)

- `backend/storage/analysis/<job_id>/pose_tracks_refined.json`
- `backend/storage/analysis/<job_id>/tracking_comparison.json`
- `backend/storage/analysis/<job_id>/tracking_comparison.md`

**Preserved:** `pose_observations.json`, `pose_tracks.json`, audit output `tracking_quality_report.json`, and all M5C.3 source code.

### Next responsible step

Review gap links and ambiguous cases against actual frames (not just a summary table), create a small verified identity annotation set, and report ID switches / identity precision before using tracking downstream for combat commentary. No jab/cross/hook, contact, actual hit, or fighter identity claims are made at this stage.
