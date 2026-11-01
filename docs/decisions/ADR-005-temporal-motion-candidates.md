# ADR-005 — Bounded temporal motion-candidate detection

Status: **Accepted as experimental baseline**, with v2 scene-transition suppression.

## Context

Milestone 3 extracted six still frames, too sparse to capture fast strikes. Milestone 4
adds bounded temporal measurements without claiming trained combat-action recognition.
A review of `putra.mp4` showed that v1's highest grayscale-change peaks frequently
coincided with editing/camera transitions; a raw top-eight ranking is misleading.

## Decision

- Preserve Spring Boot POST `/api/videos/{videoId}/processing`, Python POST `/analyze`,
  and GET `/api/videos/{videoId}/moments`. No Java endpoint or migration changes for v2.
- Preserve media metadata, six still JPEG frames and optional WAV preview from M3.
- Sample up to the first 120 seconds at 4 FPS as 96×96 grayscale proxy frames;
  decode with FFmpeg, with bounded output and a 90-second subprocess timeout.
- For each neighboring-frame pair, calculate normalized mean absolute grayscale
  difference and a 16-bin grayscale-histogram distance.
- Flag **likely** cuts when the motion change is extremely high, or when an
  unusually abrupt **local** motion spike accompanies a histogram change.
  Exclude the suspected cut and one adjacent pair on each side.
- Require at least two above-baseline nonexcluded samples in each candidate burst;
  tolerate one below-threshold gap. Rank bursts by median motion strength, not
  just the largest isolated spike. Keep at most eight temporally separated bursts.
- Write `motion_candidates.json` atomically with method
  `grayscale-motion-with-scene-suppression-v2` and statistics on likely cuts,
  suppressed neighbors and discarded short bursts. Continue to return the same
  general `motionCandidates` response contract and `verifiedCombatEvent: false`.
- The manifest remains schemaVersion 2, stage `MOTION_CANDIDATES_ONLY`, with
  `combatEventsDetected: false`; its motionAnalysis method points to v2. No new
  PostgreSQL migration is necessary.
- Record a minimal development regression against the user-provided `putra.mp4`
  without checking in the video or generated artifacts.

## Observed v1 → v2 comparison on `putra.mp4`

- Video duration: about 55.5 seconds; 222 proxy frames and 221 frame pairs.
- v1: eight candidates including peaks at ~4.75, 10.25, 15.00, 24.75, 28.25,
  30.50, 34.25, 48.25 seconds.
- v2 local test: seven candidate peaks around 15.75, 17.25, 23.75, 25.50,
  28.25, 36.25, 48.25 seconds. Ten **likely** cut samples were flagged.
- These are **diagnostic observations on one clip**, not ground-truth labels or
  validated accuracy metrics. Some true actions may be removed; other candidates
  may still arise from camera movement or editing.

## Known limitations and next phase

- No trained action model, no boxer tracking, no punch-type classification, no
  knockdown/fighter-identity verification. `motionScore` is **not confidence**.
- Grayscale global differences and histograms cannot reliably distinguish
  camera translation from fighter movement. Cut filtering is heuristic.
- A 4-FPS proxy can entirely miss punches shorter than the sample interval.
  Longer videos are truncated at 120 seconds. Timestamps are approximate.
- Actual punch recognition will require clip-based temporal models, an annotated
  validation set across multiple matches and camera styles, and quantitative
  recall / precision evaluation with hard negatives (cuts, camera shake, overlays).
- The local implementation shares filesystem access between services. Production
  security, durable workers, object storage and auth remain separate work.
