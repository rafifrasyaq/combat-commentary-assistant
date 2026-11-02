# ADR-006: Bounded action-clip preparation before model integration

Status: Proposed (Milestone 5A)

## Context

M4 produces intervals of **motion activity**, not identified punches or fighters.
A combat action-recognition model will require temporally dense video clips,
not just six still images or 4 FPS motion scores. M4 timestamps may contain
camera cuts and other false positives.

## Decision

- Prepare at most eight short clips per job, each at most five seconds.
- Include temporal context around a candidate's peak; clamp to media duration.
- Decode video again at **12 FPS** and scale to max width 640; the candidate
  finder at 4 FPS is not the frame rate supplied to future models.
- Write `action_clips.json` and a job-local `action_clips/` folder.
- Set `actionPreparation.modelInferenceStatus = NOT_CONFIGURED` in manifest.
- Do **not** generate technique labels, confidence, strike counts, or
  `verifiedCombatEvent=true`: **no classifier has been installed**.
- Preserve `/api/videos/{videoId}/moments` and existing M4 artifacts.
- Treat clips as local, internal artifacts; there is no public clip endpoint.

## Consequences / tradeoffs

- Bounded costs and reproducible input for future video models.
- Extra FFmpeg time, disk space, and re-encoding; suitable for development,
  not a production video pipeline at scale.
- Shared local filesystem still couples Java and Python; object storage is
  a separate architecture decision for multi-host deployments.
- Choosing a trained model and obtaining labeled ground-truth data remain
  explicit future tasks. Generic action-recognition models might not know
  jab, cross, hook or whether contact occurred.
