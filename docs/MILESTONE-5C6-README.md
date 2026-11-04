# M5C.6 — Active Fighter Pair Selection (experimental)

**Purpose:** generate frame-by-frame *tentative* two-person interaction pairs from the **saved** pose observations and refined tracking. It is a pure analysis step, not a trained fighter classifier.

## What it does

For each of the sampled frames, the script filters small/low-confidence person detections, ranks possible pairs using bounding-box proximity, relative sizes, vertical alignment, wrist proximity to the other person's upper body, and a *very small* unverified continuity term. It proposes a pair only if its score passes a minimum and is clearly higher than the runner-up. Otherwise the result is `AMBIGUOUS_COMPETING_PAIRS`, `UNKNOWN_WEAK_PAIR_EVIDENCE`, or `UNKNOWN_INSUFFICIENT_PEOPLE`.

**Important:** A close wrist does not mean a punch landed or even that someone tried to punch. Two nearby people are not necessarily the fighters. This algorithm does not label Fighter A/B or recognize techniques. Its scores **are not probabilities** and no ground-truth accuracy has been established. IDs are still clip-local, may switch, and must not be interpreted as fighter identities. `UNASSIGNED` means no proposal, not a verified non-fighter.

## Installation and execution

1. Download `combat-commentary-m5c6-active-pair-update.zip` to `Downloads`.
2. In repo root use:

```powershell
$zip = "$env:USERPROFILE\Downloads\combat-commentary-m5c6-active-pair-update.zip"
Expand-Archive -LiteralPath $zip -DestinationPath "D:\Milestone\cca-m5c6" -Force
& "D:\Milestone\cca-m5c6\INSTALL-M5C6.ps1" -ProjectPath "D:\Milestone\combat-commentary-assistant"
& .\scripts\verify-milestone5c6.ps1
& .\scripts\run-milestone5c6.ps1 -JobId 'c22eb136-112b-4529-b3a2-b72834624993'
```

3. Look at `backend/storage/analysis/<jobId>/active_pair_review/`.

## Outputs

- `active_pair_proposals.json`: source-linked frame-level candidate pairs and ranked alternatives, **unverified**.
- `active_pair_report.md`: per-clip counts, UNKNOWN and ambiguous breakdown.
- `candidate_XX_pair_proposals.jpg`: 7 contact sheets. **Yellow** denotes an uncertain pair proposal, **gray** means unassigned, not a verified bystander.
- `active_pair_review.csv`: one row per frame, user-editable `reviewStatus`, `humanPairVerdict` and `notes`.

**Do not run twice without moving the existing CSV**: the script refuses to overwrite human review edits. The source JSON, original M5C.3 and refined M5C.4 output remain unchanged. No Docker, Spring, FastAPI, YOLO, VideoMAE, or Git operations required.

## Minimum human validation

Examine each contact sheet left-to-right and top-to-bottom. Mark at least frames where the algorithm selected someone wearing red near the ring or visibly off-ring. Review statuses: `PASS`, `FAIL`, or `UNCLEAR`. Keep uncertainty rather than inventing evidence. Compare annotations with the real video if the image alone cannot distinguish the fighters.

To progress to technique classification, gather video-level ground-truth fighter IDs, exclude non-fighters, and ideally increase temporal sampling. Static poses from only 12 frames/clip cannot verify strikes or impact.
