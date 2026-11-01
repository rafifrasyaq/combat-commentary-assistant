# Milestone 4 — v2 local verification

## Scope

This update is **motion-candidate filtering**, not verified combat recognition.
The existing upload/processing REST APIs and Flyway database schema are unchanged.

## Verification checklist

1. Apply the `milestone4-refinement` package ONLY on `feature/combat-moment-detection`
   while HEAD is still `915815b` and the 6 files match the previous M4 package.
   The installer backs up all 6 pre-update files outside the Git repository.
2. Python: `cd ai-service; .\.venv\Scripts\python.exe -m pytest -q` → **21 passed**
   in the packaged Linux verification; test locally on Windows as well.
3. Java: `cd backend; .\mvnw.cmd clean test` with PostgreSQL running. The
   expected **13 tests / BUILD SUCCESS** from M4 are unaffected by v2 source
   changes; rerun to verify the integrated environment.
4. Restart both Python and Java processes so the running server picks up new code.
   Python still reports `combat-commentary-ai-motion-candidates` on `/health`.
5. Upload a **new** video ID (a previously completed video cannot be reprocessed).
6. POST processing **only once**, keep the returned `jobId`, GET processing until
   `COMPLETED`, then GET moments and inspect the `motionAnalysis` manifest field.
7. Verify `method` is `grayscale-motion-with-scene-suppression-v2`, check
   `excludedLikelySceneChanges` and `motionCandidates`. No specific count is
   guaranteed on other clips.
8. Compare timestamps against the real footage. Do **not** count a candidate as a
   confirmed strike or check videos/results into Git. Commit only after review.

## PowerShell, COMMAND terminal

```powershell
cd D:\Milestone\combat-commentary-assistant
$videoFile = 'C:\Users\massr\Downloads\putra.mp4'
Test-Path $videoFile
$upload = curl.exe -fsS -F "file=@$videoFile;type=video/mp4" http://localhost:8080/api/videos | ConvertFrom-Json
$videoId = $upload.videoId
$jobResp = curl.exe -fsS -X POST "http://localhost:8080/api/videos/$videoId/processing" | ConvertFrom-Json
$jobId = $jobResp.jobId
$jobResp
# Do not POST again. Check status using GET instead.
curl.exe -sS "http://localhost:8080/api/videos/$videoId/processing"
curl.exe -sS "http://localhost:8080/api/videos/$videoId/moments"
$manifestPath = ".\backend\storage\analysis\$jobId\manifest.json"
Get-Content $manifestPath -Raw | ConvertFrom-Json | Select-Object stage, combatEventsDetected, motionAnalysis
```

## Expected interpretation

`MOTION_CANDIDATES_ONLY` is intentional. `combatEventsDetected` stays false.
`motionScore` is normalized frame difference (NOT the likelihood of a punch).
`excludedLikelySceneChanges` describes likely editing transitions, not perfectly
labeled scene cuts. A zero-candidate result is valid for still footage.
