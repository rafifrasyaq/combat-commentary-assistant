param(
    [Parameter(Mandatory=$true)][string]$JobId
)
$ErrorActionPreference = 'Stop'
if ($JobId -notmatch '^[a-fA-F0-9]{8}(-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}$') { throw 'JobId must be a UUID.' }
$root = Split-Path -Parent $PSScriptRoot
$ai = Join-Path $root 'ai-service'
$python = Join-Path $ai '.venv\Scripts\python.exe'
$jobDir = Join-Path (Join-Path $root 'backend\storage\analysis') $JobId
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Python venv missing: $python" }
foreach ($name in @('action_clips.json','pose_observations.json','pose_tracks_refined.json')) {
    if (-not (Test-Path -LiteralPath (Join-Path $jobDir $name) -PathType Leaf)) {throw "Missing input: $name"}
}
Write-Host 'M5C.5: rendering existing clip frames with unverified IDs (CPU only, no model inference)...' -ForegroundColor Cyan
Push-Location $ai
try {
    & $python -m app.action.identity_review --job-dir $jobDir
    if ($LASTEXITCODE -ne 0) { throw "Identity review rendering failed (exit $LASTEXITCODE)" }
} finally { Pop-Location }
$dest = Join-Path $jobDir 'identity_review'
if (-not (Test-Path -LiteralPath (Join-Path $dest 'manual_clip_review.csv'))) { throw 'Manual review output not created' }
Write-Host "Open review images: $dest" -ForegroundColor Green
