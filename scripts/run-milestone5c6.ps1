param(
    [Parameter(Mandatory=$true)][string]$JobId
)
$ErrorActionPreference = 'Stop'
if ($JobId -notmatch '^[a-fA-F0-9]{8}(-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}$') { throw 'JobId must be a UUID.' }
$root = Split-Path -Parent $PSScriptRoot
$ai = Join-Path $root 'ai-service'
$python = Join-Path $ai '.venv\Scripts\python.exe'
$jobDir = Join-Path (Join-Path $root 'backend\storage\analysis') $JobId
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {throw "Missing Python environment: $python"}
foreach ($name in @('pose_observations.json', 'pose_tracks_refined.json', 'action_clips.json')) {
    if (-not (Test-Path -LiteralPath (Join-Path $jobDir $name) -PathType Leaf)) {throw "Missing previous milestone artifact: $name"}
}
$csv = Join-Path (Join-Path $jobDir 'active_pair_review') 'active_pair_review.csv'
if (Test-Path -LiteralPath $csv) {throw 'Existing active_pair_review.csv: refusing to overwrite human annotations. Do not rerun.'}
Write-Host 'M5C.6: proposing active pairs from existing pose and tracking (CPU only, no YOLO)...' -ForegroundColor Cyan
Push-Location $ai
try {
    & $python -m app.action.active_pair_selection --job-dir $jobDir
    if ($LASTEXITCODE -ne 0) {throw "Active pair selection failed: exit $LASTEXITCODE"}
} finally {Pop-Location}
Write-Host "Open candidate images: $(Join-Path $jobDir 'active_pair_review')" -ForegroundColor Green
