param(
    [Parameter(Mandatory=$true)]
    [string]$JobId
)
$ErrorActionPreference = 'Stop'
if ($JobId -notmatch '^[a-fA-F0-9]{8}(-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}$') {
    throw 'JobId must be a valid UUID.'
}
$root = Split-Path -Parent $PSScriptRoot
$ai = Join-Path $root 'ai-service'
$python = Join-Path $ai '.venv\Scripts\python.exe'
$jobDir = Join-Path (Join-Path $root 'backend\storage\analysis') $JobId
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Python venv missing: $python" }
if (-not (Test-Path -LiteralPath (Join-Path $jobDir 'pose_observations.json') -PathType Leaf)) {
    throw "Prior M5C.2 observations missing: $jobDir\pose_observations.json"
}
Write-Host 'M5C.4: compare refined associations with existing M5C.3 (no GPU, no re-upload)...' -ForegroundColor Cyan
Push-Location $ai
try {
    & $python -m app.action.tracking_refined --job-dir $jobDir
    if ($LASTEXITCODE -ne 0) { throw "M5C.4 processing failed (exit $LASTEXITCODE)" }
} finally { Pop-Location }
Write-Host "Readable comparison: $(Join-Path $jobDir 'tracking_comparison.md')" -ForegroundColor Green
