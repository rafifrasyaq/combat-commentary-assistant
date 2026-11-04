param(
    [Parameter(Mandatory = $true)][Guid]$JobId,
    [string]$ProjectPath = (Split-Path -Parent $PSScriptRoot)
)
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath $ProjectPath -ErrorAction Stop).Path
$python = Join-Path $project 'ai-service\.venv\Scripts\python.exe'
$jobDirectory = Join-Path $project ('backend\storage\analysis\' + $JobId.ToString())
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'Python virtual environment missing.' }
$source = Join-Path $jobDirectory 'pose_observations.json'
if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "M5C.2 pose JSON missing: $source" }
Write-Host 'M5C.3: associating observed people and measuring wrist motion (no GPU, no re-upload)...'
Push-Location (Join-Path $project 'ai-service')
try {
    & $python -u -m app.action.tracking --job-dir $jobDirectory
    if ($LASTEXITCODE -ne 0) { throw 'Pose association failed. See Python error above.' }
} finally { Pop-Location }
$output = Join-Path $jobDirectory 'pose_tracks.json'
$data = Get-Content -LiteralPath $output -Raw | ConvertFrom-Json
if ($data.stage -ne 'UNVERIFIED_CLIP_LOCAL_POSE_ASSOCIATIONS' -or
    $data.jobId -ne $JobId.ToString() -or
    $data.stableFighterIdentities -ne $false -or
    $data.combatEventsDetected -ne $false -or
    $data.punchesDetected -ne $false -or
    @($data.annotations).Count -ne $data.clipCount) {
    throw 'Unexpected tracking report metadata.'
}
Write-Host "M5C.3 SUCCESS. Clips: $($data.clipCount); tentative tracks: $($data.totalClipLocalTracks)"
Write-Host "Adjacent associations: $($data.matchedAdjacentDetections); measurable hand motions: $($data.measuredHandMotionSamples)"
Write-Host "Ambiguity-induced restarts: $($data.ambiguousRestarts)"
Write-Host "Output: $output"
Write-Host 'These IDs are clip-local associations only, NOT confirmed fighter identities or punches.'
