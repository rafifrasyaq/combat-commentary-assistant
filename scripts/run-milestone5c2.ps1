param(
    [Parameter(Mandatory = $true)][Guid]$JobId,
    [string]$ProjectPath = (Split-Path -Parent $PSScriptRoot)
)
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath $ProjectPath -ErrorAction Stop).Path
$python = Join-Path $project 'ai-service\.venv\Scripts\python.exe'
$jobDirectory = Join-Path $project ("backend\storage\analysis\" + $JobId.ToString())
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'Python venv missing.' }
if (-not (Test-Path -LiteralPath (Join-Path $jobDirectory 'action_clips.json') -PathType Leaf)) {
    throw "M5A action clips are missing for this job: $jobDirectory"
}
Write-Host 'Running YOLO11n Pose on existing M5A clips (no re-upload/backend needed)...'
Push-Location (Join-Path $project 'ai-service')
try {
    & $python -u -m app.action.pose --job-dir $jobDirectory
    if ($LASTEXITCODE -ne 0) { throw 'Pose pilot failed. Read the Python exception above.' }
} finally { Pop-Location }
$path = Join-Path $jobDirectory 'pose_observations.json'
$data = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
if ($data.stage -ne 'UNVERIFIED_POSE_OBSERVATIONS' -or
    $data.combatEventsDetected -ne $false -or
    $data.jobId -ne $JobId.ToString() -or
    @($data.annotations).Count -ne $data.clipCount) {
    throw 'Pose output has inconsistent metadata.'
}
Write-Host "Pose clips: $($data.clipCount); analyzed frames: $($data.analyzedFrameCount); previews: $($data.previewCount)"
foreach ($item in $data.annotations) {
    Write-Host ("Clip {0}: {1}/{2} frames with a person; 2+ people in {3} frames" -f
        $item.candidateIndex, $item.framesWithPerson, $item.sampledFrameCount, $item.framesWithTwoOrMorePeople)
}
Write-Host "Pose JSON: $path"
Write-Host "Open previews: $(Join-Path $jobDirectory 'pose_previews')"
Write-Host 'NOTICE: these results do not identify actual punches or stable fighters.'
