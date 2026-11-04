param(
    [Parameter(Mandatory=$true)][Guid]$JobId,
    [string]$ProjectPath = (Split-Path -Parent $PSScriptRoot)
)
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath $ProjectPath).Path
$python = Join-Path $project 'ai-service\.venv\Scripts\python.exe'
$jobDirectory = Join-Path $project ("backend\storage\analysis\" + $JobId.ToString())
if (-not (Test-Path -LiteralPath (Join-Path $jobDirectory 'action_clips.json') -PathType Leaf)) {
    throw "M5A clips for this job do not exist: $jobDirectory"
}
Write-Host 'Running VideoMAE on existing M5A clips (no re-upload, no backend required)...'
Push-Location (Join-Path $project 'ai-service')
try {
    & $python -u -m app.action.recognition --job-dir $jobDirectory
    if ($LASTEXITCODE -ne 0) { throw 'VideoMAE inference failed. Check the Python error above.' }
} finally { Pop-Location }
$path = Join-Path $jobDirectory 'action_predictions.json'
$data = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
if ($data.stage -ne 'UNVERIFIED_ACTIVITY_PREDICTIONS' -or
    $data.combatEventsDetected -ne $false -or
    @($data.predictions).Count -ne $data.clipCount) {
    throw 'Prediction JSON is inconsistent.'
}
Write-Host "Predictions: $($data.clipCount); Unknown: $($data.unknownCount)"
foreach ($entry in $data.predictions) {
    Write-Host ("Clip {0}: {1} ({2})" -f $entry.candidateIndex, $entry.activityLabel, $entry.status)
}
Write-Host "Saved: $path"
