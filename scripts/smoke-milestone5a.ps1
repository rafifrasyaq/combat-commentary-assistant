param(
    [Parameter(Mandatory=$true)][string]$VideoFile,
    [string]$ProjectPath = (Split-Path -Parent $PSScriptRoot)
)
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath $ProjectPath).Path
$video = (Resolve-Path -LiteralPath $VideoFile).Path
$backend = 'http://127.0.0.1:8080'
$ai = 'http://127.0.0.1:8000'

function Invoke-CurlJson([string[]]$CurlArgs) {
    $json = & curl.exe @CurlArgs
    if ($LASTEXITCODE -ne 0) { throw "curl.exe failed. Arguments: $($CurlArgs -join ' ')" }
    return $json | ConvertFrom-Json
}

Write-Host 'Checking server health...'
$backendHealth = Invoke-CurlJson @('-fsS', "$backend/api/health")
$aiHealth = Invoke-CurlJson @('-fsS', "$ai/health")
if ($backendHealth.status -ne 'UP' -or $aiHealth.status -ne 'UP') {
    throw 'One or more services did not return UP.'
}
Write-Host "Backend: $($backendHealth.status); AI: $($aiHealth.status)"

Write-Host 'Uploading a new video...'
$upload = Invoke-CurlJson @('-fsS', '-F', "file=@$video;type=video/mp4", "$backend/api/videos")
if (-not $upload.videoId) { throw 'Upload returned no videoId' }
$videoId = $upload.videoId
Write-Host "videoId: $videoId"

Write-Host 'Starting processing exactly once...'
$job = Invoke-CurlJson @('-fsS', '-X', 'POST', "$backend/api/videos/$videoId/processing")
if (-not $job.jobId) { throw 'Processing request returned no jobId' }
$jobId = $job.jobId
Write-Host "jobId: $jobId"

# GET only from now on; never issue another POST for the same video.
$latest = $null
for ($attempt=1; $attempt -le 30; $attempt++) {
    $latest = Invoke-CurlJson @('-fsS', "$backend/api/videos/$videoId/processing")
    Write-Host "Attempt ${attempt}: $($latest.status)"
    if ($latest.status -eq 'COMPLETED' -or $latest.status -eq 'FAILED') { break }
    Start-Sleep -Seconds 2
}
if ($latest.status -ne 'COMPLETED') {
    throw "Job did not complete successfully. Status: $($latest.status); error: $($latest.errorMessage)"
}

$manifestPath = Join-Path $project ("backend/storage/analysis/$jobId/manifest.json")
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Manifest not found at $manifestPath"
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.stage -ne 'ACTION_CLIPS_PREPARED_ONLY') {
    throw "Unexpected manifest stage: $($manifest.stage). Is the Python service restarted?"
}
if ($manifest.combatEventsDetected -ne $false -or
    $manifest.actionPreparation.modelInferenceStatus -ne 'NOT_CONFIGURED') {
    throw 'Model inference status must remain NOT_CONFIGURED and combatEventsDetected false.'
}
$clipFile = $manifest.actionPreparation.resultPath
if (-not (Test-Path -LiteralPath $clipFile -PathType Leaf)) { throw "Missing clip index: $clipFile" }
$clipData = Get-Content -LiteralPath $clipFile -Raw | ConvertFrom-Json
if ($clipData.clipCount -ne $manifest.actionPreparation.clipCount) { throw 'Clip count mismatch' }
foreach ($clip in $clipData.clips) {
    if (-not (Test-Path -LiteralPath $clip.path -PathType Leaf)) { throw "Missing video clip: $($clip.path)" }
}

Write-Host 'M5A end-to-end SUCCESS.'
Write-Host "videoId: $videoId"
Write-Host "jobId: $jobId"
Write-Host "clips prepared: $($clipData.clipCount)"
Write-Host 'model predictions: NOT RUN (expected for Milestone 5A)'
if ($clipData.clipCount -gt 0) { Write-Host "first clip: $($clipData.clips[0].path)" }
