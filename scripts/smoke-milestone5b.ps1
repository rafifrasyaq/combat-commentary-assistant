<#
  M5B automatic end-to-end smoke test.
  Prerequisite: Docker/Postgres, Spring Boot on :8080 and FastAPI :8000 with
  ACTION_RECOGNITION_ENABLED=true started before running this script.
  Performs exactly one new upload and exactly one new POST /processing.
  Then polls GET, reads on-disk artifacts, and checks they are unverified labels.
#>
param(
    [Parameter(Mandatory = $true)][string]$VideoFile,
    [string]$ProjectPath = (Split-Path -Parent $PSScriptRoot),
    [ValidateRange(1, 300)][int]$PollAttempts = 90,
    [ValidateRange(1, 30)][int]$PollSeconds = 2
)

$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath $ProjectPath -ErrorAction Stop).Path
$video = (Resolve-Path -LiteralPath $VideoFile -ErrorAction Stop).Path
if (-not (Test-Path -LiteralPath $video -PathType Leaf)) { throw "Video file not found: $video" }
if ([IO.Path]::GetExtension($video).ToLowerInvariant() -ne '.mp4') { throw 'This smoke test expects an MP4 file.' }

$backend = 'http://127.0.0.1:8080'
$ai = 'http://127.0.0.1:8000'

function Invoke-CurlJson([string[]]$CurlArgs) {
    $raw = & curl.exe @CurlArgs
    if ($LASTEXITCODE -ne 0) { throw "HTTP request failed: $($CurlArgs -join ' ')" }
    try { return ($raw | ConvertFrom-Json -ErrorAction Stop) }
    catch { throw 'HTTP response is not valid JSON.' }
}

Write-Host 'Checking backend and AI health...'
$backendHealth = Invoke-CurlJson @('-fsS', "$backend/api/health")
$aiHealth = Invoke-CurlJson @('-fsS', "$ai/health")
if ($backendHealth.status -ne 'UP' -or $aiHealth.status -ne 'UP') {
    throw "Expected UP for both services; backend=$($backendHealth.status) ai=$($aiHealth.status)"
}
Write-Host 'Both services report UP. M5B feature flag is checked via the output artifacts.'

Write-Host 'Uploading a NEW video (POST once)...'
$upload = Invoke-CurlJson @('-fsS', '-F', "file=@$video;type=video/mp4", "$backend/api/videos")
if (-not $upload.videoId) { throw 'Video upload did not return videoId.' }
$videoId = [string]$upload.videoId
Write-Host "videoId: $videoId"

Write-Host 'Requesting processing ONCE...'
$submitted = Invoke-CurlJson @('-fsS', '-X', 'POST', "$backend/api/videos/$videoId/processing")
if (-not $submitted.jobId) { throw 'Processing request did not return jobId.' }
$jobId = [string]$submitted.jobId
Write-Host "jobId: $jobId"
Write-Host 'Now only GET requests: polling the existing job...'

$latest = $null
for ($attempt = 1; $attempt -le $PollAttempts; $attempt++) {
    $latest = Invoke-CurlJson @('-fsS', "$backend/api/videos/$videoId/processing")
    if ($latest.jobId -ne $jobId) { throw 'Backend returned a different latest jobId.' }
    Write-Host "Poll ${attempt}: $($latest.status)"
    if ($latest.status -in @('COMPLETED', 'FAILED')) { break }
    Start-Sleep -Seconds $PollSeconds
}
if ($latest.status -ne 'COMPLETED') {
    throw "Processing did not complete. Last status=$($latest.status), jobId=$jobId. Check FastAPI/backend logs. Error=$($latest.errorMessage)"
}
if ($latest.resultSummary -notlike '*VideoMAE classified*') {
    throw 'Backend says COMPLETED, but summary does not confirm automatic VideoMAE inference. Ensure FastAPI was started with ACTION_RECOGNITION_ENABLED=true.'
}

$jobDir = Join-Path $project "backend\storage\analysis\$jobId"
$manifestPath = Join-Path $jobDir 'manifest.json'
$predictionsPath = Join-Path $jobDir 'action_predictions.json'
$clipsPath = Join-Path $jobDir 'action_clips.json'
foreach ($path in @($manifestPath, $predictionsPath, $clipsPath)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Expected artifact missing: $path" }
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$predictions = Get-Content -LiteralPath $predictionsPath -Raw | ConvertFrom-Json
$clips = Get-Content -LiteralPath $clipsPath -Raw | ConvertFrom-Json

if ($manifest.stage -ne 'UNVERIFIED_ACTIVITY_PREDICTIONS' -or
    $manifest.actionPreparation.modelInferenceStatus -ne 'COMPLETED' -or
    $manifest.actionRecognition.modelInferenceStatus -ne 'COMPLETED' -or
    $manifest.combatEventsDetected -ne $false -or
    $manifest.actionRecognition.combatEventsVerified -ne $false) {
    throw 'Manifest does not report completed, unverified M5B inference.'
}
if ($predictions.stage -ne 'UNVERIFIED_ACTIVITY_PREDICTIONS' -or
    $predictions.analysisType -ne 'PRETRAINED_KINETICS400_ACTIVITY_ONLY' -or
    $predictions.combatEventsDetected -ne $false -or
    $predictions.jobId -ne $jobId -or $predictions.videoId -ne $videoId) {
    throw 'Prediction metadata or job/video identity mismatch.'
}
if ($predictions.clipCount -ne $clips.clipCount -or
    $predictions.clipCount -ne $manifest.actionRecognition.clipCount -or
    $predictions.clipCount -ne @($predictions.predictions).Count -or
    $predictions.clipCount -le 0) {
    throw 'The number of predictions does not match the new job clips.'
}
foreach ($item in $predictions.predictions) {
    if ($item.status -notin @('UNKNOWN_LOW_SCORE', 'UNVERIFIED_ACTIVITY_LABEL') -or
        $item.inferenceDevice -ne 'cuda') {
        throw 'A prediction has an unexpected status or was not inferred on CUDA.'
    }
}
Write-Host ''
Write-Host 'M5B FULL END-TO-END: SUCCESS.'
Write-Host "videoId: $videoId"
Write-Host "jobId:   $jobId"
Write-Host "Clips classified: $($predictions.clipCount)"
Write-Host "UNKNOWN labels:   $($predictions.unknownCount)"
foreach ($item in $predictions.predictions) {
    $score = $item.topPredictions[0].softmaxScore
    Write-Host ('  Clip {0}: {1} (top softmax {2:P1}, {3})' -f $item.candidateIndex, $item.activityLabel, [double]$score, $item.status)
}
Write-Host "Output: $predictionsPath"
Write-Host 'Important: classifications are unverified activities, NOT confirmed punches or impacts.'
