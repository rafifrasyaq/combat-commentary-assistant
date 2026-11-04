param(
    [Parameter(Mandatory = $true)][Guid]$JobId,
    [string]$ProjectPath = (Split-Path -Parent $PSScriptRoot)
)
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath $ProjectPath -ErrorAction Stop).Path
$python = Join-Path $project 'ai-service\.venv\Scripts\python.exe'
$jobDir = Join-Path $project ('backend\storage\analysis\' + $JobId.ToString())
foreach ($file in @('pose_observations.json','pose_tracks.json')) {
    $source = Join-Path $jobDir $file
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing $source. Run M5C.3 first." }
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'Project Python virtual environment is missing.' }
Push-Location (Join-Path $project 'ai-service')
try {
    & $python -u -m app.action.tracking_audit --job-dir $jobDir
    if ($LASTEXITCODE -ne 0) { throw 'M5C.4 audit failed.' }
} finally { Pop-Location }
