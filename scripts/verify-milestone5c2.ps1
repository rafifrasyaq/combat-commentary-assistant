param([string]$ProjectPath = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath $ProjectPath -ErrorAction Stop).Path
$python = Join-Path $project 'ai-service\.venv\Scripts\python.exe'
Write-Host 'Running Python tests for existing milestones + pose pilot (no model download)...'
Push-Location (Join-Path $project 'ai-service')
try {
    & $python -m pytest -q tests
    if ($LASTEXITCODE -ne 0) { throw 'Python unit tests failed.' }
} finally { Pop-Location }
