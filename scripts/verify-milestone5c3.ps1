param([string]$ProjectPath = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath $ProjectPath -ErrorAction Stop).Path
$python = Join-Path $project 'ai-service\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'Python virtual environment missing.' }
Push-Location (Join-Path $project 'ai-service')
try {
    Write-Host 'Running M5C.3 and existing Python tests without GPU or model download...'
    & $python -m pytest -q tests
    if ($LASTEXITCODE -ne 0) { throw 'Python tests failed.' }
} finally { Pop-Location }
