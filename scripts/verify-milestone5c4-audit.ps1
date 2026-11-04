param([string]$ProjectPath = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath $ProjectPath -ErrorAction Stop).Path
$python = Join-Path $project 'ai-service\.venv\Scripts\python.exe'
Push-Location (Join-Path $project 'ai-service')
try {
    & $python -m pytest -q tests/test_tracking_audit.py
    if ($LASTEXITCODE -ne 0) { throw 'M5C.4 audit unit tests failed.' }
} finally { Pop-Location }
