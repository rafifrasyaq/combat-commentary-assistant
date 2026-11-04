$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$ai = Join-Path $root 'ai-service'
$python = Join-Path $ai '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Python venv missing: $python" }
Push-Location $ai
try {
    & $python -m pytest -q tests/test_identity_review.py
    if ($LASTEXITCODE -ne 0) { throw 'M5C.5 review tests failed' }
} finally { Pop-Location }
