$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$ai = Join-Path $root 'ai-service'
$python = Join-Path $ai '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {throw 'Python .venv not found'}
Write-Host 'M5C.6: running new active pair tests (no GPU or model inference)...' -ForegroundColor Cyan
Push-Location $ai
try {
    & $python -m pytest -q .\tests\test_active_pair_selection.py
    if ($LASTEXITCODE -ne 0) {throw "M5C.6 tests failed: exit $LASTEXITCODE"}
} finally {Pop-Location}
