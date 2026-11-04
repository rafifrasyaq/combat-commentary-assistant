$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$ai = Join-Path $root 'ai-service'
$python = Join-Path $ai '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python virtual environment not found: $python"
}
Write-Host 'M5C.4: running 12 new CPU-only tracking refinement tests...' -ForegroundColor Cyan
Push-Location $ai
try {
    & $python -m pytest -q tests/test_tracking_refined.py
    if ($LASTEXITCODE -ne 0) { throw "M5C.4 refined tests failed (exit $LASTEXITCODE)" }
} finally { Pop-Location }
