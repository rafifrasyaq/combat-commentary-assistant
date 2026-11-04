param([string]$ProjectPath = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath $ProjectPath).Path
$python = Join-Path $project 'ai-service\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'Python virtual environment not found.'
}
Write-Host 'Running M5B + M5A + M4 Python tests (no GPU or model download needed)...'
Push-Location (Join-Path $project 'ai-service')
try {
    & $python -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw 'Python tests failed.' }
} finally { Pop-Location }
& git -C $project diff --check
if ($LASTEXITCODE -ne 0) { throw 'Git whitespace check failed.' }
Write-Host 'M5B tests completed. No Git commit or PR was created.'
