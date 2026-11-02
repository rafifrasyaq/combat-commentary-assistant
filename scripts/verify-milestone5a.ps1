param([string]$ProjectPath = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath $ProjectPath).Path
$python = Join-Path $project 'ai-service\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'Python virtual environment is missing. Install ai-service requirements-dev.txt first.'
}
Write-Host 'Running Python tests (M5A, M4 regression, HTTP boundary)...'
Push-Location (Join-Path $project 'ai-service')
try {
    & $python -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw 'Python tests failed. Do NOT commit.' }
} finally { Pop-Location }
Write-Host 'Checking Git diff whitespace...'
& git -C $project diff --check
if ($LASTEXITCODE -ne 0) { throw 'git diff --check failed. Do NOT commit.' }
Write-Host 'Current Git change list:'
& git -C $project status --short --untracked-files=all
if ($LASTEXITCODE -ne 0) { throw 'git status failed.' }
Write-Host 'M5A code verification succeeded. Run one new upload E2E before commit.'
