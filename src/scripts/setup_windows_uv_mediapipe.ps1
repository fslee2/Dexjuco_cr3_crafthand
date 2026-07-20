param(
    [string]$Venv = ".venv-mediapipe-win",
    [string]$Python = "3.11"
)

$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is not installed or not on PATH. Install uv first, then rerun this script."
}

Write-Host "[setup] repo: $Repo"
Write-Host "[setup] venv: $Venv"
Write-Host "[setup] python: $Python"

uv venv $Venv --python $Python

$PythonExe = Join-Path $Repo "$Venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    throw "Could not find venv Python at $PythonExe"
}

uv pip install --python $PythonExe -r requirements-windows-mediapipe.txt

Write-Host ""
Write-Host "[setup] done"
Write-Host "Run:"
Write-Host "  .\scripts\run_mediapipe_cr3_windows_uv.ps1 -CameraId 0"
