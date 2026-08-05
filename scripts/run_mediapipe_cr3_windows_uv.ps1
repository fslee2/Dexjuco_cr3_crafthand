param(
    [int]$CameraId = 0,
    [double]$KeyboardXStep = 0.01,
    [int]$Steps = 2000,
    [switch]$NoViewer,
    [switch]$NoPreview,
    [string]$Venv = ".venv-mediapipe-win"
)

$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$PythonExe = Join-Path $Repo "$Venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    throw "Missing uv environment: $PythonExe. Run .\scripts\setup_windows_uv_mediapipe.ps1 first."
}

$env:PYTHONPATH = Join-Path $Repo "dexjoco"
$env:MUJOCO_GL = "glfw"

$ArgsList = @(
    "scripts\mediapipe_cr3_craft_click_mouse_shell.py",
    "--camera-id", "$CameraId",
    "--steps", "$Steps",
    "--keyboard-x-step", "$KeyboardXStep"
)

if (-not $NoViewer) {
    $ArgsList += "--viewer"
}
if (-not $NoPreview) {
    $ArgsList += "--preview"
}

Write-Host "[run] $PythonExe $($ArgsList -join ' ')"
& $PythonExe @ArgsList
