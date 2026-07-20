# Windows uv Environment: MediaPipe CR3+CRAFT Teleop

This is the lightweight Windows environment for the current validation path:

```text
scripts/mediapipe_cr3_craft_click_mouse_shell.py
```

It intentionally does not install HaMeR, PyTorch, pytorch3d, ViTPose, or training/distillation dependencies.

## Create Environment

From PowerShell:

```powershell
cd E:\material_for_uci_experiment\DexJoCo-CRAFT-Teleop
.\scripts\setup_windows_uv_mediapipe.ps1
```

This creates:

```text
.venv-mediapipe-win
```

Python version:

```text
3.11
```

## Run Current Validation Script

```powershell
cd E:\material_for_uci_experiment\DexJoCo-CRAFT-Teleop
.\scripts\run_mediapipe_cr3_windows_uv.ps1 -CameraId 0
```

Equivalent raw command:

```powershell
$env:PYTHONPATH="E:\material_for_uci_experiment\DexJoCo-CRAFT-Teleop\dexjoco"
$env:MUJOCO_GL="glfw"

.\.venv-mediapipe-win\Scripts\python.exe scripts\mediapipe_cr3_craft_click_mouse_shell.py `
  --camera-id 0 `
  --viewer `
  --preview `
  --keyboard-x-step 0.01
```

## Controls

```text
Palm left/right/up/down -> TCP Y/Z
Fingers -> CRAFT hand
Z or numpad 7 -> TCP X-
X or numpad 9 -> TCP X+
H or numpad 5 -> clear keyboard X offset
R -> recalibrate MediaPipe center
Q/Esc -> quit
```

## Why Windows Uses glfw

Windows MuJoCo rejects:

```text
MUJOCO_GL=egl
```

Use:

```text
MUJOCO_GL=glfw
```

The WSL/Linux path can still use `egl`.

## Troubleshooting

If camera 0 is wrong:

```powershell
.\scripts\run_mediapipe_cr3_windows_uv.ps1 -CameraId 1
```

If movement in X is too aggressive:

```powershell
.\scripts\run_mediapipe_cr3_windows_uv.ps1 -KeyboardXStep 0.003
```

If the key controls do not respond, click the OpenCV preview/key window first.

If MediaPipe cannot find the hand landmarker task, confirm this file exists:

```text
E:\material_for_uci_experiment\DexJoCo-CRAFT-Teleop\hand_landmarker.task
```

## Optional: Dual-Camera Test

Use this when two ordinary webcams are available:

```powershell
.\scripts\run_dual_mediapipe_cr3_windows_uv.ps1 `
  -FrontCameraId 0 `
  -SideCameraId 1
```

Mapping:

```text
front camera -> TCP Y/Z and CRAFT hand
side camera  -> TCP X
```

If the side-camera X direction is reversed:

```powershell
.\scripts\run_dual_mediapipe_cr3_windows_uv.ps1 -FrontCameraId 0 -SideCameraId 1 -SideXSign -1
```

If X is too aggressive:

```powershell
.\scripts\run_dual_mediapipe_cr3_windows_uv.ps1 -SideXGain 0.04
```
