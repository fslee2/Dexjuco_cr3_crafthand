# CR3 + CRAFT Teleop Handoff

Date: 2026-07-20

This document summarizes the current DexJoCo CR3+CRAFT status, whether Linux/WSL is required, what went wrong during teleop exploration, how those issues were handled, and what should be done next.

## 1. Short Answer: Is Linux Required?

Not strictly for the DexJoCo environment layer.

The core CR3+CRAFT envs are Python + MuJoCo + Gymnasium code. In principle they can run on either Linux/WSL or Windows if the Python environment, MuJoCo rendering backend, paths, and assets are set up correctly.

However, for this project right now, WSL/Linux is the recommended primary runtime.

Reason:

- Most repo commands, docs, paths, and virtualenv setup assume WSL:
  - `<DEXJOCO_ROOT>`
  - `source ~/.venv-hamer/bin/activate`
  - `export PYTHONPATH=.../dexjoco`
- Several scripts still default to `MUJOCO_GL=egl`, which is correct for WSL/Linux headless rendering but invalid on Windows MuJoCo in the tested conda env.
- HaMeR integration is easiest in the existing WSL `.venv-hamer` environment and HaMeR checkout.
- Camera and OpenCV behavior differs between WSL and Windows.
- The currently audited CR3+CRAFT smoke tests were mainly validated from WSL/Linux.

Windows is still useful for quick local experiments when using `conda activate env_mujoco`, but do not assume every DexJoCo script will run unchanged there.

Practical rule:

- Use WSL/Linux for DexJoCo + CR3+CRAFT environment validation and main teleop work.
- Use Windows only for isolated demos or when the script explicitly handles `MUJOCO_GL=glfw`.

## 2. Current Integration State

CR3+CRAFT is already integrated into DexJoCo at the environment/interface level.

Canonical env:

```python
from dexjoco.sim.envs.cr3_craft_click_mouse_shell_env import Cr3CraftClickMouseShellEnv
```

Action contract:

```text
action[0:3]   target_xyz
action[3:7]   target_quat_wxyz
action[7:22]  craft15
```

Important entry points:

```text
scripts/smoke_cr3_craft_envs.py
scripts/mediapipe_cr3_craft_click_mouse_shell.py
scripts/hamer_cr3_craft_click_mouse_shell.py
scripts/keyboard_cr3_craft_click_mouse_shell.py
scripts/quest_cr3_craft_click_mouse_shell.py
```

The macro conclusion remains:

- CR3 arm + CRAFT hand + click_mouse shell scene are integrated.
- The 22D action interface exists and works.
- The weak part is teleop control quality, not the DexJoCo env/scenario layer.

## 3. Recommended Runtime Commands

WSL/Linux:

```bash
cd <DEXJOCO_ROOT>
source ~/.venv-hamer/bin/activate
export PYTHONPATH=<DEXJOCO_ROOT>/dexjoco
export MUJOCO_GL=egl
```

Smoke test:

```bash
python scripts/smoke_cr3_craft_envs.py --env all --steps 2
python scripts/smoke_cr3_craft_envs.py --env click_mouse_shell --steps 5 --render-check
```

Current practical MediaPipe teleop path:

```bash
python scripts/mediapipe_cr3_craft_click_mouse_shell.py \
  --camera-id 0 \
  --viewer \
  --preview \
  --keyboard-x-step 0.01
```

Controls in that MediaPipe script:

```text
Palm left/right/up/down -> TCP Y/Z
Fingers -> CRAFT hand
Z or numpad 7 -> TCP X-
X or numpad 9 -> TCP X+
H or numpad 5 -> clear keyboard X offset
R -> recalibrate MediaPipe center
Q/Esc -> quit
```

## 4. What Was Changed Most Recently

The correct DexJoCo hand teleop file is:

```text
scripts/mediapipe_cr3_craft_click_mouse_shell.py
```

Recent intent:

- Keep the existing MediaPipe hand teleop behavior.
- Add a simple keyboard-controlled X offset for forward/backward TCP motion.
- Do not replace the existing teleop file with a separate toy script.

Implementation shape:

- MediaPipe palm still controls Y/Z.
- MediaPipe fingers still control CRAFT.
- Keyboard offset is added only at the final `action[:3]` stage.
- X offset resets on recalibration or hand-loss recalibration.

Validation already done:

```text
python -m py_compile scripts/mediapipe_cr3_craft_click_mouse_shell.py
conda run -n env_mujoco python scripts/mediapipe_cr3_craft_click_mouse_shell.py --help
```

Not yet validated:

- Full live camera + viewer behavior after keyboard-X addition.

## 5. Problems Encountered And Fixes

### Problem: Confusing env integration with teleop quality

Symptom:

- We spent too much time on HaMeR/MediaPipe/Quest/depth performance while the CR3+CRAFT environment itself was already mostly complete.

Resolution:

- Separate layers:
  - Environment layer: CR3+CRAFT XML, env classes, 22D action, reset/step/render.
  - Input layer: MediaPipe, HaMeR, Quest, keyboard.
  - Retargeting layer: hand/controller pose to target_xyz/quaternion/craft15.

How to avoid next time:

- First run smoke tests and confirm the env contract.
- Then test one input source at a time.
- Do not treat bad teleop behavior as proof that the env layer is broken.

### Problem: HaMeR root path mismatch

Symptom:

```text
FileNotFoundError: Could not find HaMeR checkout under /mnt/e/hand-hamer-vision-teleop
```

Actual root:

```text
E:\material_for_uci_experiment\HaMeR-CRAFT-Teleop
/mnt/e/material_for_uci_experiment/HaMeR-CRAFT-Teleop
```

Resolution:

- Pass `--hamer-root /mnt/e/material_for_uci_experiment/HaMeR-CRAFT-Teleop`.
- Update scripts/configs to search `HaMeR-CRAFT-Teleop`.

How to avoid:

- Always pass explicit model/repo roots for large external projects.

### Problem: HaMeR inference too slow

Symptom:

- Teleop was visibly slow and not suitable for low-latency robot control.

Explored options:

- Async inference.
- Model distillation.
- MobileHand lightweight teacher.
- MediaPipe geometry.
- Quest 3 WebXR.

Current conclusion:

- HaMeR is not the practical first choice for a responsive teleop loop.
- For now, use MediaPipe hand control plus keyboard X, or use Quest/controller input later.

How to avoid:

- Do not build the whole control loop around a heavy model until latency is measured.
- Keep the robot control path independent from inference FPS.

### Problem: Student/distillation pipeline was overcomplicated and brittle

Symptoms:

- Missing `hand_landmarker.task`.
- Downloaded HaMeR demo data unexpectedly.
- Segmentation fault during labeling/training.

Resolution:

- Paused distillation work.

How to avoid:

- Do not start training before the online teleop mapping is known to be useful.
- If training resumes, isolate it in a separate pipeline with a small test video and fixed dependency versions.

### Problem: MediaPipe geometry depth was not reliable enough

Symptom:

- Apparent depth jumped when the hand moved slightly.
- `dz` was unstable because it came from hand bbox/scale geometry.

Resolution:

- Do not rely on pure geometry depth as the main depth channel.
- Use a manual keyboard X offset for now.

How to avoid:

- Do not infer metric depth from single-camera 2D landmarks without validation.
- Treat it as a fallback or weak signal only.

### Problem: MobileHand whole-frame inference detected the face

Symptom:

- MobileHand keypoints appeared around the user's face instead of the hand.

Resolution:

- Add MediaPipe hand crop before MobileHand.

Remaining issue:

- MobileHand still gives relative weak-perspective depth, not true metric depth.
- Inference is still not free, and it did not become the cleanest path.

### Problem: Quest 3 setup blocked by device/ADB/network/controller issues

Symptoms:

- Quest initially demanded controllers.
- ADB did not detect the headset even after Meta Quest Developer Hub installed.
- Network/eduroam made direct web flow uncertain.

Resolution:

- Built a first-pass WebXR/Quest architecture, but did not complete real-device end-to-end testing.

How to avoid:

- Before building WebXR features, verify:
  - Quest developer mode.
  - `adb devices` detects device.
  - Quest and workstation have same reachable network or USB bridge path.
  - HTTPS certificate flow works.

### Problem: Windows MuJoCo rejected `MUJOCO_GL=egl`

Symptom:

```text
RuntimeError: invalid value for environment variable MUJOCO_GL: egl
```

Resolution:

- On Windows, use `MUJOCO_GL=glfw`.
- In scripts that may run on Windows, use:

```python
os.environ.setdefault("MUJOCO_GL", "glfw" if os.name == "nt" else "egl")
```

How to avoid:

- Do not hardcode `egl` in cross-platform scripts.
- Keep WSL/Linux command docs separate from Windows conda command docs.

### Problem: Wrong direction during last request

Symptom:

- A separate minimal keyboard-X script was created even though the user meant the existing DexJoCo hand teleop file.

Resolution:

- Deleted the separate minimal script.
- Modified the existing MediaPipe hand teleop file instead.

How to avoid:

- When the user says "DexJoCo 之前有用手部进行 teleop 的文件", first identify and patch the existing file:

```text
scripts/mediapipe_cr3_craft_click_mouse_shell.py
```

- Do not create a parallel entrypoint unless explicitly asked.

## 6. Open Problems

### A. Live validation of MediaPipe + keyboard X

Status:

- Syntax/help validated.
- Full live run still needs user test.

Next command:

```bash
python scripts/mediapipe_cr3_craft_click_mouse_shell.py \
  --camera-id 0 \
  --viewer \
  --preview \
  --keyboard-x-step 0.01
```

Expected:

- Palm controls Y/Z.
- Keyboard controls X.
- CRAFT fingers respond to finger curl.

If key focus is wrong:

- Click the OpenCV preview/key window before pressing keys.

### B. HaMeR remains too slow

Possible solutions:

- Use HaMeR only for offline labeling.
- Use lighter input sources for live teleop.
- Keep robot control loop decoupled from model inference.

Recommended for now:

- Do not prioritize HaMeR live teleop unless there is a strong reason.

### C. True monocular depth remains unresolved

Possible solutions:

- Manual keyboard X offset, current simplest option.
- Dual ordinary cameras with two-view 2D teleop mapping.
- Quest/controller 6DoF pose, likely cleaner than monocular depth.
- AprilTag/ArUco wrist marker for supervised depth data.
- Lightweight pretrained hand model as auxiliary signal, but validate carefully.

Recommended:

- Use keyboard X now.
- Test dual-camera mapping next if two webcams are available.
- If hardware permits, move to Quest/controller pose later.

### D. Quest real-device flow incomplete

Possible solutions:

- Fix ADB detection first.
- Use a non-eduroam hotspot/router.
- Use WebXR with HTTPS and WebSocket.
- Or bypass browser/WebXR and use controller data through another runtime if available.

Recommended:

- Do not block current CR3+CRAFT validation on Quest.

### E. Windows support is partial

Possible solutions:

- Audit all scripts for `MUJOCO_GL=egl`.
- Add platform-aware GL defaults.
- Add a Windows-specific quickstart using `conda activate env_mujoco`.

Recommended:

- Treat WSL as canonical until Windows is explicitly productized.

## 7. Is There A Simpler Method?

Yes.

Simplest reliable path right now:

```text
DexJoCo Cr3CraftClickMouseShellEnv
  + existing MediaPipe hand teleop for Y/Z and CRAFT
  + keyboard X offset for forward/backward
```

This avoids:

- HaMeR latency.
- Monocular depth instability.
- Quest setup.
- Training/distillation.
- Large new abstractions.

Even simpler for env-only testing:

```text
scripts/keyboard_cr3_craft_click_mouse_shell.py --viewer
```

That existing script already controls full Cartesian xyz and rotation with keyboard. Use it when no hand input is needed.

But for the user's current intended path, the better simple approach is not a new toy script. It is:

```text
Patch scripts/mediapipe_cr3_craft_click_mouse_shell.py in place.
```

## 8. Next Agent Checklist

1. Do not start with HaMeR or training.
2. Confirm environment smoke tests pass.
3. Run the MediaPipe teleop entry with `--preview --viewer`.
4. Verify keyboard focus and X offset keys.
5. If X direction or step size feels wrong, only tune `--keyboard-x-step` or swap key signs.
6. If MediaPipe Y/Z feels bad, tune `WORKSPACE_DELTA`, `PALM_CENTER_DEADBAND`, `FILTER_ALPHA`, and `FILTER_MAX_STEP`.
7. Keep every change inside the existing relevant entrypoint unless a new entrypoint is explicitly requested.
8. Do not assume Windows unless the command is being run from Windows conda.

## 9. Current Best Mental Model

The project is not blocked on CR3+CRAFT integration.

The project is blocked on choosing a usable human input source.

Current best input source for immediate progress:

```text
MediaPipe for easy Y/Z + fingers
Keyboard for X depth
DexJoCo 22D action as the stable backend interface
```

## 10. Dual-Camera Ordinary Webcam Test

Added on 2026-07-20:

```text
scripts/dual_mediapipe_cr3_craft_click_mouse_shell.py
scripts/run_dual_mediapipe_cr3_windows_uv.ps1
```

Purpose:

```text
Use two ordinary webcams to avoid single-camera depth estimation.
```

This is not stereo reconstruction.  There is no checkerboard calibration, no
camera extrinsic solve, and no triangulation.  It is a simpler two-view control
mapping:

```text
front camera:
  palm left/right -> TCP Y
  palm up/down    -> TCP Z
  fingers         -> CRAFT hand

side camera:
  palm horizontal motion -> TCP X
```

Recommended physical layout:

```text
front camera: facing the operator/hand from the front
side camera:  placed roughly 90 degrees to the side, so the user's forward/back
              hand motion appears as horizontal motion in the side image
```

Windows uv command:

```powershell
cd <DEXJOCO_ROOT>
.\scripts\run_dual_mediapipe_cr3_windows_uv.ps1 `
  -FrontCameraId 0 `
  -SideCameraId 1
```

Raw command:

```powershell
.\.venv-mediapipe-win\Scripts\python.exe scripts\dual_mediapipe_cr3_craft_click_mouse_shell.py `
  --front-camera-id 0 `
  --side-camera-id 1 `
  --viewer `
  --preview
```

Runtime controls:

```text
Z or numpad 7 -> manual TCP X-
X or numpad 9 -> manual TCP X+
H or numpad 5 -> clear manual X
R              -> recalibrate both camera centers
Q/Esc          -> quit
```

Tuning:

```powershell
# Reverse X if side camera direction is wrong
.\scripts\run_dual_mediapipe_cr3_windows_uv.ps1 -SideXSign -1

# Reduce X gain if motion is too aggressive
.\scripts\run_dual_mediapipe_cr3_windows_uv.ps1 -SideXGain 0.04
```

What to test tomorrow:

1. Confirm both camera IDs open.
2. Put hand in a neutral pose visible to both cameras for the first calibration seconds.
3. Move hand left/right/up/down in front camera and check Y/Z.
4. Move hand forward/back relative to the front camera and check whether side camera produces X.
5. If X is reversed, use `-SideXSign -1`.
6. If X is too large or noisy, reduce `-SideXGain`.

Known limitations:

- Requires both cameras to see the same hand.
- Not metric 3D; it is teleop mapping.
- If either camera loses the hand, that camera will recalibrate on the next detection.
- Side camera placement matters a lot.
