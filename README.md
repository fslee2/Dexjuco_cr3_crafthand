# CR3 × CRAFT Teleoperation Showcase

[English](README.md) · [中文版](README.zh-CN.md)

CR3 six-axis robot arm and CRAFT dexterous hand integration, task environments, and teleoperation experiments in DexJoCo/MuJoCo.

> This repository is a curated showcase and handoff package extracted from a larger research workspace. It contains the CR3+CRAFT integration layer, MuJoCo scenes, teleoperation scripts, tests, documentation, and selected media. The full DexJoCo base project, third-party model weights, CAD files, and local environments are intentionally excluded.

[![GitHub](https://img.shields.io/badge/GitHub-fslee2%2FDexjuco__cr3__crafthand-181717?logo=github)](https://github.com/fslee2/Dexjuco_cr3_crafthand)
[![Simulation](https://img.shields.io/badge/Simulation-MuJoCo%20%2F%20DexJoCo-1f6feb)](#project-scope)
[![Teleoperation](https://img.shields.io/badge/Teleoperation-MediaPipe%20%2B%20Dual--Camera-2ea44f)](#teleoperation-pipelines)

![CR3 + CRAFT camera grid](assets/images/cr3_craft_camera_grid.png)

![CR3 + CRAFT teleoperation](assets/gifs/cr3_craft_x_axis_teleop.gif)

## Project scope

The project addresses a concrete integration problem: replacing a Panda/Allegro-style robot setup in DexJoCo with a CR3 arm and a CRAFT dexterous hand, while allowing different human-input devices to drive the same robot action interface.

It is a research-oriented simulation and teleoperation platform, not a standalone hardware controller or a polished product. The system connects:

```text
MediaPipe / keyboard / dual cameras / Quest WebXR
                         ↓
             end-effector pose + hand commands
                         ↓
                    unified 22D action
                         ↓
                  CR3 + CRAFT environment
                         ↓
                 MuJoCo physics and task feedback
```

## Action and observation interfaces

All three environments share the following 22-dimensional action layout:

```text
action[0:3]   = target_xyz        # target end-effector position
action[3:7]   = target_quat_wxyz  # target orientation, scalar-first
action[7:22]  = craft15           # 15 direct CRAFT finger commands
```

The 15 hand commands are the main control inputs for five fingers. The distal joints are coupled to the proximal joints through MuJoCo equality constraints, so the external command is 15D while the internal CRAFT state contains 20 joints.

The observation dictionary exposes the current TCP pose, CR3 joint positions, CRAFT joint positions, target TCP pose, and a front-view image.

## Environments

| Environment | Purpose | Status |
| --- | --- | --- |
| `Cr3CraftReachDebugEnv` | Minimal reach and controller debugging | Fastest environment for IK, TCP, and hand checks |
| `Cr3CraftClickMouseEnv` | Standalone click-mouse task | Simplified scene for task development |
| `Cr3CraftClickMouseShellEnv` | Main task environment | Preserves the DexJoCo/Panda arena while replacing the robot with CR3+CRAFT |

`Cr3CraftClickMouseShellEnv` is the recommended mainline environment. It includes mouse and mousepad randomization, click detection, display feedback, and success counting.

## Teleoperation pipelines

### MediaPipe single-camera teleoperation

- Palm motion controls TCP Y/Z.
- Finger curl controls the 15 CRAFT hand commands.
- Keyboard input compensates for the difficult forward/backward TCP X direction.
- Calibration, filtering, deadbands, and maximum-step limits are included.

This is a lightweight, runnable prototype rather than full 3D hand reconstruction. Monocular depth is its main limitation.

### Dual-camera MediaPipe prototype

```text
Front camera → TCP Y/Z + CRAFT fingers
Side camera  → TCP X
```

This is a pragmatic two-view 2D mapping, not calibrated stereo reconstruction.

### Keyboard control

Keyboard control is useful for environment and task debugging:

```text
Z / Numpad 7  → TCP X-
X / Numpad 9  → TCP X+
H / Numpad 5  → clear X offset
R             → recalibrate
Q / Esc       → quit
```

### Quest / WebXR MVP

`dexjoco/tasks/quest_teleop.py` provides a prototype bridge for Quest pose and button states. It handles quaternion conversion, relative-pose calibration, position smoothing, grip-to-hand mapping, and WebSocket state exchange. It should be treated as an experimental interface, not a production Quest controller.

## Repository layout

```text
.
├── assets/                         # Selected images and GIFs for documentation
├── docs/                           # Architecture, audits, handoff, and setup notes
├── dexjoco/                        # Standalone importable Python package
│   ├── sim/envs/                   # CR3+CRAFT environments and MuJoCo XMLs
│   ├── sim/controllers/             # Operational-space controller
│   └── tasks/                      # Quest state receiver and action mapping
├── scripts/                        # Keyboard, MediaPipe, dual-camera, and Quest entrypoints
├── configs/                        # CR3+CRAFT teleoperation configurations
├── tests/                          # Lightweight environment and mapping tests
├── pyproject.toml                  # Installable package definition
└── requirements-windows-mediapipe.txt
├── tools/                          # Showcase-media rendering utility
├── PROJECT_SCOPE.md                # Publishing boundary
└── README.zh-CN.md                 # Chinese documentation
```

## Runtime requirements

This repository is self-contained for the CR3+CRAFT simulation and the lightweight MediaPipe/keyboard teleoperation paths. It includes the Python package, controllers, CR3 and CRAFT meshes, mouse/display assets, task XMLs, scripts, and setup files.

The repository includes the lightweight MediaPipe `hand_landmarker.task` model, so a fresh clone has the runtime asset needed by the camera demo. It also includes five CR3+CRAFT manipulation tasks: hammer-and-nail, bucket picking, tongs, watering a plant, and folding glasses.

On Windows, keep the clone path ASCII-only (for example `E:\\src\\Dexjuco_cr3_crafthand`). Some MuJoCo XML loading paths fail when the repository is nested under a directory containing Chinese characters.

## Quick validation

### Environment smoke tests

```powershell
cd <CLONE_DIR>
python scripts/smoke_cr3_craft_envs.py --env all --steps 2
python scripts/smoke_cr3_craft_envs.py --env click_mouse_shell --steps 5 --render-check
python scripts/smoke_cr3_craft_tasks.py
```

### Windows MediaPipe

```powershell
cd <CLONE_DIR>
.\scripts\setup_windows_uv_mediapipe.ps1
.\scripts\run_mediapipe_cr3_windows_uv.ps1 -CameraId 0
```

### Dual-camera prototype

```powershell
.\scripts\run_dual_mediapipe_cr3_windows_uv.ps1 `
  -FrontCameraId 0 `
  -SideCameraId 1
```

Use `-SideXSign -1` if the side-camera X direction is reversed.

## Completion status

### Implemented or substantially complete

- CR3 and CRAFT MuJoCo integration scenes;
- three CR3+CRAFT environment classes;
- unified 22D action and standard environment lifecycle;
- CRAFT distal-joint coupling and actuator control;
- click-mouse task adaptation and shell environment;
- keyboard, MediaPipe, and dual-camera teleoperation entrypoints;
- Quest/WebXR state and action-mapping prototype;
- smoke tests, handoff notes, and selected showcase media.

### Experimental or still in progress

- hand-to-robot calibration and teleoperation naturalness;
- monocular depth and dual-camera X mapping;
- real Quest browser, TLS, and network deployment;
- full HaMeR/MobileHand integration;
- real CR3+CRAFT hardware closed-loop control and Sim2Real safety validation;
- large-scale data collection and policy evaluation.

## Excluded from this repository

- Full third-party DexJoCo, HaMeR, or MobileHand checkouts;
- model weights, caches, local virtual environments, and serialized datasets;
- SolidWorks/CAD and 3D-print source files;
- Quest Developer Hub installers;
- paper raw data, training logs, and large experiment outputs.

## Documentation

- [Architecture and code guide](docs/architecture_guide.md)
- [DexJoCo integration status](docs/cr3_craft_dexjoco_integration_status.md)
- [Integration audit](docs/cr3_craft_dexjoco_integration_audit.md)
- [Teleoperation handoff](docs/cr3_craft_teleop_handoff.md)
- [Windows + uv + MediaPipe](docs/windows_uv_mediapipe_env.md)
- [Quest 3 WebXR MVP](docs/quest3_teleop_mvp.md)
- [Publishing scope](PROJECT_SCOPE.md)

## One-sentence summary

This repository connects a CR3 arm, a CRAFT dexterous hand, MuJoCo task environments, and multiple human-input pipelines into a research-oriented simulation and teleoperation platform.
