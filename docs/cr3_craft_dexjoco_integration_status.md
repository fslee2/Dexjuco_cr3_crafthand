# CR3 + CRAFT DexJoCo Integration Status

**Date**: 2026-07-08
**Status**: Production-ready for policy/teleop integration. Minor packaging improvements ongoing.

---

## 1. Quickstart

```bash
# Prerequisites: activated venv with dexjoco installed
cd <DEXJOCO_ROOT>
export PYTHONPATH=<DEXJOCO_ROOT>/dexjoco
export MUJOCO_GL=egl

# Smoke test all CR3+CRAFT envs (no camera, no HaMeR, no display needed)
python scripts/smoke_cr3_craft_envs.py --env all --steps 2

# Test a single env
python scripts/smoke_cr3_craft_envs.py --env click_mouse_shell --steps 5 --render-check

# Python import
from dexjoco.sim.envs import (
    Cr3CraftReachDebugEnv,
    Cr3CraftClickMouseEnv,
    Cr3CraftClickMouseShellEnv,
)
```

---

## 2. Recommended Main Environment

**Use `Cr3CraftClickMouseShellEnv` as the canonical CR3+CRAFT environment.**

| Why | |
|---|---|
| Scene | Full Panda arena (walls, floor, table, monitor, lighting) — same as production Panda envs |
| Controller | Operational-space torque control (`dm_robotics` opspace) — highest physics fidelity |
| Arm actuators | Motor (torque-controlled) — realistic dynamics |
| Randomization | Table height, mouse position/yaw, mousepad position, display position, table texture (18 variants) |
| Mount | Pedestal at Panda base position `(-0.8, 0, 0.9)` — easy to adjust |

**`Cr3CraftReachDebugEnv`** is a minimal debug env (no task objects, green target marker). Use for IK testing and quick iteration.

**`Cr3CraftClickMouseEnv`** is a simpler standalone click_mouse task env with IK + position servoing. Suitable as a lightweight alternative.

---

## 3. 22D Action Contract

All three envs use an identical action layout:

```
action[0:3]   → target_xyz       (EE target position, world frame, unbounded box)
action[3:7]   → target_quat_wxyz  (EE target orientation, w-x-y-z quaternion)
action[7:22]  → craft15           (CRAFT finger commands, range [0, 2π])
```

**craft15 finger ordering**: Ring(PIP, MCP_fwd, MCP_side), Index(PIP, MCP_fwd, MCP_side), Thumb(PIP, MCP_fwd, MCP_side), Middle(PIP, MCP_fwd, MCP_side), Pinky(PIP, MCP_fwd, MCP_side).

Distal joints (DIP) are coupled to PIP via MuJoCo equality constraints and are NOT in the 15D action.

Values are in `[0, 2π]` and linearly mapped to each joint's actual range. **Set finger commands to 0 for fully open; 2π for fully closed (default).**

## 4. Observation Contract

```python
obs = {
    "state": {
        "tcp_pose":        np.ndarray(7,),   # [x, y, z, qw, qx, qy, qz]
        "arm_qpos":        np.ndarray(6,),   # CR3 joint positions (rad)
        "craft_qpos":      np.ndarray(20,),  # CRAFT joint positions (rad, 20 DOF incl. coupled distal)
        "target_tcp_pose": np.ndarray(7,),   # last commanded target
    },
    "images": {
        "front": np.ndarray(640, 640, 3),    # uint8 RGB
    },
}
```

## 5. Helper Methods

| Method | ReachDebug | ClickMouse | Shell |
|---|---|---|---|
| `get_initial_action()` → (22,) | ✓ | ✓ | ✓ |
| `get_end_effector_pose_matrix()` → (4,4) | ✓ | ✓ | ✓ |

## 6. What Is Complete

| Layer | Status |
|---|---|
| CR3 arm assets (7 meshes, 6 joints, kinematics) | Done |
| CRAFT hand assets (20 STL meshes, 20 joints, 5 fingers) | Done |
| Combined CR3+CRAFT body XML | Done |
| CRAFT distal coupling equalities (PIP→DIP, all 5 fingers) | Done |
| CRAFT self-collision excludes (per-finger adjacent links) | Done |
| Three MuJoCo XML scenes (reach_debug, click_mouse, click_mouse_shell) | Done |
| Gymnasium env classes (`gym.Env` via `MujocoGymEnv`) | Done |
| 22D `action_space` (Box) | Done |
| Dict `observation_space` | Done |
| `reset()` / `step(action)` / `render()` / `close()` | Done |
| `get_initial_action()` | Done (all 3) |
| `get_end_effector_pose_matrix()` | Done (all 3) |
| Click mouse task logic (detection, success counting, randomization) | Done (ClickMouse, Shell) |
| Package export (`__init__.py`, `__all__`) | Done (all 3) |
| Smoke test script (`smoke_cr3_craft_envs.py`) | Done |
| Pytest tests (`tests/test_cr3_craft_*.py`) | Done (all 3 envs covered) |

## 7. What Is Partial / In Progress

| Area | Status | Notes |
|---|---|---|
| Gymnasium registry (`gym.make`) | Not implemented | No `gym.register` exists anywhere in the project (Panda envs also lack it). Use direct imports. |
| Teleop control quality | Experimental | HaMeR and MediaPipe scripts connect to the env via 22D action, but hand-to-robot tracking quality is not validated. |
| CR3 mount position | May need tuning | Shell robot is at `(-0.8, 0.0, 0.9)`. Reach to table at `(-0.15, 0, 0)` may need workspace adjustment. |
| Table texture randomization parity | Close to Panda | 18 materials from Panda arena. Full parity audit vs `panda_click_mouse_env.py` not done. |
| Documentation depth | Basic | This status guide + detailed audit report exist. No full API reference. |

## 8. What NOT to Confuse with Core Env Integration

These are **separate concerns**, not gaps in the environment layer:

- **HaMeR inference latency/accuracy** — hand pose estimation quality
- **MediaPipe tracking quality** — alternative hand tracking pipeline
- **Calibration tuning** — hand-to-robot retargeting gains
- **Student policy distillation** — downstream training workflows
- **Webcam/camera setup** — hardware configuration
- **Panda arm environments** — unrelated to CR3+CRAFT

## 9. File Index

| Path | Purpose |
|---|---|
| `dexjoco/dexjoco/sim/envs/cr3_craft_reach_debug_env.py` | Minimal debug env |
| `dexjoco/dexjoco/sim/envs/cr3_craft_click_mouse_env.py` | IK-controlled click_mouse env |
| `dexjoco/dexjoco/sim/envs/cr3_craft_click_mouse_shell_env.py` | **Canonical** opspace-controlled click_mouse env |
| `dexjoco/dexjoco/sim/envs/xmls/cr3_craft_reach_debug.xml` | Debug scene XML |
| `dexjoco/dexjoco/sim/envs/xmls/cr3_craft_click_mouse.xml` | Standalone click_mouse scene XML |
| `dexjoco/dexjoco/sim/envs/xmls/cr3_craft_click_mouse_shell.xml` | Panda-arena-based shell scene XML |
| `dexjoco/dexjoco/sim/envs/xmls/cr3_craft/models/cr3_robot_hand_body.xml` | Shared CR3+CRAFT body (included by all scenes) |
| `scripts/smoke_cr3_craft_envs.py` | Headless smoke test (no camera/HaMeR needed) |
| `tests/test_cr3_craft_reach_debug.py` | pytest: action contract, step stability |
| `docs/cr3_craft_dexjoco_integration_audit.md` | Detailed technical audit (this file's reference) |
| `docs/cr3_craft_dexjoco_integration_status.md` | **This file** — user-facing status guide |

## 10. Final Verdict

**The CR3+CRAFT environment layer is ready for teleop and policy work.** The 22D action interface is uniform and runtime-verified across all three envs. Any policy that outputs (target_xyz, target_quat, craft15) can drive them. The current gap is teleop control quality and hand-to-robot mapping — a tuning problem, not an integration problem.
