"""Keyboard Cartesian control for the CR3+CRAFT click_mouse shell environment.

Sends 22D actions: [target_xyz(3), target_quat_wxyz(4), craft15(15)].
Translation xyz and rotation quaternion (world-frame Euler increments) are
controlled via keyboard; the hand is toggled open/closed.

Rotation convention: world-frame composition ``delta_rot * target_rot``
where *delta_rot* is the incremental Euler rotation (roll/pitch/yaw)
expressed in the fixed world frame.

Numpad controls:
  8/2: Z up / down
  4/6: Y left / right
  7/9: X backward / forward
  1/3: roll/RX  -/+
  0/.: pitch/RY -/+
  -/+: yaw/RZ  -/+
  5: home / reset

WASD+ fallback (when numpad is unavailable):
  W/S: Z up / down
  A/D: Y left / right
  Q/E: X backward / forward
  R/F: roll/RX  -/+
  T/G: pitch/RY -/+
  Y/U: yaw/RZ  -/+

Other keys:
  C: close hand (curl)
  O: open hand (extend)
  H: home / reset
  Esc: quit

Modes
-----
--viewer        Launch the MuJoCo interactive viewer alongside control.
--smoke         Run a deterministic scripted sequence;
                no keyboard or viewer required.  Good for CI / sanity.
--steps N       Maximum environment steps (default: 200).
--step-xyz S    Translation increment in metres (default: 0.02).
--step-rot D    Rotation increment in degrees (default: 5.0).
--image-observations  Render images in observation (default: off — uses
                render_mode=\"none\" for lower latency).
--control-dt D  Control timestep in seconds (default: 0.01).
--physics-dt D  Physics timestep in seconds (default: 0.002).
--benchmark     Print average step Hz at end (also implied by --smoke).

WSL usage
---------
  cd /mnt/e/material_for_uci_experiment/DexJoCo-CRAFT-Teleop
  source ~/.venv-hamer/bin/activate
  export PYTHONPATH=/mnt/e/material_for_uci_experiment/DexJoCo-CRAFT-Teleop/dexjoco

  # Interactive with viewer
  python scripts/keyboard_cr3_craft_click_mouse_shell.py --viewer

  # Headless smoke test
  python scripts/keyboard_cr3_craft_click_mouse_shell.py --smoke --steps 12

Keyboard focus
--------------
When running interactively (without --smoke), press keys in the small
OpenCV window titled "Keyboard Control (Esc=quit)".  The MuJoCo viewer
responds to its own built-in keybindings independently.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DEXJOCO = _REPO / "dexjoco"
if str(_DEXJOCO) not in sys.path:
    sys.path.insert(0, str(_DEXJOCO))

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from scipy.spatial.transform import Rotation

from dexjoco.sim.envs.cr3_craft_click_mouse_shell_env import (
    CRAFT_VALUE_MAX,
    Cr3CraftClickMouseShellEnv,
)

# ---------------------------------------------------------------------------
# Quaternion helpers  (scipy xyzw <-> DexJoCo wxyz)
# ---------------------------------------------------------------------------


def _rot_from_wxyz(wxyz: np.ndarray) -> Rotation:
    """Convert DexJoCo wxyz quaternion to scipy Rotation."""
    return Rotation.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])


def _wxyz_from_rot(rot: Rotation) -> np.ndarray:
    """Convert scipy Rotation to DexJoCo wxyz quaternion (normalised)."""
    xyzw = rot.as_quat(canonical=True)
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)


def _apply_world_euler_delta(rot: Rotation, delta_rpy: np.ndarray) -> Rotation:
    """Apply a world-frame Euler delta (roll, pitch, yaw in radians).

    World-frame composition:  ``delta_rot * rot``.
    """
    delta = Rotation.from_euler("xyz", delta_rpy)
    return delta * rot


# ---------------------------------------------------------------------------
# Key-code helpers
# ---------------------------------------------------------------------------


def _numpad_digit_codes(char: str) -> list[int]:
    codes = [ord(char)]
    if char.isdigit():
        codes.append(0x60 + int(char))  # VK_NUMPAD0 .. VK_NUMPAD9
    return codes


NUMPAD_NAV_CODES: dict[str, list[int]] = {
    "8": [2490368],  # Up
    "2": [2621440],  # Down
    "4": [2424832],  # Left
    "6": [2555904],  # Right
    "7": [2359296],  # Home
    "9": [2162688],  # PageUp
    "1": [2293760],  # End
    "3": [2228224],  # PageDown
    "0": [2949120],  # Insert
    ".": [3014656],  # Delete
}


def _all_codes(char: str) -> list[int]:
    """Every key-code that might represent *char* on a numpad, with or
    without NumLock."""
    return _numpad_digit_codes(char) + NUMPAD_NAV_CODES.get(char, [])


# ---------------------------------------------------------------------------
# Key → (axis, sign) for translation (axes 0-2) and rotation (axes 3-5)
# ---------------------------------------------------------------------------

_TRANS_KEY_MAP: dict[int, tuple[int, int]] = {}

# -- Numpad translation ---------------------------------------------------
for char, axis, sign in [
    ("8", 2, +1),  # Z+
    ("2", 2, -1),  # Z-
    ("4", 1, -1),  # Y-
    ("6", 1, +1),  # Y+
    ("7", 0, -1),  # X-
    ("9", 0, +1),  # X+
]:
    for code in _all_codes(char):
        _TRANS_KEY_MAP[code] = (axis, sign)

# -- WASD fallback translation --------------------------------------------
for char, axis, sign in [
    ("w", 2, +1),
    ("s", 2, -1),
    ("a", 1, -1),
    ("d", 1, +1),
    ("q", 0, -1),
    ("e", 0, +1),
]:
    _TRANS_KEY_MAP[ord(char)] = (axis, sign)


_ROT_KEY_MAP: dict[int, tuple[int, int]] = {}

# -- Numpad rotation  (1/3=roll, 0/.=pitch) ------------------------------
for char, axis, sign in [
    ("1", 3, -1),  # roll-
    ("3", 3, +1),  # roll+
    ("0", 4, -1),  # pitch-
    (".", 4, +1),  # pitch+
]:
    for code in _all_codes(char):
        _ROT_KEY_MAP[code] = (axis, sign)

# -- Numpad +/- (yaw) -----------------------------------------------------
_ROT_KEY_MAP[0x6B] = (5, +1)  # VK_ADD
_ROT_KEY_MAP[0x6D] = (5, -1)  # VK_SUBTRACT

# -- Keyboard fallback rotation (R/F=roll, T/G=pitch, Y/U=yaw) -----------
for char, axis, sign in [
    ("r", 3, -1),
    ("f", 3, +1),
    ("t", 4, -1),
    ("g", 4, +1),
    ("y", 5, -1),
    ("u", 5, +1),
    # Regular keyboard - / + / = for yaw
    ("-", 5, -1),
    ("+", 5, +1),
    ("=", 5, +1),
]:
    _ROT_KEY_MAP[ord(char)] = (axis, sign)


_AXIS_NAME: dict[int, str] = {
    0: "X", 1: "Y", 2: "Z",
    3: "RX", 4: "RY", 5: "RZ",
}


# ---------------------------------------------------------------------------
# Smoke-mode helpers
# ---------------------------------------------------------------------------


def _build_smoke_actions(
    initial: np.ndarray,
    n_steps: int,
    step_rot_rad: float,
) -> list[np.ndarray]:
    """Return canned actions exercising translation *and* rotation.

    The first half of the steps drive Cartesian translation; the second
    half adds a growing pitch rotation so we can verify that
    ``action[3:7]`` changes while staying normalised.
    """
    xyz0 = initial[:3].copy()
    rot0 = _rot_from_wxyz(initial[3:7])
    if n_steps <= 0:
        return []

    deltas = np.array(
        [
            [0.0, 0.0, 0.05],
            [0.0, 0.05, 0.0],
            [0.05, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    actions: list[np.ndarray] = []
    for idx in range(n_steps):
        dx, dy, dz = deltas[(idx * len(deltas)) // n_steps]
        a = initial.copy()
        a[:3] = [xyz0[0] + dx, xyz0[1] + dy, xyz0[2] + dz]

        # Pitch rotation ramps up during the second half of the sequence.
        if idx >= n_steps // 2:
            frac = (idx - n_steps // 2) / max(1, n_steps - n_steps // 2 - 1)
            delta_rpy = np.array([0.0, step_rot_rad * frac, 0.0])
            rot = _apply_world_euler_delta(rot0, delta_rpy)
            a[3:7] = _wxyz_from_rot(rot)

        actions.append(a)
    return actions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--viewer", action="store_true",
                   help="Launch MuJoCo interactive viewer.")
    p.add_argument("--smoke", action="store_true",
                   help="Run deterministic scripted motions without keyboard input.")
    p.add_argument("--steps", type=int, default=200,
                   help="Maximum environment steps (default: 200).")
    p.add_argument("--step-xyz", type=float, default=0.02,
                   help="Translation increment per key-press in metres (default: 0.02).")
    p.add_argument("--step-rot", type=float, default=5.0,
                   help="Rotation increment per key-press in degrees (default: 5.0).")
    p.add_argument("--image-observations", action="store_true",
                   help="Render images in observation (default: off).")
    p.add_argument("--control-dt", type=float, default=0.01,
                   help="Control timestep in seconds (default: 0.01).")
    p.add_argument("--physics-dt", type=float, default=0.002,
                   help="Physics timestep in seconds (default: 0.002).")
    p.add_argument("--benchmark", action="store_true",
                   help="Print average step Hz at end of run.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    step_rot_rad = np.deg2rad(args.step_rot)

    render_mode = "rgb_array" if args.image_observations else "none"
    env = Cr3CraftClickMouseShellEnv(
        seed=args.seed,
        render_mode=render_mode,
        control_dt=args.control_dt,
        physics_dt=args.physics_dt,
    )
    viewer = None
    # Only load cv2 when we actually need it (interactive mode).
    cv2 = None

    try:
        obs, info = env.reset()
        initial_action = env.get_initial_action()
        if initial_action.shape != (22,):
            raise RuntimeError(
                f"Expected initial action shape (22,), got {initial_action.shape}"
            )

        action = initial_action.copy()
        home_xyz = initial_action[:3].copy()
        home_quat = initial_action[3:7].copy()

        # Internal rotation state kept as a scipy Rotation.
        target_rot = _rot_from_wxyz(home_quat)

        print(f"Home TCP xyz:   {np.array2string(home_xyz, precision=4)}")
        print(f"Home TCP quat:  {np.array2string(home_quat, precision=4)} (wxyz)")
        print(f"Home TCP rpy:   {target_rot.as_euler('xyz', degrees=True)} deg")
        print(f"Action dim:     {action.shape[0]}")
        print(f"Step XYZ: {args.step_xyz:.3f} m  |  Step ROT: {args.step_rot:.1f} deg")

        # ------------------------------------------------------------------
        # Viewer (optional)
        # ------------------------------------------------------------------
        if args.viewer:
            import mujoco.viewer as _mv
            viewer = _mv.launch_passive(env.model, env.data)

        # ------------------------------------------------------------------
        # Smoke mode
        # ------------------------------------------------------------------
        if args.smoke:
            actions = _build_smoke_actions(initial_action, args.steps, step_rot_rad)
            print(f"Smoke mode: {len(actions)} scripted steps.\n")
            t0 = time.perf_counter()
            last_info: dict = {}
            for idx, act in enumerate(actions):
                _, _, terminated, truncated, last_info = env.step(act)
                if viewer is not None:
                    if not viewer.is_running():
                        print("[INFO] Viewer closed, exiting smoke.")
                        break
                    viewer.sync()
                if idx % max(1, len(actions) // 8) == 0 or idx == len(actions) - 1:
                    tcp = env.data.site_xpos[env._tcp_site_id]
                    quat = act[3:7]
                    qnorm = float(np.linalg.norm(quat))
                    rot = _rot_from_wxyz(quat)
                    rpy = rot.as_euler("xyz", degrees=True)
                    print(
                        f"  step={idx:4d}  "
                        f"target=({act[0]:.3f},{act[1]:.3f},{act[2]:.3f})  "
                        f"quat_wxyz=({quat[0]:.4f},{quat[1]:.4f},{quat[2]:.4f},{quat[3]:.4f})  "
                        f"|q|={qnorm:.6f}  "
                        f"rpy=({rpy[0]:.1f},{rpy[1]:.1f},{rpy[2]:.1f}) deg  "
                        f"tcp=({tcp[0]:.3f},{tcp[1]:.3f},{tcp[2]:.3f})  "
                        f"err={last_info.get('tcp_error', float('nan')):.4f}"
                    )
                if terminated or truncated:
                    break
            elapsed = time.perf_counter() - t0
            n_done = idx + 1
            tcp_final = env.data.site_xpos[env._tcp_site_id]
            print(
                f"\nSmoke done.  "
                f"final_tcp=({tcp_final[0]:.3f},{tcp_final[1]:.3f},{tcp_final[2]:.3f})  "
                f"tcp_error={last_info.get('tcp_error', float('nan')):.4f}"
            )
            print(
                f"  steps={n_done}  elapsed={elapsed:.3f}s  "
                f"avg_hz={n_done / elapsed:.1f}"
            )
            return

        # ------------------------------------------------------------------
        # Interactive keyboard loop
        # ------------------------------------------------------------------
        import cv2

        win_name = "Keyboard Control (Esc=quit)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        # Build a small control canvas with key instructions.
        _CANVAS_H, _CANVAS_W = 420, 500
        canvas = np.zeros((_CANVAS_H, _CANVAS_W, 3), dtype=np.uint8)
        _instructions = [
            "Numpad: 8/2=Z  4/6=Y  7/9=X",
            "        1/3=RX  0/.=RY  -/+=RZ  5=home",
            "WASD+:  W/S=Z  A/D=Y  Q/E=X",
            "        R/F=RX  T/G=RY  Y/U=RZ",
            "C=close hand  O=open hand  H=home",
            "Esc=quit",
        ]
        for i, line in enumerate(_instructions):
            cv2.putText(canvas, line, (10, 30 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win_name, canvas)

        step_xyz = args.step_xyz
        step_count = 0
        last_print = time.time()
        t_start = time.perf_counter()
        last_hz_print = time.perf_counter()
        last_hz_steps = 0
        last_info: dict = {}

        print()
        print("=" * 60)
        print("Numpad: 8/2=Z  4/6=Y  7/9=X  1/3=RX  0/.=RY  -/+=RZ  5=home")
        print("WASD+:  W/S=Z  A/D=Y  Q/E=X  R/F=RX  T/G=RY  Y/U=RZ")
        print("Other:  C=close hand  O=open hand  H=home  Esc=quit")
        print("(click the OpenCV 'Keyboard Control' window and press keys)")
        print(f"Controls: dt={args.control_dt}s  phys_dt={args.physics_dt}s  render_mode={render_mode}")
        print("=" * 60)

        while step_count < args.steps:
            k = cv2.waitKeyEx(1)

            if k == 27:                     # Esc
                break
            elif k in (ord("h"), ord("H"), *_numpad_digit_codes("5")):
                action[:3] = home_xyz
                action[3:7] = home_quat
                target_rot = _rot_from_wxyz(home_quat)
                print("\n[HOME]")
            elif k in (ord("c"), ord("C")):  # close hand
                action[7:22] = CRAFT_VALUE_MAX * 0.7
                print("\n[CLOSE HAND]")
            elif k in (ord("o"), ord("O")):  # open hand
                action[7:22] = 0.0
                print("\n[OPEN HAND]")
            elif k in _TRANS_KEY_MAP:
                axis, sign = _TRANS_KEY_MAP[k]
                action[axis] += sign * step_xyz
                name = _AXIS_NAME.get(axis, f"a{axis}")
                print(
                    f"\r  {name}{'+' if sign > 0 else '-'}  "
                    f"target=({action[0]:.3f},{action[1]:.3f},{action[2]:.3f})",
                    end="",
                )
            elif k in _ROT_KEY_MAP:
                axis, sign = _ROT_KEY_MAP[k]
                delta_rpy = np.zeros(3)
                delta_rpy[axis - 3] = sign * step_rot_rad  # axis 3→roll, 4→pitch, 5→yaw
                target_rot = _apply_world_euler_delta(target_rot, delta_rpy)
                action[3:7] = _wxyz_from_rot(target_rot)
                name = _AXIS_NAME.get(axis, f"a{axis}")
                rpy = target_rot.as_euler("xyz", degrees=True)
                q = action[3:7]
                print(
                    f"\r  {name}{'+' if sign > 0 else '-'}  "
                    f"rpy=({rpy[0]:.1f},{rpy[1]:.1f},{rpy[2]:.1f}) deg  "
                    f"quat=({q[0]:.3f},{q[1]:.3f},{q[2]:.3f},{q[3]:.3f})",
                    end="",
                )

            cv2.imshow(win_name, canvas)

            _, _, terminated, truncated, last_info = env.step(action)

            if viewer is not None:
                if not viewer.is_running():
                    print("\n[INFO] Viewer closed, exiting.")
                    break
                viewer.sync()

            step_count += 1
            if time.time() - last_print > 0.5:
                tcp = env.data.site_xpos[env._tcp_site_id]
                rpy = target_rot.as_euler("xyz", degrees=True)
                print(
                    f"\n  S{step_count:4d}  "
                    f"target=({action[0]:.3f},{action[1]:.3f},{action[2]:.3f})  "
                    f"rpy=({rpy[0]:.1f},{rpy[1]:.1f},{rpy[2]:.1f}) deg  "
                    f"tcp=({tcp[0]:.3f},{tcp[1]:.3f},{tcp[2]:.3f})  "
                    f"err={last_info.get('tcp_error', float('nan')):.4f}"
                )
                last_print = time.time()

            if time.perf_counter() - last_hz_print > 1.0:
                now = time.perf_counter()
                delta = now - last_hz_print
                delta_steps = step_count - last_hz_steps
                print(
                    f"  [Hz] {delta_steps / delta:.1f} steps/s  "
                    f"({step_count} total, {now - t_start:.1f}s elapsed)"
                )
                last_hz_print = now
                last_hz_steps = step_count

            if terminated or truncated:
                break

        t_elapsed = time.perf_counter() - t_start
        tcp_final = env.data.site_xpos[env._tcp_site_id]
        print(
            f"\nDone: steps={step_count}  elapsed={t_elapsed:.3f}s  "
            f"avg_hz={step_count / t_elapsed:.1f}"
        )
        print(
            f"  final_tcp=({tcp_final[0]:.3f},{tcp_final[1]:.3f},{tcp_final[2]:.3f})  "
            f"tcp_error={last_info.get('tcp_error', float('nan')):.4f}"
        )

    finally:
        if viewer is not None:
            viewer.close()
        if cv2 is not None:
            cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
