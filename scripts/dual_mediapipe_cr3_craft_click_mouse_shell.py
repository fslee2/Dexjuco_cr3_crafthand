"""Dual-camera MediaPipe CR3+CRAFT teleop for the click_mouse shell env.

This script avoids monocular depth estimation.  It uses two ordinary webcams:

  front camera: palm left/right/up/down -> TCP Y/Z, fingers -> CRAFT hand
  side camera:  palm horizontal motion  -> TCP X

It is not calibrated stereo reconstruction.  It is a pragmatic two-view 2D
teleop mapping intended for quick feasibility testing when no depth camera is
available.

Example:
  python scripts/dual_mediapipe_cr3_craft_click_mouse_shell.py \
    --front-camera-id 0 \
    --side-camera-id 1 \
    --viewer \
    --preview
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "glfw" if os.name == "nt" else "egl")

import cv2
import mujoco
import mujoco.viewer
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_DEXJOCO = _REPO / "dexjoco"
if str(_DEXJOCO) not in sys.path:
    sys.path.insert(0, str(_DEXJOCO))

from dexjoco.sim.envs.cr3_craft_click_mouse_shell_env import Cr3CraftClickMouseShellEnv

from mediapipe_cr3_craft_click_mouse_shell import (
    CALIBRATION_FRAMES,
    FILTER_ALPHA,
    FILTER_MAX_STEP,
    HAND_CURL_GAIN,
    HAND_FILTER_ALPHA,
    HAND_MAX_STEP,
    HAND_SIDE_GAIN,
    LOG_INTERVAL,
    LOST_RECALIBRATE_STEPS,
    TARGET_UPDATE_DEADBAND,
    WORKSPACE_DELTA,
    LatestFrameCamera,
    _clamp_step,
    _craft_action_from_landmarks,
    _craft_open_action_from_env,
    _create_hand_detector,
    _detect_landmarks,
    _finish_calibration,
    _handle_keyboard_x,
    _install_fast_observation,
    _make_key_canvas,
    _measurement_from_landmarks,
    _new_calibration_buffer,
    _open_camera,
    _palm_offset_fraction,
    _side_neutral_action_from_env,
    _update_calibration_buffer,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--front-camera-id", "--camera-id", dest="front_camera_id", type=int, default=0)
    parser.add_argument("--side-camera-id", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--keyboard-x-step", type=float, default=0.01)
    parser.add_argument("--no-keyboard-x", action="store_true")
    parser.add_argument("--side-x-gain", type=float, default=float(WORKSPACE_DELTA[0]))
    parser.add_argument("--side-x-sign", type=float, default=1.0, help="Use -1 if side-camera X direction is reversed.")
    parser.add_argument("--front-y-gain", type=float, default=float(WORKSPACE_DELTA[1]))
    parser.add_argument("--front-z-gain", type=float, default=float(WORKSPACE_DELTA[2]))
    return parser.parse_args()


def _draw_frame(name, frame, keypoints, lines):
    if frame is None:
        return
    if keypoints is not None:
        for x, y in keypoints:
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
    for idx, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (10, 24 + idx * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (30, 230, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imshow(name, frame)


def _camera_args(camera_id: int):
    return SimpleNamespace(camera_id=int(camera_id))


def _read_flipped(camera: LatestFrameCamera):
    ok, frame = camera.read()
    if not ok or frame is None:
        return False, None
    return True, cv2.flip(frame, 1)


def main() -> None:
    args = _parse_args()
    env = Cr3CraftClickMouseShellEnv(render_mode="rgb_array")
    _install_fast_observation(env)

    front_cap = None
    side_cap = None
    front_camera = None
    side_camera = None
    front_detector = None
    side_detector = None
    viewer = None

    try:
        front_cap = _open_camera(_camera_args(args.front_camera_id))
        side_cap = _open_camera(_camera_args(args.side_camera_id))
        front_camera = LatestFrameCamera(front_cap)
        side_camera = LatestFrameCamera(side_cap)
        front_camera.start()
        side_camera.start()

        front_mode, front_mp, front_detector = _create_hand_detector()
        side_mode, side_mp, side_detector = _create_hand_detector()

        env.reset()
        action0 = env.get_initial_action()
        action = action0.copy()
        home_xyz = action0[:3].copy()
        hold_quat = action0[3:7].copy()

        side_neutral = _side_neutral_action_from_env(env)
        hold_craft = _craft_open_action_from_env(env)
        filtered_craft = hold_craft.copy()

        target_xyz = home_xyz.copy()
        filtered_xyz = home_xyz.copy()

        front_center0 = None
        front_side0 = None
        front_calibrating = True
        front_buffer = _new_calibration_buffer()
        front_lost_count = 0

        side_center0 = None
        side_calibrating = True
        side_buffer = _new_calibration_buffer()
        side_lost_count = 0
        side_x_offset = 0.0

        manual_x_offset = 0.0
        keyboard_x_enabled = not args.no_keyboard_x
        key_canvas = None if args.preview or not keyboard_x_enabled else _make_key_canvas()

        last_t = time.perf_counter()

        if args.viewer:
            viewer = mujoco.viewer.launch_passive(env.model, env.data)

        print(
            "Ready. Dual-camera MediaPipe CR3 teleop. "
            "front camera: Y/Z + CRAFT, side camera: X. "
            "Keys: z/7=X-, x/9=X+, h/5=clear X, r=recalibrate, q/Esc=quit."
        )
        print(
            f"front_camera={args.front_camera_id} side_camera={args.side_camera_id} "
            f"home_xyz={np.round(home_xyz, 4)} "
            f"front_y_gain={args.front_y_gain:.3f} front_z_gain={args.front_z_gain:.3f} "
            f"side_x_gain={args.side_x_gain:.3f} side_x_sign={args.side_x_sign:+.1f}"
        )

        for step in range(args.steps):
            key = -1
            if key_canvas is not None:
                cv2.imshow("Dual MediaPipe CR3 teleop keys", key_canvas)
                key = cv2.waitKeyEx(1)

            front_ok, front_frame = _read_flipped(front_camera)
            side_ok, side_frame = _read_flipped(side_camera)
            if not front_ok:
                print(f"[WARN] front camera read failed at step {step}")
                time.sleep(0.005)
                continue
            if not side_ok:
                print(f"[WARN] side camera read failed at step {step}")
                time.sleep(0.005)
                continue

            front_h, front_w = front_frame.shape[:2]
            side_h, side_w = side_frame.shape[:2]

            front_landmarks = _detect_landmarks(front_mode, front_mp, front_detector, front_frame)
            side_landmarks = _detect_landmarks(side_mode, side_mp, side_detector, side_frame)
            front_keypoints = None
            side_keypoints = None
            front_detected = front_landmarks is not None
            side_detected = side_landmarks is not None

            if key in (ord("q"), ord("Q"), 27):
                break
            if keyboard_x_enabled:
                manual_x_offset = _handle_keyboard_x(key, manual_x_offset, args.keyboard_x_step)
            if key in (ord("r"), ord("R")):
                front_center0 = None
                front_side0 = None
                front_calibrating = True
                front_buffer = _new_calibration_buffer()
                side_center0 = None
                side_calibrating = True
                side_buffer = _new_calibration_buffer()
                side_x_offset = 0.0
                manual_x_offset = 0.0
                home_xyz = env.data.site_xpos[env._tcp_site_id].copy()
                target_xyz = home_xyz.copy()
                filtered_xyz = home_xyz.copy()
                front_lost_count = 0
                side_lost_count = 0
                print(f"[INFO] recalibrating both cameras at home_xyz={np.round(home_xyz, 4)}")

            if front_detected:
                front_lost_count = 0
                front_center, front_size, front_keypoints = _measurement_from_landmarks(
                    front_landmarks, front_w, front_h
                )
                if front_calibrating or front_center0 is None:
                    _update_calibration_buffer(front_buffer, front_center, front_size, front_landmarks)
                    if _calibration_ready(front_buffer):
                        front_center0, _front_size0, front_side0 = _finish_calibration(front_buffer)
                        front_calibrating = False
                        filtered_craft = hold_craft.copy()
                        print(f"[INFO] front calibrated center={np.round(front_center0, 1)}")
            else:
                front_lost_count += 1
                if front_lost_count == LOST_RECALIBRATE_STEPS:
                    front_center0 = None
                    front_side0 = None
                    front_calibrating = True
                    front_buffer = _new_calibration_buffer()
                    print("[INFO] lost front hand; next detection will recalibrate front")

            if side_detected:
                side_lost_count = 0
                side_center, side_size, side_keypoints = _measurement_from_landmarks(
                    side_landmarks, side_w, side_h
                )
                if side_calibrating or side_center0 is None:
                    _update_calibration_buffer(side_buffer, side_center, side_size, side_landmarks)
                    if _calibration_ready(side_buffer):
                        side_center0, _side_size0, _side_side0 = _finish_calibration(side_buffer)
                        side_x_offset = 0.0
                        side_calibrating = False
                        print(f"[INFO] side calibrated center={np.round(side_center0, 1)}")
                else:
                    side_frac = _palm_offset_fraction(side_center, side_center0, side_w, side_h)
                    side_x_offset = float(args.side_x_sign * args.side_x_gain * side_frac[0])
            else:
                side_lost_count += 1
                if side_lost_count == LOST_RECALIBRATE_STEPS:
                    side_center0 = None
                    side_calibrating = True
                    side_buffer = _new_calibration_buffer()
                    side_x_offset = 0.0
                    print("[INFO] lost side hand; next detection will recalibrate side")

            if front_detected and not front_calibrating and front_center0 is not None:
                palm_frac = _palm_offset_fraction(front_center, front_center0, front_w, front_h)
                raw_target = home_xyz + np.asarray(
                    [
                        side_x_offset + manual_x_offset,
                        args.front_y_gain * palm_frac[0],
                        -args.front_z_gain * palm_frac[1],
                    ],
                    dtype=np.float64,
                )
                if np.linalg.norm(raw_target - target_xyz) < TARGET_UPDATE_DEADBAND:
                    raw_target = target_xyz.copy()
                filtered_xyz = filtered_xyz + FILTER_ALPHA * (raw_target - filtered_xyz)
                filtered_xyz = _clamp_step(filtered_xyz, target_xyz, FILTER_MAX_STEP)
                target_xyz = filtered_xyz.copy()

                raw_craft = _craft_action_from_landmarks(
                    front_landmarks,
                    HAND_CURL_GAIN,
                    side_neutral,
                    front_side0,
                    HAND_SIDE_GAIN,
                )
                filtered_craft = filtered_craft + HAND_FILTER_ALPHA * (raw_craft - filtered_craft)
                filtered_craft = _clamp_step(filtered_craft, action[7:22], HAND_MAX_STEP)

            action[:3] = target_xyz
            action[3:7] = hold_quat
            action[7:22] = filtered_craft
            _obs, _reward, terminated, truncated, info = env.step(action)

            now = time.perf_counter()
            fps = 1.0 / max(now - last_t, 1e-6)
            last_t = now

            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()

            if args.preview:
                _draw_frame(
                    "front MediaPipe CR3 teleop",
                    front_frame,
                    front_keypoints,
                    [
                        f"front detected={front_detected} calibrated={not front_calibrating}",
                        f"target Y/Z=({target_xyz[1]:+.3f},{target_xyz[2]:+.3f}) fps={fps:.1f}",
                    ],
                )
                _draw_frame(
                    "side MediaPipe CR3 teleop",
                    side_frame,
                    side_keypoints,
                    [
                        f"side detected={side_detected} calibrated={not side_calibrating}",
                        f"side_x={side_x_offset:+.3f} manual_x={manual_x_offset:+.3f}",
                    ],
                )
                key = cv2.waitKeyEx(1)
                if key in (ord("q"), ord("Q"), 27):
                    break
                if keyboard_x_enabled:
                    manual_x_offset = _handle_keyboard_x(key, manual_x_offset, args.keyboard_x_step)
                if key in (ord("r"), ord("R")):
                    front_center0 = None
                    front_side0 = None
                    front_calibrating = True
                    front_buffer = _new_calibration_buffer()
                    side_center0 = None
                    side_calibrating = True
                    side_buffer = _new_calibration_buffer()
                    side_x_offset = 0.0
                    manual_x_offset = 0.0
                    home_xyz = env.data.site_xpos[env._tcp_site_id].copy()
                    target_xyz = home_xyz.copy()
                    filtered_xyz = home_xyz.copy()
                    front_lost_count = 0
                    side_lost_count = 0
                    print(f"[INFO] recalibrating both cameras at home_xyz={np.round(home_xyz, 4)}")

            if LOG_INTERVAL > 0 and (step % LOG_INTERVAL == 0 or step == args.steps - 1):
                print(
                    f"step={step:4d} front={front_detected} side={side_detected} fps={fps:5.1f} "
                    f"target={np.round(target_xyz, 4)} side_x={side_x_offset:+.3f} "
                    f"manual_x={manual_x_offset:+.3f} tcp={np.round(env.data.site_xpos[env._tcp_site_id], 4)} "
                    f"craft_mean={np.mean(action[7:22]):.2f} "
                    f"tcp_error={info.get('tcp_error', float('nan')):.4f}"
                )

            if not np.all(np.isfinite(env.data.qacc)):
                print(f"[WARN] QACC NaN/Inf at step {step}")
                break
            if terminated or truncated:
                break

    finally:
        if viewer is not None:
            viewer.close()
        for detector in (front_detector, side_detector):
            if detector is not None:
                detector.close()
        for camera in (front_camera, side_camera):
            if camera is not None:
                camera.stop()
        for cap in (front_cap, side_cap):
            if cap is not None:
                cap.release()
        cv2.destroyAllWindows()
        env.close()


def _calibration_ready(buffer) -> bool:
    return len(buffer["centers"]) >= max(1, int(CALIBRATION_FRAMES))


if __name__ == "__main__":
    main()
