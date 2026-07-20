"""MediaPipe-only CR3 teleoperation smoke test for the click_mouse shell env."""

import argparse
import os
import sys
import threading
import time
import types
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw" if os.name == "nt" else "egl")

import cv2
import mujoco
import mujoco.viewer
import numpy as np

_repo = Path(__file__).resolve().parents[1]
_dexjoco = _repo / "dexjoco"
sys.path.insert(0, str(_dexjoco))

from dexjoco.sim.envs.cr3_craft_click_mouse_shell_env import (
    CRAFT_VALUE_MAX,
    Cr3CraftClickMouseShellEnv,
)

CAMERA_WIDTH = 424
CAMERA_HEIGHT = 240
CAMERA_FPS = 30
WORKSPACE_DELTA = np.asarray((0.10, 0.14, 0.10), dtype=np.float64)
PALM_CENTER_DEADBAND = 0.07
Z_DEADBAND = 0.05
FILTER_ALPHA = 0.65
FILTER_MAX_STEP = 0.05
TARGET_UPDATE_DEADBAND = 0.008
CALIBRATION_FRAMES = 12
LOST_RECALIBRATE_STEPS = 8
HAND_FILTER_ALPHA = 0.7
HAND_MAX_STEP = 0.9
HAND_CURL_GAIN = 1.8
HAND_SIDE_GAIN = 0.8
LOG_INTERVAL = 30


FINGERS = {
    "index": {"mcp": 5, "pip": 6, "dip": 7, "tip": 8},
    "middle": {"mcp": 9, "pip": 10, "dip": 11, "tip": 12},
    "ring": {"mcp": 13, "pip": 14, "dip": 15, "tip": 16},
    "pinky": {"mcp": 17, "pip": 18, "dip": 19, "tip": 20},
    "thumb": {"cmc": 1, "mcp": 2, "ip": 3, "tip": 4},
}


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Common use: python scripts/mediapipe_cr3_craft_click_mouse_shell.py "
            "--camera-id 0 --viewer"
        ),
    )
    parser.add_argument("--camera-id", "--device-index", dest="camera_id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--keyboard-x-step", type=float, default=0.01,
                        help="Manual TCP X increment per key press in metres.")
    parser.add_argument("--no-keyboard-x", action="store_true",
                        help="Disable keyboard X offset controls.")
    return parser.parse_args()


def _install_fast_observation(env):
    image_shape = env.observation_space["images"]["front"].shape
    zero_image = np.zeros(image_shape, dtype=np.uint8)

    def fast_observation(self):
        return {
            "state": {
                "tcp_pose": self._tcp_pose(),
                "arm_qpos": self.data.qpos[self._arm_qpos_idx].copy(),
                "craft_qpos": self._craft_state(),
                "target_tcp_pose": np.concatenate((self._target_xyz.copy(), self._target_quat.copy())),
            },
            "images": {"front": zero_image},
        }

    env._observation = types.MethodType(fast_observation, env)


def _open_camera(args):
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera_id}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class LatestFrameCamera:
    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            ok, frame = self.read()
            if ok and frame is not None:
                return
            time.sleep(0.01)

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            with self.lock:
                self.ok = bool(ok)
                if ok:
                    self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ok, self.frame.copy()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)


def _create_hand_detector():
    import mediapipe as mp

    if hasattr(mp, "solutions"):
        return "solutions", mp, mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        )

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    candidates = [
        _repo / "hand_landmarker.task",
        _repo.parent / "ros2_ws" / "mujoco_ws" / "hand_landmarker.task",
        _repo.parent / "mujoco_ws" / "hand_landmarker.task",
        Path.cwd() / "hand_landmarker.task",
    ]
    model_path = next((path for path in candidates if path.exists()), None)
    if model_path is None:
        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Could not find hand_landmarker.task. Searched: {searched}")

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        num_hands=1,
        min_hand_detection_confidence=0.45,
        min_tracking_confidence=0.45,
    )
    return "tasks", mp, vision.HandLandmarker.create_from_options(options)


def _detect_landmarks(mode, mp, detector, frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if mode == "solutions":
        result = detector.process(rgb)
        if not result.multi_hand_landmarks:
            return None
        return result.multi_hand_landmarks[0].landmark

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    if not result.hand_landmarks:
        return None
    return result.hand_landmarks[0]


def _measurement_from_landmarks(landmarks, width, height):
    keypoints = np.asarray([[lm.x * width, lm.y * height] for lm in landmarks], dtype=np.float64)
    palm_ids = [0, 5, 9, 13, 17]
    center = keypoints[palm_ids].mean(axis=0)
    x1, y1 = keypoints[:, 0].min(), keypoints[:, 1].min()
    x2, y2 = keypoints[:, 0].max(), keypoints[:, 1].max()
    size = max(float(x2 - x1), float(y2 - y1), 1.0)
    return center, size, keypoints


def _landmark_xyz(landmarks):
    return {idx: np.asarray((lm.x, lm.y, lm.z), dtype=np.float64) for idx, lm in enumerate(landmarks)}


def _joint_angle(a, b, c):
    ba = np.linalg.norm(a - b)
    bc = np.linalg.norm(c - b)
    ac = np.linalg.norm(c - a)
    cos_angle = (ba * ba + bc * bc - ac * ac) / (2.0 * ba * bc + 1e-10)
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def _normalize(vec):
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.zeros_like(vec)
    return vec / norm


def _curl_fraction(angle, open_angle=0.08, closed_angle=1.2):
    return float(np.clip((angle - open_angle) / max(closed_angle - open_angle, 1e-6), 0.0, 1.0))


def _finger_curls_from_landmarks(landmarks):
    pts = _landmark_xyz(landmarks)
    curls = {}
    for finger, ids in FINGERS.items():
        if finger == "thumb":
            mcp = np.pi - _joint_angle(pts[ids["cmc"]], pts[ids["mcp"]], pts[ids["ip"]])
            ip = np.pi - _joint_angle(pts[ids["mcp"]], pts[ids["ip"]], pts[ids["tip"]])
            curls[finger] = (
                _curl_fraction(ip, open_angle=0.05, closed_angle=0.95),
                _curl_fraction(mcp, open_angle=0.05, closed_angle=0.9),
            )
        else:
            pip = np.pi - _joint_angle(pts[ids["mcp"]], pts[ids["pip"]], pts[ids["dip"]])
            dip = np.pi - _joint_angle(pts[ids["pip"]], pts[ids["dip"]], pts[ids["tip"]])
            curls[finger] = (
                _curl_fraction(0.65 * pip + 0.35 * dip),
                _curl_fraction(pip),
            )
    return curls


def _finger_sides_from_landmarks(landmarks):
    pts = _landmark_xyz(landmarks)
    knuckle_axis = _normalize(pts[17] - pts[5])
    if not np.any(knuckle_axis):
        knuckle_axis = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)

    sides = {}
    for finger in ("index", "middle", "ring", "pinky"):
        ids = FINGERS[finger]
        direction = _normalize(pts[ids["pip"]] - pts[ids["mcp"]])
        sides[finger] = float(np.dot(direction, knuckle_axis))

    thumb_ids = FINGERS["thumb"]
    thumb_direction = _normalize(pts[thumb_ids["tip"]] - pts[thumb_ids["mcp"]])
    sides["thumb"] = float(np.dot(thumb_direction, knuckle_axis))
    return sides


def _side_neutral_action_from_env(env):
    neutral = {}
    side_joints = {
        "ring": "ring_1",
        "index": "index_1",
        "thumb": "thumb_mcp",
        "middle": "middle_1",
        "pinky": "pinky_1",
    }
    for finger, joint_name in side_joints.items():
        jid = env.model.joint(joint_name).id
        low, high = env.model.jnt_range[jid]
        percent = 0.5 if high <= low else np.clip((0.0 - low) / (high - low), 0.0, 1.0)
        neutral[finger] = float(percent * CRAFT_VALUE_MAX)
    return neutral


def _craft_open_action_from_env(env):
    side_neutral = _side_neutral_action_from_env(env)
    values = []
    for finger in ("ring", "index", "thumb", "middle", "pinky"):
        values.extend((0.0, 0.0, side_neutral[finger]))
    return np.asarray(values, dtype=np.float64)


def _craft_action_from_landmarks(landmarks, curl_gain, side_neutral, side_zero=None, side_gain=0.0):
    curls = _finger_curls_from_landmarks(landmarks)
    sides = _finger_sides_from_landmarks(landmarks)

    def scale(value):
        return CRAFT_VALUE_MAX * float(np.clip(curl_gain * value, 0.0, 1.0))

    # CRAFT direct order:
    # Ring, Index, Thumb, Middle, Pinky, each [PIP, MCP forward, MCP side].
    values = []
    for finger in ("ring", "index", "thumb", "middle", "pinky"):
        pip_curl, mcp_curl = curls[finger]
        side_value = side_neutral[finger]
        if side_zero is not None:
            side_value += side_gain * (sides[finger] - side_zero[finger]) * CRAFT_VALUE_MAX
            side_value = float(np.clip(side_value, 0.0, CRAFT_VALUE_MAX))
        values.extend((scale(pip_curl), scale(mcp_curl), side_value))
    return np.asarray(values, dtype=np.float64)


def _clamp_step(target, previous, max_step):
    step = target - previous
    norm = float(np.linalg.norm(step))
    if max_step > 0.0 and norm > max_step:
        return previous + step / norm * max_step
    return target


def _apply_deadband(value, deadband):
    if abs(value) <= deadband:
        return 0.0
    return float(np.sign(value) * (abs(value) - deadband))


def _palm_offset_fraction(center, center0, width, height):
    rel_x = (center[0] - center0[0]) / max(0.5 * width, 1.0)
    rel_y = (center[1] - center0[1]) / max(0.5 * height, 1.0)
    rel_x = _apply_deadband(rel_x, PALM_CENTER_DEADBAND)
    rel_y = _apply_deadband(rel_y, PALM_CENTER_DEADBAND)
    scale = max(1.0 - PALM_CENTER_DEADBAND, 1e-6)
    return np.clip(np.asarray((rel_x / scale, rel_y / scale), dtype=np.float64), -1.0, 1.0)


def _depth_offset_fraction(size, size0):
    if size0 is None or size0 <= 0:
        return 0.0
    ratio = size / size0
    rel = 1.0 - ratio
    rel = _apply_deadband(rel, Z_DEADBAND)
    scale = max(1.0 - Z_DEADBAND, 1e-6)
    return float(np.clip(rel / scale, -1.0, 1.0))


def _draw_preview(frame, keypoints, target, tcp_error, fps):
    if keypoints is not None:
        for x, y in keypoints:
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
    text = f"target=({target[0]:+.2f},{target[1]:+.2f},{target[2]:+.2f}) err={tcp_error:.3f} fps={fps:.1f}"
    cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 230, 255), 1, cv2.LINE_AA)
    cv2.imshow("MediaPipe CR3 teleop", frame)
    return cv2.waitKeyEx(1)


def _numpad_codes(char):
    codes = [ord(char)]
    if char.isdigit():
        codes.append(0x60 + int(char))
    return codes


KEY_X_MINUS = set([ord("z"), ord("Z"), 2359296, *_numpad_codes("7")])
KEY_X_PLUS = set([ord("x"), ord("X"), 2162688, *_numpad_codes("9")])
KEY_X_HOME = set([ord("h"), ord("H"), *_numpad_codes("5")])


def _make_key_canvas():
    canvas = np.zeros((150, 560, 3), dtype=np.uint8)
    lines = [
        "MediaPipe hand teleop + keyboard X",
        "hand: palm left/right/up/down/near-far -> TCP Y/Z/X, fingers -> CRAFT",
        "keyboard: Z/7=X-   X/9=X+   H/5=clear X   R=recalibrate   Q/Esc=quit",
    ]
    for idx, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (12, 30 + idx * 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _handle_keyboard_x(key, manual_x_offset, step):
    if key in KEY_X_MINUS:
        manual_x_offset -= step
        print(f"[KEY X-] manual_x_offset={manual_x_offset:+.4f}")
    elif key in KEY_X_PLUS:
        manual_x_offset += step
        print(f"[KEY X+] manual_x_offset={manual_x_offset:+.4f}")
    elif key in KEY_X_HOME:
        manual_x_offset = 0.0
        print("[KEY X HOME] manual_x_offset=+0.0000")
    return manual_x_offset


def _with_manual_x(target_xyz, home_xyz, manual_x_offset):
    out = np.asarray(target_xyz, dtype=np.float64).copy()
    out[0] = float(home_xyz[0] + manual_x_offset)
    return out


def _new_calibration_buffer():
    return {"centers": [], "sizes": [], "sides": []}


def _update_calibration_buffer(buffer, center, size, landmarks):
    buffer["centers"].append(np.asarray(center, dtype=np.float64))
    buffer["sizes"].append(float(size))
    buffer["sides"].append(_finger_sides_from_landmarks(landmarks))


def _calibration_ready(buffer, frames):
    return len(buffer["centers"]) >= max(1, int(frames))


def _finish_calibration(buffer):
    center0 = np.mean(np.asarray(buffer["centers"], dtype=np.float64), axis=0)
    size0 = float(np.mean(np.asarray(buffer["sizes"], dtype=np.float64)))
    side0 = {
        finger: float(np.mean([side[finger] for side in buffer["sides"]]))
        for finger in ("index", "middle", "ring", "pinky", "thumb")
    }
    return center0, size0, side0


def main():
    args = _parse_args()
    env = Cr3CraftClickMouseShellEnv(render_mode="rgb_array")
    _install_fast_observation(env)

    cap = None
    camera = None
    detector = None
    viewer = None
    try:
        cap = _open_camera(args)
        camera = LatestFrameCamera(cap)
        camera.start()
        mode, mp, detector = _create_hand_detector()
        obs, info = env.reset()
        action0 = env.get_initial_action()
        action = action0.copy()
        home_xyz = action0[:3].copy()
        hold_quat = action0[3:7].copy()
        side_neutral = _side_neutral_action_from_env(env)
        hold_craft = _craft_open_action_from_env(env)
        filtered_craft = hold_craft.copy()
        target_xyz = home_xyz.copy()
        filtered_xyz = home_xyz.copy()
        center0 = None
        size0 = None
        side0 = None
        calibrating = True
        calibration = _new_calibration_buffer()
        lost_count = 0
        last_t = time.perf_counter()
        manual_x_offset = 0.0
        keyboard_x_enabled = not args.no_keyboard_x
        key_canvas = None if args.preview or not keyboard_x_enabled else _make_key_canvas()

        if args.viewer:
            viewer = mujoco.viewer.launch_passive(env.model, env.data)

        print(
            "Ready. MediaPipe-only CR3 teleop. "
            "Move palm left/right/up/down/near-far to move the TCP in Y/Z/X. "
            "Keys: z/7=X-, x/9=X+, h/5=clear X, r=recalibrate, q/Esc=quit."
        )
        print(
            f"home_xyz={np.round(home_xyz, 4)} "
            f"workspace_delta={np.round(WORKSPACE_DELTA, 4)} "
            f"calibration_frames={CALIBRATION_FRAMES}"
        )

        for step in range(args.steps):
            key = -1
            if key_canvas is not None:
                cv2.imshow("MediaPipe CR3 teleop keys", key_canvas)
                key = cv2.waitKeyEx(1)
                if key in (ord("q"), ord("Q"), 27):
                    break
                manual_x_offset = _handle_keyboard_x(key, manual_x_offset, args.keyboard_x_step)
                if key in (ord("r"), ord("R")):
                    center0 = None
                    size0 = None
                    side0 = None
                    calibrating = True
                    calibration = _new_calibration_buffer()
                    home_xyz = env.data.site_xpos[env._tcp_site_id].copy()
                    target_xyz = home_xyz.copy()
                    filtered_xyz = home_xyz.copy()
                    manual_x_offset = 0.0
                    lost_count = 0
                    print(f"[INFO] recalibrated home_xyz={np.round(home_xyz, 4)}")

            ok, frame = camera.read() if camera is not None else cap.read()
            if not ok:
                print(f"[WARN] camera read failed at step {step}")
                time.sleep(0.005)
                continue
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            landmarks = _detect_landmarks(mode, mp, detector, frame)
            keypoints = None
            detected = landmarks is not None
            if detected:
                lost_count = 0
                center, size, keypoints = _measurement_from_landmarks(landmarks, width, height)
                if calibrating or center0 is None:
                    _update_calibration_buffer(calibration, center, size, landmarks)
                    if _calibration_ready(calibration, CALIBRATION_FRAMES):
                        center0, size0, side0 = _finish_calibration(calibration)
                        filtered_xyz = env.data.site_xpos[env._tcp_site_id].copy()
                        home_xyz = filtered_xyz.copy()
                        target_xyz = filtered_xyz.copy()
                        manual_x_offset = 0.0
                        filtered_craft = hold_craft.copy()
                        action[7:22] = hold_craft
                        calibrating = False
                        print(
                            f"[INFO] MediaPipe calibrated center={np.round(center0, 1)} "
                            f"size={size0:.1f} home_xyz={np.round(home_xyz, 4)}"
                        )
                    else:
                        target_xyz = home_xyz.copy()
                        filtered_xyz = home_xyz.copy()
                        action_xyz = _with_manual_x(target_xyz, home_xyz, manual_x_offset)
                        action[:3] = action_xyz
                        action[3:7] = hold_quat
                        action[7:22] = hold_craft
                        obs, reward, terminated, truncated, info = env.step(action)
                        if viewer is not None:
                            if not viewer.is_running():
                                break
                            viewer.sync()
                        if args.preview:
                            key = _draw_preview(frame, keypoints, action_xyz, info.get("tcp_error", float("nan")), 0.0)
                            if key in (ord("q"), ord("Q"), 27):
                                break
                            manual_x_offset = _handle_keyboard_x(key, manual_x_offset, args.keyboard_x_step)
                        continue

                palm_frac = _palm_offset_fraction(center, center0, width, height)
                depth_frac = _depth_offset_fraction(size, size0)
                delta = np.asarray(
                    [WORKSPACE_DELTA[0] * depth_frac, WORKSPACE_DELTA[1] * palm_frac[0], -WORKSPACE_DELTA[2] * palm_frac[1]],
                    dtype=np.float64,
                )
                raw_target = home_xyz + delta
                if np.linalg.norm(raw_target - target_xyz) < TARGET_UPDATE_DEADBAND:
                    raw_target = target_xyz.copy()
                filtered_xyz = filtered_xyz + FILTER_ALPHA * (raw_target - filtered_xyz)
                filtered_xyz = _clamp_step(filtered_xyz, target_xyz, FILTER_MAX_STEP)
                target_xyz = filtered_xyz.copy()

                raw_craft = _craft_action_from_landmarks(
                    landmarks,
                    HAND_CURL_GAIN,
                    side_neutral,
                    side0,
                    HAND_SIDE_GAIN,
                )
                filtered_craft = filtered_craft + HAND_FILTER_ALPHA * (raw_craft - filtered_craft)
                filtered_craft = _clamp_step(filtered_craft, action[7:22], HAND_MAX_STEP)
            else:
                lost_count += 1
                if lost_count == LOST_RECALIBRATE_STEPS:
                    center0 = None
                    size0 = None
                    side0 = None
                    calibrating = True
                    calibration = _new_calibration_buffer()
                    home_xyz = env.data.site_xpos[env._tcp_site_id].copy()
                    target_xyz = home_xyz.copy()
                    filtered_xyz = home_xyz.copy()
                    manual_x_offset = 0.0
                    print(f"[INFO] lost hand; next detection will recalibrate at {np.round(home_xyz, 4)}")

            action_xyz = _with_manual_x(target_xyz, home_xyz, manual_x_offset)
            action[:3] = action_xyz
            action[3:7] = hold_quat
            action[7:22] = filtered_craft
            obs, reward, terminated, truncated, info = env.step(action)

            now = time.perf_counter()
            fps = 1.0 / max(now - last_t, 1e-6)
            last_t = now

            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()

            if args.preview:
                key = _draw_preview(frame, keypoints, action_xyz, info.get("tcp_error", float("nan")), fps)
                if key in (ord("q"), ord("Q"), 27):
                    break
                manual_x_offset = _handle_keyboard_x(key, manual_x_offset, args.keyboard_x_step)
                if key in (ord("r"), ord("R")):
                    center0 = None
                    size0 = None
                    side0 = None
                    calibrating = True
                    calibration = _new_calibration_buffer()
                    home_xyz = env.data.site_xpos[env._tcp_site_id].copy()
                    target_xyz = home_xyz.copy()
                    filtered_xyz = home_xyz.copy()
                    manual_x_offset = 0.0
                    lost_count = 0
                    print(f"[INFO] recalibrated home_xyz={np.round(home_xyz, 4)}")

            if LOG_INTERVAL > 0 and (step % LOG_INTERVAL == 0 or step == args.steps - 1):
                print(
                    f"step={step:4d} detected={detected} fps={fps:5.1f} "
                    f"target={np.round(action_xyz, 4)} "
                    f"x_offset={manual_x_offset:+.3f} "
                    f"tcp={np.round(env.data.site_xpos[env._tcp_site_id], 4)} "
                    f"craft_mean={np.mean(action[7:22]):.2f} "
                    f"tcp_error={info.get('tcp_error', float('nan')):.4f} "
                    f"ncon={info.get('ncon', 0)}"
                )

            if not np.all(np.isfinite(env.data.qacc)):
                print(f"[WARN] QACC NaN/Inf at step {step}")
                break
            if terminated or truncated:
                break

    finally:
        if viewer is not None:
            viewer.close()
        if detector is not None:
            detector.close()
        if camera is not None:
            camera.stop()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
