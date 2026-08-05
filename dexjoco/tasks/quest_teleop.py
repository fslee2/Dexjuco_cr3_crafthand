"""Quest 3 WebXR teleoperation bridge for single-arm DexJoCo tasks.

Convention: quaternions are named explicitly.
  - xyzw: scalar-last  (x, y, z, w), scipy.spatial.transform.Rotation.as_quat() format.
  - wxyz: scalar-first (w, x, y, z), DexJoCo action order.
"""

from __future__ import annotations

import json
import ssl
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------


@dataclass
class QuestTeleopState:
    timestamp: float = 0.0
    connected: bool = False
    valid: bool = False
    right_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    right_quat_xyzw: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    )
    left_pos: Optional[np.ndarray] = None
    left_quat_xyzw: Optional[np.ndarray] = None
    trigger: float = 0.0
    grip: float = 0.0
    pinch: float = 0.0
    source: str = ""

    @property
    def right_quat_wxyz(self) -> np.ndarray:
        xyzw = self.right_quat_xyzw
        return np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)

    @property
    def right_rot(self) -> R:
        return R.from_quat(self.right_quat_xyzw)

    @property
    def close_scalar(self) -> float:
        return float(max(self.trigger, self.grip, self.pinch))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "connected": self.connected,
            "valid": self.valid,
            "right_pos": self.right_pos.tolist(),
            "right_quat_xyzw": self.right_quat_xyzw.tolist(),
            "left_pos": self.left_pos.tolist() if self.left_pos is not None else None,
            "left_quat_xyzw": self.left_quat_xyzw.tolist() if self.left_quat_xyzw is not None else None,
            "trigger": self.trigger,
            "grip": self.grip,
            "pinch": self.pinch,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Thread-safe state store
# ---------------------------------------------------------------------------


class QuestStateStore:
    """Thread-safe store for the latest Quest teleop state."""

    def __init__(self):
        self._lock = threading.Lock()
        self._state = QuestTeleopState()

    def update(self, state: QuestTeleopState) -> None:
        with self._lock:
            self._state = state

    def get(self) -> QuestTeleopState:
        with self._lock:
            return self._state

    def mark_invalid(self) -> None:
        with self._lock:
            self._state.valid = False

    def mark_connected(self, connected: bool) -> None:
        with self._lock:
            self._state.connected = connected


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuestTeleopConfig:
    pose_scale: float = 1.5
    max_step: float = 0.05
    hand_close_scalar: float = 1.0
    hand_open_scalar: float = 0.0
    smoothing_alpha: float = 0.35
    auto_enable: bool = False
    debug_log_interval: int = 0

    def __post_init__(self):
        if self.smoothing_alpha is not None:
            if not 0.0 < self.smoothing_alpha <= 1.0:
                raise ValueError(f"smoothing_alpha must be in (0, 1], got {self.smoothing_alpha}")


# ---------------------------------------------------------------------------
# Quest to 22D action mapping
# ---------------------------------------------------------------------------


def _quat_wxyz_from_xyzw(xyzw: np.ndarray) -> np.ndarray:
    xyzw = np.asarray(xyzw, dtype=np.float64).reshape(4)
    return np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)


def _quat_wxyz_from_rot(rotation: R) -> np.ndarray:
    xyzw = rotation.as_quat()
    return _quat_wxyz_from_xyzw(xyzw)


class QuestActionMapper:
    """Map Quest Teleop state deltas to a 22D CR3+CRAFT action.

    On first valid state, latches start pose from the environment and initial
    Quest right-hand 4x4 transform.  Subsequent deltas are applied to produce
    the target EE pose.
    """

    def __init__(self, config: QuestTeleopConfig, craft_command_joints: tuple = ()):
        self.config = config
        self._craft_command_joints = craft_command_joints
        self._n_craft = len(craft_command_joints) if craft_command_joints else 15
        self._quest_start: Optional[QuestTeleopState] = None
        self._ee_start: Optional[np.ndarray] = None
        self._filtered_target_pos: Optional[np.ndarray] = None
        self._step_count = 0

    def reset(self) -> None:
        self._quest_start = None
        self._ee_start = None
        self._filtered_target_pos = None
        self._step_count = 0

    @property
    def is_calibrated(self) -> bool:
        return self._quest_start is not None and self._ee_start is not None

    def map(self, state: QuestTeleopState, env_ee_matrix: np.ndarray) -> np.ndarray:
        """Return a 22D action: xyz + quat_wxyz + craft15."""
        cfg = self.config
        self._step_count += 1

        if not state.valid:
            return self._hold_action(env_ee_matrix)

        if not self.is_calibrated:
            self._quest_start = state
            self._ee_start = env_ee_matrix.copy()
            self._filtered_target_pos = env_ee_matrix[:3, 3].copy()
            if cfg.debug_log_interval > 0:
                print("[Quest] Calibrated: latched EE start and Quest pose.")

        # Position delta
        delta_pos = (state.right_pos - self._quest_start.right_pos) * cfg.pose_scale
        target_pos = self._ee_start[:3, 3] + delta_pos
        target_pos = self._filter_position(target_pos)

        # Rotation delta
        delta_rot = (
            self._quest_start.right_rot.inv() * state.right_rot
        )
        ee_start_rot = R.from_matrix(self._ee_start[:3, :3])
        target_rot = ee_start_rot * delta_rot

        # craft15 from close_scalar
        craft_raw = np.full(self._n_craft, cfg.hand_close_scalar * 2.0 * np.pi, dtype=np.float64)
        close = state.close_scalar
        craft_open = np.full(self._n_craft, cfg.hand_open_scalar * 2.0 * np.pi, dtype=np.float64)
        craft_action = craft_open + close * (craft_raw - craft_open)

        if cfg.debug_log_interval > 0 and self._step_count % cfg.debug_log_interval == 0:
            print(
                f"[Quest] step={self._step_count} "
                f"delta_pos={np.round(delta_pos, 4)} "
                f"target_pos={np.round(target_pos, 4)} "
                f"close={close:.3f}"
            )

        return np.concatenate([target_pos, _quat_wxyz_from_rot(target_rot), craft_action], axis=0)

    def _hold_action(self, env_ee_matrix: np.ndarray) -> np.ndarray:
        target_pos = env_ee_matrix[:3, 3].copy()
        target_rot = R.from_matrix(env_ee_matrix[:3, :3])
        craft_open = np.full(self._n_craft, self.config.hand_open_scalar * 2.0 * np.pi, dtype=np.float64)
        return np.concatenate([target_pos, _quat_wxyz_from_rot(target_rot), craft_open], axis=0)

    def _filter_position(self, target_pos: np.ndarray) -> np.ndarray:
        if self._filtered_target_pos is None:
            self._filtered_target_pos = target_pos.copy()
            return target_pos

        alpha = self.config.smoothing_alpha
        filtered = self._filtered_target_pos + alpha * (target_pos - self._filtered_target_pos)

        max_step = self.config.max_step
        if max_step is not None and max_step > 0:
            step = filtered - self._filtered_target_pos
            step_norm = float(np.linalg.norm(step))
            if step_norm > max_step:
                filtered = self._filtered_target_pos + step / step_norm * max_step

        self._filtered_target_pos = filtered.copy()
        return filtered


# ---------------------------------------------------------------------------
# Gym ActionWrapper
# ---------------------------------------------------------------------------


class SingleArmQuestTeleopWrapper:
    """Drop-in wrapper that feeds Quest teleop state into a single-arm DexJoCo env.

    Usage::

        store = QuestStateStore()
        mapper = QuestActionMapper(config)
        wrapper = SingleArmQuestTeleopWrapper(env, store, mapper, config)
        # In the step loop, the WebSocket receiver pushes into store;
        # the wrapper reads from store on each step() call.
    """

    def __init__(
        self,
        env,
        state_store: QuestStateStore,
        mapper: QuestActionMapper,
        config: QuestTeleopConfig,
    ):
        self.env = env
        self._store = state_store
        self._mapper = mapper
        self.config = config
        self.intervened = bool(config.auto_enable)
        self.reset_trigger = False
        self._step_count = 0

    # Delegate attribute access to the wrapped env
    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        self._step_count += 1
        if not self.intervened:
            obs, rew, done, truncated, info = self.env.step(action)
            info = dict(info)
            info["intervene_action"] = action
            return obs, rew, done, truncated, info

        state = self._store.get()
        ee_matrix = self.env.get_end_effector_pose_matrix()
        teleop_action = self._mapper.map(state, ee_matrix)
        obs, rew, done, truncated, info = self.env.step(teleop_action)
        info = dict(info)
        info["intervene_action"] = teleop_action
        if self.reset_trigger:
            info["manual_reset"] = True
            self.reset_trigger = False
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.intervened = bool(self.config.auto_enable)
        self.reset_trigger = False
        self._mapper.reset()
        self._step_count = 0
        return obs, info

    def close(self):
        self.env.close()


# ---------------------------------------------------------------------------
# WebSocket receiver
# ---------------------------------------------------------------------------


def _parse_quest_packet(data: bytes) -> Optional[QuestTeleopState]:
    """Parse a JSON packet from the Quest WebXR client into QuestTeleopState."""
    try:
        msg = json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    if msg.get("type") != "quest_pose":
        return None

    state = QuestTeleopState()
    state.timestamp = float(msg.get("timestamp", time.time()))
    state.connected = True
    state.valid = True
    state.source = "quest_webxr"

    right = msg.get("right") or msg.get("controller") or {}
    state.right_pos = np.asarray(right.get("pos", [0.0, 0.0, 0.0]), dtype=np.float64)

    quat = right.get("quat")
    if quat is not None and len(quat) == 4:
        state.right_quat_xyzw = np.asarray(quat, dtype=np.float64)
    else:
        state.right_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    left = msg.get("left") or {}
    if left:
        state.left_pos = np.asarray(left.get("pos", [0.0, 0.0, 0.0]), dtype=np.float64)
        lq = left.get("quat")
        state.left_quat_xyzw = (
            np.asarray(lq, dtype=np.float64) if (lq and len(lq) == 4) else None
        )

    state.trigger = float(right.get("trigger", msg.get("trigger", 0.0)))
    state.grip = float(right.get("grip", msg.get("grip", 0.0)))
    state.pinch = float(right.get("pinch", msg.get("pinch", 0.0)))

    return state


class QuestWebSocketReceiver:
    """Start a background WebSocket server that accepts Quest WebXR pose packets.

    Requires the ``websockets`` library.  On import failure the constructor
    raises a clear RuntimeError.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        store: Optional[QuestStateStore] = None,
        certfile: Optional[str] = None,
        keyfile: Optional[str] = None,
    ):
        if (certfile is None) != (keyfile is None):
            raise ValueError(
                "certfile and keyfile must be provided together, or both omitted."
            )

        self.host = host
        self.port = port
        self.store = store or QuestStateStore()
        self.certfile = certfile
        self.keyfile = keyfile
        self._ssl_context: Optional[ssl.SSLContext] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._server_started = threading.Event()
        self._last_error: Optional[str] = None

        if certfile is not None and keyfile is not None:
            self._ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            self._ssl_context.load_cert_chain(certfile, keyfile)

        try:
            import websockets  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "The 'websockets' library is required for QuestWebSocketReceiver. "
                "Install it with: pip install websockets"
            )

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self._server_started.clear()
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        self._server_started.wait(timeout=5.0)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.store.mark_connected(False)

    def _run_server(self) -> None:
        import asyncio
        import websockets

        async def handler(websocket):
            self.store.mark_connected(True)
            try:
                async for message in websocket:
                    if self._stop_event.is_set():
                        break
                    state = _parse_quest_packet(
                        message if isinstance(message, bytes) else message.encode("utf-8")
                    )
                    if state is not None:
                        self.store.update(state)
            except Exception:
                pass
            finally:
                self.store.mark_connected(False)

        async def serve():
            self._server_started.set()
            serve_kwargs = {"host": self.host, "port": self.port}
            if self._ssl_context is not None:
                serve_kwargs["ssl"] = self._ssl_context
            async with websockets.serve(handler, **serve_kwargs):
                while not self._stop_event.is_set():
                    await asyncio.sleep(0.2)

        try:
            asyncio.run(serve())
        except Exception as exc:
            self._last_error = str(exc)
            self._server_started.set()


# ---------------------------------------------------------------------------
# Synthetic (no-Quest) data source for smoke testing
# ---------------------------------------------------------------------------


class SyntheticQuestSource:
    """Produces synthetic QuestTeleopState packets for smoke testing.

    Moves the right hand in a small circle + opens/closes the hand over time.
    """

    def __init__(self, store: QuestStateStore, radius: float = 0.05, period: int = 40):
        self.store = store
        self.radius = radius
        self.period = period
        self._step = 0

    def step(self) -> QuestTeleopState:
        self._step += 1
        t = self._step

        theta = 2.0 * np.pi * t / self.period
        pos = np.array(
            [self.radius * np.cos(theta), self.radius * np.sin(theta), 0.0],
            dtype=np.float64,
        )

        quat = R.from_euler("z", theta / 2.0).as_quat()

        close_value = 0.5 + 0.5 * np.sin(2.0 * np.pi * t / (self.period * 2))

        state = QuestTeleopState(
            timestamp=time.time(),
            connected=True,
            valid=True,
            right_pos=pos,
            right_quat_xyzw=quat,
            trigger=close_value,
            grip=close_value,
            pinch=close_value,
            source="synthetic",
        )
        self.store.update(state)
        return state
