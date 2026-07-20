from pathlib import Path
from typing import Literal

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from ..mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "cr3_craft_reach_debug.xml"

ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
ARM_ACTUATOR_NAMES = tuple(f"arm_joint{i}" for i in range(1, 7))
HOME_Q = np.asarray((0.0, 0.6072, -1.7223, -0.2949, 1.6134, 0.0), dtype=np.float64)

# Direct CRAFT command order used by the HaMeR-CRAFT project:
# Ring, Index, Thumb, Middle, Pinky; each [PIP, MCP forward, MCP side].
CRAFT_COMMAND_JOINTS = (
    "ring_3",
    "ring_2",
    "ring_1",
    "index_3",
    "index_2",
    "index_1",
    "thumb_3",
    "thumb_2",
    "thumb_mcp",
    "middle_3",
    "middle_2",
    "middle_1",
    "pinky_3",
    "pinky_2",
    "pinky_1",
)
CRAFT_COUPLED_DISTAL = {
    "index_3": "index_4",
    "middle_3": "middle_4",
    "ring_3": "ring_4",
    "pinky_3": "pinky_4",
    "thumb_3": "thumb_4",
}
CRAFT_VALUE_MAX = 2.0 * np.pi
CRAFT_ACTUATOR_NAMES = tuple(f"h_{name}" for name in (
    "index_1",
    "index_2",
    "index_3",
    "index_4",
    "middle_1",
    "middle_2",
    "middle_3",
    "middle_4",
    "ring_1",
    "ring_2",
    "ring_3",
    "ring_4",
    "pinky_1",
    "pinky_2",
    "pinky_3",
    "pinky_4",
    "thumb_mcp",
    "thumb_2",
    "thumb_3",
    "thumb_4",
))


def _wxyz_to_rotation(quat: np.ndarray) -> Rotation:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return Rotation.identity()
    quat = quat / norm
    return Rotation.from_quat((quat[1], quat[2], quat[3], quat[0]))


def _rotation_to_wxyz(rotation: Rotation) -> np.ndarray:
    xyzw = rotation.as_quat()
    return np.asarray((xyzw[3], xyzw[0], xyzw[1], xyzw[2]), dtype=np.float64)


def _site_rotation(data: mujoco.MjData, site_id: int) -> Rotation:
    return Rotation.from_matrix(data.site_xmat[site_id].reshape(3, 3))


def _damped_least_squares_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    arm_dof_idx: np.ndarray,
    command: np.ndarray,
    damping: float,
    max_norm: float,
) -> np.ndarray:
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    jac = np.vstack((jacp[:, arm_dof_idx], jacr[:, arm_dof_idx]))
    lhs = jac @ jac.T + damping * np.eye(6)
    dq = jac.T @ np.linalg.solve(lhs, command)
    norm = np.linalg.norm(dq)
    if norm > max_norm:
        dq *= max_norm / norm
    return dq


class Cr3CraftReachDebugEnv(MujocoGymEnv):
    """Minimal CR3 + CRAFT single-arm DexJoCo backend.

    Action layout is `[target_xyz, target_quat_wxyz, craft15]`, where craft15
    follows the HaMeR-CRAFT direct order documented above.
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        render_spec: GymRenderingSpec = GymRenderingSpec(camera_id="front"),
        time_limit: float = 30.0,
    ):
        super().__init__(
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
        )
        self.render_mode = render_mode
        self.env_step = 0

        self._arm_joint_ids = np.asarray([self.model.joint(name).id for name in ARM_JOINT_NAMES])
        self._arm_qpos_idx = np.asarray([self.model.jnt_qposadr[jid] for jid in self._arm_joint_ids])
        self._arm_dof_idx = np.asarray([self.model.jnt_dofadr[jid] for jid in self._arm_joint_ids])
        self._arm_ctrl_ids = np.asarray([self.model.actuator(name).id for name in ARM_ACTUATOR_NAMES])
        self._tcp_site_id = self.model.site("tcp").id

        self._hand_joint_ids = {name: self.model.joint(name).id for name in self._hand_joint_names()}
        self._hand_ctrl_ids = {
            name[2:]: self.model.actuator(name).id
            for name in CRAFT_ACTUATOR_NAMES
            if name in [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)]
        }
        self._controlled_dof_idx = np.asarray(
            [self.model.jnt_dofadr[jid] for jid in (*self._arm_joint_ids, *self._hand_joint_ids.values())],
            dtype=np.int32,
        )

        self._q_des = np.zeros(self.model.nq, dtype=np.float64)
        self._q_cmd = np.zeros(self.model.nq, dtype=np.float64)
        self._target_xyz = np.zeros(3, dtype=np.float64)
        self._target_quat = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
        self._workspace_delta_low = np.asarray((-0.55, -0.60, -0.55), dtype=np.float64)
        self._workspace_delta_high = np.asarray((0.55, 0.60, 0.55), dtype=np.float64)
        self._max_ee_step = 0.04
        self._max_resolved_dq = 0.04
        self._damping = 0.1
        self._arm_ctrl_step = 0.02
        self._hand_ctrl_step = 5.0
        self._collision_hold = 0
        self._home_tcp_xyz = None

        xyz_low = np.full(3, -np.inf, dtype=np.float32)
        xyz_high = np.full(3, np.inf, dtype=np.float32)
        quat_low = np.full(4, -1.0, dtype=np.float32)
        quat_high = np.full(4, 1.0, dtype=np.float32)
        craft_low = np.zeros(15, dtype=np.float64)
        craft_high = np.full(15, CRAFT_VALUE_MAX, dtype=np.float64)
        self.action_space = gym.spaces.Box(
            low=np.concatenate((xyz_low, quat_low, craft_low.astype(np.float32))),
            high=np.concatenate((xyz_high, quat_high, craft_high.astype(np.float32))),
            dtype=np.float32,
        )
        image_h = int(self.model.vis.global_.offheight)
        image_w = int(self.model.vis.global_.offwidth)
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
                        "arm_qpos": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float64),
                        "craft_qpos": spaces.Box(-np.inf, np.inf, shape=(20,), dtype=np.float64),
                        "target_tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
                    }
                ),
                "images": gym.spaces.Dict(
                    {"front": spaces.Box(0, 255, shape=(image_h, image_w, 3), dtype=np.uint8)}
                ),
            }
        )

    def _hand_joint_names(self) -> tuple[str, ...]:
        ordered = []
        for name in CRAFT_COMMAND_JOINTS:
            ordered.append(name)
            coupled = CRAFT_COUPLED_DISTAL.get(name)
            if coupled:
                ordered.append(coupled)
        return tuple(dict.fromkeys(ordered))

    def _craft_action_ranges(self) -> tuple[np.ndarray, np.ndarray]:
        lows = []
        highs = []
        for joint_name in CRAFT_COMMAND_JOINTS:
            jid = self.model.joint(joint_name).id
            low, high = self.model.jnt_range[jid]
            lows.append(low)
            highs.append(high)
        return np.asarray(lows, dtype=np.float64), np.asarray(highs, dtype=np.float64)

    def _tcp_pose(self) -> np.ndarray:
        quat = _rotation_to_wxyz(_site_rotation(self.data, self._tcp_site_id))
        return np.concatenate((self.data.site_xpos[self._tcp_site_id].copy(), quat))

    def get_end_effector_pose_matrix(self) -> np.ndarray:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = self.data.site_xpos[self._tcp_site_id]
        pose[:3, :3] = self.data.site_xmat[self._tcp_site_id].reshape(3, 3)
        return pose

    def get_initial_action(self) -> np.ndarray:
        return np.concatenate((self._tcp_pose(), self._craft_action_from_qpos()))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        if seed is not None:
            self._random = np.random.RandomState(seed)
        self.env_step = 0
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[self._arm_qpos_idx] = HOME_Q
        mujoco.mj_forward(self.model, self.data)
        self._q_des[:] = self.data.qpos
        self._q_cmd[:] = self.data.qpos
        self._home_tcp_xyz = self.data.site_xpos[self._tcp_site_id].copy()
        self._target_xyz[:] = self._home_tcp_xyz
        self._target_quat[:] = _rotation_to_wxyz(_site_rotation(self.data, self._tcp_site_id))
        self._write_all_ctrls()
        mujoco.mj_forward(self.model, self.data)
        return self._observation(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).reshape(22)
        self._set_target(action)
        self._ik_update()
        for _ in range(self._n_substeps):
            self._write_all_ctrls()
            mujoco.mj_step(self.model, self.data)
        self.env_step += 1
        obs = self._observation()
        err = float(np.linalg.norm(self._target_xyz - self.data.site_xpos[self._tcp_site_id]))
        reward = -err
        terminated = False
        truncated = self.time_limit_exceeded()
        info = {
            "tcp_error": err,
            "target_tcp_pose": np.concatenate((self._target_xyz.copy(), self._target_quat.copy())),
            "ncon": int(self.data.ncon),
            "intervene_action": action.copy(),
        }
        return obs, reward, terminated, truncated, info

    def _set_target(self, action: np.ndarray) -> None:
        if self._home_tcp_xyz is None:
            self._home_tcp_xyz = self.data.site_xpos[self._tcp_site_id].copy()
        low = self._home_tcp_xyz + self._workspace_delta_low
        high = self._home_tcp_xyz + self._workspace_delta_high
        self._target_xyz = np.clip(action[:3], low, high)
        self._target_quat = _rotation_to_wxyz(_wxyz_to_rotation(action[3:7]))
        self._apply_craft_action(action[7:22])

    def _apply_craft_action(self, craft_action: np.ndarray) -> None:
        for value, joint_name in zip(craft_action, CRAFT_COMMAND_JOINTS):
            target = self._craft_value_to_joint_target(joint_name, value)
            self._set_joint_target(joint_name, target)
            coupled = CRAFT_COUPLED_DISTAL.get(joint_name)
            if coupled:
                self._set_joint_target(coupled, target)

    def _craft_value_to_joint_target(self, joint_name: str, value: float) -> float:
        jid = self.model.joint(joint_name).id
        low, high = self.model.jnt_range[jid]
        percent = np.clip(float(value) / CRAFT_VALUE_MAX, 0.0, 1.0)
        return float(low + percent * (high - low))

    def _set_joint_target(self, joint_name: str, value: float) -> None:
        jid = self.model.joint(joint_name).id
        qadr = self.model.jnt_qposadr[jid]
        low, high = self.model.jnt_range[jid]
        self._q_des[qadr] = np.clip(float(value), low, high)

    def _ik_update(self) -> None:
        pos_err = self._target_xyz - self.data.site_xpos[self._tcp_site_id]
        pos_norm = np.linalg.norm(pos_err)
        if pos_norm > self._max_ee_step:
            pos_err *= self._max_ee_step / pos_norm

        current_rot = _site_rotation(self.data, self._tcp_site_id)
        target_rot = _wxyz_to_rotation(self._target_quat)
        rot_err = (target_rot * current_rot.inv()).as_rotvec()
        rot_norm = np.linalg.norm(rot_err)
        if rot_norm > 0.015:
            rot_err *= 0.015 / rot_norm

        cmd = np.concatenate((pos_err, rot_err))
        if self._collision_hold > 0:
            self._collision_hold -= 1
        else:
            dq = _damped_least_squares_step(
                self.model,
                self.data,
                self._tcp_site_id,
                self._arm_dof_idx,
                cmd,
                self._damping,
                self._max_resolved_dq,
            )
            self._q_des[self._arm_qpos_idx] += dq
            self._clamp_limited_joints()

    def _clamp_limited_joints(self) -> None:
        for jid in range(self.model.njnt):
            if self.model.jnt_limited[jid]:
                qadr = self.model.jnt_qposadr[jid]
                low, high = self.model.jnt_range[jid]
                self._q_des[qadr] = np.clip(self._q_des[qadr], low, high)

    def _write_all_ctrls(self) -> None:
        mujoco.mj_forward(self.model, self.data)
        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[self._controlled_dof_idx] = self.data.qfrc_bias[self._controlled_dof_idx]

        for qadr, ctrl_id in zip(self._arm_qpos_idx, self._arm_ctrl_ids):
            self._q_cmd[qadr] += np.clip(self._q_des[qadr] - self._q_cmd[qadr], -self._arm_ctrl_step, self._arm_ctrl_step)
            self.data.ctrl[ctrl_id] = self._q_cmd[qadr]

        for joint_name, ctrl_id in self._hand_ctrl_ids.items():
            qadr = self.model.jnt_qposadr[self.model.joint(joint_name).id]
            self._q_cmd[qadr] += np.clip(self._q_des[qadr] - self._q_cmd[qadr], -self._hand_ctrl_step, self._hand_ctrl_step)
            self.data.ctrl[ctrl_id] = self._q_cmd[qadr]

    def _craft_action_from_qpos(self) -> np.ndarray:
        values = []
        for name in CRAFT_COMMAND_JOINTS:
            jid = self.model.joint(name).id
            qpos = self.data.qpos[self.model.jnt_qposadr[jid]]
            low, high = self.model.jnt_range[jid]
            percent = 0.5 if high <= low else np.clip((qpos - low) / (high - low), 0.0, 1.0)
            values.append(percent * CRAFT_VALUE_MAX)
        return np.asarray(values, dtype=np.float64)

    def _craft_state(self) -> np.ndarray:
        return np.asarray(
            [self.data.qpos[self.model.jnt_qposadr[self.model.joint(name).id]] for name in self._hand_joint_names()],
            dtype=np.float64,
        )

    def _observation(self):
        return {
            "state": {
                "tcp_pose": self._tcp_pose(),
                "arm_qpos": self.data.qpos[self._arm_qpos_idx].copy(),
                "craft_qpos": self._craft_state(),
                "target_tcp_pose": np.concatenate((self._target_xyz.copy(), self._target_quat.copy())),
            },
            "images": {"front": self.render()},
        }
