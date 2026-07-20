"""CR3+CRAFT click_mouse shell env — true Panda scene + CR3 robot replacement."""

from pathlib import Path
from typing import Any, Dict, Tuple

import mujoco
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from ..controllers import opspace
from ..mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "cr3_craft_click_mouse_shell.xml"

ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
ARM_ACTUATOR_NAMES = tuple(f"arm_joint{i}" for i in range(1, 7))
HOME_Q = np.asarray((0.0, 0.6072, -1.7223, -0.2949, 1.6134, 0.0), dtype=np.float64)

CRAFT_COMMAND_JOINTS = (
    "ring_3", "ring_2", "ring_1",
    "index_3", "index_2", "index_1",
    "thumb_3", "thumb_2", "thumb_mcp",
    "middle_3", "middle_2", "middle_1",
    "pinky_3", "pinky_2", "pinky_1",
)
CRAFT_COUPLED_DISTAL = {
    "index_3": "index_4", "middle_3": "middle_4",
    "ring_3": "ring_4", "pinky_3": "pinky_4", "thumb_3": "thumb_4",
}
CRAFT_VALUE_MAX = 2.0 * np.pi
CRAFT_ACTUATOR_NAMES = tuple(f"h_{name}" for name in (
    "index_1", "index_2", "index_3", "index_4",
    "middle_1", "middle_2", "middle_3", "middle_4",
    "ring_1", "ring_2", "ring_3", "ring_4",
    "pinky_1", "pinky_2", "pinky_3", "pinky_4",
    "thumb_mcp", "thumb_2", "thumb_3", "thumb_4",
))

# From Panda click_mouse env
_MOUSE_XY_BOUNDS = np.asarray([[-0.2, 0.0], [-0.25, 0.05]])
_MOUSEPAD_OFFSET = np.asarray([0.0, -0.25, 0.0])
_PLANT_XY_BOUNDS = np.asarray([[0.12, 0.3], [0.12, 0.3]])
_MOUSEPAD_Z_OFFSET = 0.002
_MOUSEPAD_Z_TOL = 0.05
_CLICK_THRESHOLD = 0.001
_YAW_PERTURB_BOUNDS = np.array([-10, 10])


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


class Cr3CraftClickMouseShellEnv(MujocoGymEnv):
    metadata = {"render_modes": ["rgb_array", "human", "none"]}

    def __init__(
        self, seed: int = 0, control_dt: float = 0.02, physics_dt: float = 0.002,
        render_mode: str = "rgb_array", time_limit: float = 30.0,
    ):
        super().__init__(
            xml_path=_XML_PATH, seed=seed, control_dt=control_dt,
            physics_dt=physics_dt, time_limit=time_limit,
            render_spec=GymRenderingSpec(camera_id="front"),
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
        self._hand_dof_idx = np.asarray(
            [self.model.jnt_dofadr[jid] for jid in self._hand_joint_ids.values()],
            dtype=np.int32,
        )

        self._q_des = np.zeros(self.model.nq, dtype=np.float64)
        self._q_cmd = np.zeros(self.model.nq, dtype=np.float64)
        self._target_xyz = np.zeros(3, dtype=np.float64)
        self._target_quat = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
        self._workspace_delta_low = np.asarray((-0.55, -0.60, -0.55), dtype=np.float64)
        self._workspace_delta_high = np.asarray((0.55, 0.60, 0.55), dtype=np.float64)
        self._hand_ctrl_step = 5.0
        self._opspace_pos_gains = np.asarray((250.0, 250.0, 250.0), dtype=np.float64)
        self._opspace_ori_gains = np.asarray((50.0, 50.0, 50.0), dtype=np.float64)
        self._opspace_damping_ratio = 4.0
        self._opspace_nullspace_stiffness = 0.5
        self._home_tcp_xyz = None

        xyz_low = np.full(3, -np.inf, dtype=np.float32)
        xyz_high = np.full(3, np.inf, dtype=np.float32)
        quat_low = np.full(4, -1.0, dtype=np.float32)
        quat_high = np.full(4, 1.0, dtype=np.float32)
        craft_low = np.zeros(15, dtype=np.float64)
        craft_high = np.full(15, CRAFT_VALUE_MAX, dtype=np.float64)
        self.action_space = spaces.Box(
            low=np.concatenate((xyz_low, quat_low, craft_low.astype(np.float32))),
            high=np.concatenate((xyz_high, quat_high, craft_high.astype(np.float32))),
            dtype=np.float32,
        )
        image_h = int(self.model.vis.global_.offheight)
        image_w = int(self.model.vis.global_.offwidth)
        self.observation_space = spaces.Dict({
            "state": spaces.Dict({
                "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
                "arm_qpos": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float64),
                "craft_qpos": spaces.Box(-np.inf, np.inf, shape=(20,), dtype=np.float64),
                "target_tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64),
            }),
            "images": spaces.Dict({})
            if self.render_mode == "none"
            else spaces.Dict({"front": spaces.Box(0, 255, shape=(image_h, image_w, 3), dtype=np.uint8)}),
        })

        # Task objects
        self._table_body_id = self.model.body("table").id
        self._table_body_z0 = float(self.model.body("table").pos[2])
        self._table_leg_geom_ids = [
            gid for gid in range(self.model.ngeom)
            if self.model.geom_bodyid[gid] == self._table_body_id
            and self.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CYLINDER
        ]
        self._table_leg_half_len0 = {
            gid: float(self.model.geom_size[gid, 1]) for gid in self._table_leg_geom_ids
        }
        self._mousepad_geom_id = self.model.geom("mousepad").id
        self._mousepad_radius = float(self.model.geom_size[self._mousepad_geom_id][0])
        self._mousepad_body_id = self.model.geom_bodyid[self._mousepad_geom_id]
        self._table_site_id = self.model.site("table_top").id

        self._display_root_z0 = float(self.data.jnt("display_root").qpos[2])
        self._mouse_joint0_init = 0.0
        self._success_trigger_target = 10
        self._success_trigger_count = 0
        self._display_blue = False

    def _hand_joint_names(self) -> tuple:
        ordered = []
        for name in CRAFT_COMMAND_JOINTS:
            ordered.append(name)
            coupled = CRAFT_COUPLED_DISTAL.get(name)
            if coupled:
                ordered.append(coupled)
        return tuple(dict.fromkeys(ordered))

    def _tcp_pose(self) -> np.ndarray:
        quat = _rotation_to_wxyz(_site_rotation(self.data, self._tcp_site_id))
        return np.concatenate((self.data.site_xpos[self._tcp_site_id].copy(), quat))

    def get_initial_action(self) -> np.ndarray:
        return np.concatenate((self._tcp_pose(), self._craft_action_from_qpos()))

    def get_end_effector_pose_matrix(self) -> np.ndarray:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = self.data.site_xpos[self._tcp_site_id]
        pose[:3, :3] = self.data.site_xmat[self._tcp_site_id].reshape(3, 3)
        return pose

    def _set_display_blue(self):
        try:
            mat_id = self.model.material("display_new-0_material-0").id
            self.model.mat_rgba[mat_id] = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
            self._display_blue = True
        except Exception:
            pass

    def _restore_display_color(self):
        try:
            mat_id = self.model.material("display_new-0_material-0").id
            self.model.mat_rgba[mat_id] = np.array([0.11372549, 0.11372549, 0.11372549, 1.0], dtype=np.float32)
            self._display_blue = False
        except Exception:
            pass

    def _detect_click(self) -> bool:
        try:
            joint0 = float(self.data.sensor("mouse_joint0_pos").data[0])
        except Exception:
            return False
        return (joint0 - self._mouse_joint0_init) > _CLICK_THRESHOLD

    def _mouse_in_mousepad(self) -> bool:
        try:
            mouse_pos = self.data.sensor("mouse_pos").data
        except Exception:
            return False
        pad_center = self.data.geom_xpos[self._mousepad_geom_id]
        dx, dy, dz = mouse_pos - pad_center
        in_xy = (dx * dx + dy * dy) <= (self._mousepad_radius * self._mousepad_radius)
        in_z = abs(dz) <= _MOUSEPAD_Z_TOL
        return in_xy and in_z

    def _compute_success(self) -> bool:
        trigger = self._mouse_in_mousepad() and self._display_blue
        self._success_trigger_count = self._success_trigger_count + 1 if trigger else 0
        return self._success_trigger_count >= self._success_trigger_target

    def reset(self, *, seed=None, options=None) -> Tuple[Dict, Dict]:
        if seed is not None:
            self._random = np.random.RandomState(seed)
        self.env_step = 0

        mujoco.mj_resetData(self.model, self.data)

        # Randomize table height (from Panda env)
        self.delta_h = float(np.random.uniform(0.0, 0.05))
        self.model.body_pos[self._table_body_id, 2] = self._table_body_z0 + self.delta_h
        for gid in self._table_leg_geom_ids:
            self.model.geom_size[gid, 1] = self._table_leg_half_len0[gid] + self.delta_h

        # Set CR3 arm to home. Do not zero all qpos here: the Panda shell
        # contains free joints whose default quaternions define object poses.
        self.data.qvel[:] = 0.0
        self.data.qpos[self._arm_qpos_idx] = HOME_Q
        mujoco.mj_forward(self.model, self.data)

        # Mouse position (from Panda env)
        table_z = float(self.data.site_xpos[self._table_site_id][2])
        self._mouse_z = table_z + _MOUSEPAD_Z_OFFSET
        mouse_xy = np.random.uniform(*_MOUSE_XY_BOUNDS)
        yaw = np.deg2rad(np.random.uniform(*_YAW_PERTURB_BOUNDS))
        orig_quat = np.array(self.data.jnt("mouse_root").qpos[3:7], dtype=np.float64)
        qw, qz = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
        w2, x2, y2, z2 = orig_quat
        q_new = np.array(
            [
                qw * w2 - qz * z2,
                qw * x2 - qz * y2,
                qw * y2 + qz * x2,
                qw * z2 + qz * w2,
            ],
            dtype=np.float64,
        )
        q_new /= np.linalg.norm(q_new)
        self.data.jnt("mouse_root").qpos = np.concatenate([mouse_xy, [self._mouse_z], q_new]).astype(np.float64)

        # Mousepad position (from Panda env)
        if self._mousepad_body_id >= 0:
            self.model.body_pos[self._mousepad_body_id][:3] = (
                np.array([mouse_xy[0], mouse_xy[1], table_z]) + _MOUSEPAD_OFFSET
            )

        # Display position (from Panda env)
        monitor_xy = np.random.uniform(*_PLANT_XY_BOUNDS)
        self.data.jnt("display_root").qpos[:3] = (monitor_xy[0], monitor_xy[1], self._display_root_z0 + self.delta_h)
        self.data.jnt("display_root").qpos[3:7] = np.array([0.5, 0.5, -0.5, -0.5])

        self._restore_display_color()
        try:
            self._mouse_joint0_init = float(self.data.sensor("mouse_joint0_pos").data[0])
        except Exception:
            self._mouse_joint0_init = 0.0

        self._success_trigger_count = 0
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._q_des[:] = self.data.qpos
        self._q_cmd[:] = self.data.qpos
        self._home_tcp_xyz = self.data.site_xpos[self._tcp_site_id].copy()
        self._target_xyz[:] = self._home_tcp_xyz
        self._target_quat[:] = _rotation_to_wxyz(_site_rotation(self.data, self._tcp_site_id))
        self._write_all_ctrls()
        mujoco.mj_forward(self.model, self.data)

        return self._observation(), {"succeed": False}

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64).reshape(22)
        self._set_target(action)
        for _ in range(self._n_substeps):
            self._write_all_ctrls()
            mujoco.mj_step(self.model, self.data)

        if self._mouse_in_mousepad() and self._detect_click() and not self._display_blue:
            self._set_display_blue()

        self.env_step += 1
        obs = self._observation()
        err = float(np.linalg.norm(self._target_xyz - self.data.site_xpos[self._tcp_site_id]))
        success = self._compute_success()
        terminated = success or self.time_limit_exceeded()
        info = {
            "tcp_error": err, "ncon": int(self.data.ncon), "succeed": success,
            "mouse_in_pad": self._mouse_in_mousepad(), "click_detected": self._detect_click(),
            "display_blue": self._display_blue,
        }
        return obs, -err, terminated, False, info

    def _set_target(self, action):
        if self._home_tcp_xyz is None:
            self._home_tcp_xyz = self.data.site_xpos[self._tcp_site_id].copy()
        low = self._home_tcp_xyz + self._workspace_delta_low
        high = self._home_tcp_xyz + self._workspace_delta_high
        self._target_xyz = np.clip(action[:3], low, high)
        self._target_quat = _rotation_to_wxyz(_wxyz_to_rotation(action[3:7]))
        self._apply_craft_action(action[7:22])

    def _apply_craft_action(self, craft_action):
        for value, joint_name in zip(craft_action, CRAFT_COMMAND_JOINTS):
            target = self._craft_value_to_joint_target(joint_name, value)
            self._set_joint_target(joint_name, target)
            coupled = CRAFT_COUPLED_DISTAL.get(joint_name)
            if coupled:
                self._set_joint_target(coupled, target)

    def _craft_value_to_joint_target(self, joint_name, value):
        jid = self.model.joint(joint_name).id
        low, high = self.model.jnt_range[jid]
        return float(low + np.clip(float(value) / CRAFT_VALUE_MAX, 0.0, 1.0) * (high - low))

    def _set_joint_target(self, joint_name, value):
        jid = self.model.joint(joint_name).id
        qadr = self.model.jnt_qposadr[jid]
        low, high = self.model.jnt_range[jid]
        self._q_des[qadr] = np.clip(float(value), low, high)

    def _write_all_ctrls(self):
        mujoco.mj_forward(self.model, self.data)
        self.data.qfrc_applied[:] = 0.0
        if self._hand_dof_idx.size:
            self.data.qfrc_applied[self._hand_dof_idx] = self.data.qfrc_bias[self._hand_dof_idx]

        arm_tau = opspace(
            model=self.model,
            data=self.data,
            site_id=self._tcp_site_id,
            dof_ids=self._arm_dof_idx,
            pos=self._target_xyz,
            ori=self._target_quat,
            ori_gains=self._opspace_ori_gains,
            joint=HOME_Q,
            gravity_comp=True,
            pos_gains=self._opspace_pos_gains,
            damping_ratio=self._opspace_damping_ratio,
            nullspace_stiffness=self._opspace_nullspace_stiffness,
        )
        ctrl_low = self.model.actuator_ctrlrange[self._arm_ctrl_ids, 0]
        ctrl_high = self.model.actuator_ctrlrange[self._arm_ctrl_ids, 1]
        self.data.ctrl[self._arm_ctrl_ids] = np.clip(arm_tau, ctrl_low, ctrl_high)

        for joint_name, ctrl_id in self._hand_ctrl_ids.items():
            qadr = self.model.jnt_qposadr[self.model.joint(joint_name).id]
            self._q_cmd[qadr] += np.clip(self._q_des[qadr] - self._q_cmd[qadr], -self._hand_ctrl_step, self._hand_ctrl_step)
            self.data.ctrl[ctrl_id] = self._q_cmd[qadr]

    def _craft_action_from_qpos(self):
        values = []
        for name in CRAFT_COMMAND_JOINTS:
            jid = self.model.joint(name).id
            qpos = self.data.qpos[self.model.jnt_qposadr[jid]]
            low, high = self.model.jnt_range[jid]
            percent = 0.5 if high <= low else np.clip((qpos - low) / (high - low), 0.0, 1.0)
            values.append(percent * CRAFT_VALUE_MAX)
        return np.asarray(values, dtype=np.float64)

    def _craft_state(self):
        return np.asarray(
            [self.data.qpos[self.model.jnt_qposadr[self.model.joint(name).id]] for name in self._hand_joint_names()],
            dtype=np.float64,
        )

    def _observation(self):
        if self.render_mode == "none":
            images = {}
        else:
            images = {"front": self.render()}
        return {
            "state": {
                "tcp_pose": self._tcp_pose(),
                "arm_qpos": self.data.qpos[self._arm_qpos_idx].copy(),
                "craft_qpos": self._craft_state(),
                "target_tcp_pose": np.concatenate((self._target_xyz.copy(), self._target_quat.copy())),
            },
            "images": images,
        }
