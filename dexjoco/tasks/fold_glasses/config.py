from typing import Literal

from ...sim.envs.cr3_craft_fold_glasses_env import Cr3CraftFoldGlassesEnv
from ..config import TaskConfigBase
from ..obs_adapters import DexjocoObsAdapter
from ..sim_teleop import SingleArmTeleopConfig
from ..single_arm_teleop import wrap_single_arm_teleop


class TaskConfig(TaskConfigBase):
    proprio_keys = [
        "tcp_pose",
        "gripper_pose",
        "glass_ori_pose",
        "box_ori_pose",
        "table_delta_height",
    ]
    teleop = SingleArmTeleopConfig(pose_scale=2.0)

    def get_environment(
        self,
        policy_mode=False,
        render_mode: Literal["rgb_array", "human"] = "human",
        randomize=False,
        **kwargs,
    ):
        teleop_source = kwargs.pop("teleop_source", "vive")
        hand = kwargs.get("hand", "allegro")
        hamer_config = kwargs.pop("hamer_config", None)
        hamer_config_path = kwargs.pop("hamer_config_path", None)
        hamer_overrides = kwargs.pop("hamer_overrides", None)
        env = Cr3CraftFoldGlassesEnv(
            render_mode=render_mode, randomize=randomize, hz=30, **kwargs
        )
        env = wrap_single_arm_teleop(
            env,
            policy_mode=policy_mode,
            teleop_source=teleop_source,
            hand=hand,
            vive_config=self.teleop,
            hamer_config=hamer_config,
            hamer_config_path=hamer_config_path,
            hamer_overrides=hamer_overrides,
        )
        env = DexjocoObsAdapter(env, proprio_keys=self.proprio_keys)
        return env

    def process_demos(self, demo):
        return demo
