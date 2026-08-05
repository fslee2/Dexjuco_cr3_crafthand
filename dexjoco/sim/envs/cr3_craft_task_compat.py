"""Shared CR3+CRAFT compatibility setup for the original single-arm tasks."""

from __future__ import annotations

import numpy as np

from .hand_models import SingleArmHand

CR3_HOME_Q = np.asarray((0.0, 0.6072, -1.7223, -0.2949, 1.6134, 0.0), dtype=np.float64)


def is_cr3_model(model) -> bool:
    """Return whether a compiled scene contains the embedded CR3 interface."""
    try:
        return model.actuator("arm_joint6").id >= 0 and model.site("tcp").id >= 0
    except Exception:
        return False


def configure_cr3_task_env(env) -> None:
    """Replace Panda/Allegro caches with CR3/CRAFT equivalents on an env.

    The task-specific classes keep their existing object randomization and
    reward logic. This function only supplies the arm, TCP, hand and action
    interface expected by those classes.
    """
    env._is_cr3 = True
    env._panda_dof_ids = np.asarray(
        [env._model.jnt_dofadr[env._model.joint(f"joint{i}").id] for i in range(1, 7)],
        dtype=np.int32,
    )
    env._panda_qpos_ids = np.asarray(
        [env._model.jnt_qposadr[env._model.joint(f"joint{i}").id] for i in range(1, 7)],
        dtype=np.int32,
    )
    env._panda_ctrl_ids = np.asarray(
        [env._model.actuator(f"arm_joint{i}").id for i in range(1, 7)],
        dtype=np.int32,
    )
    env._panda_mocap_id = int(env._model.body("target").mocapid[0])
    env._site_id = int(env._model.site("tcp").id)
    env._hand = SingleArmHand(env._model, "craft")
    env._arm_home = CR3_HOME_Q.copy()

