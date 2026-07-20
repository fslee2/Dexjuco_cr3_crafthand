import numpy as np

from dexjoco.sim.envs.cr3_craft_reach_debug_env import (
    CRAFT_COMMAND_JOINTS,
    CRAFT_VALUE_MAX,
    Cr3CraftReachDebugEnv,
)


def test_cr3_craft_action_contract_is_22d():
    env = Cr3CraftReachDebugEnv()
    try:
        obs, _info = env.reset()
        action = env.get_initial_action()
        assert action.shape == (22,)
        assert env.action_space.shape == (22,)
        np.testing.assert_allclose(env.action_space.low[7:22], 0.0)
        np.testing.assert_allclose(env.action_space.high[7:22], CRAFT_VALUE_MAX)
        assert obs["state"]["tcp_pose"].shape == (7,)
        assert obs["state"]["craft_qpos"].shape == (20,)
    finally:
        env.close()


def test_cr3_craft_direct_command_order():
    assert CRAFT_COMMAND_JOINTS == (
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


def test_cr3_craft_small_tcp_step_is_finite():
    env = Cr3CraftReachDebugEnv()
    try:
        _obs, _info = env.reset()
        action = env.get_initial_action()
        action[0] += 0.03
        action[1] += 0.02
        action[2] += 0.01
        for _ in range(10):
            obs, _reward, _terminated, _truncated, info = env.step(action)
        assert np.isfinite(obs["state"]["arm_qpos"]).all()
        assert np.isfinite(obs["state"]["craft_qpos"]).all()
        assert np.isfinite(info["tcp_error"])
    finally:
        env.close()


# ── Click mouse env tests ──


def test_click_mouse_reset_step():
    from dexjoco.sim.envs.cr3_craft_click_mouse_env import Cr3CraftClickMouseEnv

    env = Cr3CraftClickMouseEnv()
    try:
        obs, info = env.reset()
        assert env.action_space.shape == (22,)
        init_act = env.get_initial_action()
        assert init_act.shape == (22,)
        assert obs["state"]["tcp_pose"].shape == (7,)
        assert obs["state"]["craft_qpos"].shape == (20,)
        assert "front" in obs["images"]
        assert "succeed" in info

        action = env.get_initial_action()
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "mouse_in_pad" in info
        assert "click_detected" in info
        assert "display_blue" in info
    finally:
        env.close()


def test_click_mouse_has_ee_pose_matrix():
    from dexjoco.sim.envs.cr3_craft_click_mouse_env import Cr3CraftClickMouseEnv

    env = Cr3CraftClickMouseEnv()
    try:
        env.reset()
        ee = env.get_end_effector_pose_matrix()
        assert ee.shape == (4, 4)
        assert np.allclose(ee[3, :], [0, 0, 0, 1])
    finally:
        env.close()


# ── Shell env tests ──


def test_shell_reset_step():
    from dexjoco.sim.envs.cr3_craft_click_mouse_shell_env import Cr3CraftClickMouseShellEnv

    env = Cr3CraftClickMouseShellEnv()
    try:
        obs, info = env.reset()
        assert env.action_space.shape == (22,)
        init_act = env.get_initial_action()
        assert init_act.shape == (22,)
        assert obs["state"]["tcp_pose"].shape == (7,)
        assert obs["state"]["craft_qpos"].shape == (20,)
        assert "front" in obs["images"]
        assert "succeed" in info

        action = env.get_initial_action()
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "mouse_in_pad" in info
        assert "click_detected" in info
        assert "display_blue" in info
    finally:
        env.close()


def test_shell_has_ee_pose_matrix():
    from dexjoco.sim.envs.cr3_craft_click_mouse_shell_env import Cr3CraftClickMouseShellEnv

    env = Cr3CraftClickMouseShellEnv()
    try:
        env.reset()
        ee = env.get_end_effector_pose_matrix()
        assert ee.shape == (4, 4)
        assert np.allclose(ee[3, :], [0, 0, 0, 1])
    finally:
        env.close()


def test_shell_qacc_finite_after_step():
    from dexjoco.sim.envs.cr3_craft_click_mouse_shell_env import Cr3CraftClickMouseShellEnv

    env = Cr3CraftClickMouseShellEnv()
    try:
        env.reset()
        action = env.get_initial_action()
        for _ in range(5):
            obs, _reward, _term, _trunc, _info = env.step(action)
        assert np.isfinite(env.data.qacc).all(), "qacc contains NaN/Inf after steps"
    finally:
        env.close()
