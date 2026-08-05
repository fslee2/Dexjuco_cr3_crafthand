"""Smoke tests for Quest 3 teleop bridge; no Quest device or network required."""

import numpy as np
import pytest

from dexjoco.tasks.quest_teleop import (
    QuestActionMapper,
    QuestStateStore,
    QuestTeleopConfig,
    QuestTeleopState,
    SyntheticQuestSource,
    _parse_quest_packet,
)


# State dataclass.


def test_state_defaults():
    s = QuestTeleopState()
    assert not s.valid
    assert not s.connected
    assert s.right_pos.shape == (3,)
    assert np.allclose(s.right_quat_xyzw, [0, 0, 0, 1])
    assert s.close_scalar == 0.0


def test_state_wxyz_property():
    s = QuestTeleopState(right_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]))
    wxyz = s.right_quat_wxyz
    assert np.allclose(wxyz, [1.0, 0.0, 0.0, 0.0])


def test_state_close_scalar():
    s = QuestTeleopState(trigger=0.3, grip=0.7, pinch=0.2)
    assert s.close_scalar == 0.7


# State store (thread-safe).


def test_store_update_and_get():
    store = QuestStateStore()
    s = QuestTeleopState(valid=True, right_pos=np.array([1.0, 2.0, 3.0]))
    store.update(s)
    got = store.get()
    assert got.valid
    assert np.allclose(got.right_pos, [1.0, 2.0, 3.0])


def test_store_mark_invalid():
    store = QuestStateStore()
    store.update(QuestTeleopState(valid=True))
    store.mark_invalid()
    assert not store.get().valid


# Config.


def test_config_defaults():
    cfg = QuestTeleopConfig()
    assert cfg.pose_scale == 1.5
    assert cfg.max_step == 0.05


def test_config_bad_alpha():
    with pytest.raises(ValueError):
        QuestTeleopConfig(smoothing_alpha=0.0)
    with pytest.raises(ValueError):
        QuestTeleopConfig(smoothing_alpha=1.5)


# Packet parsing.


def test_parse_valid_packet():
    payload = (
        b'{"type":"quest_pose","timestamp":1.0,'
        b'"right":{"pos":[1,2,3],"quat":[0,0,0,1]},'
        b'"trigger":0.5,"grip":0.3,"pinch":0.1}'
    )
    s = _parse_quest_packet(payload)
    assert s is not None
    assert s.valid
    assert np.allclose(s.right_pos, [1, 2, 3])
    assert s.trigger == 0.5
    assert s.grip == 0.3


def test_parse_invalid_type():
    payload = b'{"type":"other"}'
    assert _parse_quest_packet(payload) is None


def test_parse_malformed_json():
    assert _parse_quest_packet(b"not json") is None


# Action mapper: shape and hold.


def test_mapper_hold_action_shape():
    cfg = QuestTeleopConfig()
    mapper = QuestActionMapper(cfg)
    ee = np.eye(4)
    state = QuestTeleopState(valid=False)
    action = mapper.map(state, ee)
    assert action.shape == (22,)
    # craft15 defaults to open (hand_open_scalar=0.0)
    assert np.allclose(action[7:22], 0.0)


# Action mapper: calibration on first valid state.


def test_mapper_calibrates_on_first_valid():
    cfg = QuestTeleopConfig(max_step=999.0)  # large so clamp doesn't interfere
    mapper = QuestActionMapper(cfg)
    ee = np.eye(4)
    ee[:3, 3] = [0.5, 0.0, 0.8]

    state = QuestTeleopState(
        valid=True,
        right_pos=np.array([0.1, 0.0, 0.0]),
        right_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    action = mapper.map(state, ee)
    assert mapper.is_calibrated
    # With zero delta, target should be near ee_start
    assert np.allclose(action[:3], ee[:3, 3], atol=0.01)


# Action mapper: position delta moves target_xyz.


def test_mapper_position_delta_changes_target():
    cfg = QuestTeleopConfig(pose_scale=2.0, max_step=999.0)
    mapper = QuestActionMapper(cfg)
    ee = np.eye(4)
    ee[:3, 3] = [0.5, 0.0, 0.8]

    # First state calibrates.
    s0 = QuestTeleopState(
        valid=True,
        right_pos=np.array([0.0, 0.0, 0.0]),
        right_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    mapper.map(s0, ee)

    # Second state moves right by 0.1.
    s1 = QuestTeleopState(
        valid=True,
        right_pos=np.array([0.1, 0.0, 0.0]),
        right_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        trigger=1.0,
        grip=1.0,
    )
    action = mapper.map(s1, ee)
    # With smoothing_alpha=0.35, filtered is about 0.5 + 0.35 * 0.2 = 0.57.
    assert action[0] > 0.55  # moved from 0.5 toward 0.7


# Action mapper: craft15 responds to trigger/grip.


def test_mapper_craft15_responds_to_close():
    cfg = QuestTeleopConfig(max_step=999.0)
    mapper = QuestActionMapper(cfg)
    ee = np.eye(4)

    s0 = QuestTeleopState(
        valid=True,
        right_pos=np.array([0.0, 0.0, 0.0]),
        right_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    mapper.map(s0, ee)

    # Open hand
    s_open = QuestTeleopState(
        valid=True,
        right_pos=np.array([0.0, 0.0, 0.0]),
        right_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        trigger=0.0,
        grip=0.0,
    )
    action_open = mapper.map(s_open, ee)
    assert np.allclose(action_open[7:22], 0.0)

    # Closed hand
    s_closed = QuestTeleopState(
        valid=True,
        right_pos=np.array([0.0, 0.0, 0.0]),
        right_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        trigger=1.0,
        grip=1.0,
    )
    action_closed = mapper.map(s_closed, ee)
    assert np.all(action_closed[7:22] > 0.0)


# WebSocketReceiver TLS constructor validation.


def test_receiver_certfile_only_raises():
    """certfile without keyfile must raise ValueError."""
    from dexjoco.tasks.quest_teleop import QuestWebSocketReceiver

    with pytest.raises(ValueError, match="certfile and keyfile must be provided together"):
        QuestWebSocketReceiver(certfile="cert.pem")


def test_receiver_keyfile_only_raises():
    """keyfile without certfile must raise ValueError."""
    from dexjoco.tasks.quest_teleop import QuestWebSocketReceiver

    with pytest.raises(ValueError, match="certfile and keyfile must be provided together"):
        QuestWebSocketReceiver(keyfile="key.pem")


def test_receiver_no_tls_no_error():
    """Neither certfile nor keyfile: no ValueError, ssl_context is None."""
    import importlib

    try:
        importlib.import_module("websockets")
    except ImportError:
        pytest.skip("websockets not installed")

    from dexjoco.tasks.quest_teleop import QuestWebSocketReceiver

    r = QuestWebSocketReceiver(host="127.0.0.1", port=0)
    assert r._ssl_context is None


def test_receiver_both_tls_creates_ssl_context(tmp_path):
    """Both certfile and keyfile: SSL context created."""
    import importlib
    import subprocess
    import sys

    try:
        importlib.import_module("websockets")
    except ImportError:
        pytest.skip("websockets not installed")

    cert_path = tmp_path / "cert.pem"
    key_path = tmp_path / "key.pem"

    # Generate a self-signed cert with openssl.
    result = subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", str(key_path), "-out", str(cert_path),
            "-days", "1", "-nodes", "-subj", "/CN=localhost",
        ],
        capture_output=True, timeout=30,
    )
    if result.returncode != 0:
        pytest.skip("openssl not available or failed")

    from dexjoco.tasks.quest_teleop import QuestWebSocketReceiver

    r = QuestWebSocketReceiver(
        host="127.0.0.1", port=0,
        certfile=str(cert_path), keyfile=str(key_path),
    )
    assert r._ssl_context is not None


# Synthetic source.


def test_synthetic_source_produces_valid_state():
    store = QuestStateStore()
    src = SyntheticQuestSource(store)
    state = src.step()
    assert state.valid
    assert state.source == "synthetic"
    assert state.right_pos.shape == (3,)
    assert state.close_scalar >= 0.0


def test_synthetic_source_positions_change():
    store = QuestStateStore()
    src = SyntheticQuestSource(store)
    s0 = src.step()
    s1 = src.step()
    # Position should differ (circular motion)
    assert not np.allclose(s0.right_pos, s1.right_pos)


# Synthetic source: full env integration smoke.


def test_synthetic_full_env_loop():
    """Run the full env loop with synthetic Quest data for a few steps."""
    import os
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"

    from dexjoco.sim.envs.cr3_craft_click_mouse_shell_env import (
        CRAFT_COMMAND_JOINTS,
        Cr3CraftClickMouseShellEnv,
    )

    config = QuestTeleopConfig(auto_enable=True)
    env = Cr3CraftClickMouseShellEnv()

    try:
        store = QuestStateStore()
        mapper = QuestActionMapper(config, craft_command_joints=CRAFT_COMMAND_JOINTS)
        from dexjoco.tasks.quest_teleop import SingleArmQuestTeleopWrapper
        wrapper = SingleArmQuestTeleopWrapper(env, store, mapper, config)
        synthetic = SyntheticQuestSource(store)

        obs, info = wrapper.reset()
        assert obs["state"]["tcp_pose"].shape == (7,)
        assert obs["state"]["craft_qpos"].shape == (20,)

        for _ in range(10):
            synthetic.step()
            obs, reward, terminated, truncated, info = wrapper.step(
                env.get_initial_action()
            )
            assert np.isfinite(reward)
            ia = info.get("intervene_action")
            assert ia is not None
            assert ia.shape == (22,)

        assert not terminated  # Should not finish click task from small random moves.
    finally:
        env.close()
        try:
            wrapper.close()
        except Exception:
            pass
