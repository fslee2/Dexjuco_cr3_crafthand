"""Quest 3 WebXR teleop driver for Cr3CraftClickMouseShellEnv.

Usage:
    # Smoke test with synthetic data (no Quest, no network)
    python scripts/quest_cr3_craft_click_mouse_shell.py --synthetic --steps 20

    # Real Quest WebSocket server + viewer
    python scripts/quest_cr3_craft_click_mouse_shell.py --viewer

    # Custom host/port
    python scripts/quest_cr3_craft_click_mouse_shell.py --host 0.0.0.0 --port 8765 --viewer

    # WSS (secure WebSocket) with TLS certificate
    python scripts/quest_cr3_craft_click_mouse_shell.py --viewer \
        --certfile cert.pem --keyfile key.pem
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np


def _ensure_mujoco_gl() -> None:
    """Set MUJOCO_GL=egl only on MuJoCo 2.x; MuJoCo 3.x removed the env var."""
    if "MUJOCO_GL" in os.environ:
        return
    try:
        import mujoco
        v = tuple(int(x) for x in mujoco.__version__.split(".")[:2])
        if v < (3, 0):
            os.environ["MUJOCO_GL"] = "egl"
    except Exception:
        os.environ["MUJOCO_GL"] = "egl"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quest 3 WebXR teleop driver for CR3+CRAFT click_mouse_shell env",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="WebSocket server host")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket server port")
    parser.add_argument("--viewer", action="store_true", default=False, help="Open MuJoCo viewer")
    parser.add_argument("--steps", type=int, default=2000, help="Max environment steps")
    parser.add_argument("--pose-scale", type=float, default=1.5, help="Position delta scaling")
    parser.add_argument("--max-step", type=float, default=0.05, help="Per-step position clamp")
    parser.add_argument("--synthetic", action="store_true", default=False,
                        help="Use synthetic Quest data (no network required)")
    parser.add_argument("--no-server", action="store_true", default=False,
                        help="Do not start WebSocket server (use with --synthetic)")
    parser.add_argument("--auto-enable", action="store_true", default=False,
                        help="Enable teleop immediately without key press")
    parser.add_argument("--certfile", type=str, default=None,
                        help="TLS certificate file for WSS (requires --keyfile)")
    parser.add_argument("--keyfile", type=str, default=None,
                        help="TLS private key file for WSS (requires --certfile)")
    return parser.parse_args()


def main() -> int:
    _ensure_mujoco_gl()
    args = _parse_args()

    # Late imports after env vars are set.
    from dexjoco.sim.envs.cr3_craft_click_mouse_shell_env import (
        CRAFT_COMMAND_JOINTS,
        Cr3CraftClickMouseShellEnv,
    )
    from dexjoco.tasks.quest_teleop import (
        QuestActionMapper,
        QuestStateStore,
        QuestTeleopConfig,
        SingleArmQuestTeleopWrapper,
        SyntheticQuestSource,
    )

    # Config.
    config = QuestTeleopConfig(
        pose_scale=args.pose_scale,
        max_step=args.max_step,
        hand_close_scalar=1.0,
        hand_open_scalar=0.0,
        smoothing_alpha=0.35,
        auto_enable=args.auto_enable or args.synthetic,
        debug_log_interval=50,
    )

    # Env.
    render_mode = "human" if args.viewer else "rgb_array"
    env = Cr3CraftClickMouseShellEnv(render_mode=render_mode)

    store = QuestStateStore()
    mapper = QuestActionMapper(config, craft_command_joints=CRAFT_COMMAND_JOINTS)
    wrapper = SingleArmQuestTeleopWrapper(env, store, mapper, config)

    # Quest source (real or synthetic).
    receiver: Optional[object] = None
    synthetic_src: Optional[SyntheticQuestSource] = None

    if args.synthetic or args.no_server:
        synthetic_src = SyntheticQuestSource(store)
        print("[Quest] Using synthetic data source.")
    else:
        from dexjoco.tasks.quest_teleop import QuestWebSocketReceiver

        try:
            receiver = QuestWebSocketReceiver(
                host=args.host, port=args.port, store=store,
                certfile=args.certfile, keyfile=args.keyfile,
            )
            receiver.start()
            if receiver.is_running:
                scheme = "wss" if (args.certfile and args.keyfile) else "ws"
                print(f"[Quest] WebSocket server listening on {scheme}://{args.host}:{args.port}")
                print(f"[Quest] Quest Browser URL: {scheme}://<your-ip>:{args.port}")
            else:
                print("[Quest] WARNING: WebSocket server failed to start. "
                      "Falling back to synthetic source.")
                synthetic_src = SyntheticQuestSource(store)
        except RuntimeError as e:
            print(f"[Quest] {e}")
            print("[Quest] Falling back to synthetic source.")
            synthetic_src = SyntheticQuestSource(store)

    # Reset.
    obs, info = wrapper.reset()
    print(f"[Env] Reset complete. Action shape: {env.action_space.shape}")
    print(f"[Env] Initial EE pose: {np.round(obs['state']['tcp_pose'], 4)}")
    if args.viewer:
        print("[Env] Close the viewer window or press Ctrl+C to stop.")

    # Main loop.
    action = env.get_initial_action()
    start_time = time.time()
    last_print = 0

    try:
        for step_i in range(args.steps):
            if synthetic_src is not None:
                synthetic_src.step()

            obs, reward, terminated, truncated, info = wrapper.step(action)

            now = time.time()
            if now - last_print >= 2.0:
                tcp = obs["state"]["tcp_pose"]
                print(
                    f"[Step {step_i}] "
                    f"tcp_xyz={np.round(tcp[:3], 4)} "
                    f"reward={float(reward):.4f} "
                    f"tcp_err={info.get('tcp_error', float('nan')):.4f}"
                )
                last_print = now
                if synthetic_src is not None:
                    state = store.get()
                    print(
                        f"  synth: pos={np.round(state.right_pos, 4)} "
                        f"close={state.close_scalar:.2f}"
                    )

            if terminated or truncated:
                print(f"[Env] Episode ended at step {step_i}: "
                      f"terminated={terminated} truncated={truncated} "
                      f"succeed={info.get('succeed', False)}")
                break

            # Manual reset via info
            if info.get("manual_reset"):
                print("[Env] Manual reset triggered.")
                obs, info = wrapper.reset()

    except KeyboardInterrupt:
        print("\n[Quest] Interrupted by user.")

    finally:
        elapsed = time.time() - start_time
        print(f"[Quest] Ran {step_i + 1} steps in {elapsed:.1f}s "
              f"({(step_i + 1) / max(elapsed, 0.001):.1f} steps/s)")

        if receiver is not None:
            receiver.stop()
            print("[Quest] WebSocket server stopped.")
        wrapper.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
