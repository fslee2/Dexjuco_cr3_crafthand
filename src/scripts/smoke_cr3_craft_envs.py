"""Smoke test for CR3+CRAFT DexJoCo envs: import, reset, step, action/obs shapes.

Usage:
    python scripts/smoke_cr3_craft_envs.py --env all --steps 2
    python scripts/smoke_cr3_craft_envs.py --env click_mouse_shell
    python scripts/smoke_cr3_craft_envs.py --env all --steps 10 --render-check

This script does NOT require HaMeR, MediaPipe, camera, or a display.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Dict, List, Tuple, Type

import numpy as np


def _ensure_mujoco_gl() -> None:
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"


def _env_name_to_class(name: str):
    if name == "reach_debug":
        from dexjoco.sim.envs.cr3_craft_reach_debug_env import Cr3CraftReachDebugEnv
        return Cr3CraftReachDebugEnv
    elif name == "click_mouse":
        from dexjoco.sim.envs.cr3_craft_click_mouse_env import Cr3CraftClickMouseEnv
        return Cr3CraftClickMouseEnv
    elif name == "click_mouse_shell":
        from dexjoco.sim.envs.cr3_craft_click_mouse_shell_env import Cr3CraftClickMouseShellEnv
        return Cr3CraftClickMouseShellEnv
    else:
        raise ValueError(f"Unknown env name: {name}")


def _all_env_names() -> List[str]:
    return ["reach_debug", "click_mouse", "click_mouse_shell"]


def _dict_keys(d: dict, prefix: str = "") -> List[str]:
    """Flatten nested dict keys for display, e.g. state.tcp_pose."""
    out = []
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.extend(_dict_keys(v, full))
        else:
            out.append(full)
    return out


def smoke_one_env(
    env_cls: Type,
    env_name: str,
    n_steps: int = 2,
    render_check: bool = False,
) -> Tuple[bool, str]:
    """Run reset + N steps on one env instance. Returns (ok, detail_str)."""
    env = None
    try:
        env = env_cls()
        lines: List[str] = []

        # reset
        obs, info = env.reset()
        lines.append(f"  reset ok")

        # action space
        act_shape = env.action_space.shape
        lines.append(f"  action_space shape: {act_shape}")

        # initial action
        init_act = env.get_initial_action()
        lines.append(f"  initial_action shape: {init_act.shape}")

        # observation keys
        lines.append(f"  obs keys: {list(obs.keys())}")
        flat_keys = _dict_keys(obs)
        lines.append(f"  obs flat keys: {flat_keys}")

        # state keys
        state_keys = list(obs.get("state", {}).keys())
        lines.append(f"  state keys: {state_keys}")

        # image keys and shapes
        img_keys: List[str] = []
        for k, v in obs.get("images", {}).items():
            img_keys.append(k)
            lines.append(f"  image {k} shape: {v.shape}")
        lines.append(f"  image keys: {img_keys}")

        # info keys (reset)
        lines.append(f"  info keys (reset): {list(info.keys())}")

        # end-effector matrix
        if hasattr(env, "get_end_effector_pose_matrix"):
            ee = env.get_end_effector_pose_matrix()
            lines.append(f"  ee_matrix shape: {ee.shape}")

        # step loop
        action = env.get_initial_action()
        for step_i in range(n_steps):
            obs, reward, terminated, truncated, info = env.step(action)
            lines.append(
                f"  step {step_i}: reward={float(reward):.6f} "
                f"terminated={terminated} truncated={truncated}"
            )
            lines.append(f"    info keys: {list(info.keys())}")

        # render check
        if render_check:
            img = env.render()
            lines.append(f"  render: {img.shape} dtype={img.dtype}")

        # close
        env.close()
        lines.append(f"  close ok")

        return True, "\n".join(lines)

    except Exception:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        return False, traceback.format_exc()


def main() -> int:
    _ensure_mujoco_gl()

    parser = argparse.ArgumentParser(
        description="Smoke test CR3+CRAFT DexJoCo environments",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="all",
        choices=["reach_debug", "click_mouse", "click_mouse_shell", "all"],
        help="Which env(s) to test (default: all)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2,
        help="Number of step() calls per env (default: 2)",
    )
    parser.add_argument(
        "--render-check",
        action="store_true",
        default=False,
        help="Also call render() and print shape (default: false)",
    )
    args = parser.parse_args()

    names = _all_env_names() if args.env == "all" else [args.env]

    failures: List[Tuple[str, str]] = []
    for name in names:
        env_cls = _env_name_to_class(name)
        print(f"[{name}] ({env_cls.__name__})")
        ok, detail = smoke_one_env(env_cls, name, n_steps=args.steps, render_check=args.render_check)
        print(detail)
        if ok:
            print(f"[{name}] PASS\n")
        else:
            print(f"[{name}] FAIL\n")
            failures.append((name, detail))

    if failures:
        print("=" * 60)
        print(f"FAILED: {len(failures)} env(s)")
        for name, detail in failures:
            print(f"\n--- {name} traceback ---")
            print(detail)
        return 1

    print(f"ALL {len(names)} env(s) passed smoke test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
