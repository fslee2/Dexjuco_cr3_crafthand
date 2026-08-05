"""Render small README media assets for the CR3+CRAFT showcase.

This utility renders a short MuJoCo GIF from the standalone CR3+CRAFT package.
Use --source-repo only when rendering from another checkout.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw" if os.name == "nt" else "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-repo",
        type=Path,
        default=Path(os.environ.get("DEXJOCO_ROOT", Path(__file__).resolve().parents[1])),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("assets/gifs"))
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--fps", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_repo = args.source_repo.resolve()
    sys.path.insert(0, str(source_repo / "dexjoco"))

    from dexjoco.sim.envs.cr3_craft_click_mouse_shell_env import (
        CRAFT_VALUE_MAX,
        Cr3CraftClickMouseShellEnv,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = Cr3CraftClickMouseShellEnv(render_mode="none")
    renderer = None
    frames = []
    try:
        env.reset()
        action0 = env.get_initial_action()
        home = action0[:3].copy()
        action = action0.copy()

        renderer = mujoco.Renderer(env.model, height=480, width=640)
        try:
            cam_id = env.model.camera("front").id
        except Exception:
            cam_id = -1

        for idx in range(max(3, args.frames)):
            phase = idx / max(1, args.frames - 1)
            action[:] = action0
            action[0] = home[0] + 0.08 * np.sin(2.0 * np.pi * phase)
            action[7:22] = CRAFT_VALUE_MAX * (0.15 + 0.55 * phase)
            env.step(action)
            mujoco.mj_forward(env.model, env.data)
            renderer.update_scene(env.data, cam_id)
            frames.append(renderer.render().copy())

        gif_path = args.out_dir / "cr3_craft_x_axis_teleop.gif"
        png_path = args.out_dir.parent / "images" / "cr3_craft_shell_preview.png"
        png_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(png_path, frames[len(frames) // 2])
        imageio.mimsave(gif_path, frames, fps=args.fps)
        print(f"wrote {png_path}")
        print(f"wrote {gif_path}")
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


if __name__ == "__main__":
    main()
