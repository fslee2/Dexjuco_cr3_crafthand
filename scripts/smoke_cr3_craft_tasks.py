"""Smoke test for the five CR3+CRAFT manipulation task scenes."""

from __future__ import annotations

import importlib
import traceback

import numpy as np


TASKS = (
    ("hammer_nail", "cr3_craft_hammer_nail_env", "Cr3CraftHammerNailEnv", {"render_mode": "rgb_array", "randomize": False}),
    ("pick_bucket", "cr3_craft_pick_bucket_env", "Cr3CraftPickBucketEnv", {"render_mode": "rgb_array", "randomize": False}),
    ("pinch_tongs", "cr3_craft_pinch_tongs_env", "Cr3CraftPinchTongsEnv", {"render_mode": "rgb_array", "randomize": False}),
    ("water_plant", "cr3_craft_water_plant_env", "Cr3CraftWaterPlantEnv", {"render_mode": "rgb_array", "randomize": False}),
    ("fold_glasses", "cr3_craft_fold_glasses_env", "Cr3CraftFoldGlassesEnv", {"render_mode": "rgb_array", "randomize": False}),
)


def main() -> int:
    failures = 0
    for name, module_name, class_name, kwargs in TASKS:
        env = None
        try:
            module = importlib.import_module(f"dexjoco.sim.envs.{module_name}")
            env = getattr(module, class_name)(hand="craft", **kwargs)
            obs, _ = env.reset(seed=0)
            obs, reward, terminated, truncated, info = env.step(np.zeros(22, dtype=np.float32))
            print(f"{name}: PASS action={env.action_space.shape} info={sorted(info)}")
        except Exception:
            failures += 1
            print(f"{name}: FAIL")
            traceback.print_exc()
        finally:
            if env is not None:
                env.close()
    print(f"TASK_TOTAL_FAILED={failures}")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
