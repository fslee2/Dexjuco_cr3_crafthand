"""Smoke test for the five CR3+CRAFT manipulation task scenes."""

from __future__ import annotations

import importlib
import traceback

import numpy as np


TASKS = (
    ("hammer_nail", "panda_hammer_nail_env", "PandaHammerNailGymEnv", {"render_mode": "rgb_array", "randomize": False}),
    ("pick_bucket", "panda_pick_bucket_env", "PandaPickBucketGymEnv", {"render_mode": "rgb_array", "randomize": False}),
    ("pinch_tongs", "panda_pinch_tongs_env", "PandaPinchTongsGymEnv", {"render_mode": "rgb_array", "randomize": False}),
    ("water_plant", "panda_water_plant_env", "PandaWaterPlantGymEnv", {"render_mode": "rgb_array", "randomize": False}),
    ("fold_glasses", "panda_fold_glasses_env", "PandaFoldGlassesGymEnv", {"render_mode": "rgb_array", "randomize": False}),
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
