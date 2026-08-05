"""CR3 + CRAFT environments shipped with this standalone repository."""

from .cr3_craft_click_mouse_env import Cr3CraftClickMouseEnv
from .cr3_craft_click_mouse_shell_env import Cr3CraftClickMouseShellEnv
from .cr3_craft_reach_debug_env import Cr3CraftReachDebugEnv
from .panda_hammer_nail_env import PandaHammerNailGymEnv
from .panda_pick_bucket_env import PandaPickBucketGymEnv
from .panda_pinch_tongs_env import PandaPinchTongsGymEnv
from .panda_water_plant_env import PandaWaterPlantGymEnv
from .panda_fold_glasses_env import PandaFoldGlassesGymEnv

Cr3CraftHammerNailEnv = PandaHammerNailGymEnv
Cr3CraftPickBucketEnv = PandaPickBucketGymEnv
Cr3CraftPinchTongsEnv = PandaPinchTongsGymEnv
Cr3CraftWaterPlantEnv = PandaWaterPlantGymEnv
Cr3CraftFoldGlassesEnv = PandaFoldGlassesGymEnv

__all__ = [
    "Cr3CraftClickMouseEnv",
    "Cr3CraftClickMouseShellEnv",
    "Cr3CraftReachDebugEnv",
    "Cr3CraftHammerNailEnv",
    "Cr3CraftPickBucketEnv",
    "Cr3CraftPinchTongsEnv",
    "Cr3CraftWaterPlantEnv",
    "Cr3CraftFoldGlassesEnv",
]
