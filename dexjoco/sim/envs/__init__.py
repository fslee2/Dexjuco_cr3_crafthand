"""CR3 + CRAFT environments shipped with this standalone repository."""

from .cr3_craft_click_mouse_env import Cr3CraftClickMouseEnv
from .cr3_craft_click_mouse_shell_env import Cr3CraftClickMouseShellEnv
from .cr3_craft_reach_debug_env import Cr3CraftReachDebugEnv
from .cr3_craft_hammer_nail_env import Cr3CraftHammerNailEnv
from .cr3_craft_pick_bucket_env import Cr3CraftPickBucketEnv
from .cr3_craft_pinch_tongs_env import Cr3CraftPinchTongsEnv
from .cr3_craft_water_plant_env import Cr3CraftWaterPlantEnv
from .cr3_craft_fold_glasses_env import Cr3CraftFoldGlassesEnv

# Legacy aliases kept for existing DexJoCo task configs and downstream scripts.
PandaHammerNailGymEnv = Cr3CraftHammerNailEnv
PandaPickBucketGymEnv = Cr3CraftPickBucketEnv
PandaPinchTongsGymEnv = Cr3CraftPinchTongsEnv
PandaWaterPlantGymEnv = Cr3CraftWaterPlantEnv
PandaFoldGlassesGymEnv = Cr3CraftFoldGlassesEnv

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
