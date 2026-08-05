"""Hand model composition and Allegro-compatible CRAFT control mapping."""

from pathlib import Path
from typing import Literal

import mujoco
import numpy as np

HandType = Literal["allegro", "craft"]

ALLEGRO_JOINT_NAMES = (
    "ffj0",
    "ffj1",
    "ffj2",
    "ffj3",
    "mfj0",
    "mfj1",
    "mfj2",
    "mfj3",
    "rfj0",
    "rfj1",
    "rfj2",
    "rfj3",
    "thj0",
    "thj1",
    "thj2",
    "thj3",
)
ALLEGRO_ACTUATOR_NAMES = (
    "ffa0",
    "ffa1",
    "ffa2",
    "ffa3",
    "mfa0",
    "mfa1",
    "mfa2",
    "mfa3",
    "rfa0",
    "rfa1",
    "rfa2",
    "rfa3",
    "tha0",
    "tha1",
    "tha2",
    "tha3",
)
ALLEGRO_HOME = np.asarray(
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.263, 0, 0, 0),
    dtype=np.float64,
)
ALLEGRO_CTRL_RANGE = np.asarray(
    (
        *((-0.47, 0.47), (-0.196, 1.61), (-0.174, 1.709), (-0.227, 1.618))
        * 3,
        (0.263, 1.396),
        (-0.105, 1.163),
        (-0.189, 1.644),
        (-0.162, 1.719),
    ),
    dtype=np.float64,
)
ALLEGRO_CTRL_LOW = ALLEGRO_CTRL_RANGE[:, 0]
ALLEGRO_CTRL_HIGH = ALLEGRO_CTRL_RANGE[:, 1]

CRAFT_PREFIX = "craft_"
CRAFT_ACTUATOR_NAMES = tuple(
    CRAFT_PREFIX + name
    for name in (
        "index_1",
        "index_2",
        "index_3",
        "middle_1",
        "middle_2",
        "middle_3",
        "ring_1",
        "ring_2",
        "ring_3",
        "pinky_1",
        "pinky_2",
        "pinky_3",
        "thumb_mcp",
        "thumb_2",
        "thumb_3",
    )
)
CRAFT_JOINT_NAMES = tuple(
    CRAFT_PREFIX + name
    for name in (
        "index_1",
        "index_2",
        "index_3",
        "index_4",
        "middle_1",
        "middle_2",
        "middle_3",
        "middle_4",
        "ring_1",
        "ring_2",
        "ring_3",
        "ring_4",
        "pinky_1",
        "pinky_2",
        "pinky_3",
        "pinky_4",
        "thumb_mcp",
        "thumb_2",
        "thumb_3",
        "thumb_4",
    )
)
CRAFT_ACTUATED_JOINT_INDICES = np.asarray(
    (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18)
)
CRAFT_CTRL_RANGE = np.asarray(
    ((-1.571, 1.571), (-0.392, 2.0), (0.0, 1.920)) * 5,
    dtype=np.float64,
)
CRAFT_CTRL_LOW = CRAFT_CTRL_RANGE[:, 0]
CRAFT_CTRL_HIGH = CRAFT_CTRL_RANGE[:, 1]
CRAFT_VALUE_MAX = 2.0 * np.pi
CRAFT_COMMAND_ACTUATOR_NAMES = tuple(
    CRAFT_PREFIX + name
    for name in (
        # HaMeR/CRAFT command order:
        # Ring, Index, Thumb, Middle, Pinky, each [PIP, MCP forward, MCP side].
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
)
CRAFT_FINGER_SIDEWAYS_SCALE = 0.1
CRAFT_COUPLED_MEDIAL_DISTAL_SCALE = 0.75
CRAFT_THUMB_PROXIMAL_HIGH_SCALE = 0.5
# CRAFT permits much wider non-thumb abduction/adduction than Allegro. Keep
# the XML limits intact, but use this narrower range at the compatibility layer.
CRAFT_MAPPING_CTRL_RANGE = CRAFT_CTRL_RANGE.copy()
_CRAFT_FINGER_SIDEWAYS_INDICES = np.asarray((0, 3, 6, 9))
CRAFT_MAPPING_CTRL_RANGE[_CRAFT_FINGER_SIDEWAYS_INDICES] *= (
    CRAFT_FINGER_SIDEWAYS_SCALE
)
_CRAFT_COUPLED_MEDIAL_DISTAL_INDICES = np.asarray((2, 5, 8, 11))
CRAFT_MAPPING_CTRL_RANGE[_CRAFT_COUPLED_MEDIAL_DISTAL_INDICES, 1] = (
    CRAFT_MAPPING_CTRL_RANGE[_CRAFT_COUPLED_MEDIAL_DISTAL_INDICES, 0]
    + CRAFT_COUPLED_MEDIAL_DISTAL_SCALE
    * np.diff(
        CRAFT_MAPPING_CTRL_RANGE[_CRAFT_COUPLED_MEDIAL_DISTAL_INDICES],
        axis=1,
    ).ravel()
)

# Keep the CRAFT thumb's physical range broad, but start the thumb MCP
# compatibility range from Allegro thj0's lower/home-adjacent limit. This avoids
# mapping Allegro home to CRAFT's fully tucked physical lower stop.
CRAFT_MAPPING_CTRL_RANGE[12, 0] = ALLEGRO_CTRL_RANGE[12, 0]
CRAFT_MAPPING_CTRL_RANGE[13, 1] = (
    ALLEGRO_CTRL_RANGE[13, 1] * CRAFT_THUMB_PROXIMAL_HIGH_SCALE
)

CRAFT_MAPPING_CTRL_LOW = CRAFT_MAPPING_CTRL_RANGE[:, 0]
CRAFT_MAPPING_CTRL_HIGH = CRAFT_MAPPING_CTRL_RANGE[:, 1]

_CRAFT_REPO = Path(__file__).resolve().parents[5] / "CRAFT-hand-MJCF"
_CRAFT_XML = _CRAFT_REPO / "craft_hand_robosuite.xml"

# Solved from corresponding neutral-pose finger-link mesh centers. This places
# CRAFT's finger length and spread axes in the same attachment-site frame as
# the original Allegro hand.
CRAFT_MOUNT_POS = np.asarray((-0.01324133, -0.01188763, -0.05111532))
CRAFT_MOUNT_QUAT = np.asarray(
    (0.50200836, -0.49846561, -0.49800248, 0.50151089)
)

CRAFT_DIGIT_QUATS = {
    # Keep the thumb outward-facing, but pitch it slightly toward the other
    # fingers so the palm-down reset is less downward-facing.
    # (w, x, y, z) 
    # +ve x rotates clockwise around thumb's principal axis
    # +ve y moves thumb towards the direction palm is facing 
    # +ve z moves thumb towards the other fingers
    "thumbmcp": (0.900949749, 0, 0.42648511, 0.08),
}
# Fitted in the standard Allegro-aligned CRAFT body frames. These keep the
# first two CRAFT thumb controls in Allegro-like directions while preserving
# the broader CRAFT thumb control ranges.
CRAFT_ALLEGRO_THUMB_JOINTS = {
    "thumb_mcp": {
        # Positive MCP motion should steer the thumb toward the palm center,
        # not directly toward the other finger roots.
        # +ve x makes mcp action rotate thumb towards arm (back of wrist)
        # +ve y makes mcp action rotate thumb away from other fingers
        # +ve z makes mcp action rotate thumb to create opposition with other fingers
        "axis": (-0.7416198, 0.3, 0.6),
    },
    "thumb_2": {
        "axis": (-0.31841560, -0.42771621, -0.84597302),
        "pos": (-0.02210169, -0.12649075, -0.04814923),
    },
}
CRAFT_MOVING_BODY_NAMES = tuple(
    f"{digit}{link}"
    for digit in ("index", "middle", "ring", "pinky")
    for link in (1, 2, 3, 4)
) + ("thumbmcp", "thumb2", "thumb3", "thumb4")


def _normalize(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _denormalize(percent: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return low + np.clip(percent, 0.0, 1.0) * (high - low)


def allegro_to_craft(allegro: np.ndarray) -> np.ndarray:
    """Map Allegro targets to equivalent percentages of CRAFT actuator ranges."""
    allegro = np.asarray(allegro, dtype=np.float64)
    if allegro.shape != (16,):
        raise ValueError(f"Expected 16 Allegro targets, got shape {allegro.shape}.")

    def digit(values: np.ndarray) -> tuple[float, float, float]:
        return float(values[0]), float(values[1]), float(values[2])

    percent = _normalize(allegro, ALLEGRO_CTRL_LOW, ALLEGRO_CTRL_HIGH)
    index = digit(percent[0:4])
    middle = digit(percent[4:8])
    ring = digit(percent[8:12])
    thumb = digit(percent[12:16])
    craft_percent = np.asarray(
        (*index, *middle, *ring, *ring, *thumb), dtype=np.float64
    )
    return _denormalize(
        craft_percent, CRAFT_MAPPING_CTRL_LOW, CRAFT_MAPPING_CTRL_HIGH
    )


def craft_to_allegro(craft: np.ndarray) -> np.ndarray:
    """Expose CRAFT state as equivalent percentages of Allegro control ranges."""
    craft = np.asarray(craft, dtype=np.float64)
    if craft.shape != (15,):
        raise ValueError(f"Expected 15 CRAFT targets, got shape {craft.shape}.")

    def digit(values: np.ndarray) -> tuple[float, float, float, float]:
        return float(values[0]), float(values[1]), float(values[2]), float(values[2])

    craft = np.clip(craft, CRAFT_MAPPING_CTRL_LOW, CRAFT_MAPPING_CTRL_HIGH)
    percent = _normalize(craft, CRAFT_MAPPING_CTRL_LOW, CRAFT_MAPPING_CTRL_HIGH)
    ring = 0.5 * (percent[6:9] + percent[9:12])
    allegro_percent = np.asarray(
        (
            *digit(percent[0:3]),
            *digit(percent[3:6]),
            *digit(ring),
            *digit(percent[12:15]),
        ),
        dtype=np.float64,
    )
    return _denormalize(allegro_percent, ALLEGRO_CTRL_LOW, ALLEGRO_CTRL_HIGH)


def craft_joint_positions(craft_targets: np.ndarray) -> np.ndarray:
    """Expand 15 CRAFT actuator targets to the 20 coupled joint positions."""
    craft_targets = np.asarray(craft_targets, dtype=np.float64)
    if craft_targets.shape != (15,):
        raise ValueError(f"Expected 15 CRAFT targets, got shape {craft_targets.shape}.")

    positions = np.empty(20, dtype=np.float64)
    for digit in range(5):
        actuator_start = 3 * digit
        joint_start = 4 * digit
        positions[joint_start : joint_start + 3] = craft_targets[
            actuator_start : actuator_start + 3
        ]
        positions[joint_start + 3] = craft_targets[actuator_start + 2]
    return positions


def configure_allegro_compatible_craft_thumb(craft: mujoco.MjSpec) -> None:
    """Apply reusable Allegro-compatible kinematics to a CRAFT thumb spec."""
    for joint_name, properties in CRAFT_ALLEGRO_THUMB_JOINTS.items():
        joint = craft.joint(joint_name)
        joint.axis = properties["axis"]
        if "pos" in properties:
            joint.pos = properties["pos"]


def align_craft_spec_to_allegro(craft: mujoco.MjSpec) -> None:
    """Align a standalone CRAFT spec with DexJoCo's Allegro attachment frame."""
    craft_base = craft.body("base")
    craft_base.pos = CRAFT_MOUNT_POS
    craft_base.quat = CRAFT_MOUNT_QUAT

    for body_name, quat in CRAFT_DIGIT_QUATS.items():
        craft.body(body_name).quat = quat
    configure_allegro_compatible_craft_thumb(craft)


def compose_craft_hand_model(arena_xml: Path) -> mujoco.MjModel:
    """Compose an Allegro-palm / CRAFT-finger hybrid into a single-arm arena."""
    if not _CRAFT_XML.is_file():
        raise FileNotFoundError(
            f"CRAFT model not found at {_CRAFT_XML}. Keep CRAFT-hand-MJCF next to dexjoco."
        )

    scene = mujoco.MjSpec.from_file(str(arena_xml))
    for name in ALLEGRO_ACTUATOR_NAMES:
        scene.delete(scene.actuator(name))
    for sensor in list(scene.sensors):
        if sensor.name.startswith("allegro_right/"):
            scene.delete(sensor)
    for exclude in list(scene.excludes):
        if exclude.bodyname1.startswith("allegro_"):
            scene.delete(exclude)
    for name in ("ff_base", "mf_base", "rf_base", "th_base"):
        scene.delete(scene.body(name))

    craft = mujoco.MjSpec.from_file(str(_CRAFT_XML))
    align_craft_spec_to_allegro(craft)
    for geom in list(craft.body("palm").geoms):
        craft.delete(geom)
    craft.delete(craft.body("eef"))
    scene.attach(craft, site="attachment_site", prefix=CRAFT_PREFIX)
    for body_name in CRAFT_MOVING_BODY_NAMES:
        exclude = scene.add_exclude()
        exclude.bodyname1 = "allegro_palm"
        exclude.bodyname2 = CRAFT_PREFIX + body_name
    return scene.compile()


def model_for_hand(arena_xml: Path, hand: HandType) -> mujoco.MjModel | None:
    """Return a composed model when the selected hand differs from the arena XML."""
    if hand not in ("allegro", "craft"):
        raise ValueError(f"Unsupported hand {hand!r}; expected 'allegro' or 'craft'.")
    return compose_craft_hand_model(arena_xml) if hand == "craft" else None


def hand_actuator_names(hand: HandType) -> tuple[str, ...]:
    return ALLEGRO_ACTUATOR_NAMES if hand == "allegro" else CRAFT_ACTUATOR_NAMES


def hand_joint_names(hand: HandType) -> tuple[str, ...]:
    return ALLEGRO_JOINT_NAMES if hand == "allegro" else CRAFT_JOINT_NAMES


def hand_home(hand: HandType) -> np.ndarray:
    if hand == "allegro":
        return ALLEGRO_HOME.copy()
    return craft_joint_positions(allegro_to_craft(ALLEGRO_HOME))


def hand_targets(hand: HandType, allegro_targets: np.ndarray) -> np.ndarray:
    return allegro_targets if hand == "allegro" else allegro_to_craft(allegro_targets)


def allegro_compatible_state(hand: HandType, hand_state: np.ndarray) -> np.ndarray:
    return hand_state if hand == "allegro" else craft_to_allegro(hand_state)


def craft_physical_targets(craft_targets: np.ndarray) -> np.ndarray:
    """Map CRAFT 0..2pi command values into DexJoCo actuator order/ranges."""
    craft_targets = np.asarray(craft_targets, dtype=np.float64)
    if craft_targets.shape != (15,):
        raise ValueError(f"Expected 15 CRAFT targets, got shape {craft_targets.shape}.")

    targets_by_actuator = {}
    for command_value, actuator_name in zip(craft_targets, CRAFT_COMMAND_ACTUATOR_NAMES):
        actuator_index = CRAFT_ACTUATOR_NAMES.index(actuator_name)
        low, high = CRAFT_CTRL_RANGE[actuator_index]
        normalized = np.clip(float(command_value) / CRAFT_VALUE_MAX, 0.0, 1.0)
        targets_by_actuator[actuator_name] = low + normalized * (high - low)

    return np.asarray(
        [targets_by_actuator[name] for name in CRAFT_ACTUATOR_NAMES],
        dtype=np.float64,
    )


class SingleArmHand:
    """Adapt a physical single-arm hand to DexJoCo's 16-value Allegro interface."""

    def __init__(self, model: mujoco.MjModel, hand: HandType):
        if hand not in ("allegro", "craft"):
            raise ValueError(f"Unsupported hand {hand!r}; expected 'allegro' or 'craft'.")
        self.hand = hand
        self.model = model
        # CR3+CRAFT scenes embed the hand directly and use the historical
        # ``h_*`` actuator names. Composed Allegro-compatible scenes use the
        # prefixed ``craft_*`` names above. Support both layouts here so task
        # environments can share the same task logic.
        self._direct_cr3_craft = bool(
            hand == "craft"
            and model.nu > 0
            and mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                "h_index_1",
            ) >= 0
        )
        if self._direct_cr3_craft:
            self.joint_names = tuple(
                name.removeprefix("craft_") for name in CRAFT_JOINT_NAMES
            )
            self.actuator_names = tuple(
                "h_" + name.removeprefix("craft_") for name in CRAFT_ACTUATOR_NAMES
            )
        else:
            self.joint_names = hand_joint_names(hand)
            self.actuator_names = hand_actuator_names(hand)
        self.qpos_ids = np.asarray(
            [int(model.joint(name).qposadr[0]) for name in self.joint_names], dtype=int
        )
        self.ctrl_ids = np.asarray(
            [int(model.actuator(name).id) for name in self.actuator_names], dtype=int
        )
        self.controlled_qpos_ids = np.asarray(
            [
                int(model.joint(model.actuator(name).trnid[0]).qposadr[0])
                for name in self.actuator_names
            ],
            dtype=int,
        )
        self.ctrl_range = model.actuator_ctrlrange[self.ctrl_ids].copy()

    def physical_targets(self, policy_targets: np.ndarray) -> np.ndarray:
        """Map and clip 16 Allegro-compatible targets for the physical hand."""
        policy_targets = np.asarray(policy_targets, dtype=np.float64)
        if policy_targets.shape != (len(ALLEGRO_JOINT_NAMES),):
            raise ValueError(
                f"Expected {len(ALLEGRO_JOINT_NAMES)} policy hand targets, "
                f"got shape {policy_targets.shape}."
            )
        targets = hand_targets(self.hand, policy_targets)
        return np.clip(targets, self.ctrl_range[:, 0], self.ctrl_range[:, 1])

    def direct_targets(self, craft_targets: np.ndarray) -> np.ndarray:
        """Convert CRAFT command values into physical actuator targets."""
        if self.hand != "craft":
            raise ValueError("Direct CRAFT targets are only supported when hand='craft'.")
        craft_targets = np.asarray(craft_targets, dtype=np.float64)
        if self._direct_cr3_craft:
            values = {}
            for command_value, command_name in zip(
                craft_targets, CRAFT_COMMAND_ACTUATOR_NAMES
            ):
                actuator_name = "h_" + command_name.removeprefix("craft_")
                actuator_id = int(self.model.actuator(actuator_name).id)
                low, high = self.model.actuator_ctrlrange[actuator_id]
                values[actuator_name] = low + np.clip(
                    float(command_value) / CRAFT_VALUE_MAX, 0.0, 1.0
                ) * (high - low)
            return np.asarray(
                [values.get(name, self.ctrl_range[i, 0]) for i, name in enumerate(self.actuator_names)],
                dtype=np.float64,
            )
        return np.clip(
            craft_physical_targets(craft_targets),
            self.ctrl_range[:, 0],
            self.ctrl_range[:, 1],
        )

    def apply(self, data: mujoco.MjData, policy_targets: np.ndarray) -> None:
        """Write Allegro-compatible policy targets to the physical actuators."""
        data.ctrl[self.ctrl_ids] = self.physical_targets(policy_targets)

    def apply_direct(self, data: mujoco.MjData, craft_targets: np.ndarray) -> None:
        """Write direct CRAFT actuator targets to the physical actuators."""
        data.ctrl[self.ctrl_ids] = self.direct_targets(craft_targets)

    def apply_action(self, data: mujoco.MjData, action: np.ndarray, offset: int = 7) -> None:
        """Apply the hand slice from a single-arm action.

        Single-arm policy/Vive actions carry 16 Allegro-compatible hand values.
        HaMeR CRAFT teleop actions carry 15 CRAFT command values instead. Keep
        this convention switch in one place so all single-arm tasks can support
        both paths.
        """
        action = np.asarray(action, dtype=np.float64)
        direct_craft_dim = len(CRAFT_COMMAND_ACTUATOR_NAMES)
        policy_dim = len(ALLEGRO_JOINT_NAMES)
        if (
            self.hand == "craft"
            and action.shape[0] >= offset + direct_craft_dim
            and action.shape[0] < offset + policy_dim
        ):
            self.apply_direct(data, action[offset : offset + direct_craft_dim])
        elif action.shape[0] >= offset + policy_dim:
            self.apply(data, action[offset : offset + policy_dim])
        else:
            self.apply(data, np.zeros(policy_dim, dtype=np.float64))

    def set_pose(self, data: mujoco.MjData, policy_targets: np.ndarray) -> None:
        """Set matching physical joint positions and actuator targets immediately."""
        targets = self.physical_targets(policy_targets)
        positions = targets if self.hand == "allegro" else craft_joint_positions(targets)
        data.qpos[self.qpos_ids] = positions
        data.ctrl[self.ctrl_ids] = targets

    def set_pose_direct(self, data: mujoco.MjData, craft_targets: np.ndarray) -> None:
        """Set CRAFT joint positions and actuator targets from direct CRAFT commands."""
        targets = self.direct_targets(craft_targets)
        data.qpos[self.qpos_ids] = craft_joint_positions(targets)
        data.ctrl[self.ctrl_ids] = targets

    def reset(self, data: mujoco.MjData) -> None:
        """Reset the physical hand to the Allegro-compatible home pose."""
        if self._direct_cr3_craft:
            self.set_pose_direct(data, np.zeros(15, dtype=np.float64))
        else:
            self.set_pose(data, ALLEGRO_HOME)

    def policy_state(self, data: mujoco.MjData) -> np.ndarray:
        """Read physical hand state through the 16-value Allegro interface."""
        if self._direct_cr3_craft:
            values = []
            for command_name in CRAFT_COMMAND_ACTUATOR_NAMES:
                actuator_name = "h_" + command_name.removeprefix("craft_")
                actuator_id = int(self.model.actuator(actuator_name).id)
                qpos_id = int(
                    self.model.joint(self.model.actuator(actuator_name).trnid[0]).qposadr[0]
                )
                low, high = self.model.actuator_ctrlrange[actuator_id]
                percent = _normalize(
                    np.asarray((data.qpos[qpos_id],), dtype=np.float64),
                    np.asarray((low,)),
                    np.asarray((high,)),
                )[0]
                values.append(percent * CRAFT_VALUE_MAX)
            return np.asarray(values, dtype=np.float32)
        state = data.qpos[self.controlled_qpos_ids]
        return allegro_compatible_state(self.hand, state).astype(np.float32)
