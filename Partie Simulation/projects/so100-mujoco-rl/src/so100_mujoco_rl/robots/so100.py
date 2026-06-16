"""SO-100 robot descriptor.

Encapsulates robot-level MuJoCo bookkeeping: joint IDs, actuator IDs,
end-effector site ID, and limits.  Everything robot-specific lives here so
task/env code stays robot-agnostic.

TODO: Once the MuJoCo Menagerie so_arm100.xml is available, verify that
    the joint/actuator names in configs/robot/so100.yaml match the XML.
    Run `scripts/check_env.py` to see what names are actually in the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from so100_mujoco_rl.utils.config import load_config, project_root
from so100_mujoco_rl.utils.mujoco_utils import (
    get_actuator_ids,
    get_joint_ids,
    get_site_id,
)

_DEFAULT_CONFIG = project_root() / "configs" / "robot" / "so100.yaml"


@dataclass
class SO100Robot:
    """Robot descriptor built from a loaded MuJoCo model.

    Parameters
    ----------
    model:
        Loaded ``mujoco.MjModel``.
    robot_cfg:
        Dict loaded from ``configs/robot/so100.yaml`` (or equivalent).
    """

    model: mujoco.MjModel
    robot_cfg: dict[str, Any]

    # Populated in __post_init__
    arm_joint_ids: list[int] = field(init=False)
    gripper_joint_ids: list[int] = field(init=False)
    arm_actuator_ids: list[int] = field(init=False)
    gripper_actuator_ids: list[int] = field(init=False)
    ee_site_id: int = field(init=False)
    has_gripper: bool = field(init=False)

    def __post_init__(self) -> None:
        cfg = self.robot_cfg

        # Arm joints — joint names may span arm + gripper in the config
        arm_joint_names: list[str] = cfg.get("joints", [])
        gripper_actuator_names: list[str] = cfg.get("actuators", {}).get("gripper", [])
        arm_actuator_names: list[str] = cfg.get("actuators", {}).get("arm", [])

        # has_gripper is config-requested AND the gripper joint/actuator must actually
        # exist in the loaded model (allows the same robot config to work with scenes
        # that lack a gripper, e.g. the reach-only scene).
        config_wants_gripper = bool(cfg.get("has_gripper", False)) and bool(gripper_actuator_names)
        gripper_present = config_wants_gripper and _joints_exist(self.model, gripper_actuator_names)
        self.has_gripper = gripper_present

        # Always exclude known gripper joint names from the arm list,
        # even when the gripper is absent in this scene's XML.
        arm_only_joint_names = [n for n in arm_joint_names if n not in gripper_actuator_names]
        gripper_joint_names = gripper_actuator_names if self.has_gripper else []

        self.arm_joint_ids = get_joint_ids(self.model, arm_only_joint_names)
        self.gripper_joint_ids = (
            get_joint_ids(self.model, gripper_joint_names) if self.has_gripper else []
        )
        self.arm_actuator_ids = get_actuator_ids(self.model, arm_actuator_names)
        self.gripper_actuator_ids = (
            get_actuator_ids(self.model, gripper_actuator_names) if self.has_gripper else []
        )

        ee_site_name: str = cfg.get("end_effector_site", "attachment_site")
        self.ee_site_id = get_site_id(self.model, ee_site_name)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_arm_joints(self) -> int:
        return len(self.arm_joint_ids)

    @property
    def n_gripper_joints(self) -> int:
        return len(self.gripper_joint_ids)

    @property
    def n_arm_actuators(self) -> int:
        return len(self.arm_actuator_ids)

    @property
    def n_gripper_actuators(self) -> int:
        return len(self.gripper_actuator_ids)

    @property
    def action_dim(self) -> int:
        """Total number of actuated DOF (arm + optional gripper)."""
        return self.n_arm_actuators + (self.n_gripper_actuators if self.has_gripper else 0)

    def get_ee_pos(self, data: mujoco.MjData) -> np.ndarray:
        """Return end-effector Cartesian position (3,)."""
        return data.site_xpos[self.ee_site_id].copy()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_model(
        cls,
        model: mujoco.MjModel,
        config_path: str | None = None,
    ) -> "SO100Robot":
        """Build from a loaded model and optional YAML path."""
        cfg_path = config_path or _DEFAULT_CONFIG
        robot_cfg = load_config(cfg_path)
        return cls(model=model, robot_cfg=robot_cfg)


def _joints_exist(model: mujoco.MjModel, joint_names: list[str]) -> bool:
    """Return True only if ALL named joints are present in the model."""
    for name in joint_names:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) == -1:
            return False
    return True
