"""Pick-and-Place task logic.

Geometry reference (from Menagerie trs_so_arm100/scene_pick_place.xml):
  cube:  size="0.018 0.018 0.018"  (half-extent, 36 mm edge)
         initial pos="0.00 -0.31 0.018" (resting on floor z=0)
  goal:  site name="goal_site" pos="0.18 -0.08 0.08" (free-standing in worldbody)

The goal site is a site element directly in the worldbody, not attached to
a body.  We randomise it via model.site_pos (no body_pos lookup needed).
The cube uses a free joint (type=free, 7 DOF qpos).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from so100_mujoco_rl.utils.mujoco_utils import get_body_id, get_site_id

# Half-extent of the cube in the Menagerie model.
_CUBE_HALF_SIZE = 0.018


@dataclass
class PickPlaceTask:
    """Stateless pick-and-place task.

    Parameters
    ----------
    model:
        Loaded ``MjModel``.
    task_cfg:
        Sub-dict from env YAML under ``task``.
    reward_cfg:
        Sub-dict from env YAML under ``reward``.
    """

    model: mujoco.MjModel
    task_cfg: dict[str, Any]
    reward_cfg: dict[str, Any]

    ee_site_id: int = -1
    target_site_id: int = -1
    object_body_id: int = -1

    _cube_qpos_adr: int = -1
    _cube_dof_adr: int = -1
    model_cube_joint_id: int = -1

    def __post_init__(self) -> None:
        ee_name     = self.task_cfg.get("end_effector_site", "ee_site")
        target_name = self.task_cfg.get("target_site", "goal_site")
        object_name = self.task_cfg.get("object_name", "cube")

        self.ee_site_id     = get_site_id(self.model, ee_name)
        self.target_site_id = get_site_id(self.model, target_name)
        self.object_body_id = get_body_id(self.model, object_name)

        # Find the free joint on the cube body.
        for jid in range(self.model.njnt):
            if self.model.jnt_bodyid[jid] == self.object_body_id:
                self.model_cube_joint_id = jid
                self._cube_qpos_adr = int(self.model.jnt_qposadr[jid])
                self._cube_dof_adr  = int(self.model.jnt_dofadr[jid])
                break
        else:
            raise ValueError(
                f"No joint found for body '{object_name}'. "
                "The cube body must have a <joint type='free'> in the scene XML."
            )

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def get_object_pos(self, data: mujoco.MjData) -> np.ndarray:
        """Return cube position (3,) from xpos (updated by mj_forward)."""
        return data.xpos[self.object_body_id].copy()

    def get_target_pos(self, data: mujoco.MjData) -> np.ndarray:
        """Return goal site position (3,) from xpos."""
        return data.site_xpos[self.target_site_id].copy()

    def get_ee_pos(self, data: mujoco.MjData) -> np.ndarray:
        return data.site_xpos[self.ee_site_id].copy()

    # ------------------------------------------------------------------
    # Reward and termination
    # ------------------------------------------------------------------

    def compute_reward(
        self,
        data: mujoco.MjData,
        action: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        ee_pos  = self.get_ee_pos(data)
        obj_pos = self.get_object_pos(data)
        tgt_pos = self.get_target_pos(data)

        floor_height = float(self.task_cfg.get("floor_height", 0.0))
        lift_height  = float(self.task_cfg.get("lift_height", 0.04))

        rw_cfg       = self.reward_cfg
        reach_w      = float(rw_cfg.get("reach_weight",   1.0))
        lift_w       = float(rw_cfg.get("lift_weight",    2.0))
        place_w      = float(rw_cfg.get("place_weight",   4.0))
        success_b    = float(rw_cfg.get("success_bonus",  10.0))
        action_pen   = float(rw_cfg.get("action_penalty", 0.01))

        dist_ee_obj   = float(np.linalg.norm(ee_pos - obj_pos))
        dist_obj_tgt  = float(np.linalg.norm(obj_pos - tgt_pos))
        is_lifted     = obj_pos[2] > (floor_height + _CUBE_HALF_SIZE + lift_height)

        reach_reward  = -reach_w  * dist_ee_obj
        lift_reward   =  lift_w   * float(is_lifted)
        place_reward  = -place_w  * dist_obj_tgt * float(is_lifted)
        bonus         =  success_b if self.is_success(data) else 0.0
        act_pen       = -action_pen * float(np.sum(action ** 2))

        total = reach_reward + lift_reward + place_reward + bonus + act_pen
        info  = {
            "reach_reward":    reach_reward,
            "lift_reward":     lift_reward,
            "place_reward":    place_reward,
            "success_bonus":   bonus,
            "action_penalty":  act_pen,
            "dist_ee_to_obj":  dist_ee_obj,
            "dist_obj_to_target": dist_obj_tgt,
            "is_lifted":       float(is_lifted),
        }
        return total, info

    def is_success(self, data: mujoco.MjData) -> bool:
        obj_pos = self.get_object_pos(data)
        tgt_pos = self.get_target_pos(data)
        threshold = float(self.task_cfg.get("success_threshold", 0.04))
        return bool(np.linalg.norm(obj_pos - tgt_pos) < threshold)

    # ------------------------------------------------------------------
    # Randomisation
    # ------------------------------------------------------------------

    def randomize_scene(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        rng: np.random.Generator,
        rand_cfg: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Randomise cube and goal positions.

        Returns a dict with keys ``object_pos`` and ``target_pos``.
        """
        obj_range = rand_cfg.get("object_range", {})
        tgt_range = rand_cfg.get("target_range", {})

        obj_x_range = obj_range.get("x", [-0.04, 0.04])
        obj_y_range = obj_range.get("y", [-0.33, -0.26])
        tgt_x_range = tgt_range.get("x", [0.10, 0.22])
        tgt_y_range = tgt_range.get("y", [-0.15, -0.04])
        tgt_z_range = tgt_range.get("z", [0.06, 0.10])

        floor_h = float(self.task_cfg.get("floor_height", 0.0))
        obj_z   = floor_h + _CUBE_HALF_SIZE + 0.001  # just above the floor

        obj_pos = np.array([
            float(rng.uniform(*obj_x_range)),
            float(rng.uniform(*obj_y_range)),
            obj_z,
        ])
        tgt_pos = np.array([
            float(rng.uniform(*tgt_x_range)),
            float(rng.uniform(*tgt_y_range)),
            float(rng.uniform(*tgt_z_range)),
        ])

        # Place cube via its freejoint qpos:
        # [x, y, z, qw, qx, qy, qz]
        adr = self._cube_qpos_adr
        data.qpos[adr : adr + 3] = obj_pos
        data.qpos[adr + 3]       = 1.0   # identity quaternion
        data.qpos[adr + 4 : adr + 7] = 0.0
        data.qvel[self._cube_dof_adr : self._cube_dof_adr + 6] = 0.0

        # Move the goal site (free-standing site in worldbody) via model.site_pos.
        model.site_pos[self.target_site_id] = tgt_pos

        mujoco.mj_forward(model, data)
        return {"object_pos": obj_pos, "target_pos": tgt_pos}
