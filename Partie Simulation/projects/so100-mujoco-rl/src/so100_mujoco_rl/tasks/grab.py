"""Grab task logic.

"Grab" = reach the end-effector to within `success_threshold` of a cube
resting on the Feuille checkerboard, with the gripper OPEN, and hold that
position for `hold_steps` consecutive steps. This confirms the arm is
correctly positioned and ready to close on the cube — closing the gripper
itself is a separate (later) phase and is not rewarded here.

Geometry reference (so100_feuille_scene.xml):
  Feuille (sheet): 0.24 x 0.16 m, centered at world (0, 0), top surface z=0.002
  cube: box, half-extent 0.015 m (3 cm edge), spawns resting on the sheet
        (z = 0.017), free joint -> qpos = [x, y, z, qw, qx, qy, qz]

Jaw joint convention (measured empirically on so_arm100.xml):
  Jaw == jnt_range[0] (≈ -0.174) -> jaws together (CLOSED)
  Jaw == jnt_range[1] (≈  1.75)  -> jaws apart    (OPEN)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from so100_mujoco_rl.utils.mujoco_utils import get_body_id, get_joint_ids, get_site_id

# Half-extent of the cube used in this task.
CUBE_HALF_SIZE = 0.015


@dataclass
class GrabTask:
    """Stateless reach-and-grab task.

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
    object_body_id: int = -1
    jaw_joint_id: int = -1
    jaw_closed: float = 0.0
    jaw_open: float = 0.0

    _cube_qpos_adr: int = -1
    _cube_dof_adr: int = -1

    # Consecutive steps the end-effector has been within `success_threshold`
    # of the cube with the gripper open. Reset on each episode reset.
    _hold_counter: int = 0
    _is_holding: bool = False

    def __post_init__(self) -> None:
        ee_name     = self.task_cfg.get("end_effector_site", "ee_site")
        object_name = self.task_cfg.get("object_name", "cube")
        jaw_name    = self.task_cfg.get("gripper_joint", "Jaw")

        self.ee_site_id     = get_site_id(self.model, ee_name)
        self.object_body_id = get_body_id(self.model, object_name)
        self.jaw_joint_id   = get_joint_ids(self.model, [jaw_name])[0]

        jaw_range = self.model.jnt_range[self.jaw_joint_id]
        self.jaw_closed = float(jaw_range[0])  # jaws together
        self.jaw_open   = float(jaw_range[1])  # jaws apart

        # Find the free joint on the cube body.
        for jid in range(self.model.njnt):
            if self.model.jnt_bodyid[jid] == self.object_body_id:
                self._cube_qpos_adr = int(self.model.jnt_qposadr[jid])
                self._cube_dof_adr  = int(self.model.jnt_dofadr[jid])
                break
        else:
            raise ValueError(
                f"No joint found for body '{object_name}'. "
                "The cube body must have a <joint type='free'> (added via _patch_spec)."
            )

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def get_object_pos(self, data: mujoco.MjData) -> np.ndarray:
        return data.xpos[self.object_body_id].copy()

    def get_ee_pos(self, data: mujoco.MjData) -> np.ndarray:
        return data.site_xpos[self.ee_site_id].copy()

    def get_close_fraction(self, data: mujoco.MjData) -> float:
        """Return gripper closedness in [0, 1]; 1.0 == fully closed."""
        jaw = float(data.qpos[self.model.jnt_qposadr[self.jaw_joint_id]])
        span = self.jaw_open - self.jaw_closed
        return float(np.clip((self.jaw_open - jaw) / span, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Reward and termination
    # ------------------------------------------------------------------

    def _update_hold(self, data: mujoco.MjData) -> bool:
        """Advance the consecutive in-position-with-open-gripper counter.

        Must be called exactly once per step (from ``compute_reward``).
        ``is_success`` simply reads the cached result so it can be called
        again later in the same step (e.g. from termination checks) without
        double-counting.
        """
        ee_pos  = self.get_ee_pos(data)
        obj_pos = self.get_object_pos(data)
        dist    = float(np.linalg.norm(ee_pos - obj_pos))
        close_frac = self.get_close_fraction(data)

        success_threshold = float(self.task_cfg.get("success_threshold", 0.01))
        gripper_open_threshold = float(self.task_cfg.get("gripper_open_threshold", 0.5))
        hold_steps = int(self.task_cfg.get("hold_steps", 5))

        in_position = dist < success_threshold and close_frac < gripper_open_threshold

        self._hold_counter = self._hold_counter + 1 if in_position else 0
        self._is_holding = self._hold_counter >= hold_steps
        return self._is_holding

    def compute_reward(
        self,
        data: mujoco.MjData,
        action: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        ee_pos  = self.get_ee_pos(data)
        obj_pos = self.get_object_pos(data)
        dist    = float(np.linalg.norm(ee_pos - obj_pos))

        rw_cfg       = self.reward_cfg
        reach_w      = float(rw_cfg.get("reach_weight", 1.0))
        success_b    = float(rw_cfg.get("success_bonus", 10.0))
        action_pen_w = float(rw_cfg.get("action_penalty", 0.01))

        reach_reward = -reach_w * dist
        bonus = success_b if self._update_hold(data) else 0.0
        act_pen = -action_pen_w * float(np.sum(action ** 2))

        total = reach_reward + bonus + act_pen
        info = {
            "reach_reward": reach_reward,
            "success_bonus": bonus,
            "action_penalty": act_pen,
            "dist_ee_to_obj": dist,
            "close_fraction": self.get_close_fraction(data),
            "hold_counter": self._hold_counter,
        }
        return total, info

    def is_success(self, _data: mujoco.MjData) -> bool:
        return self._is_holding

    # ------------------------------------------------------------------
    # Randomisation
    # ------------------------------------------------------------------

    def randomize_scene(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        rng: np.random.Generator,
        rand_cfg: dict[str, Any],
    ) -> np.ndarray:
        """Place the cube at a random position on the Feuille.

        Returns the new cube position (3,).
        """
        self._hold_counter = 0
        self._is_holding = False

        cube_range = rand_cfg.get("cube_range", {})
        x_range = cube_range.get("x", [-0.10, 0.10])
        y_range = cube_range.get("y", [-0.065, 0.065])
        z = float(self.task_cfg.get("cube_spawn_z", 0.017))

        cube_pos = np.array([
            float(rng.uniform(*x_range)),
            float(rng.uniform(*y_range)),
            z,
        ])

        # Free joint qpos layout: [x, y, z, qw, qx, qy, qz]
        adr = self._cube_qpos_adr
        data.qpos[adr : adr + 3] = cube_pos
        data.qpos[adr + 3] = 1.0
        data.qpos[adr + 4 : adr + 7] = 0.0
        data.qvel[self._cube_dof_adr : self._cube_dof_adr + 6] = 0.0

        mujoco.mj_forward(model, data)
        return cube_pos
