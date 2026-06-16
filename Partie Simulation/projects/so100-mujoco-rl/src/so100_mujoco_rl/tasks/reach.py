"""Reaching task logic.

Handles reward computation and termination for the reaching task.
Kept separate from the Gymnasium environment so the same task can be
re-used with different environment wrappers (e.g. LeRobot, real robot).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from so100_mujoco_rl.utils.mujoco_utils import get_site_id


@dataclass
class ReachTask:
    """Stateless task definition for end-effector reaching.

    Parameters
    ----------
    model:
        Loaded ``MjModel`` — used to resolve site IDs once at construction.
    task_cfg:
        Sub-dict from the environment YAML under the key ``task``.
    reward_cfg:
        Sub-dict from the environment YAML under the key ``reward``.
    """

    model: mujoco.MjModel
    task_cfg: dict[str, Any]
    reward_cfg: dict[str, Any]

    ee_site_id: int = -1
    target_site_id: int = -1

    def __post_init__(self) -> None:
        ee_name = self.task_cfg.get("end_effector_site", "attachment_site")
        target_name = self.task_cfg.get("target_site", "target_site")
        self.ee_site_id = get_site_id(self.model, ee_name)
        self.target_site_id = get_site_id(self.model, target_name)

    # ------------------------------------------------------------------
    # Core methods (called by the Gymnasium env)
    # ------------------------------------------------------------------

    def compute_reward(
        self,
        data: mujoco.MjData,
        action: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        """Return (scalar reward, info dict with reward components)."""
        ee_pos = data.site_xpos[self.ee_site_id]
        target_pos = data.site_xpos[self.target_site_id]
        dist = float(np.linalg.norm(ee_pos - target_pos))

        rw_cfg = self.reward_cfg
        dist_w = float(rw_cfg.get("distance_weight", 1.0))
        success_bonus = float(rw_cfg.get("success_bonus", 10.0))
        action_penalty = float(rw_cfg.get("action_penalty", 0.01))

        reach_reward = -dist_w * dist
        action_pen = -action_penalty * float(np.sum(action**2))
        bonus = success_bonus if self.is_success(data) else 0.0

        total = reach_reward + action_pen + bonus
        info = {
            "reach_reward": reach_reward,
            "action_penalty": action_pen,
            "success_bonus": bonus,
            "dist_to_target": dist,
        }
        return total, info

    def is_success(self, data: mujoco.MjData) -> bool:
        ee_pos = data.site_xpos[self.ee_site_id]
        target_pos = data.site_xpos[self.target_site_id]
        threshold = float(self.task_cfg.get("success_threshold", 0.04))
        return bool(np.linalg.norm(ee_pos - target_pos) < threshold)

    def randomize_target(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        rng: np.random.Generator,
        rand_cfg: dict[str, Any],
    ) -> np.ndarray:
        """Sample a new target position and move the target site.

        The target site lives directly in the worldbody (injected via MjSpec),
        so we update its position via ``model.site_pos``.

        Returns the new target xyz.
        """
        target_range = rand_cfg.get("target_range", {})
        x_range = target_range.get("x", [-0.10, 0.10])
        y_range = target_range.get("y", [-0.30, -0.10])
        z_range = target_range.get("z", [0.05, 0.18])

        x = float(rng.uniform(*x_range))
        y = float(rng.uniform(*y_range))
        z = float(rng.uniform(*z_range))
        new_pos = np.array([x, y, z], dtype=np.float64)

        # The reach target is a site injected directly into the worldbody;
        # update its position via model.site_pos then re-forward.
        self.target_site_id
        model.site_pos[self.target_site_id] = new_pos
        mujoco.mj_forward(model, data)
        return new_pos
