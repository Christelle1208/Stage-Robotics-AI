"""
Agent 3 — BOX PLACE Environment

Variant of PlaceEnv where the goal is an open box (scene_box.xml).
The robot must transport the cube over the box opening and drop it inside.

Key differences from PlaceEnv:
  - Loads scene_box.xml instead of scene_table.xml
  - Goal position = centre of the box floor (at box wall height z ~ 0.003)
  - Success: cube is inside the box (XY within inner walls, Z above floor)
  - The box is a physical object — the cube actually falls into it
  - goal_pos z is set to the release_height so the agent aims ABOVE the box
    (the r_pos gradient pushes EE+cube horizontally over the box, then drop)
"""

import numpy as np
import mujoco as _mj

from envs.base_env import CONFIG
from envs.place_env import PlaceEnv


# Inner half-width of the box (wall inner edge = 5.5 cm - 0.5 cm wall = 5 cm)
_BOX_INNER_HALF = 0.050
# Height of the box floor surface
_BOX_FLOOR_Z    = 0.006   # floor geom size[2]*2 = 0.003*2 + ground offset

class BoxPlaceEnv(PlaceEnv):
    """
    Place variant: drop the cube into an open box.

    The scene uses scene_box.xml which has physical box walls so the cube
    actually bounces/rests inside.  Goal position is set to the box centre
    at release_height so the dense r_pos reward guides the arm horizontally
    over the box before the agent opens the gripper.

    Success condition: cube is at rest inside the box (XY within inner walls,
    Z between floor and top of walls).
    """

    # Box inner half-dimensions (must match scene_box.xml geom sizes)
    BOX_INNER_HALF_XY = _BOX_INNER_HALF   # inner clear half-width in X and Y
    BOX_FLOOR_Z       = _BOX_FLOOR_Z      # z of the box floor surface
    BOX_WALL_HEIGHT   = 0.063             # floor thickness + wall height

    def __init__(self, render_mode=None):
        # Temporarily redirect xml_path to the box scene before super().__init__
        # loads the MuJoCo model.
        _orig_xml = CONFIG["env"]["xml_path"]
        CONFIG["env"]["xml_path"] = CONFIG["scene_variants"]["box"]["xml_path"]

        super().__init__(render_mode=render_mode)

        # Restore original so other envs are unaffected
        CONFIG["env"]["xml_path"] = _orig_xml

        # Override place_threshold with box-specific value
        box_cfg = CONFIG["scene_variants"]["box"]
        self.place_threshold = box_cfg["place_threshold_box"]
        self.release_height  = box_cfg["release_height"]

    # Fixed box position (world frame) — not randomised
    BOX_FIXED_X = 0.20
    BOX_FIXED_Y = -0.12

    # ─────────────────────────────────────────────────────────────────────────
    # Goal randomisation helpers (shared by reset and reset_from_grasp)
    # ─────────────────────────────────────────────────────────────────────────

    def _set_box_goal(self):
        """Place the box at its fixed world position and set goal_pos."""
        gx, gy = self.BOX_FIXED_X, self.BOX_FIXED_Y
        goal_body_pos = self.model.body("goal").pos
        goal_body_pos[0] = gx
        goal_body_pos[1] = gy
        goal_body_pos[2] = 0.0
        _mj.mj_forward(self.model, self.data)
        self.goal_pos = np.array([gx, gy, self.release_height], dtype=np.float64)

    # ─────────────────────────────────────────────────────────────────────────
    # Reset overrides
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._set_box_goal()
        return self._get_obs(), info

    def reset_from_grasp(self, qpos: np.ndarray, qctrl: np.ndarray):
        obs, info = super().reset_from_grasp(qpos, qctrl)
        self._set_box_goal()
        return self._get_obs(), info

    def step(self, action: np.ndarray):
        """Force jaw closed before every step."""
        action = np.array(action, dtype=np.float32)
        action[5] = -1.0  # always close the gripper
        return super().step(action)

    # ─────────────────────────────────────────────────────────────────────────
    # Reward override — phase-based for box task
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_reward(self) -> tuple[float, dict]:
        """
        Pure 3D distance reward toward the target point above the box.
        The jaw is already forced closed by step(), so no jaw terms needed.
        """
        d       = self._obs_dict()
        obj_pos = d["obj_pos"]
        goal    = self.goal_pos if self.goal_pos is not None else np.zeros(3)

        dx = float(obj_pos[0] - goal[0])
        dy = float(obj_pos[1] - goal[1])
        dz = float(obj_pos[2] - goal[2])
        dist3d  = float(np.sqrt(dx**2 + dy**2 + dz**2))
        xy_dist = float(np.sqrt(dx**2 + dy**2))
        obj_z   = float(obj_pos[2])

        r_pos    = -5.0 * (dx**2 + dy**2 + dz**2)
        r_energy = -self.energy_weight * float(np.sum(np.abs(self.data.actuator_force)))
        r_success = self.release_bonus if dist3d < self.place_threshold else 0.0

        b = {
            "r_pos":    r_pos,
            "r_energy": r_energy,
            "r_success": r_success,
            "obj_z":    round(obj_z, 4),
            "xy_dist":  round(xy_dist, 4),
            "dist3d":   round(dist3d, 4),
        }
        debug_keys = ("obj_z", "xy_dist", "dist3d")
        total = sum(v for k, v in b.items() if k not in debug_keys)
        return float(total), {"reward_breakdown": b}

    # ─────────────────────────────────────────────────────────────────────────
    # Success condition override
    # ─────────────────────────────────────────────────────────────────────────

    def _is_success(self) -> bool:
        """Success: cube is close to the 3D target above the box."""
        obj_pos = self.data.qpos[self.object_qpos_addr:self.object_qpos_addr + 3]
        goal    = self.goal_pos if self.goal_pos is not None else np.zeros(3)
        dist3d  = float(np.linalg.norm(obj_pos - goal))
        return dist3d < self.place_threshold
