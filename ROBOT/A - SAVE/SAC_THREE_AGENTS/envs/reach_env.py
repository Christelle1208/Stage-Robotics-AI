"""
Agent 1 — REACH Environment (SAC Three-Agent Pick-and-Place)

Paper reference: MDPI Biomimetics 8(2):240
  "Sub-task 1: the Reach agent learns to bring the end-effector to a
   pre-grasp position directly above the target object, with the gripper
   open. This decomposition prevents the agent from having to reason about
   grasping before positioning."

Task:
  Start : home joint configuration, gripper open
  Goal  : end-effector within `reach_threshold` of pre-grasp point
          (object XY + offset_z above object)
  End   : success (EE close enough) OR truncation (max steps)

Observation (19-dim flat):
  [q(6) | ee_pos(3) | obj_pos(3) | ee_vel(3) | jaw(1) | rel_ee_obj(3)]

Reward (Eq. 12–18 from paper):
  r = r_pos + r_energy
  r_pos = Kx*(x_ee-x_g)^2 + Ky*(y_ee-y_g)^2 + Kz*(z_ee-z_g)^2  (Eq. 13–14)
  Kx=-2, Ky=-2, Kz=-0.5  (Eq. 17 — reaching task, prioritize XY)
  (x_g, y_g, z_g) = P_pick = pre-grasp point above object  (Eq. 16)
  r_energy = -0.00003 * sum(|F_i|)  (Eq. 18)
  + gripper_open_bonus × (jaw / jaw_open_max)   ← keep gripper open
  + success_bonus  [sparse, once]               ← reached pre-grasp
"""

import numpy as np
from gymnasium import spaces

from envs.base_env import SO100BaseEnv, CONFIG


class ReachEnv(SO100BaseEnv):
    """
    SAC Agent 1: reach the pre-grasp position above the object.

    The pre-grasp point is computed dynamically each step as:
        pre_grasp = object_pos + [0, 0, pre_grasp_offset_z]
    so it tracks the object even with domain randomisation.
    """

    # Observation Keys (order matters → determines feature vector layout)
    _OBS_KEYS = ["q", "ee_pos", "obj_pos", "ee_vel", "jaw", "rel_ee_obj"]
    OBS_DIM   = 6 + 3 + 3 + 3 + 1 + 3  # = 19

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)

        self.max_episode_steps = CONFIG["reach"]["max_episode_steps"]

        # Reward weights (loaded from config for easy tuning)
        rc = CONFIG["reach"]["rewards"]
        self.kx                  = rc["kx"]
        self.ky                  = rc["ky"]
        self.kz                  = rc["kz"]
        self.energy_weight       = rc["energy_weight"]
        self.success_bonus       = rc["success_bonus"]
        self.gripper_open_bonus  = rc["gripper_open_bonus"]

        # Success criteria
        sc = CONFIG["reach"]["success"]
        self.reach_threshold    = sc["reach_threshold"]
        self.pre_grasp_offset_z = sc["pre_grasp_offset_z"]

        # Gripper reference values from robot config
        self.gripper_open_jaw = CONFIG["robot"]["gripper_open_jaw"]

        # Define flat observation space
        obs_low  = np.full(self.OBS_DIM, -np.inf, dtype=np.float32)
        obs_high = np.full(self.OBS_DIM,  np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # Episode state
        self._success_rewarded = False  # ensure sparse bonus fires only once

    # ─────────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """Reset to home pose, randomise object position."""
        self._reset_mujoco(seed)

        # Open the gripper at episode start (jaw fully open)
        open_jaw = self.gripper_open_jaw
        self.data.qpos[5] = open_jaw
        self.data.ctrl[5] = open_jaw

        self._success_rewarded = False

        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        """Apply delta action, compute reward, check termination."""
        self._apply_action(action)

        obs    = self._get_obs()
        reward, info = self._compute_reward()
        success   = self._is_success()
        truncated = self._step_count >= self.max_episode_steps

        info["success"] = success
        return obs, reward, success, truncated, info

    # ─────────────────────────────────────────────────────────────────────────
    # Observation
    # ─────────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Build flat 19-dim observation vector."""
        d = self._obs_dict()
        return self._flat_obs(d, self._OBS_KEYS)

    # ─────────────────────────────────────────────────────────────────────────
    # Reward (see module docstring for breakdown)
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_reward(self) -> tuple[float, dict]:
        """
        Paper reward (Eq. 12–18) for the reach task.

        r = r_pos + r_energy
        r_pos = Kx*(x_ee - x_g)^2 + Ky*(y_ee - y_g)^2 + Kz*(z_ee - z_g)^2
        r_energy = -energy_weight * sum(|F_i|)

        Reaching task coefficients: Kx=-2, Ky=-2, Kz=-0.5
        This prioritises matching (x, y) first, then lowers to the correct z.

        Returns (total_reward, breakdown_dict).
        """
        d = self._obs_dict()
        ee_pos  = d["ee_pos"]
        obj_pos = d["obj_pos"]
        jaw     = float(d["jaw"][0])

        # Goal position: pre-grasp point directly above the object centre
        pre_grasp = obj_pos + np.array([0.0, 0.0, self.pre_grasp_offset_z])

        # Position reward (Eq. 13–14)
        dx = float(ee_pos[0] - pre_grasp[0])
        dy = float(ee_pos[1] - pre_grasp[1])
        dz = float(ee_pos[2] - pre_grasp[2])
        r_x = self.kx * dx ** 2
        r_y = self.ky * dy ** 2
        r_z = self.kz * dz ** 2
        r_pos = r_x + r_y + r_z

        # Energy reward (Eq. 18)
        actuator_forces = self.data.actuator_force
        r_energy = -self.energy_weight * float(np.sum(np.abs(actuator_forces)))

        b: dict[str, float] = {}

        # Core paper reward
        b["r_pos"]    = r_pos
        b["r_energy"] = r_energy

        # Gripper-open incentive (normalised: 0 = closed, 1 = open)
        open_fraction = min(1.0, jaw / self.gripper_open_jaw)
        b["jaw_open"] = self.gripper_open_bonus * open_fraction

        # Sparse success bonus (fires once on first success)
        dist = float(np.linalg.norm(ee_pos - pre_grasp))
        b["success"] = 0.0
        if dist < self.reach_threshold and not self._success_rewarded:
            b["success"]          = self.success_bonus
            self._success_rewarded = True

        # Debug scalars (excluded from reward total)
        b["dist_m"]   = round(dist, 4)
        b["jaw_rad"]  = round(jaw, 3)

        total = sum(v for k, v in b.items() if k not in ("dist_m", "jaw_rad"))
        return float(total), {"reward_breakdown": b}

    # ─────────────────────────────────────────────────────────────────────────
    # Termination
    # ─────────────────────────────────────────────────────────────────────────

    def _is_success(self) -> bool:
        """
        Success criterion: EE is within reach_threshold of pre-grasp point
        AND gripper is open (jaw > half-open threshold).
        """
        d       = self._obs_dict()
        ee_pos  = d["ee_pos"]
        obj_pos = d["obj_pos"]
        jaw     = float(d["jaw"][0])

        pre_grasp = obj_pos + np.array([0.0, 0.0, self.pre_grasp_offset_z])
        dist      = float(np.linalg.norm(ee_pos - pre_grasp))

        gripper_open = jaw > (self.gripper_open_jaw * 0.4)  # at least 40% open
        return dist < self.reach_threshold and gripper_open
