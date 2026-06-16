"""
Agent 3 — PLACE Environment (SAC Three-Agent Pick-and-Place)

Paper reference: MDPI Biomimetics 8(2):240
  "Sub-task 3: the Place agent starts from the state left by the Grasp agent
   (object lifted and held by the gripper). It must transport the object to
   the goal position and release it there. The reward uses potential-based
   shaping to guide the arm toward the goal without biasing the optimal policy."

Task:
  Start : state snapshot from a successful Grasp episode
          (object held above table, gripper closed)
  Goal  : object resting within place_threshold of goal_pos, gripper open
  End   : success OR truncation (max steps)

Observation (22-dim flat):
  [q(6) | ee_pos(3) | obj_pos(3) | obj_vel(3) | jaw(1) | rel_ee_obj(3) | rel_obj_goal(3)]
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 19-dim base ^^^^^^^^^^^^^^^^^^^^^^^  +3 goal vector

  The goal vector is the key additional input vs ReachEnv/GraspEnv.
  Without it the agent has no directional signal toward the drop zone.

Reward (Eq. 12–18 from paper — placing task coefficients):
  r = r_pos + r_energy + release_bonus
  r_pos = Kx*(x_obj-x_g)^2 + Ky*(y_obj-y_g)^2 + Kz*(z_obj-z_g)^2  (Eq. 13–14)
  Kx=-1, Ky=-1, Kz=-1  (Eq. 17 — placing task, equal weighting)
  (x_g, y_g, z_g) = P_place = goal position   (Eq. 16)
  r_energy = -0.00003 * sum(|F_i|)  (Eq. 18)
  + release_bonus  [sparse, once when object placed at goal]
"""

import numpy as np
from gymnasium import spaces

from envs.base_env import SO100BaseEnv, CONFIG
from envs.grasp_env import ScriptedGrasp


class PlaceEnv(SO100BaseEnv):
    """
    SAC Agent 3: carry the held object to the goal position and release it.

    Must be initialised via reset_from_grasp(qpos, qctrl) at inference time.
    Standalone reset() initialises with object synthetically placed in gripper.
    """

    _OBS_KEYS = ["q", "ee_pos", "obj_pos", "obj_vel", "jaw", "rel_ee_obj"]
    OBS_DIM   = 6 + 3 + 3 + 3 + 1 + 3 + 3  # = 22 (base 19 + rel_obj_goal 3)

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)

        self.max_episode_steps = CONFIG["place"]["max_episode_steps"]

        # Place needs bigger joint steps to transport the cube over 10+ cm
        self.delta_scale = CONFIG["place"].get("delta_scale", CONFIG["env"]["delta_scale"])

        pc = CONFIG["place"]["rewards"]
        self.kx                = pc["kx"]
        self.ky                = pc["ky"]
        self.kz                = pc["kz"]
        self.energy_weight     = pc["energy_weight"]
        self.release_bonus     = pc["release_bonus"]
        self.place_threshold   = pc["place_threshold"]

        self.lift_height       = CONFIG["grasp"]["success"]["lift_height"]
        self.gripper_threshold = CONFIG["robot"]["gripper_cube_jaw"]

        obs_low  = np.full(self.OBS_DIM, -np.inf, dtype=np.float32)
        obs_high = np.full(self.OBS_DIM,  np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # Per-episode state
        self._release_rewarded = False

    # ─────────────────────────────────────────────────────────────────────────
    # Reset variants
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """
        Standalone reset: synthetically place the object in the gripper.
        Used during isolated PlaceEnv training (no chaining needed).
        """
        self._reset_mujoco(seed)

        # Set up a realistic post-grasp state: arm lowered to ground, cube gripped
        import mujoco as _mj

        # Arm configuration that puts the EE at ground level (~z=0.025)
        # This mimics the real pipeline where Reach brings EE to the cube
        ground_joints = [0.0, -1.0, 1.2, 1.57, -1.57]
        self.data.qpos[:5] = ground_joints
        self.data.ctrl[:5] = ground_joints

        # Open the jaw so cube fits between jaws
        open_jaw = CONFIG["robot"]["gripper_open_jaw"]
        self.data.qpos[5] = open_jaw
        self.data.ctrl[5] = open_jaw
        _mj.mj_forward(self.model, self.data)

        # Place the cube at the grasp site (on the ground, supported by floor)
        grasp_pos = self.data.site_xpos[self.grasp_site_id].copy()
        self.data.qpos[self.object_qpos_addr:self.object_qpos_addr + 3] = grasp_pos
        self.data.qpos[self.object_qpos_addr + 3] = 1.0
        self.data.qpos[self.object_qpos_addr + 4:self.object_qpos_addr + 7] = 0.0
        _mj.mj_forward(self.model, self.data)

        # Use ScriptedGrasp to close the jaw naturally (ground supports cube)
        grasp = ScriptedGrasp()
        grasp.run(self)

        # Re-randomise goal position relative to actual cube position (post-grip)
        # Goal between 5 cm and 7 cm away from the cube
        actual_obj = self.data.qpos[self.object_qpos_addr:self.object_qpos_addr + 3].copy()
        for _ in range(50):
            gx = actual_obj[0] + self.np_random.uniform(-0.07, 0.07)
            gy = actual_obj[1] + self.np_random.uniform(-0.07, 0.07)
            if np.sqrt((gx - actual_obj[0]) ** 2 + (gy - actual_obj[1]) ** 2) > 0.05:
                break
        goal_body_pos = self.model.body("goal").pos
        goal_body_pos[0] = gx
        goal_body_pos[1] = gy
        goal_body_pos[2] = self.object_geom_height
        _mj.mj_forward(self.model, self.data)
        self.goal_pos = goal_body_pos.copy()

        self._release_rewarded = False

        return self._get_obs(), {}

    def reset_from_grasp(self, qpos: np.ndarray, qctrl: np.ndarray):
        """
        State injection: initialise from Grasp agent's terminal state.

        Args:
            qpos:  full qpos array from GraspEnv.get_state_snapshot()
            qctrl: full ctrl array from GraspEnv.get_state_snapshot()

        Returns:
            obs, info  (Gymnasium-style)
        """
        import mujoco as _mj
        self._restore_snapshot(qpos, qctrl)

        # Randomise goal position relative to where the object currently is
        # (the goal body in Place's model is NOT shared with Reach's model)
        # Goal between 5 cm and 7 cm away from the cube
        actual_obj = self.data.qpos[self.object_qpos_addr:self.object_qpos_addr + 3].copy()
        for _ in range(50):
            gx = actual_obj[0] + self.np_random.uniform(-0.07, 0.07)
            gy = actual_obj[1] + self.np_random.uniform(-0.07, 0.07)
            if np.sqrt((gx - actual_obj[0]) ** 2 + (gy - actual_obj[1]) ** 2) > 0.05:
                break
        goal_body_pos = self.model.body("goal").pos
        goal_body_pos[0] = gx
        goal_body_pos[1] = gy
        goal_body_pos[2] = self.object_geom_height
        _mj.mj_forward(self.model, self.data)
        self.goal_pos = goal_body_pos.copy()

        self._release_rewarded = False

        return self._get_obs(), {}

    # ─────────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ─────────────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        self._apply_action(action)
        obs       = self._get_obs()
        reward, info = self._compute_reward()
        success   = self._is_success()
        truncated = self._step_count >= self.max_episode_steps
        info["success"] = success
        return obs, reward, success, truncated, info

    # ─────────────────────────────────────────────────────────────────────────
    # Observation
    # ─────────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Build 22-dim observation: base 19 + relative obj→goal vector."""
        d = self._obs_dict()

        obj_pos  = d["obj_pos"]
        goal_pos = self.goal_pos if self.goal_pos is not None else np.zeros(3, dtype=np.float32)
        rel_obj_goal = (goal_pos - obj_pos).astype(np.float32)

        # Base keys (19-dim) + goal direction (3-dim) = 22-dim
        base = self._flat_obs(d, self._OBS_KEYS)
        return np.concatenate([base, rel_obj_goal], dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # Reward
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_reward(self) -> tuple[float, dict]:
        """
        Paper reward (Eq. 12–18) for the place task.

        r = r_pos + r_energy + release_bonus
        r_pos = Kx*(x_obj-x_g)^2 + Ky*(y_obj-y_g)^2 + Kz*(z_obj-z_g)^2
        Kx=Ky=Kz=-1  (Eq. 17 — placing task)
        """
        d        = self._obs_dict()
        obj_pos  = d["obj_pos"]
        jaw      = float(d["jaw"][0])
        goal_pos = self.goal_pos if self.goal_pos is not None else np.zeros(3)

        goal_dist = float(np.linalg.norm(obj_pos - goal_pos))

        # Position reward (Eq. 13–14) — object vs goal
        dx = float(obj_pos[0] - goal_pos[0])
        dy = float(obj_pos[1] - goal_pos[1])
        dz = float(obj_pos[2] - goal_pos[2])
        r_pos = self.kx * dx ** 2 + self.ky * dy ** 2 + self.kz * dz ** 2

        # Energy reward (Eq. 18)
        r_energy = -self.energy_weight * float(np.sum(np.abs(self.data.actuator_force)))

        b: dict[str, float] = {}
        b["r_pos"]    = r_pos
        b["r_energy"] = r_energy

        # Hold bonus — reward keeping the jaw closed while transporting.
        # Only positive signal: no drop penalty (too negative, causes stagnation).
        ee_to_obj = float(np.linalg.norm(d["ee_pos"] - obj_pos))
        is_held   = jaw < self.gripper_threshold and ee_to_obj < 0.06
        b["hold"] = 0.2 if (is_held and goal_dist > self.place_threshold) else 0.0

        # Sparse release bonus — fires once when cube is within threshold (distance only)
        b["release"] = 0.0
        if goal_dist < self.place_threshold and not self._release_rewarded:
            b["release"]           = self.release_bonus
            self._release_rewarded = True

        # Dense release shaping — once cube is at goal, reward opening the jaw.
        # This teaches the agent to release the cube rather than hold it forever.
        b["release_shape"] = 0.0
        if goal_dist < self.place_threshold:
            open_frac = max(0.0, jaw - self.gripper_threshold) / (1.2 - self.gripper_threshold)
            b["release_shape"] = 0.5 * open_frac

        # Home shaping — once gripper is open, reward arm returning to home EE pos.
        # Teaches the robot to retract after placing instead of staying put.
        b["home_shape"] = 0.0
        if jaw > 1.0 and goal_dist < self.place_threshold:
            ee_pos    = d["ee_pos"]
            home_ee   = np.array([-0.017, -0.239, 0.082])  # EE at home keyframe
            home_dist = float(np.linalg.norm(ee_pos - home_ee))
            b["home_shape"] = -0.3 * home_dist ** 2

        # Debug scalars
        b["goal_dist_m"] = round(goal_dist, 4)
        b["jaw_rad"]     = round(jaw, 3)

        debug_keys = ("goal_dist_m", "jaw_rad")
        total = sum(v for k, v in b.items() if k not in debug_keys)
        return float(total), {"reward_breakdown": b}

    # ─────────────────────────────────────────────────────────────────────────
    # Termination
    # ─────────────────────────────────────────────────────────────────────────

    def _is_success(self) -> bool:
        """
        Place success: object within place_threshold of goal position and stable.
        Gripper state is not required — if the cube is at the goal, the task is done.
        """
        obj_pos   = self.data.qpos[
            self.object_qpos_addr:self.object_qpos_addr + 3
        ]
        obj_vel_z = abs(float(self.data.qvel[self.object_qvel_addr + 2]))
        goal_pos  = self.goal_pos if self.goal_pos is not None else np.zeros(3)

        goal_dist = float(np.linalg.norm(obj_pos - goal_pos))
        stable    = obj_vel_z < 0.05  # z velocity < 5 cm/s

        return goal_dist < self.place_threshold and stable
