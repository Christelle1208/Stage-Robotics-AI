"""
Two separate Gym environments for the SO-100 arm pick-and-place task.

  PickEnv  : episode starts at home pose with the cube on the table.
             Goal = grasp the cube and lift it above lift_height.
             Observation (flat, 18-dim):
               [6 joint angles | 3 grasp_site_pos | 3 object_pos | 3 object_vel
                | 3 relative vec (grasp_site → object)]

  PlaceEnv : episode starts with the cube already held by the gripper
             (simulator state is injected by Task_manager after a successful pick).
             Goal = carry the cube to goal_pos and release it.
             Observation (flat, 21-dim):
               [6 joint angles | 3 grasp_site_pos | 3 object_pos | 3 object_vel
                | 3 relative vec (grasp_site → object) | 3 relative vec (object → goal)]

Both environments use a flat Box observation space so that PPO's default
MlpPolicy (or our custom MlpExtractor in Models.py) can be used directly.
"""

import os
import yaml
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

# ---------------------------------------------------------------------------
# Load config relative to this file so it works from any cwd
# ---------------------------------------------------------------------------
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Shared base: MuJoCo bookkeeping + common reset logic
# ---------------------------------------------------------------------------
class _SO100BaseEnv(gym.Env):
    """Internal base — do not instantiate directly."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None):
        super().__init__()

        # ---- MuJoCo ----
        xml_path = os.path.join(os.path.dirname(__file__),
                                config["env"]["xml_path"])
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)
        self.frame_skip       = config["env"]["frame_skip"]
        self.max_episode_steps = config["env"]["max_episode_steps"]
        self._step_count = 0

        # ---- IDs ----
        self.grasp_site_id    = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,  "grasp_site")
        self.object_body_id   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,  "object")
        self.object_qpos_addr = self.model.jnt_qposadr[self.model.body_jntadr[self.object_body_id]]
        self.object_qvel_addr = self.model.jnt_dofadr [self.model.body_jntadr[self.object_body_id]]
        self.object_geom_height = self.model.geom("object_geom").size[2]
        self.goal_geom_height   = self.model.geom("g_goal_geom").size[1]

        # ---- Action space: normalised deltas [-1, 1]^6 ----
        # The policy outputs increments applied to the current ctrl target.
        # delta_scale (radians) limits how far each joint moves per step.
        # This is MUCH easier to learn than absolute targets because a random
        # policy just jitters slightly instead of flailing to random poses.
        self.delta_scale = config["env"]["delta_scale"]
        self.action_space = spaces.Box(
            low=-np.ones(self.model.nu, dtype=np.float32),
            high= np.ones(self.model.nu, dtype=np.float32),
            dtype=np.float32
        )

        # ---- Rendering ----
        self.render_mode = render_mode
        self._viewer = None

        # ---- Placeholders (filled in reset) ----
        self.goal_pos          = None
        self.np_random         = np.random.default_rng()

    # ------------------------------------------------------------------
    def _reset_common(self, seed):
        """Resets MuJoCo data to home keyframe and places the object."""
        self.np_random = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.model.keyframe("home").qpos
        # Sync ctrl to the actual home joint angles so position controllers
        # target the home pose, not zero (which would make the arm collapse).
        self.data.ctrl[:] = self.data.qpos[:self.model.nu]

        # Randomise object XY position slightly
        new_obj_pos = self.model.body("object").pos.copy()
        new_obj_pos[0] += self.np_random.uniform(-0.05, 0.05)
        new_obj_pos[1] += self.np_random.uniform(-0.05, 0.05)
        self.data.qpos[self.object_qpos_addr:self.object_qpos_addr + 3] = new_obj_pos

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0

        # Green goal is the default target (g_goal body)
        self.goal_pos = (
            self.model.body("g_goal").pos.copy()
            + np.array([0.0, 0.0, self.goal_geom_height + self.object_geom_height])
        )

    # ------------------------------------------------------------------
    def _get_common_obs(self):
        """Returns the 18 shared observation components as a numpy array."""
        joint_angles    = self.data.qpos[:6].astype(np.float32)
        grasp_site_pos  = self.data.site_xpos[self.grasp_site_id].astype(np.float32)
        object_pos      = self.data.qpos[self.object_qpos_addr:self.object_qpos_addr + 3].astype(np.float32)
        object_vel      = self.data.qvel[self.object_qvel_addr:self.object_qvel_addr + 3].astype(np.float32)
        rel_grasp_obj   = (grasp_site_pos - object_pos)  # direction from hand to object
        return joint_angles, grasp_site_pos, object_pos, object_vel, rel_grasp_obj

    # ------------------------------------------------------------------
    def _do_step(self, action):
        # Apply delta: new target = current target + action * scale
        new_ctrl = self.data.ctrl[:] + action * self.delta_scale
        self.data.ctrl[:] = np.clip(new_ctrl,
                                    self.model.actuator_ctrlrange[:, 0],
                                    self.model.actuator_ctrlrange[:, 1])
        mujoco.mj_step(self.model, self.data, self.frame_skip)
        self._step_count += 1

    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()

    def close_viewer(self):
        """Close only the viewer, keeping the environment usable."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def close(self):
        self.close_viewer()


# ===========================================================================
# PICK environment
# ===========================================================================
class PickEnv(_SO100BaseEnv):
    """
    Task: move the gripper to the cube, grasp it, and lift it above lift_height.

    Observation (18-dim flat):
        [joint_angles(6) | grasp_site_pos(3) | object_pos(3) | object_vel(3)
         | rel_grasp_obj(3)]

    Termination (success): object_pos.z > lift_height AND gripper closed.
    """

    OBS_DIM = 18  # 6 + 3 + 3 + 3 + 3 (no goal needed)

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)

        # Reward weights — v1
        self.dist_weight          = config["rewards_pick"]["dist_weight"]
        self.dist_progress_bonus  = config["rewards_pick"]["dist_progress_bonus"]
        self.alignment_bonus      = config["rewards_pick"]["alignment_bonus"]
        self.approach_open_weight = config["rewards_pick"]["approach_open_weight"]
        self.grasp_bonus          = config["rewards_pick"]["grasp_bonus"]
        self.grasp_sigma          = config["rewards_pick"]["grasp_sigma"]
        self.jaw_close_bonus      = config["rewards_pick"]["jaw_close_bonus"]
        self.hold_bonus           = config["rewards_pick"]["hold_bonus"]
        self.jaw_nudge_weight     = config["rewards_pick"]["jaw_nudge_weight"]
        self.lift_pull_weight     = config["rewards_pick"]["lift_pull_weight"]
        self.lift_weight          = config["rewards_pick"]["lift_weight"]
        self.orient_bonus         = config["rewards_pick"]["orient_bonus"]
        self.lift_bonus           = config["rewards_pick"]["lift_bonus"]
        self.collision_penalty    = config["rewards_pick"]["collision_penalty"]
        self.step_penalty         = config["rewards_pick"]["step_penalty"]
        self.lift_height          = config["rewards_pick"]["lift_height"]

        # Reward weights — v2 (staged top-down grasp)
        c2 = config["rewards_pick_v2"]
        self.v2_above_xy_weight   = c2["above_xy_weight"]
        self.v2_above_xy_bonus    = c2["above_xy_bonus"]
        self.v2_above_open_weight = c2["above_open_weight"]
        self.v2_above_z_min       = c2["above_z_min"]
        self.v2_descend_weight    = c2["descend_weight"]
        self.v2_descend_bonus     = c2["descend_bonus"]
        self.v2_xy_lock_penalty   = c2["xy_lock_penalty"]
        self.v2_jaw_align_base    = c2["jaw_align_base"]
        self.v2_jaw_align_bonus   = c2["jaw_align_bonus"]
        self.v2_hold_bonus        = c2["hold_bonus"]
        self.v2_lift_weight       = c2["lift_weight"]
        self.v2_lift_pull_weight  = c2["lift_pull_weight"]
        self.v2_lift_bonus        = c2["lift_bonus"]
        self.v2_drop_penalty      = c2["drop_penalty"]
        self.v2_collision_penalty = c2["collision_penalty"]
        self.v2_step_penalty      = c2["step_penalty"]
        self.v2_lift_height       = c2["lift_height"]

        # Active version
        self._reward_version = config["env"].get("reward_version", "v1")

        # Gripper thresholds
        self.gripper_threshold = config["robot"]["gripper_threshold"]
        self.gripper_cube_jaw  = config["robot"]["gripper_cube_jaw"]

        obs_low  = np.full(self.OBS_DIM, -np.inf, dtype=np.float32)
        obs_high = np.full(self.OBS_DIM,  np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self._best_dist         = None       # personal-best 3d distance (v1)
        self._best_xy_dist      = None       # personal-best horizontal distance (v2)
        self._best_z_gap        = None       # personal-best z gap gripper→cube (v2, smaller=better)
        self._best_obj_z        = 0.0        # set properly at reset() to actual cube z
        self._best_wrist_score  = 0.0        # personal-best orient (v1)
        self._best_align_score  = 0.0        # personal-best align (v1)
        self._best_jaw_close    = 0.0        # personal-best closedness in jaw_align zone (v2)
        self._hold_collected    = False      # one-shot hold bonus

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        self._reset_common(seed)
        self._best_dist        = None
        self._best_xy_dist     = None
        self._best_z_gap       = None
        self._best_obj_z       = float(self.data.qpos[self.object_qpos_addr + 2])
        self._best_wrist_score = 0.0
        self._best_align_score = 0.0
        self._best_jaw_close   = 0.0
        self._hold_collected   = False
        obs = self._get_obs()
        return obs, {}

    # ------------------------------------------------------------------
    def step(self, action):
        self._do_step(action)

        obs    = self._get_obs()
        if self._reward_version == "v2":
            reward, breakdown = self._compute_reward_v2()
        else:
            reward, breakdown = self._compute_reward()
        success = self._is_success()
        truncated = self._step_count >= self.max_episode_steps
        terminated = success

        return obs, reward, terminated, truncated, {"success": success, "reward_breakdown": breakdown}

    # ------------------------------------------------------------------
    def _get_obs(self):
        ja, gs, op, ov, rgo = self._get_common_obs()
        return np.concatenate([ja, gs, op, ov, rgo]).astype(np.float32)

    # ------------------------------------------------------------------
    def _compute_reward(self):
        _, grasp_site_pos, object_pos, _, _ = self._get_common_obs()
        jaw_angle = float(self.data.qpos[5])  # 0=closed, 1.75=fully open
        dist  = float(np.linalg.norm(grasp_site_pos - object_pos))
        obj_z = float(object_pos[2])

        centering_score = float(np.exp(-(dist ** 2) / (self.grasp_sigma ** 2)))
        closedness = 1.0 - jaw_angle / 1.75  # 1=fermée, 0=ouverte

        b = {}  # reward breakdown dict

        # 1. Reach: base negative penalty (always pulls toward cube) +
        #    PERSONAL-BEST progress bonus: only rewards setting a new closest distance.
        #    Eliminates yo-yo exploit (approach → back off → approach → collect bonus again).
        b["dist"] = -self.dist_weight * dist
        if self._best_dist is None or dist < self._best_dist:
            gain = (self._best_dist - dist) if self._best_dist is not None else 0.0
            b["dist"] += self.dist_progress_bonus * gain
            self._best_dist = dist

        # 2. Alignment bonus — personal-best, lifetime for the episode.
        #    No zone reset: oscillating out and back can never re-collect.
        b["align"] = 0.0
        if dist < 0.10:
            align_improvement = max(0.0, centering_score - self._best_align_score)
            b["align"] = self.alignment_bonus * align_improvement
            self._best_align_score = max(self._best_align_score, centering_score)
        # no reset on exit — best is kept for the whole episode

        # 3. Approach open penalty
        b["appr_open"] = -self.approach_open_weight * closedness if dist > 0.05 else 0.0

        # 4. Wrist-roll orientation bonus — personal-best, lifetime for the episode.
        #    No zone reset: oscillating across the 10cm boundary no longer re-triggers.
        b["orient"] = 0.0
        if dist < 0.15:
            wrist_roll  = float(self.data.qpos[4])
            dev         = min(abs(wrist_roll - 1.5707963), abs(wrist_roll + 1.5707963))
            wrist_score = max(0.0, 1.0 - dev / 1.5707963)
            improvement = max(0.0, wrist_score - self._best_wrist_score)
            b["orient"] = self.orient_bonus * improvement
            self._best_wrist_score = max(self._best_wrist_score, wrist_score)
        # no reset on exit — best is kept for the whole episode

        # 5. Grasp bonus — PERSONAL-BEST: only fires when cube is at a new height record.
        #    Prevents micro-oscillation exploit (cube_rising with 0.1mm delta was retriggerable).
        at_new_height = obj_z > self._best_obj_z + 1e-3
        b["grasp"] = self.grasp_bonus * centering_score * closedness if (dist < 0.03 and at_new_height) else 0.0

        # 6. Jaw close bonus — same personal-best height condition
        b["jaw"] = self.jaw_close_bonus * closedness if (dist < 0.03 and at_new_height) else 0.0

        # 6b. Small unconditional jaw nudge when well-positioned (dist < 3cm).
        #     Disabled once hold is collected — chicken-and-egg already resolved,
        #     and keeping it active creates a local minimum (net positive reward doing nothing).
        b["jaw_nudge"] = self.jaw_nudge_weight * closedness if (dist < 0.03 and not self._hold_collected) else 0.0

        # 7. Hold bonus — ONE-SHOT per episode.
        #    Robot gets the bonus once for achieving a grip, then must lift for more reward.
        #    Per-step hold was exploitable: ~8/step with no incentive to lift further.
        b["hold"] = 0.0
        if dist < 0.02 and jaw_angle < self.gripper_cube_jaw and not self._hold_collected:
            b["hold"] = self.hold_bonus
            self._hold_collected = True

        # 7b. Lift pull — continuous penalty proportional to remaining height once cube is held.
        #     After hold fires, net per-step was slightly positive (jaw_nudge dominated).
        #     This creates constant upward pressure: -lift_pull × (lift_height - obj_z).
        #     At rest (obj_z=0.22): -2.0 × 0.06 = -0.12/step. Shrinks as cube rises. Zero at success.
        b["lift_pull"] = 0.0
        if self._hold_collected and jaw_angle < self.gripper_cube_jaw:
            b["lift_pull"] = -self.lift_pull_weight * max(0.0, self.lift_height - obj_z)

        # 8. Dense lift reward — PERSONAL-BEST: only rewards new height records.
        #    Lowering the cube then re-lifting to the same height = 0 reward.
        #    Robot must keep going higher to keep earning lift rewards.
        b["lift_dense"] = 0.0
        if jaw_angle < self.gripper_cube_jaw and dist < 0.04:
            height_gain = max(0.0, obj_z - self._best_obj_z)
            b["lift_dense"] = self.lift_weight * height_gain
            self._best_obj_z = max(self._best_obj_z, obj_z)

        # _best_obj_z updated above — no _prev_obj_z needed anymore

        # 9. Lift success (sparse)
        b["lift_sparse"] = self.lift_bonus if (
            obj_z > self.lift_height and jaw_angle < self.gripper_threshold) else 0.0

        # 10. Collision penalty
        b["collision"] = 0.0
        for i in range(self.data.ncon):
            c  = self.data.contact[i]
            g1 = self.model.geom(c.geom1).name
            g2 = self.model.geom(c.geom2).name
            if "table_geom" in (g1, g2) and "object_geom" not in (g1, g2):
                b["collision"] = self.collision_penalty
                break

        # 11. Per-step penalty
        b["step"] = -self.step_penalty

        # Debug scalars (excluded from total)
        b["dist_m"]  = round(dist, 4)
        b["jaw_rad"] = round(jaw_angle, 3)
        b["obj_z"]   = round(obj_z, 3)

        total = sum(v for k, v in b.items() if k not in ("dist_m", "jaw_rad", "obj_z"))
        return float(total), b

    # ------------------------------------------------------------------
    def _is_success(self):
        object_z   = float(self.data.qpos[self.object_qpos_addr + 2])
        jaw_angle  = float(self.data.qpos[5])
        return object_z > self.lift_height and jaw_angle < self.gripper_threshold

    # ------------------------------------------------------------------
    def _compute_reward_v2(self):
        """
        Staged top-down grasp reward:
          Phase 1 — approach from above with jaw open
          Phase 2 — descend to cube height while xy-centred
          Phase 3 — close jaw once aligned
          Phase 4 — lift
        All distance rewards use personal-best to prevent oscillation exploits.
        """
        _, grasp_site_pos, object_pos, _, _ = self._get_common_obs()
        jaw_angle = float(self.data.qpos[5])
        obj_z     = float(object_pos[2])
        grip_z    = float(grasp_site_pos[2])

        xy_dist = float(np.linalg.norm(grasp_site_pos[:2] - object_pos[:2]))
        z_gap   = float(grasp_site_pos[2] - object_pos[2])   # positive = above cube
        dist3d  = float(np.linalg.norm(grasp_site_pos - object_pos))
        closedness = 1.0 - jaw_angle / 1.75

        b = {}

        # --- Phase 1: approach from above (jaw should be open) ----------
        # Base pull: always penalise horizontal distance to the cube.
        b["above_xy"] = -self.v2_above_xy_weight * xy_dist
        # Personal-best XY bonus — earn only on new closest horizontal approach.
        if self._best_xy_dist is None or xy_dist < self._best_xy_dist:
            gain = (self._best_xy_dist - xy_dist) if self._best_xy_dist is not None else 0.0
            b["above_xy"] += self.v2_above_xy_bonus * gain
            self._best_xy_dist = xy_dist
        # Jaw-open penalty: discourage approaching with closed jaw.
        # Per-step PENALTY (not bonus) to avoid the hovering-harvest exploit.
        # A bonus per step was farmable: net = +openness - 3*xy_dist > 0 at xy≈3cm.
        openness = 1.0 - closedness
        b["above_open"] = -self.v2_above_open_weight * closedness if xy_dist > 0.05 else 0.0

        # --- Phase 2: descend to cube height (active when xy < 5cm) -----
        # Only reward/penalise z once horizontally aligned enough.
        b["descend"] = 0.0
        b["xy_lock"]  = 0.0
        if xy_dist < 0.05:
            # Base pull: penalise remaining z gap above cube.
            z_gap_clamped = max(0.0, z_gap)
            b["descend"] = -self.v2_descend_weight * z_gap_clamped
            # Personal-best z gap bonus.
            if self._best_z_gap is None or z_gap_clamped < self._best_z_gap:
                gain = (self._best_z_gap - z_gap_clamped) if self._best_z_gap is not None else 0.0
                b["descend"] += self.v2_descend_bonus * gain
                self._best_z_gap = z_gap_clamped
            # Penalty for xy drift while descending.
            if z_gap_clamped < 0.04 and xy_dist > 0.03:
                b["xy_lock"] = -self.v2_xy_lock_penalty * xy_dist

        # --- Phase 3: close jaw when well-aligned ----------------------
        # Base pull (small, continuous) + personal-best bonus.
        # Base is needed to bootstrap gradient when jaw is fully open (improvement=0 from cold start).
        # Base (0.3) < lift_pull (0.36) so parking fully closed is still net negative.
        b["jaw_align"] = 0.0
        if xy_dist < 0.03 and abs(z_gap) < 0.015 and not self._hold_collected:
            b["jaw_align"] = self.v2_jaw_align_base * closedness
            improvement = max(0.0, closedness - self._best_jaw_close)
            if improvement > 0.0:
                b["jaw_align"] += self.v2_jaw_align_bonus * improvement
                self._best_jaw_close = max(self._best_jaw_close, closedness)

        # Hold bonus — one-shot when grip fully confirmed.
        b["hold"] = 0.0
        if dist3d < 0.02 and jaw_angle < self.gripper_cube_jaw and not self._hold_collected:
            b["hold"] = self.v2_hold_bonus
            self._hold_collected = True

        # --- Phase 4: lift ---------------------------------------------
        b["lift_dense"] = 0.0
        b["lift_pull"]  = 0.0
        b["drop"]       = 0.0
        if self._hold_collected:
            if jaw_angle < self.gripper_cube_jaw:
                # Personal-best height.
                height_gain = max(0.0, obj_z - self._best_obj_z)
                b["lift_dense"] = self.v2_lift_weight * height_gain
                self._best_obj_z = max(self._best_obj_z, obj_z)
                # Continuous upward pressure.
                b["lift_pull"] = -self.v2_lift_pull_weight * max(0.0, self.v2_lift_height - obj_z)
            else:
                # Drop penalty — lâcher le cube après hold est puni.
                # Sans ça, ouvrir la mâchoire soulage la pression lift_pull (exploit).
                b["drop"] = self.v2_drop_penalty

        # Lift success (sparse).
        b["lift_sparse"] = self.v2_lift_bonus if (
            obj_z > self.v2_lift_height and jaw_angle < self.gripper_threshold) else 0.0

        # Collision penalty.
        b["collision"] = 0.0
        for i in range(self.data.ncon):
            c  = self.data.contact[i]
            g1 = self.model.geom(c.geom1).name
            g2 = self.model.geom(c.geom2).name
            if "table_geom" in (g1, g2) and "object_geom" not in (g1, g2):
                b["collision"] = self.v2_collision_penalty
                break

        # Per-step penalty.
        b["step"] = -self.v2_step_penalty

        # Debug scalars.
        b["dist_m"]  = round(dist3d, 4)
        b["jaw_rad"] = round(jaw_angle, 3)
        b["obj_z"]   = round(obj_z, 3)

        total = sum(v for k, v in b.items() if k not in ("dist_m", "jaw_rad", "obj_z"))
        return float(total), b

    # ------------------------------------------------------------------
    def step_absolute(self, action):
        """
        Step with absolute joint-angle targets (6-DOF ctrl array).
        Used by the classical planner — bypasses the delta action space.
        """
        self.data.ctrl[:] = np.clip(
            action,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )
        mujoco.mj_step(self.model, self.data, self.frame_skip)
        self._step_count += 1
        success   = self._is_success()
        truncated = self._step_count >= self.max_episode_steps
        return self._get_obs(), 0.0, success, truncated, {"success": success}

    # ------------------------------------------------------------------
    def get_state_snapshot(self):
        """Returns (qpos, qctrl) so Task_manager can hand state to PlaceEnv."""
        return self.data.qpos.copy(), self.data.ctrl.copy()


# ===========================================================================
# PLACE environment
# ===========================================================================
class PlaceEnv(_SO100BaseEnv):
    """
    Task: carry the already-grasped cube to goal_pos and release it there.

    Observation (21-dim flat):
        [joint_angles(6) | grasp_site_pos(3) | object_pos(3) | object_vel(3)
         | rel_grasp_obj(3) | rel_obj_goal(3)]

    Initialisation: call env.reset_from_pick(qpos, qctrl) instead of env.reset()
    after a successful pick to inject the picked state.
    """

    OBS_DIM = 23  # 18 + 3 (obj→goal) + 1 (jaw_angle) + 1 (dist grasp→obj)

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)

        # Reward weights
        self.goal_dist_weight  = config["rewards_place"]["goal_dist_weight"]
        self.dist_dense_weight = config["rewards_place"]["dist_dense_weight"]
        self.drop_penalty      = config["rewards_place"]["drop_penalty"]
        self.release_bonus     = config["rewards_place"]["release_bonus"]
        self.place_threshold   = config["rewards_place"]["place_threshold"]

        obs_low  = np.full(self.OBS_DIM, -np.inf, dtype=np.float32)
        obs_high = np.full(self.OBS_DIM,  np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self._prev_goal_dist = None

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        """Standard reset: places the cube in the gripper at a lifted start pose."""
        self._reset_common(seed)
        # Move object to just above the gripper so the episode can start held
        grasp_pos = self.data.site_xpos[self.grasp_site_id].copy()
        self.data.qpos[self.object_qpos_addr:self.object_qpos_addr + 3] = grasp_pos
        # Close the jaw
        self.data.qpos[5] = 0.0
        self.data.ctrl[5] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._prev_goal_dist = None
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def reset_from_pick(self, qpos, qctrl):
        """
        Inject the MuJoCo state directly from a finished PickEnv episode.
        Call this instead of reset() when chaining with Task_manager.
        """
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos
        self.data.ctrl[:] = qctrl
        # Re-derive goal_pos (same logic as _reset_common, but without randomising object)
        self.goal_pos = (
            self.model.body("g_goal").pos.copy()
            + np.array([0.0, 0.0, self.goal_geom_height + self.object_geom_height])
        )
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        self._prev_goal_dist = None
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        self._do_step(action)

        obs    = self._get_obs()
        reward = self._compute_reward()
        success = self._is_success()
        truncated = self._step_count >= self.max_episode_steps
        terminated = success

        return obs, reward, terminated, truncated, {"success": success}

    # ------------------------------------------------------------------
    def _get_obs(self):
        ja, gs, op, ov, rgo = self._get_common_obs()
        rel_obj_goal = (op - self.goal_pos.astype(np.float32))  # direction object → goal
        jaw_angle    = np.array([self.data.qpos[5]], dtype=np.float32)
        dist_grasp   = np.array([np.linalg.norm(gs - op)], dtype=np.float32)
        return np.concatenate([ja, gs, op, ov, rgo, rel_obj_goal, jaw_angle, dist_grasp]).astype(np.float32)

    # ------------------------------------------------------------------
    def _compute_reward(self):
        _, grasp_site_pos, object_pos, _, _ = self._get_common_obs()
        jaw_angle  = float(self.data.qpos[5])
        goal_dist  = float(np.linalg.norm(object_pos - self.goal_pos))
        grasp_dist = float(np.linalg.norm(grasp_site_pos - object_pos))
        cube_dropped = grasp_dist > 0.08  # cube no longer held

        reward = 0.0

        if cube_dropped:
            # Flat per-step penalty regardless of where the cube landed
            reward += self.drop_penalty
        else:
            # Potential-based: reward reducing distance to goal while holding
            if self._prev_goal_dist is not None:
                reward += self.goal_dist_weight * (self._prev_goal_dist - goal_dist)
            # Absolute density: always pull toward goal even from still
            reward -= self.dist_dense_weight * goal_dist

        self._prev_goal_dist = goal_dist

        # Sparse: bonus for releasing at the goal
        if goal_dist < self.place_threshold and jaw_angle > 0.5:
            reward += self.release_bonus

        return float(reward)

    # ------------------------------------------------------------------
    def _is_success(self):
        object_pos = self.data.qpos[self.object_qpos_addr:self.object_qpos_addr + 3]
        jaw_angle  = float(self.data.qpos[5])
        goal_dist  = float(np.linalg.norm(object_pos - self.goal_pos))
        return goal_dist < self.place_threshold and jaw_angle > 0.5


# ===========================================================================
# V3 — REACH environment
# ===========================================================================
class ReachEnv(_SO100BaseEnv):
    """
    V3 Stage 1: center the open gripper over the cube.

    The robot starts at home pose with the cube at a random XY position.
    Goal: minimize 3D distance from grasp_site to cube, jaw must stay open.

    Success: dist3d < success_dist AND jaw_angle > 1.0 rad (clearly open).
    Observation (18-dim): same layout as PickEnv.
    """

    OBS_DIM = 18

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        c = config["rewards_v3_reach"]
        self.dist_weight          = c["dist_weight"]
        self.dist_bonus           = c["dist_bonus"]
        self.centering_bonus      = c["centering_bonus"]
        self.centering_sigma      = c["centering_sigma"]
        self.jaw_open_weight      = c["jaw_open_weight"]
        self.wrist_orient_penalty    = c["wrist_orient_penalty"]
        self.wrist_pitch_penalty     = c["wrist_pitch_penalty"]  # penalise Wrist_Pitch ≠ 0 (qpos[3])
        self.approach_height_penalty = c["approach_height_penalty"]  # top-down approach penalty
        self.approach_z_clearance    = c["approach_z_clearance"]     # min z gap during lateral approach
        self.hold_steps_required  = c["hold_steps_required"]  # steps at 50Hz for 2s = 100
        self.hold_per_step_bonus  = c["hold_per_step_bonus"]
        self.success_dist         = c["success_dist"]
        self.success_bonus        = c["success_bonus"]
        self.collision_penalty    = c["collision_penalty"]
        self.step_penalty         = c["step_penalty"]
        self.gripper_cube_jaw     = config["robot"]["gripper_cube_jaw"]

        obs_low  = np.full(self.OBS_DIM, -np.inf, dtype=np.float32)
        obs_high = np.full(self.OBS_DIM,  np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self._best_dist      = None
        self._best_centering = 0.0
        self._hold_steps     = 0   # consecutive steps in success zone

    def reset(self, seed=None, options=None):
        self._reset_common(seed)
        self._best_dist      = None
        self._best_centering = 0.0
        self._hold_steps     = 0
        return self._get_obs(), {}

    def _get_obs(self):
        ja, gs, op, ov, rgo = self._get_common_obs()
        return np.concatenate([ja, gs, op, ov, rgo]).astype(np.float32)

    def step(self, action):
        self._do_step(action)
        obs = self._get_obs()
        reward, breakdown = self._compute_reward()
        success   = self._is_success()
        truncated = self._step_count >= self.max_episode_steps
        return obs, reward, success, truncated, {"success": success, "reward_breakdown": breakdown}

    def _compute_reward(self):
        _, gs, op, _, _ = self._get_common_obs()
        jaw_angle  = float(self.data.qpos[5])
        dist       = float(np.linalg.norm(gs - op))
        closedness = 1.0 - jaw_angle / 1.75
        centering  = float(np.exp(-(dist ** 2) / (self.centering_sigma ** 2)))
        b = {}

        # Personal-best approach pull.
        # Soft reset: if the robot has backed off > 8cm beyond its best distance
        # (e.g. blocked, repositioning for a better angle), the personal-best is
        # reset so it can earn the full approach bonus again on re-approach.
        b["dist"] = -self.dist_weight * dist
        if self._best_dist is not None and dist > self._best_dist + 0.08:
            self._best_dist = None   # allow fresh bonuses on re-approach
        if self._best_dist is None or dist < self._best_dist:
            gain = (self._best_dist - dist) if self._best_dist is not None else 0.0
            b["dist"] += self.dist_bonus * gain
            self._best_dist = dist

        # Personal-best centering score (active when dist < 8cm)
        b["centering"] = 0.0
        if dist < 0.08:
            improvement = max(0.0, centering - self._best_centering)
            b["centering"] = self.centering_bonus * improvement
            self._best_centering = max(self._best_centering, centering)

        # Penalise closed jaw during approach
        b["jaw_open"] = -self.jaw_open_weight * closedness if dist > 0.03 else 0.0

        # Top-down approach: require gripper to stay above cube during lateral approach.
        # When xy_dist > 5cm the gripper must maintain at least approach_z_clearance
        # above the cube, preventing a side collision on the way in.
        # Penalty is inactive inside 5cm so the robot can freely descend in the final phase.
        xy_dist_r = float(np.linalg.norm(gs[:2] - op[:2]))
        z_gap_r   = float(gs[2] - op[2])   # positive = gripper above cube
        b["approach_z"] = 0.0
        if xy_dist_r > 0.05:
            gap_shortfall = max(0.0, self.approach_z_clearance - z_gap_r)
            b["approach_z"] = -self.approach_height_penalty * gap_shortfall
        wrist_roll  = float(self.data.qpos[4])
        dev_roll    = min(abs(wrist_roll - 1.5707963), abs(wrist_roll + 1.5707963))
        wrist_score = max(0.0, 1.0 - dev_roll / 1.5707963)  # 1=horizontal, 0=vertical
        b["wrist"] = -self.wrist_orient_penalty * (1.0 - wrist_score)

        # Wrist Pitch orientation: penalise pitch ≠ 0 (joint 3 should be 0 for top-down grasp)
        # Home pose has Wrist_Pitch=1.57 so this penalty fires immediately and drives it to 0.
        # Normalised by π/2: score=1 at pitch=0, score=0 at pitch=±π/2.
        wrist_pitch = float(self.data.qpos[3])
        pitch_score = max(0.0, 1.0 - abs(wrist_pitch) / 1.5707963)
        b["wrist_pitch"] = -self.wrist_pitch_penalty * (1.0 - pitch_score)

        # Hold-in-zone: small per-step bonus for staying in success zone.
        # Counter increments while in zone, resets when leaving.
        # Sparse success fires only after hold_steps_required consecutive steps.
        in_zone = dist < self.success_dist and jaw_angle > 1.0 and wrist_score > 0.5
        if in_zone:
            self._hold_steps += 1
        else:
            self._hold_steps = 0
        b["hold"] = self.hold_per_step_bonus if (in_zone and not self._is_success()) else 0.0

        # Sparse success bonus (fires once hold requirement is met)
        b["success_bonus"] = self.success_bonus if self._is_success() else 0.0

        b["collision"] = 0.0
        for i in range(self.data.ncon):
            c  = self.data.contact[i]
            g1 = self.model.geom(c.geom1).name
            g2 = self.model.geom(c.geom2).name
            if "table_geom" in (g1, g2) and "object_geom" not in (g1, g2):
                b["collision"] = self.collision_penalty
                break

        b["step"]      = -self.step_penalty
        b["dist_m"]    = round(dist, 4)
        b["jaw_rad"]   = round(jaw_angle, 3)
        b["hold_steps"] = self._hold_steps

        total = sum(v for k, v in b.items() if k not in ("dist_m", "jaw_rad", "hold_steps"))
        return float(total), b

    def _is_success(self):
        _, gs, op, _, _ = self._get_common_obs()
        dist = float(np.linalg.norm(gs - op))
        jaw  = float(self.data.qpos[5])
        wrist_roll  = float(self.data.qpos[4])
        dev         = min(abs(wrist_roll - 1.5707963), abs(wrist_roll + 1.5707963))
        wrist_score = max(0.0, 1.0 - dev / 1.5707963)
        in_zone = dist < self.success_dist and jaw > 1.0 and wrist_score > 0.5
        return in_zone and self._hold_steps >= self.hold_steps_required

    def get_state_snapshot(self):
        return self.data.qpos.copy(), self.data.ctrl.copy()


# ===========================================================================
# V3 — GRASP environment
# ===========================================================================
class GraspEnv(_SO100BaseEnv):
    """
    V3 Stage 2: close the gripper firmly on the cube.

    Standalone reset: cube is placed directly under the home-pose grasp_site
    so the robot only needs to lower slightly and close — not re-learn approach.
    Curriculum reset: inject the successful-reach qpos/qctrl from ReachEnv.

    Success: dist3d < success_dist AND jaw < gripper_cube_jaw.
    Observation (18-dim): same layout as PickEnv.
    """

    OBS_DIM = 18

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        c = config["rewards_v3_grasp"]
        self.jaw_base_weight      = c["jaw_base_weight"]
        self.jaw_bonus            = c["jaw_bonus"]
        self.dist_penalty_weight  = c["dist_penalty_weight"]
        self.wrist_orient_penalty = c["wrist_orient_penalty"]
        self.hold_bonus           = c["hold_bonus"]
        self.drift_penalty        = c["drift_penalty"]
        self.collision_penalty    = c["collision_penalty"]
        self.step_penalty         = c["step_penalty"]
        self.success_dist         = c["success_dist"]
        self.gripper_cube_jaw     = config["robot"]["gripper_cube_jaw"]

        obs_low  = np.full(self.OBS_DIM, -np.inf, dtype=np.float32)
        obs_high = np.full(self.OBS_DIM,  np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self._best_jaw_close = 0.0
        self._hold_collected = False

    def reset(self, seed=None, options=None):
        """Standalone: place cube under home grasp_site so gripper starts near cube."""
        self._reset_common(seed)
        gs = self.data.site_xpos[self.grasp_site_id].copy()
        cube_xy_noise = self.np_random.uniform(-0.025, 0.025, size=2)
        cube_rest_z   = self.model.body("object").pos[2]
        self.data.qpos[self.object_qpos_addr:self.object_qpos_addr + 3] = [
            gs[0] + cube_xy_noise[0],
            gs[1] + cube_xy_noise[1],
            cube_rest_z,
        ]
        mujoco.mj_forward(self.model, self.data)
        self._best_jaw_close = 0.0
        self._hold_collected = False
        return self._get_obs(), {}

    def reset_from_reach(self, qpos, qctrl):
        """Curriculum: inject the successful-reach state from ReachEnv."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos
        self.data.ctrl[:] = qctrl
        mujoco.mj_forward(self.model, self.data)
        self._step_count     = 0
        self._best_jaw_close = 0.0
        self._hold_collected = False
        # goal_pos not needed in GraspEnv but keep base attribute consistent
        self.goal_pos = (
            self.model.body("g_goal").pos.copy()
            + np.array([0.0, 0.0, self.goal_geom_height + self.object_geom_height])
        )
        return self._get_obs(), {}

    def get_state_snapshot(self):
        return self.data.qpos.copy(), self.data.ctrl.copy()

    def _get_obs(self):
        ja, gs, op, ov, rgo = self._get_common_obs()
        return np.concatenate([ja, gs, op, ov, rgo]).astype(np.float32)

    def step(self, action):
        self._do_step(action)
        obs = self._get_obs()
        reward, breakdown = self._compute_reward()
        success   = self._is_success()
        truncated = self._step_count >= self.max_episode_steps
        return obs, reward, success, truncated, {"success": success, "reward_breakdown": breakdown}

    def _compute_reward(self):
        _, gs, op, _, _ = self._get_common_obs()
        jaw_angle  = float(self.data.qpos[5])
        dist       = float(np.linalg.norm(gs - op))
        closedness = 1.0 - jaw_angle / 1.75
        b = {}

        # Wrist orientation penalty: horizontal wrist (joint 4 ≈ ±π/2) is required
        # to grasp the cube from the side. Vertical wrist shifts grasp_site away
        # from the cube, making dist > 0.06 unfixable without re-orienting.
        # Same formula as v1 orient_bonus but as a continuous per-step penalty.
        wrist_roll  = float(self.data.qpos[4])
        dev         = min(abs(wrist_roll - 1.5707963), abs(wrist_roll + 1.5707963))
        wrist_score = max(0.0, 1.0 - dev / 1.5707963)   # 1=horizontal, 0=vertical
        b["wrist"] = -self.wrist_orient_penalty * (1.0 - wrist_score)

        # Jaw closing: small base + personal-best progress.
        # Requires wrist_score > 0.4: robot cannot earn closing reward with vertical wrist,
        # forcing it to maintain horizontal orientation while gripping.
        b["jaw_close"] = 0.0
        if dist < 0.05 and wrist_score > 0.4:
            b["jaw_close"] = self.jaw_base_weight * closedness
            improvement = max(0.0, closedness - self._best_jaw_close)
            if improvement > 0.0:
                b["jaw_close"] += self.jaw_bonus * improvement
                self._best_jaw_close = max(self._best_jaw_close, closedness)

        # Always penalise distance from cube (stay near cube)
        b["dist_pen"] = -self.dist_penalty_weight * dist

        # Extra drift penalty when not yet gripped
        b["drift"] = 0.0
        if not self._hold_collected and dist > 0.05:
            b["drift"] = -self.drift_penalty * (dist - 0.05)

        # One-shot hold confirmation
        b["hold"] = 0.0
        if dist < 0.025 and jaw_angle < self.gripper_cube_jaw and not self._hold_collected:
            b["hold"] = self.hold_bonus
            self._hold_collected = True

        b["collision"] = 0.0
        for i in range(self.data.ncon):
            c  = self.data.contact[i]
            g1 = self.model.geom(c.geom1).name
            g2 = self.model.geom(c.geom2).name
            if "table_geom" in (g1, g2) and "object_geom" not in (g1, g2):
                b["collision"] = self.collision_penalty
                break

        b["step"]    = -self.step_penalty
        b["dist_m"]  = round(dist, 4)
        b["jaw_rad"] = round(jaw_angle, 3)

        total = sum(v for k, v in b.items() if k not in ("dist_m", "jaw_rad"))
        return float(total), b

    def _is_success(self):
        _, gs, op, _, _ = self._get_common_obs()
        dist = float(np.linalg.norm(gs - op))
        jaw  = float(self.data.qpos[5])
        wrist_roll  = float(self.data.qpos[4])
        dev         = min(abs(wrist_roll - 1.5707963), abs(wrist_roll + 1.5707963))
        wrist_score = max(0.0, 1.0 - dev / 1.5707963)
        return (self._hold_collected
                and dist < self.success_dist
                and jaw < self.gripper_cube_jaw
                and wrist_score > 0.5)   # wrist must be roughly horizontal


# ===========================================================================
# V3 — CARRY environment
# ===========================================================================
class CarryEnv(_SO100BaseEnv):
    """
    V3 Stage 3: carry the grasped cube to goal_pos, then release it there.

    Standalone reset: synthetic grasped state (home pose + closed jaw + cube
    placed at grasp_site, then physics-settled for 20 steps).
    Curriculum reset: inject the successful-grasp state from GraspEnv.

    Reward design:
      - Drop penalty (per-step) if jaw opens before reaching goal
      - Lift pull + dense lift while gripping
      - Goal-distance pull once lifted
      - One-shot release bonus when cube dropped at goal

    Success: cube within place_threshold of goal AND jaw open (> gripper_cube_jaw).
    Observation (21-dim): 18-dim base + 3-dim (cube → goal) direction.
    """

    OBS_DIM = 21  # 18 + 3 (cube→goal)

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        c = config["rewards_v3_carry"]
        self.drop_penalty         = c["drop_penalty"]
        self.goal_dist_weight     = c["goal_dist_weight"]
        self.goal_dist_progress   = c["goal_dist_progress"]  # per-step progress reward
        self.release_bonus        = c["release_bonus"]
        self.place_threshold   = c["place_threshold"]
        self.collision_penalty = c["collision_penalty"]
        self.step_penalty      = c["step_penalty"]
        self.gripper_cube_jaw  = config["robot"]["gripper_cube_jaw"]

        obs_low  = np.full(self.OBS_DIM, -np.inf, dtype=np.float32)
        obs_high = np.full(self.OBS_DIM,  np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self._prev_goal_dist    = None   # for per-step progress reward
        self._release_collected = False

    def reset(self, seed=None, options=None):
        """Synthetic grasped reset: close jaw, place cube at grasp_site, settle.

        To prevent the policy from collapsing to "return to home pose and stop",
        we randomise the arm configuration before computing grasp_site.
        This covers the space of plausible curry starting poses so the policy
        generalises when receiving a curriculum injection from GraspEnv.
        """
        self._reset_common(seed)

        # --- Randomise arm joints (joints 0-4, keep jaw closed) ---
        # Joint limits from COMMANDES.md / so_arm100.xml:
        #   0 Rotation   [-1.92,  1.92]  home=0
        #   1 Pitch       [-3.32,  0.17]  home=-1.57
        #   2 Elbow       [-0.17,  3.14]  home=1.57
        #   3 Wrist_Pitch [-1.66,  1.66]  home=1.57
        #   4 Wrist_Roll  [-2.79,  2.79]  home=-1.57
        arm_perturb = self.np_random.uniform(-0.4, 0.4, size=5).astype(np.float32)
        for i in range(5):
            lo = float(self.model.actuator_ctrlrange[i, 0])
            hi = float(self.model.actuator_ctrlrange[i, 1])
            self.data.qpos[i] = np.clip(self.data.qpos[i] + arm_perturb[i], lo, hi)
        self.data.ctrl[:5] = self.data.qpos[:5]

        # Close the jaw to grip level
        self.data.qpos[5] = self.gripper_cube_jaw
        self.data.ctrl[5] = self.gripper_cube_jaw
        mujoco.mj_forward(self.model, self.data)

        # Place cube at grasp_site of the PERTURBED configuration
        gs = self.data.site_xpos[self.grasp_site_id].copy()
        self.data.qpos[self.object_qpos_addr:self.object_qpos_addr + 3] = gs

        # Let physics settle
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        self._init_episode()
        return self._get_obs(), {}

    def reset_from_grasp(self, qpos, qctrl):
        """Curriculum: inject the successful-grasp state from GraspEnv."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos
        self.data.ctrl[:] = qctrl
        self.goal_pos = (
            self.model.body("g_goal").pos.copy()
            + np.array([0.0, 0.0, self.goal_geom_height + self.object_geom_height])
        )
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        self._init_episode()
        return self._get_obs(), {}

    def _init_episode(self):
        self._prev_goal_dist    = None
        self._release_collected = False

    def _get_obs(self):
        ja, gs, op, ov, rgo = self._get_common_obs()
        goal_dir = (self.goal_pos - op).astype(np.float32)
        return np.concatenate([ja, gs, op, ov, rgo, goal_dir]).astype(np.float32)

    def step(self, action):
        self._do_step(action)
        obs = self._get_obs()
        reward, breakdown = self._compute_reward()
        success   = self._is_success()
        truncated = self._step_count >= self.max_episode_steps
        return obs, reward, success, truncated, {"success": success, "reward_breakdown": breakdown}

    def _compute_reward(self):
        _, gs, op, _, _ = self._get_common_obs()
        jaw_angle  = float(self.data.qpos[5])
        goal_dist  = float(np.linalg.norm(op - self.goal_pos))
        jaw_open   = jaw_angle > self.gripper_cube_jaw
        at_goal    = goal_dist < self.place_threshold
        b = {}

        # Drop penalty: opening jaw before reaching goal
        b["drop"] = self.drop_penalty if (jaw_open and not at_goal and not self._release_collected) else 0.0

        # Goal approach: base pull + per-step progress reward.
        # Progress fires every step the cube gets closer — persistent gradient
        # that doesn't plateau like personal-best.
        # Gated on cube being physically held (grasp_dist < 0.07m).
        grasp_dist_r = float(np.linalg.norm(gs - op))
        cube_held    = (not jaw_open) and grasp_dist_r < 0.07
        b["goal_dist"] = 0.0
        if cube_held:
            b["goal_dist"] = -self.goal_dist_weight * goal_dist
            if self._prev_goal_dist is not None:
                progress = self._prev_goal_dist - goal_dist   # positive = closer
                b["goal_dist"] += self.goal_dist_progress * max(0.0, progress)
        self._prev_goal_dist = goal_dist if cube_held else None

        # Joint velocity penalty removed: the grasp_dist < 0.07 gate already cuts
        # off goal_dist if the cube flies away, making joint_vel redundant.
        # Keeping it caused movement cost > approach benefit, paralyzing the policy.

        # One-shot release bonus
        b["release"] = 0.0
        if jaw_open and at_goal and not self._release_collected:
            b["release"] = self.release_bonus
            self._release_collected = True

        b["collision"] = 0.0
        for i in range(self.data.ncon):
            c  = self.data.contact[i]
            g1 = self.model.geom(c.geom1).name
            g2 = self.model.geom(c.geom2).name
            if "table_geom" in (g1, g2) and "object_geom" not in (g1, g2):
                b["collision"] = self.collision_penalty
                break

        b["step"]   = -self.step_penalty
        b["dist_m"] = round(goal_dist, 4)
        b["jaw_rad"] = round(jaw_angle, 3)

        total = sum(v for k, v in b.items() if k not in ("dist_m", "jaw_rad"))
        return float(total), b

    def _is_success(self):
        op        = self.data.qpos[self.object_qpos_addr:self.object_qpos_addr + 3]
        goal_dist = float(np.linalg.norm(op - self.goal_pos))
        jaw       = float(self.data.qpos[5])
        return goal_dist < self.place_threshold and jaw > self.gripper_cube_jaw