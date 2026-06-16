"""
reach_cube_env.py
-----------------
Gymnasium environment for the SO-ARM100 to reach the green cube in TEST_SCENE.xml.

Designed for YOLO + SAC pipeline (inspired by sahand1807/Reacher_ObjectDetection):
  - Phase 1: YOLO detects the cube position once per episode (constant obs)
  - Phase 2: SAC learns to move the end-effector to that position

Extensible to pick-and-place: set task="pick_place" for future phases.

Run from SIMULATION/ directory:
    python reach_cube_env.py   # quick smoke-test
"""

from __future__ import annotations

import math
import pathlib
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium.spaces import Box

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = pathlib.Path(__file__).parent
SCENE_PATH = _THIS_DIR / "trs_so_arm100" / "TEST_SCENE.xml"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Home pose from so_arm100.xml keyframe (rad):  Rotation Pitch Elbow Wrist_Pitch Wrist_Roll Jaw
HOME_QPOS = np.array([0.0, -1.57, 1.57, 1.57, -1.57, 0.0], dtype=np.float64)

# Maximum per-step joint delta (rad) — scales the [-1,1] action output
# Jaw (index 5) has a non-zero scale so the policy can learn to open/close for pick-place.
# During reach-only training you may keep jaw frozen by passing a mask.
JOINT_DELTA_SCALE = np.array([0.08, 0.05, 0.05, 0.04, 0.04, 0.02], dtype=np.float64)

# Cube rests on table: table top z=0.15 + cube half-height 0.02 = 0.17 m.
# XML places it at 0.22 — sim settles it to table quickly so 0.22 is a safe constant.
CUBE_REST_Z = 0.22

# End-effector reach success radius (m)
REACH_THRESHOLD = 0.05

# Cube randomisation on table surface (matches table bounds in TEST_SCENE.xml)
CUBE_X_RANGE = (0.26, 0.54)   # table centre 0.4, half-width 0.15 + margin
CUBE_Y_RANGE = (-0.13, 0.0)

# Actuator names in order
ACTUATOR_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

# Safe ctrl limits to avoid singularities (indices match ACTUATOR_NAMES)
SAFE_PITCH_LOW  = -1.95   # prevent shoulder folding too far back
SAFE_ELBOW_LOW  =  0.05   # keep elbow away from fully-folded singularity

# --- Pick-and-place specific ---
# Goal platform top centers (x, y, z) for cube placement.
# z = body_z(0.075) + cylinder_half_h(0.075) + cube_half(0.02) = 0.17
GOAL_POSITIONS = np.array([
    [ 0.0,  0.4, 0.17],   # r_goal (red)
    [-0.4,  0.0, 0.17],   # g_goal (green)
    [ 0.0, -0.4, 0.17],   # b_goal (blue)
], dtype=np.float32)

GRASP_REACH        = 0.05   # EE must be within this distance of cube for assist-grasp (m)
GRASP_JAW_THRESH   = 0.20   # jaw open fraction below which jaw is considered "closing"
RELEASE_JAW_THRESH = 0.55   # jaw open fraction above which cube is released
LIFT_HEIGHT        = 0.05   # cube must rise this far above CUBE_REST_Z to count as lifted (m)
PLACE_THRESHOLD    = 0.08   # cube-to-goal distance for placement success (m)


class ReachCubeEnv(gym.Env):
    """
    SO-ARM100 reach & pick-and-place task.

    task="reach"  — Observation (21-dim):
        [0:6]   joint positions  (rad)
        [6:12]  joint velocities (rad/s)
        [12:15] end-effector position  (m, world frame)
        [15:18] cube position from YOLO or GT  (m, world frame)
        [18:21] vector EE → cube  (m)

    task="pick_place"  — Observation (29-dim = reach 21 + extra 8):
        [21:24] goal platform position  (m, world frame)
        [24:27] vector cube → goal  (m)
        [27]    jaw open fraction  [0=closed, 1=open]
        [28]    cube grasped flag  {0, 1}

    Action (6-dim):  normalised joint deltas in [-1, 1].

    Reward design:
        reach      : -dist(EE, cube) + bonuses when within REACH_THRESHOLD
        pick_place : hierarchical dense reward (approach → grasp → lift → transport → place)

    Parameters
    ----------
    use_yolo : bool
        Use YOLO detector for the cube position observation (requires a trained
        model at SIMULATION/models/cube_yolov10.pt).  When False, GT position is
        used — recommended during initial SAC training.
    task : "reach" | "pick_place"
    render_mode : None | "human" | "rgb_array"
    max_episode_steps : int
    frame_skip : int
    random_cube : bool
    cube_x_range / cube_y_range : override default table placement bounds
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        scene_path: str | pathlib.Path | None = None,
        use_yolo: bool = False,
        task: str = "reach",
        render_mode: str | None = None,
        max_episode_steps: int = 500,
        frame_skip: int = 8,
        random_cube: bool = True,
        seed: int | None = None,
        cube_x_range: tuple[float, float] | None = None,
        cube_y_range: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()

        assert task in ("reach", "pick_place"), f"Unknown task '{task}'"
        assert render_mode in (None, "human", "rgb_array"), f"Unknown render_mode '{render_mode}'"

        self.scene_path     = pathlib.Path(scene_path) if scene_path else SCENE_PATH
        self.use_yolo       = use_yolo
        self.task           = task
        self.render_mode    = render_mode
        self.max_episode_steps = max_episode_steps
        self.frame_skip     = frame_skip
        self.random_cube    = random_cube
        self.cube_x_range   = cube_x_range if cube_x_range is not None else CUBE_X_RANGE
        self.cube_y_range   = cube_y_range if cube_y_range is not None else CUBE_Y_RANGE

        # ------------------------------------------------------------------
        # Load MuJoCo model
        # ------------------------------------------------------------------
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data  = mujoco.MjData(self.model)
        self.np_random = np.random.default_rng(seed)

        # ------------------------------------------------------------------
        # Resolve body / joint / actuator IDs
        # ------------------------------------------------------------------
        self._resolve_ids()

        # Safe control limits
        self.ctrl_low  = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        pitch_idx  = self._act_ids.index(self._act_id("Pitch"))
        elbow_idx  = self._act_ids.index(self._act_id("Elbow"))
        self.ctrl_low[pitch_idx] = max(self.ctrl_low[pitch_idx], SAFE_PITCH_LOW)
        self.ctrl_low[elbow_idx] = max(self.ctrl_low[elbow_idx], SAFE_ELBOW_LOW)

        # ------------------------------------------------------------------
        # Spaces  (obs dim depends on task)
        # reach: 21-dim  |  pick_place: 29-dim
        # ------------------------------------------------------------------
        obs_dim = 21 if self.task == "reach" else 29
        self.action_space      = Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self.observation_space = Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        # ------------------------------------------------------------------
        # Runtime state
        # ------------------------------------------------------------------
        self._episode_step    = 0
        self._reached_once    = False
        self._grasped_once    = False
        self._placed_once     = False
        self._attached        = False
        self._attach_offset   = np.zeros(3, dtype=np.float64)  # cube pos relative to EE when grasped
        self._goal_pos        = GOAL_POSITIONS[0].copy()        # picked at each reset for pick_place
        self._cube_obs_pos    = np.zeros(3, dtype=np.float32)   # set at each reset
        self.last_reward      = 0.0
        self.reward_info: dict[str, float] = {}

        # ------------------------------------------------------------------
        # Rendering
        # ------------------------------------------------------------------
        self._renderer: mujoco.Renderer | None = None
        self._viewer: Any | None = None
        if render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self.model, 480, 640)
        elif render_mode == "human":
            import mujoco.viewer as mv
            self._viewer = mv.launch_passive(self.model, self.data)

        # YOLO detector (lazy-loaded on first use)
        self._yolo_detector = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _act_id(self, name: str) -> int:
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if i < 0:
            raise RuntimeError(f"Actuator '{name}' not found in MuJoCo model.")
        return i

    def _resolve_ids(self) -> None:
        def _body(name: str) -> int:
            i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if i < 0: raise RuntimeError(f"Body '{name}' not found.")
            return i
        def _jnt(name: str) -> int:
            i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if i < 0: raise RuntimeError(f"Joint '{name}' not found.")
            return i

        self._obj_body_id   = _body("object")
        self._fixed_jaw_id  = _body("Fixed_Jaw")
        self._moving_jaw_id = _body("Moving_Jaw")

        obj_jnt = _jnt("object_joint")
        self._obj_qpos_adr = int(self.model.jnt_qposadr[obj_jnt])
        self._obj_qvel_adr = int(self.model.jnt_dofadr[obj_jnt])

        rot_jnt = _jnt("Rotation")
        self._arm_qpos_start = int(self.model.jnt_qposadr[rot_jnt])
        self._arm_qvel_start = int(self.model.jnt_dofadr[rot_jnt])

        # Ordered actuator id list matching ACTUATOR_NAMES / JOINT_DELTA_SCALE
        self._act_ids = [self._act_id(n) for n in ACTUATOR_NAMES]

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _ee_pos(self) -> np.ndarray:
        """Midpoint between Fixed_Jaw and Moving_Jaw tips."""
        fixed  = self.data.xpos[self._fixed_jaw_id]
        moving = self.data.xpos[self._moving_jaw_id]
        return ((fixed + moving) * 0.5).astype(np.float32)

    def _cube_pos_gt(self) -> np.ndarray:
        return self.data.xpos[self._obj_body_id].astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos[self._arm_qpos_start : self._arm_qpos_start + 6].astype(np.float32)
        qvel = self.data.qvel[self._arm_qvel_start : self._arm_qvel_start + 6].astype(np.float32)
        ee   = self._ee_pos()
        cube = self._cube_obs_pos
        vec  = cube - ee
        base = np.concatenate([qpos, qvel, ee, cube, vec], dtype=np.float32)
        if self.task == "reach":
            return base
        # pick_place extra fields
        goal         = self._goal_pos.astype(np.float32)
        cube_to_goal = goal - cube
        jaw_open     = np.array([self._jaw_open_frac()], dtype=np.float32)
        grasped      = np.array([1.0 if self._attached else 0.0], dtype=np.float32)
        return np.concatenate([base, goal, cube_to_goal, jaw_open, grasped], dtype=np.float32)

    # ------------------------------------------------------------------
    # Pick-and-place helpers
    # ------------------------------------------------------------------

    def _jaw_open_frac(self) -> float:
        """Jaw open fraction in [0, 1].  0 = fully closed, 1 = fully open."""
        jaw_act_id = self._act_ids[5]  # Jaw is last in ACTUATOR_NAMES
        ctrl = float(self.data.ctrl[jaw_act_id])
        lo   = float(self.ctrl_low[jaw_act_id])
        hi   = float(self.ctrl_high[jaw_act_id])
        return float(np.clip((ctrl - lo) / (hi - lo + 1e-9), 0.0, 1.0))

    def _apply_assist_grasp(self) -> None:
        """
        Teleport the cube to follow the EE when the jaw is closing near the cube.
        Release when the jaw opens again above RELEASE_JAW_THRESH.
        """
        jaw_frac = self._jaw_open_frac()
        if not self._attached:
            ee   = self._ee_pos().astype(np.float64)
            cube = self._cube_pos_gt().astype(np.float64)
            dist = float(np.linalg.norm(ee - cube))
            if dist < GRASP_REACH and jaw_frac < GRASP_JAW_THRESH:
                self._attached      = True
                self._grasped_once  = True
                self._attach_offset = cube - ee
        if self._attached:
            if jaw_frac > RELEASE_JAW_THRESH:
                self._attached = False
            else:
                ee      = self._ee_pos().astype(np.float64)
                new_pos = ee + self._attach_offset
                self.data.qpos[self._obj_qpos_adr     : self._obj_qpos_adr + 3] = new_pos
                self.data.qpos[self._obj_qpos_adr + 3 : self._obj_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
                self.data.qvel[self._obj_qvel_adr      : self._obj_qvel_adr + 6] = 0.0

    def _compute_pick_place_reward(self) -> float:
        """
        Hierarchical dense reward:
          1. Approach: drive EE → cube     (-dist_ee_cube)
          2. Grasp:    first-attach bonus  (+5)
          3. Lift:     cube height bonus   (height * 5)
          4. Transport: drive cube → goal  (-dist_cube_goal * 2)
          5. Place:    placement success   (+30 once, +10 on every step close)
        """
        r = 0.0
        ee       = self._ee_pos()
        cube_gt  = self._cube_pos_gt()
        goal     = self._goal_pos.astype(np.float32)

        dist_ee_cube   = float(np.linalg.norm(ee   - cube_gt))
        dist_cube_goal = float(np.linalg.norm(cube_gt - goal))
        cube_z         = float(cube_gt[2])

        self.reward_info = {
            "dist_ee_cube":   dist_ee_cube,
            "dist_cube_goal": dist_cube_goal,
            "attached":       float(self._attached),
        }

        if not self._attached:
            # Phase 1: approach cube
            r += -dist_ee_cube
        else:
            # Phase 2: once-per-episode grasp bonus
            if self._grasped_once:
                r += 5.0
                self._grasped_once = False   # consume

            # Phase 3: lift bonus
            lift = max(0.0, cube_z - CUBE_REST_Z)
            r += lift * 5.0

            # Phase 4: transport towards goal
            r += -dist_cube_goal * 2.0

        # Phase 5: placed successfully (cube near goal and not held)
        if not self._attached and dist_cube_goal < PLACE_THRESHOLD:
            r += 10.0
            if not self._placed_once:
                r += 30.0
                self._placed_once = True
                self.reward_info["placed"] = 1.0

        r += self._joint_limit_penalty()
        return r

    # ------------------------------------------------------------------
    # Reset helpers
    # ------------------------------------------------------------------

    def _place_cube(self) -> None:
        """Place cube on table, optionally at a random position."""
        if self.random_cube:
            x = float(self.np_random.uniform(*self.cube_x_range))
            y = float(self.np_random.uniform(*self.cube_y_range))
        else:
            x = float((self.cube_x_range[0] + self.cube_x_range[1]) / 2)
            y = 0.0
        z = CUBE_REST_Z
        self.data.qpos[self._obj_qpos_adr     : self._obj_qpos_adr + 3] = [x, y, z]
        self.data.qpos[self._obj_qpos_adr + 3 : self._obj_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self._obj_qvel_adr      : self._obj_qvel_adr + 6] = 0.0

    def _get_cube_detection(self) -> np.ndarray:
        """
        Return the cube's world position.
        If use_yolo=True, runs the YOLO detector (lazy-loaded).
        Falls back to GT if YOLO fails.
        """
        if not self.use_yolo:
            return self._cube_pos_gt()

        if self._yolo_detector is None:
            from yolo_detector import CubeDetector
            self._yolo_detector = CubeDetector(self.model, self.data)

        detected = self._yolo_detector.detect(self.data, cube_world_z=CUBE_REST_Z)
        if detected is None:
            print("[ReachCubeEnv] YOLO detection failed — using GT fallback.")
            return self._cube_pos_gt()
        return detected

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, ee: np.ndarray, cube: np.ndarray) -> float:
        if self.task == "pick_place":
            return self._compute_pick_place_reward()

        # --- reach reward ---
        dist = float(np.linalg.norm(ee - cube))
        r = -dist
        self.reward_info = {"dist": dist}

        if dist < REACH_THRESHOLD:
            r += 2.0
            self.reward_info["close_bonus"] = 2.0
            if not self._reached_once:
                r += 10.0
                self._reached_once = True
                self.reward_info["reach_bonus"] = 10.0

        joint_penalty = self._joint_limit_penalty()
        r += joint_penalty
        if joint_penalty < 0.0:
            self.reward_info["joint_penalty"] = joint_penalty

        return r

    def _joint_limit_penalty(self) -> float:
        penalty = 0.0
        qpos = self.data.qpos[self._arm_qpos_start : self._arm_qpos_start + 6]
        for i, act_id in enumerate(self._act_ids):
            lo, hi = self.ctrl_low[act_id], self.ctrl_high[act_id]
            margin = 0.05 * (hi - lo)
            if qpos[i] < lo + margin:
                penalty -= (lo + margin - qpos[i]) * 10.0
            elif qpos[i] > hi - margin:
                penalty -= (qpos[i] - (hi - margin)) * 10.0
        return penalty

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        # Arm home position
        self.data.qpos[self._arm_qpos_start : self._arm_qpos_start + 6] = HOME_QPOS
        for i, act_id in enumerate(self._act_ids):
            self.data.ctrl[act_id] = HOME_QPOS[i]

        # Cube placement
        self._place_cube()
        mujoco.mj_forward(self.model, self.data)

        # Cube position observation — fixed for the entire episode (YOLO or GT)
        self._cube_obs_pos = self._get_cube_detection()

        self._episode_step  = 0
        self._reached_once  = False
        self._grasped_once  = False
        self._placed_once   = False
        self._attached      = False
        self._attach_offset = np.zeros(3, dtype=np.float64)

        # Pick a random goal platform for this episode
        if self.task == "pick_place":
            self._goal_pos = GOAL_POSITIONS[
                int(self.np_random.integers(0, len(GOAL_POSITIONS)))
            ].copy()

        self.last_reward  = 0.0
        self.reward_info  = {}

        if self._viewer is not None:
            self._viewer.sync()

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float64)

        # Apply joint deltas, clipped to safe limits
        for i, act_id in enumerate(self._act_ids):
            current  = float(self.data.ctrl[act_id])
            delta    = float(action[i]) * JOINT_DELTA_SCALE[i]
            new_ctrl = float(np.clip(current + delta, self.ctrl_low[act_id], self.ctrl_high[act_id]))
            self.data.ctrl[act_id] = new_ctrl

        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        # For pick_place: apply assist-grasp and keep cube obs up-to-date (cube moves)
        if self.task == "pick_place":
            self._apply_assist_grasp()
            self._cube_obs_pos = self._cube_pos_gt()

        ee     = self._ee_pos()
        cube   = self._cube_obs_pos
        reward = self._compute_reward(ee, cube)
        self.last_reward = reward

        self._episode_step += 1
        terminated = False
        truncated  = self._episode_step >= self.max_episode_steps

        # Early termination: cube fell off table (only penalise if not being held)
        cube_gt = self._cube_pos_gt()
        if cube_gt[2] < 0.05 and not self._attached:
            terminated = True
            reward -= 5.0

        # Early success termination for pick_place
        if self.task == "pick_place" and self._placed_once:
            terminated = True

        if self._viewer is not None:
            self._viewer.sync()

        dist_key = "dist_cube_goal" if self.task == "pick_place" else "distance"
        success  = self._placed_once if self.task == "pick_place" else self._reached_once
        info = {
            "distance":    float(self.reward_info.get(dist_key, np.linalg.norm(ee - cube))),
            "success":     success,
            **self.reward_info,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array" and self._renderer is not None:
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
        if self._viewer is not None:
            self._viewer.close()
        if self._yolo_detector is not None:
            self._yolo_detector.close()


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for _task in ("reach", "pick_place"):
        print(f"\n=== ReachCubeEnv smoke-test  task='{_task}' ===")
        env = ReachCubeEnv(task=_task, random_cube=True, render_mode=None, max_episode_steps=20)
        obs, _ = env.reset(seed=42)
        print(f"  Obs shape  : {obs.shape}")
        print(f"  Action     : {env.action_space}")
        if _task == "pick_place":
            print(f"  Goal pos   : {obs[21:24]}")
        for step_i in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  step {step_i:2d}  reward={reward:.4f}  dist={info['distance']:.4f}  "
                  f"success={info['success']}  done={terminated or truncated}")
            if terminated or truncated:
                break
        env.close()
    print("\nSmoke-test passed.")
