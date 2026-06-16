from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np


class SO100PhysicalLimitsEnv(gym.Env[np.ndarray, np.ndarray]):
    """
    SO100 environment with EXACT physical joint limits.
    
    Based on actual robot specifications:
    - Motor 0 (Rotation): -1.92 to +1.92
    - Motor 1 (Pitch):    -3.32 to +0.174  (mostly negative!)
    - Motor 2 (Elbow):    -0.174 to +3.14
    - Motor 3 (Wrist_Pitch): -1.66 to +1.66
    - Motor 4 (Wrist_Roll):  -2.79 to +2.79
    - Motor 5 (Jaw):      -0.174 to +1.75
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 50}

    def __init__(
        self,
        scene_path: str | Path | None = None,
        task: str = "pick_place",
        render_mode: str | None = None,
        max_episode_steps: int = 900,
        frame_skip: int = 8,
        control_dt: float = 0.04,
        assist_grasp: bool = True,
        random_reset_pose: bool = True,
        seed: int | None = None,
    ) -> None:
        if task not in {"reach", "pick", "pick_place"}:
            raise ValueError(f"Unsupported task '{task}'. Expected reach, pick or pick_place.")
        if render_mode not in {None, "rgb_array", "human"}:
            raise ValueError("Only render_mode=None, 'rgb_array' or 'human' is supported.")

        self.task = task
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.frame_skip = frame_skip
        self.control_dt = control_dt
        self.assist_grasp = assist_grasp
        self.random_reset_pose = random_reset_pose

        self.scene_path = (
            Path(scene_path)
            if scene_path
            else Path(__file__).resolve().parent.parent
            / "ROBOT"
            / "mujoco_menagerie"
            / "trs_so_arm100"
            / "scene_pick_place.xml"
        )
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)

        self.np_random = np.random.default_rng(seed)

        self.n_actuators = self.model.nu
        if self.n_actuators != 6:
            raise RuntimeError(f"Expected 6 actuators for SO100, got {self.n_actuators}.")

        # CRITICAL: Use EXACT physical limits from the actual robot
        self.physical_ctrl_low = np.array([-1.92, -3.32, -0.174, -1.66, -2.79, -0.174], dtype=np.float32)
        self.physical_ctrl_high = np.array([1.92, 0.174, 3.14, 1.66, 2.79, 1.75], dtype=np.float32)
        
        # Safe subset of physical limits (avoid extremes)
        self.safe_ctrl_low = self.physical_ctrl_low.copy()
        self.safe_ctrl_high = self.physical_ctrl_high.copy()
        
        # For pitch (motor 1): Range is -3.32 to +0.174
        # Safe working range: -1.2 to +0.1 (avoid extreme backward fold)
        self.safe_ctrl_low[1] = max(self.physical_ctrl_low[1], -1.2)
        self.safe_ctrl_high[1] = min(self.physical_ctrl_high[1], 0.1)
        
        # For elbow (motor 2): Range is -0.174 to +3.14
        # Safe working range: 0.3 to 2.8 (avoid over-extension)
        self.safe_ctrl_low[2] = max(self.physical_ctrl_low[2], 0.3)
        self.safe_ctrl_high[2] = min(self.physical_ctrl_high[2], 2.8)

        self.act_rotation_id = 0
        self.act_pitch_id = 1
        self.act_elbow_id = 2
        self.jaw_act_id = 5

        # Smooth, careful action scaling
        self.joint_delta_scale = np.array([0.10, 0.03, 0.04, 0.03, 0.03, 0.0], dtype=np.float32)
        self.ctrl_target = np.zeros((self.n_actuators,), dtype=np.float32)

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_actuators + 1,),
            dtype=np.float32,
        )

        obs_dim = 6 + 6 + 3 + 3 + 3 + 2
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.goal_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal_site")
        self.fixed_jaw_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Fixed_Jaw")
        self.moving_jaw_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Moving_Jaw")

        if min(
            self.cube_joint_id,
            self.cube_body_id,
            self.goal_site_id,
            self.fixed_jaw_body_id,
            self.moving_jaw_body_id,
        ) < 0:
            raise RuntimeError("Failed to resolve one or more required MuJoCo names in the scene.")

        self.cube_qpos_adr = int(self.model.jnt_qposadr[self.cube_joint_id])
        self.cube_qvel_adr = int(self.model.jnt_dofadr[self.cube_joint_id])

        self.renderer: mujoco.Renderer | None = None
        self.viewer: Any | None = None
        if self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model, 480, 640)
        elif self.render_mode == "human":
            try:
                import mujoco.viewer as mujoco_viewer
                self.viewer = mujoco_viewer.launch_passive(self.model, self.data)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to launch MuJoCo human viewer."
                ) from exc

        self._episode_step = 0
        self.cube_rest_height = 0.018
        self._goal_pos = np.array([0.18, -0.08, 0.05], dtype=np.float32)
        self._attached = False
        self._attach_offset = np.zeros(3, dtype=np.float32)
        self._prev_qpos = np.zeros(6, dtype=np.float32)

    def _get_safe_initial_pose(self) -> np.ndarray:
        """
        Return a known-safe initial pose respecting physical limits.
        
        Key insight: Pitch must be NEGATIVE (robot leans forward).
        """
        # Rotation=0, Pitch=-0.8 (forward lean), Elbow=1.0, Wrists neutral, Jaw open
        return np.array([0.0, -0.8, 1.0, 0.5, -0.5, 0.0], dtype=np.float32)

    def _sample_initial_qpos(self) -> np.ndarray:
        """Sample a safe initial pose, heavily biased toward the safe pose."""
        if self.np_random.random() < 0.8:  # 80% use the safe pose
            q = self._get_safe_initial_pose().copy()
            # Add small noise
            noise = self.np_random.normal(0, 0.1, size=6).astype(np.float32)
            q = q + noise
        else:
            # Sample within safe ranges
            q = np.zeros(6, dtype=np.float32)
            for i in range(6):
                if i == 1:  # Pitch - bias toward -0.8
                    q[i] = self.np_random.normal(-0.8, 0.3)
                elif i == 2:  # Elbow - bias toward 1.0
                    q[i] = self.np_random.normal(1.0, 0.4)
                else:
                    mid = (self.safe_ctrl_low[i] + self.safe_ctrl_high[i]) / 2
                    span = (self.safe_ctrl_high[i] - self.safe_ctrl_low[i]) / 4
                    q[i] = self.np_random.normal(mid, span)
        
        q = np.clip(q, self.safe_ctrl_low, self.safe_ctrl_high)
        
        # Always start with open gripper
        q[self.jaw_act_id] = self.safe_ctrl_low[self.jaw_act_id]
        return q

    @staticmethod
    def _wrap_angle(rad: float) -> float:
        return float((rad + np.pi) % (2.0 * np.pi) - np.pi)

    def _ee_pos(self) -> np.ndarray:
        fixed = self.data.xpos[self.fixed_jaw_body_id]
        moving = self.data.xpos[self.moving_jaw_body_id]
        return ((fixed + moving) * 0.5).astype(np.float32)

    def _cube_pos(self) -> np.ndarray:
        return self.data.xpos[self.cube_body_id].astype(np.float32)

    def _jaw_is_closed(self) -> bool:
        jaw_q = float(self.data.qpos[5])
        return jaw_q > 0.65

    def _set_cube_pose(self, xyz: np.ndarray, zero_cube_vel: bool = False) -> None:
        self.data.qpos[self.cube_qpos_adr : self.cube_qpos_adr + 3] = xyz
        self.data.qpos[self.cube_qpos_adr + 3 : self.cube_qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        if zero_cube_vel:
            self.data.qvel[self.cube_qvel_adr : self.cube_qvel_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _sample_cube_and_goal(self) -> tuple[np.ndarray, np.ndarray]:
        """Sample cube and goal positions."""
        cube = np.array(
            [
                self.np_random.uniform(-0.05, 0.05),
                self.np_random.uniform(-0.32, -0.25),
                self.cube_rest_height,
            ],
            dtype=np.float32,
        )
        goal = np.array(
            [
                self.np_random.uniform(0.12, 0.24),
                self.np_random.uniform(-0.13, -0.02),
                self.np_random.uniform(0.055, 0.14),
            ],
            dtype=np.float32,
        )
        return cube, goal

    def _obs(self) -> np.ndarray:
        qpos = self.data.qpos[:6].astype(np.float32)
        qvel = self.data.qvel[:6].astype(np.float32)
        ee = self._ee_pos()
        cube = self._cube_pos()
        attached = np.array([1.0 if self._attached else 0.0], dtype=np.float32)
        jaw_closed = np.array([1.0 if self._jaw_is_closed() else 0.0], dtype=np.float32)
        return np.concatenate([qpos, qvel, ee, cube, self._goal_pos, attached, jaw_closed], dtype=np.float32)

    def _apply_assist_grasp(self) -> None:
        if not self.assist_grasp:
            return

        ee = self._ee_pos()
        cube = self._cube_pos()
        dist = np.linalg.norm(ee - cube)
        jaw_closed = self._jaw_is_closed()

        if not self._attached and jaw_closed and dist < 0.03:
            self._attached = True
            self._attach_offset = cube - ee

        if self._attached:
            if not jaw_closed:
                self._attached = False
            else:
                new_cube_pos = ee + self._attach_offset
                if new_cube_pos[2] < 0.015:
                    new_cube_pos[2] = 0.015
                self._set_cube_pose(new_cube_pos, zero_cube_vel=False)

    def _compute_posture_penalty(self, q: np.ndarray) -> float:
        """
        Compute posture penalty respecting physical limits.
        
        Key: Pitch (motor 1) should be NEGATIVE (forward lean).
        """
        penalty = 0.0
        
        # CRITICAL: Pitch should be negative (forward lean)
        pitch_angle = float(q[1])
        
        # Heavily penalize positive pitch (backward fold)
        if pitch_angle > 0.0:
            penalty += 10.0 * (pitch_angle ** 2)
        
        # Penalize too negative pitch (over-forward)
        if pitch_angle < -1.1:
            penalty += 4.0 * ((pitch_angle + 1.1) ** 2)
        
        # Reward pitch near -0.8 (good working angle)
        pitch_deviation = abs(pitch_angle + 0.8)
        penalty += 0.8 * (pitch_deviation ** 2)
        
        # Elbow should be moderately extended
        elbow_angle = float(q[2])
        if elbow_angle < 0.4:
            penalty += 5.0 * ((0.4 - elbow_angle) ** 2)
        elif elbow_angle > 2.5:
            penalty += 3.0 * ((elbow_angle - 2.5) ** 2)
        
        # Reward elbow near 1.0
        elbow_deviation = abs(elbow_angle - 1.0)
        penalty += 0.5 * (elbow_deviation ** 2)
        
        # Penalize excessive velocities
        qvel = self.data.qvel[:6]
        vel_penalty = 0.12 * np.sum(np.abs(qvel))
        penalty += vel_penalty
        
        # Penalize jerky motion
        qpos_delta = q - self._prev_qpos
        accel_penalty = 0.06 * np.sum(qpos_delta ** 2)
        penalty += accel_penalty
        
        return float(penalty)

    def _reward_success(self) -> tuple[float, bool, dict[str, Any]]:
        q = self.data.qpos[:6]
        ee = self._ee_pos()
        cube = self._cube_pos()

        dist_ee_cube = float(np.linalg.norm(ee - cube))
        dist_ee_cube_xy = float(np.linalg.norm(ee[:2] - cube[:2]))
        dist_cube_goal = float(np.linalg.norm(cube - self._goal_pos))
        cube_height = float(cube[2])
        lifted = float(np.clip((cube_height - 0.04) / 0.03, 0.0, 1.0))

        posture_penalty = self._compute_posture_penalty(q)

        rotation_use = abs(float(q[0]))
        base_use_bonus = 0.5 * float(np.clip(rotation_use / 0.5, 0.0, 1.0))

        base_heading = self._wrap_angle(float(q[0]))
        cube_angle = float(np.arctan2(cube[1], cube[0]))
        heading_diff = abs(self._wrap_angle(base_heading - cube_angle))
        heading_align = 1.0 - heading_diff / np.pi
        align_bonus = 0.65 * heading_align

        over_cube_bonus = 0.95 * float(np.clip(1.0 - dist_ee_cube_xy / 0.10, 0.0, 1.0))
        side_drift_penalty = 1.20 * abs(float(ee[0] - cube[0]))

        jaw_q = float(q[5])
        jaw_close_level = float(
            np.clip((jaw_q - 0.10) / max(self.safe_ctrl_high[5] - 0.10, 1e-6), 0.0, 1.0)
        )
        near_cube = float(np.clip(1.0 - dist_ee_cube / 0.10, 0.0, 1.0))
        gripper_prep_bonus = 0.9 * near_cube * jaw_close_level
        far_close_penalty = 0.10 * (1.0 - near_cube) * jaw_close_level

        if self.task == "reach":
            reward = (
                -dist_ee_cube
                - 2.0 * posture_penalty
                + base_use_bonus
                + align_bonus
                + over_cube_bonus
                - side_drift_penalty
                - 0.25 * far_close_penalty
            )
            success = dist_ee_cube < 0.025
        elif self.task == "pick":
            reward = (
                -0.5 * dist_ee_cube
                + 5.0 * lifted
                + (1.5 if self._attached else 0.0)
                - 1.5 * posture_penalty
                + 0.6 * align_bonus
                + 0.8 * over_cube_bonus
                - 0.7 * side_drift_penalty
                + gripper_prep_bonus
                - far_close_penalty
            )
            success = cube_height > 0.11
        else:
            reward = (
                -0.35 * dist_ee_cube
                -1.2 * dist_cube_goal
                + 4.0 * lifted
                + (2.0 if self._attached else 0.0)
                -1.2 * posture_penalty
                + 0.5 * base_use_bonus
                + 0.65 * align_bonus
                + 0.55 * over_cube_bonus
                - 0.45 * side_drift_penalty
                + 0.5 * gripper_prep_bonus
                - 0.6 * far_close_penalty
            )
            success = (dist_cube_goal < 0.035) and (cube_height > 0.07)

        if success:
            reward += 10.0

        info = {
            "dist_ee_cube": dist_ee_cube,
            "dist_ee_cube_xy": dist_ee_cube_xy,
            "dist_cube_goal": dist_cube_goal,
            "cube_height": cube_height,
            "posture_penalty": float(posture_penalty),
            "base_use_bonus": float(base_use_bonus),
            "heading_align": float(heading_align),
            "over_cube_bonus": float(over_cube_bonus),
            "side_drift_penalty": float(side_drift_penalty),
            "gripper_prep_bonus": float(gripper_prep_bonus),
            "far_close_penalty": float(far_close_penalty),
            "is_attached": float(self._attached),
            "success": float(success),
            "pitch_angle": float(q[1]),
            "elbow_angle": float(q[2]),
        }
        return float(reward), bool(success), info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self._episode_step = 0
        self._attached = False

        mujoco.mj_resetData(self.model, self.data)

        reset_qpos: np.ndarray | None = None
        if options is not None and "initial_qpos" in options:
            reset_qpos = np.asarray(options["initial_qpos"], dtype=np.float32)
            if reset_qpos.shape != (6,):
                raise ValueError("options['initial_qpos'] must be a 6D vector.")
            reset_qpos = np.clip(reset_qpos, self.safe_ctrl_low, self.safe_ctrl_high)
        elif self.random_reset_pose:
            reset_qpos = self._sample_initial_qpos()
        else:
            reset_qpos = self._get_safe_initial_pose()

        self.data.qpos[:6] = reset_qpos
        self.data.qvel[:] = 0.0
        self.ctrl_target = reset_qpos.copy()
        self.data.ctrl[:] = self.ctrl_target
        self._prev_qpos = reset_qpos.copy()

        cube, goal = self._sample_cube_and_goal()
        self._goal_pos = goal
        self.model.site_pos[self.goal_site_id] = goal
        self._set_cube_pose(cube, zero_cube_vel=True)

        for _ in range(60):
            mujoco.mj_step(self.model, self.data)

        self._set_cube_pose(cube, zero_cube_vel=True)
        mujoco.mj_forward(self.model, self.data)

        obs = self._obs()
        info = {"task": self.task, "initial_qpos": reset_qpos.copy()}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        delta = action[: self.n_actuators] * self.joint_delta_scale
        self.ctrl_target = np.clip(self.ctrl_target + delta, self.safe_ctrl_low, self.safe_ctrl_high)

        self.ctrl_target[self.jaw_act_id] = float(
            np.clip(
                self.ctrl_target[self.jaw_act_id] + action[-1] * 0.18,
                self.safe_ctrl_low[self.jaw_act_id],
                self.safe_ctrl_high[self.jaw_act_id],
            )
        )

        self.data.ctrl[:] = self.ctrl_target

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            self._apply_assist_grasp()

        self._episode_step += 1
        self._prev_qpos = self.data.qpos[:6].copy()

        reward, success, info = self._reward_success()
        terminated = success
        truncated = self._episode_step >= self.max_episode_steps
        obs = self._obs()

        if self.render_mode == "human":
            self.render()
            time.sleep(self.control_dt)

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                raise RuntimeError("render_mode is not set to 'rgb_array'.")
            self.renderer.update_scene(self.data, camera="track_cam")
            return self.renderer.render()

        if self.render_mode == "human":
            if self.viewer is None:
                raise RuntimeError("Human viewer was not initialized.")
            self.viewer.sync()
            return None

        raise RuntimeError("render_mode is None. Set render_mode to 'human' or 'rgb_array'.")

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
