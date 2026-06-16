from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np


class SO100PickPlaceEnv(gym.Env[np.ndarray, np.ndarray]):
    """MuJoCo Gymnasium environment for SO100 reach / pick / pick-and-place tasks."""

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
        random_reset_pose: bool = False,
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

        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        # Use safer control limits to avoid backward-folded postures.
        self.safe_ctrl_low = self.ctrl_low.copy()
        self.safe_ctrl_high = self.ctrl_high.copy()

        self.act_rotation_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "Rotation")
        self.act_pitch_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "Pitch")
        self.act_elbow_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "Elbow")

        if min(self.act_rotation_id, self.act_pitch_id, self.act_elbow_id) < 0:
            raise RuntimeError("Failed to resolve one or more SO100 actuator names.")

        # Prevent the shoulder from folding too far backward.
        self.safe_ctrl_low[self.act_pitch_id] = max(self.safe_ctrl_low[self.act_pitch_id], -1.95)
        # Keep elbow away from the near-singular fully folded region.
        self.safe_ctrl_low[self.act_elbow_id] = max(self.safe_ctrl_low[self.act_elbow_id], 0.05)

        # Give more authority to base rotation than upper joints.
        self.joint_delta_scale = np.array([0.18, 0.05, 0.05, 0.045, 0.045, 0.0], dtype=np.float32)
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

        self.jaw_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "Jaw")
        self.cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.goal_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal_site")
        self.fixed_jaw_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Fixed_Jaw")
        self.moving_jaw_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Moving_Jaw")

        if min(
            self.jaw_act_id,
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
                    "Failed to launch MuJoCo human viewer. If you are on a headless session, use --save_gif instead."
                ) from exc

        self._episode_step = 0
        self.cube_rest_height = 0.018
        self._goal_pos = np.array([0.18, -0.08, 0.05], dtype=np.float32)
        self._attached = False
        self._attach_offset = np.zeros(3, dtype=np.float32)

    def _sample_initial_qpos(self) -> np.ndarray:
        q = self.np_random.uniform(self.safe_ctrl_low, self.safe_ctrl_high).astype(np.float32)
        # Start from open gripper so the policy must learn an explicit close action.
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
        # Keep cube farther in front of the robot (negative y), not near the base.
        cube = np.array(
            [
                self.np_random.uniform(-0.06, 0.06),
                self.np_random.uniform(-0.35, -0.27),
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

        if self._attached and not jaw_closed:
            self._attached = False

        if self._attached:
            anchored = ee + self._attach_offset
            anchored[2] = max(anchored[2], self.cube_rest_height)
            self._set_cube_pose(anchored, zero_cube_vel=True)

    def _reward_success(self) -> tuple[float, bool, dict[str, float]]:
        ee = self._ee_pos()
        cube = self._cube_pos()
        goal = self._goal_pos
        q = self.data.qpos[:6]

        dist_ee_cube = float(np.linalg.norm(ee - cube))
        dist_ee_cube_xy = float(np.linalg.norm(ee[:2] - cube[:2]))
        dist_cube_goal = float(np.linalg.norm(cube - goal))
        cube_height = float(cube[2])
        lifted = max(0.0, cube_height - self.cube_rest_height)

        pitch = float(q[1])
        elbow = float(q[2])
        rotation = float(q[0])

        # Penalize backward leaning and excessive upper-arm folding.
        posture_penalty = 0.0
        if pitch < -1.8:
            posture_penalty += 2.2 * (-1.8 - pitch)
        posture_penalty += 0.22 * abs(pitch + 1.35)
        posture_penalty += 0.10 * abs(elbow - 1.55)

        # Reward using base rotation when cube is laterally offset (x-axis when front is -y).
        lateral_factor = float(np.clip(abs(cube[0]) / 0.06, 0.0, 1.0))
        base_use_bonus = 0.20 * lateral_factor * abs(rotation)

        # Directional alignment from base->ee toward base->cube.
        base_pos = self.data.xpos[self.model.body("Base").id].astype(np.float32)
        ee_vec = ee[:2] - base_pos[:2]
        cube_vec = cube[:2] - base_pos[:2]
        ee_norm = float(np.linalg.norm(ee_vec))
        cube_norm = float(np.linalg.norm(cube_vec))
        if ee_norm > 1e-6 and cube_norm > 1e-6:
            ee_heading = np.arctan2(float(ee_vec[0]), float(-ee_vec[1]))
            cube_heading = np.arctan2(float(cube_vec[0]), float(-cube_vec[1]))
            heading_err = abs(self._wrap_angle(cube_heading - ee_heading))
            heading_align = 1.0 - heading_err / np.pi
        else:
            heading_align = 0.0

        align_bonus = 0.80 * heading_align

        # Keep end-effector centered above cube in XY to avoid side drift.
        over_cube_bonus = 0.95 * float(np.clip(1.0 - dist_ee_cube_xy / 0.10, 0.0, 1.0))
        side_drift_penalty = 1.20 * abs(float(ee[0] - cube[0]))

        jaw_q = float(q[self.jaw_act_id])
        jaw_close_level = float(
            np.clip((jaw_q - 0.10) / max(self.safe_ctrl_high[self.jaw_act_id] - 0.10, 1e-6), 0.0, 1.0)
        )
        near_cube = float(np.clip(1.0 - dist_ee_cube / 0.10, 0.0, 1.0))
        gripper_prep_bonus = 0.9 * near_cube * jaw_close_level
        far_close_penalty = 0.10 * (1.0 - near_cube) * jaw_close_level

        if self.task == "reach":
            reward = (
                -dist_ee_cube
                - posture_penalty
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
                - 0.5 * posture_penalty
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
                -0.35 * posture_penalty
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
            # Horizontal pose: arm stretched forward, elbow extended, gripper open.
            # Pitch=-1.57 (~-π/2) makes the upper arm horizontal; Elbow=0.05 keeps it extended.
            reset_qpos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            reset_qpos = np.clip(reset_qpos, self.safe_ctrl_low, self.safe_ctrl_high)

        self.data.qpos[:6] = reset_qpos
        self.data.qvel[:] = 0.0
        self.ctrl_target = reset_qpos.copy()
        self.data.ctrl[:] = self.ctrl_target

        cube, goal = self._sample_cube_and_goal()
        self._goal_pos = goal
        self.model.site_pos[self.goal_site_id] = goal
        self._set_cube_pose(cube, zero_cube_vel=True)

        for _ in range(30):
            mujoco.mj_step(self.model, self.data)

        # Keep the sampled cube spawn exact after joint settling.
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
                self.ctrl_target[self.jaw_act_id] + action[-1] * 0.22,
                self.safe_ctrl_low[self.jaw_act_id],
                self.safe_ctrl_high[self.jaw_act_id],
            )
        )

        self.data.ctrl[:] = self.ctrl_target

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            self._apply_assist_grasp()

        self._episode_step += 1

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


def register_so100_envs() -> None:
    gym.register(
        id="SO100Reach-v0",
        entry_point="so100_rl.so100_pick_place_env:SO100PickPlaceEnv",
        kwargs={"task": "reach"},
    )
    gym.register(
        id="SO100Pick-v0",
        entry_point="so100_rl.so100_pick_place_env:SO100PickPlaceEnv",
        kwargs={"task": "pick"},
    )
    gym.register(
        id="SO100PickPlace-v0",
        entry_point="so100_rl.so100_pick_place_env:SO100PickPlaceEnv",
        kwargs={"task": "pick_place"},
    )


if __name__ == "__main__":
    env = SO100PickPlaceEnv(task="pick_place", max_episode_steps=100)
    obs, _ = env.reset(seed=0)
    total_reward = 0.0
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"Smoke test done, total_reward={total_reward:.3f}, info={info}")
    env.close()
