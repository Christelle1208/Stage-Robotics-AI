"""SO-100 pick-and-place task environment.

Uses the MuJoCo Menagerie scene_pick_place.xml as the base scene
(cube, goal_site, and robot are already defined there).  Adds the ee_site
to the Fixed_Jaw body via the ``_extra_sites`` hook.

Observation vector layout (flat):
    arm_qpos(5)  arm_qvel(5)  grip_qpos(1)  grip_qvel(1)
    ee_pos(3)  obj_pos(3)  target_pos(3)  ee→obj(3)  obj→target(3)
    = 28 dimensions

Action (continuous Box — required for SAC):
    arm deltas (5)  +  gripper delta (1)  = 6 dimensions
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from gymnasium import spaces

from so100_mujoco_rl.envs.base_mujoco_env import BaseMuJoCoEnv
from so100_mujoco_rl.robots.so100 import SO100Robot
from so100_mujoco_rl.tasks.pick_place import PickPlaceTask
from so100_mujoco_rl.utils.config import load_config, project_root
from so100_mujoco_rl.utils.mujoco_utils import SiteSpec, get_joints_qpos, get_joints_qvel

_DEFAULT_ENV_CFG = project_root() / "configs" / "env" / "so100_pick_place.yaml"

# Home pose from Menagerie keyframe (6 arm joints, gripper open).
_HOME_CTRL = np.array([0.0, -1.57, 1.57, 1.57, -1.57, 0.0], dtype=np.float64)


class SO100PickPlaceEnv(BaseMuJoCoEnv):
    """Gymnasium environment for SO-100 arm pick-and-place.

    Exposes a continuous ``Box`` action space for compatibility with
    both PPO and SAC from Stable-Baselines3.

    Parameters
    ----------
    env_config:
        Path to (or dict of) the environment YAML.
    robot_config:
        Path to the robot YAML.
    render_mode:
        Pass ``"human"`` for interactive viewer, ``None`` for headless.
    """

    def __init__(
        self,
        env_config: str | Path | dict | None = None,
        robot_config: str | Path | None = None,
        render_mode: str | None = None,
    ) -> None:
        if env_config is None:
            cfg = load_config(_DEFAULT_ENV_CFG)
        elif isinstance(env_config, dict):
            cfg = env_config
        else:
            cfg = load_config(env_config)

        self._env_cfg = cfg
        self._rand_cfg = cfg.get("randomization", {})
        self._task_cfg = cfg.get("task", {})
        self._use_gripper: bool = bool(cfg.get("control", {}).get("use_gripper", True))

        super().__init__(
            xml_path=cfg["scene_xml"],
            max_episode_steps=int(cfg.get("max_episode_steps", 200)),
            render_mode=render_mode,
            control_mode=cfg.get("control", {}).get("control_mode", "joint_delta_position"),
            action_scale=float(cfg.get("control", {}).get("action_scale", 0.05)),
        )

        self._robot = SO100Robot.from_model(self.model, config_path=robot_config)
        self._task = PickPlaceTask(
            model=self.model,
            task_cfg=self._task_cfg,
            reward_cfg=cfg.get("reward", {}),
        )
        self._init_spaces()

    # ------------------------------------------------------------------
    # BaseMuJoCoEnv hooks
    # ------------------------------------------------------------------

    def _extra_sites(self) -> list[SiteSpec]:
        """Inject the ee_site into the Fixed_Jaw body."""
        task_cfg = getattr(self, "_task_cfg", self._env_cfg.get("task", {}))
        ee_name = task_cfg.get("end_effector_site", "ee_site")
        return [
            SiteSpec(
                body_name="Fixed_Jaw",
                site_name=ee_name,
                pos=[0.0, -0.088, 0.0],
            )
        ]

    def _build_observation_space(self) -> spaces.Box:
        if not hasattr(self, "_robot"):
            return spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        n_arm = self._robot.n_arm_joints
        n_g = self._robot.n_gripper_joints if self._use_gripper else 0
        obs_dim = n_arm + n_arm + n_g + n_g + 3 + 3 + 3 + 3 + 3
        return spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

    def _build_action_space(self) -> spaces.Box:
        if not hasattr(self, "_robot"):
            return spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        n_act = self._robot.n_arm_actuators
        if self._use_gripper and self._robot.has_gripper:
            n_act += self._robot.n_gripper_actuators
        return spaces.Box(low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        arm_qpos = get_joints_qpos(self.model, self.data, self._robot.arm_joint_ids)
        arm_qvel = get_joints_qvel(self.model, self.data, self._robot.arm_joint_ids)

        if self._use_gripper and self._robot.has_gripper:
            grip_qpos = get_joints_qpos(self.model, self.data, self._robot.gripper_joint_ids)
            grip_qvel = get_joints_qvel(self.model, self.data, self._robot.gripper_joint_ids)
        else:
            grip_qpos = grip_qvel = np.zeros(0, dtype=np.float64)

        ee_pos = self._robot.get_ee_pos(self.data)
        obj_pos = self._task.get_object_pos(self.data)
        tgt_pos = self._task.get_target_pos(self.data)
        vec_ee_obj = obj_pos - ee_pos
        vec_obj_tgt = tgt_pos - obj_pos

        return np.concatenate(
            [arm_qpos, arm_qvel, grip_qpos, grip_qvel,
             ee_pos, obj_pos, tgt_pos, vec_ee_obj, vec_obj_tgt]
        ).astype(np.float32)

    def _compute_reward(self, action: np.ndarray) -> tuple[float, dict[str, Any]]:
        return self._task.compute_reward(self.data, action)

    def _is_terminated(self) -> bool:
        return self._task.is_success(self.data)

    def _reset_task(self, rng: np.random.Generator) -> None:
        self._task.randomize_scene(self.model, self.data, rng, self._rand_cfg)

    def _get_home_ctrl(self) -> np.ndarray:
        return _HOME_CTRL

    # ------------------------------------------------------------------
    # Override _apply_action to split arm vs gripper
    # ------------------------------------------------------------------

    def _apply_action(self, action: np.ndarray) -> None:
        n_arm = self._robot.n_arm_actuators
        arm_action = action[:n_arm]
        grip_action = (
            action[n_arm:] if (self._use_gripper and self._robot.has_gripper) else np.zeros(0)
        )
        super()._apply_action(np.concatenate([arm_action, grip_action]))
