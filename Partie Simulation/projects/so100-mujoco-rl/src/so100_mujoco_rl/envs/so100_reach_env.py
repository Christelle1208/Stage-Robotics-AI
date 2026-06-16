"""SO-100 reaching task environment.

The environment extends BaseMuJoCoEnv with:
* ee_site injection into the Fixed_Jaw body (via _extra_sites hook).
* A target_site injected into a "target" floating body.
* Home pose initialisation at reset.
* Observation: [arm_qpos(5), arm_qvel(5), ee_pos(3), target_pos(3)] = 16-D.
* Action: normalized joint deltas for 5 arm actuators.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from gymnasium import spaces

from so100_mujoco_rl.envs.base_mujoco_env import BaseMuJoCoEnv
from so100_mujoco_rl.robots.so100 import SO100Robot
from so100_mujoco_rl.tasks.reach import ReachTask
from so100_mujoco_rl.utils.config import load_config, project_root
from so100_mujoco_rl.utils.mujoco_utils import SiteSpec, get_joints_qpos, get_joints_qvel

_DEFAULT_ENV_CFG = project_root() / "configs" / "env" / "so100_reach.yaml"

# Home pose from the Menagerie keyframe (confirmed from so_arm100.xml).
_HOME_CTRL = np.array([0.0, -1.57, 1.57, 1.57, -1.57, 0.0], dtype=np.float64)


class SO100ReachEnv(BaseMuJoCoEnv):
    """Gymnasium environment for SO-100 arm end-effector reaching.

    Parameters
    ----------
    env_config:
        Path to (or dict of) the environment YAML.  Defaults to
        ``configs/env/so100_reach.yaml``.
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
        self._robot_config_path = robot_config

        super().__init__(
            xml_path=cfg["scene_xml"],
            max_episode_steps=int(cfg.get("max_episode_steps", 200)),
            render_mode=render_mode,
            control_mode=cfg.get("control", {}).get("control_mode", "joint_delta_position"),
            action_scale=float(cfg.get("control", {}).get("action_scale", 0.05)),
        )

        # Robot descriptor and task (built after model is compiled in super().__init__).
        self._robot = SO100Robot.from_model(self.model, config_path=robot_config)
        self._task = ReachTask(
            model=self.model,
            task_cfg=self._task_cfg,
            reward_cfg=cfg.get("reward", {}),
        )
        self._init_spaces()

    # ------------------------------------------------------------------
    # BaseMuJoCoEnv hooks
    # ------------------------------------------------------------------

    def _extra_sites(self) -> list[SiteSpec]:
        """Inject ee_site into Fixed_Jaw and target_site into the worldbody."""
        task_cfg = getattr(self, "_task_cfg", self._env_cfg.get("task", {}))
        ee_name     = task_cfg.get("end_effector_site", "ee_site")
        target_name = task_cfg.get("target_site", "target_site")
        return [
            # End-effector: centred between the Fixed and Moving Jaw pads.
            SiteSpec(
                body_name="Fixed_Jaw",
                site_name=ee_name,
                pos=[0.0, -0.088, 0.0],
                rgba=[1.0, 0.3, 0.0, 0.8],
            ),
            # Reach target: free-floating sphere in world frame.
            # Position is updated each reset by ReachTask.randomize_target()
            # via model.site_pos — no body or joint needed.
            SiteSpec(
                body_name="worldbody",
                site_name=target_name,
                pos=[0.0, -0.20, 0.10],  # default; overridden at reset
                size=0.015,
                rgba=[0.2, 0.8, 0.2, 0.6],
            ),
        ]

    def _build_observation_space(self) -> spaces.Box:
        if not hasattr(self, "_robot"):
            return spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        n = self._robot.n_arm_joints
        return spaces.Box(-np.inf, np.inf, shape=(n + n + 3 + 3,), dtype=np.float32)

    def _build_action_space(self) -> spaces.Box:
        if not hasattr(self, "_robot"):
            return spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        return spaces.Box(-1.0, 1.0, shape=(self._robot.n_arm_actuators,), dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        qpos = get_joints_qpos(self.model, self.data, self._robot.arm_joint_ids)
        qvel = get_joints_qvel(self.model, self.data, self._robot.arm_joint_ids)
        ee_pos = self._robot.get_ee_pos(self.data)
        target_pos = self.data.site_xpos[self._task.target_site_id].copy()
        return np.concatenate([qpos, qvel, ee_pos, target_pos]).astype(np.float32)

    def _compute_reward(self, action: np.ndarray) -> tuple[float, dict[str, Any]]:
        return self._task.compute_reward(self.data, action)

    def _is_terminated(self) -> bool:
        return self._task.is_success(self.data)

    def _reset_task(self, rng: np.random.Generator) -> None:
        self._task.randomize_target(self.model, self.data, rng, self._rand_cfg)

    def _get_home_ctrl(self) -> np.ndarray:
        return _HOME_CTRL

    def _apply_action(self, action: np.ndarray) -> None:
        """Pad arm-only action with zeros for unused gripper actuator(s)."""
        n_arm = self._robot.n_arm_actuators
        n_extra = self.model.nu - n_arm  # gripper actuators not controlled here
        full_action = np.concatenate([action[:n_arm], np.zeros(n_extra)])
        super()._apply_action(full_action)
