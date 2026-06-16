"""SO-100 "grab cube" task environment.

Uses the Feuille scene (assets/robots/so100/so100_feuille_scene.xml) as the
base. A cube is added programmatically via ``_patch_spec`` (the scene XML has
no cube). The cube spawns at a random position on the Feuille checkerboard
each episode.

Observation vector layout (flat):
    arm_qpos(5)  arm_qvel(5)  grip_qpos(1)  grip_qvel(1)
    ee_pos(3)  cube_pos(3)  ee->cube(3)
    = 21 dimensions

Action (continuous Box):
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
from so100_mujoco_rl.tasks.grab import CUBE_HALF_SIZE, GrabTask
from so100_mujoco_rl.utils.config import load_config, project_root
from so100_mujoco_rl.utils.mujoco_utils import SiteSpec, get_joints_qpos, get_joints_qvel

_DEFAULT_ENV_CFG = project_root() / "configs" / "env" / "so100_grab.yaml"

# Start pose: arm matches the "start" keyframe in so100_feuille_scene.xml,
# while the gripper stays open for the reach-with-open-jaw task.
_HOME_CTRL = np.array([0.0, -3.32, 3.14, 1.35, -1.51, 1.2], dtype=np.float64)

# Default cube spawn height: resting on the Feuille's top surface (z=0.002).
_CUBE_DEFAULT_POS = [0.0, 0.0, 0.002 + CUBE_HALF_SIZE]


class SO100GrabEnv(BaseMuJoCoEnv):
    """Gymnasium environment for SO-100 reach-and-grab.

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
        self._control_cfg = cfg.get("control", {})

        super().__init__(
            xml_path=cfg["scene_xml"],
            max_episode_steps=int(cfg.get("max_episode_steps", 150)),
            render_mode=render_mode,
            control_mode=self._control_cfg.get("control_mode", "joint_delta_position"),
            action_scale=float(self._control_cfg.get("action_scale", 0.05)),
        )

        self._robot = SO100Robot.from_model(self.model, config_path=robot_config)
        self._task = GrabTask(
            model=self.model,
            task_cfg=self._task_cfg,
            reward_cfg=cfg.get("reward", {}),
        )
        self._init_spaces()

    # ------------------------------------------------------------------
    # BaseMuJoCoEnv hooks
    # ------------------------------------------------------------------

    def _patch_spec(self, spec: mujoco.MjSpec) -> None:
        """Add a free-floating cube to the Feuille scene."""
        object_name = self._task_cfg.get("object_name", "cube")

        cube_mat = spec.add_material(name="grab_cube_mat")
        cube_mat.rgba = [0.85, 0.2, 0.15, 1.0]

        cube = spec.worldbody.add_body(name=object_name, pos=_CUBE_DEFAULT_POS)
        cube.add_freejoint()
        cube.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[CUBE_HALF_SIZE, CUBE_HALF_SIZE, CUBE_HALF_SIZE],
            mass=0.03,
            material="grab_cube_mat",
            friction=[1.0, 0.05, 0.05],
            condim=4,
        )

    def _build_observation_space(self) -> spaces.Box:
        if not hasattr(self, "_robot"):
            return spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        n_arm = self._robot.n_arm_joints
        n_grip = self._robot.n_gripper_joints
        obs_dim = n_arm + n_arm + n_grip + n_grip + 3 + 3 + 3
        return spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

    def _build_action_space(self) -> spaces.Box:
        if not hasattr(self, "_robot"):
            return spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        n_act = self._robot.n_arm_actuators + self._robot.n_gripper_actuators
        return spaces.Box(low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        arm_qpos = get_joints_qpos(self.model, self.data, self._robot.arm_joint_ids)
        arm_qvel = get_joints_qvel(self.model, self.data, self._robot.arm_joint_ids)
        grip_qpos = get_joints_qpos(self.model, self.data, self._robot.gripper_joint_ids)
        grip_qvel = get_joints_qvel(self.model, self.data, self._robot.gripper_joint_ids)

        ee_pos = self._robot.get_ee_pos(self.data)
        cube_pos = self._task.get_object_pos(self.data)
        vec_ee_cube = cube_pos - ee_pos

        return np.concatenate(
            [arm_qpos, arm_qvel, grip_qpos, grip_qvel, ee_pos, cube_pos, vec_ee_cube]
        ).astype(np.float32)

    def _compute_reward(self, action: np.ndarray) -> tuple[float, dict[str, Any]]:
        return self._task.compute_reward(self.data, action)

    def _is_terminated(self) -> bool:
        return self._task.is_success(self.data)

    def _reset_task(self, rng: np.random.Generator) -> None:
        self._task.randomize_scene(self.model, self.data, rng, self._rand_cfg)

    def _get_home_ctrl(self) -> np.ndarray:
        return _HOME_CTRL

    def _apply_action(self, action: np.ndarray) -> None:
        """Apply action, optionally locking wrist roll for stable reaching."""
        action = np.array(action, dtype=np.float32, copy=True)

        if bool(self._control_cfg.get("lock_wrist_roll", False)):
            wrist_roll_aid = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                "Wrist_Roll",
            )
            if 0 <= wrist_roll_aid < action.shape[0]:
                action[wrist_roll_aid] = 0.0

        super()._apply_action(action)

        if bool(self._control_cfg.get("lock_wrist_roll", False)):
            wrist_roll_aid = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                "Wrist_Roll",
            )
            wrist_roll_home = float(self._control_cfg.get("wrist_roll_home", _HOME_CTRL[4]))
            if 0 <= wrist_roll_aid < self.model.nu:
                low = self.model.actuator_ctrlrange[wrist_roll_aid, 0]
                high = self.model.actuator_ctrlrange[wrist_roll_aid, 1]
                self.data.ctrl[wrist_roll_aid] = float(np.clip(wrist_roll_home, low, high))
