"""Base MuJoCo Gymnasium environment.

Model loading strategy
----------------------
MuJoCo 3.x resolves ``meshdir`` in ``<compiler>`` relative to the TOP-LEVEL
XML file, not relative to included files.  Loading a scene from
``mujoco/scenes/`` that includes the Menagerie's ``so_arm100.xml`` therefore
produces broken mesh paths.

The fix: load models via ``build_model(absolute_xml_path)`` which internally
uses ``mujoco.MjSpec.from_file(absolute_path)``.  Because the path is
absolute, the Menagerie directory itself is the base, and ``meshdir="assets/"``
resolves to the STL directory correctly.

Subclasses that need extra elements (e.g. an end-effector site not in the
original XML) override ``_extra_sites()`` to return a list of ``SiteSpec``
objects; ``build_model()`` injects them before compilation.

Abstract interface for subclasses
----------------------------------
Required:
    _build_observation_space() -> Space
    _build_action_space()      -> Box
    _get_obs()                 -> ndarray
    _compute_reward()          -> (float, dict)
    _is_terminated()           -> bool
    _reset_task()              -> None

Optional:
    _extra_sites()             -> list[SiteSpec]   (default: [])
    _get_home_ctrl()           -> ndarray | None   (default: None → zeros)
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, SupportsFloat

import mujoco
import mujoco.viewer
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from so100_mujoco_rl.utils.mujoco_utils import SiteSpec, build_model, resolve_xml_path


class BaseMuJoCoEnv(gym.Env):
    """Abstract base for MuJoCo-backed Gymnasium environments.

    Parameters
    ----------
    xml_path:
        Path to the scene XML (absolute or relative to project root).
        **Must resolve to a file that is inside or alongside its mesh assets**
        so that MuJoCo's meshdir resolution works.  For Menagerie models,
        point directly to the Menagerie scene XML, not a wrapper scene.
    max_episode_steps:
        Episode truncation length.
    render_mode:
        ``"human"`` uses the interactive viewer; ``"rgb_array"`` returns frames.
    control_mode:
        ``"joint_delta_position"`` — action is a delta added to current ctrl.
        ``"joint_position"``       — action directly sets ctrl target.
    action_scale:
        Multiplier for delta-position actions (radians per step).
    n_substeps:
        Number of ``mj_step`` calls per ``env.step()``.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str | Path,
        max_episode_steps: int = 200,
        render_mode: str | None = None,
        control_mode: str = "joint_delta_position",
        action_scale: float = 0.05,
        n_substeps: int = 5,
    ) -> None:
        super().__init__()

        self._xml_path = resolve_xml_path(xml_path)  # absolute Path
        self._max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self._control_mode = control_mode
        self._action_scale = action_scale
        self._n_substeps = n_substeps

        # Build the model (MjSpec + optional body/geom patching + site injection + compile).
        extra = self._extra_sites()
        self.model = build_model(
            self._xml_path,
            extra_sites=extra or None,
            spec_patcher=self._patch_spec if self._has_patch_spec() else None,
        )
        self.data = mujoco.MjData(self.model)

        # Spaces (placeholder until _init_spaces() is called after robot init).
        self.observation_space: spaces.Space
        self.action_space: spaces.Box
        self._init_spaces()

        # Renderer (lazy-initialised in render()).
        self._viewer: mujoco.viewer.Handle | None = None
        self._renderer: mujoco.Renderer | None = None

        # Step counter and RNG.
        self._step_count: int = 0
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _build_observation_space(self) -> spaces.Space:
        """Return the observation space."""

    @abc.abstractmethod
    def _build_action_space(self) -> spaces.Box:
        """Return a continuous Box action space (required for SAC)."""

    @abc.abstractmethod
    def _get_obs(self) -> np.ndarray:
        """Return current observation."""

    @abc.abstractmethod
    def _compute_reward(self, action: np.ndarray) -> tuple[float, dict[str, Any]]:
        """Return (reward, info_dict)."""

    @abc.abstractmethod
    def _is_terminated(self) -> bool:
        """Return True if the episode should end (success)."""

    @abc.abstractmethod
    def _reset_task(self, rng: np.random.Generator) -> None:
        """Randomise objects/targets.  Called after mj_resetData + home pose."""

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def _extra_sites(self) -> list[SiteSpec]:
        """Return any SiteSpec objects to inject before model compilation.

        Override in subclasses that need sites not present in the scene XML
        (e.g. the ee_site added to the Menagerie SO-100 model).
        """
        return []

    def _patch_spec(self, spec) -> None:
        """Mutate a loaded MjSpec before compilation.

        Override to add bodies, geoms, joints, materials, or cameras that
        cannot be expressed as SiteSpec (e.g. a table, an open-top bin).
        Called once during __init__, after MjSpec.from_file() and before
        extra_sites are injected.
        """

    def _has_patch_spec(self) -> bool:
        return type(self)._patch_spec is not BaseMuJoCoEnv._patch_spec

    def _get_home_ctrl(self) -> np.ndarray | None:
        """Return the home control vector used to initialise each episode.

        Return ``None`` to use zeros.  For the SO-100 this should be the
        home keyframe ctrl = ``[0, -1.57, 1.57, 1.57, -1.57, 0]``.
        """
        return None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[Any, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        # Apply home pose so the arm starts in a natural configuration.
        home_ctrl = self._get_home_ctrl()
        if home_ctrl is not None:
            n = min(len(home_ctrl), self.model.nu)
            self.data.ctrl[:n] = home_ctrl[:n]
            # For position actuators, set qpos to match ctrl so the arm
            # starts exactly at the home position without transient motion.
            for i, ctrl_val in enumerate(home_ctrl[:n]):
                aid = i  # actuator index
                jid = self.model.actuator_trnid[aid, 0]  # joint driven by actuator
                adr = self.model.jnt_qposadr[jid]
                jtype = self.model.jnt_type[jid]
                if jtype in (2, 3):  # slide or hinge → 1 DOF
                    self.data.qpos[adr] = ctrl_val

        self._reset_task(self._rng)
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[Any, SupportsFloat, bool, bool, dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._apply_action(action)

        for _ in range(self._n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()
        reward, info = self._compute_reward(action)
        terminated = self._is_terminated()
        truncated = self._step_count >= self._max_episode_steps

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        return None

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------

    def _apply_action(self, action: np.ndarray) -> None:
        """Write action values into ``data.ctrl``."""
        if self._control_mode == "joint_delta_position":
            self.data.ctrl[:] = np.clip(
                self.data.ctrl + action * self._action_scale,
                self.model.actuator_ctrlrange[:, 0],
                self.model.actuator_ctrlrange[:, 1],
            )
        elif self._control_mode == "joint_position":
            self.data.ctrl[:] = np.clip(
                action,
                self.model.actuator_ctrlrange[:, 0],
                self.model.actuator_ctrlrange[:, 1],
            )
        else:
            raise ValueError(
                f"Unknown control_mode '{self._control_mode}'. "
                "Use 'joint_delta_position' or 'joint_position'."
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_spaces(self) -> None:
        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()

    def seed(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
