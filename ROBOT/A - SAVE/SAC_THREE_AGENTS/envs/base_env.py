"""
Base environment for the SO-100 robotic arm — SAC Three-Agent pick-and-place.

Reference paper:
  "Pick-and-Place Using Three SAC Agents in a Robosuite-inspired Environment"
  MDPI Biomimetics 8(2):240

Design philosophy (Robosuite-inspired):
  - Modular observation construction (named dict → flattened vector)
  - Potential-based reward shaping to preserve optimal policy
  - MuJoCo physics backend (same as Robosuite uses internally)
  - Clean state injection interface for chaining agents

The SO-100 is a 6-DoF serial manipulator:
  Joint 0: Rotation  (yaw)
  Joint 1: Pitch (shoulder)
  Joint 2: Elbow
  Joint 3: Wrist Pitch
  Joint 4: Wrist Roll
  Joint 5: Jaw (gripper — 0=closed, 1.75=fully open)
"""

import os
import yaml
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

# ---------------------------------------------------------------------------
# Load YAML config relative to this file, so it works from any cwd.
# ---------------------------------------------------------------------------
_CFG = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(_CFG, "r") as _f:
    CONFIG = yaml.safe_load(_f)


class SO100BaseEnv(gym.Env):
    """
    Abstract base class shared by all three SAC agents.

    Provides:
      - MuJoCo model & data loading from scene XML
      - Shared body/site ID resolution
      - Delta-action space (normalised [-1, 1]^6)
      - Common observation components
      - Rendering via passive viewer
      - State snapshot / restore for agent chaining
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str | None = None):
        super().__init__()

        # ── Load MuJoCo Scene ──────────────────────────────────────────────
        xml_path = os.path.join(
            os.path.dirname(__file__), "..", CONFIG["env"]["xml_path"]
        )
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        self.frame_skip  = CONFIG["env"]["frame_skip"]
        self.delta_scale = CONFIG["env"]["delta_scale"]

        # ── Body / Site IDs ────────────────────────────────────────────────
        # Resolve all named references once at construction time.
        self.grasp_site_id    = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site"
        )
        self.object_body_id   = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "object"
        )
        # The 'object' body has a free joint → qpos starts at joint_qposadr
        _obj_jnt_id = self.model.body_jntadr[self.object_body_id]
        self.object_qpos_addr = self.model.jnt_qposadr[_obj_jnt_id]
        self.object_qvel_addr = self.model.jnt_dofadr[_obj_jnt_id]

        self.object_geom_height = float(self.model.geom("object_geom").size[2])
        self.goal_body_id       = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "goal"
        )
        self.goal_geom_height   = float(self.model.geom("goal_geom").size[2])

        # ── Action Space: Normalised Delta Joint Angles ────────────────────
        # Each action a ∈ [-1, 1]^6 maps to:
        #   ctrl[i] ← clip(ctrl[i] + a[i] × delta_scale, ctrl_range)
        # This "delta" formulation is critical: a random policy moves only
        # ≤1° per step, preventing wild flailing during early training.
        self.action_space = spaces.Box(
            low  = -np.ones(self.model.nu, dtype=np.float32),
            high =  np.ones(self.model.nu, dtype=np.float32),
            dtype = np.float32,
        )

        # ── Rendering ──────────────────────────────────────────────────────
        self.render_mode = render_mode
        self._viewer     = None

        # ── Per-episode state (set in reset) ──────────────────────────────
        self.goal_pos   = None
        self.np_random  = np.random.default_rng()
        self._step_count = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Core MuJoCo helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _reset_mujoco(self, seed: int | None):
        """Reset simulator to the 'home' keyframe and randomise object XY."""
        self.np_random = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        # Apply home keyframe joint positions
        self.data.qpos[:] = self.model.keyframe("home").qpos

        # Sync actuator targets to home pose so position controllers do not
        # immediately fight gravity from zero position.
        self.data.ctrl[:] = self.data.qpos[:self.model.nu]

        # Randomise object XY on ground in front of robot (±5 cm around default pos)
        obj_base = self.model.body("object").pos.copy()
        obj_pos = np.array([
            obj_base[0] + self.np_random.uniform(-0.05, 0.05),
            obj_base[1] + self.np_random.uniform(-0.05, 0.05),
            self.object_geom_height,  # sitting on the ground
        ])
        self.data.qpos[self.object_qpos_addr:self.object_qpos_addr + 3] = obj_pos

        # Randomise goal XY on ground (farther from object, min 10 cm apart)
        for _ in range(50):
            gx = obj_base[0] + self.np_random.uniform(-0.10, 0.10)
            gy = obj_base[1] + self.np_random.uniform(-0.10, 0.10)
            if np.sqrt((gx - obj_pos[0]) ** 2 + (gy - obj_pos[1]) ** 2) > 0.10:
                break
        goal_body_pos = self.model.body("goal").pos
        goal_body_pos[0] = gx
        goal_body_pos[1] = gy
        goal_body_pos[2] = self.goal_geom_height  # on the ground

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0

        # Goal = centre of transparent goal box on ground
        self.goal_pos = goal_body_pos.copy()

    def _apply_action(self, action: np.ndarray):
        """Apply normalised delta action and advance physics."""
        new_ctrl = self.data.ctrl + action * self.delta_scale
        self.data.ctrl[:] = np.clip(
            new_ctrl,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )
        mujoco.mj_step(self.model, self.data, self.frame_skip)
        self._step_count += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Shared observation building blocks
    # ─────────────────────────────────────────────────────────────────────────

    def _obs_dict(self) -> dict:
        """
        Returns a named observation dictionary (Robosuite-style).
        All sub-environments call this and extend it as needed.

        Keys and shapes:
          q          (6,)  — joint angles [rad]
          ee_pos     (3,)  — EE / grasp-site Cartesian position [m]
          ee_vel     (3,)  — EE linear velocity [m/s]  (finite-diff of site_xpos)
          obj_pos    (3,)  — object Cartesian position [m]
          obj_vel    (3,)  — object linear velocity [m/s]
          jaw        (1,)  — jaw angle [rad]
          rel_ee_obj (3,)  — (EE pos - object pos) — points FROM object TO EE
        """
        q         = self.data.qpos[:6].astype(np.float32)
        ee_pos    = self.data.site_xpos[self.grasp_site_id].astype(np.float32)
        # EE velocity via site xvelp (translational velocity of site frame)
        ee_vel    = np.zeros(3, dtype=np.float32)
        mujoco.mj_comVel(self.model, self.data)
        # Use site velocity from mj_jacSite instead of finite difference
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.grasp_site_id)
        ee_vel = (jacp @ self.data.qvel).astype(np.float32)

        obj_pos   = self.data.qpos[
            self.object_qpos_addr:self.object_qpos_addr + 3
        ].astype(np.float32)
        obj_vel   = self.data.qvel[
            self.object_qvel_addr:self.object_qvel_addr + 3
        ].astype(np.float32)
        jaw       = np.array([self.data.qpos[5]], dtype=np.float32)
        rel_ee_obj = (ee_pos - obj_pos)  # vector from object to end-effector

        return {
            "q":          q,
            "ee_pos":     ee_pos,
            "ee_vel":     ee_vel,
            "obj_pos":    obj_pos,
            "obj_vel":    obj_vel,
            "jaw":        jaw,
            "rel_ee_obj": rel_ee_obj.astype(np.float32),
        }

    def _flat_obs(self, obs_dict: dict, keys: list[str]) -> np.ndarray:
        """Concatenate selected keys into a flat float32 vector."""
        return np.concatenate([obs_dict[k] for k in keys], dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # State injection interface (agent chaining)
    # ─────────────────────────────────────────────────────────────────────────

    def get_state_snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (qpos, qctrl) for passing to the next agent.

        Called by Task_manager after a sub-task succeeds, before calling
        reset_from_snapshot() on the downstream environment.
        """
        return self.data.qpos.copy(), self.data.ctrl.copy()

    def _restore_snapshot(self, qpos: np.ndarray, qctrl: np.ndarray):
        """
        Injects an arbitrary simulator state into this environment.

        This is how the paper chains agents: the Grasp agent starts exactly
        where the Reach agent left off; the Place agent starts where Grasp
        left off. No separate reset is needed.
        """
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos
        self.data.ctrl[:] = qctrl
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Rendering
    # ─────────────────────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ─────────────────────────────────────────────────────────────────────────
    # Gymnasium interface – subclasses must implement these
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError
