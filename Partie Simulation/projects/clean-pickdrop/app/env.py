"""SO-100 pick-and-drop environment, written from scratch for a clean PPO vs SAC comparison.

The robot must pick up a cube that spawns at a random position on the table
and drop it into a fixed bin. The whole scene (table, cube, bin, end-effector
site) is built programmatically on top of the MuJoCo Menagerie SO-100 model
via the MjSpec API -- no XML files to maintain, no extra assets.

Reward (used for both PPO and SAC, as specified by the user):

    r = r_pos + r_energy                                            (12)
    r_pos = r_x + r_y + r_z                                         (13)
    r_x = Kx * (x_ee - x_g)^2,  r_y = Ky * (y_ee - y_g)^2,
    r_z = Kz * (z_ee - z_g)^2                                       (14)
    g = (x_g, y_g, z_g) = pick position while reaching,
                          place position while placing              (16)
    Kx = Ky = -2, Kz = -0.5   while reaching
    Kx = Ky = -1, Kz = -1     while placing                         (17)
    r_energy = -3e-5 * sum_i |F_i|  over actuator forces            (18)

The task has two phases, exactly mirroring the equations above:
  - "reach":  goal = the cube's position (Ppick)
  - "place":  goal = a point above the bin (Pplace), entered once the
              end effector has closed around the cube and lifted it.

Two sparse *event* bonuses are layered on top of (12): a one-off bonus when
the grasp is detected (phase reach -> place) and one when the cube comes to
rest inside the bin (episode success). Eq. (12) alone gives the policy no
gradient about *when* to open/close the gripper -- the end-effector site sits
on the fixed jaw, so r_pos does not change with jaw angle -- so these two
terminal signals are the minimum needed to make the grasp/release controllable
by RL. They are reported separately in `info` so they are easy to ablate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MENAGERIE_SCENE = (
    PROJECT_ROOT / "assets" / "mujoco_menagerie" / "trs_so_arm100" / "scene.xml"
)

ARM_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
GRIPPER_JOINT = "Jaw"
ALL_JOINTS = ARM_JOINTS + [GRIPPER_JOINT]
HOME_QPOS = np.array([0.0, -3.3, 3.14, 1.2, 1.5, -0.17])
EE_LOCAL_OFFSET = np.array([0.0, -0.088, 0.0])  # ee_site offset within Fixed_Jaw

# --- scene geometry (all positions in world frame) --------------------------
# The Menagerie scene already has a floor plane at z=0; the table top sits
# slightly above it (TABLE_TOP_Z) so the two surfaces don't z-fight visually.
TABLE_TOP_Z = 0.01
TABLE_HALF = np.array([0.34, 0.30, 0.01])
TABLE_POS = np.array([0.04, -0.20, TABLE_TOP_Z - TABLE_HALF[2]])

CUBE_HALF = 0.018
CUBE_REST_Z = TABLE_TOP_Z + CUBE_HALF
CUBE_SPAWN_X = (-0.12, 0.08)
CUBE_SPAWN_Y = (-0.34, -0.18)

BIN_POS = np.array([0.17, -0.09, TABLE_TOP_Z])
BIN_WALL_HEIGHT = 0.05
BIN_WALL_THICK = 0.0025
BIN_OUTER_HALF = 0.05
BIN_INNER_HALF = BIN_OUTER_HALF - BIN_WALL_THICK
PLACE_TARGET = BIN_POS + np.array([0.0, 0.0, 0.09])  # Pplace: hover point over the bin

# --- reward gains, eq. (17) --------------------------------------------------
REACH_GAINS = np.array([-2.0, -2.0, -0.5])  # Kx, Ky, Kz while reaching
PLACE_GAINS = np.array([-1.0, -1.0, -1.0])  # Kx, Ky, Kz while placing
ENERGY_COEF = 3e-5  # eq. (18)

# --- task thresholds ---------------------------------------------------------
GRASP_DIST = 0.045      # ee<->cube distance below which a grasp is possible
GRASP_LIFT = 0.05       # cube must rise this much above its resting height
SUCCESS_SETTLE_VEL = 0.05
GRASP_BONUS = 5.0
SUCCESS_BONUS = 50.0


def _build_model() -> mujoco.MjModel:
    """Compose the scene programmatically: Menagerie SO-100 + table + cube + bin."""
    spec = mujoco.MjSpec.from_file(str(MENAGERIE_SCENE))

    fixed_jaw = _find_body(spec.worldbody, "Fixed_Jaw")
    site = fixed_jaw.add_site(name="ee_site", pos=EE_LOCAL_OFFSET.tolist(), size=[0.006, 0, 0])
    site.type = mujoco.mjtGeom.mjGEOM_SPHERE
    site.rgba = [0.9, 0.2, 0.2, 0.6]

    table_mat = spec.add_material(name="table_mat")
    table_mat.rgba = [0.62, 0.47, 0.32, 1.0]
    cube_mat = spec.add_material(name="cube_mat")
    cube_mat.rgba = [0.85, 0.2, 0.15, 1.0]
    bin_mat = spec.add_material(name="bin_mat")
    bin_mat.rgba = [0.2, 0.45, 0.85, 1.0]

    table = spec.worldbody.add_body(name="table", pos=TABLE_POS.tolist())
    table.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=TABLE_HALF.tolist(),
        material="table_mat",
    )

    cube_spawn = [0.0, -0.26, CUBE_REST_Z]
    cube = spec.worldbody.add_body(name="cube", pos=cube_spawn)
    cube.add_freejoint(name="cube_joint")
    cube.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[CUBE_HALF, CUBE_HALF, CUBE_HALF],
        mass=0.05,
        material="cube_mat",
        # Rigid, non-bouncy "plastic brick" contact: condim=6 gives sliding +
        # torsional + rolling friction (so it doesn't skate or spin unrealistically),
        # and a stiff/heavily-damped solref+solimp removes the springy/elastic
        # feel MuJoCo's defaults can have for light boxes on contact.
        friction=[1.0, 0.05, 0.05],
        condim=6,
        solref=[0.01, 1.0],
        solimp=[0.95, 0.99, 0.001, 0.5, 2.0],
    )

    _add_bin(spec, bin_mat.name)

    return spec.compile()


def _add_bin(spec: mujoco.MjSpec, material: str) -> None:
    """A static open-top container: one floor geom + four wall geoms."""
    h = BIN_WALL_HEIGHT / 2.0
    t = BIN_WALL_THICK
    o = BIN_OUTER_HALF
    bin_body = spec.worldbody.add_body(name="bin", pos=BIN_POS.tolist())

    bin_body.add_geom(
        name="bin_floor", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[o, o, t], pos=[0, 0, t], material=material,
    )
    wall_specs = [
        ("bin_wall_xp", [o - t, 0, h], [t, o, h]),
        ("bin_wall_xm", [-(o - t), 0, h], [t, o, h]),
        ("bin_wall_yp", [0, o - t, h], [o, t, h]),
        ("bin_wall_ym", [0, -(o - t), h], [o, t, h]),
    ]
    for name, pos, size in wall_specs:
        bin_body.add_geom(
            name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size, pos=pos, material=material,
        )


def _find_body(body, name: str):
    if getattr(body, "name", None) == name:
        return body
    for child in getattr(body, "bodies", []):
        found = _find_body(child, name)
        if found is not None:
            return found
    raise ValueError(f"Body '{name}' not found while assembling the scene")


class SO100PickDropEnv(gym.Env):
    """Gymnasium env: SO-100 picks a randomly-placed cube and drops it in a bin.

    Observation (flat float32 vector, 26-d):
        arm_qpos(5) arm_qvel(5) gripper_qpos(1) gripper_qvel(1)
        ee_pos(3) cube_pos(3) goal_pos(3) goal-ee(3) phase_onehot(2)

    Action: continuous Box(6,) in [-1, 1] -- joint position deltas
        (5 arm joints + 1 gripper), integrated into a position-servo target.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 300,
        action_scale: float = 0.05,
        frame_skip: int = 5,
    ) -> None:
        super().__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._max_episode_steps = max_episode_steps
        self._action_scale = action_scale
        self._frame_skip = frame_skip

        self.model = _build_model()
        self.data = mujoco.MjData(self.model)
        self._renderer: mujoco.Renderer | None = None

        self._arm_jids = [self._jid(n) for n in ARM_JOINTS]
        self._gripper_jid = self._jid(GRIPPER_JOINT)
        self._arm_qadr = self.model.jnt_qposadr[self._arm_jids]
        self._arm_dadr = self.model.jnt_dofadr[self._arm_jids]
        self._grip_qadr = self.model.jnt_qposadr[self._gripper_jid]
        self._grip_dadr = self.model.jnt_dofadr[self._gripper_jid]

        self._ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self._cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cube_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        self._cube_qadr = self.model.jnt_qposadr[cube_jid]
        self._cube_dadr = self.model.jnt_dofadr[cube_jid]

        self._ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(26,), dtype=np.float32)

        self._ctrl = HOME_QPOS.copy()
        self._phase = "reach"
        self._step_count = 0
        self._rng = np.random.default_rng()

    def _jid(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid == -1:
            raise ValueError(f"Joint '{name}' not present in the compiled model")
        return jid

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[self._arm_qadr] = HOME_QPOS[:5]
        self.data.qpos[self._grip_qadr] = HOME_QPOS[5]

        x = self._rng.uniform(*CUBE_SPAWN_X)
        y = self._rng.uniform(*CUBE_SPAWN_Y)
        yaw = self._rng.uniform(-np.pi, np.pi)
        quat = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]  # rotation about world z
        self.data.qpos[self._cube_qadr : self._cube_qadr + 7] = [x, y, CUBE_REST_Z, *quat]

        self._ctrl = HOME_QPOS.copy()
        self.data.ctrl[:] = self._ctrl
        mujoco.mj_forward(self.model, self.data)

        self._phase = "reach"
        self._step_count = 0

        return self._get_obs(), self._get_info(reward_terms={})

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._ctrl = np.clip(self._ctrl + action * self._action_scale, self._ctrl_low, self._ctrl_high)
        self.data.ctrl[:] = self._ctrl
        for _ in range(self._frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        reward, reward_terms = self._compute_reward()

        cube_pos = self._cube_pos()
        success = self._check_success(cube_pos)
        fell_off = cube_pos[2] < -0.05

        terminated = bool(success or fell_off)
        truncated = self._step_count >= self._max_episode_steps

        info = self._get_info(reward_terms)
        info["is_success"] = bool(success)
        if fell_off and not success:
            info["fell_off_table"] = True

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera="track_cam" if self._has_camera() else -1)
        return self._renderer.render()

    def _has_camera(self) -> bool:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "track_cam") != -1

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Task internals
    # ------------------------------------------------------------------

    def _ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._ee_site_id].copy()

    def _cube_pos(self) -> np.ndarray:
        return self.data.xpos[self._cube_body_id].copy()

    def _cube_vel(self) -> np.ndarray:
        return self.data.qvel[self._cube_dadr : self._cube_dadr + 3].copy()

    def _current_goal(self) -> np.ndarray:
        return self._cube_pos() if self._phase == "reach" else PLACE_TARGET

    def _check_success(self, cube_pos: np.ndarray) -> bool:
        dx, dy = cube_pos[:2] - BIN_POS[:2]
        in_footprint = abs(dx) < BIN_INNER_HALF and abs(dy) < BIN_INNER_HALF
        in_height = TABLE_TOP_Z < cube_pos[2] < TABLE_TOP_Z + BIN_WALL_HEIGHT
        settled = np.linalg.norm(self._cube_vel()) < SUCCESS_SETTLE_VEL
        return bool(in_footprint and in_height and settled)

    def _maybe_advance_phase(self, ee_pos: np.ndarray, cube_pos: np.ndarray) -> float:
        """Reach -> place transition once the cube has been grasped and lifted."""
        if self._phase != "reach":
            return 0.0
        grasped = (
            np.linalg.norm(ee_pos - cube_pos) < GRASP_DIST
            and cube_pos[2] > CUBE_REST_Z + GRASP_LIFT
        )
        if grasped:
            self._phase = "place"
            return GRASP_BONUS
        return 0.0

    def _compute_reward(self) -> tuple[float, dict[str, float]]:
        ee_pos = self._ee_pos()
        cube_pos = self._cube_pos()

        bonus = self._maybe_advance_phase(ee_pos, cube_pos)

        goal = self._current_goal()
        gains = REACH_GAINS if self._phase == "reach" else PLACE_GAINS
        diff = ee_pos - goal
        r_x, r_y, r_z = gains * diff**2          # eq. (14), gains are negative
        r_pos = r_x + r_y + r_z                   # eq. (13)

        forces = self.data.actuator_force
        r_energy = -ENERGY_COEF * np.sum(np.abs(forces))  # eq. (18)

        success_bonus = 0.0
        if self._check_success(cube_pos):
            success_bonus = SUCCESS_BONUS

        reward = float(r_pos + r_energy + bonus + success_bonus)
        terms = {
            "r_pos": float(r_pos),
            "r_energy": float(r_energy),
            "grasp_bonus": float(bonus),
            "success_bonus": float(success_bonus),
        }
        return reward, terms

    # ------------------------------------------------------------------
    # Observation / info
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        arm_qpos = self.data.qpos[self._arm_qadr]
        arm_qvel = self.data.qvel[self._arm_dadr]
        grip_qpos = self.data.qpos[self._grip_qadr : self._grip_qadr + 1]
        grip_qvel = self.data.qvel[self._grip_dadr : self._grip_dadr + 1]

        ee_pos = self._ee_pos()
        cube_pos = self._cube_pos()
        goal = self._current_goal()
        phase_onehot = np.array([1.0, 0.0] if self._phase == "reach" else [0.0, 1.0])

        return np.concatenate(
            [arm_qpos, arm_qvel, grip_qpos, grip_qvel,
             ee_pos, cube_pos, goal, goal - ee_pos, phase_onehot]
        ).astype(np.float32)

    def _get_info(self, reward_terms: dict[str, float]) -> dict[str, Any]:
        return {
            "phase": self._phase,
            "reward_terms": reward_terms,
            "cube_pos": self._cube_pos(),
            "ee_pos": self._ee_pos(),
        }


def make_env(render_mode: str | None = None, **kwargs) -> SO100PickDropEnv:
    return SO100PickDropEnv(render_mode=render_mode, **kwargs)