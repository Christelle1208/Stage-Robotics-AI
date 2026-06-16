"""SO-100 Real-Life Pick-and-Place environment.

Reproduces a physical workbench setup:
  • Large table (~80×60 cm), arm clamped at the near edge
  • 5 cm cube as the pick object
  • 10×10×5 cm open bin as the drop target (fixed position on the table)

Because the scene XML cannot be loaded via MjSpec when it includes the
Menagerie's so_arm100.xml from a different directory (meshdir issue), this
env loads the Menagerie's own ``scene.xml`` (just robot + floor) and builds
the full scene programmatically in ``_patch_spec()``.

Observation and action spaces are inherited from SO100PickPlaceEnv (28-D obs,
6-D continuous action), making this env compatible with both PPO and SAC.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from so100_mujoco_rl.envs.so100_pick_place_env import SO100PickPlaceEnv
from so100_mujoco_rl.utils.config import load_config, project_root
from so100_mujoco_rl.utils.mujoco_utils import SiteSpec

_DEFAULT_ENV_CFG = project_root() / "configs" / "env" / "so100_real_pick_place.yaml"

# Real dimensions (all in metres)
_TABLE_HALF_X  = 0.40   # 80 cm wide  → half = 40 cm
_TABLE_HALF_Y  = 0.30   # 60 cm deep  → half = 30 cm
_TABLE_HALF_Z  = 0.010  # 20 mm thick slab → half = 10 mm
_TABLE_Y_CENTRE = -0.30  # table centre is 30 cm in front of the arm
_TABLE_HEIGHT  = 0.75   # distance from floor to table surface (real workbench height)

_CUBE_HALF = 0.025       # 5 cm edge   → half = 25 mm
_CUBE_MASS = 0.08        # 80 g

_BIN_X_CENTRE  =  0.15  # 15 cm to the right of the robot axis
_BIN_Y_CENTRE  = -0.25  # 25 cm forward
_BIN_OUTER     =  0.050 # bin half-extent x/y (10 cm total)
_BIN_WALL_T    =  0.005 # wall half-thickness (10 mm total)
_BIN_BOTTOM_T  =  0.003 # bottom half-thickness (6 mm)
_BIN_WALL_H    =  0.025 # wall half-height → 5 cm total bin height
_GOAL_Z        =  0.040 # goal site height inside bin (4 cm above table)


class SO100RealPickPlaceEnv(SO100PickPlaceEnv):
    """Gymnasium environment matching a physical workbench pick-and-place setup.

    Inherits all observation/action/reward logic from SO100PickPlaceEnv.
    Overrides model building to add the table and bin programmatically.
    """

    def __init__(
        self,
        env_config: str | Path | dict | None = None,
        robot_config: str | Path | None = None,
        render_mode: str | None = None,
    ) -> None:
        if env_config is None:
            env_config = _DEFAULT_ENV_CFG
        super().__init__(
            env_config=env_config,
            robot_config=robot_config,
            render_mode=render_mode,
        )

    # ------------------------------------------------------------------
    # MjSpec patching — adds table + cube + bin to the Menagerie base scene
    # ------------------------------------------------------------------

    def _patch_spec(self, spec: mujoco.MjSpec) -> None:
        """Add table, cube, and bin to the base Menagerie scene via MjSpec."""
        self._lower_floor(spec)   # move Menagerie's z=0 floor down to table-leg level
        self._add_materials(spec)
        self._add_table(spec)
        self._add_cube(spec)
        self._add_bin(spec)

    def _lower_floor(self, spec: mujoco.MjSpec) -> None:
        """Move the Menagerie's ground plane down 75 cm below the table surface.

        The robot base and table top both sit at z=0 (table surface level).
        Without this, the floor and table top are coplanar and the table
        appears to be sitting on — rather than above — the ground.
        """
        for geom in spec.worldbody.geoms:
            if geom.name == "floor":
                geom.pos = [0.0, 0.0, -_TABLE_HEIGHT]
                return

    def _add_materials(self, spec: mujoco.MjSpec) -> None:
        mat_wood = spec.add_material()
        mat_wood.name = "wood"
        mat_wood.rgba = [0.72, 0.56, 0.34, 1.0]
        mat_wood.reflectance = 0.05

        mat_metal = spec.add_material()
        mat_metal.name = "metal"
        mat_metal.rgba = [0.55, 0.55, 0.60, 1.0]
        mat_metal.reflectance = 0.25

        mat_cube = spec.add_material()
        mat_cube.name = "cube_mat"
        mat_cube.rgba = [0.88, 0.30, 0.12, 1.0]

        mat_bin = spec.add_material()
        mat_bin.name = "bin_mat"
        mat_bin.rgba = [0.25, 0.45, 0.80, 1.0]
        mat_bin.reflectance = 0.05

    def _add_table(self, spec: mujoco.MjSpec) -> None:
        """80×60 cm table, arm clamped at the near edge.

        Table top surface is at z = 0 (coincides with the arm base).
        The body origin is at the table centre, z = 0.
        """
        table = spec.worldbody.add_body()
        table.name = "table"
        table.pos = [0.0, _TABLE_Y_CENTRE, 0.0]

        # ---- Table top: physics surface ----
        top = table.add_geom()
        top.name = "table_top"
        top.type = mujoco.mjtGeom.mjGEOM_BOX
        top.size = [_TABLE_HALF_X, _TABLE_HALF_Y, _TABLE_HALF_Z]
        top.pos = [0.0, 0.0, -_TABLE_HALF_Z]   # top face at z = 0 in world
        top.material = "wood"
        top.contype = 1
        top.conaffinity = 1

        # ---- Legs (visual only, no collision) ----
        # Span from table-top underside (z = -2*_TABLE_HALF_Z = -0.020)
        # down to the floor (z = -_TABLE_HEIGHT = -0.750).
        # half-height = (_TABLE_HEIGHT - 2*_TABLE_HALF_Z) / 2
        leg_h = (_TABLE_HEIGHT - 2 * _TABLE_HALF_Z) / 2   # = 0.365
        leg_z = -(2 * _TABLE_HALF_Z + leg_h)               # centre z = -0.385
        for lx, ly in [(-0.37, -0.27), (0.37, -0.27), (-0.37, 0.27), (0.37, 0.27)]:
            leg = table.add_geom()
            leg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            leg.size = [0.015, leg_h, 0.0]   # MjSpec needs 3 values; pad with 0
            leg.pos = [lx, ly, leg_z]
            leg.material = "metal"
            leg.contype = 0
            leg.conaffinity = 0

    def _add_cube(self, spec: mujoco.MjSpec) -> None:
        """5 cm cube with a free joint — initial position from config."""
        cube_body = spec.worldbody.add_body()
        cube_body.name = "cube"
        cube_body.pos = [0.0, -0.25, _CUBE_HALF]  # resting on table

        jnt = cube_body.add_joint()
        jnt.name = "cube_joint"
        jnt.type = mujoco.mjtJoint.mjJNT_FREE

        geom = cube_body.add_geom()
        geom.name = "cube_geom"
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = [_CUBE_HALF, _CUBE_HALF, _CUBE_HALF]
        geom.mass = _CUBE_MASS
        geom.material = "cube_mat"
        geom.friction = [1.2, 0.03, 0.002]
        geom.contype = 1
        geom.conaffinity = 1
        geom.solimp = [0.9, 0.95, 0.001, 0.5, 2]
        geom.solref = [0.02, 1]

    def _add_bin(self, spec: mujoco.MjSpec) -> None:
        """10×10×5 cm open-top bin — 5 mm walls, 6 mm bottom. Fixed on table."""
        bin_body = spec.worldbody.add_body()
        bin_body.name = "bin"
        bin_body.pos = [_BIN_X_CENTRE, _BIN_Y_CENTRE, 0.0]

        def wall(name, size, pos):
            g = bin_body.add_geom()
            g.name = name
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = size
            g.pos = pos
            g.material = "bin_mat"
            g.contype = 1
            g.conaffinity = 1

        inner = _BIN_OUTER - _BIN_WALL_T   # = 0.045
        wall_cx = inner + _BIN_WALL_T      # = 0.050 — not needed; use direct offsets
        wall_top = _BIN_BOTTOM_T * 2 + _BIN_WALL_H   # bottom centre + wall half-h

        # Bottom slab
        wall("bin_bottom",
             [_BIN_OUTER, _BIN_OUTER, _BIN_BOTTOM_T],
             [0, 0, _BIN_BOTTOM_T])

        # Left (−x), Right (+x)
        wall("bin_wall_l",
             [_BIN_WALL_T, _BIN_OUTER, _BIN_WALL_H],
             [-(inner), 0, wall_top])
        wall("bin_wall_r",
             [_BIN_WALL_T, _BIN_OUTER, _BIN_WALL_H],
             [+(inner), 0, wall_top])

        # Near (−y, robot side), Far (+y)
        wall("bin_wall_near",
             [_BIN_OUTER, _BIN_WALL_T, _BIN_WALL_H],
             [0, -(inner), wall_top])
        wall("bin_wall_far",
             [_BIN_OUTER, _BIN_WALL_T, _BIN_WALL_H],
             [0, +(inner), wall_top])

    # ------------------------------------------------------------------
    # Extra sites: ee_site (Fixed_Jaw) + goal_site (worldbody, inside bin)
    # ------------------------------------------------------------------

    def _extra_sites(self) -> list[SiteSpec]:
        task_cfg = getattr(self, "_task_cfg", self._env_cfg.get("task", {}))
        ee_name     = task_cfg.get("end_effector_site", "ee_site")
        target_name = task_cfg.get("target_site", "goal_site")
        return [
            SiteSpec(
                body_name="Fixed_Jaw",
                site_name=ee_name,
                pos=[0.0, -0.088, 0.0],
                rgba=[1.0, 0.3, 0.0, 0.8],
            ),
            # Goal site is in the worldbody at the bin centre, inside the bin opening.
            # model.site_pos for worldbody sites = world coordinates directly.
            SiteSpec(
                body_name="worldbody",
                site_name=target_name,
                pos=[_BIN_X_CENTRE, _BIN_Y_CENTRE, _GOAL_Z],
                size=0.020,
                rgba=[0.15, 0.85, 0.35, 0.65],
            ),
        ]
