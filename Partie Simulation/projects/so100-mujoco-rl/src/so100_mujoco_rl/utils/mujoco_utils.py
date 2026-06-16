"""MuJoCo helper utilities.

Thin wrappers around the mujoco Python bindings providing:
* MjSpec-based model loading with programmatic site injection.
* Path validation with informative error messages.
* Named-element lookup helpers (joints, actuators, sites, bodies).
* Safe qpos/qvel accessors for scalar and slice joints.

Background: MuJoCo 3.x resolves meshdir relative to the TOP-LEVEL XML file,
not relative to included files.  Loading a scene from mujoco/scenes/ that
includes the Menagerie's so_arm100.xml therefore produces broken mesh paths.
The solution is to use MjSpec.from_file() with the ABSOLUTE path to the scene
XML (so the Menagerie directory is the base), then patch in extra sites via the
MjSpec API before compiling.  This is what build_model() does below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

from so100_mujoco_rl.utils.config import resolve_path


# ---------------------------------------------------------------------------
# Model building via MjSpec
# ---------------------------------------------------------------------------

@dataclass
class SiteSpec:
    """A site to inject into a body (or the worldbody) before model compilation.

    Set ``body_name = "worldbody"`` to add the site directly to the world body
    (useful for free-floating markers like reaching targets).
    """
    body_name: str          # Use "worldbody" for world-frame sites
    site_name: str
    pos: list[float]
    size: float = 0.007
    rgba: list[float] = field(default_factory=lambda: [0.2, 0.8, 0.2, 0.6])


def build_model(
    xml_path: str | Path,
    extra_sites: list[SiteSpec] | None = None,
    spec_patcher=None,
) -> mujoco.MjModel:
    """Load a MuJoCo model via MjSpec, inject optional sites, and compile.

    Using MjSpec (instead of MjModel.from_xml_path) lets us add sites to
    bodies defined in included files — circumventing the meshdir resolution
    problem described in this module's docstring.

    Parameters
    ----------
    xml_path:
        Path to the top-level scene XML.  Must be **absolute** so that
        MuJoCo resolves meshdir and includes relative to the correct base
        directory (the one that contains the mesh assets).
    extra_sites:
        Sites to inject before compilation.  Each SiteSpec names the parent
        body and provides position/size/colour.
    spec_patcher:
        Optional callable ``(spec: mujoco.MjSpec) -> None`` called after
        loading and before site injection.  Use it to add bodies, geoms,
        joints, materials, etc. programmatically (e.g. a table + bin for
        the real-world reproduction env).

    Raises
    ------
    FileNotFoundError
        If the XML file does not exist.
    ValueError
        If a named body is not found in the spec.
    """
    path = Path(xml_path)
    if not path.is_absolute():
        raise ValueError(
            f"build_model() requires an absolute path, got: {path}\n"
            "Use resolve_xml_path() to convert a config-relative path first."
        )
    if not path.exists():
        raise FileNotFoundError(
            f"MuJoCo scene XML not found: {path}\n\n"
            "Check that:\n"
            "  1. assets/mujoco_menagerie symlinks to a Menagerie checkout.\n"
            "  2. The scene_xml in your env config resolves to an existing file.\n\n"
            "To set up the symlink:\n"
            "  ln -s /path/to/mujoco_menagerie assets/mujoco_menagerie"
        )

    spec = mujoco.MjSpec.from_file(str(path))

    # Allow subclasses to add bodies, geoms, joints, etc. before site injection.
    if spec_patcher is not None:
        spec_patcher(spec)

    if extra_sites:
        for site_spec in extra_sites:
            if site_spec.body_name in ("worldbody", "world", ""):
                parent = spec.worldbody
            else:
                parent = _find_body(spec.worldbody, site_spec.body_name)
                if parent is None:
                    raise ValueError(
                        f"Body '{site_spec.body_name}' not found in model {path}.\n"
                        "Update the body name in the env's _extra_sites() method."
                    )
            site = parent.add_site()
            site.name = site_spec.site_name
            site.pos = list(site_spec.pos)
            site.size = [site_spec.size, 0.0, 0.0]
            site.type = mujoco.mjtGeom.mjGEOM_SPHERE
            site.rgba = list(site_spec.rgba)

    return spec.compile()


def _find_body(
    body: mujoco.MjSpec | object,
    name: str,
) -> object | None:
    """Recursively search for a named body in the MjSpec worldbody tree."""
    if getattr(body, "name", None) == name:
        return body
    for child in getattr(body, "bodies", []):
        result = _find_body(child, name)
        if result is not None:
            return result
    return None


# ---------------------------------------------------------------------------
# XML path resolution
# ---------------------------------------------------------------------------

def resolve_xml_path(xml_path: str | Path) -> Path:
    """Return the absolute path to a scene XML, relative to the project root.

    This does NOT load the XML — call build_model(resolve_xml_path(...)) to
    actually construct the MjModel.

    Raises
    ------
    FileNotFoundError
        If the XML does not exist.
    """
    path = resolve_path(xml_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Scene XML not found: {path}\n\n"
            "Check the scene_xml key in your env config.\n"
            "Run: python scripts/check_env.py --env <env_id>  to diagnose."
        )
    return path.resolve()


# ---------------------------------------------------------------------------
# Named-element lookup helpers
# ---------------------------------------------------------------------------

def get_joint_ids(model: mujoco.MjModel, joint_names: Sequence[str]) -> list[int]:
    """Return joint address indices for a list of named joints."""
    ids = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid == -1:
            _raise_missing("joint", name, model, mujoco.mjtObj.mjOBJ_JOINT)
        ids.append(jid)
    return ids


def get_actuator_ids(model: mujoco.MjModel, actuator_names: Sequence[str]) -> list[int]:
    """Return actuator indices for a list of named actuators."""
    ids = []
    for name in actuator_names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid == -1:
            _raise_missing("actuator", name, model, mujoco.mjtObj.mjOBJ_ACTUATOR)
        ids.append(aid)
    return ids


def get_site_id(model: mujoco.MjModel, site_name: str) -> int:
    """Return the site id for a named site."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if sid == -1:
        _raise_missing("site", site_name, model, mujoco.mjtObj.mjOBJ_SITE)
    return sid


def get_body_id(model: mujoco.MjModel, body_name: str) -> int:
    """Return the body id for a named body."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid == -1:
        _raise_missing("body", body_name, model, mujoco.mjtObj.mjOBJ_BODY)
    return bid


# ---------------------------------------------------------------------------
# qpos / qvel helpers
# ---------------------------------------------------------------------------

def safe_get_joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, joint_id: int) -> np.ndarray:
    """Return qpos slice for a joint (handles hinge, slide, free, ball)."""
    adr = model.jnt_qposadr[joint_id]
    n = _joint_qpos_len(model, joint_id)
    return data.qpos[adr : adr + n].copy()


def safe_get_joint_qvel(model: mujoco.MjModel, data: mujoco.MjData, joint_id: int) -> np.ndarray:
    """Return qvel slice for a joint."""
    adr = model.jnt_dofadr[joint_id]
    n = _joint_qvel_len(model, joint_id)
    return data.qvel[adr : adr + n].copy()


def get_joints_qpos(
    model: mujoco.MjModel, data: mujoco.MjData, joint_ids: Sequence[int]
) -> np.ndarray:
    """Concatenate qpos for a list of joint ids."""
    return np.concatenate([safe_get_joint_qpos(model, data, jid) for jid in joint_ids])


def get_joints_qvel(
    model: mujoco.MjModel, data: mujoco.MjData, joint_ids: Sequence[int]
) -> np.ndarray:
    """Concatenate qvel for a list of joint ids."""
    return np.concatenate([safe_get_joint_qvel(model, data, jid) for jid in joint_ids])


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _joint_qpos_len(model: mujoco.MjModel, joint_id: int) -> int:
    jtype = model.jnt_type[joint_id]
    # mjJNT_FREE=0 (7 qpos), mjJNT_BALL=1 (4), mjJNT_SLIDE=2 (1), mjJNT_HINGE=3 (1)
    return {0: 7, 1: 4, 2: 1, 3: 1}[int(jtype)]


def _joint_qvel_len(model: mujoco.MjModel, joint_id: int) -> int:
    jtype = model.jnt_type[joint_id]
    return {0: 6, 1: 3, 2: 1, 3: 1}[int(jtype)]


def _raise_missing(
    element_type: str,
    name: str,
    model: mujoco.MjModel,
    obj_type: int,
) -> None:
    count = {
        mujoco.mjtObj.mjOBJ_JOINT:    model.njnt,
        mujoco.mjtObj.mjOBJ_ACTUATOR: model.nu,
        mujoco.mjtObj.mjOBJ_SITE:     model.nsite,
        mujoco.mjtObj.mjOBJ_BODY:     model.nbody,
    }.get(obj_type, 0)
    all_names = [
        n for i in range(count)
        if (n := mujoco.mj_id2name(model, obj_type, i))
    ]
    raise ValueError(
        f"MuJoCo {element_type} '{name}' not found in model.\n"
        f"Available {element_type}s: {all_names}\n"
        "Check that the name matches the XML and update the config."
    )
