#!/usr/bin/env python3
"""Interactive MuJoCo scene viewer.

For named environments, creates the actual Gymnasium env and grabs its
compiled model — this guarantees the viewer shows exactly what the RL
policy trains on (same table, cube, bin, ee_site, goal_site, etc.).

For a raw scene XML (--scene flag), falls back to build_model().

Usage
-----
    # View the real pick-and-place scene (default):
    mjpython scripts/view_scene.py

    # View the Menagerie pick-and-place scene:
    mjpython scripts/view_scene.py --env SO100PickPlace-v0

    # View the reaching scene:
    mjpython scripts/view_scene.py --env SO100Reach-v0

    # View an arbitrary scene XML directly:
    mjpython scripts/view_scene.py \
        --scene assets/mujoco_menagerie/trs_so_arm100/scene_pick_place.xml

    # Longer session:
    mjpython scripts/view_scene.py --duration 120
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import so100_mujoco_rl.envs  # noqa: F401 — registers all environments
import gymnasium as gym
from so100_mujoco_rl.utils.mujoco_utils import SiteSpec, build_model, resolve_xml_path

_KNOWN_ENVS = [
    "SO100Grab-v0",
    "SO100RealPickPlace-v0",
    "SO100PickPlace-v0",
    "SO100Reach-v0",
]

# Extra sites to inject when loading a raw XML (--scene flag).
# For named envs these are added by the env class itself.
_SCENE_EXTRA_SITES: dict[str, list[SiteSpec]] = {
    "SO100PickPlace-v0": [
        SiteSpec("Fixed_Jaw", "ee_site", [0.0, -0.088, 0.0], rgba=[1, 0.3, 0, 0.8]),
    ],
    "SO100Reach-v0": [
        SiteSpec("Fixed_Jaw", "ee_site", [0.0, -0.088, 0.0], rgba=[1, 0.3, 0, 0.8]),
        SiteSpec("worldbody", "target_site", [0.0, -0.20, 0.10],
                 size=0.015, rgba=[0.2, 0.8, 0.2, 0.6]),
    ],
}

_SO100_HOME = [0.0, -1.57, 1.57, 1.57, -1.57, 0.0]


def _apply_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Apply home keyframe or fall back to the known SO-100 home pose."""
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        print(f"  keyframe  : 'home'  qpos[:6]={data.qpos[:6].round(3).tolist()}")
    else:
        n = min(len(_SO100_HOME), model.nu)
        data.ctrl[:n] = _SO100_HOME[:n]
        for i in range(n):
            jid = model.actuator_trnid[i, 0]
            adr = model.jnt_qposadr[jid]
            if model.jnt_type[jid] in (2, 3):
                data.qpos[adr] = _SO100_HOME[i]
        print(f"  keyframe  : none — SO-100 home pose applied manually")


def _print_summary(model: mujoco.MjModel, data: mujoco.MjData, label: str) -> None:
    joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,    i) for i in range(model.njnt)]
    sites  = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE,     i) for i in range(model.nsite)]
    bodies = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,     i) for i in range(model.nbody)]
    print(f"\nModel     : {label}")
    print(f"  nq={model.nq}  nu={model.nu}  nbody={model.nbody}  nsite={model.nsite}")
    print(f"  bodies    : {bodies}")
    print(f"  joints    : {joints}")
    print(f"  sites     : {sites}")
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    if ee_id >= 0:
        print(f"  ee_site   : {data.site_xpos[ee_id].round(4).tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="View a MuJoCo scene.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--env",
        default="SO100RealPickPlace-v0",
        choices=_KNOWN_ENVS,
        help="Named env to visualise (default: SO100RealPickPlace-v0).\n"
             "Uses gym.make() so the model is identical to what the policy trains on.",
    )
    group.add_argument(
        "--scene",
        default=None,
        help="Path to a raw scene XML (bypasses the env class).",
    )
    parser.add_argument(
        "--duration", type=float, default=60.0,
        help="Viewer duration in seconds (default: 60).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Build model + data
    # ------------------------------------------------------------------
    if args.scene:
        # Raw XML path — use build_model() directly.
        xml_path = resolve_xml_path(args.scene)
        print(f"\nLoading raw scene: {xml_path}")
        model = build_model(xml_path)
        data  = mujoco.MjData(model)
        label = xml_path.name
        _apply_home(model, data)

    else:
        # Named env — create the real env so _patch_spec() and _extra_sites()
        # run exactly as they do during training.
        env_id = args.env
        print(f"\nCreating env: {env_id}  (render_mode=None, seed=0)")
        env = gym.make(env_id, render_mode=None)
        obs, _ = env.reset(seed=0)

        inner = env.unwrapped               # type: ignore[attr-defined]
        model = inner.model
        data  = inner.data
        label = env_id

        # Home pose is already applied by env.reset() via _get_home_ctrl().

        env.close()                         # close wrappers; we keep model/data alive
        print(f"  env built — {model.nbody} bodies, {model.nsite} sites")

    mujoco.mj_forward(model, data)
    _print_summary(model, data, label)

    # ------------------------------------------------------------------
    # Launch viewer
    # ------------------------------------------------------------------
    print("\nLaunching viewer — press ESC or close the window to exit.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running() and (time.time() - start) < args.duration:
            t0 = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - t0)))

    print("Viewer closed.")


if __name__ == "__main__":
    main()
