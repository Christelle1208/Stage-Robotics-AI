#!/usr/bin/env python3
"""Sanity-check an environment: reset + N steps, print obs/act/reward shapes.

Usage
-----
    python scripts/check_env.py --env SO100PickPlace-v0
    python scripts/check_env.py --env SO100Reach-v0 --steps 10
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

# Add the src directory to sys.path for direct script execution.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import so100_mujoco_rl.envs  # noqa: F401 — registers envs


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a registered Gymnasium env.")
    parser.add_argument(
        "--env",
        default="SO100PickPlace-v0",
        help="Gymnasium env id (default: SO100PickPlace-v0)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Number of random steps to run (default: 5)",
    )
    args = parser.parse_args()

    import gymnasium as gym

    print(f"\nChecking environment: {args.env}")
    print("-" * 50)

    try:
        env = gym.make(args.env, render_mode=None)
    except Exception:
        print(f"ERROR: Failed to create environment '{args.env}'")
        traceback.print_exc()
        return

    print(f"Observation space : {env.observation_space}")
    print(f"Action space      : {env.action_space}")

    try:
        obs, info = env.reset(seed=0)
        print(f"\nreset() OK")
        print(f"  obs shape : {np.asarray(obs).shape}")
        print(f"  obs dtype : {np.asarray(obs).dtype}")

        for i in range(args.steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(
                f"  step {i+1:02d}: reward={reward:+.4f}  "
                f"terminated={terminated}  truncated={truncated}"
            )
            if terminated or truncated:
                obs, info = env.reset()
                print("       → episode reset")

        print("\nAll checks passed.")
    except Exception:
        print("\nERROR during environment interaction:")
        traceback.print_exc()
    finally:
        env.close()

    # Also list all joints/actuators in the model for debugging.
    print("\n--- Model inspection (for verifying actuator/joint names) ---")
    try:
        import mujoco
        from so100_mujoco_rl.utils.config import project_root, load_config

        cfg_map = {
            "SO100PickPlace-v0": "configs/env/so100_pick_place.yaml",
            "SO100Reach-v0": "configs/env/so100_reach.yaml",
        }
        cfg_file = cfg_map.get(args.env)
        if cfg_file:
            cfg = load_config(cfg_file)
            from so100_mujoco_rl.utils.mujoco_utils import build_model, resolve_xml_path
            xml = resolve_xml_path(cfg["scene_xml"])
            m = build_model(xml)  # MjSpec loader — handles meshdir correctly

            print("\nJoints:")
            for i in range(m.njnt):
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
                print(f"  [{i}] {name}")

            print("\nActuators:")
            for i in range(m.nu):
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                print(f"  [{i}] {name}")

            print("\nSites:")
            for i in range(m.nsite):
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i)
                print(f"  [{i}] {name}")
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()
