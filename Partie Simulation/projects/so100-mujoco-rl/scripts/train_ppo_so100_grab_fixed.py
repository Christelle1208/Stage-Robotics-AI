#!/usr/bin/env python3
"""Train a PPO policy on the SO-100 grab task with FIXED cube position.

This trains the model to reach a cube at a fixed location (center of the Feuille sheet)
for easier sim-to-real validation before generalizing with randomization.

Usage
-----
    python scripts/train_ppo_so100_grab_fixed.py
    python scripts/train_ppo_so100_grab_fixed.py --timesteps 300000
    python scripts/train_ppo_so100_grab_fixed.py \
        --resume-from outputs/logs/ppo_so100_grab_fixed/best_model/best_model.zip \
        --timesteps 300000
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from so100_mujoco_rl.train.train_sb3 import train


def main() -> None:
    """Train or resume PPO on the fixed-cube grab task."""
    parser = argparse.ArgumentParser(
        description="Train or continue PPO on SO100Grab-v0 with the fixed-cube config."
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Path relative to the project root of a saved SB3 .zip model to continue from.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Additional timesteps to train for. Overrides total_timesteps from YAML.",
    )
    args = parser.parse_args()

    overrides = {
        "resume_from": args.resume_from,
        "total_timesteps": args.timesteps,
    }
    train("configs/train/ppo_so100_grab_fixed.yaml", overrides=overrides)


if __name__ == "__main__":
    main()
