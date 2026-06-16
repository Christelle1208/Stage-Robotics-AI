#!/usr/bin/env python3
"""Train a SAC policy on the SO-100 pick-and-place task.

Usage
-----
    python scripts/train_sac_so100_pick_place.py
"""

from so100_mujoco_rl.train.train_sb3 import train

if __name__ == "__main__":
    train("configs/train/sac_so100_pick_place.yaml")
