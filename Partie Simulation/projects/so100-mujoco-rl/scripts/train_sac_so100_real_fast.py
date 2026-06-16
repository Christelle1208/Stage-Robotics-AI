#!/usr/bin/env python3
"""Quick 20-minute SAC smoke test on the real pick-and-place scene.

Run this first. If ep_rew_mean rises in TensorBoard, the full run will work.
Then run: python scripts/train_sac_so100_real.py

Usage
-----
    python scripts/train_sac_so100_real_fast.py
"""

from so100_mujoco_rl.train.train_sb3 import train

if __name__ == "__main__":
    train("configs/train/sac_so100_real_fast.yaml")
