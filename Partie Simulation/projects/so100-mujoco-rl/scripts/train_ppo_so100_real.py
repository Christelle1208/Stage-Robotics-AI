#!/usr/bin/env python3
"""Train PPO on the real-life reproduction scene (table + cube + bin).

Usage
-----
    python scripts/train_ppo_so100_real.py
"""

from so100_mujoco_rl.train.train_sb3 import train

if __name__ == "__main__":
    train("configs/train/ppo_so100_real.yaml")
