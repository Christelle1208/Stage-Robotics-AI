#!/usr/bin/env python3
"""Train SAC on the real-life reproduction scene (table + cube + bin).

SAC is recommended for this task — the denser reward shaping and
automatic entropy tuning handle the manipulation better than PPO.

Usage
-----
    python scripts/train_sac_so100_real.py
"""

from so100_mujoco_rl.train.train_sb3 import train

if __name__ == "__main__":
    train("configs/train/sac_so100_real.yaml")
