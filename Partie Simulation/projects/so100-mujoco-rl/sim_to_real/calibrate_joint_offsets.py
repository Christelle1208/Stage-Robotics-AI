#!/usr/bin/env python3
"""One-time calibration: LeRobot units (degrees / 0-100%) -> MuJoCo radians.

Why
---
LeRobot's ``SO100Follower`` reports each arm joint in DEGREES relative to
ITS OWN calibrated zero (set during ``SO100Follower.calibrate()``, which you
should run/confirm BEFORE this script — it triggers automatically on
``connect()`` if no calibration file exists yet). The gripper is reported as
0-100% of LeRobot's own calibrated range of motion.

Neither of those zero-references/ranges matches the MuJoCo model's ``qpos``
convention (``configs/robot/so100.yaml`` -> ``joint_limits``). This script
finds, per joint, the affine map

    qpos_rad = scale * lerobot_value + offset

by moving each joint to its two mechanical end-stops and recording the
LeRobot reading at each, then asking you which MuJoCo limit
(``joint_limits[joint][0]`` = min or ``[1]`` = max) each end-stop
corresponds to.

How to answer the min/max question
-----------------------------------
Open the MuJoCo viewer on a fresh env reset in another terminal, e.g.:

    PYTHONPATH=src python3 -c "
    import gymnasium as gym, so100_mujoco_rl.envs
    env = gym.make('SO100Grab-v0', render_mode='human')
    env.reset()
    import time
    while True:
        env.unwrapped.data.qpos[JOINT_QPOS_ADR] = <try min then max>
        env.render(); time.sleep(0.01)
    "

...or more simply: drive ``scripts/evaluate_policy.py --render`` and watch
which way the arm moves as a sanity check after calibration (Step 3 below
verifies this automatically using ``forward_kinematics.py``).

If you don't have a reference handy, a reasonable default is to assume the
mechanical end-stop you reach FIRST (when rotating the joint in whatever you
consider its "natural" increasing direction) corresponds to
``joint_limits[joint][1]`` (max) — this matches the SO-100's typical
zero-at-rest convention. You can always re-run this script later if the
policy moves the wrong way (Step 3 will tell you).

Usage
-----
    python3 sim_to_real/calibrate_joint_offsets.py --port /dev/ttyUSB0

Controls
--------
For each joint: torque is disabled, move the joint BY HAND to one end-stop
and press ENTER, then to the other end-stop and press ENTER, then answer
min/max for the first position.

Output
------
``sim_to_real/calibration/joint_calibration.yaml``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from lerobot_servo_bus import ARM_JOINT_ORDER, CALIBRATION_DIR, GRIPPER_MOTOR, JOINT_MAP
from so100_mujoco_rl.utils.config import load_config, project_root

_ROBOT_CFG = project_root() / "configs" / "robot" / "so100.yaml"

try:
    from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
except ImportError as e:  # pragma: no cover
    raise SystemExit("This script requires lerobot (pip install lerobot).") from e


def read_motor(robot: SO100Follower, motor: str) -> float:
    obs = robot.get_observation()
    return obs[f"{motor}.pos"]


def calibrate_joint(robot: SO100Follower, joint: str, qpos_limits: tuple[float, float]) -> dict:
    motor = JOINT_MAP[joint]
    print(f"\n--- {joint} (motor '{motor}') ---")
    print(f"  MuJoCo qpos range: [{qpos_limits[0]:+.3f}, {qpos_limits[1]:+.3f}] rad")

    input(f"  Move '{joint}' BY HAND to one mechanical end-stop, then press ENTER...")
    val_a = read_motor(robot, motor)
    print(f"    reading A = {val_a:.2f}")

    input(f"  Now move '{joint}' to the OTHER end-stop, then press ENTER...")
    val_b = read_motor(robot, motor)
    print(f"    reading B = {val_b:.2f}")

    while True:
        answer = input(
            f"  Does position A correspond to the MIN ({qpos_limits[0]:+.3f}) or "
            f"MAX ({qpos_limits[1]:+.3f}) of the MuJoCo range? [min/max]: "
        ).strip().lower()
        if answer in ("min", "max"):
            break
        print("    please type 'min' or 'max'")

    qpos_a, qpos_b = (qpos_limits if answer == "min" else qpos_limits[::-1])

    scale = (qpos_b - qpos_a) / (val_b - val_a)
    offset = qpos_a - scale * val_a
    print(f"    -> scale={scale:.6f}  offset={offset:.6f}")
    return {"scale": float(scale), "offset": float(offset)}


def calibrate_gripper(robot: SO100Follower, jaw_limits: tuple[float, float]) -> dict:
    print(f"\n--- Jaw (motor '{GRIPPER_MOTOR}') ---")
    jaw_closed, jaw_open = jaw_limits  # joint_limits["Jaw"] = [closed, open]
    print(f"  MuJoCo Jaw range: closed={jaw_closed:+.3f}  open={jaw_open:+.3f} rad")

    input("  Move the gripper to FULLY OPEN, then press ENTER...")
    pct_open = read_motor(robot, GRIPPER_MOTOR)
    print(f"    reading (open) = {pct_open:.2f}%")

    input("  Move the gripper to FULLY CLOSED, then press ENTER...")
    pct_closed = read_motor(robot, GRIPPER_MOTOR)
    print(f"    reading (closed) = {pct_closed:.2f}%")

    scale = (jaw_closed - jaw_open) / (pct_closed - pct_open)
    offset = jaw_open - scale * pct_open
    print(f"    -> scale={scale:.6f}  offset={offset:.6f}")
    return {"scale": float(scale), "offset": float(offset)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate LeRobot units -> MuJoCo radians.")
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/ttyUSB0 or /dev/tty.usbmodemXXXX")
    parser.add_argument("--robot-id", default="so100_grab")
    args = parser.parse_args()

    robot_cfg = load_config(_ROBOT_CFG)
    joint_limits = robot_cfg["joint_limits"]

    config = SO100FollowerConfig(port=args.port, id=args.robot_id)
    robot = SO100Follower(config)
    robot.connect(calibrate=True)  # runs LeRobot's own homing/range calibration if needed

    robot.bus.disable_torque()
    print("\nTorque disabled — you can move the arm freely by hand.")

    joints_cal = {}
    for joint in ARM_JOINT_ORDER:
        lo, hi = joint_limits[joint]
        joints_cal[joint] = calibrate_joint(robot, joint, (lo, hi))

    gripper_cal = calibrate_gripper(robot, tuple(joint_limits["Jaw"]))

    robot.disconnect()

    out = {"joints": joints_cal, "gripper": gripper_cal}
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CALIBRATION_DIR / "joint_calibration.yaml"
    with open(out_path, "w") as fh:
        yaml.safe_dump(out, fh, sort_keys=False)

    print(f"\nSaved {out_path}")
    print("\nNext: run forward_kinematics.py-based checks via run_policy.py, or "
          "verify with a small motion test (Step 3 in the script docstring).")


if __name__ == "__main__":
    main()
