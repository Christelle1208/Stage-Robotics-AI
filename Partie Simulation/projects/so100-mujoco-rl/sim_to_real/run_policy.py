#!/usr/bin/env python3
"""Main control loop skeleton — run the trained policy on the real SO-100.

This wires together everything else in ``sim_to_real/``:

    ServoBus (you implement) --read--> RealObsBuilder --21-dim obs--> PPO policy
                  ^                                                        |
                  |                                                        v
                  +-------------------- write joint targets <-- action (6,)

Action mapping (must match ``BaseMuJoCoEnv._apply_action``,
``control_mode: joint_delta_position`` in configs/env/so100_grab.yaml)
-----------------------------------------------------------------------
    ctrl_target = clip(ctrl_target + action * action_scale, joint_limits)

where ``ctrl_target`` is the position setpoint sent to each of the 6
position-servo actuators (5 arm + Jaw), and ``action_scale = 0.05`` rad/step
(see configs/env/so100_grab.yaml -> control.action_scale). ``ctrl_target`` is
maintained IN THIS SCRIPT across steps — it starts at the robot's current
measured pose (so the first command doesn't snap the arm anywhere), then
accumulates the policy's deltas.

Servo backend
-------------
Uses ``LeRobotServoBus`` (``lerobot_servo_bus.py``), which wraps LeRobot's
``SO100Follower`` (handles the Feetech serial protocol) and converts its
units to the radians/convention in ``configs/robot/so100.yaml`` using
``calibration/joint_calibration.yaml`` — run ``calibrate_joint_offsets.py``
once before this script.

Usage
-----
    PYTHONPATH=src python3 sim_to_real/run_policy.py \
        --model outputs/models/ppo_SO100Grab-v0_final.zip \
        --port /dev/ttyUSB0 \
        --episode-steps 150

Safety
------
- Every command is clipped to the joint limits from
  ``configs/robot/so100.yaml`` (same ranges MuJoCo enforces via
  ``actuator_ctrlrange``).
- ``action_scale`` bounds the max position change per step to 0.05 rad
  (~2.9 deg) — same as in sim. Don't increase this for real hardware without
  separately verifying it's safe at your control loop's actual achieved rate.
- Start with the robot in a safe, collision-free pose and be ready to cut
  power — this skeleton does NOT include e-stop logic.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
from stable_baselines3 import PPO

from build_observation import RealObsBuilder
from lerobot_servo_bus import LeRobotServoBus


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained policy on the real SO-100.")
    parser.add_argument("--model", required=True, help="Path to .zip SB3 PPO model")
    parser.add_argument("--episode-steps", type=int, default=150,
                         help="Matches max_episode_steps in configs/env/so100_grab.yaml")
    parser.add_argument("--action-scale", type=float, default=0.05,
                         help="Matches control.action_scale in configs/env/so100_grab.yaml")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--cube-marker-id", type=int, default=0)
    parser.add_argument("--port", required=True, help="Serial port for the SO-100 follower arm.")
    parser.add_argument("--robot-id", default="so100_grab")
    parser.add_argument("--hz", type=float, default=50.0,
                         help="Control loop rate. Matches render_fps=50 (n_substeps*timestep).")
    args = parser.parse_args()

    model = PPO.load(args.model)
    servo = LeRobotServoBus(port=args.port, robot_id=args.robot_id)
    obs_builder = RealObsBuilder(servo, cube_marker_id=args.cube_marker_id)

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera_id}")

    # Joint limits, in the SAME order as arm_qpos (5) + [Jaw] (1).
    # Pulled from the FK model so they exactly match what MuJoCo enforces.
    ctrl_low = obs_builder.fk.model.actuator_ctrlrange[:, 0]
    ctrl_high = obs_builder.fk.model.actuator_ctrlrange[:, 1]

    print("Waiting for cube marker to be visible...")
    while not obs_builder.refresh_cube_pos(cap.read()[1]):
        time.sleep(0.05)
    print(f"cube_pos = {obs_builder._cube_pos}")

    obs_builder.reset()
    obs_builder.refresh_cube_pos(cap.read()[1])  # re-cache after reset() clears it

    # Start ctrl targets at the robot's CURRENT pose — avoids a snap move.
    ctrl = np.concatenate([servo.read_arm_qpos(), [servo.read_jaw_qpos()]])

    dt = 1.0 / args.hz
    for step in range(args.episode_steps):
        t0 = time.monotonic()

        obs = obs_builder.build()
        action, _ = model.predict(obs, deterministic=True)
        action = np.clip(action, -1.0, 1.0)

        ctrl = np.clip(ctrl + action * args.action_scale, ctrl_low, ctrl_high)
        servo.write_targets(ctrl[:5], float(ctrl[5]))

        # obs layout: arm_qpos(0:5) arm_qvel(5:10) grip_qpos(10) grip_qvel(11)
        #             ee_pos(12:15) cube_pos(15:18) ee->cube(18:21)
        dist_ee_to_cube = float(np.linalg.norm(obs[18:21]))
        print(f"step {step:3d}  dist_ee_to_cube={dist_ee_to_cube:.4f} m", end="\r")

        elapsed = time.monotonic() - t0
        time.sleep(max(0.0, dt - elapsed))

    cap.release()
    servo.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
