# Sim-to-real pipeline for `SO100Grab-v0`

This folder contains everything needed to feed the trained PPO policy
(`outputs/models/ppo_SO100Grab-v0_final.zip`) with real-world observations
and turn its actions into real servo commands.

It uses **ArUco markers** for cube tracking and a **MuJoCo "digital twin"**
for end-effector forward kinematics — no depth camera, no YOLO, no ML
training required for perception.

## What the policy needs (recap)

From `SO100GrabEnv._get_obs()` (`src/so100_mujoco_rl/envs/so100_grab_env.py`),
the observation is a 21-dim vector:

```
arm_qpos(5)  arm_qvel(5)  grip_qpos(1)  grip_qvel(1)
ee_pos(3)    cube_pos(3)  ee->cube(3)
```

| Piece              | Source on real hardware                                   |
|--------------------|-------------------------------------------------------------|
| `arm_qpos`, `grip_qpos` | Servo encoder readback (you implement this)            |
| `arm_qvel`, `grip_qvel` | Finite difference of qpos, EMA-smoothed (`build_observation.py`) |
| `ee_pos`           | Forward kinematics via MuJoCo digital twin (`forward_kinematics.py`) |
| `cube_pos`         | ArUco marker on the cube + camera (`cube_pose_estimator.py`) |
| `ee->cube`         | `cube_pos - ee_pos` (computed) |

## Frame convention — read this first

Everything (`ee_pos`, `cube_pos`, the homography calibration points) must be
expressed in the **same world frame** the policy was trained in: the
"Feuille" sheet, centered at `(0, 0)`, with the robot's `Base` body mounted
at `pos=(0.06, -0.265, 0)`, rotated 180° about Z so the arm faces the sheet
(`+y` direction). See `assets/robots/so100/so_arm100.xml` lines 83-94 and
`assets/robots/so100/so100_feuille_scene.xml`.

**Practical implication**: if you physically replicate this layout —
robot base mounted 6 cm to the right and 26.5 cm "below" the center of your
physical sheet (in the robot's frame of reference: `+x` = right, `+y` =
toward the sheet) — then:

- `forward_kinematics.py` outputs `ee_pos` directly in this frame (verified —
  running it for the home pose gives `ee_pos = [0.06, -0.026, 0.089]`,
  consistent with the robot base sitting at `x=0.06`).
- `calibrate_homography.py` just needs you to measure its calibration points
  in this same sheet-centered frame (ruler from the sheet's center).
- No extra transform is needed anywhere — `ee_pos` and `cube_pos` are
  directly comparable, exactly like in sim.

If your physical setup doesn't match this layout, you can still make it
work: measure your homography points from whatever origin you like, and pass
a matching `world_offset` / `world_rotation_deg` to
`SO100ForwardKinematics(...)` in `forward_kinematics.py` so both frames line
up. The important thing is *consistency*, not matching the sim's absolute
numbers.

## Hardware checklist

- A camera with a clear, fixed view of the whole sheet (the area
  `x ∈ [-0.10, 0.10]`, `y ∈ [-0.065, 0.065]` from `configs/env/so100_grab.yaml`
  -> `randomization.cube_range`), mounted so it won't move between
  calibration and inference.
- Two printed ArUco markers (`DICT_4X4_50`):
  - **Marker 0** — stuck to the **top face of the cube**, centered.
  - **Marker 1** — a single loose marker used only during homography
    calibration (move it to each calibration point in turn).
- A checkerboard pattern for camera intrinsic calibration (any standard one,
  e.g. 9x6 internal corners).
- The SO-100 follower arm connected over USB/serial. Joint read/write goes
  through **LeRobot** (`lerobot` v0.4.4, already installed) via
  `lerobot_servo_bus.py` — see "Servo backend (LeRobot)" below.

## Servo backend (LeRobot)

`lerobot_servo_bus.py` wraps LeRobot's `SO100Follower`
(`lerobot.robots.so_follower`), which talks to the Feetech servos over
serial. It converts LeRobot's units (degrees for the 5 arm joints, 0-100%
for the gripper — both relative to LeRobot's OWN calibration) into the
radians/convention MuJoCo and the policy use
(`configs/robot/so100.yaml` -> `joint_limits`).

This conversion needs a **one-time calibration**, because LeRobot's
calibrated "zero" for each joint doesn't match MuJoCo's zero pose:

```bash
python3 sim_to_real/calibrate_joint_offsets.py --port /dev/ttyUSB0
```

This will (if not already done) run LeRobot's own motor calibration
(homing offsets / range of motion — prompts you to move the arm through its
full range), then walk you through moving each joint to its two mechanical
end-stops and asking which MuJoCo `joint_limits` bound (min/max) each
corresponds to. See the script's docstring for how to figure out the
min/max answer (compare against the MuJoCo viewer). Saves
`calibration/joint_calibration.yaml`.

Motor name mapping (`lerobot_servo_bus.JOINT_MAP`) assumes the standard
SO-100 layout (motor IDs 1-6 = shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex, wrist_roll, gripper = Rotation, Pitch, Elbow, Wrist_Pitch,
Wrist_Roll, Jaw). Fix this dict if your unit is wired differently.

## Setup order

```bash
cd "Partie Simulation"
pip install -r sim_to_real/requirements.txt

# 0. Generate ArUco markers to print.
python3 sim_to_real/generate_markers.py --ids 0 1

# 1. Camera intrinsics (one-time per camera).
python3 sim_to_real/calibrate_camera.py --camera-id 0 --cols 9 --rows 6 --square-size-m 0.025

# 2. Pixel -> table-plane homography (one-time per camera mount).
#    z-plane = cube resting height = cube_spawn_z from configs/env/so100_grab.yaml (0.017).
#    Pick >= 4 points spanning the cube_range area, e.g. its corners + center:
python3 sim_to_real/calibrate_homography.py --camera-id 0 --z-plane 0.017 \
    --points "-0.10,-0.065" "0.10,-0.065" "0.10,0.065" "-0.10,0.065" "0,0"

# 3. Sanity-check FK (no camera needed).
python3 sim_to_real/forward_kinematics.py

# 4. Sanity-check live cube tracking (marker 0 on the cube).
python3 sim_to_real/cube_pose_estimator.py --camera-id 0 --marker-id 0

# 5. One-time LeRobot <-> MuJoCo joint calibration.
python3 sim_to_real/calibrate_joint_offsets.py --port /dev/ttyUSB0

# 6. Run the policy.
PYTHONPATH=src python3 sim_to_real/run_policy.py \
    --model outputs/models/ppo_SO100Grab-v0_final.zip \
    --port /dev/ttyUSB0
```

Steps 1 and 2 are one-time (per camera/mount); re-run step 2 if the camera
moves. Step 0 is one-time ever (just print the markers once).

## File reference

| File | Purpose |
|------|---------|
| `aruco_utils.py` | Shared ArUco detection + calibration file I/O. Imported by everything else. |
| `generate_markers.py` | Generates printable ArUco marker PNGs. |
| `calibrate_camera.py` | Step 1 — camera intrinsics (matrix + distortion) via checkerboard. |
| `calibrate_homography.py` | Step 2 — pixel -> table-plane `(x, y)` homography via one ArUco marker moved to known points. |
| `forward_kinematics.py` | `SO100ForwardKinematics` — `ee_pos` + gripper `close_fraction` from joint angles, via a headless MuJoCo model. |
| `cube_pose_estimator.py` | `ArucoCubePoseEstimator` — live `cube_pos` from the cube's ArUco marker + homography. |
| `build_observation.py` | `RealObsBuilder` — assembles the full 21-dim observation (servo readback + FK + cube pose + velocity estimation). |
| `lerobot_servo_bus.py` | `LeRobotServoBus` — wraps LeRobot's `SO100Follower`, converts units via `joint_calibration.yaml`. |
| `calibrate_joint_offsets.py` | One-time LeRobot units <-> MuJoCo radians calibration (see below). |
| `run_policy.py` | Main control loop — loads the PPO model, runs `RealObsBuilder` + `model.predict` + `LeRobotServoBus` writes at 50 Hz. |
| `calibration/` | Generated `.npz`/`.yaml` calibration files + marker PNGs (gitignored-worthy, machine-specific). |

## Sanity check after calibration

Command the real arm to the `home_qpos` from `configs/robot/so100.yaml`
(`[0.0, -1.57, 1.57, 1.57, -1.57, 0.0]`, jaw fairly open) and confirm
`forward_kinematics.py`'s output for that pose
(`ee_pos = [0.06, -0.026, 0.089]`, `close_fraction = 0.286`) matches what
you'd expect physically — gripper at that height/reach, fairly open. If a
joint moves the wrong direction or the offset is clearly off, re-run
`calibrate_joint_offsets.py` for that joint (you likely answered min/max
backwards).

## Known sim/real gaps to expect

- **Dynamics**: real friction, backlash, and actuator response won't match
  MuJoCo exactly. The success criterion (reach within 1 cm + gripper open +
  hold 5 steps — see `src/so100_mujoco_rl/tasks/grab.py`) is a *positioning*
  task, which tends to transfer better than contact-rich grasping, but expect
  some degradation.
- **Latency**: the control loop here targets 50 Hz to match
  `n_substeps=5 * timestep=0.004s`. If your servo read/write round-trip can't
  keep up, the policy will see "older" state than it expects — measure your
  loop's actual achieved Hz and consider it as an extra source of
  observation noise/latency the policy wasn't trained on.
- **`arm_qvel`/`grip_qvel` noise**: sim velocities are exact; real
  finite-difference velocities are noisy. `EmaVelocityEstimator` in
  `build_observation.py` smooths this — tune `velocity_alpha` if the policy
  seems jittery (lower = smoother but more lag).
- If transfer quality is poor, the standard fix is **domain randomization**
  during training (randomize friction, add action delay, add observation
  noise to `qpos`/`qvel`/`cube_pos` in sim) rather than trying to make the
  real setup match sim perfectly.
