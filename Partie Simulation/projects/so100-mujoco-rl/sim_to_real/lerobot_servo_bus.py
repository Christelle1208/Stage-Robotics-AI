"""``ServoInterface``/``ServoBus`` implementation backed by LeRobot.

LeRobot (https://github.com/huggingface/lerobot, v0.4.4 — already installed
in this environment) ships ``SO100Follower``
(``lerobot.robots.so_follower.SO100Follower``), which wraps
``FeetechMotorsBus`` and handles the serial protocol, motor calibration
(homing offsets, range of motion), and read/write of the 6 SO-100 servos.
This module is a thin adapter that:

1. Connects to that bus.
2. Converts LeRobot's units (degrees for the 5 arm joints, 0-100 for the
   gripper, both relative to LeRobot's OWN calibration) into the radians /
   joint convention used by ``configs/robot/so100.yaml`` and the MuJoCo
   model (what ``forward_kinematics.py`` and the trained policy expect).
2'. Converts back the other way when writing position targets.

Why a separate calibration step is still needed
-------------------------------------------------
LeRobot's `DEGREES` mode reports each arm joint's angle relative to ITS OWN
calibrated zero (the midpoint of the range of motion recorded during
``SO100Follower.calibrate()``), not relative to the MuJoCo URDF/MJCF's zero
pose — different zero reference and possibly a flipped sign/direction.
Similarly the gripper's ``RANGE_0_100`` is relative to LeRobot's own
calibrated range of motion, not MuJoCo's ``Jaw`` joint range.

Both are handled with one affine map per joint (arm: degrees -> radians,
gripper: percent -> radians):

    qpos_rad = scale * lerobot_value + offset

``scale``/``offset`` are stored in ``calibration/joint_calibration.yaml`` and
produced by ``calibrate_joint_offsets.py`` — a ONE-TIME calibration you run
with the physical robot in hand (see that script's docstring).

Joint name mapping (LeRobot <-> MuJoCo / configs/robot/so100.yaml)
---------------------------------------------------------------------
    shoulder_pan  <-> Rotation
    shoulder_lift <-> Pitch
    elbow_flex    <-> Elbow
    wrist_flex    <-> Wrist_Pitch
    wrist_roll    <-> Wrist_Roll
    gripper       <-> Jaw

This is the standard SO-100 motor layout (motor IDs 1-6 in that order). If
your physical unit was wired/configured differently, fix ``JOINT_MAP`` below.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

CALIBRATION_DIR = Path(__file__).parent / "calibration"
JOINT_CALIBRATION_PATH = CALIBRATION_DIR / "joint_calibration.yaml"

# MuJoCo joint name (configs/robot/so100.yaml order) -> LeRobot motor name.
JOINT_MAP = {
    "Rotation": "shoulder_pan",
    "Pitch": "shoulder_lift",
    "Elbow": "elbow_flex",
    "Wrist_Pitch": "wrist_flex",
    "Wrist_Roll": "wrist_roll",
}
GRIPPER_JOINT = "Jaw"
GRIPPER_MOTOR = "gripper"

ARM_JOINT_ORDER = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]


def load_joint_calibration(path: Path | str = JOINT_CALIBRATION_PATH) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Run calibrate_joint_offsets.py first "
            "(one-time calibration with the physical robot)."
        )
    with open(path) as fh:
        return yaml.safe_load(fh)


class LeRobotServoBus:
    """Adapter exposing ``read_arm_qpos`` / ``read_jaw_qpos`` / ``write_targets``.

    Satisfies both ``ServoInterface`` (build_observation.py) and
    ``ServoBus`` (run_policy.py) protocols.

    Parameters
    ----------
    port:
        Serial port for the SO-100 follower arm, e.g. ``/dev/ttyUSB0``
        (Linux) or ``/dev/tty.usbmodemXXXX`` (macOS).
    robot_id:
        LeRobot robot id — used to locate/save its own motor calibration
        file (homing offsets etc.), separate from
        ``joint_calibration.yaml``.
    calibration_path:
        Path to this project's ``joint_calibration.yaml`` (produced by
        ``calibrate_joint_offsets.py``).
    """

    def __init__(
        self,
        port: str,
        robot_id: str = "so100_grab",
        calibration_path: Path | str = JOINT_CALIBRATION_PATH,
    ) -> None:
        config = SO100FollowerConfig(port=port, id=robot_id)
        self.robot = SO100Follower(config)
        self.robot.connect(calibrate=True)

        cal = load_joint_calibration(calibration_path)
        self._joint_cal = cal["joints"]
        self._gripper_cal = cal["gripper"]

    def close(self) -> None:
        self.robot.disconnect()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def _read_all(self) -> dict[str, float]:
        obs = self.robot.get_observation()
        return {k.removesuffix(".pos"): v for k, v in obs.items() if k.endswith(".pos")}

    def read_arm_qpos(self) -> np.ndarray:
        motor_vals = self._read_all()
        qpos = []
        for joint in ARM_JOINT_ORDER:
            motor = JOINT_MAP[joint]
            cal = self._joint_cal[joint]
            qpos.append(cal["scale"] * motor_vals[motor] + cal["offset"])
        return np.array(qpos, dtype=np.float64)

    def read_jaw_qpos(self) -> float:
        motor_vals = self._read_all()
        pct = motor_vals[GRIPPER_MOTOR]
        cal = self._gripper_cal
        return cal["scale"] * pct + cal["offset"]

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def write_targets(self, arm_targets: np.ndarray, jaw_target: float) -> None:
        action: dict[str, float] = {}
        for joint, qpos_rad in zip(ARM_JOINT_ORDER, arm_targets):
            motor = JOINT_MAP[joint]
            cal = self._joint_cal[joint]
            deg = (qpos_rad - cal["offset"]) / cal["scale"]
            action[f"{motor}.pos"] = float(deg)

        cal = self._gripper_cal
        pct = (jaw_target - cal["offset"]) / cal["scale"]
        action[f"{GRIPPER_MOTOR}.pos"] = float(np.clip(pct, 0.0, 100.0))

        self.robot.send_action(action)
