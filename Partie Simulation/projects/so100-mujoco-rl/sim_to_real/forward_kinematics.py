"""End-effector forward kinematics via a MuJoCo "digital twin".

Why this approach
------------------
Re-deriving the SO-100's forward kinematics by hand (DH parameters, link
transforms, etc.) is error-prone and easy to get subtly wrong relative to the
simulator the policy was trained in. Instead, we reuse the EXACT same MJCF
model (``assets/robots/so100/so_arm100.xml``) that the policy's environment
uses:

    1. Read the 5 arm joint angles + jaw angle from the real robot's servo
       encoders.
    2. Write them into a headless ``mujoco.MjData.qpos``.
    3. Call ``mujoco.mj_forward()`` (no simulation stepping, just kinematics).
    4. Read ``data.site_xpos[ee_site_id]`` — this is ``ee_pos``, computed with
       the identical kinematic chain the policy saw during training.

Frame convention — IMPORTANT
-----------------------------
``so_arm100.xml`` places its own "Base" body at
``pos="0.06 -0.265 0" quat="0 0 0 1"`` (a 180 deg rotation about Z) relative
to ITS OWN worldbody origin (see the comment in that file). This placement is
the SAME one used in ``assets/robots/so100/so100_feuille_scene.xml``, where
that worldbody origin is the center of the "Feuille" sheet — i.e. the same
world frame the policy's observations (``cube_pos``, ``ee_pos``) are
expressed in.

Net result: if you load ``so_arm100.xml`` standalone (as this module does)
and compute FK, ``ee_pos`` comes out ALREADY in the sheet-centered world
frame — no extra offset/rotation needed, AS LONG AS you physically mount the
real robot the same way: base positioned at (0.06, -0.265) relative to your
sheet's center, facing the sheet (+y direction), matching
``calibrate_homography.py``'s frame convention.

If your physical mounting differs from this, everything still works as long
as you (a) measure your homography calibration points in a frame whose
origin/axes match wherever you treat (0,0,0) as being for the robot's FK
output, and (b) optionally pass a ``world_offset`` / ``world_rotation_deg``
to ``SO100ForwardKinematics`` below to shift FK's output into that frame.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from so100_mujoco_rl.utils.config import load_config, project_root
from so100_mujoco_rl.utils.mujoco_utils import get_joint_ids, get_site_id

_ARM_XML = project_root() / "assets" / "robots" / "so100" / "so_arm100.xml"
_ROBOT_CFG = project_root() / "configs" / "robot" / "so100.yaml"


def _rot_z(deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class SO100ForwardKinematics:
    """Compute ``ee_pos`` and gripper ``close_fraction`` from joint angles.

    Parameters
    ----------
    xml_path:
        Path to the standalone arm MJCF. Defaults to
        ``assets/robots/so100/so_arm100.xml``.
    robot_config:
        Path to ``configs/robot/so100.yaml`` (joint names / ee site name).
    world_offset, world_rotation_deg:
        Optional extra transform applied to FK's raw output:
        ``world_pos = R_z(world_rotation_deg) @ ee_pos + world_offset``.
        Leave at the defaults (0, 0 deg) if your physical robot mount
        matches the sim placement described in the module docstring.
    """

    def __init__(
        self,
        xml_path: str | Path = _ARM_XML,
        robot_config: str | Path = _ROBOT_CFG,
        world_offset: np.ndarray | None = None,
        world_rotation_deg: float = 0.0,
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        cfg = load_config(robot_config)
        arm_joint_names = [j for j in cfg["joints"] if j not in cfg["actuators"]["gripper"]]
        gripper_joint_names = cfg["actuators"]["gripper"]

        self.arm_joint_ids = get_joint_ids(self.model, arm_joint_names)
        self.jaw_joint_id = get_joint_ids(self.model, gripper_joint_names)[0]
        self.ee_site_id = get_site_id(self.model, cfg["end_effector_site"])

        jaw_range = self.model.jnt_range[self.jaw_joint_id]
        self.jaw_closed = float(jaw_range[0])
        self.jaw_open = float(jaw_range[1])

        self.world_offset = np.zeros(3) if world_offset is None else np.asarray(world_offset, dtype=np.float64)
        self.world_rotation = _rot_z(world_rotation_deg)

    def compute(self, arm_qpos: np.ndarray, jaw_qpos: float) -> tuple[np.ndarray, float]:
        """Run FK for the given joint angles (radians).

        Parameters
        ----------
        arm_qpos:
            (5,) array, order = ``[Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll]``
            (same order as ``configs/robot/so100.yaml`` -> ``joints`` minus
            the gripper joint, and the same order used in the env's
            ``arm_qpos`` observation block).
        jaw_qpos:
            Jaw joint angle (radians).

        Returns
        -------
        ee_pos:
            (3,) end-effector position in the world frame (see module
            docstring for frame convention).
        close_fraction:
            Gripper closedness in [0, 1] — 1.0 = fully closed, 0.0 = fully
            open. Same convention as ``GrabTask.get_close_fraction``.
        """
        for jid, q in zip(self.arm_joint_ids, arm_qpos):
            self.data.qpos[self.model.jnt_qposadr[jid]] = q
        self.data.qpos[self.model.jnt_qposadr[self.jaw_joint_id]] = jaw_qpos

        mujoco.mj_forward(self.model, self.data)

        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        ee_pos = self.world_rotation @ ee_pos + self.world_offset

        span = self.jaw_open - self.jaw_closed
        close_fraction = float(np.clip((self.jaw_open - jaw_qpos) / span, 0.0, 1.0))

        return ee_pos, close_fraction


if __name__ == "__main__":
    # Quick sanity check: FK at the "home" pose should match the sim's
    # ee_pos at reset (compare against a fresh SO100Grab-v0 reset if you want
    # to double check against the simulator).
    fk = SO100ForwardKinematics()
    home_arm = np.array([0.0, -1.57, 1.57, 1.57, -1.57])
    home_jaw = 1.2  # matches _HOME_CTRL in so100_grab_env.py
    ee_pos, close_frac = fk.compute(home_arm, home_jaw)
    print(f"ee_pos at home pose = {ee_pos}")
    print(f"close_fraction      = {close_frac:.3f}")
