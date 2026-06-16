"""Assemble the 21-dim observation vector the trained policy expects.

Must exactly match ``SO100GrabEnv._get_obs()``
(``src/so100_mujoco_rl/envs/so100_grab_env.py``):

    arm_qpos(5)  arm_qvel(5)  grip_qpos(1)  grip_qvel(1)
    ee_pos(3)    cube_pos(3)  ee->cube(3)
    = 21 dimensions

Where each piece comes from on real hardware
----------------------------------------------
- ``arm_qpos`` / ``grip_qpos``: read directly from servo position feedback
  (you must implement ``ServoInterface`` for your actual servo bus, e.g.
  Feetech SCS/STS via their serial protocol).
- ``arm_qvel`` / ``grip_qvel``: NOT measured directly. Estimated here via
  finite difference between consecutive ``qpos`` reads, smoothed with an
  exponential moving average (EMA) — sim ``qvel`` is noise-free, raw
  encoder-diff velocity is not, and feeding raw noisy velocities to the
  policy can hurt performance.
- ``ee_pos``: forward kinematics from ``arm_qpos``/``grip_qpos``
  (see ``forward_kinematics.py``).
- ``cube_pos``: ArUco marker + homography (see ``cube_pose_estimator.py``).
- ``ee->cube``: ``cube_pos - ee_pos``, computed here.

All positions must be in the SAME world frame — see the frame-convention
notes in ``forward_kinematics.py`` and ``calibrate_homography.py``.
"""

from __future__ import annotations

import time
from typing import Protocol

import numpy as np

from cube_pose_estimator import ArucoCubePoseEstimator
from forward_kinematics import SO100ForwardKinematics


class ServoInterface(Protocol):
    """Implement this against your actual servo bus (e.g. Feetech SCS/STS).

    Both methods must return joint angles in RADIANS, in the same convention
    as ``configs/robot/so100.yaml`` (``joint_limits``) — i.e. the same units
    and sign convention MuJoCo uses. If your servo SDK returns raw ticks or
    degrees, convert before returning.
    """

    def read_arm_qpos(self) -> np.ndarray:
        """Return (5,) = [Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll] in radians."""
        ...

    def read_jaw_qpos(self) -> float:
        """Return the Jaw joint angle in radians."""
        ...


class EmaVelocityEstimator:
    """Finite-difference velocity with exponential smoothing.

    Parameters
    ----------
    alpha:
        Smoothing factor in (0, 1]. 1.0 = no smoothing (raw finite diff);
        smaller = more smoothing but more lag. 0.3-0.5 is a reasonable start.
    """

    def __init__(self, n_dims: int, alpha: float = 0.4) -> None:
        self.alpha = alpha
        self._prev_pos: np.ndarray | None = None
        self._prev_time: float | None = None
        self._vel = np.zeros(n_dims)

    def update(self, pos: np.ndarray, t: float | None = None) -> np.ndarray:
        t = time.monotonic() if t is None else t

        if self._prev_pos is not None and self._prev_time is not None:
            dt = max(t - self._prev_time, 1e-6)
            raw_vel = (pos - self._prev_pos) / dt
            self._vel = self.alpha * raw_vel + (1.0 - self.alpha) * self._vel

        self._prev_pos = pos.copy()
        self._prev_time = t
        return self._vel.copy()

    def reset(self) -> None:
        self._prev_pos = None
        self._prev_time = None
        self._vel[:] = 0.0


class RealObsBuilder:
    """Builds the 21-dim ``SO100Grab-v0`` observation vector from real sensors.

    Parameters
    ----------
    servo:
        Your ``ServoInterface`` implementation.
    cube_marker_id:
        ArUco marker ID stuck to the cube (default 0, see
        ``generate_markers.py``).
    velocity_alpha:
        EMA smoothing factor for the finite-difference velocity estimates
        (see ``EmaVelocityEstimator``).
    """

    def __init__(
        self,
        servo: ServoInterface,
        cube_marker_id: int = 0,
        velocity_alpha: float = 0.4,
    ) -> None:
        self.servo = servo
        self.fk = SO100ForwardKinematics()
        self.cube_estimator = ArucoCubePoseEstimator(marker_id=cube_marker_id)

        self._arm_vel = EmaVelocityEstimator(n_dims=5, alpha=velocity_alpha)
        self._jaw_vel = EmaVelocityEstimator(n_dims=1, alpha=velocity_alpha)

        # Cached cube_pos — the cube is static within an episode, and ArUco
        # detection may not keep up with the 50Hz control loop. Call
        # ``refresh_cube_pos()`` once at episode start (and whenever you want
        # to re-check), and ``build()`` will keep using the cached value.
        self._cube_pos: np.ndarray | None = None

    def refresh_cube_pos(self, frame: np.ndarray) -> bool:
        """Update the cached cube position from a camera frame.

        Returns ``True`` if the marker was found and the cache was updated,
        ``False`` otherwise (cache is left unchanged).
        """
        cube_pos = self.cube_estimator.estimate(frame)
        if cube_pos is None:
            return False
        self._cube_pos = cube_pos
        return True

    def reset(self) -> None:
        """Call at the start of each episode (clears velocity estimator state)."""
        self._arm_vel.reset()
        self._jaw_vel.reset()
        self._cube_pos = None

    def build(self) -> np.ndarray:
        """Return the (21,) observation vector for the current real state.

        Raises
        ------
        RuntimeError
            If ``refresh_cube_pos`` hasn't successfully run yet this episode.
        """
        if self._cube_pos is None:
            raise RuntimeError(
                "cube_pos not set — call refresh_cube_pos(frame) at least once "
                "(e.g. at episode start) before build()."
            )

        arm_qpos = self.servo.read_arm_qpos()
        jaw_qpos = self.servo.read_jaw_qpos()

        arm_qvel = self._arm_vel.update(arm_qpos)
        jaw_qvel = self._jaw_vel.update(np.array([jaw_qpos]))

        ee_pos, _close_fraction = self.fk.compute(arm_qpos, jaw_qpos)
        cube_pos = self._cube_pos
        vec_ee_cube = cube_pos - ee_pos

        return np.concatenate(
            [arm_qpos, arm_qvel, [jaw_qpos], jaw_qvel, ee_pos, cube_pos, vec_ee_cube]
        ).astype(np.float32)
