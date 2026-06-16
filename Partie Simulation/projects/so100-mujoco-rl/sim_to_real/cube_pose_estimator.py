#!/usr/bin/env python3
"""Live ``cube_pos`` estimation from an ArUco marker stuck to the cube.

Pipeline (per frame)
--------------------
1. Detect the ArUco marker with the configured ID (default 0 — stick this
   marker on the TOP FACE of the cube, centered).
2. Undistort its pixel center using the camera intrinsics from
   ``calibrate_camera.py``.
3. Apply the homography from ``calibrate_homography.py`` to get
   ``(x, y)`` on the table plane.
4. Append ``z_plane`` (the fixed height stored alongside the homography,
   e.g. ``cube_spawn_z``) to get the full ``(x, y, z)`` ``cube_pos``.

Requires
--------
``sim_to_real/calibration/camera_intrinsics.npz`` and
``sim_to_real/calibration/homography.npz`` (run the two calibration scripts
first).

Usage
-----
    python3 sim_to_real/cube_pose_estimator.py --camera-id 0 --marker-id 0

Prints the live ``cube_pos`` and shows a debug window with the detected
marker highlighted. Press 'q' to quit.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from aruco_utils import (
    detect_marker_centers,
    get_detector,
    load_camera_calibration,
    load_homography,
    pixel_to_world_xy,
)


class ArucoCubePoseEstimator:
    """Returns ``cube_pos`` (3,) from a single camera frame, or ``None``."""

    def __init__(self, marker_id: int = 0, dict_name: str | None = None) -> None:
        self.marker_id = marker_id
        self.camera_matrix, self.dist_coeffs = load_camera_calibration()
        self.homography, self.z_plane = load_homography()
        self.detector = get_detector(dict_name) if dict_name else get_detector()

    def estimate(self, frame: np.ndarray) -> np.ndarray | None:
        centers = detect_marker_centers(frame, self.detector, self.camera_matrix, self.dist_coeffs)
        if self.marker_id not in centers:
            return None

        xy = pixel_to_world_xy(centers[self.marker_id], self.homography)
        return np.array([xy[0], xy[1], self.z_plane], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live cube pose from an ArUco marker.")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--marker-id", type=int, default=0)
    args = parser.parse_args()

    estimator = ArucoCubePoseEstimator(marker_id=args.marker_id)

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera_id}")

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        cube_pos = estimator.estimate(frame)
        display = frame.copy()
        if cube_pos is not None:
            text = f"cube_pos = ({cube_pos[0]:+.3f}, {cube_pos[1]:+.3f}, {cube_pos[2]:+.3f})"
            cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            print(text, end="\r")
        else:
            cv2.putText(display, "marker not visible", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("cube_pose_estimator", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
