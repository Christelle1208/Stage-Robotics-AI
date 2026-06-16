#!/usr/bin/env python3
"""Step 1 — camera intrinsic calibration (camera matrix + distortion).

Why
---
The ArUco-based homography in ``calibrate_homography.py`` assumes a
distortion-free pinhole camera. Almost no real lens is distortion-free, so
we measure the camera's intrinsics once and undistort every pixel coordinate
before using it.

What you need
-------------
A printed checkerboard pattern (e.g. a standard 9x6-internal-corner board,
available as a free PDF from any OpenCV calibration tutorial), taped to a
flat, rigid surface.

Usage
-----
    python3 sim_to_real/calibrate_camera.py --camera-id 0 --cols 9 --rows 6 \
        --square-size-m 0.025

    # Rectangular cells (different spacing in x/y):
    python3 sim_to_real/calibrate_camera.py --camera-id 0 --cols 3 --rows 3 \
        --cell-size-x-m 0.06 --cell-size-y-m 0.04

Controls
--------
    SPACE  - capture the current frame as a calibration sample
             (only works if the checkerboard is fully visible)
    q      - finish capturing and run calibration

Capture >= 12 frames, moving/tilting the board to cover different positions,
distances and angles in the frame (corners of the image especially — that's
where distortion is largest).

Output
------
``sim_to_real/calibration/camera_intrinsics.npz`` containing
``camera_matrix`` (3x3) and ``dist_coeffs``.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from aruco_utils import save_camera_calibration


def _frame_rms_errors(
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
    rvecs: tuple[np.ndarray, ...],
    tvecs: tuple[np.ndarray, ...],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    """Return per-frame RMS reprojection errors (pixels)."""
    errors: list[float] = []
    for i, objp in enumerate(objpoints):
        proj, _ = cv2.projectPoints(objp, rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        diff = imgpoints[i].reshape(-1, 2) - proj.reshape(-1, 2)
        rmse = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
        errors.append(rmse)
    return np.asarray(errors, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Camera intrinsic calibration via checkerboard.")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--cols", type=int, default=9, help="Inner corners per row.")
    parser.add_argument("--rows", type=int, default=6, help="Inner corners per column.")
    parser.add_argument("--square-size-m", type=float, default=0.025,
                         help="Physical size of one checkerboard square, in metres.")
    parser.add_argument(
        "--cell-size-x-m",
        type=float,
        default=None,
        help="Physical spacing between adjacent inner corners along x (metres).",
    )
    parser.add_argument(
        "--cell-size-y-m",
        type=float,
        default=None,
        help="Physical spacing between adjacent inner corners along y (metres).",
    )
    parser.add_argument("--min-frames", type=int, default=12)
    parser.add_argument(
        "--outlier-trim-ratio",
        type=float,
        default=0.25,
        help=(
            "Fraction of worst frames to discard before final calibration "
            "(0 disables filtering)."
        ),
    )
    args = parser.parse_args()

    pattern_size = (args.cols, args.rows)

    if (args.cell_size_x_m is None) != (args.cell_size_y_m is None):
        raise ValueError("Provide both --cell-size-x-m and --cell-size-y-m, or neither.")

    cell_size_x = args.cell_size_x_m if args.cell_size_x_m is not None else args.square_size_m
    cell_size_y = args.cell_size_y_m if args.cell_size_y_m is not None else args.square_size_m

    # 3D coordinates of checkerboard corners in board frame (z=0).
    # Supports rectangular grids by allowing different x/y corner spacing.
    objp = np.zeros((args.cols * args.rows, 3), dtype=np.float32)
    grid = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp[:, 0] = grid[:, 0] * cell_size_x
    objp[:, 1] = grid[:, 1] * cell_size_y

    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera_id}")

    print("Move the checkerboard around the frame. Press SPACE to capture "
          "(only works when the full board is detected), 'q' to calibrate.")

    image_size: tuple[int, int] | None = None
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        image_size = (frame.shape[1], frame.shape[0])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern_size)

        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(display, pattern_size, corners, found)

        cv2.putText(display, f"captures: {len(objpoints)}/{args.min_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("calibrate_camera", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and found:
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            objpoints.append(objp.copy())
            imgpoints.append(corners)
            print(f"Captured frame {len(objpoints)}")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(objpoints) < args.min_frames:
        raise RuntimeError(
            f"Only {len(objpoints)} captures, need at least {args.min_frames}."
        )

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    # Optional outlier rejection: remove a fraction of frames with the worst
    # per-frame reprojection RMSE, then recalibrate.
    if args.outlier_trim_ratio > 0.0 and len(objpoints) > args.min_frames:
        trim_count = int(round(len(objpoints) * args.outlier_trim_ratio))
        keep_count = max(args.min_frames, len(objpoints) - trim_count)

        if keep_count < len(objpoints):
            frame_errors = _frame_rms_errors(
                objpoints,
                imgpoints,
                rvecs,
                tvecs,
                camera_matrix,
                dist_coeffs,
            )
            keep_idx = np.argsort(frame_errors)[:keep_count]
            keep_idx = np.sort(keep_idx)

            objpoints = [objpoints[i] for i in keep_idx]
            imgpoints = [imgpoints[i] for i in keep_idx]

            rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                objpoints,
                imgpoints,
                image_size,
                None,
                None,
            )

            print(
                f"\nOutlier filtering: kept {keep_count}/{len(frame_errors)} frames "
                f"(trimmed {len(frame_errors) - keep_count})."
            )

    print(f"\nRMS reprojection error: {rms:.4f} px (lower is better, <1 px is good)")
    print(f"camera_matrix=\n{camera_matrix}")
    print(f"dist_coeffs=\n{dist_coeffs.ravel()}")

    save_camera_calibration(camera_matrix, dist_coeffs)
    print("\nSaved to sim_to_real/calibration/camera_intrinsics.npz")


if __name__ == "__main__":
    main()
