#!/usr/bin/env python3
"""Step 2 — pixel -> table-plane world (x, y) homography.

Why
---
The cube always rests on the same flat surface (the "Feuille"/sheet), so a
single planar homography is enough to turn "where is the marker in the
image" into "where is the marker on the table, in metres" — no depth camera
needed.

World frame convention (must match the rest of this pipeline and the policy)
------------------------------------------------------------------------
Use the CENTER of your physical sheet/tabletop as the origin (0, 0), with
+x / +y oriented the same way as in the MuJoCo scene
(``assets/robots/so100/so100_feuille_scene.xml``): the robot base sits at
(0.06, -0.265) in this frame. If you physically mount the robot at that same
offset from your sheet's center (and facing the sheet), then this homography
frame, the FK frame (``forward_kinematics.py``), and the frame the policy was
trained in are ALL THE SAME — no extra transform needed anywhere.

If your physical setup doesn't match that exactly, that's fine: just measure
your calibration points consistently from whatever origin/axes you pick, and
also build your FK transform relative to the same origin (see
``forward_kinematics.py`` docstring for the one-line offset to add).

What you need
-------------
- One printed ArUco marker (any ID; default marker 1 — keep marker 0 for the
  cube itself, see ``generate_markers.py``).
- >= 4 points marked on the table at KNOWN (x, y) coordinates in the world
  frame above (e.g. masking tape + ruler). More points = better fit; use
  points spread across the whole area the cube can spawn in
  (``cube_range`` in configs/env/so100_grab.yaml: x in [-0.10, 0.10],
  y in [-0.065, 0.065]).

Usage
-----
    python3 sim_to_real/calibrate_homography.py \
        --camera-id 0 \
        --z-plane 0.017 \
        --points "-0.10,-0.065" "0.10,-0.065" "0.10,0.065" "-0.10,0.065" "0,0"

Controls
--------
For each point listed in --points (in order): place the marker centered on
that point on the table, then press SPACE once it's detected (it averages
30 frames for stability). Press 'q' to abort.

Output
------
``sim_to_real/calibration/homography.npz`` containing ``homography`` (3x3)
and ``z_plane`` (the fixed height, in metres, at which the cube's marker
sits — e.g. cube resting height ``cube_spawn_z`` from
configs/env/so100_grab.yaml).
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from aruco_utils import (
    detect_marker_centers,
    get_detector,
    load_camera_calibration,
    save_homography,
)


def parse_point(s: str) -> tuple[float, float]:
    x_str, y_str = s.split(",")
    return float(x_str), float(y_str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pixel -> table-plane world homography calibration.")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--marker-id", type=int, default=1,
                         help="ArUco marker ID used as the calibration probe.")
    parser.add_argument("--points", nargs="+", required=True,
                         help='World (x,y) points in metres, e.g. "-0.10,-0.065" "0.10,0.065" ...')
    parser.add_argument("--z-plane", type=float, required=True,
                         help="Height (m) of the marker's plane, e.g. cube_spawn_z=0.017.")
    parser.add_argument("--samples-per-point", type=int, default=30)
    args = parser.parse_args()

    world_points = [parse_point(p) for p in args.points]
    if len(world_points) < 4:
        raise ValueError("Need at least 4 points for a homography fit.")

    camera_matrix, dist_coeffs = load_camera_calibration()
    detector = get_detector()

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera_id}")

    pixel_points: list[np.ndarray] = []

    for i, (wx, wy) in enumerate(world_points):
        print(f"\nPoint {i + 1}/{len(world_points)}: place marker {args.marker_id} at "
              f"world ({wx:.3f}, {wy:.3f}). Press SPACE when ready, 'q' to abort.")

        captured = False
        while not captured:
            ok, frame = cap.read()
            if not ok:
                continue

            centers = detect_marker_centers(frame, detector, camera_matrix, dist_coeffs)
            display = frame.copy()
            if args.marker_id in centers:
                u, v = centers[args.marker_id]
                cv2.circle(display, (int(u), int(v)), 8, (0, 255, 0), 2)
                cv2.putText(display, "marker found - press SPACE", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(display, "marker not visible", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("calibrate_homography", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" ") and args.marker_id in centers:
                samples = []
                for _ in range(args.samples_per_point):
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    centers = detect_marker_centers(frame, detector, camera_matrix, dist_coeffs)
                    if args.marker_id in centers:
                        samples.append(centers[args.marker_id])
                avg = np.mean(samples, axis=0)
                pixel_points.append(avg)
                print(f"  -> captured pixel {avg}")
                captured = True
            elif key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()

    src = np.array(pixel_points, dtype=np.float64)       # (N, 2) pixel
    dst = np.array(world_points, dtype=np.float64)        # (N, 2) world xy

    homography, _ = cv2.findHomography(src, dst, method=0)

    # Sanity check: reproject the calibration points and report the error.
    errors = []
    for px, wxy in zip(src, dst):
        pt = np.array([px[0], px[1], 1.0])
        proj = homography @ pt
        proj /= proj[2]
        errors.append(np.linalg.norm(proj[:2] - wxy))
    errors = np.array(errors)
    print(f"\nReprojection error: mean={errors.mean()*1000:.2f}mm  max={errors.max()*1000:.2f}mm")
    if errors.max() > 0.005:
        print("WARNING: max error > 5mm — consider re-measuring your points or "
              "adding more spread-out calibration points.")

    save_homography(homography, args.z_plane)
    print(f"\nSaved to sim_to_real/calibration/homography.npz (z_plane={args.z_plane})")


if __name__ == "__main__":
    main()
