#!/usr/bin/env python3
"""Generate printable ArUco marker images (DICT_4X4_50 by default).

Usage
-----
    python3 sim_to_real/generate_markers.py --ids 0 1 2 --size-px 600 \
        --out sim_to_real/calibration/markers

Only the marker's pixel CENTER is used (for homography lookups), so the
printed physical size doesn't need to be measured — just make sure it's big
enough to be detected reliably from your camera's distance (a few cm is
usually plenty).

Suggested marker assignment (change as you like, just stay consistent):
    ID 0       -> stuck on top of the cube (the object we track at runtime)
    ID 1, 2... -> "anchor" markers used only during homography calibration
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from aruco_utils import DEFAULT_ARUCO_DICT, get_aruco_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ArUco marker images.")
    parser.add_argument("--ids", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                         help="Marker IDs to generate.")
    parser.add_argument("--dict", default=DEFAULT_ARUCO_DICT)
    parser.add_argument("--size-px", type=int, default=600, help="Image size in pixels.")
    parser.add_argument("--out", default="sim_to_real/calibration/markers",
                         help="Output directory.")
    args = parser.parse_args()

    aruco_dict = get_aruco_dict(args.dict)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for marker_id in args.ids:
        img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, args.size_px)
        path = out_dir / f"marker_{marker_id}.png"
        cv2.imwrite(str(path), img)
        print(f"Saved {path}")

    print(
        "\nPrint these. Marker 0 goes on top of the cube; marker 1 is the "
        "loose probe used by calibrate_homography.py. Physical size doesn't "
        "need to be measured — only the marker's pixel CENTER is used."
    )


if __name__ == "__main__":
    main()
