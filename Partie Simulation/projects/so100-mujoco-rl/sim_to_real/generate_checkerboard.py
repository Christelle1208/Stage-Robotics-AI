#!/usr/bin/env python3
"""Generate a printable checkerboard image for camera calibration.

This creates a high-resolution black/white checkerboard PNG where the number
of *inner corners* matches OpenCV calibration parameters.

Example
-------
python3 sim_to_real/generate_checkerboard.py \
  --cols 9 --rows 6 --cell-mm 25 \
  --out sim_to_real/calibration/checkerboard_9x6_25mm.png

Notes
-----
- OpenCV `findChessboardCorners` expects inner-corner counts.
- A 9x6-inner-corner board has 10x7 squares.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate printable checkerboard PNG.")
    parser.add_argument("--cols", type=int, default=9, help="Inner corners per row.")
    parser.add_argument("--rows", type=int, default=6, help="Inner corners per column.")
    parser.add_argument("--cell-mm", type=float, default=25.0, help="Square size in millimetres.")
    parser.add_argument("--dpi", type=int, default=300, help="Output image DPI.")
    parser.add_argument(
        "--margin-mm",
        type=float,
        default=12.0,
        help="White border margin around checkerboard in millimetres.",
    )
    parser.add_argument(
        "--out",
        default="sim_to_real/calibration/checkerboard.png",
        help="Output PNG path.",
    )
    args = parser.parse_args()

    squares_x = args.cols + 1
    squares_y = args.rows + 1

    px_per_mm = args.dpi / 25.4
    cell_px = max(4, int(round(args.cell_mm * px_per_mm)))
    margin_px = max(2, int(round(args.margin_mm * px_per_mm)))

    board_w = squares_x * cell_px
    board_h = squares_y * cell_px

    img_h = board_h + 2 * margin_px
    img_w = board_w + 2 * margin_px
    img = np.full((img_h, img_w), 255, dtype=np.uint8)

    for y in range(squares_y):
        for x in range(squares_x):
            if (x + y) % 2 == 0:
                x0 = margin_px + x * cell_px
                y0 = margin_px + y * cell_px
                img[y0:y0 + cell_px, x0:x0 + cell_px] = 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write with DPI metadata for accurate print scaling.
    cv2.imwrite(str(out_path), img)

    print(f"Saved checkerboard to: {out_path}")
    print(f"Inner corners: {args.cols} x {args.rows}")
    print(f"Squares      : {squares_x} x {squares_y}")
    print(f"Square size  : {args.cell_mm:.2f} mm")
    print(f"Resolution   : {img_w} x {img_h} px @ {args.dpi} DPI")
    print("Print at 100% scale (disable 'fit to page').")


if __name__ == "__main__":
    main()
