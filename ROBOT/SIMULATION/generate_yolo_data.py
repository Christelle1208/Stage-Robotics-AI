"""
generate_yolo_data.py
----------------------
Generate a synthetic top-down dataset for YOLOv10 cube detection.

The script:
  1. Loads TEST_SCENE.xml in MuJoCo.
  2. Randomises the cube position on the table at each sample.
  3. Renders the 'topdown_cam' view (320×320).
  4. Projects the known cube 3D centre to image (u,v).
  5. Writes YOLO-format labels (.txt) alongside the images.

Output layout:
    data/cube_yolo/
    ├── images/
    │   ├── train/   (80 % of samples)
    │   └── val/     (20 % of samples)
    └── labels/
        ├── train/
        └── val/

Usage:
    cd SIMULATION/
    python generate_yolo_data.py --n_images 2000 --output_dir data/cube_yolo
"""

from __future__ import annotations

import argparse
import pathlib
import random
import sys

import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# Defaults matching reach_cube_env.py and TEST_SCENE.xml
# ---------------------------------------------------------------------------
_THIS_DIR  = pathlib.Path(__file__).parent
SCENE_PATH = _THIS_DIR / "trs_so_arm100" / "TEST_SCENE.xml"

TOPDOWN_CAM_NAME = "topdown_cam"
IMG_W, IMG_H     = 320, 320

CUBE_X_RANGE = (0.26, 0.54)
CUBE_Y_RANGE = (-0.13, 0.13)
CUBE_REST_Z  = 0.22

# Bounding-box half-size in metres (cube visible top-face is ~40 mm wide)
CUBE_VIS_HALF = 0.022   # slightly larger than geom half-size for reliable label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _world_to_pixel(
    xyz: np.ndarray,
    cam_pos: np.ndarray,
    cam_xmat: np.ndarray,
    fovy_deg: float,
    img_w: int,
    img_h: int,
) -> tuple[float, float]:
    """
    Project a 3D world point to image pixel (u, v).

    Inverse of yolo_detector._pixel_to_world_ray.
    """
    fovy_rad = np.deg2rad(fovy_deg)
    focal_y  = (img_h / 2.0) / np.tan(fovy_rad / 2.0)
    focal_x  = focal_y

    # Transform world point to camera frame
    p_rel   = xyz - cam_pos
    p_cam   = cam_xmat.T @ p_rel   # world -> camera  (cam_xmat is world<-cam, so .T is cam<-world)

    if p_cam[2] >= 0:
        # Behind or on the camera plane — shouldn't happen for top-down view
        raise ValueError(f"Point {xyz} is behind the camera.")

    # Perspective divide (camera -Z is forward)
    z_div = -p_cam[2]
    u = ( p_cam[0] / z_div) * focal_x + img_w / 2.0
    v = (-p_cam[1] / z_div) * focal_y + img_h / 2.0   # y flip
    return float(u), float(v)


def _metres_to_pixels(metres: float, cam_pos_z: float, cube_z: float,
                       focal_y: float, img_h: int) -> float:
    """Convert a horizontal metric size to pixels at the cube depth."""
    depth  = abs(cam_pos_z - cube_z)
    pixels = (metres / depth) * focal_y
    return pixels


def generate_dataset(
    n_images: int,
    output_dir: pathlib.Path,
    val_fraction: float = 0.20,
    seed: int = 42,
    add_home_arm: bool = True,
) -> None:
    rng = np.random.default_rng(seed)
    random.seed(seed)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data  = mujoco.MjData(model)

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, TOPDOWN_CAM_NAME)
    if cam_id < 0:
        sys.exit(f"[ERROR] Camera '{TOPDOWN_CAM_NAME}' not found. "
                 "Did you update TEST_SCENE.xml?")

    fovy_deg = float(np.rad2deg(model.cam_fovy[cam_id]))

    # Object joint address
    obj_jnt_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_joint")
    obj_qpos    = int(model.jnt_qposadr[obj_jnt_id])
    obj_qvel    = int(model.jnt_dofadr[obj_jnt_id])

    # Arm home position (set once, unless random arm pose is requested)
    HOME = np.array([0.0, -1.57, 1.57, 1.57, -1.57, 0.0])
    # Vary arm posture slightly for domain randomisation
    ARM_NOISE_STD = np.array([0.3, 0.2, 0.2, 0.2, 0.2, 0.0])

    renderer = mujoco.Renderer(model, IMG_H, IMG_W)

    # ------------------------------------------------------------------
    # Output folders
    # ------------------------------------------------------------------
    splits = {"train": 1 - val_fraction, "val": val_fraction}
    for split in splits:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    split_names = ["train"] * int(n_images * (1 - val_fraction)) + \
                  ["val"]   * (n_images - int(n_images * (1 - val_fraction)))
    random.shuffle(split_names)

    print(f"Generating {n_images} images ({IMG_W}×{IMG_H}) → {output_dir}")
    print(f"  Camera '{TOPDOWN_CAM_NAME}': fovy={fovy_deg:.1f}°")
    print(f"  Cube X ∈ {CUBE_X_RANGE}, Y ∈ {CUBE_Y_RANGE}, Z={CUBE_REST_Z}")

    # Pixel size of the cube bounding box annotation
    focal_y   = (IMG_H / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
    img_idx   = 0

    for i, split in enumerate(split_names):
        mujoco.mj_resetData(model, data)

        # -- Arm posture (with small random noise for visual diversity) --
        arm_qpos = HOME + rng.normal(0, ARM_NOISE_STD)
        data.qpos[:6] = arm_qpos
        data.ctrl[:6] = arm_qpos

        # -- Cube position --
        cx = float(rng.uniform(*CUBE_X_RANGE))
        cy = float(rng.uniform(*CUBE_Y_RANGE))
        cz = CUBE_REST_Z
        data.qpos[obj_qpos     : obj_qpos + 3] = [cx, cy, cz]
        data.qpos[obj_qpos + 3 : obj_qpos + 7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[obj_qvel     : obj_qvel + 6] = 0.0
        mujoco.mj_forward(model, data)

        # -- Render --
        renderer.update_scene(data, camera=TOPDOWN_CAM_NAME)
        frame = renderer.render()   # RGB uint8 (IMG_H, IMG_W, 3)

        # -- Get camera pose for projection --
        cam_pos  = data.cam_xpos[cam_id].copy()
        cam_xmat = data.cam_xmat[cam_id].reshape(3, 3).copy()

        # -- Project cube centre to image --
        try:
            u, v = _world_to_pixel(
                np.array([cx, cy, cz]), cam_pos, cam_xmat, fovy_deg, IMG_W, IMG_H
            )
        except ValueError:
            continue  # skip degenerate sample

        # Skip if projected outside image (shouldn't happen with our ranges)
        if not (0 <= u < IMG_W and 0 <= v < IMG_H):
            print(f"  [skip] cube projected outside image: u={u:.1f}, v={v:.1f}")
            continue

        # -- Bounding box (CUBE_VIS_HALF visible face, in pixels) --
        depth   = abs(cam_pos[2] - cz)
        half_px = float(CUBE_VIS_HALF / depth * focal_y)
        half_px = max(half_px, 2.0)   # minimum 2px

        # YOLO label: class_id x_centre y_centre width height  (all normalised)
        xc_n = u / IMG_W
        yc_n = v / IMG_H
        w_n  = (2 * half_px) / IMG_W
        h_n  = (2 * half_px) / IMG_H
        w_n  = min(w_n, 1.0)
        h_n  = min(h_n, 1.0)

        # -- Save image (PIL not required — use numpy + built-in bytes) --
        img_name = f"{img_idx:06d}.png"
        img_path = output_dir / "images" / split / img_name
        _save_png(frame, img_path)

        # -- Save label --
        lbl_path = output_dir / "labels" / split / f"{img_idx:06d}.txt"
        lbl_path.write_text(f"0 {xc_n:.6f} {yc_n:.6f} {w_n:.6f} {h_n:.6f}\n")

        img_idx += 1
        if (img_idx) % 200 == 0:
            print(f"  {img_idx}/{n_images} images generated …")

    renderer.close()
    print(f"Done. {img_idx} images saved to {output_dir}")


def _save_png(frame: np.ndarray, path: pathlib.Path) -> None:
    """Save an RGB uint8 numpy array as PNG using PIL (or cv2 as fallback)."""
    try:
        from PIL import Image
        Image.fromarray(frame).save(path)
        return
    except ImportError:
        pass
    try:
        import cv2
        cv2.imwrite(str(path), frame[:, :, ::-1])   # RGB → BGR
    except ImportError:
        # Last resort: write raw PPM
        ppm_path = path.with_suffix(".ppm")
        h, w, _ = frame.shape
        with open(ppm_path, "wb") as f:
            f.write(f"P6\n{w} {h}\n255\n".encode())
            f.write(frame.tobytes())
        print(f"  [warn] PIL/cv2 not available — saved {ppm_path} instead.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic YOLO cube dataset.")
    p.add_argument("--n_images",    type=int,   default=2000,
                   help="Total number of images (default: 2000).")
    p.add_argument("--output_dir",  type=str,   default="data/cube_yolo",
                   help="Root output directory (default: data/cube_yolo).")
    p.add_argument("--val_fraction",type=float, default=0.20,
                   help="Fraction used for validation (default: 0.20).")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_dataset(
        n_images     = args.n_images,
        output_dir   = pathlib.Path(args.output_dir),
        val_fraction = args.val_fraction,
        seed         = args.seed,
    )
