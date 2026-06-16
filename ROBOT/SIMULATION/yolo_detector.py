"""
yolo_detector.py
-----------------
YOLOv10-based cube detector for the SO-ARM100 TEST_SCENE.

Workflow:
  1. Render the 'topdown_cam' MuJoCo camera (top-down view over the table).
  2. Run ultralytics YOLOv10 on the image.
  3. Unproject the bounding-box centre from 2D image → 3D world coordinates
     via ray-plane intersection at a known cube height.

The trained YOLO model is expected at:
    SIMULATION/models/cube_yolov10.pt

To train it, first generate the dataset:
    python generate_yolo_data.py
then run:
    yolo train  model=yolov10n.pt  data=configs/cube_yolo.yaml  epochs=50  imgsz=320
"""

from __future__ import annotations

import pathlib
from typing import Sequence

import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# Constants — must match TEST_SCENE.xml camera definition
# ---------------------------------------------------------------------------
TOPDOWN_CAM_NAME = "topdown_cam"
IMG_W  = 320
IMG_H  = 320
CONF_THRESHOLD = 0.30

_THIS_DIR        = pathlib.Path(__file__).parent
DEFAULT_MODEL_PT = _THIS_DIR / "models" / "cube_yolov10.pt"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _pixel_to_world_ray(
    u: float,
    v: float,
    cam_pos: np.ndarray,
    cam_xmat: np.ndarray,
    fovy_deg: float,
    img_w: int,
    img_h: int,
    world_z: float,
) -> np.ndarray:
    """
    Unproject image pixel (u, v) to world 3D by intersecting camera ray with
    the horizontal plane  z = world_z.

    MuJoCo camera convention (OpenGL):
        * Camera looks along its local -Z axis.
        * Camera +X is right in the image, +Y is up.
        * cam_xmat (3×3) maps camera-frame vectors to world-frame vectors.

    Parameters
    ----------
    u, v     : pixel coordinates in [0, img_w) × [0, img_h)
    cam_pos  : camera world position  (3,)
    cam_xmat : camera rotation matrix (3×3), world <- camera
    fovy_deg : vertical field-of-view in degrees
    world_z  : Z-height of the intersection plane (e.g. cube rest height)

    Returns
    -------
    np.ndarray of shape (3,) — world position at the intersection.
    """
    fovy_rad = np.deg2rad(fovy_deg)
    focal_y  = (img_h / 2.0) / np.tan(fovy_rad / 2.0)
    focal_x  = focal_y  # square pixels

    # Direction in camera frame (-Z forward, +X right, +Y up)
    x_cam =  (u - img_w / 2.0) / focal_x
    y_cam = -(v - img_h / 2.0) / focal_y   # image y is flipped
    z_cam = -1.0

    ray_cam   = np.array([x_cam, y_cam, z_cam], dtype=np.float64)
    ray_world = cam_xmat @ ray_cam  # world-frame direction

    # Intersect with plane  P.z = world_z
    # cam_pos + t * ray_world = ? where result.z = world_z
    if abs(ray_world[2]) < 1e-9:
        raise ValueError("Ray is parallel to the horizontal plane — cannot intersect.")

    t   = (world_z - cam_pos[2]) / ray_world[2]
    hit = cam_pos + t * ray_world
    return hit.astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class CubeDetector:
    """
    Detects the green cube in the MuJoCo scene using a YOLOv10 model.

    Usage
    -----
    detector = CubeDetector(model, data)
    cube_pos_3d = detector.detect(data, cube_world_z=0.22)  # → np.ndarray(3,) or None

    Parameters
    ----------
    model      : mujoco.MjModel
    data       : mujoco.MjData (initial reference; detect() accepts updated data)
    model_path : path to trained YOLO .pt file (default: SIMULATION/models/cube_yolov10.pt)
    img_w/h    : rendering resolution (must match training resolution)
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        model_path: str | pathlib.Path | None = None,
        img_w: int = IMG_W,
        img_h: int = IMG_H,
    ) -> None:
        self._model = model
        self._data  = data
        self._img_w = img_w
        self._img_h = img_h

        # Resolve top-down camera
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, TOPDOWN_CAM_NAME)
        if cam_id < 0:
            raise RuntimeError(
                f"Camera '{TOPDOWN_CAM_NAME}' not found in model. "
                "Make sure TEST_SCENE.xml has been updated."
            )
        self._cam_id = cam_id

        # Read FOV from model (stored in radians internally)
        self._fovy_deg = float(np.rad2deg(model.cam_fovy[cam_id]))

        # Offscreen renderer
        self._renderer = mujoco.Renderer(model, img_h, img_w)

        # Load YOLOv10 model
        pt_path = pathlib.Path(model_path) if model_path else DEFAULT_MODEL_PT
        if not pt_path.exists():
            raise FileNotFoundError(
                f"YOLO model not found at '{pt_path}'. "
                "Generate synthetic data and train it first:\n"
                "  python generate_yolo_data.py\n"
                "  yolo train model=yolov10n.pt data=configs/cube_yolo.yaml epochs=50 imgsz=320"
            )
        from ultralytics import YOLO as _YOLO
        self._yolo = _YOLO(str(pt_path))

    # ------------------------------------------------------------------

    def render_topdown(self) -> np.ndarray:
        """
        Render the top-down camera view.

        Returns
        -------
        np.ndarray of shape (img_h, img_w, 3), dtype uint8, RGB.
        """
        self._renderer.update_scene(self._data, camera=TOPDOWN_CAM_NAME)
        return self._renderer.render()

    def _cam_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (cam_pos, cam_xmat_3x3) from current simulation state."""
        mujoco.mj_forward(self._model, self._data)
        pos  = self._data.cam_xpos[self._cam_id].copy()
        xmat = self._data.cam_xmat[self._cam_id].reshape(3, 3).copy()
        return pos, xmat

    # ------------------------------------------------------------------

    def detect(
        self,
        data: mujoco.MjData,
        cube_world_z: float = 0.22,
        visualize: bool = False,
    ) -> np.ndarray | None:
        """
        Run YOLO on the top-down camera frame and return the 3D world position
        of the highest-confidence detected cube, or None if nothing is found.

        Parameters
        ----------
        data         : up-to-date MjData (after mj_forward)
        cube_world_z : Z height of the cube centre (used for ray-plane intersection)
        visualize    : if True, show the annotated detection frame with cv2
        """
        self._data = data
        frame = self.render_topdown()   # RGB uint8

        results = self._yolo.predict(frame, conf=CONF_THRESHOLD, verbose=False)

        if not results or len(results[0].boxes) == 0:
            return None

        boxes    = results[0].boxes
        best_idx = int(boxes.conf.argmax())
        xyxy     = boxes.xyxy[best_idx].cpu().numpy()
        u        = float((xyxy[0] + xyxy[2]) / 2.0)
        v        = float((xyxy[1] + xyxy[3]) / 2.0)

        cam_pos, cam_xmat = self._cam_pose()
        world_pos = _pixel_to_world_ray(
            u, v, cam_pos, cam_xmat, self._fovy_deg,
            self._img_w, self._img_h, cube_world_z,
        )

        if visualize:
            self._show_detection(frame, xyxy, world_pos)

        return world_pos

    def _show_detection(
        self,
        frame: np.ndarray,
        xyxy: Sequence[float],
        world_pos: np.ndarray,
    ) -> None:
        """Draw bounding box and world position on the frame (cv2 required)."""
        try:
            import cv2
        except ImportError:
            print("[CubeDetector] cv2 not available — skipping visualisation.")
            return

        vis = frame[:, :, ::-1].copy()  # RGB → BGR
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"cube ({world_pos[0]:.2f}, {world_pos[1]:.2f}, {world_pos[2]:.2f})"
        cv2.putText(vis, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cx, cy = int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)
        cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)
        cv2.imshow("YOLO top-down detection", vis)
        cv2.waitKey(1)

    def close(self) -> None:
        self._renderer.close()
