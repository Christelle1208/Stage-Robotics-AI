"""Shared ArUco + calibration I/O helpers used by every script in this folder.

Keeping this logic in one place means the camera matrix, distortion
coefficients, ArUco dictionary, and homography are always loaded/applied the
same way during calibration AND during inference.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

CALIBRATION_DIR = Path(__file__).parent / "calibration"
CAMERA_CALIB_PATH = CALIBRATION_DIR / "camera_intrinsics.npz"
HOMOGRAPHY_PATH = CALIBRATION_DIR / "homography.npz"

# All scripts default to this dictionary. 4x4_50 markers are small, easy to
# print at a few cm, and 50 unique IDs is more than enough (cube + a couple
# of calibration anchors).
DEFAULT_ARUCO_DICT = "DICT_4X4_50"


def get_aruco_dict(name: str = DEFAULT_ARUCO_DICT) -> cv2.aruco.Dictionary:
    dict_id = getattr(cv2.aruco, name)
    return cv2.aruco.getPredefinedDictionary(dict_id)


def get_detector(dict_name: str = DEFAULT_ARUCO_DICT) -> cv2.aruco.ArucoDetector:
    aruco_dict = get_aruco_dict(dict_name)
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(aruco_dict, params)


# ---------------------------------------------------------------------------
# Camera intrinsics
# ---------------------------------------------------------------------------

def save_camera_calibration(camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> None:
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CAMERA_CALIB_PATH, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)


def load_camera_calibration(path: Path | str = CAMERA_CALIB_PATH) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["camera_matrix"], data["dist_coeffs"]


# ---------------------------------------------------------------------------
# Homography (pixel -> table-plane world XY)
# ---------------------------------------------------------------------------

def save_homography(homography: np.ndarray, z_plane: float) -> None:
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(HOMOGRAPHY_PATH, homography=homography, z_plane=z_plane)


def load_homography(path: Path | str = HOMOGRAPHY_PATH) -> tuple[np.ndarray, float]:
    data = np.load(path)
    return data["homography"], float(data["z_plane"])


def pixel_to_world_xy(pixel_xy: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Apply a homography to a single (u, v) pixel -> (x, y) world point.

    ``pixel_xy`` must already be undistorted (see ``undistort_points``).
    """
    pt = np.array([pixel_xy[0], pixel_xy[1], 1.0])
    world = homography @ pt
    world /= world[2]
    return world[:2]


# ---------------------------------------------------------------------------
# Marker detection
# ---------------------------------------------------------------------------

def undistort_points(points: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    """Undistort an (N, 2) array of pixel points, returns (N, 2)."""
    pts = points.reshape(-1, 1, 2).astype(np.float64)
    undistorted = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, P=camera_matrix)
    return undistorted.reshape(-1, 2)


def detect_marker_centers(
    frame: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
    camera_matrix: np.ndarray | None = None,
    dist_coeffs: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    """Detect all ArUco markers in ``frame``.

    Returns a dict mapping marker id -> (u, v) pixel center. If
    ``camera_matrix``/``dist_coeffs`` are given, centers are undistorted
    before being returned (recommended — homography assumes a pinhole model
    with no lens distortion).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    corners, ids, _ = detector.detectMarkers(gray)

    centers: dict[int, np.ndarray] = {}
    if ids is None:
        return centers

    for marker_corners, marker_id in zip(corners, ids.flatten()):
        center = marker_corners[0].mean(axis=0)  # (4, 2) -> (2,)
        if camera_matrix is not None and dist_coeffs is not None:
            center = undistort_points(center[None, :], camera_matrix, dist_coeffs)[0]
        centers[int(marker_id)] = center

    return centers
