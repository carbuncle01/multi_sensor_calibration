"""Checkerboard-based pinhole lens calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .calibration_images import CalibrationImage
from .imaging import to_uint8


@dataclass(frozen=True)
class CheckerboardObservation:
    time_s: float
    corners: Any
    image_size: tuple[int, int]
    source: str


def detect_checkerboard(
    images: Iterable[CalibrationImage],
    target: dict[str, Any],
) -> list[CheckerboardObservation]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for checkerboard detection") from exc

    pattern_size = (int(target["columns"]), int(target["rows"]))
    observations = []
    for item in images:
        gray = to_uint8(item.image)
        found = False
        corners = None
        if hasattr(cv2, "findChessboardCornersSB"):
            found, corners = cv2.findChessboardCornersSB(
                gray,
                pattern_size,
                flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
            )
        if not found:
            found, corners = cv2.findChessboardCorners(
                gray,
                pattern_size,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if found:
                criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    1e-3,
                )
                corners = cv2.cornerSubPix(
                    gray, corners, (5, 5), (-1, -1), criteria
                )
        if found:
            observations.append(
                CheckerboardObservation(
                    time_s=item.time_s,
                    corners=corners,
                    image_size=(gray.shape[1], gray.shape[0]),
                    source=item.source,
                )
            )
    return observations


def object_points(target: dict[str, Any]):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for lens calibration") from exc

    columns = int(target["columns"])
    rows = int(target["rows"])
    square = float(target["square_size_m"])
    points = np.zeros((columns * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    points[:, :2] *= square
    return points


def calibrate_intrinsics(
    observations: list[CheckerboardObservation],
    target: dict[str, Any],
    *,
    camera_name: str,
    frame_id: str,
) -> dict[str, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy and OpenCV are required for lens calibration") from exc

    if len(observations) < 8:
        raise ValueError(
            f"at least 8 checkerboard observations are required; got {len(observations)}"
        )
    image_size = observations[0].image_size
    if any(item.image_size != image_size for item in observations):
        raise ValueError("all calibration images must have the same dimensions")

    target_points = object_points(target)
    object_sets = [target_points.copy() for _ in observations]
    image_sets = [item.corners for item in observations]
    rms, camera_matrix, distortion, rotations, translations = cv2.calibrateCamera(
        object_sets,
        image_sets,
        image_size,
        None,
        None,
    )

    per_view_errors = []
    for points_3d, points_2d, rotation, translation in zip(
        object_sets, image_sets, rotations, translations
    ):
        projected, _ = cv2.projectPoints(
            points_3d, rotation, translation, camera_matrix, distortion
        )
        error = cv2.norm(points_2d, projected, cv2.NORM_L2) / len(projected)
        per_view_errors.append(float(error))

    return {
        "schema_version": 1,
        "method": "opencv_checkerboard_pinhole_radtan",
        "camera_name": camera_name,
        "frame_id": frame_id,
        "image_width": image_size[0],
        "image_height": image_size[1],
        "distortion_model": "plumb_bob",
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": [float(value) for value in camera_matrix.reshape(-1)],
        },
        "distortion_coefficients": {
            "rows": 1,
            "cols": int(distortion.size),
            "data": [float(value) for value in distortion.reshape(-1)],
        },
        "rectification_matrix": {
            "rows": 3,
            "cols": 3,
            "data": [float(value) for value in np.eye(3).reshape(-1)],
        },
        "projection_matrix": {
            "rows": 3,
            "cols": 4,
            "data": [
                float(camera_matrix[0, 0]),
                0.0,
                float(camera_matrix[0, 2]),
                0.0,
                0.0,
                float(camera_matrix[1, 1]),
                float(camera_matrix[1, 2]),
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ],
        },
        "quality": {
            "rms_reprojection_error_px": float(rms),
            "observation_count": len(observations),
            "mean_view_error_px": float(np.mean(per_view_errors)),
            "maximum_view_error_px": float(np.max(per_view_errors)),
        },
        "target": dict(target),
        "observations": [
            {"time_s": item.time_s, "source": item.source}
            for item in observations
        ],
    }
