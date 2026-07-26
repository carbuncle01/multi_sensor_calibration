"""Pairwise checkerboard extrinsic calibration and manual fallback."""

from __future__ import annotations

import math
from typing import Any

from .intrinsics import CheckerboardObservation, object_points
from .transforms import quaternion_from_rpy


def pair_observations(
    reference: list[CheckerboardObservation],
    sensor: list[CheckerboardObservation],
    max_delta_s: float,
) -> list[tuple[CheckerboardObservation, CheckerboardObservation]]:
    if max_delta_s < 0.0:
        raise ValueError("max_delta_s must be non-negative")
    reference = sorted(reference, key=lambda item: item.time_s)
    sensor = sorted(sensor, key=lambda item: item.time_s)
    candidates = []
    for reference_index, reference_item in enumerate(reference):
        for sensor_index, sensor_item in enumerate(sensor):
            delta = abs(sensor_item.time_s - reference_item.time_s)
            if delta <= max_delta_s:
                candidates.append((delta, reference_index, sensor_index))

    used_reference = set()
    used_sensor = set()
    selected = []
    for _, reference_index, sensor_index in sorted(candidates):
        if reference_index in used_reference or sensor_index in used_sensor:
            continue
        used_reference.add(reference_index)
        used_sensor.add(sensor_index)
        selected.append((reference[reference_index], sensor[sensor_index]))
    return sorted(selected, key=lambda pair: pair[0].time_s)


def _intrinsic_arrays(value: dict[str, Any]):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for extrinsic calibration") from exc

    matrix = np.asarray(value["camera_matrix"]["data"], dtype=float).reshape(3, 3)
    distortion = np.asarray(
        value["distortion_coefficients"]["data"], dtype=float
    ).reshape(-1, 1)
    return matrix, distortion


def _quaternion_from_matrix(matrix) -> tuple[float, float, float, float]:
    m00, m01, m02 = (float(value) for value in matrix[0])
    m10, m11, m12 = (float(value) for value in matrix[1])
    m20, m21, m22 = (float(value) for value in matrix[2])
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return ((m21 - m12) / scale, (m02 - m20) / scale, (m10 - m01) / scale, 0.25 * scale)
    if m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        return (0.25 * scale, (m01 + m10) / scale, (m02 + m20) / scale, (m21 - m12) / scale)
    if m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        return ((m01 + m10) / scale, 0.25 * scale, (m12 + m21) / scale, (m02 - m20) / scale)
    scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
    return ((m02 + m20) / scale, (m12 + m21) / scale, 0.25 * scale, (m10 - m01) / scale)


def calibrate_pair(
    pairs: list[tuple[CheckerboardObservation, CheckerboardObservation]],
    target: dict[str, Any],
    reference_intrinsics: dict[str, Any],
    sensor_intrinsics: dict[str, Any],
    *,
    reference_frame: str,
    sensor_frame: str,
    max_pair_delta_s: float,
) -> dict[str, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy and OpenCV are required for extrinsic calibration") from exc

    if len(pairs) < 6:
        raise ValueError(f"at least 6 synchronized target observations are required; got {len(pairs)}")
    reference_matrix, reference_distortion = _intrinsic_arrays(reference_intrinsics)
    sensor_matrix, sensor_distortion = _intrinsic_arrays(sensor_intrinsics)
    target_points = object_points(target)
    object_sets = [target_points.copy() for _ in pairs]
    reference_points = [pair[0].corners for pair in pairs]
    sensor_points = [pair[1].corners for pair in pairs]
    image_size = pairs[0][0].image_size

    rms, _, _, _, _, rotation, translation, essential, fundamental = cv2.stereoCalibrate(
        object_sets,
        reference_points,
        sensor_points,
        reference_matrix,
        reference_distortion,
        sensor_matrix,
        sensor_distortion,
        image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )

    # OpenCV returns p_sensor = R * p_reference + T.  TF needs the pose of
    # child=sensor in parent=reference, which is the inverse transform.
    rotation_parent_child = rotation.T
    translation_parent_child = -rotation.T @ translation
    quaternion = _quaternion_from_matrix(rotation_parent_child)
    deltas = [abs(pair[0].time_s - pair[1].time_s) for pair in pairs]

    return {
        "schema_version": 1,
        "method": "opencv_stereo_checkerboard_fixed_intrinsics",
        "reference_frame": reference_frame,
        "sensor_frame": sensor_frame,
        "coordinate_equation": "p_sensor = R_sensor_reference * p_reference + t_sensor_reference",
        "R_sensor_reference": [float(value) for value in rotation.reshape(-1)],
        "t_sensor_reference_m": [float(value) for value in translation.reshape(-1)],
        "tf_parent_child": {
            "parent_frame": reference_frame,
            "child_frame": sensor_frame,
            "translation_m": [
                float(value) for value in translation_parent_child.reshape(-1)
            ],
            "quaternion_xyzw": [float(value) for value in quaternion],
        },
        "quality": {
            "rms_reprojection_error_px": float(rms),
            "pair_count": len(pairs),
            "maximum_allowed_pair_delta_s": max_pair_delta_s,
            "mean_pair_delta_s": float(np.mean(deltas)),
            "maximum_pair_delta_s": float(np.max(deltas)),
        },
        "essential_matrix": [float(value) for value in essential.reshape(-1)],
        "fundamental_matrix": [float(value) for value in fundamental.reshape(-1)],
        "target": dict(target),
        "pairs": [
            {
                "reference_time_s": reference_item.time_s,
                "sensor_time_s": sensor_item.time_s,
                "delta_s": abs(reference_item.time_s - sensor_item.time_s),
                "reference_source": reference_item.source,
                "sensor_source": sensor_item.source,
            }
            for reference_item, sensor_item in pairs
        ],
    }


def manual_extrinsic(
    *,
    parent_frame: str,
    child_frame: str,
    xyz_m: list[float],
    rpy_rad: list[float],
) -> dict[str, Any]:
    if len(xyz_m) != 3 or len(rpy_rad) != 3:
        raise ValueError("xyz_m and rpy_rad must each contain three values")
    quaternion = quaternion_from_rpy(*rpy_rad)
    return {
        "schema_version": 1,
        "method": "manual",
        "tf_parent_child": {
            "parent_frame": parent_frame,
            "child_frame": child_frame,
            "translation_m": [float(value) for value in xyz_m],
            "rpy_rad": [float(value) for value in rpy_rad],
            "quaternion_xyzw": [float(value) for value in quaternion],
        },
        "quality": {
            "status": "unvalidated",
            "note": "Validate this transform with independent data before runtime use.",
        },
    }
