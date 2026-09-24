"""Render spatial-validation overlays from a Kalibr camera chain."""

from __future__ import annotations

import csv
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any


def _camera_matrix(intrinsics, np):
    if len(intrinsics) != 4:
        raise ValueError("pinhole intrinsics must contain [fx, fy, cx, cy]")
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def _camera(document: dict[str, Any], name: str, np):
    value = document.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"camera chain does not contain {name}")
    if value.get("camera_model") != "pinhole":
        raise ValueError(f"{name} must use the pinhole camera model")
    if value.get("distortion_model") != "radtan":
        raise ValueError(f"{name} must use the radtan distortion model")
    resolution = value.get("resolution")
    if not isinstance(resolution, list) or len(resolution) != 2:
        raise ValueError(f"{name}.resolution must contain [width, height]")
    return {
        "matrix": _camera_matrix(value.get("intrinsics", []), np),
        "distortion": np.asarray(value.get("distortion_coeffs", []), dtype=float),
        "size": (int(resolution[0]), int(resolution[1])),
    }


def _paired_images(dataset: Path, evs_camera: str, rgb_camera: str):
    manifest = dataset / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Kalibr manifest not found: {manifest}")
    groups: dict[int, dict[str, Path]] = {}
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            camera = str(row.get("camera", ""))
            if camera not in {evs_camera, rgb_camera}:
                continue
            target = int(row["target_reference_timestamp_ns"])
            image = (dataset / row["image"]).resolve()
            if dataset.resolve() not in image.parents or not image.is_file():
                raise FileNotFoundError(f"manifest image not found: {image}")
            groups.setdefault(target, {})[camera] = image
    result = []
    for target, images in sorted(groups.items()):
        if evs_camera in images and rgb_camera in images:
            result.append((target, images[evs_camera], images[rgb_camera]))
    if not result:
        raise ValueError("manifest contains no paired EVS/RGB images")
    return result


def _checkerboard(cv2, image, pattern_size):
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = (
            cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_EXHAUSTIVE
            | cv2.CALIB_CB_ACCURACY
        )
        found, corners = cv2.findChessboardCornersSB(image, pattern_size, flags)
        if found:
            return corners.astype("float32")
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(image, pattern_size, flags)
    if not found:
        return None
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        40,
        1e-4,
    )
    return cv2.cornerSubPix(image, corners, (5, 5), (-1, -1), criteria)


def _object_points(columns: int, rows: int, spacing_m: float, np):
    points = np.zeros((columns * rows, 3), dtype="float32")
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    points[:, :2] *= float(spacing_m)
    return points


def _plane_homography(cv2, rgb_corners, object_points, evs, rgb, transform, np):
    solved, rotation_vector, translation_rgb = cv2.solvePnP(
        object_points,
        rgb_corners,
        rgb["matrix"],
        rgb["distortion"],
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not solved:
        return None
    rotation_rgb_board, _ = cv2.Rodrigues(rotation_vector)
    rotation_rgb_evs = transform[:3, :3]
    translation_rgb_evs = transform[:3, 3:4]
    rotation_evs_rgb = rotation_rgb_evs.T
    translation_evs_rgb = -rotation_evs_rgb @ translation_rgb_evs
    rotation_evs_board = rotation_evs_rgb @ rotation_rgb_board
    translation_evs_board = (
        rotation_evs_rgb @ translation_rgb + translation_evs_rgb
    )
    board_to_rgb = rgb["matrix"] @ np.column_stack(
        (rotation_rgb_board[:, 0], rotation_rgb_board[:, 1], translation_rgb[:, 0])
    )
    board_to_evs = evs["matrix"] @ np.column_stack(
        (rotation_evs_board[:, 0], rotation_evs_board[:, 1], translation_evs_board[:, 0])
    )
    determinant = float(np.linalg.det(board_to_evs))
    if not math.isfinite(determinant) or abs(determinant) < 1e-12:
        return None
    return board_to_rgb @ np.linalg.inv(board_to_evs)


def _undistort_points(cv2, corners, camera):
    return cv2.undistortPoints(
        corners,
        camera["matrix"],
        camera["distortion"],
        P=camera["matrix"],
    )


def _alignment_error(cv2, evs_corners, rgb_corners, homography, evs, rgb, np):
    evs_points = _undistort_points(cv2, evs_corners, evs)
    rgb_points = _undistort_points(cv2, rgb_corners, rgb)
    mapped = cv2.perspectiveTransform(evs_points, homography)
    direct = np.linalg.norm(mapped[:, 0] - rgb_points[:, 0], axis=1)
    reverse = np.linalg.norm(mapped[::-1, 0] - rgb_points[:, 0], axis=1)
    if float(reverse.mean()) < float(direct.mean()):
        mapped = mapped[::-1]
        direct = reverse
    return mapped, rgb_points, direct


def _blend_overlay(base, warped, valid, alpha: float, cv2, np):
    result = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    event_color = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    event_color[:, :, 2] = (event_color[:, :, 2].astype("float32") * 0.2).astype("uint8")
    mask = valid > 0
    result[mask] = np.clip(
        (1.0 - alpha) * result[mask] + alpha * event_color[mask],
        0,
        255,
    ).astype("uint8")
    return result


def _edge_overlay(base, warped, valid, alpha: float, cv2, np):
    result = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    edges = cv2.Canny(warped, 40, 120)
    edges = cv2.dilate(edges, np.ones((2, 2), dtype="uint8"))
    mask = (edges > 0) & (valid > 0)
    color = np.array([255.0, 255.0, 0.0])
    result[mask] = np.clip(
        (1.0 - alpha) * result[mask] + alpha * color,
        0,
        255,
    ).astype("uint8")
    return result


def _annotate(image, text: str, cv2):
    cv2.rectangle(image, (0, 0), (min(image.shape[1], 560), 31), (0, 0, 0), -1)
    cv2.putText(
        image,
        text,
        (9, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def render_calibration_overlay(
    dataset_dir: str | Path,
    camchain_path: str | Path,
    output_dir: str | Path,
    *,
    evs_camera: str = "cam0",
    rgb_camera: str = "cam1",
    alpha: float = 0.45,
    fps: float | None = None,
    every_n: int = 1,
    max_frames: int | None = None,
) -> dict[str, Any]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if every_n <= 0:
        raise ValueError("every_n must be positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("OpenCV and NumPy are required for overlay rendering") from exc

    from .io import load_yaml, write_yaml

    dataset = Path(dataset_dir).resolve()
    camchain_file = Path(camchain_path).resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(f"Kalibr dataset not found: {dataset}")
    if not camchain_file.is_file():
        raise FileNotFoundError(f"Kalibr camera chain not found: {camchain_file}")
    job = load_yaml(dataset / "job.yaml")
    target = load_yaml(dataset / str(job.get("target", "target.yaml")))
    camchain = load_yaml(camchain_file)
    evs = _camera(camchain, evs_camera, np)
    rgb = _camera(camchain, rgb_camera, np)
    transform_value = camchain.get(rgb_camera, {}).get("T_cn_cnm1")
    transform = np.asarray(transform_value, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"{rgb_camera}.T_cn_cnm1 must be a 4x4 matrix")

    columns = int(target["targetCols"])
    rows = int(target["targetRows"])
    spacing_m = float(target["rowSpacingMeters"])
    pattern_size = (columns, rows)
    object_points = _object_points(columns, rows, spacing_m, np)
    pairs = _paired_images(dataset, evs_camera, rgb_camera)[::every_n]
    if max_frames is not None:
        pairs = pairs[:max_frames]
    if not pairs:
        raise ValueError("no frames remain after applying frame selection")

    output = Path(output_dir).resolve()
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f"output directory is not empty: {output}")
        output.rmdir()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    snapshot_dir = staging / "snapshots"
    snapshot_dir.mkdir()
    rate = float(fps if fps is not None else job.get("export_rate_hz", 4.0))
    if rate <= 0.0:
        raise ValueError("fps must be positive")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    blend_writer = cv2.VideoWriter(
        str(staging / "overlay_blend.mp4"), fourcc, rate, rgb["size"]
    )
    edge_writer = cv2.VideoWriter(
        str(staging / "overlay_edges.mp4"), fourcc, rate, rgb["size"]
    )
    if not blend_writer.isOpened() or not edge_writer.isOpened():
        blend_writer.release()
        edge_writer.release()
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError("OpenCV could not open the MP4 video writer")

    map0 = cv2.initUndistortRectifyMap(
        evs["matrix"], evs["distortion"], None, evs["matrix"], evs["size"], cv2.CV_32FC1
    )
    map1 = cv2.initUndistortRectifyMap(
        rgb["matrix"], rgb["distortion"], None, rgb["matrix"], rgb["size"], cv2.CV_32FC1
    )
    snapshot_indices = {
        round(index * (len(pairs) - 1) / min(11, len(pairs) - 1))
        for index in range(min(12, len(pairs)))
    } if len(pairs) > 1 else {0}
    errors = []
    rendered = 0
    detected_rgb = 0
    detected_both = 0
    try:
        try:
            for pair_index, (timestamp_ns, evs_path, rgb_path) in enumerate(pairs):
                evs_image = cv2.imread(str(evs_path), cv2.IMREAD_GRAYSCALE)
                rgb_image = cv2.imread(str(rgb_path), cv2.IMREAD_GRAYSCALE)
                if evs_image is None or rgb_image is None:
                    continue
                if (evs_image.shape[1], evs_image.shape[0]) != evs["size"]:
                    raise ValueError(f"unexpected EVS image size: {evs_path}")
                if (rgb_image.shape[1], rgb_image.shape[0]) != rgb["size"]:
                    raise ValueError(f"unexpected RGB image size: {rgb_path}")
                rgb_corners = _checkerboard(cv2, rgb_image, pattern_size)
                if rgb_corners is None:
                    continue
                detected_rgb += 1
                homography = _plane_homography(
                    cv2, rgb_corners, object_points, evs, rgb, transform, np
                )
                if homography is None or not np.isfinite(homography).all():
                    continue
                evs_undistorted = cv2.remap(
                    evs_image, map0[0], map0[1], cv2.INTER_LINEAR
                )
                rgb_undistorted = cv2.remap(
                    rgb_image, map1[0], map1[1], cv2.INTER_LINEAR
                )
                warped = cv2.warpPerspective(
                    evs_undistorted,
                    homography,
                    rgb["size"],
                    flags=cv2.INTER_LINEAR,
                )
                valid = cv2.warpPerspective(
                    np.full(evs["size"][::-1], 255, dtype="uint8"),
                    homography,
                    rgb["size"],
                    flags=cv2.INTER_NEAREST,
                )
                blend = _blend_overlay(
                    rgb_undistorted, warped, valid, alpha, cv2, np
                )
                edge = _edge_overlay(
                    rgb_undistorted, warped, valid, alpha, cv2, np
                )
                evs_corners = _checkerboard(cv2, evs_image, pattern_size)
                error_text = ""
                if evs_corners is not None:
                    mapped, rgb_points, frame_errors = _alignment_error(
                        cv2,
                        evs_corners,
                        rgb_corners,
                        homography,
                        evs,
                        rgb,
                        np,
                    )
                    detected_both += 1
                    errors.extend(float(value) for value in frame_errors)
                    error_text = f" corner mean={float(frame_errors.mean()):.2f}px"
                    for index in range(0, len(mapped), 3):
                        evs_point = tuple(
                            int(round(value)) for value in mapped[index, 0]
                        )
                        rgb_point = tuple(
                            int(round(value)) for value in rgb_points[index, 0]
                        )
                        cv2.drawMarker(
                            edge,
                            evs_point,
                            (255, 0, 255),
                            cv2.MARKER_CROSS,
                            8,
                            1,
                        )
                        cv2.circle(
                            edge, rgb_point, 3, (0, 255, 0), 1, cv2.LINE_AA
                        )
                seconds = timestamp_ns / 1_000_000_000.0
                _annotate(
                    blend,
                    f"EVS -> RGB plane overlay  alpha={alpha:.2f}  t={seconds:.3f}s",
                    cv2,
                )
                _annotate(
                    edge,
                    f"EVS edges cyan / RGB gray{error_text}  t={seconds:.3f}s",
                    cv2,
                )
                blend_writer.write(blend)
                edge_writer.write(edge)
                if pair_index in snapshot_indices:
                    cv2.imwrite(
                        str(snapshot_dir / f"{timestamp_ns}_blend.png"), blend
                    )
                    cv2.imwrite(
                        str(snapshot_dir / f"{timestamp_ns}_edges.png"), edge
                    )
                rendered += 1
        finally:
            blend_writer.release()
            edge_writer.release()
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if rendered == 0:
        shutil.rmtree(staging, ignore_errors=True)
        raise ValueError("checkerboard was not detected in any RGB frame")
    error_array = np.asarray(errors, dtype=float)
    summary = {
        "schema_version": 1,
        "method": "checkerboard_plane_homography",
        "dataset": str(dataset),
        "camchain": str(camchain_file),
        "evs_camera": evs_camera,
        "rgb_camera": rgb_camera,
        "alpha": alpha,
        "fps": rate,
        "selected_pairs": len(pairs),
        "rgb_checkerboard_detections": detected_rgb,
        "dual_checkerboard_detections": detected_both,
        "rendered_frames": rendered,
        "corner_alignment_px": {
            "count": int(error_array.size),
            "mean": float(error_array.mean()) if error_array.size else None,
            "median": float(np.median(error_array)) if error_array.size else None,
            "p95": float(np.percentile(error_array, 95)) if error_array.size else None,
            "maximum": float(error_array.max()) if error_array.size else None,
        },
        "limitations": (
            "EVS is warped onto the RGB image only on the detected checkerboard "
            "plane; objects at other depths retain parallax."
        ),
    }
    write_yaml(staging / "summary.yaml", summary)
    staging.replace(output)
    return summary
