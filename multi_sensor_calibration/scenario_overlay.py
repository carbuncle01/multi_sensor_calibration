"""Render full-sequence, time-corrected RAW-event overlays on RGB frames."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from .calibration_overlay import (
    _annotate,
    _camera,
    _polarity_images,
)


def _projection_homography(evs, rgb, transform, mode: str, depth_m: float, np):
    rotation = transform[:3, :3]
    if mode == "rotation-only":
        projective = rotation
    elif mode == "fixed-depth":
        if depth_m <= 0.0:
            raise ValueError("depth_m must be positive for fixed-depth projection")
        normal = np.array([[0.0, 0.0, 1.0]], dtype=float)
        projective = rotation + transform[:3, 3:4] @ normal / depth_m
    else:
        raise ValueError("projection must be rotation-only or fixed-depth")
    return rgb["matrix"] @ projective @ np.linalg.inv(evs["matrix"])


def _rgb_frames(bag_path, topic, timestamp_source, selected_indices):
    from .imaging import decode_ros_image_bgr
    from .rosbag import iter_messages, selected_time_ns

    wanted = iter(selected_indices)
    target = next(wanted, None)
    for index, item in enumerate(iter_messages(bag_path, {topic})):
        if target is None:
            return
        if index != target:
            continue
        yield (
            selected_time_ns(item, timestamp_source) / 1_000_000_000.0,
            decode_ros_image_bgr(item.message, item.message_type),
        )
        target = next(wanted, None)


def _selected_rgb_times(
    bag_path,
    topic,
    timestamp_source,
    *,
    start_s,
    duration_s,
    every_n,
    max_frames,
):
    from .rosbag import iter_messages, selected_time_ns

    all_times = [
        selected_time_ns(item, timestamp_source) / 1_000_000_000.0
        for item in iter_messages(bag_path, {topic})
    ]
    if not all_times:
        raise ValueError(f"RGB topic contains no frames: {topic}")
    origin = all_times[0]
    begin = origin + start_s
    end = None if duration_s is None else begin + duration_s
    indices = []
    times = []
    accepted = 0
    for index, timestamp in enumerate(all_times):
        if timestamp < begin:
            continue
        if end is not None and timestamp >= end:
            break
        if accepted % every_n:
            accepted += 1
            continue
        indices.append(index)
        times.append(timestamp)
        accepted += 1
        if max_frames is not None and len(indices) >= max_frames:
            break
    if not times:
        raise ValueError("no RGB frames remain in the selected time range")
    return origin, indices, times


def _event_frames(event_file, time_sync, reference_times, window_ms, position):
    from .event_windows import WindowDefinition, reference_aligned_window_ends
    from .evs_pipeline import generate_event_frames
    from .evs_sources import MetavisionFileSource
    from .io import load_yaml
    from .models import ClockEstimate

    sync = load_yaml(time_sync)
    models = sync.get("models")
    if not isinstance(models, dict) or not isinstance(models.get("evs"), dict):
        raise ValueError("time-sync result has no evs clock model")
    clock = ClockEstimate.from_dict(models["evs"])
    accumulation_us = int(round(window_ms * 1000.0))
    policy = {"before": "end", "center": "center", "after": "start"}[position]
    definition = WindowDefinition(
        accumulation_us=accumulation_us,
        timestamp_policy=policy,
        schedule="reference_aligned",
        drop_partial_windows=False,
    )
    source = MetavisionFileSource(
        event_file,
        chunk_us=max(1_000, min(10_000, accumulation_us)),
    )
    if source.anchor is None:
        raise ValueError("EVS RAW sidecar metadata is required")

    def reference_to_event_time(reference_time_s: float) -> float:
        provisional_s = clock.inverse(reference_time_s)
        return source.anchor.to_source_us(provisional_s) / 1_000_000.0

    ends = reference_aligned_window_ends(
        reference_times,
        reference_to_event_time=reference_to_event_time,
        definition=definition,
    )
    frames = generate_event_frames(
        source,
        definition,
        end_times_us=ends,
        representation="polarity",
    )
    return frames, clock


def _overlay_rgb(rgb, event_only, event_mask, alpha, np):
    result = rgb.copy()
    result[event_mask] = np.clip(
        (1.0 - alpha) * result[event_mask] + alpha * event_only[event_mask],
        0,
        255,
    ).astype("uint8")
    return result


def render_scenario_overlay(
    bag_path: str | Path,
    event_file: str | Path,
    time_sync_path: str | Path,
    camchain_path: str | Path,
    output_dir: str | Path,
    *,
    rgb_topic: str = "/realsense/color/image_raw",
    rgb_timestamp_source: str = "bag",
    evs_camera: str = "cam0",
    rgb_camera: str = "cam1",
    projection: str = "rotation-only",
    depth_m: float = 1.0,
    event_window_ms: float = 10.0,
    event_window_position: str = "center",
    event_dilate_px: int = 1,
    alpha: float = 0.85,
    fps: float | None = None,
    start_s: float = 0.0,
    duration_s: float | None = None,
    every_n: int = 1,
    max_frames: int | None = None,
    macos_compatible: bool = True,
) -> dict[str, Any]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if event_window_ms <= 0.0:
        raise ValueError("event_window_ms must be positive")
    if event_window_position not in {"before", "center", "after"}:
        raise ValueError("event_window_position must be before, center, or after")
    if event_dilate_px <= 0 or every_n <= 0:
        raise ValueError("event_dilate_px and every_n must be positive")
    if start_s < 0.0 or (duration_s is not None and duration_s <= 0.0):
        raise ValueError("start_s must be non-negative and duration_s positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("OpenCV and NumPy are required for overlay rendering") from exc

    from .io import load_yaml, write_yaml

    bag = Path(bag_path).resolve()
    raw = Path(event_file).resolve()
    sync_file = Path(time_sync_path).resolve()
    camchain_file = Path(camchain_path).resolve()
    for path, label in (
        (bag, "bag"),
        (raw, "event file"),
        (sync_file, "time-sync result"),
        (camchain_file, "camera chain"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    camchain = load_yaml(camchain_file)
    evs = _camera(camchain, evs_camera, np)
    rgb = _camera(camchain, rgb_camera, np)
    transform = np.asarray(
        camchain.get(rgb_camera, {}).get("T_cn_cnm1"), dtype=float
    )
    if transform.shape != (4, 4):
        raise ValueError(f"{rgb_camera}.T_cn_cnm1 must be a 4x4 matrix")
    homography = _projection_homography(
        evs, rgb, transform, projection, depth_m, np
    )

    origin, selected_indices, reference_times = _selected_rgb_times(
        bag,
        rgb_topic,
        rgb_timestamp_source,
        start_s=start_s,
        duration_s=duration_s,
        every_n=every_n,
        max_frames=max_frames,
    )
    event_frames, clock = _event_frames(
        raw,
        sync_file,
        reference_times,
        event_window_ms,
        event_window_position,
    )
    rgb_frames = _rgb_frames(
        bag, rgb_topic, rgb_timestamp_source, selected_indices
    )

    if fps is None:
        differences = np.diff(np.asarray(reference_times, dtype=float))
        rate = float(1.0 / np.median(differences)) if len(differences) else 60.0
    else:
        rate = float(fps)
    if rate <= 0.0:
        raise ValueError("fps must be positive")

    output = Path(output_dir).resolve()
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f"output directory is not empty: {output}")
        output.rmdir()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    snapshots = staging / "snapshots"
    snapshots.mkdir()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    overlay_writer = cv2.VideoWriter(
        str(staging / "overlay_polarity.mp4"), fourcc, rate, rgb["size"]
    )
    event_writer = cv2.VideoWriter(
        str(staging / "polarity_only.mp4"), fourcc, rate, rgb["size"]
    )
    comparison_size = (rgb["size"][0] * 2, rgb["size"][1])
    comparison_writer = cv2.VideoWriter(
        str(staging / "rgb_vs_overlay.mp4"), fourcc, rate, comparison_size
    )
    writers = (overlay_writer, event_writer, comparison_writer)
    if not all(writer.isOpened() for writer in writers):
        for writer in writers:
            writer.release()
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError("OpenCV could not open the scenario MP4 writers")

    map_evs = cv2.initUndistortRectifyMap(
        evs["matrix"], evs["distortion"], None, evs["matrix"], evs["size"], cv2.CV_32FC1
    )
    map_rgb = cv2.initUndistortRectifyMap(
        rgb["matrix"], rgb["distortion"], None, rgb["matrix"], rgb["size"], cv2.CV_32FC1
    )
    snapshot_indices = {
        round(index * (len(reference_times) - 1) / min(11, len(reference_times) - 1))
        for index in range(min(12, len(reference_times)))
    } if len(reference_times) > 1 else {0}

    rendered = 0
    try:
        try:
            for index, ((rgb_time, rgb_image), event_frame) in enumerate(
                zip(rgb_frames, event_frames)
            ):
                if (rgb_image.shape[1], rgb_image.shape[0]) != rgb["size"]:
                    raise ValueError("RGB frame dimensions do not match the calibration")
                polarity = event_frame.image
                if (polarity.shape[1], polarity.shape[0]) != evs["size"]:
                    raise ValueError("EVS RAW dimensions do not match the calibration")
                rgb_undistorted = cv2.remap(
                    rgb_image, map_rgb[0], map_rgb[1], cv2.INTER_LINEAR
                )
                event_only, event_mask = _polarity_images(
                    polarity,
                    map_evs,
                    homography,
                    rgb["size"],
                    event_dilate_px,
                    cv2,
                    np,
                )
                overlay = _overlay_rgb(
                    rgb_undistorted, event_only, event_mask, alpha, np
                )
                effective_offset_ms = (
                    rgb_time - clock.inverse(rgb_time)
                ) * 1000.0
                relative_s = rgb_time - origin
                detail = (
                    f"{projection}  dt={effective_offset_ms:+.2f}ms  "
                    f"events={event_window_ms:g}ms  t={relative_s:.3f}s"
                )
                rgb_labeled = rgb_undistorted.copy()
                _annotate(rgb_labeled, f"RGB  t={relative_s:.3f}s", cv2)
                _annotate(overlay, detail, cv2)
                _annotate(event_only, detail, cv2)
                comparison = np.concatenate((rgb_labeled, overlay), axis=1)
                overlay_writer.write(overlay)
                event_writer.write(event_only)
                comparison_writer.write(comparison)
                if index in snapshot_indices:
                    stamp = int(round(rgb_time * 1_000_000_000.0))
                    cv2.imwrite(str(snapshots / f"{stamp}_overlay.png"), overlay)
                    cv2.imwrite(str(snapshots / f"{stamp}_comparison.png"), comparison)
                rendered += 1
        finally:
            for writer in writers:
                writer.release()
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if rendered == 0:
        shutil.rmtree(staging, ignore_errors=True)
        raise ValueError("no synchronized RGB/EVS frames were rendered")
    video_encoding: dict[str, str] = {
        "codec": "mpeg4-part2",
        "encoder": "opencv-mp4v",
        "container": "mp4",
    }
    if macos_compatible:
        from .video_compat import make_macos_compatible_mp4

        try:
            encodings = [
                make_macos_compatible_mp4(staging / name)
                for name in (
                    "overlay_polarity.mp4",
                    "polarity_only.mp4",
                    "rgb_vs_overlay.mp4",
                )
            ]
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        video_encoding = encodings[0]
    summary = {
        "schema_version": 1,
        "bag": str(bag),
        "event_file": str(raw),
        "time_sync": str(sync_file),
        "camchain": str(camchain_file),
        "projection": projection,
        "depth_m": float(depth_m) if projection == "fixed-depth" else None,
        "event_window_ms": float(event_window_ms),
        "event_window_position": event_window_position,
        "event_dilate_px": int(event_dilate_px),
        "alpha": float(alpha),
        "fps": rate,
        "requested_frames": len(reference_times),
        "rendered_frames": rendered,
        "duration_s": rendered / rate,
        "video_encoding": video_encoding,
        "outputs": [
            "overlay_polarity.mp4",
            "polarity_only.mp4",
            "rgb_vs_overlay.mp4",
        ],
        "limitations": (
            "rotation-only ignores translation and therefore retains parallax"
            if projection == "rotation-only"
            else "fixed-depth projection is exact only on the declared fronto-parallel EVS-depth plane"
        ),
    }
    write_yaml(staging / "summary.yaml", summary)
    staging.replace(output)
    return summary
