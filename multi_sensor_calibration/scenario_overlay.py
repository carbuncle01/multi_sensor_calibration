"""Render full-sequence, time-corrected RAW-event overlays on RGB frames."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from bisect import bisect_right
import tempfile
from pathlib import Path
from typing import Any

from .calibration_overlay import (
    _annotate,
    _camera,
    _polarity_images,
)


def _fov_guides(valid, np):
    """Precompute diagonal outside hatching and a six-pixel boundary band."""
    height, width = valid.shape
    yy, xx = np.indices(valid.shape)
    hatch = ~valid & ((xx + yy) % 24 < 5)
    padded = np.pad(valid, 3, mode="edge")
    inner, outer = valid.copy(), valid.copy()
    for dy in range(7):
        for dx in range(7):
            neighbor = padded[dy:dy + height, dx:dx + width]
            inner &= neighbor
            outer |= neighbor
    return hatch, outer & ~inner


def _draw_fov_guides(image, valid, guides):
    # BGR amber is distinct from red/blue event polarity colors.
    hatch, boundary = guides
    image[~valid] = (image[~valid] * 0.25).astype("uint8")
    image[hatch] = (40, 170, 230)
    image[boundary] = (0, 255, 255)


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


def _event_timeline(rgb_times, *, start_s, duration_s, step_ms, max_frames):
    """Return regular reference times and the RGB index range needed for holding.

    Stop at the last observed RGB timestamp: do not invent coverage after EOF.
    A crop between RGB frames retains the preceding frame, not a future one.
    """
    if not math.isfinite(step_ms) or step_ms < 0.001:
        raise ValueError("step_ms must be finite and at least 0.001 ms")
    if not rgb_times or any(not math.isfinite(t) for t in rgb_times):
        raise ValueError("RGB timestamps must be finite and non-empty")
    if any(b <= a for a, b in zip(rgb_times, rgb_times[1:])):
        raise ValueError("event timeline requires strictly increasing RGB timestamps")
    begin = rgb_times[0] + start_s
    last = rgb_times[-1]
    if begin > last:
        raise ValueError("no RGB coverage in the selected time range")
    step_s = step_ms / 1000.0
    span = last - begin
    if duration_s is not None:
        span = min(span, duration_s)
    count = int(math.floor(span / step_s)) + 1
    if max_frames is not None:
        count = min(count, max_frames)
    times = [begin + i * step_s for i in range(count)
             if begin + i * step_s <= last
             and (duration_s is None or i * step_s < duration_s)]
    if not times:
        raise ValueError("no frames remain in the selected time range")
    first_index = bisect_right(rgb_times, times[0]) - 1
    stop_index = bisect_right(rgb_times, times[-1])
    return times, list(range(first_index, stop_index))


def _hold_rgb_frames(rgb_frames, reference_times):
    """Stream the latest RGB frame at or before each reference timestamp."""
    frames = iter(rgb_frames)
    upcoming = next(frames, None)
    current = None
    for reference_time in reference_times:
        while upcoming is not None and upcoming[0] <= reference_time:
            current = upcoming
            upcoming = next(frames, None)
        if current is None:
            raise ValueError("no preceding RGB frame for display timestamp")
        yield current


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
    view_frame: str = "rgb",
    depth_m: float = 1.0,
    event_window_ms: float | None = None,
    event_window_position: str | None = None,
    event_dilate_px: int = 1,
    alpha: float = 0.85,
    fps: float | None = None,
    timeline: str = "rgb",
    step_ms: float = 1.0,
    start_s: float = 0.0,
    duration_s: float | None = None,
    every_n: int = 1,
    max_frames: int | None = None,
    macos_compatible: bool = True,
) -> dict[str, Any]:
    if view_frame not in {"rgb", "evs", "rgb-common"}:
        raise ValueError("view_frame must be rgb, evs or rgb-common")
    if timeline not in {"rgb", "event"}:
        raise ValueError("timeline must be rgb or event")
    if timeline == "event" and every_n != 1:
        raise ValueError("event timeline requires every_n=1 to preserve RGB updates")
    if event_window_ms is None:
        event_window_ms = 2.0 if timeline == "event" else 10.0
    if event_window_position is None:
        event_window_position = "before" if timeline == "event" else "center"
    if timeline == "event" and event_window_position != "before":
        raise ValueError("event timeline requires event_window_position=before")
    if not math.isfinite(start_s) or (duration_s is not None and not math.isfinite(duration_s)):
        raise ValueError("start_s and duration_s must be finite")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if not math.isfinite(event_window_ms) or event_window_ms < 0.001:
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

    output_size = evs["size"] if view_frame == "evs" else rgb["size"]
    rgb_to_view = np.linalg.inv(homography) if view_frame == "evs" else np.eye(3)
    event_to_view = np.eye(3) if view_frame == "evs" else homography

    origin, selected_indices, reference_times = _selected_rgb_times(
        bag,
        rgb_topic,
        rgb_timestamp_source,
        start_s=0.0 if timeline == "event" else start_s,
        duration_s=None if timeline == "event" else duration_s,
        every_n=every_n,
        max_frames=None if timeline == "event" else max_frames,
    )
    if timeline == "event":
        reference_times, selected_indices = _event_timeline(
            reference_times, start_s=start_s, duration_s=duration_s,
            step_ms=step_ms, max_frames=max_frames,
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

    if timeline == "event":
        rgb_frames = _hold_rgb_frames(rgb_frames, reference_times)

    if fps is None and timeline == "event":
        rate = 60.0
    elif fps is None:
        differences = np.diff(np.asarray(reference_times, dtype=float))
        rate = float(1.0 / np.median(differences)) if len(differences) else 60.0
    else:
        rate = float(fps)
    if not math.isfinite(rate) or rate <= 0.0:
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
        str(staging / "overlay_polarity.mp4"), fourcc, rate, output_size
    )
    event_writer = cv2.VideoWriter(
        str(staging / "polarity_only.mp4"), fourcc, rate, output_size
    )
    comparison_size = (output_size[0] * 2, output_size[1])
    comparison_writer = cv2.VideoWriter(
        str(staging / "rgb_vs_overlay.mp4"), fourcc, rate, comparison_size
    )
    writers = (overlay_writer, event_writer, comparison_writer)
    if not all(writer.isOpened() for writer in writers):
        for writer in writers:
            writer.release()
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError("OpenCV could not open the scenario MP4 writers")

    try:
        map_evs = cv2.initUndistortRectifyMap(
            evs["matrix"], evs["distortion"], None, evs["matrix"], evs["size"], cv2.CV_32FC1
        )
        map_rgb = cv2.initUndistortRectifyMap(
            rgb["matrix"], rgb["distortion"], None, rgb["matrix"], rgb["size"], cv2.CV_32FC1
        )
        common_valid = None
        if view_frame != "rgb":
            # Linear sampling support must be valid in both sensors (including undistortion).
            supports = []
            for camera, maps, transform_view in ((rgb, map_rgb, rgb_to_view), (evs, map_evs, event_to_view)):
                valid = np.ones((camera["size"][1], camera["size"][0]), dtype=np.float32)
                valid = cv2.remap(valid, maps[0], maps[1], cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                valid = cv2.warpPerspective(valid, transform_view, output_size,
                                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                supports.append(valid >= 0.999)
            common_valid = supports[0] & supports[1]
            if not common_valid.any():
                raise ValueError("calibration yields no common field of view")
            if not cv2.imwrite(str(staging / "common_valid_mask.png"), common_valid.astype("uint8") * 255):
                raise RuntimeError("could not write common field-of-view mask")

        fov_guides = _fov_guides(common_valid, np) if view_frame == "rgb-common" else None

        snapshot_indices = {
            round(index * (len(reference_times) - 1) / min(11, len(reference_times) - 1))
            for index in range(min(12, len(reference_times)))
        } if len(reference_times) > 1 else {0}

        rendered = 0
        slowdown = 1000.0 / (step_ms * rate) if timeline == "event" else None
        manifest = (staging / "frames.csv").open("w", newline="", encoding="utf-8")
        frame_csv = csv.writer(manifest)
        frame_csv.writerow([
            "frame", "video_time_s", "reference_time_s", "relative_time_s",
            "rgb_time_s", "rgb_age_ms", "event_start_us", "event_end_us", "event_count",
        ])
    except Exception:
        for writer in writers:
            writer.release()
        shutil.rmtree(staging, ignore_errors=True)
        raise

    cached_rgb_time = None
    try:
        try:
            for index, (reference_time, (rgb_time, rgb_image), event_frame) in enumerate(
                zip(reference_times, rgb_frames, event_frames)
            ):
                if (rgb_image.shape[1], rgb_image.shape[0]) != rgb["size"]:
                    raise ValueError("RGB frame dimensions do not match the calibration")
                polarity = event_frame.image
                if (polarity.shape[1], polarity.shape[0]) != evs["size"]:
                    raise ValueError("EVS RAW dimensions do not match the calibration")
                if cached_rgb_time != rgb_time:
                    rgb_undistorted = cv2.remap(
                        rgb_image, map_rgb[0], map_rgb[1], cv2.INTER_LINEAR
                    )
                    if view_frame == "evs":
                        rgb_undistorted = cv2.warpPerspective(rgb_undistorted, rgb_to_view, output_size)
                        rgb_undistorted[~common_valid] = (35, 35, 35)

                    cached_rgb_time = rgb_time
                event_only, event_mask = _polarity_images(
                    polarity,
                    map_evs,
                    event_to_view,
                    output_size,
                    event_dilate_px,
                    cv2,
                    np,
                )
                if common_valid is not None:
                    event_mask &= common_valid
                    event_only[~common_valid] = (35, 35, 35)
                overlay = _overlay_rgb(
                    rgb_undistorted, event_only, event_mask, alpha, np
                )
                effective_offset_ms = (
                    reference_time - clock.inverse(reference_time)
                ) * 1000.0
                relative_s = reference_time - origin
                rgb_age_ms = (reference_time - rgb_time) * 1000.0
                detail = (
                    f"{projection}  dt={effective_offset_ms:+.2f}ms  "
                    f"events={event_window_ms:g}ms  t={relative_s:.3f}s"
                )
                rgb_labeled = rgb_undistorted.copy()
                rgb_label = f"RGB  t={rgb_time - origin:.3f}s"
                if timeline == "event":
                    rgb_label += f"  held={rgb_age_ms:.1f}ms"
                    detail = f"EVS t={relative_s:.3f}s  past {event_window_ms:g}ms  {slowdown:.2f}x slow"
                if view_frame == "rgb-common":
                    for image in (rgb_labeled, overlay, event_only):
                        _draw_fov_guides(image, common_valid, fov_guides)
                    rgb_label += "  YELLOW: FOV boundary / HATCH: outside"
                _annotate(rgb_labeled, rgb_label, cv2)
                _annotate(overlay, detail, cv2)
                _annotate(event_only, detail, cv2)
                comparison = np.concatenate((rgb_labeled, overlay), axis=1)
                overlay_writer.write(overlay)
                event_writer.write(event_only)
                comparison_writer.write(comparison)
                if index in snapshot_indices:
                    stamp = int(round(reference_time * 1_000_000_000.0))
                    cv2.imwrite(str(snapshots / f"{stamp}_overlay.png"), overlay)
                    cv2.imwrite(str(snapshots / f"{stamp}_comparison.png"), comparison)
                frame_csv.writerow([
                    index, index / rate, reference_time, relative_s, rgb_time, rgb_age_ms,
                    event_frame.window.start_us, event_frame.window.end_us,
                    event_frame.window.event_count,
                ])
                rendered += 1
        finally:
            manifest.close()
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
        "view_frame": view_frame,
        "fov_visualization": "yellow-boundary-amber-hatch" if view_frame == "rgb-common" else None,
        "output_size": list(output_size),
        "common_valid_mask": "common_valid_mask.png" if view_frame != "rgb" else None,
        "camchain_sha256": hashlib.sha256(camchain_file.read_bytes()).hexdigest(),
        "time_sync_sha256": hashlib.sha256(sync_file.read_bytes()).hexdigest(),
        "reference_origin_s": origin,
        "rgb_to_view_homography": rgb_to_view.tolist(),
        "event_to_view_homography": event_to_view.tolist(),
        "depth_m": float(depth_m) if projection == "fixed-depth" else None,
        "event_window_ms": float(event_window_ms),
        "event_window_position": event_window_position,
        "event_dilate_px": int(event_dilate_px),
        "alpha": float(alpha),
        "fps": rate,
        "timeline": timeline,
        "step_ms": step_ms if timeline == "event" else None,
        "slowdown_factor": slowdown,
        "rgb_timestamp_source": rgb_timestamp_source,
        "rgb_display_policy": "latest_at_or_before" if timeline == "event" else "rgb_frame",
        "reference_start_s": reference_times[0],
        "reference_end_s": reference_times[rendered - 1],
        "truncated": rendered < len(reference_times),
        "timing_note": "Offline timestamp visualization; not a measurement of sensor delivery or inference latency.",
        "frame_manifest": "frames.csv",
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
    (staging / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    staging.replace(output)
    return summary
