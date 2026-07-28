"""Export synchronized mono8 image datasets for Kalibr."""

from __future__ import annotations

import csv
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .imaging import decode_ros_image, to_uint8
from .models import ClockEstimate
from .rosbag import iter_messages, selected_time_ns


SUPPORTED_CAMERA_MODELS = {
    "pinhole-radtan",
    "pinhole-equi",
    "pinhole-fov",
    "omni-none",
    "omni-radtan",
    "eucm-none",
    "ds-none",
}


@dataclass(frozen=True)
class CameraExport:
    sensor: str
    camera: str
    model: str
    topic: str


@dataclass(frozen=True)
class FrameMetadata:
    index: int
    sensor_time_s: float
    corrected_time_s: float
    source: str


@dataclass(frozen=True)
class MatchedFrame:
    target_time_s: float
    frame: FrameMetadata

    @property
    def delta_s(self) -> float:
        return self.frame.corrected_time_s - self.target_time_s


def seconds_to_nanoseconds(value: float) -> int:
    return int(round(value * 1_000_000_000.0))


def parse_camera_exports(config: dict[str, Any]) -> list[CameraExport]:
    kalibr = config.get("kalibr")
    if not isinstance(kalibr, dict):
        raise ValueError("kalibr configuration is required")
    cameras = kalibr.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("kalibr.cameras must be a non-empty list")

    result: list[CameraExport] = []
    seen_sensors: set[str] = set()
    for index, value in enumerate(cameras):
        if not isinstance(value, dict):
            raise ValueError(f"kalibr.cameras[{index}] must be a mapping")
        sensor = str(value.get("sensor", "")).strip()
        if not sensor:
            raise ValueError(f"kalibr.cameras[{index}].sensor is required")
        if sensor in seen_sensors:
            raise ValueError(f"sensor {sensor!r} appears more than once in kalibr.cameras")
        model = str(value.get("model", "pinhole-radtan"))
        if model not in SUPPORTED_CAMERA_MODELS:
            raise ValueError(f"unsupported Kalibr camera model: {model}")
        result.append(
            CameraExport(
                sensor=sensor,
                camera=f"cam{index}",
                model=model,
                topic=f"/cam{index}/image_raw",
            )
        )
        seen_sensors.add(sensor)
    return result


def select_reference_frames(
    frames: Sequence[FrameMetadata],
    max_rate_hz: float,
    max_frames: int | None = None,
) -> list[FrameMetadata]:
    if max_rate_hz <= 0.0:
        raise ValueError("max_rate_hz must be positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")

    minimum_period_s = 1.0 / max_rate_hz
    selected: list[FrameMetadata] = []
    last_time_s: float | None = None
    for frame in frames:
        if last_time_s is not None and frame.corrected_time_s <= last_time_s:
            if frame.corrected_time_s == last_time_s:
                continue
            raise ValueError("corrected frame timestamps must be increasing")
        if (
            not selected
            or frame.corrected_time_s - selected[-1].corrected_time_s
            >= minimum_period_s - 1e-12
        ):
            selected.append(frame)
            if max_frames is not None and len(selected) >= max_frames:
                break
        last_time_s = frame.corrected_time_s
    return selected


def match_frames(
    targets: Sequence[FrameMetadata],
    candidates: Sequence[FrameMetadata],
    max_delta_s: float,
) -> dict[int, MatchedFrame]:
    """Match each target to one chronological candidate without reuse."""

    if max_delta_s < 0.0:
        raise ValueError("max_delta_s must be non-negative")
    result: dict[int, MatchedFrame] = {}
    candidate_index = 0
    last_used_index = -1

    for target in targets:
        while (
            candidate_index < len(candidates)
            and candidates[candidate_index].corrected_time_s
            < target.corrected_time_s
        ):
            candidate_index += 1

        possible_indices = []
        if candidate_index < len(candidates):
            possible_indices.append(candidate_index)
        if candidate_index > 0:
            possible_indices.append(candidate_index - 1)
        possible_indices = [
            index for index in possible_indices if index > last_used_index
        ]
        if not possible_indices:
            continue

        closest_index = min(
            possible_indices,
            key=lambda index: (
                abs(
                    candidates[index].corrected_time_s
                    - target.corrected_time_s
                ),
                index,
            ),
        )
        closest = candidates[closest_index]
        if abs(closest.corrected_time_s - target.corrected_time_s) > max_delta_s:
            continue
        result[target.index] = MatchedFrame(
            target_time_s=target.corrected_time_s,
            frame=closest,
        )
        last_used_index = closest_index
    return result


def common_matches(
    reference_frames: Sequence[FrameMetadata],
    frames_by_sensor: dict[str, Sequence[FrameMetadata]],
    reference_sensor: str,
    max_delta_s: float,
) -> tuple[list[FrameMetadata], dict[str, dict[int, MatchedFrame]]]:
    matches: dict[str, dict[int, MatchedFrame]] = {}
    for sensor, frames in frames_by_sensor.items():
        if sensor == reference_sensor:
            matches[sensor] = {
                target.index: MatchedFrame(target.corrected_time_s, target)
                for target in reference_frames
            }
        else:
            matches[sensor] = match_frames(reference_frames, frames, max_delta_s)

    common_target_indices = {
        target.index for target in reference_frames
    }
    for sensor_matches in matches.values():
        common_target_indices.intersection_update(sensor_matches)
    common_reference = [
        target for target in reference_frames if target.index in common_target_indices
    ]
    return common_reference, matches


def bag_frame_metadata(
    bag_path: str | Path,
    topic: str,
    *,
    timestamp_source: str,
    clock: ClockEstimate,
) -> list[FrameMetadata]:
    result = []
    previous_corrected_s: float | None = None
    for index, item in enumerate(iter_messages(bag_path, {topic})):
        sensor_time_s = selected_time_ns(item, timestamp_source) / 1_000_000_000.0
        corrected_time_s = clock.apply(sensor_time_s)
        if (
            previous_corrected_s is not None
            and corrected_time_s < previous_corrected_s
        ):
            raise ValueError(f"corrected timestamps are not monotonic for {topic}")
        result.append(
            FrameMetadata(
                index=index,
                sensor_time_s=sensor_time_s,
                corrected_time_s=corrected_time_s,
                source=f"{bag_path}:{topic}:{index}",
            )
        )
        previous_corrected_s = corrected_time_s
    return result


def generated_frame_metadata(
    directory: str | Path,
    *,
    clock: ClockEstimate,
) -> list[FrameMetadata]:
    root = Path(directory)
    manifest = root / "frames.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"generated image manifest not found: {manifest}")

    result = []
    previous_corrected_s: float | None = None
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            timestamp = row.get("reference_timestamp_s", "")
            if not timestamp:
                raise ValueError(
                    f"{manifest} has no reference timestamp for row {index + 2}"
                )
            sensor_time_s = float(timestamp)
            corrected_time_s = clock.apply(sensor_time_s)
            image_path = root / row["image"]
            if not image_path.is_file():
                raise FileNotFoundError(f"generated image not found: {image_path}")
            if (
                previous_corrected_s is not None
                and corrected_time_s < previous_corrected_s
            ):
                raise ValueError(
                    f"corrected timestamps are not monotonic in {manifest}"
                )
            result.append(
                FrameMetadata(
                    index=index,
                    sensor_time_s=sensor_time_s,
                    corrected_time_s=corrected_time_s,
                    source=str(image_path.resolve()),
                )
            )
            previous_corrected_s = corrected_time_s
    return result


def _selected_bag_images(
    bag_path: str | Path,
    topic: str,
    selected_indices: set[int],
) -> Iterator[tuple[int, Any]]:
    for index, item in enumerate(iter_messages(bag_path, {topic})):
        if index in selected_indices:
            yield index, decode_ros_image(item.message, item.message_type)


def _selected_generated_images(
    frames: Iterable[FrameMetadata],
) -> Iterator[tuple[int, Any]]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to export generated images") from exc

    for frame in frames:
        image = cv2.imread(frame.source, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"failed to read {frame.source}")
        yield frame.index, image


def _write_camera_images(
    destination: Path,
    camera: CameraExport,
    selected: Sequence[MatchedFrame],
    *,
    bag_path: str | Path | None,
    image_directory: str | Path | None,
    sensor_config: dict[str, Any],
) -> list[dict[str, Any]]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to export Kalibr images") from exc

    camera_dir = destination / camera.camera
    camera_dir.mkdir(parents=True)
    by_index = {match.frame.index: match for match in selected}
    if image_directory is not None:
        source_images = _selected_generated_images(
            match.frame for match in selected
        )
        source_kind = "generated"
    else:
        if bag_path is None:
            raise ValueError(f"bag input is required for sensor {camera.sensor}")
        topic = sensor_config.get("image_topic")
        if not topic:
            raise ValueError(f"sensors.{camera.sensor}.image_topic is required")
        source_images = _selected_bag_images(
            bag_path, str(topic), set(by_index)
        )
        source_kind = "rosbag"

    rows = []
    seen_timestamps: set[int] = set()
    expected_shape = None
    for index, image in source_images:
        match = by_index[index]
        timestamp_ns = seconds_to_nanoseconds(match.frame.corrected_time_s)
        if timestamp_ns in seen_timestamps:
            raise ValueError(
                f"duplicate corrected timestamp for {camera.sensor}: {timestamp_ns}"
            )
        seen_timestamps.add(timestamp_ns)
        mono8 = image if getattr(image, "dtype", None) == "uint8" else to_uint8(image)
        if mono8.ndim != 2:
            raise ValueError(f"{camera.sensor} image is not single-channel")
        if expected_shape is None:
            expected_shape = tuple(mono8.shape)
        elif tuple(mono8.shape) != expected_shape:
            raise ValueError(f"{camera.sensor} image dimensions changed during export")

        relative = Path(camera.camera) / f"{timestamp_ns}.png"
        if not cv2.imwrite(str(destination / relative), mono8):
            raise RuntimeError(f"failed to write {destination / relative}")
        rows.append(
            {
                "camera": camera.camera,
                "sensor": camera.sensor,
                "topic": camera.topic,
                "model": camera.model,
                "source_kind": source_kind,
                "source_index": index,
                "input_timestamp_ns": seconds_to_nanoseconds(
                    match.frame.sensor_time_s
                ),
                "corrected_timestamp_ns": timestamp_ns,
                "clock_correction_ns": timestamp_ns
                - seconds_to_nanoseconds(match.frame.sensor_time_s),
                "target_reference_timestamp_ns": seconds_to_nanoseconds(
                    match.target_time_s
                ),
                "delta_to_target_ns": seconds_to_nanoseconds(match.delta_s),
                "image": relative.as_posix(),
                "source": match.frame.source,
            }
        )

    if len(rows) != len(selected):
        raise RuntimeError(
            f"exported {len(rows)} of {len(selected)} selected images for {camera.sensor}"
        )
    return rows


def export_dataset(
    output_dir: str | Path,
    cameras: Sequence[CameraExport],
    reference_frames: Sequence[FrameMetadata],
    matches: dict[str, dict[int, MatchedFrame]],
    *,
    reference_sensor: str,
    bag_path: str | Path | None,
    image_directories: dict[str, str | Path],
    sensor_configs: dict[str, dict[str, Any]],
    target: dict[str, Any],
    approximate_sync_s: float,
    export_rate_hz: float,
    time_sync_source: str | Path,
) -> dict[str, Any]:
    from .io import write_yaml

    output = Path(output_dir).resolve()
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f"output directory is not empty: {output}")
        output.rmdir()
    output.parent.mkdir(parents=True, exist_ok=True)

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    try:
        manifest_rows = []
        camera_summaries = []
        for camera in cameras:
            selected = [
                matches[camera.sensor][target_frame.index]
                for target_frame in reference_frames
            ]
            rows = _write_camera_images(
                staging,
                camera,
                selected,
                bag_path=bag_path,
                image_directory=image_directories.get(camera.sensor),
                sensor_config=sensor_configs[camera.sensor],
            )
            manifest_rows.extend(rows)
            camera_summaries.append(
                {
                    "camera": camera.camera,
                    "sensor": camera.sensor,
                    "topic": camera.topic,
                    "model": camera.model,
                    "frames": len(rows),
                }
            )

        if not manifest_rows:
            raise ValueError("no synchronized images were selected for export")

        manifest_path = staging / "manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(manifest_rows[0]),
            )
            writer.writeheader()
            writer.writerows(manifest_rows)

        target_yaml = {
            "target_type": "checkerboard",
            "targetCols": int(target["columns"]),
            "targetRows": int(target["rows"]),
            "rowSpacingMeters": float(target["square_size_m"]),
            "colSpacingMeters": float(target["square_size_m"]),
        }
        write_yaml(staging / "target.yaml", target_yaml)
        time_sync_path = Path(time_sync_source).resolve()
        if not time_sync_path.is_file():
            raise FileNotFoundError(f"time sync result not found: {time_sync_path}")
        shutil.copyfile(time_sync_path, staging / "time_sync.yaml")
        job = {
            "schema_version": 1,
            "reference_sensor": reference_sensor,
            "cameras": camera_summaries,
            "target": "target.yaml",
            "manifest": "manifest.csv",
            "time_sync": "time_sync.yaml",
            "approximate_sync_s": float(approximate_sync_s),
            "export_rate_hz": float(export_rate_hz),
            "time_sync_source": str(time_sync_path),
            "timestamp_contract": (
                "PNG filename is the sensor timestamp mapped once onto the "
                "reference clock, expressed in integer nanoseconds."
            ),
        }
        write_yaml(staging / "job.yaml", job)
        staging.replace(output)
        return job
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
