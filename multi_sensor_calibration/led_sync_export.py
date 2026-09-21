"""Export LED ROI activity signals for the browser synchronization inspector."""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .imaging import decode_ros_image_intensity
from .rosbag import iter_messages, selected_time_ns


@dataclass(frozen=True)
class Roi:
    x: int
    y: int
    width: int
    height: int

    @classmethod
    def parse(cls, value: str) -> "Roi":
        try:
            parts = [int(part.strip()) for part in value.split(",")]
        except ValueError as exc:
            raise ValueError("ROI must contain integer x,y,width,height values") from exc
        if len(parts) != 4:
            raise ValueError("ROI must use x,y,width,height format")
        roi = cls(*parts)
        if roi.x < 0 or roi.y < 0 or roi.width <= 0 or roi.height <= 0:
            raise ValueError("ROI x/y must be non-negative and width/height positive")
        return roi

    def validate(self, image_width: int, image_height: int, sensor: str) -> None:
        if self.x + self.width > image_width or self.y + self.height > image_height:
            raise ValueError(
                f"{sensor} ROI {self.x},{self.y},{self.width},{self.height} "
                f"exceeds image size {image_width}x{image_height}"
            )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def _estimate_rate_hz(timestamps_s: list[float]) -> float | None:
    deltas = [
        after - before
        for before, after in zip(timestamps_s, timestamps_s[1:])
        if after > before
    ]
    if not deltas:
        return None
    median_delta = statistics.median(deltas)
    return 1.0 / median_delta if median_delta > 0.0 else None


def _rgb_samples(
    bag_path: str | Path,
    topic: str,
    timestamp_source: str,
    roi: Roi,
    max_frames: int | None,
) -> tuple[list[tuple[float, float]], tuple[int, int]]:
    samples: list[tuple[float, float]] = []
    image_size: tuple[int, int] | None = None
    for item in iter_messages(bag_path, {topic}):
        image = decode_ros_image_intensity(item.message, item.message_type)
        height, width = image.shape[:2]
        if image_size is None:
            roi.validate(width, height, "RGB")
            image_size = (width, height)
        elif image_size != (width, height):
            raise ValueError("RGB image size changed during the recording")
        crop = image[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
        samples.append(
            (
                selected_time_ns(item, timestamp_source) / 1_000_000_000.0,
                float(crop.mean()),
            )
        )
        if max_frames is not None and len(samples) >= max_frames:
            break
    if not samples or image_size is None:
        raise ValueError(f"RGB topic {topic} produced no images")
    return samples, image_size


def _evs_bin_counts(
    source,
    roi: Roi,
    origin_s: float,
    bin_width_s: float,
) -> tuple[dict[int, list[int]], tuple[int, int], int]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for LED synchronization export") from exc

    if source.anchor is None:
        raise ValueError(
            "EVS input has no reference clock anchor; keep the RAW metadata sidecar "
            "beside the recording or configure an explicit anchor"
        )

    counts: dict[int, list[int]] = {}
    image_size: tuple[int, int] | None = None
    selected_event_count = 0
    for batch in source.batches():
        events = batch.events
        if image_size is None:
            if source.width <= 0 or source.height <= 0:
                raise ValueError("EVS source did not report a valid image size")
            roi.validate(source.width, source.height, "EVS")
            image_size = (source.width, source.height)
        inside = (
            (events["x"] >= roi.x)
            & (events["x"] < roi.x + roi.width)
            & (events["y"] >= roi.y)
            & (events["y"] < roi.y + roi.height)
        )
        selected = events[inside]
        if len(selected) == 0:
            continue
        selected_event_count += len(selected)
        reference_times_s = (
            source.anchor.reference_time_s
            + source.anchor.scale
            * (selected["t"].astype(np.float64) - source.anchor.source_time_us)
            / 1_000_000.0
        )
        indices = np.floor((reference_times_s - origin_s) / bin_width_s).astype(
            np.int64
        )
        for polarity_index, polarity_mask in (
            (0, selected["p"] > 0),
            (1, selected["p"] <= 0),
        ):
            unique, frequencies = np.unique(indices[polarity_mask], return_counts=True)
            for index, frequency in zip(unique.tolist(), frequencies.tolist()):
                value = counts.setdefault(int(index), [0, 0])
                value[polarity_index] += int(frequency)

    if image_size is None:
        raise ValueError("EVS input produced no event batches")
    return counts, image_size, selected_event_count


def export_led_sync_data(
    *,
    bag_path: str | Path,
    event_source,
    output_dir: str | Path,
    rgb_topic: str,
    rgb_timestamp_source: str,
    rgb_roi: Roi,
    evs_roi: Roi,
    bin_ms: float = 1.0,
    session_name: str | None = None,
    max_rgb_frames: int | None = None,
) -> dict[str, Any]:
    if bin_ms <= 0.0:
        raise ValueError("--bin-ms must be positive")
    if event_source.anchor is None:
        raise ValueError(
            "EVS source has no clock anchor; check the RAW metadata sidecar"
        )

    rgb_absolute, rgb_image_size = _rgb_samples(
        bag_path,
        rgb_topic,
        rgb_timestamp_source,
        rgb_roi,
        max_rgb_frames,
    )
    origin_s = min(rgb_absolute[0][0], event_source.anchor.reference_time_s)
    bin_width_s = bin_ms / 1000.0
    counts, evs_image_size, selected_event_count = _evs_bin_counts(
        event_source,
        evs_roi,
        origin_s,
        bin_width_s,
    )
    if not counts:
        raise ValueError("the EVS ROI contains no events")

    rgb_rows = [
        {"t": f"{timestamp_s - origin_s:.9f}", "rgb": f"{value:.9f}"}
        for timestamp_s, value in rgb_absolute
    ]
    first_bin = min(counts)
    last_bin = max(counts)
    evs_rows = []
    for index in range(first_bin, last_bin + 1):
        positive, negative = counts.get(index, [0, 0])
        evs_rows.append(
            {
                "t": f"{(index + 0.5) * bin_width_s:.9f}",
                "pos": positive,
                "neg": negative,
            }
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rgb_csv = destination / "rgb_led_signal.csv"
    evs_csv = destination / "evs_led_signal.csv"
    json_path = destination / "led_sync_data.json"
    _write_csv(rgb_csv, ["t", "rgb"], rgb_rows)
    _write_csv(evs_csv, ["t", "pos", "neg"], evs_rows)

    rgb_times_relative = [timestamp_s - origin_s for timestamp_s, _ in rgb_absolute]
    data = {
        "schema_version": 1,
        "meta": {
            "session": session_name or Path(bag_path).name,
            "rgbFps": _estimate_rate_hz(rgb_times_relative),
            "time_unit": "s",
            "time_origin_reference_s": origin_s,
            "evs_bin_ms": bin_ms,
            "evs_bin_timestamp": "center",
            "rgb_topic": rgb_topic,
            "rgb_timestamp_source": rgb_timestamp_source,
            "rgb_roi": asdict(rgb_roi),
            "evs_roi": asdict(evs_roi),
            "rgb_image_size": {
                "width": rgb_image_size[0],
                "height": rgb_image_size[1],
            },
            "evs_image_size": {
                "width": evs_image_size[0],
                "height": evs_image_size[1],
            },
            "evs_anchor": event_source.anchor.to_dict(),
        },
        "rgb": [
            {"t": timestamp_s - origin_s, "v": value}
            for timestamp_s, value in rgb_absolute
        ],
        "evs": [
            {
                "t": (index + 0.5) * bin_width_s,
                "pos": counts.get(index, [0, 0])[0],
                "neg": counts.get(index, [0, 0])[1],
            }
            for index in range(first_bin, last_bin + 1)
        ],
    }
    _write_json(json_path, data)

    return {
        "rgb_csv": str(rgb_csv),
        "evs_csv": str(evs_csv),
        "json": str(json_path),
        "rgb_samples": len(rgb_rows),
        "evs_bins": len(evs_rows),
        "evs_events_in_roi": selected_event_count,
        "time_origin_reference_s": origin_s,
    }
