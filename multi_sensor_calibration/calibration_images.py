"""Common image inputs for intrinsic and extrinsic calibration."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .imaging import decode_ros_image
from .rosbag import iter_messages, selected_time_ns


@dataclass(frozen=True)
class CalibrationImage:
    time_s: float
    image: Any
    source: str


def bag_images(
    bag_path: str | Path,
    topic: str,
    *,
    timestamp_source: str = "bag",
    every_n: int = 1,
    max_frames: int | None = None,
) -> Iterator[CalibrationImage]:
    if every_n <= 0:
        raise ValueError("every_n must be positive")
    accepted = 0
    for index, item in enumerate(iter_messages(bag_path, {topic})):
        if index % every_n:
            continue
        yield CalibrationImage(
            time_s=selected_time_ns(item, timestamp_source) / 1_000_000_000.0,
            image=decode_ros_image(item.message, item.message_type),
            source=f"{bag_path}:{topic}:{index}",
        )
        accepted += 1
        if max_frames is not None and accepted >= max_frames:
            return


def generated_evs_images(
    directory: str | Path,
    *,
    every_n: int = 1,
    max_frames: int | None = None,
) -> Iterator[CalibrationImage]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to read generated EVS images") from exc

    root = Path(directory)
    manifest = root / "frames.csv"
    accepted = 0
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            if index % every_n:
                continue
            image_path = root / row["image"]
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError(f"failed to read {image_path}")
            time_text = row.get("reference_timestamp_s", "")
            if not time_text:
                raise ValueError(
                    f"{manifest} has no reference timestamp for {row['image']}"
                )
            yield CalibrationImage(
                time_s=float(time_text),
                image=image.astype("float32") / 255.0,
                source=str(image_path),
            )
            accepted += 1
            if max_frames is not None and accepted >= max_frames:
                return
