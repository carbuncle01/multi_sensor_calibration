"""EVS activity extraction and frame generation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .event_windows import EventWindow, WindowDefinition, representative_time_us
from .imaging import decode_ros_image
from .models import TimedValue
from .rosbag import iter_messages, selected_time_ns
from .transforms import TimeAnchor


@dataclass
class GeneratedFrame:
    image: Any
    window: EventWindow
    reference_time_s: float | None


def event_activity(source) -> list[TimedValue]:
    """Create an event-rate signal on the provisional reference timeline."""

    result: list[TimedValue] = []
    for batch in source.batches():
        events = batch.events
        if len(events) == 0:
            continue
        start_us = int(events["t"][0])
        end_us = int(events["t"][-1])
        duration_s = max((end_us - start_us) / 1_000_000.0, 1e-6)
        center_us = (start_us + end_us) / 2.0
        if source.anchor is None:
            raise RuntimeError(
                "EVS source has no reference clock anchor; provide metadata or an explicit anchor"
            )
        result.append(
            TimedValue(
                time_s=source.anchor.to_reference_s(center_us),
                value=len(events) / duration_s,
            )
        )
    return result


def image_activity(
    bag_path: str | Path,
    topic: str,
    *,
    timestamp_source: str = "bag",
    max_frames: int | None = None,
) -> list[TimedValue]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for image activity extraction") from exc

    result: list[TimedValue] = []
    previous = None
    frame_count = 0
    for item in iter_messages(bag_path, {topic}):
        gray = decode_ros_image(item.message, item.message_type)
        if previous is not None:
            result.append(
                TimedValue(
                    time_s=selected_time_ns(item, timestamp_source) / 1_000_000_000.0,
                    value=float(np.mean(np.abs(gray - previous))),
                )
            )
        previous = gray
        frame_count += 1
        if max_frames is not None and frame_count >= max_frames:
            break
    return result


def image_timestamps(
    bag_path: str | Path,
    topic: str,
    *,
    timestamp_source: str = "bag",
) -> list[float]:
    return [
        selected_time_ns(item, timestamp_source) / 1_000_000_000.0
        for item in iter_messages(bag_path, {topic})
    ]


def _stream_frames(
    source,
    definition: WindowDefinition,
    end_times_us: Sequence[int] | None,
    representation: str,
) -> Iterator[GeneratedFrame]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for EVS frame generation") from exc

    from .imaging import render_events

    ends = iter(end_times_us) if end_times_us is not None else None
    next_end_us = next(ends, None) if ends is not None else None
    buffered = []
    first_source_event_us = None

    for batch in source.batches():
        events = batch.events
        if len(events) == 0:
            continue
        if first_source_event_us is None:
            first_source_event_us = int(events["t"][0])
        buffered.append(events)

        if next_end_us is None and ends is None:
            period_us = definition.period_us
            if period_us is None:
                raise ValueError("period_us is required for periodic generation")
            earliest = int(events["t"][0])
            if definition.drop_partial_windows:
                earliest += definition.accumulation_us
            next_end_us = ((earliest + period_us - 1) // period_us) * period_us

        latest_us = int(events["t"][-1])
        while next_end_us is not None and next_end_us <= latest_us:
            start_us = next_end_us - definition.accumulation_us
            if (
                definition.drop_partial_windows
                and first_source_event_us is not None
                and start_us < first_source_event_us
            ):
                if ends is not None:
                    next_end_us = next(ends, None)
                else:
                    next_end_us += int(definition.period_us)
                continue
            all_events = np.concatenate(buffered) if len(buffered) > 1 else buffered[0]
            selected = all_events[
                (all_events["t"] >= start_us) & (all_events["t"] < next_end_us)
            ]
            representative_us, event_mean_us = representative_time_us(
                definition,
                start_us,
                next_end_us,
                selected["t"] if len(selected) else None,
            )
            reference_time_s = (
                source.anchor.to_reference_s(representative_us)
                if source.anchor is not None
                else None
            )
            yield GeneratedFrame(
                image=render_events(
                    selected, source.width, source.height, representation
                ),
                window=EventWindow(
                    start_us=start_us,
                    end_us=next_end_us,
                    representative_us=representative_us,
                    event_count=len(selected),
                    event_mean_us=event_mean_us,
                ),
                reference_time_s=reference_time_s,
            )

            if ends is not None:
                next_end_us = next(ends, None)
            else:
                next_end_us += int(definition.period_us)

            if next_end_us is not None:
                keep_from = next_end_us - definition.accumulation_us
                all_events = all_events[all_events["t"] >= keep_from]
                buffered = [all_events] if len(all_events) else []


def generate_event_frames(
    source,
    definition: WindowDefinition,
    *,
    end_times_us: Sequence[int] | None = None,
    representation: str = "polarity",
) -> Iterator[GeneratedFrame]:
    if definition.schedule == "reference_aligned" and end_times_us is None:
        raise ValueError("reference_aligned generation requires end_times_us")
    yield from _stream_frames(source, definition, end_times_us, representation)


def extract_ros_event_images(
    bag_path: str | Path,
    topic: str,
    definition: WindowDefinition,
    *,
    timestamp_source: str = "bag",
) -> Iterator[GeneratedFrame]:
    """Expose already-rendered event_image messages through the same contract."""

    for item in iter_messages(bag_path, {topic}):
        timestamp_us = selected_time_ns(item, timestamp_source) // 1_000
        if definition.timestamp_policy == "end":
            end_us = timestamp_us
        elif definition.timestamp_policy == "start":
            end_us = timestamp_us + definition.accumulation_us
        else:
            end_us = timestamp_us + definition.accumulation_us // 2
        start_us = end_us - definition.accumulation_us
        representative_us, _ = representative_time_us(definition, start_us, end_us)
        yield GeneratedFrame(
            image=decode_ros_image(item.message, item.message_type),
            window=EventWindow(
                start_us=start_us,
                end_us=end_us,
                representative_us=representative_us,
                event_count=-1,
                event_mean_us=None,
            ),
            reference_time_s=timestamp_us / 1_000_000.0,
        )


def write_frames(
    frames: Iterable[GeneratedFrame],
    output_dir: str | Path,
    *,
    source_type: str,
    definition: WindowDefinition,
    anchor: TimeAnchor | Any | None,
) -> dict[str, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy and OpenCV are required to write EVS images") from exc

    from .io import write_yaml

    destination = Path(output_dir)
    image_dir = destination / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, frame in enumerate(frames):
        image = frame.image
        if image.dtype != np.uint8:
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        relative = Path("images") / f"{index:08d}.png"
        if not cv2.imwrite(str(destination / relative), image):
            raise RuntimeError(f"failed to write {destination / relative}")
        rows.append(
            {
                "image": relative.as_posix(),
                "window_start_us": frame.window.start_us,
                "window_end_us": frame.window.end_us,
                "generation_timestamp_us": frame.window.end_us,
                "representative_timestamp_us": f"{frame.window.representative_us:.3f}",
                "reference_timestamp_s": (
                    f"{frame.reference_time_s:.9f}"
                    if frame.reference_time_s is not None
                    else ""
                ),
                "event_count": frame.window.event_count,
                "event_mean_us": (
                    f"{frame.window.event_mean_us:.3f}"
                    if frame.window.event_mean_us is not None
                    else ""
                ),
            }
        )

    manifest_path = destination / "frames.csv"
    fieldnames = list(rows[0]) if rows else [
        "image",
        "window_start_us",
        "window_end_us",
        "generation_timestamp_us",
        "representative_timestamp_us",
        "reference_timestamp_s",
        "event_count",
        "event_mean_us",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    resolved_anchor = getattr(anchor, "anchor", anchor)
    metadata = {
        "source_type": source_type,
        "frame_count": len(rows),
        "window": definition.to_dict(),
        "clock_anchor": (
            resolved_anchor.to_dict() if resolved_anchor is not None else None
        ),
        "frames_manifest": str(manifest_path),
    }
    write_yaml(destination / "metadata.yaml", metadata)
    return metadata
