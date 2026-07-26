"""Canonical event-window definitions shared by every EVS input backend."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class WindowDefinition:
    """Describe how events become a calibration image.

    A frame generated at ``end_us`` contains events in the half-open interval
    ``[end_us - accumulation_us, end_us)``.  This matches the Metavision frame
    generator convention. ``timestamp_policy`` only controls the timestamp used
    to match the resulting image with another camera; it never changes which
    events belong to the frame.
    """

    accumulation_us: int
    timestamp_policy: str = "center"
    schedule: str = "periodic"
    period_us: int | None = None
    drop_partial_windows: bool = True

    def __post_init__(self) -> None:
        if self.accumulation_us <= 0:
            raise ValueError("accumulation_us must be positive")
        if self.timestamp_policy not in {"start", "center", "end", "event_mean"}:
            raise ValueError(
                "timestamp_policy must be start, center, end, or event_mean"
            )
        if self.schedule not in {"periodic", "reference_aligned"}:
            raise ValueError("schedule must be periodic or reference_aligned")
        if self.schedule == "periodic" and (self.period_us is None or self.period_us <= 0):
            raise ValueError("period_us must be positive for a periodic schedule")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WindowDefinition":
        return cls(
            accumulation_us=int(value["accumulation_us"]),
            timestamp_policy=str(value.get("timestamp_policy", "center")),
            schedule=str(value.get("schedule", "periodic")),
            period_us=(
                int(value["period_us"]) if value.get("period_us") is not None else None
            ),
            drop_partial_windows=bool(value.get("drop_partial_windows", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["interval"] = "[window_end_us - accumulation_us, window_end_us)"
        result["generation_timestamp"] = "window_end_us"
        return result


@dataclass(frozen=True)
class EventWindow:
    start_us: int
    end_us: int
    representative_us: float
    event_count: int
    event_mean_us: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def representative_time_us(
    definition: WindowDefinition,
    start_us: int,
    end_us: int,
    event_timestamps_us: Sequence[int] | None = None,
) -> tuple[float, float | None]:
    event_mean = None
    if event_timestamps_us is not None and len(event_timestamps_us):
        event_mean = sum(int(value) for value in event_timestamps_us) / len(
            event_timestamps_us
        )

    if definition.timestamp_policy == "start":
        return float(start_us), event_mean
    if definition.timestamp_policy == "end":
        return float(end_us), event_mean
    if definition.timestamp_policy == "event_mean":
        if event_mean is None:
            return (start_us + end_us) / 2.0, None
        return event_mean, event_mean
    return (start_us + end_us) / 2.0, event_mean


def periodic_window_ends(
    first_event_us: int,
    last_event_us: int,
    definition: WindowDefinition,
    *,
    origin_us: int = 0,
) -> list[int]:
    if definition.schedule != "periodic" or definition.period_us is None:
        raise ValueError("periodic WindowDefinition is required")
    if last_event_us < first_event_us:
        return []

    earliest_end = first_event_us
    if definition.drop_partial_windows:
        earliest_end += definition.accumulation_us
    period_us = definition.period_us
    first_index = math.ceil((earliest_end - origin_us) / period_us)
    first_end = origin_us + first_index * period_us
    return list(range(first_end, last_event_us + 1, period_us))


def reference_aligned_window_ends(
    reference_times_s: Iterable[float],
    *,
    reference_to_event_time,
    definition: WindowDefinition,
) -> list[int]:
    """Align the configured representative time with reference frames."""

    result = []
    for reference_time_s in reference_times_s:
        representative_us = float(reference_to_event_time(reference_time_s)) * 1_000_000.0
        if definition.timestamp_policy == "end":
            end_us = representative_us
        elif definition.timestamp_policy == "start":
            end_us = representative_us + definition.accumulation_us
        else:
            # event_mean cannot be predicted before rendering, so use the
            # nominal center as the scheduling target and still record the
            # measured event mean in the output manifest.
            end_us = representative_us + definition.accumulation_us / 2.0
        result.append(int(round(end_us)))
    if any(after <= before for before, after in zip(result, result[1:])):
        raise ValueError("reference-aligned EVS window times must be strictly increasing")
    return result
