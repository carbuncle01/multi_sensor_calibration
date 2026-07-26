"""Shared data models for calibration results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TimedValue:
    """A scalar activity measurement at a timestamp in seconds."""

    time_s: float
    value: float


@dataclass(frozen=True)
class ClockEstimate:
    """Affine mapping from a sensor clock onto the reference timeline.

    The mapping is expressed around an anchor to keep floating-point precision:

        reference_time = sensor_time
                       + offset_at_anchor_s
                       + drift * (sensor_time - anchor_sensor_time_s)
    """

    anchor_sensor_time_s: float
    offset_at_anchor_s: float
    drift: float
    correlation: float
    window_count: int
    bin_width_s: float

    @property
    def drift_ppm(self) -> float:
        return self.drift * 1_000_000.0

    def apply(self, sensor_time_s: float) -> float:
        return (
            sensor_time_s
            + self.offset_at_anchor_s
            + self.drift * (sensor_time_s - self.anchor_sensor_time_s)
        )

    def inverse(self, reference_time_s: float) -> float:
        scale = 1.0 + self.drift
        if abs(scale) <= 1e-15:
            raise ValueError("clock estimate is not invertible")
        return (
            reference_time_s
            - self.offset_at_anchor_s
            + self.drift * self.anchor_sensor_time_s
        ) / scale

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["drift_ppm"] = self.drift_ppm
        result["equation"] = (
            "reference_time = sensor_time + offset_at_anchor_s "
            "+ drift * (sensor_time - anchor_sensor_time_s)"
        )
        return result

    @classmethod
    def identity(cls, anchor_sensor_time_s: float, bin_width_s: float) -> "ClockEstimate":
        return cls(
            anchor_sensor_time_s=anchor_sensor_time_s,
            offset_at_anchor_s=0.0,
            drift=0.0,
            correlation=1.0,
            window_count=1,
            bin_width_s=bin_width_s,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ClockEstimate":
        return cls(
            anchor_sensor_time_s=float(value["anchor_sensor_time_s"]),
            offset_at_anchor_s=float(value["offset_at_anchor_s"]),
            drift=float(value.get("drift", float(value.get("drift_ppm", 0.0)) / 1_000_000.0)),
            correlation=float(value.get("correlation", 0.0)),
            window_count=int(value.get("window_count", 0)),
            bin_width_s=float(value.get("bin_width_s", 0.0)),
        )
