"""Small clock and rigid-transform helpers without third-party dependencies."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TimeAnchor:
    """Provisional mapping between EVS source time and the ROS/reference clock."""

    source_time_us: float
    reference_time_s: float
    scale: float = 1.0
    provenance: str = ""

    def to_reference_s(self, source_time_us: float) -> float:
        delta_s = (float(source_time_us) - self.source_time_us) / 1_000_000.0
        return self.reference_time_s + self.scale * delta_s

    def to_source_us(self, reference_time_s: float) -> float:
        delta_s = (float(reference_time_s) - self.reference_time_s) / self.scale
        return self.source_time_us + delta_s * 1_000_000.0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["equation"] = (
            "reference_time_s = reference_anchor_s + scale * "
            "(source_time_us - source_anchor_us) / 1e6"
        )
        return result


def quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, ...]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )
