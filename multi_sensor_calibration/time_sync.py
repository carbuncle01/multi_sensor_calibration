"""Signal-based temporal offset and drift estimation."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Iterable, Sequence

from .models import ClockEstimate, TimedValue


@dataclass(frozen=True)
class LagEstimate:
    lag_bins: float
    correlation: float


def _pearson_at_lag(reference: Sequence[float], sensor: Sequence[float], lag: int) -> float:
    """Return correlation for sensor shifted right by ``lag`` bins."""

    count = min(len(reference), len(sensor))
    if lag >= 0:
        ref_values = reference[: count - lag]
        sensor_values = sensor[lag:count]
    else:
        ref_values = reference[-lag:count]
        sensor_values = sensor[: count + lag]

    if len(ref_values) < 4:
        return float("-inf")

    ref_mean = sum(ref_values) / len(ref_values)
    sensor_mean = sum(sensor_values) / len(sensor_values)
    numerator = 0.0
    ref_energy = 0.0
    sensor_energy = 0.0
    for ref_value, sensor_value in zip(ref_values, sensor_values):
        ref_delta = ref_value - ref_mean
        sensor_delta = sensor_value - sensor_mean
        numerator += ref_delta * sensor_delta
        ref_energy += ref_delta * ref_delta
        sensor_energy += sensor_delta * sensor_delta

    denominator = math.sqrt(ref_energy * sensor_energy)
    if denominator <= 1e-18:
        return float("-inf")
    return numerator / denominator


def estimate_lag(
    reference: Sequence[float],
    sensor: Sequence[float],
    max_lag_bins: int,
) -> LagEstimate:
    """Estimate a possibly fractional lag using normalized correlation."""

    if max_lag_bins < 0:
        raise ValueError("max_lag_bins must be non-negative")
    if min(len(reference), len(sensor)) < max(8, 2 * max_lag_bins + 3):
        raise ValueError("not enough samples for the requested lag range")

    correlations = {
        lag: _pearson_at_lag(reference, sensor, lag)
        for lag in range(-max_lag_bins, max_lag_bins + 1)
    }
    best_lag = max(correlations, key=correlations.get)
    best_correlation = correlations[best_lag]
    fractional_lag = float(best_lag)

    before = correlations.get(best_lag - 1)
    after = correlations.get(best_lag + 1)
    if before is not None and after is not None and math.isfinite(best_correlation):
        curvature = before - 2.0 * best_correlation + after
        if abs(curvature) > 1e-12:
            delta = 0.5 * (before - after) / curvature
            if abs(delta) <= 1.0:
                fractional_lag += delta

    return LagEstimate(lag_bins=fractional_lag, correlation=best_correlation)


def _bin_samples(
    samples: Iterable[TimedValue],
    start_s: float,
    end_s: float,
    bin_width_s: float,
) -> list[float]:
    count = int(math.floor((end_s - start_s) / bin_width_s))
    if count <= 0:
        return []

    sums = [0.0] * count
    sample_counts = [0] * count
    for sample in samples:
        index = int((sample.time_s - start_s) / bin_width_s)
        if 0 <= index < count:
            sums[index] += float(sample.value)
            sample_counts[index] += 1

    values = [
        sums[index] / sample_counts[index] if sample_counts[index] else 0.0
        for index in range(count)
    ]
    return values


def _linear_fit(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """Fit y = intercept + slope * (x - mean_x)."""

    if not points:
        raise ValueError("at least one point is required")
    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    denominator = sum((point[0] - mean_x) ** 2 for point in points)
    if denominator <= 1e-18:
        return mean_y, 0.0
    slope = sum(
        (point[0] - mean_x) * (point[1] - mean_y) for point in points
    ) / denominator
    return mean_y, slope


def estimate_clock(
    reference_samples: Sequence[TimedValue],
    sensor_samples: Sequence[TimedValue],
    *,
    bin_width_s: float = 0.01,
    max_lag_s: float = 0.5,
    window_s: float = 15.0,
    min_correlation: float = 0.1,
) -> ClockEstimate:
    """Estimate offset and linear drift from two correlated activity signals."""

    if bin_width_s <= 0.0:
        raise ValueError("bin_width_s must be positive")
    if max_lag_s < 0.0:
        raise ValueError("max_lag_s must be non-negative")
    if window_s <= 0.0:
        raise ValueError("window_s must be positive")
    if not reference_samples or not sensor_samples:
        raise ValueError("both activity signals must contain samples")

    reference_samples = sorted(reference_samples, key=lambda sample: sample.time_s)
    sensor_samples = sorted(sensor_samples, key=lambda sample: sample.time_s)
    start_s = max(reference_samples[0].time_s, sensor_samples[0].time_s)
    end_s = min(reference_samples[-1].time_s, sensor_samples[-1].time_s)
    if end_s - start_s < max(window_s, 8.0 * bin_width_s):
        raise ValueError("activity signals do not have enough overlapping duration")

    reference = _bin_samples(reference_samples, start_s, end_s, bin_width_s)
    sensor = _bin_samples(sensor_samples, start_s, end_s, bin_width_s)
    max_lag_bins = max(1, int(round(max_lag_s / bin_width_s)))
    window_bins = max(2 * max_lag_bins + 5, int(round(window_s / bin_width_s)))
    step_bins = max(1, window_bins // 2)

    estimates: list[tuple[float, float, float]] = []
    for begin in range(0, len(reference) - window_bins + 1, step_bins):
        finish = begin + window_bins
        try:
            lag = estimate_lag(
                reference[begin:finish],
                sensor[begin:finish],
                max_lag_bins,
            )
        except ValueError:
            continue
        if not math.isfinite(lag.correlation) or lag.correlation < min_correlation:
            continue

        center_s = start_s + (begin + window_bins / 2.0) * bin_width_s
        # Positive lag means the sensor signal appears later on the common
        # timestamp axis, so sensor timestamps need a negative correction.
        offset_s = -lag.lag_bins * bin_width_s
        estimates.append((center_s, offset_s, lag.correlation))

    if not estimates:
        raise ValueError(
            "no correlated windows were found; use a stronger common stimulus "
            "or relax the correlation/lag settings"
        )

    # A median filter removes occasional false correlation peaks before the
    # affine clock fit.
    median_offset = statistics.median(value[1] for value in estimates)
    max_deviation = max(3.0 * bin_width_s, max_lag_s * 0.25)
    filtered = [
        value for value in estimates if abs(value[1] - median_offset) <= max_deviation
    ]
    if not filtered:
        filtered = estimates

    anchor_s = sum(value[0] for value in filtered) / len(filtered)
    centered_points = [(value[0], value[1]) for value in filtered]
    offset_at_anchor_s, drift = _linear_fit(centered_points)
    correlation = statistics.median(value[2] for value in filtered)

    return ClockEstimate(
        anchor_sensor_time_s=anchor_s,
        offset_at_anchor_s=offset_at_anchor_s,
        drift=drift,
        correlation=correlation,
        window_count=len(filtered),
        bin_width_s=bin_width_s,
    )
