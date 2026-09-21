from __future__ import annotations

import pytest

from multi_sensor_calibration.led_sync_export import Roi, _estimate_rate_hz


def test_roi_parse() -> None:
    assert Roi.parse("10, 20, 30, 40") == Roi(10, 20, 30, 40)


@pytest.mark.parametrize(
    "value",
    ["1,2,3", "1,2,3,4,5", "-1,2,3,4", "1,2,0,4", "a,2,3,4"],
)
def test_roi_parse_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        Roi.parse(value)


def test_roi_validate() -> None:
    Roi(10, 20, 30, 40).validate(100, 100, "test")
    with pytest.raises(ValueError, match="exceeds image size"):
        Roi(80, 20, 30, 40).validate(100, 100, "test")


def test_estimate_rate_hz_uses_median_delta() -> None:
    assert _estimate_rate_hz([0.0, 1.0 / 60.0, 2.0 / 60.0]) == pytest.approx(60.0)
