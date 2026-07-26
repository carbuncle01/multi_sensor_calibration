import math
import unittest

from multi_sensor_calibration.models import ClockEstimate, TimedValue
from multi_sensor_calibration.time_sync import estimate_clock


def activity(physical_time_s):
    value = 0.0
    for center in (8.0, 15.0, 27.0, 41.0, 53.0):
        value += math.exp(-((physical_time_s - center) / 0.22) ** 2)
    return value


class TimeSyncTest(unittest.TestCase):
    def test_recovers_constant_sensor_timestamp_delay(self):
        step_s = 0.01
        delay_s = 0.12
        reference = []
        sensor = []
        for index in range(6000):
            physical_s = index * step_s
            value = activity(physical_s)
            reference.append(TimedValue(physical_s, value))
            sensor.append(TimedValue(physical_s + delay_s, value))

        estimate = estimate_clock(
            reference,
            sensor,
            bin_width_s=step_s,
            max_lag_s=0.3,
            window_s=12.0,
            min_correlation=0.2,
        )
        self.assertAlmostEqual(estimate.offset_at_anchor_s, -delay_s, delta=0.015)
        self.assertGreater(estimate.correlation, 0.9)

    def test_clock_inverse_round_trip(self):
        estimate = ClockEstimate(
            anchor_sensor_time_s=100.0,
            offset_at_anchor_s=-0.08,
            drift=75e-6,
            correlation=0.9,
            window_count=4,
            bin_width_s=0.01,
        )
        for sensor_time_s in (90.0, 100.0, 130.0):
            reference_time_s = estimate.apply(sensor_time_s)
            self.assertAlmostEqual(
                estimate.inverse(reference_time_s), sensor_time_s, places=9
            )


if __name__ == "__main__":
    unittest.main()
