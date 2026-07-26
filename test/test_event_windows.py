import unittest

from multi_sensor_calibration.event_windows import (
    WindowDefinition,
    periodic_window_ends,
    reference_aligned_window_ends,
    representative_time_us,
)


class EventWindowTest(unittest.TestCase):
    def test_full_accumulation_periodic_windows(self):
        definition = WindowDefinition(
            accumulation_us=20_000,
            period_us=20_000,
            timestamp_policy="center",
        )
        self.assertEqual(
            periodic_window_ends(5_000, 70_000, definition),
            [40_000, 60_000],
        )

    def test_center_timestamp_does_not_change_interval(self):
        definition = WindowDefinition(
            accumulation_us=20_000,
            period_us=10_000,
            timestamp_policy="center",
        )
        representative, event_mean = representative_time_us(
            definition,
            10_000,
            30_000,
            [12_000, 18_000, 29_000],
        )
        self.assertEqual(representative, 20_000.0)
        self.assertAlmostEqual(event_mean, 19_666.666666666668)

    def test_event_mean_falls_back_to_center_for_empty_window(self):
        definition = WindowDefinition(
            accumulation_us=10_000,
            period_us=10_000,
            timestamp_policy="event_mean",
        )
        representative, event_mean = representative_time_us(
            definition, 20_000, 30_000, []
        )
        self.assertEqual(representative, 25_000.0)
        self.assertIsNone(event_mean)

    def test_reference_alignment_targets_representative_center(self):
        definition = WindowDefinition(
            accumulation_us=20_000,
            timestamp_policy="center",
            schedule="reference_aligned",
        )
        ends = reference_aligned_window_ends(
            [1.0, 1.02],
            reference_to_event_time=lambda value: value - 0.5,
            definition=definition,
        )
        self.assertEqual(ends, [510_000, 530_000])


if __name__ == "__main__":
    unittest.main()
