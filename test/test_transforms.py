import math
import unittest

from multi_sensor_calibration.transforms import TimeAnchor, quaternion_from_rpy


class TransformTest(unittest.TestCase):
    def test_time_anchor_round_trip(self):
        anchor = TimeAnchor(
            source_time_us=250.0,
            reference_time_s=1000.0,
            scale=1.0001,
        )
        source_us = 1_500_000.0
        self.assertAlmostEqual(
            anchor.to_source_us(anchor.to_reference_s(source_us)),
            source_us,
            places=5,
        )

    def test_zero_rpy_is_identity_quaternion(self):
        self.assertEqual(quaternion_from_rpy(0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

    def test_yaw_quaternion(self):
        quaternion = quaternion_from_rpy(0.0, 0.0, math.pi)
        self.assertAlmostEqual(quaternion[2], 1.0)
        self.assertAlmostEqual(quaternion[3], 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
