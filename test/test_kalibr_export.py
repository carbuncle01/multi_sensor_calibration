import csv
import tempfile
import unittest
from pathlib import Path

from multi_sensor_calibration.kalibr_export import (
    FrameMetadata,
    common_matches,
    generated_frame_metadata,
    match_frames,
    parse_camera_exports,
    select_reference_frames,
)
from multi_sensor_calibration.models import ClockEstimate


def frame(index, timestamp):
    return FrameMetadata(
        index=index,
        sensor_time_s=timestamp,
        corrected_time_s=timestamp,
        source=f"frame-{index}",
    )


class KalibrExportTest(unittest.TestCase):
    def test_camera_order_becomes_contiguous_kalibr_chain(self):
        cameras = parse_camera_exports(
            {
                "kalibr": {
                    "cameras": [
                        {"sensor": "evs", "model": "pinhole-radtan"},
                        {"sensor": "rgb", "model": "pinhole-radtan"},
                        {"sensor": "thermal", "model": "pinhole-radtan"},
                    ]
                }
            }
        )
        self.assertEqual(
            [(item.camera, item.sensor, item.topic) for item in cameras],
            [
                ("cam0", "evs", "/cam0/image_raw"),
                ("cam1", "rgb", "/cam1/image_raw"),
                ("cam2", "thermal", "/cam2/image_raw"),
            ],
        )

    def test_reference_rate_selection(self):
        values = [frame(index, index * 0.05) for index in range(21)]
        selected = select_reference_frames(values, max_rate_hz=4.0)
        self.assertEqual(
            [round(item.corrected_time_s, 2) for item in selected],
            [0.0, 0.25, 0.5, 0.75, 1.0],
        )

    def test_nearest_match_is_not_reused(self):
        targets = [frame(0, 1.0), frame(1, 1.01)]
        candidates = [frame(10, 1.005)]
        matches = match_frames(targets, candidates, max_delta_s=0.02)
        self.assertEqual(list(matches), [0])

    def test_common_matches_remove_incomplete_views(self):
        reference = [frame(0, 1.0), frame(1, 2.0), frame(2, 3.0)]
        frames_by_sensor = {
            "rgb": reference,
            "evs": [frame(10, 1.001), frame(11, 2.001), frame(12, 3.001)],
            "thermal": [frame(20, 1.002), frame(22, 3.002)],
        }
        common, matches = common_matches(
            reference,
            frames_by_sensor,
            reference_sensor="rgb",
            max_delta_s=0.01,
        )
        self.assertEqual([item.index for item in common], [0, 2])
        self.assertEqual(matches["thermal"][2].frame.index, 22)

    def test_generated_timestamp_receives_clock_model_once(self):
        clock = ClockEstimate(
            anchor_sensor_time_s=10.0,
            offset_at_anchor_s=0.25,
            drift=100e-6,
            correlation=0.9,
            window_count=3,
            bin_width_s=0.01,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "frame.png").touch()
            with (root / "frames.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["image", "reference_timestamp_s"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "image": "frame.png",
                        "reference_timestamp_s": "20.0",
                    }
                )

            frames = generated_frame_metadata(root, clock=clock)

        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].sensor_time_s, 20.0)
        self.assertAlmostEqual(frames[0].corrected_time_s, 20.251)


if __name__ == "__main__":
    unittest.main()
