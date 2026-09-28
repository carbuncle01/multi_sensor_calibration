import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from multi_sensor_calibration.led_sync_export import _tile_means_u8
from multi_sensor_calibration.led_sync_auto import (
    _side_range, _side_for_time, _rgb_phase_store, _load_rgb_tiles,
)


class LedSpatialRegressionTest(unittest.TestCase):
    def test_normalized_intensity_preserves_eight_bit_range(self):
        values, width, height = _tile_means_u8(np.array([[0., .25, .5, .75, 1.]]), 1)
        self.assertEqual(values.tolist(), [0, 64, 128, 191, 255])
        self.assertEqual((width, height), (5, 1))

    def test_mean_before_quantization_and_partial_tiles(self):
        values, width, height = _tile_means_u8(np.array([[0., 1., .25], [0., 1., .25]]), 2)
        self.assertEqual(values.tolist(), [128, 64])
        self.assertEqual((width, height), (2, 1))

    def test_merged_export_range_splits_into_distinct_sides(self):
        meta = {'ranges': [[0., 23.3]]}
        self.assertEqual(_side_range(meta, 'start'), (0., 11.65))
        self.assertEqual(_side_range(meta, 'end'), (11.65, 23.3))
        self.assertEqual(_side_for_time(20., meta['ranges']), 'end')
        self.assertEqual(_side_for_time(2., meta['ranges']), 'start')

    def test_rgb_end_phase_receives_signal_for_short_recording(self):
        times = np.array([1., 1.02, 20., 20.02])
        tiles = np.array([[0, 0], [100, 0], [0, 0], [100, 0]], dtype=np.uint8)
        stores, _ = _rgb_phase_store(times, tiles, {'ranges': [[0., 23.3]]})
        self.assertGreater(stores['start']['pos'].sum(), 0)
        self.assertGreater(stores['end']['pos'].sum(), 0)

    def test_separate_ranges_unchanged(self):
        meta = {'ranges': [[0., 12.], [15., 27.]]}
        self.assertEqual(_side_range(meta, 'start'), (0., 12.))
        self.assertEqual(_side_range(meta, 'end'), (15., 27.))

    def test_legacy_quantized_tiles_require_reexport(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'tiles.bin').write_bytes(struct.pack('<dB', 0., 0) + struct.pack('<dB', 1., 1))
            meta = dict(path='tiles.bin', grid_width=1, grid_height=1, frame_count=2, record_bytes=9)
            with self.assertRaisesRegex(ValueError, 'force-export'):
                _load_rgb_tiles(path, meta)
            # A correctly versioned dark recording is valid, despite low signal.
            meta['intensity_scale'] = 'uint8_0_255_v1'
            _, tiles = _load_rgb_tiles(path, meta)
            self.assertEqual(tiles[:, 0].tolist(), [0, 1])


if __name__ == '__main__':
    unittest.main()
