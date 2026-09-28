"""Timing regressions: hold real RGB frames and never include future events."""
import csv
import json
import tempfile
from contextlib import ExitStack
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from multi_sensor_calibration.scenario_overlay import (
    _event_frames, _event_timeline, _hold_rgb_frames, render_scenario_overlay,
)


class SlowMotionTimingTest(unittest.TestCase):
    def timeline(self, rgb_times, **kwargs):
        options = dict(start_s=0.0, duration_s=None, step_ms=1.0, max_frames=None)
        options.update(kwargs)
        return _event_timeline(rgb_times, **options)

    def test_rgb_holds_until_actual_next_frame(self):
        rgb = [(0.0, 'a'), (0.0167, 'b'), (0.0334, 'c')]
        times, indices = self.timeline([t for t, _ in rgb])
        held = list(_hold_rgb_frames((rgb[i] for i in indices), times))
        self.assertEqual([image for _, image in held[:17]], ['a'] * 17)
        self.assertEqual(held[17], rgb[1])
        self.assertTrue(all(t_rgb <= t for t, (t_rgb, _) in zip(times, held)))
        self.assertEqual(len(times), 34)

    def test_crop_keeps_preceding_frame(self):
        rgb = [100.0, 100.016, 100.032]
        times, indices = self.timeline(rgb, start_s=0.010, duration_s=0.010)
        self.assertEqual(indices, [0, 1])
        held = list(_hold_rgb_frames(((rgb[i], i) for i in indices), times))
        self.assertEqual(held[0], (100.0, 0))
        self.assertEqual(held[-1], (100.016, 1))
        self.assertEqual(len(times), 10)

    def test_exact_rgb_boundary_uses_new_frame(self):
        held = list(_hold_rgb_frames([(0.0, 'a'), (0.01, 'b')], [0.009, 0.01]))
        self.assertEqual(held, [(0.0, 'a'), (0.01, 'b')])

    def test_gap_preserves_last_frame_without_interpolation(self):
        times, indices = self.timeline([0.0, 0.01, 0.10])
        held = list(_hold_rgb_frames(((t, t) for t in [0.0, 0.01, 0.10]), times))
        self.assertEqual(held[99], (0.01, 0.01))
        self.assertEqual(held[100], (0.10, 0.10))

    def test_epoch_timestamps_and_output_cap(self):
        origin = 1_789_984_564.0
        times, _ = self.timeline([origin, origin + 0.1], max_frames=5)
        self.assertEqual(len(times), 5)
        self.assertAlmostEqual(times[-1] - origin, 0.004, places=6)

    def test_duration_is_exclusive_and_stops_at_recorded_coverage(self):
        times, _ = self.timeline([0.0, 0.10], duration_s=0.003)
        self.assertEqual(times, [0.0, 0.001, 0.002])
        times, _ = self.timeline([0.0, 0.002], duration_s=1.0)
        self.assertEqual(times[-1], 0.002)

    def test_bad_timestamps_and_step_are_rejected(self):
        for values in ([], [1.0, 0.0], [0.0, 0.0], [float('nan')]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.timeline(values)
        for step in (0.0, -1.0, float('nan'), float('inf'), 0.0001):
            with self.subTest(step=step), self.assertRaises(ValueError):
                self.timeline([0.0, 1.0], step_ms=step)
        with self.assertRaises(ValueError):
            self.timeline([0.0, 1.0], start_s=2.0)

    def test_no_future_rgb_fallback(self):
        with self.assertRaises(ValueError):
            list(_hold_rgb_frames([(0.01, 'future')], [0.0]))

    def test_noncausal_windows_and_rgb_decimation_rejected(self):
        for options in (dict(event_window_position='center'),
                        dict(event_window_position='after'), dict(every_n=2)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                render_scenario_overlay('a', 'b', 'c', 'd', 'e', timeline='event', **options)

    def test_cli_preserves_legacy_and_exposes_event_mode(self):
        from multi_sensor_calibration.cli import build_parser, command_scenario_overlay
        args = build_parser().parse_args([
            'scenario-overlay', '--bag', 'bag', '--event-file', 'raw',
            '--time-sync', 'sync', '--camchain', 'cam', '--output-dir', 'out',
            '--timeline', 'event', '--step-ms', '0.5', '--fps', '60',
        ])
        with patch('multi_sensor_calibration.scenario_overlay.render_scenario_overlay', return_value={}) as render:
            command_scenario_overlay(args)
        self.assertEqual(render.call_args.kwargs['timeline'], 'event')
        self.assertEqual(render.call_args.kwargs['step_ms'], 0.5)
        self.assertIsNone(render.call_args.kwargs['event_window_position'])

    def test_event_window_excludes_future_and_exact_endpoint(self):
        import numpy as np
        events = np.array([(0, 0, 1, t) for t in [0, 999, 1000, 1999, 2000, 3000]],
                          dtype=[('x', 'i2'), ('y', 'i2'), ('p', 'i2'), ('t', 'i8')])
        # Identity clock and a zero anchor make reference seconds equal sensor seconds.
        source = SimpleNamespace(width=1, height=1,
            anchor=SimpleNamespace(to_source_us=lambda t: t * 1e6, to_reference_s=lambda t: t / 1e6),
            batches=lambda: iter([SimpleNamespace(events=events)]))
        clock = {'models': {'evs': {'anchor_sensor_time_s': 0.0,
                 'offset_at_anchor_s': 0.0, 'drift': 0.0}}}
        with patch('multi_sensor_calibration.io.load_yaml', return_value=clock), \
             patch('multi_sensor_calibration.evs_sources.MetavisionFileSource', return_value=source):
            frames, _ = _event_frames('unused.raw', 'unused.yaml', [0.001, 0.002], 1.0, 'before')
            frames = list(frames)
        self.assertEqual([(f.window.start_us, f.window.end_us, f.window.event_count)
                          for f in frames], [(0, 1000, 2), (1000, 2000, 2)])

    def test_renderer_writes_held_rgb_and_frame_manifest(self):
        import numpy as np
        from multi_sensor_calibration import scenario_overlay as module
        for timeline, event_limit in (("rgb", None), ("event", None), ("event", 2)):
            with self.subTest(timeline=timeline, event_limit=event_limit), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                root = Path(tmp)
                source = root / 'input'
                source.touch()
                output = root / 'output'
                writers = []

                class Writer:
                    def __init__(self, path, fourcc, rate, size):
                        Path(path).touch()
                        self.frames = []
                        writers.append(self)
                    def isOpened(self):
                        return True
                    def write(self, frame):
                        self.frames.append(frame.copy())
                    def release(self):
                        pass

                cv2 = MagicMock()
                cv2.VideoWriter.side_effect = Writer
                cv2.remap.side_effect = lambda image, *a: image.copy()
                cv2.initUndistortRectifyMap.return_value = (None, None)
                stack.enter_context(patch.dict('sys.modules', {'cv2': cv2}))
                camera = {'matrix': np.eye(3), 'distortion': np.zeros(5), 'size': (2, 2)}
                stack.enter_context(patch.object(module, '_camera', return_value=camera))
                stack.enter_context(patch('multi_sensor_calibration.io.load_yaml',
                    return_value={'cam1': {'T_cn_cnm1': np.eye(4).tolist()}}))
                stack.enter_context(patch('multi_sensor_calibration.io.write_yaml',
                    side_effect=lambda path, value: path.write_text(json.dumps(value))))
                stack.enter_context(patch.object(module, '_selected_rgb_times',
                    return_value=(0.0, [0, 1, 2], [0.0, 0.002, 0.004])))
                frames = [(t, np.full((2, 2, 3), value, dtype=np.uint8))
                          for t, value in [(0.0, 10), (0.002, 20), (0.004, 30)]]
                stack.enter_context(patch.object(module, '_rgb_frames', return_value=iter(frames)))
                captured = {}

                def event_frames(raw, sync, times, window, position):
                    captured.update(window=window, position=position)
                    times = times[:event_limit]
                    events = (SimpleNamespace(image=np.zeros((2, 2), dtype=np.uint8),
                        window=SimpleNamespace(start_us=round(t*1e6-window*1000),
                                               end_us=round(t*1e6), event_count=0)) for t in times)
                    return events, SimpleNamespace(inverse=lambda t: t)

                stack.enter_context(patch.object(module, '_event_frames', side_effect=event_frames))
                stack.enter_context(patch.object(module, '_polarity_images', return_value=(
                    np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((2, 2), dtype=bool))))
                stack.enter_context(patch.object(module, '_annotate'))
                summary = module.render_scenario_overlay(
                    source, source, source, source, output,
                    timeline=timeline, macos_compatible=False,
                )
                with (output / 'frames.csv').open() as stream:
                    rows = list(csv.DictReader(stream))
                if timeline == 'event':
                    expected_count = event_limit or 5
                    self.assertEqual(summary['rendered_frames'], expected_count)
                    self.assertEqual(summary['fps'], 60)
                    self.assertAlmostEqual(summary['slowdown_factor'], 1000/60)
                    self.assertEqual([int(f[0, 0, 0]) for f in writers[0].frames],
                                     [10, 10, 20, 20, 30][:expected_count])
                    self.assertEqual([float(r['rgb_age_ms']) for r in rows], [0, 1, 0, 1, 0][:expected_count])
                    self.assertEqual(captured, {'window': 2.0, 'position': 'before'})
                else:
                    self.assertEqual(summary['rendered_frames'], 3)
                    self.assertEqual(captured, {'window': 10.0, 'position': 'center'})
                self.assertEqual(summary['truncated'], event_limit is not None)
                self.assertEqual(summary['reference_end_s'], float(rows[-1]['reference_time_s']))
                self.assertEqual(len(rows), summary['rendered_frames'])
                self.assertTrue((output / 'summary.yaml').is_file())


if __name__ == '__main__':
    unittest.main()
