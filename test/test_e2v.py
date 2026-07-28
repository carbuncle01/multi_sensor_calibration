import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from multi_sensor_calibration.e2v import resolve_checkpoint


class E2VConfigurationTest(unittest.TestCase):
    def test_explicit_checkpoint_takes_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "explicit.ckpt"
            checkpoint.touch()
            with patch.dict(
                os.environ,
                {"METAVISION_E2V_CHECKPOINT": str(Path(directory) / "missing.ckpt")},
            ):
                self.assertEqual(resolve_checkpoint(checkpoint), checkpoint.resolve())

    def test_environment_checkpoint_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "environment.ckpt"
            checkpoint.touch()
            with patch.dict(
                os.environ,
                {"METAVISION_E2V_CHECKPOINT": str(checkpoint)},
            ):
                self.assertEqual(resolve_checkpoint(None), checkpoint.resolve())


if __name__ == "__main__":
    unittest.main()
