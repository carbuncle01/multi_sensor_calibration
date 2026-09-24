# RC pop-out default EVS/RGB calibration

This directory contains the default spatial calibration for the fixed
SilkyEvCam + Intel RealSense D455 mount used by the RC-car pop-out experiment.

- `kalibr-camchain.yaml` is the directly usable Kalibr camera chain. `cam0` is
  EVS (640x480), `cam1` is RGB (848x480), and `T_cn_cnm1` maps EVS coordinates
  into RGB coordinates.
- `profile.yaml` records acquisition conditions, calibration quality, transform
  convention, and the reuse contract.
- `time-sync-reference.yaml` preserves the calibration-session LED result for
  provenance. It is not a universal time offset; verify or estimate time sync
  for every recording session.

The spatial calibration may be reused while the cameras remain rigidly mounted
and their resolution, lens, focus, and image pipeline are unchanged. A spatial
extrinsic does not by itself define a depth-independent pixel-to-pixel mapping
for cameras with different optical centers. For a pop-out recording, declare
one of the following projection models when visualizing alignment:

1. fixed obstacle/RC-car plane or distance (recommended for quantitative use),
2. rotation-only DSEC-like mapping (preview only), or
3. measured per-pixel depth.
