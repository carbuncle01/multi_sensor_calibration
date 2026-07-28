"""OpenEB Event-to-Video inference without the training-only Lightning wrapper."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


DEFAULT_CHECKPOINT_CANDIDATES = (
    "/opt/openeb/sdk/modules/core_ml/models/e2v.ckpt",
    "/usr/local/share/metavision/sdk/core_ml/models/e2v.ckpt",
)


def resolve_checkpoint(path: str | Path | None) -> Path:
    """Resolve an explicit, environment-provided, or Docker OpenEB checkpoint."""

    candidates = []
    if path:
        candidates.append(Path(path))
    environment_path = os.environ.get("METAVISION_E2V_CHECKPOINT")
    if environment_path:
        candidates.append(Path(environment_path))
    candidates.extend(Path(value) for value in DEFAULT_CHECKPOINT_CANDIDATES)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "OpenEB E2V checkpoint was not found; set evs.e2v.checkpoint_path, "
        f"--e2v-checkpoint, or METAVISION_E2V_CHECKPOINT (searched: {searched})"
    )


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        return vars(value)
    except TypeError as exc:
        raise ValueError("E2V checkpoint hyper_parameters must be a mapping") from exc


class E2VReconstructor:
    """Stateful OpenEB E2V grayscale reconstructor.

    Only inference dependencies are imported. In particular, this intentionally
    avoids ``EventToVideoLightningModel``, which constructs training losses and
    pulls in PyTorch Lightning, Kornia, and torchvision.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None,
        *,
        device: str = "auto",
        normalize_num_stds: float = 6.0,
    ) -> None:
        if normalize_num_stds <= 0.0:
            raise ValueError("evs.e2v.normalize_num_stds must be positive")

        self.checkpoint_path = resolve_checkpoint(checkpoint_path)
        self.requested_device = str(device)
        self.normalize_num_stds = float(normalize_num_stds)

        try:
            import torch
            from metavision_core_ml.event_to_video.event_to_video import EventToVideo
        except ImportError as exc:
            raise RuntimeError(
                "E2V reconstruction requires PyTorch and the OpenEB "
                "metavision_core_ml Python package inside the Docker image"
            ) from exc

        self._torch = torch
        self.device = self._resolve_device(torch, self.requested_device)
        checkpoint = self._load_checkpoint(torch)
        hparams = _as_mapping(checkpoint.get("hyper_parameters", {}))
        required_hparams = ("event_volume_depth", "cin", "cout")
        missing_hparams = [key for key in required_hparams if key not in hparams]
        if missing_hparams:
            raise ValueError(
                "E2V checkpoint is missing hyperparameters: "
                + ", ".join(missing_hparams)
            )
        self.event_volume_depth = int(hparams["event_volume_depth"])
        if self.event_volume_depth <= 0:
            raise ValueError("E2V checkpoint event_volume_depth must be positive")
        if int(hparams["cin"]) != self.event_volume_depth:
            raise ValueError(
                "E2V checkpoint cin does not match event_volume_depth"
            )

        model = EventToVideo(
            int(hparams["cin"]),
            int(hparams["cout"]),
            int(hparams.get("num_layers", 3)),
            int(hparams.get("base", 4)),
            str(hparams.get("cell", "lstm")),
            bool(hparams.get("separable", False)),
            bool(hparams.get("separable_hidden", False)),
            str(hparams.get("archi", "all_rnn")),
        )
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("E2V checkpoint does not contain state_dict")
        model_state = {
            key[len("model.") :]: value
            for key, value in state_dict.items()
            if key.startswith("model.")
        }
        if not model_state:
            raise ValueError("E2V checkpoint contains no model weights")
        model.load_state_dict(model_state, strict=True)
        self.model = model.eval().to(self.device)
        self.num_layers = int(hparams.get("num_layers", 3))

    @staticmethod
    def _resolve_device(torch, requested: str):
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"E2V device {requested!r} was requested but CUDA is unavailable"
            )
        try:
            return torch.device(requested)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"invalid E2V device: {requested}") from exc

    def _load_checkpoint(self, torch):
        try:
            # OpenEB's bundled checkpoint is trusted project data. Explicitly
            # opt out of the newer weights-only default because the historical
            # checkpoint stores argparse.Namespace hyperparameters.
            return torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            # PyTorch releases predating ``weights_only``.
            return torch.load(self.checkpoint_path, map_location="cpu")
        except Exception as exc:
            raise RuntimeError(
                f"failed to load OpenEB E2V checkpoint {self.checkpoint_path}"
            ) from exc

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "openeb_core_ml_event_to_video",
            "checkpoint_path": str(self.checkpoint_path),
            "device": str(self.device),
            "event_volume_depth": self.event_volume_depth,
            "normalize_num_stds": self.normalize_num_stds,
            "stateful": True,
            "timestamp_semantics": "frame state after consuming events before frame end",
        }

    def _event_volume(
        self,
        events,
        *,
        width: int,
        height: int,
        start_us: int,
        end_us: int,
    ):
        torch = self._torch
        volume = torch.zeros(
            (1, self.event_volume_depth, height, width),
            dtype=torch.float32,
            device=self.device,
        )
        if len(events) == 0:
            return volume

        x = torch.as_tensor(events["x"].astype("int64"), device=self.device)
        y = torch.as_tensor(events["y"].astype("int64"), device=self.device)
        polarity = torch.as_tensor(
            events["p"].astype("int64"), device=self.device
        )
        timestamp = torch.as_tensor(
            events["t"].astype("int64"), device=self.device
        )
        valid = (
            (x >= 0)
            & (x < width)
            & (y >= 0)
            & (y < height)
            & (timestamp >= int(start_us))
            & (timestamp < int(end_us))
        )
        x = x[valid]
        y = y[valid]
        polarity = polarity[valid]
        timestamp = timestamp[valid]
        if timestamp.numel() == 0:
            return volume

        duration_us = max(float(end_us - start_us), 1.0)
        bin_position = (
            (timestamp - int(start_us)).float()
            * float(self.event_volume_depth)
            / duration_us
            - 0.5
        )
        left_bin = torch.floor(bin_position).clamp(
            min=0, max=self.event_volume_depth - 1
        )
        right_bin = (left_bin + 1).clamp(max=self.event_volume_depth - 1)
        left_weight = (1.0 - torch.abs(left_bin - bin_position)).clamp(min=0.0)
        right_weight = 1.0 - left_weight
        signed_polarity = polarity.float() * 2.0 - 1.0

        pixels_per_bin = height * width
        spatial_index = y * width + x
        flat_volume = volume.view(-1)
        flat_volume.scatter_add_(
            0,
            left_bin.long() * pixels_per_bin + spatial_index,
            signed_polarity * left_weight,
        )
        flat_volume.scatter_add_(
            0,
            right_bin.long() * pixels_per_bin + spatial_index,
            signed_polarity * right_weight,
        )
        return volume

    def _normalize(self, image):
        torch = self._torch
        flat = image.reshape(-1)
        mean = flat.mean()
        spread = flat.std() * self.normalize_num_stds
        clipped = torch.clamp(image, min=mean - spread, max=mean + spread)
        low = clipped.min()
        high = clipped.max()
        return (clipped - low) / (high - low + 1e-5)

    def reconstruct(
        self,
        events,
        *,
        width: int,
        height: int,
        start_us: int,
        end_us: int,
    ):
        """Consume one chronological event slice and return uint8 grayscale."""

        if width <= 0 or height <= 0:
            raise ValueError("E2V source geometry must be positive")
        if end_us <= start_us:
            raise ValueError("E2V event slice end must be after its start")

        torch = self._torch
        import numpy as np
        import torch.nn.functional as functional

        volume = self._event_volume(
            events,
            width=width,
            height=height,
            start_us=start_us,
            end_us=end_us,
        )
        divisor = 2 ** max(self.num_layers, 0)
        padded_height = ((height + divisor - 1) // divisor) * divisor
        padded_width = ((width + divisor - 1) // divisor) * divisor
        if padded_height != height or padded_width != width:
            volume = functional.pad(
                volume,
                (0, padded_width - width, 0, padded_height - height),
            )
        sequence = volume.view(
            1,
            1,
            self.event_volume_depth,
            padded_height,
            padded_width,
        )

        with torch.inference_mode():
            state = self.model(sequence)
            gray = self.model.predict_gray(state).reshape(
                padded_height, padded_width
            )
            gray = self._normalize(gray)[:height, :width]
            return (
                gray.mul(255.0)
                .clamp(0.0, 255.0)
                .byte()
                .cpu()
                .numpy()
                .astype(np.uint8, copy=False)
            )
