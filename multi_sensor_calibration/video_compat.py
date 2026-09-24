"""Video container normalization for macOS and browser playback."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


def _gstreamer_has_x264() -> bool:
    inspect = shutil.which("gst-inspect-1.0")
    if inspect is None:
        return False
    return _run([inspect, "x264enc"]).returncode == 0


def _file_uri(path: Path) -> str:
    return "file://" + quote(str(path), safe="/")


def make_macos_compatible_mp4(path: str | Path) -> dict[str, str]:
    """Atomically replace an OpenCV MP4 with H.264/yuv420p/fast-start MP4.

    The Isaac container installs GStreamer's ugly plugin set, which provides
    x264enc even when the separately built FFmpeg intentionally omits GPL
    encoders. FFmpeg/libx264 remains a useful fallback on other hosts.
    """

    source = Path(path).resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"video is missing or empty: {source}")
    target = source.with_name(f".{source.stem}.macos{source.suffix}")
    target.unlink(missing_ok=True)
    attempts: list[tuple[str, list[str]]] = []

    gst = shutil.which("gst-launch-1.0")
    if gst is not None and _gstreamer_has_x264():
        attempts.append(
            (
                "gstreamer-x264",
                [
                    gst,
                    "-q",
                    "-e",
                    "uridecodebin",
                    f"uri={_file_uri(source)}",
                    "!",
                    "queue",
                    "!",
                    "videoconvert",
                    "!",
                    "video/x-raw,format=I420",
                    "!",
                    "x264enc",
                    "speed-preset=medium",
                    "bitrate=8000",
                    "key-int-max=120",
                    "!",
                    "video/x-h264,profile=high",
                    "!",
                    "h264parse",
                    "!",
                    "mp4mux",
                    "faststart=true",
                    "!",
                    "filesink",
                    f"location={target}",
                ],
            )
        )

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        attempts.append(
            (
                "ffmpeg-libx264",
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source),
                    "-map",
                    "0:v:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(target),
                ],
            )
        )

    failures = []
    for encoder, command in attempts:
        result = _run(command)
        if result.returncode == 0 and target.is_file() and target.stat().st_size > 0:
            target.replace(source)
            return {
                "codec": "h264",
                "pixel_format": "yuv420p",
                "container": "mp4",
                "encoder": encoder,
                "faststart": "true",
            }
        target.unlink(missing_ok=True)
        message = result.stdout.strip().replace("\n", " ")
        failures.append(f"{encoder}: {message[-500:] or 'failed'}")

    if not attempts:
        failures.append("neither GStreamer/x264enc nor FFmpeg is available")
    raise RuntimeError(
        "could not create a macOS-compatible H.264 MP4; " + "; ".join(failures)
    )
