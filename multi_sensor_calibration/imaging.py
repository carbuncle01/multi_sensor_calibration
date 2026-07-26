"""ROS image decoding and EVS event rendering."""

from __future__ import annotations

import sys
from typing import Any


def decode_ros_image(message: Any, message_type: str):
    """Decode a ROS Image/CompressedImage into a grayscale float32 array."""

    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy and OpenCV are required for image processing") from exc

    data_bytes = bytes(message.data)
    if message_type.endswith("/CompressedImage"):
        encoded = np.frombuffer(data_bytes, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("failed to decode CompressedImage")
    else:
        width = int(message.width)
        height = int(message.height)
        step = int(message.step)
        encoding = str(message.encoding).lower()
        if encoding in {"mono8", "8uc1"}:
            image = np.frombuffer(data_bytes, dtype=np.uint8).reshape(height, step)[:, :width]
        elif encoding in {"mono16", "16uc1"}:
            dtype = np.dtype(">u2" if bool(message.is_bigendian) else "<u2")
            pixels_per_row = step // 2
            image = np.frombuffer(data_bytes, dtype=dtype).reshape(
                height, pixels_per_row
            )[:, :width]
            if (sys.byteorder == "little") == bool(message.is_bigendian):
                image = image.byteswap().view(image.dtype.newbyteorder("="))
        elif encoding in {"bgr8", "rgb8"}:
            row_bytes = width * 3
            image = np.frombuffer(data_bytes, dtype=np.uint8).reshape(
                height, step
            )[:, :row_bytes].reshape(height, width, 3)
            conversion = cv2.COLOR_BGR2GRAY if encoding == "bgr8" else cv2.COLOR_RGB2GRAY
            image = cv2.cvtColor(image, conversion)
        elif encoding in {"bgra8", "rgba8"}:
            row_bytes = width * 4
            image = np.frombuffer(data_bytes, dtype=np.uint8).reshape(
                height, step
            )[:, :row_bytes].reshape(height, width, 4)
            conversion = (
                cv2.COLOR_BGRA2GRAY if encoding == "bgra8" else cv2.COLOR_RGBA2GRAY
            )
            image = cv2.cvtColor(image, conversion)
        else:
            raise ValueError(f"unsupported ROS image encoding: {message.encoding}")

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float32)
    low, high = np.percentile(image, (1.0, 99.0))
    if high <= low:
        high = low + 1.0
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def to_uint8(gray_float):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for image processing") from exc
    return np.clip(gray_float * 255.0, 0.0, 255.0).astype(np.uint8)


def render_events(events, width: int, height: int, representation: str = "polarity"):
    """Render structured ``x,y,p,t`` events to an 8-bit image."""

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for EVS frame generation") from exc

    if representation == "polarity":
        image = np.full((height, width), 127, dtype=np.uint8)
        if len(events):
            valid = (
                (events["x"] < width)
                & (events["y"] < height)
                & (events["x"] >= 0)
                & (events["y"] >= 0)
            )
            filtered = events[valid]
            image[filtered["y"], filtered["x"]] = np.where(
                filtered["p"] > 0, 255, 0
            )
        return image

    if representation == "count":
        positive = np.zeros((height, width), dtype=np.float32)
        negative = np.zeros((height, width), dtype=np.float32)
        if len(events):
            valid = (
                (events["x"] < width)
                & (events["y"] < height)
                & (events["x"] >= 0)
                & (events["y"] >= 0)
            )
            filtered = events[valid]
            on = filtered[filtered["p"] > 0]
            off = filtered[filtered["p"] <= 0]
            np.add.at(positive, (on["y"], on["x"]), 1.0)
            np.add.at(negative, (off["y"], off["x"]), 1.0)
        scale = max(float(positive.max()), float(negative.max()), 1.0)
        return np.stack(
            [
                np.clip(negative / scale * 255.0, 0, 255),
                np.zeros_like(positive),
                np.clip(positive / scale * 255.0, 0, 255),
            ],
            axis=2,
        ).astype(np.uint8)

    raise ValueError("representation must be polarity or count")
