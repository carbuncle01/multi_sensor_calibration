"""EVS event sources for Metavision files and ROS EventPacket bags."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol

from .io import load_yaml
from .rosbag import header_stamp_ns, iter_messages
from .transforms import TimeAnchor


@dataclass(frozen=True)
class EventBatch:
    """A structured NumPy array with x, y, p and continuous t[us] fields."""

    events: Any


class EventSource(Protocol):
    width: int
    height: int
    anchor: TimeAnchor | None

    def batches(self) -> Iterator[EventBatch]:
        ...


def _events_with_int64_time(events, unwrapped_time):
    import numpy as np

    result = np.empty(
        len(events),
        dtype=[("x", "<u2"), ("y", "<u2"), ("p", "i1"), ("t", "<i8")],
    )
    result["x"] = events["x"]
    result["y"] = events["y"]
    result["p"] = events["p"]
    result["t"] = unwrapped_time
    return result


class MetavisionFileSource:
    """Read RAW or HDF5 event files through Metavision EventsIterator."""

    def __init__(
        self,
        path: str | Path,
        *,
        metadata_path: str | Path | None = None,
        chunk_us: int = 10_000,
        source_time_at_anchor_us: float = 0.0,
        reference_time_at_anchor_s: float | None = None,
    ) -> None:
        self.path = Path(path)
        self.chunk_us = int(chunk_us)
        if self.chunk_us <= 0:
            raise ValueError("chunk_us must be positive")
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.anchor = self._load_anchor(
            source_time_at_anchor_us, reference_time_at_anchor_s
        )
        self.width = 0
        self.height = 0

    def _load_anchor(
        self,
        source_time_at_anchor_us: float,
        reference_time_at_anchor_s: float | None,
    ) -> TimeAnchor | None:
        metadata_path = self.metadata_path
        if metadata_path is None:
            candidate = Path(str(self.path) + ".metadata.yaml")
            if candidate.exists():
                metadata_path = candidate

        if reference_time_at_anchor_s is not None:
            return TimeAnchor(
                source_time_us=float(source_time_at_anchor_us),
                reference_time_s=float(reference_time_at_anchor_s),
                provenance="explicit configuration",
            )
        if metadata_path and metadata_path.exists():
            metadata = load_yaml(metadata_path)
            if "recording_start_ros_time_nanoseconds" in metadata:
                reference_time_s = (
                    float(metadata["recording_start_ros_time_nanoseconds"])
                    / 1_000_000_000.0
                )
            elif "recording_start_ros_time_sec" in metadata:
                reference_time_s = float(metadata["recording_start_ros_time_sec"])
            else:
                reference_time_s = None
            if reference_time_s is not None:
                return TimeAnchor(
                    source_time_us=float(source_time_at_anchor_us),
                    reference_time_s=reference_time_s,
                    provenance=f"OpenEB RAW sidecar: {metadata_path}",
                )
        return None

    def batches(self) -> Iterator[EventBatch]:
        try:
            from metavision_core.event_io import EventsIterator
        except ImportError as exc:
            raise RuntimeError(
                "Metavision SDK Python bindings are required for RAW/HDF5 EVS input"
            ) from exc

        iterator = EventsIterator(
            str(self.path),
            mode="delta_t",
            delta_t=self.chunk_us,
            relative_timestamps=False,
        )
        self.height, self.width = iterator.get_size()
        for events in iterator:
            if len(events) == 0:
                continue
            # HDF5 events normally already use int64. Copying to the canonical
            # dtype also keeps RAW/HDF5 and ROS backends identical downstream.
            yield EventBatch(
                _events_with_int64_time(events, events["t"].astype("int64"))
            )


class RosEventPacketSource:
    """Decode event_camera_msgs/EventPacket from a ROS 2 bag."""

    def __init__(
        self,
        bag_path: str | Path,
        topic: str,
        *,
        anchor_timestamp_source: str = "header",
    ) -> None:
        self.bag_path = Path(bag_path)
        self.topic = topic
        self.anchor_timestamp_source = anchor_timestamp_source
        if anchor_timestamp_source not in {"header", "bag"}:
            raise ValueError("anchor_timestamp_source must be header or bag")
        self.width = 0
        self.height = 0
        self.anchor: TimeAnchor | None = None
        self._rollover_offset_us = 0
        self._last_time_us: int | None = None

    def _unwrap(self, timestamps):
        import numpy as np

        values = np.asarray(timestamps, dtype=np.int64)
        if len(values) == 0:
            return values

        # event_camera_py exposes signed int32 microseconds. Calibration
        # recordings should remain below one 2^32-us wrap, but handle the first
        # signed rollover explicitly and reject non-monotonic output otherwise.
        values = np.where(values < 0, values + 2**32, values)
        values += self._rollover_offset_us
        if self._last_time_us is not None and int(values[0]) < self._last_time_us:
            if self._last_time_us - int(values[0]) > 2**31:
                self._rollover_offset_us += 2**32
                values += 2**32
            else:
                raise RuntimeError("decoded ROS event timestamps are not monotonic")
        self._last_time_us = int(values[-1])
        return values

    def batches(self) -> Iterator[EventBatch]:
        try:
            from event_camera_py import Decoder
        except ImportError as exc:
            raise RuntimeError(
                "event_camera_py is required for ROS EventPacket input"
            ) from exc

        decoder = Decoder()
        self._rollover_offset_us = 0
        self._last_time_us = None
        for item in iter_messages(self.bag_path, {self.topic}):
            message = item.message
            self.width = int(message.width)
            self.height = int(message.height)
            decoder.decode_bytes(
                str(message.encoding),
                self.width,
                self.height,
                int(message.time_base),
                bytes(message.events),
            )
            events = decoder.get_cd_events()
            if len(events) == 0:
                continue
            times = self._unwrap(events["t"])
            canonical = _events_with_int64_time(events, times)

            if self.anchor is None:
                if self.anchor_timestamp_source == "header":
                    reference_ns = header_stamp_ns(message)
                    if reference_ns is None:
                        raise RuntimeError(
                            f"{self.topic} has no header timestamp for clock anchoring"
                        )
                else:
                    reference_ns = item.bag_time_ns
                self.anchor = TimeAnchor(
                    source_time_us=float(canonical["t"][0]),
                    reference_time_s=reference_ns / 1_000_000_000.0,
                    provenance=(
                        f"first ROS EventPacket {self.anchor_timestamp_source} timestamp"
                    ),
                )
            yield EventBatch(canonical)


def build_event_source(
    source_type: str,
    source_config: dict[str, Any],
    *,
    bag_path: str | Path | None = None,
    event_file: str | Path | None = None,
) -> EventSource:
    if source_type == "metavision_file":
        path = event_file or source_config.get("path")
        if not path:
            raise ValueError("an EVS RAW/HDF5 path is required")
        return MetavisionFileSource(
            path,
            metadata_path=source_config.get("metadata_path"),
            chunk_us=int(source_config.get("chunk_us", 10_000)),
            source_time_at_anchor_us=float(
                source_config.get("source_time_at_anchor_us", 0.0)
            ),
            reference_time_at_anchor_s=source_config.get(
                "reference_time_at_anchor_s"
            ),
        )
    if source_type == "ros_events":
        if not bag_path:
            raise ValueError("--bag is required for ros_events input")
        return RosEventPacketSource(
            bag_path,
            str(source_config.get("topic", "/event_camera/events")),
            anchor_timestamp_source=str(
                source_config.get("anchor_timestamp_source", "header")
            ),
        )
    raise ValueError("event source must be metavision_file or ros_events")
