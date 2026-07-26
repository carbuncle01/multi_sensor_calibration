"""Direct ROS 2 bag access using the standalone rosbags package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class BagMessage:
    topic: str
    message_type: str
    bag_time_ns: int
    message: Any


def header_stamp_ns(message: Any) -> int | None:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    sec = int(getattr(stamp, "sec", 0))
    nanosec = int(getattr(stamp, "nanosec", 0))
    if sec == 0 and nanosec == 0:
        return None
    return sec * 1_000_000_000 + nanosec


def selected_time_ns(item: BagMessage, source: str) -> int:
    if source == "bag":
        return item.bag_time_ns
    if source == "header":
        stamp = header_stamp_ns(item.message)
        if stamp is None:
            raise ValueError(f"{item.topic} message has no non-zero header timestamp")
        return stamp
    raise ValueError("timestamp source must be bag or header")


def iter_messages(
    bag_path: str | Path,
    topics: set[str] | None = None,
) -> Iterator[BagMessage]:
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as exc:
        raise RuntimeError(
            "rosbags is required for ROS 2 bag input; install requirements.txt"
        ) from exc

    source = Path(bag_path)
    with AnyReader([source]) as reader:
        connections = [
            connection
            for connection in reader.connections
            if topics is None or connection.topic in topics
        ]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            try:
                message = reader.deserialize(rawdata, connection.msgtype)
            except Exception as exc:
                raise RuntimeError(
                    f"failed to deserialize {connection.topic} ({connection.msgtype}); "
                    "use an MCAP with embedded definitions or source the message package"
                ) from exc
            yield BagMessage(
                topic=connection.topic,
                message_type=connection.msgtype,
                bag_time_ns=int(timestamp),
                message=message,
            )


def inspect_topics(
    bag_path: str | Path,
    topics: set[str] | None = None,
    max_messages_per_topic: int = 500,
) -> dict[str, Any]:
    values: dict[str, dict[str, Any]] = {}
    for item in iter_messages(bag_path, topics):
        value = values.setdefault(
            item.topic,
            {
                "message_type": item.message_type,
                "sampled_messages": 0,
                "first_bag_time_ns": item.bag_time_ns,
                "last_bag_time_ns": item.bag_time_ns,
                "header_minus_bag_ns": [],
            },
        )
        if value["sampled_messages"] >= max_messages_per_topic:
            continue
        value["sampled_messages"] += 1
        value["last_bag_time_ns"] = item.bag_time_ns
        stamp = header_stamp_ns(item.message)
        if stamp is not None:
            value["header_minus_bag_ns"].append(stamp - item.bag_time_ns)

    for value in values.values():
        offsets = value.pop("header_minus_bag_ns")
        if offsets:
            offsets.sort()
            value["header_minus_bag_ns"] = {
                "minimum": offsets[0],
                "median": offsets[len(offsets) // 2],
                "maximum": offsets[-1],
            }
        else:
            value["header_minus_bag_ns"] = None
    return values
