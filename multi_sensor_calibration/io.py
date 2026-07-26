"""Configuration and result serialization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read calibration configuration") from exc

    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain a YAML mapping")
    return value


def write_yaml(path: str | Path, value: dict[str, Any]) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to write calibration results") from exc

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(value, stream, sort_keys=False, allow_unicode=True)
    temporary.replace(destination)


def sensor_config(config: dict[str, Any], sensor_name: str) -> dict[str, Any]:
    sensors = config.get("sensors")
    if not isinstance(sensors, dict) or sensor_name not in sensors:
        raise ValueError(f"sensor {sensor_name!r} is not defined in the configuration")
    value = sensors[sensor_name]
    if not isinstance(value, dict):
        raise ValueError(f"sensor {sensor_name!r} configuration must be a mapping")
    return value


def target_config(config: dict[str, Any]) -> dict[str, Any]:
    target = config.get("target")
    if not isinstance(target, dict):
        raise ValueError("target configuration is required")
    if target.get("type", "checkerboard") != "checkerboard":
        raise ValueError("the current MVP supports target.type=checkerboard only")
    for key in ("columns", "rows", "square_size_m"):
        if key not in target:
            raise ValueError(f"target.{key} is required")
    return target
