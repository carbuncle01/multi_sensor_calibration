"""Command-line interface for multi-sensor calibration."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from .calibration_images import CalibrationImage, bag_images, generated_evs_images
from .event_windows import WindowDefinition, reference_aligned_window_ends
from .io import load_yaml, sensor_config, target_config, write_yaml
from .models import ClockEstimate


def _evs_config(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("evs")
    if not isinstance(value, dict):
        raise ValueError("evs configuration is required")
    return value


def _source_config(evs: dict[str, Any], source_type: str) -> dict[str, Any]:
    sources = evs.get("sources")
    if not isinstance(sources, dict) or source_type not in sources:
        raise ValueError(f"evs.sources.{source_type} is not configured")
    value = sources[source_type]
    if not isinstance(value, dict):
        raise ValueError(f"evs.sources.{source_type} must be a mapping")
    return value


def _load_clock_models(path: str | Path | None) -> dict[str, ClockEstimate]:
    if not path:
        return {}
    value = load_yaml(path)
    models = value.get("models")
    if not isinstance(models, dict):
        raise ValueError(f"{path} does not contain clock models")
    return {
        str(name): ClockEstimate.from_dict(model)
        for name, model in models.items()
    }


def _corrected_images(
    images: Iterable[CalibrationImage],
    clock: ClockEstimate | None,
) -> Iterable[CalibrationImage]:
    if clock is None:
        yield from images
        return
    for item in images:
        yield replace(item, time_s=clock.apply(item.time_s))


def _sensor_images(
    config: dict[str, Any],
    sensor_name: str,
    *,
    bag_path: str | Path | None,
    generated_dir: str | Path | None,
    every_n: int,
    max_frames: int | None,
) -> Iterable[CalibrationImage]:
    sensor = sensor_config(config, sensor_name)
    if generated_dir:
        return generated_evs_images(
            generated_dir, every_n=every_n, max_frames=max_frames
        )
    if bag_path is None:
        raise ValueError(f"--bag or a generated image directory is required for {sensor_name}")
    topic = sensor.get("image_topic")
    if not topic:
        raise ValueError(f"sensors.{sensor_name}.image_topic is required")
    return bag_images(
        bag_path,
        str(topic),
        timestamp_source=str(sensor.get("timestamp_source", "bag")),
        every_n=every_n,
        max_frames=max_frames,
    )


def command_inspect_bag(args: argparse.Namespace) -> None:
    from .rosbag import inspect_topics

    config = load_yaml(args.config) if args.config else {}
    topics = None
    if config:
        topics = {
            str(sensor["image_topic"])
            for sensor in config.get("sensors", {}).values()
            if isinstance(sensor, dict) and sensor.get("image_topic")
        }
        evs = config.get("evs", {})
        for source in evs.get("sources", {}).values():
            if isinstance(source, dict) and source.get("topic"):
                topics.add(str(source["topic"]))
    result = {
        "bag": str(Path(args.bag).resolve()),
        "topics": inspect_topics(args.bag, topics, args.max_messages),
    }
    if args.output:
        write_yaml(args.output, result)
    else:
        import pprint

        pprint.pp(result)


def command_generate_evs(args: argparse.Namespace) -> None:
    from .evs_pipeline import (
        extract_ros_event_images,
        generate_e2v_frames,
        generate_event_frames,
        image_timestamps,
        write_frames,
    )
    from .evs_sources import build_event_source

    config = load_yaml(args.config)
    evs = _evs_config(config)
    source_type = args.source or str(evs.get("source", "metavision_file"))
    source_value = _source_config(evs, source_type)
    definition = WindowDefinition.from_dict(evs["window"])
    representation = args.representation or str(
        evs.get("representation", "polarity")
    )
    if representation == "e2v" and definition.timestamp_policy != "end":
        definition = replace(definition, timestamp_policy="end")

    if source_type == "ros_event_image":
        if representation == "e2v":
            raise ValueError(
                "E2V reconstruction requires metavision_file or ros_events input; "
                "ros_event_image is already rendered"
            )
        if not args.bag:
            raise ValueError("--bag is required for ros_event_image input")
        frames = extract_ros_event_images(
            args.bag,
            str(source_value.get("topic", "/event_camera/event_image")),
            definition,
            timestamp_source=str(source_value.get("timestamp_source", "bag")),
        )
        write_frames(
            frames,
            args.output_dir,
            source_type=source_type,
            definition=definition,
            anchor=None,
            representation="source_image",
        )
        return

    source = build_event_source(
        source_type,
        source_value,
        bag_path=args.bag,
        event_file=args.event_file,
    )
    end_times_us = None
    if definition.schedule == "reference_aligned":
        if not args.bag or not args.time_sync:
            raise ValueError(
                "reference_aligned generation requires --bag and --time-sync"
            )
        if source.anchor is None:
            probe = source.batches()
            try:
                next(probe)
            except StopIteration as exc:
                raise ValueError("EVS input contains no events") from exc
            finally:
                probe.close()
        if source.anchor is None:
            raise ValueError("EVS input has no reference clock anchor")

        clocks = _load_clock_models(args.time_sync)
        evs_name = str(evs.get("sensor_name", "evs"))
        if evs_name not in clocks:
            raise ValueError(f"time sync result has no model for {evs_name}")
        evs_clock = clocks[evs_name]
        reference_name = str(config.get("reference_sensor", "rgb"))
        reference = sensor_config(config, reference_name)
        reference_times = image_timestamps(
            args.bag,
            str(reference["image_topic"]),
            timestamp_source=str(reference.get("timestamp_source", "bag")),
        )

        def reference_to_event_time(reference_time_s: float) -> float:
            provisional_reference_s = evs_clock.inverse(reference_time_s)
            return source.anchor.to_source_us(provisional_reference_s) / 1_000_000.0

        end_times_us = reference_aligned_window_ends(
            reference_times,
            reference_to_event_time=reference_to_event_time,
            definition=definition,
        )

    representation_metadata = None
    if representation == "e2v":
        from .e2v import E2VReconstructor

        e2v_config = evs.get("e2v", {})
        if not isinstance(e2v_config, dict):
            raise ValueError("evs.e2v must be a mapping")
        checkpoint = args.e2v_checkpoint or e2v_config.get("checkpoint_path")
        device = args.e2v_device or str(e2v_config.get("device", "auto"))
        warmup_frames = (
            args.e2v_warmup_frames
            if args.e2v_warmup_frames is not None
            else int(e2v_config.get("warmup_frames", 5))
        )
        reconstructor = E2VReconstructor(
            checkpoint,
            device=device,
            normalize_num_stds=float(
                e2v_config.get("normalize_num_stds", 6.0)
            ),
        )
        frames = generate_e2v_frames(
            source,
            definition,
            reconstructor,
            end_times_us=end_times_us,
            warmup_frames=warmup_frames,
        )
        representation_metadata = reconstructor.metadata()
        representation_metadata["warmup_frames_discarded"] = warmup_frames
    else:
        frames = generate_event_frames(
            source,
            definition,
            end_times_us=end_times_us,
            representation=representation,
        )
    write_frames(
        frames,
        args.output_dir,
        source_type=source_type,
        definition=definition,
        anchor=source,
        representation=representation,
        representation_metadata=representation_metadata,
    )


def command_time_sync(args: argparse.Namespace) -> None:
    from .evs_pipeline import event_activity, image_activity
    from .evs_sources import build_event_source
    from .time_sync import estimate_clock

    config = load_yaml(args.config)
    reference_name = str(config.get("reference_sensor", "rgb"))
    reference = sensor_config(config, reference_name)
    reference_signal = image_activity(
        args.bag,
        str(reference["image_topic"]),
        timestamp_source=str(reference.get("timestamp_source", "bag")),
        max_frames=args.max_frames,
    )
    if not reference_signal:
        raise ValueError("reference camera produced no activity samples")

    estimates: dict[str, ClockEstimate] = {
        reference_name: ClockEstimate.identity(
            reference_signal[0].time_s, args.bin_ms / 1000.0
        )
    }
    signal_summary = {
        reference_name: {
            "kind": "image_difference",
            "samples": len(reference_signal),
        }
    }

    for sensor_name, sensor in config.get("sensors", {}).items():
        if sensor_name in {reference_name, str(_evs_config(config).get("sensor_name", "evs"))}:
            continue
        signal = image_activity(
            args.bag,
            str(sensor["image_topic"]),
            timestamp_source=str(sensor.get("timestamp_source", "bag")),
            max_frames=args.max_frames,
        )
        estimates[sensor_name] = estimate_clock(
            reference_signal,
            signal,
            bin_width_s=args.bin_ms / 1000.0,
            max_lag_s=args.max_lag_ms / 1000.0,
            window_s=args.window_s,
            min_correlation=args.min_correlation,
        )
        signal_summary[sensor_name] = {
            "kind": "image_difference",
            "samples": len(signal),
        }

    evs = _evs_config(config)
    evs_name = str(evs.get("sensor_name", "evs"))
    source_type = args.evs_source or str(evs.get("source", "metavision_file"))
    source_value = _source_config(evs, source_type)
    if source_type == "ros_event_image":
        evs_signal = image_activity(
            args.bag,
            str(source_value.get("topic", "/event_camera/event_image")),
            timestamp_source=str(source_value.get("timestamp_source", "bag")),
            max_frames=args.max_frames,
        )
        anchor = None
        signal_kind = "event_image_difference"
    else:
        source = build_event_source(
            source_type,
            source_value,
            bag_path=args.bag,
            event_file=args.event_file,
        )
        evs_signal = event_activity(source)
        anchor = source.anchor
        signal_kind = "event_rate"

    estimates[evs_name] = estimate_clock(
        reference_signal,
        evs_signal,
        bin_width_s=args.bin_ms / 1000.0,
        max_lag_s=args.max_lag_ms / 1000.0,
        window_s=args.window_s,
        min_correlation=args.min_correlation,
    )
    signal_summary[evs_name] = {
        "kind": signal_kind,
        "samples": len(evs_signal),
        "source_type": source_type,
        "provisional_anchor": anchor.to_dict() if anchor is not None else None,
    }

    write_yaml(
        args.output,
        {
            "schema_version": 1,
            "reference_sensor": reference_name,
            "method": "windowed_activity_cross_correlation_affine_clock",
            "models": {
                name: estimate.to_dict() for name, estimate in estimates.items()
            },
            "signals": signal_summary,
            "limitations": [
                "Software-only estimation; it does not provide hardware simultaneity.",
                "Offset includes sensor exposure/window semantics and transport latency.",
                "Validate the result on an independent common-motion recording.",
            ],
        },
    )


def command_led_sync_export(args: argparse.Namespace) -> None:
    from .evs_sources import build_event_source
    from .led_sync_export import Roi, export_led_sync_data

    config = load_yaml(args.config)
    reference_name = str(config.get("reference_sensor", "rgb"))
    reference = sensor_config(config, reference_name)
    rgb_topic = reference.get("image_topic")
    if not rgb_topic:
        raise ValueError(f"sensors.{reference_name}.image_topic is required")

    evs = _evs_config(config)
    source_type = args.evs_source or str(evs.get("source", "metavision_file"))
    if source_type != "metavision_file":
        raise ValueError(
            "led-sync-export currently requires --evs-source metavision_file"
        )
    source_value = _source_config(evs, source_type)
    source = build_event_source(
        source_type,
        source_value,
        bag_path=args.bag,
        event_file=args.event_file,
    )
    result = export_led_sync_data(
        bag_path=args.bag,
        event_source=source,
        output_dir=args.output_dir,
        rgb_topic=str(rgb_topic),
        rgb_timestamp_source=str(reference.get("timestamp_source", "bag")),
        rgb_roi=Roi.parse(args.rgb_roi),
        evs_roi=Roi.parse(args.evs_roi),
        bin_ms=args.bin_ms,
        session_name=args.session_name,
        max_rgb_frames=args.max_rgb_frames,
        preview_fps=args.preview_fps,
        preview_window_s=args.preview_window_s,
        roi_tile_size=args.roi_tile_size,
        export_roi_data=not args.no_roi_data,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def command_intrinsics(args: argparse.Namespace) -> None:
    from .intrinsics import calibrate_intrinsics, detect_checkerboard

    config = load_yaml(args.config)
    sensor = sensor_config(config, args.sensor)
    images = _sensor_images(
        config,
        args.sensor,
        bag_path=args.bag,
        generated_dir=args.images_dir,
        every_n=args.every_n,
        max_frames=args.max_frames,
    )
    observations = detect_checkerboard(images, target_config(config))
    result = calibrate_intrinsics(
        observations,
        target_config(config),
        camera_name=args.sensor,
        frame_id=str(sensor.get("frame_id", args.sensor)),
    )
    write_yaml(args.output, result)


def command_extrinsics(args: argparse.Namespace) -> None:
    from .extrinsics import calibrate_pair, pair_observations
    from .intrinsics import detect_checkerboard

    config = load_yaml(args.config)
    clocks = _load_clock_models(args.time_sync)
    reference_images = _corrected_images(
        _sensor_images(
            config,
            args.reference,
            bag_path=args.bag,
            generated_dir=args.reference_images_dir,
            every_n=args.every_n,
            max_frames=args.max_frames,
        ),
        clocks.get(args.reference),
    )
    sensor_images = _corrected_images(
        _sensor_images(
            config,
            args.sensor,
            bag_path=args.bag,
            generated_dir=args.sensor_images_dir,
            every_n=args.every_n,
            max_frames=args.max_frames,
        ),
        clocks.get(args.sensor),
    )
    target = target_config(config)
    reference_observations = detect_checkerboard(reference_images, target)
    sensor_observations = detect_checkerboard(sensor_images, target)
    pairs = pair_observations(
        reference_observations,
        sensor_observations,
        args.max_pair_delta_ms / 1000.0,
    )
    reference_sensor = sensor_config(config, args.reference)
    calibrated_sensor = sensor_config(config, args.sensor)
    result = calibrate_pair(
        pairs,
        target,
        load_yaml(args.reference_intrinsics),
        load_yaml(args.sensor_intrinsics),
        reference_frame=str(reference_sensor.get("frame_id", args.reference)),
        sensor_frame=str(calibrated_sensor.get("frame_id", args.sensor)),
        max_pair_delta_s=args.max_pair_delta_ms / 1000.0,
    )
    write_yaml(args.output, result)


def command_manual_extrinsic(args: argparse.Namespace) -> None:
    from .extrinsics import manual_extrinsic

    write_yaml(
        args.output,
        manual_extrinsic(
            parent_frame=args.parent_frame,
            child_frame=args.child_frame,
            xyz_m=args.xyz,
            rpy_rad=args.rpy,
        ),
    )


def _named_directories(values: Iterable[str]) -> dict[str, str]:
    result = {}
    for value in values:
        sensor, separator, directory = value.partition("=")
        sensor = sensor.strip()
        directory = directory.strip()
        if not separator or not sensor or not directory:
            raise ValueError(
                "--images-dir must use SENSOR=PATH format, for example "
                "--images-dir evs=result/evs_e2v"
            )
        if sensor in result:
            raise ValueError(f"--images-dir was specified more than once for {sensor}")
        result[sensor] = directory
    return result


def command_export_kalibr(args: argparse.Namespace) -> None:
    from .kalibr_export import (
        bag_frame_metadata,
        common_matches,
        export_dataset,
        generated_frame_metadata,
        parse_camera_exports,
        select_reference_frames,
    )

    config = load_yaml(args.config)
    kalibr = config.get("kalibr")
    if not isinstance(kalibr, dict):
        raise ValueError("kalibr configuration is required")
    cameras = parse_camera_exports(config)
    camera_sensors = {camera.sensor for camera in cameras}
    image_directories = _named_directories(args.images_dir)
    unknown_directories = set(image_directories) - camera_sensors
    if unknown_directories:
        raise ValueError(
            "--images-dir contains sensors not present in kalibr.cameras: "
            + ", ".join(sorted(unknown_directories))
        )

    reference_sensor = str(
        kalibr.get("reference_sensor", config.get("reference_sensor", "rgb"))
    )
    if reference_sensor not in camera_sensors:
        raise ValueError("kalibr.reference_sensor must be present in kalibr.cameras")
    if not args.time_sync:
        raise ValueError(
            "--time-sync is required so Kalibr never receives uncorrected "
            "multi-camera timestamps"
        )
    sync_document = load_yaml(args.time_sync)
    sync_reference = str(sync_document.get("reference_sensor", ""))
    if sync_reference != reference_sensor:
        raise ValueError(
            f"time sync reference is {sync_reference!r}, but Kalibr export "
            f"requires {reference_sensor!r}"
        )
    clocks = _load_clock_models(args.time_sync)
    missing_clocks = camera_sensors - set(clocks)
    if missing_clocks:
        raise ValueError(
            "time sync result has no clock model for: "
            + ", ".join(sorted(missing_clocks))
        )

    sensor_configs = {
        camera.sensor: sensor_config(config, camera.sensor) for camera in cameras
    }
    frames_by_sensor = {}
    for camera in cameras:
        if camera.sensor in image_directories:
            frames = generated_frame_metadata(
                image_directories[camera.sensor],
                clock=clocks[camera.sensor],
            )
        else:
            if not args.bag:
                raise ValueError(
                    f"--bag is required because {camera.sensor} is not supplied "
                    "with --images-dir"
                )
            topic = sensor_configs[camera.sensor].get("image_topic")
            if not topic:
                raise ValueError(
                    f"sensors.{camera.sensor}.image_topic is required or provide "
                    f"--images-dir {camera.sensor}=PATH"
                )
            frames = bag_frame_metadata(
                args.bag,
                str(topic),
                timestamp_source=str(
                    sensor_configs[camera.sensor].get("timestamp_source", "bag")
                ),
                clock=clocks[camera.sensor],
            )
        if not frames:
            raise ValueError(f"sensor {camera.sensor} produced no input frames")
        frames_by_sensor[camera.sensor] = frames

    export_rate_hz = (
        args.rate_hz
        if args.rate_hz is not None
        else float(kalibr.get("export_rate_hz", 4.0))
    )
    approximate_sync_s = (
        args.max_pair_delta_ms / 1000.0
        if args.max_pair_delta_ms is not None
        else float(kalibr.get("approximate_sync_s", 0.02))
    )
    reference_frames = select_reference_frames(
        frames_by_sensor[reference_sensor],
        export_rate_hz,
    )
    common_reference, matches = common_matches(
        reference_frames,
        frames_by_sensor,
        reference_sensor,
        approximate_sync_s,
    )
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise ValueError("--max-frames must be positive")
        common_reference = common_reference[: args.max_frames]
    minimum_frames = int(kalibr.get("minimum_frames", 20))
    if minimum_frames <= 0:
        raise ValueError("kalibr.minimum_frames must be positive")
    if len(common_reference) < minimum_frames:
        raise ValueError(
            f"only {len(common_reference)} synchronized frames were found; "
            f"kalibr.minimum_frames is {minimum_frames}. Check time-sync, "
            "generated EVS coverage, and approximate_sync_s."
        )

    export_dataset(
        args.output_dir,
        cameras,
        common_reference,
        matches,
        reference_sensor=reference_sensor,
        bag_path=args.bag,
        image_directories=image_directories,
        sensor_configs=sensor_configs,
        target=target_config(config),
        approximate_sync_s=approximate_sync_s,
        export_rate_hz=export_rate_hz,
        time_sync_source=args.time_sync,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multi-sensor-calibration",
        description="Offline EVS/RGB/thermal calibration from ROS 2 bags and event files.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-bag")
    inspect_parser.add_argument("--bag", required=True)
    inspect_parser.add_argument("--config")
    inspect_parser.add_argument("--max-messages", type=int, default=500)
    inspect_parser.add_argument("--output")
    inspect_parser.set_defaults(function=command_inspect_bag)

    generate_parser = subparsers.add_parser("generate-evs")
    generate_parser.add_argument("--config", required=True)
    generate_parser.add_argument(
        "--source",
        choices=("metavision_file", "ros_events", "ros_event_image"),
    )
    generate_parser.add_argument(
        "--representation",
        choices=("polarity", "count", "e2v"),
    )
    generate_parser.add_argument("--bag")
    generate_parser.add_argument("--event-file")
    generate_parser.add_argument("--time-sync")
    generate_parser.add_argument("--e2v-checkpoint")
    generate_parser.add_argument("--e2v-device")
    generate_parser.add_argument("--e2v-warmup-frames", type=int)
    generate_parser.add_argument("--output-dir", required=True)
    generate_parser.set_defaults(function=command_generate_evs)

    sync_parser = subparsers.add_parser("time-sync")
    sync_parser.add_argument("--config", required=True)
    sync_parser.add_argument("--bag", required=True)
    sync_parser.add_argument(
        "--evs-source",
        choices=("metavision_file", "ros_events", "ros_event_image"),
    )
    sync_parser.add_argument("--event-file")
    sync_parser.add_argument("--output", required=True)
    sync_parser.add_argument("--bin-ms", type=float, default=10.0)
    sync_parser.add_argument("--max-lag-ms", type=float, default=500.0)
    sync_parser.add_argument("--window-s", type=float, default=15.0)
    sync_parser.add_argument("--min-correlation", type=float, default=0.1)
    sync_parser.add_argument("--max-frames", type=int)
    sync_parser.set_defaults(function=command_time_sync)

    led_export_parser = subparsers.add_parser(
        "led-sync-export",
        help="Export RGB LED brightness and EVS ROI event counts for the browser UI.",
    )
    led_export_parser.add_argument("--config", required=True)
    led_export_parser.add_argument("--bag", required=True)
    led_export_parser.add_argument(
        "--evs-source",
        choices=("metavision_file",),
    )
    led_export_parser.add_argument("--event-file")
    led_export_parser.add_argument(
        "--rgb-roi",
        required=True,
        metavar="X,Y,W,H",
        help="LED rectangle in the RGB image.",
    )
    led_export_parser.add_argument(
        "--evs-roi",
        required=True,
        metavar="X,Y,W,H",
        help="LED rectangle in the EVS image.",
    )
    led_export_parser.add_argument("--bin-ms", type=float, default=1.0)
    led_export_parser.add_argument("--session-name")
    led_export_parser.add_argument("--max-rgb-frames", type=int)
    led_export_parser.add_argument(
        "--preview-fps",
        type=float,
        default=60.0,
        help="Preview images per second for the start/end windows; use 0 to disable.",
    )
    led_export_parser.add_argument(
        "--preview-window-s",
        type=float,
        default=12.0,
        help="Seconds exported at both the start and end for visual LED confirmation.",
    )
    led_export_parser.add_argument(
        "--roi-tile-size",
        type=int,
        default=16,
        help="Tile size used for browser-side dynamic ROI recalculation.",
    )
    led_export_parser.add_argument(
        "--no-roi-data",
        action="store_true",
        help="Do not export browser-side dynamic ROI data.",
    )
    led_export_parser.add_argument("--output-dir", required=True)
    led_export_parser.set_defaults(function=command_led_sync_export)

    intrinsics_parser = subparsers.add_parser("intrinsics")
    intrinsics_parser.add_argument("--config", required=True)
    intrinsics_parser.add_argument("--sensor", required=True)
    intrinsics_parser.add_argument("--bag")
    intrinsics_parser.add_argument("--images-dir")
    intrinsics_parser.add_argument("--every-n", type=int, default=1)
    intrinsics_parser.add_argument("--max-frames", type=int)
    intrinsics_parser.add_argument("--output", required=True)
    intrinsics_parser.set_defaults(function=command_intrinsics)

    extrinsics_parser = subparsers.add_parser("extrinsics")
    extrinsics_parser.add_argument("--config", required=True)
    extrinsics_parser.add_argument("--bag", required=True)
    extrinsics_parser.add_argument("--reference", required=True)
    extrinsics_parser.add_argument("--sensor", required=True)
    extrinsics_parser.add_argument("--reference-images-dir")
    extrinsics_parser.add_argument("--sensor-images-dir")
    extrinsics_parser.add_argument("--reference-intrinsics", required=True)
    extrinsics_parser.add_argument("--sensor-intrinsics", required=True)
    extrinsics_parser.add_argument("--time-sync")
    extrinsics_parser.add_argument("--max-pair-delta-ms", type=float, default=20.0)
    extrinsics_parser.add_argument("--every-n", type=int, default=1)
    extrinsics_parser.add_argument("--max-frames", type=int)
    extrinsics_parser.add_argument("--output", required=True)
    extrinsics_parser.set_defaults(function=command_extrinsics)

    manual_parser = subparsers.add_parser("manual-extrinsic")
    manual_parser.add_argument("--parent-frame", required=True)
    manual_parser.add_argument("--child-frame", required=True)
    manual_parser.add_argument("--xyz", type=float, nargs=3, required=True)
    manual_parser.add_argument("--rpy", type=float, nargs=3, required=True)
    manual_parser.add_argument("--output", required=True)
    manual_parser.set_defaults(function=command_manual_extrinsic)

    kalibr_parser = subparsers.add_parser(
        "export-kalibr",
        help="Export synchronized mono8 images for the Kalibr Docker pipeline.",
    )
    kalibr_parser.add_argument("--config", required=True)
    kalibr_parser.add_argument("--bag")
    kalibr_parser.add_argument("--time-sync", required=True)
    kalibr_parser.add_argument(
        "--images-dir",
        action="append",
        default=[],
        metavar="SENSOR=PATH",
        help="Use generated images for a sensor; may be specified more than once.",
    )
    kalibr_parser.add_argument("--rate-hz", type=float)
    kalibr_parser.add_argument("--max-pair-delta-ms", type=float)
    kalibr_parser.add_argument("--max-frames", type=int)
    kalibr_parser.add_argument("--output-dir", required=True)
    kalibr_parser.set_defaults(function=command_export_kalibr)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.function(args)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
