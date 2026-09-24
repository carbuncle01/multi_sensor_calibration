"""Portable visual review artifacts for automatic LED synchronization."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _side_range(data: dict[str, Any], sensor: str, side: str) -> tuple[float, float]:
    ranges = data["roi_data"][sensor]["ranges"]
    value = ranges[0] if side == "start" else ranges[-1]
    return float(value[0]), float(value[1])


def _target_times(
    data: dict[str, Any], result: dict[str, Any], side: str
) -> dict[str, float]:
    rgb_range = _side_range(data, "rgb", side)
    evs_range = _side_range(data, "evs", side)
    center = (rgb_range[0] + rgb_range[1] + evs_range[0] + evs_range[1]) / 4.0
    candidates = [
        pair
        for pair in result.get("pairs", [])
        if rgb_range[0] <= float(pair["rgb_time_s"]) <= rgb_range[1]
        and evs_range[0] <= float(pair["evs_time_s"]) <= evs_range[1]
    ]
    if candidates:
        selected = min(
            candidates,
            key=lambda pair: abs(
                (float(pair["rgb_time_s"]) + float(pair["evs_time_s"])) / 2.0
                - center
            ),
        )
        return {
            "rgb": float(selected["rgb_time_s"]),
            "evs": float(selected["evs_time_s"]),
        }
    return {
        "rgb": (rgb_range[0] + rgb_range[1]) / 2.0,
        "evs": (evs_range[0] + evs_range[1]) / 2.0,
    }


def _nearest_preview(data: dict[str, Any], sensor: str, target_s: float) -> dict[str, Any]:
    frames = data.get("preview", {}).get(sensor, [])
    if not frames:
        raise ValueError(f"no {sensor.upper()} preview frames are available")
    return min(frames, key=lambda frame: abs(float(frame["t"]) - target_s))


def _annotated_image(image_path, roi, title, detail, cv2, np):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read preview image: {image_path}")
    x, y = int(roi["x"]), int(roi["y"])
    width, height = int(roi["width"]), int(roi["height"])
    color = (30, 230, 80)
    cv2.rectangle(image, (x, y), (x + width - 1, y + height - 1), color, 3)

    padding = max(width, height) * 2
    x0, y0 = max(0, x - padding), max(0, y - padding)
    x1 = min(image.shape[1], x + width + padding)
    y1 = min(image.shape[0], y + height + padding)
    crop = image[y0:y1, x0:x1]
    inset_size = min(240, image.shape[0] // 2, image.shape[1] // 2)
    if crop.size and inset_size >= 80:
        inset = cv2.resize(crop, (inset_size, inset_size), interpolation=cv2.INTER_NEAREST)
        ix0 = image.shape[1] - inset_size - 12
        iy0 = 12
        image[iy0 : iy0 + inset_size, ix0 : ix0 + inset_size] = inset
        cv2.rectangle(
            image,
            (ix0 - 2, iy0 - 2),
            (ix0 + inset_size + 1, iy0 + inset_size + 1),
            (255, 255, 255),
            2,
        )

    header = np.zeros((64, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, title, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.67, color, 2)
    cv2.putText(
        header,
        detail,
        (12, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (230, 230, 230),
        1,
    )
    return np.concatenate((header, image), axis=0)


def _write_html(output: Path, result: dict[str, Any], video_dir: Path | None) -> None:
    rows = []
    for side in ("start", "end"):
        for sensor in ("rgb", "evs"):
            value = result["roi"][side][sensor]
            rows.append(
                "<tr>"
                f"<td>{side}</td><td>{sensor.upper()}</td>"
                f"<td class='{html.escape(value['confidence'])}'>{html.escape(value['confidence'])}</td>"
                f"<td>{value['matched_edges']} / {value['expected_edges']}</td>"
                f"<td>{value['coverage']:.3f}</td><td>{value['edge_precision']:.3f}</td>"
                f"<td>{value['candidate_gap']:.3f}</td>"
                f"<td><code>{html.escape(str(value['roi']))}</code></td></tr>"
            )
    clock = result["clock"]
    videos = []
    if video_dir is not None:
        for name, label in (
            ("rgb_vs_overlay.mp4", "RGB / Event overlay"),
            ("overlay_polarity.mp4", "Overlay only"),
            ("polarity_only.mp4", "Events only"),
        ):
            path = video_dir / name
            if path.is_file():
                relative = Path(os.path.relpath(path, output)).as_posix()
                videos.append(
                    f"<section><h2>{label}</h2><video controls preload='metadata' src='{html.escape(relative)}'></video></section>"
                )
    document = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>LED sync review - {html.escape(str(result.get('session', 'session')))}</title>
<style>
body{{margin:24px;background:#07111d;color:#e8f0f8;font:15px system-ui,sans-serif}}
h1,h2{{margin:12px 0}} .summary,.card{{background:#0d1b2a;border:1px solid #294057;border-radius:12px;padding:16px;margin:14px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:14px}} img,video{{width:100%;background:#000;border-radius:8px}}
table{{width:100%;border-collapse:collapse}} th,td{{padding:8px;border-bottom:1px solid #294057;text-align:left}}
.high{{color:#62e6a7}} .medium{{color:#ffd166}} .low{{color:#ff7b7b}} code{{font-size:12px}}
</style></head><body>
<h1>LED sync visual review</h1><p>{html.escape(str(result.get('session', '')))}</p>
<div class="summary"><b>Overall:</b> <span class="{result['overall_confidence']}">{result['overall_confidence']}</span>
 &nbsp; <b>offset:</b> {clock['offset_at_anchor_s'] * 1000.0:+.3f} ms
 &nbsp; <b>drift:</b> {clock['drift_ppm']:+.2f} ppm
 &nbsp; <b>RMS:</b> {clock['residual_rms_s'] * 1000.0:.3f} ms
 &nbsp; <b>edges:</b> {clock['matched_edges']}</div>
<div class="grid"><section class="card"><h2>Start ROI</h2><img src="start_roi_debug.jpg"></section>
<section class="card"><h2>End ROI</h2><img src="end_roi_debug.jpg"></section></div>
<section class="card"><h2>ROI metrics</h2><table><thead><tr><th>Side</th><th>Sensor</th><th>Confidence</th><th>Edges</th><th>Coverage</th><th>Precision</th><th>Gap</th><th>ROI</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
{''.join(videos)}
</body></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")


def generate_led_sync_review(
    data_json: str | Path,
    auto_result_json: str | Path,
    output_dir: str | Path,
    *,
    video_dir: str | Path | None = None,
) -> dict[str, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("OpenCV and NumPy are required for ROI review images") from exc

    data_path = Path(data_json).resolve()
    result_path = Path(auto_result_json).resolve()
    data = _load_json(data_path)
    result = _load_json(result_path)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    generated: dict[str, dict[str, Any]] = {"start": {}, "end": {}}

    for side in ("start", "end"):
        targets = _target_times(data, result, side)
        images = []
        for sensor in ("rgb", "evs"):
            frame = _nearest_preview(data, sensor, targets[sensor])
            source = data_path.parent / str(frame["path"])
            metrics = result["roi"][side][sensor]
            detail = (
                f"t={float(frame['t']):.3f}s  confidence={metrics['confidence']}  "
                f"edges={metrics['matched_edges']}/{metrics['expected_edges']}  "
                f"precision={metrics['edge_precision']:.3f}  gap={metrics['candidate_gap']:.3f}"
            )
            annotated = _annotated_image(
                source,
                metrics["roi"],
                f"{side.upper()} / {sensor.upper()}",
                detail,
                cv2,
                np,
            )
            name = f"{side}_{sensor}_roi.jpg"
            if not cv2.imwrite(str(output / name), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise RuntimeError(f"failed to write ROI review image: {output / name}")
            images.append(annotated)
            generated[side][sensor] = {"image": name, "preview_time_s": float(frame["t"])}

        height = max(image.shape[0] for image in images)
        padded = [
            cv2.copyMakeBorder(image, 0, height - image.shape[0], 0, 0, cv2.BORDER_CONSTANT)
            for image in images
        ]
        combined = np.concatenate(padded, axis=1)
        combined_name = f"{side}_roi_debug.jpg"
        if not cv2.imwrite(str(output / combined_name), combined, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"failed to write ROI debug sheet: {output / combined_name}")
        generated[side]["combined"] = combined_name

    resolved_video = Path(video_dir).resolve() if video_dir else None
    _write_html(output, result, resolved_video)
    manifest = {
        "schema_version": 1,
        "session": result.get("session"),
        "overall_confidence": result.get("overall_confidence"),
        "source_data": str(data_path),
        "auto_result": str(result_path),
        "video_dir": str(resolved_video) if resolved_video else None,
        "generated": generated,
        "review_page": str(output / "index.html"),
    }
    (output / "review.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
