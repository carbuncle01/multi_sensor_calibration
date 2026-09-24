"""Automatic LED ROI search and RGB/EVS affine-clock estimation.

The input is the schema-v3 output of ``led-sync-export``.  Four ROIs are
estimated independently: RGB/EVS at the beginning/end of the recording.
Only the exported 16 px spatial tiles are used, so no bag or RAW decoding is
required for this stage.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PERIOD_S = 2.3
TRANSITIONS = (
    (0.5, "on"),
    (0.6, "off"),
    (0.7, "on"),
    (0.8, "off"),
    (0.9, "on"),
    (1.2, "off"),
    (1.3, "on"),
    (1.4, "off"),
    (1.5, "on"),
    (1.8, "off"),
)
PHASE_BINS = 230
WINDOW_TILES = (2, 3, 4)


@dataclass(frozen=True)
class Edge:
    time_s: float
    kind: str
    strength: float
    low_s: float | None = None
    high_s: float | None = None


@dataclass
class PatternFit:
    phase_s: float
    assignments: dict[tuple[int, int], tuple[float, Edge]]
    residual_sum_s: float


@dataclass
class Candidate:
    tile_x: int
    tile_y: int
    size_tiles: int
    roi: dict[str, int]
    phase_score: float
    phase_s: float
    fit: PatternFit | None = None
    edges: list[Edge] | None = None
    detection_score: float = -1e9

    @property
    def matched(self) -> int:
        return len(self.fit.assignments) if self.fit else 0

    @property
    def precision(self) -> float:
        return self.matched / max(1, len(self.edges or ()))

    @property
    def rank(self) -> float:
        return self.detection_score * 1000.0 + self.phase_score


def _np():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for automatic LED ROI detection") from exc
    return np


def _positive_modulo(value: float, modulus: float) -> float:
    return (value % modulus + modulus) % modulus


def _side_range(meta: dict[str, Any], side: str) -> tuple[float, float]:
    ranges = meta.get("ranges")
    if not isinstance(ranges, list) or not ranges:
        raise ValueError("ROI tile metadata has no start/end ranges")
    value = ranges[0] if side == "start" else ranges[-1]
    return float(value[0]), float(value[1])


def _side_for_time(time_s: float, ranges: list[list[float]]) -> str:
    if len(ranges) < 2:
        return "start"
    first_center = (float(ranges[0][0]) + float(ranges[0][1])) / 2.0
    last_center = (float(ranges[-1][0]) + float(ranges[-1][1])) / 2.0
    return "start" if abs(time_s - first_center) <= abs(time_s - last_center) else "end"


def _load_rgb_tiles(base: Path, meta: dict[str, Any]):
    np = _np()
    path = (base / str(meta["path"])).resolve()
    tile_count = int(meta["grid_width"]) * int(meta["grid_height"])
    record_bytes = int(meta.get("record_bytes", 8 + tile_count))
    raw = np.memmap(path, mode="r", dtype=np.uint8)
    frame_count = min(int(meta.get("frame_count", 0)), raw.size // record_bytes)
    if frame_count <= 1:
        raise ValueError(f"RGB ROI tile data is empty: {path}")
    records = raw[: frame_count * record_bytes].reshape(frame_count, record_bytes)
    times = np.frombuffer(records[:, :8].copy().tobytes(), dtype="<f8")
    tiles = records[:, 8 : 8 + tile_count]
    return times, tiles


def _load_evs_records(base: Path, meta: dict[str, Any]):
    np = _np()
    path = (base / str(meta["path"])).resolve()
    dtype = np.dtype(
        [
            ("bin", "<u4"),
            ("tile", "<u2"),
            ("pos", "<u2"),
            ("neg", "<u2"),
            ("reserved", "<u2"),
        ]
    )
    records = np.fromfile(path, dtype=dtype)
    if records.size == 0:
        raise ValueError(f"EVS ROI tile data is empty: {path}")
    return records


def _empty_phase_store(tile_count: int):
    np = _np()
    return {
        side: {
            "pos": np.zeros((tile_count, PHASE_BINS), dtype=np.float32),
            "neg": np.zeros((tile_count, PHASE_BINS), dtype=np.float32),
        }
        for side in ("start", "end")
    }


def _phase_index(time_s: float) -> int:
    return int(math.floor(_positive_modulo(time_s, PERIOD_S) / PERIOD_S * PHASE_BINS)) % PHASE_BINS


def _rgb_phase_store(times, tiles, meta: dict[str, Any]):
    np = _np()
    tile_count = tiles.shape[1]
    store = _empty_phase_store(tile_count)
    means = np.asarray(tiles, dtype=np.float32).mean(axis=1)
    ranges = meta["ranges"]
    for index in range(1, len(times)):
        side = _side_for_time(float(times[index]), ranges)
        if _side_for_time(float(times[index - 1]), ranges) != side:
            continue
        if float(times[index] - times[index - 1]) > 0.06:
            continue
        delta = (
            np.asarray(tiles[index], dtype=np.float32)
            - np.asarray(tiles[index - 1], dtype=np.float32)
            - float(means[index] - means[index - 1])
        )
        phase = _phase_index(float(times[index] + times[index - 1]) / 2.0)
        store[side]["pos"][:, phase] += np.maximum(delta, 0.0)
        store[side]["neg"][:, phase] += np.maximum(-delta, 0.0)
    return store, means


def _evs_phase_store(records, meta: dict[str, Any]):
    np = _np()
    tile_count = int(meta["grid_width"]) * int(meta["grid_height"])
    store = _empty_phase_store(tile_count)
    bin_width_s = float(meta.get("bin_ms", 1.0)) / 1000.0
    ranges = meta["ranges"]
    for side in ("start", "end"):
        low, high = _side_range(meta, side)
        first_bin = max(0, int(math.floor(low / bin_width_s)))
        last_bin = int(math.ceil(high / bin_width_s))
        selected = records[(records["bin"] >= first_bin) & (records["bin"] <= last_bin)]
        if selected.size == 0:
            continue
        times = (selected["bin"].astype(np.float64) + 0.5) * bin_width_s
        phases = np.floor(np.mod(times, PERIOD_S) / PERIOD_S * PHASE_BINS).astype(np.int64)
        valid = selected["tile"] < tile_count
        tiles = selected["tile"][valid].astype(np.int64)
        phases = phases[valid]
        np.add.at(store[side]["pos"], (tiles, phases), selected["pos"][valid])
        np.add.at(store[side]["neg"], (tiles, phases), selected["neg"][valid])
    return store


def _candidate_tiles(candidate: Candidate, grid_width: int) -> list[int]:
    return [
        y * grid_width + x
        for y in range(candidate.tile_y, candidate.tile_y + candidate.size_tiles)
        for x in range(candidate.tile_x, candidate.tile_x + candidate.size_tiles)
    ]


def _phase_score(positive, negative) -> tuple[float, float]:
    np = _np()
    total = float(positive.sum() + negative.sum())
    if total <= 0.0:
        return -math.inf, 0.0
    baseline = total / (PHASE_BINS * 2.0)
    scores = np.zeros(PHASE_BINS, dtype=np.float64)
    for transition_s, kind in TRANSITIONS:
        transition_bin = int(round(transition_s / PERIOD_S * PHASE_BINS))
        correct = positive if kind == "on" else negative
        wrong = negative if kind == "on" else positive
        correct_peak = np.maximum.reduce(
            [np.roll(correct, -(transition_bin + delta)) for delta in (-1, 0, 1)]
        )
        wrong_peak = np.maximum.reduce(
            [np.roll(wrong, -(transition_bin + delta)) for delta in (-1, 0, 1)]
        )
        scores += correct_peak - 0.45 * wrong_peak
    scores -= baseline * len(TRANSITIONS) * 2.2
    scores /= math.sqrt(total) + 1.0
    best = int(np.argmax(scores))
    return float(scores[best]), best / PHASE_BINS * PERIOD_S


def _spatial_candidates(
    meta: dict[str, Any],
    phase_store: dict[str, Any],
    image_size: dict[str, Any],
) -> list[Candidate]:
    grid_width = int(meta["grid_width"])
    grid_height = int(meta["grid_height"])
    tile_size = int(meta["tile_size"])
    image_width = int(image_size.get("width", grid_width * tile_size))
    image_height = int(image_size.get("height", grid_height * tile_size))
    candidates: list[Candidate] = []
    for size_tiles in WINDOW_TILES:
        if size_tiles > grid_width or size_tiles > grid_height:
            continue
        for tile_y in range(grid_height - size_tiles + 1):
            for tile_x in range(grid_width - size_tiles + 1):
                placeholder = Candidate(
                    tile_x=tile_x,
                    tile_y=tile_y,
                    size_tiles=size_tiles,
                    roi={
                        "x": tile_x * tile_size,
                        "y": tile_y * tile_size,
                        "width": min(size_tiles * tile_size, image_width - tile_x * tile_size),
                        "height": min(size_tiles * tile_size, image_height - tile_y * tile_size),
                    },
                    phase_score=0.0,
                    phase_s=0.0,
                )
                tiles = _candidate_tiles(placeholder, grid_width)
                score, phase_s = _phase_score(
                    phase_store["pos"][tiles].sum(axis=0),
                    phase_store["neg"][tiles].sum(axis=0),
                )
                placeholder.phase_score = score
                placeholder.phase_s = phase_s
                candidates.append(placeholder)
    return sorted(candidates, key=lambda item: item.phase_score, reverse=True)[:24]


def _fit_pattern(edges: list[Edge], tolerance_s: float) -> PatternFit | None:
    if not edges:
        return None
    strongest = sorted(edges, key=lambda edge: edge.strength, reverse=True)[:160]
    phases = [
        _positive_modulo(edge.time_s - transition_s, PERIOD_S)
        for edge in strongest
        for transition_s, kind in TRANSITIONS
        if edge.kind == kind
    ]
    best: PatternFit | None = None
    best_score = -math.inf
    for phase_s in phases:
        assignments: dict[tuple[int, int], tuple[float, Edge]] = {}
        for edge in edges:
            nearest = None
            for pattern_index, (transition_s, kind) in enumerate(TRANSITIONS):
                if edge.kind != kind:
                    continue
                cycle = round((edge.time_s - phase_s - transition_s) / PERIOD_S)
                expected = phase_s + transition_s + cycle * PERIOD_S
                residual = edge.time_s - expected
                if nearest is None or abs(residual) < abs(nearest[2]):
                    nearest = (cycle, pattern_index, residual)
            if nearest is None or abs(nearest[2]) > tolerance_s:
                continue
            cycle, pattern_index, residual = nearest
            key = (cycle, pattern_index)
            previous = assignments.get(key)
            if previous is None or abs(residual) < abs(previous[0]):
                assignments[key] = (residual, edge)
        if not assignments:
            continue
        residual_sum = sum(abs(value[0]) for value in assignments.values())
        strength = sum(math.log1p(max(0.0, value[1].strength)) for value in assignments.values())
        score = len(assignments) * 1000.0 - residual_sum / max(1e-6, tolerance_s) + strength * 0.01
        if score > best_score:
            best_score = score
            best = PatternFit(phase_s=phase_s, assignments=assignments, residual_sum_s=residual_sum)
    return best


def _detection_quality(fit: PatternFit | None, edges: list[Edge], tolerance_s: float) -> float:
    if fit is None or not edges:
        return -1e9
    matched = len(fit.assignments)
    precision = matched / len(edges)
    extras = max(0, len(edges) - matched)
    return matched * 10.0 + precision * 25.0 - extras * 0.2 - fit.residual_sum_s / max(1e-6, tolerance_s)


def _group_rgb_edges(times, values, threshold: float) -> list[Edge]:
    raw: list[Edge] = []
    for index in range(1, len(times)):
        delta = float(values[index] - values[index - 1])
        if abs(delta) < threshold:
            continue
        raw.append(
            Edge(
                time_s=float(times[index] + times[index - 1]) / 2.0,
                kind="on" if delta >= 0.0 else "off",
                strength=abs(delta),
                low_s=float(times[index - 1]),
                high_s=float(times[index]),
            )
        )
    grouped: list[Edge] = []
    for edge in raw:
        if not grouped or edge.time_s - grouped[-1].time_s > 0.04:
            grouped.append(edge)
        elif edge.strength > grouped[-1].strength:
            grouped[-1] = edge
    return grouped


def _evaluate_rgb_candidates(candidates, side, times, tiles, means, meta, rgb_fps):
    np = _np()
    grid_width = int(meta["grid_width"])
    low, high = _side_range(meta, side)
    selected = (times >= low) & (times <= high)
    selected_times = times[selected]
    selected_means = means[selected]
    selected_tiles = tiles[selected]
    tolerance = max(0.035, 1.5 / max(1.0, float(rgb_fps)))
    for candidate in candidates:
        indices = _candidate_tiles(candidate, grid_width)
        values = np.asarray(selected_tiles[:, indices], dtype=np.float32).mean(axis=1) - selected_means
        magnitudes = np.abs(np.diff(values))
        if magnitudes.size == 0:
            continue
        center = float(np.median(magnitudes))
        mad = float(np.median(np.abs(magnitudes - center)))
        thresholds = {
            max(0.05, center + 4.0 * 1.4826 * mad),
            max(0.05, float(np.quantile(magnitudes, 0.90))),
            max(0.05, float(np.quantile(magnitudes, 0.95))),
        }
        for threshold in thresholds:
            edges = _group_rgb_edges(selected_times, values, threshold)
            fit = _fit_pattern(edges, tolerance)
            quality = _detection_quality(fit, edges, tolerance)
            if quality > candidate.detection_score:
                candidate.fit = fit
                candidate.edges = edges
                candidate.detection_score = quality
    return sorted(candidates, key=lambda item: item.rank, reverse=True)


def _group_evs_edges(positive, negative, first_bin, bin_width_s, threshold) -> list[Edge]:
    np = _np()
    indices = np.flatnonzero(positive + negative >= threshold)
    raw = [
        Edge(
            time_s=(first_bin + int(index) + 0.5) * bin_width_s,
            kind="on" if positive[index] >= negative[index] else "off",
            strength=float(positive[index] + negative[index]),
        )
        for index in indices.tolist()
    ]
    groups: list[list[Edge]] = []
    for edge in raw:
        if not groups or edge.time_s - groups[-1][-1].time_s > 0.008:
            groups.append([edge])
        else:
            groups[-1].append(edge)
    return [max(group, key=lambda edge: edge.strength) for group in groups]


def _evaluate_evs_candidates(candidates, side, records, meta):
    np = _np()
    grid_width = int(meta["grid_width"])
    bin_width_s = float(meta.get("bin_ms", 1.0)) / 1000.0
    low, high = _side_range(meta, side)
    first_bin = max(0, int(math.floor(low / bin_width_s)))
    last_bin = int(math.ceil(high / bin_width_s))
    selected = records[(records["bin"] >= first_bin) & (records["bin"] <= last_bin)]
    length = max(1, last_bin - first_bin + 1)
    for candidate in candidates:
        wanted = np.asarray(_candidate_tiles(candidate, grid_width), dtype=np.uint16)
        local = selected[np.isin(selected["tile"], wanted)]
        positive = np.zeros(length, dtype=np.uint32)
        negative = np.zeros(length, dtype=np.uint32)
        if local.size:
            indices = local["bin"].astype(np.int64) - first_bin
            np.add.at(positive, indices, local["pos"])
            np.add.at(negative, indices, local["neg"])
        totals = positive + negative
        thresholds = {max(2.0, float(np.quantile(totals, level))) for level in (0.97, 0.985, 0.995)}
        for threshold in thresholds:
            edges = _group_evs_edges(positive, negative, first_bin, bin_width_s, threshold)
            fit = _fit_pattern(edges, 0.035)
            quality = _detection_quality(fit, edges, 0.035)
            if quality > candidate.detection_score:
                candidate.fit = fit
                candidate.edges = edges
                candidate.detection_score = quality
    return sorted(candidates, key=lambda item: item.rank, reverse=True)


def _roi_iou(first: dict[str, int], second: dict[str, int]) -> float:
    x0, y0 = max(first["x"], second["x"]), max(first["y"], second["y"])
    x1 = min(first["x"] + first["width"], second["x"] + second["width"])
    y1 = min(first["y"] + first["height"], second["y"] + second["height"])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    union = first["width"] * first["height"] + second["width"] * second["height"] - intersection
    return intersection / union if union > 0 else 0.0


def _expected_edges(time_range: tuple[float, float], phase_s: float) -> int:
    count = 0
    for transition_s, _ in TRANSITIONS:
        first = math.ceil((time_range[0] - phase_s - transition_s) / PERIOD_S)
        last = math.floor((time_range[1] - phase_s - transition_s) / PERIOD_S)
        count += max(0, last - first + 1)
    return count


def _finalize_candidate(evaluated: list[Candidate], time_range):
    if not evaluated:
        raise ValueError("automatic ROI search produced no candidates")
    best = evaluated[0]
    second = next((item for item in evaluated[1:] if _roi_iou(item.roi, best.roi) < 0.15), None)
    expected = max(1, _expected_edges(time_range, best.fit.phase_s if best.fit else best.phase_s))
    coverage = best.matched / expected
    gap = best.detection_score - second.detection_score if second else 1e9
    confidence = "low"
    if best.matched >= 18 and coverage >= 0.35 and best.precision >= 0.45 and gap >= 15.0:
        confidence = "high"
    elif best.matched >= 9 and coverage >= 0.18 and best.precision >= 0.20 and gap >= 3.0:
        confidence = "medium"
    return best, {
        "roi": best.roi,
        "confidence": confidence,
        "matched_edges": best.matched,
        "expected_edges": expected,
        "coverage": coverage,
        "edge_precision": best.precision,
        "candidate_gap": gap,
        "phase_s": best.fit.phase_s if best.fit else best.phase_s,
    }


def _pair_side(evs: Candidate, rgb: Candidate, max_delta_s: float = 0.2):
    if evs.fit is None or rgb.fit is None:
        return []
    best: list[tuple[Edge, Edge]] = []
    best_distance = math.inf
    for cycle_shift in range(-3, 4):
        pairs = []
        distance = 0.0
        for (cycle, pattern_index), (_, evs_edge) in evs.fit.assignments.items():
            rgb_value = rgb.fit.assignments.get((cycle + cycle_shift, pattern_index))
            if rgb_value is None:
                continue
            rgb_edge = rgb_value[1]
            delta = abs(rgb_edge.time_s - evs_edge.time_s)
            if delta > max_delta_s:
                continue
            distance += delta
            pairs.append((evs_edge, rgb_edge))
        if len(pairs) > len(best) or (len(pairs) == len(best) and distance < best_distance):
            best = pairs
            best_distance = distance
    return best


def _clock_fit(pairs: list[tuple[Edge, Edge]]):
    np = _np()
    if len(pairs) < 4:
        raise ValueError(f"at least 4 matched LED edges are required; got {len(pairs)}")
    evs_times = np.asarray([pair[0].time_s for pair in pairs], dtype=np.float64)
    offsets = np.asarray([pair[1].time_s - pair[0].time_s for pair in pairs], dtype=np.float64)
    anchor = float(evs_times.mean())
    x = evs_times - anchor
    offset = float(offsets.mean())
    denominator = float(np.dot(x, x))
    drift = float(np.dot(x, offsets - offset) / denominator) if denominator > 0.0 else 0.0
    residuals = offsets - (offset + drift * x)
    rms = float(math.sqrt(float(np.mean(residuals * residuals))))
    return anchor, offset, drift, rms, residuals


def auto_led_sync(data_json: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(data_json).resolve()
    data = json.loads(source.read_text(encoding="utf-8"))
    meta = data.get("meta")
    roi_data = data.get("roi_data")
    if not isinstance(meta, dict) or not isinstance(roi_data, dict):
        raise ValueError("led_sync_data.json has no meta/roi_data; re-export with ROI data")
    rgb_meta, evs_meta = roi_data.get("rgb"), roi_data.get("evs")
    if not isinstance(rgb_meta, dict) or not isinstance(evs_meta, dict):
        raise ValueError("led_sync_data.json has incomplete RGB/EVS ROI metadata")

    times, rgb_tiles = _load_rgb_tiles(source.parent, rgb_meta)
    evs_records = _load_evs_records(source.parent, evs_meta)
    rgb_phases, rgb_means = _rgb_phase_store(times, rgb_tiles, rgb_meta)
    evs_phases = _evs_phase_store(evs_records, evs_meta)
    selected: dict[str, dict[str, Candidate]] = {"start": {}, "end": {}}
    result_rois: dict[str, dict[str, Any]] = {"start": {}, "end": {}}

    for side in ("start", "end"):
        rgb_candidates = _spatial_candidates(rgb_meta, rgb_phases[side], meta["rgb_image_size"])
        rgb_evaluated = _evaluate_rgb_candidates(
            rgb_candidates, side, times, rgb_tiles, rgb_means, rgb_meta, meta.get("rgbFps", 60.0)
        )
        selected[side]["rgb"], result_rois[side]["rgb"] = _finalize_candidate(
            rgb_evaluated, _side_range(rgb_meta, side)
        )
        evs_candidates = _spatial_candidates(evs_meta, evs_phases[side], meta["evs_image_size"])
        evs_evaluated = _evaluate_evs_candidates(evs_candidates, side, evs_records, evs_meta)
        selected[side]["evs"], result_rois[side]["evs"] = _finalize_candidate(
            evs_evaluated, _side_range(evs_meta, side)
        )

    pairs = []
    for side in ("start", "end"):
        pairs.extend(_pair_side(selected[side]["evs"], selected[side]["rgb"]))
    pairs.sort(key=lambda pair: pair[0].time_s)
    anchor, offset, drift, rms, residuals = _clock_fit(pairs)
    origin = float(meta["time_origin_reference_s"])
    rgb_period = 1.0 / max(1.0, float(meta.get("rgbFps", 60.0)))
    evs_bin = float(meta.get("evs_bin_ms", 1.0)) / 1000.0
    all_confidences = [result_rois[side][sensor]["confidence"] for side in ("start", "end") for sensor in ("rgb", "evs")]
    overall = "low" if "low" in all_confidences else "medium" if "medium" in all_confidences else "high"

    time_sync = {
        "schema_version": 1,
        "reference_sensor": "rgb",
        "method": "automatic_known_led_pattern_alignment",
        "models": {
            "rgb": {
                "anchor_sensor_time_s": origin + anchor,
                "offset_at_anchor_s": 0.0,
                "drift": 0.0,
                "correlation": 1.0,
                "window_count": len(pairs),
                "bin_width_s": rgb_period,
            },
            "evs": {
                "anchor_sensor_time_s": origin + anchor,
                "offset_at_anchor_s": offset,
                "drift": drift,
                "correlation": 0.0,
                "window_count": len(pairs),
                "bin_width_s": evs_bin,
            },
        },
        "quality": {
            "residual_rms_s": rms,
            "matched_edges": len(pairs),
            "drift_ppm": drift * 1_000_000.0,
            "led_pattern_used": True,
            "roi_confidence": overall,
        },
    }
    result = {
        "schema_version": 1,
        "session": meta.get("session", source.parent.name),
        "source": str(source),
        "method": "sliding_window_known_led_pattern",
        "window_sizes_px": [
            int(rgb_meta["tile_size"]) * value for value in WINDOW_TILES
        ],
        "roi": result_rois,
        "overall_confidence": overall,
        "clock": {
            "anchor_relative_s": anchor,
            "offset_at_anchor_s": offset,
            "drift": drift,
            "drift_ppm": drift * 1_000_000.0,
            "residual_rms_s": rms,
            "matched_edges": len(pairs),
        },
        "pairs": [
            {
                "kind": evs.kind,
                "evs_time_s": evs.time_s,
                "rgb_time_s": rgb.time_s,
                "rgb_interval_s": [rgb.low_s, rgb.high_s],
                "residual_s": float(residuals[index]),
            }
            for index, (evs, rgb) in enumerate(pairs)
        ],
    }
    return time_sync, result
