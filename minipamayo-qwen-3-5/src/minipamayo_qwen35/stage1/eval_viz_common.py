"""Shared helpers for Stage 1 eval visualization entrypoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(values)
    probs = np.linspace(1.0 / len(ordered), 1.0, len(ordered))
    return ordered, probs


def nested_get(payload: dict[str, Any], dotted_key: str) -> Any:
    value: Any = payload
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(dotted_key)
        value = value[key]
    return value


def metric_array(samples: list[dict[str, Any]], dotted_key: str) -> np.ndarray:
    return np.asarray([float(nested_get(sample, dotted_key)) for sample in samples], dtype=np.float64)


def select_rank_groups(
    samples: list[dict[str, Any]],
    *,
    count: int,
    metric_key: str = "fde_m",
) -> dict[str, list[dict[str, Any]]]:
    ordered = sorted(samples, key=lambda sample: float(nested_get(sample, metric_key)))
    count = min(count, len(ordered))
    median_start = max((len(ordered) // 2) - (count // 2), 0)
    median_end = min(median_start + count, len(ordered))
    median_start = max(0, median_end - count)
    return {
        "best": ordered[:count],
        "median": ordered[median_start:median_end],
        "worst": list(reversed(ordered[-count:])),
    }


def trajectory_limits(
    samples: list[dict[str, Any]],
    *,
    waypoint_keys: list[str],
) -> tuple[tuple[float, float], tuple[float, float]]:
    xs: list[float] = []
    ys: list[float] = []
    for sample in samples:
        for key in waypoint_keys:
            try:
                points = nested_get(sample, key)
            except KeyError:
                continue
            if not points:
                continue
            for point in points:
                xs.append(float(point[0]))
                ys.append(float(point[1]))
    if not xs or not ys:
        raise RuntimeError("Could not derive trajectory limits from the provided samples.")
    xpad = max(0.5, (max(xs) - min(xs)) * 0.08)
    ypad = max(0.5, (max(ys) - min(ys)) * 0.08)
    return (min(xs) - xpad, max(xs) + xpad), (min(ys) - ypad, max(ys) + ypad)
