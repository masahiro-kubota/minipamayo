"""Diagnose curve-only failure modes from Stage1 per-sample eval artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ...utils.artifact_paths import bundle_dir, resolve_bundle_dir, scope_from_owner_json_path


MORPHOLOGY_STRAIGHT = "straight_through"
MORPHOLOGY_LATE_ONSET = "late_turn_onset"
MORPHOLOGY_WRONG_DIRECTION = "wrong_turn_direction"
MORPHOLOGY_LATE_DIVERGENCE = "late_divergence"
MORPHOLOGY_OSCILLATORY = "oscillatory"
MORPHOLOGY_ORDER = (
    MORPHOLOGY_STRAIGHT,
    MORPHOLOGY_LATE_ONSET,
    MORPHOLOGY_WRONG_DIRECTION,
    MORPHOLOGY_LATE_DIVERGENCE,
    MORPHOLOGY_OSCILLATORY,
)


@dataclass(frozen=True)
class Stage1ASample:
    sample_id: str
    command: str
    image_path: str
    ade_m: float
    fde_m: float
    action_mae_kappa: float
    gt_waypoints: np.ndarray
    pred_waypoints: np.ndarray
    payload: dict


@dataclass(frozen=True)
class Stage1BSample:
    sample_id: str
    command: str
    image_path: str
    ade_m: float
    fde_m: float
    max_lateral_error_m: float
    action_mae_kappa: float
    gt_waypoints: np.ndarray
    pred_waypoints: np.ndarray
    pid_ade_m: float
    pid_fde_m: float
    pid_max_lateral_error_m: float
    pid_action_mae_kappa: float
    pid_pred_waypoints: np.ndarray
    payload: dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate curve-failure diagnostics from Stage1A/Stage1B per-sample eval artifacts."
    )
    parser.add_argument("--stage1a-per-sample", type=str, required=True)
    parser.add_argument("--stage1b-per-sample", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--bucket-size", type=int, default=50)
    return parser


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _to_waypoint_array(points: list[list[float]]) -> np.ndarray:
    if not points:
        raise RuntimeError("Expected non-empty waypoint list.")
    raw = np.asarray(points, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise RuntimeError(f"Expected waypoint array shape (N, 2), got {raw.shape}.")
    return raw


def _load_stage1a_samples(path: Path) -> dict[str, Stage1ASample]:
    records = _read_jsonl(path)
    samples: dict[str, Stage1ASample] = {}
    for record in records:
        sample_id = str(record["sample_id"])
        metrics = record["metrics"]
        samples[sample_id] = Stage1ASample(
            sample_id=sample_id,
            command=str(record["command"]),
            image_path=str(record["image_path"]),
            ade_m=float(record["ade_m"]),
            fde_m=float(record["fde_m"]),
            action_mae_kappa=float(metrics["action_mae_kappa"]),
            gt_waypoints=_to_waypoint_array(record["gt_waypoints"]),
            pred_waypoints=_to_waypoint_array(record["pred_waypoints"]),
            payload=record,
        )
    return samples


def _load_stage1b_samples(path: Path) -> dict[str, Stage1BSample]:
    records = _read_jsonl(path)
    samples: dict[str, Stage1BSample] = {}
    for record in records:
        sample_id = str(record["sample_id"])
        metrics = record["metrics"]
        pid = record["pid_override"]
        pid_metrics = pid["metrics"]
        samples[sample_id] = Stage1BSample(
            sample_id=sample_id,
            command=str(record["command"]),
            image_path=str(record["image_path"]),
            ade_m=float(record["ade_m"]),
            fde_m=float(record["fde_m"]),
            max_lateral_error_m=float(record["max_lateral_error_m"]),
            action_mae_kappa=float(metrics["action_mae_kappa"]),
            gt_waypoints=_to_waypoint_array(record["gt_waypoints"]),
            pred_waypoints=_to_waypoint_array(record["pred_waypoints"]),
            pid_ade_m=float(pid["ade_m"]),
            pid_fde_m=float(pid["fde_m"]),
            pid_max_lateral_error_m=float(pid["max_lateral_error_m"]),
            pid_action_mae_kappa=float(pid_metrics["action_mae_kappa"]),
            pid_pred_waypoints=_to_waypoint_array(pid["pred_waypoints"]),
            payload=record,
        )
    return samples


def _summary_path_from_per_sample(path: Path) -> Path:
    if not path.name.endswith(".per_sample.jsonl"):
        raise RuntimeError(f"Expected `.per_sample.jsonl` suffix, got: {path}")
    return path.with_name(path.name.replace(".per_sample.jsonl", ".json"))


def _read_summary_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _scope_for_stage1a_per_sample(stage1a_per_sample_path: Path):
    return scope_from_owner_json_path(
        _summary_path_from_per_sample(stage1a_per_sample_path),
        kind="eval",
        stage="stage1",
        component="vlm_ce",
        target_component="curve_failure_diagnosis",
    )


def _default_output_dir(stage1a_per_sample_path: Path) -> Path:
    summary_path = _summary_path_from_per_sample(stage1a_per_sample_path)
    run_name = summary_path.stem
    return bundle_dir(_scope_for_stage1a_per_sample(stage1a_per_sample_path), run_name)


def _resolve_output_dir(stage1a_per_sample_path: Path, output_dir: str) -> Path:
    if output_dir:
        return Path(output_dir).resolve()
    default_output_dir = _default_output_dir(stage1a_per_sample_path)
    return resolve_bundle_dir(
        "",
        scope=_scope_for_stage1a_per_sample(stage1a_per_sample_path),
        run_name=default_output_dir.name,
    )


def _dominant_turn_sign(lateral: np.ndarray, *, threshold: float) -> int:
    max_index = int(np.argmax(np.abs(lateral)))
    dominant_value = float(lateral[max_index])
    if abs(dominant_value) < threshold:
        return 0
    return 1 if dominant_value > 0.0 else -1


def _turn_threshold(gt_lateral: np.ndarray) -> float:
    gt_peak = float(np.max(np.abs(gt_lateral)))
    return min(1.5, max(0.5, 0.1 * gt_peak))


def _turn_onset_index(lateral: np.ndarray, *, threshold: float, sign_hint: int) -> int | None:
    if sign_hint == 0:
        sign_hint = _dominant_turn_sign(lateral, threshold=threshold)
    if sign_hint == 0:
        return None
    signed = lateral * float(sign_hint)
    indices = np.flatnonzero(signed >= threshold)
    if indices.size == 0:
        return None
    return int(indices[0])


def _sign_flip_count(lateral: np.ndarray, *, threshold: float) -> int:
    significant = lateral[np.abs(lateral) >= threshold]
    if significant.size == 0:
        return 0
    signs = np.sign(significant).astype(int)
    collapsed: list[int] = []
    for sign in signs.tolist():
        if sign == 0:
            continue
        if not collapsed or collapsed[-1] != sign:
            collapsed.append(sign)
    if len(collapsed) <= 1:
        return 0
    return len(collapsed) - 1


def _displacement_series(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - gt, axis=1)


def compute_max_lateral_error(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.max(np.abs(pred[:, 1] - gt[:, 1])))


def classify_failure_morphology(sample: Stage1ASample) -> str:
    gt_lateral = sample.gt_waypoints[:, 1]
    pred_lateral = sample.pred_waypoints[:, 1]
    threshold = _turn_threshold(gt_lateral)
    gt_sign = _dominant_turn_sign(gt_lateral, threshold=threshold)
    pred_sign = _dominant_turn_sign(pred_lateral, threshold=threshold)
    pred_abs_peak = float(np.max(np.abs(pred_lateral)))
    gt_abs_peak = float(np.max(np.abs(gt_lateral)))
    gt_onset = _turn_onset_index(gt_lateral, threshold=threshold, sign_hint=gt_sign)
    pred_onset = _turn_onset_index(pred_lateral, threshold=threshold, sign_hint=gt_sign)
    sign_flips = _sign_flip_count(pred_lateral, threshold=max(0.35, threshold * 0.5))
    displacement = _displacement_series(sample.pred_waypoints, sample.gt_waypoints)
    prefix_len = max(8, int(math.ceil(len(displacement) * 0.25)))
    prefix_mean_error = float(displacement[:prefix_len].mean())

    if pred_abs_peak <= max(0.75, gt_abs_peak * 0.25):
        return MORPHOLOGY_STRAIGHT
    if gt_sign != 0 and pred_sign != 0 and gt_sign != pred_sign:
        return MORPHOLOGY_WRONG_DIRECTION
    if sign_flips >= 2:
        return MORPHOLOGY_OSCILLATORY
    if gt_onset is not None and pred_onset is not None and pred_onset - gt_onset >= 3:
        return MORPHOLOGY_LATE_ONSET
    if prefix_mean_error <= min(1.5, max(0.75, sample.fde_m * 0.35)):
        return MORPHOLOGY_LATE_DIVERGENCE
    return MORPHOLOGY_LATE_DIVERGENCE


def _to_plot_frame(points: np.ndarray) -> np.ndarray:
    forward = points[:, 0]
    lateral = points[:, 1]
    return np.stack([-lateral, forward], axis=1)


def _trajectory_limits(samples: list[Stage1ASample]) -> tuple[tuple[float, float], tuple[float, float]]:
    xs: list[float] = []
    ys: list[float] = []
    for sample in samples:
        for points in (sample.gt_waypoints, sample.pred_waypoints):
            plot_points = _to_plot_frame(points)
            xs.extend(float(value) for value in plot_points[:, 0].tolist())
            ys.extend(float(value) for value in plot_points[:, 1].tolist())
    if not xs or not ys:
        return (-1.0, 1.0), (-1.0, 1.0)
    xpad = max(0.5, (max(xs) - min(xs)) * 0.08)
    ypad = max(0.5, (max(ys) - min(ys)) * 0.08)
    return (min(xs) - xpad, max(xs) + xpad), (min(ys) - ypad, max(ys) + ypad)


def _bucket_ranges(total: int, bucket_size: int) -> tuple[slice, slice, slice]:
    bucket_size = min(bucket_size, total)
    worst = slice(total - bucket_size, total)
    midpoint = total // 2
    start = max(0, midpoint - bucket_size // 2)
    end = min(total, start + bucket_size)
    start = max(0, end - bucket_size)
    median = slice(start, end)
    best = slice(0, bucket_size)
    return worst, median, best


def _render_overlay_grid(
    *,
    samples: list[Stage1ASample],
    morphology_map: dict[str, str],
    title: str,
    output_path: Path,
) -> None:
    if not samples:
        raise RuntimeError(f"No samples provided for overlay grid: {title}")
    ncols = 5
    nrows = int(math.ceil(len(samples) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.0, nrows * 3.4), dpi=180)
    axes_array = np.atleast_1d(axes).reshape(nrows, ncols)
    xlim, ylim = _trajectory_limits(samples)
    for axis in axes_array.flatten():
        axis.axis("off")
    for axis, sample in zip(axes_array.flatten(), samples):
        axis.axis("on")
        gt = _to_plot_frame(sample.gt_waypoints)
        pred = _to_plot_frame(sample.pred_waypoints)
        axis.plot(gt[:, 0], gt[:, 1], color="#111111", linewidth=1.5)
        axis.plot(pred[:, 0], pred[:, 1], color="#d62728", linewidth=1.3, linestyle="--")
        axis.scatter(gt[0, 0], gt[0, 1], color="#2ca02c", s=10, zorder=4)
        axis.scatter(gt[-1, 0], gt[-1, 1], color="#111111", s=10, marker="s", zorder=4)
        axis.scatter(pred[-1, 0], pred[-1, 1], color="#d62728", s=10, marker="x", zorder=4)
        axis.axhline(0.0, color="#d9d9d9", linewidth=0.6)
        axis.axvline(0.0, color="#d9d9d9", linewidth=0.6)
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.18)
        axis.tick_params(labelsize=6)
        axis.set_title(
            f"{sample.sample_id}\nFDE {sample.fde_m:.2f} | {morphology_map[sample.sample_id]}",
            fontsize=6,
        )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _correlation(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) < 2 or len(values_b) < 2:
        return float("nan")
    arr_a = np.asarray(values_a, dtype=np.float64)
    arr_b = np.asarray(values_b, dtype=np.float64)
    if np.allclose(arr_a, arr_a[0]) or np.allclose(arr_b, arr_b[0]):
        return float("nan")
    return float(np.corrcoef(arr_a, arr_b)[0, 1])


def _render_kappa_scatter(
    *,
    stage1a_samples: list[Stage1ASample],
    stage1b_samples: list[Stage1BSample],
    output_path: Path,
) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), dpi=180, sharey=True)
    scatter_specs = [
        (
            axes[0],
            "Stage1A",
            [sample.action_mae_kappa for sample in stage1a_samples],
            [sample.fde_m for sample in stage1a_samples],
            "#d62728",
        ),
        (
            axes[1],
            "Stage1B",
            [sample.action_mae_kappa for sample in stage1b_samples],
            [sample.fde_m for sample in stage1b_samples],
            "#1f77b4",
        ),
    ]
    stats: dict[str, dict[str, float]] = {}
    for axis, label, xs, ys, color in scatter_specs:
        axis.scatter(xs, ys, alpha=0.65, s=12, color=color)
        axis.set_xlabel("action_mae_kappa")
        axis.set_title(label)
        axis.grid(alpha=0.22)
        corr = _correlation(xs, ys)
        stats[label.lower()] = {
            "pearson_r": corr,
            "min_fde_m": float(min(ys)),
            "max_fde_m": float(max(ys)),
        }
        axis.text(
            0.04,
            0.96,
            f"pearson r = {corr:.3f}" if not math.isnan(corr) else "pearson r = n/a",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
        )
    axes[0].set_ylabel("FDE (m)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return stats


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_bucket_manifest(path: Path, samples: list[Stage1ASample], morphology_map: dict[str, str]) -> None:
    payload = [
        {
            "sample_id": sample.sample_id,
            "command": sample.command,
            "fde_m": sample.fde_m,
            "ade_m": sample.ade_m,
            "action_mae_kappa": sample.action_mae_kappa,
            "morphology": morphology_map[sample.sample_id],
            "image_path": sample.image_path,
        }
        for sample in samples
    ]
    _write_json(path, {"samples": payload})


def _counter_payload(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter.get(key, 0)) for key in MORPHOLOGY_ORDER}


def _format_markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |" for row in rows
    ]
    return "\n".join([header, divider, *body])


def build_stage1a_vs_stage1b_rows(
    *,
    stage1a_samples: dict[str, Stage1ASample],
    stage1b_samples: dict[str, Stage1BSample],
    morphology_map: dict[str, str],
) -> list[dict]:
    rows: list[dict] = []
    common_ids = sorted(set(stage1a_samples) & set(stage1b_samples))
    for sample_id in common_ids:
        stage1a_sample = stage1a_samples[sample_id]
        stage1b_sample = stage1b_samples[sample_id]
        row = {
            "sample_id": sample_id,
            "command": stage1a_sample.command,
            "morphology": morphology_map[sample_id],
            "stage1a_fde_m": round(stage1a_sample.fde_m, 6),
            "stage1b_fde_m": round(stage1b_sample.fde_m, 6),
            "delta_fde_m": round(stage1b_sample.fde_m - stage1a_sample.fde_m, 6),
            "stage1a_ade_m": round(stage1a_sample.ade_m, 6),
            "stage1b_ade_m": round(stage1b_sample.ade_m, 6),
            "delta_ade_m": round(stage1b_sample.ade_m - stage1a_sample.ade_m, 6),
            "stage1a_max_lateral_error_m": round(
                compute_max_lateral_error(stage1a_sample.pred_waypoints, stage1a_sample.gt_waypoints),
                6,
            ),
            "stage1b_max_lateral_error_m": round(stage1b_sample.max_lateral_error_m, 6),
            "delta_max_lateral_error_m": round(
                stage1b_sample.max_lateral_error_m
                - compute_max_lateral_error(
                    stage1a_sample.pred_waypoints,
                    stage1a_sample.gt_waypoints,
                ),
                6,
            ),
            "stage1a_action_mae_kappa": round(stage1a_sample.action_mae_kappa, 6),
            "stage1b_action_mae_kappa": round(stage1b_sample.action_mae_kappa, 6),
            "delta_action_mae_kappa": round(
                stage1b_sample.action_mae_kappa - stage1a_sample.action_mae_kappa,
                6,
            ),
            "image_path": stage1a_sample.image_path,
        }
        rows.append(row)
    rows.sort(key=lambda row: float(row["delta_fde_m"]))
    return rows


def build_pid_rows(stage1b_samples: dict[str, Stage1BSample], morphology_map: dict[str, str]) -> list[dict]:
    rows: list[dict] = []
    for sample in stage1b_samples.values():
        rows.append(
            {
                "sample_id": sample.sample_id,
                "command": sample.command,
                "morphology": morphology_map[sample.sample_id],
                "canonical_fde_m": round(sample.fde_m, 6),
                "pid_fde_m": round(sample.pid_fde_m, 6),
                "delta_fde_m": round(sample.pid_fde_m - sample.fde_m, 6),
                "canonical_ade_m": round(sample.ade_m, 6),
                "pid_ade_m": round(sample.pid_ade_m, 6),
                "delta_ade_m": round(sample.pid_ade_m - sample.ade_m, 6),
                "canonical_max_lateral_error_m": round(sample.max_lateral_error_m, 6),
                "pid_max_lateral_error_m": round(sample.pid_max_lateral_error_m, 6),
                "delta_max_lateral_error_m": round(
                    sample.pid_max_lateral_error_m - sample.max_lateral_error_m,
                    6,
                ),
                "canonical_action_mae_kappa": round(sample.action_mae_kappa, 6),
                "pid_action_mae_kappa": round(sample.pid_action_mae_kappa, 6),
                "delta_action_mae_kappa": round(
                    sample.pid_action_mae_kappa - sample.action_mae_kappa,
                    6,
                ),
                "image_path": sample.image_path,
            }
        )
    rows.sort(key=lambda row: float(row["delta_fde_m"]))
    return rows


def _stage1a_bucket_samples(
    stage1a_samples: dict[str, Stage1ASample],
    bucket_size: int,
) -> tuple[list[Stage1ASample], list[Stage1ASample], list[Stage1ASample]]:
    sorted_samples = sorted(stage1a_samples.values(), key=lambda sample: sample.fde_m)
    worst_slice, median_slice, best_slice = _bucket_ranges(len(sorted_samples), bucket_size)
    worst = sorted_samples[worst_slice]
    median = sorted_samples[median_slice]
    best = sorted_samples[best_slice]
    return worst, median, best


def _select_markdown_rows(rows: list[dict], *, top_k: int) -> tuple[list[dict], list[dict]]:
    improved = rows[:top_k]
    worsened = list(reversed(rows[-top_k:]))
    return improved, worsened


def _report_text(
    *,
    label: str,
    stage1a_summary: dict,
    stage1b_summary: dict,
    morphology_overall: Counter[str],
    morphology_worst: Counter[str],
    morphology_median: Counter[str],
    stage1a_vs_stage1b_rows: list[dict],
    pid_rows: list[dict],
    scatter_stats: dict,
    bucket_size: int,
    output_dir: Path,
) -> str:
    improved_count = sum(1 for row in stage1a_vs_stage1b_rows if float(row["delta_fde_m"]) < 0.0)
    worsened_count = sum(1 for row in stage1a_vs_stage1b_rows if float(row["delta_fde_m"]) > 0.0)
    pid_improved_count = sum(1 for row in pid_rows if float(row["delta_fde_m"]) < 0.0)
    pid_worsened_count = sum(1 for row in pid_rows if float(row["delta_fde_m"]) > 0.0)
    stage1a_to_stage1b_improved, stage1a_to_stage1b_worsened = _select_markdown_rows(
        stage1a_vs_stage1b_rows,
        top_k=min(15, len(stage1a_vs_stage1b_rows)),
    )
    pid_improved, pid_worsened = _select_markdown_rows(pid_rows, top_k=min(15, len(pid_rows)))
    lines = [
        f"# Curve Failure Diagnosis Report: {label}",
        "",
        "## Summary",
        "",
        f"- `Stage1A`: `FDE={stage1a_summary.get('fde_m', float('nan')):.4f}`, "
        f"`ADE={stage1a_summary.get('ade_m', float('nan')):.4f}`, "
        f"`action_mae_kappa={stage1a_summary.get('action_mae_kappa', float('nan')):.4f}`",
        f"- `Stage1B`: `FDE={stage1b_summary.get('fde_m', float('nan')):.4f}`, "
        f"`ADE={stage1b_summary.get('ade_m', float('nan')):.4f}`, "
        f"`max_lateral_error={stage1b_summary.get('global_max_lateral_error_m', float('nan')):.4f}`",
        f"- `Stage1A -> Stage1B` per-sample delta: improved `{improved_count}`, worsened `{worsened_count}`",
        f"- `Stage1B canonical -> pid_override` per-sample delta: improved `{pid_improved_count}`, worsened `{pid_worsened_count}`",
        "",
        "## Stage1A Morphology Counts",
        "",
        f"- overall: `{json.dumps(_counter_payload(morphology_overall), ensure_ascii=False)}`",
        f"- worst {bucket_size}: `{json.dumps(_counter_payload(morphology_worst), ensure_ascii=False)}`",
        f"- median {bucket_size}: `{json.dumps(_counter_payload(morphology_median), ensure_ascii=False)}`",
        "",
        "## Kappa vs FDE",
        "",
        f"- Stage1A pearson r: `{scatter_stats['stage1a']['pearson_r']:.4f}`",
        f"- Stage1B pearson r: `{scatter_stats['stage1b']['pearson_r']:.4f}`",
        "",
        "## Generated Artifacts",
        "",
        f"- `stage1a_worst{bucket_size}_overlays.png`",
        f"- `stage1a_median{bucket_size}_overlays.png`",
        f"- `stage1a_best{bucket_size}_overlays.png`",
        "- `stage1a_vs_stage1b_join.csv`",
        "- `stage1b_pid_override_diff.csv`",
        "- `kappa_mae_vs_fde_scatter.png`",
        "",
        "## Stage1A -> Stage1B Top Improvements",
        "",
        _format_markdown_table(
            stage1a_to_stage1b_improved,
            [
                ("sample_id", "sample_id"),
                ("morphology", "morphology"),
                ("stage1a_fde_m", "stage1a_fde"),
                ("stage1b_fde_m", "stage1b_fde"),
                ("delta_fde_m", "delta_fde"),
            ],
        ),
        "",
        "## Stage1A -> Stage1B Top Regressions",
        "",
        _format_markdown_table(
            stage1a_to_stage1b_worsened,
            [
                ("sample_id", "sample_id"),
                ("morphology", "morphology"),
                ("stage1a_fde_m", "stage1a_fde"),
                ("stage1b_fde_m", "stage1b_fde"),
                ("delta_fde_m", "delta_fde"),
            ],
        ),
        "",
        "## PID Override Top Improvements",
        "",
        _format_markdown_table(
            pid_improved,
            [
                ("sample_id", "sample_id"),
                ("morphology", "morphology"),
                ("canonical_fde_m", "canonical_fde"),
                ("pid_fde_m", "pid_fde"),
                ("delta_fde_m", "delta_fde"),
            ],
        ),
        "",
        "## PID Override Top Regressions",
        "",
        _format_markdown_table(
            pid_worsened,
            [
                ("sample_id", "sample_id"),
                ("morphology", "morphology"),
                ("canonical_fde_m", "canonical_fde"),
                ("pid_fde_m", "pid_fde"),
                ("delta_fde_m", "delta_fde"),
            ],
        ),
        "",
        f"_output_dir: `{output_dir}`_",
        "",
    ]
    return "\n".join(lines)


def diagnose_curve_failures(
    *,
    stage1a_per_sample_path: Path,
    stage1b_per_sample_path: Path,
    output_dir: Path,
    label: str,
    bucket_size: int,
) -> dict:
    stage1a_samples = _load_stage1a_samples(stage1a_per_sample_path)
    stage1b_samples = _load_stage1b_samples(stage1b_per_sample_path)
    stage1a_summary = _read_summary_if_exists(_summary_path_from_per_sample(stage1a_per_sample_path))
    stage1b_summary = _read_summary_if_exists(_summary_path_from_per_sample(stage1b_per_sample_path))

    common_ids = sorted(set(stage1a_samples) & set(stage1b_samples))
    if not common_ids:
        raise RuntimeError("Stage1A and Stage1B per-sample files do not share any sample_id values.")

    morphology_map = {
        sample_id: classify_failure_morphology(stage1a_samples[sample_id]) for sample_id in common_ids
    }
    morphology_overall = Counter(morphology_map.values())

    worst_bucket, median_bucket, best_bucket = _stage1a_bucket_samples(stage1a_samples, bucket_size)
    morphology_worst = Counter(morphology_map[sample.sample_id] for sample in worst_bucket)
    morphology_median = Counter(morphology_map[sample.sample_id] for sample in median_bucket)
    morphology_best = Counter(morphology_map[sample.sample_id] for sample in best_bucket)

    output_dir.mkdir(parents=True, exist_ok=True)
    _render_overlay_grid(
        samples=worst_bucket,
        morphology_map=morphology_map,
        title=f"{label} | Stage1A worst {len(worst_bucket)} by FDE",
        output_path=output_dir / f"stage1a_worst{len(worst_bucket)}_overlays.png",
    )
    _render_overlay_grid(
        samples=median_bucket,
        morphology_map=morphology_map,
        title=f"{label} | Stage1A median {len(median_bucket)} by FDE",
        output_path=output_dir / f"stage1a_median{len(median_bucket)}_overlays.png",
    )
    _render_overlay_grid(
        samples=best_bucket,
        morphology_map=morphology_map,
        title=f"{label} | Stage1A best {len(best_bucket)} by FDE",
        output_path=output_dir / f"stage1a_best{len(best_bucket)}_overlays.png",
    )

    _write_bucket_manifest(
        output_dir / f"stage1a_worst{len(worst_bucket)}_samples.json",
        worst_bucket,
        morphology_map,
    )
    _write_bucket_manifest(
        output_dir / f"stage1a_median{len(median_bucket)}_samples.json",
        median_bucket,
        morphology_map,
    )
    _write_bucket_manifest(
        output_dir / f"stage1a_best{len(best_bucket)}_samples.json",
        best_bucket,
        morphology_map,
    )

    stage1a_vs_stage1b_rows = build_stage1a_vs_stage1b_rows(
        stage1a_samples=stage1a_samples,
        stage1b_samples=stage1b_samples,
        morphology_map=morphology_map,
    )
    stage1a_vs_stage1b_fields = [
        "sample_id",
        "command",
        "morphology",
        "stage1a_fde_m",
        "stage1b_fde_m",
        "delta_fde_m",
        "stage1a_ade_m",
        "stage1b_ade_m",
        "delta_ade_m",
        "stage1a_max_lateral_error_m",
        "stage1b_max_lateral_error_m",
        "delta_max_lateral_error_m",
        "stage1a_action_mae_kappa",
        "stage1b_action_mae_kappa",
        "delta_action_mae_kappa",
        "image_path",
    ]
    _write_csv(output_dir / "stage1a_vs_stage1b_join.csv", stage1a_vs_stage1b_rows, stage1a_vs_stage1b_fields)

    pid_rows = build_pid_rows(stage1b_samples, morphology_map)
    pid_fields = [
        "sample_id",
        "command",
        "morphology",
        "canonical_fde_m",
        "pid_fde_m",
        "delta_fde_m",
        "canonical_ade_m",
        "pid_ade_m",
        "delta_ade_m",
        "canonical_max_lateral_error_m",
        "pid_max_lateral_error_m",
        "delta_max_lateral_error_m",
        "canonical_action_mae_kappa",
        "pid_action_mae_kappa",
        "delta_action_mae_kappa",
        "image_path",
    ]
    _write_csv(output_dir / "stage1b_pid_override_diff.csv", pid_rows, pid_fields)

    scatter_stats = _render_kappa_scatter(
        stage1a_samples=[stage1a_samples[sample_id] for sample_id in common_ids],
        stage1b_samples=[stage1b_samples[sample_id] for sample_id in common_ids],
        output_path=output_dir / "kappa_mae_vs_fde_scatter.png",
    )

    summary_payload = {
        "label": label,
        "num_samples": len(common_ids),
        "stage1a_summary": stage1a_summary,
        "stage1b_summary": stage1b_summary,
        "stage1a_morphology_counts": {
            "overall": _counter_payload(morphology_overall),
            "worst_bucket": _counter_payload(morphology_worst),
            "median_bucket": _counter_payload(morphology_median),
            "best_bucket": _counter_payload(morphology_best),
        },
        "stage1a_vs_stage1b": {
            "num_improved_fde": int(sum(1 for row in stage1a_vs_stage1b_rows if float(row["delta_fde_m"]) < 0.0)),
            "num_worsened_fde": int(sum(1 for row in stage1a_vs_stage1b_rows if float(row["delta_fde_m"]) > 0.0)),
            "mean_delta_fde_m": float(
                np.mean([float(row["delta_fde_m"]) for row in stage1a_vs_stage1b_rows], dtype=np.float64)
            ),
            "mean_delta_max_lateral_error_m": float(
                np.mean(
                    [float(row["delta_max_lateral_error_m"]) for row in stage1a_vs_stage1b_rows],
                    dtype=np.float64,
                )
            ),
        },
        "stage1b_pid_override": {
            "num_improved_fde": int(sum(1 for row in pid_rows if float(row["delta_fde_m"]) < 0.0)),
            "num_worsened_fde": int(sum(1 for row in pid_rows if float(row["delta_fde_m"]) > 0.0)),
            "mean_delta_fde_m": float(
                np.mean([float(row["delta_fde_m"]) for row in pid_rows], dtype=np.float64)
            ),
            "mean_delta_max_lateral_error_m": float(
                np.mean([float(row["delta_max_lateral_error_m"]) for row in pid_rows], dtype=np.float64)
            ),
        },
        "kappa_vs_fde_scatter": scatter_stats,
    }
    _write_json(output_dir / "summary.json", summary_payload)
    report_text = _report_text(
        label=label,
        stage1a_summary=stage1a_summary,
        stage1b_summary=stage1b_summary,
        morphology_overall=morphology_overall,
        morphology_worst=morphology_worst,
        morphology_median=morphology_median,
        stage1a_vs_stage1b_rows=stage1a_vs_stage1b_rows,
        pid_rows=pid_rows,
        scatter_stats=scatter_stats,
        bucket_size=min(bucket_size, len(common_ids)),
        output_dir=output_dir,
    )
    (output_dir / "report.md").write_text(report_text, encoding="utf-8")
    return summary_payload


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    stage1a_per_sample_path = Path(args.stage1a_per_sample).resolve()
    stage1b_per_sample_path = Path(args.stage1b_per_sample).resolve()
    output_dir = _resolve_output_dir(stage1a_per_sample_path, args.output_dir)
    label = args.label.strip() or stage1a_per_sample_path.stem.replace(".per_sample", "")
    summary = diagnose_curve_failures(
        stage1a_per_sample_path=stage1a_per_sample_path,
        stage1b_per_sample_path=stage1b_per_sample_path,
        output_dir=output_dir,
        label=label,
        bucket_size=args.bucket_size,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
