"""Plot curve-block selections on top of Stage 1 sample positions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ...utils.artifact_paths import bundle_dir, resolve_bundle_dir, scope_from_owner_json_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot curve-only block selections on top of Stage 1 sample positions."
    )
    parser.add_argument("--curve-json", type=str, required=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional output directory. Defaults under artifacts/preprocess/stage1/curve_block_plots/canonical/.",
    )
    return parser


def _default_output_dir(curve_json_path: Path) -> Path:
    return bundle_dir(
        _scope_for_curve_json(curve_json_path, component="curve_block_plots"),
        curve_json_path.stem,
    )


def _scope_for_curve_json(curve_json_path: Path, *, component: str):
    return scope_from_owner_json_path(
        curve_json_path,
        kind="preprocess",
        stage="stage1",
        component="curve_thresholds",
        target_component=component,
    )


def _load_positions(jsonl_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_indices: list[int] = []
    xs: list[float] = []
    ys: list[float] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            ego_pose = record.get("ego_pose")
            if not isinstance(ego_pose, dict):
                raise RuntimeError(f"Record is missing ego_pose in {jsonl_path}")
            sample_indices.append(int(record["sample_index"]))
            xs.append(float(ego_pose["x"]))
            ys.append(float(ego_pose["y"]))
    return np.asarray(sample_indices), np.asarray(xs), np.asarray(ys)


def _apply_equal_axes(ax: plt.Axes, xs: np.ndarray, ys: np.ndarray) -> None:
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())
    center_x = 0.5 * (min_x + max_x)
    center_y = 0.5 * (min_y + max_y)
    half_span = max(max_x - min_x, max_y - min_y) * 0.55
    half_span = max(half_span, 5.0)
    ax.set_xlim(center_x - half_span, center_x + half_span)
    ax.set_ylim(center_y - half_span, center_y + half_span)
    ax.set_aspect("equal", adjustable="box")


def _plot_run(
    *,
    run_summary: dict,
    output_dir: Path,
) -> dict:
    jsonl_path = Path(run_summary["jsonl_path"])
    sample_indices, xs, ys = _load_positions(jsonl_path)
    blocks = run_summary["curve_blocks"]["blocks"]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
    ax.plot(xs, ys, color="#c7c7c7", linewidth=1.2, alpha=0.95, zorder=1)
    ax.scatter(xs, ys, s=5, color="#d9d9d9", alpha=0.7, zorder=2)

    cmap = plt.get_cmap("tab20")
    for block in blocks:
        start_index = int(block["start_sample_index"])
        end_index = int(block["end_sample_index"])
        mask = (sample_indices >= start_index) & (sample_indices <= end_index)
        color = cmap(block["block_index"] % cmap.N)
        ax.plot(xs[mask], ys[mask], color=color, linewidth=2.4, alpha=0.95, zorder=3)
        ax.scatter(xs[mask], ys[mask], s=9, color=color, alpha=0.95, zorder=4)
        ax.scatter(xs[mask][0], ys[mask][0], s=28, color="#2ca02c", edgecolors="black", linewidths=0.4, zorder=5)
        ax.scatter(xs[mask][-1], ys[mask][-1], s=28, color="#d62728", edgecolors="black", linewidths=0.4, zorder=5)

    _apply_equal_axes(ax, xs, ys)
    curve_blocks = run_summary["curve_blocks"]
    ax.set_title(
        f"{run_summary['run_name']}\n"
        f"curve blocks={curve_blocks['num_blocks']}, "
        f"curve sample frac={curve_blocks['block_sample_fraction']:.3f}",
        fontsize=11,
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    output_path = output_dir / f"{run_summary['run_name']}.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "run_name": run_summary["run_name"],
        "plot_path": str(output_path),
    }


def _plot_overview(*, run_summaries: list[dict], output_dir: Path) -> Path:
    num_runs = len(run_summaries)
    fig, axes = plt.subplots(1, num_runs, figsize=(8 * num_runs, 8), dpi=160)
    if num_runs == 1:
        axes = [axes]

    for ax, run_summary in zip(axes, run_summaries):
        jsonl_path = Path(run_summary["jsonl_path"])
        sample_indices, xs, ys = _load_positions(jsonl_path)
        blocks = run_summary["curve_blocks"]["blocks"]

        ax.plot(xs, ys, color="#c7c7c7", linewidth=1.2, alpha=0.95, zorder=1)
        ax.scatter(xs, ys, s=5, color="#d9d9d9", alpha=0.7, zorder=2)

        cmap = plt.get_cmap("tab20")
        for block in blocks:
            start_index = int(block["start_sample_index"])
            end_index = int(block["end_sample_index"])
            mask = (sample_indices >= start_index) & (sample_indices <= end_index)
            color = cmap(block["block_index"] % cmap.N)
            ax.plot(xs[mask], ys[mask], color=color, linewidth=2.0, alpha=0.95, zorder=3)

        _apply_equal_axes(ax, xs, ys)
        ax.set_title(
            f"{run_summary['run_name']}\n"
            f"blocks={run_summary['curve_blocks']['num_blocks']}, "
            f"frac={run_summary['curve_blocks']['block_sample_fraction']:.3f}",
            fontsize=10,
        )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    output_path = output_dir / "overview.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = build_parser().parse_args()
    curve_json_path = Path(args.curve_json).resolve()
    with curve_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    run_summaries = payload.get("runs")
    if not isinstance(run_summaries, list) or not run_summaries:
        raise RuntimeError("curve-json must contain a non-empty `runs` list.")

    output_dir = (
        resolve_bundle_dir(
            args.output_dir,
            scope=_scope_for_curve_json(curve_json_path, component="curve_block_plots"),
            run_name=curve_json_path.stem,
        )
        if args.output_dir
        else _default_output_dir(curve_json_path)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    run_outputs = [_plot_run(run_summary=run_summary, output_dir=output_dir) for run_summary in run_summaries]
    overview_path = _plot_overview(run_summaries=run_summaries, output_dir=output_dir)

    summary_path = output_dir / "plot_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "curve_json": str(curve_json_path),
                "overview_path": str(overview_path),
                "runs": run_outputs,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        json.dumps(
            {
                "curve_json": str(curve_json_path),
                "overview_path": str(overview_path),
                "runs": run_outputs,
                "plot_summary_path": str(summary_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
