"""Visualize canonical Stage 1B eval outputs with local plots and W&B logging."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ...inspector.manifests import update_manifest_plots
from ...utils.preflight import init_required_online_wandb
from ..eval_viz_common import (
    cdf,
    load_json,
    load_jsonl,
    metric_array,
    nested_get,
    select_rank_groups,
    trajectory_limits,
    write_json,
)
from ..stage1_json_cli import parse_stage1_json_only_args

CONFIG_PATH_KEYS = {
    "summary_json",
    "per_sample_jsonl",
    "output_dir",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize canonical Stage 1B eval outputs.")
    parser.add_argument("--config-json", type=str, default="")
    parser.add_argument("--summary-json", type=str, default="")
    parser.add_argument("--per-sample-jsonl", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--wandb-project", type=str, default="")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--overlay-count", type=int, default=16)
    parser.add_argument("--worst-table-count", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parse_stage1_json_only_args(
        parser,
        path_keys=CONFIG_PATH_KEYS,
        error_message="Stage 1B visualization accepts only --config-json. Put all settings in the JSON file.",
    )
    for key in ["summary_json", "per_sample_jsonl", "output_dir", "wandb_project", "wandb_run_name"]:
        if not str(getattr(args, key, "")):
            raise RuntimeError(f"`{key}` must be defined in the config JSON.")
    if int(args.overlay_count) <= 0:
        raise RuntimeError("`overlay_count` must be > 0.")
    if int(args.worst_table_count) <= 0:
        raise RuntimeError("`worst_table_count` must be > 0.")
    if int(args.dpi) <= 0:
        raise RuntimeError("`dpi` must be > 0.")
    return args


def _ensure_plot_fields(samples: list[dict[str, Any]]) -> None:
    required_top_level = {
        "sample_id",
        "sample_index",
        "image_path",
        "command",
        "gt_action",
        "pred_action",
        "gt_waypoints",
        "pred_waypoints",
        "ade_m",
        "fde_m",
        "max_lateral_error_m",
    }
    required_metrics = {
        "action_mae_accel",
        "action_mae_kappa",
        "ade_m",
        "fde_m",
        "max_lateral_error_m",
    }
    if not samples:
        raise RuntimeError("Per-sample JSONL is empty.")
    for key in required_top_level:
        if key not in samples[0]:
            raise RuntimeError(f"Per-sample JSONL is missing required field `{key}`.")
    metrics = samples[0].get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError("Per-sample JSONL is missing `metrics` objects.")
    for key in required_metrics:
        if key not in metrics:
            raise RuntimeError(f"Per-sample JSONL metrics are missing required field `{key}`.")


def _pid_enabled(samples: list[dict[str, Any]], summary: dict[str, Any]) -> bool:
    return bool(summary.get("pid_override_enabled", False)) and all(
        isinstance(sample.get("pid_override"), dict) for sample in samples
    )


def _save_histograms(samples: list[dict[str, Any]], output_path: Path, *, dpi: int) -> None:
    metrics = {
        "ADE (m)": metric_array(samples, "ade_m"),
        "FDE (m)": metric_array(samples, "fde_m"),
        "Kappa MAE": metric_array(samples, "metrics.action_mae_kappa"),
        "Max Lateral (m)": metric_array(samples, "max_lateral_error_m"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=dpi)
    for ax, (title, values) in zip(axes.flat, metrics.items()):
        ax.hist(values, bins=32, color="#1f77b4", alpha=0.82, edgecolor="white")
        ax.axvline(
            values.mean(),
            color="#d62728",
            linestyle="--",
            linewidth=1.6,
            label=f"mean={values.mean():.3f}",
        )
        ax.axvline(
            np.median(values),
            color="#2ca02c",
            linestyle=":",
            linewidth=1.6,
            label=f"median={np.median(values):.3f}",
        )
        ax.set_title(title)
        ax.set_xlabel(title)
        ax.set_ylabel("Count")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Stage1B Curve Eval Distributions", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_cdfs(samples: list[dict[str, Any]], output_path: Path, *, dpi: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=dpi)
    metric_specs = [
        ("ADE (m)", metric_array(samples, "ade_m")),
        ("FDE (m)", metric_array(samples, "fde_m")),
        ("Max Lateral (m)", metric_array(samples, "max_lateral_error_m")),
    ]
    for ax, (title, values) in zip(axes, metric_specs):
        ordered, probs = cdf(values)
        ax.plot(ordered, probs, color="#1f77b4", linewidth=2.0)
        ax.set_title(f"{title} CDF")
        ax.set_xlabel(title)
        ax.set_ylabel("CDF")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_scatter(samples: list[dict[str, Any]], output_path: Path, *, dpi: int) -> None:
    fde = metric_array(samples, "fde_m")
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.5), dpi=dpi)
    scatter_specs = [
        ("Kappa MAE", metric_array(samples, "metrics.action_mae_kappa"), "#d62728"),
        ("Accel MAE", metric_array(samples, "metrics.action_mae_accel"), "#8c564b"),
        ("Max Lateral (m)", metric_array(samples, "max_lateral_error_m"), "#1f77b4"),
    ]
    for ax, (xlabel, xvals, color) in zip(axes, scatter_specs):
        ax.scatter(xvals, fde, s=18, alpha=0.55, color=color, edgecolors="none")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("FDE (m)")
        ax.set_title(f"{xlabel} vs FDE")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_pid_comparison(
    samples: list[dict[str, Any]],
    output_path: Path,
    *,
    dpi: int,
    pid_enabled: bool,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.5), dpi=dpi)
    if not pid_enabled:
        for ax in axes:
            ax.axis("off")
        fig.suptitle("Stage1B PID Comparison (PID override disabled)", fontsize=14)
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return

    canonical_fde = metric_array(samples, "fde_m")
    pid_fde = metric_array(samples, "pid_override.fde_m")
    canonical_lat = metric_array(samples, "max_lateral_error_m")
    pid_lat = metric_array(samples, "pid_override.max_lateral_error_m")
    delta_fde = pid_fde - canonical_fde

    limit_fde = max(float(canonical_fde.max()), float(pid_fde.max()))
    axes[0].scatter(canonical_fde, pid_fde, s=18, alpha=0.55, color="#1f77b4", edgecolors="none")
    axes[0].plot([0.0, limit_fde], [0.0, limit_fde], color="#111111", linestyle="--", linewidth=1.2)
    axes[0].set_xlabel("Canonical FDE (m)")
    axes[0].set_ylabel("PID FDE (m)")
    axes[0].set_title("Canonical vs PID FDE")
    axes[0].grid(alpha=0.25)

    limit_lat = max(float(canonical_lat.max()), float(pid_lat.max()))
    axes[1].scatter(canonical_lat, pid_lat, s=18, alpha=0.55, color="#2ca02c", edgecolors="none")
    axes[1].plot([0.0, limit_lat], [0.0, limit_lat], color="#111111", linestyle="--", linewidth=1.2)
    axes[1].set_xlabel("Canonical Max Lateral (m)")
    axes[1].set_ylabel("PID Max Lateral (m)")
    axes[1].set_title("Canonical vs PID Max Lateral")
    axes[1].grid(alpha=0.25)

    axes[2].hist(delta_fde, bins=32, color="#9467bd", alpha=0.82, edgecolor="white")
    axes[2].axvline(0.0, color="#111111", linestyle="--", linewidth=1.2)
    axes[2].set_xlabel("PID FDE - Canonical FDE (m)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("PID FDE Delta")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_sample_order(
    samples: list[dict[str, Any]],
    output_path: Path,
    *,
    dpi: int,
    pid_enabled: bool,
) -> None:
    x = np.arange(len(samples))
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), dpi=dpi, sharex=True)
    axes[0].plot(x, metric_array(samples, "fde_m"), label="Canonical FDE", color="#d62728", linewidth=1.4)
    axes[0].plot(x, metric_array(samples, "ade_m"), label="Canonical ADE", color="#1f77b4", linewidth=1.2)
    if pid_enabled:
        axes[0].plot(
            x,
            metric_array(samples, "pid_override.fde_m"),
            label="PID FDE",
            color="#17becf",
            linewidth=1.1,
        )
    axes[0].set_ylabel("Meters")
    axes[0].set_title("Stage1B Error by Eval Order")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        x,
        metric_array(samples, "metrics.action_mae_kappa"),
        label="Canonical Kappa MAE",
        color="#9467bd",
        linewidth=1.3,
    )
    axes[1].plot(
        x,
        metric_array(samples, "metrics.action_mae_accel"),
        label="Canonical Accel MAE",
        color="#8c564b",
        linewidth=1.1,
    )
    if pid_enabled:
        axes[1].plot(
            x,
            metric_array(samples, "pid_override.metrics.action_mae_accel"),
            label="PID Accel MAE",
            color="#2ca02c",
            linewidth=1.1,
        )
    axes[1].set_ylabel("Action MAE")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].plot(
        x,
        metric_array(samples, "max_lateral_error_m"),
        label="Canonical Max Lateral",
        color="#ff7f0e",
        linewidth=1.2,
    )
    if pid_enabled:
        axes[2].plot(
            x,
            metric_array(samples, "pid_override.max_lateral_error_m"),
            label="PID Max Lateral",
            color="#1f77b4",
            linewidth=1.2,
        )
    axes[2].set_xlabel("Eval Sample Index")
    axes[2].set_ylabel("Meters")
    axes[2].legend()
    axes[2].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_single_trajectory(
    ax,
    sample: dict[str, Any],
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    pid_enabled: bool,
) -> None:
    gt = np.asarray(sample["gt_waypoints"], dtype=np.float64)
    pred = np.asarray(sample["pred_waypoints"], dtype=np.float64)
    ax.plot(gt[:, 0], gt[:, 1], color="#111111", linewidth=2.2, label="GT")
    ax.plot(pred[:, 0], pred[:, 1], color="#d62728", linewidth=2.0, linestyle="--", label="Canonical")
    if pid_enabled and isinstance(sample.get("pid_override"), dict):
        pid = np.asarray(nested_get(sample, "pid_override.pred_waypoints"), dtype=np.float64)
        ax.plot(pid[:, 0], pid[:, 1], color="#1f77b4", linewidth=1.8, linestyle=":", label="PID")
        ax.scatter(pid[-1, 0], pid[-1, 1], color="#1f77b4", s=22, marker="^", zorder=4)
    ax.scatter(gt[0, 0], gt[0, 1], color="#2ca02c", s=22, zorder=4)
    ax.scatter(gt[-1, 0], gt[-1, 1], color="#111111", s=22, marker="s", zorder=4)
    ax.scatter(pred[-1, 0], pred[-1, 1], color="#d62728", s=22, marker="x", zorder=4)
    ax.axhline(0.0, color="#d0d0d0", linewidth=0.7)
    ax.axvline(0.0, color="#d0d0d0", linewidth=0.7)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    title = f"{sample['sample_id']} | FDE={sample['fde_m']:.2f}"
    if pid_enabled and isinstance(sample.get("pid_override"), dict):
        title += f" | PID={sample['pid_override']['fde_m']:.2f}"
    ax.set_title(title, fontsize=8)


def _save_trajectory_grid(
    samples: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
    dpi: int,
    pid_enabled: bool,
) -> None:
    cols = 4
    rows = math.ceil(len(samples) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.1, rows * 4.0), dpi=dpi)
    axes_array = np.asarray(axes).reshape(rows, cols)
    waypoint_keys = ["gt_waypoints", "pred_waypoints"]
    if pid_enabled:
        waypoint_keys.append("pid_override.pred_waypoints")
    xlim, ylim = trajectory_limits(samples, waypoint_keys=waypoint_keys)
    for ax, sample in zip(axes_array.flat, samples):
        _plot_single_trajectory(ax, sample, xlim=xlim, ylim=ylim, pid_enabled=pid_enabled)
    for ax in axes_array.flat[len(samples) :]:
        ax.axis("off")
    handles = [
        Line2D([0], [0], color="#111111", linewidth=2.2, label="GT"),
        Line2D([0], [0], color="#d62728", linewidth=2.0, linestyle="--", label="Canonical"),
    ]
    if pid_enabled:
        handles.append(Line2D([0], [0], color="#1f77b4", linewidth=1.8, linestyle=":", label="PID"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False)
    fig.suptitle(title, fontsize=16, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_single_overlay(
    sample: dict[str, Any],
    output_path: Path,
    *,
    dpi: int,
    pid_enabled: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=dpi)
    waypoint_keys = ["gt_waypoints", "pred_waypoints"]
    if pid_enabled:
        waypoint_keys.append("pid_override.pred_waypoints")
    xlim, ylim = trajectory_limits([sample], waypoint_keys=waypoint_keys)
    _plot_single_trajectory(ax, sample, xlim=xlim, ylim=ylim, pid_enabled=pid_enabled)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_json).resolve()
    per_sample_path = Path(args.per_sample_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = output_dir / "worst_sample_overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    summary = load_json(summary_path)
    samples = load_jsonl(per_sample_path)
    _ensure_plot_fields(samples)
    pid_enabled = _pid_enabled(samples, summary)

    expected_samples = int(summary.get("num_samples", 0))
    if expected_samples > 0 and expected_samples != len(samples):
        raise RuntimeError(
            f"Per-sample JSONL count {len(samples)} does not match summary num_samples {expected_samples}."
        )

    plot_paths = {
        "histograms": output_dir / "metric_histograms.png",
        "cdfs": output_dir / "metric_cdfs.png",
        "scatter": output_dir / "metric_scatter.png",
        "pid_comparison": output_dir / "pid_comparison.png",
        "sample_order": output_dir / "sample_order.png",
        "trajectory_best": output_dir / "trajectory_best.png",
        "trajectory_median": output_dir / "trajectory_median.png",
        "trajectory_worst": output_dir / "trajectory_worst.png",
        "manifest": output_dir / "visualization_manifest.json",
        "worst_samples": output_dir / "worst_samples.json",
    }

    _save_histograms(samples, plot_paths["histograms"], dpi=int(args.dpi))
    _save_cdfs(samples, plot_paths["cdfs"], dpi=int(args.dpi))
    _save_scatter(samples, plot_paths["scatter"], dpi=int(args.dpi))
    _save_pid_comparison(samples, plot_paths["pid_comparison"], dpi=int(args.dpi), pid_enabled=pid_enabled)
    _save_sample_order(samples, plot_paths["sample_order"], dpi=int(args.dpi), pid_enabled=pid_enabled)

    rank_groups = select_rank_groups(samples, count=int(args.overlay_count))
    _save_trajectory_grid(
        rank_groups["best"],
        plot_paths["trajectory_best"],
        title="Best FDE Trajectories",
        dpi=int(args.dpi),
        pid_enabled=pid_enabled,
    )
    _save_trajectory_grid(
        rank_groups["median"],
        plot_paths["trajectory_median"],
        title="Median FDE Trajectories",
        dpi=int(args.dpi),
        pid_enabled=pid_enabled,
    )
    _save_trajectory_grid(
        rank_groups["worst"],
        plot_paths["trajectory_worst"],
        title="Worst FDE Trajectories",
        dpi=int(args.dpi),
        pid_enabled=pid_enabled,
    )

    worst_table_count = min(int(args.worst_table_count), len(samples))
    worst_samples = sorted(samples, key=lambda sample: float(sample["fde_m"]), reverse=True)[:worst_table_count]
    worst_manifest_rows: list[dict[str, Any]] = []
    for rank, sample in enumerate(worst_samples, start=1):
        overlay_path = overlays_dir / f"rank_{rank:02d}_{sample['sample_id']}.png"
        _save_single_overlay(sample, overlay_path, dpi=int(args.dpi), pid_enabled=pid_enabled)
        row = {
            "rank": rank,
            "sample_id": sample["sample_id"],
            "sample_index": int(sample["sample_index"]),
            "image_path": sample["image_path"],
            "command": sample["command"],
            "overlay_path": str(overlay_path),
            "ade_m": float(sample["ade_m"]),
            "fde_m": float(sample["fde_m"]),
            "max_lateral_error_m": float(sample["max_lateral_error_m"]),
            "action_mae_kappa": float(sample["metrics"]["action_mae_kappa"]),
        }
        if pid_enabled and isinstance(sample.get("pid_override"), dict):
            row["pid_fde_m"] = float(sample["pid_override"]["fde_m"])
            row["pid_max_lateral_error_m"] = float(sample["pid_override"]["max_lateral_error_m"])
            row["pid_action_mae_accel"] = float(sample["pid_override"]["metrics"]["action_mae_accel"])
        worst_manifest_rows.append(row)
    write_json(plot_paths["worst_samples"], {"rows": worst_manifest_rows})

    run = init_required_online_wandb(
        project=str(args.wandb_project),
        entity=str(getattr(args, "wandb_entity", "")),
        name=str(args.wandb_run_name),
        config={
            "entrypoint": "stage1.expert_cfm.visualize_eval",
            "config_json": str(args.config_json),
            "summary_json": str(summary_path),
            "per_sample_jsonl": str(per_sample_path),
            "output_dir": str(output_dir),
            "num_samples": len(samples),
            "overlay_count": int(args.overlay_count),
            "worst_table_count": worst_table_count,
            "pid_override_enabled": pid_enabled,
        },
    )
    try:
        import wandb

        run.summary["checkpoint"] = summary.get("checkpoint", "")
        run.summary["stage1_checkpoint"] = summary.get("stage1_checkpoint", "")
        run.summary["num_samples"] = len(samples)
        for metric_key in [
            "cfm_loss",
            "ade_m",
            "fde_m",
            "mean_max_lateral_error_m",
            "global_max_lateral_error_m",
            "action_mae_accel",
            "action_mae_kappa",
        ]:
            if metric_key in summary:
                run.summary[f"summary/{metric_key}"] = summary[metric_key]
        if pid_enabled and isinstance(summary.get("pid_override"), dict):
            for metric_key in [
                "ade_m",
                "fde_m",
                "mean_max_lateral_error_m",
                "global_max_lateral_error_m",
                "action_mae_accel",
                "action_mae_kappa",
            ]:
                if metric_key in summary["pid_override"]:
                    run.summary[f"summary/pid_override/{metric_key}"] = summary["pid_override"][metric_key]

        run.log(
            {
                "plots/histograms": wandb.Image(str(plot_paths["histograms"])),
                "plots/cdfs": wandb.Image(str(plot_paths["cdfs"])),
                "plots/scatter": wandb.Image(str(plot_paths["scatter"])),
                "plots/pid_comparison": wandb.Image(str(plot_paths["pid_comparison"])),
                "plots/sample_order": wandb.Image(str(plot_paths["sample_order"])),
                "plots/trajectory_best": wandb.Image(str(plot_paths["trajectory_best"])),
                "plots/trajectory_median": wandb.Image(str(plot_paths["trajectory_median"])),
                "plots/trajectory_worst": wandb.Image(str(plot_paths["trajectory_worst"])),
            },
            step=len(samples),
        )

        table_columns = [
            "rank",
            "sample_id",
            "sample_index",
            "command",
            "ade_m",
            "fde_m",
            "max_lateral_error_m",
            "action_mae_kappa",
        ]
        if pid_enabled:
            table_columns.extend(["pid_fde_m", "pid_max_lateral_error_m", "pid_action_mae_accel"])
        table_columns.extend(["camera_image", "trajectory_overlay"])
        table = wandb.Table(columns=table_columns)
        for row in worst_manifest_rows:
            values = [
                row["rank"],
                row["sample_id"],
                row["sample_index"],
                row["command"],
                row["ade_m"],
                row["fde_m"],
                row["max_lateral_error_m"],
                row["action_mae_kappa"],
            ]
            if pid_enabled:
                values.extend(
                    [
                        row.get("pid_fde_m"),
                        row.get("pid_max_lateral_error_m"),
                        row.get("pid_action_mae_accel"),
                    ]
                )
            values.extend(
                [
                    wandb.Image(str(row["image_path"])),
                    wandb.Image(str(row["overlay_path"])),
                ]
            )
            table.add_data(*values)
        run.log({"tables/worst_samples": table}, step=len(samples))

        manifest = {
            "summary_json": str(summary_path),
            "per_sample_jsonl": str(per_sample_path),
            "output_dir": str(output_dir),
            "plots": {key: str(path) for key, path in plot_paths.items() if key != "manifest"},
            "worst_samples": worst_manifest_rows,
            "wandb": {
                "project": args.wandb_project,
                "run_name": args.wandb_run_name,
                "run_url": getattr(run, "url", ""),
            },
        }
        write_json(plot_paths["manifest"], manifest)
        update_manifest_plots(
            summary_json=summary_path,
            plots_dir=output_dir,
            plots={key: str(path) for key, path in plot_paths.items() if key != "manifest"},
            wandb_run_url=getattr(run, "url", ""),
        )
        run.summary["manifest_path"] = str(plot_paths["manifest"])
        run.summary["output_dir"] = str(output_dir)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
