"""Visualize canonical Stage 1A eval outputs with local plots and W&B logging."""

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

from ..eval_viz_common import (
    cdf,
    metric_array,
    select_rank_groups,
    trajectory_limits,
    write_json,
)
from ..eval_visualizer import (
    CONFIG_PATH_KEYS,
    Stage1VisualizationContext,
    build_stage1_visualize_parser,
    finalize_stage1_visualization,
    load_visualization_context,
    parse_stage1_visualize_args,
)
from .cli import artifact_scope_for_config, parse_config_json_only_args

def build_parser() -> argparse.ArgumentParser:
    return build_stage1_visualize_parser("Visualize canonical Stage 1A eval outputs.")


def parse_args() -> argparse.Namespace:
    return parse_stage1_visualize_args(
        parser=build_parser(),
        parse_json_only_args=lambda parser: parse_config_json_only_args(
            parser,
            path_keys=CONFIG_PATH_KEYS,
            error_message="Stage 1A visualization accepts only --config-json. Put all settings in the JSON file.",
        ),
        scope_for_config=lambda config_json: artifact_scope_for_config(config_json, kind="eval"),
    )

def _ensure_plot_fields(samples: list[dict[str, Any]]) -> None:
    required_top_level = {
        "sample_id",
        "sample_index",
        "image_path",
        "gt_waypoints",
        "pred_waypoints",
        "gt_action_bins",
        "pred_action_bins",
    }
    required_metrics = {
        "teacher_forced_token_accuracy",
        "autoregressive_token_accuracy",
        "action_mae_accel",
        "action_mae_kappa",
        "ade_m",
        "fde_m",
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

def _metric(samples: list[dict[str, Any]], metric_key: str) -> np.ndarray:
    if metric_key in {"ade_m", "fde_m"}:
        return metric_array(samples, metric_key)
    return metric_array(samples, f"metrics.{metric_key}")


def _save_histograms(samples: list[dict[str, Any]], output_path: Path, *, dpi: int) -> None:
    metrics = {
        "ADE (m)": _metric(samples, "ade_m"),
        "FDE (m)": _metric(samples, "fde_m"),
        "Kappa MAE": _metric(samples, "action_mae_kappa"),
        "AR Token Accuracy": _metric(samples, "autoregressive_token_accuracy"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=dpi)
    for ax, (title, values) in zip(axes.flat, metrics.items()):
        ax.hist(values, bins=32, color="#1f77b4", alpha=0.82, edgecolor="white")
        ax.axvline(values.mean(), color="#d62728", linestyle="--", linewidth=1.6, label=f"mean={values.mean():.3f}")
        ax.axvline(np.median(values), color="#2ca02c", linestyle=":", linewidth=1.6, label=f"median={np.median(values):.3f}")
        ax.set_title(title)
        ax.set_xlabel(title)
        ax.set_ylabel("Count")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Stage1A Curve Eval Distributions", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_cdfs(samples: list[dict[str, Any]], output_path: Path, *, dpi: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=dpi)
    metric_specs = [
        ("ADE (m)", _metric(samples, "ade_m")),
        ("FDE (m)", _metric(samples, "fde_m")),
        ("Kappa MAE", _metric(samples, "action_mae_kappa")),
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
    fde = _metric(samples, "fde_m")
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.5), dpi=dpi)
    scatter_specs = [
        ("TF Token Accuracy", _metric(samples, "teacher_forced_token_accuracy"), "#2ca02c"),
        ("AR Token Accuracy", _metric(samples, "autoregressive_token_accuracy"), "#1f77b4"),
        ("Kappa MAE", _metric(samples, "action_mae_kappa"), "#d62728"),
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


def _save_sample_order(samples: list[dict[str, Any]], output_path: Path, *, dpi: int) -> None:
    x = np.arange(len(samples))
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), dpi=dpi, sharex=True)
    axes[0].plot(x, _metric(samples, "fde_m"), label="FDE", color="#d62728", linewidth=1.4)
    axes[0].plot(x, _metric(samples, "ade_m"), label="ADE", color="#1f77b4", linewidth=1.2)
    axes[0].set_ylabel("Meters")
    axes[0].set_title("Trajectory Error by Eval Order")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, _metric(samples, "action_mae_kappa"), label="Kappa MAE", color="#9467bd", linewidth=1.3)
    axes[1].plot(x, _metric(samples, "action_mae_accel"), label="Accel MAE", color="#8c564b", linewidth=1.1)
    axes[1].set_ylabel("Action MAE")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].plot(
        x,
        _metric(samples, "teacher_forced_token_accuracy"),
        label="TF Token Acc",
        color="#2ca02c",
        linewidth=1.1,
    )
    axes[2].plot(
        x,
        _metric(samples, "autoregressive_token_accuracy"),
        label="AR Token Acc",
        color="#ff7f0e",
        linewidth=1.1,
    )
    axes[2].set_xlabel("Eval Sample Index")
    axes[2].set_ylabel("Accuracy")
    axes[2].legend()
    axes[2].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_bin_usage(samples: list[dict[str, Any]], output_path: Path, *, dpi: int) -> None:
    gt_bins: list[int] = []
    pred_bins: list[int] = []
    for sample in samples:
        gt_bins.extend(int(x) for x in sample["gt_action_bins"])
        pred_bins.extend(int(x) for x in sample["pred_action_bins"])
    max_bin = max(gt_bins + pred_bins)
    min_bin = min(gt_bins + pred_bins)
    bins = np.arange(min_bin, max_bin + 1)
    gt_counts = np.bincount(np.asarray(gt_bins) - min_bin, minlength=len(bins))
    pred_counts = np.bincount(np.asarray(pred_bins) - min_bin, minlength=len(bins))

    fig, ax = plt.subplots(figsize=(14, 5), dpi=dpi)
    ax.step(bins, gt_counts, where="mid", label="GT bins", color="#2ca02c", linewidth=1.8)
    ax.step(bins, pred_counts, where="mid", label="Pred bins", color="#d62728", linewidth=1.6)
    ax.set_title("Action Bin Usage")
    ax.set_xlabel("Bin ID")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

def _plot_single_trajectory(ax, sample: dict[str, Any], *, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    gt = np.asarray(sample["gt_waypoints"], dtype=np.float64)
    pred = np.asarray(sample["pred_waypoints"], dtype=np.float64)
    ax.plot(gt[:, 0], gt[:, 1], color="#111111", linewidth=2.2, label="GT")
    ax.plot(pred[:, 0], pred[:, 1], color="#d62728", linewidth=2.0, linestyle="--", label="Pred")
    ax.scatter(gt[0, 0], gt[0, 1], color="#2ca02c", s=22, zorder=4)
    ax.scatter(gt[-1, 0], gt[-1, 1], color="#111111", s=22, marker="s", zorder=4)
    ax.scatter(pred[-1, 0], pred[-1, 1], color="#d62728", s=22, marker="x", zorder=4)
    ax.axhline(0.0, color="#d0d0d0", linewidth=0.7)
    ax.axvline(0.0, color="#d0d0d0", linewidth=0.7)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.set_title(
        f"{sample['sample_id']} | FDE={sample['fde_m']:.2f} | AR={sample['metrics']['autoregressive_token_accuracy']:.2f}",
        fontsize=8,
    )


def _save_trajectory_grid(
    samples: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
    dpi: int,
) -> None:
    cols = 4
    rows = math.ceil(len(samples) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.1, rows * 4.0), dpi=dpi)
    axes_array = np.asarray(axes).reshape(rows, cols)
    xlim, ylim = trajectory_limits(samples, waypoint_keys=["gt_waypoints", "pred_waypoints"])
    for ax, sample in zip(axes_array.flat, samples):
        _plot_single_trajectory(ax, sample, xlim=xlim, ylim=ylim)
    for ax in axes_array.flat[len(samples) :]:
        ax.axis("off")
    handles = [
        Line2D([0], [0], color="#111111", linewidth=2.2, label="GT"),
        Line2D([0], [0], color="#d62728", linewidth=2.0, linestyle="--", label="Pred"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(title, fontsize=16, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_single_overlay(sample: dict[str, Any], output_path: Path, *, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=dpi)
    xlim, ylim = trajectory_limits([sample], waypoint_keys=["gt_waypoints", "pred_waypoints"])
    _plot_single_trajectory(ax, sample, xlim=xlim, ylim=ylim)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _populate_run_summary(run, context: Stage1VisualizationContext) -> None:
    run.summary["checkpoint"] = context.summary.get("checkpoint", "")
    run.summary["test_jsonl"] = context.summary.get("test_jsonl", "")
    run.summary["num_samples"] = len(context.samples)
    for metric_key in [
        "teacher_forced_loss",
        "teacher_forced_token_accuracy",
        "autoregressive_token_accuracy",
        "action_mae_accel",
        "action_mae_kappa",
        "ade_m",
        "fde_m",
    ]:
        if metric_key in context.summary:
            run.summary[f"summary/{metric_key}"] = context.summary[metric_key]


def _log_wandb_artifacts(
    run,
    wandb,
    context: Stage1VisualizationContext,
    plot_paths: dict[str, Path],
    worst_manifest_rows: list[dict[str, Any]],
) -> None:
    run.log(
        {
            "plots/histograms": wandb.Image(str(plot_paths["histograms"])),
            "plots/cdfs": wandb.Image(str(plot_paths["cdfs"])),
            "plots/scatter": wandb.Image(str(plot_paths["scatter"])),
            "plots/sample_order": wandb.Image(str(plot_paths["sample_order"])),
            "plots/bin_usage": wandb.Image(str(plot_paths["bin_usage"])),
            "plots/trajectory_best": wandb.Image(str(plot_paths["trajectory_best"])),
            "plots/trajectory_median": wandb.Image(str(plot_paths["trajectory_median"])),
            "plots/trajectory_worst": wandb.Image(str(plot_paths["trajectory_worst"])),
        },
        step=len(context.samples),
    )

    table = wandb.Table(
        columns=[
            "rank",
            "sample_id",
            "sample_index",
            "record_sample_index",
            "ade_m",
            "fde_m",
            "tf_token_acc",
            "ar_token_acc",
            "action_mae_kappa",
            "camera_image",
            "trajectory_overlay",
        ]
    )
    for row in worst_manifest_rows:
        table.add_data(
            row["rank"],
            row["sample_id"],
            row["sample_index"],
            row["record_sample_index"],
            row["ade_m"],
            row["fde_m"],
            row["teacher_forced_token_accuracy"],
            row["autoregressive_token_accuracy"],
            row["action_mae_kappa"],
            wandb.Image(str(row["image_path"])),
            wandb.Image(str(row["overlay_path"])),
        )
    run.log({"tables/worst_samples": table}, step=len(context.samples))

def main() -> None:
    args = parse_args()
    context = load_visualization_context(args, ensure_plot_fields=_ensure_plot_fields)
    output_dir = context.output_dir
    overlays_dir = context.overlays_dir
    samples = context.samples

    plot_paths = {
        "histograms": output_dir / "metric_histograms.png",
        "cdfs": output_dir / "metric_cdfs.png",
        "scatter": output_dir / "metric_scatter.png",
        "sample_order": output_dir / "sample_order.png",
        "bin_usage": output_dir / "bin_usage.png",
        "trajectory_best": output_dir / "trajectory_best.png",
        "trajectory_median": output_dir / "trajectory_median.png",
        "trajectory_worst": output_dir / "trajectory_worst.png",
        "manifest": output_dir / "visualization_manifest.json",
        "worst_samples": output_dir / "worst_samples.json",
    }

    _save_histograms(samples, plot_paths["histograms"], dpi=int(args.dpi))
    _save_cdfs(samples, plot_paths["cdfs"], dpi=int(args.dpi))
    _save_scatter(samples, plot_paths["scatter"], dpi=int(args.dpi))
    _save_sample_order(samples, plot_paths["sample_order"], dpi=int(args.dpi))
    _save_bin_usage(samples, plot_paths["bin_usage"], dpi=int(args.dpi))

    rank_groups = select_rank_groups(samples, count=int(args.overlay_count))
    _save_trajectory_grid(rank_groups["best"], plot_paths["trajectory_best"], title="Best FDE Trajectories", dpi=int(args.dpi))
    _save_trajectory_grid(
        rank_groups["median"],
        plot_paths["trajectory_median"],
        title="Median FDE Trajectories",
        dpi=int(args.dpi),
    )
    _save_trajectory_grid(
        rank_groups["worst"],
        plot_paths["trajectory_worst"],
        title="Worst FDE Trajectories",
        dpi=int(args.dpi),
    )

    worst_table_count = context.worst_table_count
    worst_samples = sorted(samples, key=lambda sample: float(sample["fde_m"]), reverse=True)[:worst_table_count]
    worst_manifest_rows: list[dict[str, Any]] = []
    for rank, sample in enumerate(worst_samples, start=1):
        overlay_path = overlays_dir / f"rank_{rank:02d}_{sample['sample_id']}.png"
        _save_single_overlay(sample, overlay_path, dpi=int(args.dpi))
        worst_manifest_rows.append(
            {
                "rank": rank,
                "sample_id": sample["sample_id"],
                "sample_index": int(sample["sample_index"]),
                "record_sample_index": int(sample["record_sample_index"]),
                "image_path": sample["image_path"],
                "overlay_path": str(overlay_path),
                "ade_m": float(sample["ade_m"]),
                "fde_m": float(sample["fde_m"]),
                "teacher_forced_token_accuracy": float(sample["metrics"]["teacher_forced_token_accuracy"]),
                "autoregressive_token_accuracy": float(sample["metrics"]["autoregressive_token_accuracy"]),
                "action_mae_kappa": float(sample["metrics"]["action_mae_kappa"]),
            }
        )
    write_json(plot_paths["worst_samples"], {"rows": worst_manifest_rows})

    finalize_stage1_visualization(
        context=context,
        entrypoint="stage1.vlm_ce.visualize_eval",
        plot_paths=plot_paths,
        worst_manifest_rows=worst_manifest_rows,
        populate_run_summary=_populate_run_summary,
        log_wandb_artifacts=_log_wandb_artifacts,
    )


if __name__ == "__main__":
    main()
