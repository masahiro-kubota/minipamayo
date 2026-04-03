"""Matplotlib helpers for the local eval inspector."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .models import NormalizedGroup, NormalizedSample


def _to_plot_frame(points: list[list[float]]) -> np.ndarray:
    if not points:
        return np.zeros((0, 2), dtype=np.float64)
    raw = np.asarray(points, dtype=np.float64)
    forward = raw[:, 0]
    lateral = raw[:, 1]
    # Canonical waypoints are (forward, lateral[left+]). Rotate so forward points up.
    return np.stack([-lateral, forward], axis=1)


def _trajectory_limits(sample: NormalizedSample) -> tuple[tuple[float, float], tuple[float, float]]:
    xs: list[float] = []
    ys: list[float] = []
    for points in [sample.gt_waypoints, sample.pred_waypoints, sample.pid_pred_waypoints or []]:
        for point in _to_plot_frame(points):
            xs.append(float(point[0]))
            ys.append(float(point[1]))
    if not xs or not ys:
        return (-1.0, 1.0), (-1.0, 1.0)
    xpad = max(0.5, (max(xs) - min(xs)) * 0.08)
    ypad = max(0.5, (max(ys) - min(ys)) * 0.08)
    return (min(xs) - xpad, max(xs) + xpad), (min(ys) - ypad, max(ys) + ypad)


def build_trajectory_overlay_figure(sample: NormalizedSample):
    fig, ax = plt.subplots(figsize=(3.6, 4.4), dpi=180)
    xlim, ylim = _trajectory_limits(sample)
    if sample.gt_waypoints:
        gt = _to_plot_frame(sample.gt_waypoints)
        ax.plot(gt[:, 0], gt[:, 1], color="#111111", linewidth=2.0, label="GT")
        ax.scatter(gt[0, 0], gt[0, 1], color="#2ca02c", s=20, zorder=4)
        ax.scatter(gt[-1, 0], gt[-1, 1], color="#111111", s=20, marker="s", zorder=4)
    if sample.pred_waypoints:
        pred = _to_plot_frame(sample.pred_waypoints)
        ax.plot(pred[:, 0], pred[:, 1], color="#d62728", linewidth=1.8, linestyle="--", label="Pred")
        ax.scatter(pred[-1, 0], pred[-1, 1], color="#d62728", s=20, marker="x", zorder=4)
    if sample.pid_pred_waypoints:
        pid = _to_plot_frame(sample.pid_pred_waypoints)
        ax.plot(pid[:, 0], pid[:, 1], color="#1f77b4", linewidth=1.6, linestyle=":", label="PID")
        ax.scatter(pid[-1, 0], pid[-1, 1], color="#1f77b4", s=20, marker="^", zorder=4)
    ax.axhline(0.0, color="#d0d0d0", linewidth=0.7)
    ax.axvline(0.0, color="#d0d0d0", linewidth=0.7)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.set_xlabel("Lateral", fontsize=8)
    ax.set_ylabel("Forward", fontsize=8)
    ax.set_title(f"{sample.stage} | {sample.sample_id}", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.legend(
        loc="upper center",
        ncol=3 if sample.pid_pred_waypoints else 2,
        frameon=False,
        fontsize=7,
        handlelength=2.0,
        borderaxespad=0.3,
    )
    fig.tight_layout(pad=0.8)
    return fig


def save_trajectory_overlay(sample: NormalizedSample, output_path: str | Path) -> Path:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = build_trajectory_overlay_figure(sample)
    try:
        fig.savefig(path, bbox_inches="tight")
    finally:
        plt.close(fig)
    return path


def build_group_metric_timeline_figure(
    group: NormalizedGroup,
    *,
    current_frame_index: int | None = None,
):
    metric_specs = [
        ("fde_m", "FDE", "#d62728"),
        ("ade_m", "ADE", "#ff7f0e"),
        ("action_mae_kappa", "Kappa MAE", "#2ca02c"),
        ("token_accuracy", "Token Acc", "#1f77b4"),
    ]
    available_specs = [
        (attr_name, label, color)
        for attr_name, label, color in metric_specs
        if any(getattr(sample, attr_name) is not None for sample in group.samples)
    ]
    if not available_specs:
        fig, ax = plt.subplots(figsize=(7.0, 1.8), dpi=180)
        ax.text(0.5, 0.5, "No per-frame metrics available.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig

    fig, axes = plt.subplots(
        len(available_specs),
        1,
        figsize=(7.0, max(2.2, 1.9 * len(available_specs))),
        dpi=180,
        sharex=True,
    )
    if len(available_specs) == 1:
        axes = [axes]
    x_values = [int(sample.group_frame_index) for sample in group.samples]
    for axis, (attr_name, label, color) in zip(axes, available_specs):
        y_values = [
            float(getattr(sample, attr_name)) if getattr(sample, attr_name) is not None else np.nan
            for sample in group.samples
        ]
        axis.plot(x_values, y_values, color=color, linewidth=1.8)
        axis.scatter(x_values, y_values, color=color, s=10, alpha=0.85)
        if current_frame_index is not None:
            axis.axvline(int(current_frame_index), color="#111111", linewidth=1.1, linestyle="--")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Frame In Block")
    fig.suptitle(group.group_id, fontsize=10)
    fig.tight_layout()
    return fig
