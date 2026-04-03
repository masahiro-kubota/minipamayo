"""Matplotlib helpers for the local eval inspector."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .models import NormalizedSample


def _trajectory_limits(sample: NormalizedSample) -> tuple[tuple[float, float], tuple[float, float]]:
    xs: list[float] = []
    ys: list[float] = []
    for points in [sample.gt_waypoints, sample.pred_waypoints, sample.pid_pred_waypoints or []]:
        for point in points:
            xs.append(float(point[0]))
            ys.append(float(point[1]))
    if not xs or not ys:
        return (-1.0, 1.0), (-1.0, 1.0)
    xpad = max(0.5, (max(xs) - min(xs)) * 0.08)
    ypad = max(0.5, (max(ys) - min(ys)) * 0.08)
    return (min(xs) - xpad, max(xs) + xpad), (min(ys) - ypad, max(ys) + ypad)


def build_trajectory_overlay_figure(sample: NormalizedSample):
    fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=180)
    xlim, ylim = _trajectory_limits(sample)
    if sample.gt_waypoints:
        gt = np.asarray(sample.gt_waypoints, dtype=np.float64)
        ax.plot(gt[:, 0], gt[:, 1], color="#111111", linewidth=2.2, label="GT")
        ax.scatter(gt[0, 0], gt[0, 1], color="#2ca02c", s=28, zorder=4)
        ax.scatter(gt[-1, 0], gt[-1, 1], color="#111111", s=28, marker="s", zorder=4)
    if sample.pred_waypoints:
        pred = np.asarray(sample.pred_waypoints, dtype=np.float64)
        ax.plot(pred[:, 0], pred[:, 1], color="#d62728", linewidth=2.0, linestyle="--", label="Pred")
        ax.scatter(pred[-1, 0], pred[-1, 1], color="#d62728", s=28, marker="x", zorder=4)
    if sample.pid_pred_waypoints:
        pid = np.asarray(sample.pid_pred_waypoints, dtype=np.float64)
        ax.plot(pid[:, 0], pid[:, 1], color="#1f77b4", linewidth=1.8, linestyle=":", label="PID")
        ax.scatter(pid[-1, 0], pid[-1, 1], color="#1f77b4", s=28, marker="^", zorder=4)
    ax.axhline(0.0, color="#d0d0d0", linewidth=0.7)
    ax.axvline(0.0, color="#d0d0d0", linewidth=0.7)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.set_title(f"{sample.stage} | {sample.sample_id}", fontsize=10)
    ax.legend(loc="upper center", ncol=3 if sample.pid_pred_waypoints else 2, frameon=False)
    fig.tight_layout()
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
