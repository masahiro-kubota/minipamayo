"""Small geometry and dynamics helpers used by the Stage 1 prototype."""

from __future__ import annotations

import numpy as np
import torch


def forward_dynamics_np(
    a: np.ndarray,
    kappa: np.ndarray,
    v0: float,
    dt: float = 0.5,
) -> np.ndarray:
    """Forward unicycle dynamics for a single control sequence."""

    k_steps = len(a)
    x, y, theta, v = 0.0, 0.0, 0.0, float(v0)
    waypoints = np.zeros((k_steps, 2), dtype=np.float32)

    for i in range(k_steps):
        v_new = v + dt * float(a[i])
        theta_new = theta + dt * float(kappa[i]) * v + (dt**2 / 2.0) * float(kappa[i]) * float(a[i])
        x_new = x + (dt / 2.0) * (v * np.cos(theta) + v_new * np.cos(theta_new))
        y_new = y + (dt / 2.0) * (v * np.sin(theta) + v_new * np.sin(theta_new))

        waypoints[i] = [x_new, y_new]
        x, y, theta, v = x_new, y_new, theta_new, v_new

    return waypoints


def forward_dynamics_batch(
    a: torch.Tensor,
    kappa: torch.Tensor,
    v0: torch.Tensor,
    dt: float = 0.5,
) -> torch.Tensor:
    """Forward unicycle dynamics for a batch of control sequences."""

    batch_size, k_steps = a.shape
    device = a.device
    dtype = torch.float32

    x = torch.zeros(batch_size, device=device, dtype=dtype)
    y = torch.zeros(batch_size, device=device, dtype=dtype)
    theta = torch.zeros(batch_size, device=device, dtype=dtype)
    v = v0.to(dtype)

    waypoints = torch.zeros(batch_size, k_steps, 2, device=device, dtype=dtype)

    for i in range(k_steps):
        v_new = v + dt * a[:, i]
        theta_new = theta + dt * kappa[:, i] * v + (dt**2 / 2.0) * kappa[:, i] * a[:, i]
        x_new = x + (dt / 2.0) * (v * theta.cos() + v_new * theta_new.cos())
        y_new = y + (dt / 2.0) * (v * theta.sin() + v_new * theta_new.sin())

        waypoints[:, i, 0] = x_new
        waypoints[:, i, 1] = y_new
        x, y, theta, v = x_new, y_new, theta_new, v_new

    return waypoints


def _second_order_diff_matrix(length: int) -> np.ndarray:
    rows = max(length - 2, 0)
    matrix = np.zeros((rows, length), dtype=np.float64)
    for i in range(rows):
        matrix[i, i] = -1.0
        matrix[i, i + 1] = 2.0
        matrix[i, i + 2] = -1.0
    return matrix


def inverse_dynamics_np(
    positions: np.ndarray,
    headings: np.ndarray,
    dt: float = 0.5,
    v_threshold: float = 0.1,
    a_lambda: float = 1e-4,
    a_ridge: float = 1e-4,
    kappa_lambda: float = 1e-4,
    kappa_ridge: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer smooth (a, kappa) controls from K+2 ego poses."""

    num_poses = len(positions)
    k_steps = num_poses - 2
    displacements = np.diff(positions, axis=0)
    speeds = np.linalg.norm(displacements, axis=1) / dt

    heading_diffs = np.diff(headings)
    heading_diffs = np.arctan2(np.sin(heading_diffs), np.cos(heading_diffs))

    raw_a = np.diff(speeds) / dt
    d2_a = _second_order_diff_matrix(k_steps)
    a_system = np.eye(k_steps) + (a_lambda / dt**4) * (d2_a.T @ d2_a) + a_ridge * np.eye(k_steps)
    accel = np.linalg.solve(a_system, raw_a)

    arc_lengths = dt * speeds[:k_steps] + (dt**2 / 2.0) * accel
    raw_kappa = np.zeros(k_steps, dtype=np.float64)
    for i in range(k_steps):
        if abs(arc_lengths[i]) > v_threshold * dt:
            raw_kappa[i] = heading_diffs[i] / arc_lengths[i]

    s_mat = np.diag(arc_lengths)
    d2_kappa = _second_order_diff_matrix(k_steps)
    kappa_system = (
        s_mat.T @ s_mat
        + (kappa_lambda / dt**4) * (d2_kappa.T @ d2_kappa)
        + kappa_ridge * np.eye(k_steps)
    )
    kappa = np.linalg.solve(kappa_system, s_mat.T @ heading_diffs[:k_steps])
    return accel.astype(np.float32), kappa.astype(np.float32)


def to_ego_centric(
    global_positions: np.ndarray,
    ego_position: np.ndarray,
    ego_heading: float,
) -> np.ndarray:
    """Transform global XY positions into the t0 ego frame."""

    centered = global_positions - ego_position
    cos_h = np.cos(ego_heading)
    sin_h = np.sin(ego_heading)
    ego_x = cos_h * centered[:, 0] + sin_h * centered[:, 1]
    ego_y = -sin_h * centered[:, 0] + cos_h * centered[:, 1]
    return np.stack([ego_x, ego_y], axis=1).astype(np.float32)


def interleave_action(accel: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """Return [a0, k0, a1, k1, ...]."""

    out = np.empty(accel.shape[0] * 2, dtype=np.float32)
    out[0::2] = accel
    out[1::2] = kappa
    return out
