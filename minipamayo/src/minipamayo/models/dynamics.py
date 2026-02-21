"""Unicycle dynamics: forward and inverse transformations.

Forward: (a, κ) control inputs → (x, y, θ, v) trajectory
Inverse: ego poses → (a, κ) GT control inputs

Used across Stage 0 (Phase 4+), Stage 1, Stage 2.
"""

import numpy as np
import torch


def forward_dynamics_np(
    a: np.ndarray,
    kappa: np.ndarray,
    v0: float,
    dt: float = 0.5,
) -> np.ndarray:
    """Forward unicycle dynamics (numpy, single sample).

    Ego-centric frame: starts at (0, 0, θ=0, v=v0).

    Args:
        a: (K,) acceleration sequence
        kappa: (K,) curvature sequence
        v0: initial speed (m/s)
        dt: time step (s)

    Returns:
        waypoints: (K, 2) — ego-centric [x, y] positions
    """
    K = len(a)
    x, y, theta, v = 0.0, 0.0, 0.0, v0
    waypoints = np.zeros((K, 2))

    for i in range(K):
        v_new = v + dt * a[i]
        theta_new = theta + dt * kappa[i] * v + dt**2 / 2 * kappa[i] * a[i]
        x_new = x + dt / 2 * (v * np.cos(theta) + v_new * np.cos(theta_new))
        y_new = y + dt / 2 * (v * np.sin(theta) + v_new * np.sin(theta_new))

        waypoints[i] = [x_new, y_new]
        x, y, theta, v = x_new, y_new, theta_new, v_new

    return waypoints


def forward_dynamics_batch(
    a: torch.Tensor,
    kappa: torch.Tensor,
    v0: torch.Tensor,
    dt: float = 0.5,
) -> torch.Tensor:
    """Batch forward unicycle dynamics (torch).

    Args:
        a: (B, K) acceleration
        kappa: (B, K) curvature
        v0: (B,) initial speed
        dt: time step

    Returns:
        waypoints: (B, K, 2) — ego-centric [x, y]
    """
    B, K = a.shape
    device, dtype = a.device, torch.float32

    x = torch.zeros(B, device=device, dtype=dtype)
    y = torch.zeros(B, device=device, dtype=dtype)
    theta = torch.zeros(B, device=device, dtype=dtype)
    v = v0.to(dtype)

    waypoints = torch.zeros(B, K, 2, device=device, dtype=dtype)

    for i in range(K):
        v_new = v + dt * a[:, i]
        theta_new = theta + dt * kappa[:, i] * v + dt**2 / 2 * kappa[:, i] * a[:, i]
        x_new = x + dt / 2 * (v * theta.cos() + v_new * theta_new.cos())
        y_new = y + dt / 2 * (v * theta.sin() + v_new * theta_new.sin())

        waypoints[:, i, 0] = x_new
        waypoints[:, i, 1] = y_new
        x, y, theta, v = x_new, y_new, theta_new, v_new

    return waypoints


def inverse_dynamics_np(
    positions: np.ndarray,
    headings: np.ndarray,
    dt: float = 0.5,
    v_threshold: float = 0.1,
    lambda_reg: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse dynamics: K+2 ego poses → K (a, κ) control inputs.

    Uses Tikhonov regularization (Alpamayo §5.1) for smooth GT extraction:
      min ||Dx - b||² + λ||x||²
      → x = (D^T D + λI)^{-1} D^T b

    Args:
        positions: (K+2, 2) — consecutive [x, y] positions
        headings: (K+2,) — consecutive yaw angles (rad)
        dt: time step (s)
        v_threshold: min speed for curvature computation (m/s)
        lambda_reg: Tikhonov regularization strength

    Returns:
        a: (K,) acceleration (m/s²)
        kappa: (K,) curvature (1/m)
    """
    n_poses = len(positions)
    K = n_poses - 2

    # K+1 speeds from K+2 positions
    displacements = np.diff(positions, axis=0)  # (K+1, 2)
    speeds = np.linalg.norm(displacements, axis=1) / dt  # (K+1,)

    # K+1 heading differences, normalized to [-π, π]
    heading_diffs = np.diff(headings)  # (K+1,)
    heading_diffs = (heading_diffs + np.pi) % (2 * np.pi) - np.pi

    # Tikhonov-regularized acceleration: min ||a - raw_a||² + λ||L_a @ a||²
    # raw_a: finite differences of speeds
    # L_a: first-order difference matrix for smoothness
    raw_a = np.diff(speeds) / dt  # (K,) — unregularized acceleration
    L_a = np.zeros((K - 1, K)) if K > 1 else np.zeros((0, K))
    for i in range(K - 1):
        L_a[i, i] = -1.0
        L_a[i, i + 1] = 1.0
    a = np.linalg.solve(np.eye(K) + lambda_reg * L_a.T @ L_a, raw_a)  # (K,)

    # Tikhonov-regularized curvature: kappa = dθ/(dt·v), smoothed
    raw_kappa = np.zeros(K)
    for i in range(K):
        if speeds[i] > v_threshold:
            raw_kappa[i] = heading_diffs[i] / (dt * speeds[i])
    # Regularize: min ||x - raw_kappa||² + λ||Δx||²
    # Smoothing matrix L: first-order difference on kappa
    L = np.zeros((K - 1, K)) if K > 1 else np.zeros((0, K))
    for i in range(K - 1):
        L[i, i] = -1.0
        L[i, i + 1] = 1.0
    kappa = np.linalg.solve(np.eye(K) + lambda_reg * L.T @ L, raw_kappa)  # (K,)

    return a, kappa


def to_ego_centric(
    global_positions: np.ndarray,
    ego_position: np.ndarray,
    ego_heading: float,
) -> np.ndarray:
    """Transform global positions to ego-centric frame.

    Args:
        global_positions: (N, 2) — [x, y] in global coords
        ego_position: (2,) — ego [x, y] in global coords
        ego_heading: ego yaw angle (rad)

    Returns:
        ego_positions: (N, 2) — [x, y] in ego-centric coords
            (ego at origin, heading = +x direction)
    """
    centered = global_positions - ego_position
    cos_h = np.cos(ego_heading)
    sin_h = np.sin(ego_heading)
    # Rotate by -heading to align ego forward with +x
    ego_x = cos_h * centered[:, 0] + sin_h * centered[:, 1]
    ego_y = -sin_h * centered[:, 0] + cos_h * centered[:, 1]
    return np.stack([ego_x, ego_y], axis=1)
