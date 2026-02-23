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


def _second_order_diff_matrix(N: int) -> np.ndarray:
    """Build 2nd-order finite difference matrix (Alpamayo w_smooth2).

    D2[i,i]=-1, D2[i,i+1]=2, D2[i,i+2]=-1
    Penalizes changes in the 1st derivative (i.e., smoothness of the derivative).
    """
    rows = max(N - 2, 0)
    D = np.zeros((rows, N))
    for i in range(rows):
        D[i, i] = -1.0
        D[i, i + 1] = 2.0
        D[i, i + 2] = -1.0
    return D


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
    """Inverse dynamics: K+2 ego poses → K (a, κ) control inputs.

    Uses 2nd-order Tikhonov regularization (Alpamayo §5.1) for smooth GT:
      Acceleration: min ||a - Δv/dt||² + (λ/dt⁴)||D₂a||² + ridge||a||²
      Curvature:    min ||s·κ - Δθ||²   + (λ/dt⁴)||D₂κ||² + ridge||κ||²
        where s = dt·v + dt²/2·a (Alpamayo kinematic denominator)

    Args:
        positions: (K+2, 2) — consecutive [x, y] positions
        headings: (K+2,) — consecutive yaw angles (rad)
        dt: time step (s)
        v_threshold: min speed for curvature computation (m/s)
        a_lambda: Tikhonov smoothing weight for acceleration
        a_ridge: Ridge regularization for acceleration
        kappa_lambda: Tikhonov smoothing weight for curvature
        kappa_ridge: Ridge regularization for curvature

    Returns:
        a: (K,) acceleration (m/s²)
        kappa: (K,) curvature (1/m)
    """
    n_poses = len(positions)
    K = n_poses - 2

    # K+1 speeds from K+2 positions
    displacements = np.diff(positions, axis=0)  # (K+1, 2)
    speeds = np.linalg.norm(displacements, axis=1) / dt  # (K+1,)

    # K+1 heading differences, normalized to [-π, π] (atan2 for stability, Alpamayo)
    heading_diffs = np.diff(headings)  # (K+1,)
    heading_diffs = np.arctan2(np.sin(heading_diffs), np.cos(heading_diffs))

    # --- Acceleration (2nd-order Tikhonov, Alpamayo _v_to_a) ---
    raw_a = np.diff(speeds) / dt  # (K,)
    D2_a = _second_order_diff_matrix(K)
    DTD_a = (a_lambda / dt**4) * D2_a.T @ D2_a
    a = np.linalg.solve(np.eye(K) + DTD_a + a_ridge * np.eye(K), raw_a)  # (K,)

    # --- Curvature (2nd-order Tikhonov, Alpamayo _theta_v_a_to_kappa) ---
    # Kinematic denominator: s = dt * v + dt²/2 * a (matches forward dynamics)
    s = dt * speeds[:K] + (dt**2) / 2.0 * a  # (K,)

    # Solve: min ||diag(s) @ kappa - dtheta||² + smooth + ridge
    raw_kappa = np.zeros(K)
    for i in range(K):
        if abs(s[i]) > v_threshold * dt:  # threshold on arc length, not speed
            raw_kappa[i] = heading_diffs[i] / s[i]

    S = np.diag(s)
    STS = S.T @ S
    D2_k = _second_order_diff_matrix(K)
    DTD_k = (kappa_lambda / dt**4) * D2_k.T @ D2_k
    kappa = np.linalg.solve(STS + DTD_k + kappa_ridge * np.eye(K), S.T @ heading_diffs[:K])

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
