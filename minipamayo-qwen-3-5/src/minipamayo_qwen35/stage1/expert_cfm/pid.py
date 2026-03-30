"""PID helpers for canonical Stage 1B lateral-isolation experiments."""

from __future__ import annotations

import torch


def build_longitudinal_pid_accel_sequence(
    *,
    v0: torch.Tensor,
    target_speed_mps: float,
    dt: float,
    horizon: int,
    kp: float,
    ki: float,
    kd: float,
    accel_bounds: tuple[float, float],
) -> torch.Tensor:
    """Build a fixed-target longitudinal PID acceleration sequence."""
    if horizon <= 0:
        raise RuntimeError("`horizon` must be > 0.")
    v = v0.to(dtype=torch.float32).reshape(-1)
    batch_size = int(v.shape[0])
    device = v.device
    target = torch.full((batch_size,), float(target_speed_mps), device=device, dtype=torch.float32)
    accel_min, accel_max = float(accel_bounds[0]), float(accel_bounds[1])
    integral = torch.zeros_like(v)
    prev_error = torch.zeros_like(v)
    accel_rows: list[torch.Tensor] = []

    for step_idx in range(horizon):
        error = target - v
        integral = integral + error * float(dt)
        if step_idx == 0:
            derivative = torch.zeros_like(error)
        else:
            derivative = (error - prev_error) / float(dt)
        accel = kp * error + ki * integral + kd * derivative
        accel = accel.clamp(min=accel_min, max=accel_max)
        accel_rows.append(accel)
        v = (v + float(dt) * accel).clamp(min=0.0)
        prev_error = error

    return torch.stack(accel_rows, dim=1)


def apply_longitudinal_pid_override(
    *,
    pred_action: torch.Tensor,
    v0: torch.Tensor,
    dt: float,
    target_speed_kmh: float,
    kp: float,
    ki: float,
    kd: float,
    accel_bounds: tuple[float, float],
) -> torch.Tensor:
    """Replace the longitudinal channel of a canonical `(accel, kappa)` action."""
    if pred_action.dim() != 3 or pred_action.shape[-1] != 2:
        raise RuntimeError(
            "Expected `pred_action` shaped (batch, k, 2) for PID override.\n"
            f"found={tuple(pred_action.shape)!r}"
        )
    pid_action = pred_action.to(dtype=torch.float32).clone()
    pid_accel = build_longitudinal_pid_accel_sequence(
        v0=v0,
        target_speed_mps=float(target_speed_kmh) / 3.6,
        dt=float(dt),
        horizon=int(pred_action.shape[1]),
        kp=float(kp),
        ki=float(ki),
        kd=float(kd),
        accel_bounds=accel_bounds,
    )
    pid_action[:, :, 0] = pid_accel
    return pid_action
