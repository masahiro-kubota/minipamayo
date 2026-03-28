"""Canonical Stage 1B action space aligned with Alpamayo's public API surface."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from ..tokenization.history import canonicalize_history_batch_tensors


def _yaw_from_rot(rot: torch.Tensor) -> torch.Tensor:
    return torch.atan2(rot[..., 1, 0], rot[..., 0, 0])


def _yaw_to_rotmat(yaw: torch.Tensor) -> torch.Tensor:
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    rot = torch.zeros((*yaw.shape, 3, 3), device=yaw.device, dtype=yaw.dtype)
    rot[..., 0, 0] = cos_yaw
    rot[..., 0, 1] = -sin_yaw
    rot[..., 1, 0] = sin_yaw
    rot[..., 1, 1] = cos_yaw
    rot[..., 2, 2] = 1.0
    return rot


def _canonicalize_traj_group_tensor(name: str, value: torch.Tensor) -> torch.Tensor:
    if value.dim() == 3:
        return value
    if value.dim() == 4 and value.shape[1] == 1:
        return value[:, 0]
    raise RuntimeError(
        f"Expected `{name}` shape (batch, T, ...) or (batch, 1, T, ...), got {tuple(value.shape)!r}."
    )


class UnicycleAccelCurvatureActionSpace(nn.Module):
    """Unicycle action space with Alpamayo-like API and canonical single-group support."""

    def __init__(
        self,
        *,
        k: int | None = None,
        n_waypoints: int | None = None,
        dt: float = 0.1,
        accel_mean: float = 0.0,
        accel_std: float = 1.0,
        curvature_mean: float = 0.0,
        curvature_std: float = 1.0,
        accel_bounds: tuple[float, float] = (-9.8, 9.8),
        curvature_bounds: tuple[float, float] = (-0.2, 0.2),
    ):
        super().__init__()
        resolved_k = int(n_waypoints if n_waypoints is not None else (k if k is not None else 64))
        if resolved_k <= 0:
            raise RuntimeError("`k` / `n_waypoints` must be > 0.")
        if dt <= 0.0:
            raise RuntimeError("`dt` must be > 0.")
        if accel_std <= 0.0 or curvature_std <= 0.0:
            raise RuntimeError("Action normalization std must be > 0.")

        self.k = resolved_k
        self.n_waypoints = resolved_k
        self.dt = float(dt)
        self.accel_bounds = tuple(float(v) for v in accel_bounds)
        self.curvature_bounds = tuple(float(v) for v in curvature_bounds)
        self.register_buffer("accel_mean", torch.tensor(float(accel_mean), dtype=torch.float32))
        self.register_buffer("accel_std", torch.tensor(float(accel_std), dtype=torch.float32))
        self.register_buffer(
            "curvature_mean",
            torch.tensor(float(curvature_mean), dtype=torch.float32),
        )
        self.register_buffer(
            "curvature_std",
            torch.tensor(float(curvature_std), dtype=torch.float32),
        )

    def get_action_space_dims(self) -> tuple[int, int]:
        return (self.k, 2)

    def is_within_bounds(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-2:] != (self.k, 2):
            raise RuntimeError(
                "Action tensor has incompatible shape for bounds check.\n"
                f"expected=(*, {self.k}, 2)\n"
                f"found={tuple(action.shape)!r}"
            )
        accel = action[..., 0] * self.accel_std + self.accel_mean
        curvature = action[..., 1] * self.curvature_std + self.curvature_mean
        accel_ok = (accel >= self.accel_bounds[0]) & (accel <= self.accel_bounds[1])
        curvature_ok = (curvature >= self.curvature_bounds[0]) & (
            curvature <= self.curvature_bounds[1]
        )
        return torch.all(accel_ok & curvature_ok, dim=-1)

    def estimate_t0_states(
        self,
        traj_history_xyz: torch.Tensor,
        traj_history_rot: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        history_xyz, history_rot = canonicalize_history_batch_tensors(traj_history_xyz, traj_history_rot)
        hist_xyz = history_xyz[:, 0]
        hist_rot = history_rot[:, 0]
        if hist_xyz.shape[1] < 2:
            raise RuntimeError("Need at least two history steps to estimate t0 state.")
        current_xy = hist_xyz[:, -1, :2]
        prev_xy = hist_xyz[:, -2, :2]
        speed = torch.linalg.norm(current_xy - prev_xy, dim=-1) / self.dt
        yaw = _yaw_from_rot(hist_rot[:, -1])
        return {"v": speed, "yaw": yaw}

    def _denormalize_action(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        accel = action[..., 0] * self.accel_std.to(action.device) + self.accel_mean.to(action.device)
        curvature = (
            action[..., 1] * self.curvature_std.to(action.device)
            + self.curvature_mean.to(action.device)
        )
        return accel, curvature

    def _normalize_action(self, accel: torch.Tensor, curvature: torch.Tensor) -> torch.Tensor:
        accel = (accel - self.accel_mean.to(accel.device)) / self.accel_std.to(accel.device)
        curvature = (
            curvature - self.curvature_mean.to(curvature.device)
        ) / self.curvature_std.to(curvature.device)
        return torch.stack([accel, curvature], dim=-1)

    def traj_to_action(
        self,
        traj_history_xyz: torch.Tensor,
        traj_history_rot: torch.Tensor,
        traj_future_xyz: torch.Tensor,
        traj_future_rot: torch.Tensor,
        t0_states: dict[str, torch.Tensor] | None = None,
        output_all_states: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del args, kwargs
        history_xyz, history_rot = canonicalize_history_batch_tensors(traj_history_xyz, traj_history_rot)
        future_xyz = _canonicalize_traj_group_tensor("traj_future_xyz", traj_future_xyz)
        future_rot = _canonicalize_traj_group_tensor("traj_future_rot", traj_future_rot)
        if future_xyz.shape[1] != self.k:
            raise RuntimeError(
                "Future trajectory length does not match canonical action horizon.\n"
                f"expected_k={self.k}\n"
                f"found_future_steps={future_xyz.shape[1]}"
            )
        if t0_states is None:
            t0_states = self.estimate_t0_states(history_xyz, history_rot)

        full_xy = torch.cat([history_xyz[:, 0, -1:, :2], future_xyz[:, :, :2]], dim=1)
        delta_xy = full_xy[:, 1:] - full_xy[:, :-1]
        step_speed = torch.linalg.norm(delta_xy, dim=-1) / self.dt
        prev_speed = torch.cat([t0_states["v"].unsqueeze(1), step_speed[:, :-1]], dim=1)
        accel = (step_speed - prev_speed) / self.dt

        future_yaw = _yaw_from_rot(future_rot)
        prev_yaw = t0_states["yaw"].unsqueeze(1)
        yaw_seq = torch.cat([prev_yaw, future_yaw], dim=1)
        delta_yaw = torch.atan2(
            torch.sin(yaw_seq[:, 1:] - yaw_seq[:, :-1]),
            torch.cos(yaw_seq[:, 1:] - yaw_seq[:, :-1]),
        )
        distance = torch.linalg.norm(delta_xy, dim=-1).clamp_min(1e-4)
        curvature = delta_yaw / distance

        action = self._normalize_action(accel, curvature)
        if not output_all_states:
            return action
        states = torch.stack([prev_speed, accel, yaw_seq[:, :-1]], dim=-1)
        return action, states

    def action_to_traj(
        self,
        action: torch.Tensor,
        traj_history_xyz: torch.Tensor,
        traj_history_rot: torch.Tensor,
        t0_states: dict[str, torch.Tensor] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del args, kwargs
        history_xyz, history_rot = canonicalize_history_batch_tensors(traj_history_xyz, traj_history_rot)
        if action.dim() == 2:
            action = action.unsqueeze(0)
        if action.dim() == 4 and action.shape[1] == 1:
            action = action[:, 0]
        if action.shape[-2:] != (self.k, 2):
            raise RuntimeError(
                "Canonical action tensor must have shape (batch, k, 2).\n"
                f"expected=(*, {self.k}, 2)\n"
                f"found={tuple(action.shape)!r}"
            )
        if t0_states is None:
            t0_states = self.estimate_t0_states(history_xyz, history_rot)

        accel, curvature = self._denormalize_action(action.to(torch.float32))
        batch_size = accel.shape[0]
        device = accel.device
        dtype = accel.dtype

        x = torch.zeros((batch_size, self.k), device=device, dtype=dtype)
        y = torch.zeros((batch_size, self.k), device=device, dtype=dtype)
        theta = torch.zeros((batch_size, self.k + 1), device=device, dtype=dtype)
        theta[:, 0] = t0_states["yaw"].to(device=device, dtype=dtype)
        v = t0_states["v"].to(device=device, dtype=dtype)

        for step_idx in range(self.k):
            prev_v = v
            ds = prev_v * self.dt + 0.5 * accel[:, step_idx] * (self.dt**2)
            theta[:, step_idx + 1] = theta[:, step_idx] + curvature[:, step_idx] * ds
            if step_idx == 0:
                x[:, step_idx] = ds * torch.cos(theta[:, step_idx])
                y[:, step_idx] = ds * torch.sin(theta[:, step_idx])
            else:
                x[:, step_idx] = x[:, step_idx - 1] + ds * torch.cos(theta[:, step_idx])
                y[:, step_idx] = y[:, step_idx - 1] + ds * torch.sin(theta[:, step_idx])
            v = torch.clamp(prev_v + accel[:, step_idx] * self.dt, min=0.0)

        pred_xyz = torch.zeros((batch_size, 1, self.k, 3), device=device, dtype=dtype)
        pred_xyz[:, 0, :, 0] = x
        pred_xyz[:, 0, :, 1] = y
        pred_xyz[:, 0, :, 2] = history_xyz[:, 0, -1:, 2].expand(-1, self.k)

        pred_rot = _yaw_to_rotmat(theta[:, 1:]).unsqueeze(1)
        return pred_xyz, pred_rot
