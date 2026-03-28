"""Alpamayo-style action-space wrapper for canonical Stage 1B."""

from __future__ import annotations

import torch

from ...utils.dynamics import forward_dynamics_batch
from ..tokenization.history import canonicalize_history_batch_tensors


def _estimate_t0_speed(history_xyz: torch.Tensor, dt: float) -> torch.Tensor:
    if dt <= 0.0:
        raise RuntimeError("`dt` must be > 0 for action-space rollout.")
    if history_xyz.shape[-2] < 2:
        raise RuntimeError(
            "Canonical Stage 1B action space requires at least two history steps to estimate speed."
        )
    current_xy = history_xyz[:, 0, -1, :2]
    prev_xy = history_xyz[:, 0, -2, :2]
    displacement = torch.linalg.norm(current_xy - prev_xy, dim=-1)
    return displacement / float(dt)


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


class UnicycleAccelCurvatureActionSpace:
    """Minimal single-trajectory-group action-space API."""

    def __init__(self, *, k: int, dt: float):
        self.k = int(k)
        self.dt = float(dt)

    def get_action_space_dims(self) -> tuple[int, int]:
        return (self.k, 2)

    def action_to_traj(
        self,
        *,
        traj_history_xyz: torch.Tensor,
        traj_history_rot: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_xyz, history_rot = canonicalize_history_batch_tensors(
            traj_history_xyz,
            traj_history_rot,
        )
        if action.dim() == 2:
            action = action.reshape(action.shape[0], self.k, 2)
        if action.dim() != 3 or tuple(action.shape[-2:]) != (self.k, 2):
            raise RuntimeError(
                "Canonical Stage 1B action space expects action shape "
                f"(batch, {self.k}, 2), got {tuple(action.shape)!r}."
            )
        v0 = _estimate_t0_speed(history_xyz, self.dt)
        pred_xy = forward_dynamics_batch(
            action[:, :, 0],
            action[:, :, 1],
            v0.to(device=action.device, dtype=action.dtype),
            dt=self.dt,
        )
        pred_xyz = torch.cat(
            [
                pred_xy,
                torch.zeros(
                    pred_xy.shape[0],
                    pred_xy.shape[1],
                    1,
                    device=pred_xy.device,
                    dtype=pred_xy.dtype,
                ),
            ],
            dim=-1,
        ).unsqueeze(1)

        if pred_xy.shape[1] == 1:
            yaw = torch.zeros(pred_xy.shape[0], 1, device=pred_xy.device, dtype=pred_xy.dtype)
        else:
            deltas = pred_xy[:, 1:] - pred_xy[:, :-1]
            step_yaw = torch.atan2(deltas[..., 1], deltas[..., 0])
            yaw = torch.cat([step_yaw[:, :1], step_yaw], dim=1)
        pred_rot = _yaw_to_rotmat(yaw).unsqueeze(1)
        return pred_xyz, pred_rot
