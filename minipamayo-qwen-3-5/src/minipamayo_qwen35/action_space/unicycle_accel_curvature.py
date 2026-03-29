"""Canonical action space aligned with Alpamayo's unicycle accel+curvature file."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from ..contract.history_tokens import (
    canonicalize_history_batch_tensors,
    canonicalize_history_sample_tensors,
)
from ..geometry.rotation import rot_2d_to_3d, rotation_matrix_torch, so3_to_yaw_torch
from .action_space import ActionSpace
from .utils import (
    dxy_theta_to_v,
    dxy_theta_to_v_without_v0,
    solve_xs_eq_y,
    theta_smooth,
    unwrap_angle,
)


def _canonicalize_traj_group_tensor(name: str, value: torch.Tensor) -> torch.Tensor:
    if value.dim() == 3:
        return value
    if value.dim() == 4 and value.shape[-2:] == (3, 3):
        return value
    if value.dim() == 4 and value.shape[1] == 1:
        return value[:, 0]
    if value.dim() == 5 and value.shape[1] == 1:
        return value[:, 0]
    raise RuntimeError(
        f"Expected `{name}` shape (batch, T, ...) or (batch, 1, T, ...), got {tuple(value.shape)!r}."
    )


def _rotation_matrix_from_yaw(yaw_rad: float) -> np.ndarray:
    cos_yaw = math.cos(float(yaw_rad))
    sin_yaw = math.sin(float(yaw_rad))
    return np.asarray(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def canonicalize_future_sample_tensors(
    future_xyz: torch.Tensor,
    future_rot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if future_xyz.dim() == 2:
        future_xyz = future_xyz.unsqueeze(0)
    if future_rot.dim() == 3:
        future_rot = future_rot.unsqueeze(0)
    if future_xyz.dim() != 3 or future_xyz.shape[0] != 1 or future_xyz.shape[-1] != 3:
        raise RuntimeError(
            "Expected canonical `ego_future_xyz` sample tensor shape "
            f"(1, K, 3), got {tuple(future_xyz.shape)!r}."
        )
    if future_rot.dim() != 4 or future_rot.shape[0] != 1 or future_rot.shape[-2:] != (3, 3):
        raise RuntimeError(
            "Expected canonical `ego_future_rot` sample tensor shape "
            f"(1, K, 3, 3), got {tuple(future_rot.shape)!r}."
        )
    return future_xyz, future_rot


def derive_future_tensors_from_global_poses(record: dict) -> tuple[torch.Tensor, torch.Tensor]:
    if "ego_pose" not in record or "future_poses_global" not in record:
        raise RuntimeError(
            "Stage 1 record is missing `ego_pose` or `future_poses_global`, "
            "which are required to derive canonical future trajectory tensors."
        )
    ego_pose = record["ego_pose"]
    future_poses = record["future_poses_global"]
    if not isinstance(future_poses, list) or not future_poses:
        raise RuntimeError("Stage 1 record has invalid `future_poses_global`.")
    if "gt_waypoints" not in record or not isinstance(record["gt_waypoints"], list):
        raise RuntimeError("Stage 1 record is missing canonical `gt_waypoints` for future alignment.")
    target_steps = len(record["gt_waypoints"])
    if target_steps <= 0:
        raise RuntimeError("Stage 1 record has empty `gt_waypoints`.")
    future_poses = future_poses[:target_steps]

    origin_x = float(ego_pose["x"])
    origin_y = float(ego_pose["y"])
    origin_yaw = math.radians(float(ego_pose["yaw_deg"]))
    cos_yaw = math.cos(origin_yaw)
    sin_yaw = math.sin(origin_yaw)

    future_xyz = np.zeros((len(future_poses), 3), dtype=np.float32)
    future_rot = np.zeros((len(future_poses), 3, 3), dtype=np.float32)
    for pose_idx, pose in enumerate(future_poses):
        dx = float(pose["x"]) - origin_x
        dy = float(pose["y"]) - origin_y
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        future_xyz[pose_idx, 0] = local_x
        future_xyz[pose_idx, 1] = local_y

        yaw_rad = math.radians(float(pose["yaw_deg"]))
        local_yaw = math.atan2(math.sin(yaw_rad - origin_yaw), math.cos(yaw_rad - origin_yaw))
        future_rot[pose_idx] = _rotation_matrix_from_yaw(local_yaw)

    return (
        torch.from_numpy(future_xyz).unsqueeze(0),
        torch.from_numpy(future_rot).unsqueeze(0),
    )


class UnicycleAccelCurvatureActionSpace(ActionSpace):
    """Unicycle action space ported from Alpamayo with canonical single-group wrappers."""

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
        theta_lambda: float = 1e-6,
        theta_ridge: float = 1e-8,
        v_lambda: float = 1e-6,
        v_ridge: float = 1e-4,
        a_lambda: float = 1e-4,
        a_ridge: float = 1e-4,
        kappa_lambda: float = 1e-4,
        kappa_ridge: float = 1e-4,
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
        self.theta_lambda = float(theta_lambda)
        self.theta_ridge = float(theta_ridge)
        self.v_lambda = float(v_lambda)
        self.v_ridge = float(v_ridge)
        self.a_lambda = float(a_lambda)
        self.a_ridge = float(a_ridge)
        self.kappa_lambda = float(kappa_lambda)
        self.kappa_ridge = float(kappa_ridge)
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

    @torch.no_grad()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def _v_to_a(self, v: torch.Tensor) -> torch.Tensor:
        dv = (v[..., 1:] - v[..., :-1]) / self.dt
        return solve_xs_eq_y(
            s=torch.ones_like(dv),
            y=dv,
            dt=self.dt,
            lam=self.a_lambda,
            ridge=self.a_ridge,
            w_smooth1=None,
            w_smooth2=1.0,
            w_smooth3=None,
        )

    @torch.no_grad()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def _theta_v_a_to_kappa(
        self,
        theta: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        dtheta = theta[..., 1:] - theta[..., :-1]
        s = self.dt * v[..., :-1] + (self.dt**2) / 2.0 * a
        w = torch.ones_like(dtheta)
        return solve_xs_eq_y(
            s=s,
            y=dtheta,
            w_data=w,
            w_smooth1=None,
            w_smooth2=1.0,
            w_smooth3=None,
            lam=self.kappa_lambda,
            ridge=self.kappa_ridge,
            dt=self.dt,
        )

    @torch.no_grad()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def estimate_t0_states(
        self,
        traj_history_xyz: torch.Tensor,
        traj_history_rot: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        history_xyz, history_rot = canonicalize_history_batch_tensors(
            traj_history_xyz, traj_history_rot
        )
        hist_xyz = history_xyz[:, 0]
        hist_rot = history_rot[:, 0]
        full_xy = hist_xyz[..., :2]
        dxy = full_xy[..., 1:, :] - full_xy[..., :-1, :]
        theta = so3_to_yaw_torch(hist_rot)
        theta = unwrap_angle(theta)
        v = dxy_theta_to_v_without_v0(
            dxy=dxy,
            theta=theta,
            dt=self.dt,
            v_lambda=self.v_lambda,
            v_ridge=self.v_ridge,
        )
        return {"v": v[..., -1]}

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
        history_xyz, history_rot = canonicalize_history_batch_tensors(
            traj_history_xyz, traj_history_rot
        )
        history_xyz = history_xyz[:, 0]
        history_rot = history_rot[:, 0]
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

        full_xy = torch.cat([history_xyz[..., -1:, :], future_xyz], dim=-2)[..., :2]
        dxy = full_xy[..., 1:, :] - full_xy[..., :-1, :]
        theta = theta_smooth(
            traj_future_rot=future_rot,
            dt=self.dt,
            theta_lambda=self.theta_lambda,
            theta_ridge=self.theta_ridge,
        )
        v0 = t0_states["v"]
        v = dxy_theta_to_v(
            dxy=dxy,
            theta=theta,
            v0=v0,
            dt=self.dt,
            v_lambda=self.v_lambda,
            v_ridge=self.v_ridge,
        )
        accel = self._v_to_a(v)
        kappa = self._theta_v_a_to_kappa(theta, v, accel)

        accel = (accel - self.accel_mean.to(accel.device)) / self.accel_std.to(accel.device)
        kappa = (kappa - self.curvature_mean.to(kappa.device)) / self.curvature_std.to(
            kappa.device
        )
        action = torch.stack([accel, kappa], dim=-1)
        if not output_all_states:
            return action
        states = torch.stack([v[:, :-1], accel, theta[:, :-1]], dim=-1)
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
        history_xyz, history_rot = canonicalize_history_batch_tensors(
            traj_history_xyz, traj_history_rot
        )
        history_xyz = history_xyz[:, 0]
        history_rot = history_rot[:, 0]
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

        accel = action[..., 0]
        kappa = action[..., 1]

        accel_mean = self.accel_mean.to(accel.device)
        accel_std = self.accel_std.to(accel.device)
        kappa_mean = self.curvature_mean.to(kappa.device)
        kappa_std = self.curvature_std.to(kappa.device)
        accel = accel * accel_std + accel_mean
        kappa = kappa * kappa_std + kappa_mean

        if t0_states is None:
            t0_states = self.estimate_t0_states(history_xyz, history_rot)

        v0 = t0_states["v"]
        dt = self.dt
        dt_2_term = 0.5 * (self.dt**2)
        velocity = torch.cat(
            [v0.unsqueeze(-1), (v0.unsqueeze(-1) + torch.cumsum(accel * dt, dim=-1))],
            dim=-1,
        )
        initial_yaw = torch.zeros_like(v0)
        theta = torch.cat(
            [
                initial_yaw.unsqueeze(-1),
                (
                    initial_yaw.unsqueeze(-1)
                    + torch.cumsum(kappa * velocity[..., :-1] * dt, dim=-1)
                    + torch.cumsum(kappa * accel * dt_2_term, dim=-1)
                ),
            ],
            dim=-1,
        )
        half_dt_term = 0.5 * dt
        initial_x = torch.zeros_like(v0)
        initial_y = torch.zeros_like(v0)
        x = (
            initial_x.unsqueeze(-1)
            + torch.cumsum(
                velocity[..., :-1] * torch.cos(theta[..., :-1]) * half_dt_term,
                dim=-1,
            )
            + torch.cumsum(
                velocity[..., 1:] * torch.cos(theta[..., 1:]) * half_dt_term,
                dim=-1,
            )
        )
        y = (
            initial_y.unsqueeze(-1)
            + torch.cumsum(
                velocity[..., :-1] * torch.sin(theta[..., :-1]) * half_dt_term,
                dim=-1,
            )
            + torch.cumsum(
                velocity[..., 1:] * torch.sin(theta[..., 1:]) * half_dt_term,
                dim=-1,
            )
        )

        batch_dim = history_xyz.shape[:-2]
        pred_xyz = torch.zeros(
            *batch_dim,
            self.n_waypoints,
            3,
            device=history_xyz.device,
            dtype=history_xyz.dtype,
        )
        pred_xyz[..., 0] = x.to(dtype=pred_xyz.dtype)
        pred_xyz[..., 1] = y.to(dtype=pred_xyz.dtype)
        pred_xyz[..., 2] = history_xyz[..., -1:, 2]
        pred_rot = rot_2d_to_3d(rotation_matrix_torch(theta[..., 1:])).to(
            dtype=history_rot.dtype
        )
        pred_xyz = pred_xyz.unsqueeze(1)
        pred_rot = pred_rot.unsqueeze(1)
        return pred_xyz, pred_rot


def canonical_action_tensor_from_tensors(
    *,
    history_xyz: torch.Tensor,
    history_rot: torch.Tensor,
    future_xyz: torch.Tensor,
    future_rot: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    history_xyz, history_rot = canonicalize_history_sample_tensors(history_xyz, history_rot)
    future_xyz, future_rot = canonicalize_future_sample_tensors(future_xyz, future_rot)
    k_steps = int(future_xyz.shape[1])
    action_space = UnicycleAccelCurvatureActionSpace(n_waypoints=k_steps, dt=float(dt))
    action = action_space.traj_to_action(
        traj_history_xyz=history_xyz,
        traj_history_rot=history_rot,
        traj_future_xyz=future_xyz,
        traj_future_rot=future_rot,
    )
    return action.reshape(-1).to(torch.float32)


def canonical_action_tensor_from_record(record: dict) -> torch.Tensor:
    required_keys = ["ego_history_xyz", "ego_history_rot", "dt"]
    missing_keys = [key for key in required_keys if key not in record]
    if missing_keys:
        raise RuntimeError(
            "Stage 1 record is missing canonical action-contract fields:\n"
            + "\n".join(missing_keys)
        )
    history_xyz, history_rot = canonicalize_history_sample_tensors(
        torch.tensor(record["ego_history_xyz"], dtype=torch.float32),
        torch.tensor(record["ego_history_rot"], dtype=torch.float32),
    )
    if "ego_future_xyz" in record and "ego_future_rot" in record:
        future_xyz, future_rot = canonicalize_future_sample_tensors(
            torch.tensor(record["ego_future_xyz"], dtype=torch.float32),
            torch.tensor(record["ego_future_rot"], dtype=torch.float32),
        )
    else:
        future_xyz, future_rot = derive_future_tensors_from_global_poses(record)
    return canonical_action_tensor_from_tensors(
        history_xyz=history_xyz,
        history_rot=history_rot,
        future_xyz=future_xyz,
        future_rot=future_rot,
        dt=float(record["dt"]),
    )


def canonical_action_array_from_record(record: dict) -> np.ndarray:
    return canonical_action_tensor_from_record(record).detach().cpu().numpy()


def saved_action_tensor_from_record(record: dict) -> torch.Tensor:
    if "action" not in record:
        raise RuntimeError("Stage 1 record is missing canonical saved `action`.")
    action = torch.tensor(record["action"], dtype=torch.float32).reshape(-1)
    if action.numel() == 0 or action.numel() % 2 != 0:
        raise RuntimeError(
            "Stage 1 record has invalid saved `action` layout.\n"
            f"found={tuple(action.shape)!r}"
        )
    if "gt_waypoints" in record and isinstance(record["gt_waypoints"], list) and record["gt_waypoints"]:
        expected_dim = len(record["gt_waypoints"]) * 2
        if action.numel() != expected_dim:
            raise RuntimeError(
                "Saved `action` length does not match canonical waypoint count.\n"
                f"expected={expected_dim}\n"
                f"found={action.numel()}"
            )
    return action


def saved_action_array_from_record(record: dict) -> np.ndarray:
    return saved_action_tensor_from_record(record).detach().cpu().numpy()


def rollout_waypoints_from_action_tensor(
    *,
    action: torch.Tensor,
    history_xyz: torch.Tensor,
    history_rot: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    history_xyz, history_rot = canonicalize_history_batch_tensors(history_xyz, history_rot)
    if action.dim() == 1:
        if action.numel() % 2 != 0:
            raise RuntimeError(
                "Flat canonical action tensor must have an even number of scalars.\n"
                f"found={tuple(action.shape)!r}"
            )
        action = action.view(1, -1, 2)
    elif action.dim() == 2:
        if action.shape[-1] == 2:
            action = action.unsqueeze(0)
        else:
            if action.shape[-1] % 2 != 0:
                raise RuntimeError(
                    "Batched flat canonical action tensor must have an even trailing dimension.\n"
                    f"found={tuple(action.shape)!r}"
                )
            action = action.view(action.shape[0], -1, 2)
    elif action.dim() != 3 or action.shape[-1] != 2:
        raise RuntimeError(
            "Expected canonical action shaped (2*k,), (batch, 2*k), (k, 2), or (batch, k, 2).\n"
            f"found={tuple(action.shape)!r}"
        )

    k_steps = int(action.shape[-2])
    action_space = UnicycleAccelCurvatureActionSpace(n_waypoints=k_steps, dt=float(dt))
    future_xyz, _future_rot = action_space.action_to_traj(
        action.to(dtype=torch.float32),
        history_xyz.to(dtype=torch.float32),
        history_rot.to(dtype=torch.float32),
    )
    return future_xyz[:, 0, :, :2].to(torch.float32)
