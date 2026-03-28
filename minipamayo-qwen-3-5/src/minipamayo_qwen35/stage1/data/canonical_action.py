"""Canonical Stage 1 action-contract helpers.

These helpers keep Stage 1A discrete-token supervision, Stage 1B continuous
expert supervision, extraction metadata, and evaluation rollout on the same
Alpamayo-aligned unicycle action space.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..expert_cfm.action_space import UnicycleAccelCurvatureActionSpace
from ..tokenization.history import (
    canonicalize_history_batch_tensors,
    canonicalize_history_sample_tensors,
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
