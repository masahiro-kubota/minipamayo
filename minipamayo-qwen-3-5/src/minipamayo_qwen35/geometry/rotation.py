"""Torch rotation helpers aligned with Alpamayo geometry naming."""

from __future__ import annotations

import torch


def so3_to_yaw_torch(rot_mat: torch.Tensor) -> torch.Tensor:
    """Compute yaw from SO(3) rotation matrices assuming xyz Euler order."""

    cos_th_cos_phi = rot_mat[..., 0, 0]
    cos_th_sin_phi = rot_mat[..., 1, 0]
    return torch.atan2(cos_th_sin_phi, cos_th_cos_phi)


def rotation_matrix_torch(angle: torch.Tensor) -> torch.Tensor:
    """Create one or more 2D rotation matrices from yaw angles."""

    return torch.stack(
        [
            torch.stack([torch.cos(angle), -torch.sin(angle)], dim=-1),
            torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1),
        ],
        dim=-2,
    )


def rot_2d_to_3d(rot: torch.Tensor) -> torch.Tensor:
    """Lift 2D rotation matrices to flat-ground 3D rotation matrices."""

    return torch.cat(
        [
            torch.cat([rot, torch.zeros_like(rot[..., :1])], dim=-1),
            torch.tensor([0.0, 0.0, 1.0], device=rot.device, dtype=rot.dtype).repeat(
                rot.shape[:-2] + (1, 1)
            ),
        ],
        dim=-2,
    )


def round_2pi_torch(x: torch.Tensor) -> torch.Tensor:
    """Normalize angles to [-pi, pi]."""

    return torch.atan2(torch.sin(x), torch.cos(x))
