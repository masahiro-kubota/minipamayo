"""Action-space base class aligned with Alpamayo naming."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class ActionSpace(ABC, nn.Module):
    """Action space base class for trajectory generation."""

    @abstractmethod
    def traj_to_action(
        self,
        traj_history_xyz: torch.Tensor,
        traj_history_rot: torch.Tensor,
        traj_future_xyz: torch.Tensor,
        traj_future_rot: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Transform the future trajectory to the action space."""

    @abstractmethod
    def action_to_traj(
        self,
        action: torch.Tensor,
        traj_history_xyz: torch.Tensor,
        traj_history_rot: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform the action space back to a future trajectory."""

    @abstractmethod
    def get_action_space_dims(self) -> tuple[int, ...]:
        """Return the action-space dimensions."""

    def is_within_bounds(self, action: torch.Tensor) -> torch.Tensor:
        """Dummy bounds check used by action spaces without explicit bounds."""

        num_action_dims = len(self.get_action_space_dims())
        batch_shape = action.shape[:-num_action_dims] if num_action_dims > 0 else action.shape
        return torch.ones(batch_shape, dtype=torch.bool, device=action.device)
