"""Action prediction heads for MiniPamayo.

Phase 3: MLPActionHead — simple MLP for [steer, throttle] regression
Phase 4: MLPActionHead with expanded output for (a, kappa) x 64 waypoints
"""

import torch
import torch.nn as nn


class MLPActionHead(nn.Module):
    """MLP regression head for action prediction.

    Phase 3: (896,) → (2,) [steer, throttle]
    Phase 4: (896,) -> (128,) = (64, 2) [(a, kappa) x 64 waypoints]
    """

    def __init__(self, input_dim: int = 896, output_dim: int = 2, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: (B, input_dim) — LLM last hidden state
        Returns:
            action: (B, output_dim)
        """
        return self.mlp(hidden_state)
