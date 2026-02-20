"""DINOv2 ViT-B/14 vision encoder wrapper."""

import torch
import torch.nn as nn
from transformers import Dinov2Model


class VisionEncoder(nn.Module):
    """DINOv2 ViT-B/14 wrapper. Returns patch tokens only (CLS excluded).

    Output: (B, 256, 768) for ViT-B/14 with 224x224 input.
    """

    def __init__(self, model_name: str = "facebook/dinov2-base"):
        super().__init__()
        self.dinov2 = Dinov2Model.from_pretrained(model_name)
        self.hidden_size = self.dinov2.config.hidden_size  # 768

    def freeze(self):
        self.dinov2.requires_grad_(False)

    def unfreeze(self):
        self.dinov2.requires_grad_(True)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, 224, 224) ImageNet-normalized
        Returns:
            patch_tokens: (B, 256, 768)
        """
        output = self.dinov2(pixel_values=pixel_values)
        # last_hidden_state: (B, 1+256, 768) — CLS + 256 patch tokens
        return output.last_hidden_state[:, 1:, :]
