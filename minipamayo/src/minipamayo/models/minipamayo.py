"""MiniPamayo: integrated VLA model.

Architecture: DINOv2 ViT-B/14 → Adapter → Qwen2.5-0.5B → ActionHead
"""

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from .action_head import MLPActionHead
from .adapter import MeanPoolAdapter, PerTokenAdapter
from .vision_encoder import VisionEncoder


class MiniPamayo(nn.Module):
    """MiniPamayo VLA model.

    Forward: image → VisionEncoder → Adapter → LLM → ActionHead → action
    """

    def __init__(
        self,
        vision_model_name: str = "facebook/dinov2-base",
        llm_model_name: str = "Qwen/Qwen2.5-0.5B",
        adapter_type: str = "mean_pool",
        action_dim: int = 2,
    ):
        super().__init__()
        self.vision_encoder = VisionEncoder(vision_model_name)

        if adapter_type == "mean_pool":
            self.adapter = MeanPoolAdapter(vision_dim=self.vision_encoder.hidden_size, llm_dim=896)
        elif adapter_type == "per_token":
            self.adapter = PerTokenAdapter(vision_dim=self.vision_encoder.hidden_size, llm_dim=896)
        else:
            raise ValueError(f"Unknown adapter_type: {adapter_type}")

        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name, dtype=torch.bfloat16)
        self.action_head = MLPActionHead(input_dim=896, output_dim=action_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, 224, 224) ImageNet-normalized images
        Returns:
            action: (B, action_dim)
        """
        # 1. Vision Encoder: (B, 3, 224, 224) → (B, 256, 768)
        patch_features = self.vision_encoder(pixel_values)

        # 2. Adapter: (B, 256, 768) → (B, N_vis, 896)
        visual_tokens = self.adapter(patch_features)

        # 3. LLM: (B, N_vis, 896) → hidden states
        outputs = self.llm(
            inputs_embeds=visual_tokens,
            output_hidden_states=True,
        )
        # Take the last hidden state at the last token position
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # (B, 896)

        # 4. Action Head: (B, 896) → (B, action_dim)
        action = self.action_head(last_hidden)
        return action

    def set_stage0(self):
        """Stage 0: all modules trainable."""
        self.vision_encoder.unfreeze()
        self.adapter.requires_grad_(True)
        self.llm.requires_grad_(True)
        self.action_head.requires_grad_(True)

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.vision_encoder.dinov2.gradient_checkpointing_enable()
        self.llm.gradient_checkpointing_enable()

    def load_vlm_checkpoint(self, checkpoint_path: str | Path):
        """Load pre-trained VLM weights (VisionEncoder + Adapter + LLM).

        Adapter weights are loaded only if the architecture matches
        (PerTokenAdapter for the Cosmos Reason Mini checkpoint).
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        # Vision Encoder
        missing, unexpected = self.vision_encoder.load_state_dict(
            ckpt["vision_encoder_state_dict"], strict=False
        )
        if missing:
            print(f"[VisionEncoder] Missing keys: {missing}")
        if unexpected:
            print(f"[VisionEncoder] Unexpected keys: {unexpected}")

        # LLM
        missing, unexpected = self.llm.load_state_dict(ckpt["llm_state_dict"], strict=False)
        if missing:
            print(f"[LLM] Missing keys: {missing}")
        if unexpected:
            print(f"[LLM] Unexpected keys: {unexpected}")

        # Adapter: only load if architecture matches
        if isinstance(self.adapter, PerTokenAdapter):
            missing, unexpected = self.adapter.load_state_dict(
                ckpt["adapter_state_dict"], strict=False
            )
            if missing:
                print(f"[Adapter] Missing keys: {missing}")
            if unexpected:
                print(f"[Adapter] Unexpected keys: {unexpected}")
        else:
            print(
                f"[Adapter] Skipping checkpoint load: "
                f"checkpoint has PerTokenAdapter but model uses {type(self.adapter).__name__}"
            )

        print(
            f"Loaded VLM checkpoint from {checkpoint_path} (iteration {ckpt.get('iteration', '?')})"
        )

    def count_parameters(self) -> dict[str, int]:
        """Count parameters per module."""

        def _count(module: nn.Module) -> tuple[int, int]:
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return total, trainable

        modules = {
            "vision_encoder": self.vision_encoder,
            "adapter": self.adapter,
            "llm": self.llm,
            "action_head": self.action_head,
        }
        result = {}
        for name, mod in modules.items():
            total, trainable = _count(mod)
            result[name] = {"total": total, "trainable": trainable}
        return result
