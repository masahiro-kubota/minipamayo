"""Vision → LLM Adapter modules.

Phase 3 (fail-fast): MeanPoolAdapter — mean pool + linear → 1 visual token
Phase 4 (control-based): CrossAttentionAdapter — learnable queries (16 tokens)
PerTokenAdapter — per-token MLP (from checkpoint, 256 tokens)
"""

import torch
import torch.nn as nn


class MeanPoolAdapter(nn.Module):
    """Mean pool + linear projection → 1 visual token.

    Simplest adapter for Phase 3 fail-fast. Averages all patch tokens
    then projects to LLM dimension.
    """

    def __init__(self, vision_dim: int = 768, llm_dim: int = 896):
        super().__init__()
        self.proj = nn.Linear(vision_dim, llm_dim)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (B, N_patches, vision_dim)  e.g. (B, 256, 768)
        Returns:
            visual_tokens: (B, 1, llm_dim)  e.g. (B, 1, 896)
        """
        pooled = vision_features.mean(dim=1)  # (B, vision_dim)
        projected = self.proj(pooled)  # (B, llm_dim)
        return projected.unsqueeze(1)  # (B, 1, llm_dim)


class PerTokenAdapter(nn.Module):
    """Per-token 2-layer MLP projector (from Qwen2.5-VL Mini / Cosmos Reason Mini).

    Projects each patch token independently. Compatible with pre-trained checkpoint.
    Output: (B, 256, 896) — 256 visual tokens.
    """

    def __init__(self, vision_dim: int = 768, llm_dim: int = 896, ratio: int = 4):
        super().__init__()
        hidden_dim = vision_dim * ratio  # 3072
        self.mlp = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_dim),
        )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (B, N_patches, vision_dim)  e.g. (B, 256, 768)
        Returns:
            visual_tokens: (B, N_patches, llm_dim)  e.g. (B, 256, 896)
        """
        return self.mlp(vision_features)


class CrossAttentionAdapter(nn.Module):
    """Cross-Attention Pooling adapter with learnable queries.

    16 learnable queries attend to DINOv2 patch features via cross-attention,
    producing 16 visual tokens for LLM input. Phase 4 upgrade from MeanPoolAdapter.

    Output: (B, num_queries, llm_dim)  e.g. (B, 16, 896)
    """

    def __init__(
        self,
        vision_dim: int = 768,
        llm_dim: int = 896,
        num_queries: int = 16,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, llm_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=llm_dim,
            num_heads=num_heads,
            kdim=vision_dim,
            vdim=vision_dim,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(llm_dim)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: (B, N_patches, vision_dim)  e.g. (B, 256, 768)
        Returns:
            visual_tokens: (B, num_queries, llm_dim)  e.g. (B, 16, 896)
        """
        B = vision_features.shape[0]
        queries = self.queries.expand(B, -1, -1)
        attn_out, _ = self.cross_attn(queries, vision_features, vision_features)
        return self.norm(attn_out + queries)
