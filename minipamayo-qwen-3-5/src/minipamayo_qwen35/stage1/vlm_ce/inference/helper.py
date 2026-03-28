"""Alpamayo helper-compatible Stage 1 inference utilities."""

from __future__ import annotations

import collections.abc
from typing import Any

import torch
from transformers import AutoProcessor

from ....utils.image_budget import CANONICAL_IMAGE_MAX_PIXELS, CANONICAL_IMAGE_MIN_PIXELS

MIN_PIXELS = CANONICAL_IMAGE_MIN_PIXELS
MAX_PIXELS = CANONICAL_IMAGE_MAX_PIXELS
def create_message(frames: torch.Tensor):
    """Construct the Alpamayo-style message using images and CoT prefill."""
    if frames.ndim != 4:
        raise ValueError(f"{frames.ndim=}, expected 4 (N, C, H, W)")

    num_traj_token = 48
    hist_traj_placeholder = (
        f"<|traj_history_start|>{'<|traj_history|>' * num_traj_token}<|traj_history_end|>"
    )

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a driving assistant that generates safe and accurate actions.",
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "image", "image": frame} for frame in frames]
            + [
                {
                    "type": "text",
                    "text": (
                        f"{hist_traj_placeholder}"
                        "output the chain-of-thought reasoning of the driving process, "
                        "then output the future trajectory."
                    ),
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "<|cot_start|>",
                }
            ],
        },
    ]


def get_processor(processor_path: str) -> AutoProcessor:
    """Load the saved canonical processor with Alpamayo-style pixel settings."""
    processor = AutoProcessor.from_pretrained(
        processor_path,
        trust_remote_code=True,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    return processor


def to_device(
    data: Any,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Any:
    """Recursively cast data into the specified device, dtype."""
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)
    if isinstance(data, collections.abc.Mapping):
        return {key: to_device(data[key], device=device, dtype=dtype) for key in data}
    if isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes)):
        return [to_device(elem, device=device, dtype=dtype) for elem in data]
    return data
