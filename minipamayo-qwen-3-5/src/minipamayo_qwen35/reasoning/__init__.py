"""Shared synthetic-reasoning helpers used by Stage 2 and Stage 3."""

from .dataset import ReasoningSftJsonlDataset, reasoning_sft_collate
from .synthetic import (
    ACTION_SECTION_HEADER,
    build_reasoning_text,
    build_stage3_prompt_text,
    build_stage3_target_text,
    build_stage3_user_prompt,
    infer_driving_decision,
    normalize_label,
)
from .synthetic_dataset import SyntheticReasoningJsonlDataset, synthetic_reasoning_collate

__all__ = [
    "ACTION_SECTION_HEADER",
    "ReasoningSftJsonlDataset",
    "build_reasoning_text",
    "build_stage3_prompt_text",
    "build_stage3_target_text",
    "build_stage3_user_prompt",
    "infer_driving_decision",
    "normalize_label",
    "reasoning_sft_collate",
    "SyntheticReasoningJsonlDataset",
    "synthetic_reasoning_collate",
]
