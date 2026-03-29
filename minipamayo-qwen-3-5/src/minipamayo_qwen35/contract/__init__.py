"""Shared Stage 1 / Stage 2 contract exports."""

from ..action_space.discrete_action_space import DiscreteTrajectoryTokenizer
from .history_tokens import (
    HISTORY_END_TOKEN,
    HISTORY_PLACEHOLDER_TOKEN,
    HISTORY_START_TOKEN,
    HISTORY_SPECIAL_TOKENS,
    HistoryTokenRegistry,
    HistoryTrajectoryQuantizer,
    canonicalize_history_batch_tensors,
    canonicalize_history_sample_numpy,
    canonicalize_history_sample_tensors,
    encode_history_token_id_rows,
)
from .prompt import (
    ALPAMAYO_REASONING_USER_TEXT,
    COT_END_TOKEN,
    COT_START_TOKEN,
    DEFAULT_QUESTION,
    DEFAULT_SYSTEM_PROMPT,
    PROMPT_SPECIAL_TOKENS,
    TRAJ_FUTURE_END_TOKEN,
    TRAJ_FUTURE_START_TOKEN,
    add_prompt_special_tokens,
    build_chat_prompt_text,
    build_history_placeholder,
    build_messages,
    build_prompt_text,
    build_reasoning_prompt_text,
)
from .sequence_layout import (
    STAGE1A_TARGET_LAYOUT,
    STAGE2_PROMPT_CONTRACT,
    STAGE2_TARGET_LAYOUT,
)
from .task_spec import CanonicalStage1Spec, KappaOnlyStage1Spec, Stage1TaskSpec
from .trajectory_tokens import Stage1TokenRegistry, format_stage1_token

__all__ = [
    "ALPAMAYO_REASONING_USER_TEXT",
    "COT_END_TOKEN",
    "COT_START_TOKEN",
    "CanonicalStage1Spec",
    "DEFAULT_QUESTION",
    "DEFAULT_SYSTEM_PROMPT",
    "DiscreteTrajectoryTokenizer",
    "HISTORY_END_TOKEN",
    "HISTORY_PLACEHOLDER_TOKEN",
    "HISTORY_SPECIAL_TOKENS",
    "HISTORY_START_TOKEN",
    "HistoryTokenRegistry",
    "HistoryTrajectoryQuantizer",
    "KappaOnlyStage1Spec",
    "PROMPT_SPECIAL_TOKENS",
    "STAGE1A_TARGET_LAYOUT",
    "STAGE2_PROMPT_CONTRACT",
    "STAGE2_TARGET_LAYOUT",
    "Stage1TaskSpec",
    "Stage1TokenRegistry",
    "TRAJ_FUTURE_END_TOKEN",
    "TRAJ_FUTURE_START_TOKEN",
    "add_prompt_special_tokens",
    "build_chat_prompt_text",
    "build_history_placeholder",
    "build_messages",
    "build_prompt_text",
    "build_reasoning_prompt_text",
    "canonicalize_history_batch_tensors",
    "canonicalize_history_sample_numpy",
    "canonicalize_history_sample_tensors",
    "encode_history_token_id_rows",
    "format_stage1_token",
]
