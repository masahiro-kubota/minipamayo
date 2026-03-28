"""Shared Stage 1 prompt and target-spec utilities."""

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
from .task_spec import CanonicalStage1Spec, KappaOnlyStage1Spec, Stage1TaskSpec

__all__ = [
    "ALPAMAYO_REASONING_USER_TEXT",
    "COT_END_TOKEN",
    "COT_START_TOKEN",
    "DEFAULT_QUESTION",
    "DEFAULT_SYSTEM_PROMPT",
    "PROMPT_SPECIAL_TOKENS",
    "TRAJ_FUTURE_END_TOKEN",
    "TRAJ_FUTURE_START_TOKEN",
    "add_prompt_special_tokens",
    "build_chat_prompt_text",
    "build_history_placeholder",
    "build_messages",
    "build_prompt_text",
    "build_reasoning_prompt_text",
    "CanonicalStage1Spec",
    "KappaOnlyStage1Spec",
    "Stage1TaskSpec",
]
