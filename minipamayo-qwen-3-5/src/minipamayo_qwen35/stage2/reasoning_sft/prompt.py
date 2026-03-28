"""Stage 2-local prompt builders."""

from __future__ import annotations

from ...stage1.common.prompt import build_reasoning_prompt_text


def build_stage2_prompt_text(processor, v0: float, history_token_count: int = 0) -> str:
    del v0
    return build_reasoning_prompt_text(processor, history_token_count=history_token_count)
