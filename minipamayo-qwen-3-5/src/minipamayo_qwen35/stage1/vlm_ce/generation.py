from __future__ import annotations

import copy

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from ...contract.trajectory_tokens import Stage1TokenRegistry


class ActionTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids: list[int]):
        self.allowed_token_ids = tuple(int(token_id) for token_id in allowed_token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        del input_ids
        allowed_token_ids = torch.tensor(
            self.allowed_token_ids,
            device=scores.device,
            dtype=torch.long,
        )
        masked_scores = torch.full_like(scores, torch.finfo(scores.dtype).min)
        masked_scores.index_copy_(1, allowed_token_ids, scores.index_select(1, allowed_token_ids))
        return masked_scores


@torch.no_grad()
def greedy_generate_action_tokens(
    model,
    prompt_inputs: dict,
    registry: Stage1TokenRegistry,
    action_len: int,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.do_sample = False
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = int(action_len)
    generation_config.min_new_tokens = int(action_len)
    generation_config.return_dict_in_generate = True
    pad_token_id = generation_config.pad_token_id
    if pad_token_id is None:
        pad_token_id = getattr(model.config, "pad_token_id", None)
    if pad_token_id is None:
        eos_token_id = generation_config.eos_token_id
        if isinstance(eos_token_id, list):
            pad_token_id = int(eos_token_id[0]) if eos_token_id else None
        elif eos_token_id is not None:
            pad_token_id = int(eos_token_id)
    if pad_token_id is None:
        raise RuntimeError("Stage 1 greedy generation requires `pad_token_id` or `eos_token_id`.")
    generation_config.pad_token_id = int(pad_token_id)
    logits_processor = LogitsProcessorList([ActionTokenLogitsProcessor(list(registry.token_ids))])
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    with torch.autocast("cuda", dtype=model_dtype):
        outputs = model.generate(
            **prompt_inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
        )
    generated_ids = outputs.sequences[:, prompt_len:]
    if generated_ids.shape[1] != int(action_len):
        raise RuntimeError(
            "Stage 1 greedy generation returned an unexpected number of action tokens.\n"
            f"expected_action_len={int(action_len)}\n"
            f"generated_shape={tuple(generated_ids.shape)!r}"
        )
    return generated_ids
