"""Grouped rollout sampling for canonical Stage 3."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch

from ....contract.prompt import build_reasoning_prompt_text
from ....models.prompt_inputs import (
    append_token_to_model_inputs,
    model_forward_inputs,
    prepare_prompt_inputs_with_history,
)
from ..common import clone_model_inputs, maybe_autocast
from .parser import ParsedStage3Sequence, parse_generated_sequence


@dataclass
class Stage3Rollout:
    parsed: ParsedStage3Sequence
    policy_token_logprobs: torch.Tensor
    ref_token_logprobs: torch.Tensor
    pred_future_xyz: torch.Tensor
    pred_future_rot: torch.Tensor

    @property
    def policy_logprob_sum(self) -> torch.Tensor:
        return self.policy_token_logprobs.sum()

    @property
    def ref_logprob_sum(self) -> torch.Tensor:
        return self.ref_token_logprobs.sum()


def _generate_rollout_token_ids(
    model,
    prompt_inputs: dict,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    pad_token_id: int | None,
) -> list[int]:
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.do_sample = True
    generation_config.num_return_sequences = 1
    generation_config.temperature = temperature
    generation_config.top_p = top_p
    generation_config.top_k = top_k if top_k > 0 else None
    generation_config.max_new_tokens = max_new_tokens
    generation_config.return_dict_in_generate = True
    generation_config.output_logits = False
    generation_config.pad_token_id = pad_token_id

    with torch.inference_mode():
        outputs = model.generate(
            **prompt_inputs,
            generation_config=generation_config,
        )
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    return [int(token_id) for token_id in outputs.sequences[0, prompt_len:].tolist()]


def _compute_sequence_log_probs(
    model,
    prompt_inputs: dict,
    token_ids: list[int],
    *,
    model_dtype: torch.dtype,
    device: torch.device,
    with_grad: bool,
) -> torch.Tensor:
    if not token_ids:
        raise RuntimeError("Stage 3 rollout generation returned an empty sequence.")
    full_inputs = clone_model_inputs(prompt_inputs)
    for token_id in token_ids:
        append_token_to_model_inputs(
            model,
            full_inputs,
            torch.tensor([token_id], device=device, dtype=torch.long),
        )
    context = maybe_autocast(device, model_dtype)
    grad_context = torch.enable_grad() if with_grad else torch.no_grad()
    with grad_context:
        with context:
            outputs = model(**model_forward_inputs(full_inputs))
    prefix_len = prompt_inputs["input_ids"].shape[1]
    logits = outputs.logits[:, prefix_len - 1 : -1, :]
    token_tensor = torch.tensor(token_ids, device=logits.device, dtype=torch.long)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return log_probs[0, torch.arange(len(token_ids), device=logits.device), token_tensor]


def prepare_stage3_prompt_inputs(bundle, batch: dict) -> dict:
    prompt_text = build_reasoning_prompt_text(
        bundle.processor,
        history_token_count=bundle.history_quantizer.token_count,
    )
    return prepare_prompt_inputs_with_history(
        model=bundle.policy_model,
        batch=batch,
        processor=bundle.processor,
        history_registry=bundle.history_registry,
        history_quantizer=bundle.history_quantizer,
        prompt_text=prompt_text,
        device=bundle.device,
    )


def _decode_motion_rollout(bundle, batch: dict, parsed: ParsedStage3Sequence) -> tuple[torch.Tensor, torch.Tensor]:
    action_token_tensor = torch.tensor(
        [parsed.padded_action_token_ids],
        device=bundle.device,
        dtype=torch.long,
    )
    hist_xyz = batch["ego_history_xyz"].to(device=bundle.device, dtype=torch.float32)
    hist_rot = batch["ego_history_rot"].to(device=bundle.device, dtype=torch.float32)
    # Canonical Stage 3 routes motion decode through the frozen wrapper bundle so
    # the Stage 2 policy and frozen motion stack share the same tokenizer contract.
    pred_future_xyz, pred_future_rot, _ = bundle.wrapper.traj_tokenizer.decode(
        hist_xyz,
        hist_rot,
        action_token_tensor,
    )
    return pred_future_xyz[0].detach().cpu(), pred_future_rot[0].detach().cpu()


def generate_grouped_rollouts(
    *,
    bundle,
    batch: dict,
    num_rollouts: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    requires_policy_grad: bool,
) -> list[Stage3Rollout]:
    prompt_inputs = prepare_stage3_prompt_inputs(bundle, batch)
    rollouts: list[Stage3Rollout] = []
    for _ in range(num_rollouts):
        token_ids = _generate_rollout_token_ids(
            bundle.policy_model,
            prompt_inputs,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            pad_token_id=bundle.processor.tokenizer.pad_token_id,
        )
        parsed = parse_generated_sequence(
            token_ids=token_ids,
            tokenizer=bundle.processor.tokenizer,
            registry=bundle.registry,
            quantizer=bundle.quantizer,
            expected_action_token_count=bundle.expected_action_token_count,
        )
        policy_token_logprobs = _compute_sequence_log_probs(
            bundle.policy_model,
            prompt_inputs,
            parsed.generated_token_ids,
            model_dtype=bundle.model_dtype,
            device=bundle.device,
            with_grad=requires_policy_grad,
        )
        ref_token_logprobs = _compute_sequence_log_probs(
            bundle.reference_model,
            prompt_inputs,
            parsed.generated_token_ids,
            model_dtype=bundle.model_dtype,
            device=bundle.device,
            with_grad=False,
        )
        pred_future_xyz, pred_future_rot = _decode_motion_rollout(bundle, batch, parsed)
        rollouts.append(
            Stage3Rollout(
                parsed=parsed,
                policy_token_logprobs=policy_token_logprobs,
                ref_token_logprobs=ref_token_logprobs,
                pred_future_xyz=pred_future_xyz,
                pred_future_rot=pred_future_rot,
            )
        )
    return rollouts
