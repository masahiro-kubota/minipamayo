from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from ..contract.prompt import build_prompt_text
from ..contract.record_adapter import rollout_waypoints_from_action_tensor
from ..contract.task_spec import CanonicalStage1Spec, Stage1TaskSpec
from .stage1a_components import load_components
from .stage1a_prompting import (
    build_full_inputs_from_prompt_inputs,
    model_forward_inputs,
    prepare_alpamayo_prompt_inputs_with_history,
    prepare_batch,
)


@dataclass
class Stage1ARuntime:
    checkpoint: dict
    model: Any
    processor: Any
    registry: Any
    history_registry: Any
    history_quantizer: Any
    quantizer: Any
    model_dtype: torch.dtype
    task_spec: Stage1TaskSpec
    stage1_metadata: dict
    question: str
    history_token_count: int
    target_dim: int
    full_action_dim: int
    k_steps: int
    dt: float


REQUIRED_STAGE1_METADATA_KEYS = [
    "question",
    "target_dim",
    "full_action_dim",
    "k",
    "dt",
    "history_steps",
    "history_token_count",
    "action_representation",
    "rollout_accel_source",
]


class _ActionTokenLogitsProcessor(LogitsProcessor):
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


def compute_token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    shifted_logits = logits[:, :-1, :].argmax(dim=-1)
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != -100
    correct = ((shifted_logits == shifted_labels) & mask).sum().item()
    total = mask.sum().item()
    return int(correct), int(total)


@torch.no_grad()
def greedy_generate_action_tokens(
    model,
    prompt_inputs: dict,
    registry,
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
    logits_processor = LogitsProcessorList([_ActionTokenLogitsProcessor(list(registry.token_ids))])
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


def load_stage1a_runtime(
    args,
    task_spec: Stage1TaskSpec | None = None,
    *,
    device: torch.device | None = None,
) -> Stage1ARuntime:
    task_spec = task_spec or CanonicalStage1Spec()
    (
        checkpoint,
        model,
        processor,
        registry,
        history_registry,
        history_quantizer,
        quantizer,
        model_dtype,
    ) = load_components(args, task_spec)
    if device is not None:
        model.to(device)
    stage1_metadata = checkpoint.get("stage1_metadata")
    if not isinstance(stage1_metadata, dict):
        raise RuntimeError("Checkpoint is missing canonical `stage1_metadata`.")
    task_spec.validate_checkpoint(stage1_metadata)
    missing_stage1_keys = [key for key in REQUIRED_STAGE1_METADATA_KEYS if key not in stage1_metadata]
    if missing_stage1_keys:
        raise RuntimeError(
            "Checkpoint is missing canonical Stage 1 metadata:\n" + "\n".join(missing_stage1_keys)
        )
    return Stage1ARuntime(
        checkpoint=checkpoint,
        model=model,
        processor=processor,
        registry=registry,
        history_registry=history_registry,
        history_quantizer=history_quantizer,
        quantizer=quantizer,
        model_dtype=model_dtype,
        task_spec=task_spec,
        stage1_metadata=stage1_metadata,
        question=str(stage1_metadata["question"]),
        history_token_count=int(stage1_metadata["history_token_count"]),
        target_dim=int(stage1_metadata["target_dim"]),
        full_action_dim=int(stage1_metadata["full_action_dim"]),
        k_steps=int(stage1_metadata["k"]),
        dt=float(stage1_metadata["dt"]),
    )


def build_stage1a_prompt_text(runtime: Stage1ARuntime) -> str:
    return build_prompt_text(
        runtime.processor,
        runtime.question,
        history_token_count=runtime.history_quantizer.token_count,
    )


def prepare_stage1a_prompt_inputs(
    runtime: Stage1ARuntime,
    batch: dict,
    *,
    device: torch.device,
    prompt_mode: str,
) -> dict:
    if prompt_mode == "canonical_teacher_forced":
        return prepare_batch(
            runtime.model,
            batch,
            runtime.processor,
            runtime.registry,
            runtime.history_registry,
            runtime.history_quantizer,
            runtime.quantizer,
            runtime.task_spec,
            build_stage1a_prompt_text(runtime),
            device,
        )[0]
    if prompt_mode == "alpamayo_message_rollout":
        return prepare_alpamayo_prompt_inputs_with_history(
            model=runtime.model,
            batch=batch,
            processor=runtime.processor,
            history_registry=runtime.history_registry,
            history_quantizer=runtime.history_quantizer,
            question=runtime.question,
            history_token_count=runtime.history_token_count,
            device=device,
        )
    raise RuntimeError(f"Unsupported prompt_mode: {prompt_mode!r}")


def prepare_stage1a_training_batch(
    runtime: Stage1ARuntime,
    batch: dict,
    *,
    device: torch.device,
) -> tuple[dict, torch.Tensor]:
    return prepare_batch(
        runtime.model,
        batch,
        runtime.processor,
        runtime.registry,
        runtime.history_registry,
        runtime.history_quantizer,
        runtime.quantizer,
        runtime.task_spec,
        build_stage1a_prompt_text(runtime),
        device,
    )


def run_stage1a_teacher_forced_batch(
    runtime: Stage1ARuntime,
    batch: dict,
    *,
    device: torch.device,
    prompt_mode: str,
) -> dict:
    if prompt_mode == "canonical_teacher_forced":
        full_inputs, labels = prepare_stage1a_training_batch(runtime, batch, device=device)
    elif prompt_mode == "alpamayo_message_rollout":
        prompt_inputs = prepare_alpamayo_prompt_inputs_with_history(
            model=runtime.model,
            batch=batch,
            processor=runtime.processor,
            history_registry=runtime.history_registry,
            history_quantizer=runtime.history_quantizer,
            question=runtime.question,
            history_token_count=runtime.history_token_count,
            device=device,
        )
        full_inputs, labels = build_full_inputs_from_prompt_inputs(
            model=runtime.model,
            prompt_inputs=prompt_inputs,
            batch=batch,
            registry=runtime.registry,
            quantizer=runtime.quantizer,
            task_spec=runtime.task_spec,
            device=device,
        )
    else:
        raise RuntimeError(f"Unsupported prompt_mode: {prompt_mode!r}")
    with torch.autocast("cuda", dtype=runtime.model_dtype):
        outputs = runtime.model(**model_forward_inputs(full_inputs), labels=labels)
    correct, total = compute_token_accuracy(outputs.logits, labels)
    return {
        "full_inputs": full_inputs,
        "labels": labels,
        "outputs": outputs,
        "correct": correct,
        "total": total,
    }


def decode_stage1a_generated_batch(
    runtime: Stage1ARuntime,
    batch: dict,
    generated_token_ids: torch.Tensor,
) -> list[dict]:
    rows: list[dict] = []
    for row_idx in range(generated_token_ids.shape[0]):
        pred_token_ids = [int(token_id) for token_id in generated_token_ids[row_idx].detach().cpu().tolist()]
        pred_bin_ids = runtime.registry.decode_token_ids_to_bin_ids(pred_token_ids)
        pred_target = runtime.registry.decode_target_token_ids(pred_token_ids, runtime.quantizer)
        pred_target_tensor = torch.tensor(pred_target, dtype=torch.float32)
        gt_action_tensor = batch["action"][row_idx].detach().cpu().reshape(-1).to(torch.float32)
        pred_action_tensor = runtime.task_spec.full_action_from_target_tensor(
            pred_target_tensor,
            gt_action_tensor=gt_action_tensor,
        )
        gt_target_tensor = runtime.task_spec.target_from_action_tensor(gt_action_tensor)
        gt_waypoints = batch["gt_waypoints"][row_idx].detach().cpu().reshape(-1, 2).to(torch.float32)
        pred_waypoints = rollout_waypoints_from_action_tensor(
            action=pred_action_tensor.view(1, runtime.k_steps, 2),
            history_xyz=batch["ego_history_xyz"][row_idx].detach().cpu().to(dtype=torch.float32),
            history_rot=batch["ego_history_rot"][row_idx].detach().cpu().to(dtype=torch.float32),
            dt=runtime.dt,
        )[0].detach().cpu()
        waypoint_errors = torch.norm(pred_waypoints - gt_waypoints, dim=-1)
        rows.append(
            {
                "pred_token_ids": pred_token_ids,
                "pred_bin_ids": pred_bin_ids,
                "pred_target_tensor": pred_target_tensor.detach().cpu(),
                "gt_target_tensor": gt_target_tensor.detach().cpu(),
                "pred_action_tensor": pred_action_tensor.detach().cpu(),
                "gt_action_tensor": gt_action_tensor.detach().cpu(),
                "pred_waypoints": pred_waypoints,
                "gt_waypoints": gt_waypoints.detach().cpu(),
                "target_mae": float(
                    torch.mean(torch.abs(pred_target_tensor.detach().cpu() - gt_target_tensor.detach().cpu())).item()
                ),
                "ade_m": float(waypoint_errors.mean().item()),
                "fde_m": float(waypoint_errors[-1].item()),
                "waypoint_errors": waypoint_errors,
            }
        )
    return rows


def run_stage1a_rollout_batch(
    runtime: Stage1ARuntime,
    batch: dict,
    *,
    device: torch.device,
    prompt_mode: str = "alpamayo_message_rollout",
) -> dict:
    if prompt_mode != "alpamayo_message_rollout":
        raise RuntimeError("Stage 1 rollout currently expects `alpamayo_message_rollout`.")
    prompt_inputs = prepare_alpamayo_prompt_inputs_with_history(
        model=runtime.model,
        batch=batch,
        processor=runtime.processor,
        history_registry=runtime.history_registry,
        history_quantizer=runtime.history_quantizer,
        question=runtime.question,
        history_token_count=runtime.history_token_count,
        device=device,
    )
    generated_token_ids = greedy_generate_action_tokens(
        runtime.model,
        prompt_inputs,
        runtime.registry,
        runtime.target_dim,
        runtime.model_dtype,
    )
    gt_token_rows = runtime.task_spec.encode_target_token_rows_from_batch(
        batch,
        runtime.registry,
        runtime.quantizer,
    )
    gt_token_ids = torch.tensor(gt_token_rows, device=device, dtype=torch.long)
    return {
        "prompt_inputs": prompt_inputs,
        "generated_token_ids": generated_token_ids,
        "gt_token_ids": gt_token_ids,
        "decoded_rows": decode_stage1a_generated_batch(runtime, batch, generated_token_ids),
    }


def extract_prompt_cache(model, prompt_inputs: dict) -> tuple[object, torch.Tensor]:
    outputs = model(
        **model_forward_inputs(prompt_inputs),
        use_cache=True,
        output_hidden_states=False,
        return_dict=True,
    )
    past_key_values = outputs.past_key_values
    if not past_key_values:
        raise RuntimeError("Frozen Stage 1A VLM did not return `past_key_values`.")
    attention_mask = prompt_inputs["attention_mask"].detach()
    return past_key_values, attention_mask
