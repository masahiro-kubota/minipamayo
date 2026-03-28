"""Shared helpers for canonical Stage 1B expert CFM training/evaluation."""

from __future__ import annotations

from pathlib import Path

import torch

from .. import CanonicalStage1Spec
from ..eval import load_components
from ..prompt import DEFAULT_QUESTION, build_prompt_text
from ..train import prepare_prompt_inputs_with_history


def freeze_module(module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def infer_prompt_text(checkpoint: dict, processor) -> str:
    stage1_metadata = checkpoint.get("stage1_metadata", {})
    question = str(stage1_metadata.get("question") or DEFAULT_QUESTION)
    history_token_count = int(stage1_metadata.get("history_token_count") or 0)
    return build_prompt_text(
        processor=processor,
        question=question,
        history_token_count=history_token_count,
    )


def load_stage1_condition_components(args):
    stage1_args = type(
        "Stage1ExpertArgs",
        (),
        {
            "checkpoint": str(Path(args.stage1_checkpoint).resolve()),
            "image_min_pixels": int(args.image_min_pixels),
            "image_max_pixels": int(args.image_max_pixels),
        },
    )()
    return load_components(stage1_args, task_spec=CanonicalStage1Spec())


def prepare_condition_inputs(
    batch: dict,
    processor,
    history_registry,
    history_quantizer,
    prompt_text: str,
    device: torch.device,
) -> dict:
    return prepare_prompt_inputs_with_history(
        batch=batch,
        processor=processor,
        history_registry=history_registry,
        history_quantizer=history_quantizer,
        prompt_text=prompt_text,
        device=device,
    )


def extract_last_layer_kv_cache(
    model,
    prompt_inputs: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(
        **prompt_inputs,
        use_cache=True,
        output_hidden_states=False,
        return_dict=True,
    )
    past_key_values = outputs.past_key_values
    if not past_key_values:
        raise RuntimeError("Frozen Stage 1 VLM did not return `past_key_values` for Stage 1B.")
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        key_cache = past_key_values.key_cache[-1]
        value_cache = past_key_values.value_cache[-1]
    else:
        last_layer = past_key_values[-1]
        if not isinstance(last_layer, tuple) or len(last_layer) < 2:
            raise RuntimeError("Unexpected `past_key_values` payload shape in Stage 1B.")
        key_cache = last_layer[0]
        value_cache = last_layer[1]
    if key_cache.dim() != 4 or value_cache.dim() != 4:
        raise RuntimeError("Stage 1B expects 4D key/value caches from the frozen VLM.")

    batch_size = prompt_inputs["input_ids"].shape[0]
    seq_len = prompt_inputs["input_ids"].shape[1]
    if key_cache.shape[0] != batch_size:
        if key_cache.shape[1] == batch_size:
            key_cache = key_cache.permute(1, 0, 2, 3)
            value_cache = value_cache.permute(1, 0, 2, 3)
        else:
            raise RuntimeError(
                "Could not align Stage 1B key/value cache batch dimension with prompt batch size."
            )

    if key_cache.shape[2] == seq_len:
        key_context = key_cache.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        value_context = value_cache.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
    elif key_cache.shape[1] == seq_len:
        key_context = key_cache.reshape(batch_size, seq_len, -1)
        value_context = value_cache.reshape(batch_size, seq_len, -1)
    else:
        raise RuntimeError(
            "Could not align key/value cache sequence length with the prompt attention mask in Stage 1B."
        )
    context = torch.cat([key_context, value_context], dim=-1).detach()
    attention_mask = prompt_inputs["attention_mask"].detach()
    return context, attention_mask


def compute_action_stats(dataset) -> dict[str, float]:
    accel_rows = []
    kappa_rows = []
    for index in range(len(dataset)):
        action = dataset[index]["action"].cpu().numpy()
        accel_rows.append(action[0::2])
        kappa_rows.append(action[1::2])
    accel = torch.tensor(accel_rows, dtype=torch.float32).flatten()
    kappa = torch.tensor(kappa_rows, dtype=torch.float32).flatten()
    accel_std = float(torch.std(accel, unbiased=False).item())
    kappa_std = float(torch.std(kappa, unbiased=False).item())
    if accel_std <= 0.0 or kappa_std <= 0.0:
        raise RuntimeError("Stage 1B action normalization requires non-zero accel and kappa std.")
    return {
        "accel_mean": float(torch.mean(accel).item()),
        "accel_std": accel_std,
        "kappa_mean": float(torch.mean(kappa).item()),
        "kappa_std": kappa_std,
    }


def build_stage1b_metadata(dataset, args, condition_dim: int) -> dict:
    record = dataset[0]
    action = record["action"]
    gt_waypoints = record["gt_waypoints"]
    dt_value = float(record["dt"].item()) if hasattr(record["dt"], "item") else float(record["dt"])
    return {
        "stage1_checkpoint": args.stage1_checkpoint,
        "train_jsonl": list(args.train_jsonl),
        "val_jsonl": list(args.val_jsonl) if args.val_jsonl is not None else None,
        "sample_format": "jsonl+images",
        "condition_source": "last_layer_past_key_value",
        "conditioning_contract": "detached_kv_cache_from_stage1a_prompt",
        "k": len(gt_waypoints),
        "action_dim": int(action.shape[0]),
        "dt": dt_value,
        "condition_dim": int(condition_dim),
    }
