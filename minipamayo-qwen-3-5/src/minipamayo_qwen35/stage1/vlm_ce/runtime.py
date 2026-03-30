from __future__ import annotations

import gc
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ...contract.prompt import build_prompt_text
from ...contract.record_adapter import rollout_waypoints_from_action_tensor
from ...contract.task_spec import CanonicalStage1Spec, Stage1TaskSpec
from ...utils.preflight import collect_gpu_preflight_snapshot
from ...contract.trajectory_tokens import Stage1TokenRegistry
from .components import load_components
from .generation import greedy_generate_action_tokens
from .metrics import compute_token_accuracy
from .prompting import (
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_gib(num_bytes: int) -> float:
    return round(num_bytes / (1024**3), 3)


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def move_value_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_value_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_value_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_value_to_device(item, device) for item in value)
    return value


def move_optimizer_state_to_device(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            state[key] = move_value_to_device(value, device)


def log_gpu_preflight(device: torch.device) -> dict:
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    snapshot = collect_gpu_preflight_snapshot(gpu_index=device_index)
    print(json.dumps({"event": "gpu_preflight", **snapshot}, ensure_ascii=False))
    if snapshot["warning_reasons"]:
        print(
            json.dumps(
                {
                    "event": "gpu_preflight_warning",
                    "gpu_index": device_index,
                    "warning_reasons": snapshot["warning_reasons"],
                    "non_self_compute_processes": snapshot.get("non_self_compute_processes", []),
                },
                ensure_ascii=False,
            )
        )
    return snapshot


def write_run_config(save_dir: Path, args, run_metadata: dict) -> None:
    with (save_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config_json": args.config_json,
                "config_payload": args.config_payload,
                "resolved_args": vars(args),
                "run_metadata": run_metadata,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


def maybe_wandb_log(run, data: dict, step: int | None = None) -> None:
    if run is None:
        raise RuntimeError("W&B run is unexpectedly unavailable.")
    run.log(data, step=step)


def maybe_wandb_finish(run) -> None:
    if run is None:
        raise RuntimeError("W&B run is unexpectedly unavailable.")
    run.finish()


def metric_improved(current: float, best: float, min_delta: float) -> bool:
    if math.isinf(best):
        return True
    return current < (best - min_delta)


def best_metric_from_history(metrics_history: list[dict], metric_name: str) -> tuple[float, int]:
    best_metric = float("inf")
    best_epoch = 0
    for metrics in metrics_history:
        if metric_name not in metrics or "epoch" not in metrics:
            raise RuntimeError(
                f"Metrics history is missing canonical fields `{metric_name}` or `epoch`: {metrics!r}"
            )
        value = metrics[metric_name]
        if value is None:
            raise RuntimeError(f"Metrics history contains null `{metric_name}`: {metrics!r}")
        value = float(value)
        if value < best_metric:
            best_metric = value
            best_epoch = int(metrics["epoch"])
    return best_metric, best_epoch


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
