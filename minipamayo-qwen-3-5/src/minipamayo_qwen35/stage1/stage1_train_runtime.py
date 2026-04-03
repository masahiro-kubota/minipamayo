from __future__ import annotations

import gc
import math
from pathlib import Path

import torch

from ..utils.train_runtime import (
    format_gib,
    log_gpu_preflight,
    maybe_wandb_finish,
    maybe_wandb_log,
    set_seed,
    write_run_config,
)


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
