from __future__ import annotations

from pathlib import Path

import torch

from ..utils.train_runtime import (
    best_metric_from_history,
    format_gib,
    log_gpu_preflight,
    metric_improved,
    maybe_wandb_finish,
    maybe_wandb_log,
    release_cuda_memory,
    set_seed,
    write_run_config,
)


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
