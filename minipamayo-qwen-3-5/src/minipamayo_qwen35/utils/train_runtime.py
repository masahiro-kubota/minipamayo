"""Shared training/runtime helpers reused across stages."""

from __future__ import annotations

import gc
import json
import math
import random
from pathlib import Path

import torch

from .preflight import collect_gpu_preflight_snapshot


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def format_gib(num_bytes: int) -> float:
    return round(num_bytes / (1024**3), 3)


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
