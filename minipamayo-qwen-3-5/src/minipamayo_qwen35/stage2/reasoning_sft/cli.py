"""Shared CLI helpers for canonical Stage 2 reasoning-SFT entrypoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ...stage1.stage1_json_cli import load_stage1_config_args, parse_stage1_json_only_args
from ...utils.preflight import enforce_runtime_prerequisites


def load_stage2_config_args(
    config_json: str,
    parser: argparse.ArgumentParser,
    *,
    path_keys: set[str],
    list_keys: set[str] | None = None,
) -> tuple[str, dict, dict]:
    return load_stage1_config_args(
        config_json,
        parser,
        path_keys=path_keys,
        list_keys=list_keys,
    )


def parse_stage2_json_only_args(
    *,
    parser: argparse.ArgumentParser,
    path_keys: set[str],
    list_keys: set[str] | None = None,
    json_only_error: str,
) -> argparse.Namespace:
    return parse_stage1_json_only_args(
        parser=parser,
        path_keys=path_keys,
        list_keys=list_keys,
        error_message=json_only_error,
    )


def resolve_stage2_device(device_name: str) -> torch.device:
    return torch.device(
        device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )


def require_stage2_cuda_device(*, device_name: str, git_cwd: str | Path, error_message: str) -> torch.device:
    device = resolve_stage2_device(device_name)
    if device.type != "cuda":
        raise RuntimeError(error_message)
    enforce_runtime_prerequisites(git_cwd=Path(git_cwd))
    return device
