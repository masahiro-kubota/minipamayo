"""Shared CLI helpers for canonical Stage 1B entrypoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ...utils.artifact_paths import ArtifactScope, scope_from_config_path
from ...utils.preflight import require_cuda_device
from ..stage1_json_cli import load_stage1_config_args, parse_stage1_json_only_args

COMMON_CONFIG_PATH_KEYS = {"checkpoint", "stage1_checkpoint", "output_json"}


def add_stage1b_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config-json", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--stage1-checkpoint", type=str, default="")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--flow-steps", type=int, default=0)
    parser.add_argument("--include-pid-override", action="store_true")
    parser.add_argument("--pid-target-speed-kmh", type=float, default=0.0)
    parser.add_argument("--pid-kp", type=float, default=-1.0)
    parser.add_argument("--pid-ki", type=float, default=-1.0)
    parser.add_argument("--pid-kd", type=float, default=-1.0)


def load_stage1b_config_args(
    *,
    config_json: str,
    parser: argparse.ArgumentParser,
    path_keys: set[str],
    list_keys: set[str] | None = None,
) -> tuple[str, dict, dict]:
    return load_stage1_config_args(
        config_json,
        parser,
        path_keys=path_keys,
        list_keys=list_keys,
    )


def parse_stage1b_json_only_args(
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


def validate_stage1b_runtime_args(args: argparse.Namespace) -> None:
    if not args.checkpoint:
        raise RuntimeError("`checkpoint` must be defined in the config JSON.")
    if not args.stage1_checkpoint:
        raise RuntimeError("`stage1_checkpoint` must be defined in the config JSON.")
    if args.flow_steps <= 0:
        raise RuntimeError("`flow_steps` must be > 0.")
    if bool(getattr(args, "include_pid_override", False)):
        if args.pid_target_speed_kmh <= 0.0:
            raise RuntimeError("`pid_target_speed_kmh` must be > 0 when `include_pid_override=true`.")
        if args.pid_kp < 0.0:
            raise RuntimeError("`pid_kp` must be >= 0 when `include_pid_override=true`.")
        if args.pid_ki < 0.0:
            raise RuntimeError("`pid_ki` must be >= 0 when `include_pid_override=true`.")
        if args.pid_kd < 0.0:
            raise RuntimeError("`pid_kd` must be >= 0 when `include_pid_override=true`.")


def require_stage1b_cuda_device() -> torch.device:
    return require_cuda_device(
        device_name="cuda",
        git_cwd=Path(__file__).resolve().parent,
        error_message="Canonical Stage 1B entrypoints require CUDA.",
    )


def artifact_scope_for_config(config_json: str, *, kind: str) -> ArtifactScope:
    return scope_from_config_path(
        config_json,
        kind=kind,
        stage="stage1",
        component="expert_cfm",
        default_track="canonical",
    )
