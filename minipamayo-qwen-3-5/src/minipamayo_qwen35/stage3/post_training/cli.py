"""Shared CLI helpers for Stage 3 post-training entrypoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ...stage1.stage1_json_cli import load_stage1_config_args, parse_stage1_json_only_args
from ...utils.artifact_paths import ArtifactScope, scope_from_config_path
from ...utils.preflight import require_cuda_device, resolve_runtime_device


def load_stage3_config_args(
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


def parse_stage3_json_only_args(
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


def resolve_stage3_device(device_name: str) -> torch.device:
    return resolve_runtime_device(device_name)


def require_stage3_cuda_device(*, device_name: str, git_cwd: str | Path, error_message: str) -> torch.device:
    return require_cuda_device(
        device_name=device_name,
        git_cwd=Path(git_cwd),
        error_message=error_message,
    )


def artifact_scope_for_config(config_json: str, *, kind: str) -> ArtifactScope:
    return scope_from_config_path(
        config_json,
        kind=kind,
        stage="stage3",
        component="post_training",
        default_track="canonical",
    )
