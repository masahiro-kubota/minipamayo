"""Shared JSON-config CLI helpers for stage-scoped entrypoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..stage1.stage1_json_cli import load_stage1_config_args, parse_stage1_json_only_args
from .artifact_paths import ArtifactScope, scope_from_config_path
from .preflight import require_cuda_device, resolve_runtime_device


def load_stage_config_args(
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


def parse_stage_json_only_args(
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


def artifact_scope_for_config(
    config_json: str,
    *,
    kind: str,
    stage: str,
    component: str,
    default_track: str = "canonical",
) -> ArtifactScope:
    return scope_from_config_path(
        config_json,
        kind=kind,
        stage=stage,
        component=component,
        default_track=default_track,
    )


def resolve_stage_device(device_name: str) -> torch.device:
    return resolve_runtime_device(device_name)


def require_stage_cuda_device(*, device_name: str, git_cwd: str | Path, error_message: str) -> torch.device:
    return require_cuda_device(
        device_name=device_name,
        git_cwd=Path(git_cwd),
        error_message=error_message,
    )
