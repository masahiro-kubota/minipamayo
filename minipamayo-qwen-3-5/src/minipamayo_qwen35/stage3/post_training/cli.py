"""Shared CLI helpers for Stage 3 post-training entrypoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ...utils.artifact_paths import ArtifactScope
from ...utils.stage_cli import (
    artifact_scope_for_config as build_artifact_scope_for_config,
    load_stage_config_args,
    parse_stage_json_only_args,
    require_stage_cuda_device,
    resolve_stage_device,
)


def load_stage3_config_args(
    config_json: str,
    parser: argparse.ArgumentParser,
    *,
    path_keys: set[str],
    list_keys: set[str] | None = None,
) -> tuple[str, dict, dict]:
    return load_stage_config_args(
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
    return parse_stage_json_only_args(
        parser=parser,
        path_keys=path_keys,
        list_keys=list_keys,
        json_only_error=json_only_error,
    )


def resolve_stage3_device(device_name: str) -> torch.device:
    return resolve_stage_device(device_name)


def require_stage3_cuda_device(*, device_name: str, git_cwd: str | Path, error_message: str) -> torch.device:
    return require_stage_cuda_device(
        device_name=device_name,
        git_cwd=git_cwd,
        error_message=error_message,
    )


def artifact_scope_for_config(config_json: str, *, kind: str) -> ArtifactScope:
    return build_artifact_scope_for_config(
        config_json,
        kind=kind,
        stage="stage3",
        component="post_training",
        default_track="canonical",
    )
