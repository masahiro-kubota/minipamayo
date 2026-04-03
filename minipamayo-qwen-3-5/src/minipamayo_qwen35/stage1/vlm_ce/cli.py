from __future__ import annotations

import argparse

from ...utils.artifact_paths import ArtifactScope, scope_from_config_path
from ..stage1_json_cli import load_stage1_config_args, parse_stage1_json_only_args


def load_config_args(
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


def parse_config_json_only_args(
    parser: argparse.ArgumentParser,
    *,
    path_keys: set[str],
    list_keys: set[str] | None = None,
    error_message: str,
) -> argparse.Namespace:
    return parse_stage1_json_only_args(
        parser,
        path_keys=path_keys,
        list_keys=list_keys,
        error_message=error_message,
    )


def artifact_scope_for_config(config_json: str, *, kind: str) -> ArtifactScope:
    return scope_from_config_path(
        config_json,
        kind=kind,
        stage="stage1",
        component="vlm_ce",
        default_track="canonical",
    )
