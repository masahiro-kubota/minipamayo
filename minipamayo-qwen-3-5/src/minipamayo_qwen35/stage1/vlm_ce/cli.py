from __future__ import annotations

import argparse

from ...utils.artifact_paths import ArtifactScope
from ...utils.stage_cli import (
    artifact_scope_for_config as build_artifact_scope_for_config,
    load_stage_config_args,
    parse_stage_json_only_args,
)


def load_config_args(
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


def parse_config_json_only_args(
    parser: argparse.ArgumentParser,
    *,
    path_keys: set[str],
    list_keys: set[str] | None = None,
    error_message: str,
) -> argparse.Namespace:
    return parse_stage_json_only_args(
        parser=parser,
        path_keys=path_keys,
        list_keys=list_keys,
        json_only_error=error_message,
    )


def artifact_scope_for_config(config_json: str, *, kind: str) -> ArtifactScope:
    return build_artifact_scope_for_config(
        config_json,
        kind=kind,
        stage="stage1",
        component="vlm_ce",
        default_track="canonical",
    )
