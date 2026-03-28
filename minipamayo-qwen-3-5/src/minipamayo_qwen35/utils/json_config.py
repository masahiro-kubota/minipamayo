"""Helpers for JSON-backed script configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json_payload(config_json: str) -> tuple[Path, object]:
    config_path = Path(config_json).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return config_path, payload


def resolve_path_base(
    config_path: Path,
    payload,
    *,
    default_base: str,
    base_dirs: dict[str, Path],
) -> Path:
    path_base = default_base
    if isinstance(payload, dict):
        path_base = str(payload.get("path_base", default_base))

    if path_base not in base_dirs:
        choices = ", ".join(sorted(base_dirs))
        raise RuntimeError(f"Config path_base must be one of: {choices}.")
    return base_dirs[path_base]


def coerce_config_value(action, value):
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        if not isinstance(value, bool):
            raise RuntimeError(f"Config key `{action.dest}` must be a boolean.")
        return value

    if action.type is not None and value is not None:
        try:
            value = action.type(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Config key `{action.dest}` could not be parsed: {value!r}") from exc

    if action.choices is not None and value not in action.choices:
        choices = ", ".join(str(choice) for choice in action.choices)
        raise RuntimeError(f"Config key `{action.dest}` must be one of: {choices}.")

    return value


def normalize_arg_config(
    raw_config: dict,
    parser: argparse.ArgumentParser,
    *,
    exclude_dests: set[str] | None = None,
    path_keys: set[str] | None = None,
    base_dir: Path | None = None,
) -> dict:
    exclude = exclude_dests or set()
    valid_actions = {
        action.dest: action
        for action in parser._actions
        if action.dest not in exclude
    }
    unknown_keys = sorted(set(raw_config) - set(valid_actions))
    if unknown_keys:
        raise RuntimeError(f"Unknown config keys: {', '.join(unknown_keys)}")

    config_args: dict = {}
    path_arg_keys = path_keys or set()
    for key, value in raw_config.items():
        normalized = coerce_config_value(valid_actions[key], value)
        if key in path_arg_keys and normalized:
            path = Path(normalized)
            if not path.is_absolute():
                if base_dir is None:
                    raise RuntimeError(f"Relative path key `{key}` requires a base_dir.")
                normalized = str((base_dir / path).resolve())
            else:
                normalized = str(path)
        config_args[key] = normalized
    return config_args
