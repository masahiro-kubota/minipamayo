"""Shared CLI helpers for canonical Stage 1B entrypoints."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

from ...utils.json_config import load_json_payload, normalize_arg_config, resolve_path_base
from ...utils.preflight import require_expected_cuda_toolkit

PROJECT_ROOT = Path(__file__).resolve().parents[4]
COMMON_CONFIG_PATH_KEYS = {"checkpoint", "stage1_checkpoint", "output_json"}


def add_stage1b_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config-json", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--stage1-checkpoint", type=str, default="")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--include-pid-override", action="store_true")
    parser.add_argument("--pid-target-speed-kmh", type=float, default=24.0)
    parser.add_argument("--pid-kp", type=float, default=1.0)
    parser.add_argument("--pid-ki", type=float, default=0.05)
    parser.add_argument("--pid-kd", type=float, default=0.0)


def load_stage1b_config_args(
    *,
    config_json: str,
    parser: argparse.ArgumentParser,
    path_keys: set[str],
    list_keys: set[str] | None = None,
) -> tuple[str, dict, dict]:
    config_path, payload = load_json_payload(config_json)
    raw_config = payload.get("args") if isinstance(payload, dict) and "args" in payload else payload
    if not isinstance(raw_config, dict):
        raise RuntimeError("Config JSON must be an object or an object with an `args` object.")
    base_dir = resolve_path_base(
        config_path,
        payload,
        default_base="project_root",
        base_dirs={"project_root": PROJECT_ROOT, "config_dir": config_path.parent},
    )
    config_args = normalize_arg_config(
        raw_config,
        parser,
        exclude_dests={"help", "config_json"},
        path_keys=path_keys,
        list_keys=list_keys or set(),
        base_dir=base_dir,
    )
    return str(config_path), payload, config_args


def parse_stage1b_json_only_args(
    *,
    parser: argparse.ArgumentParser,
    path_keys: set[str],
    list_keys: set[str] | None = None,
    json_only_error: str,
) -> argparse.Namespace:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        return parser.parse_args()
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config-json", type=str, required=True)
    pre_args, remaining = pre_parser.parse_known_args()
    if remaining:
        raise RuntimeError(json_only_error)
    config_path, config_payload, config_args = load_stage1b_config_args(
        config_json=pre_args.config_json,
        parser=parser,
        path_keys=path_keys,
        list_keys=list_keys,
    )
    parser.set_defaults(**config_args, config_json=config_path)
    args = parser.parse_args()
    args.config_json = config_path
    args.config_payload = config_payload
    args.config_args = config_args
    return args


def validate_stage1b_runtime_args(args: argparse.Namespace) -> None:
    if not args.checkpoint:
        raise RuntimeError("`checkpoint` must be defined in the config JSON.")
    if not args.stage1_checkpoint:
        raise RuntimeError("`stage1_checkpoint` must be defined in the config JSON.")
    if args.flow_steps <= 0:
        raise RuntimeError("`flow_steps` must be > 0.")
    if args.pid_target_speed_kmh <= 0.0:
        raise RuntimeError("`pid_target_speed_kmh` must be > 0.")


def require_stage1b_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("Canonical Stage 1B entrypoints require CUDA.")
    require_expected_cuda_toolkit()
    return torch.device("cuda")
