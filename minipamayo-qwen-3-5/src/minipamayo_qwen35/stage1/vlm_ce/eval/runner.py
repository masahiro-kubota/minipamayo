"""Canonical Stage 1A evaluation for the Qwen3.5 branch.

Evaluates:
- teacher-forced loss / token accuracy
- autoregressive token accuracy
- action-space MAE
- trajectory ADE / FDE via forward dynamics
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mcap.well_known import MessageEncoding, SchemaEncoding
from mcap.writer import CompressionType, Writer
from PIL import Image
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    LogitsProcessor,
    LogitsProcessorList,
)

from ....utils.image_budget import (
    CANONICAL_IMAGE_MAX_PIXELS,
    CANONICAL_IMAGE_MIN_PIXELS,
    validate_canonical_image_budget,
)
from ....utils.json_config import load_json_payload, normalize_arg_config, resolve_path_base
from ....utils.preflight import enforce_runtime_prerequisites
from ....utils.run_metadata import (
    collect_dataset_view_fingerprint,
    collect_git_metadata,
    collect_gpu_info,
    collect_processor_settings,
)
from ...checkpoint_completion import require_completed_training_run
from ....contract.task_spec import CanonicalStage1Spec, Stage1TaskSpec
from ...dataset import Stage1JsonlDataset
from ....contract.record_adapter import rollout_waypoints_from_action_tensor
from ....contract.prompt import (
    add_prompt_special_tokens,
    build_multimodal_messages,
    build_stage1_question_user_text,
)
from ....contract.history_tokens import HistoryTokenRegistry, HistoryTrajectoryQuantizer
from ....contract.trajectory_tokens import Stage1TokenRegistry
from ....helper import to_device
from ...dataset import stage1_collate
from ..train import (
    CHECKPOINT_KIND_FULL,
    CHECKPOINT_KIND_MODEL_ONLY,
    build_full_inputs_from_prompt_inputs,
    build_processor_kwargs,
    build_model_load_kwargs,
    compute_token_accuracy,
    format_gib,
    inject_history_inputs_embeds,
    load_checkpoint,
    model_forward_inputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[5]
CONFIG_PATH_KEYS = {
    "checkpoint",
    "test_jsonl",
    "output_json",
    "output_mcap",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate canonical Stage 1A checkpoints.")
    parser.add_argument("--config-json", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--test-jsonl", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--show-samples", type=int, default=10)
    parser.add_argument("--image-min-pixels", type=int, default=CANONICAL_IMAGE_MIN_PIXELS)
    parser.add_argument("--image-max-pixels", type=int, default=CANONICAL_IMAGE_MAX_PIXELS)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--output-mcap", type=str, default="")
    return parser


def _load_config_args(config_json: str, parser: argparse.ArgumentParser) -> tuple[str, dict, dict]:
    config_path, payload = load_json_payload(config_json)
    raw_config = payload.get("args") if isinstance(payload, dict) and "args" in payload else payload
    if not isinstance(raw_config, dict):
        raise RuntimeError("Config JSON must be an object or an object with an `args` object.")

    base_dir = resolve_path_base(
        config_path,
        payload,
        default_base="project_root",
        base_dirs={
            "project_root": PROJECT_ROOT,
            "config_dir": config_path.parent,
        },
    )
    config_args = normalize_arg_config(
        raw_config,
        parser,
        exclude_dests={"help", "config_json"},
        path_keys=CONFIG_PATH_KEYS,
        base_dir=base_dir,
    )
    return str(config_path), payload, config_args


def parse_args() -> argparse.Namespace:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        return build_parser().parse_args()

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config-json", type=str, required=True)
    pre_args, remaining = pre_parser.parse_known_args()
    if remaining:
        raise RuntimeError(
            "Stage 1 evaluation accepts only --config-json. Put all settings in the JSON file."
        )

    parser = build_parser()
    config_path, config_payload, config_args = _load_config_args(pre_args.config_json, parser)
    parser.set_defaults(**config_args, config_json=config_path)
    args = parser.parse_args()
    args.config_json = config_path
    args.config_payload = config_payload
    args.config_args = config_args
    if not args.checkpoint:
        raise RuntimeError("`checkpoint` must be defined in the config JSON.")
    if not args.test_jsonl:
        raise RuntimeError("`test_jsonl` must be defined in the config JSON.")
    validate_canonical_image_budget(args.image_min_pixels, args.image_max_pixels)
    return args


def resolve_checkpoint_args(checkpoint: dict) -> dict:
    if "checkpoint_kind" not in checkpoint:
        raise RuntimeError("Checkpoint is missing canonical `checkpoint_kind`.")
    checkpoint_kind = checkpoint["checkpoint_kind"]
    if checkpoint_kind not in {CHECKPOINT_KIND_FULL, CHECKPOINT_KIND_MODEL_ONLY}:
        raise RuntimeError(f"Unsupported checkpoint_kind: {checkpoint_kind!r}")
    if "args" not in checkpoint:
        raise RuntimeError("Checkpoint is missing canonical `args` metadata.")
    checkpoint_args = checkpoint["args"]
    if not isinstance(checkpoint_args, dict):
        raise RuntimeError("Checkpoint is missing canonical `args` metadata.")
    required_keys = ["model_path", "dtype"]
    missing_keys = [key for key in required_keys if key not in checkpoint_args]
    if missing_keys:
        raise RuntimeError(
            "Checkpoint is missing canonical `args` fields:\n" + "\n".join(missing_keys)
        )
    return checkpoint_args


def resolve_processor_path(checkpoint_path: Path) -> str:
    saved_processor = checkpoint_path.parent / "processor"
    if not saved_processor.exists():
        raise RuntimeError(
            f"Checkpoint is missing the canonical saved processor directory: {saved_processor}"
        )
    return str(saved_processor)


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "fp16":
        return torch.float16
    return torch.bfloat16


def prepare_alpamayo_prompt_inputs_with_history(
    *,
    model,
    batch: dict,
    processor,
    history_registry: HistoryTokenRegistry,
    history_quantizer: HistoryTrajectoryQuantizer,
    question: str,
    history_token_count: int,
    device: torch.device,
) -> dict:
    user_text = build_stage1_question_user_text(question, history_token_count)
    message_batch: list[list[dict]] = []
    for image_path in batch["image_path"]:
        with Image.open(image_path) as raw_image:
            image = raw_image.convert("RGB")
            frame_tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).unsqueeze(0)
        message_batch.append(build_multimodal_messages(frames=frame_tensor, user_text=user_text))
    prompt_inputs = processor.apply_chat_template(
        message_batch,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt_inputs = to_device(prompt_inputs, device=device)
    return inject_history_inputs_embeds(
        model=model,
        prompt_inputs=prompt_inputs,
        history_registry=history_registry,
        history_quantizer=history_quantizer,
        history_xyz=batch["ego_history_xyz"].to(device=device, dtype=torch.float32),
        history_rot=batch["ego_history_rot"].to(device=device, dtype=torch.float32),
    )


def load_components(
    args: argparse.Namespace,
    task_spec: Stage1TaskSpec | None = None,
) -> tuple[
    dict,
    object,
    object,
    Stage1TokenRegistry,
    HistoryTokenRegistry,
    HistoryTrajectoryQuantizer,
    Any,
    torch.dtype,
]:
    task_spec = task_spec or CanonicalStage1Spec()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_args = resolve_checkpoint_args(checkpoint)
    if "stage1_metadata" not in checkpoint or not isinstance(checkpoint["stage1_metadata"], dict):
        raise RuntimeError("Checkpoint is missing canonical `stage1_metadata`.")
    task_spec.validate_checkpoint(checkpoint["stage1_metadata"])
    model_path = str(checkpoint_args["model_path"])
    processor_path = resolve_processor_path(checkpoint_path)
    model_dtype = resolve_dtype(str(checkpoint_args["dtype"]))
    processor_kwargs = build_processor_kwargs(args.image_min_pixels, args.image_max_pixels)
    processor = AutoProcessor.from_pretrained(
        processor_path, trust_remote_code=True, **processor_kwargs
    )
    add_prompt_special_tokens(processor.tokenizer)

    if "history_registry" not in checkpoint or not isinstance(checkpoint["history_registry"], dict):
        raise RuntimeError("Checkpoint is missing canonical `history_registry` metadata.")
    history_cfg = checkpoint["history_registry"]
    token_cfg = checkpoint["token_registry"]
    required_token_cfg_keys = ["n_bins", "token_prefix", "start_index"]
    missing_token_cfg_keys = [key for key in required_token_cfg_keys if key not in token_cfg]
    if missing_token_cfg_keys:
        raise RuntimeError(
            "Checkpoint token_registry is missing canonical fields:\n"
            + "\n".join(missing_token_cfg_keys)
        )
    required_history_cfg_keys = ["n_bins", "token_prefix", "start_index"]
    missing_history_cfg_keys = [key for key in required_history_cfg_keys if key not in history_cfg]
    if missing_history_cfg_keys:
        raise RuntimeError(
            "Checkpoint history_registry is missing canonical fields:\n"
            + "\n".join(missing_history_cfg_keys)
        )
    registry = Stage1TokenRegistry(
        n_bins=int(token_cfg["n_bins"]),
        token_prefix=str(token_cfg["token_prefix"]),
        start_index=int(token_cfg["start_index"]),
    )
    registry.add_to_tokenizer(processor.tokenizer)
    history_registry = HistoryTokenRegistry(
        n_bins=int(history_cfg["n_bins"]),
        token_prefix=str(history_cfg["token_prefix"]),
        start_index=int(history_cfg["start_index"]),
    )
    history_registry.add_to_tokenizer(processor.tokenizer)

    if "history_quantizer" not in checkpoint or not isinstance(
        checkpoint["history_quantizer"], dict
    ):
        raise RuntimeError("Checkpoint is missing canonical `history_quantizer` metadata.")
    history_quantizer_cfg = checkpoint["history_quantizer"]
    required_history_quantizer_keys = [
        "history_steps",
        "n_bins",
        "x_range",
        "y_range",
        "z_range",
        "yaw_range",
        "quantization_mode",
    ]
    missing_history_quantizer_keys = [
        key for key in required_history_quantizer_keys if key not in history_quantizer_cfg
    ]
    if missing_history_quantizer_keys:
        raise RuntimeError(
            "Checkpoint history_quantizer is missing canonical fields:\n"
            + "\n".join(missing_history_quantizer_keys)
        )
    history_quantizer = HistoryTrajectoryQuantizer(
        history_steps=int(history_quantizer_cfg["history_steps"]),
        n_bins=int(history_quantizer_cfg["n_bins"]),
        x_range=tuple(history_quantizer_cfg["x_range"]),
        y_range=tuple(history_quantizer_cfg["y_range"]),
        z_range=tuple(history_quantizer_cfg["z_range"]),
        yaw_range=(
            tuple(history_quantizer_cfg["yaw_range"])
            if history_quantizer_cfg["yaw_range"] is not None
            else None
        ),
        quantization_mode=str(history_quantizer_cfg["quantization_mode"]),
    )

    if "quantizer" not in checkpoint or not isinstance(checkpoint["quantizer"], dict):
        raise RuntimeError("Checkpoint is missing canonical `quantizer` metadata.")
    quantizer = task_spec.quantizer_from_checkpoint(
        checkpoint["quantizer"],
        stage1_metadata=checkpoint["stage1_metadata"],
    )

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        **build_model_load_kwargs(model_dtype),
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.config.use_cache = True
    model.eval()
    return (
        checkpoint,
        model,
        processor,
        registry,
        history_registry,
        history_quantizer,
        quantizer,
        model_dtype,
    )


def require_checkpoint_run_metadata(checkpoint: dict) -> dict:
    if "run_metadata" not in checkpoint:
        raise RuntimeError("Checkpoint is missing canonical `run_metadata`.")
    run_metadata = checkpoint["run_metadata"]
    if not isinstance(run_metadata, dict):
        raise RuntimeError("Checkpoint is missing canonical `run_metadata`.")
    return run_metadata


def normalize_image_format(image_path: str) -> str:
    suffix = Path(image_path).suffix.lower().lstrip(".")
    if suffix == "jpg":
        return "jpeg"
    if suffix in {"jpeg", "png", "webp", "avif"}:
        return suffix
    return "jpeg"


def ns_to_timestamp(ns: int) -> dict:
    return {
        "sec": int(ns // 1_000_000_000),
        "nsec": int(ns % 1_000_000_000),
    }


def require_extract_summary(test_jsonl: str) -> dict:
    summary_path = Path(test_jsonl).parent / "extract_summary.json"
    if not summary_path.exists():
        with Path(test_jsonl).open("r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        if not first_line:
            raise RuntimeError(f"Test dataset is empty: {test_jsonl}")
        first_record = json.loads(first_line)
        image_path = first_record.get("image_path")
        if not isinstance(image_path, str) or not image_path:
            raise RuntimeError(f"Test dataset is missing extract_summary.json: {summary_path}")
        candidate_path = Path(image_path).resolve().parent.parent / "extract_summary.json"
        if not candidate_path.exists():
            raise RuntimeError(f"Test dataset is missing extract_summary.json: {summary_path}")
        summary_path = candidate_path
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    if not isinstance(summary, dict):
        raise RuntimeError(f"Extract summary must be a JSON object: {summary_path}")
    required_keys = ["episode_dir", "record_hz"]
    missing_keys = [key for key in required_keys if key not in summary]
    if missing_keys:
        raise RuntimeError(
            "Extract summary is missing canonical fields:\n" + "\n".join(missing_keys)
        )
    return summary


def infer_episode_id(extract_summary: dict) -> str:
    episode_dir = extract_summary["episode_dir"]
    return Path(str(episode_dir)).name


def require_record_field(record: dict, key: str):
    if key not in record:
        raise RuntimeError(f"Evaluation record is missing canonical field `{key}`: {record!r}")
    return record[key]


def require_ego_pose(record: dict) -> dict:
    ego_pose = require_record_field(record, "ego_pose")
    if not isinstance(ego_pose, dict):
        raise RuntimeError(f"`ego_pose` must be an object: {record!r}")
    for key in ["x", "y", "yaw_deg"]:
        if key not in ego_pose:
            raise RuntimeError(f"`ego_pose` is missing `{key}`: {record!r}")
    return ego_pose


def elapsed_seconds(record: dict, extract_summary: dict) -> float:
    return record_time_ns(record, extract_summary) / 1_000_000_000.0


def record_time_ns(record: dict, extract_summary: dict) -> int:
    record_hz = float(extract_summary["record_hz"])
    if record_hz <= 0:
        raise RuntimeError(f"`record_hz` must be positive in extract_summary: {extract_summary!r}")
    source_frame_id = require_record_field(record, "source_frame_id")
    return int(round(float(source_frame_id) / record_hz * 1_000_000_000))


def yaw_deg_to_quaternion(yaw_deg: float) -> dict:
    yaw_rad = np.deg2rad(float(yaw_deg))
    return {
        "x": 0.0,
        "y": 0.0,
        "z": float(np.sin(yaw_rad / 2.0)),
        "w": float(np.cos(yaw_rad / 2.0)),
    }


def foxglove_frame_transforms_schema() -> bytes:
    return json.dumps(
        {
            "type": "object",
            "required": ["transforms"],
            "properties": {
                "transforms": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                            "timestamp",
                            "parent_frame_id",
                            "child_frame_id",
                            "translation",
                            "rotation",
                        ],
                        "properties": {
                            "timestamp": {
                                "type": "object",
                                "required": ["sec", "nsec"],
                                "properties": {
                                    "sec": {"type": "integer"},
                                    "nsec": {"type": "integer"},
                                },
                            },
                            "parent_frame_id": {"type": "string"},
                            "child_frame_id": {"type": "string"},
                            "translation": {
                                "type": "object",
                                "required": ["x", "y", "z"],
                                "properties": {
                                    "x": {"type": "number"},
                                    "y": {"type": "number"},
                                    "z": {"type": "number"},
                                },
                            },
                            "rotation": {
                                "type": "object",
                                "required": ["x", "y", "z", "w"],
                                "properties": {
                                    "x": {"type": "number"},
                                    "y": {"type": "number"},
                                    "z": {"type": "number"},
                                    "w": {"type": "number"},
                                },
                            },
                        },
                    },
                },
            },
        }
    ).encode("utf-8")


def ego_state_schema() -> bytes:
    return json.dumps(
        {
            "type": "object",
            "required": [
                "timestamp",
                "episode_id",
                "frame_id",
                "elapsed_seconds",
                "speed_mps",
                "route_completion_ratio",
                "distance_to_goal_m",
                "pose",
            ],
            "properties": {
                "timestamp": {
                    "type": "object",
                    "required": ["sec", "nsec"],
                    "properties": {
                        "sec": {"type": "integer"},
                        "nsec": {"type": "integer"},
                    },
                },
                "episode_id": {"type": "string"},
                "frame_id": {"type": "integer"},
                "elapsed_seconds": {"type": "number"},
                "speed_mps": {"type": "number"},
                "route_completion_ratio": {"type": ["number", "null"]},
                "distance_to_goal_m": {"type": ["number", "null"]},
                "pose": {
                    "type": "object",
                    "required": ["x", "y", "z", "yaw_deg", "pitch_deg", "roll_deg"],
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "z": {"type": "number"},
                        "yaw_deg": {"type": "number"},
                        "pitch_deg": {"type": "number"},
                        "roll_deg": {"type": "number"},
                    },
                },
            },
        }
    ).encode("utf-8")


def ego_planning_schema() -> bytes:
    return json.dumps(
        {
            "type": "object",
            "required": [
                "timestamp",
                "episode_id",
                "frame_id",
                "elapsed_seconds",
                "behavior",
                "planner_state",
                "traffic_light_state",
                "overtake_state",
                "target_lane_id",
                "min_ttc",
            ],
            "properties": {
                "timestamp": {
                    "type": "object",
                    "required": ["sec", "nsec"],
                    "properties": {
                        "sec": {"type": "integer"},
                        "nsec": {"type": "integer"},
                    },
                },
                "episode_id": {"type": "string"},
                "frame_id": {"type": "integer"},
                "elapsed_seconds": {"type": "number"},
                "behavior": {"type": ["string", "null"]},
                "planner_state": {"type": ["string", "null"]},
                "traffic_light_state": {"type": ["string", "null"]},
                "overtake_state": {"type": ["string", "null"]},
                "target_lane_id": {"type": ["string", "null"]},
                "min_ttc": {"type": ["number", "null"]},
            },
        }
    ).encode("utf-8")


def foxglove_compressed_image_schema() -> bytes:
    return json.dumps(
        {
            "type": "object",
            "required": ["timestamp", "frame_id", "data", "format"],
            "properties": {
                "timestamp": {
                    "type": "object",
                    "required": ["sec", "nsec"],
                    "properties": {
                        "sec": {"type": "integer"},
                        "nsec": {"type": "integer"},
                    },
                },
                "frame_id": {"type": "string"},
                "data": {"type": "string", "contentEncoding": "base64"},
                "format": {"type": "string", "enum": ["jpeg", "png", "webp", "avif"]},
            },
        }
    ).encode("utf-8")


def stage1_eval_sample_schema() -> bytes:
    return json.dumps(
        {
            "type": "object",
            "required": [
                "timestamp",
                "sample_id",
                "sample_index",
                "source_frame_id",
                "v0_mps",
                "dt",
                "command",
                "planner_state",
                "ego_pose",
                "gt_action",
                "pred_action",
                "gt_action_bins",
                "pred_action_bins",
                "gt_waypoints",
                "pred_waypoints",
                "metrics",
                "image_topic",
                "coordinate_frame",
            ],
            "properties": {
                "timestamp": {
                    "type": "object",
                    "required": ["sec", "nsec"],
                    "properties": {
                        "sec": {"type": "integer"},
                        "nsec": {"type": "integer"},
                    },
                },
                "sample_id": {"type": "string"},
                "sample_index": {"type": "integer"},
                "source_frame_id": {"type": "integer"},
                "v0_mps": {"type": "number"},
                "dt": {"type": "number"},
                "command": {"type": "string"},
                "planner_state": {"type": "string"},
                "image_topic": {"type": "string"},
                "coordinate_frame": {"type": "string"},
                "ego_pose": {
                    "type": "object",
                    "required": ["x", "y", "yaw_deg"],
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "yaw_deg": {"type": "number"},
                    },
                },
                "gt_action": {"type": "array", "items": {"type": "number"}},
                "pred_action": {"type": "array", "items": {"type": "number"}},
                "gt_action_bins": {"type": "array", "items": {"type": "integer"}},
                "pred_action_bins": {"type": "array", "items": {"type": "integer"}},
                "gt_waypoints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["x", "y"],
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                        },
                    },
                },
                "pred_waypoints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["x", "y"],
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                        },
                    },
                },
                "metrics": {
                    "type": "object",
                    "required": [
                        "teacher_forced_match_count",
                        "teacher_forced_token_accuracy",
                        "autoregressive_match_count",
                        "autoregressive_token_accuracy",
                        "action_mae_accel",
                        "action_mae_kappa",
                        "ade_m",
                        "fde_m",
                    ],
                    "properties": {
                        "teacher_forced_match_count": {"type": "integer"},
                        "teacher_forced_token_accuracy": {"type": "number"},
                        "autoregressive_match_count": {"type": "integer"},
                        "autoregressive_token_accuracy": {"type": "number"},
                        "action_mae_accel": {"type": "number"},
                        "action_mae_kappa": {"type": "number"},
                        "ade_m": {"type": "number"},
                        "fde_m": {"type": "number"},
                    },
                },
            },
        }
    ).encode("utf-8")


def stage1_eval_summary_schema() -> bytes:
    return json.dumps(
        {
            "type": "object",
            "required": [
                "checkpoint",
                "test_jsonl",
                "num_samples",
                "teacher_forced_loss",
                "teacher_forced_token_accuracy",
                "autoregressive_token_accuracy",
                "action_mae_accel",
                "action_mae_kappa",
                "ade_m",
                "fde_m",
                "peak_allocated_gib",
                "peak_reserved_gib",
                "unique_bins_used",
                "min_bin_used",
                "max_bin_used",
                "k",
                "dt",
                "generation_mode",
            ],
            "properties": {
                "checkpoint": {"type": "string"},
                "test_jsonl": {"type": "string"},
                "num_samples": {"type": "integer"},
                "teacher_forced_loss": {"type": "number"},
                "teacher_forced_token_accuracy": {"type": "number"},
                "autoregressive_token_accuracy": {"type": "number"},
                "action_mae_accel": {"type": "number"},
                "action_mae_kappa": {"type": "number"},
                "ade_m": {"type": "number"},
                "fde_m": {"type": "number"},
                "peak_allocated_gib": {"type": "number"},
                "peak_reserved_gib": {"type": "number"},
                "unique_bins_used": {"type": "integer"},
                "min_bin_used": {"type": "integer"},
                "max_bin_used": {"type": "integer"},
                "k": {"type": "integer"},
                "dt": {"type": "number"},
                "generation_mode": {"type": "string"},
            },
        }
    ).encode("utf-8")


def init_mcap_writer(
    output_mcap: str, episode_metadata: dict[str, str], camera_metadata: dict[str, str]
):
    output_path = Path(output_mcap)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stream = output_path.open("wb")
    writer = Writer(stream, compression=CompressionType.ZSTD, use_chunking=True)
    writer.start(profile="carla_alpamayo.route_loop", library="minipamayo_qwen35")
    writer.add_metadata("episode", episode_metadata)

    image_schema_id = writer.register_schema(
        name="foxglove.CompressedImage",
        encoding=SchemaEncoding.JSONSchema,
        data=foxglove_compressed_image_schema(),
    )
    tf_schema_id = writer.register_schema(
        name="foxglove.FrameTransforms",
        encoding=SchemaEncoding.JSONSchema,
        data=foxglove_frame_transforms_schema(),
    )
    ego_state_schema_id = writer.register_schema(
        name="carla_alpamayo.EgoState",
        encoding=SchemaEncoding.JSONSchema,
        data=ego_state_schema(),
    )
    ego_planning_schema_id = writer.register_schema(
        name="carla_alpamayo.EgoPlanning",
        encoding=SchemaEncoding.JSONSchema,
        data=ego_planning_schema(),
    )
    sample_schema_id = writer.register_schema(
        name="minipamayo.Stage1EvalSample",
        encoding=SchemaEncoding.JSONSchema,
        data=stage1_eval_sample_schema(),
    )
    summary_schema_id = writer.register_schema(
        name="minipamayo.Stage1EvalSummary",
        encoding=SchemaEncoding.JSONSchema,
        data=stage1_eval_summary_schema(),
    )

    image_channel_id = writer.register_channel(
        topic="/camera/front/compressed",
        message_encoding=MessageEncoding.JSON,
        schema_id=image_schema_id,
        metadata=camera_metadata,
    )
    tf_channel_id = writer.register_channel(
        topic="/tf",
        message_encoding=MessageEncoding.JSON,
        schema_id=tf_schema_id,
    )
    ego_state_channel_id = writer.register_channel(
        topic="/ego/state",
        message_encoding=MessageEncoding.JSON,
        schema_id=ego_state_schema_id,
    )
    ego_planning_channel_id = writer.register_channel(
        topic="/ego/planning",
        message_encoding=MessageEncoding.JSON,
        schema_id=ego_planning_schema_id,
    )
    sample_channel_id = writer.register_channel(
        topic="/eval/stage1/sample",
        message_encoding=MessageEncoding.JSON,
        schema_id=sample_schema_id,
    )
    summary_channel_id = writer.register_channel(
        topic="/eval/stage1/summary",
        message_encoding=MessageEncoding.JSON,
        schema_id=summary_schema_id,
    )
    return (
        stream,
        writer,
        image_channel_id,
        tf_channel_id,
        ego_state_channel_id,
        ego_planning_channel_id,
        sample_channel_id,
        summary_channel_id,
    )


def write_single_segment_index(
    output_mcap: str,
    episode_metadata: dict[str, str],
    start_elapsed_seconds: float,
    end_elapsed_seconds: float,
    frame_count: int,
) -> None:
    output_path = Path(output_mcap)
    index_payload = {
        "episode_id": episode_metadata["episode_id"],
        "route_name": episode_metadata["route_name"],
        "town": episode_metadata["town"],
        "weather": episode_metadata["weather"],
        "segment_seconds": 0.0,
        "segments": [
            {
                "segment_index": 0,
                "path": output_path.name,
                "start_elapsed_seconds": start_elapsed_seconds,
                "end_elapsed_seconds": end_elapsed_seconds,
                "frame_count": frame_count,
            }
        ],
    }
    with (output_path.parent / "index.json").open("w", encoding="utf-8") as f:
        json.dump(index_payload, f, indent=2, ensure_ascii=False)


def write_json_message(
    writer: Writer, channel_id: int, payload: dict, log_time_ns: int, sequence: int
) -> None:
    writer.add_message(
        channel_id=channel_id,
        log_time=log_time_ns,
        publish_time=log_time_ns,
        sequence=sequence,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    )


class ActionTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids: list[int]):
        self.allowed_token_ids = tuple(int(token_id) for token_id in allowed_token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        del input_ids
        allowed_token_ids = torch.tensor(
            self.allowed_token_ids,
            device=scores.device,
            dtype=torch.long,
        )
        masked_scores = torch.full_like(scores, torch.finfo(scores.dtype).min)
        masked_scores.index_copy_(1, allowed_token_ids, scores.index_select(1, allowed_token_ids))
        return masked_scores


@torch.no_grad()
def greedy_generate_action_tokens(
    model,
    prompt_inputs: dict,
    registry: Stage1TokenRegistry,
    action_len: int,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.do_sample = False
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = int(action_len)
    generation_config.min_new_tokens = int(action_len)
    generation_config.return_dict_in_generate = True
    pad_token_id = generation_config.pad_token_id
    if pad_token_id is None:
        pad_token_id = getattr(model.config, "pad_token_id", None)
    if pad_token_id is None:
        eos_token_id = generation_config.eos_token_id
        if isinstance(eos_token_id, list):
            pad_token_id = int(eos_token_id[0]) if eos_token_id else None
        elif eos_token_id is not None:
            pad_token_id = int(eos_token_id)
    if pad_token_id is None:
        raise RuntimeError("Stage 1 greedy generation requires `pad_token_id` or `eos_token_id`.")
    generation_config.pad_token_id = int(pad_token_id)
    logits_processor = LogitsProcessorList([ActionTokenLogitsProcessor(list(registry.token_ids))])
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    with torch.autocast("cuda", dtype=model_dtype):
        outputs = model.generate(
            **prompt_inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
        )
    generated_ids = outputs.sequences[:, prompt_len:]
    if generated_ids.shape[1] != int(action_len):
        raise RuntimeError(
            "Stage 1 greedy generation returned an unexpected number of action tokens.\n"
            f"expected_action_len={int(action_len)}\n"
            f"generated_shape={tuple(generated_ids.shape)!r}"
        )
    return generated_ids


def main(task_spec: Stage1TaskSpec | None = None) -> None:
    task_spec = task_spec or CanonicalStage1Spec()
    args = parse_args()
    require_completed_training_run(
        args.checkpoint,
        checkpoint_label="Stage 1A checkpoint",
        required_summary_keys=["completed_epochs", "best_epoch", "stop_reason"],
        allowed_stop_reasons={"max_epochs", "early_stopping"},
    )
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type != "cuda":
        raise RuntimeError("Stage 1 evaluation currently expects CUDA.")
    enforce_runtime_prerequisites(git_cwd=Path(__file__).resolve().parent)
    git_metadata = collect_git_metadata(Path(__file__).resolve().parent)
    gpu_info = collect_gpu_info(device)

    (
        checkpoint,
        model,
        processor,
        registry,
        history_registry,
        history_quantizer,
        quantizer,
        model_dtype,
    ) = load_components(args, task_spec)
    checkpoint_run_metadata = require_checkpoint_run_metadata(checkpoint)
    model.to(device)
    processor_settings = collect_processor_settings(
        processor,
        requested_min_pixels=args.image_min_pixels or None,
        requested_max_pixels=args.image_max_pixels or None,
    )

    test_jsonl = args.test_jsonl
    dataset = Stage1JsonlDataset(test_jsonl, max_samples=args.max_samples)
    dataset_fingerprint = collect_dataset_view_fingerprint(dataset)
    extract_summary = require_extract_summary(test_jsonl)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=stage1_collate,
    )

    if "stage1_metadata" not in checkpoint:
        raise RuntimeError("Checkpoint is missing canonical `stage1_metadata`.")
    stage1_metadata = checkpoint["stage1_metadata"]
    if not isinstance(stage1_metadata, dict):
        raise RuntimeError("Checkpoint is missing canonical `stage1_metadata`.")
    task_spec.validate_checkpoint(stage1_metadata)
    required_stage1_keys = [
        "question",
        "target_dim",
        "full_action_dim",
        "k",
        "dt",
        "history_steps",
        "history_token_count",
        "action_representation",
        "rollout_accel_source",
    ]
    missing_stage1_keys = [key for key in required_stage1_keys if key not in stage1_metadata]
    if missing_stage1_keys:
        raise RuntimeError(
            "Checkpoint is missing canonical Stage 1 metadata:\n" + "\n".join(missing_stage1_keys)
        )
    question = str(stage1_metadata["question"])
    history_token_count = int(stage1_metadata["history_token_count"])
    target_dim = int(stage1_metadata["target_dim"])
    full_action_dim = int(stage1_metadata["full_action_dim"])
    k_steps = int(stage1_metadata["k"])
    dt = float(stage1_metadata["dt"])
    episode_id = infer_episode_id(extract_summary)
    episode_metadata = {
        "episode_id": episode_id,
        "route_name": "stage1_eval",
        "town": "",
        "weather": "",
        "compression": "zstd_chunked",
    }

    torch.cuda.reset_peak_memory_stats(device)

    mcap_stream = None
    mcap_writer = None
    image_channel_id = None
    tf_channel_id = None
    ego_state_channel_id = None
    ego_planning_channel_id = None
    sample_channel_id = None
    summary_channel_id = None
    if args.output_mcap:
        with Image.open(dataset[0]["image_path"]) as first_image:
            width, height = first_image.size
        camera_metadata = {
            "frame_id": "ego/front_camera",
            "camera_width": str(width),
            "camera_height": str(height),
            "jpeg_quality": "0",
        }
        (
            mcap_stream,
            mcap_writer,
            image_channel_id,
            tf_channel_id,
            ego_state_channel_id,
            ego_planning_channel_id,
            sample_channel_id,
            summary_channel_id,
        ) = init_mcap_writer(args.output_mcap, episode_metadata, camera_metadata)

    print(
        json.dumps(
            {
                "event": "stage1_eval_setup",
                "checkpoint": args.checkpoint,
                "test_jsonl": test_jsonl,
                "num_samples": len(dataset),
                "batch_size": args.batch_size,
                "target_dim": target_dim,
                "full_action_dim": full_action_dim,
                "k": k_steps,
                "dt": dt,
                "history_steps": int(stage1_metadata["history_steps"]),
                "history_token_count": int(stage1_metadata["history_token_count"]),
                "dtype": "bf16" if model_dtype == torch.bfloat16 else "fp16",
                "image_min_pixels": args.image_min_pixels or None,
                "image_max_pixels": args.image_max_pixels or None,
                "processor_settings": processor_settings,
                "action_representation": stage1_metadata["action_representation"],
                "rollout_accel_source": stage1_metadata["rollout_accel_source"],
            },
            ensure_ascii=False,
        )
    )

    tf_loss_total = 0.0
    tf_batches = 0
    tf_correct = 0
    tf_total_tokens = 0
    ar_correct = 0
    ar_total_tokens = 0

    pred_actions_list: list[torch.Tensor] = []
    gt_actions_list: list[torch.Tensor] = []
    pred_waypoints_list: list[torch.Tensor] = []
    gt_waypoints_list: list[torch.Tensor] = []
    generated_bins: list[int] = []
    record_cursor = 0
    last_log_time_ns = 0
    first_elapsed_s: float | None = None
    last_elapsed_s: float | None = None

    try:
        with torch.no_grad():
            sample_index = 0
            for batch in loader:
                prompt_inputs = prepare_alpamayo_prompt_inputs_with_history(
                    model=model,
                    batch=batch,
                    processor=processor,
                    history_registry=history_registry,
                    history_quantizer=history_quantizer,
                    question=question,
                    history_token_count=history_token_count,
                    device=device,
                )
                full_inputs, labels = build_full_inputs_from_prompt_inputs(
                    model=model,
                    prompt_inputs=prompt_inputs,
                    batch=batch,
                    registry=registry,
                    quantizer=quantizer,
                    task_spec=task_spec,
                    device=device,
                )
                with torch.autocast("cuda", dtype=model_dtype):
                    outputs = model(**model_forward_inputs(full_inputs), labels=labels)

                correct, total = compute_token_accuracy(outputs.logits, labels)
                tf_loss_total += float(outputs.loss.detach().cpu())
                tf_correct += correct
                tf_total_tokens += total
                tf_batches += 1

                shifted_preds = outputs.logits[:, :-1, :].argmax(dim=-1)
                shifted_labels = labels[:, 1:]
                shifted_mask = shifted_labels != -100
                tf_per_sample_total = shifted_mask.sum(dim=1)
                tf_per_sample_correct = ((shifted_preds == shifted_labels) & shifted_mask).sum(
                    dim=1
                )

                generated_token_ids = greedy_generate_action_tokens(
                    model=model,
                    prompt_inputs=prompt_inputs,
                    registry=registry,
                    action_len=target_dim,
                    model_dtype=model_dtype,
                )

                gt_token_rows = task_spec.encode_target_token_rows_from_batch(
                    batch,
                    registry,
                    quantizer,
                )
                gt_token_ids = torch.tensor(gt_token_rows, device=device, dtype=torch.long)
                ar_matches = generated_token_ids == gt_token_ids
                ar_correct += int(ar_matches.sum().item())
                ar_total_tokens += int(gt_token_ids.numel())

                for row_idx in range(generated_token_ids.shape[0]):
                    record = dataset.records[record_cursor]
                    sample_id = str(require_record_field(record, "sample_id"))
                    sample_record_index = int(require_record_field(record, "sample_index"))
                    source_frame_id = int(require_record_field(record, "source_frame_id"))
                    command = str(require_record_field(record, "command"))
                    planner_state = str(require_record_field(record, "planner_state"))
                    ego_pose = require_ego_pose(record)
                    pred_token_ids = generated_token_ids[row_idx].detach().cpu().tolist()
                    gt_row = gt_token_ids[row_idx].detach().cpu().tolist()
                    pred_bins = registry.decode_token_ids_to_bin_ids(pred_token_ids)
                    gt_bins = registry.decode_token_ids_to_bin_ids(gt_row)
                    pred_target = registry.decode_target_token_ids(pred_token_ids, quantizer)
                    pred_target_tensor = torch.tensor(pred_target, dtype=torch.float32)
                    gt_action_tensor = batch["action"][row_idx].detach().cpu()
                    pred_action_tensor = task_spec.full_action_from_target_tensor(
                        pred_target_tensor,
                        gt_action_tensor=gt_action_tensor,
                    )
                    gt_waypoint_tensor = batch["gt_waypoints"][row_idx].detach().cpu()
                    v0_tensor = batch["v0"][row_idx].detach().cpu()

                    pred_actions_list.append(pred_action_tensor)
                    gt_actions_list.append(gt_action_tensor)
                    gt_waypoints_list.append(gt_waypoint_tensor)
                    generated_bins.extend(pred_bins)

                    pred_waypoint_tensor = (
                        rollout_waypoints_from_action_tensor(
                            action=pred_action_tensor.view(1, k_steps, 2),
                            history_xyz=batch["ego_history_xyz"][row_idx].detach().cpu(),
                            history_rot=batch["ego_history_rot"][row_idx].detach().cpu(),
                            dt=dt,
                        )
                        .squeeze(0)
                    )
                    pred_waypoints_list.append(pred_waypoint_tensor)
                    displacement = torch.norm(pred_waypoint_tensor - gt_waypoint_tensor, dim=1)
                    tf_match_count = int(tf_per_sample_correct[row_idx].item())
                    tf_token_count = int(tf_per_sample_total[row_idx].item())
                    ar_match_count = int(ar_matches[row_idx].sum().item())

                    if sample_index < args.show_samples:
                        print(
                            json.dumps(
                                {
                                    "event": "sample",
                                    "sample_index": sample_index,
                                    "sample_id": sample_id,
                                    "match_tokens": ar_match_count,
                                    "target_dim": target_dim,
                                    "ade_m": float(displacement.mean().item()),
                                    "fde_m": float(displacement[-1].item()),
                                },
                                ensure_ascii=False,
                            )
                        )

                    if mcap_writer is not None:
                        log_time_ns = record_time_ns(record, extract_summary)
                        last_log_time_ns = max(last_log_time_ns, log_time_ns)
                        sample_elapsed_s = elapsed_seconds(record, extract_summary)
                        if first_elapsed_s is None:
                            first_elapsed_s = sample_elapsed_s
                        last_elapsed_s = sample_elapsed_s

                        image_payload = {
                            "timestamp": ns_to_timestamp(log_time_ns),
                            "frame_id": "ego/front_camera",
                            "data": base64.b64encode(
                                Path(batch["image_path"][row_idx]).read_bytes()
                            ).decode("ascii"),
                            "format": normalize_image_format(batch["image_path"][row_idx]),
                        }
                        write_json_message(
                            writer=mcap_writer,
                            channel_id=image_channel_id,
                            payload=image_payload,
                            log_time_ns=log_time_ns,
                            sequence=sample_index,
                        )

                        state_payload = {
                            "timestamp": ns_to_timestamp(log_time_ns),
                            "episode_id": episode_id,
                            "frame_id": source_frame_id,
                            "elapsed_seconds": sample_elapsed_s,
                            "speed_mps": float(v0_tensor.item()),
                            "route_completion_ratio": None,
                            "distance_to_goal_m": None,
                            "pose": {
                                "x": float(ego_pose["x"]),
                                "y": float(ego_pose["y"]),
                                "z": 0.0,
                                "yaw_deg": float(ego_pose["yaw_deg"]),
                                "pitch_deg": 0.0,
                                "roll_deg": 0.0,
                            },
                        }
                        write_json_message(
                            writer=mcap_writer,
                            channel_id=ego_state_channel_id,
                            payload=state_payload,
                            log_time_ns=log_time_ns,
                            sequence=sample_index,
                        )

                        planning_payload = {
                            "timestamp": ns_to_timestamp(log_time_ns),
                            "episode_id": episode_id,
                            "frame_id": source_frame_id,
                            "elapsed_seconds": sample_elapsed_s,
                            "behavior": command,
                            "planner_state": planner_state,
                            "traffic_light_state": None,
                            "overtake_state": None,
                            "target_lane_id": None,
                            "min_ttc": None,
                        }
                        write_json_message(
                            writer=mcap_writer,
                            channel_id=ego_planning_channel_id,
                            payload=planning_payload,
                            log_time_ns=log_time_ns,
                            sequence=sample_index,
                        )

                        tf_payload = {
                            "transforms": [
                                {
                                    "timestamp": ns_to_timestamp(log_time_ns),
                                    "parent_frame_id": "map",
                                    "child_frame_id": "ego/base_link",
                                    "translation": {
                                        "x": float(ego_pose["x"]),
                                        "y": float(ego_pose["y"]),
                                        "z": 0.0,
                                    },
                                    "rotation": yaw_deg_to_quaternion(float(ego_pose["yaw_deg"])),
                                },
                                {
                                    "timestamp": ns_to_timestamp(log_time_ns),
                                    "parent_frame_id": "ego/base_link",
                                    "child_frame_id": "ego/front_camera",
                                    "translation": {"x": 1.5, "y": 0.0, "z": 2.4},
                                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                                },
                            ]
                        }
                        write_json_message(
                            writer=mcap_writer,
                            channel_id=tf_channel_id,
                            payload=tf_payload,
                            log_time_ns=log_time_ns,
                            sequence=sample_index,
                        )

                        sample_payload = {
                            "timestamp": ns_to_timestamp(log_time_ns),
                            "sample_id": sample_id,
                            "sample_index": sample_record_index,
                            "source_frame_id": source_frame_id,
                            "v0_mps": float(v0_tensor.item()),
                            "dt": dt,
                            "command": command,
                            "planner_state": planner_state,
                            "image_topic": "/camera/front/compressed",
                            "coordinate_frame": "ego_xy_meters",
                            "action_representation": stage1_metadata["action_representation"],
                            "ego_pose": {
                                "x": float(ego_pose["x"]),
                                "y": float(ego_pose["y"]),
                                "yaw_deg": float(ego_pose["yaw_deg"]),
                            },
                            "gt_action": [float(x) for x in gt_action_tensor.tolist()],
                            "pred_action": [float(x) for x in pred_action_tensor.tolist()],
                            "gt_action_bins": [int(x) for x in gt_bins],
                            "pred_action_bins": [int(x) for x in pred_bins],
                            "gt_waypoints": [
                                {"x": float(point[0]), "y": float(point[1])}
                                for point in gt_waypoint_tensor.tolist()
                            ],
                            "pred_waypoints": [
                                {"x": float(point[0]), "y": float(point[1])}
                                for point in pred_waypoint_tensor.tolist()
                            ],
                            "metrics": {
                                "teacher_forced_match_count": tf_match_count,
                                "teacher_forced_token_accuracy": tf_match_count
                                / max(tf_token_count, 1),
                                "autoregressive_match_count": ar_match_count,
                                "autoregressive_token_accuracy": ar_match_count
                                / max(target_dim, 1),
                                "action_mae_accel": float(
                                    (
                                        pred_action_tensor.view(k_steps, 2)[:, 0]
                                        - gt_action_tensor.view(k_steps, 2)[:, 0]
                                    )
                                    .abs()
                                    .mean()
                                    .item()
                                ),
                                "action_mae_kappa": float(
                                    (
                                        pred_action_tensor.view(k_steps, 2)[:, 1]
                                        - gt_action_tensor.view(k_steps, 2)[:, 1]
                                    )
                                    .abs()
                                    .mean()
                                    .item()
                                ),
                                "ade_m": float(displacement.mean().item()),
                                "fde_m": float(displacement[-1].item()),
                            },
                        }
                        write_json_message(
                            writer=mcap_writer,
                            channel_id=sample_channel_id,
                            payload=sample_payload,
                            log_time_ns=log_time_ns,
                            sequence=sample_index,
                        )

                    sample_index += 1
                    record_cursor += 1

        if not pred_actions_list:
            raise RuntimeError("No samples were evaluated.")

        pred_actions = torch.stack(pred_actions_list)
        gt_actions = torch.stack(gt_actions_list)
        pred_waypoints = torch.stack(pred_waypoints_list)
        gt_waypoints = torch.stack(gt_waypoints_list)

        pred_kv = pred_actions.reshape(-1, k_steps, 2)
        gt_kv = gt_actions.reshape(-1, k_steps, 2)
        displacement_errors = torch.norm(pred_waypoints - gt_waypoints, dim=2)

        summary = {
            "config_json": args.config_json,
            "config_payload": args.config_payload,
            "config_args": args.config_args,
            "run_args": vars(args),
            "checkpoint": args.checkpoint,
            "test_jsonl": test_jsonl,
            "num_samples": len(pred_actions_list),
            "teacher_forced_loss": tf_loss_total / max(tf_batches, 1),
            "teacher_forced_token_accuracy": tf_correct / max(tf_total_tokens, 1),
            "autoregressive_token_accuracy": ar_correct / max(ar_total_tokens, 1),
            "action_mae_accel": float((pred_kv[:, :, 0] - gt_kv[:, :, 0]).abs().mean().item()),
            "action_mae_kappa": float((pred_kv[:, :, 1] - gt_kv[:, :, 1]).abs().mean().item()),
            "ade_m": float(displacement_errors.mean().item()),
            "fde_m": float(displacement_errors[:, -1].mean().item()),
            "peak_allocated_gib": format_gib(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_gib": format_gib(torch.cuda.max_memory_reserved(device)),
            "unique_bins_used": len(set(generated_bins)),
            "min_bin_used": min(generated_bins),
            "max_bin_used": max(generated_bins),
            "action_representation": stage1_metadata["action_representation"],
            "rollout_accel_source": stage1_metadata["rollout_accel_source"],
            "target_dim": target_dim,
            "full_action_dim": full_action_dim,
            "k": k_steps,
            "dt": dt,
            "generation_mode": "greedy_action_vocab_only",
            "run_metadata": {
                "git": git_metadata,
                "gpu": gpu_info,
                "datasets": {
                    "test": dataset_fingerprint,
                },
                "processor": processor_settings,
                "checkpoint_run_metadata": checkpoint_run_metadata,
            },
        }

        print(json.dumps({"event": "stage1_eval_summary", **summary}, ensure_ascii=False, indent=2))

        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

        if mcap_writer is not None:
            if first_elapsed_s is None or last_elapsed_s is None:
                raise RuntimeError(
                    "MCAP output requested, but elapsed time metadata was not recorded."
                )
            write_json_message(
                writer=mcap_writer,
                channel_id=summary_channel_id,
                payload=summary,
                log_time_ns=last_log_time_ns + 1,
                sequence=len(pred_actions_list),
            )
            write_single_segment_index(
                output_mcap=args.output_mcap,
                episode_metadata=episode_metadata,
                start_elapsed_seconds=first_elapsed_s,
                end_elapsed_seconds=last_elapsed_s,
                frame_count=len(pred_actions_list),
            )
    finally:
        if mcap_writer is not None:
            mcap_writer.finish()
        if mcap_stream is not None:
            mcap_stream.close()


if __name__ == "__main__":
    main()
