"""Eval-only artifact helpers for canonical Stage 1A evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from mcap.well_known import MessageEncoding, SchemaEncoding
from mcap.writer import CompressionType, Writer

from .metrics import require_record_field


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
