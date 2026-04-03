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
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from PIL import Image

from ...utils.image_budget import (
    validate_canonical_image_budget,
)
from ...utils.eval_reporting import (
    EvalReporter,
    add_eval_reporting_args,
    reporting_path_keys,
    validate_eval_reporting_args,
)
from ...utils.preflight import enforce_runtime_prerequisites
from ...utils.run_metadata import (
    collect_dataset_view_fingerprint,
    collect_git_metadata,
    collect_gpu_info,
    collect_processor_settings,
)
from ..stage1_train_runtime import format_gib
from ..stage1a_runtime import (
    load_stage1a_runtime,
    run_stage1a_rollout_batch,
    run_stage1a_teacher_forced_batch,
)
from ..checkpoint_completion import require_completed_training_run
from ...contract.task_spec import CanonicalStage1Spec, Stage1TaskSpec
from ..dataset import Stage1JsonlDataset, stage1_collate
from .cli import parse_config_json_only_args
from .eval_artifacts import (
    elapsed_seconds,
    infer_episode_id,
    init_mcap_writer,
    normalize_image_format,
    ns_to_timestamp,
    record_time_ns,
    require_ego_pose,
    require_extract_summary,
    write_json_message,
    write_single_segment_index,
    yaw_deg_to_quaternion,
)
from .metrics import require_record_field

CONFIG_PATH_KEYS = {
    "checkpoint",
    "test_jsonl",
    "output_json",
    "output_mcap",
} | reporting_path_keys(include_per_sample_jsonl=True)


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
    parser.add_argument("--image-min-pixels", type=int, default=0)
    parser.add_argument("--image-max-pixels", type=int, default=0)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--output-mcap", type=str, default="")
    add_eval_reporting_args(parser, include_per_sample_jsonl=True)
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parse_config_json_only_args(
        parser,
        path_keys=CONFIG_PATH_KEYS,
        error_message="Stage 1 evaluation accepts only --config-json. Put all settings in the JSON file.",
    )
    if not args.checkpoint:
        raise RuntimeError("`checkpoint` must be defined in the config JSON.")
    if not args.test_jsonl:
        raise RuntimeError("`test_jsonl` must be defined in the config JSON.")
    validate_canonical_image_budget(args.image_min_pixels, args.image_max_pixels)
    validate_eval_reporting_args(args, require_per_sample_jsonl=True)
    return args


def require_checkpoint_run_metadata(checkpoint: dict) -> dict:
    if "run_metadata" not in checkpoint:
        raise RuntimeError("Checkpoint is missing canonical `run_metadata`.")
    run_metadata = checkpoint["run_metadata"]
    if not isinstance(run_metadata, dict):
        raise RuntimeError("Checkpoint is missing canonical `run_metadata`.")
    return run_metadata


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

    runtime = load_stage1a_runtime(args, task_spec, device=device)
    checkpoint_run_metadata = require_checkpoint_run_metadata(runtime.checkpoint)
    processor_settings = collect_processor_settings(
        runtime.processor,
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

    stage1_metadata = runtime.stage1_metadata
    target_dim = runtime.target_dim
    full_action_dim = runtime.full_action_dim
    k_steps = runtime.k_steps
    dt = runtime.dt
    episode_id = infer_episode_id(extract_summary)
    episode_metadata = {
        "episode_id": episode_id,
        "route_name": "stage1_eval",
        "town": "",
        "weather": "",
        "compression": "zstd_chunked",
    }
    reporter = EvalReporter.from_args(
        args=args,
        stage="stage1a_eval",
        total_samples=len(dataset),
        checkpoint=args.checkpoint,
        dataset_path=test_jsonl,
        extra_wandb_config={
            "entrypoint": "stage1.vlm_ce.eval",
            "batch_size": int(args.batch_size),
            "max_samples": int(args.max_samples),
            "show_samples": int(args.show_samples),
        },
    )

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

    reporter.emit_setup(
        "stage1_eval_setup",
        {
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
            "dtype": "bf16" if runtime.model_dtype == torch.bfloat16 else "fp16",
            "image_min_pixels": args.image_min_pixels or None,
            "image_max_pixels": args.image_max_pixels or None,
            "processor_settings": processor_settings,
            "action_representation": stage1_metadata["action_representation"],
            "rollout_accel_source": stage1_metadata["rollout_accel_source"],
        },
    )

    tf_loss_total = 0.0
    tf_batches = 0
    tf_correct = 0
    tf_total_tokens = 0
    ar_correct = 0
    ar_total_tokens = 0
    action_mae_accel_total = 0.0
    action_mae_kappa_total = 0.0
    ade_total = 0.0
    fde_total = 0.0

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
        try:
            with torch.no_grad():
                sample_index = 0
                for batch in loader:
                    teacher_forced = run_stage1a_teacher_forced_batch(
                        runtime,
                        batch,
                        device=device,
                        prompt_mode="alpamayo_message_rollout",
                    )
                    rollout = run_stage1a_rollout_batch(runtime, batch, device=device)
                    outputs = teacher_forced["outputs"]
                    labels = teacher_forced["labels"]
                    correct = teacher_forced["correct"]
                    total = teacher_forced["total"]
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

                    generated_token_ids = rollout["generated_token_ids"]
                    gt_token_ids = rollout["gt_token_ids"]
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
                        decoded_row = rollout["decoded_rows"][row_idx]
                        pred_token_ids = decoded_row["pred_token_ids"]
                        gt_row = gt_token_ids[row_idx].detach().cpu().tolist()
                        pred_bins = decoded_row["pred_bin_ids"]
                        gt_bins = runtime.registry.decode_token_ids_to_bin_ids(gt_row)
                        gt_action_tensor = decoded_row["gt_action_tensor"]
                        pred_action_tensor = decoded_row["pred_action_tensor"]
                        gt_waypoint_tensor = decoded_row["gt_waypoints"]
                        v0_tensor = batch["v0"][row_idx].detach().cpu()

                        pred_actions_list.append(pred_action_tensor)
                        gt_actions_list.append(gt_action_tensor)
                        gt_waypoints_list.append(gt_waypoint_tensor)
                        generated_bins.extend(pred_bins)

                        pred_waypoint_tensor = decoded_row["pred_waypoints"]
                        pred_waypoints_list.append(pred_waypoint_tensor)
                        displacement = torch.norm(pred_waypoint_tensor - gt_waypoint_tensor, dim=1)
                        tf_match_count = int(tf_per_sample_correct[row_idx].item())
                        tf_token_count = int(tf_per_sample_total[row_idx].item())
                        ar_match_count = int(ar_matches[row_idx].sum().item())
                        sample_action_mae_accel = float(
                            (
                                pred_action_tensor.view(k_steps, 2)[:, 0]
                                - gt_action_tensor.view(k_steps, 2)[:, 0]
                            )
                            .abs()
                            .mean()
                            .item()
                        )
                        sample_action_mae_kappa = float(
                            (
                                pred_action_tensor.view(k_steps, 2)[:, 1]
                                - gt_action_tensor.view(k_steps, 2)[:, 1]
                            )
                            .abs()
                            .mean()
                            .item()
                        )
                        sample_ade = float(displacement.mean().item())
                        sample_fde = float(displacement[-1].item())
                        action_mae_accel_total += sample_action_mae_accel
                        action_mae_kappa_total += sample_action_mae_kappa
                        ade_total += sample_ade
                        fde_total += sample_fde

                        reporter.emit_sample(
                            {
                                "event": "sample",
                                "sample_index": sample_index,
                                "record_sample_index": sample_record_index,
                                "sample_id": sample_id,
                                "source_frame_id": source_frame_id,
                                "image_path": str(batch["image_path"][row_idx]),
                                "match_tokens": ar_match_count,
                                "target_dim": target_dim,
                                "dt": dt,
                                "command": command,
                                "planner_state": planner_state,
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
                                    [float(point[0]), float(point[1])]
                                    for point in gt_waypoint_tensor.tolist()
                                ],
                                "pred_waypoints": [
                                    [float(point[0]), float(point[1])]
                                    for point in pred_waypoint_tensor.tolist()
                                ],
                                "ade_m": sample_ade,
                                "fde_m": sample_fde,
                                "metrics": {
                                    "teacher_forced_match_count": tf_match_count,
                                    "teacher_forced_token_accuracy": tf_match_count
                                    / max(tf_token_count, 1),
                                    "autoregressive_match_count": ar_match_count,
                                    "autoregressive_token_accuracy": ar_match_count
                                    / max(target_dim, 1),
                                    "action_mae_accel": sample_action_mae_accel,
                                    "action_mae_kappa": sample_action_mae_kappa,
                                    "ade_m": sample_ade,
                                    "fde_m": sample_fde,
                                },
                            },
                            print_to_stdout=sample_index < args.show_samples,
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
                                    "action_mae_accel": sample_action_mae_accel,
                                    "action_mae_kappa": sample_action_mae_kappa,
                                    "ade_m": sample_ade,
                                    "fde_m": sample_fde,
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
                        reporter.emit_progress(
                            processed_samples=sample_index,
                            running_metrics={
                                "teacher_forced_loss": tf_loss_total / max(tf_batches, 1),
                                "teacher_forced_token_accuracy": tf_correct / max(tf_total_tokens, 1),
                                "autoregressive_token_accuracy": ar_correct / max(ar_total_tokens, 1),
                                "action_mae_accel": action_mae_accel_total / max(sample_index, 1),
                                "action_mae_kappa": action_mae_kappa_total / max(sample_index, 1),
                                "ade_m": ade_total / max(sample_index, 1),
                                "fde_m": fde_total / max(sample_index, 1),
                            },
                        )

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
            reporter.emit_summary("stage1_eval_summary", summary)
        except Exception as exc:
            reporter.emit_failure("stage1_eval_failure", exc)
            raise
    finally:
        reporter.close()
        if mcap_writer is not None:
            mcap_writer.finish()
        if mcap_stream is not None:
            mcap_stream.close()


if __name__ == "__main__":
    main()
