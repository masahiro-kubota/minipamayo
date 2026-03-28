"""Convert CARLA telemetry MCAP episodes into Stage 1 JSONL samples."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from mcap.reader import make_reader

from ...utils.json_config import load_json_payload, normalize_arg_config, resolve_path_base
from ...utils.preflight import require_clean_git_worktree
from ...utils.dynamics import to_ego_centric
from .canonical_action import canonical_action_tensor_from_tensors

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATASETS_ROOT = PROJECT_ROOT / "datasets"


@dataclass
class FrameRecord:
    frame_id: int
    image_b64: str
    image_format: str
    speed_mps: float
    x: float
    y: float
    yaw_deg: float
    command: str
    planner_state: str


@dataclass
class ExtractionJob:
    output_dir: Path
    episode_dir: Path | None = None
    summary_path: Path | None = None
    mcap_paths: list[Path] | None = None


def build_config_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract Stage 1 samples from MCAP telemetry.")
    parser.add_argument("--config-json", type=str, default="")
    return parser


def build_extract_settings_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--k", type=int, default=6, help="Number of predicted action steps.")
    parser.add_argument("--dt", type=float, default=0.5, help="Training step size in seconds.")
    parser.add_argument(
        "--history-steps",
        type=int,
        default=16,
        help="Number of past ego-motion frames, inclusive of the current frame.",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=0,
        help="Stride in source frames between samples. 0 = use future stride.",
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--log-every",
        type=int,
        default=200,
        help="Emit extraction progress every N samples to stderr. 0 disables periodic progress logs.",
    )
    return parser


def _resolve_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _resolve_config_base_dir(config_path: Path, payload) -> Path:
    return resolve_path_base(
        config_path,
        payload,
        default_base="project_root",
        base_dirs={
            "project_root": PROJECT_ROOT,
            "datasets_root": DATASETS_ROOT,
            "config_dir": config_path.parent,
        },
    )


def _read_record_hz(summary_path: Path) -> float:
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    return float(summary["record_hz"])


def _resolve_mcap_paths(episode_dir: Path) -> list[Path]:
    legacy_mcap_path = episode_dir / "telemetry.mcap"
    if legacy_mcap_path.exists():
        return [legacy_mcap_path]

    telemetry_dir = episode_dir / "telemetry"
    if not telemetry_dir.exists():
        raise RuntimeError(
            "Could not find telemetry input. Expected either "
            f"{legacy_mcap_path} or a telemetry/ directory under {episode_dir}."
        )

    index_path = telemetry_dir / "index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        segments = index.get("segments", [])
        if not segments:
            raise RuntimeError(f"Telemetry index exists but contains no segments: {index_path}")

        mcap_paths = [telemetry_dir / str(segment["path"]) for segment in segments]
        missing_paths = [str(path) for path in mcap_paths if not path.exists()]
        if missing_paths:
            raise RuntimeError(
                "Telemetry index referenced missing segment files:\n" + "\n".join(missing_paths)
            )
        return mcap_paths

    mcap_paths = sorted(telemetry_dir.glob("segment_*.mcap"))
    if mcap_paths:
        return mcap_paths

    raise RuntimeError(
        "Could not find any MCAP telemetry files. Expected telemetry.mcap, "
        "telemetry/index.json, or telemetry/segment_*.mcap."
    )


def _infer_summary_path(mcap_paths: list[Path]) -> Path:
    if not mcap_paths:
        raise RuntimeError("Config job has no MCAP paths.")
    telemetry_dirs = {path.resolve().parent for path in mcap_paths}
    if len(telemetry_dirs) != 1:
        raise RuntimeError("Config jobs with explicit MCAP paths must share the same telemetry directory.")
    telemetry_dir = telemetry_dirs.pop()
    summary_path = telemetry_dir.parent / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"Could not infer summary.json for telemetry directory: {telemetry_dir}")
    return summary_path


def _normalize_job(raw_job: dict, base_dir: Path) -> ExtractionJob:
    if not isinstance(raw_job, dict):
        raise RuntimeError("Each config job must be a JSON object.")

    output_dir_value = raw_job.get("output_dir")
    if not output_dir_value:
        raise RuntimeError("Each config job must define output_dir.")
    output_dir = _resolve_path(str(output_dir_value), base_dir)

    episode_dir_value = raw_job.get("episode_dir")
    if episode_dir_value:
        episode_dir = _resolve_path(str(episode_dir_value), base_dir)
        return ExtractionJob(output_dir=output_dir, episode_dir=episode_dir)

    mcap_path_values = raw_job.get("mcap_paths")
    if not isinstance(mcap_path_values, list) or not mcap_path_values:
        raise RuntimeError("Each config job must define either episode_dir or a non-empty mcap_paths list.")

    mcap_paths = [_resolve_path(str(path), base_dir) for path in mcap_path_values]
    missing_paths = [str(path) for path in mcap_paths if not path.exists()]
    if missing_paths:
        raise RuntimeError("Config referenced missing MCAP files:\n" + "\n".join(missing_paths))

    summary_path_value = raw_job.get("summary_path")
    summary_path = (
        _resolve_path(str(summary_path_value), base_dir)
        if summary_path_value
        else _infer_summary_path(mcap_paths)
    )
    if not summary_path.exists():
        raise RuntimeError(f"Config referenced missing summary.json: {summary_path}")

    return ExtractionJob(
        output_dir=output_dir,
        summary_path=summary_path,
        mcap_paths=mcap_paths,
    )


def _load_jobs(payload, base_dir: Path) -> list[ExtractionJob]:
    if isinstance(payload, dict):
        unknown_keys = sorted(set(payload) - {"path_base", "extract", "jobs"})
        if unknown_keys:
            raise RuntimeError(f"Unknown config keys: {', '.join(unknown_keys)}")
    raw_jobs = payload.get("jobs") if isinstance(payload, dict) else payload
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise RuntimeError("Config JSON must be a non-empty list or an object with a non-empty jobs list.")

    return [_normalize_job(raw_job, base_dir) for raw_job in raw_jobs]


def _load_extract_config(payload) -> dict:
    extract_parser = build_extract_settings_parser()
    raw_extract = {}
    if isinstance(payload, dict):
        raw_extract = payload.get("extract", {})
    if not isinstance(raw_extract, dict):
        raise RuntimeError("Config `extract` must be a JSON object.")

    extract_config = normalize_arg_config(
        raw_extract,
        extract_parser,
        exclude_dests={"help"},
    )
    extract_parser.set_defaults(**extract_config)
    return vars(extract_parser.parse_args([]))


def parse_args() -> argparse.Namespace:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        return build_config_parser().parse_args()

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config-json", type=str, required=True)
    pre_args, remaining = pre_parser.parse_known_args()
    if remaining:
        raise RuntimeError(
            "Stage 1 extraction accepts only --config-json. Put jobs and extraction settings in the JSON file."
        )

    parser = build_config_parser()
    args = parser.parse_args()
    config_path, payload = load_json_payload(pre_args.config_json)
    base_dir = _resolve_config_base_dir(config_path, payload)
    extract_config = _load_extract_config(payload)
    jobs = _load_jobs(payload, base_dir)

    args.config_json = str(config_path)
    args.config_payload = payload
    args.extract_config = extract_config
    args.jobs = jobs
    for key, value in extract_config.items():
        setattr(args, key, value)
    return args


def _load_frames(mcap_paths: list[Path]) -> list[FrameRecord]:
    topics = {
        "/camera/front/compressed",
        "/ego/state",
        "/ego/planning",
    }
    raw_frames: dict[tuple[int, int], dict] = {}
    for mcap_path in mcap_paths:
        with mcap_path.open("rb") as f:
            reader = make_reader(f)
            for _, channel, message in reader.iter_messages(topics=topics):
                payload = json.loads(message.data)
                try:
                    timestamp = payload["timestamp"]
                    ts_key = (int(timestamp["sec"]), int(timestamp["nsec"]))
                    frame = raw_frames.setdefault(ts_key, {})
                    if channel.topic == "/camera/front/compressed":
                        frame["image_b64"] = payload["data"]
                        frame["image_format"] = str(payload["format"])
                    elif channel.topic == "/ego/state":
                        frame["frame_id"] = int(payload["frame_id"])
                        pose = payload["pose"]
                        frame["speed_mps"] = float(payload["speed_mps"])
                        frame["x"] = float(pose["x"])
                        frame["y"] = float(pose["y"])
                        frame["yaw_deg"] = float(pose["yaw_deg"])
                    elif channel.topic == "/ego/planning":
                        frame["frame_id"] = int(payload["frame_id"])
                        frame["command"] = str(payload["behavior"])
                        frame["planner_state"] = str(payload["planner_state"])
                except KeyError as exc:
                    raise RuntimeError(
                        f"Missing required field {exc.args[0]!r} in {channel.topic} message from {mcap_path}."
                    ) from exc

    frames: list[FrameRecord] = []
    for ts_key in sorted(raw_frames):
        frame = raw_frames[ts_key]
        required = {
            "frame_id",
            "image_b64",
            "image_format",
            "speed_mps",
            "x",
            "y",
            "yaw_deg",
            "command",
            "planner_state",
        }
        if not required.issubset(frame):
            continue
        frames.append(FrameRecord(**frame))
    return frames


def _record_to_json(
    sample_index: int,
    image_rel_path: str,
    source_frame: FrameRecord,
    history_pose_frames: list[FrameRecord],
    ego_history_xyz: np.ndarray,
    ego_history_rot: np.ndarray,
    ego_future_xyz: np.ndarray,
    ego_future_rot: np.ndarray,
    future_pose_frames: list[FrameRecord],
    gt_waypoints: np.ndarray,
    action: np.ndarray,
    dt: float,
) -> dict:
    return {
        "sample_id": f"{source_frame.frame_id:06d}",
        "sample_index": sample_index,
        "source_frame_id": source_frame.frame_id,
        "image_path": image_rel_path,
        "v0": float(source_frame.speed_mps),
        "command": source_frame.command,
        "planner_state": source_frame.planner_state,
        "dt": dt,
        "ego_pose": {
            "x": float(source_frame.x),
            "y": float(source_frame.y),
            "yaw_deg": float(source_frame.yaw_deg),
        },
        "ego_history_xyz": np.expand_dims(ego_history_xyz, axis=0).tolist(),
        "ego_history_rot": np.expand_dims(ego_history_rot, axis=0).tolist(),
        "ego_future_xyz": np.expand_dims(ego_future_xyz, axis=0).tolist(),
        "ego_future_rot": np.expand_dims(ego_future_rot, axis=0).tolist(),
        "history_poses_global": [
            {
                "frame_id": frame.frame_id,
                "x": float(frame.x),
                "y": float(frame.y),
                "yaw_deg": float(frame.yaw_deg),
            }
            for frame in history_pose_frames
        ],
        "gt_waypoints": gt_waypoints.tolist(),
        "action": action.tolist(),
        "future_poses_global": [
            {
                "frame_id": frame.frame_id,
                "x": float(frame.x),
                "y": float(frame.y),
                "yaw_deg": float(frame.yaw_deg),
            }
            for frame in future_pose_frames
        ],
    }


def _emit_progress(event: str, **payload) -> None:
    record = {"event": event, **payload}
    print(json.dumps(record, ensure_ascii=False), file=sys.stderr, flush=True)


def _rotation_matrix_from_yaw(yaw_rad: float) -> np.ndarray:
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)
    return np.asarray(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _describe_job(job: ExtractionJob) -> str:
    if job.episode_dir is not None:
        return job.episode_dir.name
    return job.output_dir.name


def _job_config_record(job: ExtractionJob) -> dict:
    return {
        "episode_dir": str(job.episode_dir) if job.episode_dir is not None else None,
        "summary_path": str(job.summary_path) if job.summary_path is not None else None,
        "mcap_paths": [str(path) for path in job.mcap_paths] if job.mcap_paths is not None else None,
        "output_dir": str(job.output_dir),
    }


def _extract_job(job: ExtractionJob, args: argparse.Namespace, *, job_index: int, total_jobs: int) -> dict:
    output_dir = job.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    job_name = _describe_job(job)

    summary_path = (job.episode_dir / "summary.json") if job.episode_dir is not None else job.summary_path
    if summary_path is None:
        raise RuntimeError("Extraction job has neither episode_dir nor summary_path.")

    record_hz = _read_record_hz(summary_path)
    future_stride = max(1, int(round(args.dt * record_hz)))
    sample_stride = args.sample_stride if args.sample_stride > 0 else future_stride

    mcap_paths = _resolve_mcap_paths(job.episode_dir) if job.episode_dir is not None else job.mcap_paths
    if not mcap_paths:
        raise RuntimeError("Extraction job resolved no MCAP paths.")

    frames = _load_frames(mcap_paths)
    min_start = args.history_steps - 1
    max_start = len(frames) - 1 - (args.k + 1) * future_stride
    if max_start < min_start:
        raise RuntimeError("Not enough frames in the episode for the requested horizon.")
    max_possible_samples = ((max_start - min_start) // sample_stride) + 1
    target_samples = min(max_possible_samples, args.max_samples) if args.max_samples > 0 else max_possible_samples

    _emit_progress(
        "extract_start",
        job_index=job_index,
        total_jobs=total_jobs,
        job_name=job_name,
        episode_dir=str(job.episode_dir) if job.episode_dir is not None else None,
        output_dir=str(output_dir),
        num_mcap_files=len(mcap_paths),
        num_frames=len(frames),
        target_samples=target_samples,
        dt=args.dt,
        k=args.k,
        history_steps=args.history_steps,
    )

    samples: list[dict] = []
    sample_count = 0
    for start_idx in range(min_start, max_start + 1, sample_stride):
        if args.max_samples > 0 and sample_count >= args.max_samples:
            break

        source_frame = frames[start_idx]
        history_pose_frames = frames[start_idx - args.history_steps + 1 : start_idx + 1]
        pose_window = [frames[start_idx + step * future_stride] for step in range(args.k + 2)]
        positions = np.asarray([[frame.x, frame.y] for frame in pose_window], dtype=np.float64)
        headings = np.deg2rad(np.asarray([frame.yaw_deg for frame in pose_window], dtype=np.float64))

        gt_waypoints = to_ego_centric(
            positions[1 : args.k + 1],
            positions[0],
            headings[0],
        )
        history_positions = np.asarray(
            [[frame.x, frame.y] for frame in history_pose_frames],
            dtype=np.float64,
        )
        history_headings = np.deg2rad(
            np.asarray([frame.yaw_deg for frame in history_pose_frames], dtype=np.float64)
        )
        ego_history_xy = to_ego_centric(history_positions, positions[0], headings[0]).astype(np.float32)
        ego_history_xyz = np.concatenate(
            [ego_history_xy, np.zeros((args.history_steps, 1), dtype=np.float32)],
            axis=1,
        )
        ego_history_yaw = np.arctan2(
            np.sin(history_headings - headings[0]),
            np.cos(history_headings - headings[0]),
        ).astype(np.float32)
        ego_history_rot = np.stack(
            [_rotation_matrix_from_yaw(float(yaw)) for yaw in ego_history_yaw],
            axis=0,
        )
        ego_future_xyz = np.concatenate(
            [gt_waypoints.astype(np.float32), np.zeros((args.k, 1), dtype=np.float32)],
            axis=1,
        )
        ego_future_yaw = np.arctan2(
            np.sin(headings[1 : args.k + 1] - headings[0]),
            np.cos(headings[1 : args.k + 1] - headings[0]),
        ).astype(np.float32)
        ego_future_rot = np.stack(
            [_rotation_matrix_from_yaw(float(yaw)) for yaw in ego_future_yaw],
            axis=0,
        )
        action = (
            canonical_action_tensor_from_tensors(
                history_xyz=torch.from_numpy(ego_history_xyz).unsqueeze(0),
                history_rot=torch.from_numpy(ego_history_rot).unsqueeze(0),
                future_xyz=torch.from_numpy(ego_future_xyz).unsqueeze(0),
                future_rot=torch.from_numpy(ego_future_rot).unsqueeze(0),
                dt=args.dt,
            )
            .cpu()
            .numpy()
        )

        image_rel_path = f"images/{source_frame.frame_id:06d}.{source_frame.image_format}"
        image_path = output_dir / image_rel_path
        image_path.write_bytes(base64.b64decode(source_frame.image_b64))

        sample = _record_to_json(
            sample_index=sample_count,
            image_rel_path=image_rel_path,
            source_frame=source_frame,
            history_pose_frames=history_pose_frames,
            ego_history_xyz=ego_history_xyz,
            ego_history_rot=ego_history_rot,
            ego_future_xyz=ego_future_xyz,
            ego_future_rot=ego_future_rot,
            future_pose_frames=pose_window[1:],
            gt_waypoints=gt_waypoints,
            action=action,
            dt=args.dt,
        )
        samples.append(sample)
        sample_count += 1
        if args.log_every > 0 and sample_count % args.log_every == 0:
            _emit_progress(
                "extract_progress",
                job_index=job_index,
                total_jobs=total_jobs,
                job_name=job_name,
                samples_written=sample_count,
                target_samples=target_samples,
            )

    jsonl_path = output_dir / "samples.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    run_config_path = output_dir / "run_config.json"
    with run_config_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config_json": args.config_json,
                "config_payload": args.config_payload,
                "resolved_extract_config": args.extract_config,
                "job": _job_config_record(job),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    summary = {
        "config_json": args.config_json,
        "extract_config": args.extract_config,
        "episode_dir": str(job.episode_dir) if job.episode_dir is not None else None,
        "summary_path": str(summary_path),
        "record_hz": record_hz,
        "dt": args.dt,
        "num_mcap_files": len(mcap_paths),
        "mcap_paths": [str(path) for path in mcap_paths],
        "future_stride_frames": future_stride,
        "sample_stride_frames": sample_stride,
        "k": args.k,
        "history_steps": args.history_steps,
        "num_frames": len(frames),
        "num_samples": len(samples),
        "jsonl_path": str(jsonl_path),
        "image_dir": str(image_dir),
        "run_config_path": str(run_config_path),
    }
    with (output_dir / "extract_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _emit_progress(
        "extract_complete",
        job_index=job_index,
        total_jobs=total_jobs,
        job_name=job_name,
        num_samples=len(samples),
        jsonl_path=str(jsonl_path),
    )

    return summary


def main() -> None:
    args = parse_args()
    require_clean_git_worktree(Path(__file__).resolve().parent)
    jobs = args.jobs
    summaries = [_extract_job(job, args, job_index=index, total_jobs=len(jobs)) for index, job in enumerate(jobs, start=1)]

    if len(summaries) == 1:
        print(json.dumps(summaries[0], indent=2, ensure_ascii=False))
        return

    print(json.dumps({"jobs": summaries}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
