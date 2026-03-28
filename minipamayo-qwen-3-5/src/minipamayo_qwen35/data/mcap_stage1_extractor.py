"""Convert CARLA telemetry MCAP episodes into Stage 1 JSONL samples."""

from __future__ import annotations

import argparse
import base64
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mcap.reader import make_reader

from ..utils.dynamics import interleave_action, inverse_dynamics_np, to_ego_centric
from ..utils.preflight import require_clean_git_worktree


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Stage 1 samples from MCAP telemetry.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--episode-dir", type=str, help="Single episode directory to extract.")
    input_group.add_argument("--manifest-json", type=str, help="JSON manifest describing one or more extraction jobs.")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory for --episode-dir mode.")
    parser.add_argument("--k", type=int, default=6, help="Number of predicted action steps.")
    parser.add_argument("--dt", type=float, default=0.5, help="Training step size in seconds.")
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=0,
        help="Stride in source frames between samples. 0 = use future stride.",
    )
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def _resolve_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


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
        raise RuntimeError("Manifest job has no MCAP paths.")
    telemetry_dirs = {path.resolve().parent for path in mcap_paths}
    if len(telemetry_dirs) != 1:
        raise RuntimeError("Manifest jobs with explicit MCAP paths must share the same telemetry directory.")
    telemetry_dir = telemetry_dirs.pop()
    summary_path = telemetry_dir.parent / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"Could not infer summary.json for telemetry directory: {telemetry_dir}")
    return summary_path


def _normalize_job(raw_job: dict, base_dir: Path) -> ExtractionJob:
    if not isinstance(raw_job, dict):
        raise RuntimeError("Each manifest job must be a JSON object.")

    output_dir_value = raw_job.get("output_dir")
    if not output_dir_value:
        raise RuntimeError("Each manifest job must define output_dir.")
    output_dir = _resolve_path(str(output_dir_value), base_dir)

    episode_dir_value = raw_job.get("episode_dir")
    if episode_dir_value:
        episode_dir = _resolve_path(str(episode_dir_value), base_dir)
        return ExtractionJob(output_dir=output_dir, episode_dir=episode_dir)

    mcap_path_values = raw_job.get("mcap_paths")
    if not isinstance(mcap_path_values, list) or not mcap_path_values:
        raise RuntimeError("Each manifest job must define either episode_dir or a non-empty mcap_paths list.")

    mcap_paths = [_resolve_path(str(path), base_dir) for path in mcap_path_values]
    missing_paths = [str(path) for path in mcap_paths if not path.exists()]
    if missing_paths:
        raise RuntimeError("Manifest referenced missing MCAP files:\n" + "\n".join(missing_paths))

    summary_path_value = raw_job.get("summary_path")
    summary_path = (
        _resolve_path(str(summary_path_value), base_dir)
        if summary_path_value
        else _infer_summary_path(mcap_paths)
    )
    if not summary_path.exists():
        raise RuntimeError(f"Manifest referenced missing summary.json: {summary_path}")

    return ExtractionJob(
        output_dir=output_dir,
        summary_path=summary_path,
        mcap_paths=mcap_paths,
    )


def _load_jobs(args: argparse.Namespace) -> list[ExtractionJob]:
    if args.episode_dir:
        if not args.output_dir:
            raise RuntimeError("--output-dir is required when using --episode-dir.")
        return [ExtractionJob(output_dir=Path(args.output_dir), episode_dir=Path(args.episode_dir))]

    manifest_path = Path(args.manifest_json).resolve()
    with manifest_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_jobs = payload.get("jobs") if isinstance(payload, dict) else payload
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise RuntimeError("Manifest JSON must be a non-empty list or an object with a non-empty jobs list.")

    return [_normalize_job(raw_job, manifest_path.parent) for raw_job in raw_jobs]


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
                timestamp = payload["timestamp"]
                ts_key = (int(timestamp["sec"]), int(timestamp["nsec"]))
                frame = raw_frames.setdefault(ts_key, {})
                if channel.topic == "/camera/front/compressed":
                    frame["image_b64"] = payload["data"]
                    frame["image_format"] = payload.get("format", "jpeg")
                elif channel.topic == "/ego/state":
                    frame["frame_id"] = int(payload["frame_id"])
                    pose = payload["pose"]
                    frame["speed_mps"] = float(payload["speed_mps"])
                    frame["x"] = float(pose["x"])
                    frame["y"] = float(pose["y"])
                    frame["yaw_deg"] = float(pose["yaw_deg"])
                elif channel.topic == "/ego/planning":
                    frame["frame_id"] = int(payload["frame_id"])
                    frame["command"] = str(payload.get("behavior", "lanefollow"))
                    frame["planner_state"] = str(payload.get("planner_state", "unknown"))

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


def _extract_job(job: ExtractionJob, args: argparse.Namespace) -> dict:
    output_dir = job.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

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
    max_start = len(frames) - 1 - (args.k + 1) * future_stride
    if max_start < 0:
        raise RuntimeError("Not enough frames in the episode for the requested horizon.")

    samples: list[dict] = []
    sample_count = 0
    for start_idx in range(0, max_start + 1, sample_stride):
        if args.max_samples > 0 and sample_count >= args.max_samples:
            break

        source_frame = frames[start_idx]
        pose_window = [frames[start_idx + step * future_stride] for step in range(args.k + 2)]
        positions = np.asarray([[frame.x, frame.y] for frame in pose_window], dtype=np.float64)
        headings = np.deg2rad(np.asarray([frame.yaw_deg for frame in pose_window], dtype=np.float64))

        accel, kappa = inverse_dynamics_np(positions, headings, dt=args.dt)
        action = interleave_action(accel, kappa)
        gt_waypoints = to_ego_centric(
            positions[1 : args.k + 1],
            positions[0],
            headings[0],
        )

        image_rel_path = f"images/{source_frame.frame_id:06d}.{source_frame.image_format}"
        image_path = output_dir / image_rel_path
        image_path.write_bytes(base64.b64decode(source_frame.image_b64))

        sample = _record_to_json(
            sample_index=sample_count,
            image_rel_path=image_rel_path,
            source_frame=source_frame,
            future_pose_frames=pose_window[1:],
            gt_waypoints=gt_waypoints,
            action=action,
            dt=args.dt,
        )
        samples.append(sample)
        sample_count += 1

    jsonl_path = output_dir / "samples.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    summary = {
        "episode_dir": str(job.episode_dir) if job.episode_dir is not None else None,
        "summary_path": str(summary_path),
        "record_hz": record_hz,
        "dt": args.dt,
        "num_mcap_files": len(mcap_paths),
        "mcap_paths": [str(path) for path in mcap_paths],
        "future_stride_frames": future_stride,
        "sample_stride_frames": sample_stride,
        "k": args.k,
        "num_frames": len(frames),
        "num_samples": len(samples),
        "jsonl_path": str(jsonl_path),
        "image_dir": str(image_dir),
    }
    with (output_dir / "extract_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main() -> None:
    require_clean_git_worktree(Path(__file__).resolve().parent)
    args = parse_args()
    jobs = _load_jobs(args)
    summaries = [_extract_job(job, args) for job in jobs]

    if len(summaries) == 1:
        print(json.dumps(summaries[0], indent=2, ensure_ascii=False))
        return

    print(json.dumps({"jobs": summaries}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
