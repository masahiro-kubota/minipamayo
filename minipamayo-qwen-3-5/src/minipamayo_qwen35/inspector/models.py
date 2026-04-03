"""Shared dataclasses for the local eval inspector."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactManifest:
    artifact_kind: str
    stage: str
    run_name: str
    summary_json: str
    checkpoint: str | None = None
    dataset_path: str | None = None
    progress_json: str | None = None
    per_sample_jsonl: str | None = None
    plots_dir: str | None = None
    plots: dict[str, str] = field(default_factory=dict)
    wandb_run_url: str | None = None

    @property
    def manifest_path(self) -> Path:
        summary_path = Path(self.summary_json).resolve()
        return summary_path.with_name(f"{summary_path.stem}.manifest.json")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "artifact_kind": self.artifact_kind,
            "stage": self.stage,
            "run_name": self.run_name,
            "summary_json": str(Path(self.summary_json).resolve()),
        }
        if self.plots:
            payload["plots"] = dict(self.plots)
        optional_scalars = {
            "checkpoint": self.checkpoint,
            "dataset_path": self.dataset_path,
            "progress_json": self.progress_json,
            "per_sample_jsonl": self.per_sample_jsonl,
            "plots_dir": self.plots_dir,
            "wandb_run_url": self.wandb_run_url,
        }
        for key, value in optional_scalars.items():
            if value is not None:
                payload[key] = value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ArtifactManifest":
        return cls(
            artifact_kind=str(payload["artifact_kind"]),
            stage=str(payload["stage"]),
            run_name=str(payload["run_name"]),
            summary_json=str(payload["summary_json"]),
            checkpoint=str(payload["checkpoint"]) if "checkpoint" in payload else None,
            dataset_path=str(payload["dataset_path"]) if "dataset_path" in payload else None,
            progress_json=str(payload["progress_json"]) if "progress_json" in payload else None,
            per_sample_jsonl=(
                str(payload["per_sample_jsonl"]) if "per_sample_jsonl" in payload else None
            ),
            plots_dir=str(payload["plots_dir"]) if "plots_dir" in payload else None,
            plots={str(key): str(value) for key, value in dict(payload.get("plots", {})).items()},
            wandb_run_url=str(payload["wandb_run_url"]) if "wandb_run_url" in payload else None,
        )


@dataclass(frozen=True)
class NormalizedSample:
    stage: str
    run_name: str
    sample_id: str
    sample_index: int
    image_path: str
    source_frame_id: str = ""
    group_id: str = ""
    group_frame_index: int = 0
    group_length: int = 1
    command: str = ""
    gt_waypoints: list[list[float]] = field(default_factory=list)
    pred_waypoints: list[list[float]] = field(default_factory=list)
    pid_pred_waypoints: list[list[float]] | None = None
    ade_m: float | None = None
    fde_m: float | None = None
    max_lateral_error_m: float | None = None
    token_accuracy: float | None = None
    action_mae_kappa: float | None = None
    reasoning_text_gt: str = ""
    reasoning_text_pred: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "run_name": self.run_name,
            "sample_id": self.sample_id,
            "sample_index": int(self.sample_index),
            "image_path": self.image_path,
            "source_frame_id": self.source_frame_id,
            "group_id": self.group_id,
            "group_frame_index": int(self.group_frame_index),
            "group_length": int(self.group_length),
            "command": self.command,
            "ade_m": self.ade_m,
            "fde_m": self.fde_m,
            "max_lateral_error_m": self.max_lateral_error_m,
            "token_accuracy": self.token_accuracy,
            "action_mae_kappa": self.action_mae_kappa,
            "has_pred_waypoints": bool(self.pred_waypoints),
            "has_pid_pred_waypoints": bool(self.pid_pred_waypoints),
            "has_reasoning_text_pred": bool(self.reasoning_text_pred),
            "has_reasoning_text_gt": bool(self.reasoning_text_gt),
        }


@dataclass(frozen=True)
class NormalizedGroup:
    group_id: str
    run_name: str
    stage: str
    samples: tuple[NormalizedSample, ...]
    start_sample_index: int
    end_sample_index: int
    metrics_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.samples)

    def to_row(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "stage": self.stage,
            "run_name": self.run_name,
            "length": int(self.length),
            "start_sample_index": int(self.start_sample_index),
            "end_sample_index": int(self.end_sample_index),
            "commands": ", ".join(self.metrics_summary.get("commands", [])),
            "mean_ade_m": self.metrics_summary.get("mean_ade_m"),
            "max_ade_m": self.metrics_summary.get("max_ade_m"),
            "mean_fde_m": self.metrics_summary.get("mean_fde_m"),
            "max_fde_m": self.metrics_summary.get("max_fde_m"),
            "mean_action_mae_kappa": self.metrics_summary.get("mean_action_mae_kappa"),
            "mean_token_accuracy": self.metrics_summary.get("mean_token_accuracy"),
        }


@dataclass(frozen=True)
class NormalizedRun:
    manifest: ArtifactManifest
    summary: dict[str, Any]
    samples: list[NormalizedSample]
    groups: list[NormalizedGroup] = field(default_factory=list)
    invalid_reason: str | None = None

    @property
    def stage(self) -> str:
        return self.manifest.stage

    @property
    def run_name(self) -> str:
        return self.manifest.run_name

    @property
    def browser_unavailable_reason(self) -> str | None:
        if self.manifest.per_sample_jsonl is None:
            return "Sample/Block view unavailable for summary-only artifact."
        if not self.samples:
            return "No sample payload available for this artifact."
        return None


@dataclass(frozen=True)
class Stage2RunBundle:
    run_name: str
    eval_manifest: ArtifactManifest | None
    inference_manifest: ArtifactManifest | None
