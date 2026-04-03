"""Shared progress reporting for eval and inference entrypoints."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .preflight import init_required_online_wandb

DEFAULT_WANDB_PROJECT = ""
DEFAULT_PROGRESS_EVERY_SAMPLES = 0
DEFAULT_PROGRESS_EVERY_SECONDS = 0.0


def add_eval_reporting_args(
    parser: argparse.ArgumentParser,
    *,
    include_per_sample_jsonl: bool,
) -> None:
    parser.add_argument("--progress-json", type=str, default="")
    if include_per_sample_jsonl:
        parser.add_argument("--per-sample-jsonl", type=str, default="")
    parser.add_argument("--progress-every-samples", type=int, default=DEFAULT_PROGRESS_EVERY_SAMPLES)
    parser.add_argument("--progress-every-seconds", type=float, default=DEFAULT_PROGRESS_EVERY_SECONDS)
    parser.add_argument("--wandb-project", type=str, default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")


def reporting_path_keys(*, include_per_sample_jsonl: bool) -> set[str]:
    keys = {"progress_json"}
    if include_per_sample_jsonl:
        keys.add("per_sample_jsonl")
    return keys


def _require_non_empty_string_arg(args: argparse.Namespace, arg_name: str) -> None:
    if not hasattr(args, arg_name):
        raise RuntimeError(f"`{arg_name}` must be defined on the argparse namespace.")
    if not str(getattr(args, arg_name, "")):
        raise RuntimeError(f"`{arg_name}` must be defined in the config JSON.")


def validate_eval_reporting_args(
    args: argparse.Namespace,
    *,
    require_output_json: bool = True,
    require_progress_json: bool = True,
    require_per_sample_jsonl: bool = False,
) -> None:
    if require_output_json:
        _require_non_empty_string_arg(args, "output_json")
    if require_progress_json:
        _require_non_empty_string_arg(args, "progress_json")
    if require_per_sample_jsonl:
        _require_non_empty_string_arg(args, "per_sample_jsonl")
    if int(args.progress_every_samples) <= 0:
        raise RuntimeError("`progress_every_samples` must be > 0.")
    if float(args.progress_every_seconds) <= 0.0:
        raise RuntimeError("`progress_every_seconds` must be > 0.")
    _require_non_empty_string_arg(args, "wandb_project")
    _require_non_empty_string_arg(args, "wandb_run_name")


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _resolve_optional_path(path_value: str) -> Path | None:
    if path_value:
        return Path(path_value).resolve()
    return None


def _resolve_progress_json_path(progress_json: str) -> Path | None:
    if progress_json:
        return Path(progress_json).resolve()
    return None


def _resolve_per_sample_jsonl_path(per_sample_jsonl: str) -> Path | None:
    if per_sample_jsonl:
        return Path(per_sample_jsonl).resolve()
    return None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    temp_path.replace(path)


def _flatten_scalars(payload: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        key_name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_scalars(value, prefix=key_name))
            continue
        if isinstance(value, bool | int | float | str):
            flattened[key_name] = value
    return flattened


def _numeric_scalars(payload: dict[str, Any]) -> dict[str, float | int | bool]:
    scalars: dict[str, float | int | bool] = {}
    for key, value in _flatten_scalars(payload).items():
        if isinstance(value, bool | int | float):
            scalars[key] = value
    return scalars


def _json_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


@dataclass
class EvalReporter:
    stage: str
    total_samples: int
    output_json_path: Path | None
    progress_json_path: Path | None
    per_sample_jsonl_path: Path | None
    progress_every_samples: int
    progress_every_seconds: float
    wandb_run: Any
    start_time_monotonic: float = field(default_factory=time.monotonic)
    processed_samples: int = 0
    last_metrics: dict[str, Any] = field(default_factory=dict)
    last_event: str = "initialized"
    _last_reported_samples: int = 0
    _last_report_time_monotonic: float = field(default_factory=time.monotonic)
    _closed: bool = False

    @classmethod
    def from_args(
        cls,
        *,
        args: argparse.Namespace,
        stage: str,
        total_samples: int,
        checkpoint: str,
        dataset_path: str,
        extra_wandb_config: dict[str, Any] | None = None,
    ) -> "EvalReporter":
        validate_eval_reporting_args(args)
        output_json = str(getattr(args, "output_json", ""))
        progress_json = _resolve_progress_json_path(str(getattr(args, "progress_json", "")))
        per_sample_jsonl = _resolve_per_sample_jsonl_path(str(getattr(args, "per_sample_jsonl", "")))
        run_name = str(getattr(args, "wandb_run_name", ""))
        wandb_config = {
            "stage": stage,
            "config_json": str(getattr(args, "config_json", "")),
            "checkpoint": checkpoint,
            "dataset_path": dataset_path,
            "output_json": output_json,
            "progress_json": str(progress_json) if progress_json is not None else "",
            "per_sample_jsonl": str(per_sample_jsonl) if per_sample_jsonl is not None else "",
            "total_samples": total_samples,
        }
        if extra_wandb_config:
            wandb_config.update(extra_wandb_config)
        wandb_run = init_required_online_wandb(
            project=str(args.wandb_project),
            entity=str(getattr(args, "wandb_entity", "")),
            name=run_name,
            config=wandb_config,
        )
        return cls(
            stage=stage,
            total_samples=total_samples,
            output_json_path=_resolve_optional_path(output_json),
            progress_json_path=progress_json,
            per_sample_jsonl_path=per_sample_jsonl,
            progress_every_samples=int(args.progress_every_samples),
            progress_every_seconds=float(args.progress_every_seconds),
            wandb_run=wandb_run,
        )

    def emit_event(self, event_name: str, payload: dict[str, Any]) -> None:
        event_payload = {"event": event_name, "stage": self.stage, **payload}
        print(_json_line(event_payload), flush=True)

    def emit_setup(self, event_name: str, payload: dict[str, Any]) -> None:
        self.last_event = event_name
        self.emit_event(event_name, payload)
        self._write_progress_snapshot(state="running")
        self._update_wandb_summary({"setup": payload})

    def emit_sample(self, payload: dict[str, Any], *, print_to_stdout: bool) -> None:
        sample_payload = {"stage": self.stage, **payload}
        if self.per_sample_jsonl_path is not None:
            self.per_sample_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self.per_sample_jsonl_path.open("a", encoding="utf-8") as f:
                f.write(_json_line(sample_payload))
                f.write("\n")
        if print_to_stdout:
            print(_json_line(sample_payload), flush=True)

    def emit_progress(
        self,
        *,
        processed_samples: int,
        running_metrics: dict[str, Any],
        extra_payload: dict[str, Any] | None = None,
        force: bool = False,
    ) -> None:
        self.processed_samples = int(processed_samples)
        self.last_metrics = dict(running_metrics)
        now = time.monotonic()
        should_report = force
        if not should_report:
            if self.processed_samples == 0:
                should_report = False
            elif self.processed_samples >= self.total_samples:
                should_report = True
            elif (self.processed_samples - self._last_reported_samples) >= self.progress_every_samples:
                should_report = True
            elif (now - self._last_report_time_monotonic) >= self.progress_every_seconds:
                should_report = True
        if not should_report:
            return

        self.last_event = "eval_progress"
        elapsed_s = max(now - self.start_time_monotonic, 1e-6)
        samples_per_s = self.processed_samples / elapsed_s
        remaining_samples = max(self.total_samples - self.processed_samples, 0)
        eta_s = remaining_samples / samples_per_s if self.processed_samples > 0 else None
        payload = {
            "processed_samples": self.processed_samples,
            "total_samples": self.total_samples,
            "percent": round((self.processed_samples / max(self.total_samples, 1)) * 100.0, 3),
            "elapsed_s": round(elapsed_s, 3),
            "samples_per_s": round(samples_per_s, 6),
            "eta_s": round(eta_s, 3) if eta_s is not None else None,
            "metrics": running_metrics,
        }
        if extra_payload:
            payload.update(extra_payload)
        self.emit_event("eval_progress", payload)
        self._write_progress_snapshot(state="running")
        self.wandb_run.log(
            {
                "progress/processed_samples": self.processed_samples,
                "progress/total_samples": self.total_samples,
                "progress/percent": payload["percent"],
                "progress/elapsed_s": payload["elapsed_s"],
                "progress/samples_per_s": payload["samples_per_s"],
                **{f"running/{key}": value for key, value in _numeric_scalars(running_metrics).items()},
            },
            step=self.processed_samples,
        )
        self._last_reported_samples = self.processed_samples
        self._last_report_time_monotonic = now

    def emit_summary(self, event_name: str, summary: dict[str, Any]) -> None:
        self.processed_samples = max(self.processed_samples, self.total_samples)
        self.last_metrics = dict(summary.get("metrics", {})) if isinstance(summary.get("metrics"), dict) else dict(summary)
        if self.output_json_path is not None:
            _atomic_write_json(self.output_json_path, summary)
        self.last_event = event_name
        self.emit_event(event_name, summary)
        self._write_progress_snapshot(state="completed", summary=summary)
        final_scalars = _flatten_scalars(summary)
        final_numeric_scalars = _numeric_scalars(summary)
        if final_numeric_scalars:
            self.wandb_run.log(
                {f"final/{key}": value for key, value in final_numeric_scalars.items()},
                step=self.processed_samples,
            )
        if final_scalars:
            self._update_wandb_summary(final_scalars)
        self.close()

    def emit_failure(self, event_name: str, error: Exception) -> None:
        payload = {
            "error_type": error.__class__.__name__,
            "error_message": str(error),
            "processed_samples": self.processed_samples,
            "total_samples": self.total_samples,
        }
        self.last_event = event_name
        self.emit_event(event_name, payload)
        self._write_progress_snapshot(state="failed", error=payload)
        self._update_wandb_summary({"failure": payload})
        self.close(exit_code=1)

    def close(self, *, exit_code: int = 0) -> None:
        if self._closed:
            return
        self._closed = True
        self.wandb_run.summary["exit_code"] = exit_code
        self.wandb_run.finish()

    def _update_wandb_summary(self, payload: dict[str, Any]) -> None:
        for key, value in _flatten_scalars(payload).items():
            self.wandb_run.summary[key] = value

    def _write_progress_snapshot(
        self,
        *,
        state: str,
        summary: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        if self.progress_json_path is None:
            return
        now = time.monotonic()
        elapsed_s = max(now - self.start_time_monotonic, 0.0)
        samples_per_s = (self.processed_samples / elapsed_s) if elapsed_s > 0.0 and self.processed_samples > 0 else None
        eta_s = None
        if samples_per_s and self.total_samples > self.processed_samples:
            eta_s = (self.total_samples - self.processed_samples) / samples_per_s
        payload = {
            "stage": self.stage,
            "state": state,
            "current_event": self.last_event,
            "processed_samples": self.processed_samples,
            "total_samples": self.total_samples,
            "percent": round((self.processed_samples / max(self.total_samples, 1)) * 100.0, 3),
            "elapsed_s": round(elapsed_s, 3),
            "samples_per_s": round(samples_per_s, 6) if samples_per_s is not None else None,
            "eta_s": round(eta_s, 3) if eta_s is not None else None,
            "updated_at": _now_iso(),
            "running_metrics": self.last_metrics,
        }
        if self.output_json_path is not None:
            payload["output_json"] = str(self.output_json_path)
        if self.per_sample_jsonl_path is not None:
            payload["per_sample_jsonl"] = str(self.per_sample_jsonl_path)
        if summary is not None:
            payload["summary"] = summary
        if error is not None:
            payload["error"] = error
        _atomic_write_json(self.progress_json_path, payload)
