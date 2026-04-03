"""Canonical helpers for repo-local checkpoint run directories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .artifact_paths import derive_track_from_config_path, normalize_track


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"


@dataclass(frozen=True)
class CheckpointScope:
    stage: str
    component: str
    track: str

    def normalized_track(self) -> tuple[str, ...]:
        return normalize_track(self.track)


def _normalize_path_like(value: str | Path) -> Path:
    return Path(str(value)).resolve()


def _validate_non_empty_segment(value: str, *, label: str) -> str:
    if not value or "/" in value or value in {".", ".."}:
        raise RuntimeError(f"`{label}` must be a non-empty path segment, got {value!r}.")
    return value


def checkpoint_scope_dir(scope: CheckpointScope, *, project_root: Path | None = None) -> Path:
    stage = _validate_non_empty_segment(scope.stage, label="stage")
    component = _validate_non_empty_segment(scope.component, label="component")
    root = (project_root or PROJECT_ROOT).resolve() / "checkpoints"
    return root.joinpath(stage, component, *scope.normalized_track())


def checkpoint_scope_from_config_path(
    config_path: str | Path,
    *,
    stage: str,
    component: str,
    project_root: Path | None = None,
    default_track: str | None = None,
) -> CheckpointScope:
    return CheckpointScope(
        stage=stage,
        component=component,
        track=derive_track_from_config_path(
            config_path,
            project_root=project_root,
            default_track=default_track,
        ),
    )


def checkpoint_run_dir(
    scope: CheckpointScope,
    run_name: str,
    *,
    project_root: Path | None = None,
) -> Path:
    return checkpoint_scope_dir(scope, project_root=project_root) / _validate_non_empty_segment(
        run_name,
        label="run_name",
    )


def validate_checkpoint_run_dir(
    path_value: str | Path,
    *,
    scope: CheckpointScope,
    run_name: str,
    project_root: Path | None = None,
) -> Path:
    path = _normalize_path_like(path_value)
    expected = checkpoint_run_dir(scope, run_name, project_root=project_root).resolve()
    if path != expected:
        raise RuntimeError(
            "Checkpoint run directory must match the canonical location.\n"
            f"expected={expected}\n"
            f"actual={path}"
        )
    return expected


def resolve_checkpoint_run_dir(
    path_value: str | Path,
    *,
    scope: CheckpointScope,
    run_name: str,
    project_root: Path | None = None,
) -> Path:
    if str(path_value):
        return validate_checkpoint_run_dir(
            path_value,
            scope=scope,
            run_name=run_name,
            project_root=project_root,
        )
    return checkpoint_run_dir(scope, run_name, project_root=project_root).resolve()

