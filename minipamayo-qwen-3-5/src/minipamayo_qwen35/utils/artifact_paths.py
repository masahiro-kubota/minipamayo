"""Canonical helpers for repo-local artifact paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
ALLOWED_ARTIFACT_KINDS = frozenset({"preprocess", "profile", "eval", "inference", "run_logs"})
ALLOWED_TRACK_ROOTS = frozenset({"canonical", "experiments", "debug"})


@dataclass(frozen=True)
class ArtifactScope:
    kind: str
    stage: str
    component: str
    track: str

    def normalized_track(self) -> tuple[str, ...]:
        return normalize_track(self.track)


@dataclass(frozen=True)
class ArtifactReportingPaths:
    output_json: Path
    progress_json: Path
    per_sample_jsonl: Path | None
    manifest_json: Path
    plots_dir: Path
    output_mcap: Path


def default_artifact_root() -> Path:
    return ARTIFACT_ROOT


def _normalize_path_like(value: str | Path) -> Path:
    return Path(str(value)).resolve()


def _validate_non_empty_segment(value: str, *, label: str) -> str:
    if not value or "/" in value or value in {".", ".."}:
        raise RuntimeError(f"`{label}` must be a non-empty path segment, got {value!r}.")
    return value


def normalize_track(track: str) -> tuple[str, ...]:
    raw_parts = tuple(part for part in Path(track).parts if part not in {"", "."})
    if not raw_parts:
        raise RuntimeError("Artifact `track` must not be empty.")
    if any(part == ".." for part in raw_parts):
        raise RuntimeError("Artifact `track` must not contain `..`.")
    track_root = raw_parts[0]
    if track_root not in ALLOWED_TRACK_ROOTS:
        choices = ", ".join(sorted(ALLOWED_TRACK_ROOTS))
        raise RuntimeError(f"Artifact `track` root must be one of: {choices}.")
    if track_root == "experiments":
        if len(raw_parts) < 2:
            raise RuntimeError("Artifact `track` under `experiments` must include an experiment name.")
    elif len(raw_parts) != 1:
        raise RuntimeError(f"Artifact `track` `{track}` must not contain nested directories.")
    for part in raw_parts:
        _validate_non_empty_segment(part, label="track")
    return raw_parts


def artifact_scope_dir(scope: ArtifactScope, *, project_root: Path | None = None) -> Path:
    kind = _validate_non_empty_segment(scope.kind, label="kind")
    if kind not in ALLOWED_ARTIFACT_KINDS - {"run_logs"}:
        choices = ", ".join(sorted(ALLOWED_ARTIFACT_KINDS - {"run_logs"}))
        raise RuntimeError(f"Artifact `kind` must be one of: {choices}.")
    stage = _validate_non_empty_segment(scope.stage, label="stage")
    component = _validate_non_empty_segment(scope.component, label="component")
    root = (project_root or PROJECT_ROOT).resolve() / "artifacts"
    return root.joinpath(kind, stage, component, *normalize_track(scope.track))


def owner_json_path(scope: ArtifactScope, run_name: str, *, project_root: Path | None = None) -> Path:
    return artifact_scope_dir(scope, project_root=project_root) / f"{_validate_non_empty_segment(run_name, label='run_name')}.json"


def bundle_dir(scope: ArtifactScope, run_name: str, *, project_root: Path | None = None) -> Path:
    return artifact_scope_dir(scope, project_root=project_root) / _validate_non_empty_segment(
        run_name,
        label="run_name",
    )


def run_logs_root(workflow: str, attempt_name: str, *, project_root: Path | None = None) -> Path:
    root = (project_root or PROJECT_ROOT).resolve() / "artifacts" / "run_logs"
    return root / _validate_non_empty_segment(workflow, label="workflow") / _validate_non_empty_segment(
        attempt_name,
        label="attempt_name",
    )


def derive_track_from_config_path(
    config_path: str | Path,
    *,
    project_root: Path | None = None,
    default_track: str | None = None,
) -> str:
    path = _normalize_path_like(config_path)
    root = (project_root or PROJECT_ROOT).resolve()
    try:
        relative_path = path.relative_to(root)
    except ValueError:
        if default_track is not None:
            normalize_track(default_track)
            return default_track
        raise RuntimeError(f"Config path must live under repo root: {path}")
    if not relative_path.parts or relative_path.parts[0] != "configs":
        if default_track is not None:
            normalize_track(default_track)
            return default_track
        raise RuntimeError(f"Config path must live under `configs/`: {path}")
    for index, part in enumerate(relative_path.parts[1:-1], start=1):
        if part not in ALLOWED_TRACK_ROOTS:
            continue
        if part == "experiments":
            track_parts = relative_path.parts[index:-1]
            normalize_track("/".join(track_parts))
            return "/".join(track_parts)
        normalize_track(part)
        return part
    if default_track is not None:
        normalize_track(default_track)
        return default_track
    raise RuntimeError(f"Could not derive artifact track from config path: {path}")


def scope_from_config_path(
    config_path: str | Path,
    *,
    kind: str,
    stage: str,
    component: str,
    project_root: Path | None = None,
    default_track: str | None = None,
) -> ArtifactScope:
    return ArtifactScope(
        kind=kind,
        stage=stage,
        component=component,
        track=derive_track_from_config_path(
            config_path,
            project_root=project_root,
            default_track=default_track,
        ),
    )


def scope_from_owner_json_path(
    path_value: str | Path,
    *,
    kind: str,
    stage: str,
    component: str,
    target_kind: str | None = None,
    target_stage: str | None = None,
    target_component: str | None = None,
    project_root: Path | None = None,
) -> ArtifactScope:
    path = validate_generic_artifact_path(path_value, project_root=project_root)
    root = (project_root or PROJECT_ROOT).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Artifact owner path must live under repo root: {path}") from exc
    if len(relative.parts) < 6 or relative.parts[0] != "artifacts":
        raise RuntimeError(f"Artifact owner path must live under `artifacts/`: {path}")

    actual_kind, actual_stage, actual_component = relative.parts[1:4]
    if (actual_kind, actual_stage, actual_component) != (kind, stage, component):
        raise RuntimeError(
            "Artifact owner path scope does not match the expected source scope.\n"
            f"expected=({kind}, {stage}, {component})\n"
            f"actual=({actual_kind}, {actual_stage}, {actual_component})\n"
            f"path={path}"
        )

    source_scope = ArtifactScope(
        kind=actual_kind,
        stage=actual_stage,
        component=actual_component,
        track="/".join(relative.parts[4:-1]),
    )
    validate_owner_json_path(path, scope=source_scope, project_root=project_root)
    return ArtifactScope(
        kind=target_kind or source_scope.kind,
        stage=target_stage or source_scope.stage,
        component=target_component or source_scope.component,
        track=source_scope.track,
    )


def is_artifact_path(path_value: str | Path, *, project_root: Path | None = None) -> bool:
    path = _normalize_path_like(path_value)
    root = (project_root or PROJECT_ROOT).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    return bool(relative.parts) and relative.parts[0] == "artifacts"


def validate_generic_artifact_path(path_value: str | Path, *, project_root: Path | None = None) -> Path:
    path = _normalize_path_like(path_value)
    root = (project_root or PROJECT_ROOT).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError:
        return path
    if not relative.parts or relative.parts[0] != "artifacts":
        return path
    if len(relative.parts) < 2:
        raise RuntimeError(f"Artifact path must not point to the artifact root directly: {path}")
    kind = relative.parts[1]
    if kind not in ALLOWED_ARTIFACT_KINDS:
        choices = ", ".join(sorted(ALLOWED_ARTIFACT_KINDS))
        raise RuntimeError(f"Artifact path kind must be one of: {choices}. Got: {path}")
    if kind == "run_logs":
        if len(relative.parts) < 4:
            raise RuntimeError(
                "Run-log paths must live under `artifacts/run_logs/<workflow>/<attempt_name>/...`."
            )
        _validate_non_empty_segment(relative.parts[2], label="workflow")
        _validate_non_empty_segment(relative.parts[3], label="attempt_name")
        return path
    if len(relative.parts) < 6:
        raise RuntimeError(
            "Artifact paths must live under `artifacts/<kind>/<stage>/<component>/<track>/...`."
        )
    _validate_non_empty_segment(relative.parts[2], label="stage")
    _validate_non_empty_segment(relative.parts[3], label="component")
    track_root = relative.parts[4]
    if track_root == "experiments":
        if len(relative.parts) < 7:
            raise RuntimeError(
                "Artifact experiment paths must live under "
                "`artifacts/<kind>/<stage>/<component>/experiments/<name>/...`."
            )
        normalize_track("/".join(relative.parts[4:6]))
    else:
        normalize_track(track_root)
    return path


def validate_artifact_config_values(
    config_args: dict[str, Any],
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    for value in config_args.values():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    validate_generic_artifact_path(item, project_root=project_root)
        elif isinstance(value, str):
            validate_generic_artifact_path(value, project_root=project_root)
    return config_args


def validate_owner_json_path(
    path_value: str | Path,
    *,
    scope: ArtifactScope,
    project_root: Path | None = None,
) -> Path:
    path = validate_generic_artifact_path(path_value, project_root=project_root)
    if path.suffix != ".json":
        raise RuntimeError(f"Artifact owner path must end with `.json`: {path}")
    expected_dir = artifact_scope_dir(scope, project_root=project_root)
    if path.parent != expected_dir:
        raise RuntimeError(
            "Artifact owner JSON must live directly under the canonical scope directory.\n"
            f"expected_dir={expected_dir}\n"
            f"actual_path={path}"
        )
    return path


def resolve_owner_json_path(
    path_value: str | Path,
    *,
    scope: ArtifactScope,
    run_name: str | None = None,
    project_root: Path | None = None,
) -> Path:
    if str(path_value):
        return validate_owner_json_path(path_value, scope=scope, project_root=project_root)
    if run_name is None:
        raise RuntimeError("Artifact owner JSON path is required when `run_name` is not provided.")
    return owner_json_path(scope, run_name, project_root=project_root)


def reporting_paths_for_output(
    output_json: str | Path,
    *,
    include_per_sample_jsonl: bool,
) -> ArtifactReportingPaths:
    output_path = _normalize_path_like(output_json)
    return ArtifactReportingPaths(
        output_json=output_path,
        progress_json=output_path.with_name(f"{output_path.stem}.progress.json"),
        per_sample_jsonl=(
            output_path.with_name(f"{output_path.stem}.per_sample.jsonl")
            if include_per_sample_jsonl
            else None
        ),
        manifest_json=output_path.with_name(f"{output_path.stem}.manifest.json"),
        plots_dir=output_path.parent / f"{output_path.stem}_plots",
        output_mcap=output_path.with_suffix(".mcap"),
    )


def resolve_expected_artifact_path(
    path_value: str | Path,
    *,
    expected_path: Path,
) -> Path:
    resolved = validate_generic_artifact_path(path_value)
    expected = expected_path.resolve()
    if resolved != expected:
        raise RuntimeError(f"Artifact path must match the canonical location.\nexpected={expected}\nactual={resolved}")
    return expected


def resolve_expected_artifact_dir(
    path_value: str | Path,
    *,
    expected_dir: Path,
) -> Path:
    resolved = validate_generic_artifact_path(path_value)
    expected = expected_dir.resolve()
    if resolved != expected:
        raise RuntimeError(
            f"Artifact directory must match the canonical location.\nexpected={expected}\nactual={resolved}"
        )
    return expected


def resolve_bundle_dir(
    path_value: str | Path,
    *,
    scope: ArtifactScope,
    run_name: str,
    project_root: Path | None = None,
) -> Path:
    expected = bundle_dir(scope, run_name, project_root=project_root).resolve()
    if str(path_value):
        return resolve_expected_artifact_dir(path_value, expected_dir=expected)
    return expected


def apply_reporting_artifact_policy(
    args,
    *,
    scope: ArtifactScope,
    include_per_sample_jsonl: bool,
    allow_default_output_json: bool = False,
    default_run_name: str | None = None,
    include_output_mcap: bool = False,
) -> ArtifactReportingPaths:
    output_value = str(getattr(args, "output_json", ""))
    if not output_value and not allow_default_output_json:
        raise RuntimeError("`output_json` must be defined before artifact policy validation.")
    output_json = resolve_owner_json_path(
        output_value,
        scope=scope,
        run_name=default_run_name if allow_default_output_json else None,
    )
    paths = reporting_paths_for_output(output_json, include_per_sample_jsonl=include_per_sample_jsonl)
    args.output_json = str(paths.output_json)

    if hasattr(args, "progress_json"):
        progress_value = str(getattr(args, "progress_json", ""))
        args.progress_json = str(
            resolve_expected_artifact_path(progress_value, expected_path=paths.progress_json)
            if progress_value
            else paths.progress_json
        )
    if include_per_sample_jsonl and hasattr(args, "per_sample_jsonl"):
        per_sample_value = str(getattr(args, "per_sample_jsonl", ""))
        args.per_sample_jsonl = str(
            resolve_expected_artifact_path(
                per_sample_value,
                expected_path=paths.per_sample_jsonl,
            )
            if per_sample_value
            else paths.per_sample_jsonl
        )
    if include_output_mcap and hasattr(args, "output_mcap") and str(getattr(args, "output_mcap", "")):
        args.output_mcap = str(
            resolve_expected_artifact_path(str(getattr(args, "output_mcap", "")), expected_path=paths.output_mcap)
        )
    return paths


def apply_visualization_artifact_policy(
    args,
    *,
    scope: ArtifactScope,
) -> ArtifactReportingPaths:
    summary_json = validate_owner_json_path(str(getattr(args, "summary_json", "")), scope=scope)
    paths = reporting_paths_for_output(summary_json, include_per_sample_jsonl=True)
    args.summary_json = str(summary_json)
    args.per_sample_jsonl = str(
        resolve_expected_artifact_path(str(getattr(args, "per_sample_jsonl", "")), expected_path=paths.per_sample_jsonl)
    )
    output_dir_value = str(getattr(args, "output_dir", ""))
    args.output_dir = str(
        resolve_expected_artifact_dir(output_dir_value, expected_dir=paths.plots_dir)
        if output_dir_value
        else paths.plots_dir
    )
    return paths
