"""Shared helpers for canonical training artifact files."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainArtifactPaths:
    save_dir: Path
    run_config_json: Path
    history_json: Path
    summary_json: Path
    best_pt: Path
    last_pt: Path
    final_pt: Path

    def checkpoint_path(self, filename: str) -> Path:
        return self.save_dir / filename


def resolve_train_artifact_paths(save_dir: str | Path, *, create_dir: bool = True) -> TrainArtifactPaths:
    root = Path(save_dir).resolve()
    if create_dir:
        root.mkdir(parents=True, exist_ok=True)
    return TrainArtifactPaths(
        save_dir=root,
        run_config_json=root / "run_config.json",
        history_json=root / "history.json",
        summary_json=root / "summary.json",
        best_pt=root / "best.pt",
        last_pt=root / "last.pt",
        final_pt=root / "final.pt",
    )


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        Path(tmp_name).replace(path)
    finally:
        tmp_path = Path(tmp_name)
        if tmp_path.exists():
            tmp_path.unlink()


def _resolve_json_path(paths_or_path: TrainArtifactPaths | str | Path, attr_name: str) -> Path:
    if isinstance(paths_or_path, TrainArtifactPaths):
        return getattr(paths_or_path, attr_name)
    return Path(paths_or_path).resolve()


def write_history_json(paths_or_path: TrainArtifactPaths | str | Path, payload: Any) -> None:
    _atomic_write_json(_resolve_json_path(paths_or_path, "history_json"), payload)


def write_summary_json(paths_or_path: TrainArtifactPaths | str | Path, payload: Any) -> None:
    _atomic_write_json(_resolve_json_path(paths_or_path, "summary_json"), payload)


def write_run_config_json(
    paths_or_path: TrainArtifactPaths | str | Path,
    *,
    config_json: str,
    config_payload: dict,
    resolved_args: dict[str, Any],
    run_metadata: dict,
) -> None:
    _atomic_write_json(
        _resolve_json_path(paths_or_path, "run_config_json"),
        {
            "config_json": config_json,
            "config_payload": config_payload,
            "resolved_args": resolved_args,
            "run_metadata": run_metadata,
        },
    )
