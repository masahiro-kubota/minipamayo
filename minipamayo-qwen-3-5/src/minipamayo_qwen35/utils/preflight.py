"""Shared preflight checks for repo entrypoints."""

from __future__ import annotations

import csv
import os
import subprocess
from io import StringIO
from pathlib import Path
from typing import Any

GPU_OTHER_USED_WARN_MIB = 1024
GPU_FREE_WARN_MIB = 4096


def resolve_git_repo_root(cwd: str | Path | None = None) -> Path:
    base_dir = Path(cwd or Path.cwd()).resolve()
    try:
        result = subprocess.run(
            ["git", "-C", str(base_dir), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("This command requires git, but the `git` executable is not available.") from exc
    if result.returncode != 0:
        stderr = result.stderr.strip() or "unknown git error"
        raise RuntimeError(
            f"This command requires running inside a git repository, but git could not resolve the repo root from {base_dir}: {stderr}"
        )
    return Path(result.stdout.strip())


def require_clean_git_worktree(cwd: str | Path | None = None) -> Path:
    repo_root = resolve_git_repo_root(cwd)
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("This command requires git, but the `git` executable is not available.") from exc
    if result.returncode != 0:
        stderr = result.stderr.strip() or "unknown git error"
        raise RuntimeError(f"This command requires a clean git worktree, but git status failed in {repo_root}: {stderr}")
    dirty_entries = result.stdout.strip()
    if dirty_entries:
        raise RuntimeError(
            "This command requires a clean git worktree with no staged, unstaged, or untracked files. "
            f"Commit, stash, or remove changes before running.\nDirty entries:\n{dirty_entries}"
        )
    return repo_root


def init_required_online_wandb(
    *,
    project: str,
    config: dict[str, Any],
    entity: str = "",
    name: str = "",
):
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("Training requires wandb to be installed and available for online logging.") from exc

    init_kwargs = {
        "project": project,
        "config": config,
        "mode": "online",
    }
    if entity:
        init_kwargs["entity"] = entity
    if name:
        init_kwargs["name"] = name

    try:
        run = wandb.init(**init_kwargs)
    except Exception as exc:
        raise RuntimeError(
            "Training requires W&B online mode. Confirm network access and `wandb login` before running."
        ) from exc

    run_mode = getattr(getattr(run, "settings", None), "mode", None)
    if run_mode != "online":
        if run is not None:
            run.finish()
        raise RuntimeError(f"Training requires W&B online mode, but wandb started in {run_mode!r} mode.")
    return run


def enforce_training_prerequisites(
    *,
    project: str,
    config: dict[str, Any],
    entity: str = "",
    name: str = "",
    git_cwd: str | Path | None = None,
):
    require_clean_git_worktree(git_cwd)
    return init_required_online_wandb(
        project=project,
        config=config,
        entity=entity,
        name=name,
    )


def _run_nvidia_smi_query(query_target: str, fields: list[str]) -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-{query_target}={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("`nvidia-smi` is not available.") from exc
    if result.returncode != 0:
        stderr = result.stderr.strip() or "unknown nvidia-smi error"
        raise RuntimeError(f"`nvidia-smi` query for {query_target} failed: {stderr}")

    rows: list[dict[str, str]] = []
    for row in csv.reader(StringIO(result.stdout)):
        if not row:
            continue
        if len(row) != len(fields):
            raise RuntimeError(f"Unexpected `nvidia-smi` output for {query_target}: {row!r}")
        rows.append({field: value.strip() for field, value in zip(fields, row)})
    return rows


def collect_gpu_preflight_snapshot(
    *,
    gpu_index: int,
    other_used_warn_mib: int = GPU_OTHER_USED_WARN_MIB,
    free_warn_mib: int = GPU_FREE_WARN_MIB,
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "query_ok": False,
        "gpu_index": gpu_index,
        "warning_reasons": [],
    }
    try:
        gpu_rows = _run_nvidia_smi_query(
            "gpu",
            ["index", "uuid", "name", "memory.total", "memory.used", "memory.free"],
        )
        gpu_row = next((row for row in gpu_rows if int(row["index"]) == gpu_index), None)
        if gpu_row is None:
            raise RuntimeError(f"GPU index {gpu_index} was not found in `nvidia-smi` output.")

        app_rows = _run_nvidia_smi_query(
            "compute-apps",
            ["gpu_uuid", "pid", "process_name", "used_memory"],
        )
        current_pid = os.getpid()
        non_self_compute_processes = []
        for row in app_rows:
            if row["gpu_uuid"] != gpu_row["uuid"]:
                continue
            pid = int(row["pid"])
            if pid == current_pid:
                continue
            used_memory_mib = int(row["used_memory"])
            non_self_compute_processes.append(
                {
                    "pid": pid,
                    "process_name": row["process_name"],
                    "used_memory_mib": used_memory_mib,
                }
            )

        total_mib = int(gpu_row["memory.total"])
        used_mib = int(gpu_row["memory.used"])
        free_mib = int(gpu_row["memory.free"])
        non_self_compute_used_mib = sum(proc["used_memory_mib"] for proc in non_self_compute_processes)
        other_used_mib = max(0, used_mib - non_self_compute_used_mib)

        warning_reasons: list[str] = []
        if non_self_compute_processes:
            warning_reasons.append(
                f"Detected {len(non_self_compute_processes)} other compute process(es) using {non_self_compute_used_mib} MiB."
            )
        if other_used_mib >= other_used_warn_mib:
            warning_reasons.append(
                f"Detected {other_used_mib} MiB already in use outside non-self compute processes."
            )
        if free_mib < free_warn_mib:
            warning_reasons.append(f"Only {free_mib} MiB free before model load.")

        snapshot.update(
            {
                "query_ok": True,
                "gpu_name": gpu_row["name"],
                "total_mib": total_mib,
                "used_mib": used_mib,
                "free_mib": free_mib,
                "non_self_compute_used_mib": non_self_compute_used_mib,
                "other_used_mib": other_used_mib,
                "non_self_compute_processes": non_self_compute_processes,
                "warning_reasons": warning_reasons,
            }
        )
    except Exception as exc:
        snapshot["query_error"] = str(exc)
        snapshot["warning_reasons"] = [f"Could not inspect GPU state before training: {exc}"]
    return snapshot
