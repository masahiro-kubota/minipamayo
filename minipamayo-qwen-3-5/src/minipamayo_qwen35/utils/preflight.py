"""Shared preflight checks for repo entrypoints."""

from __future__ import annotations

import csv
import os
import re
import subprocess
from io import StringIO
from pathlib import Path
from typing import Any

GPU_OTHER_USED_WARN_MIB = 1024
GPU_FREE_WARN_MIB = 4096
EXPECTED_CUDA_TOOLKIT_VERSION = "12.8"
EXPECTED_CUDA_HOME = Path("/usr/local/cuda-12.8")
EXPECTED_CUDA_SYMLINK = Path("/usr/local/cuda")


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
    require_expected_cuda_toolkit()
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


def _parse_nvcc_release(stdout: str) -> str | None:
    match = re.search(r"release\s+(\d+\.\d+)", stdout)
    if match is None:
        return None
    return match.group(1)


def collect_cuda_toolkit_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "query_ok": False,
        "expected_toolkit_version": EXPECTED_CUDA_TOOLKIT_VERSION,
        "expected_cuda_home": str(EXPECTED_CUDA_HOME),
    }
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        snapshot["error"] = f"`nvcc` is not available: {exc}"
        return snapshot
    if result.returncode != 0:
        stderr = result.stderr.strip() or "unknown nvcc error"
        snapshot["error"] = f"`nvcc --version` failed: {stderr}"
        return snapshot

    nvcc_release = _parse_nvcc_release(result.stdout)
    cuda_home_env = os.environ.get("CUDA_HOME", "")
    symlink_target = ""
    if EXPECTED_CUDA_SYMLINK.exists():
        symlink_target = str(EXPECTED_CUDA_SYMLINK.resolve())

    torch_cuda = ""
    try:
        import torch

        torch_cuda = str(torch.version.cuda or "")
    except Exception:
        torch_cuda = ""

    snapshot.update(
        {
            "query_ok": True,
            "nvcc_release": nvcc_release,
            "cuda_home_env": cuda_home_env,
            "cuda_symlink_target": symlink_target,
            "torch_cuda": torch_cuda,
            "nvcc_output": result.stdout.strip(),
        }
    )
    return snapshot


def require_expected_cuda_toolkit() -> dict[str, Any]:
    snapshot = collect_cuda_toolkit_snapshot()
    if not snapshot["query_ok"]:
        raise RuntimeError(
            "Canonical runtime requires CUDA toolkit 12.8, but CUDA preflight failed.\n"
            f"{snapshot.get('error', 'unknown CUDA preflight error')}"
        )

    problems: list[str] = []
    nvcc_release = snapshot.get("nvcc_release")
    if nvcc_release != EXPECTED_CUDA_TOOLKIT_VERSION:
        problems.append(
            f"`nvcc --version` reported {nvcc_release!r}, expected {EXPECTED_CUDA_TOOLKIT_VERSION!r}."
        )

    cuda_home_env = snapshot.get("cuda_home_env", "")
    if cuda_home_env != str(EXPECTED_CUDA_HOME):
        problems.append(
            f"`CUDA_HOME` is {cuda_home_env!r}, expected {str(EXPECTED_CUDA_HOME)!r}."
        )

    cuda_symlink_target = snapshot.get("cuda_symlink_target", "")
    if cuda_symlink_target and cuda_symlink_target != str(EXPECTED_CUDA_HOME):
        problems.append(
            f"`/usr/local/cuda` resolves to {cuda_symlink_target!r}, expected {str(EXPECTED_CUDA_HOME)!r}."
        )

    torch_cuda = snapshot.get("torch_cuda", "")
    if torch_cuda and torch_cuda != EXPECTED_CUDA_TOOLKIT_VERSION:
        problems.append(
            f"`torch.version.cuda` is {torch_cuda!r}, expected {EXPECTED_CUDA_TOOLKIT_VERSION!r}."
        )

    if problems:
        raise RuntimeError(
            "Canonical runtime requires CUDA toolkit 12.8 alignment for flash-attn.\n"
            + "\n".join(problems)
            + "\nFix shell exports so PATH / LD_LIBRARY_PATH / CUDA_HOME point to /usr/local/cuda-12.8."
        )
    return snapshot
