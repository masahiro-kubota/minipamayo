"""Shared preflight checks for repo entrypoints."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


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
