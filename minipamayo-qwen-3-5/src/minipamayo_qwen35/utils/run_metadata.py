"""Run metadata helpers for train/eval entrypoints."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Subset

from .preflight import resolve_git_repo_root

if TYPE_CHECKING:
    from collections.abc import Iterable


def _run_git(args: list[str], cwd: str | Path | None = None) -> str:
    repo_root = resolve_git_repo_root(cwd)
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or "unknown git error"
        raise RuntimeError(f"git {' '.join(args)} failed in {repo_root}: {stderr}")
    return result.stdout.strip()


def collect_git_metadata(cwd: str | Path | None = None) -> dict[str, Any]:
    repo_root = resolve_git_repo_root(cwd)
    branch = _run_git(["branch", "--show-current"], repo_root)
    commit = _run_git(["rev-parse", "HEAD"], repo_root)
    short_commit = _run_git(["rev-parse", "--short", "HEAD"], repo_root)
    return {
        "repo_root": str(repo_root),
        "branch": branch or None,
        "commit": commit,
        "short_commit": short_commit,
    }


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_non_empty_lines(path: str | Path) -> int:
    count = 0
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _sha256_int_sequence(values: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(f"{int(value)}\n".encode())
    return digest.hexdigest()


def _jsonl_source_fingerprint(jsonl_path: str | Path) -> dict[str, Any]:
    path = Path(jsonl_path).resolve()
    if not path.exists():
        raise RuntimeError(f"Dataset JSONL does not exist: {path}")

    extract_summary_path = path.parent / "extract_summary.json"
    image_dir = path.parent / "images"
    if not image_dir.exists():
        raise RuntimeError(f"Dataset image directory does not exist: {image_dir}")
    if not extract_summary_path.exists():
        raise RuntimeError(f"Dataset extract summary does not exist: {extract_summary_path}")

    image_files = sorted(child for child in image_dir.iterdir() if child.is_file())

    fingerprint = {
        "jsonl_path": str(path),
        "jsonl_sha256": _sha256_file(path),
        "jsonl_size_bytes": path.stat().st_size,
        "num_records": _count_non_empty_lines(path),
        "dataset_root": str(path.parent),
        "image_dir": str(image_dir),
        "image_file_count": len(image_files),
        "extract_summary_path": str(extract_summary_path),
        "extract_summary_sha256": _sha256_file(extract_summary_path),
        "extract_summary_size_bytes": extract_summary_path.stat().st_size,
    }
    return fingerprint


def _dataset_source_fingerprints(dataset) -> dict[str, Any]:
    if hasattr(dataset, "jsonl_paths"):
        jsonl_paths = dataset.jsonl_paths
        if not isinstance(jsonl_paths, list) or not jsonl_paths:
            raise RuntimeError("Dataset `jsonl_paths` must be a non-empty list.")
        source_fingerprints = [_jsonl_source_fingerprint(path) for path in jsonl_paths]
        if len(source_fingerprints) == 1:
            return {
                "num_sources": 1,
                "source": source_fingerprints[0],
            }
        return {
            "num_sources": len(source_fingerprints),
            "sources": source_fingerprints,
        }

    if hasattr(dataset, "jsonl_path"):
        return {
            "num_sources": 1,
            "source": _jsonl_source_fingerprint(dataset.jsonl_path),
        }

    raise RuntimeError("Dataset does not expose canonical JSONL source paths for fingerprinting.")


def collect_dataset_view_fingerprint(dataset) -> dict[str, Any]:
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        indices = list(dataset.indices)
        return {
            "view_type": "subset",
            "selected_count": len(dataset),
            "indices_sha256": _sha256_int_sequence(indices),
            **_dataset_source_fingerprints(base_dataset),
        }

    return {
        "view_type": "dataset",
        "selected_count": len(dataset),
        **_dataset_source_fingerprints(dataset),
    }


def collect_gpu_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version()
        if torch.backends.cudnn.is_available()
        else None,
    }
    if device.type != "cuda":
        return info

    gpu_index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(gpu_index)
    info.update(
        {
            "gpu_index": gpu_index,
            "gpu_name": props.name,
            "gpu_total_memory_mib": int(props.total_memory / (1024**2)),
            "gpu_multiprocessor_count": props.multi_processor_count,
            "gpu_compute_capability": f"{props.major}.{props.minor}",
            "bf16_supported": torch.cuda.is_bf16_supported(),
        }
    )
    return info


def collect_processor_settings(
    processor,
    *,
    requested_min_pixels: int | None = None,
    requested_max_pixels: int | None = None,
) -> dict[str, Any]:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise RuntimeError("Processor is missing the canonical `image_processor` component.")

    size = getattr(image_processor, "size", None)
    if size is not None:
        size = dict(size)

    return {
        "processor_class": processor.__class__.__name__,
        "image_processor_class": image_processor.__class__.__name__,
        "requested_min_pixels": requested_min_pixels,
        "requested_max_pixels": requested_max_pixels,
        "min_pixels": getattr(image_processor, "min_pixels", None),
        "max_pixels": getattr(image_processor, "max_pixels", None),
        "size": size,
        "patch_size": getattr(image_processor, "patch_size", None),
        "temporal_patch_size": getattr(image_processor, "temporal_patch_size", None),
        "merge_size": getattr(image_processor, "merge_size", None),
    }
