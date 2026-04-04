"""Materialize curve-block-only train pools from curve-threshold analysis output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ...utils.artifact_paths import bundle_dir, resolve_bundle_dir, scope_from_owner_json_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build included/excluded curve-block train pools from curve-threshold output."
    )
    parser.add_argument("--curve-json", type=str, required=True)
    parser.add_argument("--exclude-split-manifest", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    return parser


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _materialize_record(record: dict, source_jsonl_path: Path) -> dict:
    out = dict(record)
    if "image_path" in out:
        out["image_path"] = str((source_jsonl_path.parent / str(out["image_path"])).resolve())
    return out


def _scope_for_curve_json(curve_json_path: Path):
    return scope_from_owner_json_path(
        curve_json_path,
        kind="preprocess",
        stage="stage1",
        component="curve_thresholds",
        target_component="curve_train_pools",
    )


def _default_output_dir(curve_json_path: Path) -> Path:
    return bundle_dir(_scope_for_curve_json(curve_json_path), curve_json_path.stem)


def _resolve_output_dir(curve_json_path: Path, output_dir: str) -> Path:
    default_output_dir = _default_output_dir(curve_json_path)
    return resolve_bundle_dir(
        output_dir,
        scope=_scope_for_curve_json(curve_json_path),
        run_name=default_output_dir.name,
    )


def _sha256_lines(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_curve_payload(curve_json_path: Path) -> dict:
    payload = json.loads(curve_json_path.read_text(encoding="utf-8"))
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise RuntimeError("Curve JSON is missing a non-empty `runs` list.")
    return payload


def _load_exclude_manifest(path: Path | None) -> dict | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "run_name" not in payload or "source_jsonl" not in payload:
        raise RuntimeError("Exclude split manifest is missing `run_name` or `source_jsonl`.")
    holdout = payload.get("holdout_selection")
    curve_holdout = payload.get("curve_holdout")
    if not isinstance(holdout, dict) or not isinstance(curve_holdout, dict):
        raise RuntimeError("Exclude split manifest is missing holdout metadata.")
    block_indices = holdout.get("block_indices")
    if not isinstance(block_indices, list):
        raise RuntimeError("Exclude split manifest is missing `holdout_selection.block_indices`.")
    if "num_samples" not in curve_holdout:
        raise RuntimeError("Exclude split manifest is missing `curve_holdout.num_samples`.")
    return payload


def _block_index_set(block: dict, *, total_records: int) -> set[int]:
    start_idx = int(block["start_sample_index"])
    end_idx = int(block["end_sample_index"])
    if start_idx < 0 or end_idx < start_idx or end_idx >= total_records:
        raise RuntimeError(
            "Curve block indices are invalid.\n"
            f"start_sample_index={start_idx}\n"
            f"end_sample_index={end_idx}\n"
            f"total_records={total_records}"
        )
    index_set = set(range(start_idx, end_idx + 1))
    expected_count = int(block.get("num_samples", len(index_set)))
    if len(index_set) != expected_count:
        raise RuntimeError(
            "Curve block sample count does not match the declared `num_samples`.\n"
            f"block_index={block.get('block_index')}\n"
            f"expected={expected_count}\n"
            f"actual={len(index_set)}"
        )
    return index_set


def _collect_curve_block_indices(run: dict, *, total_records: int) -> set[int]:
    curve_blocks = run.get("curve_blocks")
    if not isinstance(curve_blocks, dict):
        raise RuntimeError(f"Run is missing `curve_blocks`: {run.get('run_name')}")
    blocks = curve_blocks.get("blocks")
    if not isinstance(blocks, list):
        raise RuntimeError(f"Run is missing `curve_blocks.blocks`: {run.get('run_name')}")
    indices: set[int] = set()
    for block in blocks:
        indices.update(_block_index_set(block, total_records=total_records))
    declared_count = int(curve_blocks.get("block_sample_count", len(indices)))
    if len(indices) != declared_count:
        raise RuntimeError(
            "Unioned curve block samples do not match `block_sample_count`.\n"
            f"run_name={run.get('run_name')}\n"
            f"expected={declared_count}\n"
            f"actual={len(indices)}"
        )
    return indices


def _collect_holdout_indices(run: dict, *, total_records: int, block_indices: set[int]) -> set[int]:
    curve_blocks = run["curve_blocks"]
    blocks = curve_blocks["blocks"]
    known_block_indices = {int(block["block_index"]) for block in blocks}
    missing = sorted(block_indices - known_block_indices)
    if missing:
        raise RuntimeError(
            "Exclude split manifest refers to unknown curve block indices.\n"
            f"run_name={run.get('run_name')}\n"
            f"missing_block_indices={missing}"
        )
    indices: set[int] = set()
    for block in blocks:
        if int(block["block_index"]) not in block_indices:
            continue
        indices.update(_block_index_set(block, total_records=total_records))
    return indices


def build_curve_block_train_pools(
    *,
    curve_json_path: Path,
    output_dir: Path,
    exclude_split_manifest_path: Path | None = None,
) -> dict:
    payload = _load_curve_payload(curve_json_path)
    exclude_manifest = _load_exclude_manifest(exclude_split_manifest_path)
    exclude_run_name = str(exclude_manifest["run_name"]) if exclude_manifest is not None else ""
    exclude_source_jsonl = (
        Path(str(exclude_manifest["source_jsonl"])).resolve() if exclude_manifest is not None else None
    )
    exclude_block_indices = (
        {int(value) for value in exclude_manifest["holdout_selection"]["block_indices"]}
        if exclude_manifest is not None
        else set()
    )

    included_records: list[dict] = []
    excluded_records: list[dict] = []
    excluded_holdout_sample_ids: list[str] = []
    matched_exclude_manifest = exclude_manifest is None
    run_summaries: list[dict] = []

    for run in payload["runs"]:
        run_name = str(run["run_name"])
        source_jsonl_path = Path(str(run["jsonl_path"])).resolve()
        records = _read_jsonl(source_jsonl_path)
        if len(records) != int(run["num_samples"]):
            raise RuntimeError(
                "Curve JSON sample count does not match source JSONL.\n"
                f"run_name={run_name}\n"
                f"curve_json_num_samples={run['num_samples']}\n"
                f"jsonl_records={len(records)}"
            )

        included_indices = _collect_curve_block_indices(run, total_records=len(records))
        included_records.extend(
            _materialize_record(records[idx], source_jsonl_path) for idx in sorted(included_indices)
        )

        run_excluded_holdout_indices: set[int] = set()
        run_excluded_block_indices: list[int] = []
        if exclude_manifest is not None and (
            run_name == exclude_run_name or source_jsonl_path == exclude_source_jsonl
        ):
            matched_exclude_manifest = True
            run_excluded_holdout_indices = _collect_holdout_indices(
                run,
                total_records=len(records),
                block_indices=exclude_block_indices,
            )
            run_excluded_block_indices = sorted(exclude_block_indices)
            excluded_holdout_sample_ids.extend(
                str(records[idx]["sample_id"]) for idx in sorted(run_excluded_holdout_indices)
            )

        excluded_indices = included_indices - run_excluded_holdout_indices
        excluded_records.extend(
            _materialize_record(records[idx], source_jsonl_path) for idx in sorted(excluded_indices)
        )

        run_summaries.append(
            {
                "run_name": run_name,
                "jsonl_path": str(source_jsonl_path),
                "num_samples": len(records),
                "num_curve_blocks": int(run["curve_blocks"]["num_blocks"]),
                "included_curve_block_samples": len(included_indices),
                "excluded_curve_block_samples": len(excluded_indices),
                "excluded_holdout_samples": len(run_excluded_holdout_indices),
                "excluded_holdout_block_indices": run_excluded_block_indices,
            }
        )

    if exclude_manifest is not None and not matched_exclude_manifest:
        raise RuntimeError(
            "Exclude split manifest did not match any run in the curve JSON.\n"
            f"run_name={exclude_run_name}\n"
            f"source_jsonl={exclude_source_jsonl}"
        )

    expected_excluded_holdout = (
        int(exclude_manifest["curve_holdout"]["num_samples"]) if exclude_manifest is not None else 0
    )
    if len(excluded_holdout_sample_ids) != expected_excluded_holdout:
        raise RuntimeError(
            "Excluded holdout sample count does not match the split manifest.\n"
            f"expected={expected_excluded_holdout}\n"
            f"actual={len(excluded_holdout_sample_ids)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    included_jsonl = output_dir / "curve_block_train_pool_included.jsonl"
    excluded_jsonl = output_dir / "curve_block_train_pool_excluded.jsonl"
    manifest_path = output_dir / "curve_block_train_pool.manifest.json"
    _write_jsonl(included_jsonl, included_records)
    _write_jsonl(excluded_jsonl, excluded_records)

    manifest = {
        "curve_json": str(curve_json_path),
        "exclude_split_manifest": (
            str(exclude_split_manifest_path) if exclude_split_manifest_path is not None else ""
        ),
        "curve_block_config": payload.get("curve_block_config"),
        "runs": run_summaries,
        "included_pool": {
            "jsonl_path": str(included_jsonl),
            "num_samples": len(included_records),
        },
        "excluded_pool": {
            "jsonl_path": str(excluded_jsonl),
            "num_samples": len(excluded_records),
        },
        "excluded_holdout": {
            "run_name": exclude_run_name,
            "num_samples": len(excluded_holdout_sample_ids),
            "sample_id_sha256": _sha256_lines(excluded_holdout_sample_ids),
            "block_indices": sorted(exclude_block_indices),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    args = build_parser().parse_args()
    curve_json_path = Path(args.curve_json).resolve()
    if not curve_json_path.exists():
        raise RuntimeError(f"Curve JSON does not exist: {curve_json_path}")
    exclude_split_manifest_path = None
    if args.exclude_split_manifest:
        exclude_split_manifest_path = Path(args.exclude_split_manifest).resolve()
        if not exclude_split_manifest_path.exists():
            raise RuntimeError(
                f"Exclude split manifest does not exist: {exclude_split_manifest_path}"
            )

    output_dir = _resolve_output_dir(curve_json_path, args.output_dir)
    manifest = build_curve_block_train_pools(
        curve_json_path=curve_json_path,
        output_dir=output_dir,
        exclude_split_manifest_path=exclude_split_manifest_path,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
