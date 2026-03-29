"""Build train/holdout JSONL splits from curve-block analysis output."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "stage1" / "preprocess"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build train/holdout JSONL splits from curve-threshold block output."
    )
    parser.add_argument("--curve-json", type=str, required=True)
    parser.add_argument("--run-name-contains", type=str, default="perimeter_cw")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--holdout-stride", type=int, default=4)
    parser.add_argument("--holdout-offset", type=int, default=0)
    parser.add_argument("--subset-sizes", type=str, default="128,512,2048")
    parser.add_argument("--seed", type=int, default=7)
    return parser


def _slug(text: str) -> str:
    return text.replace("/", "_").replace(" ", "_")


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


def _parse_int_csv(csv: str) -> list[int]:
    values = [entry.strip() for entry in csv.split(",") if entry.strip()]
    if not values:
        raise RuntimeError("Expected one or more comma-separated integer subset sizes.")
    out = [int(value) for value in values]
    if any(value <= 0 for value in out):
        raise RuntimeError("Subset sizes must be > 0.")
    return out


def _load_curve_run(curve_json_path: Path, run_name_contains: str) -> dict:
    payload = json.loads(curve_json_path.read_text(encoding="utf-8"))
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise RuntimeError("Curve JSON is missing a non-empty `runs` list.")
    matches = [run for run in runs if run_name_contains in str(run.get("run_name", ""))]
    if not matches:
        raise RuntimeError(f"No run matched run-name-contains={run_name_contains!r}.")
    if len(matches) > 1:
        matched_names = "\n".join(str(run.get("run_name", "")) for run in matches)
        raise RuntimeError(
            "run-name-contains matched multiple runs. Make it more specific.\n" + matched_names
        )
    return matches[0]


def _materialize_record(record: dict, source_jsonl_path: Path) -> dict:
    out = dict(record)
    if "image_path" in out:
        out["image_path"] = str((source_jsonl_path.parent / str(out["image_path"])).resolve())
    return out


def _default_output_dir(curve_json_path: Path, run_name: str) -> Path:
    return (
        ARTIFACTS_ROOT
        / "curve_splits"
        / f"{curve_json_path.stem}__run-{_slug(run_name)}"
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.holdout_stride <= 0:
        raise RuntimeError("`holdout_stride` must be > 0.")
    if args.holdout_offset < 0 or args.holdout_offset >= args.holdout_stride:
        raise RuntimeError("`holdout_offset` must satisfy 0 <= offset < holdout_stride.")

    curve_json_path = Path(args.curve_json).resolve()
    if not curve_json_path.exists():
        raise RuntimeError(f"Curve JSON does not exist: {curve_json_path}")
    run = _load_curve_run(curve_json_path, args.run_name_contains)
    run_name = str(run["run_name"])
    source_jsonl_path = Path(str(run["jsonl_path"])).resolve()
    records = _read_jsonl(source_jsonl_path)
    if len(records) != int(run["num_samples"]):
        raise RuntimeError(
            "Curve JSON sample count does not match source JSONL.\n"
            f"curve_json_num_samples={run['num_samples']}\n"
            f"jsonl_records={len(records)}"
        )

    blocks = run["curve_blocks"]["blocks"]
    holdout_blocks = [block for idx, block in enumerate(blocks) if idx % args.holdout_stride == args.holdout_offset]
    holdout_indices: set[int] = set()
    selected_block_indices: list[int] = []
    for block in holdout_blocks:
        selected_block_indices.append(int(block["block_index"]))
        start_idx = int(block["start_sample_index"])
        end_idx = int(block["end_sample_index"])
        holdout_indices.update(range(start_idx, end_idx + 1))

    train_indices = [idx for idx in range(len(records)) if idx not in holdout_indices]
    holdout_index_list = sorted(holdout_indices)
    if not holdout_index_list:
        raise RuntimeError("Holdout selection produced no samples.")
    if not train_indices:
        raise RuntimeError("Holdout selection left no training samples.")

    subset_sizes = _parse_int_csv(args.subset_sizes)
    max_subset = max(subset_sizes)
    if max_subset > len(train_indices):
        raise RuntimeError(
            "Largest subset size exceeds available training samples.\n"
            f"largest_subset={max_subset}\n"
            f"available_train_samples={len(train_indices)}"
        )

    rng = random.Random(args.seed)
    shuffled_train_indices = list(train_indices)
    rng.shuffle(shuffled_train_indices)

    train_pool_records = [_materialize_record(records[idx], source_jsonl_path) for idx in train_indices]
    holdout_records = [_materialize_record(records[idx], source_jsonl_path) for idx in holdout_index_list]

    output_dir = Path(args.output_dir).resolve() if args.output_dir else _default_output_dir(curve_json_path, run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_pool_jsonl = output_dir / "train_pool.jsonl"
    holdout_jsonl = output_dir / "curve_holdout.jsonl"
    _write_jsonl(train_pool_jsonl, train_pool_records)
    _write_jsonl(holdout_jsonl, holdout_records)

    subset_payloads: dict[str, dict] = {}
    for subset_size in subset_sizes:
        subset_indices = sorted(shuffled_train_indices[:subset_size])
        subset_records = [_materialize_record(records[idx], source_jsonl_path) for idx in subset_indices]
        subset_jsonl = output_dir / f"train_{subset_size}.jsonl"
        _write_jsonl(subset_jsonl, subset_records)
        subset_payloads[str(subset_size)] = {
            "jsonl_path": str(subset_jsonl),
            "num_samples": subset_size,
            "sample_index_range": [int(subset_indices[0]), int(subset_indices[-1])],
            "sample_ids_preview": [
                str(subset_records[0]["sample_id"]),
                str(subset_records[-1]["sample_id"]),
            ],
        }

    manifest = {
        "curve_json": str(curve_json_path),
        "run_name": run_name,
        "source_jsonl": str(source_jsonl_path),
        "holdout_selection": {
            "block_count": len(holdout_blocks),
            "block_indices": selected_block_indices,
            "holdout_stride": int(args.holdout_stride),
            "holdout_offset": int(args.holdout_offset),
            "num_holdout_samples": len(holdout_index_list),
        },
        "train_pool": {
            "jsonl_path": str(train_pool_jsonl),
            "num_samples": len(train_indices),
        },
        "curve_holdout": {
            "jsonl_path": str(holdout_jsonl),
            "num_samples": len(holdout_index_list),
        },
        "subsets": subset_payloads,
    }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
