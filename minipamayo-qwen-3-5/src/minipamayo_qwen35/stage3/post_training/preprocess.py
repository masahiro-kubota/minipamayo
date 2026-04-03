"""Build a Stage 3 curation manifest from disagreement scores."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ...utils.artifact_paths import (
    ArtifactScope,
    artifact_scope_dir,
    resolve_expected_artifact_path,
    validate_generic_artifact_path,
)
from ...utils.jsonl import read_jsonl


def select_curated_records(
    records: list[dict],
    *,
    top_fraction: float,
    random_fraction: float,
    seed: int,
) -> list[dict]:
    if not 0.0 <= top_fraction <= 1.0:
        raise RuntimeError("`top_fraction` must be in [0, 1].")
    if not 0.0 <= random_fraction <= 1.0:
        raise RuntimeError("`random_fraction` must be in [0, 1].")
    if not records:
        return []

    sorted_records = sorted(
        records,
        key=lambda row: float(row.get("disagreement_score", 0.0)),
        reverse=True,
    )
    top_count = min(len(sorted_records), max(0, int(round(len(sorted_records) * top_fraction))))
    top_records = sorted_records[:top_count]
    remaining = sorted_records[top_count:]
    rng = random.Random(seed)
    random_count = min(len(remaining), max(0, int(round(len(sorted_records) * random_fraction))))
    random_records = rng.sample(remaining, random_count) if random_count > 0 else []
    selected = top_records + random_records
    selected.sort(key=lambda row: str(row["sample_id"]))
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Stage 3 curation manifest.")
    parser.add_argument("--scores-jsonl", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, default="")
    parser.add_argument("--top-fraction", type=float, default=0.3)
    parser.add_argument("--random-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def _float_slug(value: float) -> str:
    text = format(float(value), ".10g")
    return text.replace("-", "neg").replace(".", "p")


def _derived_run_name(*, scores_jsonl: str | Path, top_fraction: float, random_fraction: float, seed: int) -> str:
    stem = Path(str(scores_jsonl)).resolve().stem
    return (
        f"{stem}__top-{_float_slug(top_fraction)}"
        f"__rand-{_float_slug(random_fraction)}"
        f"__seed-{int(seed)}"
    )


def _scope_from_output_jsonl(path_value: str | Path) -> ArtifactScope:
    path = validate_generic_artifact_path(path_value)
    parts = path.parts
    try:
        artifacts_index = parts.index("artifacts")
    except ValueError as exc:
        raise RuntimeError(
            "Stage 3 preprocess output must live under "
            "`artifacts/preprocess/stage3/post_training/<track>/...`.\n"
            f"actual={path}"
        ) from exc
    try:
        kind = parts[artifacts_index + 1]
        stage = parts[artifacts_index + 2]
        component = parts[artifacts_index + 3]
        track = "/".join(parts[artifacts_index + 4 : -1])
    except IndexError as exc:
        raise RuntimeError(f"Stage 3 preprocess output must live under artifact scope: {path}") from exc
    if (kind, stage, component) != ("preprocess", "stage3", "post_training"):
        raise RuntimeError(
            "Stage 3 preprocess output must live under "
            "`artifacts/preprocess/stage3/post_training/<track>/...`.\n"
            f"actual={path}"
        )
    return ArtifactScope(kind=kind, stage=stage, component=component, track=track)


def resolve_output_jsonl(
    output_jsonl: str,
    *,
    scores_jsonl: str | Path,
    top_fraction: float,
    random_fraction: float,
    seed: int,
) -> Path:
    run_name = _derived_run_name(
        scores_jsonl=scores_jsonl,
        top_fraction=top_fraction,
        random_fraction=random_fraction,
        seed=seed,
    )
    if output_jsonl:
        scope = _scope_from_output_jsonl(output_jsonl)
    else:
        scope = ArtifactScope(
            kind="preprocess",
            stage="stage3",
            component="post_training",
            track="canonical",
        )
    expected_path = artifact_scope_dir(scope) / f"{run_name}.jsonl"
    if output_jsonl:
        resolved = resolve_expected_artifact_path(output_jsonl, expected_path=expected_path)
    else:
        resolved = expected_path.resolve()
    if resolved.suffix != ".jsonl":
        raise RuntimeError(f"Stage 3 preprocess output must end with `.jsonl`: {resolved}")
    return resolved


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    args.output_jsonl = str(
        resolve_output_jsonl(
            args.output_jsonl,
            scores_jsonl=args.scores_jsonl,
            top_fraction=args.top_fraction,
            random_fraction=args.random_fraction,
            seed=args.seed,
        )
    )
    return args


def main() -> None:
    args = parse_args()
    input_path = Path(args.scores_jsonl).resolve()
    output_path = Path(args.output_jsonl).resolve()
    records = read_jsonl(input_path)
    selected = select_curated_records(
        records,
        top_fraction=args.top_fraction,
        random_fraction=args.random_fraction,
        seed=args.seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in selected:
            payload = {
                "sample_id": str(record["sample_id"]),
                "weight": float(record.get("weight", 1.0)),
                "disagreement_score": float(record.get("disagreement_score", 0.0)),
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


__all__ = [
    "build_parser",
    "main",
    "parse_args",
    "resolve_output_jsonl",
    "select_curated_records",
]


if __name__ == "__main__":
    main()
