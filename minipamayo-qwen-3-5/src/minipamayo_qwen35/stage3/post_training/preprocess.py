"""Build a Stage 3 curation manifest from disagreement scores."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

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
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--top-fraction", type=float, default=0.3)
    parser.add_argument("--random-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
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


__all__ = ["build_parser", "main", "select_curated_records"]


if __name__ == "__main__":
    main()
