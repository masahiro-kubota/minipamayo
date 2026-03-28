"""Build canonical Stage 2 reasoning-SFT JSONL files from Stage 1 records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ....sequence.stage3_builder import build_reasoning_text, infer_driving_decision
from ....utils.json_config import load_json_payload, resolve_path_base

PROJECT_ROOT = Path(__file__).resolve().parents[5]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build `samples_reasoning_sft.jsonl` from canonical Stage 1 JSONL files."
    )
    parser.add_argument("--config-json", type=str, default="")
    return parser


def parse_args() -> tuple[argparse.Namespace, Path, dict]:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        return build_parser().parse_args(), Path(), {}

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config-json", type=str, required=True)
    pre_args, remaining = pre_parser.parse_known_args()
    if remaining:
        raise RuntimeError(
            "Reasoning JSONL builder accepts only --config-json. Put all settings in the JSON file."
        )

    config_path, payload = load_json_payload(pre_args.config_json)
    if not isinstance(payload, dict):
        raise RuntimeError("Config JSON must be an object.")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise RuntimeError("Config JSON must contain a non-empty `jobs` list.")
    base_dir = resolve_path_base(
        config_path,
        payload,
        default_base="project_root",
        base_dirs={"project_root": PROJECT_ROOT, "config_dir": config_path.parent},
    )
    parser = build_parser()
    args = parser.parse_args([f"--config-json={config_path}"])
    args.config_json = str(config_path)
    args.config_payload = payload
    args.base_dir = base_dir
    return args, config_path, payload


def _resolve_job_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _build_reasoning_record(record: dict) -> dict:
    command = str(record.get("command", "")).strip()
    planner_state = str(record.get("planner_state", "")).strip()
    if not command or not planner_state:
        raise RuntimeError(
            "Stage 2 reasoning JSONL builder requires `command` and `planner_state` in each record."
        )
    decision = infer_driving_decision(command, planner_state)
    enriched = dict(record)
    enriched["decision_longitudinal"] = decision["longitudinal"]
    enriched["decision_lateral"] = decision["lateral"]
    enriched["reasoning_text"] = build_reasoning_text(
        command=command,
        planner_state=planner_state,
        decision=decision,
    )
    return enriched


def _process_job(base_dir: Path, job: dict) -> dict:
    if not isinstance(job, dict):
        raise RuntimeError("Each job must be an object.")
    input_jsonl = job.get("input_jsonl")
    output_jsonl = job.get("output_jsonl")
    if not isinstance(input_jsonl, str) or not input_jsonl:
        raise RuntimeError("Each job requires non-empty `input_jsonl`.")
    if not isinstance(output_jsonl, str) or not output_jsonl:
        raise RuntimeError("Each job requires non-empty `output_jsonl`.")

    input_path = _resolve_job_path(base_dir, input_jsonl)
    output_path = _resolve_job_path(base_dir, output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_records = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            enriched = _build_reasoning_record(record)
            dst.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            num_records += 1

    return {
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "num_records": num_records,
    }


def main() -> None:
    args, _config_path, payload = parse_args()
    jobs = payload["jobs"]
    summaries = []
    for index, job in enumerate(jobs, start=1):
        summary = _process_job(args.base_dir, job)
        summaries.append(summary)
        print(
            json.dumps(
                {
                    "event": "reasoning_jsonl_built",
                    "job_index": index,
                    "total_jobs": len(jobs),
                    **summary,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )

    print(json.dumps({"jobs": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
