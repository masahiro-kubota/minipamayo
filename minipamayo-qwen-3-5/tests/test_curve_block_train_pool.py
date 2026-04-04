from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from minipamayo_qwen35.stage1.preprocess.build_curve_block_train_pool import (
    build_curve_block_train_pools,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_records(prefix: str, count: int) -> list[dict]:
    return [
        {
            "sample_id": f"{prefix}_{idx:03d}",
            "image_path": f"images/{idx:03d}.png",
            "command": "lanefollow",
        }
        for idx in range(count)
    ]


class CurveBlockTrainPoolTest(unittest.TestCase):
    def test_build_curve_block_train_pools_excludes_holdout_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            run1_jsonl = tmp_path / "run1" / "samples.jsonl"
            run2_jsonl = tmp_path / "run2" / "samples.jsonl"
            run3_jsonl = tmp_path / "run3" / "samples.jsonl"
            _write_jsonl(run1_jsonl, _build_records("ccw", 5))
            _write_jsonl(run2_jsonl, _build_records("cw", 4))
            _write_jsonl(run3_jsonl, _build_records("per", 6))

            curve_json_path = tmp_path / "curve.json"
            curve_json_path.write_text(
                json.dumps(
                    {
                        "curve_block_config": {
                            "anchor_mode": "or",
                            "block_kappa_threshold": 0.08,
                            "block_yaw_threshold": 0.5,
                            "block_pre_seconds": 1.0,
                            "block_post_seconds": 2.0,
                        },
                        "runs": [
                            {
                                "run_name": "run_ccw",
                                "jsonl_path": str(run1_jsonl),
                                "num_samples": 5,
                                "curve_blocks": {
                                    "num_blocks": 2,
                                    "block_sample_count": 4,
                                    "blocks": [
                                        {
                                            "block_index": 0,
                                            "start_sample_index": 0,
                                            "end_sample_index": 1,
                                            "num_samples": 2,
                                        },
                                        {
                                            "block_index": 1,
                                            "start_sample_index": 3,
                                            "end_sample_index": 4,
                                            "num_samples": 2,
                                        },
                                    ],
                                },
                            },
                            {
                                "run_name": "run_cw",
                                "jsonl_path": str(run2_jsonl),
                                "num_samples": 4,
                                "curve_blocks": {
                                    "num_blocks": 1,
                                    "block_sample_count": 2,
                                    "blocks": [
                                        {
                                            "block_index": 0,
                                            "start_sample_index": 1,
                                            "end_sample_index": 2,
                                            "num_samples": 2,
                                        }
                                    ],
                                },
                            },
                            {
                                "run_name": "run_perimeter",
                                "jsonl_path": str(run3_jsonl),
                                "num_samples": 6,
                                "curve_blocks": {
                                    "num_blocks": 2,
                                    "block_sample_count": 4,
                                    "blocks": [
                                        {
                                            "block_index": 0,
                                            "start_sample_index": 0,
                                            "end_sample_index": 1,
                                            "num_samples": 2,
                                        },
                                        {
                                            "block_index": 1,
                                            "start_sample_index": 3,
                                            "end_sample_index": 4,
                                            "num_samples": 2,
                                        },
                                    ],
                                },
                            },
                        ],
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            split_manifest_path = tmp_path / "split_manifest.json"
            split_manifest_path.write_text(
                json.dumps(
                    {
                        "run_name": "run_perimeter",
                        "source_jsonl": str(run3_jsonl),
                        "holdout_selection": {
                            "block_indices": [0],
                        },
                        "curve_holdout": {
                            "num_samples": 2,
                        },
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            output_dir = tmp_path / "output"
            manifest = build_curve_block_train_pools(
                curve_json_path=curve_json_path,
                exclude_split_manifest_path=split_manifest_path,
                output_dir=output_dir,
            )

            self.assertEqual(manifest["included_pool"]["num_samples"], 10)
            self.assertEqual(manifest["excluded_pool"]["num_samples"], 8)
            self.assertEqual(manifest["excluded_holdout"]["num_samples"], 2)

            excluded_records = [
                json.loads(line)
                for line in (output_dir / "curve_block_train_pool_excluded.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            excluded_sample_ids = {record["sample_id"] for record in excluded_records}
            self.assertNotIn("per_000", excluded_sample_ids)
            self.assertNotIn("per_001", excluded_sample_ids)
            self.assertIn("per_003", excluded_sample_ids)
            self.assertIn("per_004", excluded_sample_ids)

            included_records = [
                json.loads(line)
                for line in (output_dir / "curve_block_train_pool_included.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            included_sample_ids = {record["sample_id"] for record in included_records}
            self.assertIn("per_000", included_sample_ids)
            self.assertIn("per_001", included_sample_ids)

            saved_manifest = json.loads(
                (output_dir / "curve_block_train_pool.manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_manifest["excluded_holdout"]["num_samples"], 2)
            self.assertEqual(saved_manifest["runs"][2]["excluded_holdout_samples"], 2)
