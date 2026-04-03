from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from minipamayo_qwen35.inspector.adapters import (
    load_stage1a_run,
    load_stage1b_run,
    load_stage2_eval_run,
    load_stage2_inference_run,
)
from minipamayo_qwen35.inspector.models import ArtifactManifest


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


class AdapterTests(unittest.TestCase):
    def test_stage1a_adapter_normalizes_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            summary_path = root / "stage1a.json"
            per_sample_path = root / "stage1a.per_sample.jsonl"
            _write_json(summary_path, {"num_samples": 1})
            _write_jsonl(
                per_sample_path,
                [
                    {
                        "sample_id": "sample-001",
                        "sample_index": 0,
                        "image_path": "/tmp/image.jpeg",
                        "command": "right",
                        "gt_waypoints": [[0.0, 0.0], [1.0, 1.0]],
                        "pred_waypoints": [[0.0, 0.0], [1.2, 1.1]],
                        "ade_m": 0.4,
                        "fde_m": 0.7,
                        "metrics": {
                            "autoregressive_token_accuracy": 0.75,
                            "action_mae_kappa": 0.02,
                        },
                    }
                ],
            )
            manifest = ArtifactManifest(
                artifact_kind="eval",
                stage="stage1a_eval",
                run_name="stage1a",
                summary_json=str(summary_path),
                per_sample_jsonl=str(per_sample_path),
            )
            run = load_stage1a_run(manifest)
            self.assertEqual(run.samples[0].sample_id, "sample-001")
            self.assertAlmostEqual(run.samples[0].token_accuracy or 0.0, 0.75)
            self.assertIsNone(run.invalid_reason)

    def test_stage1b_adapter_normalizes_pid_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            summary_path = root / "stage1b.json"
            per_sample_path = root / "stage1b.per_sample.jsonl"
            _write_json(summary_path, {"num_samples": 1})
            _write_jsonl(
                per_sample_path,
                [
                    {
                        "sample_id": "sample-001",
                        "sample_index": 0,
                        "image_path": "/tmp/image.jpeg",
                        "command": "left",
                        "gt_waypoints": [[0.0, 0.0], [1.0, 1.0]],
                        "pred_waypoints": [[0.0, 0.0], [1.5, 1.3]],
                        "ade_m": 0.5,
                        "fde_m": 0.8,
                        "max_lateral_error_m": 0.9,
                        "metrics": {"action_mae_kappa": 0.03},
                        "pid_override": {
                            "pred_waypoints": [[0.0, 0.0], [1.1, 1.0]],
                        },
                    }
                ],
            )
            manifest = ArtifactManifest(
                artifact_kind="eval",
                stage="stage1b_eval",
                run_name="stage1b",
                summary_json=str(summary_path),
                per_sample_jsonl=str(per_sample_path),
            )
            run = load_stage1b_run(manifest)
            self.assertEqual(run.samples[0].sample_id, "sample-001")
            self.assertEqual(run.samples[0].pid_pred_waypoints, [[0.0, 0.0], [1.1, 1.0]])

    def test_stage2_eval_adapter_normalizes_reasoning_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            summary_path = root / "stage2_eval.json"
            per_sample_path = root / "stage2_eval.per_sample.jsonl"
            _write_json(summary_path, {"num_samples": 1})
            _write_jsonl(
                per_sample_path,
                [
                    {
                        "sample_id": "sample-001",
                        "sample_index": 0,
                        "image_path": "/tmp/image.jpeg",
                        "command": "forward",
                        "reasoning_text": "ground truth reasoning",
                        "metrics": {"teacher_forced_token_accuracy": 0.6},
                    }
                ],
            )
            manifest = ArtifactManifest(
                artifact_kind="eval",
                stage="stage2_eval",
                run_name="stage2_eval",
                summary_json=str(summary_path),
                per_sample_jsonl=str(per_sample_path),
            )
            run = load_stage2_eval_run(manifest)
            self.assertEqual(run.samples[0].reasoning_text_gt, "ground truth reasoning")
            self.assertAlmostEqual(run.samples[0].token_accuracy or 0.0, 0.6)

    def test_stage2_inference_adapter_normalizes_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            summary_path = root / "stage2_inference.json"
            per_sample_path = root / "stage2_inference.per_sample.jsonl"
            _write_json(summary_path, {"num_samples": 1})
            _write_jsonl(
                per_sample_path,
                [
                    {
                        "sample_id": "sample-001",
                        "sample_index": 0,
                        "image_path": "/tmp/image.jpeg",
                        "command": "right",
                        "reasoning": {"text": "pred reasoning"},
                        "prediction": {"waypoints": [[0.0, 0.0], [2.0, 2.0]]},
                        "ground_truth": {
                            "waypoints": [[0.0, 0.0], [1.0, 1.0]],
                            "reasoning_text": "gt reasoning",
                        },
                        "metrics": {"ade_m": 0.7, "fde_m": 1.1},
                    }
                ],
            )
            manifest = ArtifactManifest(
                artifact_kind="inference",
                stage="stage2_inference",
                run_name="stage2_inference",
                summary_json=str(summary_path),
                per_sample_jsonl=str(per_sample_path),
            )
            run = load_stage2_inference_run(manifest)
            self.assertEqual(run.samples[0].reasoning_text_pred, "pred reasoning")
            self.assertEqual(run.samples[0].reasoning_text_gt, "gt reasoning")
            self.assertAlmostEqual(run.samples[0].fde_m or 0.0, 1.1)

    def test_duplicate_sample_ids_mark_run_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            summary_path = root / "dup.json"
            per_sample_path = root / "dup.per_sample.jsonl"
            _write_json(summary_path, {"num_samples": 2})
            _write_jsonl(
                per_sample_path,
                [
                    {
                        "sample_id": "sample-dup",
                        "sample_index": 0,
                        "image_path": "/tmp/a.jpeg",
                        "gt_waypoints": [],
                        "pred_waypoints": [],
                        "metrics": {},
                    },
                    {
                        "sample_id": "sample-dup",
                        "sample_index": 1,
                        "image_path": "/tmp/b.jpeg",
                        "gt_waypoints": [],
                        "pred_waypoints": [],
                        "metrics": {},
                    },
                ],
            )
            manifest = ArtifactManifest(
                artifact_kind="eval",
                stage="stage1a_eval",
                run_name="dup",
                summary_json=str(summary_path),
                per_sample_jsonl=str(per_sample_path),
            )
            run = load_stage1a_run(manifest)
            self.assertIsNotNone(run.invalid_reason)
            self.assertIn("sample-dup", run.invalid_reason or "")


if __name__ == "__main__":
    unittest.main()
