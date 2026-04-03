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

from minipamayo_qwen35.inspector.manifests import write_manifest
from minipamayo_qwen35.inspector.models import ArtifactManifest
from minipamayo_qwen35.inspector.registry import (
    compare_runs_by_sample_id,
    load_manifest_registry,
    load_normalized_run,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


class RegistryTests(unittest.TestCase):
    def test_registry_sorts_per_sample_manifests_ahead_of_summary_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "artifacts"

            summary_only_path = artifact_root / "eval/stage1/vlm_ce/canonical/a_summary_only.json"
            _write_json(summary_only_path, {"num_samples": 0})
            write_manifest(
                ArtifactManifest(
                    artifact_kind="eval",
                    stage="stage1a_eval",
                    run_name="summary-only",
                    summary_json=str(summary_only_path),
                )
            )

            rich_summary_path = artifact_root / "eval/stage1/vlm_ce/canonical/b_curve_eval.json"
            rich_samples_path = rich_summary_path.with_name("b_curve_eval.per_sample.jsonl")
            _write_json(rich_summary_path, {"num_samples": 1})
            _write_jsonl(
                rich_samples_path,
                [
                    {
                        "sample_id": "sample-001",
                        "sample_index": 0,
                        "image_path": "/tmp/001.jpeg",
                        "gt_waypoints": [[0.0, 0.0], [1.0, 1.0]],
                        "pred_waypoints": [[0.0, 0.0], [1.1, 1.1]],
                        "metrics": {"autoregressive_token_accuracy": 0.7, "action_mae_kappa": 0.02},
                    }
                ],
            )
            write_manifest(
                ArtifactManifest(
                    artifact_kind="eval",
                    stage="stage1a_eval",
                    run_name="curve-eval",
                    summary_json=str(rich_summary_path),
                    per_sample_jsonl=str(rich_samples_path),
                )
            )

            registry = load_manifest_registry(artifact_root)
            self.assertEqual(registry.stage1a_manifests[0].run_name, "curve-eval")
            rich_run = load_normalized_run(registry.stage1a_manifests[0])
            summary_only_run = load_normalized_run(registry.stage1a_manifests[1])
            self.assertIsNone(rich_run.browser_unavailable_reason)
            self.assertEqual(
                summary_only_run.browser_unavailable_reason,
                "Sample/Block view unavailable for summary-only artifact.",
            )

    def test_registry_discovers_and_compares_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "artifacts"

            stage1a_summary = artifact_root / "eval/stage1/vlm_ce/canonical/stage1a.json"
            stage1a_samples = stage1a_summary.with_name("stage1a.per_sample.jsonl")
            _write_json(stage1a_summary, {"num_samples": 1})
            _write_jsonl(
                stage1a_samples,
                [
                    {
                        "sample_id": "sample-001",
                        "sample_index": 0,
                        "image_path": "/tmp/001.jpeg",
                        "gt_waypoints": [[0.0, 0.0], [1.0, 1.0]],
                        "pred_waypoints": [[0.0, 0.0], [1.1, 1.1]],
                        "metrics": {"autoregressive_token_accuracy": 0.7, "action_mae_kappa": 0.02},
                    }
                ],
            )
            write_manifest(
                ArtifactManifest(
                    artifact_kind="eval",
                    stage="stage1a_eval",
                    run_name="stage1a",
                    summary_json=str(stage1a_summary),
                    per_sample_jsonl=str(stage1a_samples),
                )
            )

            stage2_eval_summary = artifact_root / "eval/stage2/reasoning_sft/canonical/stage2_eval.json"
            stage2_eval_samples = stage2_eval_summary.with_name("stage2_eval.per_sample.jsonl")
            _write_json(stage2_eval_summary, {"metrics": {"token_accuracy": 0.8}})
            _write_jsonl(
                stage2_eval_samples,
                [
                    {
                        "sample_id": "sample-001",
                        "sample_index": 0,
                        "image_path": "/tmp/001.jpeg",
                        "reasoning_text": "gt",
                        "metrics": {"teacher_forced_token_accuracy": 0.8},
                    }
                ],
            )
            write_manifest(
                ArtifactManifest(
                    artifact_kind="eval",
                    stage="stage2_eval",
                    run_name="stage2_eval",
                    summary_json=str(stage2_eval_summary),
                    per_sample_jsonl=str(stage2_eval_samples),
                )
            )

            stage2_inf_summary = artifact_root / "inference/stage2/reasoning_sft/canonical/stage2_inference.json"
            stage2_inf_samples = stage2_inf_summary.with_name("stage2_inference.per_sample.jsonl")
            _write_json(stage2_inf_summary, {"metrics": {"fde_m": 1.2}})
            _write_jsonl(
                stage2_inf_samples,
                [
                    {
                        "sample_id": "sample-001",
                        "sample_index": 0,
                        "image_path": "/tmp/001.jpeg",
                        "reasoning": {"text": "pred"},
                        "prediction": {"waypoints": [[0.0, 0.0], [2.0, 2.0]]},
                        "ground_truth": {"waypoints": [[0.0, 0.0], [1.0, 1.0]], "reasoning_text": "gt"},
                        "metrics": {"ade_m": 0.6, "fde_m": 1.2},
                    }
                ],
            )
            write_manifest(
                ArtifactManifest(
                    artifact_kind="inference",
                    stage="stage2_inference",
                    run_name="stage2_inference",
                    summary_json=str(stage2_inf_summary),
                    per_sample_jsonl=str(stage2_inf_samples),
                )
            )

            stage3_eval_summary = artifact_root / "eval/stage3/post_training/canonical/stage3_eval.json"
            stage3_eval_samples = stage3_eval_summary.with_name("stage3_eval.per_sample.jsonl")
            _write_json(stage3_eval_summary, {"metrics": {"reward": 0.4, "fde_m": 1.0}})
            _write_jsonl(
                stage3_eval_samples,
                [
                    {
                        "sample_id": "sample-001",
                        "sample_index": 0,
                        "image_path": "/tmp/001.jpeg",
                        "reasoning_text": "gt",
                        "reasoning_text_pred": "pred",
                        "gt_waypoints": [[0.0, 0.0], [1.0, 1.0]],
                        "pred_waypoints": [[0.0, 0.0], [1.2, 1.1]],
                        "metrics": {"ade_m": 0.5, "fde_m": 1.0},
                    }
                ],
            )
            write_manifest(
                ArtifactManifest(
                    artifact_kind="eval",
                    stage="stage3_eval",
                    run_name="stage3_eval",
                    summary_json=str(stage3_eval_summary),
                    per_sample_jsonl=str(stage3_eval_samples),
                )
            )

            registry = load_manifest_registry(artifact_root)
            self.assertEqual(len(registry.stage1a_manifests), 1)
            self.assertEqual(len(registry.stage2_eval_manifests), 1)
            self.assertEqual(len(registry.stage2_inference_manifests), 1)
            self.assertEqual(len(registry.stage3_eval_manifests), 1)

            stage1a_run = load_normalized_run(registry.stage1a_manifests[0])
            stage2_eval_run = load_normalized_run(registry.stage2_eval_manifests[0])
            stage2_inference_run = load_normalized_run(registry.stage2_inference_manifests[0])
            stage3_eval_run = load_normalized_run(registry.stage3_eval_manifests[0])
            self.assertEqual(len(stage1a_run.groups), 1)
            self.assertEqual(stage3_eval_run.samples[0].reasoning_text_pred, "pred")
            rows = compare_runs_by_sample_id(
                stage1a_run=stage1a_run,
                stage2_eval_run=stage2_eval_run,
                stage2_inference_run=stage2_inference_run,
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sample_id"], "sample-001")
            self.assertIsNotNone(rows[0]["stage1a"])
            self.assertIsNotNone(rows[0]["stage2_eval"])
            self.assertIsNotNone(rows[0]["stage2_inference"])


if __name__ == "__main__":
    unittest.main()
