from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from minipamayo_qwen35.stage2.reasoning_sft import batch_inference, eval as stage2_eval


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class _FakeWandbRun:
    def __init__(self) -> None:
        self.summary: dict = {}
        self.url = "https://wandb.example/fake"

    def log(self, *_args, **_kwargs) -> None:
        return None

    def finish(self) -> None:
        return None


class _DummyReasoningDataset:
    def __init__(self, _jsonl, max_samples: int = 0) -> None:
        self.samples = [
            {
                "sample_id": "sample-001",
                "image_path": "/tmp/001.jpeg",
                "reasoning_text": "reason-1",
                "gt_waypoints": torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
                "command": "right",
                "planner_state": "curve",
                "decision_longitudinal": "go",
                "decision_lateral": "turn",
            },
            {
                "sample_id": "sample-002",
                "image_path": "/tmp/002.jpeg",
                "reasoning_text": "reason-2",
                "gt_waypoints": torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
                "command": "left",
                "planner_state": "curve",
                "decision_longitudinal": "go",
                "decision_lateral": "turn",
            },
        ]
        if max_samples > 0:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


class Stage2OutputTests(unittest.TestCase):
    def test_stage2_eval_writes_per_sample_outputs_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "stage2_eval_config.json"
            _write_json(
                config_path,
                {
                    "path_base": "config_dir",
                    "args": {
                        "checkpoint": "stage2.pt",
                        "eval_jsonl": "eval.jsonl",
                        "output_json": "stage2_eval.json",
                        "progress_json": "stage2_eval.progress.json",
                        "per_sample_jsonl": "stage2_eval.per_sample.jsonl",
                        "device": "cuda",
                        "batch_size": 2,
                        "num_workers": 0,
                        "max_samples": 0,
                        "image_min_pixels": 163840,
                        "image_max_pixels": 196608,
                        "progress_every_samples": 1,
                        "progress_every_seconds": 1,
                        "wandb_project": "test-project",
                        "wandb_run_name": "stage2-eval-test",
                    },
                },
            )
            batch = {
                "sample_id": ["sample-001", "sample-002"],
                "image_path": ["/tmp/001.jpeg", "/tmp/002.jpeg"],
                "command": ["right", "left"],
                "planner_state": ["curve", "curve"],
                "decision_longitudinal": ["go", "go"],
                "decision_lateral": ["turn", "turn"],
                "reasoning_text": ["gt-1", "gt-2"],
            }

            def fake_evaluate_stage2(*, progress_callback, sample_callback, **_kwargs):
                result = {
                    "per_sample_loss": torch.tensor([0.4, 0.6], dtype=torch.float32),
                    "per_sample_correct": torch.tensor([3, 2], dtype=torch.long),
                    "per_sample_total": torch.tensor([4, 4], dtype=torch.long),
                }
                sample_callback(0, batch, result)
                progress_callback(2, {"loss": 0.5, "token_accuracy": 0.625})
                return {"loss": 0.5, "token_accuracy": 0.625}

            with (
                mock.patch(
                    "minipamayo_qwen35.utils.eval_reporting.init_required_online_wandb",
                    return_value=_FakeWandbRun(),
                ),
                mock.patch.object(stage2_eval, "require_stage2_cuda_device", return_value=torch.device("cpu")),
                mock.patch(
                    "minipamayo_qwen35.models.checkpoint_loader.load_stage2_checkpoint_bundle",
                    return_value={
                        "checkpoint": {"stage2_metadata": {"name": "dummy"}},
                        "checkpoint_args": {
                            "stage1a_checkpoint": str(root / "stage1a.pt"),
                            "handoff_loss_weight": 8.0,
                        },
                        "stage1_checkpoint": {"stage1_metadata": {"name": "dummy"}},
                        "model": SimpleNamespace(train=lambda: None, eval=lambda: None),
                        "processor": SimpleNamespace(),
                        "history_registry": SimpleNamespace(),
                        "history_quantizer": SimpleNamespace(token_count=8),
                        "model_dtype": torch.float32,
                    },
                ),
                mock.patch(
                    "minipamayo_qwen35.stage2.reasoning_sft.dataset.ReasoningSftJsonlDataset",
                    _DummyReasoningDataset,
                ),
                mock.patch(
                    "minipamayo_qwen35.stage2.reasoning_sft.dataset.build_reasoning_sft_dataloader",
                    return_value=[batch],
                ),
                mock.patch(
                    "minipamayo_qwen35.stage2.reasoning_sft.runtime.evaluate_stage2",
                    side_effect=fake_evaluate_stage2,
                ),
                mock.patch.object(stage2_eval, "collect_dataset_view_fingerprint", return_value={"rows": 2}),
                mock.patch.object(stage2_eval, "collect_processor_settings", return_value={"processor": "dummy"}),
                mock.patch.object(sys, "argv", ["stage2_eval.py", "--config-json", str(config_path)]),
            ):
                stage2_eval.main()

            summary_path = root / "stage2_eval.json"
            per_sample_path = root / "stage2_eval.per_sample.jsonl"
            manifest_path = root / "stage2_eval.manifest.json"
            self.assertTrue(summary_path.exists())
            self.assertTrue(per_sample_path.exists())
            self.assertTrue(manifest_path.exists())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rows = [json.loads(line) for line in per_sample_path.read_text(encoding="utf-8").splitlines() if line]
            self.assertEqual(len(rows), 2)
            aggregate_accuracy = sum(row["metrics"]["teacher_forced_correct_tokens"] for row in rows) / sum(
                row["metrics"]["teacher_forced_total_tokens"] for row in rows
            )
            self.assertAlmostEqual(summary["metrics"]["token_accuracy"], aggregate_accuracy)

    def test_stage2_batch_inference_writes_summary_progress_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "batch_inference_config.json"
            _write_json(
                config_path,
                {
                    "path_base": "config_dir",
                    "args": {
                        "checkpoint": "stage2.pt",
                        "stage1b_checkpoint": "stage1b.pt",
                        "input_jsonl": ["eval.jsonl"],
                        "output_json": "batch_inference.json",
                        "progress_json": "batch_inference.progress.json",
                        "per_sample_jsonl": "batch_inference.per_sample.jsonl",
                        "device": "cuda",
                        "max_samples": 0,
                        "image_min_pixels": 163840,
                        "image_max_pixels": 196608,
                        "max_reasoning_tokens": 64,
                        "flow_steps": 10,
                        "temperature": 0.6,
                        "top_p": 0.98,
                        "top_k": 0,
                        "progress_every_samples": 1,
                        "progress_every_seconds": 1,
                        "wandb_project": "test-project",
                        "wandb_run_name": "stage2-batch-test",
                    },
                },
            )

            def fake_payload(*, sample: dict, sample_index: int, **_kwargs):
                return {
                    "sample_id": sample["sample_id"],
                    "sample_index": sample_index,
                    "image_path": sample["image_path"],
                    "command": sample["command"],
                    "reasoning": {"text": f"pred-{sample['sample_id']}"},
                    "prediction": {"waypoints": [[0.0, 0.0], [1.0 + sample_index, 1.0]]},
                    "ground_truth": {
                        "waypoints": [[0.0, 0.0], [1.0, 1.0]],
                        "reasoning_text": sample["reasoning_text"],
                    },
                    "metrics": {"ade_m": 0.5 + sample_index, "fde_m": 1.0 + sample_index},
                }

            with (
                mock.patch(
                    "minipamayo_qwen35.utils.eval_reporting.init_required_online_wandb",
                    return_value=_FakeWandbRun(),
                ),
                mock.patch.object(batch_inference, "require_stage2_cuda_device", return_value=torch.device("cpu")),
                mock.patch.object(batch_inference, "ReasoningSftJsonlDataset", _DummyReasoningDataset),
                mock.patch(
                    "minipamayo_qwen35.models.checkpoint_loader.load_stage2_inference_bundle",
                    return_value={"bundle": "dummy"},
                ),
                mock.patch.object(batch_inference, "build_stage2_inference_payload", side_effect=fake_payload),
                mock.patch.object(sys, "argv", ["batch_inference.py", "--config-json", str(config_path)]),
            ):
                batch_inference.main()

            summary_path = root / "batch_inference.json"
            progress_path = root / "batch_inference.progress.json"
            per_sample_path = root / "batch_inference.per_sample.jsonl"
            manifest_path = root / "batch_inference.manifest.json"
            self.assertTrue(summary_path.exists())
            self.assertTrue(progress_path.exists())
            self.assertTrue(per_sample_path.exists())
            self.assertTrue(manifest_path.exists())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rows = [json.loads(line) for line in per_sample_path.read_text(encoding="utf-8").splitlines() if line]
            self.assertEqual(summary["num_samples"], 2)
            self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
