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

from minipamayo_qwen35.inspector.backfill import backfill_artifact_manifests
from minipamayo_qwen35.inspector.manifests import (
    load_manifest,
    update_manifest_plots,
    upsert_manifest,
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


class ManifestTests(unittest.TestCase):
    def test_upsert_manifest_omits_empty_optional_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            summary_path = root / "artifacts/eval/stage1/vlm_ce/canonical/run.json"
            _write_json(summary_path, {"checkpoint": "checkpoint.pt", "test_jsonl": "test.jsonl"})

            manifest = upsert_manifest(
                artifact_kind="eval",
                stage="stage1a_eval",
                run_name="run",
                summary_json=summary_path,
                checkpoint=str(root / "checkpoint.pt"),
                dataset_path=str(root / "test.jsonl"),
            )
            payload = json.loads(manifest.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["artifact_kind"], "eval")
            self.assertNotIn("progress_json", payload)
            self.assertNotIn("plots", payload)

            plots_dir = root / "artifacts/eval/stage1/vlm_ce/canonical/run_plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            (plots_dir / "metric_histograms.png").write_bytes(b"png")
            (plots_dir / "worst_samples.json").write_text('{"rows": []}', encoding="utf-8")
            updated = update_manifest_plots(
                summary_json=summary_path,
                plots_dir=plots_dir,
                plots={
                    "metric_histograms": str(plots_dir / "metric_histograms.png"),
                    "worst_samples": str(plots_dir / "worst_samples.json"),
                },
                wandb_run_url="https://wandb.example/run",
            )
            reloaded = load_manifest(updated.manifest_path)
            self.assertEqual(reloaded.wandb_run_url, "https://wandb.example/run")
            self.assertIn("metric_histograms", reloaded.plots)

    def test_backfill_reconstructs_existing_eval_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_root = Path(tmp_dir) / "artifacts"
            summary_path = artifact_root / "eval/stage1/expert_cfm/canonical/run.json"
            progress_path = summary_path.with_name("run.progress.json")
            per_sample_path = summary_path.with_name("run.per_sample.jsonl")
            plots_dir = summary_path.parent / "run_plots"
            _write_json(
                summary_path,
                {
                    "checkpoint": "/tmp/checkpoints/stage1b.pt",
                    "eval_jsonl": ["/tmp/eval.jsonl"],
                    "num_samples": 1,
                },
            )
            _write_json(progress_path, {"state": "completed"})
            _write_jsonl(
                per_sample_path,
                [
                    {
                        "sample_id": "sample-001",
                        "sample_index": 0,
                        "image_path": "/tmp/frame.jpeg",
                    }
                ],
            )
            plots_dir.mkdir(parents=True, exist_ok=True)
            (plots_dir / "metric_histograms.png").write_bytes(b"png")
            _write_json(
                plots_dir / "visualization_manifest.json",
                {"wandb": {"run_url": "https://wandb.example/visualizer"}},
            )

            written = backfill_artifact_manifests(artifact_root)
            self.assertEqual(len(written), 1)
            manifest = load_manifest(written[0])
            self.assertEqual(manifest.stage, "stage1b_eval")
            self.assertEqual(manifest.artifact_kind, "eval")
            self.assertEqual(manifest.wandb_run_url, "https://wandb.example/visualizer")
            self.assertTrue(manifest.per_sample_jsonl)
            self.assertIn("metric_histograms", manifest.plots)


if __name__ == "__main__":
    unittest.main()
