from __future__ import annotations

import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from minipamayo_qwen35.ops.ignore_rule_paths import IgnoreRulePaths, build_backup_path
from minipamayo_qwen35.ops.ignore_rule_run import (
    build_parser,
    enforce_eval_prerequisites,
    enforce_full_start_prerequisites,
    require_exact_line_count,
)


class IgnoreRuleOpsTests(unittest.TestCase):
    def test_ignore_rule_paths_match_expected_completion_defaults(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = IgnoreRulePaths.for_attempt(
                "completion_ignore_rule_full_001",
                "ignore-rule-completion-001",
                project_root=root,
            )

            self.assertEqual(
                paths.log_root,
                root / "artifacts" / "run_logs" / "completion_ignore_rule_full_001",
            )
            self.assertEqual(
                paths.stage1a_train_config,
                root
                / "configs"
                / "stage1"
                / "vlm_ce"
                / "train"
                / "canonical"
                / "ignore_rule_data_k64_dt01_completion_001_12gb.json",
            )
            self.assertEqual(
                paths.stage1b_eval_progress,
                root
                / "artifacts"
                / "eval"
                / "stage1"
                / "expert_cfm"
                / "canonical"
                / "ignore_rule_data_k64_dt01_completion_001_curve_eval_safe.progress.json",
            )
            self.assertEqual(paths.train_preprocess_expected_counts, (6423, 4676, 6167))
            self.assertEqual(paths.curve_preprocess_expected_counts, (569,))

    def test_build_backup_path_appends_expected_suffix(self) -> None:
        original = Path("/tmp/example.json")
        backup = build_backup_path(original, tag="20260404_010203")
        self.assertEqual(str(backup), "/tmp/example.json_bak_20260404_010203")

    def test_parser_rejects_invalid_stage_values(self) -> None:
        parser = build_parser()
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["full", "--start-stage", "bad"])
            with self.assertRaises(SystemExit):
                parser.parse_args(["eval", "--target-stage", "bad"])

    def test_require_exact_line_count_validates(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            path.write_text('{"a":1}\n{"a":2}\n', encoding="utf-8")
            require_exact_line_count(path, 2)
            with self.assertRaisesRegex(RuntimeError, "Unexpected line count"):
                require_exact_line_count(path, 3)

    def test_stage1b_start_requires_stage1a_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = IgnoreRulePaths.for_attempt("attempt", "session", project_root=Path(tmp))
            with self.assertRaisesRegex(RuntimeError, "Stage1A start-stage prerequisites"):
                enforce_full_start_prerequisites(paths, start_stage="stage1b")

    def test_stage2_eval_requires_stage2_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            paths = IgnoreRulePaths.for_attempt("attempt", "session", project_root=Path(tmp))
            with self.assertRaisesRegex(RuntimeError, "Stage2 eval prerequisites"):
                enforce_eval_prerequisites(
                    paths,
                    target_stage="stage2",
                    run_stage2_sample_inference=True,
                )


if __name__ == "__main__":
    unittest.main()
