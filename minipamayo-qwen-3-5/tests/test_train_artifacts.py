from __future__ import annotations

import json
from pathlib import Path

from minipamayo_qwen35.utils.train_artifacts import (
    resolve_train_artifact_paths,
    write_history_json,
    write_run_config_json,
    write_summary_json,
)


def test_resolve_train_artifact_paths_returns_standard_filenames(tmp_path: Path) -> None:
    paths = resolve_train_artifact_paths(tmp_path / "checkpoints" / "stage2" / "run", create_dir=False)

    assert paths.run_config_json == paths.save_dir / "run_config.json"
    assert paths.history_json == paths.save_dir / "history.json"
    assert paths.summary_json == paths.save_dir / "summary.json"
    assert paths.best_pt == paths.save_dir / "best.pt"
    assert paths.last_pt == paths.save_dir / "last.pt"
    assert paths.final_pt == paths.save_dir / "final.pt"
    assert paths.checkpoint_path("best_handoff.pt") == paths.save_dir / "best_handoff.pt"


def test_train_artifact_json_writers_are_atomic_and_leave_no_tmp_files(tmp_path: Path) -> None:
    paths = resolve_train_artifact_paths(tmp_path / "checkpoints" / "stage3" / "run")

    write_run_config_json(
        paths,
        config_json="configs/stage3/post_training/canonical/run.json",
        config_payload={"args": {"lr": 1e-5}},
        resolved_args={"lr": 1e-5},
        run_metadata={"git": {"commit": "abc123"}},
    )
    write_history_json(paths, [{"epoch": 1, "reward": 0.5}])
    write_summary_json(paths, {"best_reward": 0.5})

    assert json.loads(paths.run_config_json.read_text(encoding="utf-8")) == {
        "config_json": "configs/stage3/post_training/canonical/run.json",
        "config_payload": {"args": {"lr": 1e-5}},
        "resolved_args": {"lr": 1e-5},
        "run_metadata": {"git": {"commit": "abc123"}},
    }
    assert json.loads(paths.history_json.read_text(encoding="utf-8")) == [{"epoch": 1, "reward": 0.5}]
    assert json.loads(paths.summary_json.read_text(encoding="utf-8")) == {"best_reward": 0.5}
    assert list(paths.save_dir.glob(".*.tmp")) == []
