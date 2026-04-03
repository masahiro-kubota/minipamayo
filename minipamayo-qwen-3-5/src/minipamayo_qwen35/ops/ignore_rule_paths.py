"""Path helpers for the canonical ignore-rule completion workflow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CUDA_HOME = Path("/usr/local/cuda-12.8")
CUDA_BIN = CUDA_HOME / "bin"
CUDA_LIB64 = CUDA_HOME / "lib64"


def timestamp_iso() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def timestamp_tag() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def build_backup_path(path: Path, *, tag: str | None = None) -> Path:
    suffix = tag or timestamp_tag()
    return Path(f"{path}_bak_{suffix}")


@dataclass(frozen=True)
class IgnoreRulePaths:
    project_root: Path
    attempt_name: str
    session_name: str

    log_root: Path
    state_dir: Path
    master_log: Path
    run_status_file: Path
    run_exitcode_file: Path
    monitor_alert_file: Path

    stage1a_train_config: Path
    stage1a_eval_config: Path
    stage1b_train_config: Path
    stage1b_eval_config: Path
    stage2_preprocess_config: Path
    stage2_curve_preprocess_config: Path
    stage2_train_config: Path
    stage2_eval_config: Path
    stage2_infer_config: Path

    stage1a_save_dir: Path
    stage1b_save_dir: Path
    stage2_save_dir: Path

    stage1a_summary: Path
    stage1a_best: Path
    stage1a_final: Path
    stage1b_summary: Path
    stage1b_best: Path
    stage1b_last: Path
    stage2_summary: Path
    stage2_best: Path

    stage1a_eval_output: Path
    stage1a_eval_progress: Path
    stage1a_eval_per_sample: Path
    stage1b_eval_output: Path
    stage1b_eval_progress: Path
    stage2_eval_output: Path
    stage2_eval_progress: Path
    stage2_infer_output: Path
    stage2_infer_progress: Path

    train_preprocess_outputs: tuple[Path, ...]
    train_preprocess_expected_counts: tuple[int, ...]
    curve_preprocess_outputs: tuple[Path, ...]
    curve_preprocess_expected_counts: tuple[int, ...]

    @classmethod
    def for_attempt(
        cls,
        attempt_name: str,
        session_name: str,
        *,
        project_root: Path | None = None,
    ) -> "IgnoreRulePaths":
        root = (project_root or REPO_ROOT).resolve()
        log_root = root / "artifacts" / "run_logs" / attempt_name
        state_dir = log_root / "state"

        stage1a_save_dir = root / "checkpoints" / "stage1" / "vlm_ce" / "canonical" / "ignore_rule_data_k64_dt01_completion_001_12gb"
        stage1b_save_dir = root / "checkpoints" / "stage1" / "expert_cfm" / "canonical" / "ignore_rule_data_k64_dt01_completion_001_12gb_safe"
        stage2_save_dir = root / "checkpoints" / "stage2" / "reasoning_sft" / "canonical" / "ignore_rule_data_k64_dt01_completion_001_12gb"

        return cls(
            project_root=root,
            attempt_name=attempt_name,
            session_name=session_name,
            log_root=log_root,
            state_dir=state_dir,
            master_log=log_root / "master.log",
            run_status_file=log_root / "run.status.json",
            run_exitcode_file=log_root / "run.exitcode",
            monitor_alert_file=log_root / "monitor.alert",
            stage1a_train_config=root / "configs" / "stage1" / "vlm_ce" / "train" / "canonical" / "ignore_rule_data_k64_dt01_completion_001_12gb.json",
            stage1a_eval_config=root / "configs" / "stage1" / "vlm_ce" / "eval" / "canonical" / "ignore_rule_data_k64_dt01_completion_001_curve_eval.json",
            stage1b_train_config=root / "configs" / "stage1" / "expert_cfm" / "train" / "canonical" / "ignore_rule_data_k64_dt01_completion_001_12gb_safe.json",
            stage1b_eval_config=root / "configs" / "stage1" / "expert_cfm" / "eval" / "canonical" / "ignore_rule_data_k64_dt01_completion_001_curve_eval_safe.json",
            stage2_preprocess_config=root / "configs" / "stage2" / "reasoning_sft" / "data" / "ignore_rule_data_k64_dt01_completion_001.json",
            stage2_curve_preprocess_config=root / "configs" / "stage2" / "reasoning_sft" / "data" / "ignore_rule_data_k64_dt01_completion_001_curve_eval.json",
            stage2_train_config=root / "configs" / "stage2" / "reasoning_sft" / "canonical" / "ignore_rule_data_k64_dt01_completion_001_12gb.json",
            stage2_eval_config=root / "configs" / "stage2" / "reasoning_sft" / "canonical" / "ignore_rule_data_k64_dt01_completion_001_curve_eval.json",
            stage2_infer_config=root / "configs" / "stage2" / "reasoning_sft" / "inference" / "canonical" / "ignore_rule_data_k64_dt01_completion_001_curve_sample_stage1b_safe.json",
            stage1a_save_dir=stage1a_save_dir,
            stage1b_save_dir=stage1b_save_dir,
            stage2_save_dir=stage2_save_dir,
            stage1a_summary=stage1a_save_dir / "summary.json",
            stage1a_best=stage1a_save_dir / "best.pt",
            stage1a_final=stage1a_save_dir / "final.pt",
            stage1b_summary=stage1b_save_dir / "summary.json",
            stage1b_best=stage1b_save_dir / "best.pt",
            stage1b_last=stage1b_save_dir / "last.pt",
            stage2_summary=stage2_save_dir / "summary.json",
            stage2_best=stage2_save_dir / "best.pt",
            stage1a_eval_output=root / "artifacts" / "eval" / "stage1" / "vlm_ce" / "canonical" / "ignore_rule_data_k64_dt01_completion_001_curve_eval.json",
            stage1a_eval_progress=root / "artifacts" / "eval" / "stage1" / "vlm_ce" / "canonical" / "ignore_rule_data_k64_dt01_completion_001_curve_eval.progress.json",
            stage1a_eval_per_sample=root / "artifacts" / "eval" / "stage1" / "vlm_ce" / "canonical" / "ignore_rule_data_k64_dt01_completion_001_curve_eval.per_sample.jsonl",
            stage1b_eval_output=root / "artifacts" / "eval" / "stage1" / "expert_cfm" / "canonical" / "ignore_rule_data_k64_dt01_completion_001_curve_eval_safe.json",
            stage1b_eval_progress=root / "artifacts" / "eval" / "stage1" / "expert_cfm" / "canonical" / "ignore_rule_data_k64_dt01_completion_001_curve_eval_safe.progress.json",
            stage2_eval_output=root / "artifacts" / "eval" / "stage2" / "reasoning_sft" / "canonical" / "ignore_rule_data_k64_dt01_completion_001_curve_eval.json",
            stage2_eval_progress=root / "artifacts" / "eval" / "stage2" / "reasoning_sft" / "canonical" / "ignore_rule_data_k64_dt01_completion_001_curve_eval.progress.json",
            stage2_infer_output=root / "artifacts" / "inference" / "stage2" / "reasoning_sft" / "canonical" / "ignore_rule_data_k64_dt01_completion_001_curve_sample_stage1b_safe.json",
            stage2_infer_progress=root / "artifacts" / "inference" / "stage2" / "reasoning_sft" / "canonical" / "ignore_rule_data_k64_dt01_completion_001_curve_sample_stage1b_safe.progress.json",
            train_preprocess_outputs=(
                root / "datasets" / "processed" / "stage2" / "reasoning_sft" / "ignore_rule_data_k64_dt01_completion_001" / "20260327_231917_town01_intersection_weave_ccw_expert_eval_0025824a9fa8" / "samples_reasoning_sft.jsonl",
                root / "datasets" / "processed" / "stage2" / "reasoning_sft" / "ignore_rule_data_k64_dt01_completion_001" / "20260327_231917_town01_intersection_weave_cw_expert_eval_0025824a9fa8" / "samples_reasoning_sft.jsonl",
                root / "datasets" / "processed" / "stage2" / "reasoning_sft" / "ignore_rule_data_k64_dt01_completion_001" / "20260327_231917_town01_perimeter_cw_expert_eval_0025824a9fa8" / "samples_reasoning_sft.jsonl",
            ),
            train_preprocess_expected_counts=(6423, 4676, 6167),
            curve_preprocess_outputs=(
                root / "datasets" / "processed" / "stage2" / "reasoning_sft" / "ignore_rule_data_k64_dt01_completion_001_curve_eval" / "perimeter_cw_holdout_v1" / "samples_reasoning_sft.jsonl",
            ),
            curve_preprocess_expected_counts=(569,),
        )

    def stage1a_required_artifacts(self) -> tuple[Path, ...]:
        return (self.stage1a_summary, self.stage1a_best, self.stage1a_final)

    def stage1b_required_artifacts(self) -> tuple[Path, ...]:
        return (self.stage1b_summary, self.stage1b_best, self.stage1b_last)

    def stage2_required_artifacts(self) -> tuple[Path, ...]:
        return (self.stage2_summary, self.stage2_best)

