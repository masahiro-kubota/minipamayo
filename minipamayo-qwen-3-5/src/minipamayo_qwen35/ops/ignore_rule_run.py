"""Python orchestration CLI for the canonical ignore-rule completion workflow."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from .ignore_rule_paths import (
    CUDA_BIN,
    CUDA_HOME,
    CUDA_LIB64,
    IgnoreRulePaths,
    build_backup_path,
    timestamp_iso,
)


FULL_START_STAGES = ("stage1a", "stage1b", "stage2")
EVAL_TARGET_STAGES = ("stage1a", "stage1b", "stage2", "all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orchestrate the canonical ignore-rule completion workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    full_parser = subparsers.add_parser("full", description="Run the full Stage1A -> Stage1B -> Stage2 workflow.")
    full_parser.add_argument("--attempt-name", type=str, default="completion_ignore_rule_full_001")
    full_parser.add_argument("--session-name", type=str, default="ignore-rule-completion-001")
    full_parser.add_argument("--start-stage", type=str, choices=FULL_START_STAGES, default="stage1a")
    full_parser.add_argument("--max-stage1a-attempts", type=int, default=0)
    full_parser.add_argument("--stage1a-retry-sleep-s", type=int, default=30)
    full_parser.add_argument("--max-stage1b-attempts", type=int, default=0)
    full_parser.add_argument("--stage1b-retry-sleep-s", type=int, default=30)
    full_parser.add_argument("--max-stage2-attempts", type=int, default=0)
    full_parser.add_argument("--stage2-retry-sleep-s", type=int, default=30)
    full_parser.set_defaults(handler=handle_full_command)

    eval_parser = subparsers.add_parser("eval", description="Run curve-only evaluation for ignore-rule completion artifacts.")
    eval_parser.add_argument("--attempt-name", type=str, default="completion_ignore_rule_curve_eval_001")
    eval_parser.add_argument("--session-name", type=str, default="ignore-rule-curve-eval-001")
    eval_parser.add_argument("--target-stage", type=str, choices=EVAL_TARGET_STAGES, default="all")
    eval_parser.add_argument(
        "--run-stage2-sample-inference",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    eval_parser.set_defaults(handler=handle_eval_command)

    watch_parser = subparsers.add_parser("watch", description="Watch an existing ignore-rule completion attempt.")
    watch_parser.add_argument("--attempt-name", type=str, required=True)
    watch_parser.add_argument("--interval-s", type=int, default=600)
    watch_parser.add_argument("--tail-lines", type=int, default=30)
    watch_parser.set_defaults(handler=handle_watch_command)
    return parser


def prepend_env_path(existing: str | None, prefix: str) -> str:
    if existing:
        return f"{prefix}:{existing}"
    return prefix


def build_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = prepend_env_path(env.get("PATH"), str(CUDA_BIN))
    env["LD_LIBRARY_PATH"] = prepend_env_path(env.get("LD_LIBRARY_PATH"), str(CUDA_LIB64))
    env["CUDA_HOME"] = str(CUDA_HOME)
    return env


def require_existing_paths(paths: tuple[Path, ...], *, label: str) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"{label} is missing required artifacts:\n" + "\n".join(missing))


def require_exact_line_count(path: Path, expected: int) -> None:
    if not path.exists():
        raise RuntimeError(f"Required file does not exist: {path}")
    with path.open("r", encoding="utf-8") as fh:
        actual = sum(1 for _ in fh)
    if actual != expected:
        raise RuntimeError(f"Unexpected line count for {path}: expected={expected} actual={actual}")


@dataclass
class RunContext:
    paths: IgnoreRulePaths
    master_log_fp: TextIO
    current_stage: str = "bootstrap"

    def log_line(self, message: str) -> None:
        line = f"[{timestamp_iso()}] {message}"
        print(line, flush=True)
        self.master_log_fp.write(line + "\n")
        self.master_log_fp.flush()

    def write_run_status(self, state: str, rc: int) -> None:
        payload = {
            "attempt_name": self.paths.attempt_name,
            "session_name": self.paths.session_name,
            "state": state,
            "current_stage": self.current_stage,
            "exit_code": rc,
            "updated_at": timestamp_iso(),
            "log_root": str(self.paths.log_root),
        }
        self.paths.run_status_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self.paths.run_exitcode_file.write_text(f"{rc}\n", encoding="utf-8")

    def run_stage(self, stage_name: str, command: list[str]) -> int:
        self.current_stage = stage_name
        self.write_run_status("running", 999)
        stage_log = self.paths.log_root / f"{stage_name}.log"
        rc_file = self.paths.state_dir / f"{stage_name}.exitcode"
        status_file = self.paths.state_dir / f"{stage_name}.status.json"
        started_at = timestamp_iso()
        self.log_line(f"stage_start stage={stage_name} command={shlex.join(command)}")

        env = build_subprocess_env()
        rc = 1
        with stage_log.open("w", encoding="utf-8") as stage_fp:
            process = subprocess.Popen(
                command,
                cwd=self.paths.project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                stage_fp.write(line)
                stage_fp.flush()
                self.master_log_fp.write(line)
                self.master_log_fp.flush()
            rc = process.wait()

        finished_at = timestamp_iso()
        rc_file.write_text(f"{rc}\n", encoding="utf-8")
        status_file.write_text(
            json.dumps(
                {
                    "stage": stage_name,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "exit_code": rc,
                    "log_path": str(stage_log),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self.log_line(f"stage_end stage={stage_name} exit_code={rc} log={stage_log}")
        return rc


def backup_path_if_exists(path: Path, ctx: RunContext | None = None) -> Path | None:
    if not path.exists():
        return None
    backup_path = build_backup_path(path)
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    path.rename(backup_path)
    if ctx is not None:
        ctx.log_line(f"moved_aside path={path} backup={backup_path}")
    return backup_path


def prepare_log_root(paths: IgnoreRulePaths) -> RunContext:
    paths.log_root.parent.mkdir(parents=True, exist_ok=True)
    backup_path_if_exists(paths.log_root)
    paths.log_root.mkdir(parents=True, exist_ok=True)
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    paths.master_log.write_text("", encoding="utf-8")
    master_log_fp = paths.master_log.open("a", encoding="utf-8")
    ctx = RunContext(paths=paths, master_log_fp=master_log_fp)
    ctx.log_line(
        f"launcher_ready attempt={paths.attempt_name} session={paths.session_name} log_root={paths.log_root}"
    )
    ctx.write_run_status("running", 999)
    return ctx


def close_context(ctx: RunContext) -> None:
    ctx.master_log_fp.close()


def build_stage_entrypoint_command(entrypoint: str, config_json: Path) -> list[str]:
    return [sys.executable, "-m", entrypoint, "--config-json", str(config_json)]


def build_self_eval_command(
    *,
    attempt_name: str,
    session_name: str,
    target_stage: str,
    run_stage2_sample_inference: bool = True,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "minipamayo_qwen35.ops.ignore_rule_run",
        "eval",
        "--attempt-name",
        attempt_name,
        "--session-name",
        session_name,
        "--target-stage",
        target_stage,
    ]
    command.append(
        "--run-stage2-sample-inference"
        if run_stage2_sample_inference
        else "--no-run-stage2-sample-inference"
    )
    return command


def prepare_train_preprocess_outputs(ctx: RunContext) -> None:
    for output_path in ctx.paths.train_preprocess_outputs:
        backup_path_if_exists(output_path, ctx)


def prepare_train_artifacts(ctx: RunContext, *, start_stage: str) -> None:
    if start_stage == "stage1a":
        backup_path_if_exists(ctx.paths.stage1a_save_dir, ctx)
    if start_stage in {"stage1a", "stage1b"}:
        backup_path_if_exists(ctx.paths.stage1b_save_dir, ctx)
    backup_path_if_exists(ctx.paths.stage2_save_dir, ctx)


def prepare_eval_outputs(ctx: RunContext, *, target_stage: str, run_stage2_sample_inference: bool) -> None:
    if target_stage in {"stage1a", "all"}:
        backup_path_if_exists(ctx.paths.stage1a_eval_output, ctx)
        backup_path_if_exists(ctx.paths.stage1a_eval_progress, ctx)
        backup_path_if_exists(ctx.paths.stage1a_eval_per_sample, ctx)
    if target_stage in {"stage1b", "all"}:
        backup_path_if_exists(ctx.paths.stage1b_eval_output, ctx)
        backup_path_if_exists(ctx.paths.stage1b_eval_progress, ctx)
    if target_stage in {"stage2", "all"}:
        for output_path in ctx.paths.curve_preprocess_outputs:
            backup_path_if_exists(output_path, ctx)
        backup_path_if_exists(ctx.paths.stage2_eval_output, ctx)
        backup_path_if_exists(ctx.paths.stage2_eval_progress, ctx)
        if run_stage2_sample_inference:
            backup_path_if_exists(ctx.paths.stage2_infer_output, ctx)
            backup_path_if_exists(ctx.paths.stage2_infer_progress, ctx)


def validate_train_preprocess_outputs(ctx: RunContext) -> None:
    for output_path, expected in zip(
        ctx.paths.train_preprocess_outputs,
        ctx.paths.train_preprocess_expected_counts,
        strict=True,
    ):
        require_exact_line_count(output_path, expected)
        ctx.log_line(f"validated_line_count path={output_path} expected={expected}")


def validate_curve_preprocess_outputs(ctx: RunContext) -> None:
    for output_path, expected in zip(
        ctx.paths.curve_preprocess_outputs,
        ctx.paths.curve_preprocess_expected_counts,
        strict=True,
    ):
        require_exact_line_count(output_path, expected)
        ctx.log_line(f"validated_line_count path={output_path} expected={expected}")


def run_retry_loop(
    ctx: RunContext,
    *,
    stage_label: str,
    base_stage_name: str,
    command: list[str],
    required_artifacts: tuple[Path, ...],
    max_attempts: int,
    retry_sleep_s: int,
    save_dir: Path,
) -> None:
    attempt = 0
    while True:
        attempt += 1
        backup_path_if_exists(save_dir, ctx)
        rc = ctx.run_stage(f"{base_stage_name}_attempt_{attempt:03d}", command)
        if rc == 0:
            try:
                require_existing_paths(required_artifacts, label=stage_label)
                ctx.log_line(f"{stage_label.lower()}_success attempt={attempt}")
                return
            except RuntimeError as exc:
                ctx.log_line(str(exc))
                rc = 1

        ctx.log_line(
            f"{stage_label.lower()}_retry_required attempt={attempt} exit_code={rc} sleep_s={retry_sleep_s}"
        )
        if max_attempts > 0 and attempt >= max_attempts:
            raise RuntimeError(f"{stage_label} retry exhausted after {attempt} attempts.")
        time.sleep(retry_sleep_s)


def run_eval_subcommand_non_blocking(ctx: RunContext, *, target_stage: str, stage_name: str) -> None:
    attempt_name = f"{ctx.paths.attempt_name}_{target_stage}_curve_eval"
    session_name = f"{ctx.paths.session_name}-{target_stage}-curve-eval"
    rc = ctx.run_stage(
        stage_name,
        build_self_eval_command(
            attempt_name=attempt_name,
            session_name=session_name,
            target_stage=target_stage,
            run_stage2_sample_inference=True,
        ),
    )
    if rc != 0:
        ctx.log_line(f"non_blocking_failure stage={stage_name}")


def enforce_full_start_prerequisites(paths: IgnoreRulePaths, *, start_stage: str) -> None:
    if start_stage == "stage1b":
        require_existing_paths(paths.stage1a_required_artifacts(), label="Stage1A start-stage prerequisites")
    elif start_stage == "stage2":
        require_existing_paths(paths.stage1a_required_artifacts(), label="Stage1A start-stage prerequisites")
        require_existing_paths(paths.stage1b_required_artifacts(), label="Stage1B start-stage prerequisites")


def enforce_eval_prerequisites(
    paths: IgnoreRulePaths,
    *,
    target_stage: str,
    run_stage2_sample_inference: bool,
) -> None:
    if target_stage in {"stage1a", "all"}:
        require_existing_paths(paths.stage1a_required_artifacts(), label="Stage1A eval prerequisites")
    if target_stage in {"stage1b", "all"}:
        require_existing_paths(paths.stage1b_required_artifacts(), label="Stage1B eval prerequisites")
    if target_stage in {"stage2", "all"}:
        require_existing_paths(paths.stage2_required_artifacts(), label="Stage2 eval prerequisites")
        if run_stage2_sample_inference:
            require_existing_paths(paths.stage1b_required_artifacts(), label="Stage2 sample inference prerequisites")


def run_stage1a_eval(ctx: RunContext) -> None:
    rc = ctx.run_stage(
        "stage1a_curve_eval",
        build_stage_entrypoint_command(
            "minipamayo_qwen35.stage1.vlm_ce.eval",
            ctx.paths.stage1a_eval_config,
        ),
    )
    if rc != 0:
        raise RuntimeError("Stage1A curve eval failed.")


def run_stage1b_eval(ctx: RunContext) -> None:
    rc = ctx.run_stage(
        "stage1b_curve_eval",
        build_stage_entrypoint_command(
            "minipamayo_qwen35.stage1.expert_cfm.eval",
            ctx.paths.stage1b_eval_config,
        ),
    )
    if rc != 0:
        raise RuntimeError("Stage1B curve eval failed.")


def run_stage2_eval_suite(ctx: RunContext, *, run_stage2_sample_inference: bool) -> None:
    rc = ctx.run_stage(
        "stage2_curve_eval_preprocess",
        build_stage_entrypoint_command(
            "minipamayo_qwen35.stage2.reasoning_sft.preprocess",
            ctx.paths.stage2_curve_preprocess_config,
        ),
    )
    if rc != 0:
        raise RuntimeError("Stage2 curve eval preprocess failed.")
    validate_curve_preprocess_outputs(ctx)

    rc = ctx.run_stage(
        "stage2_curve_eval",
        build_stage_entrypoint_command(
            "minipamayo_qwen35.stage2.reasoning_sft.eval",
            ctx.paths.stage2_eval_config,
        ),
    )
    if rc != 0:
        raise RuntimeError("Stage2 curve eval failed.")

    if run_stage2_sample_inference:
        rc = ctx.run_stage(
            "stage2_curve_sample_inference",
            build_stage_entrypoint_command(
                "minipamayo_qwen35.stage2.reasoning_sft.inference",
                ctx.paths.stage2_infer_config,
            ),
        )
        if rc != 0:
            raise RuntimeError("Stage2 curve sample inference failed.")


def handle_full_command(args: argparse.Namespace) -> int:
    paths = IgnoreRulePaths.for_attempt(args.attempt_name, args.session_name)
    ctx = prepare_log_root(paths)
    try:
        ctx.log_line(
            f"launcher_config start_stage={args.start_stage} attempt={args.attempt_name} session={args.session_name}"
        )
        prepare_train_preprocess_outputs(ctx)
        prepare_train_artifacts(ctx, start_stage=args.start_stage)

        rc = ctx.run_stage(
            "stage2_preprocess",
            build_stage_entrypoint_command(
                "minipamayo_qwen35.stage2.reasoning_sft.preprocess",
                paths.stage2_preprocess_config,
            ),
        )
        if rc != 0:
            ctx.write_run_status("failed", 10)
            return 10
        validate_train_preprocess_outputs(ctx)

        try:
            enforce_full_start_prerequisites(paths, start_stage=args.start_stage)
        except RuntimeError as exc:
            ctx.log_line(str(exc))
            code = 21 if args.start_stage == "stage1b" else 31
            ctx.write_run_status("failed", code)
            return code

        if args.start_stage == "stage1a":
            try:
                run_retry_loop(
                    ctx,
                    stage_label="Stage1A",
                    base_stage_name="stage1a_train",
                    command=build_stage_entrypoint_command(
                        "minipamayo_qwen35.stage1.vlm_ce.train",
                        paths.stage1a_train_config,
                    ),
                    required_artifacts=paths.stage1a_required_artifacts(),
                    max_attempts=args.max_stage1a_attempts,
                    retry_sleep_s=args.stage1a_retry_sleep_s,
                    save_dir=paths.stage1a_save_dir,
                )
            except RuntimeError as exc:
                ctx.log_line(str(exc))
                ctx.write_run_status("failed", 20)
                return 20
            run_eval_subcommand_non_blocking(ctx, target_stage="stage1a", stage_name="stage1a_curve_eval")

            try:
                run_retry_loop(
                    ctx,
                    stage_label="Stage1B",
                    base_stage_name="stage1b_train",
                    command=build_stage_entrypoint_command(
                        "minipamayo_qwen35.stage1.expert_cfm.train",
                        paths.stage1b_train_config,
                    ),
                    required_artifacts=paths.stage1b_required_artifacts(),
                    max_attempts=args.max_stage1b_attempts,
                    retry_sleep_s=args.stage1b_retry_sleep_s,
                    save_dir=paths.stage1b_save_dir,
                )
            except RuntimeError as exc:
                ctx.log_line(str(exc))
                ctx.write_run_status("failed", 30)
                return 30
            run_eval_subcommand_non_blocking(ctx, target_stage="stage1b", stage_name="stage1b_curve_eval")
        elif args.start_stage == "stage1b":
            run_eval_subcommand_non_blocking(ctx, target_stage="stage1a", stage_name="stage1a_curve_eval")
            try:
                run_retry_loop(
                    ctx,
                    stage_label="Stage1B",
                    base_stage_name="stage1b_train",
                    command=build_stage_entrypoint_command(
                        "minipamayo_qwen35.stage1.expert_cfm.train",
                        paths.stage1b_train_config,
                    ),
                    required_artifacts=paths.stage1b_required_artifacts(),
                    max_attempts=args.max_stage1b_attempts,
                    retry_sleep_s=args.stage1b_retry_sleep_s,
                    save_dir=paths.stage1b_save_dir,
                )
            except RuntimeError as exc:
                ctx.log_line(str(exc))
                ctx.write_run_status("failed", 30)
                return 30
            run_eval_subcommand_non_blocking(ctx, target_stage="stage1b", stage_name="stage1b_curve_eval")
        elif args.start_stage == "stage2":
            run_eval_subcommand_non_blocking(ctx, target_stage="stage1a", stage_name="stage1a_curve_eval")
            run_eval_subcommand_non_blocking(ctx, target_stage="stage1b", stage_name="stage1b_curve_eval")

        try:
            run_retry_loop(
                ctx,
                stage_label="Stage2",
                base_stage_name="stage2_train",
                command=build_stage_entrypoint_command(
                    "minipamayo_qwen35.stage2.reasoning_sft.train",
                    paths.stage2_train_config,
                ),
                required_artifacts=paths.stage2_required_artifacts(),
                max_attempts=args.max_stage2_attempts,
                retry_sleep_s=args.stage2_retry_sleep_s,
                save_dir=paths.stage2_save_dir,
            )
        except RuntimeError as exc:
            ctx.log_line(str(exc))
            ctx.write_run_status("failed", 40)
            return 40

        run_eval_subcommand_non_blocking(ctx, target_stage="stage2", stage_name="stage2_curve_eval_suite")
        ctx.write_run_status("completed", 0)
        ctx.log_line("run_completed exit_code=0")
        return 0
    except RuntimeError as exc:
        ctx.log_line(str(exc))
        ctx.write_run_status("failed", 1)
        return 1
    finally:
        close_context(ctx)


def handle_eval_command(args: argparse.Namespace) -> int:
    paths = IgnoreRulePaths.for_attempt(args.attempt_name, args.session_name)
    ctx = prepare_log_root(paths)
    try:
        prepare_eval_outputs(
            ctx,
            target_stage=args.target_stage,
            run_stage2_sample_inference=args.run_stage2_sample_inference,
        )
        enforce_eval_prerequisites(
            paths,
            target_stage=args.target_stage,
            run_stage2_sample_inference=args.run_stage2_sample_inference,
        )

        if args.target_stage in {"stage1a", "all"}:
            run_stage1a_eval(ctx)
        if args.target_stage in {"stage1b", "all"}:
            run_stage1b_eval(ctx)
        if args.target_stage in {"stage2", "all"}:
            run_stage2_eval_suite(ctx, run_stage2_sample_inference=args.run_stage2_sample_inference)

        ctx.write_run_status("completed", 0)
        ctx.log_line("run_completed exit_code=0")
        return 0
    except RuntimeError as exc:
        ctx.log_line(str(exc))
        ctx.write_run_status("failed", 1)
        return 1
    finally:
        close_context(ctx)


def _read_watch_state(status_file: Path) -> str:
    if not status_file.exists():
        return "missing"
    payload = json.loads(status_file.read_text(encoding="utf-8"))
    return str(payload.get("state", "missing"))


def _tail_lines(path: Path, line_count: int) -> list[str]:
    if not path.exists():
        return ["missing"]
    return path.read_text(encoding="utf-8").splitlines()[-line_count:] or [""]


def print_watch_block(label: str, lines: list[str]) -> None:
    print(f"\n[{label}]")
    for line in lines:
        print(line)


def handle_watch_command(args: argparse.Namespace) -> int:
    paths = IgnoreRulePaths.for_attempt(args.attempt_name, session_name="")
    while True:
        print(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S %Z')} ===")
        if paths.run_status_file.exists():
            status_lines = paths.run_status_file.read_text(encoding="utf-8").splitlines()
        else:
            status_lines = ["missing"]
        print_watch_block("run.status.json", status_lines)

        if paths.monitor_alert_file.exists():
            alert_lines = paths.monitor_alert_file.read_text(encoding="utf-8").splitlines() or [""]
        else:
            alert_lines = ["none"]
        print_watch_block("monitor.alert", alert_lines)

        print_watch_block("master.log tail", _tail_lines(paths.master_log, args.tail_lines))

        state = _read_watch_state(paths.run_status_file)
        if state in {"completed", "failed", "interrupted"}:
            print(f"\nwatch_exit state={state}")
            return 0
        time.sleep(args.interval_s)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
