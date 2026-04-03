"""Shared runtime helpers for Stage 3 post-training train/eval entrypoints."""

from __future__ import annotations

import json
from pathlib import Path

from .rewards import RewardWeights, aggregate_rewards
from .rewards.consistency import score_consistency
from .rewards.trajectory import score_trajectory


def sample_view_from_batch(batch: dict) -> dict:
    sample = {
        "sample_id": batch["sample_id"][0],
        "image_path": batch["image_path"][0],
        "action": batch["action"][0].detach().cpu(),
        "v0": batch["v0"][0].detach().cpu(),
        "gt_waypoints": batch["gt_waypoints"][0].detach().cpu(),
        "dt": float(batch["dt"][0]),
        "ego_history_xyz": batch["ego_history_xyz"][0].detach().cpu(),
        "ego_history_rot": batch["ego_history_rot"][0].detach().cpu(),
        "ego_future_xyz": batch["ego_future_xyz"][0].detach().cpu(),
        "ego_future_rot": batch["ego_future_rot"][0].detach().cpu(),
        "reasoning_text": batch["reasoning_text"][0],
    }
    if "sample_weight" in batch:
        sample["sample_weight"] = float(batch["sample_weight"][0].item())
    for key in ("command", "planner_state", "decision_longitudinal", "decision_lateral"):
        if key in batch:
            sample[key] = batch[key][0]
    return sample


def score_stage3_rollout(
    *,
    sample: dict,
    rollout,
    reasoning_scorer,
    reward_weights: RewardWeights,
    traj_l2_weight: float,
    traj_jerk_weight: float,
):
    reasoning_result = reasoning_scorer.score(sample, rollout.parsed.reasoning_text)
    consistency_result = score_consistency(
        reasoning_text=rollout.parsed.reasoning_text,
        pred_future_xyz=rollout.pred_future_xyz,
        v0=float(sample["v0"].item()),
        dt=float(sample["dt"]),
    )
    trajectory_result = score_trajectory(
        sample=sample,
        pred_future_xyz=rollout.pred_future_xyz,
        l2_weight=traj_l2_weight,
        jerk_weight=traj_jerk_weight,
    )
    return aggregate_rewards(
        reasoning=reasoning_result,
        consistency=consistency_result,
        trajectory=trajectory_result,
        weights=reward_weights,
    )


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
