"""Reward functions for Stage 4 GRPO.

Three reward signals (matching Alpamayo design):
  1. r_reason: LLM-based reasoning quality scoring (0-5 scale)
  2. r_consistency: CoC-Action consistency (binary)
  3. r_traj: Low-level trajectory quality (L2 + collision + jerk)

Composite reward:
  R = w_reason * r_reason/5 + w_consistency * r_consistency + w_traj * r_traj
"""

import hashlib
import json
import math
from pathlib import Path

import torch

from .models.dynamics import forward_dynamics_batch

# --- Ego vehicle safety margin for collision checking ---
# Half-dimensions of typical ego vehicle (added to obstacle boxes)
EGO_HALF_WIDTH = 1.0  # meters
EGO_HALF_LENGTH = 2.25  # meters

# --- r_reason prompt template (from stage4-rl.md) ---
REASON_REWARD_PROMPT = """\
以下は自動運転シーンに対する推論トレースです。
0-5 のスケールで評価してください。

評価基準:
- 5: 因果要因が正確に特定され、運転行動と一貫した推論
- 3: おおよそ妥当だが、因果要因の一部が不正確または欠落
- 1: 推論が視覚入力と矛盾、または行動と不整合
- 0: 無関係な推論

[推論トレース]
{reasoning_trace}

[予測された行動]
{predicted_action}

スコア (0-5):"""


# ============================================================
# r_reason: Reasoning quality reward via external LLM API
# ============================================================


class ReasonReward:
    """Reasoning quality reward via external LLM API (0-5 scale).

    Uses OpenAI API to score reasoning traces. Results are cached to disk
    to avoid redundant API calls across runs.

    Usage:
        rr = ReasonReward(cache_dir="data/reason_reward_cache")
        score = rr.compute("reasoning text...", "predicted action text...")
        # Returns float in [0, 5]
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/reason_reward_cache",
        model: str = "gpt-4o-mini",
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def _cache_key(self, reasoning_text: str, action_text: str) -> str:
        content = f"{reasoning_text}||{action_text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def compute(self, reasoning_text: str, action_text: str) -> float:
        """Score reasoning quality (0-5 scale).

        Args:
            reasoning_text: Generated CoC reasoning trace
            action_text: Predicted driving action description

        Returns:
            score: float in [0.0, 5.0]
        """
        # Check cache
        key = self._cache_key(reasoning_text, action_text)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.loads(f.read())["score"]

        # API call
        prompt = REASON_REWARD_PROMPT.format(
            reasoning_trace=reasoning_text,
            predicted_action=action_text,
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip()

        # Parse score
        try:
            score = float(text.split()[0])
            score = max(0.0, min(5.0, score))
        except (ValueError, IndexError):
            score = 2.5  # fallback

        # Cache result
        with open(cache_file, "w") as f:
            json.dump(
                {
                    "score": score,
                    "reasoning": reasoning_text[:500],
                    "action": action_text[:200],
                },
                f,
                ensure_ascii=False,
            )

        return score


# ============================================================
# r_collision: Obstacle collision penalty
# ============================================================


def _point_in_inflated_obb(
    px: float,
    py: float,
    cx: float,
    cy: float,
    half_w: float,
    half_l: float,
    heading: float,
) -> bool:
    """Check if point (px, py) is inside an inflated oriented bounding box.

    The box is inflated by ego vehicle half-dimensions for safety.
    """
    dx = px - cx
    dy = py - cy
    cos_h = math.cos(-heading)
    sin_h = math.sin(-heading)
    # Rotate to box-local frame
    local_x = cos_h * dx - sin_h * dy
    local_y = sin_h * dx + cos_h * dy
    # Inflated half-dimensions (obstacle + ego margin)
    return abs(local_x) <= (half_l + EGO_HALF_LENGTH) and abs(local_y) <= (half_w + EGO_HALF_WIDTH)


def collision_reward(
    pred_waypoints: torch.Tensor,
    obstacles: list[dict],
) -> float:
    """Collision penalty reward.

    Checks each predicted waypoint against obstacle bounding boxes.

    Args:
        pred_waypoints: (K, 2) predicted ego-centric waypoints
        obstacles: list of dicts with keys: center [x, y], size [w, l], heading

    Returns:
        reward: float in [0, 1]. 1.0 = no collisions, 0.0 = all waypoints collide.
    """
    if not obstacles:
        return 1.0

    K = pred_waypoints.shape[0]
    n_collisions = 0

    for k in range(K):
        px = pred_waypoints[k, 0].item()
        py = pred_waypoints[k, 1].item()

        for obs in obstacles:
            cx, cy = obs["center"]
            w, length = obs["size"]
            heading = obs["heading"]

            if _point_in_inflated_obb(px, py, cx, cy, w / 2, length / 2, heading):
                n_collisions += 1
                break  # One collision per waypoint is enough

    return 1.0 - n_collisions / K


# ============================================================
# r_consistency: CoC-Action consistency (binary)
# ============================================================


def extract_meta_action(
    pred_a: torch.Tensor,
    pred_kappa: torch.Tensor,
    v0: torch.Tensor,
    dt: float = 0.5,
) -> dict[str, str]:
    """Extract meta-action from predicted control inputs.

    Args:
        pred_a: (K,) acceleration sequence
        pred_kappa: (K,) curvature sequence
        v0: scalar initial speed
        dt: timestep

    Returns:
        meta_action: {"longitudinal": str, "lateral": str}
    """
    K = pred_a.shape[0]

    # Compute velocity sequence
    velocities = [v0.item()]
    for i in range(K):
        v_next = velocities[-1] + pred_a[i].item() * dt
        velocities.append(max(0, v_next))

    final_v = velocities[-1]
    v_change = final_v - velocities[0]

    # Longitudinal
    if final_v < 0.3:
        longitudinal = "stop"
    elif v_change < -1.0:
        longitudinal = "yield"
    else:
        longitudinal = "go_straight"

    # Lateral: compute trajectory and heading changes
    waypoints = forward_dynamics_batch(
        pred_a.unsqueeze(0), pred_kappa.unsqueeze(0), v0.unsqueeze(0), dt=dt
    ).squeeze(0)  # (K, 2)

    # Cumulative heading change from curvature
    heading = 0.0
    v = v0.item()
    for i in range(K):
        heading += v * pred_kappa[i].item() * dt
        v = max(0, v + pred_a[i].item() * dt)

    heading_deg = math.degrees(heading)

    if heading_deg > 30:
        lateral = "turn_left"
    elif heading_deg < -30:
        lateral = "turn_right"
    elif abs(waypoints[-1, 1].item()) > 1.5 and abs(heading_deg) < 30:
        lateral = "lane_change_left" if waypoints[-1, 1].item() > 0 else "lane_change_right"
    else:
        lateral = "lane_keeping"

    return {"longitudinal": longitudinal, "lateral": lateral}


def consistency_reward(
    pred_a: torch.Tensor,
    pred_kappa: torch.Tensor,
    v0: torch.Tensor,
    decision: dict[str, str],
    dt: float = 0.5,
) -> float:
    """CoC-Action consistency reward (binary).

    Compares the meta-action extracted from trajectory with the CoC decision.

    Returns:
        reward: 0.0 or 1.0
    """
    meta = extract_meta_action(pred_a, pred_kappa, v0, dt)

    # Longitudinal match (follow_lead treated as go_straight)
    gt_long = decision["longitudinal"]
    pred_long = meta["longitudinal"]
    if gt_long == "follow_lead":
        gt_long = "go_straight"
    long_match = pred_long == gt_long

    # Lateral match
    lat_match = meta["lateral"] == decision["lateral"]

    return 1.0 if (long_match and lat_match) else 0.0


# ============================================================
# r_traj: Low-level trajectory quality (L2 + collision + jerk)
# ============================================================


def trajectory_reward(
    pred_a: torch.Tensor,
    pred_kappa: torch.Tensor,
    gt_waypoints: torch.Tensor,
    v0: torch.Tensor,
    obstacles: list[dict] | None = None,
    dt: float = 0.5,
    alpha: float = 0.5,
    gamma: float = 2.0,
    w_l2: float = 0.5,
    w_col: float = 0.3,
    w_jerk: float = 0.2,
) -> float:
    """Low-level trajectory quality reward.

    r_traj = w_l2 * r_l2 + w_col * r_collision + w_jerk * r_jerk

    Args:
        pred_a: (K,) acceleration
        pred_kappa: (K,) curvature
        gt_waypoints: (K, 2) ground truth waypoints
        v0: scalar initial speed
        obstacles: list of obstacle dicts (None = no collision penalty)
        dt: timestep
        alpha: L2 reward scaling
        gamma: jerk penalty scaling
        w_l2: weight for L2 distance reward
        w_col: weight for collision penalty reward
        w_jerk: weight for jerk suppression reward

    Returns:
        reward: float in [0, 1]
    """
    # Forward dynamics to get predicted waypoints
    pred_wp = forward_dynamics_batch(
        pred_a.unsqueeze(0), pred_kappa.unsqueeze(0), v0.unsqueeze(0), dt=dt
    ).squeeze(0)  # (K, 2)

    # L2 reward: r_l2 = exp(-alpha * mean_l2_distance)
    l2_dist = torch.norm(pred_wp - gt_waypoints, dim=1).mean().item()
    r_l2 = math.exp(-alpha * l2_dist)

    # Collision reward: r_collision = 1 - n_collisions / n_waypoints
    r_col = collision_reward(pred_wp, obstacles) if obstacles is not None else 1.0

    # Jerk penalty: r_jerk = exp(-gamma * mean_jerk)
    if pred_a.shape[0] >= 3:
        a_jerk = torch.diff(pred_a, n=2).abs().mean().item()
        k_jerk = torch.diff(pred_kappa, n=2).abs().mean().item()
        jerk = a_jerk + k_jerk * 100  # scale kappa jerk
        r_jerk = math.exp(-gamma * jerk)
    else:
        r_jerk = 1.0

    # Weighted combination (design: w_l2=0.5, w_col=0.3, w_jerk=0.2)
    return w_l2 * r_l2 + w_col * r_col + w_jerk * r_jerk


# ============================================================
# Composite reward: R = w_reason * r_reason/5 + w_consistency * r_consistency + w_traj * r_traj
# ============================================================


def composite_reward(
    pred_a: torch.Tensor,
    pred_kappa: torch.Tensor,
    gt_waypoints: torch.Tensor,
    v0: torch.Tensor,
    decision: dict[str, str],
    obstacles: list[dict] | None = None,
    r_reason: float | None = None,
    dt: float = 0.5,
    w_reason: float = 0.4,
    w_consistency: float = 0.3,
    w_traj: float = 0.3,
) -> float:
    """Composite reward combining all three signals.

    R = w_reason * (r_reason/5) + w_consistency * r_consistency + w_traj * r_traj

    When r_reason is None, weights are renormalized over the remaining signals:
      R = (w_consistency / (w_consistency + w_traj)) * r_consistency
        + (w_traj / (w_consistency + w_traj)) * r_traj

    Args:
        pred_a: (K,) acceleration
        pred_kappa: (K,) curvature
        gt_waypoints: (K, 2) ground truth waypoints
        v0: scalar initial speed
        decision: GT driving decision dict
        obstacles: obstacle list (None = skip collision penalty)
        r_reason: pre-computed reasoning quality score in [0, 5] (None = skip)
        dt: timestep
        w_reason: weight for reasoning quality reward
        w_consistency: weight for CoC-Action consistency reward
        w_traj: weight for trajectory quality reward

    Returns:
        reward: float in [0, 1]
    """
    r_c = consistency_reward(pred_a, pred_kappa, v0, decision, dt)
    r_t = trajectory_reward(pred_a, pred_kappa, gt_waypoints, v0, obstacles, dt)

    if r_reason is not None:
        # Normalize r_reason from [0, 5] to [0, 1]
        r_r_normalized = max(0.0, min(1.0, r_reason / 5.0))
        return w_reason * r_r_normalized + w_consistency * r_c + w_traj * r_t
    else:
        # Renormalize weights when r_reason is unavailable
        w_sum = w_consistency + w_traj
        return (w_consistency / w_sum) * r_c + (w_traj / w_sum) * r_t
