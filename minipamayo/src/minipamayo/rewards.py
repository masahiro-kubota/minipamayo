"""Reward functions for Stage 4 GRPO.

Three reward signals (matching Alpamayo design):
  1. r_reason: LLM-based reasoning quality scoring (0-5 scale)
  2. r_consistency: CoC-Action consistency (binary)
  3. r_traj: Low-level trajectory quality (L2 + collision + jerk)

Composite reward:
  R = w_reason * r_reason/5 + w_consistency * r_consistency + w_traj * r_traj
"""

import base64
import hashlib
import json
import math
import re
from pathlib import Path

import torch

from .models.dynamics import forward_dynamics_batch

# --- Ego vehicle safety margin for collision checking ---
# Half-dimensions of typical ego vehicle (added to obstacle boxes)
EGO_HALF_WIDTH = 1.0  # meters
EGO_HALF_LENGTH = 2.25  # meters

# --- r_reason prompt template (Alpamayo §5.3.2 aligned) ---
# LRM critic receives: image + GT CoC + PRED CoC, scores PRED vs GT.
REASON_REWARD_PROMPT = """\
You are an expert evaluator for autonomous driving reasoning traces. \
The reasoning trace describes what the ego vehicle should be doing and \
the reasons and factors that lead to the behavior. Your task is to score \
how well a predicted reasoning trace (PRED) aligns with the ground truth \
(GT) in terms of behavior consistency and causal reasoning.

The attached image shows the driving scene from the front camera. \
Use this visual context to verify the reasoning.

[Ground Truth Reasoning (GT)]
{gt_reasoning}

[Predicted Reasoning (PRED)]
{pred_reasoning}

Scoring rubric (0-5):
5 Behavior & causal reasoning fully consistent.
4 Behavior correct; causal reasoning mostly consistent.
3 Behavior roughly correct, but incomplete or slightly incorrect reasoning.
2 Behavior partially incorrect or reasoning largely inconsistent.
1 Behavior is wrong or contradicts GT.
0 Completely unrelated or opposite.

Respond with ONLY a single integer (0-5). No explanation."""


# ============================================================
# r_reason: Reasoning quality reward via external LLM API
# ============================================================


def _encode_image_base64(image_path: str | Path) -> str:
    """Encode image file to base64 data URL for OpenAI Vision API."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(
        suffix, "image/jpeg"
    )
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{data}"


class ReasonReward:
    """Reasoning quality reward via multimodal LLM API (0-5 scale).

    Alpamayo §5.3.2 aligned: LRM critic receives image + GT CoC + PRED CoC,
    scores how well PRED matches GT on behavior alignment and causal reasoning.

    Uses OpenAI Vision API (gpt-4o). Results are cached to disk.

    Usage:
        rr = ReasonReward(model="gpt-4o")
        score = rr.compute(
            image_path="path/to/image.jpg",
            gt_reasoning="GT CoC trace...",
            pred_reasoning="PRED CoC trace...",
        )
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

    def _cache_key(self, image_path: str, gt_reasoning: str, pred_reasoning: str) -> str:
        content = f"{image_path}||{gt_reasoning}||{pred_reasoning}"
        return hashlib.sha256(content.encode()).hexdigest()

    def compute(
        self,
        image_path: str | Path,
        gt_reasoning: str,
        pred_reasoning: str,
    ) -> float:
        """Score reasoning quality (0-5 scale).

        Alpamayo §5.3.2: LRM receives image + GT + PRED, scores PRED vs GT
        on behavior alignment and causal reasoning quality.

        Args:
            image_path: Path to the driving scene image
            gt_reasoning: Ground truth CoC reasoning trace
            pred_reasoning: Model-generated CoC reasoning trace

        Returns:
            score: float in [0.0, 5.0]
        """
        image_path = str(image_path)

        # Check cache
        key = self._cache_key(image_path, gt_reasoning, pred_reasoning)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.loads(f.read())["score"]

        # Build prompt text
        prompt_text = REASON_REWARD_PROMPT.format(
            gt_reasoning=gt_reasoning,
            pred_reasoning=pred_reasoning,
        )

        # Build multimodal message (image + text)
        image_url = _encode_image_base64(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=10,
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip()

        # Parse score — extract first number from response
        match = re.search(r"[0-5](?:\.\d+)?", text)
        score = max(0.0, min(5.0, float(match.group()))) if match else 2.5

        # Cache result
        with open(cache_file, "w") as f:
            json.dump(
                {
                    "score": score,
                    "image_path": image_path,
                    "gt_reasoning": gt_reasoning[:500],
                    "pred_reasoning": pred_reasoning[:500],
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


def has_collision(
    pred_waypoints: torch.Tensor,
    obstacles: list[dict],
) -> bool:
    """Check if any predicted waypoint collides with obstacles (binary).

    Alpamayo §5.3.2: binary collision indicator I[collision(x_pred)].

    Args:
        pred_waypoints: (K, 2) predicted ego-centric waypoints
        obstacles: list of dicts with keys: center [x, y], size [w, l], heading

    Returns:
        True if any waypoint collides with any obstacle.
    """
    if not obstacles:
        return False

    K = pred_waypoints.shape[0]

    for k in range(K):
        px = pred_waypoints[k, 0].item()
        py = pred_waypoints[k, 1].item()

        for obs in obstacles:
            cx, cy = obs["center"]
            w, length = obs["size"]
            heading = obs["heading"]

            if _point_in_inflated_obb(px, py, cx, cy, w / 2, length / 2, heading):
                return True

    return False


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
    lambda_l2: float = 1.0,
    lambda_coll: float = 5.0,
    lambda_jerk: float = 0.1,
) -> float:
    """Low-level trajectory quality reward (Alpamayo §5.3.2).

    r_traj = -(λ_L2 · ||x_pred - x_expert||²_2 + λ_coll · I[collision] + λ_jerk · J(x_pred))

    Penalty formulation: negative values indicate worse trajectories.

    Args:
        pred_a: (K,) acceleration
        pred_kappa: (K,) curvature
        gt_waypoints: (K, 2) ground truth waypoints
        v0: scalar initial speed
        obstacles: list of obstacle dicts (None = no collision penalty)
        dt: timestep
        lambda_l2: weight for L2 distance penalty
        lambda_coll: weight for binary collision penalty
        lambda_jerk: weight for jerk penalty

    Returns:
        reward: float (negative penalty)
    """
    # Forward dynamics to get predicted waypoints
    pred_wp = forward_dynamics_batch(
        pred_a.unsqueeze(0), pred_kappa.unsqueeze(0), v0.unsqueeze(0), dt=dt
    ).squeeze(0)  # (K, 2)

    # L2 distance penalty: ||x_pred - x_expert||²_2
    l2_penalty = (pred_wp - gt_waypoints).pow(2).sum().item()

    # Binary collision indicator: I[collision(x_pred)]
    coll_indicator = 1.0 if (obstacles and has_collision(pred_wp, obstacles)) else 0.0

    # Jerk penalty: J(x_pred) = sum|a_{i+1} - a_i|
    jerk = torch.diff(pred_a).abs().sum().item() if pred_a.shape[0] >= 2 else 0.0

    return -(lambda_l2 * l2_penalty + lambda_coll * coll_indicator + lambda_jerk * jerk)


# ============================================================
# Composite reward: R = w_reason * r_reason/5 + w_consistency * r_consistency + r_traj
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
    """Composite reward combining all three signals (Alpamayo §5.3.2).

    R = w_reason * (r_reason/5) + w_consistency * r_consistency + w_traj * r_traj

    r_traj is a negative penalty term from trajectory_reward().
    When r_reason is None, weights are renormalized over the remaining signals.

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
        reward: float
    """
    r_c = consistency_reward(pred_a, pred_kappa, v0, decision, dt)
    r_t = trajectory_reward(pred_a, pred_kappa, gt_waypoints, v0, obstacles, dt)

    if r_reason is not None:
        r_r_normalized = max(0.0, min(1.0, r_reason / 5.0))
        return w_reason * r_r_normalized + w_consistency * r_c + w_traj * r_t
    else:
        w_sum = w_consistency + w_traj
        return (w_consistency / w_sum) * r_c + (w_traj / w_sum) * r_t
