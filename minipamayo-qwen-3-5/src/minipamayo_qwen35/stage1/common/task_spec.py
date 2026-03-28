"""Shared Stage 1 target specifications and quantizers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Subset

from ..data.canonical_action import canonical_action_array_from_record
from ..tokenization.quantizer import ActionQuantizer


def _record_action_array(record: dict) -> np.ndarray:
    action = canonical_action_array_from_record(record)
    if action.ndim != 1 or action.size == 0 or action.size % 2 != 0:
        raise RuntimeError(f"Stage 1 record has invalid `action` layout: {record!r}")
    return action


def iter_stage1_records(dataset) -> list[dict]:
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        if not hasattr(base_dataset, "records"):
            raise RuntimeError("Stage 1 subset base dataset is missing canonical `records`.")
        return [base_dataset.records[index] for index in dataset.indices]

    if not hasattr(dataset, "records"):
        raise RuntimeError("Stage 1 dataset is missing canonical `records`.")
    return list(dataset.records)


@dataclass(frozen=True)
class KappaOnlyQuantizer:
    """Single-channel quantizer for steer-only Stage 1 experiments."""

    n_bins: int = 256
    kappa_range: tuple[float, float] = (-0.2, 0.2)
    range_source: str = "train_corpus_percentile"
    range_percentile: float = 99.5
    range_margin_ratio: float = 0.1
    observed_min: float = 0.0
    observed_max: float = 0.0
    derived_radius: float = 0.2
    num_values: int = 0

    def quantize_bin(self, value: float) -> int:
        v_min, v_max = self.kappa_range
        clamped = max(v_min, min(v_max - 1e-8, float(value)))
        bin_idx = int((clamped - v_min) / (v_max - v_min) * self.n_bins)
        return min(max(bin_idx, 0), self.n_bins - 1)

    def dequantize_bin(self, bin_idx: int) -> float:
        v_min, v_max = self.kappa_range
        return v_min + (bin_idx + 0.5) * (v_max - v_min) / self.n_bins

    def encode_bin_ids(self, values: np.ndarray) -> list[int]:
        return [self.quantize_bin(float(value)) for value in values]

    def decode_bin_ids(self, bin_ids: list[int]) -> np.ndarray:
        values = [self.dequantize_bin(int(bin_idx)) for bin_idx in bin_ids]
        return np.asarray(values, dtype=np.float32)


class Stage1TaskSpec(ABC):
    """Defines how Stage 1 targets are quantized and interpreted."""

    name: str
    action_representation: str
    rollout_accel_source: str

    @abstractmethod
    def build_quantizer(self, train_dataset):
        """Construct the quantizer for a Stage 1 training run."""

    @abstractmethod
    def quantizer_from_checkpoint(self, quantizer_payload: dict):
        """Recreate the quantizer from checkpoint metadata."""

    @abstractmethod
    def target_from_action_array(self, action: np.ndarray) -> np.ndarray:
        """Return the generated token target from an interleaved full action array."""

    @abstractmethod
    def target_from_action_tensor(self, action: torch.Tensor) -> torch.Tensor:
        """Return the generated token target from an interleaved full action tensor."""

    @abstractmethod
    def full_action_from_target_tensor(
        self,
        target: torch.Tensor,
        *,
        gt_action_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reconstruct the full interleaved [a0, k0, ...] action vector for rollout and logging."""

    @abstractmethod
    def quantizer_metadata(self, quantizer) -> dict[str, Any]:
        """Serialize quantizer metadata for checkpoint and run summaries."""

    def validate_checkpoint(self, stage1_metadata: dict) -> None:
        action_representation = stage1_metadata.get("action_representation")
        if action_representation != self.action_representation:
            raise RuntimeError(
                "Checkpoint action representation does not match the entrypoint:\n"
                f"expected={self.action_representation!r}\n"
                f"found={action_representation!r}"
            )

    def target_dim_from_record(self, record: dict) -> int:
        return int(self.target_from_action_array(_record_action_array(record)).shape[0])

    def full_action_dim_from_record(self, record: dict) -> int:
        return int(_record_action_array(record).shape[0])

    def metadata(self, quantizer) -> dict[str, Any]:
        return {
            "action_representation": self.action_representation,
            "rollout_accel_source": self.rollout_accel_source,
            **self.quantizer_metadata(quantizer),
        }

    def encode_target_token_rows_from_batch(
        self,
        batch: dict,
        registry,
        quantizer,
    ) -> list[list[int]]:
        rows: list[list[int]] = []
        for action in batch["action"]:
            target = self.target_from_action_tensor(action).cpu().numpy()
            rows.append(registry.encode_target_token_ids(target, quantizer))
        return rows


@dataclass(frozen=True)
class CanonicalStage1Spec(Stage1TaskSpec):
    """Canonical Stage 1 target: interleaved acceleration and curvature."""

    name: str = "canonical"
    action_representation: str = "accel_kappa"
    rollout_accel_source: str = "predicted"

    def build_quantizer(self, train_dataset) -> ActionQuantizer:
        return ActionQuantizer()

    def quantizer_from_checkpoint(self, quantizer_payload: dict) -> ActionQuantizer:
        required_keys = ["n_bins", "a_range", "kappa_range"]
        missing_keys = [key for key in required_keys if key not in quantizer_payload]
        if missing_keys:
            raise RuntimeError(
                "Canonical Stage 1 checkpoint is missing quantizer metadata:\n" + "\n".join(missing_keys)
            )
        return ActionQuantizer(
            n_bins=int(quantizer_payload["n_bins"]),
            a_range=tuple(quantizer_payload["a_range"]),
            kappa_range=tuple(quantizer_payload["kappa_range"]),
        )

    def target_from_action_array(self, action: np.ndarray) -> np.ndarray:
        return np.asarray(action, dtype=np.float32)

    def target_from_action_tensor(self, action: torch.Tensor) -> torch.Tensor:
        return action.to(torch.float32)

    def full_action_from_target_tensor(
        self,
        target: torch.Tensor,
        *,
        gt_action_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return target.reshape(-1).to(torch.float32)

    def quantizer_metadata(self, quantizer: ActionQuantizer) -> dict[str, Any]:
        return {
            "quantizer_kind": "accel_kappa",
            "n_bins": quantizer.n_bins,
            "a_range": list(quantizer.a_range),
            "kappa_range": list(quantizer.kappa_range),
        }

    def encode_target_token_rows_from_batch(
        self,
        batch: dict,
        registry,
        quantizer: ActionQuantizer,
    ) -> list[list[int]]:
        if "ego_future_xyz" not in batch or "ego_future_rot" not in batch:
            return super().encode_target_token_rows_from_batch(batch, registry, quantizer)
        token_rows = quantizer.encode(
            hist_xyz=batch["ego_history_xyz"].to(dtype=torch.float32),
            hist_rot=batch["ego_history_rot"].to(dtype=torch.float32),
            fut_xyz=batch["ego_future_xyz"].to(dtype=torch.float32),
            fut_rot=batch["ego_future_rot"].to(dtype=torch.float32),
        )
        return [registry.encode_bin_token_ids(row.tolist()) for row in token_rows.detach().cpu()]


@dataclass(frozen=True)
class KappaOnlyStage1Spec(Stage1TaskSpec):
    """Steer-only Stage 1 target that predicts curvature bins only."""

    name: str = "steer_only"
    action_representation: str = "kappa_only"
    rollout_accel_source: str = "ground_truth"
    n_bins: int = 256
    range_percentile: float = 99.5
    range_margin_ratio: float = 0.1

    def build_quantizer(self, train_dataset) -> KappaOnlyQuantizer:
        kappa_values: list[float] = []
        for record in iter_stage1_records(train_dataset):
            action = _record_action_array(record)
            kappa_values.extend(float(value) for value in action[1::2])
        if not kappa_values:
            raise RuntimeError("Steer-only Stage 1 requires non-empty kappa values in the train corpus.")

        kappa_array = np.asarray(kappa_values, dtype=np.float32)
        observed_abs = float(np.percentile(np.abs(kappa_array), self.range_percentile))
        derived_radius = max(observed_abs * (1.0 + self.range_margin_ratio), 1e-6)
        return KappaOnlyQuantizer(
            n_bins=self.n_bins,
            kappa_range=(-derived_radius, derived_radius),
            range_source="train_corpus_percentile",
            range_percentile=self.range_percentile,
            range_margin_ratio=self.range_margin_ratio,
            observed_min=float(kappa_array.min()),
            observed_max=float(kappa_array.max()),
            derived_radius=derived_radius,
            num_values=int(kappa_array.size),
        )

    def quantizer_from_checkpoint(self, quantizer_payload: dict) -> KappaOnlyQuantizer:
        required_keys = [
            "n_bins",
            "kappa_range",
            "range_source",
            "range_percentile",
            "range_margin_ratio",
            "observed_min",
            "observed_max",
            "derived_radius",
            "num_values",
        ]
        missing_keys = [key for key in required_keys if key not in quantizer_payload]
        if missing_keys:
            raise RuntimeError(
                "Steer-only Stage 1 checkpoint is missing quantizer metadata:\n" + "\n".join(missing_keys)
            )
        return KappaOnlyQuantizer(
            n_bins=int(quantizer_payload["n_bins"]),
            kappa_range=tuple(quantizer_payload["kappa_range"]),
            range_source=str(quantizer_payload["range_source"]),
            range_percentile=float(quantizer_payload["range_percentile"]),
            range_margin_ratio=float(quantizer_payload["range_margin_ratio"]),
            observed_min=float(quantizer_payload["observed_min"]),
            observed_max=float(quantizer_payload["observed_max"]),
            derived_radius=float(quantizer_payload["derived_radius"]),
            num_values=int(quantizer_payload["num_values"]),
        )

    def target_from_action_array(self, action: np.ndarray) -> np.ndarray:
        return np.asarray(action[1::2], dtype=np.float32)

    def target_from_action_tensor(self, action: torch.Tensor) -> torch.Tensor:
        return action[..., 1::2].to(torch.float32)

    def full_action_from_target_tensor(
        self,
        target: torch.Tensor,
        *,
        gt_action_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if gt_action_tensor is None:
            raise RuntimeError("Steer-only Stage 1 rollout requires the canonical ground-truth acceleration.")
        flat_target = target.reshape(-1).to(torch.float32)
        flat_gt_action = gt_action_tensor.reshape(-1).to(torch.float32)
        if flat_gt_action.numel() != flat_target.numel() * 2:
            raise RuntimeError(
                "Ground-truth action does not match the steer-only target length for rollout reconstruction."
            )
        full_action = torch.empty(flat_gt_action.numel(), dtype=torch.float32)
        full_action[0::2] = flat_gt_action[0::2]
        full_action[1::2] = flat_target
        return full_action

    def quantizer_metadata(self, quantizer: KappaOnlyQuantizer) -> dict[str, Any]:
        return {
            "quantizer_kind": "kappa_only",
            "n_bins": quantizer.n_bins,
            "kappa_range": list(quantizer.kappa_range),
            "range_source": quantizer.range_source,
            "range_percentile": quantizer.range_percentile,
            "range_margin_ratio": quantizer.range_margin_ratio,
            "observed_min": quantizer.observed_min,
            "observed_max": quantizer.observed_max,
            "derived_radius": quantizer.derived_radius,
            "num_values": quantizer.num_values,
        }
