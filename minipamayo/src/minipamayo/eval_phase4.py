"""Phase 4 evaluation: control-based trajectory prediction.

Computes action MAE, ADE, FDE using forward dynamics.

Usage:
    cd minipamayo && uv run python -m minipamayo.eval_phase4 \
        --checkpoint checkpoints/phase4/best.pt
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data.nuscenes_trajectory_dataset import NuScenesTrajectoryDataset
from .models.dynamics import forward_dynamics_batch
from .models.minipamayo import MiniPamayo


def parse_args():
    parser = argparse.ArgumentParser(description="MiniPamayo Phase 4 evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--nuscenes_root",
        type=str,
        default="../cosmos-reason-mini/data/nuscenes",
    )
    parser.add_argument("--nuscenes_version", type=str, default="v1.0-mini")
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--show_samples", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset (full, no split)
    print("Loading dataset...")
    dataset = NuScenesTrajectoryDataset(
        nuscenes_root=args.nuscenes_root,
        version=args.nuscenes_version,
        K=args.K,
    )
    print(f"Total samples: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # Model
    print("Building model...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    K = ckpt.get("K", args.K)
    adapter_type = ckpt.get("adapter_type", "cross_attention")

    model = MiniPamayo(adapter_type=adapter_type, action_dim=K * 2)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded: {args.checkpoint}")
    print(f"  K={K}, adapter={adapter_type}")
    if "metrics" in ckpt:
        print(f"  Saved metrics: {ckpt['metrics']}")

    # Evaluate
    print(f"\n{'=' * 70}")
    print(f"Per-sample predictions (first {args.show_samples})")
    print(f"{'=' * 70}")

    all_preds = []
    all_gts = []
    all_v0 = []
    all_gt_waypoints = []
    total_loss = 0.0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            pixel_values = batch["pixel_values"].to(device)
            gt_action = batch["action"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred_action = model(pixel_values)
                loss = F.huber_loss(pred_action.float(), gt_action, delta=1.0)

            pred = pred_action.float().cpu().squeeze()
            gt = gt_action.cpu().squeeze()
            total_loss += loss.item()
            all_preds.append(pred)
            all_gts.append(gt)
            all_v0.append(batch["v0"].squeeze())
            all_gt_waypoints.append(batch["gt_waypoints"].squeeze())

            if i < args.show_samples:
                pred_kv = pred.reshape(K, 2)
                gt_kv = gt.reshape(K, 2)
                a_err = (pred_kv[:, 0] - gt_kv[:, 0]).abs().mean().item()
                k_err = (pred_kv[:, 1] - gt_kv[:, 1]).abs().mean().item()
                print(f"  [{i:3d}] a_MAE={a_err:.4f}  kappa_MAE={k_err:.4f}")

    preds = torch.stack(all_preds)  # (N, K*2)
    gts = torch.stack(all_gts)
    v0s = torch.stack(all_v0)  # (N,)
    gt_waypoints = torch.stack(all_gt_waypoints)  # (N, K, 2)

    # Reshape to (N, K, 2) for analysis
    preds_kv = preds.reshape(-1, K, 2)
    gts_kv = gts.reshape(-1, K, 2)
    errors_kv = (preds_kv - gts_kv).abs()

    # Action-space metrics
    print(f"\n{'=' * 70}")
    print("Action-Space Metrics")
    print(f"{'=' * 70}")
    print(f"  Huber Loss:    {total_loss / len(loader):.6f}")
    print(f"  a MAE:         {errors_kv[:, :, 0].mean():.6f}")
    print(f"  kappa MAE:     {errors_kv[:, :, 1].mean():.6f}")

    # Trajectory metrics (ADE/FDE)
    pred_a = preds_kv[:, :, 0]  # (N, K)
    pred_kappa = preds_kv[:, :, 1]  # (N, K)
    pred_waypoints = forward_dynamics_batch(pred_a, pred_kappa, v0s, dt=0.1)  # (N, K, 2)

    displacement_errors = torch.norm(pred_waypoints - gt_waypoints, dim=2)  # (N, K)
    ade = displacement_errors.mean().item()
    fde = displacement_errors[:, -1].mean().item()

    print(f"\n{'=' * 70}")
    print("Trajectory Metrics (via forward dynamics)")
    print(f"{'=' * 70}")
    print(f"  ADE:           {ade:.4f} m")
    print(f"  FDE:           {fde:.4f} m")

    # Per-timestep ADE
    print("\n  Per-timestep ADE:")
    for t in range(K):
        t_ade = displacement_errors[:, t].mean().item()
        print(f"    t={t} ({(t + 1) * 0.1:.1f}s): {t_ade:.4f} m")

    # Distribution
    print(f"\n{'=' * 70}")
    print("Distribution Statistics")
    print(f"{'=' * 70}")
    for ch, name in [(0, "Acceleration (a)"), (1, "Curvature (kappa)")]:
        p = preds_kv[:, :, ch]
        g = gts_kv[:, :, ch]
        print(f"\n  {name}:")
        print(
            f"    GT   — mean: {g.mean():+.5f}, std: {g.std():.5f}, "
            f"min: {g.min():+.5f}, max: {g.max():+.5f}"
        )
        print(
            f"    Pred — mean: {p.mean():+.5f}, std: {p.std():.5f}, "
            f"min: {p.min():+.5f}, max: {p.max():+.5f}"
        )

    # Worst predictions
    print(f"\n{'=' * 70}")
    print("Worst 5 (by FDE)")
    print(f"{'=' * 70}")
    fde_per_sample = displacement_errors[:, -1]
    worst_indices = fde_per_sample.argsort(descending=True)[:5]
    for rank, idx in enumerate(worst_indices):
        i = idx.item()
        img_name = Path(dataset.samples[i]["image_path"]).name
        print(
            f"  #{rank + 1} (sample {i}, {img_name}): "
            f"FDE={fde_per_sample[i]:.3f}m, "
            f"ADE={displacement_errors[i].mean():.3f}m"
        )

    # Input dependency
    print(f"\n{'=' * 70}")
    print("Input Dependency")
    print(f"{'=' * 70}")
    pred_var = preds.var(dim=0).mean()
    print(f"  Mean pred variance: {pred_var:.6f}")
    print(f"  Status: {'COLLAPSED' if pred_var < 1e-8 else 'OK (input-dependent)'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
