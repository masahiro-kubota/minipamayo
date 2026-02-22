"""Stage 0 evaluation: load checkpoint and sanity-check predictions.

Evaluates on the full dataset (no train/val split) to verify the training
code works correctly.

Usage:
    cd minipamayo && uv run python -m minipamayo.eval_stage0 \
        --checkpoint checkpoints/stage0/best.pt
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data.nuscenes_dataset import NuScenesSteerThrottleDataset
from .models.minipamayo import MiniPamayo


def parse_args():
    parser = argparse.ArgumentParser(description="MiniPamayo Stage 0 evaluation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint (e.g. checkpoints/stage0/best.pt)",
    )
    parser.add_argument(
        "--nuscenes_root",
        type=str,
        default="/mnt/ssd/nuscenes",
    )
    parser.add_argument("--nuscenes_version", type=str, default="v1.0-trainval")
    parser.add_argument(
        "--adapter_type", type=str, default="mean_pool", choices=["mean_pool", "per_token"]
    )
    parser.add_argument(
        "--show_samples", type=int, default=20, help="Number of per-sample predictions to show"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Full dataset (no split — sanity check only)
    print("Loading nuScenes dataset...")
    dataset = NuScenesSteerThrottleDataset(
        nuscenes_root=args.nuscenes_root,
        version=args.nuscenes_version,
    )
    print(f"Total samples: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # Model
    print("Building model...")
    model = MiniPamayo(adapter_type=args.adapter_type, action_dim=2)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}, Step: {ckpt.get('global_step', '?')}")
    if "metrics" in ckpt:
        print(f"  Saved metrics: {ckpt['metrics']}")

    # Evaluate
    print("\n" + "=" * 70)
    print(f"Per-sample predictions (first {args.show_samples} samples)")
    print("=" * 70)
    print(
        f"{'Idx':>4} | {'Pred Steer':>11} {'GT Steer':>11} {'Err':>8} | "
        f"{'Pred Throt':>11} {'GT Throt':>11} {'Err':>8}"
    )
    print("-" * 70)

    all_preds = []
    all_gts = []
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

            if i < args.show_samples:
                s_err = abs(pred[0].item() - gt[0].item())
                t_err = abs(pred[1].item() - gt[1].item())
                print(
                    f"{i:4d} | "
                    f"{pred[0]:+11.5f} {gt[0]:+11.5f} {s_err:8.5f} | "
                    f"{pred[1]:+11.5f} {gt[1]:+11.5f} {t_err:8.5f}"
                )

    preds = torch.stack(all_preds)  # (N, 2)
    gts = torch.stack(all_gts)  # (N, 2)
    errors = (preds - gts).abs()

    # Overall metrics
    print("\n" + "=" * 70)
    print("Overall Metrics")
    print("=" * 70)
    print(f"  Loss (Huber):        {total_loss / len(loader):.6f}")
    print(f"  Steer MAE:           {errors[:, 0].mean():.6f}")
    print(f"  Throttle MAE:        {errors[:, 1].mean():.6f}")

    # Distribution statistics
    print("\n" + "=" * 70)
    print("Distribution Statistics")
    print("=" * 70)
    for ch, name in [(0, "Steer"), (1, "Throttle")]:
        p = preds[:, ch]
        g = gts[:, ch]
        print(f"\n  {name}:")
        print(
            f"    GT       — mean: {g.mean():+.5f}, std: {g.std():.5f}, "
            f"min: {g.min():+.5f}, max: {g.max():+.5f}"
        )
        print(
            f"    Pred     — mean: {p.mean():+.5f}, std: {p.std():.5f}, "
            f"min: {p.min():+.5f}, max: {p.max():+.5f}"
        )
        print(
            f"    Error    — mean: {errors[:, ch].mean():.5f}, "
            f"max: {errors[:, ch].max():.5f}, "
            f"median: {errors[:, ch].median():.5f}"
        )

    # Worst predictions
    print("\n" + "=" * 70)
    print("Worst 5 predictions (by total error)")
    print("=" * 70)
    total_err = errors.sum(dim=1)
    worst_indices = total_err.argsort(descending=True)[:5]
    for rank, idx in enumerate(worst_indices):
        i = idx.item()
        p = preds[i]
        g = gts[i]
        img_path = dataset.samples[i]["image_path"]
        img_name = Path(img_path).name
        print(
            f"  #{rank + 1} (sample {i}, {img_name}): "
            f"steer {p[0]:+.5f} vs {g[0]:+.5f} (err {errors[i, 0]:.5f}), "
            f"throttle {p[1]:+.5f} vs {g[1]:+.5f} (err {errors[i, 1]:.5f})"
        )

    # Input dependency check
    print("\n" + "=" * 70)
    print("Input Dependency")
    print("=" * 70)
    pred_var = preds.var(dim=0)
    print(f"  Pred variance — steer: {pred_var[0]:.6f}, throttle: {pred_var[1]:.6f}")
    collapsed = pred_var.sum() < 1e-8
    print(f"  Status: {'COLLAPSED (all same output!)' if collapsed else 'OK (input-dependent)'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
