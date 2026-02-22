"""Stage 0 Phase 3: fail-fast pipeline validation.

Verifies that gradients flow through DINO → Adapter → LLM → ActionHead.
Predicts [steer, throttle] from a single CAM_FRONT image.

Usage:
    cd minipamayo && uv run python -m minipamayo.train_stage0 \
        --nuscenes_root /mnt/ssd/nuscenes \
        --checkpoint ../cosmos-reason-mini/checkpoints/rl-mini-merged/checkpoint-final.pt
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from .data.nuscenes_dataset import NuScenesSteerThrottleDataset
from .models.minipamayo import MiniPamayo


def parse_args():
    parser = argparse.ArgumentParser(description="MiniPamayo Stage 0 Phase 3 training")
    parser.add_argument(
        "--nuscenes_root",
        type=str,
        default="/mnt/ssd/nuscenes",
    )
    parser.add_argument("--nuscenes_version", type=str, default="v1.0-trainval")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../cosmos-reason-mini/checkpoints/rl-mini-merged/checkpoint-final.pt",
    )
    parser.add_argument(
        "--adapter_type", type=str, default="mean_pool", choices=["mean_pool", "per_token"]
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=1, help="Log every N optimizer steps")
    parser.add_argument("--save_dir", type=str, default="checkpoints/stage0")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="minipamayo")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, dataloader, device) -> dict:
    """Evaluate on val set. Returns loss and per-channel MAE."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_gts = []
    n = 0

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        gt_action = batch["action"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred_action = model(pixel_values)
            loss = F.huber_loss(pred_action.float(), gt_action, delta=1.0)

        total_loss += loss.item()
        all_preds.append(pred_action.float().cpu())
        all_gts.append(gt_action.cpu())
        n += 1

    preds = torch.cat(all_preds, dim=0)  # (N, 2)
    gts = torch.cat(all_gts, dim=0)  # (N, 2)
    errors = (preds - gts).abs()

    return {
        "val_loss": total_loss / max(n, 1),
        "steer_mae": errors[:, 0].mean().item(),
        "throttle_mae": errors[:, 1].mean().item(),
        "steer_std_pred": preds[:, 0].std().item(),
        "throttle_std_pred": preds[:, 1].std().item(),
        "steer_std_gt": gts[:, 0].std().item(),
        "throttle_std_gt": gts[:, 1].std().item(),
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.use_wandb:
        import wandb

        wandb.init(project=args.wandb_project, name="stage0-phase3", config=vars(args))

    # Dataset with train/val split
    print("Loading nuScenes dataset...")
    full_dataset = NuScenesSteerThrottleDataset(
        nuscenes_root=args.nuscenes_root,
        version=args.nuscenes_version,
    )
    n_val = int(len(full_dataset) * args.val_ratio)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    print(f"Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Model
    print("Building model...")
    model = MiniPamayo(adapter_type=args.adapter_type, action_dim=2)
    if args.checkpoint and Path(args.checkpoint).exists():
        model.load_vlm_checkpoint(args.checkpoint)
    model.set_stage0()
    model.enable_gradient_checkpointing()
    model = model.to(device)

    # Parameter count
    param_info = model.count_parameters()
    total_params = sum(v["total"] for v in param_info.values())
    trainable_params = sum(v["trainable"] for v in param_info.values())
    print("\nParameters:")
    for name, info in param_info.items():
        print(f"  {name}: {info['total']:,} total, {info['trainable']:,} trainable")
    print(f"  TOTAL: {total_params:,} total, {trainable_params:,} trainable\n")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.max_epochs // args.grad_accum
    warmup_steps = max(1, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(progress * math.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Initial evaluation (before training)
    print("=== Initial Evaluation (before training) ===")
    init_metrics = evaluate(model, val_loader, device)
    for k, v in init_metrics.items():
        print(f"  {k}: {v:.6f}")

    # Training loop
    print("\n=== Starting Training ===")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_val_loss = float("inf")
    optimizer.zero_grad()

    for epoch in range(args.max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            gt_action = batch["action"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred_action = model(pixel_values)
                loss = F.huber_loss(pred_action.float(), gt_action, delta=1.0)
                loss = loss / args.grad_accum

            loss.backward()
            epoch_loss += loss.item() * args.grad_accum
            epoch_samples += 1

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                global_step += 1

                if global_step % args.log_every == 0:
                    avg_loss = epoch_loss / epoch_samples
                    lr = scheduler.get_last_lr()[0]

                    # Per-module gradient norms (before optimizer.step/zero_grad)
                    grad_norms = {}
                    for name, module in [
                        ("VE", model.vision_encoder),
                        ("Adp", model.adapter),
                        ("LLM", model.llm),
                        ("AH", model.action_head),
                    ]:
                        grads = [
                            p.grad.norm().item() for p in module.parameters() if p.grad is not None
                        ]
                        grad_norms[name] = sum(grads) / len(grads) if grads else 0.0

                    gn_str = " ".join(f"{k}={v:.4f}" for k, v in grad_norms.items())
                    print(
                        f"[E{epoch + 1}] Step {global_step:3d} | "
                        f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                        f"Grad: {gn_str}"
                    )

                    if args.use_wandb:
                        import wandb

                        wandb.log(
                            {
                                "train/loss": avg_loss,
                                "train/lr": lr,
                                **{f"grad_norm/{k}": v for k, v in grad_norms.items()},
                            },
                            step=global_step,
                        )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # Epoch evaluation
        metrics = evaluate(model, val_loader, device)
        avg_train_loss = epoch_loss / max(epoch_samples, 1)

        print(
            f"\n=== Epoch {epoch + 1}/{args.max_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {metrics['val_loss']:.4f} | "
            f"Steer MAE: {metrics['steer_mae']:.5f} | "
            f"Throttle MAE: {metrics['throttle_mae']:.5f} ===\n"
        )

        if args.use_wandb:
            import wandb

            wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=global_step)

        # Save best
        if metrics["val_loss"] < best_val_loss:
            best_val_loss = metrics["val_loss"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                },
                save_dir / "best.pt",
            )

    # Save final
    torch.save(
        {
            "epoch": args.max_epochs,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        save_dir / "final.pt",
    )

    # Final evaluation with detailed output
    print("\n=== Final Evaluation ===")
    final_metrics = evaluate(model, val_loader, device)
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.6f}")

    # Compare with initial
    print("\n=== Improvement ===")
    print(f"  Val loss:      {init_metrics['val_loss']:.6f} → {final_metrics['val_loss']:.6f}")
    print(f"  Steer MAE:     {init_metrics['steer_mae']:.6f} → {final_metrics['steer_mae']:.6f}")
    print(
        f"  Throttle MAE:  {init_metrics['throttle_mae']:.6f} → {final_metrics['throttle_mae']:.6f}"
    )

    # Input-dependent output check
    print("\n=== Input Dependency Check (different images → different outputs?) ===")
    model.eval()
    with torch.no_grad():
        outputs = []
        for i in range(min(5, len(val_dataset))):
            sample = val_dataset[i]
            pv = sample["pixel_values"].unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(pv)
            outputs.append(out.float().cpu().squeeze())
            gt = sample["action"]
            print(
                f"  Sample {i}: pred=[{out[0, 0]:.4f}, {out[0, 1]:.4f}], gt=[{gt[0]:.4f}, {gt[1]:.4f}]"
            )

        # Check variance across predictions
        stacked = torch.stack(outputs)
        print(
            f"  Pred variance: steer={stacked[:, 0].var():.6f}, throttle={stacked[:, 1].var():.6f}"
        )
        all_same = stacked.var(dim=0).sum() < 1e-8
        print(f"  Input-dependent: {'NO (all same!)' if all_same else 'YES'}")

    # Gradient flow verification
    print("\n=== Gradient Flow Verification ===")
    for name, module in [
        ("vision_encoder", model.vision_encoder),
        ("adapter", model.adapter),
        ("llm", model.llm),
        ("action_head", model.action_head),
    ]:
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
        print(f"  {name}: {'OK' if has_grad else 'NO GRADIENT'}")

    if torch.cuda.is_available():
        print(f"\nPeak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    print("\nDone.")


if __name__ == "__main__":
    main()
