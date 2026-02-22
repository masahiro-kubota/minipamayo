"""Phase 4: Stage 0 with control-based representation.

CrossAttentionAdapter (16 queries) + (a, kappa) x K trajectory prediction.
Builds on Phase 3 (steer/throttle) to use proper control-based actions.

Usage:
    cd minipamayo && uv run python -m minipamayo.train_phase4 \
        --checkpoint ../cosmos-reason-mini/checkpoints/rl-mini-merged/checkpoint-final.pt
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data.nuscenes_trajectory_dataset import NuScenesTrajectoryDataset
from .models.minipamayo import MiniPamayo


def parse_args():
    parser = argparse.ArgumentParser(description="MiniPamayo Phase 4 training")
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
    parser.add_argument("--K", type=int, default=6, help="Number of future waypoints")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="checkpoints/phase4")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="minipamayo")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, dataloader, device, K) -> dict:
    """Evaluate: Huber loss + per-channel MAE (a, kappa)."""
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

    preds = torch.cat(all_preds, dim=0)  # (N, K*2)
    gts = torch.cat(all_gts, dim=0)

    # Reshape to (N, K, 2) for per-channel analysis
    preds_kv = preds.reshape(-1, K, 2)
    gts_kv = gts.reshape(-1, K, 2)
    errors_kv = (preds_kv - gts_kv).abs()

    return {
        "val_loss": total_loss / max(n, 1),
        "a_mae": errors_kv[:, :, 0].mean().item(),
        "kappa_mae": errors_kv[:, :, 1].mean().item(),
        "action_mae": (preds - gts).abs().mean().item(),
        "a_std_pred": preds_kv[:, :, 0].std().item(),
        "kappa_std_pred": preds_kv[:, :, 1].std().item(),
    }


def main():
    t_start = time.time()
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"K (waypoints): {args.K}, action_dim: {args.K * 2}")

    if args.use_wandb:
        import wandb

        wandb.init(project=args.wandb_project, name="phase4-ctrl", config=vars(args))

    # Dataset (scene-level split)
    print("Loading nuScenes trajectory dataset...")
    use_split = args.nuscenes_version == "v1.0-trainval"
    train_dataset = NuScenesTrajectoryDataset(
        nuscenes_root=args.nuscenes_root,
        version=args.nuscenes_version,
        K=args.K,
        split="train" if use_split else None,
    )
    val_dataset = NuScenesTrajectoryDataset(
        nuscenes_root=args.nuscenes_root,
        version=args.nuscenes_version,
        K=args.K,
        split="val" if use_split else None,
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Model with CrossAttentionAdapter
    print("Building model...")
    model = MiniPamayo(adapter_type="cross_attention", action_dim=args.K * 2)
    if args.checkpoint and Path(args.checkpoint).exists():
        model.load_vlm_checkpoint(args.checkpoint)
    model.set_stage0()  # all modules trainable
    model.enable_gradient_checkpointing()
    model = model.to(device)

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

    # Initial evaluation
    print("=== Initial Evaluation ===")
    init_metrics = evaluate(model, val_loader, device, args.K)
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
        t_epoch = time.time()
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
        metrics = evaluate(model, val_loader, device, args.K)
        avg_train_loss = epoch_loss / max(epoch_samples, 1)

        epoch_time = time.time() - t_epoch
        print(
            f"\n=== Epoch {epoch + 1}/{args.max_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {metrics['val_loss']:.4f} | "
            f"a MAE: {metrics['a_mae']:.5f} | "
            f"kappa MAE: {metrics['kappa_mae']:.5f} | "
            f"Time: {epoch_time:.0f}s ===\n"
        )

        if args.use_wandb:
            import wandb

            wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=global_step)

        if metrics["val_loss"] < best_val_loss:
            best_val_loss = metrics["val_loss"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                    "K": args.K,
                    "adapter_type": "cross_attention",
                },
                save_dir / "best.pt",
            )

    torch.save(
        {
            "epoch": args.max_epochs,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "K": args.K,
            "adapter_type": "cross_attention",
        },
        save_dir / "final.pt",
    )

    # Summary
    print("\n=== Improvement ===")
    final_metrics = evaluate(model, val_loader, device, args.K)
    print(f"  Val loss:   {init_metrics['val_loss']:.6f} -> {final_metrics['val_loss']:.6f}")
    print(f"  a MAE:      {init_metrics['a_mae']:.6f} -> {final_metrics['a_mae']:.6f}")
    print(f"  kappa MAE:  {init_metrics['kappa_mae']:.6f} -> {final_metrics['kappa_mae']:.6f}")

    # Input dependency check
    print("\n=== Input Dependency ===")
    model.eval()
    with torch.no_grad():
        outputs = []
        for i in range(min(5, len(val_dataset))):
            sample = val_dataset[i]
            pv = sample["pixel_values"].unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(pv)
            outputs.append(out.float().cpu().squeeze())

        stacked = torch.stack(outputs)
        pred_var = stacked.var(dim=0).mean()
        print(f"  Mean pred variance: {pred_var:.6f}")
        print(f"  Input-dependent: {'NO' if pred_var < 1e-8 else 'YES'}")

    # Gradient flow
    print("\n=== Gradient Flow ===")
    for name, module in [
        ("vision_encoder", model.vision_encoder),
        ("adapter", model.adapter),
        ("llm", model.llm),
        ("action_head", model.action_head),
    ]:
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
        print(f"  {name}: {'OK' if has_grad else 'NO GRADIENT'}")

    total_time = time.time() - t_start
    if torch.cuda.is_available():
        print(f"\nPeak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    print(f"Total time: {total_time:.0f}s ({total_time / 60:.1f}min)")
    print("\nDone.")


if __name__ == "__main__":
    main()
