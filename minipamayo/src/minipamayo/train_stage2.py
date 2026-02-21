"""Stage 2: Flow Matching trajectory decoder training.

Freezes VLM (VisionEncoder + Adapter + LLM) as feature extractor.
Trains TrajectoryDecoder with Conditional Flow Matching loss.

Usage:
    cd minipamayo && uv run python -m minipamayo.train_stage2 \
        --phase4_checkpoint checkpoints/phase4/best.pt
"""

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from .data.nuscenes_trajectory_dataset import NuScenesTrajectoryDataset
from .models.minipamayo import MiniPamayo
from .models.trajectory_decoder import TrajectoryDecoder, cfm_loss


def parse_args():
    parser = argparse.ArgumentParser(description="MiniPamayo Stage 2 training")
    parser.add_argument("--nuscenes_root", type=str, default="../cosmos-reason-mini/data/nuscenes")
    parser.add_argument("--nuscenes_version", type=str, default="v1.0-mini")
    parser.add_argument(
        "--phase4_checkpoint",
        type=str,
        default="checkpoints/phase4/best.pt",
        help="Phase 4 checkpoint for frozen VLM",
    )
    parser.add_argument(
        "--vlm_checkpoint",
        type=str,
        default="../cosmos-reason-mini/checkpoints/rl-mini-merged/checkpoint-final.pt",
        help="Fallback VLM checkpoint if phase4 not found",
    )
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--condition_dim", type=int, default=896)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="checkpoints/stage2")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    return parser.parse_args()


@torch.no_grad()
def extract_conditions(vlm, pixel_values):
    """Extract LLM hidden states as condition for flow decoder.

    Args:
        vlm: frozen MiniPamayo model
        pixel_values: (B, 3, 224, 224)

    Returns:
        condition: (B, condition_dim) mean-pooled LLM hidden states
    """
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        patch_features = vlm.vision_encoder(pixel_values)
        visual_tokens = vlm.adapter(patch_features)
        outputs = vlm.llm(inputs_embeds=visual_tokens, output_hidden_states=True)
    # Mean pool over sequence length for stable conditioning
    last_hidden = outputs.hidden_states[-1]  # (B, N_vis, 896)
    condition = last_hidden.mean(dim=1).float()  # (B, 896)
    return condition


@torch.no_grad()
def evaluate(decoder, vlm, dataloader, device):
    """Evaluate CFM loss on validation set."""
    decoder.eval()
    total_loss = 0.0
    n = 0

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        gt_action = batch["action"].to(device)

        condition = extract_conditions(vlm, pixel_values)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = cfm_loss(decoder, gt_action, condition)
        total_loss += loss.item()
        n += 1

    return {"val_cfm_loss": total_loss / max(n, 1)}


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_dim = args.K * 2
    print(f"Device: {device}")
    print(f"K={args.K}, action_dim={action_dim}")

    # Dataset
    print("Loading dataset...")
    full_dataset = NuScenesTrajectoryDataset(
        nuscenes_root=args.nuscenes_root,
        version=args.nuscenes_version,
        K=args.K,
    )
    n_val = int(len(full_dataset) * args.val_ratio)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    print(f"Total: {len(full_dataset)}, Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Frozen VLM (feature extractor)
    print("Building frozen VLM...")
    vlm = MiniPamayo(adapter_type="cross_attention", action_dim=action_dim)

    phase4_path = Path(args.phase4_checkpoint)
    vlm_path = Path(args.vlm_checkpoint)

    if phase4_path.exists():
        print(f"Loading Phase 4 checkpoint: {phase4_path}")
        ckpt = torch.load(phase4_path, map_location="cpu", weights_only=True)
        state_dict = ckpt["model_state_dict"]
        missing, unexpected = vlm.load_state_dict(state_dict, strict=False)
        print(f"  Missing: {len(missing)} keys, Unexpected: {len(unexpected)} keys")
    elif vlm_path.exists():
        print(f"Phase 4 not found, loading VLM checkpoint: {vlm_path}")
        vlm.load_vlm_checkpoint(vlm_path)
    else:
        print("WARNING: No checkpoint found, using random VLM weights")

    vlm.requires_grad_(False)
    vlm.eval()
    vlm = vlm.to(device)

    # Flow Matching Decoder (trainable)
    print("Building TrajectoryDecoder...")
    decoder = TrajectoryDecoder(
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        condition_dim=args.condition_dim,
    )
    decoder = decoder.to(device)

    n_params = sum(p.numel() for p in decoder.parameters())
    n_trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Decoder: {n_params:,} total, {n_trainable:,} trainable")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.max_epochs // args.grad_accum
    warmup_steps = 500

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(progress * math.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Initial evaluation
    print("\n=== Initial Evaluation ===")
    init_metrics = evaluate(decoder, vlm, val_loader, device)
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
        decoder.train()
        epoch_loss = 0.0
        epoch_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            gt_action = batch["action"].to(device)

            # Extract condition from frozen VLM
            condition = extract_conditions(vlm, pixel_values)

            # CFM loss (bf16 for decoder)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = cfm_loss(decoder, gt_action, condition)
            loss = loss / args.grad_accum
            loss.backward()

            epoch_loss += loss.item() * args.grad_accum
            epoch_samples += 1

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.max_grad_norm)
                global_step += 1

                if global_step % args.log_every == 0:
                    avg_loss = epoch_loss / epoch_samples
                    lr = scheduler.get_last_lr()[0]
                    print(
                        f"[E{epoch + 1}] Step {global_step:3d} | "
                        f"CFM Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                    )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # Epoch evaluation
        metrics = evaluate(decoder, vlm, val_loader, device)
        avg_train_loss = epoch_loss / max(epoch_samples, 1)

        print(
            f"\n=== Epoch {epoch + 1}/{args.max_epochs} | "
            f"Train CFM: {avg_train_loss:.4f} | "
            f"Val CFM: {metrics['val_cfm_loss']:.4f} ===\n"
        )

        if metrics["val_cfm_loss"] < best_val_loss:
            best_val_loss = metrics["val_cfm_loss"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "decoder_state_dict": decoder.state_dict(),
                    "metrics": metrics,
                    "K": args.K,
                    "action_dim": action_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "num_heads": args.num_heads,
                    "condition_dim": args.condition_dim,
                },
                save_dir / "best.pt",
            )

    torch.save(
        {
            "epoch": args.max_epochs,
            "global_step": global_step,
            "decoder_state_dict": decoder.state_dict(),
            "metrics": metrics,
            "K": args.K,
            "action_dim": action_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "condition_dim": args.condition_dim,
        },
        save_dir / "final.pt",
    )

    # Summary
    print("\n=== Summary ===")
    final_metrics = evaluate(decoder, vlm, val_loader, device)
    print(f"  Val CFM: {init_metrics['val_cfm_loss']:.4f} -> {final_metrics['val_cfm_loss']:.4f}")
    print(f"  Decoder params: {n_trainable:,}")

    if torch.cuda.is_available():
        print(f"\nPeak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    print("\nDone.")


if __name__ == "__main__":
    main()
