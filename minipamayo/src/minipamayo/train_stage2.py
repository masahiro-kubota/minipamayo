"""Stage 2: Flow Matching Expert training (Alpamayo-faithful).

Freezes VLM (VisionEncoder + Adapter + LLM) and trains Expert Transformer
with Conditional Flow Matching loss. Expert is conditioned on VLM KV-cache
via past_key_values (Alpamayo §5.1-5.2).

Usage:
    cd minipamayo && uv run python -m minipamayo.train_stage2 \
        --phase4_checkpoint checkpoints/phase4/best.pt

    # Full config (24 layers, 896 hidden, ~280M)
    uv run python -m minipamayo.train_stage2 \
        --hidden_size 896 --num_attention_heads 14
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data.nuscenes_trajectory_dataset import NuScenesTrajectoryDataset
from .models.minipamayo import MiniPamayo
from .models.trajectory_decoder import TrajectoryDecoder, cfm_loss


def parse_args():
    parser = argparse.ArgumentParser(description="MiniPamayo Stage 2 Expert training")
    parser.add_argument("--nuscenes_root", type=str, default="/mnt/ssd/nuscenes")
    parser.add_argument("--nuscenes_version", type=str, default="v1.0-trainval")
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
    parser.add_argument("--K", type=int, default=6)
    # Expert architecture (must satisfy: hidden_size = num_attention_heads * 64)
    parser.add_argument(
        "--hidden_size", type=int, default=640, help="Expert hidden dim (must be heads*64)"
    )
    parser.add_argument("--num_hidden_layers", type=int, default=24, help="Must match VLM (24)")
    parser.add_argument("--num_attention_heads", type=int, default=10, help="Expert Q heads")
    parser.add_argument(
        "--intermediate_size", type=int, default=None, help="FFN dim (default: hidden*4)"
    )
    # Fourier encoding
    parser.add_argument("--num_fourier_feats", type=int, default=20)
    parser.add_argument("--fourier_max_freq", type=float, default=100.0)
    parser.add_argument("--mlp_hidden_size", type=int, default=1024)
    parser.add_argument("--mlp_num_layers", type=int, default=4)
    # Training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="checkpoints/stage2")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="minipamayo")
    # Data augmentation
    parser.add_argument(
        "--curve_oversample",
        type=float,
        default=3.0,
        help="Weight multiplier for curve scenes (kappa > threshold)",
    )
    parser.add_argument(
        "--curve_kappa_threshold",
        type=float,
        default=0.01,
        help="Kappa threshold to identify curve scenes",
    )
    return parser.parse_args()


@torch.no_grad()
def extract_kv_cache(vlm, pixel_values):
    """Extract VLM KV-cache as condition for Expert decoder (Alpamayo §5.1).

    Runs VLM forward with use_cache=True to produce past_key_values,
    which are passed directly to the Expert Transformer.

    Args:
        vlm: frozen MiniPamayo model
        pixel_values: (B, 3, 224, 224)

    Returns:
        kv_cache: DynamicCache with VLM KV-cache
        prefill_seq_len: int, sequence length of the KV-cache
    """
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        patch_features = vlm.vision_encoder(pixel_values)
        visual_tokens = vlm.adapter(patch_features)
        outputs = vlm.llm(inputs_embeds=visual_tokens, use_cache=True)
    kv_cache = outputs.past_key_values
    prefill_seq_len = kv_cache.get_seq_length()
    return kv_cache, prefill_seq_len


@torch.no_grad()
def evaluate(decoder, vlm, dataloader, device):
    """Evaluate CFM loss on validation set."""
    decoder.eval()
    total_loss = 0.0
    n = 0

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        gt_action = batch["action"].to(device)

        kv_cache, prefill_seq_len = extract_kv_cache(vlm, pixel_values)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = cfm_loss(decoder, gt_action, kv_cache, prefill_seq_len)
        total_loss += loss.item()
        n += 1

    return {"val_cfm_loss": total_loss / max(n, 1)}


def _compute_action_stats(dataset) -> dict[str, float]:
    """Compute mean/std of acceleration and curvature from dataset.

    Used to normalize actions for Flow Matching (Alpamayo §5.1).
    """
    all_accel = []
    all_kappa = []
    for i in range(len(dataset)):
        action = dataset.samples[i]["action"]  # (K*2,) interleaved
        all_accel.append(action[0::2])  # even indices = acceleration
        all_kappa.append(action[1::2])  # odd indices = curvature
    all_accel = np.concatenate(all_accel)
    all_kappa = np.concatenate(all_kappa)
    stats = {
        "accel_mean": float(np.mean(all_accel)),
        "accel_std": float(np.std(all_accel)),
        "kappa_mean": float(np.mean(all_kappa)),
        "kappa_std": float(np.std(all_kappa)),
    }
    print(
        f"  Action stats: accel={stats['accel_mean']:.4f}±{stats['accel_std']:.4f}, "
        f"kappa={stats['kappa_mean']:.6f}±{stats['kappa_std']:.6f}"
    )
    return stats


def _build_curve_sampler(dataset, kappa_threshold: float, oversample: float):
    """Build WeightedRandomSampler that oversamples curve scenes."""
    weights = []
    n_curve = 0
    for i in range(len(dataset)):
        action = dataset.samples[i]["action"]  # (K*2,) interleaved
        kappas = action[1::2]  # every other element is kappa
        max_kappa = float(np.max(np.abs(kappas)))
        if max_kappa > kappa_threshold:
            weights.append(oversample)
            n_curve += 1
        else:
            weights.append(1.0)
    print(
        f"  Curve scenes (|kappa| > {kappa_threshold}): {n_curve}/{len(dataset)} "
        f"({n_curve / len(dataset) * 100:.1f}%), weight={oversample}x"
    )
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)


def main():
    t_start = time.time()
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_dim = args.K * 2
    print(f"Device: {device}")
    print(f"K={args.K}, action_dim={action_dim}")
    print(
        f"Expert: hidden={args.hidden_size}, layers={args.num_hidden_layers}, heads={args.num_attention_heads}"
    )

    # Dataset (scene-level split)
    print("Loading dataset...")
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

    # Curve oversampling
    train_sampler = None
    if args.curve_oversample > 1.0:
        print("Building curve oversampler...")
        train_sampler = _build_curve_sampler(
            train_dataset, args.curve_kappa_threshold, args.curve_oversample
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
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

    # Compute action normalization stats from training set (Alpamayo §5.1)
    print("Computing action normalization stats...")
    action_stats = _compute_action_stats(train_dataset)

    # Flow Matching Expert (trainable)
    print("Building Expert TrajectoryDecoder...")
    decoder = TrajectoryDecoder(
        K=args.K,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        num_fourier_feats=args.num_fourier_feats,
        fourier_max_freq=args.fourier_max_freq,
        mlp_hidden_size=args.mlp_hidden_size,
        mlp_num_layers=args.mlp_num_layers,
        **action_stats,
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

    if args.use_wandb:
        import wandb

        wandb.init(project=args.wandb_project, name="stage2-flow", config=vars(args))

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
        t_epoch = time.time()
        decoder.train()
        epoch_loss = 0.0
        epoch_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            gt_action = batch["action"].to(device)

            # Extract KV-cache from frozen VLM
            kv_cache, prefill_seq_len = extract_kv_cache(vlm, pixel_values)

            # CFM loss (bf16 for decoder)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = cfm_loss(decoder, gt_action, kv_cache, prefill_seq_len)
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

                    if args.use_wandb:
                        import wandb

                        wandb.log({"train/cfm_loss": avg_loss, "train/lr": lr}, step=global_step)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # Epoch evaluation
        metrics = evaluate(decoder, vlm, val_loader, device)
        avg_train_loss = epoch_loss / max(epoch_samples, 1)

        epoch_time = time.time() - t_epoch
        print(
            f"\n=== Epoch {epoch + 1}/{args.max_epochs} | "
            f"Train CFM: {avg_train_loss:.4f} | "
            f"Val CFM: {metrics['val_cfm_loss']:.4f} | "
            f"Time: {epoch_time:.0f}s ===\n"
        )

        if args.use_wandb:
            import wandb

            wandb.log(
                {f"val/{k}": v for k, v in metrics.items()}
                | {"train/epoch_loss": avg_train_loss, "epoch": epoch + 1},
                step=global_step,
            )

        if metrics["val_cfm_loss"] < best_val_loss:
            best_val_loss = metrics["val_cfm_loss"]
            ckpt_data = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "decoder_state_dict": decoder.state_dict(),
                "metrics": metrics,
                "K": args.K,
                "action_dim": action_dim,
                "hidden_size": args.hidden_size,
                "num_hidden_layers": args.num_hidden_layers,
                "num_attention_heads": args.num_attention_heads,
                "intermediate_size": args.intermediate_size or args.hidden_size * 4,
                "num_fourier_feats": args.num_fourier_feats,
                "fourier_max_freq": args.fourier_max_freq,
                "mlp_hidden_size": args.mlp_hidden_size,
                "mlp_num_layers": args.mlp_num_layers,
                "architecture": "expert_kv_cache",  # distinguish from old cross-attn decoder
            }
            torch.save(ckpt_data, save_dir / "best.pt")

    ckpt_data = {
        "epoch": args.max_epochs,
        "global_step": global_step,
        "decoder_state_dict": decoder.state_dict(),
        "metrics": metrics,
        "K": args.K,
        "action_dim": action_dim,
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.num_hidden_layers,
        "num_attention_heads": args.num_attention_heads,
        "intermediate_size": args.intermediate_size or args.hidden_size * 4,
        "num_fourier_feats": args.num_fourier_feats,
        "fourier_max_freq": args.fourier_max_freq,
        "mlp_hidden_size": args.mlp_hidden_size,
        "mlp_num_layers": args.mlp_num_layers,
        "architecture": "expert_kv_cache",
    }
    torch.save(ckpt_data, save_dir / "final.pt")

    # Summary
    print("\n=== Summary ===")
    final_metrics = evaluate(decoder, vlm, val_loader, device)
    print(f"  Val CFM: {init_metrics['val_cfm_loss']:.4f} -> {final_metrics['val_cfm_loss']:.4f}")
    print(f"  Decoder params: {n_trainable:,}")

    total_time = time.time() - t_start
    if torch.cuda.is_available():
        print(f"\nPeak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    print(f"Total time: {total_time:.0f}s ({total_time / 60:.1f}min)")
    print("\nDone.")


if __name__ == "__main__":
    main()
