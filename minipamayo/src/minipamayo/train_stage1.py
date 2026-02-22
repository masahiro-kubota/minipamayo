"""Stage 1: Discrete action token training.

Quantizes (a, kappa) x K into discrete tokens and trains with cross-entropy
loss via teacher forcing. LLM vocabulary extended with 256 action bins.

Usage:
    cd minipamayo && uv run python -m minipamayo.train_stage1 \
        --phase4_checkpoint checkpoints/phase4/best.pt
"""

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data.nuscenes_trajectory_dataset import NuScenesTrajectoryDataset
from .models.discrete_head import DiscreteActionTokenizer
from .models.minipamayo import MiniPamayo


def parse_args():
    parser = argparse.ArgumentParser(description="MiniPamayo Stage 1 training")
    parser.add_argument("--nuscenes_root", type=str, default="/mnt/ssd/nuscenes")
    parser.add_argument("--nuscenes_version", type=str, default="v1.0-trainval")
    parser.add_argument(
        "--phase4_checkpoint",
        type=str,
        default="checkpoints/phase4/best.pt",
        help="Phase 4 checkpoint to initialize from",
    )
    parser.add_argument(
        "--vlm_checkpoint",
        type=str,
        default="../cosmos-reason-mini/checkpoints/rl-mini-merged/checkpoint-final.pt",
        help="Fallback VLM checkpoint if phase4 not found",
    )
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--n_bins", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="checkpoints/stage1")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="minipamayo")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, tokenizer, dataloader, device, K) -> dict:
    """Evaluate: CE loss + token accuracy."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n = 0

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        gt_action = batch["action"].to(device)

        # Encode GT to token IDs
        gt_token_ids = tokenizer.encode_batch(gt_action)  # (B, K*2)

        # Vision + Adapter
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            patch_features = model.vision_encoder(pixel_values)
            visual_tokens = model.adapter(patch_features)

        # Build teacher-forced sequence
        B = visual_tokens.shape[0]
        n_vis = visual_tokens.shape[1]
        action_embeds = model.llm.get_input_embeddings()(gt_token_ids)
        inputs_embeds = torch.cat([visual_tokens.to(action_embeds.dtype), action_embeds], dim=1)

        labels = torch.full((B, n_vis + K * 2), -100, dtype=torch.long, device=device)
        labels[:, n_vis:] = gt_token_ids

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.llm(inputs_embeds=inputs_embeds, labels=labels)

        total_loss += outputs.loss.float().item()

        # Token accuracy (compare shifted predictions with labels)
        logits = outputs.logits[:, n_vis - 1 : -1, :]  # (B, K*2, vocab)
        pred_tokens = logits.argmax(dim=-1)  # (B, K*2)
        total_correct += (pred_tokens == gt_token_ids).sum().item()
        total_tokens += gt_token_ids.numel()
        n += 1

    return {
        "val_loss": total_loss / max(n, 1),
        "token_accuracy": total_correct / max(total_tokens, 1),
    }


def main():
    t_start = time.time()
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"K={args.K}, n_bins={args.n_bins}, action_tokens={args.K * 2}")

    # Tokenizer
    tokenizer = DiscreteActionTokenizer(n_bins=args.n_bins)

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

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )

    # Model — load checkpoint FIRST, then extend vocab
    print("Building model...")
    model = MiniPamayo(adapter_type="cross_attention", action_dim=args.K * 2)

    phase4_path = Path(args.phase4_checkpoint)
    vlm_path = Path(args.vlm_checkpoint)

    if phase4_path.exists():
        print(f"Loading Phase 4 checkpoint: {phase4_path}")
        ckpt = torch.load(phase4_path, map_location="cpu", weights_only=True)
        state_dict = ckpt["model_state_dict"]
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("action_head")}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Missing: {len(missing)} keys (action_head)")
        print(f"  Unexpected: {len(unexpected)} keys")
    elif vlm_path.exists():
        print(f"Phase 4 not found, loading VLM checkpoint: {vlm_path}")
        model.load_vlm_checkpoint(vlm_path)
    else:
        print("No checkpoint found, training from scratch")

    # Extend LLM vocabulary AFTER loading checkpoint
    original_vocab = model.llm.config.vocab_size
    new_vocab = tokenizer.vocab_offset + args.n_bins
    model.llm.resize_token_embeddings(new_vocab)
    print(f"Vocab extended: {original_vocab} -> {new_vocab} (+{args.n_bins} action tokens)")

    # All modules trainable (same as Stage 0)
    model.set_stage0()
    model.enable_gradient_checkpointing()
    model = model.to(device)

    param_info = model.count_parameters()
    total_params = sum(v["total"] for v in param_info.values())
    trainable_params = sum(v["trainable"] for v in param_info.values())
    print(f"\nParameters: {total_params:,} total, {trainable_params:,} trainable")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.max_epochs // args.grad_accum
    warmup_steps = max(1, int(total_steps * 0.05))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(progress * math.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if args.use_wandb:
        import wandb

        wandb.init(project=args.wandb_project, name="stage1-discrete", config=vars(args))

    # Initial evaluation
    print("\n=== Initial Evaluation ===")
    init_metrics = evaluate(model, tokenizer, val_loader, device, args.K)
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

            # Encode GT actions to token IDs
            gt_token_ids = tokenizer.encode_batch(gt_action)

            # Vision + Adapter
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                patch_features = model.vision_encoder(pixel_values)
                visual_tokens = model.adapter(patch_features)

            # Teacher-forced sequence
            B = visual_tokens.shape[0]
            n_vis = visual_tokens.shape[1]
            action_embeds = model.llm.get_input_embeddings()(gt_token_ids)
            inputs_embeds = torch.cat([visual_tokens.to(action_embeds.dtype), action_embeds], dim=1)

            labels = torch.full((B, n_vis + args.K * 2), -100, dtype=torch.long, device=device)
            labels[:, n_vis:] = gt_token_ids

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model.llm(inputs_embeds=inputs_embeds, labels=labels)
                loss = outputs.loss / args.grad_accum

            loss.backward()
            epoch_loss += outputs.loss.float().item()
            epoch_samples += 1

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                global_step += 1

                if global_step % args.log_every == 0:
                    avg_loss = epoch_loss / epoch_samples
                    lr = scheduler.get_last_lr()[0]
                    print(
                        f"[E{epoch + 1}] Step {global_step:3d} | "
                        f"CE Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                    )

                    if args.use_wandb:
                        import wandb

                        wandb.log({"train/ce_loss": avg_loss, "train/lr": lr}, step=global_step)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # Epoch evaluation
        metrics = evaluate(model, tokenizer, val_loader, device, args.K)
        avg_train_loss = epoch_loss / max(epoch_samples, 1)

        epoch_time = time.time() - t_epoch
        print(
            f"\n=== Epoch {epoch + 1}/{args.max_epochs} | "
            f"Train CE: {avg_train_loss:.4f} | "
            f"Val CE: {metrics['val_loss']:.4f} | "
            f"Token Acc: {metrics['token_accuracy']:.4f} | "
            f"Time: {epoch_time:.0f}s ===\n"
        )

        if args.use_wandb:
            import wandb

            wandb.log(
                {f"val/{k}": v for k, v in metrics.items()}
                | {"train/epoch_loss": avg_train_loss, "epoch": epoch + 1},
                step=global_step,
            )

        if metrics["val_loss"] < best_val_loss:
            best_val_loss = metrics["val_loss"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                    "K": args.K,
                    "n_bins": args.n_bins,
                    "tokenizer_config": {
                        "n_bins": tokenizer.n_bins,
                        "a_range": (tokenizer.a_min, tokenizer.a_max),
                        "kappa_range": (tokenizer.kappa_min, tokenizer.kappa_max),
                        "vocab_offset": tokenizer.vocab_offset,
                    },
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
            "n_bins": args.n_bins,
            "tokenizer_config": {
                "n_bins": tokenizer.n_bins,
                "a_range": (tokenizer.a_min, tokenizer.a_max),
                "kappa_range": (tokenizer.kappa_min, tokenizer.kappa_max),
                "vocab_offset": tokenizer.vocab_offset,
            },
        },
        save_dir / "final.pt",
    )

    # Summary
    print("\n=== Summary ===")
    final_metrics = evaluate(model, tokenizer, val_loader, device, args.K)
    print(f"  Val CE:    {init_metrics['val_loss']:.4f} -> {final_metrics['val_loss']:.4f}")
    print(
        f"  Token Acc: {init_metrics['token_accuracy']:.4f} -> {final_metrics['token_accuracy']:.4f}"
    )

    total_time = time.time() - t_start
    if torch.cuda.is_available():
        print(f"\nPeak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    print(f"Total time: {total_time:.0f}s ({total_time / 60:.1f}min)")
    print("\nDone.")


if __name__ == "__main__":
    main()
