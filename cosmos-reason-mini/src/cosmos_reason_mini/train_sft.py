"""Physical AI SFT 学習スクリプト。

Usage:
    cd cosmos-reason-mini && uv run python -m cosmos_reason_mini.train_sft \
        --json_path data/sft/qa_mini.json \
        --image_root data/nuscenes \
        --vlm_checkpoint ../qwen-vl-mini/checkpoints/stage2.1/checkpoint-5247.pt \
        --output_dir checkpoints/sft
"""

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

DEFAULTS = {
    "json_path": "data/sft/qa_train.json",
    "image_root": "data/nuscenes",
    "vlm_checkpoint": "../qwen-vl-mini/checkpoints/stage2.1/checkpoint-5247.pt",
    "output_dir": "checkpoints/sft",
    "epochs": 1,
    "lr_llm": 2e-5,
    "lr_ve": 1e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "grad_accum_steps": 16,
    "max_length": 2048,
    "neftune_alpha": 5.0,
    "save_steps": 25,
    "logging_steps": 5,
    "no_wandb": False,
}


def create_optimizer(model, lr_llm, lr_ve, weight_decay):
    """VE と LLM/Adapter で異なる LR を設定。"""
    ve_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("vision_encoder."):
            ve_params.append(param)
        else:
            other_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": other_params, "lr": lr_llm},
            {"params": ve_params, "lr": lr_ve},
        ],
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )


def create_scheduler(optimizer, num_training_steps, warmup_ratio):
    """Cosine annealing with warmup (min_lr = lr/10)."""
    warmup_steps = int(num_training_steps * warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(num_training_steps - warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def collate_fn(batch):
    """可変長シーケンスのパディング。"""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = item["attention_mask"]
        labels[i, :seq_len] = item["labels"]

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def save_checkpoint(model, optimizer, global_step, epoch, save_path):
    """train_stage2.py と同一のチェックポイント形式で保存。"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "vision_encoder_state_dict": model.vision_encoder.state_dict(),
            "adapter_state_dict": model.adapter.state_dict(),
            "llm_state_dict": model.llm.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
        },
        save_path,
    )
    print(f"Saved checkpoint: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    for key, default in DEFAULTS.items():
        if isinstance(default, bool):
            parser.add_argument(f"--{key}", action="store_true", default=default)
        else:
            parser.add_argument(f"--{key}", type=type(default), default=default)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Model ---
    from cosmos_reason_mini.model_loader import load_vlm_from_checkpoint

    print(f"Loading model from {args.vlm_checkpoint}...")
    model = load_vlm_from_checkpoint(
        args.vlm_checkpoint,
        neftune_alpha=args.neftune_alpha,
        device=device,
    )
    model.set_stage2()  # VE + Adapter + LLM 全パラメータ訓練可能
    model.train()

    # Gradient checkpointing
    if hasattr(model.llm, "gradient_checkpointing_enable"):
        model.llm.gradient_checkpointing_enable()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total_params / 1e6:.1f}M")

    # --- Dataset ---
    from cosmos_reason_mini.data.driving_dataset import DrivingQADataset

    dataset = DrivingQADataset(
        json_path=args.json_path,
        image_root=args.image_root,
        max_length=args.max_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    print(f"Dataset: {len(dataset)} samples")

    # --- Optimizer & Scheduler ---
    num_training_steps = len(dataloader) * args.epochs // args.grad_accum_steps
    optimizer = create_optimizer(model, args.lr_llm, args.lr_ve, args.weight_decay)
    scheduler = create_scheduler(optimizer, num_training_steps, args.warmup_ratio)
    print(f"Training: {num_training_steps} optimizer steps over {args.epochs} epoch(s)")

    # --- wandb ---
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb

        wandb.init(project="cosmos-reason-mini", name="phase2-sft", config=vars(args))

    # --- Training Loop ---
    global_step = 0
    accum_loss = 0.0
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss / args.grad_accum_steps

            # NaN/Inf detection
            if torch.isnan(outputs.loss) or torch.isinf(outputs.loss):
                print(f"Skipping NaN/Inf loss at batch {batch_idx}")
                continue

            loss.backward()
            accum_loss += outputs.loss.item()

            if (batch_idx + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = accum_loss / args.grad_accum_steps
                accum_loss = 0.0

                if global_step % args.logging_steps == 0:
                    lr = scheduler.get_last_lr()[0]
                    print(
                        f"[epoch {epoch}] step {global_step}/{num_training_steps} "
                        f"loss={avg_loss:.4f} lr={lr:.2e}"
                    )
                    if use_wandb:
                        wandb.log(
                            {
                                "loss": avg_loss,
                                "lr": lr,
                                "step": global_step,
                                "epoch": epoch,
                            }
                        )

                if global_step % args.save_steps == 0:
                    save_checkpoint(
                        model,
                        optimizer,
                        global_step,
                        epoch,
                        Path(args.output_dir) / f"checkpoint-{global_step}.pt",
                    )

    # Final save
    save_checkpoint(
        model,
        optimizer,
        global_step,
        epoch,
        Path(args.output_dir) / f"checkpoint-{global_step}.pt",
    )
    print(f"Training complete. {global_step} steps.")
    if use_wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
