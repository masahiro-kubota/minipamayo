"""Stage 1 Training: Feature Alignment (Adapter only)."""

import argparse
from pathlib import Path

import torch
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from qwen_vl_mini.data.pretrain_dataset import PretrainCollator, PretrainDataset
from qwen_vl_mini.model import IMAGE_TRANSFORM, QwenVLMini

DEFAULTS = {
    "json_path": "data/llava-pretrain/chat.json",
    "image_dir": "data/llava-pretrain",
    "output_dir": "checkpoints/stage1",
    "lr": 1e-3,
    "batch_size": 4,
    "grad_accum": 64,  # global batch = 256
    "epochs": 1,
    "warmup_ratio": 0.03,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "max_length": 512,
    "save_steps": 500,
    "logging_steps": 10,
    "num_workers": 4,
    "wandb_project": "qwen-vl-mini",
}


def save_checkpoint(model, optimizer, scheduler, global_step, output_dir):
    path = Path(output_dir) / f"checkpoint-{global_step}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter_state_dict": model.adapter.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": global_step,
        },
        path,
    )
    print(f"Saved checkpoint: {path}")


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Feature Alignment")
    for k, v in DEFAULTS.items():
        parser.add_argument(f"--{k}", type=type(v), default=v)
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- wandb ---
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(project=args.wandb_project, name="stage1-alignment", config=vars(args))
    else:
        print("wandb disabled")

    # --- Model ---
    print("Loading model...")
    model = QwenVLMini()
    model.set_stage1()
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params (Adapter only): {trainable:,}")

    # --- Data ---
    print("Loading dataset...")
    dataset = PretrainDataset(
        json_path=args.json_path,
        image_dir=args.image_dir,
        tokenizer=model.tokenizer,
        transform=IMAGE_TRANSFORM,
        max_length=args.max_length,
    )
    print(f"Dataset size: {len(dataset):,}")

    collator = PretrainCollator(model.tokenizer, max_length=args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    # --- Optimizer ---
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    # --- Scheduler ---
    steps_per_epoch = len(dataloader) // args.grad_accum
    num_training_steps = steps_per_epoch * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    print(f"Training steps: {num_training_steps} (warmup: {num_warmup_steps})")

    # --- Training loop ---
    model.train()
    global_step = 0
    initial_loss = None
    adapter_snapshot = None

    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    pixel_values=batch["pixel_values"].to(device),
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                loss = output.loss / args.grad_accum

            # === T2 Check 1: NaN detection ===
            if torch.isnan(output.loss):
                raise RuntimeError(f"Loss is NaN at step {step}. Aborting.")

            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                loss_value = output.loss.item()

                # === T2 Check 2: Learning progress ===
                if global_step == 0:
                    initial_loss = loss_value
                if global_step == 100 and initial_loss is not None:
                    if loss_value < initial_loss:
                        print(f"✓ Learning check passed: {initial_loss:.4f} → {loss_value:.4f}")
                    else:
                        print(f"⚠ Loss not decreasing: {initial_loss:.4f} → {loss_value:.4f}")

                # === T2 Check 3: Adapter weights updating ===
                if global_step == 0:
                    adapter_snapshot = {k: v.clone() for k, v in model.adapter.state_dict().items()}
                if global_step == 1 and adapter_snapshot is not None:
                    changed = any(
                        not torch.equal(adapter_snapshot[k], v)
                        for k, v in model.adapter.state_dict().items()
                    )
                    if changed:
                        print("✓ Adapter weights are being updated")
                    else:
                        print("⚠ Adapter weights not changing!")

                # Logging
                if global_step % args.logging_steps == 0:
                    print(
                        f"[Step {global_step}] loss={loss_value:.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e} grad_norm={grad_norm:.2f}"
                    )
                    if use_wandb:
                        wandb.log(
                            {
                                "loss": loss_value,
                                "lr": scheduler.get_last_lr()[0],
                                "grad_norm": grad_norm.item(),
                                "epoch": epoch,
                                "global_step": global_step,
                            }
                        )

                # Checkpoint
                if (global_step + 1) % args.save_steps == 0:
                    save_checkpoint(model, optimizer, scheduler, global_step, args.output_dir)

                global_step += 1

    # Final checkpoint
    save_checkpoint(model, optimizer, scheduler, global_step, args.output_dir)
    print(f"Training complete. Final step: {global_step}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
