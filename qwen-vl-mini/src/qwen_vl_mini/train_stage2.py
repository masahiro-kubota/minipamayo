"""Stage 2 Training: Visual Instruction Tuning (Full fine-tune)."""

import argparse
from pathlib import Path

import torch
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from qwen_vl_mini.data.instruct_dataset import InstructCollator, InstructDataset
from qwen_vl_mini.model import IMAGE_TRANSFORM, QwenVLMini


def log(msg: str) -> None:
    """Print with flush to avoid output buffering in background execution."""
    print(msg, flush=True)


DEFAULTS = {
    "json_path": "data/llava-instruct/llava_instruct_150k.json",
    "image_dir": "data/coco/train2014",
    "output_dir": "checkpoints/stage2",
    "stage1_checkpoint": "checkpoints/stage1/checkpoint-2325.pt",
    "resume": "",
    "lr": 2e-5,
    "ve_lr": 1e-5,
    "batch_size": 1,
    "grad_accum": 128,  # global batch = 128
    "epochs": 2,
    "warmup_ratio": 0.03,
    "weight_decay": 0.1,
    "max_grad_norm": 1.0,
    "max_length": 2048,
    "save_steps": 100,
    "logging_steps": 5,
    "num_workers": 4,
    "wandb_project": "qwen-vl-mini",
}


def save_checkpoint(model, optimizer, scheduler, global_step, epoch, output_dir):
    path = Path(output_dir) / f"checkpoint-{global_step}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "vision_encoder_state_dict": model.vision_encoder.state_dict(),
            "adapter_state_dict": model.adapter.state_dict(),
            "llm_state_dict": model.llm.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
        },
        path,
    )
    log(f"Saved checkpoint: {path}")


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Visual Instruction Tuning")
    for k, v in DEFAULTS.items():
        parser.add_argument(f"--{k}", type=type(v), default=v)
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument(
        "--ve_frozen", action="store_true", help="Keep VisionEncoder frozen (fallback)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- wandb ---
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name="stage2-instruct",
            config=vars(args),
            resume="allow",
        )
    else:
        log("wandb disabled")

    # --- Model ---
    log("Loading model...")
    model = QwenVLMini()

    # Load weights
    if args.resume:
        log(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.vision_encoder.load_state_dict(ckpt["vision_encoder_state_dict"])
        model.adapter.load_state_dict(ckpt["adapter_state_dict"])
        model.llm.load_state_dict(ckpt["llm_state_dict"])
        resume_step = ckpt["global_step"] + 1
        resume_epoch = ckpt.get("epoch", 0)
        log(f"Resumed: global_step={resume_step}, epoch={resume_epoch}")
    else:
        # Load Stage 1 adapter weights for fresh start
        ckpt = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=True)
        model.adapter.load_state_dict(ckpt["adapter_state_dict"])
        log(f"Loaded Stage 1 adapter weights from {args.stage1_checkpoint}")
        resume_step = 0
        resume_epoch = 0

    # Set training mode
    if args.ve_frozen:
        log("Mode: VE frozen + Adapter + LLM")
        model.vision_encoder.freeze()
        model.adapter.requires_grad_(True)
        model.llm.requires_grad_(True)
    else:
        log("Mode: Full fine-tune (VE + Adapter + LLM)")
        model.set_stage2()

    model = model.to(device)

    # Enable gradient checkpointing for VRAM savings
    model.llm.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"Trainable params: {trainable:,} / {total:,} ({trainable / total:.1%})")

    # --- Data ---
    log("Loading dataset...")
    dataset = InstructDataset(
        json_path=args.json_path,
        image_dir=args.image_dir,
        tokenizer=model.tokenizer,
        transform=IMAGE_TRANSFORM,
        max_length=args.max_length,
    )
    log(f"Dataset size: {len(dataset):,}")

    collator = InstructCollator(model.tokenizer, max_length=args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    # --- Optimizer (parameter groups) ---
    ve_params = [p for p in model.vision_encoder.parameters() if p.requires_grad]
    other_params = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("vision_encoder.")
    ]

    param_groups = []
    if ve_params:
        param_groups.append({"params": ve_params, "lr": args.ve_lr})
    param_groups.append({"params": other_params, "lr": args.lr})

    optimizer = AdamW(
        param_groups,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    # --- Scheduler ---
    steps_per_epoch = len(dataloader) // args.grad_accum
    num_training_steps = steps_per_epoch * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    # Resume optimizer/scheduler state
    if args.resume:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        log("Restored optimizer and scheduler state")

    log(f"Training steps: {num_training_steps} (warmup: {num_warmup_steps})")
    if resume_step > 0:
        log(f"Resuming from step {resume_step}/{num_training_steps}")

    # --- Training loop ---
    model.train()
    global_step = resume_step
    initial_loss = None
    ve_snapshot = None
    microbatch_count = 0  # Track microbatches within current accumulation window

    for epoch in range(resume_epoch, args.epochs):
        for step, batch in enumerate(dataloader):
            # Skip already-processed microbatches when resuming
            if args.resume and epoch == resume_epoch:
                steps_to_skip = resume_step * args.grad_accum
                if step < steps_to_skip:
                    continue

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    pixel_values=batch["pixel_values"].to(device),
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                loss = output.loss / args.grad_accum

            # === T2 Check 1: NaN/Inf detection ===
            if torch.isnan(output.loss) or torch.isinf(output.loss):
                raise RuntimeError(
                    f"Loss is {'NaN' if torch.isnan(output.loss) else 'Inf'} at step {step}. "
                    "Consider: lower lr, gradient clipping, or --ve_frozen."
                )

            loss.backward()
            microbatch_count += 1

            if microbatch_count % args.grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                microbatch_count = 0

                loss_value = output.loss.item()

                # === T2 Check 2: Learning progress ===
                if global_step == resume_step:
                    initial_loss = loss_value
                if global_step == resume_step + 50 and initial_loss is not None:
                    if loss_value < initial_loss:
                        log(f"✓ Learning check passed: {initial_loss:.4f} → {loss_value:.4f}")
                    else:
                        log(f"⚠ Loss not decreasing: {initial_loss:.4f} → {loss_value:.4f}")

                # === T2 Check 3: DINOv2 weights updating ===
                if not args.ve_frozen:
                    if global_step == resume_step:
                        ve_snapshot = (
                            model.vision_encoder.dinov2.encoder.layer[-1]
                            .attention.attention.query.weight.data[:3, :3]
                            .clone()
                        )
                    if global_step == resume_step + 10 and ve_snapshot is not None:
                        ve_current = model.vision_encoder.dinov2.encoder.layer[
                            -1
                        ].attention.attention.query.weight.data[:3, :3]
                        if torch.equal(ve_snapshot, ve_current):
                            log("⚠ DINOv2 weights not changing — check requires_grad")
                        else:
                            log("✓ DINOv2 weights are being updated")

                # === T2 Check 4: High gradient norm warning ===
                if grad_norm > 10.0:
                    log(f"⚠ High gradient norm: {grad_norm:.2f} at step {global_step}")

                # Logging
                if global_step % args.logging_steps == 0:
                    lr_display = scheduler.get_last_lr()[-1]
                    log(
                        f"[Epoch {epoch} Step {global_step}] loss={loss_value:.4f} "
                        f"lr={lr_display:.2e} grad_norm={grad_norm:.2f}"
                    )
                    if use_wandb:
                        log_dict = {
                            "loss": loss_value,
                            "lr": lr_display,
                            "grad_norm": grad_norm.item()
                            if hasattr(grad_norm, "item")
                            else grad_norm,
                            "epoch": epoch,
                            "global_step": global_step,
                        }
                        if len(optimizer.param_groups) > 1:
                            log_dict["lr_ve"] = optimizer.param_groups[0]["lr"]
                            log_dict["lr_llm"] = optimizer.param_groups[1]["lr"]
                        wandb.log(log_dict)

                # Checkpoint
                if (global_step + 1) % args.save_steps == 0:
                    save_checkpoint(
                        model, optimizer, scheduler, global_step, epoch, args.output_dir
                    )

                global_step += 1

    # Final checkpoint
    save_checkpoint(model, optimizer, scheduler, global_step, epoch, args.output_dir)
    log(f"Training complete. Final step: {global_step}")

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**2
        log(f"Peak VRAM: {peak:.0f} MB")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
