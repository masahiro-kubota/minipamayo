"""Stage 3: Chain of Causation (CoC) SFT.

Joint next-token prediction of reasoning text + discrete action tokens.
Uses Qwen chat template for sequence construction.

Sequence (chat format):
  <|im_start|>system\n{system_msg}<|im_end|>\n
  <|im_start|>user\n[visual_tokens] Speed: {v0} m/s. ...?<|im_end|>\n
  <|im_start|>assistant\n{CoC reasoning text}{action tokens}<|im_end|>

Loss on assistant content only (reasoning + action + eos tokens).

Design doc: stage3-coc-sft.md
- Gradient control: set_stage3() — VE+Adapter+LLM trainable, Flow frozen
- Differential LR: VE 1e-5, rest 2e-5
- Chat template with egomotion (v0)

Usage:
    cd minipamayo && uv run python -m minipamayo.train_stage3 \
        --stage1_checkpoint checkpoints/stage1/best.pt \
        --coc_data data/coc_annotations_trainval.jsonl
"""

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .data.coc_dataset import CoCDataset
from .models.discrete_head import DiscreteActionTokenizer
from .models.minipamayo import MiniPamayo


def parse_args():
    parser = argparse.ArgumentParser(description="MiniPamayo Stage 3 CoC SFT")
    parser.add_argument("--coc_data", type=str, default="data/coc_annotations_trainval.jsonl")
    parser.add_argument(
        "--stage1_checkpoint",
        type=str,
        default="checkpoints/stage1/best.pt",
        help="Stage 1 checkpoint with extended vocab",
    )
    parser.add_argument(
        "--phase4_checkpoint",
        type=str,
        default="checkpoints/phase4/best.pt",
        help="Fallback Phase 4 checkpoint",
    )
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--n_bins", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--ve_lr", type=float, default=1e-5, help="Vision Encoder LR (0.5x main)")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_accum", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="checkpoints/stage3")
    parser.add_argument("--nuscenes_root", type=str, default="/mnt/ssd/nuscenes")
    parser.add_argument("--nuscenes_version", type=str, default="v1.0-trainval")
    parser.add_argument("--max_text_len", type=int, default=2048)
    parser.add_argument(
        "--action_loss_weight",
        type=float,
        default=1.0,
        help="Weight multiplier for action token loss (vs reasoning tokens)",
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="minipamayo")
    return parser.parse_args()


def build_sequence(model, pixel_values, batch, device, action_loss_weight=1.0):
    """Build the full training sequence with Qwen chat template.

    Chat format:
      [system_ids] [user_prefix_ids] [visual_embeds(16)] [ego_question_ids]
      [asst_prefix_ids] [reasoning_ids] [action_ids] [eos_ids]

    Labels: -100 for everything up to and including asst_prefix,
            then reasoning_ids + action_ids + eos_ids.

    Returns:
        inputs_embeds: (1, total_len, hidden_dim)
        labels: (1, total_len)
        loss_weights: (1, total_len) — per-token loss weights
    """
    system_ids = batch["system_ids"].squeeze(0).to(device)
    user_prefix_ids = batch["user_prefix_ids"].squeeze(0).to(device)
    ego_question_ids = batch["ego_question_ids"].squeeze(0).to(device)
    asst_prefix_ids = batch["asst_prefix_ids"].squeeze(0).to(device)
    reasoning_ids = batch["reasoning_ids"].squeeze(0).to(device)
    action_ids = batch["action_ids"].squeeze(0).to(device)
    eos_ids = batch["eos_ids"].squeeze(0).to(device)

    # Visual embeddings
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        patch_features = model.vision_encoder(pixel_values)
        visual_embeds = model.adapter(patch_features)  # (1, 16, 896)

    embed_layer = model.llm.get_input_embeddings()

    # Text embeddings for each segment
    system_embeds = embed_layer(system_ids.unsqueeze(0))
    user_prefix_embeds = embed_layer(user_prefix_ids.unsqueeze(0))
    ego_question_embeds = embed_layer(ego_question_ids.unsqueeze(0))
    asst_prefix_embeds = embed_layer(asst_prefix_ids.unsqueeze(0))
    reasoning_embeds = embed_layer(reasoning_ids.unsqueeze(0))
    action_embeds = embed_layer(action_ids.unsqueeze(0))
    eos_embeds = embed_layer(eos_ids.unsqueeze(0))

    # Concatenate all segments
    target_dtype = system_embeds.dtype
    inputs_embeds = torch.cat(
        [
            system_embeds,
            user_prefix_embeds,
            visual_embeds.to(target_dtype),
            ego_question_embeds,
            asst_prefix_embeds,
            reasoning_embeds,
            action_embeds,
            eos_embeds,
        ],
        dim=1,
    )

    # Labels: loss only on assistant content (reasoning + action + eos)
    n_prefix = (
        system_ids.shape[0]
        + user_prefix_ids.shape[0]
        + visual_embeds.shape[1]
        + ego_question_ids.shape[0]
        + asst_prefix_ids.shape[0]
    )
    n_reasoning = reasoning_ids.shape[0]
    n_action = action_ids.shape[0]
    n_eos = eos_ids.shape[0]
    total_len = inputs_embeds.shape[1]

    labels = torch.full((1, total_len), -100, dtype=torch.long, device=device)
    offset = n_prefix
    labels[0, offset : offset + n_reasoning] = reasoning_ids
    offset += n_reasoning
    labels[0, offset : offset + n_action] = action_ids
    offset += n_action
    labels[0, offset : offset + n_eos] = eos_ids

    # Per-token loss weights: 1.0 for reasoning/eos, action_loss_weight for action tokens
    loss_weights = torch.ones((1, total_len), dtype=torch.float32, device=device)
    action_start = n_prefix + n_reasoning
    loss_weights[0, action_start : action_start + n_action] = action_loss_weight

    return inputs_embeds, labels, loss_weights


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate: CE loss + token accuracy on reasoning + action tokens."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n = 0

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)

        inputs_embeds, labels, _loss_weights = build_sequence(model, pixel_values, batch, device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.llm(inputs_embeds=inputs_embeds, labels=labels)

        total_loss += outputs.loss.float().item()

        # Token accuracy on non-masked positions
        mask = labels[0] != -100
        if mask.any():
            logits = outputs.logits[0, :-1]  # shifted
            target = labels[0, 1:]
            target_mask = target != -100
            if target_mask.any():
                preds = logits[target_mask].argmax(dim=-1)
                total_correct += (preds == target[target_mask]).sum().item()
                total_tokens += target_mask.sum().item()
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
    print(f"K={args.K}, n_bins={args.n_bins}")

    # Tokenizers
    action_tokenizer = DiscreteActionTokenizer(n_bins=args.n_bins)
    text_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    # Dataset (scene-level split via image_path filtering)
    print("Loading CoC dataset...")
    use_split = args.nuscenes_version == "v1.0-trainval"
    if use_split:
        from .data.nuscenes_trajectory_dataset import NuScenesTrajectoryDataset

        train_traj = NuScenesTrajectoryDataset(
            nuscenes_root=args.nuscenes_root,
            version=args.nuscenes_version,
            K=args.K,
            split="train",
        )
        val_traj = NuScenesTrajectoryDataset(
            nuscenes_root=args.nuscenes_root,
            version=args.nuscenes_version,
            K=args.K,
            split="val",
        )
        train_paths = {s["image_path"] for s in train_traj.samples}
        val_paths = {s["image_path"] for s in val_traj.samples}
        del train_traj, val_traj  # Free memory
    else:
        train_paths = None
        val_paths = None

    train_dataset = CoCDataset(
        annotations_path=args.coc_data,
        tokenizer=text_tokenizer,
        action_tokenizer=action_tokenizer,
        K=args.K,
        max_text_len=args.max_text_len,
        allowed_image_paths=train_paths,
    )
    val_dataset = CoCDataset(
        annotations_path=args.coc_data,
        tokenizer=text_tokenizer,
        action_tokenizer=action_tokenizer,
        K=args.K,
        max_text_len=args.max_text_len,
        allowed_image_paths=val_paths,
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Model — load Stage 1 checkpoint (has extended vocab)
    print("Building model...")
    model = MiniPamayo(adapter_type="cross_attention", action_dim=args.K * 2)

    stage1_path = Path(args.stage1_checkpoint)
    phase4_path = Path(args.phase4_checkpoint)

    if stage1_path.exists():
        print(f"Loading Stage 1 checkpoint: {stage1_path}")
        ckpt = torch.load(stage1_path, map_location="cpu", weights_only=True)
        new_vocab = action_tokenizer.vocab_offset + args.n_bins
        model.llm.resize_token_embeddings(new_vocab)
        state_dict = ckpt["model_state_dict"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Missing: {len(missing)} keys, Unexpected: {len(unexpected)} keys")
    elif phase4_path.exists():
        print(f"Stage 1 not found, loading Phase 4: {phase4_path}")
        ckpt = torch.load(phase4_path, map_location="cpu", weights_only=True)
        state_dict = {
            k: v for k, v in ckpt["model_state_dict"].items() if not k.startswith("action_head")
        }
        model.load_state_dict(state_dict, strict=False)
        new_vocab = action_tokenizer.vocab_offset + args.n_bins
        model.llm.resize_token_embeddings(new_vocab)
    else:
        print("WARNING: No checkpoint found, training from scratch")
        new_vocab = action_tokenizer.vocab_offset + args.n_bins
        model.llm.resize_token_embeddings(new_vocab)

    # Stage 3 gradient control: VE+Adapter+LLM trainable, Flow Head frozen
    model.set_stage3()
    model.enable_gradient_checkpointing()
    model = model.to(device)

    param_info = model.count_parameters()
    total_params = sum(v["total"] for v in param_info.values())
    trainable_params = sum(v["trainable"] for v in param_info.values())
    print(f"\nParameters: {total_params:,} total, {trainable_params:,} trainable")

    # Optimizer with differential learning rates (design: VE 1e-5, rest 2e-5)
    ve_params = list(model.vision_encoder.parameters())
    other_params = list(model.adapter.parameters()) + list(model.llm.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": ve_params, "lr": args.ve_lr},
            {"params": other_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.max_epochs // args.grad_accum
    warmup_steps = max(1, int(total_steps * 0.03))  # design: warmup_ratio=0.03

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(progress * math.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if args.use_wandb:
        import wandb

        wandb.init(project=args.wandb_project, name="stage3-coc-sft", config=vars(args))

    # Initial evaluation
    print("\n=== Initial Evaluation ===")
    init_metrics = evaluate(model, val_loader, device)
    for k, v in init_metrics.items():
        print(f"  {k}: {v:.6f}")

    # Training
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

            inputs_embeds, labels, loss_weights = build_sequence(
                model,
                pixel_values,
                batch,
                device,
                action_loss_weight=args.action_loss_weight,
            )

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model.llm(inputs_embeds=inputs_embeds)
                # Weighted CE loss: shift logits and labels
                logits = outputs.logits[:, :-1, :].contiguous()
                targets = labels[:, 1:].contiguous()
                weights = loss_weights[:, 1:].contiguous()
                mask = targets != -100
                if mask.any():
                    ce = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1),
                        ignore_index=-100,
                        reduction="none",
                    )
                    ce = ce.view(targets.shape)
                    weighted_loss = (ce * weights * mask.float()).sum() / mask.float().sum()
                else:
                    weighted_loss = torch.tensor(0.0, device=device)
                loss = weighted_loss / args.grad_accum

            loss.backward()
            epoch_loss += weighted_loss.float().item()
            epoch_samples += 1

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                global_step += 1

                if global_step % args.log_every == 0:
                    avg_loss = epoch_loss / epoch_samples
                    lr_ve = scheduler.get_last_lr()[0]
                    lr_main = scheduler.get_last_lr()[1]
                    print(
                        f"[E{epoch + 1}] Step {global_step:3d} | "
                        f"CE Loss: {avg_loss:.4f} | LR(VE): {lr_ve:.2e} | LR(main): {lr_main:.2e}"
                    )

                    if args.use_wandb:
                        import wandb

                        wandb.log(
                            {
                                "train/ce_loss": avg_loss,
                                "train/lr_ve": lr_ve,
                                "train/lr_main": lr_main,
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
            f"Train CE: {avg_train_loss:.4f} | "
            f"Val CE: {metrics['val_loss']:.4f} | "
            f"Token Acc: {metrics['token_accuracy']:.4f} | "
            f"Time: {time.time() - t_epoch:.0f}s ===\n"
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
        },
        save_dir / "final.pt",
    )

    # Summary
    print("\n=== Summary ===")
    final_metrics = evaluate(model, val_loader, device)
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
