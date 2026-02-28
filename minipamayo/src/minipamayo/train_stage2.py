"""Stage 2: Flow Matching Expert training (Alpamayo §5.1).

Freezes VLM and trains Expert Transformer with Conditional Flow Matching loss.
Expert is conditioned on VLM KV-cache containing [o_image, Reason] via
past_key_values. GT CoC reasoning text is teacher-forced through the VLM
to build the KV-cache (Alpamayo §5.1 CFM loss formula).

Usage:
    cd minipamayo && uv run python -m minipamayo.train_stage2 \
        --nuscenes_version v1.0-trainval \
        --coc_data data/coc_annotations_trainval.jsonl \
        --max_epochs 30 --batch_size 1 --grad_accum 64
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer

from .data.coc_dataset import CoCDataset
from .data.nuscenes_trajectory_dataset import NuScenesTrajectoryDataset
from .models.discrete_head import DiscreteActionTokenizer
from .models.minipamayo import MiniPamayo
from .models.trajectory_decoder import (
    CrossAttentionDecoder,
    SimpleDecoder,
    TrajectoryDecoder,
    cfm_loss,
    cfm_loss_cross_attn,
    cfm_loss_simple,
)


def parse_args():
    parser = argparse.ArgumentParser(description="MiniPamayo Stage 2 Expert training")
    parser.add_argument("--nuscenes_root", type=str, default="/mnt/ssd/nuscenes")
    parser.add_argument("--nuscenes_version", type=str, default="v1.0-trainval")
    parser.add_argument("--coc_data", type=str, default="data/coc_annotations_trainval.jsonl")
    parser.add_argument(
        "--vlm_checkpoint",
        type=str,
        default="checkpoints/stage1/best.pt",
        help="Stage 1 VLM checkpoint (frozen)",
    )
    parser.add_argument("--K", type=int, default=6)
    # Decoder type
    parser.add_argument(
        "--decoder_type",
        type=str,
        default="simple",
        choices=["simple", "kv_cache", "cross_attention"],
        help="Decoder: simple (旧実装 ~3M) / kv_cache (145M) / cross_attention (~6M)",
    )
    # Cross-attention decoder params
    parser.add_argument("--ca_hidden_dim", type=int, default=256)
    parser.add_argument("--ca_num_layers", type=int, default=4)
    parser.add_argument("--ca_num_heads", type=int, default=4)
    parser.add_argument("--ca_mlp_ratio", type=int, default=4)
    parser.add_argument("--ca_dropout", type=float, default=0.0)
    # Simple decoder ablation flags
    parser.add_argument(
        "--use_action_norm",
        action="store_true",
        default=False,
        help="Enable action normalization for SimpleDecoder (ablation)",
    )
    # KV-cache Expert architecture (must satisfy: hidden_size = num_attention_heads * 64)
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
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--attention_dropout", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="checkpoints/stage2")
    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--no_wandb", dest="use_wandb", action="store_false")
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
def extract_conditions(vlm, pixel_values):
    """旧実装: visual tokens → LLM → mean-pool hidden states → (B, 896)."""
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        patch_features = vlm.vision_encoder(pixel_values)
        visual_tokens = vlm.adapter(patch_features)
        outputs = vlm.llm(inputs_embeds=visual_tokens, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1]  # (B, L, 896)
    condition = last_hidden.mean(dim=1).float()  # (B, 896)
    return condition


@torch.no_grad()
def extract_kv_cache(vlm, pixel_values, batch, device):
    """Extract VLM KV-cache with teacher-forced GT reasoning (Alpamayo §5.1).

    Builds the full prompt sequence [system][user][visual][question][asst][reasoning]
    and runs VLM forward with use_cache=True.
    """
    inputs_embeds = _build_prompt_embeds(vlm, pixel_values, batch, device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = vlm.llm(inputs_embeds=inputs_embeds, use_cache=True)

    kv_cache = outputs.past_key_values
    prefill_seq_len = kv_cache.get_seq_length()
    return kv_cache, prefill_seq_len


@torch.no_grad()
def _build_prompt_embeds(vlm, pixel_values, batch, device):
    """Build full prompt embeddings (shared by extract_kv_cache and extract_hidden_states)."""
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        patch_features = vlm.vision_encoder(pixel_values)
        visual_tokens = vlm.adapter(patch_features)

    embed_layer = vlm.llm.get_input_embeddings()

    def _embed(ids_key):
        t = batch[ids_key].to(device)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return embed_layer(t)

    system_embeds = _embed("system_ids")
    user_prefix_embeds = _embed("user_prefix_ids")
    ego_question_embeds = _embed("ego_question_ids")
    asst_prefix_embeds = _embed("asst_prefix_ids")
    reasoning_embeds = _embed("reasoning_ids")

    target_dtype = system_embeds.dtype
    return torch.cat(
        [
            system_embeds,
            user_prefix_embeds,
            visual_tokens.to(target_dtype),
            ego_question_embeds,
            asst_prefix_embeds,
            reasoning_embeds,
        ],
        dim=1,
    )


@torch.no_grad()
def extract_hidden_states(vlm, pixel_values, batch, device):
    """Extract VLM hidden states for CrossAttentionDecoder conditioning.

    Same prompt as extract_kv_cache, but returns last_hidden_state instead
    of KV-cache. No KV-head bottleneck — full 896-dim hidden states.
    """
    inputs_embeds = _build_prompt_embeds(vlm, pixel_values, batch, device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = vlm.llm(inputs_embeds=inputs_embeds, output_hidden_states=True)
    return outputs.hidden_states[-1]  # (B, L, 896)


@torch.no_grad()
def evaluate(decoder, vlm, dataloader, device, decoder_type="kv_cache"):
    """Evaluate CFM loss on validation set."""
    decoder.eval()
    total_loss = 0.0
    n = 0

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        gt_action = batch["action"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if decoder_type == "simple":
                condition = extract_conditions(vlm, pixel_values)
                loss = cfm_loss_simple(decoder, gt_action, condition)
            elif decoder_type == "cross_attention":
                hidden_states = extract_hidden_states(vlm, pixel_values, batch, device)
                loss = cfm_loss_cross_attn(decoder, gt_action, hidden_states)
            else:
                kv_cache, prefill_seq_len = extract_kv_cache(vlm, pixel_values, batch, device)
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


def _build_checkpoint(args, decoder, metrics, epoch, global_step):
    """Build checkpoint dict for saving."""
    action_dim = args.K * 2
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "decoder_state_dict": decoder.state_dict(),
        "metrics": metrics,
        "K": args.K,
        "action_dim": action_dim,
        "num_fourier_feats": args.num_fourier_feats,
        "fourier_max_freq": args.fourier_max_freq,
    }
    if args.decoder_type == "simple":
        ckpt.update(
            {
                "architecture": "simple",
                "hidden_dim": args.ca_hidden_dim,
                "num_layers": args.ca_num_layers,
                "num_heads": args.ca_num_heads,
                "condition_dim": 896,
                "use_action_norm": args.use_action_norm,
            }
        )
        if args.use_action_norm:
            ckpt.update(
                {
                    "accel_mean": decoder.accel_mean.item(),
                    "accel_std": decoder.accel_std.item(),
                    "kappa_mean": decoder.kappa_mean.item(),
                    "kappa_std": decoder.kappa_std.item(),
                }
            )
    elif args.decoder_type == "cross_attention":
        ckpt.update(
            {
                "architecture": "cross_attention",
                "hidden_dim": args.ca_hidden_dim,
                "num_layers": args.ca_num_layers,
                "num_heads": args.ca_num_heads,
                "mlp_ratio": args.ca_mlp_ratio,
                "condition_dim": 896,
                "action_mlp_hidden": args.ca_hidden_dim,
                "action_mlp_layers": 2,
                "dropout": args.ca_dropout,
            }
        )
    else:
        ckpt.update(
            {
                "architecture": "expert_kv_cache",
                "hidden_size": args.hidden_size,
                "num_hidden_layers": args.num_hidden_layers,
                "num_attention_heads": args.num_attention_heads,
                "intermediate_size": args.intermediate_size or args.hidden_size * 4,
                "mlp_hidden_size": args.mlp_hidden_size,
                "mlp_num_layers": args.mlp_num_layers,
                "attention_dropout": args.attention_dropout,
            }
        )
    return ckpt


def main():
    t_start = time.time()
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_dim = args.K * 2
    print(f"Device: {device}")
    print(f"K={args.K}, action_dim={action_dim}, decoder_type={args.decoder_type}")
    if args.decoder_type == "kv_cache":
        print(
            f"Expert: hidden={args.hidden_size}, layers={args.num_hidden_layers}, "
            f"heads={args.num_attention_heads}"
        )
    elif args.decoder_type == "simple":
        print(
            f"Simple: hidden={args.ca_hidden_dim}, layers={args.ca_num_layers}, "
            f"heads={args.ca_num_heads}"
        )
    else:
        print(
            f"CrossAttn: hidden={args.ca_hidden_dim}, layers={args.ca_num_layers}, "
            f"heads={args.ca_num_heads}"
        )

    # Tokenizers (needed for CoCDataset)
    text_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    action_tokenizer = DiscreteActionTokenizer(n_bins=256)

    # Dataset (scene-level split via CoCDataset with image_path filtering)
    print("Loading CoC dataset for Expert training...")
    use_split = args.nuscenes_version == "v1.0-trainval"
    if use_split:
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
        del train_traj, val_traj
    else:
        train_paths = None
        val_paths = None

    train_dataset = CoCDataset(
        annotations_path=args.coc_data,
        tokenizer=text_tokenizer,
        action_tokenizer=action_tokenizer,
        K=args.K,
        allowed_image_paths=train_paths,
    )
    val_dataset = CoCDataset(
        annotations_path=args.coc_data,
        tokenizer=text_tokenizer,
        action_tokenizer=action_tokenizer,
        K=args.K,
        allowed_image_paths=val_paths,
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Curve oversampling
    train_sampler = None
    if args.curve_oversample > 1.0:
        print("Building curve oversampler...")
        train_sampler = _build_curve_sampler(
            train_dataset, args.curve_kappa_threshold, args.curve_oversample
        )

    # batch_size=1 because reasoning_ids length varies per sample
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Frozen VLM (feature extractor + KV-cache generator)
    print("Building frozen VLM...")
    vlm = MiniPamayo(adapter_type="cross_attention", action_dim=action_dim)

    vlm_path = Path(args.vlm_checkpoint)
    if vlm_path.exists():
        print(f"Loading VLM checkpoint: {vlm_path}")
        ckpt = torch.load(vlm_path, map_location="cpu", weights_only=True)
        state_dict = ckpt["model_state_dict"]
        missing, unexpected = vlm.load_state_dict(state_dict, strict=False)
        print(f"  Missing: {len(missing)} keys, Unexpected: {len(unexpected)} keys")
    else:
        raise FileNotFoundError(f"VLM checkpoint not found: {vlm_path}")

    vlm.requires_grad_(False)
    vlm.eval()
    vlm = vlm.to(device)

    # Compute action normalization stats from training set (Alpamayo §5.1)
    print("Computing action normalization stats...")
    action_stats = _compute_action_stats(train_dataset)

    # Flow Matching Decoder (trainable)
    if args.decoder_type == "simple":
        norm_label = "with action_norm" if args.use_action_norm else "no action_norm"
        print(f"Building SimpleDecoder (旧実装, {norm_label})...")
        decoder = SimpleDecoder(
            action_dim=action_dim,
            hidden_dim=args.ca_hidden_dim,
            num_layers=args.ca_num_layers,
            num_heads=args.ca_num_heads,
            condition_dim=896,
            use_action_norm=args.use_action_norm,
            **(action_stats if args.use_action_norm else {}),
        )
    elif args.decoder_type == "cross_attention":
        print("Building CrossAttentionDecoder...")
        decoder = CrossAttentionDecoder(
            K=args.K,
            hidden_dim=args.ca_hidden_dim,
            num_layers=args.ca_num_layers,
            num_heads=args.ca_num_heads,
            mlp_ratio=args.ca_mlp_ratio,
            condition_dim=896,
            num_fourier_feats=args.num_fourier_feats,
            fourier_max_freq=args.fourier_max_freq,
            action_mlp_hidden=args.ca_hidden_dim,
            action_mlp_layers=2,
            dropout=args.ca_dropout,
            **action_stats,
        )
    else:
        print("Building Expert TrajectoryDecoder (KV-cache)...")
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
            attention_dropout=args.attention_dropout,
            **action_stats,
        )
    decoder = decoder.to(device)

    n_params = sum(p.numel() for p in decoder.parameters())
    n_trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Decoder: {n_params:,} total, {n_trainable:,} trainable")

    # Log prompt seq_length for first sample
    sample0 = train_dataset[0]
    pv0 = sample0["pixel_values"].unsqueeze(0).to(device)
    if args.decoder_type == "simple":
        cond0 = extract_conditions(vlm, pv0)
        print(f"Condition shape: {list(cond0.shape)} (mean-pooled)")
        del cond0
    elif args.decoder_type == "cross_attention":
        hs0 = extract_hidden_states(vlm, pv0, sample0, device)
        print(f"Hidden states shape: {list(hs0.shape)} (seq_len={hs0.shape[1]})")
        del hs0
    else:
        kv0, seq0 = extract_kv_cache(vlm, pv0, sample0, device)
        print(f"KV-cache seq_length: {seq0} (was 16 with visual-only)")
        del kv0
    del pv0

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

        run_name = f"stage2-{args.decoder_type}"
        if args.decoder_type == "simple" and args.use_action_norm:
            run_name += "-norm"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    # Initial evaluation
    print("\n=== Initial Evaluation ===")
    init_metrics = evaluate(decoder, vlm, val_loader, device, args.decoder_type)
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

            # Extract condition from frozen VLM → CFM loss
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if args.decoder_type == "simple":
                    condition = extract_conditions(vlm, pixel_values)
                    loss = cfm_loss_simple(decoder, gt_action, condition)
                elif args.decoder_type == "cross_attention":
                    hidden_states = extract_hidden_states(vlm, pixel_values, batch, device)
                    loss = cfm_loss_cross_attn(decoder, gt_action, hidden_states)
                else:
                    kv_cache, prefill_seq_len = extract_kv_cache(vlm, pixel_values, batch, device)
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
        metrics = evaluate(decoder, vlm, val_loader, device, args.decoder_type)
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
            torch.save(
                _build_checkpoint(args, decoder, metrics, epoch + 1, global_step),
                save_dir / "best.pt",
            )

    torch.save(
        _build_checkpoint(args, decoder, metrics, args.max_epochs, global_step),
        save_dir / "final.pt",
    )

    # Summary
    print("\n=== Summary ===")
    final_metrics = evaluate(decoder, vlm, val_loader, device, args.decoder_type)
    print(f"  Val CFM: {init_metrics['val_cfm_loss']:.4f} -> {final_metrics['val_cfm_loss']:.4f}")
    print(f"  Decoder params: {n_trainable:,}")

    total_time = time.time() - t_start
    if torch.cuda.is_available():
        print(f"\nPeak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    print(f"Total time: {total_time:.0f}s ({total_time / 60:.1f}min)")
    print("\nDone.")


if __name__ == "__main__":
    main()
