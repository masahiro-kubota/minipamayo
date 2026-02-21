"""Stage 1 evaluation: discrete action token prediction.

Evaluates token accuracy and trajectory metrics (ADE/FDE) via dequantization.

Usage:
    cd minipamayo && uv run python -m minipamayo.eval_stage1 \
        --checkpoint checkpoints/stage1/best.pt
"""

import argparse

import torch
from torch.utils.data import DataLoader

from .data.nuscenes_trajectory_dataset import NuScenesTrajectoryDataset
from .models.discrete_head import DiscreteActionTokenizer
from .models.dynamics import forward_dynamics_batch
from .models.minipamayo import MiniPamayo


def parse_args():
    parser = argparse.ArgumentParser(description="MiniPamayo Stage 1 evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--nuscenes_root", type=str, default="../cosmos-reason-mini/data/nuscenes")
    parser.add_argument("--nuscenes_version", type=str, default="v1.0-mini")
    parser.add_argument("--show_samples", type=int, default=10)
    return parser.parse_args()


def greedy_generate(model, visual_tokens, n_tokens, device):
    """Greedy autoregressive generation of action tokens."""
    input_embeds = visual_tokens
    generated = []

    for _ in range(n_tokens):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.llm(inputs_embeds=input_embeds)
        logits = outputs.logits[:, -1, :]
        token_id = logits.argmax(dim=-1)  # (B,)
        generated.append(token_id)

        next_embed = model.llm.get_input_embeddings()(token_id.unsqueeze(1))
        input_embeds = torch.cat([input_embeds, next_embed.to(input_embeds.dtype)], dim=1)

    return torch.stack(generated, dim=1)  # (B, n_tokens)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    K = ckpt.get("K", 64)
    n_bins = ckpt.get("n_bins", 256)
    tok_cfg = ckpt.get("tokenizer_config", {})

    tokenizer = DiscreteActionTokenizer(
        n_bins=tok_cfg.get("n_bins", n_bins),
        a_range=tok_cfg.get("a_range", (-6.0, 6.0)),
        kappa_range=tok_cfg.get("kappa_range", (-0.1, 0.1)),
        vocab_offset=tok_cfg.get("vocab_offset", 151936),
    )

    # Dataset
    print("Loading dataset...")
    dataset = NuScenesTrajectoryDataset(
        nuscenes_root=args.nuscenes_root,
        version=args.nuscenes_version,
        K=K,
    )
    print(f"Total: {len(dataset)}, K={K}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # Model
    print("Building model...")
    model = MiniPamayo(adapter_type="cross_attention", action_dim=K * 2)
    new_vocab = tokenizer.vocab_offset + n_bins
    model.llm.resize_token_embeddings(new_vocab)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded: {args.checkpoint}")
    if "metrics" in ckpt:
        print(f"  Saved metrics: {ckpt['metrics']}")

    # Evaluate
    print(f"\n{'=' * 70}")
    print("Evaluation (teacher-forced + autoregressive)")
    print(f"{'=' * 70}")

    all_tf_correct = 0
    all_ar_correct = 0
    total_tokens = 0
    all_ar_actions = []
    all_gt_actions = []
    all_v0 = []
    all_gt_waypoints = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            pixel_values = batch["pixel_values"].to(device)
            gt_action = batch["action"].to(device)
            gt_token_ids = tokenizer.encode_batch(gt_action)

            # Vision + Adapter
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                patch_features = model.vision_encoder(pixel_values)
                visual_tokens = model.adapter(patch_features)

            # Teacher-forced evaluation
            n_vis = visual_tokens.shape[1]
            action_embeds = model.llm.get_input_embeddings()(gt_token_ids)
            inputs_embeds = torch.cat([visual_tokens.to(action_embeds.dtype), action_embeds], dim=1)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model.llm(inputs_embeds=inputs_embeds)

            tf_logits = outputs.logits[:, n_vis - 1 : -1, :]
            tf_pred = tf_logits.argmax(dim=-1)
            all_tf_correct += (tf_pred == gt_token_ids).sum().item()

            # Autoregressive generation
            ar_tokens = greedy_generate(model, visual_tokens, K * 2, device)
            all_ar_correct += (ar_tokens == gt_token_ids).sum().item()
            total_tokens += gt_token_ids.numel()

            # Decode AR tokens to continuous actions
            ar_action = tokenizer.decode_batch(ar_tokens.cpu())
            all_ar_actions.append(ar_action.squeeze(0))
            all_gt_actions.append(gt_action.cpu().squeeze(0))
            all_v0.append(batch["v0"].squeeze())
            all_gt_waypoints.append(batch["gt_waypoints"].squeeze(0))

            if i < args.show_samples:
                gt_ids = gt_token_ids[0].cpu().tolist()
                ar_ids = ar_tokens[0].cpu().tolist()
                match = sum(1 for a, b in zip(gt_ids, ar_ids, strict=False) if a == b)
                print(f"  [{i:3d}] AR match: {match}/{K * 2}")

    # Metrics
    tf_acc = all_tf_correct / max(total_tokens, 1)
    ar_acc = all_ar_correct / max(total_tokens, 1)

    print(f"\n{'=' * 70}")
    print("Token Accuracy")
    print(f"{'=' * 70}")
    print(f"  Teacher-forced: {tf_acc:.4f}")
    print(f"  Autoregressive: {ar_acc:.4f}")

    # Trajectory metrics from AR generation
    ar_actions = torch.stack(all_ar_actions)  # (N, K*2)
    gt_actions = torch.stack(all_gt_actions)
    v0s = torch.stack(all_v0)
    gt_wp = torch.stack(all_gt_waypoints)

    ar_kv = ar_actions.reshape(-1, K, 2)
    gt_kv = gt_actions.reshape(-1, K, 2)

    # Action-space MAE
    a_mae = (ar_kv[:, :, 0] - gt_kv[:, :, 0]).abs().mean().item()
    kappa_mae = (ar_kv[:, :, 1] - gt_kv[:, :, 1]).abs().mean().item()

    # Forward dynamics -> ADE/FDE
    pred_wp = forward_dynamics_batch(ar_kv[:, :, 0], ar_kv[:, :, 1], v0s, dt=0.5)
    disp_errors = torch.norm(pred_wp - gt_wp, dim=2)
    ade = disp_errors.mean().item()
    fde = disp_errors[:, -1].mean().item()

    print(f"\n{'=' * 70}")
    print("Action-Space Metrics (from AR generation)")
    print(f"{'=' * 70}")
    print(f"  a MAE:     {a_mae:.6f}")
    print(f"  kappa MAE: {kappa_mae:.6f}")

    print(f"\n{'=' * 70}")
    print("Trajectory Metrics (via forward dynamics)")
    print(f"{'=' * 70}")
    print(f"  ADE: {ade:.4f} m")
    print(f"  FDE: {fde:.4f} m")

    # Token distribution
    print(f"\n{'=' * 70}")
    print("Token Distribution (AR)")
    print(f"{'=' * 70}")
    all_ar_tokens = []
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                patch_features = model.vision_encoder(pixel_values)
                visual_tokens = model.adapter(patch_features)
            ar_tokens = greedy_generate(model, visual_tokens, K * 2, device)
            all_ar_tokens.append(ar_tokens.cpu())

    all_ar_tokens = torch.cat(all_ar_tokens, dim=0)  # (N, K*2)
    bins_used = (all_ar_tokens - tokenizer.vocab_offset).unique()
    print(f"  Unique bins used: {len(bins_used)} / {n_bins}")
    print(f"  Bin range: [{bins_used.min().item()}, {bins_used.max().item()}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
