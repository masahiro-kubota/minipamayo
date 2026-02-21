"""Stage 2 evaluation: Flow Matching trajectory decoder.

Evaluates CFM loss, samples trajectories, computes ADE/FDE and diversity.

Usage:
    cd minipamayo && uv run python -m minipamayo.eval_stage2 \
        --decoder_checkpoint checkpoints/stage2/best.pt \
        --phase4_checkpoint checkpoints/phase4/best.pt
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data.nuscenes_trajectory_dataset import NuScenesTrajectoryDataset
from .models.dynamics import forward_dynamics_batch
from .models.minipamayo import MiniPamayo
from .models.trajectory_decoder import TrajectoryDecoder, cfm_loss, cfm_sample


def parse_args():
    parser = argparse.ArgumentParser(description="MiniPamayo Stage 2 evaluation")
    parser.add_argument("--decoder_checkpoint", type=str, required=True)
    parser.add_argument(
        "--phase4_checkpoint",
        type=str,
        default="checkpoints/phase4/best.pt",
    )
    parser.add_argument(
        "--vlm_checkpoint",
        type=str,
        default="../cosmos-reason-mini/checkpoints/rl-mini-merged/checkpoint-final.pt",
    )
    parser.add_argument("--nuscenes_root", type=str, default="../cosmos-reason-mini/data/nuscenes")
    parser.add_argument("--nuscenes_version", type=str, default="v1.0-mini")
    parser.add_argument("--n_steps", type=int, default=20, help="Euler integration steps")
    parser.add_argument("--n_samples", type=int, default=5, help="Samples per input for diversity")
    parser.add_argument("--show_samples", type=int, default=10)
    return parser.parse_args()


@torch.no_grad()
def extract_conditions(vlm, pixel_values):
    """Extract LLM hidden states as condition for flow decoder."""
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        patch_features = vlm.vision_encoder(pixel_values)
        visual_tokens = vlm.adapter(patch_features)
        outputs = vlm.llm(inputs_embeds=visual_tokens, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1]
    return last_hidden.mean(dim=1).float()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load decoder checkpoint
    ckpt = torch.load(args.decoder_checkpoint, map_location="cpu", weights_only=True)
    K = ckpt.get("K", 6)
    action_dim = ckpt.get("action_dim", K * 2)
    hidden_dim = ckpt.get("hidden_dim", 256)
    num_layers = ckpt.get("num_layers", 4)
    num_heads = ckpt.get("num_heads", 4)
    condition_dim = ckpt.get("condition_dim", 896)

    print(
        f"Decoder config: action_dim={action_dim}, hidden={hidden_dim}, "
        f"layers={num_layers}, heads={num_heads}"
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

    # Frozen VLM
    print("Building frozen VLM...")
    vlm = MiniPamayo(adapter_type="cross_attention", action_dim=action_dim)

    phase4_path = Path(args.phase4_checkpoint)
    vlm_path = Path(args.vlm_checkpoint)
    if phase4_path.exists():
        p4_ckpt = torch.load(phase4_path, map_location="cpu", weights_only=True)
        vlm.load_state_dict(p4_ckpt["model_state_dict"], strict=False)
        print(f"Loaded VLM from {phase4_path}")
    elif vlm_path.exists():
        vlm.load_vlm_checkpoint(vlm_path)
    vlm.requires_grad_(False)
    vlm.eval()
    vlm = vlm.to(device)

    # Decoder
    print("Building TrajectoryDecoder...")
    decoder = TrajectoryDecoder(
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        condition_dim=condition_dim,
    )
    decoder.load_state_dict(ckpt["decoder_state_dict"])
    decoder = decoder.to(device)
    decoder.eval()

    print(f"Loaded decoder: {args.decoder_checkpoint}")
    if "metrics" in ckpt:
        print(f"  Saved metrics: {ckpt['metrics']}")

    # Evaluate
    print(f"\n{'=' * 70}")
    print(f"Evaluation (n_steps={args.n_steps})")
    print(f"{'=' * 70}")

    all_cfm_loss = 0.0
    all_pred_actions = []
    all_gt_actions = []
    all_v0 = []
    all_gt_waypoints = []
    n = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            pixel_values = batch["pixel_values"].to(device)
            gt_action = batch["action"].to(device)

            condition = extract_conditions(vlm, pixel_values)

            # CFM loss
            loss = cfm_loss(decoder, gt_action, condition)
            all_cfm_loss += loss.item()

            # Sample trajectory
            pred_action = cfm_sample(decoder, condition, action_dim, n_steps=args.n_steps)
            all_pred_actions.append(pred_action.cpu().squeeze(0))
            all_gt_actions.append(gt_action.cpu().squeeze(0))
            all_v0.append(batch["v0"].squeeze())
            all_gt_waypoints.append(batch["gt_waypoints"].squeeze(0))
            n += 1

            if i < args.show_samples:
                pred = pred_action.cpu().squeeze()
                gt = gt_action.cpu().squeeze()
                pred_kv = pred.reshape(K, 2)
                gt_kv = gt.reshape(K, 2)
                a_err = (pred_kv[:, 0] - gt_kv[:, 0]).abs().mean().item()
                k_err = (pred_kv[:, 1] - gt_kv[:, 1]).abs().mean().item()
                print(f"  [{i:3d}] a_MAE={a_err:.4f}  kappa_MAE={k_err:.4f}")

    # Aggregate metrics
    preds = torch.stack(all_pred_actions)  # (N, K*2)
    gts = torch.stack(all_gt_actions)
    v0s = torch.stack(all_v0)
    gt_wp = torch.stack(all_gt_waypoints)

    preds_kv = preds.reshape(-1, K, 2)
    gts_kv = gts.reshape(-1, K, 2)
    errors_kv = (preds_kv - gts_kv).abs()

    print(f"\n{'=' * 70}")
    print("CFM Loss")
    print(f"{'=' * 70}")
    print(f"  Mean CFM loss: {all_cfm_loss / n:.6f}")

    print(f"\n{'=' * 70}")
    print("Action-Space Metrics (single sample)")
    print(f"{'=' * 70}")
    print(f"  a MAE:     {errors_kv[:, :, 0].mean():.6f}")
    print(f"  kappa MAE: {errors_kv[:, :, 1].mean():.6f}")

    # Trajectory via forward dynamics
    pred_a = preds_kv[:, :, 0]
    pred_kappa = preds_kv[:, :, 1]
    pred_wp = forward_dynamics_batch(pred_a, pred_kappa, v0s, dt=0.1)

    disp_errors = torch.norm(pred_wp - gt_wp, dim=2)
    ade = disp_errors.mean().item()
    fde = disp_errors[:, -1].mean().item()

    print(f"\n{'=' * 70}")
    print("Trajectory Metrics (via forward dynamics)")
    print(f"{'=' * 70}")
    print(f"  ADE: {ade:.4f} m")
    print(f"  FDE: {fde:.4f} m")

    print("\n  Per-timestep ADE:")
    for t in range(K):
        t_ade = disp_errors[:, t].mean().item()
        print(f"    t={t} ({(t + 1) * 0.1:.1f}s): {t_ade:.4f} m")

    # Distribution
    print(f"\n{'=' * 70}")
    print("Distribution Statistics")
    print(f"{'=' * 70}")
    for ch, name in [(0, "Acceleration (a)"), (1, "Curvature (kappa)")]:
        p = preds_kv[:, :, ch]
        g = gts_kv[:, :, ch]
        print(f"\n  {name}:")
        print(
            f"    GT   -- mean: {g.mean():+.5f}, std: {g.std():.5f}, "
            f"min: {g.min():+.5f}, max: {g.max():+.5f}"
        )
        print(
            f"    Pred -- mean: {p.mean():+.5f}, std: {p.std():.5f}, "
            f"min: {p.min():+.5f}, max: {p.max():+.5f}"
        )

    # Multi-sample diversity analysis
    print(f"\n{'=' * 70}")
    print(f"Diversity Analysis ({args.n_samples} samples per input)")
    print(f"{'=' * 70}")

    all_diversity = []
    all_min_ade = []
    n_diversity = min(50, len(dataset))  # limit for speed

    with torch.no_grad():
        for i in range(n_diversity):
            sample = dataset[i]
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
            gt_action = sample["action"].unsqueeze(0).to(device)
            v0 = sample["v0"].unsqueeze(0)
            gt_waypoint = sample["gt_waypoints"].unsqueeze(0)

            condition = extract_conditions(vlm, pixel_values)

            # Generate multiple samples
            samples = []
            for _ in range(args.n_samples):
                s = cfm_sample(decoder, condition, action_dim, n_steps=args.n_steps)
                samples.append(s.cpu())
            samples = torch.stack(samples, dim=1).squeeze(0)  # (n_samples, K*2)

            # Diversity: mean pairwise L2 distance between samples
            diffs = samples.unsqueeze(0) - samples.unsqueeze(1)  # (n, n, K*2)
            pairwise_dist = diffs.norm(dim=-1)
            mask = torch.triu(
                torch.ones(args.n_samples, args.n_samples, dtype=torch.bool), diagonal=1
            )
            diversity = pairwise_dist[mask].mean().item()
            all_diversity.append(diversity)

            # minADE: best of n_samples
            sample_kv = samples.reshape(args.n_samples, K, 2)
            v0_i = v0.expand(args.n_samples)

            pred_wp_i = forward_dynamics_batch(sample_kv[:, :, 0], sample_kv[:, :, 1], v0_i, dt=0.1)
            gt_wp_i = gt_waypoint.expand(args.n_samples, -1, -1)
            disp_i = torch.norm(pred_wp_i - gt_wp_i, dim=2)
            ade_per_sample = disp_i.mean(dim=1)
            all_min_ade.append(ade_per_sample.min().item())

    mean_diversity = sum(all_diversity) / len(all_diversity)
    mean_min_ade = sum(all_min_ade) / len(all_min_ade)

    print(f"  Mean pairwise diversity: {mean_diversity:.4f}")
    print(f"  minADE ({args.n_samples} samples):  {mean_min_ade:.4f} m")
    print(f"  ADE (single sample):     {ade:.4f} m")
    print(f"  Improvement from multi:  {(ade - mean_min_ade) / ade * 100:.1f}%")

    # Input dependency
    print(f"\n{'=' * 70}")
    print("Input Dependency")
    print(f"{'=' * 70}")
    pred_var = preds.var(dim=0).mean()
    print(f"  Mean pred variance: {pred_var:.6f}")
    print(f"  Status: {'COLLAPSED' if pred_var < 1e-8 else 'OK (input-dependent)'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
