"""カーブシーンを特定し、Stage 2 Flow Matching デコーダで可視化する。

Usage:
    cd minipamayo && uv run python scripts/find_curve_scenes.py \
        --decoder_checkpoint checkpoints/stage2-simple/best.pt
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")


@torch.no_grad()
def vlm_mean_pool(vlm, pixel_values):
    """Extract mean-pooled VLM hidden states (for SimpleDecoder)."""
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        patch_features = vlm.vision_encoder(pixel_values)
        visual_tokens = vlm.adapter(patch_features)
        outputs = vlm.llm(inputs_embeds=visual_tokens, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1]  # (B, L, 896)
    return last_hidden.mean(dim=1).float()  # (B, 896)


@torch.no_grad()
def vlm_hidden_states(vlm, pixel_values):
    """Extract VLM hidden states from visual tokens only (no reasoning prompt)."""
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        patch_features = vlm.vision_encoder(pixel_values)
        visual_tokens = vlm.adapter(patch_features)
        outputs = vlm.llm(inputs_embeds=visual_tokens, output_hidden_states=True)
    return outputs.hidden_states[-1]


@torch.no_grad()
def vlm_kv_cache(vlm, pixel_values):
    """Extract VLM KV-cache from visual tokens only."""
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        patch_features = vlm.vision_encoder(pixel_values)
        visual_tokens = vlm.adapter(patch_features)
        outputs = vlm.llm(inputs_embeds=visual_tokens, use_cache=True)
    return outputs.past_key_values, outputs.past_key_values.get_seq_length()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decoder_checkpoint",
        type=str,
        default="checkpoints/stage2-simple/best.pt",
    )
    parser.add_argument(
        "--vlm_checkpoint",
        type=str,
        default="checkpoints/stage1/best.pt",
    )
    parser.add_argument("--n_vis", type=int, default=10)
    parser.add_argument("--n_flow_samples", type=int, default=10)
    return parser.parse_args()


def main():
    from minipamayo.data.nuscenes_trajectory_dataset import NuScenesTrajectoryDataset
    from minipamayo.models.dynamics import forward_dynamics_batch
    from minipamayo.models.minipamayo import MiniPamayo
    from minipamayo.models.trajectory_decoder import (
        cfm_sample,
        cfm_sample_cross_attn,
        cfm_sample_simple,
        load_decoder_from_checkpoint,
    )

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Step 1: データセット読み込み & カーブシーン抽出 ---
    print("=" * 70)
    print("Step 1: Loading dataset and finding curve scenes")
    print("=" * 70)

    print("Loading NuScenes dataset...")
    dataset = NuScenesTrajectoryDataset(
        nuscenes_root="/mnt/ssd/nuscenes", version="v1.0-trainval", K=6
    )
    total = len(dataset)
    print(f"Total samples: {total}")

    # GT軌跡の横方向変位でカーブを判定
    LAT_THRESHOLD = 3.0  # meters
    curve_scenes = []

    print("Scanning for curve scenes (using pre-computed waypoints, no image loading)...")
    for i in range(total):
        if i % 5000 == 0:
            print(f"  {i}/{total} scanned, {len(curve_scenes)} curves found so far")
        wp = dataset.samples[i]["gt_waypoints"]  # numpy array, no image load
        lat_disp = float(wp[-1, 1])  # 最終waypoint の横方向変位
        abs_lat = abs(lat_disp)
        if abs_lat > LAT_THRESHOLD:
            curve_scenes.append(
                {
                    "idx": i,
                    "abs_lat": abs_lat,
                    "final_x": float(wp[-1, 0]),
                    "final_y": lat_disp,
                    "direction": "left" if lat_disp > 0 else "right",
                }
            )
    print(f"  {total}/{total} scanned, {len(curve_scenes)} curves found")

    print(f"\nCurve scenes (|lateral| > {LAT_THRESHOLD}m): {len(curve_scenes)}")
    curve_scenes.sort(key=lambda x: -x["abs_lat"])

    print("\nTop 20 curve scenes:")
    for cs in curve_scenes[:20]:
        print(
            f"  idx={cs['idx']:5d}  |lat|={cs['abs_lat']:.2f}m  "
            f"final=({cs['final_x']:.1f}, {cs['final_y']:.1f})  dir={cs['direction']}"
        )

    if not curve_scenes:
        print("ERROR: No curve scenes found!")
        return

    # --- Step 2: モデル読み込み ---
    print("\n" + "=" * 70)
    print("Step 2: Loading VLM and Flow Matching decoder")
    print("=" * 70)

    K = 6
    action_dim = K * 2

    # VLM
    vlm = MiniPamayo(adapter_type="cross_attention", action_dim=action_dim)
    vlm_ckpt = torch.load(args.vlm_checkpoint, map_location="cpu", weights_only=True)
    vlm.load_state_dict(vlm_ckpt["model_state_dict"], strict=False)
    vlm = vlm.to(device).eval()
    vlm.requires_grad_(False)
    print(f"Loaded VLM: {args.vlm_checkpoint}")

    # Decoder (auto-detect architecture)
    decoder, _, ckpt_info = load_decoder_from_checkpoint(args.decoder_checkpoint, device)
    arch = ckpt_info.get("architecture", "expert_kv_cache")
    print(f"Loaded decoder: {args.decoder_checkpoint} (arch={arch})")

    # --- Step 3: カーブシーンで推論 ---
    print("\n" + "=" * 70)
    print("Step 3: Inference on curve scenes")
    print("=" * 70)

    n_vis = min(args.n_vis, len(curve_scenes))
    n_flow_samples = args.n_flow_samples
    selected = curve_scenes[:n_vis]

    print(f"Visualizing {n_vis} scenes, {n_flow_samples} flow samples each")
    print(f"Architecture: {arch}\n")

    fig, axes = plt.subplots(n_vis, 2, figsize=(14, 4 * n_vis))
    if n_vis == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for row, cs in enumerate(selected):
            idx = cs["idx"]
            print(f"--- [{row + 1}/{n_vis}] Processing idx={idx} ---")
            sample = dataset[idx]
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
            v0 = sample["v0"]
            gt_wp = sample["gt_waypoints"].numpy()

            # VLM conditioning (architecture-dependent)
            if arch == "simple":
                condition = vlm_mean_pool(vlm, pixel_values)
            elif arch == "cross_attention":
                hidden_states = vlm_hidden_states(vlm, pixel_values)
            else:
                kv_cache, prefill_seq_len = vlm_kv_cache(vlm, pixel_values)

            # Multiple flow samples
            flow_wps = []
            ade_list = []
            raw_actions = []
            for _s in range(n_flow_samples):
                if arch == "simple":
                    pred_action = cfm_sample_simple(decoder, condition, n_steps=20)
                elif arch == "cross_attention":
                    pred_action = cfm_sample_cross_attn(decoder, hidden_states, n_steps=20)
                else:
                    pred_action = cfm_sample(decoder, kv_cache, prefill_seq_len, n_steps=20)
                raw_actions.append(pred_action.cpu().squeeze())
                pred_kv = pred_action.cpu().squeeze().reshape(K, 2)
                wp = (
                    forward_dynamics_batch(
                        pred_kv[:, 0].unsqueeze(0),
                        pred_kv[:, 1].unsqueeze(0),
                        v0.unsqueeze(0),
                        dt=0.5,
                    )
                    .squeeze(0)
                    .numpy()
                )
                flow_wps.append(wp)
                ade = np.linalg.norm(wp - gt_wp, axis=1).mean()
                ade_list.append(ade)

            best_idx = int(np.argmin(ade_list))
            best_wp = flow_wps[best_idx]

            # Action statistics
            actions_stack = torch.stack(raw_actions)  # (n_samples, K*2)
            actions_kv = actions_stack.reshape(n_flow_samples, K, 2)
            kappa_mean = actions_kv[:, :, 1].mean(dim=0)
            kappa_std = actions_kv[:, :, 1].std(dim=0)

            # GT action
            gt_action = sample["action"].reshape(K, 2)

            print(f"[{row}] idx={idx} dir={cs['direction']} |lat|={cs['abs_lat']:.2f}m")
            print(f"  GT   kappa: {gt_action[:, 1].tolist()}")
            print(f"  Pred kappa mean: {kappa_mean.tolist()}")
            print(f"  Pred kappa std:  {kappa_std.tolist()}")
            print(
                f"  ADE: best={ade_list[best_idx]:.2f}m, worst={max(ade_list):.2f}m, "
                f"mean={np.mean(ade_list):.2f}m"
            )
            print(f"  All ADE: {[f'{a:.2f}' for a in ade_list]}")
            print()

            # --- Plot ---
            # Image
            img = Image.open(dataset.samples[idx]["image_path"]).convert("RGB")
            axes[row, 0].imshow(img)
            axes[row, 0].axis("off")
            axes[row, 0].set_title(
                f"Sample {idx} ({cs['direction']}, |lat|={cs['abs_lat']:.1f}m)", fontsize=9
            )

            # BEV: +x=forward, +y=left → plot(-y, x)
            ax = axes[row, 1]
            gt = gt_wp
            ax.plot(-gt[:, 1], gt[:, 0], "b-o", markersize=4, linewidth=2, label="GT", zorder=5)

            bw = best_wp
            ax.plot(-bw[:, 1], bw[:, 0], "r--^", markersize=4, linewidth=2, label="Best", zorder=5)

            for j, fw in enumerate(flow_wps):
                label = "Samples" if j == 0 else None
                ax.plot(
                    -fw[:, 1],
                    fw[:, 0],
                    color="salmon",
                    alpha=0.4,
                    linewidth=1,
                    label=label,
                    zorder=3,
                )

            ax.plot(0, 0, "ko", markersize=6, zorder=10)
            ax.set_aspect("equal")
            ax.legend(fontsize=7, loc="lower right")
            ax.set_xlabel("lateral (m)", fontsize=8)
            ax.set_ylabel("forward (m)", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_title(
                f"Stage 2 Flow — minADE={ade_list[best_idx]:.2f}m ({n_flow_samples} samples)",
                fontsize=9,
            )

    arch_labels = {
        "simple": "Simple",
        "cross_attention": "Cross-Attn",
        "expert_kv_cache": "KV-Cache Expert",
    }
    arch_label = arch_labels.get(arch, arch)
    fig.suptitle(
        f"Stage 2: Flow Matching ({arch_label}) on CURVE Scenes",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    suffix_map = {
        "simple": "simple",
        "cross_attention": "cross_attn",
        "expert_kv_cache": "kv_cache",
    }
    suffix = suffix_map.get(arch, arch)
    out_path = f"outputs/vis_stage2_curves_{suffix}.png"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
