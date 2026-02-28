"""Visualization for all evaluation stages.

Generates a PNG grid showing input images, BEV trajectory plots,
and (for Stage 3/4) CoC reasoning text side by side.

Usage:
    cd minipamayo && uv run python -m minipamayo.visualize \
        --stage phase4 --checkpoint checkpoints/phase4/best.pt

    uv run python -m minipamayo.visualize \
        --stage stage1 --checkpoint checkpoints/stage1/best.pt

    uv run python -m minipamayo.visualize \
        --stage stage2 --decoder_checkpoint checkpoints/stage2/best.pt \
        --vlm_checkpoint checkpoints/phase4/best.pt

    uv run python -m minipamayo.visualize \
        --stage stage3 --checkpoint checkpoints/stage3/best.pt

    uv run python -m minipamayo.visualize \
        --stage stage4 --checkpoint checkpoints/stage4/best.pt \
        --ref_checkpoint checkpoints/stage3/best.pt
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer

from .data.coc_dataset import CoCDataset, build_chat_token_ids, format_coc_text
from .data.nuscenes_trajectory_dataset import NuScenesTrajectoryDataset
from .models.discrete_head import DiscreteActionTokenizer
from .models.dynamics import forward_dynamics_batch
from .models.minipamayo import MiniPamayo
from .models.trajectory_decoder import cfm_sample, load_decoder_from_checkpoint

matplotlib.use("Agg")

N_VIS = 5  # number of samples to visualize
K_DEFAULT = 6


# ============================================================
# Argument parsing
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(description="MiniPamayo visualization")
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["phase4", "stage1", "stage2", "stage3", "stage3_rollouts", "stage4", "coc_labels"],
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--decoder_checkpoint", type=str, default=None)
    parser.add_argument("--vlm_checkpoint", type=str, default=None)
    parser.add_argument("--ref_checkpoint", type=str, default=None)
    parser.add_argument("--coc_data", type=str, default="data/coc_annotations_trainval.jsonl")
    parser.add_argument("--nuscenes_root", type=str, default="/mnt/ssd/nuscenes")
    parser.add_argument("--nuscenes_version", type=str, default="v1.0-trainval")
    parser.add_argument("--n_samples", type=int, default=5, help="Flow matching samples (stage2)")
    parser.add_argument("--n_rollouts", type=int, default=8, help="Temperature-sampled rollouts")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--n_vis", type=int, default=N_VIS, help="Number of samples to visualize")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--K", type=int, default=K_DEFAULT)
    parser.add_argument(
        "--curves", action="store_true", help="Select curve scenes (|lateral| > 3m)"
    )
    parser.add_argument(
        "--curve_offset", type=int, default=0, help="Skip first N diverse curve scenes"
    )
    return parser.parse_args()


# ============================================================
# BEV trajectory plot
# ============================================================


def draw_bev_plot(
    ax,
    gt_waypoints,
    pred_waypoints=None,
    sft_waypoints=None,
    flow_samples=None,
    obstacles=None,
    ade=None,
    fde=None,
    title=None,
):
    """Draw BEV trajectory plot on given axes.

    Coordinate convention: ego-centric +x=forward, +y=left.
    BEV display: forward=up, right=right → plot(-y, x).
    """
    # GT trajectory (blue)
    gt = np.array(gt_waypoints)
    ax.plot(-gt[:, 1], gt[:, 0], "b-o", markersize=4, linewidth=2, label="GT", zorder=5)

    # Pred trajectory (red)
    if pred_waypoints is not None:
        pred = np.array(pred_waypoints)
        ax.plot(
            -pred[:, 1],
            pred[:, 0],
            "r--^",
            markersize=4,
            linewidth=2,
            label="Pred",
            zorder=5,
        )

    # SFT trajectory (green, for Stage 4)
    if sft_waypoints is not None:
        sft = np.array(sft_waypoints)
        ax.plot(
            -sft[:, 1],
            sft[:, 0],
            "g--s",
            markersize=3,
            linewidth=1.5,
            label="SFT",
            zorder=4,
        )

    # Flow matching multi-samples (thin red)
    if flow_samples is not None:
        for j, fs in enumerate(flow_samples):
            fs = np.array(fs)
            label = "Samples" if j == 0 else None
            ax.plot(
                -fs[:, 1],
                fs[:, 0],
                color="salmon",
                alpha=0.4,
                linewidth=1,
                label=label,
                zorder=3,
            )

    # Obstacles (gray rectangles)
    if obstacles:
        for obs in obstacles:
            cx, cy = obs["center"]
            w, length = obs["size"]
            heading = obs["heading"]
            # BEV transform: display_x = -cy, display_y = cx
            rect = mpatches.FancyBboxPatch(
                (-cy - w / 2, cx - length / 2),
                w,
                length,
                boxstyle="round,pad=0",
                facecolor="gray",
                alpha=0.3,
                edgecolor="gray",
                linewidth=0.5,
                zorder=2,
            )
            t = matplotlib.transforms.Affine2D().rotate_around(-cy, cx, heading) + ax.transData
            rect.set_transform(t)
            ax.add_patch(rect)

    # Ego vehicle marker at origin
    ax.annotate(
        "",
        xy=(0, 0.8),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="black", lw=2),
        zorder=10,
    )
    ax.plot(0, 0, "ko", markersize=6, zorder=10)

    # Metrics text
    metrics_lines = []
    if ade is not None:
        metrics_lines.append(f"ADE: {ade:.2f}m")
    if fde is not None:
        metrics_lines.append(f"FDE: {fde:.2f}m")
    if metrics_lines:
        ax.text(
            0.02,
            0.98,
            "\n".join(metrics_lines),
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlabel("lateral (m)", fontsize=8)
    ax.set_ylabel("forward (m)", fontsize=8)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=9)


# ============================================================
# CoC text panel
# ============================================================


def draw_coc_text(ax, text, title="CoC"):
    """Draw CoC reasoning text on axes."""
    ax.axis("off")
    ax.set_title(title, fontsize=9)
    # Wrap long text
    wrapped = "\n".join(_wrap_text(text, width=50))
    ax.text(
        0.02,
        0.98,
        wrapped,
        transform=ax.transAxes,
        fontsize=6,
        verticalalignment="top",
        fontfamily="monospace",
        wrap=True,
    )


def _wrap_text(text, width=50):
    """Simple text wrapping."""
    lines = []
    for line in text.split("\n"):
        while len(line) > width:
            split_pos = line.rfind(" ", 0, width)
            if split_pos == -1:
                split_pos = width
            lines.append(line[:split_pos])
            line = line[split_pos:].lstrip()
        lines.append(line)
    return lines


# ============================================================
# Inference helpers (reused from eval scripts)
# ============================================================


def greedy_generate_stage1(model, visual_tokens, n_tokens, device):
    """Greedy AR generation for Stage 1."""
    input_embeds = visual_tokens
    generated = []
    for _ in range(n_tokens):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.llm(inputs_embeds=input_embeds)
        logits = outputs.logits[:, -1, :]
        token_id = logits.argmax(dim=-1)
        generated.append(token_id)
        next_embed = model.llm.get_input_embeddings()(token_id.unsqueeze(1))
        input_embeds = torch.cat([input_embeds, next_embed.to(input_embeds.dtype)], dim=1)
    return torch.stack(generated, dim=1)


@torch.no_grad()
def extract_kv_cache(vlm, pixel_values):
    """Extract VLM KV-cache for Expert conditioning."""
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        patch_features = vlm.vision_encoder(pixel_values)
        visual_tokens = vlm.adapter(patch_features)
        outputs = vlm.llm(inputs_embeds=visual_tokens, use_cache=True)
    kv_cache = outputs.past_key_values
    prefill_seq_len = kv_cache.get_seq_length()
    return kv_cache, prefill_seq_len


def autoregressive_generate(model, prompt_embeds, max_tokens):
    """Generate reasoning + action tokens autoregressively."""
    input_embeds = prompt_embeds.clone()
    generated = []
    for _ in range(max_tokens):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.llm(inputs_embeds=input_embeds)
        logits = outputs.logits[:, -1, :]
        token_id = logits.argmax(dim=-1)
        generated.append(token_id.item())
        if token_id.item() in (151643, 151645):  # EOS / <|im_end|>
            break
        next_embed = model.llm.get_input_embeddings()(token_id.unsqueeze(0))
        input_embeds = torch.cat([input_embeds, next_embed.to(input_embeds.dtype)], dim=1)
    return generated


def temperature_generate(model, prompt_embeds, max_tokens, temperature=0.8):
    """Generate with temperature sampling (same method as GRPO rollout).

    Unlike greedy autoregressive_generate, this uses multinomial sampling
    to produce diverse trajectory candidates.
    """
    input_embeds = prompt_embeds.clone()
    generated = []
    for _ in range(max_tokens):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.llm(inputs_embeds=input_embeds)
        logits = outputs.logits[:, -1, :] / temperature
        probs = torch.softmax(logits.float(), dim=-1)
        token_id = torch.multinomial(probs, 1).squeeze(-1)
        generated.append(token_id.item())
        if token_id.item() in (151643, 151645):  # EOS / <|im_end|>
            break
        next_embed = model.llm.get_input_embeddings()(token_id.unsqueeze(0))
        input_embeds = torch.cat([input_embeds, next_embed.to(input_embeds.dtype)], dim=1)
    return generated


def build_eval_prompt_embeds(model, pixel_values, v0, text_tokenizer, device):
    """Build prompt embeddings for Stage 3/4 AR generation."""
    chat_ids = build_chat_token_ids(text_tokenizer, v0)
    embed_layer = model.llm.get_input_embeddings()

    system_embeds = embed_layer(
        torch.tensor(chat_ids["system_ids"], dtype=torch.long, device=device).unsqueeze(0)
    )
    user_prefix_embeds = embed_layer(
        torch.tensor(chat_ids["user_prefix_ids"], dtype=torch.long, device=device).unsqueeze(0)
    )
    ego_question_embeds = embed_layer(
        torch.tensor(chat_ids["ego_question_ids"], dtype=torch.long, device=device).unsqueeze(0)
    )
    asst_prefix_embeds = embed_layer(
        torch.tensor(chat_ids["asst_prefix_ids"], dtype=torch.long, device=device).unsqueeze(0)
    )

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        patch_features = model.vision_encoder(pixel_values)
        visual_embeds = model.adapter(patch_features)

    target_dtype = system_embeds.dtype
    return torch.cat(
        [
            system_embeds,
            user_prefix_embeds,
            visual_embeds.to(target_dtype),
            ego_question_embeds,
            asst_prefix_embeds,
        ],
        dim=1,
    )


def parse_from_tokens(token_ids, text_tokenizer, action_tokenizer, K):
    """Parse decision, action, and text from generated tokens."""
    vocab_offset = action_tokenizer.vocab_offset
    n_bins = action_tokenizer.n_bins

    action_ids = [t for t in token_ids if vocab_offset <= t < vocab_offset + n_bins]
    if len(action_ids) >= K * 2:
        action_ids = action_ids[: K * 2]
    else:
        action_ids = action_ids + [vocab_offset] * (K * 2 - len(action_ids))
    action = action_tokenizer.decode(action_ids)

    text_tokens = [t for t in token_ids if t < vocab_offset]
    text = text_tokenizer.decode(text_tokens, skip_special_tokens=True)

    decision = {}
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("longitudinal:"):
            decision["longitudinal"] = line.split(":", 1)[1].strip()
        elif line.startswith("lateral:"):
            decision["lateral"] = line.split(":", 1)[1].strip()

    return action, decision, text


# ============================================================
# Per-stage visualization
# ============================================================


def vis_phase4(args, device):
    """Phase 4: regression-based continuous action."""
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    K = ckpt.get("K", args.K)
    adapter_type = ckpt.get("adapter_type", "cross_attention")

    model = MiniPamayo(adapter_type=adapter_type, action_dim=K * 2)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    dataset = NuScenesTrajectoryDataset(
        nuscenes_root=args.nuscenes_root, version=args.nuscenes_version, K=K
    )

    fig, axes = plt.subplots(args.n_vis, 2, figsize=(12, 4 * args.n_vis))
    if args.n_vis == 1:
        axes = axes.reshape(1, -1)

    indices = _select_indices(len(dataset), args.n_vis)

    with torch.no_grad():
        for row, idx in enumerate(indices):
            sample = dataset[idx]
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
            v0 = sample["v0"]
            gt_wp = sample["gt_waypoints"].numpy()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred_action = model(pixel_values).float().cpu().squeeze()

            pred_kv = pred_action.reshape(K, 2)
            pred_wp = (
                forward_dynamics_batch(
                    pred_kv[:, 0].unsqueeze(0), pred_kv[:, 1].unsqueeze(0), v0.unsqueeze(0)
                )
                .squeeze(0)
                .numpy()
            )

            disp = np.linalg.norm(pred_wp - gt_wp, axis=1)
            ade = disp.mean()
            fde = disp[-1]

            # Input image
            img = Image.open(dataset.samples[idx]["image_path"]).convert("RGB")
            axes[row, 0].imshow(img)
            axes[row, 0].axis("off")
            axes[row, 0].set_title(f"Sample {idx}", fontsize=9)

            # BEV
            draw_bev_plot(
                axes[row, 1],
                gt_wp,
                pred_waypoints=pred_wp,
                ade=ade,
                fde=fde,
                title=f"Phase 4 — v0={v0.item():.1f} m/s",
            )

    fig.suptitle("Phase 4: Regression Trajectory", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def vis_stage1(args, device):
    """Stage 1: discrete action tokens."""
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    K = ckpt.get("K", args.K)
    n_bins = ckpt.get("n_bins", 256)
    tok_cfg = ckpt.get("tokenizer_config", {})

    tokenizer = DiscreteActionTokenizer(
        n_bins=tok_cfg.get("n_bins", n_bins),
        a_range=tok_cfg.get("a_range", (-6.0, 6.0)),
        kappa_range=tok_cfg.get("kappa_range", (-0.1, 0.1)),
        vocab_offset=tok_cfg.get("vocab_offset", 151936),
    )

    model = MiniPamayo(adapter_type="cross_attention", action_dim=K * 2)
    model.llm.resize_token_embeddings(tokenizer.vocab_offset + n_bins)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    dataset = NuScenesTrajectoryDataset(
        nuscenes_root=args.nuscenes_root, version=args.nuscenes_version, K=K
    )

    fig, axes = plt.subplots(args.n_vis, 2, figsize=(12, 4 * args.n_vis))
    if args.n_vis == 1:
        axes = axes.reshape(1, -1)

    indices = _select_indices(len(dataset), args.n_vis)

    with torch.no_grad():
        for row, idx in enumerate(indices):
            sample = dataset[idx]
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
            gt_action = sample["action"]
            v0 = sample["v0"]
            gt_wp = sample["gt_waypoints"].numpy()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                patch_features = model.vision_encoder(pixel_values)
                visual_tokens = model.adapter(patch_features)

            ar_tokens = greedy_generate_stage1(model, visual_tokens, K * 2, device)
            ar_action = tokenizer.decode_batch(ar_tokens.cpu()).squeeze(0)
            ar_kv = ar_action.reshape(K, 2)

            pred_wp = (
                forward_dynamics_batch(
                    ar_kv[:, 0].unsqueeze(0), ar_kv[:, 1].unsqueeze(0), v0.unsqueeze(0)
                )
                .squeeze(0)
                .numpy()
            )

            # Token accuracy
            gt_token_ids = tokenizer.encode_batch(gt_action.unsqueeze(0))
            match = (ar_tokens.cpu() == gt_token_ids).sum().item()
            total = gt_token_ids.numel()

            disp = np.linalg.norm(pred_wp - gt_wp, axis=1)
            ade = disp.mean()
            fde = disp[-1]

            img = Image.open(dataset.samples[idx]["image_path"]).convert("RGB")
            axes[row, 0].imshow(img)
            axes[row, 0].axis("off")
            axes[row, 0].set_title(f"Sample {idx}", fontsize=9)

            draw_bev_plot(
                axes[row, 1],
                gt_wp,
                pred_waypoints=pred_wp,
                ade=ade,
                fde=fde,
                title=f"Stage 1 — Token Acc: {match}/{total}",
            )

    fig.suptitle("Stage 1: Discrete Action Tokens", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def vis_stage2(args, device):
    """Stage 2: flow matching trajectory decoder."""
    # Decoder checkpoint
    decoder, K, ckpt = load_decoder_from_checkpoint(args.decoder_checkpoint, device)
    action_dim = ckpt.get("action_dim", K * 2)
    print(
        f"Expert config: hidden={ckpt.get('hidden_size')}, "
        f"layers={ckpt.get('num_hidden_layers')}, heads={ckpt.get('num_attention_heads')}"
    )

    # Frozen VLM
    vlm_ckpt_path = args.vlm_checkpoint or "checkpoints/phase4/best.pt"
    vlm = MiniPamayo(adapter_type="cross_attention", action_dim=action_dim)
    vlm_ckpt = torch.load(vlm_ckpt_path, map_location="cpu", weights_only=True)
    vlm.load_state_dict(vlm_ckpt["model_state_dict"], strict=False)
    vlm = vlm.to(device).eval()
    vlm.requires_grad_(False)

    dataset = NuScenesTrajectoryDataset(
        nuscenes_root=args.nuscenes_root, version=args.nuscenes_version, K=K
    )

    fig, axes = plt.subplots(args.n_vis, 2, figsize=(12, 4 * args.n_vis))
    if args.n_vis == 1:
        axes = axes.reshape(1, -1)

    indices = _select_indices(len(dataset), args.n_vis)

    with torch.no_grad():
        for row, idx in enumerate(indices):
            sample = dataset[idx]
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
            v0 = sample["v0"]
            gt_wp = sample["gt_waypoints"].numpy()

            kv_cache, prefill_seq_len = extract_kv_cache(vlm, pixel_values)

            # Multiple flow samples
            flow_wp_list = []
            ade_list = []
            for _ in range(args.n_samples):
                pred_action = cfm_sample(decoder, kv_cache, prefill_seq_len, n_steps=20)
                pred_kv = pred_action.cpu().squeeze().reshape(K, 2)
                wp = (
                    forward_dynamics_batch(
                        pred_kv[:, 0].unsqueeze(0), pred_kv[:, 1].unsqueeze(0), v0.unsqueeze(0)
                    )
                    .squeeze(0)
                    .numpy()
                )
                flow_wp_list.append(wp)
                ade_list.append(np.linalg.norm(wp - gt_wp, axis=1).mean())

            # Best sample (min ADE)
            best_idx = int(np.argmin(ade_list))
            best_wp = flow_wp_list[best_idx]
            disp = np.linalg.norm(best_wp - gt_wp, axis=1)

            img = Image.open(dataset.samples[idx]["image_path"]).convert("RGB")
            axes[row, 0].imshow(img)
            axes[row, 0].axis("off")
            axes[row, 0].set_title(f"Sample {idx}", fontsize=9)

            draw_bev_plot(
                axes[row, 1],
                gt_wp,
                pred_waypoints=best_wp,
                flow_samples=flow_wp_list,
                ade=disp.mean(),
                fde=disp[-1],
                title=f"Stage 2 — minADE={ade_list[best_idx]:.2f}m ({args.n_samples} samples)",
            )

    fig.suptitle("Stage 2: Flow Matching Decoder", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def vis_stage3(args, device):
    """Stage 3: CoC SFT with reasoning + action."""
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    K = args.K
    n_bins = 256

    action_tokenizer = DiscreteActionTokenizer(n_bins=n_bins)
    text_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    model = MiniPamayo(adapter_type="cross_attention", action_dim=K * 2)
    model.llm.resize_token_embeddings(action_tokenizer.vocab_offset + n_bins)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    # GT annotations
    gt_annotations = []
    with open(args.coc_data) as f:
        for line in f:
            gt_annotations.append(json.loads(line))

    dataset = CoCDataset(
        annotations_path=args.coc_data,
        tokenizer=text_tokenizer,
        action_tokenizer=action_tokenizer,
        K=K,
    )

    fig, axes = plt.subplots(args.n_vis, 3, figsize=(18, 4 * args.n_vis))
    if args.n_vis == 1:
        axes = axes.reshape(1, -1)

    indices = _select_indices(len(dataset), args.n_vis)

    with torch.no_grad():
        for row, idx in enumerate(indices):
            sample = dataset[idx]
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
            v0 = sample["v0"]
            gt_wp = sample["gt_waypoints"].numpy()
            gt_ann = gt_annotations[idx]
            obstacles = gt_ann.get("obstacles", [])

            # AR generation
            prompt_embeds = build_eval_prompt_embeds(
                model, pixel_values, v0.item(), text_tokenizer, device
            )
            ar_tokens = autoregressive_generate(model, prompt_embeds, 300)
            action, _decision, ar_text = parse_from_tokens(
                ar_tokens, text_tokenizer, action_tokenizer, K
            )

            action_t = torch.tensor(action, dtype=torch.float32)
            pred_kv = action_t.reshape(K, 2)
            pred_wp = (
                forward_dynamics_batch(
                    pred_kv[:, 0].unsqueeze(0), pred_kv[:, 1].unsqueeze(0), v0.unsqueeze(0)
                )
                .squeeze(0)
                .numpy()
            )

            disp = np.linalg.norm(pred_wp - gt_wp, axis=1)
            ade = disp.mean()
            fde = disp[-1]

            # Input image
            img = Image.open(gt_ann["image_path"]).convert("RGB")
            axes[row, 0].imshow(img)
            axes[row, 0].axis("off")
            axes[row, 0].set_title(f"Sample {idx}", fontsize=9)

            # BEV
            gt_decision = gt_ann["coc"]["driving_decision"]
            draw_bev_plot(
                axes[row, 1],
                gt_wp,
                pred_waypoints=pred_wp,
                obstacles=obstacles,
                ade=ade,
                fde=fde,
                title=f"Stage 3 — GT:{gt_decision['longitudinal']}/{gt_decision['lateral']}",
            )

            # CoC text
            gt_coc_text = format_coc_text(gt_ann["coc"])
            combined_text = (
                f"=== Generated ===\n{ar_text[:400]}\n\n=== GT CoC ===\n{gt_coc_text[:400]}"
            )
            draw_coc_text(axes[row, 2], combined_text, title="CoC Reasoning")

    fig.suptitle("Stage 3: CoC SFT", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def _load_decoder(decoder_checkpoint, device):
    """Load frozen Flow Matching Expert if checkpoint exists."""
    if decoder_checkpoint is None:
        return None
    path = Path(decoder_checkpoint)
    if not path.exists():
        print(f"WARNING: No decoder at {path}, using discrete tokens")
        return None
    decoder, _K, _ckpt = load_decoder_from_checkpoint(path, device)
    print(f"Loaded Flow Matching Expert: {path}")
    return decoder


@torch.no_grad()
def _flow_trajectory(model, decoder, prompt_embeds, token_ids, device, n_steps=10):
    """Extract continuous trajectory using Flow Matching Expert (KV-cache)."""
    embed_layer = model.llm.get_input_embeddings()
    token_ids_t = torch.tensor(token_ids, dtype=torch.long, device=device)
    token_embeds = embed_layer(token_ids_t.unsqueeze(0))
    input_embeds = torch.cat([prompt_embeds, token_embeds], dim=1)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.llm(inputs_embeds=input_embeds, use_cache=True)
    kv_cache = outputs.past_key_values
    prefill_seq_len = kv_cache.get_seq_length()
    pred_action = cfm_sample(decoder, kv_cache, prefill_seq_len, n_steps=n_steps)
    return pred_action.squeeze(0)


def vis_stage4(args, device):
    """Stage 4: GRPO RL vs SFT comparison.

    When --decoder_checkpoint is provided, trajectories are generated via
    Flow Matching (conditioned on full VLM hidden states) instead of
    discrete token parsing.
    """
    K = args.K
    n_bins = 256

    action_tokenizer = DiscreteActionTokenizer(n_bins=n_bins)
    text_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    new_vocab = action_tokenizer.vocab_offset + n_bins

    # RL model
    rl_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    rl_model = MiniPamayo(adapter_type="cross_attention", action_dim=K * 2)
    rl_model.llm.resize_token_embeddings(new_vocab)
    rl_model.load_state_dict(rl_ckpt["model_state_dict"])
    rl_model = rl_model.to(device).eval()

    # SFT model
    sft_model = None
    if args.ref_checkpoint:
        sft_ckpt = torch.load(args.ref_checkpoint, map_location="cpu", weights_only=True)
        sft_model = MiniPamayo(adapter_type="cross_attention", action_dim=K * 2)
        sft_model.llm.resize_token_embeddings(new_vocab)
        sft_model.load_state_dict(sft_ckpt["model_state_dict"])
        sft_model = sft_model.to(device).eval()

    # Flow Matching decoder (optional)
    decoder = _load_decoder(args.decoder_checkpoint, device)

    # GT annotations
    gt_annotations = []
    with open(args.coc_data) as f:
        for line in f:
            gt_annotations.append(json.loads(line))

    dataset = CoCDataset(
        annotations_path=args.coc_data,
        tokenizer=text_tokenizer,
        action_tokenizer=action_tokenizer,
        K=K,
    )

    fig, axes = plt.subplots(args.n_vis, 3, figsize=(18, 4 * args.n_vis))
    if args.n_vis == 1:
        axes = axes.reshape(1, -1)

    indices = _select_indices(len(dataset), args.n_vis)

    with torch.no_grad():
        for row, idx in enumerate(indices):
            sample = dataset[idx]
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
            v0 = sample["v0"]
            gt_wp = sample["gt_waypoints"].numpy()
            gt_ann = gt_annotations[idx]
            obstacles = gt_ann.get("obstacles", [])

            # RL inference
            rl_prompt = build_eval_prompt_embeds(
                rl_model, pixel_values, v0.item(), text_tokenizer, device
            )
            rl_tokens = autoregressive_generate(rl_model, rl_prompt, 300)
            _rl_action, _rl_decision, rl_text = parse_from_tokens(
                rl_tokens, text_tokenizer, action_tokenizer, K
            )
            if decoder is not None:
                rl_action_t = _flow_trajectory(rl_model, decoder, rl_prompt, rl_tokens, device)
            else:
                rl_action_t = torch.tensor(_rl_action, dtype=torch.float32)
            rl_kv = rl_action_t.reshape(K, 2)
            rl_wp = (
                forward_dynamics_batch(
                    rl_kv[:, 0].unsqueeze(0), rl_kv[:, 1].unsqueeze(0), v0.unsqueeze(0)
                )
                .squeeze(0)
                .numpy()
            )

            # SFT inference
            sft_wp = None
            sft_text = ""
            if sft_model is not None:
                sft_prompt = build_eval_prompt_embeds(
                    sft_model, pixel_values, v0.item(), text_tokenizer, device
                )
                sft_tokens = autoregressive_generate(sft_model, sft_prompt, 300)
                _sft_action, _sft_decision, sft_text = parse_from_tokens(
                    sft_tokens, text_tokenizer, action_tokenizer, K
                )
                if decoder is not None:
                    sft_action_t = _flow_trajectory(
                        sft_model, decoder, sft_prompt, sft_tokens, device
                    )
                else:
                    sft_action_t = torch.tensor(_sft_action, dtype=torch.float32)
                sft_kv = sft_action_t.reshape(K, 2)
                sft_wp = (
                    forward_dynamics_batch(
                        sft_kv[:, 0].unsqueeze(0), sft_kv[:, 1].unsqueeze(0), v0.unsqueeze(0)
                    )
                    .squeeze(0)
                    .numpy()
                )

            disp = np.linalg.norm(rl_wp - gt_wp, axis=1)
            ade = disp.mean()
            fde = disp[-1]

            # Input image
            img = Image.open(gt_ann["image_path"]).convert("RGB")
            axes[row, 0].imshow(img)
            axes[row, 0].axis("off")
            axes[row, 0].set_title(f"Sample {idx}", fontsize=9)

            # BEV (RL=Pred, SFT=green)
            gt_decision = gt_ann["coc"]["driving_decision"]
            traj_mode = "Flow" if decoder is not None else "Discrete"
            draw_bev_plot(
                axes[row, 1],
                gt_wp,
                pred_waypoints=rl_wp,
                sft_waypoints=sft_wp,
                obstacles=obstacles,
                ade=ade,
                fde=fde,
                title=f"Stage 4 ({traj_mode}) — GT:{gt_decision['longitudinal']}/{gt_decision['lateral']}",
            )

            # CoC text: RL vs SFT
            combined_text = f"=== RL ===\n{rl_text[:350]}\n\n=== SFT ===\n{sft_text[:350]}"
            draw_coc_text(axes[row, 2], combined_text, title="RL vs SFT")

    fig.suptitle("Stage 4: GRPO RL vs SFT", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def vis_stage3_rollouts(args, device):
    """Stage 3 multi-rollout: temperature-sampled candidates (pre-GRPO check).

    Generates multiple trajectory candidates using the same temperature
    sampling method as GRPO's generate_rollout(). Shows whether the SFT
    model produces diverse candidates before RL training.

    When --decoder_checkpoint is provided, trajectories are generated via
    Flow Matching (different rollout text → different KV-cache → different
    continuous trajectories). Without decoder, uses discrete token parsing.
    """
    K = args.K
    n_bins = 256
    n_rollouts = args.n_rollouts
    temperature = args.temperature

    action_tokenizer = DiscreteActionTokenizer(n_bins=n_bins)
    text_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    model = MiniPamayo(adapter_type="cross_attention", action_dim=K * 2)
    model.llm.resize_token_embeddings(action_tokenizer.vocab_offset + n_bins)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    # Flow Matching decoder (optional)
    decoder = _load_decoder(args.decoder_checkpoint, device)

    # GT annotations
    gt_annotations = []
    with open(args.coc_data) as f:
        for line in f:
            gt_annotations.append(json.loads(line))

    dataset = CoCDataset(
        annotations_path=args.coc_data,
        tokenizer=text_tokenizer,
        action_tokenizer=action_tokenizer,
        K=K,
    )

    fig, axes = plt.subplots(args.n_vis, 3, figsize=(18, 4 * args.n_vis))
    if args.n_vis == 1:
        axes = axes.reshape(1, -1)

    if getattr(args, "curves", False):
        curve_indices = []
        for i in range(len(dataset)):
            wp = dataset[i]["gt_waypoints"].numpy()
            if abs(float(wp[-1, 1])) > 3.0:
                curve_indices.append((i, abs(float(wp[-1, 1]))))
        curve_indices.sort(key=lambda x: -x[1])
        # Spread across different scenes (skip consecutive indices)
        selected = []
        last_idx = -100
        offset = getattr(args, "curve_offset", 0)
        skipped = 0
        for ci in curve_indices:
            if abs(ci[0] - last_idx) > 50:
                if skipped < offset:
                    skipped += 1
                    last_idx = ci[0]
                    continue
                selected.append(ci)
                last_idx = ci[0]
            if len(selected) >= args.n_vis:
                break
        indices = [ci[0] for ci in selected]
        print(f"Selected {len(indices)} curve scenes (|lateral| > 3m, diverse)")
        for ci in selected:
            print(f"  idx={ci[0]:5d} |lat|={ci[1]:.2f}m")
    else:
        indices = _select_indices(len(dataset), args.n_vis)

    traj_mode = "Flow" if decoder is not None else "Discrete"
    print(
        f"Generating {n_rollouts} rollouts per sample "
        f"(temperature={temperature}, trajectory={traj_mode})"
    )
    with torch.no_grad():
        for row, idx in enumerate(indices):
            sample = dataset[idx]
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
            v0 = sample["v0"]
            gt_wp = sample["gt_waypoints"].numpy()
            gt_ann = gt_annotations[idx]
            obstacles = gt_ann.get("obstacles", [])

            prompt_embeds = build_eval_prompt_embeds(
                model, pixel_values, v0.item(), text_tokenizer, device
            )

            # Generate N rollouts with temperature sampling
            candidate_wps = []
            candidate_texts = []
            ade_list = []

            # Also generate greedy (temperature=0) for comparison
            greedy_tokens = autoregressive_generate(model, prompt_embeds, 300)
            _greedy_action, _greedy_dec, greedy_text = parse_from_tokens(
                greedy_tokens, text_tokenizer, action_tokenizer, K
            )
            if decoder is not None:
                greedy_action_t = _flow_trajectory(
                    model, decoder, prompt_embeds, greedy_tokens, device
                )
            else:
                greedy_action_t = torch.tensor(_greedy_action, dtype=torch.float32)
            greedy_kv = greedy_action_t.reshape(K, 2)
            greedy_wp = (
                forward_dynamics_batch(
                    greedy_kv[:, 0].unsqueeze(0), greedy_kv[:, 1].unsqueeze(0), v0.unsqueeze(0)
                )
                .squeeze(0)
                .numpy()
            )

            for _r in range(n_rollouts):
                tokens = temperature_generate(model, prompt_embeds, 300, temperature)
                action, _decision, text = parse_from_tokens(
                    tokens, text_tokenizer, action_tokenizer, K
                )
                if decoder is not None:
                    action_t = _flow_trajectory(model, decoder, prompt_embeds, tokens, device)
                else:
                    action_t = torch.tensor(action, dtype=torch.float32)
                kv = action_t.reshape(K, 2)
                wp = (
                    forward_dynamics_batch(
                        kv[:, 0].unsqueeze(0), kv[:, 1].unsqueeze(0), v0.unsqueeze(0)
                    )
                    .squeeze(0)
                    .numpy()
                )
                candidate_wps.append(wp)
                candidate_texts.append(text)
                ade_list.append(np.linalg.norm(wp - gt_wp, axis=1).mean())

            # Best candidate (min ADE)
            best_idx = int(np.argmin(ade_list))
            best_ade = ade_list[best_idx]
            greedy_ade = np.linalg.norm(greedy_wp - gt_wp, axis=1).mean()

            print(
                f"  Sample {idx}: greedy ADE={greedy_ade:.2f}m, "
                f"best-of-{n_rollouts} ADE={best_ade:.2f}m "
                f"(all: {[f'{a:.2f}' for a in ade_list]})"
            )

            # Input image
            img = Image.open(gt_ann["image_path"]).convert("RGB")
            axes[row, 0].imshow(img)
            axes[row, 0].axis("off")
            axes[row, 0].set_title(f"Sample {idx}", fontsize=9)

            # BEV: GT (blue), greedy (red), all candidates (salmon)
            gt_decision = gt_ann["coc"]["driving_decision"]
            draw_bev_plot(
                axes[row, 1],
                gt_wp,
                pred_waypoints=greedy_wp,
                flow_samples=candidate_wps,
                obstacles=obstacles,
                ade=greedy_ade,
                fde=np.linalg.norm(greedy_wp[-1] - gt_wp[-1]),
                title=(
                    f"Rollouts ({traj_mode}, N={n_rollouts}, T={temperature}) — "
                    f"GT:{gt_decision['longitudinal']}/{gt_decision['lateral']}"
                ),
            )

            # CoC text: greedy + best candidate
            combined_text = (
                f"=== Greedy (ADE={greedy_ade:.2f}m) ===\n{greedy_text[:300]}\n\n"
                f"=== Best Candidate #{best_idx} (ADE={best_ade:.2f}m) ===\n"
                f"{candidate_texts[best_idx][:300]}"
            )
            draw_coc_text(axes[row, 2], combined_text, title="Greedy vs Best Candidate")

    fig.suptitle(
        f"Multi-Rollout Candidates ({traj_mode}, N={n_rollouts}, temp={temperature})",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


# ============================================================
# Utility
# ============================================================


def vis_coc_labels(args):
    """Visualize CoC annotation labels: image + decision + components + trace."""
    gt_annotations = []
    with open(args.coc_data) as f:
        for line in f:
            gt_annotations.append(json.loads(line))

    n_vis = min(args.n_vis, len(gt_annotations))
    indices = _select_indices(len(gt_annotations), n_vis)

    fig, axes = plt.subplots(
        n_vis, 2, figsize=(16, 4 * n_vis), gridspec_kw={"width_ratios": [1, 1.5]}
    )
    if n_vis == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(indices):
        ann = gt_annotations[idx]
        coc = ann["coc"]
        decision = coc["driving_decision"]

        # Image
        img = Image.open(ann["image_path"]).convert("RGB")
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"[{idx}] {ann['image_path'].split('/')[-1]}", fontsize=7)
        axes[row, 0].axis("off")

        # CoC text
        lines = []
        lines.append(f"[Decision]  long: {decision['longitudinal']}  |  lat: {decision['lateral']}")
        lines.append("")
        lines.append("[Critical Components]")
        for comp in coc["critical_components"]:
            lines.append(f"  - {comp['type']}: {comp['description']}")
        lines.append("")
        lines.append("[CoC Trace]")
        lines.append(coc["coc_trace"])
        text = "\n".join(lines)

        wrapped = "\n".join(_wrap_text(text, width=70))
        axes[row, 1].axis("off")
        axes[row, 1].text(
            0.02,
            0.98,
            wrapped,
            transform=axes[row, 1].transAxes,
            fontsize=8,
            verticalalignment="top",
            fontfamily="monospace",
            wrap=True,
        )

    fig.suptitle(f"CoC Labels ({len(gt_annotations)} samples)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def _select_indices(total, n_vis):
    """Select evenly-spaced indices from dataset."""
    if total <= n_vis:
        return list(range(total))
    step = total / n_vis
    return [int(i * step) for i in range(n_vis)]


# ============================================================
# Main
# ============================================================


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Visualizing: {args.stage}")

    if args.stage == "phase4":
        fig = vis_phase4(args, device)
    elif args.stage == "stage1":
        fig = vis_stage1(args, device)
    elif args.stage == "stage2":
        fig = vis_stage2(args, device)
    elif args.stage == "stage3":
        fig = vis_stage3(args, device)
    elif args.stage == "stage3_rollouts":
        fig = vis_stage3_rollouts(args, device)
    elif args.stage == "stage4":
        fig = vis_stage4(args, device)
    elif args.stage == "coc_labels":
        fig = vis_coc_labels(args)

    # Save
    output_path = args.output or f"outputs/vis_{args.stage}.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
