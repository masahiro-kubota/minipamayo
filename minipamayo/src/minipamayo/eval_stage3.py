"""Stage 3 evaluation: CoC SFT — reasoning + action quality.

Evaluates:
  1. Token accuracy (teacher-forced + autoregressive)
  2. Driving Decision accuracy (longitudinal + lateral)
  3. Action-space metrics (MAE, ADE, FDE)
  4. Qualitative reasoning samples

Usage:
    cd minipamayo && uv run python -m minipamayo.eval_stage3 \
        --checkpoint checkpoints/stage3/best.pt \
        --coc_data data/coc_annotations.jsonl
"""

import argparse
import json

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .data.coc_dataset import CoCDataset
from .models.discrete_head import DiscreteActionTokenizer
from .models.dynamics import forward_dynamics_batch
from .models.minipamayo import MiniPamayo


def parse_args():
    parser = argparse.ArgumentParser(description="MiniPamayo Stage 3 evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--coc_data", type=str, default="data/coc_annotations.jsonl")
    parser.add_argument("--nuscenes_root", type=str, default="../cosmos-reason-mini/data/nuscenes")
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--n_bins", type=int, default=256)
    parser.add_argument("--max_text_len", type=int, default=256)
    parser.add_argument("--show_samples", type=int, default=5)
    return parser.parse_args()


def autoregressive_generate(model, prompt_embeds, max_tokens, device):
    """Generate reasoning + action tokens autoregressively."""
    input_embeds = prompt_embeds.clone()
    generated = []

    for _ in range(max_tokens):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.llm(inputs_embeds=input_embeds)
        logits = outputs.logits[:, -1, :]
        token_id = logits.argmax(dim=-1)  # (1,)
        generated.append(token_id.item())

        # Check for EOS or <|im_end|>
        if token_id.item() in (151643, 151645):
            break

        next_embed = model.llm.get_input_embeddings()(token_id.unsqueeze(0))
        input_embeds = torch.cat([input_embeds, next_embed.to(input_embeds.dtype)], dim=1)

    return generated


def parse_decision_from_text(text: str) -> dict | None:
    """Parse driving decision from generated text."""
    decision = {}
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("longitudinal:"):
            decision["longitudinal"] = line.split(":", 1)[1].strip()
        elif line.startswith("lateral:"):
            decision["lateral"] = line.split(":", 1)[1].strip()

    if "longitudinal" in decision and "lateral" in decision:
        return decision
    return None


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizers
    action_tokenizer = DiscreteActionTokenizer(n_bins=args.n_bins)
    text_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    # Dataset
    print("Loading CoC dataset...")
    dataset = CoCDataset(
        annotations_path=args.coc_data,
        tokenizer=text_tokenizer,
        action_tokenizer=action_tokenizer,
        K=args.K,
        max_text_len=args.max_text_len,
    )
    print(f"Total: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Model
    print("Building model...")
    model = MiniPamayo(adapter_type="cross_attention", action_dim=args.K * 2)
    new_vocab = action_tokenizer.vocab_offset + args.n_bins
    model.llm.resize_token_embeddings(new_vocab)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded: {args.checkpoint}")

    # Load GT annotations for decision comparison
    gt_annotations = []
    with open(args.coc_data) as f:
        for line in f:
            gt_annotations.append(json.loads(line))

    # Evaluate
    print(f"\n{'=' * 70}")
    print("Evaluation")
    print(f"{'=' * 70}")

    # Teacher-forced metrics
    total_tf_loss = 0.0
    total_tf_correct = 0
    total_tf_tokens = 0
    all_ar_actions = []
    all_gt_actions = []
    all_v0 = []
    all_gt_waypoints = []
    decision_correct = {"longitudinal": 0, "lateral": 0}
    decision_total = 0
    sample_outputs = []
    vocab_offset = action_tokenizer.vocab_offset

    with torch.no_grad():
        for i, batch in enumerate(loader):
            pixel_values = batch["pixel_values"].to(device)
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
                visual_embeds = model.adapter(patch_features)

            embed_layer = model.llm.get_input_embeddings()

            # Text embeddings for each segment
            system_embeds = embed_layer(system_ids.unsqueeze(0))
            user_prefix_embeds = embed_layer(user_prefix_ids.unsqueeze(0))
            ego_question_embeds = embed_layer(ego_question_ids.unsqueeze(0))
            asst_prefix_embeds = embed_layer(asst_prefix_ids.unsqueeze(0))
            reasoning_embeds = embed_layer(reasoning_ids.unsqueeze(0))
            action_embeds = embed_layer(action_ids.unsqueeze(0))
            eos_embeds = embed_layer(eos_ids.unsqueeze(0))

            target_dtype = system_embeds.dtype

            # Teacher-forced: full sequence
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

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model.llm(inputs_embeds=inputs_embeds, labels=labels)
            total_tf_loss += outputs.loss.float().item()

            # Token accuracy
            mask = labels[0, 1:] != -100
            if mask.any():
                logits = outputs.logits[0, :-1]
                preds = logits[mask].argmax(dim=-1)
                targets = labels[0, 1:][mask]
                total_tf_correct += (preds == targets).sum().item()
                total_tf_tokens += mask.sum().item()

            # Autoregressive generation (prompt = up to asst_prefix)
            prompt_embeds_ar = torch.cat(
                [
                    system_embeds,
                    user_prefix_embeds,
                    visual_embeds.to(target_dtype),
                    ego_question_embeds,
                    asst_prefix_embeds,
                ],
                dim=1,
            )
            max_gen = n_reasoning + n_action + 20
            ar_tokens = autoregressive_generate(model, prompt_embeds_ar, max_gen, device)

            # Extract action tokens from AR output
            ar_action_ids = [t for t in ar_tokens if vocab_offset <= t < vocab_offset + args.n_bins]
            if len(ar_action_ids) >= args.K * 2:
                ar_action_ids = ar_action_ids[: args.K * 2]
                ar_action = action_tokenizer.decode(ar_action_ids)
                all_ar_actions.append(torch.tensor(ar_action, dtype=torch.float32))
            else:
                # Pad with zeros if not enough action tokens generated
                ar_action = action_tokenizer.decode(
                    ar_action_ids + [vocab_offset] * (args.K * 2 - len(ar_action_ids))
                )
                all_ar_actions.append(torch.tensor(ar_action, dtype=torch.float32))

            all_gt_actions.append(torch.tensor(gt_annotations[i]["action"], dtype=torch.float32))
            all_v0.append(torch.tensor(gt_annotations[i]["v0"], dtype=torch.float32))
            all_gt_waypoints.append(
                torch.tensor(gt_annotations[i]["gt_waypoints"], dtype=torch.float32)
            )

            # Parse decision from AR text
            ar_text_tokens = [t for t in ar_tokens if t < vocab_offset]
            ar_text = text_tokenizer.decode(ar_text_tokens, skip_special_tokens=True)

            ar_decision = parse_decision_from_text(ar_text)
            gt_decision = gt_annotations[i]["coc"]["driving_decision"]
            decision_total += 1

            if ar_decision:
                if ar_decision.get("longitudinal") == gt_decision["longitudinal"]:
                    decision_correct["longitudinal"] += 1
                if ar_decision.get("lateral") == gt_decision["lateral"]:
                    decision_correct["lateral"] += 1

            # Save samples for display
            if i < args.show_samples:
                sample_outputs.append(
                    {
                        "index": i,
                        "ar_text": ar_text[:300],
                        "ar_decision": ar_decision,
                        "gt_decision": gt_decision,
                        "n_action_tokens": len(ar_action_ids),
                    }
                )

    # Print results
    tf_acc = total_tf_correct / max(total_tf_tokens, 1)
    print(f"\n{'=' * 70}")
    print("Teacher-Forced Metrics")
    print(f"{'=' * 70}")
    print(f"  CE Loss:       {total_tf_loss / len(loader):.4f}")
    print(f"  Token Accuracy: {tf_acc:.4f}")

    print(f"\n{'=' * 70}")
    print("Driving Decision Accuracy")
    print(f"{'=' * 70}")
    long_acc = decision_correct["longitudinal"] / max(decision_total, 1)
    lat_acc = decision_correct["lateral"] / max(decision_total, 1)
    overall = (decision_correct["longitudinal"] + decision_correct["lateral"]) / max(
        2 * decision_total, 1
    )
    print(f"  Longitudinal:  {long_acc:.4f} ({decision_correct['longitudinal']}/{decision_total})")
    print(f"  Lateral:       {lat_acc:.4f} ({decision_correct['lateral']}/{decision_total})")
    print(f"  Overall:       {overall:.4f}")

    # Trajectory metrics
    ar_actions = torch.stack(all_ar_actions)
    gt_actions = torch.stack(all_gt_actions)
    v0s = torch.stack(all_v0)
    gt_wp = torch.stack(all_gt_waypoints)

    ar_kv = ar_actions.reshape(-1, args.K, 2)
    gt_kv = gt_actions.reshape(-1, args.K, 2)

    a_mae = (ar_kv[:, :, 0] - gt_kv[:, :, 0]).abs().mean().item()
    kappa_mae = (ar_kv[:, :, 1] - gt_kv[:, :, 1]).abs().mean().item()

    pred_wp = forward_dynamics_batch(ar_kv[:, :, 0], ar_kv[:, :, 1], v0s, dt=0.5)
    disp_errors = torch.norm(pred_wp - gt_wp, dim=2)
    ade = disp_errors.mean().item()
    fde = disp_errors[:, -1].mean().item()

    print(f"\n{'=' * 70}")
    print("Action-Space Metrics (AR)")
    print(f"{'=' * 70}")
    print(f"  a MAE:     {a_mae:.6f}")
    print(f"  kappa MAE: {kappa_mae:.6f}")

    print(f"\n{'=' * 70}")
    print("Trajectory Metrics")
    print(f"{'=' * 70}")
    print(f"  ADE: {ade:.4f} m")
    print(f"  FDE: {fde:.4f} m")

    # Qualitative samples
    print(f"\n{'=' * 70}")
    print(f"Sample Outputs (first {args.show_samples})")
    print(f"{'=' * 70}")
    for s in sample_outputs:
        print(f"\n  --- Sample {s['index']} ---")
        print(f"  GT Decision: {s['gt_decision']}")
        print(f"  AR Decision: {s['ar_decision']}")
        print(f"  Action tokens generated: {s['n_action_tokens']}")
        print(f"  AR Text (truncated): {s['ar_text'][:200]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
