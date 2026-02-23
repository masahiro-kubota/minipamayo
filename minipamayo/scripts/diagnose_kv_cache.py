"""KV-cache conditioning の有効性を検証する。

正しい KV-cache vs シャッフルした KV-cache で CFM loss と予測軌道を比較。
差がなければ Expert は VLM の情報を使えていない。

Usage:
    cd minipamayo && uv run python scripts/diagnose_kv_cache.py
"""

import numpy as np
import torch

from minipamayo.data.nuscenes_trajectory_dataset import NuScenesTrajectoryDataset
from minipamayo.models.minipamayo import MiniPamayo
from minipamayo.models.trajectory_decoder import (
    cfm_loss,
    cfm_sample,
    clone_kv_cache,
    load_decoder_from_checkpoint,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K = 6

    # --- Load models ---
    print("Loading decoder...")
    decoder, _, ckpt = load_decoder_from_checkpoint("checkpoints/stage2/best.pt", device)
    print(
        f"  hidden={ckpt['hidden_size']}, layers={ckpt['num_hidden_layers']}, heads={ckpt['num_attention_heads']}"
    )

    print("Loading VLM...")
    vlm = MiniPamayo(adapter_type="cross_attention", action_dim=K * 2)
    vlm_ckpt = torch.load("checkpoints/phase4/best.pt", map_location="cpu", weights_only=True)
    vlm.load_state_dict(vlm_ckpt["model_state_dict"], strict=False)
    vlm = vlm.to(device).eval()
    vlm.requires_grad_(False)

    print("Loading dataset...")
    dataset = NuScenesTrajectoryDataset(
        nuscenes_root="/mnt/ssd/nuscenes", version="v1.0-trainval", K=K
    )
    print(f"  Total: {len(dataset)}")

    # --- Test 1: CFM loss with correct vs shuffled KV-cache ---
    print("\n" + "=" * 70)
    print("Test 1: CFM loss — correct KV vs shuffled KV vs zero KV")
    print("=" * 70)

    N = 50
    losses_correct = []
    losses_shuffled = []
    losses_zero = []

    indices = list(range(0, len(dataset), len(dataset) // N))[:N]

    with torch.no_grad():
        # Pre-compute all KV-caches
        kv_caches = []
        prefill_lens = []
        for idx in indices:
            sample = dataset[idx]
            pv = sample["pixel_values"].unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pf = vlm.vision_encoder(pv)
                vt = vlm.adapter(pf)
                out = vlm.llm(inputs_embeds=vt, use_cache=True)
            kv_caches.append(out.past_key_values)
            prefill_lens.append(out.past_key_values.get_seq_length())

        for i, idx in enumerate(indices):
            sample = dataset[idx]
            gt_action = sample["action"].unsqueeze(0).to(device)

            # Correct KV-cache
            loss_c = cfm_loss(decoder, gt_action, kv_caches[i], prefill_lens[i]).item()
            losses_correct.append(loss_c)

            # Shuffled KV-cache (use next sample's KV)
            j = (i + 1) % N
            loss_s = cfm_loss(decoder, gt_action, kv_caches[j], prefill_lens[j]).item()
            losses_shuffled.append(loss_s)

            # Zero KV-cache (create a fake zero cache)
            zero_cache = clone_kv_cache(kv_caches[i])
            for layer_idx in range(len(zero_cache.layers)):
                zero_cache.layers[layer_idx].keys.zero_()
                zero_cache.layers[layer_idx].values.zero_()
            loss_z = cfm_loss(decoder, gt_action, zero_cache, prefill_lens[i]).item()
            losses_zero.append(loss_z)

            if i < 5:
                print(f"  [{i}] correct={loss_c:.4f}  shuffled={loss_s:.4f}  zero={loss_z:.4f}")

    mean_c = np.mean(losses_correct)
    mean_s = np.mean(losses_shuffled)
    mean_z = np.mean(losses_zero)
    print(f"\n  Mean CFM loss (N={N}):")
    print(f"    Correct KV:  {mean_c:.4f}")
    print(f"    Shuffled KV: {mean_s:.4f}  (diff: {(mean_s - mean_c) / mean_c * 100:+.1f}%)")
    print(f"    Zero KV:     {mean_z:.4f}  (diff: {(mean_z - mean_c) / mean_c * 100:+.1f}%)")

    if abs(mean_s - mean_c) / mean_c < 0.05:
        print("\n  ⚠ WARNING: Shuffled KV ≈ Correct KV → Expert is NOT using VLM info!")
    else:
        print("\n  ✓ OK: Expert is using VLM conditioning.")

    # --- Test 2: Trajectory prediction with correct vs shuffled KV ---
    print("\n" + "=" * 70)
    print("Test 2: Trajectory predictions — correct vs shuffled KV")
    print("=" * 70)

    N2 = 20
    pred_diffs = []

    with torch.no_grad():
        for i in range(N2):
            idx = indices[i]
            sample = dataset[idx]

            # Predict with correct KV
            pred_c = cfm_sample(decoder, kv_caches[i], prefill_lens[i], n_steps=20).cpu().squeeze()

            # Predict with shuffled KV
            j = (i + 1) % N
            pred_s = cfm_sample(decoder, kv_caches[j], prefill_lens[j], n_steps=20).cpu().squeeze()

            # L2 distance between predictions
            diff = (pred_c - pred_s).pow(2).sum().sqrt().item()
            pred_diffs.append(diff)

            if i < 5:
                pred_c_kv = pred_c.reshape(K, 2)
                pred_s_kv = pred_s.reshape(K, 2)
                print(f"  [{i}] pred_diff={diff:.4f}")
                print(f"    correct kappa: {pred_c_kv[:, 1].tolist()}")
                print(f"    shuffled kappa: {pred_s_kv[:, 1].tolist()}")

    mean_diff = np.mean(pred_diffs)
    print(f"\n  Mean prediction L2 diff (correct vs shuffled): {mean_diff:.4f}")

    if mean_diff < 0.01:
        print("  ⚠ WARNING: Predictions nearly identical → KV-cache has NO effect!")
    elif mean_diff < 0.1:
        print("  ⚠ WEAK: Predictions slightly different → KV-cache has MINIMAL effect.")
    else:
        print(f"  ✓ OK: Predictions differ meaningfully (L2={mean_diff:.4f}).")

    # --- Test 3: KV-cache statistics ---
    print("\n" + "=" * 70)
    print("Test 3: KV-cache statistics")
    print("=" * 70)

    kv = kv_caches[0]
    n_layers = len(kv.layers)
    print(f"  Num layers: {n_layers}")
    print(f"  Key shape (layer 0): {kv.layers[0].keys.shape}")
    print(f"  Value shape (layer 0): {kv.layers[0].values.shape}")
    print(f"  Seq length: {kv.get_seq_length()}")

    # Check if KV values are meaningful
    for layer_idx in [0, n_layers // 2, n_layers - 1]:
        k = kv.layers[layer_idx].keys
        v = kv.layers[layer_idx].values
        print(
            f"  Layer {layer_idx}: K norm={k.norm():.2f}, V norm={v.norm():.2f}, "
            f"K std={k.std():.4f}, V std={v.std():.4f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
