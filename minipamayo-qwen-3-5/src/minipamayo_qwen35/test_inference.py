# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# End-to-end example script for the inference pipeline:
# This script loads a JSONL sample, runs inference, and computes the minADE.
# It can be used to test the inference pipeline.

import argparse

import numpy as np
import torch

parser = argparse.ArgumentParser(
    description="Run Alpamayo-style end-to-end inference for one JSONL sample."
)
parser.add_argument("--stage2-checkpoint", type=str, required=True)
parser.add_argument("--stage1b-checkpoint", type=str, required=True)
parser.add_argument("--sample-jsonl", type=str, required=True)
parser.add_argument("--sample-index", type=int, default=0)
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

from . import helper
from .load_reasoning_jsonl import load_reasoning_jsonl_sample
from .models.checkpoint_loader import load_stage2_inference_bundle

if args.sample_index < 0:
    raise RuntimeError("`sample_index` must be >= 0.")

device = torch.device(args.device)
if device.type != "cuda":
    raise RuntimeError("`test_inference.py` currently expects CUDA.")

print(f"Loading dataset for sample_index: {args.sample_index}...")
data = load_reasoning_jsonl_sample(args.sample_jsonl, sample_index=args.sample_index)
print("Dataset loaded.")
messages = helper.create_message(data["image_frames"].flatten(0, 1))

bundle = load_stage2_inference_bundle(
    stage2_checkpoint_path=args.stage2_checkpoint,
    stage1b_checkpoint_path=args.stage1b_checkpoint,
    image_min_pixels=helper.MIN_PIXELS,
    image_max_pixels=helper.MAX_PIXELS,
    flow_steps=10,
    device=device,
)
model = bundle["wrapper"]
processor = helper.get_processor(model.tokenizer)

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    continue_final_message=True,
    return_dict=True,
    return_tensors="pt",
)
model_inputs = {
    "tokenized_data": inputs,
    "ego_history_xyz": data["ego_history_xyz"],
    "ego_history_rot": data["ego_history_rot"],
}

model_inputs = helper.to_device(model_inputs, device)

torch.cuda.manual_seed_all(42)
with torch.autocast("cuda", dtype=torch.bfloat16):
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=model_inputs,
        top_p=0.98,
        temperature=0.6,
        num_traj_samples=1,
        max_generation_length=256,
        return_extra=True,
    )

print("Chain-of-Causation (per trajectory):\n", extra["cot"][0])

gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
min_ade = diff.min()
print("minADE:", min_ade, "meters")
print(
    "Note: VLA-reasoning models produce nondeterministic outputs due to trajectory sampling, "
    "hardware differences, etc. With num_traj_samples=1 (set for GPU memory compatibility), "
    "variance in minADE is expected. For visual sanity checks, see notebooks/inference.ipynb"
)
