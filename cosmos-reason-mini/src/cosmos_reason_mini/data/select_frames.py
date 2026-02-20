"""nuScenes からフレームを選定し、メタデータ付き JSON を出力する。"""

import argparse
import json
import os
import random

from nuscenes.nuscenes import NuScenes


def select_frames(nusc, max_per_scene=5, seed=42):
    random.seed(seed)
    frames = []
    for scene in nusc.scene:
        # シーン内の全キーフレームを収集
        sample_token = scene["first_sample_token"]
        scene_frames = []
        while sample_token:
            sample = nusc.get("sample", sample_token)
            cam_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
            ego_pose = nusc.get("ego_pose", cam_data["ego_pose_token"])
            scene_frames.append(
                {
                    "sample_token": sample_token,
                    "image_path": cam_data["filename"],  # samples/CAM_FRONT/xxx.jpg
                    "scene_name": scene["name"],
                    "scene_description": scene["description"],
                    "ego_translation": ego_pose["translation"],
                    "ego_rotation": ego_pose["rotation"],
                    "timestamp": sample["timestamp"],
                }
            )
            sample_token = sample["next"] if sample["next"] else None

        # 等間隔でサンプリング(多様性確保)
        if len(scene_frames) > max_per_scene:
            indices = [int(i * len(scene_frames) / max_per_scene) for i in range(max_per_scene)]
            scene_frames = [scene_frames[i] for i in indices]
        frames.extend(scene_frames)
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="data/nuscenes")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--max_per_scene", type=int, default=5)
    parser.add_argument("--output", default="data/sft/frames.json")
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    frames = select_frames(nusc, max_per_scene=args.max_per_scene)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(frames, f, indent=2)
    print(f"Selected {len(frames)} frames from {len(nusc.scene)} scenes")


if __name__ == "__main__":
    main()
