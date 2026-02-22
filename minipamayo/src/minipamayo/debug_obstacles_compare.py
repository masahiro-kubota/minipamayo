"""Compare obstacles: jsonl (Stage 3) vs fresh nuScenes computation.

Also tests the rotation sign of draw_bev_plot.

Usage:
    cd minipamayo && uv run python -m minipamayo.debug_obstacles_compare
"""

import json

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion

matplotlib.use("Agg")

OBSTACLE_PREFIXES = ("vehicle.", "human.pedestrian.", "movable_object.")
OBSTACLE_MAX_RANGE = 50.0


def get_obstacles_fresh(nusc, sample_token, ego_position, ego_heading):
    """Recompute obstacles from nuScenes (same logic as dataset)."""
    sample = nusc.get("sample", sample_token)
    obstacles = []
    cos_h = np.cos(ego_heading)
    sin_h = np.sin(ego_heading)

    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        if not ann["category_name"].startswith(OBSTACLE_PREFIXES):
            continue
        global_center = np.array(ann["translation"][:2])
        dist = np.linalg.norm(global_center - ego_position)
        if dist > OBSTACLE_MAX_RANGE:
            continue
        centered = global_center - ego_position
        ego_x = float(cos_h * centered[0] + sin_h * centered[1])
        ego_y = float(-sin_h * centered[0] + cos_h * centered[1])
        width = float(ann["size"][0])
        length = float(ann["size"][1])
        global_obs_heading = Quaternion(ann["rotation"]).yaw_pitch_roll[0]
        relative_heading = float(global_obs_heading - ego_heading)
        obstacles.append(
            {
                "center": [ego_x, ego_y],
                "size": [width, length],
                "heading": relative_heading,
                "category": ann["category_name"],
            }
        )
    return obstacles


def draw_bev_with_rotation_test(ax, obstacles, gt_waypoints=None, title="", rotation_sign=1):
    """Draw BEV obstacles with configurable rotation sign."""
    for obs in obstacles:
        cx, cy = obs["center"]
        w, length = obs["size"]
        heading = obs["heading"]
        cat = obs.get("category", "vehicle.car")

        if cat.startswith("vehicle."):
            color = "tab:orange"
        elif cat.startswith("human."):
            color = "tab:red"
        else:
            color = "tab:gray"

        # BEV transform: display_x = -cy, display_y = cx
        rect = mpatches.FancyBboxPatch(
            (-cy - w / 2, cx - length / 2),
            w,
            length,
            boxstyle="round,pad=0",
            facecolor=color,
            alpha=0.4,
            edgecolor=color,
            linewidth=1.0,
            zorder=3,
        )
        rot_angle = rotation_sign * heading
        t = matplotlib.transforms.Affine2D().rotate_around(-cy, cx, rot_angle) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)

        # Heading direction arrow (shows where obstacle is facing)
        arrow_len = max(length, 2.0)
        # Obstacle's forward in ego frame: (cos(heading), sin(heading))
        # In BEV display: (-sin(heading), cos(heading))
        dx_disp = -np.sin(heading) * arrow_len * 0.5
        dy_disp = np.cos(heading) * arrow_len * 0.5
        ax.annotate(
            "",
            xy=(-cy + dx_disp, cx + dy_disp),
            xytext=(-cy, cx),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
            zorder=5,
        )

    if gt_waypoints is not None:
        gt = np.array(gt_waypoints)
        ax.plot(-gt[:, 1], gt[:, 0], "b-o", markersize=3, linewidth=1.5, label="GT", zorder=5)

    # Ego marker
    ax.plot(0, 0, "gs", markersize=8, zorder=10)
    ax.annotate(
        "",
        xy=(0, 2),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="green", lw=2),
        zorder=10,
    )

    ax.set_xlim(-40, 40)
    ax.set_ylim(-20, 50)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("← Right    Left →", fontsize=7)
    ax.set_ylabel("← Back    Front →", fontsize=7)


def main():
    import argparse
    import os

    from PIL import Image as PILImage

    parser = argparse.ArgumentParser()
    parser.add_argument("--nuscenes_root", type=str, default="/mnt/ssd/nuscenes")
    parser.add_argument("--nuscenes_version", type=str, default="v1.0-trainval")
    parser.add_argument("--coc_data", type=str, default="data/coc_annotations_trainval.jsonl")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Load jsonl
    with open(args.coc_data) as f:
        annotations = [json.loads(line) for line in f]

    ann = annotations[args.sample_idx]
    jsonl_obstacles = ann.get("obstacles", [])
    gt_wp = np.array(ann["gt_waypoints"])
    image_path = ann["image_path"]

    print(f"Sample {args.sample_idx}: {image_path}")
    print(f"  jsonl obstacles: {len(jsonl_obstacles)}")

    # Find matching nuScenes sample to recompute obstacles
    from nuscenes.nuscenes import NuScenes

    print("Loading nuScenes...")
    nusc = NuScenes(version=args.nuscenes_version, dataroot=args.nuscenes_root, verbose=False)

    # Find sample by matching image path
    fresh_obstacles = None
    for sample in nusc.sample:
        cam_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        full_path = f"{args.nuscenes_root}/{cam_data['filename']}"
        if full_path == image_path or cam_data["filename"] in image_path:
            ego_pose = nusc.get("ego_pose", cam_data["ego_pose_token"])
            ego_pos = np.array(ego_pose["translation"][:2])
            ego_heading = Quaternion(ego_pose["rotation"]).yaw_pitch_roll[0]
            fresh_obstacles = get_obstacles_fresh(nusc, sample["token"], ego_pos, ego_heading)
            print(f"  fresh obstacles: {len(fresh_obstacles)}")
            break

    if fresh_obstacles is None:
        print("WARNING: Could not find matching nuScenes sample!")
        fresh_obstacles = []

    # Compare obstacle centers
    if jsonl_obstacles and fresh_obstacles:
        print("\n  === Center comparison (jsonl vs fresh) ===")
        for i, (jo, fo) in enumerate(
            zip(
                sorted(jsonl_obstacles, key=lambda o: o["center"][0]),
                sorted(fresh_obstacles, key=lambda o: o["center"][0]),
                strict=False,
            )
        ):
            jc = jo["center"]
            fc = fo["center"]
            diff = np.linalg.norm(np.array(jc) - np.array(fc))
            jh = jo["heading"]
            fh = fo["heading"]
            print(
                f"  [{i}] jsonl=({jc[0]:+.1f},{jc[1]:+.1f}) h={jh:+.2f}  "
                f"fresh=({fc[0]:+.1f},{fc[1]:+.1f}) h={fh:+.2f}  diff={diff:.3f}m"
            )
            if i >= 9:
                print(f"  ... ({len(jsonl_obstacles)} total)")
                break

    # Plot: 4 panels
    # [image] [jsonl rotate=-h] [jsonl rotate=+h] [fresh rotate=+h]
    fig, axes = plt.subplots(1, 4, figsize=(28, 7))

    # Panel 0: Input image
    img = PILImage.open(image_path).convert("RGB")
    axes[0].imshow(img)
    axes[0].axis("off")
    axes[0].set_title(f"CAM_FRONT (sample {args.sample_idx})", fontsize=9)

    # Panel 1: jsonl obstacles with CURRENT code (rotate by -heading)
    draw_bev_with_rotation_test(
        axes[1],
        jsonl_obstacles,
        gt_wp,
        title="jsonl: rotate_around(-heading)\n[CURRENT CODE]",
        rotation_sign=-1,
    )

    # Panel 2: jsonl obstacles with FIXED rotation (rotate by +heading)
    draw_bev_with_rotation_test(
        axes[2],
        jsonl_obstacles,
        gt_wp,
        title="jsonl: rotate_around(+heading)\n[PROPOSED FIX]",
        rotation_sign=+1,
    )

    # Panel 3: fresh obstacles with +heading
    draw_bev_with_rotation_test(
        axes[3],
        fresh_obstacles,
        gt_wp,
        title="fresh nuScenes: rotate_around(+heading)",
        rotation_sign=+1,
    )

    fig.suptitle("Obstacle Rotation Sign Test", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = args.output or "outputs/debug_obstacles_rotation.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
