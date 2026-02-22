"""Debug visualization: 6-camera surround view + BEV obstacle map.

Verifies that obstacle positions in ego-centric BEV match what cameras see.

Usage:
    cd minipamayo && uv run python -m minipamayo.debug_obstacles \
        --nuscenes_root /mnt/ssd/nuscenes \
        --n_vis 3
"""

import argparse

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion

matplotlib.use("Agg")

# nuScenes 6 cameras in surround order
CAMERAS = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
]

OBSTACLE_PREFIXES = ("vehicle.", "human.pedestrian.", "movable_object.")
OBSTACLE_MAX_RANGE = 50.0


def get_obstacles_ego(nusc, sample, ego_position, ego_heading):
    """Get obstacles in ego-centric frame (same logic as NuScenesTrajectoryDataset)."""
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
                "distance": dist,
            }
        )

    return obstacles


def draw_bev_debug(ax, obstacles, gt_waypoints=None):
    """Draw BEV with obstacles, ego marker, and directional guides."""
    # Coordinate convention: ego-centric +x=forward, +y=left
    # BEV display: forward=up, right=right → plot(-y, x)

    for obs in obstacles:
        cx, cy = obs["center"]
        w, length = obs["size"]
        heading = obs["heading"]

        # Category-based coloring
        cat = obs["category"]
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
        t = matplotlib.transforms.Affine2D().rotate_around(-cy, cx, -heading) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)

        # Label with category short name + distance
        short = cat.split(".")[-1][:8]
        ax.text(
            -cy,
            cx,
            f"{short}\n{obs['distance']:.0f}m",
            fontsize=5,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            zorder=4,
        )

    # GT trajectory if available
    if gt_waypoints is not None:
        gt = np.array(gt_waypoints)
        ax.plot(-gt[:, 1], gt[:, 0], "b-o", markersize=3, linewidth=1.5, label="GT", zorder=5)

    # Ego marker
    ax.annotate(
        "",
        xy=(0, 1.5),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="green", lw=2),
        zorder=10,
    )
    ax.plot(0, 0, "gs", markersize=8, zorder=10)
    ax.text(0, -1.5, "EGO", fontsize=7, ha="center", va="top", color="green", fontweight="bold")

    # FOV guide lines (approximate camera coverage)
    fov_range = 40
    # Front ~120° → ±60°
    for angle_deg in [-60, 60]:
        rad = np.radians(90 - angle_deg)  # 90° = forward in display coords
        dx = fov_range * np.cos(rad)
        dy = fov_range * np.sin(rad)
        ax.plot([0, dx], [0, dy], "g--", alpha=0.2, linewidth=0.5)

    # Quadrant labels
    ax.text(0, fov_range * 0.9, "FRONT", fontsize=7, ha="center", va="top", color="gray", alpha=0.5)
    ax.text(
        0, -fov_range * 0.9, "BACK", fontsize=7, ha="center", va="bottom", color="gray", alpha=0.5
    )
    ax.text(
        -fov_range * 0.9,
        0,
        "RIGHT",
        fontsize=7,
        ha="left",
        va="center",
        color="gray",
        alpha=0.5,
        rotation=90,
    )
    ax.text(
        fov_range * 0.9,
        0,
        "LEFT",
        fontsize=7,
        ha="right",
        va="center",
        color="gray",
        alpha=0.5,
        rotation=90,
    )

    ax.set_xlim(-fov_range, fov_range)
    ax.set_ylim(-fov_range, fov_range)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("← Right    Left →", fontsize=7)
    ax.set_ylabel("← Back    Front →", fontsize=7)
    ax.set_title("BEV (ego-centric)", fontsize=9)

    # Legend
    legend_patches = [
        mpatches.Patch(color="tab:orange", alpha=0.4, label="vehicle"),
        mpatches.Patch(color="tab:red", alpha=0.4, label="pedestrian"),
        mpatches.Patch(color="tab:gray", alpha=0.4, label="movable_obj"),
    ]
    ax.legend(handles=legend_patches, fontsize=6, loc="upper right")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nuscenes_root", type=str, default="/mnt/ssd/nuscenes")
    parser.add_argument("--nuscenes_version", type=str, default="v1.0-trainval")
    parser.add_argument("--n_vis", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--scene_idx", type=int, default=0, help="Scene index to visualize")
    parser.add_argument(
        "--sample_offset", type=int, default=5, help="Start from this sample in scene"
    )
    args = parser.parse_args()

    from nuscenes.nuscenes import NuScenes

    print(f"Loading nuScenes {args.nuscenes_version} from {args.nuscenes_root}...")
    nusc = NuScenes(version=args.nuscenes_version, dataroot=args.nuscenes_root, verbose=False)

    scene = nusc.scene[args.scene_idx]
    print(f"Scene: {scene['name']} ({scene['description']})")

    # Collect sample tokens
    sample_tokens = []
    token = scene["first_sample_token"]
    while token:
        sample_tokens.append(token)
        token = nusc.get("sample", token)["next"]

    n_vis = min(args.n_vis, len(sample_tokens) - args.sample_offset)
    selected = sample_tokens[args.sample_offset : args.sample_offset + n_vis]

    # Layout per sample: 2 rows x 4 cols — 3 cameras (top) + 3 cameras (bottom) + BEV (right, span 2 rows)
    from matplotlib.gridspec import GridSpec

    total_rows = n_vis * 2
    fig = plt.figure(figsize=(24, 7 * n_vis))
    gs = GridSpec(total_rows, 4, figure=fig, hspace=0.3, wspace=0.2)

    for vi, sample_token in enumerate(selected):
        sample = nusc.get("sample", sample_token)

        # Get ego pose from CAM_FRONT
        cam_front_data = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        ego_pose = nusc.get("ego_pose", cam_front_data["ego_pose_token"])
        ego_pos = np.array(ego_pose["translation"][:2])
        ego_heading = Quaternion(ego_pose["rotation"]).yaw_pitch_roll[0]

        obstacles = get_obstacles_ego(nusc, sample, ego_pos, ego_heading)
        print(f"  Sample {vi}: {len(obstacles)} obstacles")

        row_base = vi * 2

        # Top row cameras: FRONT_LEFT, FRONT, FRONT_RIGHT
        for ci, cam_name in enumerate(CAMERAS[:3]):
            ax = fig.add_subplot(gs[row_base, ci])
            cam_data = nusc.get("sample_data", sample["data"][cam_name])
            img_path = f"{args.nuscenes_root}/{cam_data['filename']}"
            img = plt.imread(img_path)
            ax.imshow(img)
            ax.set_title(cam_name, fontsize=8)
            ax.axis("off")

        # Bottom row cameras: BACK_LEFT, BACK, BACK_RIGHT (left-to-right spatial order)
        for ci, cam_name in enumerate(["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]):
            ax = fig.add_subplot(gs[row_base + 1, ci])
            cam_data = nusc.get("sample_data", sample["data"][cam_name])
            img_path = f"{args.nuscenes_root}/{cam_data['filename']}"
            img = plt.imread(img_path)
            ax.imshow(img)
            ax.set_title(cam_name, fontsize=8)
            ax.axis("off")

        # BEV spans 2 rows on the right column
        ax_bev = fig.add_subplot(gs[row_base : row_base + 2, 3])
        draw_bev_debug(ax_bev, obstacles)

    fig.suptitle(f"Obstacle Debug — Scene: {scene['name']}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = args.output or f"outputs/debug_obstacles_scene{args.scene_idx}.png"
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
