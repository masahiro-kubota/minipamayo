"""nuScenes dataset for Stage 0 (MLP regression).

Phase 3 GT: [steer, throttle] from ego pose differences.
Phase 4 GT: (a, kappa) x 64 from inverse dynamics (future implementation).
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from torchvision import transforms

# ImageNet normalization (DINOv2 standard)
IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def quaternion_to_yaw(q: list[float]) -> float:
    """Convert quaternion [w, x, y, z] to yaw angle (radians)."""
    quat = Quaternion(q)
    # yaw = rotation around z-axis
    return quat.yaw_pitch_roll[0]


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-π, π]."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


class NuScenesSteerThrottleDataset(Dataset):
    """nuScenes dataset for Phase 3 fail-fast: predict [steer, throttle].

    GT is computed from consecutive ego poses:
    - steer: yaw change between consecutive frames (rad)
    - throttle: speed change between consecutive frames (m/s)
    """

    def __init__(
        self,
        nuscenes_root: str | Path,
        version: str = "v1.0-mini",
        camera: str = "CAM_FRONT",
        transform: transforms.Compose | None = None,
    ):
        self.nuscenes_root = Path(nuscenes_root)
        self.camera = camera
        self.transform = transform or IMAGE_TRANSFORM

        # Lazy import to avoid hard dependency at module level
        from nuscenes.nuscenes import NuScenes

        self.nusc = NuScenes(version=version, dataroot=str(self.nuscenes_root), verbose=False)

        # Build sample list with consecutive pairs for GT computation
        self.samples = self._build_samples()

    def _build_samples(self) -> list[dict]:
        """Build list of samples with image path + GT action.

        Requires 3 consecutive frames: computes speed from position differences,
        then steer = yaw change, throttle = speed change (acceleration).
        """
        samples = []
        for scene in self.nusc.scene:
            # Collect all frames for this scene
            frames = []
            sample_token = scene["first_sample_token"]
            while sample_token:
                sample = self.nusc.get("sample", sample_token)
                cam_data = self.nusc.get("sample_data", sample["data"][self.camera])
                ego_pose = self.nusc.get("ego_pose", cam_data["ego_pose_token"])
                frames.append(
                    {
                        "image_path": str(self.nuscenes_root / cam_data["filename"]),
                        "ego_pose": ego_pose,
                        "timestamp": cam_data["timestamp"],
                    }
                )
                sample_token = sample["next"]

            # Compute speeds between consecutive frames
            speeds = []
            for i in range(len(frames) - 1):
                dt = (frames[i + 1]["timestamp"] - frames[i]["timestamp"]) / 1e6
                if dt > 0:
                    pos_a = np.array(frames[i]["ego_pose"]["translation"][:2])
                    pos_b = np.array(frames[i + 1]["ego_pose"]["translation"][:2])
                    speeds.append(np.linalg.norm(pos_b - pos_a) / dt)
                else:
                    speeds.append(0.0)

            # Build samples: need frame i with speed[i-1] and speed[i]
            for i in range(1, len(frames) - 1):
                yaw_prev = quaternion_to_yaw(frames[i - 1]["ego_pose"]["rotation"])
                yaw_curr = quaternion_to_yaw(frames[i]["ego_pose"]["rotation"])
                steer = normalize_angle(yaw_curr - yaw_prev)
                throttle = speeds[i] - speeds[i - 1]  # acceleration

                samples.append(
                    {
                        "image_path": frames[i]["image_path"],
                        "steer": float(steer),
                        "throttle": float(throttle),
                    }
                )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Load and transform image
        image = Image.open(sample["image_path"]).convert("RGB")
        pixel_values = self.transform(image)

        # GT action
        action = torch.tensor([sample["steer"], sample["throttle"]], dtype=torch.float32)

        return {
            "pixel_values": pixel_values,
            "action": action,
        }
