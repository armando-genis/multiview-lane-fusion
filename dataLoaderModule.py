import os
import sys

from pathlib import Path
import re
import numpy as np
from PIL import Image, ImageOps
from collections import defaultdict


_IMAGE_RE = re.compile(r"(racecar_camera_camera_\d+_image_raw)_(\d{5})\.jpg")
_CAMERA_INDEX_RE = re.compile(r"camera_(\d+)_image_raw")


def _camera_index(cam_name: str) -> int:
    """Extract camera number from cam_name for consistent ordering (e.g. camera_0 -> 0)."""
    m = _CAMERA_INDEX_RE.search(cam_name)
    return int(m.group(1)) if m else 0

class SyncDataset:
    def __init__(self, root: Path):
        self.image_dir = root / "individual"
        self.lidar_dir = root / "lidar_bins"
        self.samples = self._index_samples()

    def _index_samples(self):
        samples = defaultdict(lambda: {"images": {}})

        for img_path in self.image_dir.glob("*.jpg"):
            m = _IMAGE_RE.match(img_path.name)
            if not m:
                continue
            cam_name = m.group(1)
            idx = int(m.group(2))
            samples[idx]["images"][cam_name] = img_path

        for bin_path in self.lidar_dir.glob("*.bin"):
            idx = int(bin_path.stem)
            samples[idx]["lidar"] = bin_path

        synced = {
            idx: s
            for idx, s in samples.items()
            if "lidar" in s and len(s["images"]) > 0
        }
        return dict(sorted(synced.items()))

    def indices(self):
        return list(self.samples.keys())

    def num_scenes(self) -> int:
        """Return the total number of scenes in the dataset."""
        return len(self.samples)

    def load_images(self, idx: int):
        items = sorted(
            self.samples[idx]["images"].items(),
            key=lambda x: _camera_index(x[0]),
        )
        imgs = {}
        for cam_name, path in items:
            with Image.open(path) as pil_img:
                pil_img = ImageOps.exif_transpose(pil_img)
                pil_img = pil_img.rotate(180, expand=False)
                pil_img = pil_img.convert("RGB")
                img = np.array(pil_img)
            imgs[cam_name] = img
        return imgs

    def load_lidar(self, idx: int):
        bin_path = self.samples[idx]["lidar"]
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        xyz = points[:, :3]
        valid_mask = np.isfinite(xyz).all(axis=1)
        return xyz[valid_mask]
