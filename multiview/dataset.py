"""MessyTable Mini MultiView dataset for CNNGeometric.

Expected layout:
 data_messytable/{train,val,test}/{scene_id}/center.jpg,view_*.jpg,annotations.json

Bounding boxes are xyxy in the 512x512 padded image coordinates saved by
our Colab extraction script.  Instance IDs are shared within each scene.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class MessyTableMultiViewDataset(Dataset):
    def __init__(
        self,
        root: str | Path = "data_messytable",
        split: str = "train",
        image_size: int = 512,
        max_views: int | None = None,
        normalize: bool = True,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'")
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if max_views is not None and max_views < 0:
            raise ValueError("max_views must be >= 0")
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.max_views = max_views
        self.normalize = normalize
        self.samples = sorted((self.root / split).glob("*/annotations.json"))
        if not self.samples:
            raise FileNotFoundError(f"No scenes found under {self.root / split}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as im:
            im = im.convert("RGB")  # grayscale JPG -> 3 channels for pretrained VGG
            if im.size != (self.image_size, self.image_size):
                im = im.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            arr = np.array(im, dtype=np.float32, copy=True) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        if self.normalize:
            # Matches the common VGG/ImageNet normalization. If your repository's
            # NormalizeImage uses another convention, change this to match it.
            mean = t.new_tensor([0.485, 0.456, 0.406])[:, None, None]
            std = t.new_tensor([0.229, 0.224, 0.225])[:, None, None]
            t = (t - mean) / std
        return t

    def __getitem__(self, index: int) -> dict[str, Any]:
        annotation_path = self.samples[index]
        scene_dir = annotation_path.parent
        with annotation_path.open("r", encoding="utf-8") as f:
            ann = json.load(f)

        cameras = ann["cameras"]
        center_id = str(ann["center_camera"])
        side_ids = [str(i) for i in ann["side_cameras"]]
        if self.max_views is not None:
            side_ids = side_ids[:self.max_views]

        ids = [str(i) for i in ann["common_instance_ids"]]
        # The JSON generator stores only shared instances in every selected camera.
        # Recheck after max_views filtering and fail loudly on malformed samples.
        selected_ids = [center_id] + side_ids
        ids = [i for i in ids if all(i in cameras[c]["instances"] for c in selected_ids)]
        if not ids:
            raise ValueError(f"No matched instances in {annotation_path}")

        center_info = cameras[center_id]
        center = self._load_image(scene_dir / center_info["filename"])
        views = [self._load_image(scene_dir / cameras[c]["filename"]) for c in side_ids]
        views_tensor = torch.stack(views) if views else center.new_empty((0, 3, self.image_size, self.image_size))

        # Bboxes are in the extractor's saved image coordinate system.
        # Scale them if a different image_size is requested here.
        saved_size = ann.get("image_size", [512, 512])
        saved_w, saved_h = float(saved_size[0]), float(saved_size[1])
        sx, sy = self.image_size / saved_w, self.image_size / saved_h

        def boxes_for(cam_id: str) -> torch.Tensor:
            vals = []
            for instance_id in ids:
                x1, y1, x2, y2 = cameras[cam_id]["instances"][instance_id]["bbox_xyxy"]
                vals.append([x1 * sx, y1 * sy, x2 * sx, y2 * sy])
            return torch.tensor(vals, dtype=torch.float32)

        center_boxes = boxes_for(center_id)
        view_boxes = torch.stack([boxes_for(c) for c in side_ids]) if side_ids else torch.empty((0, len(ids), 4), dtype=torch.float32)

# ==================================================
# TINY OBJECT FILTER
# ==================================================

        MAX_TINY_SIZE = 32
        MIN_TINY_SIZE = 2

        # Center bounding box boyutları
        widths = center_boxes[:, 2] - center_boxes[:, 0]
        heights = center_boxes[:, 3] - center_boxes[:, 1]

        max_sides = torch.maximum(widths, heights)
        min_sides = torch.minimum(widths, heights)

        # Tiny / Very Small nesneleri seç
        tiny_mask = (
            (max_sides <= MAX_TINY_SIZE)
            & (min_sides >= MIN_TINY_SIZE)
        )

        # Aynı instance ID'leri bütün kameralarda koru
        center_boxes = center_boxes[tiny_mask]
        view_boxes = view_boxes[:, tiny_mask, :]

        ids = [
            instance_id
            for instance_id, keep in zip(
                ids, tiny_mask.tolist()
            )
            if keep
        ]


        

        return {
            "center": center,                     # (3,H,W)
            "views": views_tensor,                # (V,3,H,W)
            "center_boxes": center_boxes,         # (N,4) xyxy pixels
            "view_boxes": view_boxes,             # (V,N,4) xyxy pixels
            "instance_ids": ids,                 # N shared IDs
            "view_ids": side_ids,
            "scene_id": ann["scene_id"],
            "split": self.split,
        }



def multiview_collate_fn(batch):
    """
    Farklı sayıda kamera ve nesne içeren
    sahneleri tek bir batch'e dönüştürür.
    """

    if not batch:
        raise ValueError("Empty batch")

    B = len(batch)

    max_V = max(
        item["views"].shape[0] for item in batch
    )

    max_N = max(
        item["center_boxes"].shape[0]
        for item in batch
    )

    C, H, W = batch[0]["center"].shape

    centers = torch.stack([
        item["center"] for item in batch
    ])

    views = centers.new_zeros(
        (B, max_V, C, H, W)
    )

    center_boxes = torch.zeros(
        (B, max_N, 4),
        dtype=torch.float32
    )

    view_boxes = torch.zeros(
        (B, max_V, max_N, 4),
        dtype=torch.float32
    )

    view_mask = torch.zeros(
        (B, max_V),
        dtype=torch.bool
    )

    instance_mask = torch.zeros(
        (B, max_N),
        dtype=torch.bool
    )

    for i, item in enumerate(batch):

        V = item["views"].shape[0]
        N = item["center_boxes"].shape[0]

        views[i, :V] = item["views"]

        center_boxes[i, :N] = (
            item["center_boxes"]
        )

        view_boxes[i, :V, :N] = (
            item["view_boxes"]
        )

        view_mask[i, :V] = True
        instance_mask[i, :N] = True

    return {
        "center": centers,
        "views": views,
        "center_boxes": center_boxes,
        "view_boxes": view_boxes,
        "view_mask": view_mask,
        "instance_mask": instance_mask,
        "instance_ids": [
            item["instance_ids"]
            for item in batch
        ],
        "view_ids": [
            item["view_ids"]
            for item in batch
        ],
        "scene_ids": [
            item["scene_id"]
            for item in batch
        ],
    }
