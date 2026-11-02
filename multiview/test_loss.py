
import torch

from multiview.dataset import (
    MessyTableMultiViewDataset,
    multiview_collate_fn,
)
from multiview.loss import TinyAlignmentLoss

from torch.utils.data import DataLoader


def main():
    dataset = MessyTableMultiViewDataset(
        root="data_messytable",
        split="train",
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=multiview_collate_fn,
    )

    batch = next(iter(loader))

    B, V = batch["views"].shape[:2]

    # Başlangıçta identity affine
    theta = torch.tensor(
        [[1., 0., 0.],
         [0., 1., 0.]],
        dtype=torch.float32,
    ).view(1, 1, 2, 3).repeat(B, V, 1, 1)

    criterion = TinyAlignmentLoss(
        image_size=512,
        center_weight=1.0,
        box_weight=0.25,
        reg_weight=0.01,
    )

    result = criterion(
        theta=theta,
        center_boxes=batch["center_boxes"],
        view_boxes=batch["view_boxes"],
        view_mask=batch["view_mask"],
        instance_mask=batch["instance_mask"],
    )

    print("=== LOSS TEST ===")

    for key, value in result.items():
        print(f"{key}: {value.item():.6f}")

    print("Loss testi tamamlandı!")


if __name__ == "__main__":
    main()
