
import torch
from torch.utils.data import DataLoader

from multiview.dataset import (
    MessyTableMultiViewDataset,
    multiview_collate_fn,
)

def main():
    print("Dataset testi başlıyor...", flush=True)

    dataset = MessyTableMultiViewDataset(
        root="data_messytable",
        split="train",
    )

    print("Toplam sahne:", len(dataset))

    sample = dataset[0]

    print("\n--- Tek sahne ---")
    print("Scene ID:", sample["scene_id"])
    print("Center:", sample["center"].shape)
    print("Views:", sample["views"].shape)
    print("Center boxes:", sample["center_boxes"].shape)
    print("View boxes:", sample["view_boxes"].shape)
    print("Instance IDs:", sample["instance_ids"])
    print("Tiny nesne sayısı:", len(sample["instance_ids"]))

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=multiview_collate_fn,
    )

    batch = next(iter(loader))

    print("\n--- Batch ---")

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(key, tuple(value.shape))
        else:
            print(key, value)

    print("\nDataset testi tamamlandı!")


if __name__ == "__main__":
    main()
