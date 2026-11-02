
from collections import Counter
import numpy as np

from multiview.dataset import MessyTableMultiViewDataset


def main():
    dataset = MessyTableMultiViewDataset(
        root="data_messytable",
        split="train"
    )

    counts = []
    empty_scenes = []
    enough_scenes = []

    for i in range(len(dataset)):
        sample = dataset[i]
        n = len(sample["instance_ids"])

        counts.append(n)

        if n == 0:
            empty_scenes.append(sample["scene_id"])

        if n >= 3:
            enough_scenes.append(sample["scene_id"])

    counts = np.array(counts)

    print("Toplam sahne:", len(dataset))
    print("Toplam tiny nesne:", counts.sum())
    print("Ortalama tiny nesne:", counts.mean())
    print("Medyan tiny nesne:", np.median(counts))
    print("Maksimum tiny nesne:", counts.max())

    print("\n--- Sahne dağılımı ---")

    distribution = Counter(counts.tolist())

    for n, scene_count in sorted(distribution.items()):
        print(
            f"{n} tiny nesne içeren sahne: "
            f"{scene_count}"
        )

    print("\nBoş sahne sayısı:", len(empty_scenes))
    print(
        "En az 3 tiny nesne içeren sahne:",
        len(enough_scenes)
    )

    print(
        "En az 3 tiny nesneli sahne oranı:",
        f"%{100 * len(enough_scenes) / len(dataset):.1f}"
    )


if __name__ == "__main__":
    main()
