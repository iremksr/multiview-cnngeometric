
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from multiview.dataset import MessyTableMultiViewDataset


def to_image(tensor):
    """
    ImageNet normalize edilmiş görüntüyü
    tekrar görüntülenebilir RGB formatına çevirir.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = image * std + mean

    return np.clip(image, 0, 1)


def visualize_sample(dataset, index=0, max_objects=10):
    sample = dataset[index]

    center = sample["center"]
    views = sample["views"]

    center_boxes = sample["center_boxes"].numpy()
    view_boxes = sample["view_boxes"].numpy()

    instance_ids = sample["instance_ids"]
    view_ids = sample["view_ids"]

    num_views = views.shape[0]

    fig, axes = plt.subplots(
        1,
        num_views + 1,
        figsize=(5 * (num_views + 1), 5)
    )

    if num_views == 0:
        axes = [axes]

    images = [center] + list(views)

    titles = ["CENTER"] + [
        f"CAMERA {camera_id}"
        for camera_id in view_ids
    ]

    for view_index, (ax, image, title) in enumerate(
        zip(axes, images, titles)
    ):
        ax.imshow(to_image(image))
        ax.set_title(title)
        ax.axis("off")

        if view_index == 0:
            boxes = center_boxes
        else:
            boxes = view_boxes[view_index - 1]

        for object_index in range(
            min(max_objects, len(instance_ids))
        ):
            x1, y1, x2, y2 = boxes[object_index]

            color = plt.cm.tab20(
                object_index % 20
            )

            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=1.5,
                edgecolor=color,
                facecolor="none"
            )

            ax.add_patch(rect)

            ax.text(
                x1,
                y1 - 3,
                f"ID {instance_ids[object_index]}",
                color="white",
                fontsize=8,
                bbox={
                    "facecolor": color,
                    "alpha": 0.85,
                    "pad": 2
                }
            )

    fig.suptitle(
        f"Scene: {sample['scene_id']}",
        fontsize=14
    )

    plt.tight_layout()

    output = f"messytable_scene_{index}.png"
    plt.savefig(output, dpi=150, bbox_inches="tight")

    print("Görsel kaydedildi:", output)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="data_messytable"
    )
    parser.add_argument(
        "--split",
        default="train"
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=10
    )

    args = parser.parse_args()

    dataset = MessyTableMultiViewDataset(
        root=args.root,
        split=args.split
    )

    visualize_sample(
        dataset,
        index=args.index,
        max_objects=args.max_objects
    )
