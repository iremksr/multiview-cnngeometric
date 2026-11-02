
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from image.normalization import normalize_image
from multiview.model import MultiViewCNNGeometric


DATA_DIR = Path("data_pcb")
CHECKPOINT = "trained_models/cnngeometric_affine.pth.tar"
VIEWS = ["left", "right", "top", "bottom"]
SIZE = 1024


def load_image(path):
    image = Image.open(path).convert("RGB")

    if image.size != (SIZE, SIZE):
        image = image.resize(
            (SIZE, SIZE),
            Image.Resampling.BILINEAR
        )

    array = np.asarray(image, dtype=np.float32) / 255.0

    tensor = torch.from_numpy(array)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)

    # Pretrained CNNGeometric'in beklediği normalizasyon
    tensor = normalize_image(tensor, forward=True)

    return tensor.squeeze(0)


def main():
    print("Loading MultiView CNNGeometric...")

    model = MultiViewCNNGeometric(
        checkpoint_path=CHECKPOINT,
        freeze_backbone=True,
    )

    model.eval()

    # Bir PCB komponenti seç
    center_files = sorted(DATA_DIR.glob("center_*.png"))

    if not center_files:
        raise FileNotFoundError(
            f"Center görüntüsü bulunamadı: {DATA_DIR}"
        )

    center_path = center_files[0]
    component = center_path.stem.removeprefix("center_")

    print("Component:", component)

    # Center: (1, 3, 240, 240)
    center = load_image(center_path).unsqueeze(0)

    # Views: (1, 4, 3, 240, 240)
    views = torch.stack(
        [
            load_image(DATA_DIR / f"{view}_{component}.png")
            for view in VIEWS
        ],
        dim=0
    ).unsqueeze(0)

    print("\nInput shapes:")
    print("Views:", tuple(views.shape))
    print("Center:", tuple(center.shape))

    # Forward pass
    with torch.no_grad():
        outputs = model(views, center)

    print("\nOutput shapes:")
    print("Theta:", tuple(outputs["theta"].shape))
    print("Warped views:", tuple(outputs["warped_views"].shape))
    print("Grid:", tuple(outputs["grid"].shape))

    print("\nPredicted affine matrices:")

    for i, view in enumerate(VIEWS):
        print(f"\n{view.upper()} -> CENTER")
        print(outputs["theta"][0, i])

    print("\nForward test completed successfully!")


    # Hizalanmış görüntüleri kaydet
    from torchvision.utils import save_image

    output_dir = Path("results/multiview_before_training")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize edilmiş görüntüleri tekrar görüntülenebilir hale getir
    center_vis = normalize_image(
        center.clone(), forward=False
    ).clamp(0, 1)

    save_image(
        center_vis,
        output_dir / f"{component}_center.png"
    )

    for i, view_name in enumerate(VIEWS):
        original_vis = normalize_image(
            views[:, i].clone(), forward=False
        ).clamp(0, 1)

        warped_vis = normalize_image(
            outputs["warped_views"][:, i].clone(),
            forward=False
        ).clamp(0, 1)

        save_image(
            original_vis,
            output_dir / f"{component}_{view_name}_original.png"
        )

        save_image(
            warped_vis,
            output_dir / f"{component}_{view_name}_warped.png"
        )

    print(f"\nImages saved to: {output_dir}")



if __name__ == "__main__":
    main()
