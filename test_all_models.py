
import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from model.cnn_geometric_model import CNNGeometric
from geotnf.transformation import GeometricTnf
from image.normalization import normalize_image


CONFIGS = {
    "affine": {
        "output_dim": 6,
        "checkpoint": "trained_models/cnngeometric_affine.pth.tar",
    },
    "hom": {
        "output_dim": 8,
        "checkpoint": "trained_models/cnngeometric_homography.pth.tar",
    },
    "tps": {
        "output_dim": 18,
        "checkpoint": "trained_models/cnngeometric_tps.pth.tar",
    },
}

VIEWS = ("left", "right", "top", "bottom")


def load_image(path, size):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def to_pil(tensor):
    x = normalize_image(tensor.detach().cpu(), forward=False)
    arr = x.squeeze(0).permute(1, 2, 0).numpy()
    arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def load_model(model_name, checkpoint_path):
    cfg = CONFIGS[model_name]

    model = CNNGeometric(
        output_dim=cfg["output_dim"],
        feature_extraction_cnn="vgg",
        use_cuda=False,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = OrderedDict(
        (key.replace("vgg", "model"), value)
        for key, value in checkpoint["state_dict"].items()
    )

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def make_comparison(center, source, warped):
    overlay = Image.blend(center, warped, 0.5)

    images = [center, source, warped, overlay]
    labels = ["CENTER", "SOURCE", "WARPED", "OVERLAY"]

    width, height = center.size
    header = 28

    canvas = Image.new(
        "RGB",
        (width * 4, height + header),
        "white",
    )
    draw = ImageDraw.Draw(canvas)

    for idx, (image, label) in enumerate(zip(images, labels)):
        x = idx * width
        canvas.paste(image, (x, header))
        draw.text((x + 8, 7), label, fill="black")

    return canvas


def run(args):
    data_dir = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    centers = sorted(data_dir.glob("center_*.png"))

    if not centers:
        raise FileNotFoundError(
            f"Center görüntüsü bulunamadı: {data_dir}"
        )

    print(f"Found {len(centers)} components")

    results = {}

    for model_name in args.models:
        cfg = CONFIGS[model_name]
        checkpoint_path = Path(cfg["checkpoint"])

        print(f"\nLoading {model_name.upper()} model...")
        model = load_model(model_name, checkpoint_path)

        transformer = GeometricTnf(
            geometric_model=model_name,
            out_h=args.size,
            out_w=args.size,
            use_cuda=False,
        )

        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        results[model_name] = {}

        with torch.no_grad():
            for center_path in centers:
                component = center_path.stem.removeprefix("center_")

                target_raw = load_image(center_path, args.size)
                target = normalize_image(
                    target_raw.clone(), forward=True
                )

                for view in VIEWS:
                    source_path = data_dir / f"{view}_{component}.png"

                    if not source_path.exists():
                        print(f"Missing: {source_path}")
                        continue

                    source_raw = load_image(source_path, args.size)
                    source = normalize_image(
                        source_raw.clone(), forward=True
                    )

                    theta = model({
                        "source_image": source,
                        "target_image": target,
                    })

                    if model_name == "affine":
                        transform_params = theta.view(-1, 2, 3)
                    else:
                        transform_params = theta

                    warped = transformer(
                        source,
                        transform_params,
                    )

                    center_img = to_pil(target)
                    source_img = to_pil(source)
                    warped_img = to_pil(warped)

                    comparison = make_comparison(
                        center_img,
                        source_img,
                        warped_img,
                    )

                    name = f"{component}_{view}"
                    comparison.save(model_dir / f"{name}_comparison.png")
                    warped_img.save(model_dir / f"{name}_warped.png")

                    results[model_name][name] = {
                        "theta": theta.cpu().flatten().tolist()
                    }

                    print(
                        f"[{model_name.upper()}] "
                        f"{component} / {view} "
                        f"theta shape={tuple(theta.shape)}"
                    )

    with open(output_dir / "transforms.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFinished! Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="data_pcb")
    parser.add_argument("--output", default="results/all_models")
    parser.add_argument("--size", type=int, default=240)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(CONFIGS),
        default=["affine", "hom", "tps"],
    )

    run(parser.parse_args())
