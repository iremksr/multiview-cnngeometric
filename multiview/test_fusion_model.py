
import torch

from multiview.model import MultiViewCNNGeometric
from multiview.fusion import MaskedMeanFusion
from torch import nn

CHECKPOINT = r"D:\irem_ssh\checkpoints\streetview_affine.pth.tar"


def test_fusion_module():
    print("\n[1] Fusion module test")

    torch.manual_seed(42)

    fusion = MaskedMeanFusion(
        feature_dim=1600,
        hidden_dim=128,
    )

    features = torch.randn(2, 4, 1600)

    mask = torch.tensor([
        [True, True, True, False],
        [True, True, False, False],
    ])

    output = fusion(features, mask)

    torch.testing.assert_close(
        output, features, rtol=0, atol=0
    )

    print("PASS: Fusion starts as identity")


def test_pretrained_equivalence():
    print("\n[2] Pretrained CNNGeometric equivalence test")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = MultiViewCNNGeometric(
        checkpoint_path=CHECKPOINT,
        freeze_backbone=True,
    ).to(device).eval()

    # Two scenes, three views per scene
    views = torch.randn(
        2, 3, 3, 240, 240, device=device
    )

    center = torch.randn(
        2, 3, 240, 240, device=device
    )

    view_mask = torch.tensor([
        [True, True, False],
        [True, False, True],
    ], device=device)

    with torch.no_grad():
        theta_fusion = model(
            views, center, view_mask=view_mask
        )["theta"]

        # Run original pairwise CNNGeometric
        theta_baseline = torch.eye(
            2, 3, device=device
        ).reshape(1, 1, 2, 3).expand(
            2, 3, 2, 3
        ).clone()

        for batch_idx in range(2):
            for view_idx in range(3):
                if not view_mask[batch_idx, view_idx]:
                    continue

                source = views[
                    batch_idx, view_idx
                ].unsqueeze(0)

                target = center[
                    batch_idx
                ].unsqueeze(0)

                theta = model.registration({
                    "source_image": source,
                    "target_image": target,
                })

                theta_baseline[
                    batch_idx, view_idx
                ] = theta.reshape(2, 3)

    difference = (
        theta_fusion - theta_baseline
    ).abs().max().item()

    print(f"Maximum theta difference: {difference:.8f}")

    torch.testing.assert_close(
        theta_fusion,
        theta_baseline,
        rtol=1e-5,
        atol=1e-5,
    )

    print("PASS: Fusion preserves pretrained predictions")


def test_fusion_gradients():
    print("\n[3] Fusion gradient test")

    fusion = MaskedMeanFusion(
        feature_dim=1600,
        hidden_dim=128,
    )

    features = torch.randn(
        2, 3, 1600, requires_grad=True
    )

    mask = torch.ones(
        2, 3, dtype=torch.bool
    )

    output = fusion(features, mask)

    loss = output.square().mean()
    loss.backward()

    grad = fusion.mlp[-1].weight.grad

    assert grad is not None
    assert grad.abs().sum().item() > 0

    print("PASS: Fusion parameters receive gradients")


def test_padding_independence():
    print("\n[4] Padding independence test")

    torch.manual_seed(42)

    fusion = MaskedMeanFusion(
        feature_dim=1600,
        hidden_dim=128,
    )

    # Identity initialization dışında fusion davranışını da test et
    with torch.no_grad():
        nn.init.normal_(fusion.mlp[-1].weight, std=0.01)

    features = torch.randn(1, 3, 1600)
    mask = torch.tensor([[True, True, False]])

    output1 = fusion(features, mask)

    modified = features.clone()
    modified[:, 2] = torch.randn_like(modified[:, 2]) * 1000

    output2 = fusion(modified, mask)

    torch.testing.assert_close(
        output1[:, :2],
        output2[:, :2],
        rtol=1e-5,
        atol=1e-5,
    )

    print("PASS: Padding does not affect valid views")


if __name__ == "__main__":
    test_fusion_module()
    test_pretrained_equivalence()
    test_fusion_gradients()
    test_padding_independence()

    print("\nALL TESTS PASSED")
