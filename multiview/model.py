
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.cnn_geometric_model import CNNGeometric


class MultiViewCNNGeometric(nn.Module):
    """
    Pretrained CNNGeometric tabanlı Multi-View Affine Alignment.

    Inputs:
        views:  (B, V, 3, H, W)
        center: (B, 3, H, W)

    Outputs:
        theta: (B, V, 2, 3)

    Optional:
        warped_views: (B, V, 3, H, W)
        grid:         (B, V, H, W, 2)

    Training:
        return_warped=False

    Visualization / Inference:
        return_warped=True
    """

    def __init__(
        self,
        checkpoint_path,
        freeze_backbone=True,
        align_corners=False,
        registration_size=240,
    ):
        super().__init__()

        self.align_corners = align_corners
        self.registration_size = registration_size
        self.freeze_backbone = freeze_backbone

        # ----------------------------------------
        # 1. CNNGeometric model
        # ----------------------------------------

        self.registration = CNNGeometric(
            output_dim=6,
            feature_extraction_cnn="vgg",
            use_cuda=False,
        )

        # ----------------------------------------
        # 2. Pretrained checkpoint
        # ----------------------------------------

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        state_dict = checkpoint.get(
            "state_dict",
            checkpoint,
        )

        state_dict = OrderedDict(
            (
                k.replace("vgg", "model"),
                v,
            )
            for k, v in state_dict.items()
        )

        self.registration.load_state_dict(
            state_dict,
            strict=True,
        )

        # ----------------------------------------
        # 3. Freeze backbone
        # ----------------------------------------

        if freeze_backbone:
            self.freeze_feature_extractor()

    def freeze_feature_extractor(self):
        """
        Freeze pretrained VGG feature extractor.
        Regression layers remain trainable.
        """

        self.freeze_backbone = True

        for param in (
            self.registration.FeatureExtraction.parameters()
        ):
            param.requires_grad = False

        self.registration.FeatureExtraction.eval()

    def unfreeze_feature_extractor(self):
        """
        Optional: unfreeze backbone for later fine-tuning.
        """

        self.freeze_backbone = False

        for param in (
            self.registration.FeatureExtraction.parameters()
        ):
            param.requires_grad = True

    def train(self, mode=True):
        super().train(mode)

        if self.freeze_backbone:
            self.registration.FeatureExtraction.eval()

        return self

    def forward(
        self,
        views,
        center,
        return_warped=False,
    ):
        """
        views:
            (B, V, 3, H, W)

        center:
            (B, 3, H, W)

        return_warped=False:
            Only predict affine theta.

        return_warped=True:
            Also produce warped views and sampling grids.
        """

        B, V, C, H, W = views.shape

        if center.shape != (B, C, H, W):
            raise ValueError(
                "Center and views dimensions do not match. "
                f"views={tuple(views.shape)}, "
                f"center={tuple(center.shape)}"
            )

        # ----------------------------------------
        # 1. Handle empty views
        # ----------------------------------------

        if V == 0:
            result = {
                "theta": center.new_empty(
                    (B, 0, 2, 3)
                )
            }

            if return_warped:
                result["warped_views"] = (
                    center.new_empty((B, 0, C, H, W))
                )

                result["grid"] = (
                    center.new_empty((B, 0, H, W, 2))
                )

            return result

        # ----------------------------------------
        # 2. Resize for CNNGeometric
        # ----------------------------------------

        sources = views.reshape(
            B * V, C, H, W
        )

        sources_small = F.interpolate(
            sources,
            size=(
                self.registration_size,
                self.registration_size,
            ),
            mode="bilinear",
            align_corners=False,
        )

        # Resize center only once per scene
        center_small = F.interpolate(
            center,
            size=(
                self.registration_size,
                self.registration_size,
            ),
            mode="bilinear",
            align_corners=False,
        )

        targets_small = (
            center_small.unsqueeze(1)
            .expand(
                B,
                V,
                C,
                self.registration_size,
                self.registration_size,
            )
            .reshape(
                B * V,
                C,
                self.registration_size,
                self.registration_size,
            )
        )

        # ----------------------------------------
        # 3. Affine regression
        # ----------------------------------------

        theta = self.registration({
            "source_image": sources_small,
            "target_image": targets_small,
        })

        theta = theta.reshape(
            B, V, 2, 3
        )

        result = {
            "theta": theta
        }

        # ----------------------------------------
        # 4. Training optimization
        # ----------------------------------------

        if not return_warped:
            return result

        # ----------------------------------------
        # 5. Optional full-resolution warp
        # ----------------------------------------

        theta_flat = theta.reshape(
            B * V, 2, 3
        )

        grid = F.affine_grid(
            theta_flat,
            size=(B * V, C, H, W),
            align_corners=self.align_corners,
        )

        warped = F.grid_sample(
            sources,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=self.align_corners,
        )

        result["warped_views"] = warped.reshape(
            B, V, C, H, W
        )

        result["grid"] = grid.reshape(
            B, V, H, W, 2
        )

        return result
