
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.cnn_geometric_model import CNNGeometric


class MultiViewCNNGeometric(nn.Module):
    def __init__(
        self,
        checkpoint_path,
        freeze_backbone=True,
        align_corners=False,
    ):
        super().__init__()

        self.align_corners = align_corners

        # Pretrained CNNGeometric Affine modeli
        self.registration = CNNGeometric(
            output_dim=6,
            feature_extraction_cnn="vgg",
            use_cuda=False,
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        state_dict = OrderedDict(
            (k.replace("vgg", "model"), v)
            for k, v in checkpoint["state_dict"].items()
        )

        self.registration.load_state_dict(
            state_dict,
            strict=True,
        )

        # Önce tüm parametreleri eğitilebilir yap
        for param in self.registration.parameters():
            param.requires_grad = True

        # VGG16'yı dondur
        if freeze_backbone:
            for param in (
                self.registration.FeatureExtraction.parameters()
            ):
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)

        # Dondurulmuş backbone'u eval modunda tut.
        # Böylece ileride BN gibi katmanlar olsa da
        # istatistikleri değişmez.
        if not any(
            p.requires_grad
            for p in self.registration.FeatureExtraction.parameters()
        ):
            self.registration.FeatureExtraction.eval()

        return self

    
    def forward(self, views, center):
        """
        views:  (B, 4, 3, H, W)
        center: (B, 3, H, W)

        H ve W orijinal çözünürlük olabilir (1024x1024).
        """

        B, V, C, H, W = views.shape

        # Hiç yan görüntü gelmediyse
        if V == 0:
            return {
                "theta": center.new_empty((B, 0, 2, 3)),
                "warped_views": center.new_empty((B, 0, C, H, W)),
                "grid": center.new_empty((B, 0, H, W, 2)),
            }

        sources = views.reshape(B * V, C, H, W)

        targets = center.unsqueeze(1).expand(
            B, V, C, H, W
        ).reshape(B * V, C, H, W)

        # 1. Affine tahmini için 240x240'a küçült
        sources_small = F.interpolate(
            sources,
            size=(240, 240),
            mode="bilinear",
            align_corners=False
        )

        targets_small = F.interpolate(
            targets,
            size=(240, 240),
            mode="bilinear",
            align_corners=False
        )

        # 2. CNNGeometric ile affine matrisini tahmin et
        theta = self.registration({
            "source_image": sources_small,
            "target_image": targets_small
        }).reshape(B * V, 2, 3)

        # 3. Affine dönüşümünü ORİJİNAL çözünürlükte uygula
        grid = F.affine_grid(
            theta,
            size=sources.shape,
            align_corners=self.align_corners
        )

        warped = F.grid_sample(
            sources,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=self.align_corners
        )

        return {
            "theta": theta.reshape(B, V, 2, 3),
            "warped_views": warped.reshape(B, V, C, H, W),
            "grid": grid.reshape(B, V, H, W, 2)
        }
