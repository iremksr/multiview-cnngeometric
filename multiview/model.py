"""Multi-view affine registration using the pretrained CNNGeometric model."""
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
from model.cnn_geometric_model import CNNGeometric
from multiview.fusion import MaskedMeanFusion
from multiview.roi_context import ROIContextRefinement


class MultiViewCNNGeometric(nn.Module):
    def __init__(self, checkpoint_path, freeze_backbone=True,
                 align_corners=False, registration_size=240):
        super().__init__()
        self.align_corners = align_corners
        self.registration_size = registration_size
        self.freeze_backbone = freeze_backbone
        self.registration = CNNGeometric(output_dim=6,
                                         feature_extraction_cnn='vgg', use_cuda=False)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)
        if not isinstance(state_dict, dict):
            raise ValueError('Pretrained checkpoint must contain a state_dict.')
        state_dict = OrderedDict((k.replace('vgg', 'model'), v)
                                 for k, v in state_dict.items())
        self.registration.load_state_dict(state_dict, strict=True)
        if freeze_backbone:
            self.freeze_feature_extractor()

        self.fusion = MaskedMeanFusion(
            feature_dim=1600,
            hidden_dim=128,
        )

        self.roi_refinement = ROIContextRefinement(
            feature_channels=512,
            hidden_dim=128,
            roi_size=7,
            context_scale=3.0,
        )

    def freeze_feature_extractor(self):
        self.freeze_backbone = True
        for p in self.registration.FeatureExtraction.parameters():
            p.requires_grad_(False)
        self.registration.FeatureExtraction.eval()

    def unfreeze_feature_extractor(self):
        self.freeze_backbone = False
        for p in self.registration.FeatureExtraction.parameters():
            p.requires_grad_(True)

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
        view_mask=None,
        center_roi_boxes=None,
        view_roi_boxes=None,
        center_roi_mask=None,
        view_roi_mask=None,
    ):
        if views.ndim != 5 or center.ndim != 4:
            raise ValueError(
                "Expected views (B,V,C,H,W), center (B,C,H,W)"
            )

        b, v, c, h, w = views.shape

        if center.shape != (b, c, h, w):
            raise ValueError("Incompatible center shape")

        if view_mask is None:
            view_mask = torch.ones(
                b, v, device=views.device, dtype=torch.bool
            )
        else:
            if view_mask.shape != (b, v):
                raise ValueError("Invalid view_mask shape")
            view_mask = view_mask.to(
                device=views.device, dtype=torch.bool
            )

        identity = views.new_tensor([
            [1., 0., 0.],
            [0., 1., 0.]
        ])

        theta_flat = identity.expand(b * v, 2, 3).clone()
        flat_valid = view_mask.reshape(-1)

        # ROI refinement için feature map'ler
        center_features = None
        side_features = None

        if flat_valid.any():

            valid_indices = flat_valid.nonzero(as_tuple=True)[0]

            sources = views.reshape(
                b * v, c, h, w
            )[flat_valid]

            targets = center[:, None].expand(
                b, v, c, h, w
            ).reshape(b * v, c, h, w)[flat_valid]

            sources_small = F.interpolate(
                sources,
                size=(self.registration_size,) * 2,
                mode="bilinear",
                align_corners=False,
            )

            targets_small = F.interpolate(
                targets,
                size=(self.registration_size,) * 2,
                mode="bilinear",
                align_corners=False,
            )

            # 1. Pretrained VGG16 feature extraction
            feature_A = self.registration.FeatureExtraction(
                sources_small
            )

            feature_B = self.registration.FeatureExtraction(
                targets_small
            )

            # 2. Original CNNGeometric correlation
            correlation = self.registration.FeatureCorrelation(
                feature_A,
                feature_B
            )

            # 3. Regression convolution
            reg = self.registration.FeatureRegression

            features = reg.conv(correlation)
            features = features.flatten(start_dim=1)

            feature_dim = features.shape[-1]

            # 4. Restore multi-view structure
            all_features = features.new_zeros(
                b * v, feature_dim
            )

            all_features = all_features.index_copy(
                0, valid_indices, features
            )

            all_features = all_features.reshape(
                b, v, feature_dim
            )

            # 5. Existing Experiment C fusion
            fused = self.fusion(
                all_features,
                view_mask
            )

            fused_valid = fused.reshape(
                b * v, feature_dim
            )[flat_valid]

            # 6. Initial affine prediction
            predicted = reg.linear(
                fused_valid
            ).reshape(-1, 2, 3)

            theta_flat = theta_flat.index_copy(
                0, valid_indices, predicted
            )

            # 7. Restore VGG16 feature maps for ROI attention
            channels = feature_A.shape[1]
            fh, fw = feature_A.shape[-2:]

            side_features_flat = feature_A.new_zeros(
                b * v, channels, fh, fw
            )

            side_features_flat = side_features_flat.index_copy(
                0, valid_indices, feature_A
            )

            side_features = side_features_flat.reshape(
                b, v, channels, fh, fw
            )

            center_features_flat = feature_B.new_zeros(
                b * v, channels, fh, fw
            )

            center_features_flat = center_features_flat.index_copy(
                0, valid_indices, feature_B
            )

            # One center feature map per scene.
            # Select the first valid side-center pair.
            center_features_per_view = center_features_flat.reshape(
                b, v, channels, fh, fw
            )

            first_valid = view_mask.to(torch.int64).argmax(dim=1)

            center_features = center_features_per_view[
                torch.arange(b, device=views.device),
                first_valid
            ]

        # 8. Initial affine matrix
        theta0 = theta_flat.reshape(b, v, 2, 3)

        theta = theta0

        # 9. Experiment G: ROI Context Refinement
        roi_inputs = (
            center_roi_boxes,
            view_roi_boxes,
            center_roi_mask,
            view_roi_mask,
        )

        if any(x is not None for x in roi_inputs):
            if not all(x is not None for x in roi_inputs):
                raise ValueError(
                    "All ROI boxes and masks must be provided together"
                )

            if center_features is not None:
                theta = self.roi_refinement(
                    theta0=theta0,
                    center_features=center_features,
                    side_features=side_features,
                    center_boxes=center_roi_boxes,
                    side_boxes=view_roi_boxes,
                    center_mask=center_roi_mask,
                    side_mask=view_roi_mask,
                    view_mask=view_mask,
                    image_size=(h, w),
                )

        # Invalid views always retain identity
        theta = torch.where(
            view_mask[:, :, None, None],
            theta,
            identity.view(1, 1, 2, 3)
        )

        result = {
            "theta": theta,
            "theta0": theta0,
        }

        # 10. Optional image warping
        if return_warped:
            grid = F.affine_grid(
                theta.reshape(b * v, 2, 3),
                size=(b * v, c, h, w),
                align_corners=self.align_corners,
            )

            warped = F.grid_sample(
                views.reshape(b * v, c, h, w),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=self.align_corners,
            )

            result["warped"] = warped.reshape(
                b, v, c, h, w
            )

        return result