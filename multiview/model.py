"""Multi-view affine registration using the pretrained CNNGeometric model."""
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
from model.cnn_geometric_model import CNNGeometric


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

    def forward(self, views, center, return_warped=False, view_mask=None):
        if views.ndim != 5 or center.ndim != 4:
            raise ValueError('Expected views (B,V,C,H,W), center (B,C,H,W).')
        b, v, c, h, w = views.shape
        if center.shape != (b, c, h, w):
            raise ValueError(f'Incompatible shapes: views={views.shape}, center={center.shape}')
        if view_mask is None:
            view_mask = torch.ones((b, v), dtype=torch.bool, device=views.device)
        elif view_mask.shape != (b, v):
            raise ValueError(f'view_mask must be {(b, v)}, got {tuple(view_mask.shape)}')
        else:
            view_mask = view_mask.to(device=views.device, dtype=torch.bool)

        # Identity for invalid/padded views. Only real views are passed to CNNGeometric.
        identity = views.new_tensor([[1., 0., 0.], [0., 1., 0.]])
        theta_flat = identity.expand(b * v, 2, 3).clone()
        flat_valid = view_mask.reshape(-1)
        if flat_valid.any():
            sources = views.reshape(b * v, c, h, w)[flat_valid]
            targets = center[:, None].expand(b, v, c, h, w).reshape(b * v, c, h, w)[flat_valid]
            sources_small = F.interpolate(sources, size=(self.registration_size,) * 2,
                                          mode='bilinear', align_corners=False)
            targets_small = F.interpolate(targets, size=(self.registration_size,) * 2,
                                          mode='bilinear', align_corners=False)
            predicted = self.registration({'source_image': sources_small,
                                           'target_image': targets_small}).reshape(-1, 2, 3)
            theta_flat = theta_flat.index_copy(0, flat_valid.nonzero(as_tuple=True)[0], predicted)
        theta = theta_flat.reshape(b, v, 2, 3)
        result = {'theta': theta}
        if return_warped:
            sources = views.reshape(b * v, c, h, w)
            grid = F.affine_grid(theta_flat, (b * v, c, h, w),
                                 align_corners=self.align_corners)
            warped = F.grid_sample(sources, grid, mode='bilinear',
                                    padding_mode='zeros', align_corners=self.align_corners)
            result['warped_views'] = warped.reshape(b, v, c, h, w)
            result['grid'] = grid.reshape(b, v, h, w, 2)
        return result
