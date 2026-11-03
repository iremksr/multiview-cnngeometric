
import torch
from torch import nn


class MaskedMeanFusion(nn.Module):
    def __init__(self, feature_dim=1600, hidden_dim=128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        # Start with exactly zero correction.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, features, view_mask):
        """
        features:  (B, V, D)
        view_mask: (B, V), True for valid views
        """
        b, v, d = features.shape

        if view_mask.shape != (b, v):
            raise ValueError("Invalid view_mask shape")

        mask = view_mask.unsqueeze(-1).to(features.dtype)

        context = (features * mask).sum(dim=1)
        context = context / mask.sum(dim=1).clamp(min=1)

        context = context.unsqueeze(1).expand(-1, v, -1)

        fusion_input = torch.cat(
            [features, context], dim=-1
        )

        correction = self.mlp(fusion_input)

        return features + correction * mask
