
"""Experiment G: Geometry-guided ROI context attention and affine refinement.

Architecture:
    VGG feature maps
        -> ROI context extraction
        -> Geometry-guided cross-attention
        -> Residual affine refinement

Affine convention:
    theta0 maps CENTER normalized coordinates to SIDE normalized coordinates.

Bounding boxes:
    Original image pixel coordinates, xyxy.

Instance IDs are never used by this module.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def roi_grid_features(
    features,
    boxes,
    valid,
    image_hw,
    output_size=7,
    expand=3.0,
):
    """Extract contextual ROI features using grid_sample.

    Args:
        features: (B, C, Hf, Wf)
        boxes: (B, N, 4), original image pixel coordinates
        valid: (B, N), boolean mask
        image_hw: (H, W)
        output_size: spatial ROI feature size
        expand: ROI expansion factor

    Returns:
        (B, N, C, S, S)
    """
    b, c, hf, wf = features.shape
    n = boxes.shape[1]

    if n == 0:
        return features.new_zeros(
            b, 0, c, output_size, output_size
        )

    ih, iw = image_hw

    boxes = boxes.to(
        device=features.device,
        dtype=features.dtype,
    )
    valid = valid.to(
        device=features.device,
        dtype=torch.bool,
    )

    x1, y1, x2, y2 = boxes.unbind(-1)

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    bw = ((x2 - x1).clamp(min=1) * expand).clamp(max=iw)
    bh = ((y2 - y1).clamp(min=1) * expand).clamp(max=ih)

    t = torch.linspace(
        -0.5,
        0.5,
        output_size,
        device=features.device,
        dtype=features.dtype,
    )

    yy, xx = torch.meshgrid(t, t, indexing="ij")

    x = cx[..., None, None] + bw[..., None, None] * xx
    y = cy[..., None, None] + bh[..., None, None] * yy

    # align_corners=False normalized coordinates
    gx = 2 * x / iw - 1
    gy = 2 * y / ih - 1

    grid = torch.stack(
        (gx, gy), dim=-1
    ).reshape(b * n, output_size, output_size, 2)

    expanded_features = (
        features[:, None]
        .expand(-1, n, -1, -1, -1)
        .reshape(b * n, c, hf, wf)
    )

    sampled = F.grid_sample(
        expanded_features,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    sampled = sampled.reshape(
        b, n, c, output_size, output_size
    )

    return (
        sampled
        * valid[:, :, None, None, None].to(sampled.dtype)
    )


class ROIContextEncoder(nn.Module):
    """Encode each expanded ROI into a feature descriptor."""

    def __init__(
        self,
        in_channels,
        dim=128,
        roi_size=7,
        expand=3.0,
    ):
        super().__init__()

        self.roi_size = roi_size
        self.expand = expand

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(
        self,
        feature_map,
        boxes,
        mask,
        image_hw,
    ):
        rois = roi_grid_features(
            features=feature_map,
            boxes=boxes,
            valid=mask,
            image_hw=image_hw,
            output_size=self.roi_size,
            expand=self.expand,
        )

        b, n, c, s, _ = rois.shape

        if n == 0:
            return feature_map.new_zeros(
                b, 0, self.proj[0].out_channels
            )

        z = self.proj(
            rois.reshape(b * n, c, s, s)
        )

        z = z.flatten(1).reshape(b, n, -1)

        z = F.normalize(z, dim=-1)

        return z * mask[:, :, None].to(z.dtype)


class GeometryGuidedROIAttention(nn.Module):
    """Soft cross-attention with geometric distance prior.

    Center ROIs are queries.
    Side ROIs are keys and values.

    No hard instance matching or instance IDs are used.
    """

    def __init__(
        self,
        dim=128,
        geometry_weight=1.0,
        distance_scale=64.0,
    ):
        super().__init__()

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)

        self.geometry_weight = geometry_weight
        self.distance_scale = distance_scale

        # Optional no-match token
        self.no_match = nn.Parameter(torch.zeros(dim))

    def forward(
        self,
        center_z,
        side_z,
        side_boxes,
        center_centers_in_side,
        center_mask,
        side_mask,
    ):
        b, n, d = center_z.shape
        m = side_z.shape[1]

        if n == 0 or m == 0:
            context = center_z.new_zeros(b, d)
            weights = center_z.new_zeros(b, n, m + 1)
            return context, weights

        query = self.q(center_z)
        key = self.k(side_z)
        value = self.v(side_z)

        scores = (
            query @ key.transpose(1, 2)
        ) / math.sqrt(d)

        # Compare projected CENTER ROI centers
        # against SIDE ROI centers.
        side_xy = (
            side_boxes[..., :2]
            + side_boxes[..., 2:]
        ) / 2

        distance = torch.cdist(
            center_centers_in_side.float(),
            side_xy.float(),
        ).to(scores.dtype)

        scores = scores - (
            self.geometry_weight
            * distance
            / self.distance_scale
        )

        # Invalid side ROIs cannot receive attention.
        scores = scores.masked_fill(
            ~side_mask[:, None, :],
            torch.finfo(scores.dtype).min,
        )

        # A center ROI may have no corresponding side ROI.
        null_score = (
            query * self.no_match
        ).sum(dim=-1, keepdim=True) / math.sqrt(d)

        scores = torch.cat(
            (scores, null_score),
            dim=-1,
        )

        weights = torch.softmax(scores, dim=-1)

        weights = (
            weights
            * center_mask[:, :, None].to(weights.dtype)
        )

        # Ignore the no-match token in feature aggregation.
        context_per_roi = weights[..., :m] @ value

        context = (
            context_per_roi
            * center_mask[:, :, None].to(context_per_roi.dtype)
        ).sum(dim=1)

        context = context / (
            center_mask.sum(dim=1)
            .clamp(min=1)
            .unsqueeze(-1)
        )

        return context, weights


class ResidualAffineRefinement(nn.Module):
    """Predict an additive correction to the initial affine matrix.

    Zero initialization ensures theta_final == theta0
    before the refinement head is trained.
    """

    def __init__(self, dim=128, hidden=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 6),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, theta0, context):
        delta = self.net(context).reshape(-1, 2, 3)

        return theta0 + delta


class ExperimentGHead(nn.Module):
    """ROI attention and refinement for one center-side pair per item."""

    def __init__(
        self,
        feature_channels,
        dim=128,
        roi_size=7,
        expand=3.0,
        geometry_weight=1.0,
        distance_scale=64.0,
    ):
        super().__init__()

        self.encoder = ROIContextEncoder(
            in_channels=feature_channels,
            dim=dim,
            roi_size=roi_size,
            expand=expand,
        )

        self.attn = GeometryGuidedROIAttention(
            dim=dim,
            geometry_weight=geometry_weight,
            distance_scale=distance_scale,
        )

        self.refine = ResidualAffineRefinement(
            dim=dim,
            hidden=dim,
        )

    def forward(
        self,
        theta0,
        center_feature,
        side_feature,
        center_boxes,
        side_boxes,
        center_mask,
        side_mask,
        image_hw,
        center_centers_in_side,
    ):
        center_z = self.encoder(
            center_feature,
            center_boxes,
            center_mask,
            image_hw,
        )

        side_z = self.encoder(
            side_feature,
            side_boxes,
            side_mask,
            image_hw,
        )

        context, weights = self.attn(
            center_z=center_z,
            side_z=side_z,
            side_boxes=side_boxes,
            center_centers_in_side=center_centers_in_side,
            center_mask=center_mask,
            side_mask=side_mask,
        )

        has_pair = (
            center_mask.any(dim=-1)
            & side_mask.any(dim=-1)
        )

        theta = self.refine(theta0, context)

        theta = torch.where(
            has_pair[:, None, None],
            theta,
            theta0,
        )

        return {
            "theta": theta,
            "theta0": theta0,
            "attention": weights,
        }


class ROIContextRefinement(nn.Module):
    """Multi-view ROI refinement wrapper.

    Compatible with MultiViewCNNGeometric.forward().

    Args:
        theta0: (B, V, 2, 3)
        center_features: (B, C, Hf, Wf)
        side_features: (B, V, C, Hf, Wf)
        center_boxes: (B, Nc, 4)
        side_boxes: (B, V, Ns, 4)
        center_mask: (B, Nc)
        side_mask: (B, V, Ns)
        view_mask: (B, V)
        image_size: (H, W)
    """

    def __init__(
        self,
        feature_channels=512,
        hidden_dim=128,
        roi_size=7,
        context_scale=3.0,
        geometry_weight=1.0,
        distance_scale=64.0,
    ):
        super().__init__()

        self.head = ExperimentGHead(
            feature_channels=feature_channels,
            dim=hidden_dim,
            roi_size=roi_size,
            expand=context_scale,
            geometry_weight=geometry_weight,
            distance_scale=distance_scale,
        )

    def forward(
        self,
        theta0,
        center_features,
        side_features,
        center_boxes,
        side_boxes,
        center_mask,
        side_mask,
        view_mask,
        image_size,
    ):
        b, v = theta0.shape[:2]
        h, w = image_size

        center_mask = center_mask.to(
            device=theta0.device,
            dtype=torch.bool,
        )

        side_mask = side_mask.to(
            device=theta0.device,
            dtype=torch.bool,
        )

        view_mask = view_mask.to(
            device=theta0.device,
            dtype=torch.bool,
        )

        # Collect each view's refined theta without
        # modifying theta0 in place.
        refined_views = []

        for view_idx in range(v):
            valid = view_mask[:, view_idx]

            if not valid.any():
                refined_views.append(theta0[:, view_idx])
                continue

            current_theta = theta0[valid, view_idx]

            current_center_boxes = center_boxes[valid]
            current_side_boxes = side_boxes[valid, view_idx]

            current_center_mask = center_mask[valid]
            current_side_mask = side_mask[valid, view_idx]

            # Center ROI centers in original pixel coordinates.
            center_xy = (
                current_center_boxes[..., :2]
                + current_center_boxes[..., 2:]
            ) / 2

            scale = center_xy.new_tensor([w, h])

            # Pixel -> normalized coordinates.
            center_xy_norm = (
                2 * (center_xy + 0.5) / scale - 1
            )

            # Apply center -> side affine transformation.
            projected_norm = (
                torch.einsum(
                    "bij,bnj->bni",
                    current_theta[..., :2],
                    center_xy_norm,
                )
                + current_theta[..., 2][:, None, :]
            )

            # Normalized -> side image pixels.
            projected_pixels = (
                (projected_norm + 1) * scale / 2 - 0.5
            )

            result = self.head(
                theta0=current_theta,
                center_feature=center_features[valid],
                side_feature=side_features[valid, view_idx],
                center_boxes=current_center_boxes,
                side_boxes=current_side_boxes,
                center_mask=current_center_mask,
                side_mask=current_side_mask,
                image_hw=image_size,
                center_centers_in_side=projected_pixels,
            )

            # Reconstruct the complete batch.
            refined_for_view = theta0[:, view_idx].index_copy(
                0,
                valid.nonzero(as_tuple=True)[0],
                result["theta"],
            )

            refined_views.append(refined_for_view)

        return torch.stack(refined_views, dim=1)
