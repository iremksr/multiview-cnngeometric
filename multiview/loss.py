
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyAlignmentLoss(nn.Module):
    """
    CNNGeometric multi-view tiny-object alignment loss.

    theta:         (B, V, 2, 3)
    center_boxes:  (B, N, 4)
    view_boxes:    (B, V, N, 4)
    view_mask:     (B, V)
    instance_mask: (B, N)

    theta maps normalized CENTER coordinates
    to normalized SIDE coordinates.

    Bounding boxes use pixel-space xyxy.
    """

    def __init__(
        self,
        image_size=512,
        center_weight=1.0,
        box_weight=0.25,
        reg_weight=0.01,
        beta=0.02,
    ):
        super().__init__()

        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        self.height, self.width = image_size

        self.center_weight = center_weight
        self.box_weight = box_weight
        self.reg_weight = reg_weight
        self.beta = beta

    @staticmethod
    def box_centers(boxes):
        return (
            boxes[..., :2] + boxes[..., 2:]
        ) / 2.0

    @staticmethod
    def box_corners(boxes):
        x1, y1, x2, y2 = boxes.unbind(-1)

        return torch.stack([
            torch.stack([x1, y1], dim=-1),
            torch.stack([x2, y1], dim=-1),
            torch.stack([x2, y2], dim=-1),
            torch.stack([x1, y2], dim=-1),
        ], dim=-2)

    def pixel_to_normalized(self, points):
        scale = points.new_tensor([
            self.width,
            self.height,
        ])

        return 2.0 * (points + 0.5) / scale - 1.0

    @staticmethod
    def transform_points(points, theta):
        """
        points: (B,V,N,K,2)
        theta:  (B,V,2,3)
        """

        linear = theta[..., :2]
        translation = theta[..., 2]

        transformed = torch.einsum(
            "bvij,bvnkj->bvnki",
            linear,
            points,
        )

        return (
            transformed
            + translation[:, :, None, None, :]
        )

    @staticmethod
    def masked_mean(values, mask):
        mask = mask.to(values.dtype)

        return (
            (values * mask).sum()
            / mask.sum().clamp_min(1.0)
        )

    def forward(
        self,
        theta,
        center_boxes,
        view_boxes,
        view_mask,
        instance_mask,
    ):
        B, V = theta.shape[:2]

        valid = (
            view_mask[:, :, None]
            & instance_mask[:, None, :]
        )

        # ----------------------------------
        # 1. Center alignment
        # ----------------------------------

        center_points = self.pixel_to_normalized(
            self.box_centers(center_boxes)
        )

        side_points = self.pixel_to_normalized(
            self.box_centers(view_boxes)
        )

        # Center -> Side
        center_points = (
            center_points[:, None]
            .expand(-1, V, -1, -1)
            .unsqueeze(-2)
        )

        predicted_side_points = (
            self.transform_points(
                center_points,
                theta,
            ).squeeze(-2)
        )

        center_error = F.smooth_l1_loss(
            predicted_side_points,
            side_points,
            reduction="none",
            beta=self.beta,
        ).mean(dim=-1)

        center_loss = self.masked_mean(
            center_error,
            valid,
        )

        # ----------------------------------
        # 2. Bounding box alignment
        # ----------------------------------

        center_corners = self.pixel_to_normalized(
            self.box_corners(center_boxes)
        )

        side_corners = self.pixel_to_normalized(
            self.box_corners(view_boxes)
        )

        center_corners = (
            center_corners[:, None]
            .expand(-1, V, -1, -1, -1)
        )

        predicted_side_corners = (
            self.transform_points(
                center_corners,
                theta,
            )
        )

        box_error = F.smooth_l1_loss(
            predicted_side_corners,
            side_corners,
            reduction="none",
            beta=self.beta,
        ).mean(dim=(-1, -2))

        box_loss = self.masked_mean(
            box_error,
            valid,
        )

        # ----------------------------------
        # 3. Affine regularization
        # ----------------------------------

        identity = theta.new_tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

        reg_error = (
            theta - identity
        ).square().mean(dim=(-1, -2))

        reg_loss = self.masked_mean(
            reg_error,
            view_mask,
        )

        # ----------------------------------
        # 4. Total loss
        # ----------------------------------

        total = (
            self.center_weight * center_loss
            + self.box_weight * box_loss
            + self.reg_weight * reg_loss
        )

        return {
            "loss": total,
            "center_loss": center_loss,
            "box_loss": box_loss,
            "reg_loss": reg_loss,
        }
