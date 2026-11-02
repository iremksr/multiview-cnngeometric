
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyAlignmentLoss(nn.Module):
    """
    Multi-view CNNGeometric için tiny object alignment loss.

    Beklenen girişler:
        theta:         (B, V, 2, 3)
        center_boxes:  (B, N, 4)
        view_boxes:    (B, V, N, 4)
        view_mask:     (B, V)
        instance_mask: (B, N)

    Bounding box formatı: xyxy, piksel koordinatları.

    theta:
        affine_grid ile uyumlu, output -> input
        yönündeki normalize affine dönüşüm.

    Not:
        Bu sınıf theta'nın center(output) -> side(input)
        yönünde olduğunu varsayar.
    """

    def __init__(
        self,
        image_size=512,
        center_weight=1.0,
        box_weight=0.25,
        reg_weight=0.01,
        reg_type="identity",
    ):
        super().__init__()

        self.image_size = image_size
        self.center_weight = center_weight
        self.box_weight = box_weight
        self.reg_weight = reg_weight
        self.reg_type = reg_type

    @staticmethod
    def box_centers(boxes):
        return (boxes[..., :2] + boxes[..., 2:]) / 2

    @staticmethod
    def box_corners(boxes):
        x1, y1, x2, y2 = boxes.unbind(dim=-1)

        return torch.stack(
            [
                torch.stack([x1, y1], dim=-1),
                torch.stack([x2, y1], dim=-1),
                torch.stack([x2, y2], dim=-1),
                torch.stack([x1, y2], dim=-1),
            ],
            dim=-2,
        )

    def pixel_to_normalized(self, points):
        """
        align_corners=False için:
        x_norm = 2 * (x + 0.5) / W - 1
        """
        return 2.0 * (points + 0.5) / self.image_size - 1.0

    def normalized_to_pixel(self, points):
        return (
            (points + 1.0) * self.image_size / 2.0
        ) - 0.5

    @staticmethod
    def invert_affine(theta):
        """
        (B,V,2,3) -> (B,V,3,3) inverse.

        torch.linalg.inv kullanmak yerine
        solve üzerinden hesaplanır.
        """
        B, V = theta.shape[:2]

        bottom = theta.new_zeros((B, V, 1, 3))
        bottom[..., 0, 2] = 1.0

        matrix = torch.cat([theta, bottom], dim=-2)

        identity = torch.eye(
            3,
            device=theta.device,
            dtype=theta.dtype,
        ).expand(B, V, 3, 3)

        inverse = torch.linalg.solve(matrix, identity)

        return inverse[..., :2, :]

    @staticmethod
    def transform_points(points, affine):
        """
        points: (B,V,N,K,2)
        affine: (B,V,2,3)

        returns: (B,V,N,K,2)
        """
        linear = affine[..., :2]
        translation = affine[..., 2]

        transformed = torch.einsum(
            "bvij,bvnkj->bvnki",
            linear,
            points,
        )

        return transformed + translation[:, :, None, None, :]

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

        # Center / side box merkezleri
        center_centers = self.box_centers(
            center_boxes
        )

        view_centers = self.box_centers(
            view_boxes
        )

        # Pixel -> normalized
        center_points = self.pixel_to_normalized(
            center_centers
        )

        side_points = self.pixel_to_normalized(
            view_centers
        )

        # (B,V,N,1,2)
        side_points = side_points.unsqueeze(-2)

        # theta center -> side ise
        # inverse side -> center dönüşümüdür.
        theta_inv = self.invert_affine(theta)

        transformed_centers = self.transform_points(
            side_points,
            theta_inv,
        ).squeeze(-2)

        # Normalized center error
        center_error = (
            transformed_centers
            - center_points[:, None]
        ).abs().mean(dim=-1)

        center_loss = self.masked_mean(
            center_error,
            valid,
        )

        # ---------------------------------------
        # Bounding box corner alignment
        # ---------------------------------------

        center_corners = self.box_corners(
            center_boxes
        )

        side_corners = self.box_corners(
            view_boxes
        )

        center_corners = self.pixel_to_normalized(
            center_corners
        )

        side_corners = self.pixel_to_normalized(
            side_corners
        )

        transformed_corners = self.transform_points(
            side_corners,
            theta_inv,
        )

        corner_error = (
            transformed_corners
            - center_corners[:, None]
        ).abs().mean(dim=(-1, -2))

        box_loss = self.masked_mean(
            corner_error,
            valid,
        )

        # ---------------------------------------
        # Affine regularization
        # ---------------------------------------

        identity = torch.tensor(
            [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0]],
            device=theta.device,
            dtype=theta.dtype,
        )

        reg_error = (
            theta - identity
        ).pow(2).mean(dim=(-1, -2))

        reg_loss = self.masked_mean(
            reg_error,
            view_mask,
        )

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
