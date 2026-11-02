"""Tiny object affine alignment loss with per-view, per-instance masks."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyAlignmentLoss(nn.Module):
    def __init__(self, image_size=512, center_weight=1., box_weight=.25,
                 reg_weight=.01, beta=.02):
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.height, self.width = image_size
        self.center_weight = center_weight
        self.box_weight = box_weight
        self.reg_weight = reg_weight
        self.beta = beta

    def normalize(self, pts):
        size = pts.new_tensor([self.width, self.height])
        return 2 * (pts + .5) / size - 1

    @staticmethod
    def corners(box):
        x1, y1, x2, y2 = box.unbind(-1)
        return torch.stack([torch.stack([x1, y1], -1),
                            torch.stack([x2, y1], -1),
                            torch.stack([x2, y2], -1),
                            torch.stack([x1, y2], -1)], -2)

    @staticmethod
    def transform(points, theta):
        return (torch.einsum('bvij,bvnkj->bvnki', theta[..., :2], points)
                + theta[..., 2][:, :, None, None, :])

    @staticmethod
    def masked_mean(value, mask):
        mask = mask.to(value.dtype)
        return (value * mask).sum() / mask.sum().clamp_min(1)

    def forward(self, theta, center_boxes, view_boxes, view_mask,
                instance_mask, pair_mask):
        valid = (view_mask[:, :, None] & instance_mask[:, None, :] & pair_mask)
        # Empty or padded instances do not contribute to center/box loss.
        center_pts = self.normalize((center_boxes[..., :2] + center_boxes[..., 2:]) / 2)
        side_pts = self.normalize((view_boxes[..., :2] + view_boxes[..., 2:]) / 2)
        pred_side = self.transform(center_pts[:, None, :, None, :].expand(-1, theta.shape[1], -1, -1, -1), theta).squeeze(-2)
        center_err = F.smooth_l1_loss(pred_side, side_pts, reduction='none', beta=self.beta).mean(-1)
        center_loss = self.masked_mean(center_err, valid)
        center_corners = self.normalize(self.corners(center_boxes))
        side_corners = self.normalize(self.corners(view_boxes))
        pred_corners = self.transform(center_corners[:, None].expand(-1, theta.shape[1], -1, -1, -1), theta)
        box_err = F.smooth_l1_loss(pred_corners, side_corners, reduction='none', beta=self.beta).mean((-1, -2))
        box_loss = self.masked_mean(box_err, valid)
        identity = theta.new_tensor([[1., 0., 0.], [0., 1., 0.]])
        reg_err = (theta - identity).square().mean((-1, -2))
        # Only regularize views with at least one valid supervised pair.
        supervised_views = view_mask & valid.any(-1)
        reg_loss = self.masked_mean(reg_err, supervised_views)
        total = self.center_weight * center_loss + self.box_weight * box_loss + self.reg_weight * reg_loss
        return {'loss': total, 'center_loss': center_loss,
                'box_loss': box_loss, 'reg_loss': reg_loss}
