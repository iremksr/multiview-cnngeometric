"""Tiny-object affine alignment with partial per-view/per-instance supervision."""
import torch
from torch import nn
from torch.nn import functional as F


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
    def denormalize(pts, width, height):
        return (pts + 1) * pts.new_tensor([width, height]) / 2 - .5

    @staticmethod
    def corners(box):
        x1, y1, x2, y2 = box.unbind(-1)
        return torch.stack((torch.stack((x1, y1), -1),
                            torch.stack((x2, y1), -1),
                            torch.stack((x2, y2), -1),
                            torch.stack((x1, y2), -1)), -2)

    @staticmethod
    def transform(points, theta):
        # theta: (B,V,2,3); points: (B,V,N,K,2)
        return (torch.einsum('bvij,bvnkj->bvnki', theta[..., :2], points)
                + theta[..., 2][:, :, None, None, :])

    @staticmethod
    def masked_mean(value, mask):
        mask = mask.to(value.dtype)
        return (value * mask).sum() / mask.sum().clamp_min(1)

    def forward(self, theta, center_boxes, view_boxes, view_mask,
                instance_mask, pair_mask):
        valid = view_mask[:, :, None] & instance_mask[:, None, :] & pair_mask
        center_pts = self.normalize((center_boxes[..., :2] + center_boxes[..., 2:]) / 2)
        side_pts = self.normalize((view_boxes[..., :2] + view_boxes[..., 2:]) / 2)
        pred_side = self.transform(center_pts[:, None, :, None, :].expand(
            -1, theta.shape[1], -1, -1, -1), theta).squeeze(-2)
        center_err = F.smooth_l1_loss(pred_side, side_pts,
                                      reduction='none', beta=self.beta).mean(-1)
        center_loss = self.masked_mean(center_err, valid)

        center_corners = self.normalize(self.corners(center_boxes))
        pred_corners = self.transform(center_corners[:, None].expand(
            -1, theta.shape[1], -1, -1, -1), theta)
        # Axis-aligned enclosure of the transformed center box.
        pred_box = torch.cat((pred_corners.amin(-2), pred_corners.amax(-2)), -1)
        side_corners = self.normalize(self.corners(view_boxes))
        target_box = torch.cat((side_corners.amin(-2), side_corners.amax(-2)), -1)
        box_err = F.smooth_l1_loss(pred_box, target_box,
                                   reduction='none', beta=self.beta).mean(-1)
        box_loss = self.masked_mean(box_err, valid)

        identity = theta.new_tensor([[1., 0., 0.], [0., 1., 0.]])
        reg_err = (theta - identity).square().mean((-1, -2))
        supervised_views = view_mask & valid.any(-1)
        reg_loss = self.masked_mean(reg_err, supervised_views)
        total = (self.center_weight * center_loss + self.box_weight * box_loss
                 + self.reg_weight * reg_loss)
        return {'loss': total, 'center_loss': center_loss,
                'box_loss': box_loss, 'reg_loss': reg_loss}

    @torch.no_grad()
    def alignment_metrics(self, theta, center_boxes, view_boxes, view_mask,
                          instance_mask, pair_mask):
        """IoU in SIDE coordinates, before/after center->side affine mapping.

        Returns (sum_before, sum_after, count_after_iou_ge_0_5, valid_pair_count).
        """
        valid = view_mask[:, :, None] & instance_mask[:, None, :] & pair_mask
        count = int(valid.sum().item())
        if not count:
            return 0., 0., 0., 0
        center_corners = self.normalize(self.corners(center_boxes))
        pred_corners = self.transform(center_corners[:, None].expand(
            -1, theta.shape[1], -1, -1, -1), theta)
        pred_box_norm = torch.cat((pred_corners.amin(-2), pred_corners.amax(-2)), -1)
        pred_box = self.denormalize(pred_box_norm.reshape(*pred_box_norm.shape[:-1], 2, 2),
                                    self.width, self.height).reshape_as(pred_box_norm)
        before = center_boxes[:, None].expand_as(view_boxes)
        target = view_boxes

        def iou(a, b):
            lo = torch.maximum(a[..., :2], b[..., :2])
            hi = torch.minimum(a[..., 2:], b[..., 2:])
            intersection = (hi - lo).clamp_min(0).prod(-1)
            area_a = (a[..., 2:] - a[..., :2]).clamp_min(0).prod(-1)
            area_b = (b[..., 2:] - b[..., :2]).clamp_min(0).prod(-1)
            return intersection / (area_a + area_b - intersection).clamp_min(1e-8)

        before_iou = iou(before, target)[valid]
        after_iou = iou(pred_box, target)[valid]
        return (float(before_iou.sum().item()), float(after_iou.sum().item()),
                int((after_iou >= .5).sum().item()), count)
