"""Differentiable Generalized IoU loss for xyxy boxes."""
import torch


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Elementwise GIoU. Inputs: (..., 4) in xyxy, output: (...,)."""
    if boxes1.shape != boxes2.shape or boxes1.shape[-1] != 4:
        raise ValueError('boxes1 and boxes2 must have the same (..., 4) shape')
    a_min = torch.minimum(boxes1[..., :2], boxes1[..., 2:])
    a_max = torch.maximum(boxes1[..., :2], boxes1[..., 2:])
    b_min = torch.minimum(boxes2[..., :2], boxes2[..., 2:])
    b_max = torch.maximum(boxes2[..., :2], boxes2[..., 2:])

    inter_min = torch.maximum(a_min, b_min)
    inter_max = torch.minimum(a_max, b_max)
    inter_wh = (inter_max - inter_min).clamp(min=0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]

    area_a_wh = (a_max - a_min).clamp(min=0)
    area_b_wh = (b_max - b_min).clamp(min=0)
    area_a = area_a_wh[..., 0] * area_a_wh[..., 1]
    area_b = area_b_wh[..., 0] * area_b_wh[..., 1]
    union = area_a + area_b - inter
    iou = inter / union.clamp(min=eps)

    enclosing_min = torch.minimum(a_min, b_min)
    enclosing_max = torch.maximum(a_max, b_max)
    enclosing_wh = (enclosing_max - enclosing_min).clamp(min=0)
    enclosing_area = (enclosing_wh[..., 0] * enclosing_wh[..., 1]).clamp(min=eps)
    return iou - (enclosing_area - union) / enclosing_area


def masked_giou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor,
                     valid_mask: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Mean (1-GIoU) over valid boxes; zero with gradient if no valid boxes."""
    giou = generalized_box_iou(pred_boxes, target_boxes, eps=eps)
    if valid_mask.shape != giou.shape:
        raise ValueError('valid_mask must have shape matching boxes without last dimension')
    valid = valid_mask.bool()
    if not valid.any():
        return pred_boxes.sum() * 0.0
    return (1.0 - giou[valid]).mean()
