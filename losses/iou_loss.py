"""Custom IoU loss 
"""

import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output: 'mean' | 'sum'.
        """
        super().__init__()
        self.eps = eps
        self.reduction = reduction

        # validate reduction
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError("reduction must be one of: none | mean | sum")

        self.reduction = reduction
        # TODO: validate reduction in {"none", "mean", "sum"}.

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format."""
        # TODO: implement IoU loss.
        # unpack
        px, py, pw, ph = pred_boxes[:,0], pred_boxes[:,1], pred_boxes[:,2], pred_boxes[:,3]
        tx, ty, tw, th = target_boxes[:,0], target_boxes[:,1], target_boxes[:,2], target_boxes[:,3]

        # convert center -> corner
        p_x1 = px - pw / 2
        p_y1 = py - ph / 2
        p_x2 = px + pw / 2
        p_y2 = py + ph / 2

        t_x1 = tx - tw / 2
        t_y1 = ty - th / 2
        t_x2 = tx + tw / 2
        t_y2 = ty + th / 2

        # intersection
        x1 = torch.max(p_x1, t_x1)
        y1 = torch.max(p_y1, t_y1)
        x2 = torch.min(p_x2, t_x2)
        y2 = torch.min(p_y2, t_y2)

        inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

        # union
        area_pred = pw * ph
        area_target = tw * th

        union = area_pred + area_target - inter + self.eps

        iou = inter / union

        loss = 1 - iou

        # reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
        raise NotImplementedError("Implement IoULoss.forward")
    


