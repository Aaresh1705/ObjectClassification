import torch 

def intersection_over_union(y_pred: torch.Tensor, y_true: torch.Tensor, smooth=1e-6):
    """Computes Intersection over Union (IoU) between predicted and true masks
    Args:
        y_pred (torch.Tensor): Predicted segmentation masks
        y_true (torch.Tensor): Ground truth segmentation masks
        smooth (float): Smoothing factor to avoid division by zero
    Returns:
        float: IoU score
    """
    if y_true.ndim == 3:
        y_true = y_true.unsqueeze(1)
    # Ensure correct shape
    assert y_pred.shape == y_true.shape, "Shape mismatch between predictions and ground truth"
    # Apply sigmoid to convert logits → probabilities
    y_pred = torch.sigmoid(y_pred) 
    # Flatten across spatial dimensions
    intersection = (y_pred * y_true).sum(dim=(2, 3))
    union = y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3)) - intersection
    # Compute IoU
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()