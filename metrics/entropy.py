import torch
import torch.nn.functional as F


def binary_cross_entropy(
    y_pred, 
    y_true, 
    from_logits=True,
    batch_reduction="mean"
):
    if from_logits:
        y_pred = torch.sigmoid(y_pred)
    if batch_reduction is None:
        batch_reduction = "none"
    return F.binary_cross_entropy(y_pred, y_true, reduction=batch_reduction)