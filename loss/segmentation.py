# torch imports
import torch
import torch.nn.functional as F
from torch import Tensor

# random imports
from typing import Optional, Union
from pydantic import validate_arguments

# local imports
from .util import _loss_module_from_func
from ..util.more_functools import partial
from ..metrics.util import InputMode, Reduction
from ..metrics.segmentation import soft_dice_score, soft_jaccard_score, pixel_mse


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def soft_dice_loss(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    weights: Optional[Union[Tensor, list]] = None,
    ignore_index: Optional[int] = None,
    ignore_empty_labels: bool = False,
    from_logits: bool = False,
    smooth: float = 1e-7,
    eps: float = 1e-7,
    square_denom: bool = True,
    log_loss: bool = False,
) -> Tensor:

    score = soft_dice_score(
        y_pred,
        y_true,
        mode=mode,
        reduction=reduction,
        batch_reduction=batch_reduction,
        weights=weights,
        ignore_empty_labels=ignore_empty_labels,
        ignore_index=ignore_index,
        from_logits=from_logits,
        smooth=smooth,
        eps=eps,
        square_denom=square_denom,
    )
    # Assert that everywhere the score is between 0 and 1 (batch many items)
    assert (score >= 0).all() and (score <= 1).all(), f"Score is not between 0 and 1: {score}"

    if log_loss:
        loss = -torch.log(score.clamp_min(eps))
    else:
        loss = 1.0 - score

    return loss


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def soft_jaccard_loss(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    weights: Optional[Union[Tensor, list]] = None,
    ignore_index: Optional[int] = None,
    smooth: float = 1e-7,
    eps: float = 1e-7,
    square_denom: bool = True,
    log_loss: bool = False,
) -> Tensor:

    score = soft_jaccard_score(
        y_pred,
        y_true,
        mode=mode,
        reduction=reduction,
        batch_reduction=batch_reduction,
        weights=weights,
        ignore_index=ignore_index,
        eps=eps,
        smooth=smooth,
        square_denom=square_denom,
    )

    if log_loss:
        loss = -torch.log(score.clamp_min(eps))
    else:
        loss = 1.0 - score

    return loss


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def pixel_crossentropy_loss(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    weights: Optional[Union[Tensor, list]] = None,
    ignore_index: Optional[int] = -100,
    from_logits: bool = False,
):
    """One cross_entropy function to rule them all
    ---
    Pytorch has four CrossEntropy loss-functions
        1. Binary CrossEntropy
          - nn.BCELoss
          - F.binary_cross_entropy
        2. Sigmoid + Binary CrossEntropy (expects logits)
          - nn.BCEWithLogitsLoss
          - F.binary_cross_entropy_with_logits
        3. Categorical
          - nn.NLLLoss
          - F.nll_loss
        4. Softmax + Categorical (expects logits)
          - nn.CrossEntropyLoss
          - F.cross_entropy
    """
    assert len(y_pred.shape) > 2, "y_pred must have at least 3 dimensions."
    batch_size, num_classes = y_pred.shape[:2]
    y_true = y_true.long()

    if mode == "auto":
        if y_pred.shape == y_true.shape:
            mode = "binary" if num_classes == 1 else "onehot"
        else:
            mode = "multiclass"

    # If weights are a list turn them into a tensor
    if isinstance(weights, list):
        weights = torch.tensor(weights, device=y_pred.device, dtype=y_pred.dtype)

    if mode == "binary":
        assert y_pred.shape == y_true.shape
        assert ignore_index is None
        assert weights is None
        if from_logits:
            loss = F.binary_cross_entropy_with_logits(
                y_pred, 
                y_true, 
                reduction="none"
                )
        else:
            loss = F.binary_cross_entropy(
                y_pred, 
                y_true, 
                reduction="none"
                )
        loss = loss.squeeze(dim=1)
    else:
        # Squeeze the label, (no need for channel dimension).
        if len(y_true.shape) == len(y_pred.shape):
            y_true = y_true.squeeze(1)

        if from_logits:
            loss = F.cross_entropy(
                y_pred,
                y_true,
                reduction="none",
                weight=weights,
                ignore_index=ignore_index,
            )
        else:
            loss = F.nll_loss(
                y_pred,
                y_true,
                reduction="none",
                weight=weights,
                ignore_index=ignore_index,
            )

    # Channels have been collapsed
    spatial_dims = list(range(1, len(y_pred.shape) - 1))
    if reduction == "mean":
        loss = loss.mean(dim=spatial_dims)
    if reduction == "sum":
        loss = loss.sum(dim=spatial_dims)

    if batch_reduction == "mean":
        loss = loss.mean(dim=0)
    if batch_reduction == "sum":
        loss = loss.sum(dim=0)

    return loss


pixel_mse_loss = pixel_mse
binary_soft_dice_loss = partial(soft_dice_loss, mode="binary")
binary_soft_jaccard_loss = partial(soft_jaccard_loss, mode="binary")

SoftDiceLoss = _loss_module_from_func("SoftDiceLoss", soft_dice_loss)
SoftJaccardLoss = _loss_module_from_func("SoftJaccardLoss", soft_jaccard_loss)
PixelMSELoss = _loss_module_from_func("PixelMSELoss", pixel_mse_loss)
PixelCELoss = _loss_module_from_func("PixelCELoss", pixel_crossentropy_loss)