# torch imports
import torch
from torch import Tensor
# misc imports
from pydantic import validate_arguments
from typing import Optional, Union, List
# local imports
from .util import (
    _metric_reduction,
    _inputs_as_onehot,
    _inputs_as_longlabels,
    InputMode,
    Reduction,
)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def dice_score(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    smooth: float = 1e-7,
    eps: float = 1e-7,
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    weights: Optional[Union[Tensor, List]] = None,
    ignore_empty_labels: bool = False,
    from_logits: bool = False,
    label_index: Optional[int] = None,
    ignore_index: Optional[int] = None,
) -> Tensor:

    y_pred, y_true = _inputs_as_onehot(
        y_pred, 
        y_true, 
        mode=mode, 
        from_logits=from_logits, 
        discretize=True
    )

    intersection = torch.logical_and(y_pred == 1.0, y_true == 1.0).sum(dim=-1)
    pred_amounts = (y_pred == 1.0).sum(dim=-1)
    true_amounts = (y_true == 1.0).sum(dim=-1)
    cardinalities = pred_amounts + true_amounts

    dice_scores = (2 * intersection + smooth) / (cardinalities + smooth).clamp_min(eps)

    if ignore_empty_labels:
        existing_label = (true_amounts > 0).float()
        if weights is None:
            weights = existing_label
        else:
            weights = weights * existing_label

    # If a label index is provided, then we only consider the dice score from that label.
    if label_index is not None:
        assert ignore_empty_labels is False, "ignore_empty_labels is not supported when label_index is provided."
        assert ignore_index is None, "ignore_index is not supported when label_index is provided."
        dice_scores = dice_scores[:, label_index:label_index+1]
        weights = None
        
    return _metric_reduction(
        dice_scores,
        reduction=reduction,
        weights=weights,
        ignore_index=ignore_index,
        batch_reduction=batch_reduction,
    )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def pixel_accuracy(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    from_logits: bool = False,
    ignore_index: Optional[int] = None,
    batch_reduction: Reduction = "mean"
):
    y_pred_long, y_true_long = _inputs_as_longlabels(
        y_pred, 
        y_true, 
        mode=mode, 
        from_logits=from_logits, 
        discretize=True
    )
    # Note this only really makes sense in non-binary contexts.
    if ignore_index is not None:
        y_pred_long = y_pred_long[y_true_long != ignore_index] 
        y_true_long = y_true_long[y_true_long != ignore_index]
    
    scores = (y_pred_long == y_true_long).float()
    # Flatten the scores so it looks like [B, H* W]
    scores = scores.view(scores.size(0), -1)
    # Take the mean of the second dimension
    scores = scores.mean(dim=1)
    # If we want to take the mean of the batch, then we take the mean of the scores.
    if batch_reduction == "mean":
        return scores.mean()
    elif batch_reduction == "sum":
        return scores.sum()
    else:
        return scores


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def pixel_recall(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    from_logits: bool = False,
):
    y_pred_long, y_true_long = _inputs_as_longlabels(
        y_pred,
        y_true,
        mode,
        from_logits=from_logits,
        discretize=True
    )
    true_positives = ((y_pred_long == y_true_long) & (y_true_long == 1)).float().sum()
    false_negatives = ((y_pred_long != y_true_long) & (y_true_long == 1)).float().sum()
    return true_positives / (true_positives + false_negatives)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def pixel_precision(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    from_logits: bool = False,
):
    y_pred_long, y_true_long = _inputs_as_longlabels(
        y_pred,
        y_true,
        mode,
        from_logits=from_logits,
        discretize=True
    )
    true_positives = ((y_pred_long == y_true_long) & (y_true_long == 1)).float().sum()
    false_positives = ((y_pred_long != y_true_long) & (y_true_long == 0)).float().sum()
    return true_positives / (true_positives + false_positives) 


def pixel_mse(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    from_logits: bool = False,
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    ignore_index: Optional[int] = None,
    weights: Optional[Union[Tensor, List]] = None,
) -> Tensor:

    y_pred, y_true = _inputs_as_onehot(
        y_pred, y_true, mode=mode, from_logits=from_logits
    )
    # if per_channel:
    # NOTE: Each channel is weighted equally because of the mean reduction
    correct = (y_pred - y_true).square().mean(dim=-1)

    return _metric_reduction(
        correct,
        reduction=reduction,
        weights=weights,
        ignore_index=ignore_index,
        batch_reduction=batch_reduction,
    )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def soft_dice_score(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    smooth: float = 1e-7,
    eps: float = 1e-7,
    square_denom: bool = True,
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    ignore_empty_labels: bool = False,
    weights: Optional[Union[Tensor, List]] = None,
    ignore_index: Optional[int] = None,
    from_logits: bool = False,
) -> Tensor:
    y_pred, y_true = _inputs_as_onehot(
        y_pred, 
        y_true, 
        mode=mode,
        discretize=False,
        from_logits=from_logits
    )
    intersection = torch.sum(y_pred * y_true, dim=-1)

    if square_denom:
        pred_amounts = y_pred.square().sum(dim=-1)
        true_amounts = y_true.square().sum(dim=-1)
    else:
        pred_amounts = y_pred.sum(dim=-1)
        true_amounts = y_true.sum(dim=-1)
    
    cardinalities = pred_amounts + true_amounts
    soft_dice_score = (2 * intersection + smooth) / (cardinalities + smooth).clamp_min(eps)

    # If ignore_empty_labels is True, then we want to ignore labels that have no pixels in the ground truth.
    if ignore_empty_labels:
        existing_label = (true_amounts > 0).float()
        if weights is None:
            weights = existing_label
        else:
            weights = weights * existing_label
    
    
    return _metric_reduction(
        soft_dice_score,
        reduction=reduction,
        weights=weights,
        ignore_index=ignore_index,
        batch_reduction=batch_reduction,
    )
    

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def soft_jaccard_score(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    smooth: float = 1e-7,
    eps: float = 1e-7,
    square_denom: bool = True,
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    weights: Optional[Union[Tensor, List]] = None,
    ignore_index: Optional[int] = None,
    from_logits: bool = False,
) -> Tensor:
    y_pred, y_true = _inputs_as_onehot(
        y_pred, y_true, mode=mode, from_logits=from_logits
    )
    assert y_pred.shape == y_true.shape

    intersection = torch.sum(y_pred * y_true, dim=-1)

    if square_denom:
        cardinalities = y_pred.square().sum(dim=-1) + y_true.square().sum(dim=-1)
    else:
        cardinalities = y_pred.sum(dim=-1) + y_true.sum(dim=-1)

    union = cardinalities - intersection

    score = (intersection + smooth) / (union + smooth).clamp_min(eps)
    return _metric_reduction(
        score,
        reduction=reduction,
        weights=weights,
        ignore_index=ignore_index,
        batch_reduction=batch_reduction,
    )


def jaccard_score(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    smooth: float = 1e-7,
    eps: float = 1e-7,
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    weights: Optional[Union[Tensor, List]] = None,
    ignore_empty_labels: bool = True,
    from_logits: bool = False,
    ignore_index: Optional[int] = None,
) -> Tensor:

    y_pred, y_true = _inputs_as_onehot(
        y_pred,
        y_true,
        mode=mode,
        from_logits=from_logits,
        discretize=True,
    )

    intersection = torch.logical_and(y_pred == 1.0, y_true == 1.0).sum(dim=-1)
    true_amounts = (y_true == 1.0).sum(dim=-1)
    pred_amounts = (y_pred == 1.0).sum(dim=-1)
    cardinalities = true_amounts + pred_amounts
    union = cardinalities - intersection

    score = (intersection + smooth) / (union + smooth).clamp_min(eps)

    if ignore_empty_labels:
        existing_label = (true_amounts > 0).float()
        if weights is None:
            weights = existing_label
        else:
            weights = weights * existing_label

    return _metric_reduction(
        score,
        reduction=reduction,
        weights=weights,
        ignore_index=ignore_index,
        batch_reduction=batch_reduction,
    )